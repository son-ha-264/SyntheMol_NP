"""Hsp90alpha NTD docking oracle for generative-model reinforcement learning.

Wraps the setup from hsp90_docking_setup.sh: AutoDock Vina 1.2 with Vinardo
scoring and a box enclosing the union of the radicicol (4EGK) and NVP-AUY922
(6LTI) ligand envelopes, so it favours neither chemotype.

    from hsp90_oracle import make_oracle
    oracle = make_oracle(receptor="np", n_workers=16)
    rewards = oracle(smiles_batch)          # list of floats in [0, 1]

Backends. "python" (default) uses Vina's API with Meeko in memory: affinity maps
are computed once per worker and reused, with no temp files or process spawn per
molecule. "subprocess" shells out to vina / mk_prepare_ligand.py, which is slower
but useful when the bindings and the CLI are different versions.

Reward. The size-corrected residual, |score| - (b0 + b1*HAC), not ligand
efficiency; see SIZE_FIT below for why. Property limits are hard gates returning
0 rather than weighted terms, which could be traded off against the score.
Results are cached by canonical SMILES. Nothing raises: bad SMILES, embedding
failure, Vina error and timeout all return 0.0 with the reason recorded.

Calibration at EXHAUSTIVENESS 64, mean of seeds 1/2/3/42, crystal conformers:
    radicicol  in 4EGK: -4.877 +/- 0.027 kcal/mol, 25 heavy atoms, redock 0.26 A
    NVP-AUY922 in 6LTI: -8.022 +/- 0.006 kcal/mol, 34 heavy atoms, redock 1.78 A

Why exhaustiveness 64 rather than Vina's default 8 or the 32 used earlier:
sampling adequacy is chemotype-dependent. At 32 the rigid synthetic ligand is
reproducible to 0.014 kcal/mol while the macrocyclic natural product misses its
crystal pose in 1 of 4 seeds (spread 1.00). At 64 the macrocycle is reproducible
to 0.061 and passes 4 of 4. Choose exhaustiveness from the MACROCYCLE; validating
on a drug-like reference silently undersamples natural-product-like output.

    python hsp90_oracle.py --self-test        # reproduce the calibration
    python hsp90_oracle.py --diagnose         # environment report
    python hsp90_oracle.py --where            # resolved data root

Requires rdkit, meeko, and vina (bindings for the python backend, the executable
for subprocess; they are SEPARATE installs).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

__all__ = ["Hsp90Oracle", "DualReceptorOracle", "Hsp90LigandEfficiency",
           "Filters", "ScoreRecord", "make_oracle", "make_dual_oracle",
           "resolve_root", "docking_metric", "MODULE_DIR", "SIZE_FIT",
           "self_test", "diagnose"]

RECEPTORS = {"np": "4EGK", "synth": "6LTI"}

# Data is found relative to THIS FILE, not the working directory, so a training
# script launched from anywhere finds the receptors.
MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_SUBDIRS = ("data/hsp90", "hsp90_vina", "hsp90_v2")
_SEARCH_BASES = (MODULE_DIR, MODULE_DIR.parent, Path.cwd())

# Size correction measured for THIS scoring function and pocket:
#   |score| = b0 + b1*HAC, fitted on molecules with <= 6 rotatable bonds, where
#   size is not confounded with flexibility (r = 0.731).
# The relationship is AFFINE, so dividing by heavy atoms leaves a b0/HAC term
# that dominates and turns ligand efficiency into a measure of 1/size - which is
# why a 16-atom benzamide outscored a 1.7 nM clinical candidate. Subtracting the
# fitted trend instead gives "better than expected for this size".
# WARNING: fitted at EXHAUSTIVENESS 32; the default is now 64, which shifts
# scores (radicicol -4.78 -> -4.88). Refit at 64, on a POOLED unconditional
# sample from both priors, before publishing. size_correction_fit.json in the
# working directory overrides these, so refitting needs no code change.
SIZE_FIT = {"b0": 3.4561, "b1": 0.0796}

# "residual"   |score| - (b0 + b1*HAC). Size-neutral by construction; default.
# "efficiency" |score| / HAC. Biased toward small molecules here.
# "raw"        |score|. Biased toward large (0.08 kcal/mol per heavy atom).
NORMALIZATION = "residual"
# PROVISIONAL windows. The residual pair was widened from the measured
# (-2.73, 0.85) because at exhaustiveness 64 NVP-AUY922's residual is +1.86 and
# was clipping to reward 1.0 - a saturated arm has no gradient left. The old
# bounds came from exhaustiveness-32 probe scores, and their lower end was set
# by a single PEG-chain outlier. Refit both from residual percentiles of a
# pooled unconditional sample from BOTH priors, docked at 64.
_WINDOWS = {"residual": (-2.50, 2.50), "efficiency": (0.065, 0.415),
            "raw": (2.5, 9.0)}


def _load_size_fit(path: str = "size_correction_fit.json") -> None:
    p = Path(path)
    if not p.is_file():
        return
    try:
        d = json.loads(p.read_text())
        if "b0" in d and "b1" in d:
            SIZE_FIT["b0"], SIZE_FIT["b1"] = float(d["b0"]), float(d["b1"])
    except Exception:
        pass


_load_size_fit()


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class Filters:
    """Hard gates. Failing any scores 0 and the molecule is not docked.

    Permissive on size: the residual reward subtracts size, so extra atoms buy
    no reward and these are not anti-hacking devices. A molecule too large for
    the pocket is punished automatically, scoring worse than its size predicts.
    Rotatable bonds drive docking COST (42 vs 6.65 s/mol in the probe set), and
    RDKit does not count macrocyclic ring bonds, so that cap limits long acyclic
    chains rather than NP macrocycles. logP stays tight: it correlates with the
    score at r = 0.553 here and the residual corrects for size only, while large
    natural products sit well under 7 anyway. Report the values used.
    """
    max_heavy_atoms: int = 70          # erythromycin 51, paclitaxel 62, rapamycin 65
    min_heavy_atoms: int = 12
    max_mol_weight: float = 1000.0     # excludes cyclosporine-scale peptides
    max_rotatable_bonds: int = 15
    max_logp: float = 7.0

    def check(self, mol: Chem.Mol) -> Optional[str]:
        """Name of the first failed gate, or None."""
        hac = mol.GetNumHeavyAtoms()
        if hac > self.max_heavy_atoms:
            return "max_heavy_atoms"
        if hac < self.min_heavy_atoms:
            return "min_heavy_atoms"
        if Descriptors.MolWt(mol) > self.max_mol_weight:
            return "max_mol_weight"
        if rdMolDescriptors.CalcNumRotatableBonds(mol) > self.max_rotatable_bonds:
            return "max_rotatable_bonds"
        if Descriptors.MolLogP(mol) > self.max_logp:
            return "max_logp"
        return None


@dataclass
class ScoreRecord:
    smiles: str
    reward: float = 0.0
    raw_score: Optional[float] = None
    heavy_atoms: Optional[int] = None
    ligand_efficiency: Optional[float] = None   # holds the active metric
    failed_filter: Optional[str] = None
    error: Optional[str] = None


def parse_box_config(path: str) -> Tuple[List[float], List[float]]:
    """Meeko's Vina box file -> (center, box_size)."""
    v: Dict[str, float] = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"\s*(\w+)\s*=\s*([-\d.]+)", line)
        if m:
            v[m.group(1)] = float(m.group(2))
    try:
        return ([v["center_x"], v["center_y"], v["center_z"]],
                [v["size_x"], v["size_y"], v["size_z"]])
    except KeyError as exc:
        raise ValueError(f"{path} is missing {exc}") from None


def _find_exe(explicit: Optional[str], name: str, env_var: str) -> Optional[str]:
    """Locate an executable without requiring it on PATH: explicit argument,
    then the environment variable, then PATH. The Vina bindings and the Vina
    executable are separate installs, so an environment often has only one."""
    for cand in (explicit, os.environ.get(env_var)):
        if cand:
            p = Path(cand).expanduser()
            if p.is_file() and os.access(p, os.X_OK):
                return str(p.resolve())
            if shutil.which(cand):
                return shutil.which(cand)
    return shutil.which(name)


def docking_metric(score: float, heavy_atoms: int, norm: dict) -> float:
    """Raw docking score -> the quantity the reward is built from."""
    mag = -score                                   # positive; bigger is better
    mode = norm.get("mode", "residual")
    if mode == "raw":
        return mag
    if mode == "efficiency":
        return mag / heavy_atoms
    return mag - (norm["b0"] + norm["b1"] * heavy_atoms)


# --------------------------------------------------------------------------- #
# Molecule handling, shared by both backends
# --------------------------------------------------------------------------- #

def embed_3d(smiles: str) -> Optional[Chem.Mol]:
    """SMILES -> one 3D conformer with explicit hydrogens.

    Vina searches rotatable bonds itself and Meeko opens macrocycles so their
    ring conformation is searched too, so one ETKDG conformer suffices as a
    starting point. Substitute Gypsum-DL here for stereoisomer and protonation
    enumeration, scoring each state and taking the best.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = 0xF00D
    ps.useSmallRingTorsions = True
    if AllChem.EmbedMolecule(mol, ps) != 0:
        ps.useRandomCoords = True                  # macrocycles often need this
        if AllChem.EmbedMolecule(mol, ps) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass                                       # unoptimised still docks
    return mol


def mol_to_pdbqt_string(mol: Chem.Mol) -> Tuple[Optional[str], Optional[str]]:
    """Meeko in memory. Returns (pdbqt_string, error)."""
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
    except ImportError as exc:
        return None, f"meeko import failed: {exc}"
    try:
        setups = MoleculePreparation()(mol)        # callable, not .prepare()
        if not setups:
            return None, "meeko returned no setups"
        res = PDBQTWriterLegacy.write_string(setups[0])
        # Version-tolerant: some releases return (string, ok, error_message).
        if isinstance(res, tuple):
            pdbqt = res[0]
            if len(res) > 1 and not res[1]:
                return None, f"meeko write failed: {str(res[2:3])[:100]}"
        else:
            pdbqt = res
        if not pdbqt or "ATOM" not in pdbqt:
            return None, "meeko produced an empty pdbqt"
        return pdbqt, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"[:140]


def _pre_dock(smiles: str, filters_d: dict) -> Tuple[ScoreRecord, Optional[Chem.Mol]]:
    """Validate, filter and embed. If mol is None the record is already final."""
    rec = ScoreRecord(smiles=smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        rec.error = "invalid_smiles"
        return rec, None
    rec.heavy_atoms = mol.GetNumHeavyAtoms()
    failed = Filters(**filters_d).check(mol)
    if failed:
        rec.failed_filter = failed
        return rec, None
    mol3d = embed_3d(smiles)
    if mol3d is None:
        rec.error = "embedding_failed"
    return rec, mol3d


def _post_dock(rec: ScoreRecord, score: float, norm: dict) -> dict:
    rec.raw_score = score
    rec.ligand_efficiency = docking_metric(score, rec.heavy_atoms, norm)
    lo, hi = norm["lo"], norm["hi"]
    rec.reward = max(0.0, min(1.0, (rec.ligand_efficiency - lo) / (hi - lo)))
    return asdict(rec)


# --------------------------------------------------------------------------- #
# Backends. _W holds per-process state; Vina objects cannot be pickled.
# --------------------------------------------------------------------------- #

_W: dict = {}


def _init_worker(receptor_pdbqt: str, center: List[float], box_size: List[float],
                 scoring: str, seed: int, cpu: int) -> None:
    """Build the Vina instance and compute the affinity maps ONCE per process.
    With no ligand loaded the maps cover all 22 Vina atom types, so they serve
    every ligand this worker docks. That is the whole case for the Python API."""
    from vina import Vina
    try:
        v = Vina(sf_name=scoring, cpu=cpu, seed=seed, verbosity=0)
    except TypeError:                              # older signature, no seed
        v = Vina(sf_name=scoring, cpu=cpu, verbosity=0)
        try:
            v.set_seed(seed)
        except Exception:
            pass
    v.set_receptor(receptor_pdbqt)
    v.compute_vina_maps(center=center, box_size=box_size)
    _W["vina"] = v


def _dock_python(smiles: str, filters_d: dict, exhaustiveness: int,
                 n_poses: int, norm: dict) -> dict:
    rec, mol3d = _pre_dock(smiles, filters_d)
    if mol3d is None:
        return asdict(rec)
    pdbqt, err = mol_to_pdbqt_string(mol3d)
    if pdbqt is None:
        rec.error = err
        return asdict(rec)
    try:
        v = _W["vina"]
        v.set_ligand_from_string(pdbqt)
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        score = float(v.energies(n_poses=1)[0][0])     # best pose, total energy
    except Exception as exc:
        rec.error = f"vina: {type(exc).__name__}: {exc}"[:140]
        return asdict(rec)
    return _post_dock(rec, score, norm)


def _parse_best_score(text: str) -> Optional[float]:
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "1":
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


def _dock_subprocess(smiles: str, filters_d: dict, receptor: str, box_cfg: str,
                     scoring: str, exhaustiveness: int, n_poses: int, seed: int,
                     timeout_s: int, norm: dict, vina_exe: str = "vina",
                     meeko_exe: str = "mk_prepare_ligand.py") -> dict:
    rec, mol3d = _pre_dock(smiles, filters_d)
    if mol3d is None:
        return asdict(rec)
    tmp = tempfile.mkdtemp(prefix="hsp90dock_")
    try:
        sdf, pdbqt, out = (os.path.join(tmp, f) for f in ("l.sdf", "l.pdbqt", "o.pdbqt"))
        Chem.Kekulize(mol3d, clearAromaticFlags=True)
        Chem.MolToMolFile(mol3d, sdf, kekulize=True)
        p = subprocess.run([meeko_exe, "-i", sdf, "-o", pdbqt],
                           capture_output=True, text=True, timeout=120)
        if p.returncode != 0 or not os.path.getsize(pdbqt):
            rec.error = f"meeko: {p.stderr.strip()[:110]}"
            return asdict(rec)
        r = subprocess.run([vina_exe, "--receptor", receptor, "--ligand", pdbqt,
                            "--config", box_cfg, "--scoring", scoring,
                            "--exhaustiveness", str(exhaustiveness),
                            "--num_modes", str(n_poses), "--seed", str(seed),
                            "--cpu", "1", "--out", out],
                           capture_output=True, text=True, timeout=timeout_s)
        if r.returncode != 0:
            rec.error = f"vina: {r.stderr.strip()[:110]}"
            return asdict(rec)
        score = _parse_best_score(r.stdout)
        if score is None:
            rec.error = "no_score_parsed"
            return asdict(rec)
        return _post_dock(rec, score, norm)
    except subprocess.TimeoutExpired:
        rec.error = "timeout"
        return asdict(rec)
    except Exception as exc:
        rec.error = f"{type(exc).__name__}: {exc}"[:140]
        return asdict(rec)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Oracle
# --------------------------------------------------------------------------- #

class Hsp90Oracle:
    """Callable docking oracle. SMILES in, rewards in [0, 1] out.

    The worker pool is created once and reused across score() calls: each
    python-backend worker computes the affinity maps on startup, so a pool per
    call would recompute them every training step. Use as a context manager, or
    call close(), to shut the workers down and flush the cache.
    """

    def __init__(
        self,
        receptor_pdbqt: str,
        box_config: str,
        *,
        backend: str = "python",
        scoring: str = "vinardo",
        exhaustiveness: int = 64,
        n_poses: int = 9,
        seed: int = 42,
        timeout_s: int = 600,
        filters: Optional[Filters] = None,
        normalization: Optional[str] = None,
        le_min: Optional[float] = None,
        le_max: Optional[float] = None,
        n_workers: int = 8,
        cache_path: Optional[str] = None,
        vina_exe: Optional[str] = None,
        meeko_exe: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.receptor = str(Path(receptor_pdbqt).resolve())
        self.box_config = str(Path(box_config).resolve())
        for p in (self.receptor, self.box_config):
            if not Path(p).is_file():
                raise FileNotFoundError(p)
        self.center, self.box_size = parse_box_config(self.box_config)

        self.normalization = normalization or NORMALIZATION
        if self.normalization not in _WINDOWS:
            raise ValueError(f"unknown normalization: {self.normalization}")
        # Windows differ per normalization: the residual is a kcal/mol offset
        # near zero, efficiency ~0.1-0.4, raw ~3-10. One shared window would
        # clip everything to 0 or 1.
        d_lo, d_hi = _WINDOWS[self.normalization]
        self.le_min = d_lo if le_min is None else le_min
        self.le_max = d_hi if le_max is None else le_max

        self.scoring, self.exhaustiveness = scoring, exhaustiveness
        self.n_poses, self.seed, self.timeout_s = n_poses, seed, timeout_s
        self.filters = filters or Filters()
        self.n_workers = max(1, n_workers)
        self.verbose = verbose

        self.vina_exe = _find_exe(vina_exe, "vina", "VINA_EXE")
        self.meeko_exe = _find_exe(meeko_exe, "mk_prepare_ligand.py", "MEEKO_EXE")
        self.backend = self._resolve_backend(backend)

        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: Dict[str, dict] = {}
        if self.cache_path and self.cache_path.is_file():
            try:
                self._cache = json.loads(self.cache_path.read_text())
            except Exception:
                self._cache = {}

        self.n_docked = self.n_cached = self.n_failed = 0
        self._pool: Optional[ProcessPoolExecutor] = None

    def _resolve_backend(self, backend: str) -> str:
        if backend not in ("python", "subprocess"):
            raise ValueError("backend must be 'python' or 'subprocess'")
        if backend == "python":
            try:
                import vina  # noqa: F401
                return "python"
            except Exception as exc:
                print(f"[Hsp90Oracle] cannot import the vina bindings "
                      f"({type(exc).__name__}: {exc})")
                if not (self.vina_exe and self.meeko_exe):
                    raise RuntimeError(
                        "Neither the vina Python bindings nor the vina executable "
                        "are usable here. They are SEPARATE installs:\n"
                        "  bindings:   conda install -c conda-forge numpy swig "
                        "boost-cpp libboost && pip install vina\n"
                        "  executable: conda install -c conda-forge vina\n"
                        "  or point at one: Hsp90Oracle(..., vina_exe='/path/to/vina')"
                        " / export VINA_EXE=...\n"
                        "Run `python hsp90_oracle.py --diagnose` for a full report."
                    ) from exc
                print("[Hsp90Oracle] falling back to the subprocess backend")
                backend = "subprocess"
        missing = [n for n, p in (("vina", self.vina_exe),
                                  ("mk_prepare_ligand.py", self.meeko_exe))
                   if p is None]
        if missing:
            raise RuntimeError(
                f"subprocess backend needs {', '.join(missing)}: not on PATH, "
                f"VINA_EXE or MEEKO_EXE. See `--diagnose`.")
        return backend

    # -- plumbing --------------------------------------------------------- #

    def save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.write_text(json.dumps(self._cache))

    @staticmethod
    def _canonical(smiles: str) -> Optional[str]:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol is not None else None

    def _norm(self) -> dict:
        return {"mode": self.normalization, "b0": SIZE_FIT["b0"],
                "b1": SIZE_FIT["b1"], "lo": self.le_min, "hi": self.le_max}

    def _dock_args(self, smi: str) -> tuple:
        fd = asdict(self.filters)
        if self.backend == "python":
            return (smi, fd, self.exhaustiveness, self.n_poses, self._norm())
        return (smi, fd, self.receptor, self.box_config, self.scoring,
                self.exhaustiveness, self.n_poses, self.seed, self.timeout_s,
                self._norm(), self.vina_exe, self.meeko_exe)

    @property
    def _dock_fn(self):
        return _dock_python if self.backend == "python" else _dock_subprocess

    def _pool_kwargs(self, cpu: int = 1) -> dict:
        if self.backend == "python":
            return {"initializer": _init_worker,
                    "initargs": (self.receptor, self.center, self.box_size,
                                 self.scoring, self.seed, cpu)}
        return {}

    def _get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            if self.verbose:
                print(f"[Hsp90Oracle] starting {self.n_workers} persistent "
                      f"worker(s)", flush=True)
            self._pool = ProcessPoolExecutor(max_workers=self.n_workers,
                                             **self._pool_kwargs(cpu=1))
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        self.save_cache()

    def __enter__(self) -> "Hsp90Oracle":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- scoring ---------------------------------------------------------- #

    def score(self, smiles_list: Sequence[str]) -> List[ScoreRecord]:
        keys = [self._canonical(s) for s in smiles_list]
        todo = sorted({k for k in keys if k and k not in self._cache})

        if todo:
            # Never spawn more workers than molecules: each python-backend
            # worker computes the maps on startup, so an oversized pool pays
            # that once per worker and leaves most idle.
            n_eff = min(self.n_workers, len(todo))
            t0 = time.time()
            if self.verbose:
                print(f"[Hsp90Oracle] docking {len(todo)} molecule(s), {n_eff} "
                      f"worker(s), exhaustiveness {self.exhaustiveness}, "
                      f"backend {self.backend}", flush=True)
            if n_eff <= 1:
                # One molecule: run in-process and give Vina every core (cpu=0).
                if self.backend == "python" and "vina" not in _W:
                    _init_worker(self.receptor, self.center, self.box_size,
                                 self.scoring, self.seed, 0)
                for i, smi in enumerate(todo, 1):
                    ts = time.time()
                    self._cache[smi] = self._dock_fn(*self._dock_args(smi))
                    if self.verbose:
                        r = self._cache[smi]
                        note = (r.get("error") or r.get("failed_filter")
                                or f"{r.get('raw_score')} kcal/mol")
                        print(f"[Hsp90Oracle]   {i}/{len(todo)} "
                              f"{time.time() - ts:.1f}s  {note}", flush=True)
            else:
                pool, done = self._get_pool(), 0
                futs = {pool.submit(self._dock_fn, *self._dock_args(s)): s
                        for s in todo}
                for fut in as_completed(futs):
                    smi = futs[fut]
                    try:
                        self._cache[smi] = fut.result()
                    except Exception as exc:
                        self._cache[smi] = asdict(ScoreRecord(
                            smiles=smi, error=f"worker: {exc}"[:140]))
                    done += 1
                    if self.verbose and (done % 25 == 0 or done == len(todo)):
                        el = time.time() - t0
                        print(f"[Hsp90Oracle]   {done}/{len(todo)} in {el:.0f}s "
                              f"({el/done:.1f}s/mol)", flush=True)
            self.n_docked += len(todo)
            self.save_cache()

        out: List[ScoreRecord] = []
        for original, key in zip(smiles_list, keys):
            if key is None:
                out.append(ScoreRecord(smiles=original, error="invalid_smiles"))
                self.n_failed += 1
                continue
            rec = ScoreRecord(**self._cache[key])
            rec.smiles = original
            if rec.error:
                self.n_failed += 1
            out.append(rec)
        self.n_cached += len(smiles_list) - len(todo)
        return out

    def __call__(self, smiles_list: Sequence[str]) -> List[float]:
        return [r.reward for r in self.score(smiles_list)]

    def stats(self) -> dict:
        return {"backend": self.backend, "docked": self.n_docked,
                "cache_hits": self.n_cached, "failures": self.n_failed,
                "cache_size": len(self._cache)}


class DualReceptorOracle:
    """Both receptor arms, for the induced-fit comparison.

    score_both() gives per-receptor values and their difference. A molecule
    scoring well in the NP-induced pocket (4EGK) but poorly in the synthetic one
    (6LTI) is exploiting that conformation rather than binding Hsp90.
    """

    def __init__(self, np_oracle: Hsp90Oracle, synth_oracle: Hsp90Oracle,
                 mode: str = "np_only") -> None:
        if mode not in ("mean", "min", "np_only", "synth_only"):
            raise ValueError(f"unknown mode: {mode}")
        self.np_oracle, self.synth_oracle, self.mode = np_oracle, synth_oracle, mode

    def score_both(self, smiles_list: Sequence[str]) -> List[dict]:
        a, b = self.np_oracle.score(smiles_list), self.synth_oracle.score(smiles_list)
        return [{"smiles": x.smiles,
                 "reward_np_receptor": x.reward, "reward_synth_receptor": y.reward,
                 "raw_np": x.raw_score, "raw_synth": y.raw_score,
                 "delta": (None if x.raw_score is None or y.raw_score is None
                           else x.raw_score - y.raw_score)}
                for x, y in zip(a, b)]

    def __call__(self, smiles_list: Sequence[str]) -> List[float]:
        rows = self.score_both(smiles_list)
        if self.mode == "mean":
            return [(r["reward_np_receptor"] + r["reward_synth_receptor"]) / 2
                    for r in rows]
        if self.mode == "min":
            return [min(r["reward_np_receptor"], r["reward_synth_receptor"])
                    for r in rows]
        key = "reward_np_receptor" if self.mode == "np_only" else "reward_synth_receptor"
        return [r[key] for r in rows]


class Hsp90LigandEfficiency:
    """MolScore-compatible scoring function supplying the docking term."""

    return_metrics = ["LE", "docking_score", "heavy_atoms"]

    def __init__(self, prefix: str = "hsp90", receptor_pdbqt: str = "",
                 box_config: str = "", exhaustiveness: int = 64,
                 n_workers: int = 8, cache_path: Optional[str] = None,
                 **kwargs) -> None:
        self.prefix = prefix.strip().replace(" ", "_")
        self.oracle = Hsp90Oracle(receptor_pdbqt, box_config,
                                  exhaustiveness=exhaustiveness,
                                  n_workers=n_workers, cache_path=cache_path)

    def __call__(self, smiles: List[str], **kwargs) -> List[dict]:
        return [{"smiles": r.smiles,
                 f"{self.prefix}_LE": r.ligand_efficiency or 0.0,
                 f"{self.prefix}_docking_score": r.raw_score or 0.0,
                 f"{self.prefix}_heavy_atoms": r.heavy_atoms or 0}
                for r in self.oracle.score(smiles)]


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #

def _looks_like_root(p: Path) -> bool:
    return ((p / "receptors").is_dir() and (p / "box").is_dir()
            and any((p / "receptors").glob("*.pdbqt")))


def resolve_root(root: Optional[str] = None) -> Path:
    """Find the prep directory, failing loudly rather than silently wrongly.

    Order: the `root` argument, then HSP90_DATA_ROOT, then data/hsp90 /
    hsp90_vina / hsp90_v2 under this file's directory, its parent, and the
    working directory. A relative `root` is tried against the working directory
    first, then this file's directory.
    """
    explicit = root if root is not None else os.environ.get("HSP90_DATA_ROOT")
    if explicit:
        p = Path(explicit).expanduser()
        cands = [p] if p.is_absolute() else [Path.cwd() / p, MODULE_DIR / p]
    else:
        cands = [b / s for s in _DEFAULT_SUBDIRS for b in _SEARCH_BASES]
    for c in cands:
        if _looks_like_root(c):
            return c.resolve()
    raise FileNotFoundError(
        "Could not find the Hsp90 prep directory (needs receptors/*.pdbqt and "
        "box/).\nTried:\n" + "\n".join(f"    {c}" for c in cands) +
        "\n\nFix by one of:\n"
        "  - pass it:        make_oracle('/abs/path/to/data/hsp90', 'np')\n"
        "  - set it:         export HSP90_DATA_ROOT=/abs/path/to/data/hsp90\n"
        f"  - place it at {' / '.join(_DEFAULT_SUBDIRS)} next to {MODULE_DIR}\n"
        "  - generate it:    bash hsp90_docking_setup.sh")


def make_oracle(root: Optional[str] = None, receptor: str = "np",
                **kwargs) -> Hsp90Oracle:
    """Build an oracle from the prep directory, no paths to repeat.

        oracle = make_oracle(receptor="np", n_workers=16)   # root auto-found

    `receptor` is "np" (4EGK, radicicol-bound), "synth" (6LTI, AUY922-bound), or
    a PDB code. The cache goes in the WORKING directory, since the data
    directory may be read-only or shared while the cache is run-specific.
    """
    root_p = resolve_root(root)
    pdb = RECEPTORS.get(receptor, receptor).upper()
    rec = root_p / "receptors" / f"{pdb}.pdbqt"
    box = root_p / "box" / f"{pdb}_vina_box.txt"
    for f in (rec, box):
        if not f.is_file():
            known = sorted(p.stem for p in (root_p / "receptors").glob("*.pdbqt"))
            raise FileNotFoundError(f"{f} not found. Known receptors: {known}")
    kwargs.setdefault("cache_path", f"oracle_cache_{pdb}.json")
    return Hsp90Oracle(receptor_pdbqt=str(rec), box_config=str(box), **kwargs)


def make_dual_oracle(root: Optional[str] = None, mode: str = "np_only",
                     **kwargs) -> DualReceptorOracle:
    """Both receptor arms at once.

    DO NOT TRAIN ON 6LTI. At exhaustiveness 64 over 4 seeds, radicicol in 6LTI
    spreads 1.12 kcal/mol while every other receptor/ligand pair spreads under
    0.07: the synthetic-induced conformation gives macrocycles no deep well, so
    the search wanders. Including it in the reward would inject ~1 kcal/mol of
    noise into exactly the chemistry under study.

    Hence mode="np_only" by default: train on 4EGK, then re-score the final
    molecules on 6LTI with several seeds and report the spread. That keeps the
    induced-fit comparison an analysis of outputs, and halves the compute.
    """
    root_p = resolve_root(root)
    half = max(1, kwargs.pop("n_workers", 8) // 2)
    base_cache = kwargs.pop("cache_path", None)

    def arm(which: str) -> Hsp90Oracle:
        kw = dict(kwargs)
        if base_cache:
            # One shared path would be written by both arms, mixing receptors.
            stem = Path(base_cache)
            pdb = RECEPTORS.get(which, which).upper()
            kw["cache_path"] = str(stem.with_name(f"{stem.stem}_{pdb}{stem.suffix}"))
        return make_oracle(str(root_p), which, n_workers=half, **kw)

    return DualReceptorOracle(arm("np"), arm("synth"), mode=mode)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

def self_test(root: Optional[str] = None, backend: str = "python") -> None:
    """Dock the reference ligands from SMILES and compare with redocking.

    From-SMILES conformers differ from crystal ones, so expect a few tenths of a
    kcal/mol of difference rather than an exact match.
    """
    root_p = resolve_root(root)
    print(f"Self-test, backend={backend}, root={root_p}\n")
    for lid, rid, expect in (("RDC", "4EGK", -4.877), ("2GJ", "6LTI", -8.022)):
        sdf = root_p / "ligands" / f"{lid}_prep.sdf"
        if not sdf.is_file():
            print(f"  missing {sdf}; run hsp90_docking_setup.sh first")
            return
        smi = Chem.MolToSmiles(Chem.MolFromMolFile(str(sdf), removeHs=True))
        with make_oracle(str(root_p), rid, backend=backend, n_workers=1,
                         cache_path=None) as oracle:
            rec = oracle.score([smi])[0]
        if rec.raw_score is None:
            print(f"  {lid} in {rid}: FAILED - {rec.error or rec.failed_filter}")
            continue
        d = rec.raw_score - expect
        verdict = "OK" if abs(d) < 1.5 else "CHECK - differs from redocking"
        print(f"  {lid} in {rid}: {rec.raw_score:+.2f} kcal/mol (redock "
              f"{expect:+.2f}, delta {d:+.2f}), metric {rec.ligand_efficiency:.3f}, "
              f"reward {rec.reward:.3f}  [{verdict}]")
        print(f"      {smi}")


def diagnose(root: Optional[str] = None) -> None:
    """Environment report: interpreter, packages, executables, data root."""
    print(f"python        {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info >= (3, 13):
        print("  WARNING: RDKit has no wheel for 3.13+, so pip backtracks and "
              "installs an\n  ancient Meeko (the 'No module named rdkit.six' "
              "error). Use python=3.12.")
    for name in ("rdkit", "meeko", "vina", "numpy"):
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"{name:<13} {ver}")
            if name == "meeko":
                try:
                    if tuple(int(x) for x in ver.split(".")[:2]) < (0, 5):
                        print("  WARNING: too old; it imports the long-removed "
                              "rdkit.six and lacks mk_export.py.\n"
                              "  conda install -c conda-forge 'meeko>=0.6'")
                except ValueError:
                    pass
        except Exception as exc:
            print(f"{name:<13} IMPORT FAILED: {type(exc).__name__}: {exc}")
    print()
    for label, env in (("vina", "VINA_EXE"), ("mk_prepare_ligand.py", "MEEKO_EXE"),
                       ("mk_export.py", None)):
        found = _find_exe(None, label, env or "_UNSET_")
        print(f"{label:<22} {found or 'not found'}")
    print()
    print(f"module dir    {MODULE_DIR}")
    try:
        r = resolve_root(root)
        print(f"data root     {r}")
        for sub, pat in (("receptors", "*.pdbqt"), ("box", "*.txt"),
                         ("ligands", "*_prep.sdf")):
            names = sorted(p.name for p in (r / sub).glob(pat)) if (r / sub).is_dir() else []
            print(f"  {sub:<11} {names or 'MISSING'}")
    except FileNotFoundError as exc:
        print(f"data root     {exc}")
    print()
    print(f"size fit      b0={SIZE_FIT['b0']:.4f}  b1={SIZE_FIT['b1']:.4f}")
    print(f"normalization {NORMALIZATION}  window {_WINDOWS[NORMALIZATION]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Hsp90 docking oracle")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--where", action="store_true",
                    help="print the resolved data root and exit")
    ap.add_argument("--smiles", nargs="*")
    ap.add_argument("--root", default=None,
                    help="prep directory; auto-detected relative to this file, "
                         "or from HSP90_DATA_ROOT")
    ap.add_argument("--receptor", default="np", help="'np', 'synth', or a PDB code")
    ap.add_argument("--backend", default="python", choices=["python", "subprocess"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--exhaustiveness", type=int, default=64,
                    help="64 is validated: at 32 radicicol misses its crystal "
                         "pose in 1 of 4 seeds, at 64 in 0 of 4")
    ap.add_argument("--normalization", default=None,
                    choices=["residual", "efficiency", "raw"])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.where:
        print(f"module:    {MODULE_DIR}")
        print(f"data root: {resolve_root(args.root)}")
    elif args.diagnose:
        diagnose(args.root)
    elif args.self_test:
        self_test(args.root, args.backend)
    elif args.smiles:
        with make_oracle(args.root, args.receptor, backend=args.backend,
                         n_workers=args.workers, verbose=not args.quiet,
                         exhaustiveness=args.exhaustiveness,
                         normalization=args.normalization) as oracle:
            for rec in oracle.score(args.smiles):
                print(f"{rec.reward:.3f}  raw={rec.raw_score}  "
                      f"metric={rec.ligand_efficiency}  "
                      f"{rec.failed_filter or rec.error or ''}  {rec.smiles}")
            print(oracle.stats())
    else:
        ap.print_help()