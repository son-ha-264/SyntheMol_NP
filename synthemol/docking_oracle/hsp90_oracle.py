#!/usr/bin/env python3
"""
Hsp90alpha NTD docking oracle for generative-model reinforcement learning.

Wraps the validated setup from hsp90_docking_setup.sh: AutoDock Vina 1.2 with
Vinardo scoring and the chemotype-neutral box derived from the union of the
radicicol (4EGK) and NVP-AUY922 (6LTI) ligand envelopes.

Two backends:

  backend="python"      (default) Vina's Python API + Meeko in memory.
                        Affinity maps are computed ONCE per worker process and
                        reused for every ligand; the command line recomputes
                        them per run, so this is several times faster. No temp
                        files, no process spawn per molecule.

  backend="subprocess"  the vina / mk_prepare_ligand.py executables. Slower,
                        but a useful fallback if the Python bindings are a
                        different version from the CLI tools.

Design decisions, and why:

  * Reward is LIGAND EFFICIENCY (|score| / heavy atoms), not the raw score.
    Raw scores grow with molecular size and a natural-product-biased prior
    generates larger molecules, so a raw-score reward would reward size. On the
    reference ligands the raw gap between chemotypes is 40%, the LE gap 19%.

  * Property limits are HARD FILTERS returning reward 0, not weighted terms.
    A weighted term can be traded off against the score during optimisation.

  * Caching is by canonical SMILES. RL revisits structures constantly.

  * Nothing raises. Bad SMILES, embedding failure, Vina error or timeout all
    return 0.0 with the reason recorded. A reward that throws kills a run.

Calibration (exhaustiveness 32, seed 42, validated setup):
    radicicol  in 4EGK: -4.78 kcal/mol, 25 heavy atoms, LE 0.191, redock 0.35 A
    NVP-AUY922 in 6LTI: -8.01 kcal/mol, 34 heavy atoms, LE 0.236, redock 1.78 A
Run `python hsp90_oracle.py --self-test` before training.

Requires: rdkit, meeko, vina (pip install vina). The subprocess backend also
needs vina and mk_prepare_ligand.py on PATH.
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


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class Filters:
    """Hard gates. Failing any of these scores 0 and the molecule is not docked.

    Permissive on size by design, so natural-product-like chemistry is not
    excluded before it is measured, while still blocking the degenerate
    large/greasy solutions RL finds when left unconstrained. Report the values.
    """
    max_heavy_atoms: int = 60
    min_heavy_atoms: int = 12
    max_mol_weight: float = 1000.0
    max_rotatable_bonds: int = 15
    #max_logp: float = 7.0

    def check(self, mol: Chem.Mol) -> Optional[str]:
        if mol.GetNumHeavyAtoms() > self.max_heavy_atoms:
            return "max_heavy_atoms"
        if mol.GetNumHeavyAtoms() < self.min_heavy_atoms:
            return "min_heavy_atoms"
        if Descriptors.MolWt(mol) > self.max_mol_weight:
            return "max_mol_weight"
        if rdMolDescriptors.CalcNumRotatableBonds(mol) > self.max_rotatable_bonds:
            return "max_rotatable_bonds"
        #if Descriptors.MolLogP(mol) > self.max_logp:
        #    return "max_logp"
        return None


@dataclass
class ScoreRecord:
    smiles: str
    reward: float = 0.0
    raw_score: Optional[float] = None
    heavy_atoms: Optional[int] = None
    ligand_efficiency: Optional[float] = None
    failed_filter: Optional[str] = None
    error: Optional[str] = None


def parse_box_config(path: str) -> Tuple[List[float], List[float]]:
    """Read Meeko's Vina box file into (center, box_size)."""
    vals: Dict[str, float] = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"\s*(\w+)\s*=\s*([-\d.]+)", line)
        if m:
            vals[m.group(1)] = float(m.group(2))
    try:
        return ([vals["center_x"], vals["center_y"], vals["center_z"]],
                [vals["size_x"], vals["size_y"], vals["size_z"]])
    except KeyError as exc:
        raise ValueError(f"{path} is missing {exc}") from None


def _find_exe(explicit: Optional[str], name: str, env_var: str) -> Optional[str]:
    """Locate an executable without requiring it on PATH.

    Order: explicit argument, then the environment variable, then PATH. This
    matters because the Vina Python bindings and the Vina executable are
    separate installs, so a given environment often has one and not the other,
    and the binary may live in a different conda environment entirely.
    """
    for cand in (explicit, os.environ.get(env_var)):
        if cand:
            p = Path(cand).expanduser()
            if p.is_file() and os.access(p, os.X_OK):
                return str(p.resolve())
            found = shutil.which(cand)
            if found:
                return found
    return shutil.which(name)


# --------------------------------------------------------------------------- #
# Shared molecule handling
# --------------------------------------------------------------------------- #

def embed_3d(smiles: str) -> Optional[Chem.Mol]:
    """SMILES -> one 3D conformer with explicit hydrogens.

    Vina searches rotatable bonds itself, and Meeko opens macrocycles so their
    ring conformation is searched too, so a single ETKDG conformer suffices as
    a starting point. Substitute Gypsum-DL here if you want stereoisomer and
    protonation-state enumeration, scoring each state and taking the best.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    params.useSmallRingTorsions = True
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True          # macrocycles often need this
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass                                    # unoptimised still docks
    return mol


def mol_to_pdbqt_string(mol: Chem.Mol) -> Tuple[Optional[str], Optional[str]]:
    """Meeko, in memory. Returns (pdbqt_string, error)."""
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
    except ImportError as exc:
        return None, f"meeko import failed: {exc}"
    try:
        mk_prep = MoleculePreparation()
        setups = mk_prep(mol)                   # callable, not .prepare()
        if not setups:
            return None, "meeko returned no setups"
        res = PDBQTWriterLegacy.write_string(setups[0])
        # Version-tolerant: some releases return (string, ok, error_message).
        if isinstance(res, tuple):
            pdbqt = res[0]
            ok = res[1] if len(res) > 1 else True
            if not ok:
                msg = res[2] if len(res) > 2 else "unknown"
                return None, f"meeko write failed: {str(msg)[:100]}"
        else:
            pdbqt = res
        if not pdbqt or "ATOM" not in pdbqt:
            return None, "meeko produced an empty pdbqt"
        return pdbqt, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"[:140]


# --------------------------------------------------------------------------- #
# Python backend: one Vina instance per worker process, maps computed once
# --------------------------------------------------------------------------- #

_W: dict = {}          # per-process state; Vina objects cannot be pickled


def _init_worker(receptor_pdbqt: str, center: List[float], box_size: List[float],
                 scoring: str, seed: int, cpu: int) -> None:
    """Build the Vina instance and compute the affinity maps ONCE per process.

    With no ligand loaded, compute_vina_maps covers all 22 Vina atom types, so
    the same maps serve every ligand this worker will ever dock. This is the
    whole performance argument for the Python API over the command line.
    """
    from vina import Vina
    try:
        v = Vina(sf_name=scoring, cpu=cpu, seed=seed, verbosity=0)
    except TypeError:                            # older signature without seed
        v = Vina(sf_name=scoring, cpu=cpu, verbosity=0)
        try:
            v.set_seed(seed)
        except Exception:
            pass
    v.set_receptor(receptor_pdbqt)
    v.compute_vina_maps(center=center, box_size=box_size)
    _W["vina"] = v


def _dock_python(smiles: str, filters_d: dict, exhaustiveness: int,
                 n_poses: int, le_min: float, le_max: float) -> dict:
    rec = ScoreRecord(smiles=smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        rec.error = "invalid_smiles"
        return asdict(rec)

    rec.heavy_atoms = mol.GetNumHeavyAtoms()
    failed = Filters(**filters_d).check(mol)
    if failed:
        rec.failed_filter = failed
        return asdict(rec)

    mol3d = embed_3d(smiles)
    if mol3d is None:
        rec.error = "embedding_failed"
        return asdict(rec)

    pdbqt, err = mol_to_pdbqt_string(mol3d)
    if pdbqt is None:
        rec.error = err
        return asdict(rec)

    try:
        v = _W["vina"]
        v.set_ligand_from_string(pdbqt)
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        energies = v.energies(n_poses=1)
        score = float(energies[0][0])            # best pose, total energy
    except Exception as exc:
        rec.error = f"vina: {type(exc).__name__}: {exc}"[:140]
        return asdict(rec)

    rec.raw_score = score
    rec.ligand_efficiency = -score / rec.heavy_atoms
    rec.reward = max(0.0, min(1.0, (rec.ligand_efficiency - le_min) / (le_max - le_min)))
    return asdict(rec)


# --------------------------------------------------------------------------- #
# Subprocess backend (fallback)
# --------------------------------------------------------------------------- #

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
                     timeout_s: int, le_min: float, le_max: float,
                     vina_exe: str = "vina",
                     meeko_exe: str = "mk_prepare_ligand.py") -> dict:
    rec = ScoreRecord(smiles=smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        rec.error = "invalid_smiles"
        return asdict(rec)
    rec.heavy_atoms = mol.GetNumHeavyAtoms()
    failed = Filters(**filters_d).check(mol)
    if failed:
        rec.failed_filter = failed
        return asdict(rec)
    mol3d = embed_3d(smiles)
    if mol3d is None:
        rec.error = "embedding_failed"
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
        rec.raw_score = score
        rec.ligand_efficiency = -score / rec.heavy_atoms
        rec.reward = max(0.0, min(1.0, (rec.ligand_efficiency - le_min) / (le_max - le_min)))
        return asdict(rec)
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

        oracle = Hsp90Oracle(
            receptor_pdbqt="hsp90_v2/receptors/4EGK.pdbqt",
            box_config="hsp90_v2/box/4EGK_vina_box.txt",
            n_workers=16,
            cache_path="cache_4EGK.json",
        )
        rewards = oracle(smiles_batch)
        records = oracle.score(smiles_batch)   # raw score, LE, failure reason

    The LE window maps ligand efficiency to [0, 1]. Defaults bracket the
    reference ligands (radicicol 0.191, NVP-AUY922 0.236), putting them at
    about 0.36 and 0.54, with headroom above the best known ligand so the
    reward does not saturate early in training.
    """

    def __init__(
        self,
        receptor_pdbqt: str,
        box_config: str,
        *,
        backend: str = "python",
        scoring: str = "vinardo",
        exhaustiveness: int = 32,
        n_poses: int = 9,
        seed: int = 42,
        timeout_s: int = 600,
        filters: Optional[Filters] = None,
        le_min: float = 0.065,
        le_max: float = 0.415,
        n_workers: int = 8,
        cache_path: Optional[str] = None,
        vina_exe: Optional[str] = None,
        meeko_exe: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose
        self.receptor = str(Path(receptor_pdbqt).resolve())
        self.box_config = str(Path(box_config).resolve())
        for p in (self.receptor, self.box_config):
            if not Path(p).is_file():
                raise FileNotFoundError(p)
        self.center, self.box_size = parse_box_config(self.box_config)

        if backend not in ("python", "subprocess"):
            raise ValueError("backend must be 'python' or 'subprocess'")

        import_error: Optional[str] = None
        if backend == "python":
            try:
                import vina  # noqa: F401
            except Exception as exc:
                # Not just ImportError: a vina built against the wrong Boost or
                # a partial wheel raises OSError or AttributeError on import.
                import_error = f"{type(exc).__name__}: {exc}"
                print(f"[Hsp90Oracle] vina Python API unavailable ({import_error}); "
                      f"trying the subprocess backend")
                backend = "subprocess"

        # Explicit paths or env vars (VINA_EXE / MEEKO_EXE) are accepted, so the
        # executables need not be on PATH or in this conda environment.
        self.vina_exe = _find_exe(vina_exe, "vina", "VINA_EXE")
        self.meeko_exe = _find_exe(meeko_exe, "mk_prepare_ligand.py", "MEEKO_EXE")

        if backend == "subprocess":
            missing = [n for n, p in (("vina", self.vina_exe),
                                      ("mk_prepare_ligand.py", self.meeko_exe))
                       if p is None]
            if missing:
                lines = ["No usable Vina backend."]
                if import_error:
                    lines.append(f"  Python API: import vina failed - {import_error}")
                else:
                    lines.append("  Python API: not attempted (backend='subprocess')")
                lines += [
                    f"  Command line: {', '.join(missing)} not on PATH",
                    f"  interpreter:  {sys.executable}",
                    f"  PATH head:    {os.environ.get('PATH','').split(os.pathsep)[0]}",
                    "",
                    "Most common cause: this interpreter is not the conda env that has",
                    "vina. Check with:  which python; which vina",
                    "",
                    "Fixes:",
                    "  conda activate dock                      # the env from the prep step",
                    "  conda install -c conda-forge vina        # CLI + library",
                    f"  {sys.executable} -m pip install vina     # Python API, THIS interpreter",
                    "",
                    "The bindings and the executable are SEPARATE installs: pip install",
                    "vina gives no executable, conda install vina gives no bindings.",
                    "For the bindings, install the build deps first:",
                    "  conda install -c conda-forge numpy swig boost-cpp libboost",
                    f"  {sys.executable} -m pip install vina",
                    "",
                    "Or point at a binary in another environment, no PATH needed:",
                    "  Hsp90Oracle(..., vina_exe='/home/you/miniconda3/envs/dock/bin/vina')",
                    "  export VINA_EXE=/home/you/miniconda3/envs/dock/bin/vina",
                    "",
                    "Run `python hsp90_oracle.py --diagnose` for a fuller report.",
                ]
                raise RuntimeError("\n".join(lines))
        self.backend = backend

        self.scoring = scoring
        self.exhaustiveness = exhaustiveness
        self.n_poses = n_poses
        self.seed = seed
        self.timeout_s = timeout_s
        self.filters = filters or Filters()
        self.le_min, self.le_max = le_min, le_max
        self.n_workers = max(1, n_workers)

        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: Dict[str, dict] = {}
        if self.cache_path and self.cache_path.is_file():
            try:
                self._cache = json.loads(self.cache_path.read_text())
            except Exception:
                self._cache = {}

        self.n_docked = self.n_cached = self.n_failed = 0

        # The worker pool is created once and REUSED across score() calls. Each
        # python-backend worker computes the affinity maps on startup, so a pool
        # per call would recompute them every training step: with 16 workers and
        # 500 steps that is 8000 map computations instead of 16.
        self._pool: Optional[ProcessPoolExecutor] = None

    # -- plumbing --------------------------------------------------------- #

    def save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.write_text(json.dumps(self._cache))

    @staticmethod
    def _canonical(smiles: str) -> Optional[str]:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol is not None else None

    def _pool_kwargs(self, cpu: int = 1) -> dict:
        if self.backend == "python":
            return {"initializer": _init_worker,
                    "initargs": (self.receptor, self.center, self.box_size,
                                 self.scoring, self.seed, cpu)}
        return {}

    def _dock_args(self, smi: str) -> tuple:
        fd = asdict(self.filters)
        if self.backend == "python":
            return (smi, fd, self.exhaustiveness, self.n_poses,
                    self.le_min, self.le_max)
        return (smi, fd, self.receptor, self.box_config, self.scoring,
                self.exhaustiveness, self.n_poses, self.seed, self.timeout_s,
                self.le_min, self.le_max, self.vina_exe, self.meeko_exe)

    @property
    def _dock_fn(self):
        return _dock_python if self.backend == "python" else _dock_subprocess

    def _get_pool(self) -> ProcessPoolExecutor:
        """Lazily create the pool and keep it for the life of the oracle."""
        if self._pool is None:
            if self.verbose:
                print(f"[Hsp90Oracle] starting {self.n_workers} persistent "
                      f"worker(s); affinity maps computed once per worker",
                      flush=True)
            self._pool = ProcessPoolExecutor(max_workers=self.n_workers,
                                             **self._pool_kwargs(cpu=1))
        return self._pool

    def close(self) -> None:
        """Shut the workers down. Call at the end of training, or use the oracle
        as a context manager."""
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
            # Never spawn more workers than there are molecules. Each python-backend
            # worker computes the full affinity maps on startup, so an oversized pool
            # pays that cost once per worker and then leaves most of them idle.
            n_eff = min(self.n_workers, len(todo))
            t0 = time.time()
            if self.verbose:
                print(f"[Hsp90Oracle] docking {len(todo)} molecule(s), "
                      f"{n_eff} worker(s), exhaustiveness {self.exhaustiveness}, "
                      f"backend {self.backend}", flush=True)

            if n_eff <= 1:
                # One molecule or one worker: run in-process and give Vina every
                # core (cpu=0) instead of pinning it to one.
                if self.backend == "python":
                    if "vina" not in _W:
                        if self.verbose:
                            print("[Hsp90Oracle] computing affinity maps "
                                  "(once; reused for all ligands)...", flush=True)
                        _init_worker(self.receptor, self.center, self.box_size,
                                     self.scoring, self.seed, 0)
                        if self.verbose:
                            print(f"[Hsp90Oracle] maps ready in "
                                  f"{time.time() - t0:.1f}s", flush=True)
                for i, smi in enumerate(todo, 1):
                    ts = time.time()
                    self._cache[smi] = self._dock_fn(*self._dock_args(smi))
                    if self.verbose:
                        r = self._cache[smi]
                        note = r.get("error") or r.get("failed_filter") or \
                            f"{r.get('raw_score')} kcal/mol"
                        print(f"[Hsp90Oracle]   {i}/{len(todo)} "
                              f"{time.time() - ts:.1f}s  {note}", flush=True)
            else:
                done = 0
                pool = self._get_pool()          # persistent, not per-batch
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

    `score_both` gives the per-receptor values and their difference. A molecule
    scoring well in the natural-product-induced pocket (4EGK) but poorly in the
    synthetic one (6LTI) is exploiting that conformation rather than binding
    Hsp90, and the delta column surfaces that at scale.
    """

    def __init__(self, np_oracle: Hsp90Oracle, synth_oracle: Hsp90Oracle,
                 mode: str = "mean") -> None:
        if mode not in ("mean", "min", "np_only", "synth_only"):
            raise ValueError(f"unknown mode: {mode}")
        self.np_oracle, self.synth_oracle, self.mode = np_oracle, synth_oracle, mode

    def score_both(self, smiles_list: Sequence[str]) -> List[dict]:
        a = self.np_oracle.score(smiles_list)
        b = self.synth_oracle.score(smiles_list)
        return [{"smiles": x.smiles,
                 "reward_np_receptor": x.reward,
                 "reward_synth_receptor": y.reward,
                 "raw_np": x.raw_score, "raw_synth": y.raw_score,
                 "delta": (None if x.raw_score is None or y.raw_score is None
                           else x.raw_score - y.raw_score)}
                for x, y in zip(a, b)]

    def __call__(self, smiles_list: Sequence[str]) -> List[float]:
        rows = self.score_both(smiles_list)
        if self.mode == "mean":
            return [(r["reward_np_receptor"] + r["reward_synth_receptor"]) / 2 for r in rows]
        if self.mode == "min":
            return [min(r["reward_np_receptor"], r["reward_synth_receptor"]) for r in rows]
        if self.mode == "np_only":
            return [r["reward_np_receptor"] for r in rows]
        return [r["reward_synth_receptor"] for r in rows]


class Hsp90LigandEfficiency:
    """MolScore-compatible scoring function supplying the docking term."""

    return_metrics = ["LE", "docking_score", "heavy_atoms"]

    def __init__(self, prefix: str = "hsp90", receptor_pdbqt: str = "",
                 box_config: str = "", exhaustiveness: int = 32,
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
# Factories: the import-and-go entry points
# --------------------------------------------------------------------------- #

__all__ = ["Hsp90Oracle", "DualReceptorOracle", "Hsp90LigandEfficiency",
           "Filters", "ScoreRecord", "make_oracle", "make_dual_oracle",
           "resolve_root", "MODULE_DIR", "self_test", "benchmark", "diagnose"]

# Which receptor is which, so callers say "np" / "synth" rather than PDB codes.
RECEPTORS = {"np": "4EGK", "synth": "6LTI"}

# Anchor data lookup to this file, not the working directory. A training script
# launched from anywhere - a cluster scheduler, a notebook, another package -
# must find the receptors without the caller knowing where they live.
MODULE_DIR = Path(__file__).resolve().parent

# Searched in order when no root is given, relative to MODULE_DIR, then its
# parent (module in src/, data beside it), then the working directory.
_DEFAULT_SUBDIRS = ("data/hsp90", "hsp90_vina", "hsp90_v2")
_SEARCH_BASES = (MODULE_DIR, MODULE_DIR.parent, Path.cwd())


def _looks_like_root(p: Path) -> bool:
    """A prep root has receptors/ and box/ with at least one receptor in it."""
    return ((p / "receptors").is_dir() and (p / "box").is_dir()
            and any((p / "receptors").glob("*.pdbqt")))


def resolve_root(root: Optional[str] = None) -> Path:
    """Find the prep directory, and fail loudly rather than silently wrongly.

    Resolution order:
      1. the `root` argument, if given
      2. the HSP90_DATA_ROOT environment variable
      3. data/hsp90, hsp90_vina or hsp90_v2 under this file's directory,
         then its parent, then the working directory

    A relative `root` is tried against the working directory first (so an
    explicit relative path behaves as typed) and then against this file's
    directory. Raises FileNotFoundError listing every path tried.
    """
    tried: List[Path] = []

    candidates: List[Path] = []
    explicit = root if root is not None else os.environ.get("HSP90_DATA_ROOT")
    if explicit:
        p = Path(explicit).expanduser()
        candidates = [p] if p.is_absolute() else [Path.cwd() / p, MODULE_DIR / p]
    else:
        candidates = [base / sub for sub in _DEFAULT_SUBDIRS for base in _SEARCH_BASES]

    for cand in candidates:
        tried.append(cand)
        if _looks_like_root(cand):
            return cand.resolve()

    listing = "\n".join(f"    {t}" for t in tried)
    raise FileNotFoundError(
        "Could not find the Hsp90 prep directory (a folder containing "
        "receptors/*.pdbqt and box/).\nTried:\n" + listing +
        "\n\nFix by one of:\n"
        "  - pass it explicitly:   make_oracle('/abs/path/to/data/hsp90', 'np')\n"
        "  - set the environment:  export HSP90_DATA_ROOT=/abs/path/to/data/hsp90\n"
        "  - place the prep output at one of "
        f"{', '.join(_DEFAULT_SUBDIRS)} next to {MODULE_DIR}\n"
        "  - generate it:          bash hsp90_docking_setup.sh"
    )


def make_oracle(root: Optional[str] = None, receptor: str = "np",
                **kwargs) -> Hsp90Oracle:
    """Build an oracle from the prep directory, no paths to repeat.

        from hsp90_oracle import make_oracle
        oracle = make_oracle(receptor="np", n_workers=16)   # root auto-found
        oracle = make_oracle("data/hsp90", "np")            # or explicit
        rewards = oracle(smiles_batch)

    `root` defaults to whatever `resolve_root` finds relative to this file.
    `receptor` is "np" (4EGK, radicicol-bound) or "synth" (6LTI, AUY922-bound),
    or a PDB code directly. Any Hsp90Oracle keyword passes straight through.

    The cache goes in the WORKING directory, not the data directory, since the
    data directory may be read-only or shared and the cache is run-specific.
    """
    root_p = resolve_root(root)
    pdb = RECEPTORS.get(receptor, receptor).upper()
    rec = root_p / "receptors" / f"{pdb}.pdbqt"
    box = root_p / "box" / f"{pdb}_vina_box.txt"
    for f in (rec, box):
        if not f.is_file():
            raise FileNotFoundError(
                f"{f} not found. Known receptors in {root_p / 'receptors'}: "
                f"{sorted(p.stem for p in (root_p / 'receptors').glob('*.pdbqt'))}")
    kwargs.setdefault("cache_path", f"oracle_cache_{pdb}.json")
    return Hsp90Oracle(receptor_pdbqt=str(rec), box_config=str(box), **kwargs)


def make_dual_oracle(root: Optional[str] = None, mode: str = "mean",
                     **kwargs) -> DualReceptorOracle:
    """Both receptor arms at once.

        oracle = make_dual_oracle(mode="mean", n_workers=16)
        rewards = oracle(smiles_batch)              # one reward per molecule
        rows    = oracle.score_both(smiles_batch)   # per-receptor + delta

    `mode`: "mean" trains on the average of both conformations; "min" requires
    a molecule to work in both, which is the stricter and more defensible
    reward; "np_only" / "synth_only" train on a single arm.
    """
    root_p = resolve_root(root)                  # resolve once, share the result
    half = max(1, kwargs.pop("n_workers", 8) // 2)
    # A single cache_path would be written by both arms, mixing receptors into
    # one file. Suffix it per receptor instead.
    base_cache = kwargs.pop("cache_path", None)

    def arm(which: str) -> Hsp90Oracle:
        kw = dict(kwargs)
        if base_cache:
            stem = Path(base_cache)
            kw["cache_path"] = str(stem.with_name(
                f"{stem.stem}_{RECEPTORS.get(which, which).upper()}{stem.suffix}"))
        return make_oracle(str(root_p), which, n_workers=half, **kw)

    return DualReceptorOracle(arm("np"), arm("synth"), mode=mode)


# --------------------------------------------------------------------------- #
# Self-test and benchmark
# --------------------------------------------------------------------------- #

def self_test(root: Optional[str] = None, backend: str = "python") -> None:
    """Dock the reference ligands from SMILES and compare with the crystal
    redocking numbers. Run before training and after any environment change."""
    root_p = resolve_root(root)
    print(f"Self-test, backend={backend}, root={root_p}")
    print("From-SMILES conformers differ from crystal ones, so expect a few")
    print("tenths of a kcal/mol of difference, not an exact match.\n")
    for lid, rid, expect in (("RDC", "4EGK", -4.78), ("2GJ", "6LTI", -8.01)):
        sdf = root_p / "ligands" / f"{lid}_prep.sdf"
        if not sdf.is_file():
            print(f"  missing {sdf}; run hsp90_docking_setup.sh first")
            return
        smi = Chem.MolToSmiles(Chem.MolFromMolFile(str(sdf), removeHs=True))
        oracle = make_oracle(str(root_p), rid, backend=backend, n_workers=1,
                             cache_path=None)
        rec = oracle.score([smi])[0]
        if rec.raw_score is None:
            print(f"  {lid} in {rid}: FAILED - {rec.error or rec.failed_filter}")
            continue
        d = rec.raw_score - expect
        verdict = "OK" if abs(d) < 1.5 else "CHECK - differs from crystal redocking"
        print(f"  {lid} in {rid}: {rec.raw_score:+.2f} kcal/mol "
              f"(redock {expect:+.2f}, delta {d:+.2f}), LE {rec.ligand_efficiency:.3f}, "
              f"reward {rec.reward:.3f}  [{verdict}]")
        print(f"      {smi}")


def diagnose(root: Optional[str] = None) -> None:
    """Report which backends are usable and why, before anything else fails."""
    print("=== interpreter ===")
    print(f"  python      {sys.executable}")
    print(f"  version     {sys.version.split()[0]}")
    print(f"  cwd         {Path.cwd()}")
    print(f"  module dir  {MODULE_DIR}")
    print(f"  conda env   {os.environ.get('CONDA_DEFAULT_ENV', '(none)')}")

    print("\n=== Python packages ===")
    for name in ("rdkit", "meeko", "vina"):
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "(no __version__)")
            print(f"  {name:8s} OK    {ver}  {getattr(mod, '__file__', '')}")
        except Exception as exc:
            print(f"  {name:8s} FAIL  {type(exc).__name__}: {exc}")

    print("\n=== executables on PATH ===")
    for tool in ("vina", "mk_prepare_ligand.py", "mk_prepare_receptor.py",
                 "mk_export.py", "pymol", "pdb2pqr30", "obabel"):
        where = shutil.which(tool)
        print(f"  {tool:22s} {where or 'NOT FOUND'}")

    print("\n=== data ===")
    try:
        root_p = resolve_root(root)
        print(f"  root  {root_p}")
        for sub, pat in (("receptors", "*.pdbqt"), ("box", "*_vina_box.txt"),
                         ("ligands", "*_prep.sdf")):
            found = sorted(p.name for p in (root_p / sub).glob(pat)) \
                if (root_p / sub).is_dir() else []
            print(f"  {sub:10s} {found or 'MISSING'}")
    except FileNotFoundError as exc:
        print(f"  {exc}")

    print("\n=== verdict ===")
    py_ok = False
    try:
        import vina  # noqa: F401
        py_ok = True
    except Exception:
        pass
    cli_ok = all(shutil.which(t) for t in ("vina", "mk_prepare_ligand.py"))
    if py_ok:
        print("  backend='python' available (preferred: maps computed once per worker)")
    elif cli_ok:
        print("  backend='python' unavailable; backend='subprocess' will be used")
    else:
        print("  NO usable backend - install vina before training")


def benchmark(root: Optional[str] = None, n: int = 8, workers: int = 4) -> None:
    """Time both backends on the same molecules, so the throughput claim is
    measured on your hardware rather than assumed."""
    import time
    smis = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccc2c(c1)[nH]c1ccccc12",
            "CN1CCN(CC1)c1ccc(cc1)C(=O)Nc1ccccc1", "O=C(Nc1ccccc1)c1ccc(O)cc1",
            "CCOc1ccc(cc1)C(=O)NCc1ccncc1", "Oc1ccc(cc1)C1=NOC(=C1)c1ccccc1",
            "CC(C)c1cc(O)c(O)cc1C(=O)NCC1CCOCC1",
            "COc1cc(cc(OC)c1OC)C(=O)Nc1ccc(F)cc1"][:n]
    root_p = resolve_root(root)
    for backend in ("python", "subprocess"):
        try:
            o = make_oracle(str(root_p), "np", backend=backend,
                            n_workers=workers, cache_path=None)
            t0 = time.time()
            rewards = o(smis)
            dt = time.time() - t0
            ok = sum(1 for r in rewards if r > 0)
            print(f"  {backend:12s} {dt:6.1f} s for {len(smis)} mols "
                  f"({dt/len(smis):5.1f} s/mol, {ok} scored)")
        except Exception as exc:
            print(f"  {backend:12s} unavailable: {exc}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Hsp90 docking oracle")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--benchmark", action="store_true",
                    help="time the python and subprocess backends")
    ap.add_argument("--root", default=None,
                    help="prep directory; auto-detected relative to this file "
                         "or from HSP90_DATA_ROOT if omitted")
    ap.add_argument("--backend", default="python", choices=["python", "subprocess"])
    ap.add_argument("--smiles", nargs="*")
    ap.add_argument("--receptor", default="np",
                    help="'np' (4EGK), 'synth' (6LTI), or a PDB code")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--exhaustiveness", type=int, default=32,
                    help="32 is the validated setting; 8 is much faster for a "
                         "quick check but samples macrocycles poorly")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-molecule progress output")
    ap.add_argument("--where", action="store_true",
                    help="print the resolved data root and exit")
    ap.add_argument("--diagnose", action="store_true",
                    help="report which backends and data are usable")
    args = ap.parse_args()

    if args.diagnose:
        diagnose(args.root)
    elif args.where:
        print(f"module:    {MODULE_DIR}")
        print(f"data root: {resolve_root(args.root)}")
    elif args.self_test:
        self_test(args.root, args.backend)
    elif args.benchmark:
        benchmark(args.root, workers=args.workers)
    elif args.smiles:
        oracle = make_oracle(args.root, args.receptor, backend=args.backend,
                             n_workers=args.workers,
                             exhaustiveness=args.exhaustiveness,
                             verbose=not args.quiet)
        for rec in oracle.score(args.smiles):
            print(f"{rec.reward:.3f}  raw={rec.raw_score}  LE={rec.ligand_efficiency}  "
                  f"{rec.failed_filter or rec.error or ''}  {rec.smiles}")
        print(oracle.stats())
    else:
        ap.print_help()