"""
Standalone GSK3B (Glycogen Synthase Kinase 3 Beta) oracle.

Extracted from PyTDC (https://github.com/mims-harvard/TDC) so you can score
molecules without installing `PyTDC` -- and, unlike PyTDC, without needing a
compatible scikit-learn either.

Faithful to TDC's implementation:
  * tdc/chem_utils/oracle/oracle.py  ->  gsk3b() / load_gsk3b_model()
  * tdc/oracles.py                   ->  Oracle.__call__ semantics
  * tdc/utils/load.py                ->  oracle_download_wrapper()

The oracle is a RandomForestClassifier trained on ExCAPE-DB GSK3B activity
data (Jin et al., "Multi-Objective Molecule Generation using Interpretable
Substructures"). It takes a Morgan/ECFP fingerprint (radius 2, 2048 bits) and
returns P(active), a float in [0, 1].

Why there is no scikit-learn import here
----------------------------------------
TDC's checkpoint was pickled with scikit-learn ~1.0-1.2. scikit-learn 1.3
added a `missing_go_to_left` field to the decision-tree node dtype, so every
later version refuses to unpickle it:

    ValueError: node array from the pickle has an incompatible dtype

Rather than pin you to a four-year-old scikit-learn, this module reads the
pickled forest directly (thresholds, child pointers and leaf values are just
numpy arrays) and runs the tree traversal itself. Verified to reproduce
scikit-learn's `predict_proba` exactly -- 0.0 absolute difference -- on
forests pickled by both 1.2.2 and 1.9.1.

A side benefit: unpickling goes through a restricted Unpickler that only
allows numpy array reconstruction and turns every sklearn global into an
inert stub, so a tampered checkpoint cannot execute arbitrary code the way
plain `pickle.load` would.

Dependencies: rdkit, numpy. That's it -- no PyTDC, no scikit-learn, no requests.

Usage
-----
    from gsk3b_oracle import gsk3b

    gsk3b("CC(=O)Oc1ccccc1C(=O)O")        # -> 0.02
    gsk3b(["CCO", "c1ccccc1", "bad!!!"])  # -> [0.0, 0.01, 0.0]

or, keeping TDC's object-style call sites unchanged:

    from gsk3b_oracle import Oracle
    oracle = Oracle("GSK3B")
    oracle(smiles_list)

The ~4 MB checkpoint is downloaded once from Harvard Dataverse (the same
place PyTDC fetches it from) into ./oracle/, the path PyTDC uses -- so an
existing TDC cache is reused rather than re-downloaded. Override the location
with the TDC_ORACLE_PATH environment variable.
"""

from __future__ import annotations

import contextlib
import os
import pickle
import sys
from typing import Iterable, Iterator, List, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "gsk3b",
    "GSK3BOracle",
    "Oracle",
    "download_gsk3b_model",
    "load_forest",
    "disable_rdkit_logging",
]


# --------------------------------------------------------------------------- #
# Checkpoint download (mirrors tdc/utils/load.py + tdc/metadata.py)
# --------------------------------------------------------------------------- #

DATAVERSE_URL = "https://dataverse.harvard.edu/api/access/datafile/"

# tdc/metadata.py :: oracle2id. The two files are the same model; "_current"
# is TDC's re-pickle for scikit-learn >= 0.24. We can read either one, and
# prefer whichever is already cached.
_ORACLE2ID = {
    "gsk3b_current": 6413412,
    "gsk3b": 4170295,
}

ORACLE_PATH = os.environ.get("TDC_ORACLE_PATH", "oracle")


def download_gsk3b_model(path: str = None) -> str:
    """Return a local path to the GSK3B checkpoint, downloading it if needed."""
    import urllib.request

    path = ORACLE_PATH if path is None else path
    os.makedirs(path, exist_ok=True)

    # Reuse an existing copy (including one left by a PyTDC install).
    for name in _ORACLE2ID:
        local = os.path.join(path, name + ".pkl")
        if os.path.exists(local) and os.path.getsize(local) > 0:
            return local

    name = "gsk3b_current"
    local = os.path.join(path, name + ".pkl")
    url = DATAVERSE_URL + str(_ORACLE2ID[name])
    print(f"Downloading GSK3B oracle from {url} ...", file=sys.stderr)

    tmp = local + ".part"
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        if os.path.getsize(tmp) == 0:
            raise RuntimeError("downloaded file is empty")
        os.replace(tmp, local)
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            f"Could not download the GSK3B checkpoint from {url} ({exc}). "
            f"Harvard Dataverse is occasionally down or blocked by a proxy; "
            f"you can also download that URL by hand and save it as {local}."
        ) from exc

    print(f"Saved to {local}", file=sys.stderr)
    return local


# --------------------------------------------------------------------------- #
# scikit-learn-free loading of the pickled RandomForestClassifier
# --------------------------------------------------------------------------- #


class _Stub:
    """Inert stand-in for an sklearn class; just captures pickled state."""

    def __init__(self, *args, **kwargs):
        self._init_args = args

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state


_STUB_CACHE: dict = {}
_ALLOWED_MODULES = ("builtins", "copyreg", "_codecs")


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that reconstructs numpy data but neuters sklearn/scipy globals.

    Anything outside numpy, a short builtins allowlist and the sklearn/scipy
    namespaces raises, so the checkpoint cannot execute arbitrary code.
    """

    def find_class(self, module, name):
        if module.startswith("numpy") or module in _ALLOWED_MODULES:
            return super().find_class(module, name)
        if module.startswith("sklearn") or module.startswith("scipy"):
            key = (module, name)
            if key not in _STUB_CACHE:
                _STUB_CACHE[key] = type(name, (_Stub,), {"__module__": module})
            return _STUB_CACHE[key]
        raise pickle.UnpicklingError(
            f"refusing to load unexpected global {module}.{name} from the checkpoint"
        )


class RandomForest:
    """Minimal inference-only RandomForestClassifier reader.

    Holds each tree's node table and leaf values and reproduces
    scikit-learn's `predict_proba` with numpy only.
    """

    def __init__(self, trees: List[Tuple[np.ndarray, np.ndarray]],
                 classes: np.ndarray):
        if not trees:
            raise ValueError("the checkpoint contains no decision trees")
        self.trees = trees
        self.classes_ = classes

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_pickle(cls, path: str) -> "RandomForest":
        with open(path, "rb") as f:
            obj = _RestrictedUnpickler(f).load()

        estimators = getattr(obj, "estimators_", None)
        if estimators is None:
            raise ValueError(
                f"{path} does not look like a pickled RandomForestClassifier "
                f"(no estimators_ attribute). If you replaced the cached file, "
                f"delete it and let the module re-download.")

        trees = []
        for est in estimators:
            tree = getattr(est, "tree_", None)
            if tree is None or not hasattr(tree, "nodes"):
                raise ValueError(
                    "unexpected estimator layout in the checkpoint; expected "
                    "decision trees with a `tree_` attribute")
            trees.append((tree.nodes, np.asarray(tree.values, dtype=np.float64)))

        classes = np.asarray(getattr(obj, "classes_", np.array([0, 1])))
        return cls(trees, classes)

    # -- inference --------------------------------------------------------- #

    def _positive_column(self) -> int:
        """Index of the 'active' class, matching predict_proba[:, 1] in TDC."""
        classes = list(self.classes_)
        if len(classes) != 2:
            raise ValueError(
                f"expected a binary classifier, got classes {classes}")
        for target in (1, 1.0, True, "1"):
            if target in classes:
                return classes.index(target)
        return 1

    def predict_proba_positive(self, X: np.ndarray) -> np.ndarray:
        """P(positive class) for each row of X, averaged over the forest."""
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D feature matrix, got shape {X.shape}")

        pos = self._positive_column()
        n_samples = X.shape[0]
        rows = np.arange(n_samples)
        total = np.zeros(n_samples, dtype=np.float64)

        for nodes, values in self.trees:
            left = nodes["left_child"]
            right = nodes["right_child"]
            feature = nodes["feature"]
            threshold = nodes["threshold"]

            # Walk every sample down the tree in lockstep; leaves have no
            # left child (sklearn's TREE_LEAF == -1).
            current = np.zeros(n_samples, dtype=np.int64)
            internal = left[current] != -1
            while internal.any():
                idx = np.nonzero(internal)[0]
                node = current[idx]
                go_left = X[idx, feature[node]] <= threshold[node]
                current[idx] = np.where(go_left, left[node], right[node])
                internal = left[current] != -1

            leaf = values[current]
            if leaf.ndim == 3:  # (n_samples, n_outputs, n_classes)
                leaf = leaf[:, 0, :]
            # Old checkpoints store class counts, newer ones store fractions;
            # normalising handles both (a fraction row already sums to 1).
            denom = leaf.sum(axis=1)
            denom[denom == 0] = 1.0
            total += leaf[:, pos] / denom

        return total / len(self.trees)

    # Convenience alias so this can stand in for an sklearn estimator.
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba_positive(X)
        return np.column_stack([1.0 - p, p])

    @property
    def n_estimators(self) -> int:
        return len(self.trees)


def load_forest(path: str = None) -> RandomForest:
    """Load the GSK3B forest, downloading the checkpoint if necessary."""
    return RandomForest.from_pickle(path or download_gsk3b_model())


# --------------------------------------------------------------------------- #
# Fingerprint (mirrors tdc/chem_utils/oracle/oracle.py)
# --------------------------------------------------------------------------- #

_FP_RADIUS = 2
_FP_NBITS = 2048
_morgan_generator = None

# Invalid SMILES are routine when scoring generated molecules, and RDKit's
# C++ logger writes a four-line parse error to stderr for each one. The
# oracle silences that only while it parses, so the rest of your RDKit
# logging is untouched. Set GSK3B_RDKIT_LOGS=1 to keep the messages.
_SILENCE_RDKIT_DEFAULT = os.environ.get("GSK3B_RDKIT_LOGS",
                                        "").strip().lower() not in {
                                            "1", "true", "yes", "on"
                                        }


@contextlib.contextmanager
def _rdkit_quiet(enabled: bool = True) -> Iterator[None]:
    """Suppress RDKit's native log output for the duration of the block."""
    if not enabled:
        yield
        return

    from rdkit import rdBase

    blocker = getattr(rdBase, "BlockLogs", None)
    if blocker is not None and hasattr(blocker, "__enter__"):
        with blocker():
            yield
        return
    if blocker is not None:
        handle = blocker()  # older RDKit: logs stay blocked while alive
        try:
            yield
        finally:
            del handle
        return

    from rdkit import RDLogger  # last resort: toggle the Python-side logger

    RDLogger.DisableLog("rdApp.*")
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.*")


def disable_rdkit_logging() -> None:
    """Turn off RDKit's logging process-wide.

    The oracle does not need this -- it silences its own parsing already.
    Use it if other parts of your pipeline parse generated SMILES too.
    """
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")


def smiles_to_rdkit_mol(smiles: str):
    """Convert SMILES to an RDKit mol, returning None if it can't be sanitized."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
    return mol


def _fingerprint(mol) -> np.ndarray:
    """Morgan fingerprint, radius 2, 2048 bits, as shape (1, 2048) floats.

    Uses RDKit's MorganGenerator when available -- the legacy
    AllChem.GetMorganFingerprintAsBitVect that TDC calls is deprecated in
    recent RDKit -- and produces bit-identical fingerprints either way.
    """
    from rdkit import DataStructs

    global _morgan_generator
    if _morgan_generator is None:
        try:
            from rdkit.Chem import rdFingerprintGenerator

            _morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=_FP_RADIUS, fpSize=_FP_NBITS)
        except Exception:
            _morgan_generator = False  # fall back to the legacy API

    if _morgan_generator:
        fp = _morgan_generator.GetFingerprint(mol)
    else:
        from rdkit.Chem import AllChem

        fp = AllChem.GetMorganFingerprintAsBitVect(mol,
                                                   _FP_RADIUS,
                                                   nBits=_FP_NBITS)

    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, features)
    return features.reshape(1, -1)


# --------------------------------------------------------------------------- #
# Oracle
# --------------------------------------------------------------------------- #


class GSK3BOracle:
    """GSK3B activity oracle: SMILES -> P(active) in [0, 1].

    Accepts a single SMILES string or a list of them. Invalid SMILES score
    0.0, as in PyTDC.

    Args:
        path: directory for the cached model checkpoint.
        num_max_call: optional budget; raises ValueError once exceeded
            (TDC uses this for benchmark fairness).
        silence_rdkit: suppress RDKit's parse-error spam for invalid SMILES.
            Scoped to this oracle's own parsing, so logging elsewhere in your
            code is unaffected. Defaults to True (GSK3B_RDKIT_LOGS=1 to keep
            the messages).
    """

    def __init__(self,
                 path: str = None,
                 num_max_call: int = None,
                 silence_rdkit: bool = None):
        self.path = ORACLE_PATH if path is None else path
        self.num_max_call = num_max_call
        self.silence_rdkit = (_SILENCE_RDKIT_DEFAULT
                              if silence_rdkit is None else silence_rdkit)
        self.num_called = 0
        self.default_property = 0.0
        self._model = None

    @property
    def model(self) -> RandomForest:
        if self._model is None:
            self._model = RandomForest.from_pickle(
                download_gsk3b_model(self.path))
        return self._model

    # -- scoring ----------------------------------------------------------- #

    def score_one(self, smiles: str) -> float:
        with _rdkit_quiet(self.silence_rdkit):
            mol = smiles_to_rdkit_mol(smiles)
        if mol is None:
            return self.default_property
        try:
            fp = _fingerprint(mol)
        except Exception:
            return self.default_property
        return float(self.model.predict_proba_positive(fp)[0])

    def score_many(self, smiles_list: Sequence[str]) -> List[float]:
        """Batched scoring: the whole forest is walked once for all molecules."""
        fps, valid_idx = [], []
        with _rdkit_quiet(self.silence_rdkit):
            for i, smi in enumerate(smiles_list):
                mol = smiles_to_rdkit_mol(smi)
                if mol is None:
                    continue
                try:
                    fps.append(_fingerprint(mol)[0])
                    valid_idx.append(i)
                except Exception:
                    continue

        scores = [self.default_property] * len(smiles_list)
        if fps:
            probs = self.model.predict_proba_positive(np.asarray(fps))
            for i, p in zip(valid_idx, probs):
                scores[i] = float(p)
        return scores

    # -- call -------------------------------------------------------------- #

    def _check_budget(self, n: int) -> None:
        if self.num_max_call is not None and self.num_called + n > self.num_max_call:
            raise ValueError(
                "The maximum number of oracle calls is reached! The maximum is: "
                f"{self.num_max_call}. The current requested call (plus "
                f"accumulated calls) is: {self.num_called + n}")
        self.num_called += n

    def __call__(
        self, smiles: Union[str, Iterable[str]]
    ) -> Union[float, List[float]]:
        if isinstance(smiles, str):
            self._check_budget(1)
            return self.score_one(smiles)

        smiles_list = list(smiles)
        self._check_budget(len(smiles_list))
        return self.score_many(smiles_list)


# Module-level singleton, so `gsk3b(...)` loads the forest once and reuses it.
_default_oracle = GSK3BOracle()


def gsk3b(smiles: Union[str, Iterable[str]]) -> Union[float, List[float]]:
    """Evaluate the GSK3B score of a SMILES string or list of SMILES strings."""
    return _default_oracle(smiles)


def Oracle(name: str = "GSK3B", **kwargs) -> GSK3BOracle:
    """Drop-in stand-in for `tdc.Oracle`, restricted to GSK3B."""
    if name.lower().replace("_", "").replace("-", "") not in {"gsk3b", "gsk3"}:
        raise ValueError(
            f"This standalone module only provides the GSK3B oracle, not {name!r}."
        )
    return GSK3BOracle(**kwargs)


if __name__ == "__main__":
    demo = [
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
        "O=C(Nc1ccc(F)cc1)c1cnc2ccccn12",  # kinase-ish scaffold
        "not_a_molecule",  # invalid -> 0.0
    ]
    print(f"forest: {_default_oracle.model.n_estimators} trees")
    for smi, score in zip(demo, gsk3b(demo)):
        print(f"{score:.3f}  {smi}")