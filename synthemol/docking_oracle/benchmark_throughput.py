"""Measure steady-state docking throughput, which is what an RL run pays.

The oracle's own --benchmark misleads: with 8 molecules and 8 workers each worker
docks one, so affinity-map computation is paid per molecule and the persistent
pool buys nothing. In training a worker docks hundreds against maps it computed
once.

Two batches go through the SAME warm pool. Batch 1 includes startup; batch 2 does
not and is the planning figure. A macrocycle is timed separately, since those
dominate an NP-biased run and cost several times more.

    python benchmark_throughput.py --workers 16 --n 64
"""

from __future__ import annotations

import argparse
import time
from typing import List

from rdkit import Chem, RDLogger

import junk.docking_oracle_2.hsp90_oracle as _ora
from junk.docking_oracle_2.hsp90_oracle import Hsp90Oracle, make_oracle, resolve_root

RDLogger.DisableLog("rdApp.*")

# Without the persistent pool each batch starts fresh workers, so batch 2 is not
# warm and the headline number is wrong. Several copies of hsp90_oracle.py may
# exist in a project tree, so report which one was imported.
HAS_POOL = hasattr(Hsp90Oracle, "_get_pool")
HAS_CLOSE = hasattr(Hsp90Oracle, "close")

CORES = ["c1ccc(cc1)C(=O)N{R}", "c1ccc(cc1)S(=O)(=O)N{R}", "c1cnc(cc1)C(=O)N{R}",
         "O=C(N{R})c1ccc(O)cc1", "O=C(N{R})c1ccc(F)cc1", "COc1ccc(cc1)C(=O)N{R}",
         "Clc1ccc(cc1)C(=O)N{R}", "c1ccc2[nH]ccc2c1C(=O)N{R}",
         "O=C(N{R})c1ccc(cc1)C#N", "O=C(N{R})c1cccc(O)c1"]
SUBS = ["CC", "CCC", "CC(C)C", "Cc1ccccc1", "CCc1ccccc1", "C1CCCCC1",
        "CC1CCOCC1", "CCN1CCOCC1", "Cc1ccncc1", "C1CCNCC1", "CC(=O)NC",
        "CCO", "CCOC", "CC1CCNCC1"]   # 10 x 14 = 140, enough for --n 64


def distinct_smiles(n: int) -> List[str]:
    """Distinct valid drug-like SMILES. Distinctness matters: the oracle caches
    by canonical SMILES, so repeats would be served from cache."""
    out, seen = [], set()
    for core in CORES:
        for r in SUBS:
            mol = Chem.MolFromSmiles(core.replace("{R}", r))
            if mol is None:
                continue
            can = Chem.MolToSmiles(mol)
            if can not in seen:
                seen.add(can)
                out.append(can)
                if len(out) >= n:
                    return out
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--receptor", default="np")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n", type=int, default=64,
                    help="molecules per batch; keep >= 4x workers so maps amortise")
    ap.add_argument("--exhaustiveness", type=int, default=64)
    ap.add_argument("--backend", default="python",
                    choices=["python", "subprocess"])
    args = ap.parse_args()

    root = resolve_root(args.root)
    mols = distinct_smiles(args.n * 2)
    half = len(mols) // 2
    b1, b2 = mols[:half], mols[half:half * 2]
    if half < args.n:
        print(f"note: only {len(mols)} distinct molecules available, "
              f"using {half} per batch")

    print(f"oracle module   {_ora.__file__}")
    if not HAS_POOL:
        print("  *** This copy predates the persistent-pool fix: batch 2 will not"
              " be warm and the figure will be too high. Update it. ***")
    print(f"root {root}")
    print(f"receptor {args.receptor}   backend {args.backend}   "
          f"workers {args.workers}   exhaustiveness {args.exhaustiveness}")
    print(f"batch {half} molecules ({half / args.workers:.1f} per worker)\n")

    # cache_path=None so nothing is served from an earlier run
    oracle = make_oracle(str(root), args.receptor, backend=args.backend,
                         n_workers=args.workers, cache_path=None,
                         exhaustiveness=args.exhaustiveness)
    try:
        t0 = time.time(); r1 = oracle(b1); d1 = time.time() - t0
        t0 = time.time(); r2 = oracle(b2); d2 = time.time() - t0
        warm = d2 / max(1, len(b2))

        print("=== Drug-like ===")
        print(f"  batch 1 (cold)  {d1:7.1f} s   {d1/len(b1):5.2f} s/mol   "
              f"{sum(r > 0 for r in r1)}/{len(b1)} scored")
        print(f"  batch 2 (warm)  {d2:7.1f} s   {warm:5.2f} s/mol   "
              f"{sum(r > 0 for r in r2)}/{len(b2)} scored   <- PLANNING FIGURE")
        if d1 > d2:
            print(f"  one-off startup ~{d1 - d2:.1f} s "
                  f"({(d1 - d2)/args.workers:.1f} s per worker)")

        # Radicicol as the macrocycle cost probe, from the prepared reference
        macro = None
        sdf = root / "ligands" / "RDC_prep.sdf"
        if sdf.is_file():
            m = Chem.MolFromMolFile(str(sdf), removeHs=True)
            if m is not None:
                t0 = time.time(); rm = oracle([Chem.MolToSmiles(m)])
                macro = time.time() - t0
                print("\n=== Macrocycle (radicicol) ===")
                print(f"  {macro:7.1f} s   reward {rm[0]:.3f}   "
                      f"{macro/warm:.1f}x drug-like")
                print("  Plan with this: an NP-biased prior generates macrocycles"
                      " disproportionately.")

        print("\n=== Projection (40% cache hits, 60/40 drug-like/macrocycle) ===")
        mixed = 0.6 * warm + 0.4 * (macro or warm)
        for n in (5_000, 10_000, 20_000):
            print(f"  {n:>6,} evals   drug-like {n*warm*0.6/3600:5.1f} h"
                  f"   NP mix {n*mixed*0.6/3600:5.1f} h   per arm")
        print("  Measure your own cache-hit rate from oracle.stats() after a run.")
        print(f"\n  stats: {oracle.stats()}")
    finally:
        if HAS_CLOSE:
            oracle.close()


if __name__ == "__main__":
    main()