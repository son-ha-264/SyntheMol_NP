"""
Measure STEADY-STATE docking throughput, which is what an RL run actually pays.

Why the built-in --benchmark misleads: with 8 molecules and 8 workers each worker
docks one molecule, so the affinity-map computation is paid once per molecule and
the persistent-pool fix buys nothing. In training, a worker docks hundreds of
molecules against maps it computed once.

This script separates the two costs by running two batches through the SAME warm
pool: batch 1 includes map computation, batch 2 does not. The difference is the
startup cost; batch 2 is your planning figure.

It also times macrocycles separately, since those dominate the cost in an
NP-biased run and have a different per-molecule cost from drug-like scaffolds.

    python benchmark_throughput.py --workers 16 --n 64
    python benchmark_throughput.py --workers 16 --n 64 --exhaustiveness 16
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

from rdkit import Chem, RDLogger

import hsp90_oracle as _ora
from hsp90_oracle import Hsp90Oracle, make_oracle, resolve_root

RDLogger.DisableLog("rdApp.*")

# Several copies of hsp90_oracle.py may exist in a project tree, and Python picks
# whichever is importable first. Report which one, and whether it has the
# persistent worker pool: without it each batch starts fresh workers that
# recompute the affinity maps, so "warm" batch 2 is not warm and the headline
# throughput number is wrong.
HAS_POOL = hasattr(Hsp90Oracle, "_get_pool")
HAS_CTX = hasattr(Hsp90Oracle, "__enter__")
HAS_CLOSE = hasattr(Hsp90Oracle, "close")


def drug_like_set(n: int) -> List[str]:
    """Distinct, valid, drug-like SMILES built combinatorially.

    Distinctness matters: the oracle caches by canonical SMILES, so repeated
    molecules would be served from cache and the timing would be meaningless.
    """
    cores = [
        "c1ccc(cc1)C(=O)N{R}",
        "c1ccc(cc1)S(=O)(=O)N{R}",
        "c1ccc2[nH]ccc2c1C(=O)N{R}",
        "c1cnc(cc1)C(=O)N{R}",
        "O=C(N{R})c1ccc(O)cc1",
        "O=C(N{R})c1ccc(F)cc1",
        "COc1ccc(cc1)C(=O)N{R}",
        "Clc1ccc(cc1)C(=O)N{R}",
    ]
    rs = [
        "CC", "CCC", "CC(C)C", "Cc1ccccc1", "CCc1ccccc1", "C1CCCCC1",
        "CC1CCOCC1", "CCN1CCOCC1", "Cc1ccncc1", "C1CCNCC1", "CCO",
        "CC(=O)NC",
    ]
    out, seen = [], set()
    for core in cores:
        for r in rs:
            smi = core.replace("{R}", r)
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            can = Chem.MolToSmiles(mol)
            if can in seen:
                continue
            seen.add(can)
            out.append(can)
            if len(out) >= n:
                return out
    return out


def macrocycle_set(root: Path) -> List[str]:
    """Radicicol from the prepared reference, as the macrocycle cost probe."""
    sdf = root / "ligands" / "RDC_prep.sdf"
    if not sdf.is_file():
        return []
    mol = Chem.MolFromMolFile(str(sdf), removeHs=True)
    return [Chem.MolToSmiles(mol)] if mol else []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--receptor", default="np")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n", type=int, default=64,
                    help="molecules per batch; keep it >= 4x workers so each "
                         "worker docks several and maps are amortised")
    ap.add_argument("--exhaustiveness", type=int, default=32)
    ap.add_argument("--backend", default="python",
                    choices=["python", "subprocess"])
    args = ap.parse_args()

    root = resolve_root(args.root)
    mols = drug_like_set(args.n * 2)
    if len(mols) < args.n * 2:
        print(f"note: only {len(mols)} distinct molecules available; "
              f"using {len(mols)//2} per batch")
    half = len(mols) // 2
    batch1, batch2 = mols[:half], mols[half:half * 2]

    print(f"oracle module   {_ora.__file__}")
    if not HAS_POOL:
        print()
        print("  *** WARNING: this copy of hsp90_oracle.py predates the")
        print("      persistent-pool fix. Each score() call will start fresh")
        print("      workers and recompute the affinity maps, so batch 2 will")
        print("      not be warm and the throughput figure will be too high.")
        print("      Replace this file with the current version before trusting")
        print("      the numbers below.")
        print()
    print(f"root            {root}")
    print(f"receptor        {args.receptor}")
    print(f"backend         {args.backend}")
    print(f"workers         {args.workers}")
    print(f"exhaustiveness  {args.exhaustiveness}")
    print(f"batch size      {len(batch1)} (per worker: "
          f"{len(batch1)/args.workers:.1f})")
    print()

    # cache_path=None so nothing is served from a previous run
    oracle = make_oracle(str(root), args.receptor, backend=args.backend,
                         n_workers=args.workers,
                         exhaustiveness=args.exhaustiveness,
                         cache_path=None, verbose=False)

    # Works with or without the context-manager support added alongside the
    # persistent pool.
    try:
        t0 = time.time()
        r1 = oracle(batch1)
        d1 = time.time() - t0
        ok1 = sum(1 for r in r1 if r > 0)

        t0 = time.time()
        r2 = oracle(batch2)
        d2 = time.time() - t0
        ok2 = sum(1 for r in r2 if r > 0)

        print("=== Drug-like molecules ===")
        print(f"  batch 1 (cold: includes map computation)")
        print(f"      {d1:8.1f} s total   {d1/len(batch1):6.2f} s/mol   "
              f"{ok1}/{len(batch1)} scored")
        print(f"  batch 2 (warm pool: THIS IS THE PLANNING FIGURE)")
        print(f"      {d2:8.1f} s total   {d2/len(batch2):6.2f} s/mol   "
              f"{ok2}/{len(batch2)} scored")
        startup = d1 - d2
        if startup > 0:
            print(f"  one-off startup cost: ~{startup:.1f} s "
                  f"({startup/args.workers:.1f} s per worker)")

        warm_rate = d2 / max(1, len(batch2))

        macro = macrocycle_set(root)
        macro_rate = None
        if macro:
            print()
            print("=== Macrocycle (radicicol), same warm pool ===")
            t0 = time.time()
            rm = oracle(macro)
            dm = time.time() - t0
            macro_rate = dm
            print(f"      {dm:8.1f} s for 1 molecule   reward {rm[0]:.3f}")
            print(f"      ratio vs drug-like: {dm/warm_rate:5.1f}x")
            print("      NP-biased priors generate macrocycles "
                  "disproportionately, so plan with this number, not the")
            print("      drug-like one.")

        print()
        print("=== Projection ===")
        for evals in (5_000, 10_000, 20_000):
            # Assume a 60/40 drug-like/macrocycle mix for an NP-biased arm, and
            # 40% cache hits, which is typical once RL starts revisiting motifs.
            mixed = 0.6 * warm_rate + 0.4 * (macro_rate or warm_rate)
            for label, rate in (("drug-like only", warm_rate),
                                ("NP-biased mix", mixed)):
                hours = evals * rate * 0.6 / 3600      # 40% cache hits
                print(f"  {evals:>6,} evaluations, {label:16s} "
                      f"~{hours:6.1f} h per arm")
        print()
        print("  Multiply by the number of arms (3) and receptors (2) for the")
        print("  full study. Cache hits are assumed at 40%; measure your own")
        print("  rate from oracle.stats() after a real run.")
        print()
        print(f"  stats: {oracle.stats()}")
    finally:
        if HAS_CLOSE:
            oracle.close()


if __name__ == "__main__":
    main()