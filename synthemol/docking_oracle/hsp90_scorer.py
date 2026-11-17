"""Single-SMILES Scorer for the Hsp90 docking oracle.

The one-at-a-time interface costs ~5%: Vina parallelises the exhaustiveness runs
of a single docking across cores, so one molecule on N cores takes about as long
as N molecules on one core each. That holds while cores <= exhaustiveness; above
that Vina runs out of search runs to distribute and batching would start to pay.

    scorer = HSP90Scorer()
    reward = scorer("CCOc1ccc(cc1)C(=O)NCc1ccncc1")   # [0, 1], higher is better

Reference rewards: radicicol ~0.53, NVP-AUY922 ~0.73. Above 0.9 is more likely
reward hacking than discovery - inspect the structures.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from junk.docking_oracle_2.hsp90_oracle import Filters, make_oracle


class HSP90Scorer:
    """Reward in [0, 1] for one SMILES. Higher is better."""

    def __init__(self, receptor: str = "np", exhaustiveness: int = 64,
                 cache_path: Optional[str] = "hsp90_cache_np.json",
                 filters: Optional[Filters] = None, root: Optional[str] = None,
                 verbose: bool = False) -> None:
        # n_workers=1 is deliberate: with a single molecule the oracle runs
        # in-process and gives Vina every core. A larger value changes nothing.
        self.oracle = make_oracle(root, receptor, n_workers=1, filters=filters,
                                  exhaustiveness=exhaustiveness,
                                  cache_path=cache_path, verbose=verbose)
        self.n_calls = 0

    def __call__(self, smiles: str) -> float:
        self.n_calls += 1
        return self.oracle([smiles])[0]

    def detail(self, smiles: str):
        """Full record. Worth logging: reward 0.0 means three different things
        (weak binder, filter rejection, failure) that the scalar cannot separate."""
        return self.oracle.score([smiles])[0]

    def stats(self) -> dict:
        s = dict(self.oracle.stats(), calls=self.n_calls)
        if self.n_calls:
            s["cache_hit_rate"] = round(1 - s["docked"] / self.n_calls, 3)
        return s

    def close(self) -> None:
        if hasattr(self.oracle, "close"):
            self.oracle.close()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


def report_parallelism(scorer: HSP90Scorer) -> None:
    """Check that one docking saturates the cores, and time a few molecules."""
    cores, exh = os.cpu_count() or 1, scorer.oracle.exhaustiveness
    print(f"cores {cores}   exhaustiveness {exh}")
    if cores > exh:
        print(f"  WARNING: a single docking saturates at most {exh} cores, so "
              f"{cores - exh} sit idle. Raise exhaustiveness or batch.")
    else:
        print(f"  OK: {exh / cores:.1f} search runs per core.")
    print(f"  check CPU usage meanwhile: expect ~{min(cores, exh) * 100}% for vina\n")

    smis = ["CC(C)c1cc(O)c(O)cc1C(=O)NCC1CCOCC1",
            "O=C(Nc1ccccc1)c1ccc(O)cc1",
            "Oc1ccc(cc1)C1=NOC(=C1)c1ccccc1"]
    times = []
    for s in smis:
        t0 = time.time()
        r = scorer(s)
        times.append(time.time() - t0)
        print(f"  {times[-1]:6.1f} s   reward {r:.3f}   {s}")

    # The first call carries one-off Vina init, so steady state is the rest.
    steady = sum(times[1:]) / max(1, len(times) - 1)
    print(f"\n  first call {times[0]:.1f} s (includes init), steady state "
          f"{steady:.1f} s/mol")
    if scorer.stats()["docked"] == 0:
        print("  WARNING: every molecule came from cache - these are lookup times,"
              " not docking times. Delete the cache file and rerun.")


if __name__ == "__main__":
    with HSP90Scorer() as scorer:
        report_parallelism(scorer)
        print(f"\nstats: {scorer.stats()}")
