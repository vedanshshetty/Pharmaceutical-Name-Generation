#!/usr/bin/env python3
"""
Build and publish trained artifacts so a fresh clone starts warm.

    python scripts/train_artifacts.py --live
    python scripts/train_artifacts.py --offline

Timing the cold path shows why this exists and what it is actually for:

    screening corpus 0.03s   n-gram fit 0.01s   induced grammar 0.05s
    LIVE CORPUS FETCH 20-60s

The expensive thing is the network, not the arithmetic. So caching the corpus is a speed
decision, while persisting the trained models is a *reproducibility* decision: a
published result should be re-derivable from a committed artifact rather than from
whatever the model happened to fit that afternoon.

Keys are content-addressed on the corpus fingerprint plus the relevant config hash, so
changing the stem table produces a different key and the stale model is simply not found.
There is no "remember to clear the cache" step, because that step is always forgotten.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pharma_name_gen import build_system
from pharma_name_gen.verifier import VerifierConfig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--live", action="store_true", default=True)
    g.add_argument("--offline", dest="live", action="store_false")
    ap.add_argument("--no-publish", action="store_true",
                    help="Build the cache but do not stage anything for commit.")
    args = ap.parse_args()

    system = build_system(live=args.live, use_artifacts=True,
                          verifier_config=VerifierConfig(stem_aware_similarity=True),
                          progress=print)
    print()
    print(system.summary())
    print()
    print(system.store.summary())

    if not args.no_publish:
        staged = system.store.publish()
        print(f"\nstaged {len(staged)} artifacts with `git add`:")
        for p in staged:
            print("  ", p.name)
        print("\nThis deliberately stops short of committing. Run:")
        print("  git commit -m 'Update trained artifacts'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
