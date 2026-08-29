#!/usr/bin/env python3
"""
Refresh the committed reference snapshot from the primary regulators.

Run this when you want to update `data/existing_drug_names.csv`, then commit the result.
The point of committing a snapshot rather than always fetching live is reproducibility:
a paper's numbers must be re-derivable in six months, and openFDA's contents will have
moved by then.

    python scripts/fetch_reference_data.py --ndc-limit 25000
    python scripts/fetch_reference_data.py --sources openfda rxnorm --out data/

Sources are all primary and openly licensed: openFDA (US FDA), RxNorm (US NLM), and the
EMA medicines register (EU). No aggregator sites are used at any point, because a
screening universe scraped from a third-party site is not citable.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pharma_name_gen import data_layer as dl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=["openfda", "rxnorm", "ema"],
                    choices=["openfda", "rxnorm", "ema"])
    ap.add_argument("--ndc-limit", type=int, default=20000,
                    help="openFDA caps `skip` at 25000 without an API key.")
    ap.add_argument("--out", default="data", type=Path)
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    print(f"Fetching from: {', '.join(args.sources)}")
    snap = dl.build_snapshot(live=True, sources=args.sources,
                             ndc_limit=args.ndc_limit, timeout=args.timeout,
                             verbose=True)

    if snap.mode != "live":
        print("\nNo live source succeeded. The committed snapshot is unchanged.",
              file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "existing_drug_names.csv"
    cols = ["generic_name_raw", "brand_name_raw", "product_type", "route", "source"]
    snap.names[cols].to_csv(csv_path, index=False)

    manifest = snap.manifest()
    manifest["fetched_by"] = "scripts/fetch_reference_data.py"
    manifest["fetched_at"] = datetime.now(timezone.utc).isoformat()
    (args.out / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"\nwrote {len(snap.names)} rows to {csv_path}")
    print(f"wrote provenance to {args.out / 'PROVENANCE.json'}")
    print(f"fingerprint: {snap.fingerprint}")
    print("\nCommit both files together — a corpus without its provenance record is "
          "not a citable artifact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
