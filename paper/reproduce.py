"""Full experimental reproduction for the manuscript. Offline, seeded, reproducible.

Builds the system from committed data (no network, no API keys), then writes every
number in the manuscript to paper/results/: verifier discrimination, phonetic ablation,
architecture comparison, determinism, and a multi-seed sweep (seeds 1, 2, 3) over the
twenty default targets with a separate brand-vs-generic aggregation.

Run from the repository root:

    python paper/reproduce.py                  # everything, all three seeds in one go
    python paper/reproduce.py 2                # resume: only seed 2's sweep
    python paper/reproduce.py --aggregate      # merge the per-seed sweeps already written

Seed-mode exists because a full three-seed sweep is ~an hour on a laptop CPU: write one
seed at a time, then aggregate. Every deterministic section (discrimination, ablation,
architecture, determinism) is reproduced by seed-mode too, so a seed churn never leaves
the paper's headline numbers stale.
"""
import json
import os
import sys
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd

warnings.filterwarnings("ignore")
from pharma_name_gen import build_system
from pharma_name_gen.verifier import VerifierConfig
from pharma_name_gen import evaluation as ev
from pharma_name_gen.sweep import (
    DEFAULT_TARGETS, acceptance_by_difficulty, failure_profile,
    proposer_contribution, quality_component_correlations, run_sweep,
)

OUT = os.path.join(ROOT, "paper", "results")
os.makedirs(OUT, exist_ok=True)
log = lambda *a: print(*a, flush=True)

SEEDS = [1, 2, 3]
N_PER_CLASS = 10


def out(name: str) -> str:
    return os.path.join(OUT, name)


def build():
    t0 = time.time()
    system = build_system(live=False, use_artifacts=True,
                          verifier_config=VerifierConfig(stem_aware_similarity=True),
                          progress=log)
    log(system.summary())
    if not os.path.exists(out("system_summary.txt")):
        open(out("system_summary.txt"), "w").write(system.summary())
        json.dump(system.manifest(), open(out("manifest.json"), "w"), indent=2,
                  default=str)
    return system


def deterministic_sections(system):
    if os.path.exists(out("verifier_discrimination.json")):
        log("deterministic sections already present, skipping")
        return

    log("\n== discrimination ==")
    verif = ev.evaluate_verifier(system, n_negative=400, seed=0)
    verif["threshold_sweep"].to_csv(out("threshold_sweep.csv"), index=False)
    json.dump({k: v for k, v in verif.items() if k != "threshold_sweep"},
              open(out("verifier_discrimination.json"), "w"), indent=2, default=str)
    log({k: verif[k] for k in ("n_confusable_pairs", "n_random_pairs",
                               "mean_confusable_score", "mean_random_score",
                               "separation", "roc_auc", "optimal_threshold_youden")})

    log("\n== phonetic ablation ==")
    ab = ev.ablate_phonetic_algorithms(system, n_negative=300, seed=0)
    ab.to_csv(out("ablate_phonetic.csv"), index=False)
    log(ab.to_string())

    log("\n== v1 vs v2 ==")
    comp = ev.compare_architectures(system, n=10)
    comp["candidates"].to_csv(out("arch_candidates.csv"), index=False)
    comp["summary"].to_csv(out("arch_summary.csv"))
    comp["by_component"].to_csv(out("arch_by_component.csv"))
    log(comp["summary"].to_string())
    log(comp["by_component"].to_string())
    log(comp["candidates"][["architecture", "candidate_name", "quality_total",
                            "composite_risk_score", "risk_band",
                            "proposer"]].to_string())

    log("\n== determinism ==")
    det = ev.determinism_check(system, runs=3, n=5)
    json.dump(det, open(out("determinism.json"), "w"), indent=2)
    log(det)


def run_seed(system, s: int):
    log(f"\n== sweep seed {s} ==")
    df, summary = run_sweep(system, DEFAULT_TARGETS, n_per_class=N_PER_CLASS,
                            out_csv=out(f"sweep_all_attempts_s{s}.csv"),
                            progress=log, seed=s)
    json.dump(summary, open(out(f"sweep_summary_s{s}.json"), "w"), indent=2,
              default=str)
    log(f"seed {s}: {summary['total_candidates']} candidates, "
        f"{summary['total_accepted']} accepted ({summary['overall_accept_rate']:.1%}) "
        f"in {summary['wall_seconds']}s")


def aggregate():
    all_dfs = []
    per_seed = []
    for s in SEEDS:
        f = out(f"sweep_all_attempts_s{s}.csv")
        if not os.path.exists(f):
            sys.exit(f"missing {f} - run `python paper/reproduce.py {s}` first")
        all_dfs.append(pd.read_csv(f))
        per_seed.append(json.load(open(out(f"sweep_summary_s{s}.json"))))

    merged = pd.concat(all_dfs, ignore_index=True)
    merged.to_csv(out("sweep_all_attempts.csv"), index=False)
    json.dump({"seeds": SEEDS, "per_seed": per_seed},
              open(out("sweep_summary.json"), "w"), indent=2, default=str)
    log(f"merged {len(merged)} rows across seeds 1..3")

    headlines = []
    for s, df in zip(SEEDS, all_dfs):
        for tt, sub in df.groupby("target_type"):
            headlines.append({
                "seed": s, "target_type": tt, "candidates": int(len(sub)),
                "accepted": int(sub["accepted"].sum()),
                "accept_rate": float(sub["accepted"].mean()),
                "mean_quality": float(sub["quality_total"].mean()),
                "best_quality": float(sub["quality_total"].max()),
            })
    agg_rows = []
    for tt, g in pd.DataFrame(headlines).groupby("target_type"):
        for m in ("accept_rate", "mean_quality", "best_quality"):
            agg_rows.append({"target_type": tt, "metric": m,
                             "mean": round(float(g[m].mean()), 3),
                             "std": round(float(g[m].std()), 3),
                             "seed_values": [round(float(v), 3) for v in g[m]]})
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out("sweep_brand_vs_generic.csv"), index=False)
    log("\n-- brand vs generic (mean +/- std across seeds) --")
    log(agg.to_string(index=False))

    for name, fn in [("acceptance_by_difficulty", acceptance_by_difficulty),
                     ("proposer_contribution", proposer_contribution),
                     ("failure_profile", failure_profile),
                     ("quality_component_correlations",
                      quality_component_correlations)]:
        t = fn(merged)
        t.to_csv(out(f"{name}.csv"))
        log(f"\n-- {name} --")
        log(t.to_string())


def main():
    args = sys.argv[1:]
    system = build()
    deterministic_sections(system)

    if not args:
        for s in SEEDS:
            run_seed(system, s)
        aggregate()
        log(f"\nTOTAL {time.time() - t0:.1f}s")
        return

    if args[0] == "--aggregate":
        aggregate()
        return

    if args[0].isdigit() and int(args[0]) in SEEDS:
        run_seed(system, int(args[0]))
        return

    sys.exit(f"usage: reproduce.py [SEED|--aggregate], seed in {SEEDS}")


t0 = time.time()
main()