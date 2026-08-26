"""
NOMINA evaluation — the numbers that go in the paper.

Three families of experiment, each answering a question a reviewer will ask:

1. **Does the verifier discriminate?** Known-confusable name pairs (the FDA and ISMP
   confused-drug-name lists are the canonical source) should score high; random pairs
   from the same corpus should score low. If the two distributions overlap, the
   composite score is not measuring confusability and no threshold choice can rescue it.

2. **Does each component earn its place?** Ablations. Remove one phonetic algorithm or
   one quality term, re-measure, report the delta. A component that changes nothing when
   removed should be removed, or at minimum should not be described in the methodology
   as if it contributes.

3. **Is v2 actually better than v1?** The honest version of this compares the two
   architectures on the same corpus, same verifier, same seed, and reports the quality
   distribution of what each produces. Anything less is an assertion.

Everything here is deterministic given a seed, and everything reads from artifacts the
sweep already wrote, so the evaluation section of the paper is re-derivable rather than
transcribed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Known-confusable pairs.
# Source: FDA "Name Differentiation Project" / ISMP List of Confused Drug Names.
# These are pairs that have caused documented dispensing errors, which is a far
# stronger label than "two names that look similar to me".
# ---------------------------------------------------------------------------
LASA_PAIRS: List[Tuple[str, str]] = [
    ("clonidine", "klonopin"), ("hydralazine", "hydroxyzine"),
    ("metformin", "metronidazole"), ("prednisone", "prednisolone"),
    ("celebrex", "celexa"), ("zantac", "zyrtec"),
    ("lamictal", "lamisil"), ("amlodipine", "amiloride"),
    ("cycloserine", "cyclosporine"), ("chlorpromazine", "chlorpropamide"),
    ("dobutamine", "dopamine"), ("glipizide", "glyburide"),
    ("humalog", "humulin"), ("lorazepam", "alprazolam"),
    ("methadone", "methylphenidate"), ("nicardipine", "nifedipine"),
    ("oxycodone", "oxycontin"), ("quinine", "quinidine"),
    ("sulfadiazine", "sulfasalazine"), ("tolbutamide", "tolazamide"),
    ("vinblastine", "vincristine"), ("zolpidem", "zolmitriptan"),
    ("cefazolin", "cefprozil"), ("carboplatin", "cisplatin"),
    ("daunorubicin", "doxorubicin"), ("epinephrine", "ephedrine"),
    ("fluoxetine", "duloxetine"), ("hydromorphone", "morphine"),
    ("levothyroxine", "liothyronine"), ("mercaptopurine", "methotrexate"),
]


# ===========================================================================
# 1. Discrimination
# ===========================================================================

def score_pairs(verifier, pairs: Sequence[Tuple[str, str]]) -> List[float]:
    """Score each pair by asking the verifier how close the first is to the second.

    A single-element corpus is substituted so the score is the pair distance rather than
    the distance to the nearest of thousands of names, which would be dominated by
    whichever unrelated name happened to be closest.
    """
    out = []
    for a, b in pairs:
        try:
            ranked = verifier._rank(a, [b], {b: None}, 1)      # noqa: SLF001
            if ranked:
                out.append(float(ranked[0]["composite"]))
        except Exception:                                      # noqa: BLE001
            continue
    return out


def random_pairs(corpus: Sequence[str], n: int, seed: int = 0) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    pool = [c for c in corpus if len(c) >= 5]
    return [(rng.choice(pool), rng.choice(pool)) for _ in range(n)]


def roc_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """AUC via the Mann-Whitney U identity, with explicit tie handling.

    Written out rather than imported from sklearn so the package has no heavyweight ML
    dependency for one statistic, and so the tie convention is visible.
    """
    if not positive or not negative:
        return float("nan")
    wins = ties = 0
    for p in positive:
        for n in negative:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def threshold_sweep(positive: Sequence[float], negative: Sequence[float],
                    lo: float = 30, hi: float = 95, step: float = 2.5) -> pd.DataFrame:
    """Sensitivity and specificity across candidate cutoffs.

    This is what justifies an operating point. Choosing 70 because it is a round number
    is not a justification; choosing it because sensitivity is still X while specificity
    reaches Y is.
    """
    rows = []
    t = lo
    while t <= hi:
        tp = sum(1 for p in positive if p >= t)
        fn = len(positive) - tp
        fp = sum(1 for n in negative if n >= t)
        tn = len(negative) - fp
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        rows.append({"threshold": t, "sensitivity": round(sens, 3),
                     "specificity": round(spec, 3),
                     "youden_j": round(sens + spec - 1, 3),
                     "false_positives": fp, "false_negatives": fn})
        t += step
    return pd.DataFrame(rows)


def evaluate_verifier(system, n_negative: int = 400, seed: int = 0) -> Dict[str, Any]:
    """Full discrimination report for the screening component."""
    pos = score_pairs(system.verifier, LASA_PAIRS)
    neg = score_pairs(system.verifier, random_pairs(system.screening.all, n_negative, seed))
    sweep = threshold_sweep(pos, neg)
    best = sweep.loc[sweep["youden_j"].idxmax()] if len(sweep) else None
    return {
        "n_confusable_pairs": len(pos),
        "n_random_pairs": len(neg),
        "mean_confusable_score": round(sum(pos) / max(1, len(pos)), 2),
        "mean_random_score": round(sum(neg) / max(1, len(neg)), 2),
        "separation": round(sum(pos) / max(1, len(pos)) - sum(neg) / max(1, len(neg)), 2),
        "roc_auc": round(roc_auc(pos, neg), 4),
        "threshold_sweep": sweep,
        "optimal_threshold_youden": float(best["threshold"]) if best is not None else None,
        "configured_high_cutoff": system.verifier.config.thresholds.similarity_high,
        "configured_moderate_cutoff": system.verifier.config.thresholds.similarity_moderate,
        "positive_scores": pos,
        "negative_scores": neg,
    }


# ===========================================================================
# 2. Ablations
# ===========================================================================

def ablate_quality_components(system, df: pd.DataFrame) -> pd.DataFrame:
    """Recompute the composite with each term removed, one at a time.

    Reported as rank correlation against the full objective rather than as a score
    delta, because what matters is whether removing the term changes *which names get
    picked*, not whether it shifts every score by a constant.
    """
    cols = [c for c in df.columns if c.startswith("q_")]
    if not cols or "quality_total" not in df:
        return pd.DataFrame()
    weights = system.scorer.generic_weights
    sub = df[cols + ["quality_total"]].dropna()
    if len(sub) < 5:
        return pd.DataFrame()

    rows = []
    for drop in cols:
        keep = [c for c in cols if c != drop]
        w = {c: weights.get(c[2:], 0.0) for c in keep}
        norm = sum(w.values()) or 1.0
        recomputed = sum(sub[c] * w[c] for c in keep) / norm * 100.0
        rows.append({
            "removed_component": drop[2:],
            "weight": round(weights.get(drop[2:], 0.0), 3),
            "rank_corr_with_full": round(float(recomputed.corr(sub["quality_total"],
                                                              method="spearman")), 4),
            "mean_abs_score_shift": round(float((recomputed - sub["quality_total"]).abs().mean()), 2),
        })
    out = pd.DataFrame(rows).sort_values("rank_corr_with_full")
    out["interpretation"] = out["rank_corr_with_full"].map(
        lambda r: "load-bearing" if r < 0.90 else ("contributes" if r < 0.985 else "near-inert"))
    return out


def ablate_phonetic_algorithms(system, pairs: Sequence[Tuple[str, str]] = LASA_PAIRS,
                               n_negative: int = 300, seed: int = 0) -> pd.DataFrame:
    """Zero out one similarity algorithm's weight at a time and re-measure AUC.

    The verifier blends several string and phonetic measures. This says which of them is
    actually carrying the discrimination, which is the claim the methodology section
    needs to support.
    """
    from copy import deepcopy
    base_w = system.verifier.config.weights
    fields = [f for f in vars(base_w) if isinstance(getattr(base_w, f), (int, float))]
    neg_pairs = random_pairs(system.screening.all, n_negative, seed)

    def auc_with(weights) -> float:
        original = system.verifier.config.weights
        system.verifier.config.weights = weights
        try:
            return roc_auc(score_pairs(system.verifier, pairs),
                           score_pairs(system.verifier, neg_pairs))
        finally:
            system.verifier.config.weights = original

    baseline = auc_with(base_w)
    rows = [{"algorithm": "(none removed)", "weight": None,
             "roc_auc": round(baseline, 4), "delta": 0.0}]
    for f in fields:
        w = deepcopy(base_w)
        original_value = getattr(w, f)
        if not original_value:
            continue
        setattr(w, f, 0.0)
        a = auc_with(w)
        rows.append({"algorithm": f, "weight": original_value,
                     "roc_auc": round(a, 4), "delta": round(a - baseline, 4)})
    return pd.DataFrame(rows).sort_values("delta")


# ===========================================================================
# 3. Architecture comparison — v1 against v2
# ===========================================================================

V1_BASELINE_OUTPUT: Dict[str, List[str]] = {
    # Verbatim output from the pre-redesign four-strategy generator, same corpus, same
    # class, recorded before any change was made. Kept as a literal so the comparison in
    # the paper is against what the old system actually produced, not a reconstruction.
    "rejection_sampling": ["erythroolol", "benralimolol", "acycloolol", "avalolol", "amoxiolol"],
    "constrained_decoding": ["thodhunolol", "roanzensolol", "hiemfailolol",
                             "joatloatolol", "snabreistolol"],
    "rl_refined": ["vancicloolol", "eculiolol", "vildaolol", "famuolol", "oxaolol"],
}


def compare_architectures(system, target_class: str = "beta-blocker",
                          target_stem: str = "-olol",
                          n: int = 10) -> Dict[str, Any]:
    """Score v1's actual output and v2's output on the same objective.

    The key move is that both are scored by the *same* quality function against the
    *same* corpus. v1 had no quality function, so its names were never measured on these
    axes; measuring them now is the only way to make the improvement a number instead of
    an opinion.
    """
    from .contracts import TargetType

    v1_rows = []
    for strategy, names in V1_BASELINE_OUTPUT.items():
        for nm in names:
            resp = system.verifier.verify(nm, target_type="generic",
                                          target_class=target_class,
                                          target_stem=target_stem)
            q = system.scorer.score(nm, resp, TargetType.GENERIC, target_stem)
            row = {"architecture": "v1 (four independent strategies)",
                   "proposer": strategy, "candidate_name": nm,
                   "quality_total": round(q.total, 2),
                   "composite_risk_score": round(resp.composite_risk_score, 2),
                   "risk_band": resp.risk_band.value,
                   "accepted_under_v1_policy": resp.overall_pass}
            row.update({f"q_{c.name}": round(c.score, 3) for c in q.components})
            v1_rows.append(row)

    report = system.generic.generate(n_shortlist=n, target_class=target_class,
                                     target_stem=target_stem)
    v2_rows = []
    for c in report.shortlist:
        row = {"architecture": "v2 (pool-and-select)", "proposer": c.proposer,
               "candidate_name": c.name, "quality_total": round(c.quality.total, 2),
               "composite_risk_score": round(c.risk, 2),
               "risk_band": c.response.risk_band.value,
               "accepted_under_v1_policy": c.response.overall_pass}
        row.update({f"q_{x.name}": round(x.score, 3) for x in c.quality.components})
        v2_rows.append(row)

    df = pd.DataFrame(v1_rows + v2_rows)
    summary = df.groupby("architecture").agg(
        n=("candidate_name", "count"),
        mean_quality=("quality_total", "mean"),
        best_quality=("quality_total", "max"),
        mean_risk=("composite_risk_score", "mean"),
        low_band=("risk_band", lambda s: int((s == "low").sum())),
    ).round(2)

    comp_cols = [c for c in df.columns if c.startswith("q_")]
    by_component = df.groupby("architecture")[comp_cols].mean().round(3).T
    by_component.index = [i[2:] for i in by_component.index]

    return {"candidates": df, "summary": summary, "by_component": by_component,
            "v2_stats": report.stats}


# ===========================================================================
# 4. Reproducibility
# ===========================================================================

def determinism_check(system, target_class: str = "beta-blocker",
                      target_stem: str = "-olol", n: int = 5,
                      runs: int = 2) -> Dict[str, Any]:
    """Two runs with the same seed must produce the same shortlist.

    Cheap to run and it catches the single most common way a "reproducible" pipeline
    stops being reproducible, which is an unseeded RNG somewhere in a helper.
    """
    outputs = []
    for _ in range(runs):
        rep = system.generic.generate(n_shortlist=n, target_class=target_class,
                                      target_stem=target_stem)
        outputs.append([c.name for c in rep.shortlist])
    return {"runs": outputs, "identical": all(o == outputs[0] for o in outputs)}
