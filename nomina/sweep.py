"""
NOMINA sweep harness — run every class, record every attempt, write one CSV.

Why every attempt and not just the winners
------------------------------------------
A results table that lists only accepted names cannot answer the question a reviewer
will actually ask, which is not "what did it produce" but "what did it produce relative
to what it tried". Acceptance rate, the distribution of failure codes, which proposer
carries which class, and whether the hard cases fail for structural reasons or for
incidental ones are all invisible in a winners-only table.

So the sweep writes one row per candidate ever evaluated, accepted or not, with an
`accepted` flag, the failure codes, the full quality decomposition and the proposer that
produced it. Every number in the evaluation section of the paper is then derivable from
this single artifact, which is also what makes the results reproducible: the CSV plus
the run manifest is enough to re-derive every claim.

Class selection
---------------
The default sweep is not a random sample of stems. It deliberately spans the difficulty
range, because a sweep over easy classes only would report a flattering acceptance rate
that says nothing:

  * roomy classes    (`-pril`, `-sartan`, `-vaptan`): few siblings, short names viable
  * crowded classes  (`-olol`, `-prazole`, `-caine`): many siblings, high intra-stem pressure
  * saturated classes(`-tinib`, `-mab`, `-gliptin`): dozens of recent entrants, long stems
  * brand-mode runs  (no stem at all, opposite objective)
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .contracts import TargetType


@dataclass
class SweepTarget:
    """One cell of the sweep."""
    target_type: str            # 'generic' | 'brand'
    stem: Optional[str]
    label: str
    difficulty: str             # 'roomy' | 'crowded' | 'saturated' | 'brand'


DEFAULT_TARGETS: List[SweepTarget] = [
    SweepTarget("generic", "-pril", "ACE inhibitor", "roomy"),
    SweepTarget("generic", "-sartan", "angiotensin II receptor antagonist", "roomy"),
    SweepTarget("generic", "-vaptan", "vasopressin receptor antagonist", "roomy"),
    SweepTarget("generic", "-dipine", "dihydropyridine calcium channel blocker", "roomy"),
    SweepTarget("generic", "-olol", "beta-blocker", "crowded"),
    SweepTarget("generic", "-prazole", "proton pump inhibitor", "crowded"),
    SweepTarget("generic", "-caine", "local anaesthetic", "crowded"),
    SweepTarget("generic", "-statin", "HMG-CoA reductase inhibitor", "crowded"),
    SweepTarget("generic", "-cycline", "tetracycline antibacterial", "crowded"),
    SweepTarget("generic", "-tinib", "tyrosine kinase inhibitor", "saturated"),
    SweepTarget("generic", "-mab", "monoclonal antibody", "saturated"),
    SweepTarget("generic", "-gliptin", "DPP-4 inhibitor", "saturated"),
    SweepTarget("generic", "-gliflozin", "SGLT2 inhibitor", "saturated"),
    SweepTarget("generic", "-parib", "PARP inhibitor", "saturated"),
    SweepTarget("brand", None, "proprietary mark, cardiovascular", "brand"),
    SweepTarget("brand", None, "proprietary mark, oncology", "brand"),
]


def targets_from_stems(system, stems: Sequence[str],
                       difficulty: str = "custom") -> List[SweepTarget]:
    """Build sweep targets from arbitrary stems, labelled from the stem table."""
    table = {str(r.stem).strip().lower(): str(r.meaning)
             for r in system.snapshot.stems.itertuples(index=False)}
    out = []
    for st in stems:
        key = st if st.startswith("-") else f"-{st}"
        out.append(SweepTarget("generic", key,
                               table.get(key, table.get(key.lstrip("-"), key)),
                               difficulty))
    return out


def run_sweep(system, targets: Optional[Sequence[SweepTarget]] = None,
              n_per_class: int = 10,
              out_csv: Optional[Path] = None,
              progress: Optional[Callable[[str], None]] = None,
              continue_on_error: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run the whole sweep and return (every_attempt_dataframe, summary).

    Errors in one cell are recorded and the sweep continues. A sweep that dies on class
    nine of sixteen and loses the first eight is worse than useless, because the failure
    is usually something incidental like a stem with no siblings in the corpus.
    """
    say = progress or (lambda _m: None)
    targets = list(targets or DEFAULT_TARGETS)
    rows: List[Dict[str, Any]] = []
    per_class: List[Dict[str, Any]] = []
    started = datetime.now(timezone.utc).isoformat()
    t_all = time.perf_counter()

    for i, tgt in enumerate(targets, 1):
        say(f"[{i}/{len(targets)}] {tgt.label} ({tgt.stem or 'brand'})")
        t0 = time.perf_counter()
        try:
            pipeline = system.pipeline(tgt.target_type)
            report = pipeline.generate(n_shortlist=n_per_class,
                                       target_class=tgt.label,
                                       target_stem=tgt.stem)
            shortlist_names = {c.name for c in report.shortlist}
            for r in report.rows():
                r.update({
                    "sweep_target": tgt.label,
                    "target_type": tgt.target_type,
                    "target_stem": tgt.stem,
                    "difficulty": tgt.difficulty,
                    "in_shortlist": r["candidate_name"] in shortlist_names,
                })
                rows.append(r)
            st = report.stats
            per_class.append({
                "sweep_target": tgt.label, "target_type": tgt.target_type,
                "target_stem": tgt.stem, "difficulty": tgt.difficulty,
                "evaluated": st["candidates_evaluated"],
                "admissible": st["admissible"],
                "admissible_rate": st["admissible_rate"],
                "band_low": st.get("band_low"), "band_moderate": st.get("band_moderate"),
                "returned": st["returned"], "best_quality": st["best_quality"],
                "mean_shortlist_quality": st["mean_shortlist_quality"],
                "verifier_calls": st["verifier_calls"], "llm_calls": st["llm_calls"],
                "wall_seconds": st["wall_seconds"], "error": None,
            })
            say(f"    {st['returned']}/{n_per_class} returned, "
                f"best quality {st['best_quality']}, {st['wall_seconds']}s")
        except Exception as exc:                            # noqa: BLE001
            if not continue_on_error:
                raise
            per_class.append({
                "sweep_target": tgt.label, "target_type": tgt.target_type,
                "target_stem": tgt.stem, "difficulty": tgt.difficulty,
                "evaluated": 0, "admissible": 0, "returned": 0,
                "wall_seconds": round(time.perf_counter() - t0, 2),
                "error": f"{type(exc).__name__}: {exc}"[:200],
            })
            say(f"    FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=1)

    df = pd.DataFrame(rows)
    summary = {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(time.perf_counter() - t_all, 1),
        "targets": len(targets),
        "n_per_class": n_per_class,
        "total_candidates": int(len(df)),
        "total_accepted": int(df["accepted"].sum()) if len(df) else 0,
        "overall_accept_rate": round(float(df["accepted"].mean()), 4) if len(df) else 0.0,
        "per_class": per_class,
        "manifest": system.manifest(),
    }

    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        pd.DataFrame(per_class).to_csv(
            out_csv.with_name(out_csv.stem + "_by_class.csv"), index=False)
        say(f"wrote {len(df)} rows to {out_csv}")

    return df, summary


# ===========================================================================
# Reading the sweep back
# ===========================================================================

def acceptance_by_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Does the difficulty stratification actually predict anything?

    If roomy, crowded and saturated classes all show the same acceptance rate, the
    stratification is decoration and should be dropped from the paper rather than
    presented as a finding.
    """
    g = df.groupby("difficulty").agg(
        candidates=("candidate_name", "count"),
        accepted=("accepted", "sum"),
        accept_rate=("accepted", "mean"),
        mean_quality=("quality_total", "mean"),
        mean_risk=("composite_risk_score", "mean"),
    ).round(3)
    return g.sort_values("accept_rate", ascending=False)


def proposer_contribution(df: pd.DataFrame) -> pd.DataFrame:
    """The comparison that the v1 `compare_strategies` was reaching for.

    This is where per-proposer comparison legitimately belongs: as an experiment over
    logged pool data, not as a production code path that forces the user to pick one
    proposer and discard the others.
    """
    base = df.copy()
    base["proposer_family"] = base["proposer"].str.replace("+refined", "", regex=False)
    g = base.groupby("proposer_family").agg(
        proposed=("candidate_name", "count"),
        accepted=("accepted", "sum"),
        accept_rate=("accepted", "mean"),
        mean_quality=("quality_total", "mean"),
        best_quality=("quality_total", "max"),
        shortlisted=("in_shortlist", "sum"),
    ).round(3)
    g["shortlist_share"] = (g["shortlisted"] / max(1, g["shortlisted"].sum())).round(3)
    return g.sort_values("shortlist_share", ascending=False)


def failure_profile(df: pd.DataFrame, top: int = 12) -> pd.DataFrame:
    """Which checks actually do the rejecting.

    A check that never fires is either redundant or mis-thresholded, and either way its
    presence in the paper's methodology section is not supported by the data.
    """
    codes: Dict[str, int] = {}
    for cell in df["failure_codes"].dropna():
        for c in str(cell).split("|"):
            if c:
                codes[c] = codes.get(c, 0) + 1
    out = pd.DataFrame(sorted(codes.items(), key=lambda kv: -kv[1])[:top],
                       columns=["failure_code", "count"])
    out["share_of_rejections"] = (out["count"] / max(1, int((~df["accepted"]).sum()))).round(3)
    return out


def quality_component_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """How much each quality term moves the composite.

    A component whose correlation with the total is near 1.0 is doing all the work and
    the others are decoration; a component near 0.0 is inert. Both are things to know
    before claiming a seven-term objective.
    """
    cols = [c for c in df.columns if c.startswith("q_")]
    if not cols or "quality_total" not in df:
        return pd.DataFrame()
    rows = []
    for c in cols:
        sub = df[[c, "quality_total"]].dropna()
        rows.append({
            "component": c[2:],
            "mean": round(float(sub[c].mean()), 3) if len(sub) else None,
            "std": round(float(sub[c].std()), 3) if len(sub) else None,
            "corr_with_total": round(float(sub[c].corr(sub["quality_total"])), 3)
            if len(sub) > 2 else None,
        })
    return pd.DataFrame(rows).sort_values("corr_with_total", ascending=False)
