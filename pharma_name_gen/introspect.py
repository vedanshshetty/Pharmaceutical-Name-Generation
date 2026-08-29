"""
Introspection — `model.summary()` for a pipeline rather than a network.

Purpose
-------
The notebook has to explain the system to someone who did not build it, and prose alone
is not evidence that the prose is true. These functions print the *live* configuration:
the actual corpus sizes, the actual weights, the actual thresholds, the actual proposer
wiring, read out of the constructed objects at run time. If a number in the architecture
section is wrong, it is wrong because the system is wired that way, not because the
documentation drifted.

Every table here is derived, never hardcoded. Change a weight in `quality.py` and the
architecture summary in the notebook changes with it.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

import pandas as pd


# ===========================================================================
# Architecture
# ===========================================================================

def architecture_summary(system) -> str:
    """Layer-by-layer summary of the constructed system, Keras-style."""
    m = system.manifest()
    d, sc, tr = m["data"], m["screening_corpus"], m["training_corpus"]
    v = system.verifier.config
    rows = [
        ("DATA", "DataSnapshot",
         f"mode={d['mode']}  sources={len(d['sources'])}  fingerprint={d['fingerprint']}",
         f"{d['name_rows']} rows"),
        ("DATA", "ScreeningCorpus",
         f"generic={sc['kept_generic']}  brand={sc['kept_brand']}  "
         f"filtered={sc.get('tokens_filtered', 0)}",
         f"{sc['kept_total_unique']} names"),
        ("DATA", "TrainingCorpus",
         f"prefixes={tr['unique_prefixes']}  stem_coverage={tr['stem_coverage']:.0%}  "
         f"blocklist={tr['blocklisted_morphemes']}",
         f"{tr['generic_tokens']} tokens"),
        ("DATA", "StemTable", "INN/USAN stems, suffix + embedded matching",
         f"{d['stem_rows']} stems"),
        ("PROPOSE", "GrammarProposer",
         f"induced syllable grammar, corpus-derived inventory",
         "free / CPU"),
        ("PROPOSE", "NGramProposer",
         f"order-3 char model, guided={system.config.guided}, "
         f"reward floor + {system.config.reward_guided_draws} draws/slot",
         "free / CPU"),
        ("SCREEN", "Verifier",
         f"similarity(moderate={v.thresholds.similarity_moderate}, "
         f"high={v.thresholds.similarity_high})  stem  trademark  phonotactics  "
         f"crosslingual",
         f"v{system.verifier.__class__.__module__.split('.')[-1]}"),
        ("SCORE", "QualityScorer",
         f"{len(system.scorer.generic_weights)} generic terms / "
         f"{len(system.scorer.brand_weights)} brand terms",
         "0-100"),
        ("SELECT", "NominaPipeline",
         f"pool={system.config.pool_per_proposer}/proposer x "
         f"{system.config.max_rounds} rounds, refine top "
         f"{system.config.refine_top_k}, select by quality",
         f"seed={system.config.seed}"),
    ]
    width = [8, 18, 62, 16]
    out = ["=" * sum(width),
           f"{'STAGE':<{width[0]}}{'COMPONENT':<{width[1]}}{'CONFIGURATION':<{width[2]}}{'SIZE / COST':<{width[3]}}",
           "=" * sum(width)]
    last = None
    for stage, comp, cfg, size in rows:
        if last and stage != last:
            out.append("-" * sum(width))
        out.append(f"{stage if stage != last else '':<{width[0]}}"
                   f"{comp:<{width[1]}}{cfg[:width[2]-1]:<{width[2]}}{size:<{width[3]}}")
        last = stage
    out.append("=" * sum(width))
    out.append(f"Generative-Verifier v{m['system_version']}   git={m['git_sha'] or 'n/a'}   "
               f"schema={system.verifier.verify('a' * 6, target_type='brand').verifier_version}")
    return "\n".join(out)


def data_flow() -> str:
    """The request path, as text so it survives every rendering surface."""
    return """
      request(target_type, class, stem, N)
                    |
        +-----------+-----------+
        |                       |            <- run in PARALLEL, every request
   GrammarProposer         NGramProposer          (nothing is discarded unseen)
   (induced syllable        (order-3 chars,
    grammar)                guided by this
        |                    run's rejections)
        +-----------+-----------+
                    |
              CANDIDATE POOL
                    |
              Verifier.verify()   <- the WHOLE pool, not the first success
                    |
        +-----------+-----------+
        |                       |
   rejected                 admissible
        |                       |
  structured feedback           |          <- payloads, not prose
  -> bigram penalties           |
  -> temperature ramp           |
  -> refine top-k               |
        |                       |
        +-----------+-----------+
                    |
            QualityScorer.score()
                    |
             SELECT top N by quality
                    |
               shortlist
"""


# ===========================================================================
# Design decisions
# ===========================================================================

DESIGN_DECISIONS: List[Dict[str, str]] = [
    {
        "area": "Architecture",
        "decision": "Parallel pool-and-select, not four peer strategies and not a cascade",
        "alternative": "Early-exit cascade: try the cheapest proposer, stop on first pass",
        "reasoning": "A cascade optimises for cost, not quality. If the first proposer "
                     "clears the bar with margin 1.04 the cascade stops and never learns "
                     "that another proposer had margin 15 on the same call. Pooling "
                     "verifies everything before choosing, so selection cannot be a "
                     "local optimum of whichever proposer happened to run first.",
        "evidence": "v1 shortlists were whatever passed first; v2 selects on a scored pool.",
    },
    {
        "area": "Architecture",
        "decision": "`rl_refined` demoted from a strategy to a batch-drawing policy",
        "alternative": "Keep it as a fourth peer value of `generation_strategy`",
        "reasoning": "It used the same n-gram model as `rejection_sampling` plus a "
                     "bigram penalty and a temperature ramp. That is a boolean on one "
                     "sampler, not a fourth mechanism. On a single fresh call with no "
                     "prior rejections it is byte-identical to plain sampling; its value "
                     "only exists across a batch.",
        "evidence": "Same model object, two extra parameters.",
    },
    {
        "area": "Generation",
        "decision": "Induce the syllable grammar from the corpus",
        "alternative": "Hand-written onset/nucleus/coda inventory (what v1 had)",
        "reasoning": "The hand-written inventory encoded English orthography (nuclei "
                     "`ou`, `ea`, `ei`; codas `lt`, `nd`) and produced `skemkultolol` "
                     "and `jeimheistolol`. Induced from real fantasy prefixes it "
                     "produces `lelma`, `monite`, `veldo`. Recombining at the syllable "
                     "level also makes whole-morpheme copying structurally impossible.",
        "evidence": "Compare v1 and v2 shortlists on the same class and seed.",
    },
    {
        "area": "Data",
        "decision": "Expand the stem table from 34 to 277 INN/USAN stems",
        "alternative": "Keep the seed table and accept low coverage",
        "reasoning": "With 34 stems only 86 fantasy prefixes could be extracted from the "
                     "corpus. An order-3 character model trained on 86 short strings "
                     "memorises, which is where `erythroolol` and `amoxiolol` came from. "
                     "Expanding the table (plus splitting multi-word entries) took the "
                     "pool to 330.",
        "evidence": "86 -> 330 unique training prefixes.",
    },
    {
        "area": "Data",
        "decision": "Live-first with static fallback, and always report which ran",
        "alternative": "Static committed CSV only, or live only",
        "reasoning": "Static only means the screen ages badly and misses new entrants. "
                     "Live only means a rate-limited API kills the demo. Silent fallback "
                     "is the worst option of the three, because degrading from a 40k-name "
                     "universe to a 2k one quietly inflates every distinctiveness margin "
                     "in the run. So it falls back, and it says so.",
        "evidence": "`DataSnapshot.mode` on every manifest.",
    },
    {
        "area": "Screening",
        "decision": "Report both margins and an explicit risk band",
        "alternative": "Single margin to the hard cutoff (what v1 did)",
        "reasoning": "v1 reported `70 - score`, so a candidate at 57.0 advertised a "
                     "margin of 13.03 and read as comfortable while sitting inside the "
                     "55-70 band POCA designates for review. Every accepted v1 name was "
                     "in that band and nothing in the output said so.",
        "evidence": "`erythroolol`: margin_to_reject 13.03, margin_to_review -1.97.",
    },
    {
        "area": "Screening",
        "decision": "Detect foreign class stems inside compliant generic names",
        "alternative": "Check the terminal stem only",
        "reasoning": "`cillinolol` and `prazololol` both passed v1: correct terminal "
                     "stem, contradictory internal one. The INN system treats a "
                     "misleading internal stem the same way it treats a misleading "
                     "terminal one, so this was a conformance gap.",
        "evidence": "New failure code `STEM_FOREIGN_EMBEDDED`.",
    },
    {
        "area": "Policy",
        "decision": "The 55-70 band is admissible-with-review, not rejected",
        "alternative": "Treat moderate as failure",
        "reasoning": "Tried and reverted. Refusing the band dropped the admissible rate "
                     "to 0.5%, because in a stem-governed class the stem itself forces "
                     "high orthographic similarity to siblings, so almost every "
                     "plausible name lands there. The band is reported instead, and the "
                     "quality objective scores headroom against 55 so grey-band names "
                     "rank below clear ones automatically.",
        "evidence": "Configurable via `PipelineConfig.treat_moderate_as_failure`.",
    },
    {
        "area": "Objective",
        "decision": "Quality is orthogonal to and downstream of the verifier",
        "alternative": "Fold quality terms into the admissibility decision",
        "reasoning": "The verifier's claim is regulatory: it answers whether a name may "
                     "exist. Quality answers whether anyone would want it. Mixing them "
                     "would make the regulatory claim a preference model, which is not "
                     "defensible in a paper or in front of a reviewer. Quality ranks "
                     "what survives; it never rescues a rejection.",
        "evidence": "`QualityReport` is attached to the response, never consulted by "
                    "`overall_pass`.",
    },
    {
        "area": "Objective",
        "decision": "Typicality is scored as a band, not a maximum",
        "alternative": "Reward corpus likelihood monotonically",
        "reasoning": "Maximising corpus likelihood is exactly the objective that "
                     "produces memorised prefixes. The reward peaks mid-distribution and "
                     "falls off on both sides: too improbable reads as random letters, "
                     "too probable means the model reproduced its training data.",
        "evidence": "`score_typicality` targets 0.40-0.70.",
    },
    {
        "area": "Objective",
        "decision": "Read pure phonotactics, not the verifier's blended score",
        "alternative": "Use `checks.pronounceability.score` directly",
        "reasoning": "That score is 0.6*phonotactic + 0.4*corpus_typicality. Ranking on "
                     "it would have the objective rewarding corpus-hugging through the "
                     "pronounceability term while punishing it through the novelty term, "
                     "via two components that look independent and are not.",
        "evidence": "`score_pronounceability` reads `details['phonotactic']`.",
    },
    {
        "area": "Objective",
        "decision": "Separate generic and brand pipelines",
        "alternative": "One pipeline with a target_type branch",
        "reasoning": "The objectives are opposed. A generic name must carry its class "
                     "stem and be systematically unmemorable; a brand name must carry no "
                     "stem at all and be memorable. Shared weights would optimise both "
                     "toward the average of two incompatible targets.",
        "evidence": "`GENERIC_WEIGHTS` vs `BRAND_WEIGHTS`; `stem_avoidance` and "
                    "`memorability` exist only for brand.",
    },
    {
        "area": "Reproducibility",
        "decision": "Content-addressed artifacts keyed on a corpus fingerprint",
        "alternative": "Time-based cache expiry, or no cache",
        "reasoning": "Time-based expiry means someone has to remember to clear the "
                     "cache, and nobody does. Keying on a hash of the corpus plus the "
                     "relevant config means changing the stem table produces a different "
                     "key and the stale model is simply not found.",
        "evidence": "`artifacts.key_for()`.",
    },
    {
        "area": "Evaluation",
        "decision": "Log every attempt, not just the winners",
        "alternative": "Export the shortlist",
        "reasoning": "A winners-only table cannot answer the question a reviewer asks, "
                     "which is not 'what did it produce' but 'what did it produce "
                     "relative to what it tried'. Acceptance rate, failure-code "
                     "distribution and per-proposer contribution are all invisible "
                     "otherwise.",
        "evidence": "`sweep_all_attempts.csv`, one row per candidate ever evaluated.",
    },
]


def design_decisions(area: Optional[str] = None) -> pd.DataFrame:
    rows = DESIGN_DECISIONS
    if area:
        rows = [r for r in rows if r["area"].lower() == area.lower()]
    return pd.DataFrame(rows)


# ===========================================================================
# Component detail
# ===========================================================================

def quality_weights_table(system) -> pd.DataFrame:
    """The objective, printed from the live weights."""
    g, b = system.scorer.generic_weights, system.scorer.brand_weights
    what = {
        "distinctiveness": "headroom below the 55 review line (not the 70 reject line)",
        "novelty": "penalises verbatim reuse of a fragment of a real name",
        "morpheme_hygiene": "penalises carrying another class's stem",
        "stem_avoidance": "a stem in a brand name is a false class claim",
        "pronounceability": "phonotactic legality + articulatory ease",
        "memorability": "short, low syllable count, clean CV alternation",
        "shape": "length and syllables vs the real distribution for this profile",
        "seam": "orthographic hygiene at the prefix/stem join",
        "typicality": "plausible as a drug name without being derivative",
    }
    keys = sorted(set(g) | set(b))
    return pd.DataFrame([{
        "component": k,
        "generic_weight": g.get(k, 0.0),
        "brand_weight": b.get(k, 0.0),
        "measures": what.get(k, ""),
    } for k in keys]).sort_values("generic_weight", ascending=False)


def threshold_table(system) -> pd.DataFrame:
    t = system.verifier.config.thresholds
    return pd.DataFrame([{"threshold": k, "value": v}
                         for k, v in vars(t).items()
                         if isinstance(v, (int, float))])


def corpus_table(system) -> pd.DataFrame:
    d, sc, tr = (system.snapshot.manifest(), system.screening.stats,
                 system.training.stats)
    return pd.DataFrame([
        {"quantity": "raw snapshot rows", "value": d["name_rows"],
         "note": f"mode={d['mode']}"},
        {"quantity": "screening universe (unique names)", "value": sc["kept_total_unique"],
         "note": "what the verifier measures distance from"},
        {"quantity": "noise tokens filtered", "value": sc.get("tokens_filtered", 0),
         "note": "homeopathic Latin, product descriptors, dosage forms"},
        {"quantity": "generic tokens (training)", "value": tr["generic_tokens"],
         "note": "multi-word entries split, not dropped"},
        {"quantity": "fantasy prefixes (unique)", "value": tr["unique_prefixes"],
         "note": "v1 had 86; this is the n-gram training set"},
        {"quantity": "stem coverage", "value": f"{tr['stem_coverage']:.1%}",
         "note": f"{tr['stems_available']} stems available"},
        {"quantity": "brand marks (training)", "value": tr["brand_names"], "note": ""},
        {"quantity": "blocklisted morphemes", "value": tr["blocklisted_morphemes"],
         "note": "class-signalling fragments a new name must not reuse"},
    ])


def source_code(obj) -> str:
    """Show the actual implementation of a function or class in the notebook.

    Used so the architecture section can display the real code for the two or three
    pieces that carry the argument, rather than asking the reader to take prose on
    trust."""
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return f"<source unavailable for {obj!r}>"
