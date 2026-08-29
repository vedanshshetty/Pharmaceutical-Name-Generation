"""
System builder — one call that wires everything, with caching and a manifest.

This is the single seam between "the modules" and "a working system". It exists so that
the notebook, the Streamlit app, the sweep harness and the test suite all construct the
system identically. Divergent construction is how two halves of a project end up
silently disagreeing about what corpus they are using, which is exactly what happened in
v1 (verifier: 1,918 names; generator: 420; nobody noticed).
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import data_layer as dl
from .artifacts import (
    ArtifactStore, dump_ngram, dump_shape, key_for, load_ngram, load_shape,
)
from .contracts import TargetType
from .corpus import build_screening_corpus, build_training_corpus, siblings_for_stem
from .orchestrator import (
    GrammarProposer, NGramProposer, NominaPipeline, PipelineConfig,
)
from .phonotactics import InducedGrammar
from .quality import QualityScorer, ShapeReference

VERSION = "2.1.0"


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=Path(__file__).resolve().parent.parent,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:                                       # noqa: BLE001
        return None


@dataclass
class NominaSystem:
    """A fully wired system plus the manifest that makes a run citable."""
    snapshot: Any
    screening: Any
    training: Any
    verifier: Any
    scorer: QualityScorer
    generic: NominaPipeline
    brand: NominaPipeline
    store: ArtifactStore
    config: PipelineConfig
    build_log: List[str] = field(default_factory=list)
    built_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def pipeline(self, target_type: str) -> NominaPipeline:
        return self.generic if str(target_type).lower() == "generic" else self.brand

    def manifest(self) -> Dict[str, Any]:
        """Everything a reviewer needs to establish that two runs are comparable."""
        return {
            "system_version": VERSION,
            "git_sha": _git_sha(),
            "built_at": self.built_at,
            "data": self.snapshot.manifest(),
            "screening_corpus": self.screening.stats,
            "training_corpus": self.training.stats,
            "verifier_config": self.verifier.config.to_dict(),
            "pipeline_config": self.config.to_dict(),
            "quality_weights": {
                "generic": self.scorer.generic_weights,
                "brand": self.scorer.brand_weights,
            },
            "shape_reference": {
                "generic": vars(self.scorer.shape_generic),
                "brand": vars(self.scorer.shape_brand),
            },
        }

    def summary(self) -> str:
        m = self.manifest()
        return "\n".join([
            f"Generative-Verifier v{VERSION}   git={m['git_sha'] or 'n/a'}",
            self.snapshot.summary(),
            f"  screening universe : {self.screening.stats['kept_total_unique']} names",
            f"  training prefixes  : {self.training.stats['unique_prefixes']} unique "
            f"(stem coverage {self.training.stats['stem_coverage']:.0%} of "
            f"{self.training.stats['generic_tokens']} generic tokens)",
            f"  brand training set : {self.training.stats['brand_names']} marks",
            f"  shape target       : generic "
            f"{self.scorer.shape_generic.mean_len:.1f} chars / "
            f"{self.scorer.shape_generic.mean_syl:.1f} syll, brand "
            f"{self.scorer.shape_brand.mean_len:.1f} chars / "
            f"{self.scorer.shape_brand.mean_syl:.1f} syll",
        ])


def build_system(live: bool = True,
                 config: Optional[PipelineConfig] = None,
                 verifier_config: Optional[Any] = None,
                 use_artifacts: bool = True,
                 ndc_limit: int = 20000,
                 progress: Optional[Callable[[str], None]] = None) -> NominaSystem:
    """Construct the whole system.

    `live=True` attempts the regulator feeds and falls back to the committed snapshot on
    any failure. `use_artifacts=True` resolves derived models from the content-addressed
    cache, so a warm run skips both the network and the fits. Neither is required: with
    both off, this builds deterministically from committed files with no network at all,
    which is the mode CI uses.
    """
    say = progress or (lambda _m: None)
    cfg = config or PipelineConfig()
    store = ArtifactStore(allow_remote=use_artifacts)
    log: List[str] = []

    # -- data ---------------------------------------------------------------
    t0 = time.perf_counter()
    snap_key = key_for("snapshot", "live" if live else "static", {"ndc_limit": ndc_limit})
    if use_artifacts and live:
        from .artifacts import dump_snapshot, load_snapshot
        from .data_layer import DataSnapshot, SourceRecord
        snapshot = store.load_or_build(
            snap_key,
            build=lambda: dl.build_snapshot(live=True, ndc_limit=ndc_limit, verbose=False),
            dump=dump_snapshot,
            load=lambda p: load_snapshot(p, DataSnapshot, SourceRecord),
        )
    else:
        snapshot = dl.get_snapshot(live=live, ndc_limit=ndc_limit)
    log.append(f"data snapshot [{snapshot.mode}] in {time.perf_counter()-t0:.1f}s")
    say(log[-1])

    # -- corpora ------------------------------------------------------------
    screening = build_screening_corpus(snapshot.names)
    training = build_training_corpus(snapshot.names, snapshot.stems)
    log.append(f"corpora: {screening.stats['kept_total_unique']} screening names, "
               f"{training.stats['unique_prefixes']} training prefixes")
    say(log[-1])

    # -- verifier -----------------------------------------------------------
    from .verifier import (
        StemTable, Verifier, VerifierConfig, grapheme_to_phonemes, phonotactic_score,
        syllabify,
    )
    vcfg = verifier_config or VerifierConfig()
    verifier = Verifier(
        corpus={"generic": screening.generic, "brand": screening.brand,
                "all": screening.all, "source": screening.source,
                "stats": screening.stats},
        stem_table=StemTable.from_dataframe(snapshot.stems),
        config=vcfg,
    )
    log.append(f"verifier v{verifier.__class__.__module__} built over "
               f"{len(screening.all)} names")

    def syllable_count(name: str) -> int:
        return len(syllabify(grapheme_to_phonemes(name)))

    # -- language models (cached) ------------------------------------------
    from .generator import (
        CharNGramModel, strategy_rejection_sampling,
    )
    lm_cfg = {"order": 3, "k": 0.35}

    def _lm(kind: str, tokens: List[str]):
        k = key_for(f"ngram_{kind}", snapshot.fingerprint, lm_cfg)
        if not use_artifacts:
            return CharNGramModel(tokens, **lm_cfg)
        return store.load_or_build(
            k,
            build=lambda: CharNGramModel(tokens, **lm_cfg),
            dump=dump_ngram,
            load=lambda p: load_ngram(p, CharNGramModel),
        )

    lm_prefix = _lm("prefix", training.prefixes)
    lm_brand = _lm("brand", training.brand_names)

    # -- induced syllable grammars (cached) --------------------------------
    def _grammar(kind: str, tokens: List[str]):
        k = key_for(f"grammar_{kind}", snapshot.fingerprint, {"min_count": 2})
        if not use_artifacts:
            return InducedGrammar.induce(tokens)
        return store.load_or_build(
            k,
            build=lambda: InducedGrammar.induce(tokens),
            dump=lambda g: {"kind": "induced_grammar", **g.to_dict()},
            load=lambda p: InducedGrammar.from_dict(p),
        )

    grammar_prefix = _grammar("prefix", training.prefixes)
    grammar_brand = _grammar("brand", training.brand_names)
    log.append(f"induced grammars: {len(grammar_prefix.legal_onsets)} attested onsets "
               f"(generic), {len(grammar_brand.legal_onsets)} (brand)")

    # -- shape references (cached) -----------------------------------------
    def _shape(kind: str, names: List[str]):
        k = key_for(f"shape_{kind}", snapshot.fingerprint, {})
        if not use_artifacts:
            return ShapeReference.from_names(names, syllable_count)
        return store.load_or_build(
            k,
            build=lambda: ShapeReference.from_names(names, syllable_count),
            dump=dump_shape,
            load=lambda p: load_shape(p, ShapeReference),
        )

    # Sampling-time phonotactics. Returns the legality score and the syllable
    # structures, which is exactly what the quality objective's pronounceability term
    # consumes, so the reward used during proposal and the score used during selection
    # measure the same quantity rather than two subtly different ones.
    def pron_raw(name: str):
        phones = grapheme_to_phonemes(name)
        score, diag = phonotactic_score(phones)
        return score, diag.get("syllables", [])

    scorer = QualityScorer(
        screening_corpus=screening, training_corpus=training,
        syllable_fn=syllable_count, lm_generic=lm_prefix, lm_brand=lm_brand,
        moderate_cutoff=vcfg.thresholds.similarity_moderate,
        high_cutoff=vcfg.thresholds.similarity_high,
        pron_fn=pron_raw,
    )
    scorer.shape_generic = _shape("generic",
                                  [n for n in training.generic_tokens if 5 <= len(n) <= 18])
    scorer.shape_brand = _shape("brand", training.brand_names)

    # -- pipelines ----------------------------------------------------------
    from .generator import GeneratorConfig, refine_candidate
    gcfg = GeneratorConfig()

    # Prefix length window taken from the data rather than from the v1 default of 3-9.
    # Real fantasy prefixes are short: the `-olol` class runs 3 to 7 characters, and a
    # 9-character prefix produces a 13-character generic name that no real INN name
    # matches. Using the corpus interquartile range keeps generated names inside the
    # distribution the shape term is scoring them against.
    _plens = sorted(len(p) for p in training.prefixes) or [3, 4, 5, 6]
    prefix_min = max(3, _plens[len(_plens) // 20])
    prefix_max = max(prefix_min + 3, _plens[int(len(_plens) * 0.90)])
    log.append(f"prefix length window from corpus: {prefix_min}-{prefix_max} "
               f"(v1 default was {gcfg.min_fantasy_prefix}-{gcfg.max_fantasy_prefix})")

    generic_pipeline = NominaPipeline(
        target_type=TargetType.GENERIC, verifier=verifier, scorer=scorer,
        proposers=[
            GrammarProposer(grammar_prefix, prefix_min, prefix_max,
                            max_syllables=3, scorer=scorer,
                            draws=cfg.reward_guided_draws),
            NGramProposer(strategy_rejection_sampling, lm_prefix,
                          prefix_min, prefix_max,
                          cfg, scorer=scorer, guided=cfg.guided),
        ],
        refine_fn=refine_candidate, config=cfg,
        sibling_fn=lambda stem: siblings_for_stem(training, stem),
    )

    brand_pipeline = NominaPipeline(
        target_type=TargetType.BRAND, verifier=verifier, scorer=scorer,
        proposers=[
            # Brand marks are capped at three syllables: the induced grammar will happily
            # produce `futatafima` from the brand corpus, which is well-formed and
            # commercially useless. Memorability is the objective here, not systematicity.
            GrammarProposer(grammar_brand, gcfg.min_brand_length,
                            gcfg.max_brand_length, max_syllables=3,
                            scorer=scorer, draws=cfg.reward_guided_draws),
            NGramProposer(strategy_rejection_sampling, lm_brand,
                          gcfg.min_brand_length, gcfg.max_brand_length,
                          cfg, scorer=scorer, guided=cfg.guided),
        ],
        refine_fn=refine_candidate, config=cfg,
        sibling_fn=lambda stem: [],
    )

    return NominaSystem(
        snapshot=snapshot, screening=screening, training=training, verifier=verifier,
        scorer=scorer, generic=generic_pipeline, brand=brand_pipeline, store=store,
        config=cfg, build_log=log,
    )
