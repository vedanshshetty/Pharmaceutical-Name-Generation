"""
NOMINA test suite.

Organised around the defects that were actually found, not around code coverage. Every
test in `TestRegressions` corresponds to a specific thing the v1 pipeline got wrong and
shipped; if one of them goes red, a known failure has come back.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nomina.contracts import (
    SCHEMA_VERSION, CandidateRequest, FailureCode, RiskBand, TargetType, VerifierResponse,
)
from nomina.corpus import build_screening_corpus, build_training_corpus, fold, tokenise
from nomina.phonotactics import InducedGrammar, parse_syllables
from nomina.quality import SubstringIndex, score_seam, score_novelty


# ===========================================================================
# Regressions — one test per defect found in the v1 pipeline
# ===========================================================================

class TestRegressions:

    def test_foreign_stem_inside_generic_is_rejected(self, system):
        """`cillinolol` passed v1: a beta-blocker announcing itself as a penicillin.

        The stem check only inspected the suffix, so a compliant terminal stem hid a
        contradictory internal one.
        """
        r = system.verifier.verify("cillinolol", target_type="generic",
                                   target_class="beta-blocker", target_stem="-olol")
        assert FailureCode.STEM_FOREIGN_EMBEDDED in r.failure_codes
        assert not r.overall_pass

    def test_correct_stem_alone_is_not_flagged_as_foreign(self, system):
        """The mandated stem must never be counted against the name that carries it."""
        r = system.verifier.verify("bexolol", target_type="generic",
                                   target_class="beta-blocker", target_stem="-olol")
        assert FailureCode.STEM_FOREIGN_EMBEDDED not in r.failure_codes

    def test_grey_band_is_reported_not_hidden(self, system):
        """v1 reported margin to the 70 cutoff only, so a name at 57 advertised a
        margin of 13 while sitting inside the 55-70 review band."""
        r = system.verifier.verify("erythroolol", target_type="generic",
                                   target_class="beta-blocker", target_stem="-olol")
        sim = r.checks.similarity
        assert sim.distinctiveness_margin is not None
        assert sim.distinctiveness_margin_moderate is not None
        assert sim.distinctiveness_margin_moderate < sim.distinctiveness_margin
        assert r.risk_band in (RiskBand.LOW, RiskBand.MODERATE, RiskBand.HIGH)

    def test_memorised_morpheme_is_penalised(self, system):
        """`erythroolol` lifts seven characters from erythromycin. No similarity metric
        catches it once the stem differs, so the novelty term has to."""
        index = SubstringIndex(system.screening.all)
        borrowed, _ = score_novelty("erythroolol", index, "olol")
        invented, _ = score_novelty("bexolol", index, "olol")
        assert borrowed < 0.5
        assert invented > borrowed

    def test_vowel_collision_at_stem_seam_is_penalised(self, system):
        """`acycloolol` and `clohaolol`: a vowel-final prefix glued to a vowel-initial
        stem, which no real INN name does."""
        bad, detail = score_seam("acycloolol", "olol")
        good, _ = score_seam("bexolol", "olol")
        assert bad < good
        assert any("seam" in f or "vowel" in f for f in detail["faults"])

    def test_generator_never_emits_vowel_final_prefix_before_vowel_stem(self, system):
        report = system.generic.generate(n_shortlist=5, target_class="beta-blocker",
                                         target_stem="-olol")
        for c in report.shortlist:
            prefix = c.name[: -len("olol")]
            assert prefix and prefix[-1] not in "aeiou", f"{c.name} has a vowel seam"

    def test_training_corpus_is_large_enough_to_generalise(self, system):
        """v1 trained an order-3 character model on 86 strings, which memorises rather
        than generalises. The expanded stem table and multi-word splitting fixed it."""
        assert system.training.stats["unique_prefixes"] > 250
        assert system.snapshot.stems.shape[0] > 200

    def test_pronounceability_excludes_corpus_typicality(self, system):
        """The verifier's blended score is 0.6*phonotactic + 0.4*typicality. Ranking on
        it would have the objective rewarding corpus-hugging through one term while
        punishing it through another."""
        from nomina.quality import score_pronounceability
        r = system.verifier.verify("metoprolol", target_type="generic", target_stem="-olol")
        _, detail = score_pronounceability(r)
        assert "phonotactic" in detail and "articulatory_ease" in detail
        assert detail["phonotactic"] != detail["blended_verifier_score"]


# ===========================================================================
# Contracts
# ===========================================================================

class TestContracts:

    def test_schema_version_bumped(self):
        assert SCHEMA_VERSION == "1.1.0"

    def test_response_round_trips_through_json(self, system):
        r = system.verifier.verify("bexolol", target_type="generic", target_stem="-olol")
        restored = VerifierResponse.model_validate_json(r.model_dump_json())
        assert restored.candidate_name == r.candidate_name
        assert restored.risk_band == r.risk_band
        assert restored.composite_risk_score == pytest.approx(r.composite_risk_score)

    def test_refinement_signals_are_machine_readable(self, system):
        """The generator reads `payload`, never `human_readable`. If payloads were
        empty the feedback loop would silently become a no-op."""
        r = system.verifier.verify("atenolol", target_type="generic", target_stem="-olol")
        assert r.refinement_feedback
        assert any(s.payload for s in r.refinement_feedback)

    def test_exact_collision_with_marketed_name_fails(self, system):
        r = system.verifier.verify("metoprolol", target_type="generic", target_stem="-olol")
        assert not r.overall_pass
        assert r.risk_band == RiskBand.HIGH


# ===========================================================================
# Corpus and data layer
# ===========================================================================

class TestCorpus:

    def test_fold_is_idempotent(self):
        assert fold("Metoprolol") == "metoprolol"
        assert fold(fold("Ácido-Fólico")) == fold("Ácido-Fólico")

    def test_multiword_generics_are_split_not_dropped(self):
        """v1 discarded every multi-word entry, which is what shrank the training pool."""
        assert tokenise("acetaminophen, dextromethorphan hydrobromide") == [
            "acetaminophen", "dextromethorphan", "hydrobromide"]

    def test_homeopathic_latin_is_filtered_from_screening(self, system):
        """`sativus` and `latifolia` as nearest neighbours are noise, not protection."""
        corpus = set(system.screening.all)
        for junk in ("sativus", "latifolia", "cysteinum", "yellow"):
            assert junk not in corpus

    def test_real_marketed_names_survive_filtering(self, system):
        """The filter must not be so aggressive that it deletes the things we screen
        against. This is the guard on the previous test."""
        corpus = set(system.screening.all)
        for real in ("metoprolol", "tylenol", "advil", "atenolol", "ibuprofen"):
            assert real in corpus, f"{real} was filtered out of the screening universe"

    def test_snapshot_carries_provenance(self, system):
        m = system.snapshot.manifest()
        assert m["sources"] and m["fingerprint"]
        assert m["mode"] in ("live", "static")

    def test_fingerprint_is_content_addressed(self, system):
        """Artifact cache keys hang off this. If it were not content-derived, a corpus
        change would silently reuse a model trained on the old one."""
        import pandas as pd
        before = system.snapshot.fingerprint
        assert before == system.snapshot.fingerprint
        mutated = type(system.snapshot)(
            names=pd.concat([system.snapshot.names,
                             pd.DataFrame([{"generic_name": "zzzznewdrug"}])],
                            ignore_index=True),
            stems=system.snapshot.stems, sources=system.snapshot.sources)
        assert mutated.fingerprint != before


# ===========================================================================
# Induced grammar
# ===========================================================================

class TestPhonotactics:

    def test_grammar_is_induced_from_data(self, system):
        g = InducedGrammar.induce(system.training.prefixes)
        assert g.n_train > 100
        assert g.legal_onsets

    def test_samples_are_well_formed(self, system):
        import random
        g = InducedGrammar.induce(system.training.prefixes)
        rng = random.Random(1)
        for _ in range(200):
            s = g.sample(rng, 3, 8)
            if not s:
                continue
            assert "aaa" not in s and "eee" not in s
            assert not any(len(r) > 3 for r in
                           __import__("re").findall(r"[bcdfghjklmnpqrstvwxyz]+", s))

    def test_final_coda_constraint_is_honoured(self, system):
        import random
        g = InducedGrammar.induce(system.training.prefixes)
        rng = random.Random(2)
        for _ in range(100):
            s = g.sample(rng, 3, 8, final_coda=True)
            if s:
                assert s[-1] not in "aeiou"

    def test_grammar_serialises_losslessly(self, system):
        g = InducedGrammar.induce(system.training.prefixes)
        g2 = InducedGrammar.from_dict(g.to_dict())
        assert g2.legal_onsets == g.legal_onsets
        assert g2.n_train == g.n_train

    def test_syllabification_matches_intuition(self):
        assert [str(s) for s in parse_syllables("metopr", {"m", "t", "pr", ""})] == ["me", "topr"]


# ===========================================================================
# Pipeline behaviour
# ===========================================================================

class TestPipeline:

    def test_shortlist_is_ordered_by_quality(self, system):
        r = system.generic.generate(n_shortlist=6, target_class="beta-blocker",
                                    target_stem="-olol")
        scores = [c.quality.total for c in r.shortlist]
        assert scores == sorted(scores, reverse=True)

    def test_all_shortlisted_names_carry_the_required_stem(self, system):
        r = system.generic.generate(n_shortlist=6, target_class="ACE inhibitor",
                                    target_stem="-pril")
        assert all(c.name.endswith("pril") for c in r.shortlist)

    def test_brand_names_carry_no_stem(self, system):
        r = system.brand.generate(n_shortlist=6, target_class="proprietary mark")
        stems = [s for s in system.training.stem_index if len(s) >= 4]
        for c in r.shortlist:
            assert not any(c.name.endswith(s) for s in stems), c.name

    def test_both_proposers_contribute_to_the_pool(self, system):
        """Pool-and-select is pointless if one proposer never produces anything usable.
        The n-gram sat at a 0% accept rate until a reward floor was added inside the
        sampler."""
        r = system.generic.generate(n_shortlist=10, target_class="beta-blocker",
                                    target_stem="-olol")
        families = {c.proposer.replace("+refined", "") for c in r.all_candidates if c.accepted}
        assert len(families) >= 2, f"only {families} produced admissible candidates"

    def test_nothing_is_discarded_before_being_verified(self, system):
        """The core architectural claim: every pooled candidate is scored, so selection
        cannot be a local optimum of whichever proposer happened to go first."""
        r = system.generic.generate(n_shortlist=5, target_class="beta-blocker",
                                    target_stem="-olol")
        assert all(c.response is not None and c.quality is not None
                   for c in r.all_candidates)
        assert r.stats["verifier_calls"] >= r.stats["candidates_evaluated"]

    def test_run_is_deterministic_under_a_fixed_seed(self, system):
        a = system.generic.generate(n_shortlist=5, target_class="beta-blocker",
                                    target_stem="-olol")
        b = system.generic.generate(n_shortlist=5, target_class="beta-blocker",
                                    target_stem="-olol")
        assert [c.name for c in a.shortlist] == [c.name for c in b.shortlist]

    def test_report_exports_every_attempt(self, system):
        r = system.generic.generate(n_shortlist=5, target_class="beta-blocker",
                                    target_stem="-olol")
        df = r.to_frame()
        assert len(df) == len(r.all_candidates)
        assert "accepted" in df.columns and "quality_total" in df.columns
        assert df["accepted"].sum() >= len(r.shortlist)

    def test_llm_absence_is_not_an_error(self, system):
        """No key, no network, no LLM: the run must still complete on the free pool."""
        assert system.generic.llm is None
        r = system.generic.generate(n_shortlist=3, target_class="beta-blocker",
                                    target_stem="-olol")
        assert r.shortlist
        assert r.stats["llm_calls"] == 0


# ===========================================================================
# Quality objective
# ===========================================================================

class TestQuality:

    def test_v2_output_beats_v1_output_on_the_same_objective(self, system):
        """The headline claim, as a test rather than an assertion."""
        from nomina.evaluation import compare_architectures
        result = compare_architectures(system, n=10)
        summary = result["summary"]
        v1 = summary.loc["v1 (four independent strategies)", "mean_quality"]
        v2 = summary.loc["v2 (pool-and-select)", "mean_quality"]
        assert v2 > v1, f"v2 mean quality {v2} did not beat v1 {v1}"

    def test_quality_is_bounded(self, system):
        r = system.generic.generate(n_shortlist=5, target_class="beta-blocker",
                                    target_stem="-olol")
        for c in r.all_candidates:
            assert 0.0 <= c.quality.total <= 100.0
            assert all(0.0 <= x.score <= 1.0 for x in c.quality.components)

    def test_generic_and_brand_use_different_objectives(self, system):
        assert set(system.scorer.generic_weights) != set(system.scorer.brand_weights)
        assert "morpheme_hygiene" in system.scorer.generic_weights
        assert "stem_avoidance" in system.scorer.brand_weights

    def test_cheap_reward_tracks_the_full_objective(self, system):
        """Guided sampling optimises the cheap reward. If it did not correlate with the
        real objective, the sampler would be steering toward the wrong thing."""
        r = system.generic.generate(n_shortlist=8, target_class="beta-blocker",
                                    target_stem="-olol")
        pairs = [(system.scorer.cheap_reward(c.name, TargetType.GENERIC, "-olol"),
                  c.quality.total) for c in r.all_candidates[:120]]
        s = pd.DataFrame(pairs, columns=["cheap", "full"])
        assert s["cheap"].corr(s["full"]) > 0.3


# ===========================================================================
# Verifier discrimination
# ===========================================================================

class TestVerifierDiscrimination:

    def test_separates_known_confusable_pairs_from_random_pairs(self, system):
        from nomina.evaluation import evaluate_verifier
        r = evaluate_verifier(system, n_negative=150)
        assert r["roc_auc"] > 0.90
        assert r["separation"] > 15
