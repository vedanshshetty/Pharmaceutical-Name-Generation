"""
NOMINA orchestrator — parallel pool-and-select generation.

The architecture this replaces
------------------------------
v1 exposed four peer values of `generation_strategy`: `llm_baseline`,
`rejection_sampling`, `constrained_decoding` and `rl_refined`. Three problems, all
established by inspection and confirmed by running it:

1. **They were not four ideas.** `rl_refined` used the *same* n-gram model as
   `rejection_sampling`, adding a bigram penalty and a rising temperature. That is a
   `guided: bool` on one sampler, not a fourth mechanism. The honest inventory is three
   structurally different proposers (phonological grammar, corpus statistics, semantic
   reasoning) and one modifier.

2. **They never combined.** Each ran alone. Whichever was selected produced the whole
   shortlist, so a run learned nothing about what the others would have proposed. A
   cascade with early exit was considered and rejected for the same reason: stopping at
   the first proposer that clears the bar optimises for cost and discards, unseen, the
   possibility that another proposer had something better.

3. **Feedback was applied asymmetrically.** The guided n-gram path received this run's
   rejections. The LLM path received a static sample of real names and was never told
   what had already failed, despite being the only proposer able to reason about *why*.

The architecture here
---------------------
    propose in parallel  ->  pool  ->  verify the whole pool  ->  score  ->  select

Both free proposers run for every request. Nothing is discarded before it has been
seen. Selection is on the NameQuality objective, not on arrival order. The LLM is not
in the free pool because its unit cost is different in kind (money and seconds against
CPU microseconds); it is escalated to only when the free pool's best result is thin,
which is the one situation where paying for semantic reasoning is justified.

`rl_refined` survives as what it always was: the *batch policy* for the statistical
proposer. Drawing N candidates independently from one biased sampler produces N
near-duplicates of the same near-miss. Draw #7 is conditioned on what draws #1-6 got
rejected for, which is the only context in which "learning from rejections" means
anything on a single request.

Generic and brand are separate pipelines because their objectives are opposed: a generic
name must carry its class stem and be systematically unmemorable; a brand name must
carry no stem at all and be memorable. Shared code with a branch would have quietly
optimised both toward the average of two incompatible targets.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .contracts import (
    CandidateRequest, FailureCode, QualityReport, RiskBand, TargetType, VerifierResponse,
)

ORCHESTRATOR_VERSION = "2.0.0"


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class PipelineConfig:
    """Everything the production path can be tuned by, in one auditable place."""

    # Pool sizing
    pool_per_proposer: int = 24        # raw draws from EACH free proposer, per round
    max_rounds: int = 4                # pool refills before giving up
    shortlist_size: int = 10

    # Refinement of near-misses
    max_refinement_rounds: int = 3     # edits applied to one lineage before abandoning
    refine_top_k: int = 8              # only the most promising rejects are worth editing

    # LLM escalation
    use_llm: bool = True
    llm_quality_threshold: float = 62.0   # escalate when the free pool's best is below this
    llm_batch: int = 12
    llm_max_calls: int = 2

    # Guided (RL-style) sampling
    guided: bool = True
    guidance_strength: float = 1.0     # multiplier on the rejection bigram penalty
    temperature: float = 1.0
    temperature_ramp: float = 0.06     # per consecutive failed round
    reward_guided_draws: int = 3       # oversample, keep the best by cheap_reward

    # Admissibility policy.
    # FALSE by default, matching the published POCA convention: 55-70 is the band that
    # *warrants human review*, not the band that is refused. Discarding it outright was
    # tried and is wrong for two reasons. It throws away the best candidates in
    # saturated classes (the `-olol` space at short prefix lengths is genuinely crowded,
    # so almost everything plausible lands there), and it hides the tension instead of
    # reporting it. The band is now carried explicitly on every result, and the quality
    # objective's distinctiveness term scores headroom against the 55 line, so grey-band
    # names are ranked below clear ones automatically rather than being silently
    # accepted as v1 did or silently discarded as an over-correction would.
    treat_moderate_as_failure: bool = False
    min_shortlist_quality: float = 0.0   # optional floor on what may be returned

    seed: int = 20260826

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Candidate:
    """One proposed name and everything known about it."""
    name: str
    proposer: str                      # 'grammar' | 'ngram' | 'ngram_guided' | 'llm'
    round: int = 0
    lineage: List[str] = field(default_factory=list)
    response: Optional[VerifierResponse] = None
    quality: Optional[QualityReport] = None
    accepted: bool = False
    refined_from: Optional[str] = None

    @property
    def quality_score(self) -> float:
        return self.quality.total if self.quality else 0.0

    @property
    def risk(self) -> float:
        return self.response.composite_risk_score if self.response else 100.0

    def to_row(self) -> Dict[str, Any]:
        r, q = self.response, self.quality
        sim = r.checks.similarity if r else None
        row: Dict[str, Any] = {
            "candidate_name": self.name,
            "accepted": self.accepted,
            "proposer": self.proposer,
            "round": self.round,
            "quality_total": round(q.total, 2) if q else None,
            "composite_risk_score": round(r.composite_risk_score, 2) if r else None,
            "risk_band": r.risk_band.value if r else None,
            "margin_to_review": sim.distinctiveness_margin_moderate if sim else None,
            "margin_to_reject": sim.distinctiveness_margin if sim else None,
            "nearest_match": sim.nearest_match if sim else None,
            "pronounceability": r.checks.pronounceability.score if r else None,
            "syllables": r.checks.pronounceability.syllable_count if r else None,
            "failure_codes": "|".join(c.value for c in r.failure_codes) if r else None,
            "warning_codes": "|".join(c.value for c in r.warning_codes) if r else None,
            "refined_from": self.refined_from,
            "lineage": " -> ".join(self.lineage) if self.lineage else None,
        }
        if q:
            for c in q.components:
                row[f"q_{c.name}"] = round(c.score, 3)
        return row


@dataclass
class RunReport:
    """One complete generation run, fully reconstructable from this object."""
    shortlist: List[Candidate]
    all_candidates: List[Candidate]
    stats: Dict[str, Any]
    config: Dict[str, Any]
    request: Dict[str, Any]
    llm_log: List[Dict[str, Any]] = field(default_factory=list)

    def rows(self) -> List[Dict[str, Any]]:
        return [c.to_row() for c in self.all_candidates]

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame(self.rows())


# ===========================================================================
# Proposers
# ===========================================================================

class Proposer:
    """A proposer turns a request into raw candidate strings. Nothing more.

    Deliberately narrow: proposers do not verify, do not score, and do not decide
    anything. That is what makes them genuinely swappable and genuinely comparable,
    which the v1 'strategies' were not, since each one owned its own end-to-end loop.
    """
    name = "base"
    cost = "free"

    def propose(self, n: int, ctx: "GenerationContext") -> List[str]:
        raise NotImplementedError


@dataclass
class GenerationContext:
    """Mutable per-run state every proposer can read and the guided sampler writes to."""
    target_type: TargetType
    target_class: Optional[str]
    target_stem: Optional[str]
    rng: random.Random
    round: int = 0
    rejected: List[str] = field(default_factory=list)
    penalty_bigrams: Counter = field(default_factory=Counter)
    banned: set = field(default_factory=set)
    siblings: List[str] = field(default_factory=list)

    @property
    def bare_stem(self) -> str:
        return (self.target_stem or "").lstrip("-").lower() \
            if self.target_type == TargetType.GENERIC else ""

    @property
    def needs_consonant_final_prefix(self) -> bool:
        """True when the required stem begins with a vowel, in which case the fantasy
        prefix must end on a consonant or the join produces a vowel pile-up."""
        st = self.bare_stem
        return bool(st) and st[0] in "aeiou"


class GrammarProposer(Proposer):
    """Phonological construction from a grammar INDUCED FROM THE CORPUS.

    Pronounceability holds by construction rather than being discovered by rejection,
    which is why this proposer clears the screen in one round where the statistical one
    needs several. Crucially it recombines at the *syllable* level, so unlike the n-gram
    model it structurally cannot reproduce a whole morpheme such as `erythro`: the
    memorisation failure mode is excluded by the representation, not merely penalised
    after the fact.

    The grammar is learned, not written. A hand-specified inventory encoded English
    orthography and produced `skemkultolol` and `jeimheistolol`; the induced one draws
    onsets, nuclei and codas from the frequencies actually observed in real fantasy
    prefixes.
    """
    name = "grammar"

    def __init__(self, grammar, min_len: int, max_len: int,
                 max_syllables: Optional[int] = None, scorer=None,
                 draws: int = 3):
        self.grammar = grammar
        self.min_len, self.max_len = min_len, max_len
        self.max_syllables = max_syllables
        self.scorer = scorer
        self.draws = max(1, draws)

    def propose(self, n: int, ctx: "GenerationContext") -> List[str]:
        out, seen = [], set()
        stem = ctx.bare_stem
        guard = 0
        while len(out) < n and guard < n * self.draws * 15:
            best, best_r = None, -1.0
            for _ in range(self.draws):
                guard += 1
                k = None
                if self.max_syllables:
                    k = ctx.rng.randint(2, self.max_syllables)
                body = self.grammar.sample(
                    ctx.rng, self.min_len, self.max_len, n_syllables=k,
                    final_coda=True if ctx.needs_consonant_final_prefix else None)
                if not body:
                    continue
                cand = body + stem
                if cand in seen or cand in ctx.banned:
                    continue
                if self.scorer is None:
                    best = cand
                    break
                r = self.scorer.cheap_reward(cand, ctx.target_type, ctx.target_stem)
                if r > best_r:
                    best, best_r = cand, r
            if best:
                seen.add(best)
                out.append(best)
        return out


class NGramProposer(Proposer):
    """Corpus statistics, optionally guided by this run's rejections.

    The guided mode is what v1 called `rl_refined`. It is not a separate strategy and it
    is not a trained policy: it is a batch-drawing discipline. Two mechanisms, both
    operating across a batch rather than within a single draw:

      * bigrams that appeared in the colliding region of a rejected candidate are
        down-weighted (never zeroed, so the space stays connected), and
      * temperature rises with consecutive failed rounds, widening the search once the
        high-probability region has demonstrably failed.

    Additionally each slot is drawn `reward_guided_draws` times and the best by the
    verifier-free reward is kept. That is the cheap approximation of "optimise for
    quality during proposal", and it costs microseconds rather than a 50ms verify.
    """
    name = "ngram"

    def __init__(self, sampler: Callable, model, min_len: int, max_len: int,
                 config: PipelineConfig, scorer=None, guided: bool = True,
                 min_reward: float = 0.55):
        self._sample = sampler
        self.model = model
        self.min_len, self.max_len = min_len, max_len
        self.config = config
        self.scorer = scorer
        self.guided = guided
        # A floor on the verifier-free reward, applied INSIDE the sampler. Without it
        # the n-gram proposer contributed nothing usable: trained on a few hundred short
        # prefixes it reproduces them, every reproduction is caught by the novelty term,
        # and its accept rate sits at zero. Resampling until a draw clears the floor is
        # rejection sampling applied to the right objective, and it is what makes this
        # proposer a real contributor to the pool rather than dead weight.
        self.min_reward = min_reward
        self.name = "ngram_guided" if guided else "ngram"

    def propose(self, n: int, ctx: GenerationContext) -> List[str]:
        cfg = self.config
        temp = cfg.temperature
        avoid = None
        banned = None
        if self.guided:
            temp = cfg.temperature * (1.0 + cfg.temperature_ramp * ctx.round)
            avoid = Counter({k: v * cfg.guidance_strength
                             for k, v in ctx.penalty_bigrams.items()})
            banned = ctx.banned

        stem = ctx.bare_stem
        out, seen = [], set()
        guard = 0
        draws = max(1, cfg.reward_guided_draws if self.scorer else 1)
        budget = n * draws * 20
        while len(out) < n and guard < budget:
            best, best_r = None, -1.0
            for _ in range(draws):
                guard += 1
                body = self._sample(ctx.rng, self.model, self.min_len, self.max_len,
                                    temperature=temp, banned=banned, avoid_bigrams=avoid)
                if not body:
                    continue
                # Same seam rule the grammar proposer enforces during construction. The
                # n-gram model has no notion of the stem it is about to be concatenated
                # with, so the constraint has to be applied here.
                if ctx.needs_consonant_final_prefix and body[-1] in "aeiou":
                    continue
                cand = body + stem
                if cand in seen or cand in ctx.banned:
                    continue
                if self.scorer is None:
                    best = cand
                    break
                r = self.scorer.cheap_reward(cand, ctx.target_type, ctx.target_stem)
                if r > best_r:
                    best, best_r = cand, r
                if best_r >= self.min_reward:
                    break
            if best and (self.scorer is None or best_r >= self.min_reward
                         or guard > budget * 0.8):
                seen.add(best)
                out.append(best)
        return out


class LLMProposer(Proposer):
    """Semantic proposal. Metered, escalated to, never in the free pool."""
    name = "llm"
    cost = "metered"

    def __init__(self, client, avoid_provider: Callable[[GenerationContext], List[str]]):
        self.client = client
        self._avoid = avoid_provider
        self.log: List[Dict[str, Any]] = []

    def propose(self, n: int, ctx: GenerationContext) -> List[str]:
        res = self.client.propose(
            n=n, target_type=ctx.target_type.value, target_class=ctx.target_class,
            target_stem=ctx.target_stem, avoid_names=self._avoid(ctx),
            rejected_this_run=ctx.rejected[-20:],
        )
        self.log.append({"model": res.model_used, "attempts": res.attempts,
                         "n_returned": len(res.names), "error": res.error,
                         "latency_s": res.latency_s})
        return res.names


# ===========================================================================
# The pipeline
# ===========================================================================

class NominaPipeline:
    """Pool-and-select generation for one target type.

    Constructed by `build_pipeline()` rather than directly, so that the corpus, models,
    verifier and scorer are wired consistently and the expensive pieces are shared
    between the generic and brand pipelines instead of built twice.
    """

    def __init__(self, target_type: TargetType, verifier, scorer,
                 proposers: Sequence[Proposer], refine_fn: Callable,
                 config: Optional[PipelineConfig] = None,
                 llm_proposer: Optional[LLMProposer] = None,
                 sibling_fn: Optional[Callable[[str], List[str]]] = None):
        self.target_type = target_type
        self.verifier = verifier
        self.scorer = scorer
        self.proposers = list(proposers)
        self.refine = refine_fn
        self.config = config or PipelineConfig()
        self.llm = llm_proposer
        self._siblings = sibling_fn or (lambda stem: [])

    # -- verification ------------------------------------------------------
    def _verify(self, name: str, ctx: GenerationContext, proposer: str) -> VerifierResponse:
        req = CandidateRequest(
            candidate_name=name, target_type=self.target_type,
            target_class=ctx.target_class, target_stem=ctx.target_stem,
            generation_strategy=proposer,
            generation_metadata={"round": ctx.round, "pipeline": ORCHESTRATOR_VERSION},
        )
        return self.verifier.verify(req)

    def _admissible(self, resp: VerifierResponse) -> bool:
        """Admissibility policy, stated in one place.

        With `treat_moderate_as_failure` on, a candidate in the 55-70 review band is not
        accepted. v1 accepted these silently, and every single name it produced in
        testing sat in that band, which meant the headline pass rate was measuring
        something other than what it appeared to measure.
        """
        if not resp.overall_pass:
            return False
        if self.config.treat_moderate_as_failure and resp.risk_band == RiskBand.MODERATE:
            return False
        return True

    def _learn_from_rejection(self, ctx: GenerationContext, name: str,
                              resp: VerifierResponse) -> None:
        """Turn a structured rejection into sampling pressure.

        Reads `signal.payload`, never `human_readable`. The colliding name, the sibling
        that was too close, and the offending foreign morpheme are all in the payload as
        machine-usable data precisely so that the generator never has to parse prose or
        make a second model call to understand why it failed.
        """
        ctx.rejected.append(name)
        ctx.banned.add(name)
        for sig in resp.refinement_feedback:
            for key in ("nearest_match", "sibling", "fragment"):
                v = sig.payload.get(key)
                if isinstance(v, str) and len(v) > 1:
                    ctx.penalty_bigrams.update(v[i:i + 2] for i in range(len(v) - 1))
            for key in ("conflicts", "foreign_stems"):
                for v in sig.payload.get(key, []) or []:
                    if isinstance(v, str) and len(v) > 1:
                        ctx.penalty_bigrams.update(v[i:i + 2] for i in range(len(v) - 1))

    # -- the run -----------------------------------------------------------
    def generate(self, n_shortlist: Optional[int] = None,
                 target_class: Optional[str] = None,
                 target_stem: Optional[str] = None,
                 progress: Optional[Callable[[str], None]] = None) -> RunReport:
        cfg = self.config
        want = n_shortlist or cfg.shortlist_size
        say = progress or (lambda _m: None)
        t0 = time.perf_counter()

        ctx = GenerationContext(
            target_type=self.target_type, target_class=target_class,
            target_stem=target_stem, rng=random.Random(cfg.seed),
            siblings=self._siblings(target_stem or ""),
        )

        everything: List[Candidate] = []
        accepted: List[Candidate] = []
        verify_calls = 0
        llm_calls = 0

        for rnd in range(1, cfg.max_rounds + 1):
            ctx.round = rnd - 1

            # 1. PROPOSE — all free proposers, in parallel, into one pool.
            pool: List[Tuple[str, str]] = []
            for p in self.proposers:
                for nm in p.propose(cfg.pool_per_proposer, ctx):
                    pool.append((nm, p.name))
            say(f"round {rnd}: pooled {len(pool)} raw candidates from "
                f"{len(self.proposers)} proposers")

            # 2. VERIFY the whole pool. Nothing is discarded before it is seen.
            batch: List[Candidate] = []
            seen_names = {c.name for c in everything}
            for nm, src in pool:
                if nm in seen_names:
                    continue
                seen_names.add(nm)
                resp = self._verify(nm, ctx, src)
                verify_calls += 1
                cand = Candidate(name=nm, proposer=src, round=rnd,
                                 lineage=[nm], response=resp)
                cand.quality = self.scorer.score(nm, resp, self.target_type, target_stem)
                cand.accepted = self._admissible(resp)
                batch.append(cand)
                if not cand.accepted:
                    self._learn_from_rejection(ctx, nm, resp)

            everything.extend(batch)
            accepted.extend([c for c in batch if c.accepted])
            say(f"round {rnd}: {sum(c.accepted for c in batch)} admissible of {len(batch)}")

            # 3. REFINE the most promising rejects. Editing a candidate that scored 20
            #    is wasted work; editing one that scored 61 and missed on one code is
            #    the cheapest quality available anywhere in the system.
            near = sorted([c for c in batch if not c.accepted and c.response],
                          key=lambda c: c.quality_score, reverse=True)[: cfg.refine_top_k]
            for cand in near:
                cur, resp = cand.name, cand.response
                lineage = [cur]
                for _ in range(cfg.max_refinement_rounds):
                    nxt = self.refine(cur, resp, self.target_type, ctx.rng)
                    if not nxt or nxt == cur or nxt in seen_names:
                        break
                    seen_names.add(nxt)
                    lineage.append(nxt)
                    resp = self._verify(nxt, ctx, cand.proposer)
                    verify_calls += 1
                    ref = Candidate(name=nxt, proposer=cand.proposer + "+refined",
                                    round=rnd, lineage=list(lineage), response=resp,
                                    refined_from=cand.name)
                    ref.quality = self.scorer.score(nxt, resp, self.target_type, target_stem)
                    ref.accepted = self._admissible(resp)
                    everything.append(ref)
                    if ref.accepted:
                        accepted.append(ref)
                        break
                    self._learn_from_rejection(ctx, nxt, resp)
                    cur = nxt

            # 4. Stop when we have enough GOOD names, not merely enough passing ones.
            good = [c for c in accepted if c.quality_score >= cfg.llm_quality_threshold]
            if len(good) >= want:
                say(f"round {rnd}: {len(good)} candidates at or above quality "
                    f"{cfg.llm_quality_threshold}; stopping")
                break

        # 5. ESCALATE to the metered proposer only if the free pool came up thin.
        best_free = max((c.quality_score for c in accepted), default=0.0)
        if (self.llm and cfg.use_llm and llm_calls < cfg.llm_max_calls
                and (best_free < cfg.llm_quality_threshold or len(accepted) < want)):
            say(f"free pool best quality {best_free:.1f} < {cfg.llm_quality_threshold}; "
                f"escalating to the LLM proposer")
            ctx.round += 1
            names = self.llm.propose(cfg.llm_batch, ctx)
            llm_calls += 1
            for nm in names:
                if nm in {c.name for c in everything}:
                    continue
                resp = self._verify(nm, ctx, "llm")
                verify_calls += 1
                cand = Candidate(name=nm, proposer="llm", round=ctx.round + 1,
                                 lineage=[nm], response=resp)
                cand.quality = self.scorer.score(nm, resp, self.target_type, target_stem)
                cand.accepted = self._admissible(resp)
                everything.append(cand)
                if cand.accepted:
                    accepted.append(cand)
                else:
                    self._learn_from_rejection(ctx, nm, resp)

        # 6. SELECT on the objective, not on arrival order.
        pool_for_selection = [c for c in accepted
                              if c.quality_score >= cfg.min_shortlist_quality]
        shortlist = sorted(pool_for_selection,
                           key=lambda c: c.quality_score, reverse=True)[:want]

        by_proposer: Dict[str, Dict[str, Any]] = {}
        for c in everything:
            b = by_proposer.setdefault(c.proposer, {"proposed": 0, "accepted": 0, "quality": []})
            b["proposed"] += 1
            b["accepted"] += int(c.accepted)
            if c.accepted:
                b["quality"].append(c.quality_score)
        for b in by_proposer.values():
            qs = b.pop("quality")
            b["mean_quality"] = round(sum(qs) / len(qs), 2) if qs else None
            b["accept_rate"] = round(b["accepted"] / max(1, b["proposed"]), 3)

        stats = {
            "orchestrator_version": ORCHESTRATOR_VERSION,
            "target_type": self.target_type.value,
            "requested": want,
            "returned": len(shortlist),
            "candidates_evaluated": len(everything),
            "admissible": len(accepted),
            "admissible_rate": round(len(accepted) / max(1, len(everything)), 3),
            "band_low": sum(1 for c in accepted if c.response
                            and c.response.risk_band == RiskBand.LOW),
            "band_moderate": sum(1 for c in accepted if c.response
                                 and c.response.risk_band == RiskBand.MODERATE),
            "verifier_calls": verify_calls,
            "llm_calls": llm_calls,
            "best_quality": round(shortlist[0].quality_score, 2) if shortlist else None,
            "mean_shortlist_quality": round(
                sum(c.quality_score for c in shortlist) / len(shortlist), 2) if shortlist else None,
            "wall_seconds": round(time.perf_counter() - t0, 2),
            "by_proposer": by_proposer,
        }

        return RunReport(
            shortlist=shortlist, all_candidates=everything, stats=stats,
            config=cfg.to_dict(),
            request={"target_type": self.target_type.value, "target_class": target_class,
                     "target_stem": target_stem, "n_shortlist": want},
            llm_log=list(self.llm.log) if self.llm else [],
        )
