"""
NOMINA NameQuality — the objective that says whether a name is any *good*.

Why this module exists
----------------------
The v1 pipeline measured exactly one thing: risk. A candidate was accepted the moment
the verifier stopped objecting, and the shortlist was whatever passed first. Running it
end to end produced, among others:

    erythroolol   accepted, reported margin 13.03
    amoxiolol     accepted, reported margin 11.26
    acycloolol    accepted, reported margin  7.04
    snabreistolol accepted, reported margin 14.17
    hiemfailolol  accepted, reported margin 16.09

Every one of those is admissible under the letter of the checks and indefensible as a
name. The first three are real drug prefixes (erythromycin, amoxicillin, aciclovir) with
a beta-blocker stem stapled on, which is precisely the cross-class misidentification the
INN stem system exists to prevent. The last two are unsayable. Nothing in the pipeline
was measuring any of that, because "does not trip a check" and "is a good name" are
different questions and only the first one had an implementation.

Design
------
Quality is deliberately kept *orthogonal to* and *downstream of* the verifier. The
verifier remains the sole authority on admissibility; this module never overrides a
rejection and never rescues one. It ranks what survives, and it supplies the reward
signal that guided sampling optimises against. That separation is what keeps the
regulatory claim honest: the screen is still a screen, not a preference model.

Every component returns 0-1, higher is better, and every component is auditable — the
returned `QualityReport` carries the per-term scores and the evidence behind them, so a
shortlist can always answer "why did this name win".

Two profiles, because the objectives genuinely differ
----------------------------------------------------
GENERIC (INN/USAN) names are stem-governed. They are long, systematic, deliberately
unmemorable, and their entire job is to encode class membership unambiguously while
being distinguishable from siblings. Reusing another class's morpheme is the cardinal
sin.

BRAND names are the opposite: short, memorable, distinctive, and they must NOT carry a
stem at all, since a stem in a proprietary name falsely implies class membership. They
are optimised for recall and pronunciation across languages.

Running one objective over both would optimise brand names toward five-syllable
systematic strings and generic names toward punchy two-syllable ones. Hence two weight
profiles and two component sets.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .contracts import QualityComponent, QualityReport, TargetType, VerifierResponse

VOWELS = set("aeiou")


# ===========================================================================
# Weight profiles
# ===========================================================================

GENERIC_WEIGHTS: Dict[str, float] = {
    "distinctiveness": 0.24,   # headroom to the REVIEW line, not the hard cutoff
    "novelty": 0.20,           # is this actually new, or a remembered fragment
    "morpheme_hygiene": 0.16,  # does it smuggle another class's signal
    "pronounceability": 0.14,  # continuous, not a pass/fail floor
    "shape": 0.10,             # length and syllable count vs the real INN distribution
    "seam": 0.09,              # the prefix/stem join, where machine names betray themselves
    "typicality": 0.07,        # plausible as a drug name, without being derivative
}

BRAND_WEIGHTS: Dict[str, float] = {
    "distinctiveness": 0.24,
    "novelty": 0.14,
    "stem_avoidance": 0.16,    # a stem in a brand name is a false class claim
    "pronounceability": 0.14,
    "memorability": 0.15,      # short, low syllable count, clean onsets
    "seam": 0.09,
    "typicality": 0.08,
}


@dataclass
class ShapeReference:
    """Empirical length and syllable distribution of real names, per profile.

    Measured from the corpus rather than hardcoded, so expanding the corpus to EMA and
    RxNorm automatically re-centres the target instead of leaving a stale constant that
    was tuned against a 2,000-row US-only sample.
    """
    mean_len: float
    std_len: float
    mean_syl: float
    std_syl: float
    n: int

    @classmethod
    def from_names(cls, names: Sequence[str], syllable_fn) -> "ShapeReference":
        names = [n for n in names if n]
        if not names:
            return cls(10.0, 2.0, 4.0, 1.0, 0)
        lens = [len(n) for n in names]
        syls = []
        for n in names[:1500]:                      # bounded: G2P is the expensive part
            try:
                syls.append(max(1, syllable_fn(n)))
            except Exception:                        # noqa: BLE001
                continue
        syls = syls or [4]
        return cls(
            mean_len=sum(lens) / len(lens),
            std_len=(sum((x - sum(lens) / len(lens)) ** 2 for x in lens) / len(lens)) ** 0.5 or 1.0,
            mean_syl=sum(syls) / len(syls),
            std_syl=(sum((x - sum(syls) / len(syls)) ** 2 for x in syls) / len(syls)) ** 0.5 or 1.0,
            n=len(names),
        )


# ===========================================================================
# Substring index — the anti-memorisation machinery
# ===========================================================================

class SubstringIndex:
    """Fast 'does this candidate reuse a chunk of a real name' lookup.

    Naively comparing a candidate against every corpus name is O(corpus x len^2) per
    candidate, which is far too slow inside a sampling loop that draws hundreds of
    candidates. Instead every k-length substring of every real name is indexed once, and
    lookup becomes a set membership test per candidate substring.
    """

    def __init__(self, names: Iterable[str], min_k: int = 4, max_k: int = 9):
        self.min_k, self.max_k = min_k, max_k
        self._index: Dict[int, Dict[str, str]] = {k: {} for k in range(min_k, max_k + 1)}
        for name in names:
            n = len(name)
            for k in range(min_k, min(max_k, n) + 1):
                bucket = self._index[k]
                for i in range(n - k + 1):
                    bucket.setdefault(name[i:i + k], name)

    def longest_shared(self, candidate: str,
                       ignore_suffix: str = "") -> Tuple[int, str, str]:
        """Longest substring of `candidate` that also occurs in some real name.

        `ignore_suffix` masks the required stem before the search. Without it every
        generic candidate would score a guaranteed len(stem) overlap purely for
        complying with the regulation that forced the stem on it, which would make the
        term measure conformity rather than memorisation.
        """
        probe = candidate
        if ignore_suffix and probe.endswith(ignore_suffix):
            probe = probe[: len(probe) - len(ignore_suffix)]
        best = (0, "", "")
        for k in range(min(self.max_k, len(probe)), self.min_k - 1, -1):
            bucket = self._index.get(k, {})
            for i in range(len(probe) - k + 1):
                frag = probe[i:i + k]
                if frag in bucket:
                    return k, frag, bucket[frag]
            if best[0]:
                break
        return best


# ===========================================================================
# Individual components
# ===========================================================================

def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_distinctiveness(response: VerifierResponse, moderate_cutoff: float = 55.0,
                          high_cutoff: float = 70.0) -> Tuple[float, Dict[str, Any]]:
    """Headroom below the REVIEW line, not the rejection line.

    v1 reported `high_cutoff - score`, so a candidate at 57 against a 55 review line
    advertised a margin of 13 and looked comfortable. Under POCA convention it was
    squarely inside the band that warrants human review. Scoring against the moderate
    cutoff makes a name earn its way *out* of the grey band rather than merely out of
    the reject band.
    """
    sim = response.checks.similarity
    score = sim.nearest_match_score if sim.nearest_match_score is not None else 0.0
    margin = moderate_cutoff - score
    # 0 at the review line, 1 once ~25 points clear of it.
    val = _clamp(margin / 25.0)
    return val, {"nearest_match": sim.nearest_match, "nearest_score": round(score, 2),
                 "margin_to_review": round(margin, 2),
                 "margin_to_reject": round(high_cutoff - score, 2),
                 "band": response.risk_band.value if response.risk_band else None}


def score_novelty(candidate: str, index: SubstringIndex,
                  stem: str = "") -> Tuple[float, Dict[str, Any]]:
    """Penalises verbatim reuse of a fragment of an existing name.

    This is the term that kills `erythroolol`. The seven-character run `erythro` is
    lifted whole from erythromycin, so a clinician reads macrolide antibiotic and gets
    a beta-blocker. An n-gram model trained on a small corpus produces this constantly,
    and no similarity metric catches it, because once `-mycin` is swapped for `-olol`
    the two whole strings are genuinely far apart.
    """
    k, frag, src = index.longest_shared(candidate, ignore_suffix=stem)
    body_len = max(1, len(candidate) - len(stem))
    if k == 0:
        return 1.0, {"longest_shared": 0}
    # 4 shared characters is unremarkable; 7+ is a lifted morpheme.
    absolute = _clamp((8 - k) / 4.0)
    relative = _clamp(1.0 - (k / body_len))
    val = 0.6 * absolute + 0.4 * relative
    return val, {"longest_shared": k, "fragment": frag, "matches": src,
                 "share_of_body": round(k / body_len, 2)}


def score_morpheme_hygiene(candidate: str, blocklist: Set[str],
                           stem_index: Dict[str, str],
                           required_stem: str = "") -> Tuple[float, Dict[str, Any]]:
    """Does the name carry a *foreign* class signal anywhere inside it.

    The verifier checks whether a generic name ends in the right stem. It historically
    did not check whether the name also contains a different class's stem somewhere in
    the middle, so `cillinolol` (a beta-blocker announcing itself as a penicillin) and
    `prazololol` both passed. That gap is closed in the verifier now; this term scores
    the same property continuously so the sampler can be steered away from it rather
    than only punished at the end.
    """
    body = candidate[: len(candidate) - len(required_stem)] if required_stem else candidate
    hits = []
    for frag in blocklist:
        if len(frag) >= 4 and frag in body and frag != required_stem.lstrip("-"):
            hits.append(frag)
    hits = sorted(set(hits), key=len, reverse=True)[:4]
    if not hits:
        return 1.0, {"foreign_morphemes": []}
    worst = len(hits[0])
    val = _clamp(1.0 - (0.35 * len(hits)) - (0.06 * max(0, worst - 3)))
    return val, {"foreign_morphemes": hits,
                 "meanings": [stem_index.get(h) for h in hits if h in stem_index]}


def _orthographic_complexity(name: str) -> Tuple[float, Dict[str, Any]]:
    """Letter-level cluster complexity.

    The phonemic route is too permissive on its own: `skemkultolol` syllabifies into
    perfectly legal units and scores 1.0 phonotactically, because the G2P converter is
    happy to assign SK and LT to legal positions. Counting consonant clusters directly
    in the orthography catches what the phonemic route waves through, and it is the
    representation a prescriber actually reads off a label.
    """
    runs = re.findall(r"[bcdfghjklmnpqrstvwxyz]+", name)
    heavy = [r for r in runs if len(r) >= 3]
    medium = [r for r in runs if len(r) == 2]
    penalty = 0.45 * len(heavy) + 0.10 * max(0, len(medium) - 1)
    return _clamp(1.0 - penalty), {"consonant_runs": runs,
                                   "clusters_3plus": heavy, "clusters_2": len(medium)}


def articulatory_ease(syllables: Sequence[str], name: str = "") -> Tuple[float, Dict[str, Any]]:
    """How easy the syllable structure is to articulate.

    `phonotactic_score` is effectively a legality gate: it returns 1.0 for essentially
    every well-formed string, real or generated, so it cannot rank. What separates
    `metoprolol` (M EH / T OW / P R OW / L AO L) from `ascazalolol`
    (AE / S K AE / Z AE / L OW / L AO L) is not legality, it is structure: the
    proportion of simple onset-nucleus syllables, and the absence of consonant clusters
    in non-final position. That is what this measures.
    """
    if not syllables:
        return 0.5, {"syllables": 0}
    simple = 0
    clusters = 0
    for i, syl in enumerate(syllables):
        phones = syl.split()
        nucleus_at = next((j for j, p in enumerate(phones) if p[0] in "AEIOU"), None)
        if nucleus_at is None:
            continue
        onset, coda = phones[:nucleus_at], phones[nucleus_at + 1:]
        if len(onset) <= 1 and len(coda) <= 1:
            simple += 1
        if len(onset) > 1 and i < len(syllables) - 1:
            clusters += 1
    ease = simple / len(syllables) - 0.15 * clusters
    detail = {"simple_syllables": simple, "total_syllables": len(syllables),
              "medial_clusters": clusters}
    if name:
        ortho, od = _orthographic_complexity(name)
        ease = 0.55 * _clamp(ease) + 0.45 * ortho
        detail.update(od)
        detail["orthographic_ease"] = round(ortho, 3)
    return _clamp(ease), detail


def score_pronounceability(response: VerifierResponse,
                           reference: Optional[Sequence[float]] = None
                           ) -> Tuple[float, Dict[str, Any]]:
    """Phonotactic legality AND articulatory ease, with corpus typicality excluded.

    The verifier reports `0.6 * phonotactic + 0.4 * corpus_typicality` under the name
    "pronounceability". That blend is defensible as a single admissibility floor, but it
    is the wrong quantity to rank on here, because the typicality term rewards exactly
    the corpus-hugging behaviour the novelty term is built to punish. Blending them
    would have the objective pulling in both directions at once through two components
    that look independent and are not.

    So the pure phonotactic component is read out of the check's details, and the
    ranking signal comes from articulatory structure instead.
    """
    chk = response.checks.pronounceability
    if chk.score is None:
        return 0.5, {"raw": None}
    phono = float(chk.details.get("phonotactic", chk.score))
    ease, ease_detail = articulatory_ease(chk.syllables or [], response.candidate_name.lower())
    val = _clamp(0.45 * phono + 0.55 * ease)
    detail = {"phonotactic": round(phono, 3),
              "articulatory_ease": round(ease, 3),
              "blended_verifier_score": round(chk.score, 3),
              "syllable_count": chk.syllable_count,
              "illegal_onsets": chk.details.get("illegal_onsets", []),
              **ease_detail}
    return val, detail


def score_shape(candidate: str, syllables: int,
                ref: ShapeReference) -> Tuple[float, Dict[str, Any]]:
    """How close the name sits to the real distribution for its profile.

    `snabreistolol` is thirteen characters with a four-consonant onset cluster; real
    `-olol` names run seven to eleven. Scored as a two-sided Gaussian so both
    unpronounceably long and implausibly short names lose points, rather than a hard
    min/max that treats 11 and 19 characters as equally fine.
    """
    zl = (len(candidate) - ref.mean_len) / max(0.5, ref.std_len)
    zs = (syllables - ref.mean_syl) / max(0.5, ref.std_syl)
    val = _clamp(0.5 * math.exp(-0.5 * zl ** 2) + 0.5 * math.exp(-0.5 * zs ** 2))
    return val, {"length": len(candidate), "syllables": syllables,
                 "target_length": round(ref.mean_len, 1),
                 "target_syllables": round(ref.mean_syl, 1),
                 "z_length": round(zl, 2), "z_syllables": round(zs, 2)}


_TRIPLE = re.compile(r"(.)\1\1")
_VOWEL_RUN = re.compile(r"[aeiou]{3,}")
_CONS_RUN = re.compile(r"[bcdfghjklmnpqrstvwxz]{4,}")


def score_seam(candidate: str, stem: str = "") -> Tuple[float, Dict[str, Any]]:
    """Orthographic hygiene, with special attention to the prefix/stem join.

    Machine-composed names give themselves away at the seam. `acycloolol` and
    `amoxiolol` are both a prefix ending in a vowel glued to `-olol`, producing a vowel
    run no real INN name has. Cheap to detect and it removes an entire visible failure
    mode from the shortlist.
    """
    faults: List[str] = []
    penalty = 0.0
    if _TRIPLE.search(candidate):
        faults.append("triple letter"); penalty += 0.45
    if _VOWEL_RUN.search(candidate):
        faults.append("3+ vowel run"); penalty += 0.35
    if _CONS_RUN.search(candidate):
        faults.append("4+ consonant run"); penalty += 0.35
    if stem and candidate.endswith(stem):
        join = len(candidate) - len(stem)
        if join >= 1:
            last_of_prefix, first_of_stem = candidate[join - 1], stem[0]
            if last_of_prefix in VOWELS and first_of_stem in VOWELS:
                faults.append("vowel-vowel collision at stem seam"); penalty += 0.40
            if last_of_prefix == first_of_stem:
                faults.append("repeated letter at stem seam"); penalty += 0.25
            if (join >= 2 and candidate[join - 2:join] == stem[:2]):
                faults.append("stem echo before stem"); penalty += 0.20
    if len(set(candidate)) <= max(3, len(candidate) // 4):
        faults.append("too few distinct letters"); penalty += 0.25
    return _clamp(1.0 - penalty), {"faults": faults}


def score_typicality(candidate: str, lm, stem: str = "") -> Tuple[float, Dict[str, Any]]:
    """Plausible as a pharmaceutical name, without being a copy.

    Deliberately a band rather than a maximum. Maximising corpus likelihood is exactly
    the objective that produces memorised prefixes, so the reward peaks in the middle of
    the real distribution and falls off on *both* sides: too improbable reads as random
    letters, too probable means the model reproduced its training data.
    """
    body = candidate[: len(candidate) - len(stem)] if stem and candidate.endswith(stem) else candidate
    if lm is None or len(body) < 2:
        return 0.5, {"typicality": None}
    try:
        t = float(lm.typicality(body))
    except Exception:                              # noqa: BLE001
        return 0.5, {"typicality": None}
    val = _clamp(1.0 - abs(t - 0.55) / 0.55)
    return val, {"typicality": round(t, 3), "target_band": "0.40-0.70"}


def score_memorability(candidate: str, syllables: int) -> Tuple[float, Dict[str, Any]]:
    """Brand-only: short, few syllables, clean alternation.

    Real successful marks cluster hard at two to three syllables and six to nine letters
    (Lipitor, Ozempic, Xarelto, Humira). Longer marks exist but underperform on recall,
    which is the property a brand name is actually bought for.
    """
    len_term = _clamp(1.0 - abs(len(candidate) - 7.5) / 6.0)
    syl_term = _clamp(1.0 - abs(syllables - 3.0) / 3.0)
    alternation = 0.0
    if len(candidate) > 1:
        flips = sum(1 for a, b in zip(candidate, candidate[1:])
                    if (a in VOWELS) != (b in VOWELS))
        alternation = _clamp(flips / (len(candidate) - 1))
    val = 0.4 * len_term + 0.35 * syl_term + 0.25 * alternation
    return val, {"length": len(candidate), "syllables": syllables,
                 "cv_alternation": round(alternation, 2)}


def score_stem_avoidance(candidate: str, stem_index: Dict[str, str]) -> Tuple[float, Dict[str, Any]]:
    """Brand-only: a proprietary name must not imply a pharmacological class.

    A mark ending in a recognised stem is a false claim of class membership, and both
    the FDA and the INN programme object to it. Terminal position is weighted hardest
    because that is where a stem is read as a stem rather than as coincidence.
    """
    hits_end = [s for s in stem_index if len(s) >= 3 and candidate.endswith(s)]
    hits_in = [s for s in stem_index if len(s) >= 4 and s in candidate and not candidate.endswith(s)]
    penalty = 0.75 * bool(hits_end) + 0.20 * min(2, len(hits_in))
    return _clamp(1.0 - penalty), {"terminal_stems": sorted(hits_end, key=len, reverse=True)[:3],
                                   "embedded_stems": sorted(hits_in, key=len, reverse=True)[:3]}


# ===========================================================================
# Scorer
# ===========================================================================

class QualityScorer:
    """Builds the reference distributions once, then scores candidates cheaply."""

    def __init__(self, screening_corpus, training_corpus, syllable_fn,
                 lm_generic=None, lm_brand=None,
                 moderate_cutoff: float = 55.0, high_cutoff: float = 70.0,
                 generic_weights: Optional[Dict[str, float]] = None,
                 brand_weights: Optional[Dict[str, float]] = None,
                 pron_reference: Optional[Sequence[float]] = None,
                 pron_fn: Optional[Any] = None):
        self.screening = screening_corpus
        self.training = training_corpus
        self.syllable_fn = syllable_fn
        self.lm_generic = lm_generic
        self.lm_brand = lm_brand
        self.moderate_cutoff = moderate_cutoff
        self.high_cutoff = high_cutoff
        self.generic_weights = dict(generic_weights or GENERIC_WEIGHTS)
        self.brand_weights = dict(brand_weights or BRAND_WEIGHTS)
        self.pron_reference = list(pron_reference or [])
        # Direct phonotactic scorer, for use inside the sampling loop. Measured at
        # 0.025 ms per call (~40k/sec), so folding pronounceability into the proposal
        # reward is essentially free and lets the sampler optimise for it directly
        # instead of discovering it by rejection.
        self._pron_fn = pron_fn

        self.index = SubstringIndex(screening_corpus.all)
        self.shape_generic = ShapeReference.from_names(
            [n for n in training_corpus.generic_tokens if 5 <= len(n) <= 18], syllable_fn)
        self.shape_brand = ShapeReference.from_names(training_corpus.brand_names, syllable_fn)
        self.blocklist = training_corpus.morpheme_blocklist
        self.stem_index = training_corpus.stem_index

    # -- the objective ------------------------------------------------------
    def score(self, candidate: str, response: VerifierResponse,
              target_type: TargetType = TargetType.GENERIC,
              target_stem: Optional[str] = None) -> QualityReport:
        name = (candidate or "").strip().lower()
        stem = (target_stem or "").lstrip("-").lower() if target_type == TargetType.GENERIC else ""
        syllables = response.checks.pronounceability.syllable_count or self._syllables(name)

        parts: List[Tuple[str, float, Dict[str, Any]]] = []
        d, dd = score_distinctiveness(response, self.moderate_cutoff, self.high_cutoff)
        parts.append(("distinctiveness", d, dd))
        n, nd = score_novelty(name, self.index, stem)
        parts.append(("novelty", n, nd))
        p, pd_ = score_pronounceability(response, self.pron_reference)
        parts.append(("pronounceability", p, pd_))
        s, sd = score_seam(name, stem)
        parts.append(("seam", s, sd))

        if target_type == TargetType.GENERIC:
            weights = self.generic_weights
            m, md = score_morpheme_hygiene(name, self.blocklist, self.stem_index, stem)
            parts.append(("morpheme_hygiene", m, md))
            sh, shd = score_shape(name, syllables, self.shape_generic)
            parts.append(("shape", sh, shd))
            t, td = score_typicality(name, self.lm_generic, stem)
            parts.append(("typicality", t, td))
        else:
            weights = self.brand_weights
            a, ad = score_stem_avoidance(name, self.stem_index)
            parts.append(("stem_avoidance", a, ad))
            mm, mmd = score_memorability(name, syllables)
            parts.append(("memorability", mm, mmd))
            t, td = score_typicality(name, self.lm_brand, "")
            parts.append(("typicality", t, td))

        components = [QualityComponent(name=k, score=round(v, 4),
                                       weight=weights.get(k, 0.0), detail=det)
                      for k, v, det in parts]
        total = sum(c.score * c.weight for c in components)
        norm = sum(c.weight for c in components) or 1.0
        notes = [f"{c.name}={c.score:.2f}" for c in components if c.score < 0.35]

        return QualityReport(
            total=round(100.0 * total / norm, 2),
            components=components,
            profile=target_type.value,
            disqualified=not response.overall_pass,
            notes=notes,
        )

    def _syllables(self, name: str) -> int:
        try:
            return max(1, self.syllable_fn(name))
        except Exception:                          # noqa: BLE001
            return max(1, sum(1 for i, c in enumerate(name)
                              if c in VOWELS and (i == 0 or name[i - 1] not in VOWELS)))

    # -- reward signal for guided sampling ---------------------------------
    def cheap_reward(self, candidate: str, target_type: TargetType,
                     target_stem: Optional[str] = None) -> float:
        """Verifier-free approximation of the objective, for use *inside* the sampling
        loop where a full verify (~50ms) per draw would be prohibitive.

        Uses only the terms that need no screening: novelty, morpheme hygiene, seam and
        shape. Correlates well enough with the full objective to steer sampling, and
        every survivor is scored properly afterwards, so an approximation here can waste
        a draw but can never let a bad name onto the shortlist.
        """
        name = (candidate or "").strip().lower()
        stem = (target_stem or "").lstrip("-").lower() if target_type == TargetType.GENERIC else ""
        n, _ = score_novelty(name, self.index, stem)
        s, _ = score_seam(name, stem)
        ref = self.shape_generic if target_type == TargetType.GENERIC else self.shape_brand
        sh, _ = score_shape(name, self._syllables(name), ref)
        if target_type == TargetType.GENERIC:
            m, _ = score_morpheme_hygiene(name, self.blocklist, self.stem_index, stem)
        else:
            m, _ = score_stem_avoidance(name, self.stem_index)
        pr = 0.5
        if self._pron_fn is not None:
            try:
                phono, syls = self._pron_fn(name)
                ease, _ = articulatory_ease(syls, name)
                pr = _clamp(0.45 * float(phono) + 0.55 * ease)
            except Exception:                          # noqa: BLE001
                pr = 0.5
        return 0.26 * n + 0.20 * m + 0.16 * s + 0.14 * sh + 0.24 * pr
