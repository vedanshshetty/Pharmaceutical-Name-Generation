"""
NOMINA â€” Verifier suite (Person B).

A from-scratch, dependency-light reimplementation of the algorithm family behind the
FDA's Phonetic and Orthographic Computer Analysis (POCA) tool, plus the four additional
regulatory screens the NOMINA architecture requires.

    V0  well-formedness      basic input sanity
    V1  similarity           POCA-style composite: orthographic (normalised Levenshtein
                             + BI-SIM) and phonetic (ALINE over a rule-based G2P), scored
                             against the corpus of marketed drug names
    V2  stem_conflict        USAN/INN stem grammar: required for generic names, prohibited
                             in stem position for brand names
    V3  trademark_collision  registered-mark screening proxy (offline corpus by default,
                             optional live lookup)
    V4  pronounceability     phonotactic well-formedness + corpus-typicality
    V5  crosslingual         adverse meaning in major pharma markets + implied-claim terms

Everything here is pure Python + numpy/pandas. No model downloads, no API keys, no network
calls on the default path, so a results run is byte-for-byte reproducible.

SCOPE BOUNDARY: this screens the computationally-checkable first pass. It is not FDA or
USAN review, which additionally involve prescription-simulation human-factors studies,
full legal likelihood-of-confusion analysis, and committee judgement.
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

VERIFIER_VERSION = "1.0.0"

# ===========================================================================
# 0. Configuration
# ===========================================================================

@dataclass
class ScoringWeights:
    """Every weight POCA-style scoring uses, in one place.

    Defaults reproduce POCA's published behaviour: the orthographic and phonetic
    components are averaged with equal weight, and within the orthographic component the
    edit-distance and bigram measures are also averaged equally. Change these to run a
    weighted variant without touching any algorithm code.
    """
    w_levenshtein: float = 0.5      # within orthographic
    w_bisim: float = 0.5            # within orthographic
    w_orthographic: float = 0.5     # composite
    w_phonetic: float = 0.5         # composite

    def normalised(self) -> "ScoringWeights":
        o = self.w_levenshtein + self.w_bisim
        c = self.w_orthographic + self.w_phonetic
        o = o if o else 1.0
        c = c if c else 1.0
        return ScoringWeights(self.w_levenshtein / o, self.w_bisim / o,
                              self.w_orthographic / c, self.w_phonetic / c)


@dataclass
class Thresholds:
    """Published POCA operating points.

    Analyses of POCA output report composite scores >= 70 as highly similar (a candidate
    at this level is generally treated as unacceptable), 55-70 as a moderate-similarity
    grey band warranting review, and < 55 as low risk.
    """
    similarity_high: float = 70.0      # hard reject
    similarity_moderate: float = 55.0  # warn
    trademark_high: float = 70.0
    pronounceability_min: float = 0.45
    crosslingual_phonetic: float = 0.88
    intra_stem_high: float = 75.0      # sibling names sharing the same stem
    min_length: int = 4
    max_length: int = 20
    min_stem_prefix: int = 2           # characters of "fantasy prefix" before the stem


@dataclass
class VerifierConfig:
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    thresholds: Thresholds = field(default_factory=Thresholds)
    phonetic_algorithm: str = "aline"     # 'aline' | 'metaphone' | 'none'
    aline_mode: str = "local"             # 'local' (as published) | 'global'
    top_k: int = 5
    # Two-stage blocking. POCA compares against the whole universe; at corpus scale in
    # pure Python that is wasteful, so a cheap bigram-overlap prefilter selects a pool,
    # full orthographic scoring runs on that pool, and the expensive ALINE alignment runs
    # only on the highest-scoring subset. Pools are generous enough that the top-k is
    # unchanged in practice; set them to 0 to disable blocking entirely.
    prefilter_pool: int = 600
    phonetic_pool: int = 150
    treat_moderate_as_failure: bool = False
    treat_implied_claim_as_failure: bool = False
    # Stem-governed generic names are REQUIRED to share a suffix with every sibling in
    # their class, which inflates raw similarity for reasons the regulator mandated. With
    # this on, the required stem is removed from both strings before V1 scoring, so the
    # screen measures the distinctiveness of the fantasy prefix -- which is the part the
    # candidate actually controls. Off by default: plain POCA behaviour is the baseline,
    # and the notebook reports both.
    stem_aware_similarity: bool = False
    enable_trademark: bool = True
    enable_crosslingual: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"weights": asdict(self.weights), "thresholds": asdict(self.thresholds),
                "phonetic_algorithm": self.phonetic_algorithm, "aline_mode": self.aline_mode,
                "top_k": self.top_k, "prefilter_pool": self.prefilter_pool,
                "phonetic_pool": self.phonetic_pool,
                "treat_moderate_as_failure": self.treat_moderate_as_failure,
                "treat_implied_claim_as_failure": self.treat_implied_claim_as_failure,
                "stem_aware_similarity": self.stem_aware_similarity,
                "enable_trademark": self.enable_trademark,
                "enable_crosslingual": self.enable_crosslingual}


# ===========================================================================
# 1. Orthographic similarity
# ===========================================================================

def normalise(name: Optional[str]) -> str:
    """Fold to the comparison form: ASCII, lowercase, letters only."""
    if name is None:
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.strip().lower()
    s = re.sub(r"[^a-z]", "", s)
    return s


def levenshtein_distance(a: str, b: str) -> int:
    """Classic Wagner-Fischer edit distance (unit cost insert/delete/substitute)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1,            # deletion
                           cur[j - 1] + 1,          # insertion
                           prev[j - 1] + (ca != cb)))  # substitution
        prev = cur
    return prev[-1]


def levenshtein_similarity(a: str, b: str) -> float:
    """Edit distance rescaled to a 0-1 similarity, as POCA's orthographic component does."""
    if not a and not b:
        return 1.0
    m = max(len(a), len(b))
    if m == 0:
        return 0.0
    return 1.0 - levenshtein_distance(a, b) / m


def bi_sim(a: str, b: str) -> float:
    """BI-SIM (Kondrak & Dorr): bigram-level similarity by dynamic programming.

    Rather than counting shared bigrams as a set (which throws away order), BI-SIM aligns
    the two bigram sequences with an LCS-style recurrence in which a pair of bigrams can
    match *partially*: matching (x_i, x_i+1) against (y_j, y_j+1) scores the fraction of
    identical corresponding letters, i.e. 0, 0.5 or 1. The alignment score is normalised
    by the length of the longer bigram sequence, so identical strings score 1.0.

    This is the measure POCA's documentation refers to as "BI-SIM", and it is the reason
    POCA catches transposition-style confusables that raw edit distance under-penalises.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0
    if n == 1 or m == 1:
        # No bigrams available; degrade gracefully to a unigram containment test.
        return 1.0 if (a in b or b in a) else 0.0
    A = [a[i:i + 2] for i in range(n - 1)]
    B = [b[j:j + 2] for j in range(m - 1)]
    na, nb = len(A), len(B)
    prev = [0.0] * (nb + 1)
    for i in range(1, na + 1):
        cur = [0.0] * (nb + 1)
        ai = A[i - 1]
        for j in range(1, nb + 1):
            bj = B[j - 1]
            s = ((ai[0] == bj[0]) + (ai[1] == bj[1])) / 2.0
            cur[j] = max(prev[j - 1] + s, prev[j], cur[j - 1])
        prev = cur
    return prev[nb] / max(na, nb)


def orthographic_similarity(a: str, b: str, weights: Optional[ScoringWeights] = None
                            ) -> Tuple[float, float, float]:
    """Returns (combined, levenshtein_component, bisim_component), all in 0-1."""
    w = (weights or ScoringWeights()).normalised()
    led = levenshtein_similarity(a, b)
    bs = bi_sim(a, b)
    return w.w_levenshtein * led + w.w_bisim * bs, led, bs


# ===========================================================================
# 2. Metaphone (phonetic ablation arm)
# ===========================================================================

_MP_VOWELS = "AEIOU"

def metaphone(word: str) -> str:
    """Lawrence Philips' original Metaphone, implemented directly from the published rules.

    Used only as the *ablation* arm: it collapses a name to a coarse consonant-skeleton
    code, which is what pre-POCA phonetic screening looked like. Comparing it against
    ALINE quantifies what the feature-based phonetic model actually buys you.
    """
    w = re.sub(r"[^A-Z]", "", (word or "").upper())
    if not w:
        return ""
    # Initial-cluster exceptions
    if w[:2] in ("AE", "GN", "KN", "PN", "WR"):
        w = w[1:]
    elif w[:1] == "X":
        w = "S" + w[1:]
    elif w[:2] == "WH":
        w = "W" + w[2:]

    out: List[str] = []
    i, n = 0, len(w)
    while i < n:
        c = w[i]
        prev = w[i - 1] if i > 0 else ""
        nxt = w[i + 1] if i + 1 < n else ""
        nxt2 = w[i + 2] if i + 2 < n else ""
        # Skip doubled letters except CC
        if c == prev and c != "C":
            i += 1
            continue
        if c in _MP_VOWELS:
            if i == 0:
                out.append(c)
        elif c == "B":
            if not (i == n - 1 and prev == "M"):
                out.append("B")
        elif c == "C":
            if nxt == "I" and nxt2 == "A":
                out.append("X")
            elif nxt == "H":
                out.append("K" if prev == "S" else "X")
                i += 1
            elif nxt in "IEY":
                if prev != "S":
                    out.append("S")
            else:
                out.append("K")
        elif c == "D":
            if nxt == "G" and nxt2 in "EYI":
                out.append("J")
                i += 2
            else:
                out.append("T")
        elif c == "G":
            if nxt == "H":
                if not (i + 2 >= n or nxt2 in _MP_VOWELS):
                    i += 1
                else:
                    out.append("K")
                    i += 1
            elif nxt == "N":
                pass  # silent in GN / GNED
            elif nxt in "IEY":
                out.append("J")
            else:
                out.append("K")
        elif c == "H":
            if prev in _MP_VOWELS and nxt not in _MP_VOWELS:
                pass
            elif prev in "CSPTG":
                pass
            else:
                out.append("H")
        elif c in "FJLMNR":
            out.append(c)
        elif c == "K":
            if prev != "C":
                out.append("K")
        elif c == "P":
            if nxt == "H":
                out.append("F")
                i += 1
            else:
                out.append("P")
        elif c == "Q":
            out.append("K")
        elif c == "S":
            if nxt == "H":
                out.append("X")
                i += 1
            elif nxt == "I" and nxt2 in "OA":
                out.append("X")
            else:
                out.append("S")
        elif c == "T":
            if nxt == "H":
                out.append("0")
                i += 1
            elif nxt == "I" and nxt2 in "OA":
                out.append("X")
            else:
                out.append("T")
        elif c == "V":
            out.append("F")
        elif c == "W":
            if nxt in _MP_VOWELS:
                out.append("W")
        elif c == "X":
            out.append("KS")
        elif c == "Y":
            if nxt in _MP_VOWELS:
                out.append("Y")
        elif c == "Z":
            out.append("S")
        i += 1
    return "".join(out)


def metaphone_similarity(a: str, b: str) -> float:
    """Normalised edit similarity between the two Metaphone codes."""
    ca, cb = metaphone(a), metaphone(b)
    if not ca and not cb:
        return 1.0
    return levenshtein_similarity(ca, cb)