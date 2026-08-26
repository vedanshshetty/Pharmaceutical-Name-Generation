"""
NOMINA — Verifier suite (Person B).

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

VERIFIER_VERSION = "1.1.0"

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
    # V2b. A generic name that ENDS in the right stem may still carry a different
    # class's stem inside it: `cillinolol` and `prazololol` both passed v1, announcing
    # themselves as a penicillin and a prazole respectively while being neither. The
    # INN programme treats a misleading internal stem the same way it treats a
    # misleading terminal one, so this is a genuine conformance gap, not a nicety.
    enforce_foreign_embedded_stem: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"weights": asdict(self.weights), "thresholds": asdict(self.thresholds),
                "phonetic_algorithm": self.phonetic_algorithm, "aline_mode": self.aline_mode,
                "top_k": self.top_k, "prefilter_pool": self.prefilter_pool,
                "phonetic_pool": self.phonetic_pool,
                "treat_moderate_as_failure": self.treat_moderate_as_failure,
                "treat_implied_claim_as_failure": self.treat_implied_claim_as_failure,
                "stem_aware_similarity": self.stem_aware_similarity,
                "enable_trademark": self.enable_trademark,
                "enable_crosslingual": self.enable_crosslingual,
                "enforce_foreign_embedded_stem": self.enforce_foreign_embedded_stem}


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
# ===========================================================================
# 3. Grapheme-to-phoneme  (rule-based, tuned for pharmaceutical orthography)
# ===========================================================================
#
# POCA converts a name to a phonemic representation before phonetic comparison. A
# dictionary lookup is useless here: every candidate name is by construction a
# non-word. So the transcription has to be generated by rule.
#
# The rule set below is deliberately deterministic and self-contained (no CMUdict, no
# neural G2P, no downloads). It encodes three things: standard English grapheme rules,
# the Greek/Latinate readings that dominate pharmaceutical orthography (ch = /k/ before
# a liquid, ph = /f/, initial ps-/pn-/gn- reduction), and a table of conventional
# readings for the productive pharmaceutical suffixes.
#
# On accuracy: this transcriber is not perfect, and it does not need to be. Both the
# candidate and every corpus name pass through the *same* transcriber, so systematic
# transcription bias largely cancels in the comparison. What matters for a similarity
# screen is that two names which sound alike receive similar phone strings, and that
# property is robust to a consistent transcription convention.

VOWEL_LETTERS = set("aeiouy")

# Two-consonant sequences that are legal English syllable onsets. Used both by the
# open-syllable rule here and by the phonotactic check in V4.
LEGAL_ONSET_CLUSTERS = {
    "pl", "pr", "tr", "tw", "kl", "kr", "kw", "bl", "br", "dr", "gl", "gr",
    "fl", "fr", "thr", "thw", "sl", "sm", "sn", "sp", "st", "sk", "sw",
    "spl", "spr", "str", "skr", "skw", "shr", "sf", "vr", "hy", "my", "ny",
}

# Conventional readings for productive pharmaceutical word-endings, longest first.
# These are orthographic conventions, not a class lookup: they say how the letters are
# read, and are applied identically to real names and to novel candidates.
_PHARMA_SUFFIXES: List[Tuple[str, List[str]]] = [
    ("floxacin", ["F", "L", "AA", "K", "S", "AH", "N"]),
    ("glitazone", ["G", "L", "IH", "T", "AH", "Z", "OW", "N"]),
    ("prazole",  ["P", "R", "EY", "Z", "OW", "L"]),
    ("triptan",  ["T", "R", "IH", "P", "T", "AE", "N"]),
    ("azepam",   ["AE", "Z", "AH", "P", "AE", "M"]),
    ("gliptin",  ["G", "L", "IH", "P", "T", "IH", "N"]),
    ("cycline",  ["S", "AY", "K", "L", "IY", "N"]),
    ("sartan",   ["S", "AA", "R", "T", "AE", "N"]),
    ("statin",   ["S", "T", "AE", "T", "IH", "N"]),
    ("cillin",   ["S", "IH", "L", "IH", "N"]),
    ("dipine",   ["D", "IH", "P", "IY", "N"]),
    ("azine",    ["AH", "Z", "IY", "N"]),
    ("idine",    ["IH", "D", "IY", "N"]),
    ("mycin",    ["M", "AY", "S", "IH", "N"]),
    ("caine",    ["K", "EY", "N"]),
    ("navir",    ["N", "AH", "V", "IH", "R"]),
    ("tinib",    ["T", "IH", "N", "IH", "B"]),
    ("ciclib",   ["S", "IH", "K", "L", "IH", "B"]),
    ("leukin",   ["L", "UW", "K", "IH", "N"]),
    ("parin",    ["P", "AH", "R", "IH", "N"]),
    ("olol",     ["OW", "L", "AO", "L"]),
    ("olone",    ["OW", "L", "OW", "N"]),
    ("asone",    ["AH", "S", "OW", "N"]),
    ("tidine",   ["T", "IH", "D", "IY", "N"]),
    ("feron",    ["F", "IH", "R", "AA", "N"]),
    ("pril",     ["P", "R", "IH", "L"]),
    ("zumab",    ["Z", "UW", "M", "AE", "B"]),
    ("ximab",    ["Z", "IH", "M", "AE", "B"]),
    ("umab",     ["Y", "UW", "M", "AE", "B"]),
    ("mab",      ["M", "AE", "B"]),
    ("nib",      ["N", "IH", "B"]),
    ("vir",      ["V", "IH", "R"]),
    ("cel",      ["S", "EH", "L"]),
    ("ase",      ["EY", "Z"]),
    ("ine",      ["IY", "N"]),
    ("ide",      ["AY", "D"]),
    ("ate",      ["EY", "T"]),
    ("one",      ["OW", "N"]),
    ("ium",      ["IY", "AH", "M"]),
    ("ol",       ["AO", "L"]),
]

_SHORT_VOWEL = {"a": "AE", "e": "EH", "i": "IH", "o": "AA", "u": "AH", "y": "IH"}
_LONG_VOWEL = {"a": "EY", "e": "IY", "i": "AY", "o": "OW", "u": "UW", "y": "AY"}
_R_VOWEL = {"a": "AA", "e": "ER", "i": "ER", "o": "AO", "u": "ER", "y": "ER"}
_VOWEL_DIGRAPHS = {
    "ai": "EY", "ay": "EY", "ei": "EY", "ey": "EY", "ee": "IY", "ea": "IY",
    "ie": "IY", "oo": "UW", "ou": "AW", "ow": "AW", "oi": "OY", "oy": "OY",
    "au": "AO", "aw": "AO", "eu": "UW", "ew": "UW", "ue": "UW", "ui": "UW",
}


def _syllable_is_open(word: str, i: int) -> bool:
    """True if the vowel at position i sits in an open syllable (so it lengthens).

    Open means the following consonant material can all be parsed as the onset of the
    NEXT syllable, which is the case for a single consonant or a legal onset cluster.
    """
    j = i + 1
    while j < len(word) and word[j] not in VOWEL_LETTERS:
        j += 1
    if j >= len(word):
        return False                      # word-final -> closed
    cluster = word[i + 1:j]
    if len(cluster) == 0:
        return True                       # vowel hiatus
    if len(cluster) == 1:
        return True
    return cluster in LEGAL_ONSET_CLUSTERS


def grapheme_to_phonemes(name: str) -> List[str]:
    """Transcribe an orthographic name into a sequence of ARPAbet-style phone symbols."""
    w = normalise(name)
    if not w:
        return []

    tail: List[str] = []
    boundary = len(w)
    for suf, phones in _PHARMA_SUFFIXES:
        # Require something in front of the suffix, so the whole word is never consumed.
        if w.endswith(suf) and len(w) > len(suf) + 1:
            tail = list(phones)
            boundary = len(w) - len(suf)
            break

    # NOTE: the scanner walks the *whole* word but stops emitting at `boundary`. Slicing
    # the prefix off instead would strip away the right-hand context the open-syllable
    # and silent-e rules depend on, and mis-transcribe the seam (e.g. omeprazole).
    out: List[str] = []
    i, n = 0, len(w)
    while i < boundary:
        rest = w[i:]
        c = w[i]
        nxt = w[i + 1] if i + 1 < n else ""

        # -- word-initial reductions (Greek-derived clusters) --------------
        if i == 0:
            if rest[:2] in ("kn", "gn", "pn", "mn"):
                out.append("N"); i += 2; continue
            if rest[:2] == "ps":
                out.append("S"); i += 2; continue
            if rest[:2] == "wr":
                out.append("R"); i += 2; continue
            if c == "x":
                out.append("Z"); i += 1; continue

        # -- consonant digraphs and trigraphs -------------------------------
        if rest[:3] == "tch":
            out.append("CH"); i += 3; continue
        if rest[:3] == "sch":
            out.extend(["S", "K"]); i += 3; continue
        if rest[:2] == "ph":
            out.append("F"); i += 2; continue
        if rest[:2] == "th":
            out.append("TH"); i += 2; continue
        if rest[:2] == "sh":
            out.append("SH"); i += 2; continue
        if rest[:2] == "ch":
            # /k/ in the Greek/Latinate pattern (chlor-, chol-, chrom-, -sch-),
            # /tS/ otherwise.
            nxt2 = rest[2] if len(rest) > 2 else ""
            out.append("K" if nxt2 in ("l", "r") else "CH")
            i += 2; continue
        if rest[:2] == "ck":
            out.append("K"); i += 2; continue
        if rest[:2] == "qu":
            out.extend(["K", "W"]); i += 2; continue
        if rest[:2] == "wh":
            out.append("W"); i += 2; continue
        if rest[:2] == "gh":
            if i == 0:
                out.append("G")
            i += 2; continue
        if rest[:2] == "ng" and (i + 2 == n or w[i + 2] not in VOWEL_LETTERS):
            out.append("NG"); i += 2; continue

        # -- vowel digraphs -------------------------------------------------
        if rest[:2] in _VOWEL_DIGRAPHS:
            out.append(_VOWEL_DIGRAPHS[rest[:2]]); i += 2; continue

        # -- doubled consonants collapse ------------------------------------
        if c == nxt and c not in VOWEL_LETTERS:
            i += 1
            continue

        # -- single vowels ---------------------------------------------------
        if c in VOWEL_LETTERS:
            if c == "y" and i == 0:
                out.append("Y"); i += 1; continue
            # r-coloured: vowel + r not followed by a vowel
            if nxt == "r" and (i + 2 >= n or w[i + 2] not in VOWEL_LETTERS):
                # /ER/ is already rhotic (her = HH ER); AA/AO keep a separate R (car, for).
                rv = _R_VOWEL[c]
                out.extend([rv] if rv == "ER" else [rv, "R"])
                i += 2; continue
            # magic e:  V C e#
            if i + 2 == n - 1 and w[n - 1] == "e" and w[i + 1] not in VOWEL_LETTERS:
                out.append(_LONG_VOWEL[c]); i += 1; continue
            # word-final silent e
            if c == "e" and i == n - 1 and n > 2 and w[i - 1] not in VOWEL_LETTERS:
                i += 1; continue
            if _syllable_is_open(w, i) and c in ("o", "u"):
                # Only back/round vowels lengthen in an open syllable. Front vowels in
                # pharmaceutical orthography are read short or reduced far more often
                # than they are lengthened (metoprolol, celebrex, atorvastatin).
                out.append(_LONG_VOWEL[c])
            else:
                out.append(_SHORT_VOWEL[c])
            i += 1; continue

        # -- single consonants -----------------------------------------------
        if c == "c":
            out.append("S" if nxt in ("e", "i", "y") else "K")
        elif c == "g":
            out.append("JH" if nxt in ("e", "i", "y") else "G")
        elif c == "s":
            prev = w[i - 1] if i > 0 else ""
            out.append("Z" if (prev in VOWEL_LETTERS and nxt in VOWEL_LETTERS) else "S")
        elif c == "x":
            out.extend(["K", "S"])
        elif c == "j":
            out.append("JH")
        elif c == "h":
            out.append("HH")
        elif c in "ptkbdgfvmnlrwz":
            out.append({"p": "P", "t": "T", "k": "K", "b": "B", "d": "D", "g": "G",
                        "f": "F", "v": "V", "m": "M", "n": "N", "l": "L", "r": "R",
                        "w": "W", "z": "Z"}[c])
        i += 1

    out.extend(tail)
    return out


def phoneme_string(name: str) -> str:
    return " ".join(grapheme_to_phonemes(name))


# ===========================================================================
# 4. ALINE — feature-based phonetic alignment (Kondrak)
# ===========================================================================
#
# ALINE scores two phone strings by finding the alignment that maximises total phonetic
# similarity, where the similarity of two individual phones is a salience-weighted
# comparison of their articulatory features rather than a binary match. That is what
# lets it know that /p/ and /b/ are near-identical while /p/ and /l/ are not -- exactly
# the distinction a look-alike/sound-alike screen depends on.

C_SKIP, C_SUB, C_EXP, C_VWL = -10.0, 35.0, 45.0, 10.0

SALIENCE: Dict[str, float] = {
    "syllabic": 5, "voice": 10, "lateral": 10, "high": 5, "manner": 50, "long": 1,
    "place": 40, "nasal": 10, "aspirated": 5, "back": 5, "retroflex": 10, "round": 5,
}

# manner: stop 1.0, affricate 0.9, fricative 0.8, approximant 0.6,
#         high vowel 0.4, mid vowel 0.2, low vowel 0.0
# place:  bilabial 1.0, labiodental 0.95, dental 0.9, alveolar 0.85, retroflex 0.8,
#         palato-alveolar 0.75, palatal 0.7, velar 0.6, glottal 0.1
def _c(place, manner, voice=0.0, nasal=0.0, lateral=0.0, retroflex=0.0, rnd=0.0):
    return {"syllabic": 0.0, "place": place, "manner": manner, "voice": voice,
            "nasal": nasal, "lateral": lateral, "retroflex": retroflex,
            "aspirated": 0.0, "long": 0.0, "high": 0.0, "back": 0.0, "round": rnd}


def _v(high, back, rnd, long_=0.0, retroflex=0.0):
    manner = 0.4 if high >= 0.75 else (0.2 if high >= 0.4 else 0.0)
    return {"syllabic": 1.0, "place": 0.6, "manner": manner, "voice": 1.0,
            "nasal": 0.0, "lateral": 0.0, "retroflex": retroflex, "aspirated": 0.0,
            "long": long_, "high": high, "back": back, "round": rnd}


PHONE_FEATURES: Dict[str, Dict[str, float]] = {
    # stops
    "P": _c(1.00, 1.0), "B": _c(1.00, 1.0, voice=1),
    "T": _c(0.85, 1.0), "D": _c(0.85, 1.0, voice=1),
    "K": _c(0.60, 1.0), "G": _c(0.60, 1.0, voice=1),
    # fricatives
    "F": _c(0.95, 0.8), "V": _c(0.95, 0.8, voice=1),
    "TH": _c(0.90, 0.8), "DH": _c(0.90, 0.8, voice=1),
    "S": _c(0.85, 0.8), "Z": _c(0.85, 0.8, voice=1),
    "SH": _c(0.75, 0.8), "ZH": _c(0.75, 0.8, voice=1),
    "HH": _c(0.10, 0.8),
    # affricates
    "CH": _c(0.75, 0.9), "JH": _c(0.75, 0.9, voice=1),
    # nasals
    "M": _c(1.00, 1.0, voice=1, nasal=1), "N": _c(0.85, 1.0, voice=1, nasal=1),
    "NG": _c(0.60, 1.0, voice=1, nasal=1),
    # approximants
    "L": _c(0.85, 0.6, voice=1, lateral=1), "R": _c(0.80, 0.6, voice=1, retroflex=1),
    "Y": _c(0.70, 0.6, voice=1), "W": _c(0.60, 0.6, voice=1, rnd=1),
    # vowels  (high, back[front=1.0/central=0.5/back=0.0], round, long)
    "IY": _v(1.00, 1.0, 0.0, 1.0), "IH": _v(0.75, 1.0, 0.0, 0.0),
    "EY": _v(0.55, 1.0, 0.0, 1.0), "EH": _v(0.50, 1.0, 0.0, 0.0),
    "AE": _v(0.25, 1.0, 0.0, 0.0), "AA": _v(0.00, 0.0, 0.0, 1.0),
    "AO": _v(0.25, 0.0, 1.0, 1.0), "OW": _v(0.50, 0.0, 1.0, 1.0),
    "UH": _v(0.75, 0.0, 1.0, 0.0), "UW": _v(1.00, 0.0, 1.0, 1.0),
    "AH": _v(0.50, 0.5, 0.0, 0.0), "ER": _v(0.50, 0.5, 0.0, 1.0, retroflex=1),
    "AY": _v(0.00, 0.5, 0.0, 1.0), "AW": _v(0.00, 0.5, 1.0, 1.0),
    "OY": _v(0.25, 0.0, 1.0, 1.0),
}

_CONS_FEATURES = ("syllabic", "manner", "voice", "nasal", "retroflex",
                  "lateral", "aspirated", "long", "place")
_VOWEL_FEATURES = ("syllabic", "manner", "high", "back", "round", "long", "nasal")


def is_vowel(phone: str) -> bool:
    f = PHONE_FEATURES.get(phone)
    return bool(f and f["syllabic"] >= 1.0)


@lru_cache(maxsize=None)
def phone_delta(p: str, q: str) -> float:
    """Salience-weighted articulatory distance between two phones."""
    fp, fq = PHONE_FEATURES.get(p), PHONE_FEATURES.get(q)
    if fp is None or fq is None:
        return 0.0 if p == q else 100.0
    vp, vq = fp["syllabic"] >= 1.0, fq["syllabic"] >= 1.0
    if vp and vq:
        feats = _VOWEL_FEATURES
    elif not vp and not vq:
        feats = _CONS_FEATURES
    else:
        feats = tuple(set(_CONS_FEATURES) | set(_VOWEL_FEATURES))
    return sum(abs(fp[f] - fq[f]) * SALIENCE[f] for f in feats)


def _v_penalty(p: str) -> float:
    return C_VWL if is_vowel(p) else 0.0


@lru_cache(maxsize=None)
def sigma_sub(p: str, q: str) -> float:
    return C_SUB - phone_delta(p, q) - _v_penalty(p) - _v_penalty(q)


@lru_cache(maxsize=None)
def sigma_exp(p: str, q1: str, q2: str) -> float:
    """One phone on one side aligned against two on the other (expansion/compression)."""
    return (C_EXP - phone_delta(p, q1) - phone_delta(p, q2)
            - _v_penalty(p) - max(_v_penalty(q1), _v_penalty(q2)))


def _self_score(phones: Sequence[str]) -> float:
    return sum(sigma_sub(p, p) for p in phones)


def aline_raw(x: Sequence[str], y: Sequence[str], mode: str = "local") -> float:
    """Optimal alignment score between two phone sequences.

    mode='local'  reproduces ALINE as published (Smith-Waterman style, best-scoring
                  substring alignment).
    mode='global' forces both sequences to be consumed end to end, which is the stricter
                  reading for whole-name screening. Both are reported in the validation
                  section so the operating choice is made on evidence, not assertion.
    """
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return 0.0
    local = (mode == "local")
    S = [[0.0] * (m + 1) for _ in range(n + 1)]
    if not local:
        for i in range(1, n + 1):
            S[i][0] = S[i - 1][0] + C_SKIP
        for j in range(1, m + 1):
            S[0][j] = S[0][j - 1] + C_SKIP
    best = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cands = [S[i - 1][j - 1] + sigma_sub(x[i - 1], y[j - 1]),
                     S[i - 1][j] + C_SKIP,
                     S[i][j - 1] + C_SKIP]
            if i >= 2:
                cands.append(S[i - 2][j - 1] + sigma_exp(y[j - 1], x[i - 2], x[i - 1]))
            if j >= 2:
                cands.append(S[i - 1][j - 2] + sigma_exp(x[i - 1], y[j - 2], y[j - 1]))
            if local:
                cands.append(0.0)
            S[i][j] = max(cands)
            if S[i][j] > best:
                best = S[i][j]
    return best if local else S[n][m]


def aline_similarity(x: Sequence[str], y: Sequence[str], mode: str = "local") -> float:
    """Alignment score normalised to 0-1 (1.0 iff the phone strings are identical)."""
    if not x or not y:
        return 0.0
    raw = aline_raw(x, y, mode=mode)
    denom = _self_score(x) + _self_score(y)
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, 2.0 * raw / denom))


def phonetic_similarity(a: str, b: str, algorithm: str = "aline",
                        mode: str = "local") -> float:
    """Phonetic similarity between two orthographic names, 0-1."""
    if algorithm == "none":
        return 0.0
    if algorithm == "metaphone":
        return metaphone_similarity(a, b)
    return aline_similarity(tuple(grapheme_to_phonemes(a)),
                            tuple(grapheme_to_phonemes(b)), mode=mode)


def poca_score(a: str, b: str, config: Optional[VerifierConfig] = None) -> Dict[str, float]:
    """The full POCA-style score decomposition for one pair of names, on a 0-100 scale."""
    cfg = config or VerifierConfig()
    w = cfg.weights.normalised()
    na, nb = normalise(a), normalise(b)
    ortho, led, bs = orthographic_similarity(na, nb, w)
    phon = phonetic_similarity(na, nb, cfg.phonetic_algorithm, cfg.aline_mode)
    composite = w.w_orthographic * ortho + w.w_phonetic * phon
    return {"composite": round(100 * composite, 2), "orthographic": round(100 * ortho, 2),
            "phonetic": round(100 * phon, 2), "levenshtein": round(100 * led, 2),
            "bi_sim": round(100 * bs, 2)}
# ===========================================================================
# 5. Corpus construction and filtering
# ===========================================================================
#
# The shared data layer pulls names straight from openFDA's NDC directory, which mixes
# true drug names with OTC *product descriptions*: "target up and up morning facial
# moisturizing with spf", "meijer lidocaine pain relief patch assortment". Left in, these
# dominate the nearest-match field and distort every threshold you calibrate. This module
# reduces the raw feed to entries that are actually names.
#
# The rule, stated plainly enough to defend in the report: a corpus entry must be a
# single alphabetic token of 4-25 characters that is not a dosage form, a salt/counter-ion,
# or a marketing modifier. Multi-ingredient generic names are additionally split on
# whitespace, because each component of "acetaminophen dextromethorphan guaifenesin" is
# itself a real active-ingredient name and belongs in the comparison universe.

DOSAGE_FORM_WORDS = {
    "tablet", "tablets", "capsule", "capsules", "softgel", "softgels", "caplet", "caplets",
    "solution", "suspension", "syrup", "elixir", "injection", "injectable", "cream",
    "ointment", "gel", "lotion", "foam", "spray", "aerosol", "patch", "patches", "drops",
    "drop", "powder", "granules", "suppository", "suppositories", "lozenge", "lozenges",
    "inhaler", "inhalation", "emulsion", "paste", "shampoo", "swab", "swabs", "wipe",
    "wipes", "pad", "pads", "liquid", "kit", "syringe", "vial", "pen", "film", "strip",
    "strips", "chewable", "coated", "delayed", "extended", "release", "concentrate",
    "topical", "oral", "ophthalmic", "otic", "nasal", "rectal", "vaginal", "sterile",
    "solutions", "stick", "roll", "bar", "wash", "cleanser", "serum", "mask",
}

SALT_AND_MOIETY_WORDS = {
    "hydrochloride", "hcl", "sodium", "potassium", "calcium", "magnesium", "sulfate",
    "sulphate", "phosphate", "citrate", "tartrate", "bitartrate", "maleate", "besylate",
    "succinate", "fumarate", "acetate", "mesylate", "bromide", "chloride", "nitrate",
    "carbonate", "bicarbonate", "gluconate", "lactate", "malate", "oxalate", "stearate",
    "palmitate", "propionate", "valerate", "benzoate", "salicylate", "dihydrochloride",
    "monohydrate", "dihydrate", "anhydrous", "hydrate", "hemihydrate", "hydrobromide",
    "phosphates", "sesquihydrate", "pamoate", "decanoate", "enanthate", "cypionate",
    "acid", "base", "salt", "usp", "nf", "ph", "eur",
}

MARKETING_WORDS = {
    "extra", "strength", "maximum", "max", "original", "advanced", "complete", "daily",
    "care", "clear", "dry", "sensitive", "whitening", "mint", "cherry", "berry", "grape",
    "orange", "lemon", "honey", "night", "nighttime", "day", "daytime", "cold", "flu",
    "cough", "pain", "relief", "reliever", "fever", "headache", "allergy", "sinus",
    "chest", "sore", "throat", "muscle", "joint", "back", "body", "face", "facial",
    "hand", "foot", "feet", "skin", "hair", "scalp", "eye", "eyes", "ear", "lip",
    "baby", "kids", "kid", "children", "childrens", "child", "adult", "adults", "infant",
    "infants", "junior", "sun", "sunscreen", "sunblock", "spf", "moisturizing",
    "moisturizer", "hydrating", "protection", "protectant", "antiseptic", "sanitizer",
    "sanitizing", "antibacterial", "alcohol", "foaming", "free", "plus", "and", "with",
    "for", "the", "value", "family", "size", "count", "pack", "assortment", "formula",
    "brand", "store", "signature", "select", "premium", "natural", "organic", "pure",
    "gentle", "soothing", "cooling", "warming", "medicated", "regular", "mild",
    "professional", "clinical", "sport", "sports", "active", "ultra", "super", "mega",
    "double", "triple", "new", "improved", "wart", "remover", "acne", "anti", "aging",
    "morning", "evening", "up", "target", "meijer", "walgreens", "cvs", "equate",
    "kirkland", "rite", "aid", "good", "sense", "sunmark", "leader", "quality", "choice",
    "berkley", "jensen", "member", "mark", "basic", "essentials", "wellness", "health",
    "healthy", "first", "aid", "medicine", "medicated", "drug", "otc", "generic",
}

CORPUS_STOPWORDS = DOSAGE_FORM_WORDS | SALT_AND_MOIETY_WORDS | MARKETING_WORDS


def _acceptable_token(tok: str, min_len: int = 4, max_len: int = 25) -> bool:
    return (tok.isalpha() and min_len <= len(tok) <= max_len
            and tok not in CORPUS_STOPWORDS)


def build_corpus(names_df, split_combinations: bool = True,
                 min_len: int = 4, max_len: int = 25) -> Dict[str, Any]:
    """Turn the shared data layer's raw name table into a screening corpus.

    Returns a dict with:
        generic  : sorted list of accepted generic (nonproprietary) names
        brand    : sorted list of accepted single-token brand names
        all      : union of the two
        source   : {name -> 'generic' | 'brand' | 'both'}
        stats    : before/after counts, for the report
    """
    raw_generic = [str(x).strip().lower() for x in names_df.get("generic_name", [])
                   if isinstance(x, str) and x.strip()]
    raw_brand = [str(x).strip().lower() for x in names_df.get("brand_name", [])
                 if isinstance(x, str) and x.strip()]

    generic: set = set()
    for name in raw_generic:
        toks = [re.sub(r"[^a-z]", "", t) for t in re.split(r"[\s\-/,]+", name)]
        if len(toks) == 1:
            if _acceptable_token(toks[0], min_len, max_len):
                generic.add(toks[0])
        elif split_combinations:
            for t in toks:
                if _acceptable_token(t, min_len, max_len):
                    generic.add(t)

    brand: set = set()
    for name in raw_brand:
        toks = name.split()
        if len(toks) != 1:
            continue                      # multi-word = product description, not a mark
        t = re.sub(r"[^a-z]", "", toks[0])
        if _acceptable_token(t, min_len, max_len):
            brand.add(t)

    source: Dict[str, str] = {}
    for n in generic:
        source[n] = "generic"
    for n in brand:
        source[n] = "both" if n in generic else "brand"

    stats = {
        "raw_generic_rows": len(raw_generic),
        "raw_brand_rows": len(raw_brand),
        "raw_unique_generic": len(set(raw_generic)),
        "raw_unique_brand": len(set(raw_brand)),
        "kept_generic": len(generic),
        "kept_brand": len(brand),
        "kept_total_unique": len(set(generic) | set(brand)),
    }
    return {"generic": sorted(generic), "brand": sorted(brand),
            "all": sorted(set(generic) | set(brand)), "source": source, "stats": stats}


# A small, hand-checked sample of registered pharmaceutical marks (Nice Class 5) used to
# give the offline trademark screen coverage beyond what appears in the FDA feed. It is a
# proxy, deliberately labelled as one -- see the scope note in the module docstring.
CLASS5_TRADEMARK_SAMPLE = [
    "lipitor", "crestor", "zocor", "nexium", "prilosec", "prevacid", "plavix", "advair",
    "singulair", "synthroid", "diovan", "norvasc", "toprol", "coreg", "lopressor",
    "zestril", "prinivil", "cozaar", "avapro", "benicar", "micardis", "atacand",
    "glucophage", "januvia", "onglyza", "tradjenta", "invokana", "farxiga", "jardiance",
    "victoza", "trulicity", "ozempic", "lantus", "levemir", "humalog", "novolog",
    "humulin", "novolin", "xarelto", "eliquis", "pradaxa", "coumadin", "brilinta",
    "effient", "zetia", "vytorin", "repatha", "praluent", "lyrica", "neurontin",
    "cymbalta", "effexor", "zoloft", "paxil", "prozac", "lexapro", "celexa", "wellbutrin",
    "abilify", "seroquel", "zyprexa", "risperdal", "latuda", "vraylar", "rexulti",
    "ambien", "lunesta", "belsomra", "xanax", "ativan", "klonopin", "valium", "halcion",
    "ritalin", "adderall", "concerta", "vyvanse", "strattera", "focalin", "quillivant",
    "humira", "enbrel", "remicade", "stelara", "cosentyx", "taltz", "otezla", "xeljanz",
    "rinvoq", "dupixent", "keytruda", "opdivo", "tecentriq", "imfinzi", "yervoy",
    "herceptin", "avastin", "rituxan", "gleevec", "tarceva", "iressa", "tagrisso",
    "ibrance", "verzenio", "kisqali", "lynparza", "zejula", "rubraca", "venclexta",
    "imbruvica", "calquence", "revlimid", "pomalyst", "velcade", "kyprolis", "darzalex",
    "epogen", "procrit", "aranesp", "neulasta", "neupogen", "eliglustat", "cerezyme",
    "viagra", "cialis", "levitra", "stendra", "flomax", "avodart", "proscar", "propecia",
    "ventolin", "proair", "symbicort", "breo", "trelegy", "spiriva", "xolair", "nucala",
    "fasenra", "tezspire", "flonase", "nasonex", "zyrtec", "claritin", "allegra",
    "benadryl", "sudafed", "mucinex", "robitussin", "delsym", "afrin", "tylenol",
    "advil", "motrin", "aleve", "excedrin", "bayer", "ecotrin", "celebrex", "mobic",
    "voltaren", "vioxx", "oxycontin", "percocet", "vicodin", "dilaudid", "suboxone",
    "narcan", "lyrica", "topamax", "lamictal", "depakote", "keppra", "dilantin",
    "tegretol", "trileptal", "vimpat", "briviact", "aptiom", "fycompa", "epidiolex",
    "valtrex", "zovirax", "tamiflu", "xofluza", "paxlovid", "veklury", "truvada",
    "descovy", "biktarvy", "triumeq", "genvoya", "atripla", "isentress", "prezista",
    "harvoni", "epclusa", "mavyret", "sovaldi", "zepatier", "vosevi", "viekira",
]


# ===========================================================================
# 6. USAN / INN stem grammar  (V2)
# ===========================================================================

class StemTable:
    """The USAN/INN stem system, loaded from the shared data layer's usan_stems.csv."""

    def __init__(self, stems: Dict[str, str]):
        # stem text without the leading hyphen -> pharmacological meaning
        self.stems: Dict[str, str] = {k.strip().lstrip("-").lower(): v
                                      for k, v in stems.items() if k and k.strip("- ")}
        self._by_length = sorted(self.stems, key=len, reverse=True)

    @classmethod
    def from_dataframe(cls, df) -> "StemTable":
        return cls(dict(zip(df["stem"].astype(str), df["meaning"].astype(str))))

    def meaning(self, stem: str) -> Optional[str]:
        return self.stems.get(stem.lstrip("-").lower())

    def suffix_stems(self, name: str) -> List[str]:
        """Every stem the name ends with, longest first."""
        n = normalise(name)
        return [s for s in self._by_length if n.endswith(s) and len(n) > len(s)]

    def longest_suffix_stem(self, name: str) -> Optional[str]:
        hits = self.suffix_stems(name)
        return hits[0] if hits else None

    def embedded_stems(self, name: str, min_len: int = 4) -> List[str]:
        """Stems appearing inside the name but NOT in final position."""
        n = normalise(name)
        found = []
        for s in self._by_length:
            if len(s) < min_len:
                continue
            idx = n.find(s)
            if idx >= 0 and not n.endswith(s):
                found.append(s)
        return found

    def compatible(self, detected: str, expected: str) -> bool:
        """A longer stem that ends in the expected one is a legal specialisation.

        e.g. expected '-mab' (monoclonal antibody), detected '-zumab' (humanized
        monoclonal antibody) -- the candidate is still in the right class.
        """
        d, e = detected.lstrip("-").lower(), expected.lstrip("-").lower()
        return d == e or d.endswith(e) or e.endswith(d)


# ===========================================================================
# 7. Phonotactics and pronounceability  (V4)
# ===========================================================================

_LEGAL_ONSETS_PHONES = {
    (), ("P",), ("B",), ("T",), ("D",), ("K",), ("G",), ("F",), ("V",), ("TH",), ("DH",),
    ("S",), ("Z",), ("SH",), ("ZH",), ("HH",), ("CH",), ("JH",), ("M",), ("N",), ("L",),
    ("R",), ("Y",), ("W",),
    ("P", "L"), ("P", "R"), ("B", "L"), ("B", "R"), ("T", "R"), ("T", "W"), ("D", "R"),
    ("K", "L"), ("K", "R"), ("K", "W"), ("G", "L"), ("G", "R"), ("F", "L"), ("F", "R"),
    ("TH", "R"), ("TH", "W"), ("SH", "R"), ("S", "L"), ("S", "M"), ("S", "N"),
    ("S", "P"), ("S", "T"), ("S", "K"), ("S", "W"), ("S", "F"), ("V", "R"),
    ("P", "Y"), ("B", "Y"), ("K", "Y"), ("F", "Y"), ("V", "Y"), ("M", "Y"), ("N", "Y"),
    ("HH", "Y"), ("L", "Y"), ("G", "Y"), ("D", "Y"), ("T", "Y"),
    ("S", "P", "L"), ("S", "P", "R"), ("S", "T", "R"), ("S", "K", "R"), ("S", "K", "W"),
    ("S", "P", "Y"), ("S", "K", "Y"), ("S", "T", "Y"),
}


def syllabify(phones: Sequence[str]) -> List[List[str]]:
    """Split a phone string into syllables by the maximal-onset principle.

    Consonants between two nuclei are assigned to the following onset for as long as the
    resulting cluster stays a legal English onset; whatever is left closes the preceding
    syllable.
    """
    phones = list(phones)
    nuclei = [i for i, p in enumerate(phones) if is_vowel(p)]
    if not nuclei:
        return [phones] if phones else []

    boundaries = [0]
    for a, b in zip(nuclei, nuclei[1:]):
        cluster = phones[a + 1:b]
        split = len(cluster)                       # default: everything to the coda
        for k in range(len(cluster) + 1):
            if tuple(cluster[k:]) in _LEGAL_ONSETS_PHONES:
                split = k
                break
        boundaries.append(a + 1 + split)
    boundaries.append(len(phones))
    return [phones[boundaries[i]:boundaries[i + 1]] for i in range(len(boundaries) - 1)]


def _onset_of(syl: Sequence[str]) -> Tuple[str, ...]:
    out = []
    for p in syl:
        if is_vowel(p):
            break
        out.append(p)
    return tuple(out)


def _coda_of(syl: Sequence[str]) -> Tuple[str, ...]:
    seen_vowel = False
    out: List[str] = []
    for p in syl:
        if is_vowel(p):
            seen_vowel = True
            out = []
        elif seen_vowel:
            out.append(p)
    return tuple(out)


def phonotactic_score(phones: Sequence[str]) -> Tuple[float, Dict[str, Any]]:
    """How well-formed is this phone string as an English word? 0-1, with diagnostics."""
    if not phones:
        return 0.0, {"reason": "empty"}
    syls = syllabify(phones)
    if not any(is_vowel(p) for p in phones):
        return 0.0, {"reason": "no_vowel_nucleus", "syllables": syls}

    illegal_onsets, long_codas = [], []
    for s in syls:
        on, co = _onset_of(s), _coda_of(s)
        if on not in _LEGAL_ONSETS_PHONES:
            illegal_onsets.append(" ".join(on))
        if len(co) > 3:
            long_codas.append(" ".join(co))

    max_run, run = 0, 0
    max_vrun, vrun = 0, 0
    n_cons = 0
    for p in phones:
        if is_vowel(p):
            run = 0
            vrun += 1
        else:
            run += 1
            vrun = 0
            n_cons += 1
        max_run = max(max_run, run)
        max_vrun = max(max_vrun, vrun)

    score = 1.0
    if max_vrun > 2:
        score -= 0.20 * (max_vrun - 2)      # hiatus chains: aaaa, eoia
    if n_cons == 0:
        score -= 0.40                        # a name with no consonants at all
    score -= 0.35 * (len(illegal_onsets) / max(1, len(syls)))
    score -= 0.20 * (len(long_codas) / max(1, len(syls)))
    if max_run > 3:
        score -= 0.15 * (max_run - 3)
    n_syl = len(syls)
    if n_syl < 2:
        score -= 0.10                        # monosyllables are rare in drug nomenclature
    elif n_syl > 7:
        score -= 0.10 * (n_syl - 7)
    return max(0.0, min(1.0, score)), {
        "syllables": [" ".join(s) for s in syls], "syllable_count": n_syl,
        "illegal_onsets": illegal_onsets, "long_codas": long_codas,
        "max_consonant_run": max_run, "max_vowel_run": max_vrun,
        "consonant_count": n_cons,
    }


class CharacterTrigramModel:
    """Add-k smoothed character trigram model over the real-name corpus.

    Supplies the second half of the pronounceability score: not "is this legal English"
    but "does this look like a drug name". Reported as an empirical percentile against
    the corpus itself, so 0.5 means 'as typical as the median marketed name'.
    """

    def __init__(self, names: Iterable[str], k: float = 0.1):
        from collections import Counter
        self.k = k
        self.bi, self.tri = Counter(), Counter()
        self.vocab: set = set()
        clean = [normalise(n) for n in names]
        clean = [c for c in clean if len(c) >= 3]
        for w in clean:
            s = "^^" + w + "$"
            self.vocab.update(s)
            for i in range(len(s) - 2):
                self.bi[s[i:i + 2]] += 1
                self.tri[s[i:i + 3]] += 1
        self.v = max(1, len(self.vocab))
        self._reference = sorted(self.mean_logprob(w) for w in clean) or [0.0]
        # context -> [(next_char, count)], so the model can also generate
        self.successors: Dict[str, List[Tuple[str, int]]] = {}
        for tg, c in self.tri.items():
            self.successors.setdefault(tg[:2], []).append((tg[2], c))

    def mean_logprob(self, name: str) -> float:
        w = normalise(name)
        if len(w) < 1:
            return -99.0
        s = "^^" + w + "$"
        total, n = 0.0, 0
        for i in range(len(s) - 2):
            num = self.tri[s[i:i + 3]] + self.k
            den = self.bi[s[i:i + 2]] + self.k * self.v
            total += math.log(num / den)
            n += 1
        return total / max(1, n)

    def typicality(self, name: str) -> float:
        """Fraction of corpus names this candidate is at least as likely as. 0-1."""
        import bisect
        lp = self.mean_logprob(name)
        return bisect.bisect_left(self._reference, lp) / len(self._reference)

    def sample(self, rng, min_len: int = 5, max_len: int = 12) -> str:
        """Sample a novel string from the model. Used only to build a realistic
        stand-in generator for exercising the verifier before Person A's is ready."""
        s = "^^"
        for _ in range(200):
            opts = self.successors.get(s[-2:])
            if not opts:
                break
            chars, weights = zip(*opts)
            ch = rng.choices(chars, weights=weights, k=1)[0]
            if ch == "$":
                if len(s) - 2 >= min_len:
                    break
                continue
            s += ch
            if len(s) - 2 >= max_len:
                break
        return s[2:]


# ===========================================================================
# 8. Cross-lingual adverse meaning and implied claims  (V5)
# ===========================================================================
#
# Two distinct regulatory concerns, both handled here:
#  (a) a name that carries an alarming, offensive or clinically misleading meaning in a
#      major market language, which is a documented reason regulators and naming agencies
#      reject candidates;
#  (b) a name that implies efficacy, superiority or safety, which FDA guidance treats as
#      promotional and disallows in a proprietary name.
#
# Coverage is partial by construction: this is a curated lexicon over eight market
# languages, not a translation engine. It is a bonus module with stated limits, and the
# report must say so rather than claiming solved cross-cultural screening.

CROSSLINGUAL_LEXICON: Dict[str, List[Tuple[str, str]]] = {
    "es": [("muerte", "death"), ("morir", "to die"), ("veneno", "poison"),
           ("dolor", "pain"), ("matar", "to kill"), ("enfermo", "sick"),
           ("sangre", "blood"), ("cancer", "cancer"), ("mierda", "vulgar"),
           ("feo", "ugly"), ("loco", "insane"), ("tonto", "stupid")],
    "fr": [("mort", "death"), ("mourir", "to die"), ("poison", "poison"),
           ("douleur", "pain"), ("tuer", "to kill"), ("malade", "ill"),
           ("sang", "blood"), ("merde", "vulgar"), ("laid", "ugly"), ("fou", "insane")],
    "de": [("tod", "death"), ("toten", "to kill"), ("gift", "poison"),
           ("schmerz", "pain"), ("krank", "ill"), ("blut", "blood"),
           ("scheisse", "vulgar"), ("dumm", "stupid"), ("sterben", "to die")],
    "it": [("morte", "death"), ("veleno", "poison"), ("dolore", "pain"),
           ("malato", "ill"), ("sangue", "blood"), ("uccidere", "to kill"),
           ("merda", "vulgar"), ("brutto", "ugly")],
    "pt": [("morte", "death"), ("veneno", "poison"), ("dor", "pain"),
           ("doente", "ill"), ("sangue", "blood"), ("matar", "to kill"),
           ("merda", "vulgar"), ("burro", "stupid")],
    "ja": [("shinu", "to die"), ("shi", "death"), ("doku", "poison"),
           ("itami", "pain"), ("byouki", "illness"), ("chi", "blood"),
           ("korosu", "to kill"), ("baka", "fool")],
    "zh": [("si", "death"), ("du", "poison"), ("tong", "pain"), ("bing", "illness"),
           ("xue", "blood"), ("sha", "to kill"), ("sha", "foolish")],
    "hi": [("maut", "death"), ("zahar", "poison"), ("dard", "pain"),
           ("bimar", "ill"), ("khoon", "blood"), ("marna", "to kill")],
}

# Terms that imply efficacy, superiority or safety. FDA guidance treats a proprietary
# name carrying such a claim as promotional and unacceptable.
IMPLIED_CLAIM_TERMS: List[Tuple[str, str]] = [
    ("cure", "implies cure"), ("heal", "implies healing"), ("fix", "implies repair"),
    ("safe", "implies safety"), ("best", "implies superiority"),
    ("supreme", "implies superiority"), ("superior", "implies superiority"),
    ("perfect", "implies superiority"), ("miracle", "implies extraordinary efficacy"),
    ("power", "implies potency"), ("potent", "implies potency"),
    ("strong", "implies potency"), ("instant", "implies speed of onset"),
    ("rapid", "implies speed of onset"), ("fast", "implies speed of onset"),
    ("total", "implies completeness"), ("complete", "implies completeness"),
    ("forever", "implies permanence"), ("guard", "implies protection"),
    ("shield", "implies protection"), ("protect", "implies protection"),
    ("vital", "implies vitality"), ("youth", "implies rejuvenation"),
    ("slim", "implies weight loss"), ("thin", "implies weight loss"),
]
# ===========================================================================
# 9. The verifier
# ===========================================================================

from .contracts import (  # noqa: E402
    CandidateBatch, CandidateRequest, CheckBundle, CheckName, CrossLingualCheck,
    CrossLingualHit, FailureCode, NearestMatch, PronounceabilityCheck, RefinementSignal,
    RiskBand, Severity, SimilarityCheck, StemCheck, TargetType, TrademarkCheck,
    VerifierBatchResponse, VerifierResponse,
)


def _bigrams(s: str) -> set:
    return {s[i:i + 2] for i in range(len(s) - 1)} or {s}


class Verifier:
    """The NOMINA verifier suite: V0-V5 behind one call.

        v = Verifier.from_data_layer(data_layer)
        result = v.verify("metozolol", target_type="generic", target_stem="-olol")
        print(result.overall_pass, result.failure_codes)
    """

    def __init__(self, corpus: Dict[str, Any], stem_table: StemTable,
                 config: Optional[VerifierConfig] = None,
                 trademark_names: Optional[Sequence[str]] = None):
        self.config = config or VerifierConfig()
        self.stems = stem_table
        self.corpus = corpus
        self.names: List[str] = list(corpus["all"])
        self.source: Dict[str, str] = corpus.get("source", {})
        self._name_set = set(self.names)
        self._bigram_index = {n: _bigrams(n) for n in self.names}
        self._phoneme_cache: Dict[str, Tuple[str, ...]] = {}

        tm = set(normalise(t) for t in (trademark_names or CLASS5_TRADEMARK_SAMPLE))
        tm |= set(corpus.get("brand", []))
        self.trademark_names: List[str] = sorted(t for t in tm if t)
        self._tm_bigrams = {n: _bigrams(n) for n in self.trademark_names}

        # Generic and proprietary names occupy different orthographic distributions
        # (metoprolol vs Xanax). Scoring a brand candidate against a generic reference
        # makes every plausible brand name look atypical, so keep one model per register
        # and pick by target_type at scoring time.
        self.lm_generic = CharacterTrigramModel(corpus.get("generic") or self.names)
        brand_names = corpus.get("brand") or self.names
        self.lm_brand = CharacterTrigramModel(
            list(brand_names) + [b for b in self.trademark_names if b not in set(brand_names)])
        self.lm = self.lm_generic          # backwards-compatible default

        # names grouped by the stem they carry, for the intra-class distinctiveness test
        self._by_stem: Dict[str, List[str]] = {}
        for n in corpus.get("generic", []):
            st = self.stems.longest_suffix_stem(n)
            if st:
                self._by_stem.setdefault(st, []).append(n)

    # -- construction --------------------------------------------------------
    @classmethod
    def from_data_layer(cls, data_layer, config: Optional[VerifierConfig] = None,
                        **corpus_kwargs) -> "Verifier":
        """Build straight from the shared data layer both halves of the project import."""
        names_df = data_layer.load_existing_names()
        stems_df = data_layer.load_usan_stems()
        corpus = build_corpus(names_df, **corpus_kwargs)
        return cls(corpus, StemTable.from_dataframe(stems_df), config=config)

    # -- internals -----------------------------------------------------------
    def _phonemes(self, name: str) -> Tuple[str, ...]:
        got = self._phoneme_cache.get(name)
        if got is None:
            got = tuple(grapheme_to_phonemes(name))
            self._phoneme_cache[name] = got
        return got

    def _prefilter(self, target: str, pool: Sequence[str],
                   index: Dict[str, set], limit: int) -> List[str]:
        if not limit or limit >= len(pool):
            return list(pool)
        tb = _bigrams(target)
        scored = []
        for n in pool:
            nb = index[n]
            inter = len(tb & nb)
            if inter == 0:
                continue
            scored.append((2 * inter / (len(tb) + len(nb)), n))
        scored.sort(reverse=True)
        return [n for _, n in scored[:limit]]

    def _rank(self, target: str, pool: Sequence[str], index: Dict[str, set],
              top_k: int) -> List[Dict[str, Any]]:
        """Score `target` against `pool` and return the top_k full score decompositions."""
        cfg = self.config
        w = cfg.weights.normalised()
        shortlist = self._prefilter(target, pool, index, cfg.prefilter_pool)

        ortho_scored = []
        for n in shortlist:
            o, led, bs = orthographic_similarity(target, n, w)
            ortho_scored.append((o, led, bs, n))
        ortho_scored.sort(reverse=True)

        n_phon = cfg.phonetic_pool or len(ortho_scored)
        tp = self._phonemes(target)
        out = []
        for idx, (o, led, bs, n) in enumerate(ortho_scored):
            if idx < n_phon and cfg.phonetic_algorithm != "none":
                if cfg.phonetic_algorithm == "metaphone":
                    ph = metaphone_similarity(target, n)
                else:
                    ph = aline_similarity(tp, self._phonemes(n), mode=cfg.aline_mode)
            else:
                ph = 0.0
            comp = w.w_orthographic * o + w.w_phonetic * ph
            out.append({"name": n, "composite": round(100 * comp, 2),
                        "orthographic": round(100 * o, 2), "phonetic": round(100 * ph, 2),
                        "levenshtein": round(100 * led, 2), "bi_sim": round(100 * bs, 2),
                        "source": self.source.get(n, "corpus")})
        out.sort(key=lambda d: d["composite"], reverse=True)
        return out[:max(1, top_k)]

    def nearest(self, name: str, k: int = 5) -> List[Dict[str, Any]]:
        """Public helper: the k most similar existing names, with score decompositions."""
        return self._rank(normalise(name), self.names, self._bigram_index, k)

    # -- the checks ----------------------------------------------------------
    def _check_well_formed(self, raw: str, out: CheckBundle,
                           fb: List[RefinementSignal]) -> str:
        t = self.config.thresholds
        n = normalise(raw)
        out.well_formedness.details = {"normalised": n, "length": len(n)}
        if not n:
            out.well_formedness.passed = False
            out.well_formedness.codes.append(FailureCode.MALFORMED_CANDIDATE)
            fb.append(RefinementSignal(
                code=FailureCode.MALFORMED_CANDIDATE, check=CheckName.WELL_FORMEDNESS,
                payload={"candidate_name": raw},
                human_readable="Candidate is empty after normalisation."))
            return n
        if re.sub(r"[a-zA-Z]", "", str(raw).strip()):
            out.well_formedness.codes.append(FailureCode.NON_ALPHABETIC)
            fb.append(RefinementSignal(
                code=FailureCode.NON_ALPHABETIC, severity=Severity.WARN,
                check=CheckName.WELL_FORMEDNESS,
                payload={"candidate_name": raw, "normalised": n},
                human_readable="Non-alphabetic characters were stripped before scoring."))
        if not (t.min_length <= len(n) <= t.max_length):
            out.well_formedness.passed = False
            out.well_formedness.codes.append(FailureCode.LENGTH_OUT_OF_RANGE)
            fb.append(RefinementSignal(
                code=FailureCode.LENGTH_OUT_OF_RANGE, check=CheckName.WELL_FORMEDNESS,
                payload={"length": len(n), "min": t.min_length, "max": t.max_length},
                human_readable=f"Length {len(n)} is outside {t.min_length}-{t.max_length}."))
        return n

    def _check_similarity(self, name: str, out: CheckBundle,
                          fb: List[RefinementSignal],
                          target_stem: Optional[str] = None) -> float:
        cfg, t = self.config, self.config.thresholds
        chk = out.similarity
        chk.threshold = t.similarity_high
        stem_off = None
        if cfg.stem_aware_similarity and target_stem:
            cand = target_stem.lstrip("-").lower()
            if name.endswith(cand) and len(name) - len(cand) >= 2:
                stem_off = cand

        if name in self._name_set:
            chk.passed = False
            chk.score = 100.0
            chk.nearest_match = name
            chk.nearest_match_score = 100.0
            chk.distinctiveness_margin = t.similarity_high - 100.0
            chk.distinctiveness_margin_moderate = t.similarity_moderate - 100.0
            chk.risk_band = RiskBand.HIGH
            chk.codes.append(FailureCode.EXACT_NAME_COLLISION)
            chk.top_matches = [NearestMatch(name=name, composite=100.0, orthographic=100.0,
                                            phonetic=100.0, levenshtein=100.0, bi_sim=100.0,
                                            source=self.source.get(name, "corpus"))]
            fb.append(RefinementSignal(
                code=FailureCode.EXACT_NAME_COLLISION, check=CheckName.SIMILARITY,
                payload={"nearest_match": name, "score": 100.0},
                human_readable=f"'{name}' is already a marketed drug name."))
            return 100.0

        if stem_off:
            # Compare only the fantasy prefixes of names that share the mandated stem.
            pool = [n for n in self.names if n.endswith(stem_off)]
            trimmed = {n[:len(n) - len(stem_off)]: n for n in pool if len(n) > len(stem_off)}
            others = [n for n in self.names if not n.endswith(stem_off)]
            keys = list(trimmed)
            ranked_stem = self._rank(name[:len(name) - len(stem_off)], keys,
                                     {k: _bigrams(k) for k in keys}, cfg.top_k)
            for r in ranked_stem:
                r["name"] = trimmed[r["name"]]
                r["source"] = self.source.get(r["name"], "corpus")
            ranked_other = self._rank(name, others,
                                      {n: self._bigram_index[n] for n in others}, cfg.top_k)
            ranked = sorted(ranked_stem + ranked_other,
                            key=lambda d: d["composite"], reverse=True)[:cfg.top_k]
            chk.details = {"stem_aware": True, "discounted_stem": f"-{stem_off}"}
        else:
            ranked = self._rank(name, self.names, self._bigram_index, cfg.top_k)
        chk.top_matches = [NearestMatch(**r) for r in ranked]
        top = ranked[0] if ranked else None
        score = top["composite"] if top else 0.0
        chk.score = score
        chk.nearest_match = top["name"] if top else None
        chk.nearest_match_score = score
        chk.distinctiveness_margin = round(t.similarity_high - score, 2)
        # Both operating points are now reported. Carrying only the distance to the hard
        # cutoff let a candidate at 57 advertise a margin of 13 while sitting inside the
        # 55-70 band POCA designates for human review.
        chk.distinctiveness_margin_moderate = round(t.similarity_moderate - score, 2)
        chk.risk_band = (RiskBand.HIGH if score >= t.similarity_high
                         else RiskBand.MODERATE if score >= t.similarity_moderate
                         else RiskBand.LOW)
        chk.details.update({"scored_against": len(self.names),
                            "phonetic_algorithm": cfg.phonetic_algorithm,
                            "aline_mode": cfg.aline_mode})

        if score >= t.similarity_high:
            chk.passed = False
            chk.codes.append(FailureCode.SIMILARITY_TOO_HIGH)
            fb.append(RefinementSignal(
                code=FailureCode.SIMILARITY_TOO_HIGH, check=CheckName.SIMILARITY,
                payload={"nearest_match": top["name"], "score": score,
                         "cutoff": t.similarity_high,
                         "excess": round(score - t.similarity_high, 2),
                         "orthographic": top["orthographic"], "phonetic": top["phonetic"],
                         "shared_prefix_len": _shared_prefix_len(name, top["name"]),
                         "shared_suffix_len": _shared_suffix_len(name, top["name"])},
                human_readable=(f"Too similar to '{top['name']}' "
                                f"(composite {score:.1f} >= {t.similarity_high:.0f}).")))
        elif score >= t.similarity_moderate:
            sev = Severity.FAIL if cfg.treat_moderate_as_failure else Severity.WARN
            if cfg.treat_moderate_as_failure:
                chk.passed = False
            chk.codes.append(FailureCode.SIMILARITY_MODERATE)
            fb.append(RefinementSignal(
                code=FailureCode.SIMILARITY_MODERATE, severity=sev,
                check=CheckName.SIMILARITY,
                payload={"nearest_match": top["name"], "score": score,
                         "moderate_band": [t.similarity_moderate, t.similarity_high]},
                human_readable=(f"Moderate similarity to '{top['name']}' "
                                f"(composite {score:.1f}).")))
        return score

    def _check_stem(self, name: str, target_type: TargetType, target_stem: Optional[str],
                    out: CheckBundle, fb: List[RefinementSignal]) -> None:
        t = self.config.thresholds
        chk = out.stem_conflict
        detected = self.stems.longest_suffix_stem(name)
        chk.detected_stem = f"-{detected}" if detected else None
        chk.expected_stem = target_stem

        if target_type == TargetType.BRAND:
            if detected:
                chk.passed = False
                chk.reason = (f"proprietary name ends in the USAN stem '-{detected}' "
                              f"({self.stems.meaning(detected)})")
                chk.codes.append(FailureCode.STEM_MISUSE_IN_BRAND)
                fb.append(RefinementSignal(
                    code=FailureCode.STEM_MISUSE_IN_BRAND, check=CheckName.STEM_CONFLICT,
                    payload={"stem": f"-{detected}",
                             "stem_meaning": self.stems.meaning(detected),
                             "candidate_name": name},
                    human_readable=(f"Brand names may not carry a USAN stem in stem "
                                    f"position; '-{detected}' denotes "
                                    f"{self.stems.meaning(detected)}.")))
            else:
                for emb in self.stems.embedded_stems(name):
                    chk.codes.append(FailureCode.STEM_EMBEDDED_IN_BRAND)
                    fb.append(RefinementSignal(
                        code=FailureCode.STEM_EMBEDDED_IN_BRAND, severity=Severity.WARN,
                        check=CheckName.STEM_CONFLICT,
                        payload={"stem": f"-{emb}",
                                 "stem_meaning": self.stems.meaning(emb)},
                        human_readable=(f"Contains the stem '-{emb}' "
                                        f"({self.stems.meaning(emb)}) outside stem "
                                        f"position; reviewable but not disqualifying.")))
                    break
            return

        # ---- generic (INN/USAN) candidates ---------------------------------
        if not target_stem:
            chk.details = {"note": "no target stem supplied; stem use recorded only"}
            return

        expected = target_stem.lstrip("-").lower()
        chk.details = {"expected_meaning": self.stems.meaning(expected)}
        if not name.endswith(expected):
            chk.passed = False
            chk.reason = f"does not end in the required stem '-{expected}'"
            if detected:
                chk.codes.append(FailureCode.STEM_MISMATCH)
                code = FailureCode.STEM_MISMATCH
                msg = (f"Carries '-{detected}' ({self.stems.meaning(detected)}) but the "
                       f"target class requires '-{expected}'.")
            else:
                chk.codes.append(FailureCode.STEM_MISSING)
                code = FailureCode.STEM_MISSING
                msg = f"Generic name must end in the class stem '-{expected}'."
            fb.append(RefinementSignal(
                code=code, check=CheckName.STEM_CONFLICT,
                payload={"expected_stem": f"-{expected}",
                         "expected_meaning": self.stems.meaning(expected),
                         "detected_stem": f"-{detected}" if detected else None,
                         "candidate_name": name},
                human_readable=msg))
            return

        if detected and not self.stems.compatible(detected, expected):
            chk.passed = False
            chk.codes.append(FailureCode.STEM_MISMATCH)
            chk.reason = f"stem '-{detected}' is incompatible with '-{expected}'"
            fb.append(RefinementSignal(
                code=FailureCode.STEM_MISMATCH, check=CheckName.STEM_CONFLICT,
                payload={"expected_stem": f"-{expected}", "detected_stem": f"-{detected}"},
                human_readable=chk.reason))

        prefix = name[:len(name) - len(expected)]
        if len(prefix) < t.min_stem_prefix:
            chk.passed = False
            chk.codes.append(FailureCode.STEM_PREFIX_TOO_SHORT)
            fb.append(RefinementSignal(
                code=FailureCode.STEM_PREFIX_TOO_SHORT, check=CheckName.STEM_CONFLICT,
                payload={"prefix": prefix, "min_prefix": t.min_stem_prefix,
                         "stem": f"-{expected}"},
                human_readable=(f"Only '{prefix}' precedes the stem; the fantasy prefix "
                                f"must be at least {t.min_stem_prefix} characters.")))

        # V2b -- foreign class stems carried INSIDE a compliant generic name.
        # Terminal-stem compliance was the only thing v1 tested, so a name could end in
        # the right stem and still broadcast a second, contradictory class signal from
        # the middle of the string. Everything before the required stem is inspected;
        # the required stem itself is excluded so a name is never penalised for the
        # regulation that mandated it.
        if self.config.enforce_foreign_embedded_stem:
            body = name[: len(name) - len(expected)] if expected and name.endswith(expected) else name
            foreign = []
            for st in self.stems.stems:
                if len(st) < 4 or st == expected:
                    continue
                if st in body and not expected.endswith(st):
                    foreign.append(st)
            foreign = sorted(set(foreign), key=len, reverse=True)[:3]
            if foreign:
                chk.passed = False
                chk.codes.append(FailureCode.STEM_FOREIGN_EMBEDDED)
                chk.details["foreign_embedded_stems"] = foreign
                fb.append(RefinementSignal(
                    code=FailureCode.STEM_FOREIGN_EMBEDDED, check=CheckName.STEM_CONFLICT,
                    payload={"foreign_stems": foreign,
                             "meanings": [self.stems.meaning(f) for f in foreign],
                             "expected_stem": f"-{expected}" if expected else None,
                             "candidate_name": name,
                             "region": [0, len(body)]},
                    human_readable=(f"Carries the stem '-{foreign[0]}' "
                                    f"({self.stems.meaning(foreign[0])}) inside a name "
                                    f"declared as -{expected}; the two class signals "
                                    f"contradict each other.")))

        siblings = self._by_stem.get(expected, [])
        chk.same_stem_siblings = siblings[:20]
        if siblings:
            ranked = self._rank(name, siblings,
                                {s: self._bigram_index.get(s, _bigrams(s)) for s in siblings},
                                1)
            if ranked and ranked[0]["composite"] >= t.intra_stem_high:
                chk.passed = False
                chk.codes.append(FailureCode.INTRA_STEM_TOO_CLOSE)
                fb.append(RefinementSignal(
                    code=FailureCode.INTRA_STEM_TOO_CLOSE, check=CheckName.STEM_CONFLICT,
                    payload={"sibling": ranked[0]["name"],
                             "score": ranked[0]["composite"],
                             "cutoff": t.intra_stem_high, "stem": f"-{expected}"},
                    human_readable=(f"Not distinguishable from '{ranked[0]['name']}', "
                                    f"which shares the '-{expected}' stem "
                                    f"({ranked[0]['composite']:.1f}).")))

    def _check_trademark(self, name: str, out: CheckBundle,
                         fb: List[RefinementSignal]) -> None:
        cfg, t = self.config, self.config.thresholds
        chk = out.trademark_collision
        chk.threshold = t.trademark_high
        if not cfg.enable_trademark or not self.trademark_names:
            chk.details = {"skipped": True}
            return
        ranked = self._rank(name, self.trademark_names, self._tm_bigrams, cfg.top_k)
        hits = [r for r in ranked if r["composite"] >= t.trademark_high]
        chk.score = ranked[0]["composite"] if ranked else 0.0
        chk.conflicts = [NearestMatch(**{**r, "source": "trademark"}) for r in hits]
        chk.details = {"screened_against": len(self.trademark_names)}
        if hits:
            chk.passed = False
            chk.codes.append(FailureCode.TRADEMARK_HIT)
            fb.append(RefinementSignal(
                code=FailureCode.TRADEMARK_HIT, check=CheckName.TRADEMARK_COLLISION,
                payload={"conflicts": [h["name"] for h in hits],
                         "top_score": hits[0]["composite"], "cutoff": t.trademark_high},
                human_readable=(f"Screens as confusable with the registered mark "
                                f"'{hits[0]['name']}' ({hits[0]['composite']:.1f}).")))

    def _check_pronounceable(self, name: str, out: CheckBundle,
                             fb: List[RefinementSignal],
                             target_type: TargetType = TargetType.GENERIC) -> None:
        t = self.config.thresholds
        chk = out.pronounceability
        chk.threshold = t.pronounceability_min
        phones = list(self._phonemes(name))
        phono, diag = phonotactic_score(phones)
        lm = self.lm_brand if target_type == TargetType.BRAND else self.lm_generic
        typ = lm.typicality(name)
        score = round(0.6 * phono + 0.4 * typ, 4)
        chk.phonemes = phones
        chk.syllables = diag.get("syllables", [])
        chk.syllable_count = int(diag.get("syllable_count", 0) or 0)
        chk.score = score
        chk.details = {"phonotactic": round(phono, 4), "corpus_typicality": round(typ, 4),
                       "reference_register": ("brand" if target_type == TargetType.BRAND
                                              else "generic"),
                       "illegal_onsets": diag.get("illegal_onsets", []),
                       "max_consonant_run": diag.get("max_consonant_run"),
                       "max_vowel_run": diag.get("max_vowel_run"),
                       "phoneme_string": " ".join(phones)}
        if diag.get("reason") == "no_vowel_nucleus" or not any(is_vowel(p) for p in phones):
            chk.passed = False
            chk.codes.append(FailureCode.NO_VOWEL_NUCLEUS)
            fb.append(RefinementSignal(
                code=FailureCode.NO_VOWEL_NUCLEUS, check=CheckName.PRONOUNCEABILITY,
                payload={"phonemes": phones},
                human_readable="No vowel nucleus; the name cannot be syllabified."))
            return
        if diag.get("illegal_onsets"):
            chk.codes.append(FailureCode.ILLEGAL_ONSET_CLUSTER)
            fb.append(RefinementSignal(
                code=FailureCode.ILLEGAL_ONSET_CLUSTER, severity=Severity.WARN,
                check=CheckName.PRONOUNCEABILITY,
                payload={"onsets": diag["illegal_onsets"]},
                human_readable=(f"Illegal onset cluster(s): "
                                f"{', '.join(diag['illegal_onsets'])}.")))
        if score < t.pronounceability_min:
            chk.passed = False
            chk.codes.append(FailureCode.UNPRONOUNCEABLE)
            fb.append(RefinementSignal(
                code=FailureCode.UNPRONOUNCEABLE, check=CheckName.PRONOUNCEABILITY,
                payload={"score": score, "minimum": t.pronounceability_min,
                         "phonotactic": round(phono, 4),
                         "corpus_typicality": round(typ, 4)},
                human_readable=(f"Pronounceability {score:.2f} below the "
                                f"{t.pronounceability_min:.2f} floor.")))

    def _check_crosslingual(self, name: str, out: CheckBundle,
                            fb: List[RefinementSignal]) -> None:
        cfg, t = self.config, self.config.thresholds
        chk = out.crosslingual
        if not cfg.enable_crosslingual:
            chk.details = {"skipped": True}
            return
        chk.details = {"languages": sorted(CROSSLINGUAL_LEXICON),
                       "coverage": "curated lexicon, not a translation engine"}
        np_ = self._phonemes(name)
        hits: List[CrossLingualHit] = []
        for lang, entries in CROSSLINGUAL_LEXICON.items():
            for term, gloss in entries:
                if len(term) >= 4 and term in name:
                    hits.append(CrossLingualHit(language=lang, term=term, gloss=gloss,
                                                match_type="substring", similarity=1.0))
                    continue
                if abs(len(term) - len(name)) <= 3:
                    sim = aline_similarity(np_, self._phonemes(term), mode=cfg.aline_mode)
                    if sim >= t.crosslingual_phonetic:
                        hits.append(CrossLingualHit(language=lang, term=term, gloss=gloss,
                                                    match_type="phonetic",
                                                    similarity=round(sim, 3)))
        if hits:
            chk.passed = False
            chk.hits = hits
            chk.codes.append(FailureCode.CROSSLINGUAL_ADVERSE_MEANING)
            fb.append(RefinementSignal(
                code=FailureCode.CROSSLINGUAL_ADVERSE_MEANING, check=CheckName.CROSSLINGUAL,
                payload={"hits": [h.model_dump() for h in hits]},
                human_readable=(f"Adverse meaning in {hits[0].language}: "
                                f"'{hits[0].term}' ({hits[0].gloss}).")))

        claims = [(term, why) for term, why in IMPLIED_CLAIM_TERMS if term in name]
        if claims:
            sev = Severity.FAIL if cfg.treat_implied_claim_as_failure else Severity.WARN
            if cfg.treat_implied_claim_as_failure:
                chk.passed = False
            chk.codes.append(FailureCode.IMPLIED_CLAIM)
            chk.details["implied_claims"] = [{"term": c, "reason": w} for c, w in claims]
            fb.append(RefinementSignal(
                code=FailureCode.IMPLIED_CLAIM, severity=sev, check=CheckName.CROSSLINGUAL,
                payload={"terms": [c for c, _ in claims],
                         "reasons": [w for _, w in claims]},
                human_readable=(f"Contains '{claims[0][0]}', which {claims[0][1]}; "
                                f"promotional content is not permitted in a drug name.")))

    # -- public API ----------------------------------------------------------
    def verify(self, candidate, target_type: str = "generic",
               target_class: Optional[str] = None, target_stem: Optional[str] = None,
               **kwargs) -> VerifierResponse:
        """Screen one candidate. Accepts a CandidateRequest or a bare string."""
        t0 = time.perf_counter()
        if isinstance(candidate, CandidateRequest):
            req = candidate
        else:
            req = CandidateRequest(candidate_name=str(candidate), target_type=target_type,
                                   target_class=target_class, target_stem=target_stem,
                                   **kwargs)

        checks = CheckBundle()
        fb: List[RefinementSignal] = []
        name = self._check_well_formed(req.candidate_name, checks, fb)

        score = 0.0
        if name:
            score = self._check_similarity(name, checks, fb, req.target_stem)
            self._check_stem(name, req.target_type, req.target_stem, checks, fb)
            self._check_trademark(name, checks, fb)
            self._check_pronounceable(name, checks, fb, req.target_type)
            self._check_crosslingual(name, checks, fb)

        overall = all([checks.well_formedness.passed, checks.similarity.passed,
                       checks.stem_conflict.passed, checks.trademark_collision.passed,
                       checks.pronounceability.passed, checks.crosslingual.passed])
        band = checks.similarity.risk_band or (
            RiskBand.HIGH if score >= self.config.thresholds.similarity_high
            else RiskBand.MODERATE if score >= self.config.thresholds.similarity_moderate
            else RiskBand.LOW)
        return VerifierResponse(
            candidate_id=req.candidate_id, candidate_name=req.candidate_name,
            target_type=req.target_type, overall_pass=overall, risk_band=band,
            composite_risk_score=score, checks=checks, refinement_feedback=fb,
            verifier_version=VERIFIER_VERSION,
            timing_ms=round((time.perf_counter() - t0) * 1000, 2))

    def verify_batch(self, candidates) -> VerifierBatchResponse:
        if isinstance(candidates, CandidateBatch):
            items = candidates.candidates
        else:
            items = list(candidates)
        return VerifierBatchResponse(results=[self.verify(c) for c in items])


def _shared_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _shared_suffix_len(a: str, b: str) -> int:
    return _shared_prefix_len(a[::-1], b[::-1])
# ===========================================================================
# 10. Ground truth and evaluation
# ===========================================================================
#
# The verifier has to be shown correct ON ITS OWN TERMS -- independently of whatever the
# generator happens to emit. That means a labelled set with a real signal in it.
#
# POSITIVES: name pairs documented as confused in practice. Drawn from the ISMP list of
# confused drug names and the FDA/ISMP tall-man-lettering recommendations -- these are
# pairs that have caused real dispensing errors, which is precisely the outcome POCA-style
# screening exists to prevent.
#
# NEGATIVES: pairs of marketed names that do NOT appear on any confusion list. The
# curated set below is deliberately adversarial -- pairs that share a class, a stem or an
# initial letter, so the task is not trivially separable -- and the harness additionally
# samples random corpus pairs.

LASA_CONFUSABLE_PAIRS: List[Tuple[str, str]] = [
    ("hydralazine", "hydroxyzine"), ("clonidine", "klonopin"), ("celebrex", "celexa"),
    ("celexa", "cerebyx"), ("celebrex", "cerebyx"), ("zantac", "xanax"),
    ("lamictal", "lamisil"), ("prednisone", "prednisolone"), ("tramadol", "trazodone"),
    ("sulfadiazine", "sulfasalazine"), ("chlorpromazine", "chlorpropamide"),
    ("dobutamine", "dopamine"), ("glipizide", "glyburide"), ("humalog", "humulin"),
    ("novolog", "novolin"), ("vinblastine", "vincristine"),
    ("cycloserine", "cyclosporine"), ("daptomycin", "dactinomycin"),
    ("epinephrine", "ephedrine"), ("lorazepam", "clonazepam"), ("quinine", "quinidine"),
    ("valacyclovir", "valganciclovir"), ("risperidone", "ropinirole"),
    ("methadone", "methylphenidate"), ("hydromorphone", "morphine"),
    ("oxycodone", "oxycontin"), ("fluoxetine", "duloxetine"), ("paroxetine", "fluoxetine"),
    ("carboplatin", "cisplatin"), ("chlorpromazine", "prochlorperazine"),
    ("nicardipine", "nifedipine"), ("amlodipine", "amiodarone"),
    ("metformin", "metronidazole"), ("glucagon", "glucophage"),
    ("lantus", "lente"), ("zyprexa", "zyrtec"), ("zocor", "zyrtec"),
    ("actos", "actonel"), ("avandia", "coumadin"), ("neulasta", "lunesta"),
    ("clobetasol", "clotrimazole"), ("cefazolin", "cefotaxime"),
    ("vancomycin", "vecuronium"), ("hydrocodone", "oxycodone"),
    ("dopamine", "dobutamine"), ("tobramycin", "tobradex"),
    ("mercaptopurine", "mercaptamine"), ("azathioprine", "azithromycin"),
    ("sitagliptin", "sumatriptan"), ("zolpidem", "zolmitriptan"),
    ("acetazolamide", "acetohexamide"), ("chlorthalidone", "chlorpropamide"),
    ("desipramine", "disopyramide"), ("dimenhydrinate", "diphenhydramine"),
    ("glipizide", "glimepiride"), ("guanfacine", "guaifenesin"),
    ("lamivudine", "lamotrigine"), ("leucovorin", "leuprolide"),
    ("nelfinavir", "nevirapine"), ("olanzapine", "olsalazine"),
    ("prednisolone", "prednisone"), ("quinidine", "quinine"),
    ("sufentanil", "fentanyl"), ("tiagabine", "tizanidine"),
    ("vinorelbine", "vincristine"), ("cefepime", "cefixime"),
    ("amoxicillin", "ampicillin"), ("levothyroxine", "liothyronine"),
]
# NOTE for the write-up: verify every pair above against the CURRENT ISMP list of confused
# drug names before citing it. Those lists are revised, and a pair that is arguably
# confusable but undocumented belongs in neither column.

DISTINCT_PAIRS: List[Tuple[str, str]] = [
    # deliberately hard negatives: same class, same stem, or same initial letter,
    # but not documented as confusable
    ("metoprolol", "atenolol"), ("lisinopril", "enalapril"), ("losartan", "valsartan"),
    ("atorvastatin", "simvastatin"), ("omeprazole", "pantoprazole"),
    ("ibuprofen", "naproxen"),
    ("gabapentin", "pregabalin"), ("sertraline", "citalopram"),
    ("warfarin", "heparin"), ("furosemide", "bumetanide"),
    ("albuterol", "salmeterol"),
    ("ranitidine", "famotidine"), ("loratadine", "cetirizine"),
    ("ciprofloxacin", "levofloxacin"), ("doxycycline", "minocycline"),
    ("trastuzumab", "bevacizumab"), ("infliximab", "adalimumab"),
    ("imatinib", "dasatinib"), ("sildenafil", "tadalafil"),
    ("pioglitazone", "rosiglitazone"), ("montelukast", "zafirlukast"),
    ("clopidogrel", "prasugrel"), ("rivaroxaban", "apixaban"),
    ("metoprolol", "amoxicillin"), ("lisinopril", "trastuzumab"),
    ("omeprazole", "warfarin"), ("atorvastatin", "ibuprofen"),
    ("sildenafil", "metformin"), ("gabapentin", "rosuvastatin"),
    ("digoxin", "diltiazem"), ("spironolactone", "sotalol"),
    ("propranolol", "prednisone"), ("cephalexin", "cetirizine"),
    ("morphine", "midazolam"), ("insulin", "ibuprofen"),
    ("tamoxifen", "tamsulosin"), ("verapamil", "vardenafil"),
    ("naloxone", "naltrexone"),
]

# Candidate-level cases: each is (name, target_type, target_stem, expected failure code).
SYNTHETIC_CASES: List[Tuple[str, str, Optional[str], Optional[str]]] = [
    ("metoprolol",  "generic", "-olol",     "EXACT_NAME_COLLISION"),
    ("amoxicillin", "generic", "-cillin",   "EXACT_NAME_COLLISION"),
    ("velmab",      "brand",   None,        "STEM_MISUSE_IN_BRAND"),
    ("zextolol",    "brand",   None,        "STEM_MISUSE_IN_BRAND"),
    ("velitolol",   "generic", "-pril",     "STEM_MISMATCH"),
    ("velitanix",   "generic", "-olol",     "STEM_MISSING"),
    ("olol",        "generic", "-olol",     "STEM_PREFIX_TOO_SHORT"),
    ("bzhrkxlm",    "generic", None,        "NO_VOWEL_NUCLEUS"),
    ("aaaaaa",      "generic", None,        "UNPRONOUNCEABLE"),
    ("muerteril",   "generic", None,        "CROSSLINGUAL_ADVERSE_MEANING"),
    ("dolorexan",   "brand",   None,        "CROSSLINGUAL_ADVERSE_MEANING"),
    ("curepran",    "brand",   None,        "IMPLIED_CLAIM"),
    ("safexadol",   "brand",   None,        "IMPLIED_CLAIM"),
    ("lipitorex",   "brand",   None,        "TRADEMARK_HIT"),
    ("xanaxel",     "brand",   None,        "TRADEMARK_HIT"),
    # stem inflation: a well-formed antibody name that plain POCA scoring rejects purely
    # because every -umab name is required to share seven of its letters. Discussed in
    # the notebook; this is the case the stem_aware_similarity option exists for.
    ("nadrelumab",  "generic", "-umab",     "SIMILARITY_TOO_HIGH"),
    # controls: these should clear every check
    ("velitopril",  "generic", "-pril",     None),
    ("kirendomab",  "generic", "-mab",      None),
    ("kevantor",    "brand",   None,        None),
    ("perdaxine",   "brand",   None,        None),
]


def score_pairs(pairs: Sequence[Tuple[str, str]],
                config: Optional[VerifierConfig] = None) -> List[float]:
    cfg = config or VerifierConfig()
    return [poca_score(a, b, cfg)["composite"] for a, b in pairs]


def classification_metrics(pos_scores: Sequence[float], neg_scores: Sequence[float],
                           threshold: float) -> Dict[str, float]:
    tp = sum(1 for s in pos_scores if s >= threshold)
    fn = len(pos_scores) - tp
    fp = sum(1 for s in neg_scores if s >= threshold)
    tn = len(neg_scores) - fp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    return {"threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "specificity": round(tn / (tn + fp), 4) if (tn + fp) else 0.0}


def roc_auc(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    """Mann-Whitney U form of AUC: P(random positive scores above random negative)."""
    if not pos_scores or not neg_scores:
        return 0.5
    wins = ties = 0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos_scores) * len(neg_scores))


def threshold_sweep(pos_scores: Sequence[float], neg_scores: Sequence[float],
                    lo: float = 40.0, hi: float = 90.0, step: float = 1.0
                    ) -> List[Dict[str, float]]:
    rows, t = [], lo
    while t <= hi + 1e-9:
        rows.append(classification_metrics(pos_scores, neg_scores, round(t, 3)))
        t += step
    return rows


def sample_random_pairs(names: Sequence[str], n: int = 400, seed: int = 20260824
                        ) -> List[Tuple[str, str]]:
    """Seeded random name pairs -- the easy half of the negative set."""
    import random
    rng = random.Random(seed)
    pool = [x for x in names if len(x) >= 5]
    out, seen = [], set()
    while len(out) < n and len(pool) > 1:
        a, b = rng.choice(pool), rng.choice(pool)
        if a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b))
    return out


def evaluate_configuration(config: VerifierConfig, extra_negatives: Sequence = (),
                           label: str = "") -> Dict[str, Any]:
    """Full pair-level evaluation of one scoring configuration."""
    pos = score_pairs(LASA_CONFUSABLE_PAIRS, config)
    neg = score_pairs(list(DISTINCT_PAIRS) + list(extra_negatives), config)
    t = config.thresholds
    sweep = threshold_sweep(pos, neg)
    best = max(sweep, key=lambda r: r["f1"])
    return {
        "label": label or f"{config.phonetic_algorithm}/{config.aline_mode}",
        "n_positive": len(pos), "n_negative": len(neg),
        "auc": round(roc_auc(pos, neg), 4),
        "at_published_cutoff": classification_metrics(pos, neg, t.similarity_high),
        "at_moderate_cutoff": classification_metrics(pos, neg, t.similarity_moderate),
        "best_f1": best,
        "sweep": sweep,
        "pos_scores": pos, "neg_scores": neg,
        "mean_positive": round(sum(pos) / len(pos), 2) if pos else 0.0,
        "mean_negative": round(sum(neg) / len(neg), 2) if neg else 0.0,
    }


def evaluate_synthetic_cases(verifier: "Verifier") -> List[Dict[str, Any]]:
    """Candidate-level check: does each engineered defect trip the code it should?"""
    rows = []
    for name, ttype, stem, expected in SYNTHETIC_CASES:
        res = verifier.verify(name, target_type=ttype, target_stem=stem)
        codes = [c.value for c in res.failure_codes]
        warns = [c.value for c in res.warning_codes]
        if expected is None:
            ok = res.overall_pass
        else:
            ok = expected in codes or expected in warns
        rows.append({"candidate": name, "target_type": ttype, "target_stem": stem,
                     "expected": expected or "(should pass)", "overall_pass": res.overall_pass,
                     "codes": ", ".join(codes) or "-", "warnings": ", ".join(warns) or "-",
                     "correct": ok})
    return rows