"""
Induced phonotactics — a syllable grammar learned from real names.

The problem with a hand-written grammar
---------------------------------------
v1's constrained proposer sampled from a hardcoded inventory: nuclei
`{a e i o u ai ea ie oa ou ei}`, codas `{n r l s t d m x nd st rt lt ns}`, onsets
including `{sk sm sn sw th ch sh ph}`. That is a rough sketch of *English* orthography,
and it produced exactly what English orthography produces:

    skemkultolol   swoulveaxolol   jeimheistolol   mousnunsolol

Every one of those is pronounceable in the narrow sense the phonotactic check tests, and
none of them is remotely plausible as an INN name, because real INN fantasy prefixes do
not contain `ou`, `ei`, `ea`, or a `lt` coda. The hand-written inventory encoded the
wrong language.

The fix
-------
Induce the inventory from the fantasy-prefix corpus instead. Real prefixes are parsed
into onset / nucleus / coda units, the inventories and their frequencies are counted, and
new syllables are sampled from the empirical distribution conditioned on position.

This gives the proposer the property that neither of the others has: output that is
pronounceable *by construction* (like the hand-written grammar) and distributionally
plausible (like the n-gram model), while being a recombination at the syllable level
rather than a reproduction at the morpheme level. It cannot emit `erythro` unless
`e`, `ry`, `thro` all happen to be resampled in order, which is vanishingly unlikely,
so the memorisation failure mode is structurally excluded rather than penalised after
the fact.

Everything here is orthographic, not phonemic. The verifier owns phonemic analysis and
has a full rule-based grapheme-to-phoneme converter for it; duplicating that here would
create two transcription systems that could silently diverge. Letters are sufficient to
sample from, and the verifier remains the sole authority on whether the result is
actually sayable.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

VOWELS = set("aeiou")


def _vowel_groups(word: str) -> List[Tuple[int, int]]:
    """Index spans of maximal vowel runs. `y` counts as a vowel only between consonants,
    which is how it behaves in names like `oxytocin` but not in `yohimbine`."""
    spans: List[Tuple[int, int]] = []
    i = 0
    n = len(word)
    while i < n:
        c = word[i]
        is_v = c in VOWELS or (c == "y" and 0 < i < n - 1
                               and word[i - 1] not in VOWELS)
        if is_v:
            j = i
            while j < n and (word[j] in VOWELS
                             or (word[j] == "y" and 0 < j < n - 1 and word[j - 1] not in VOWELS)):
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


@dataclass
class Syllable:
    onset: str
    nucleus: str
    coda: str

    def __str__(self) -> str:
        return self.onset + self.nucleus + self.coda


def parse_syllables(word: str, legal_onsets: Optional[set] = None) -> List[Syllable]:
    """Orthographic syllabification by onset maximisation.

    A consonant run between two vowel groups is split so that the longest suffix of the
    run that forms an attested onset starts the next syllable, and the remainder closes
    the previous one. That is the standard maximal-onset principle, with "attested"
    meaning "observed word-initially in this corpus" rather than "listed by me", so the
    parser bootstraps from the same data as the generator.
    """
    spans = _vowel_groups(word)
    if not spans:
        return []
    syls: List[Syllable] = []
    for idx, (vs, ve) in enumerate(spans):
        prev_end = spans[idx - 1][1] if idx else 0
        run = word[prev_end:vs]
        if idx == 0:
            onset, carry = run, ""
        else:
            onset, carry = "", run
            # Longest suffix of `run` that is an attested onset becomes the next onset.
            for k in range(len(run), 0, -1):
                cand = run[len(run) - k:]
                if legal_onsets is None or cand in legal_onsets:
                    onset, carry = cand, run[: len(run) - k]
                    break
            if syls:
                syls[-1] = Syllable(syls[-1].onset, syls[-1].nucleus,
                                    syls[-1].coda + carry)
        coda = word[ve:] if idx == len(spans) - 1 else ""
        syls.append(Syllable(onset, word[vs:ve], coda))
    return syls


@dataclass
class InducedGrammar:
    """Position-conditioned syllable inventory learned from a corpus of real names."""
    onsets: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    nuclei: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    codas: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    syllable_counts: Counter = field(default_factory=Counter)
    transitions: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    legal_onsets: set = field(default_factory=set)
    n_train: int = 0

    # -- induction ---------------------------------------------------------
    @classmethod
    def induce(cls, words: Sequence[str], min_count: int = 2) -> "InducedGrammar":
        words = [w for w in (x.strip().lower() for x in words) if w and w.isalpha()]
        g = cls()
        g.n_train = len(words)

        # Pass 1: what can begin a word? That set defines legal onsets for pass 2.
        initial: Counter = Counter()
        for w in words:
            spans = _vowel_groups(w)
            if spans:
                initial[w[: spans[0][0]]] += 1
        g.legal_onsets = {o for o, c in initial.items() if c >= min_count} | {""}

        # Pass 2: parse with those onsets and count everything by position.
        for w in words:
            syls = parse_syllables(w, g.legal_onsets)
            if not syls:
                continue
            g.syllable_counts[len(syls)] += 1
            for i, s in enumerate(syls):
                pos = "initial" if i == 0 else ("final" if i == len(syls) - 1 else "medial")
                g.onsets[pos][s.onset] += 1
                g.nuclei[pos][s.nucleus] += 1
                g.codas[pos][s.coda] += 1
                if i:
                    g.transitions[syls[i - 1].coda][s.onset] += 1
        return g

    # -- sampling ----------------------------------------------------------
    def _pick(self, counter: Counter, rng: random.Random,
              exclude: Sequence[str] = ()) -> str:
        items = [(k, v) for k, v in counter.items() if k not in exclude]
        if not items:
            items = list(counter.items()) or [("", 1)]
        keys, weights = zip(*items)
        return rng.choices(keys, weights=weights, k=1)[0]

    def sample(self, rng: random.Random, min_len: int = 3, max_len: int = 9,
               n_syllables: Optional[int] = None, max_tries: int = 40,
               final_coda: Optional[bool] = None) -> str:
        """Draw a novel string from the induced distribution.

        Rejection-samples on length, and enforces three hard structural constraints the
        counts alone would occasionally violate: no vowel run longer than two letters,
        no consonant run longer than three, and an optional required final coda.

        `final_coda=True` is what stops a vowel-final prefix being glued to a
        vowel-initial stem. Every real `-olol` name in the corpus (aten-, metopr-,
        bisopr-, nad-, nebiv-, timol-) ends its prefix on a consonant, because
        `clohaolol` and `delsakoolol` are what happens otherwise. It is a genuine
        phonotactic fact about the language, and it is cheaper to enforce during
        construction than to detect and repair afterwards.
        """
        if not self.nuclei:
            return ""
        best = ""
        for _ in range(max_tries):
            k = n_syllables or self._weighted_syllable_count(rng)
            parts: List[str] = []
            prev_coda = ""
            for i in range(k):
                pos = "initial" if i == 0 else ("final" if i == k - 1 else "medial")
                if prev_coda and self.transitions.get(prev_coda):
                    onset = self._pick(self.transitions[prev_coda], rng)
                else:
                    onset = self._pick(self.onsets[pos], rng)
                nucleus = self._pick(self.nuclei[pos], rng)
                if i < k - 1:
                    coda = self._pick(self.codas[pos], rng)
                elif final_coda:
                    coda = self._pick(self.codas["final"], rng,
                                      exclude=("",)) or self._pick(self.codas["medial"],
                                                                   rng, exclude=("",))
                else:
                    coda = ""
                parts.append(onset + nucleus + coda)
                prev_coda = coda
            s = "".join(parts)
            if not self._well_formed(s):
                continue
            if final_coda is True and (not s or s[-1] in VOWELS):
                continue
            if final_coda is False and s and s[-1] not in VOWELS:
                continue
            if min_len <= len(s) <= max_len:
                return s
            if not best or abs(len(s) - (min_len + max_len) // 2) < abs(len(best) - (min_len + max_len) // 2):
                best = s
        return best

    def _weighted_syllable_count(self, rng: random.Random) -> int:
        if not self.syllable_counts:
            return 2
        keys, weights = zip(*self.syllable_counts.items())
        return rng.choices(keys, weights=weights, k=1)[0]

    @staticmethod
    def _well_formed(s: str) -> bool:
        vrun = crun = 0
        for ch in s:
            if ch in VOWELS:
                vrun += 1
                crun = 0
            else:
                crun += 1
                vrun = 0
            if vrun > 2 or crun > 3:
                return False
        return bool(s) and any(c in VOWELS for c in s)

    # -- reporting ---------------------------------------------------------
    def summary(self, top: int = 8) -> str:
        def fmt(c: Counter) -> str:
            return ", ".join(f"{k or 'Ø'}({v})" for k, v in c.most_common(top))
        return "\n".join([
            f"InducedGrammar  trained on {self.n_train} names, "
            f"{len(self.legal_onsets)} attested onsets",
            f"  onset  initial: {fmt(self.onsets['initial'])}",
            f"  nucleus initial: {fmt(self.nuclei['initial'])}",
            f"  coda    medial : {fmt(self.codas['medial'])}",
            f"  syllable counts: {dict(self.syllable_counts.most_common(5))}",
        ])

    def to_dict(self) -> Dict:
        return {
            "onsets": {k: dict(v) for k, v in self.onsets.items()},
            "nuclei": {k: dict(v) for k, v in self.nuclei.items()},
            "codas": {k: dict(v) for k, v in self.codas.items()},
            "syllable_counts": dict(self.syllable_counts),
            "transitions": {k: dict(v) for k, v in self.transitions.items()},
            "legal_onsets": sorted(self.legal_onsets),
            "n_train": self.n_train,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "InducedGrammar":
        g = cls()
        for attr in ("onsets", "nuclei", "codas"):
            store = getattr(g, attr)
            for pos, counts in d[attr].items():
                store[pos] = Counter(counts)
        g.syllable_counts = Counter({int(k): v for k, v in d["syllable_counts"].items()})
        for k, v in d["transitions"].items():
            g.transitions[k] = Counter(v)
        g.legal_onsets = set(d["legal_onsets"])
        g.n_train = d["n_train"]
        return g
