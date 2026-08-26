"""
NOMINA — Generator suite (Person A).

Produces pharmaceutical name candidates and drives them through the Verifier's
refinement loop until a target number are accepted, or the attempt budget runs out.

Four generation strategies, matching the `generation_strategy` values `contracts.py`
already anticipates:

    llm_baseline          an LLM (Claude) proposes candidates directly, given the
                          class, the required stem, and a short "avoid list" of the
                          real names it must stay distinct from
    rejection_sampling    a character n-gram model trained on the *fantasy prefixes*
                          of real USAN names samples a prefix, the required stem is
                          appended, and anything obviously broken is filtered before
                          it is ever sent to the verifier
    constrained_decoding  candidates are built syllable-by-syllable from a legal
                          onset/nucleus/coda grammar, so pronounceability (V4) is
                          satisfied by construction rather than discovered by rejection
    rl_refined            not a trained policy (out of scope for a two-person, two-week
                          build) but a genuine reward-guided loop: every verifier
                          rejection updates a running penalty over the substrings that
                          caused it, and the *next* sample is drawn away from that
                          region of the space rather than redrawn blind

All four strategies emit *raw* candidate strings only. Turning a raw candidate into an
accepted one is `refine_candidate`'s job: it reads a `VerifierResponse.refinement_feedback`
list and edits the string to address exactly the `FailureCode`s reported, branching on the
enumerated code and its structured `payload` -- never on `human_readable`, which exists for
logs, not for control flow. That is the contract's central design rule and this module
follows it throughout.

Dependency-light by design: this module imports only `contracts.py`, stdlib, and (for
`llm_baseline`) the optional `anthropic` package. It does NOT import `verifier.py` --
the two sides stay swappable, and the same reason `verifier.py` builds its own corpus
filtering instead of reusing anything generator-side applies in reverse here: the
training corpus for the generator's language models is a different artifact, built
for a different purpose (proposing plausible prefixes), from the verifier's screening
corpus (judging collisions), and the two should not silently share code that could
diverge without a conversation.
"""

from __future__ import annotations

import math
import random
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .contracts import (
    CandidateRequest,
    CandidateBatch,
    FailureCode,
    MockVerifier,
    TargetType,
    VerifierBatchResponse,
    VerifierResponse,
)

GENERATOR_VERSION = "1.1.0"


# ===========================================================================
# 0. Configuration
# ===========================================================================

@dataclass
class GeneratorConfig:
    """Every tunable the generator uses, in one place -- mirrors the shape of the
    verifier's `VerifierConfig` so both sides can be reported the same way in the
    write-up.
    """
    min_length: int = 4
    max_length: int = 20
    min_fantasy_prefix: int = 3        # letters before the stem, generic candidates
    max_fantasy_prefix: int = 9
    min_brand_length: int = 5
    max_brand_length: int = 11

    ngram_order: int = 3               # character n-gram context length
    ngram_smoothing_k: float = 0.35
    temperature: float = 1.0           # >1 = more novel / less corpus-typical

    # Refinement loop
    max_refinement_rounds: int = 6      # edits applied to ONE lineage before giving up
    max_candidates_per_accept: int = 250  # safety valve: total raw draws per accepted name

    # LLM strategy
    llm_model: str = "google/gemini-pro"
    llm_batch_size: int = 10
    llm_avoid_list_size: int = 12

    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===========================================================================
# 1. Normalisation and corpus preparation
# ===========================================================================
#
# The generator's corpus need is different from the verifier's. The verifier needs
# the comparison UNIVERSE (every real name, so it can measure a candidate's distance
# from all of them). The generator needs TRAINING DATA for "what does a plausible
# fantasy prefix look like" -- specifically the part of a real generic name that is
# left over once its class stem is stripped, since that leftover fragment is exactly
# the freedom a namer has and exactly what this module has to learn to imitate.
#
# Kept deliberately independent of verifier.py's corpus code (see module docstring).

def normalise(name: Optional[str]) -> str:
    """Fold to the comparison form: ASCII, lowercase, letters only."""
    if name is None:
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.strip().lower()
    return re.sub(r"[^a-z]", "", s)


# Abbreviated relative to the verifier's stopword list -- this one only has to keep
# obvious non-names out of a *training* corpus, not adjudicate real collisions, so it
# does not need to be exhaustive.
_TRAINING_NOISE_WORDS = {
    "tablet", "tablets", "capsule", "capsules", "solution", "suspension", "syrup",
    "injection", "cream", "ointment", "gel", "lotion", "spray", "patch", "drops",
    "powder", "suppository", "lozenge", "inhaler", "topical", "oral", "sterile",
    "hydrochloride", "sodium", "potassium", "calcium", "sulfate", "phosphate",
    "citrate", "tartrate", "maleate", "acetate", "mesylate", "chloride", "nitrate",
    "monohydrate", "dihydrate", "anhydrous", "hydrate", "extra", "strength",
    "maximum", "original", "advanced", "daily", "care", "clear", "night", "cold",
    "flu", "cough", "pain", "relief", "allergy", "sinus", "baby", "kids", "adult",
    "sun", "sunscreen", "moisturizing", "free", "plus", "and", "with", "for", "the",
    "value", "family", "size", "count", "pack", "assortment", "formula", "brand",
    "store", "premium", "natural", "organic", "gentle", "regular", "acne", "anti",
    "up", "usp", "nf",
}


def _acceptable_token(tok: str, min_len: int, max_len: int) -> bool:
    return tok.isalpha() and min_len <= len(tok) <= max_len and tok not in _TRAINING_NOISE_WORDS


def _flatten_name_column(names_df, column: str, min_len: int, max_len: int) -> List[str]:
    """Single-token entries in `column`, cleaned. Multi-word entries are almost always
    OTC product descriptions rather than names (see verifier.py Step 6's identical
    observation) so they are dropped rather than split -- prefix training wants real
    orthographic names, not combination-product ingredient lists.
    """
    out = []
    for x in names_df.get(column, []):
        if not isinstance(x, str) or not x.strip():
            continue
        toks = x.strip().lower().split()
        if len(toks) != 1:
            continue
        t = re.sub(r"[^a-z]", "", toks[0])
        if _acceptable_token(t, min_len, max_len):
            out.append(t)
    return out


def build_prefix_corpus(names_df, stems_df, min_len: int = 3, max_len: int = 20
                         ) -> Dict[str, Any]:
    """Strip every known USAN/INN stem off every matching generic name and pool the
    leftovers -- that pool IS the fantasy-prefix training corpus. Pooled across every
    stem rather than kept per-stem: most individual stem classes have too few members
    (a handful of `-sartan` drugs, say) to train a character model on, and the thing
    being learned -- how pharma fantasy syllables tend to sound -- generalises across
    classes far better than it varies between them.
    """
    stems = sorted(
        {str(s).strip().lstrip("-").lower() for s in stems_df["stem"] if str(s).strip("- ")},
        key=len, reverse=True,
    )
    generic_tokens = _flatten_name_column(names_df, "generic_name", min_len, max_len)

    prefixes: List[str] = []
    stemmed_names: List[str] = []
    for name in set(generic_tokens):
        for st in stems:
            if name.endswith(st) and len(name) > len(st) + 1:
                prefixes.append(name[: len(name) - len(st)])
                stemmed_names.append(name)
                break

    unstemmed = sorted(set(generic_tokens) - set(stemmed_names))
    return {
        "prefixes": prefixes,
        "stems_seen": sorted(set(stems)),
        "generic_tokens": sorted(set(generic_tokens)),
        "unstemmed_generic_tokens": unstemmed,  # informative only; not primary training data
        "stats": {
            "raw_generic_rows": len(generic_tokens),
            "unique_generic_tokens": len(set(generic_tokens)),
            "prefixes_extracted": len(prefixes),
            "coverage": round(len(prefixes) / max(1, len(set(generic_tokens))), 3),
        },
    }


def build_brand_corpus(names_df, min_len: int = 5, max_len: int = 20) -> List[str]:
    """Single-token brand names, cleaned -- training data for the brand-register model."""
    return sorted(set(_flatten_name_column(names_df, "brand_name", min_len, max_len)))


def stem_lookup(stems_df, class_keyword: Optional[str] = None) -> List[Tuple[str, str]]:
    """Look up stem(s) by class keyword, e.g. stem_lookup(stems_df, 'beta-blocker').
    Mirrors `data_layer.stems_for_class` but returns normalised (no leading '-') stems
    so callers can pass the result straight into the corpus / model code above.
    """
    rows = []
    for _, row in stems_df.iterrows():
        stem, meaning = str(row["stem"]).strip(), str(row["meaning"])
        if class_keyword is None or class_keyword.lower() in meaning.lower():
            rows.append((stem.lstrip("-").lower(), meaning))
    return rows


# ===========================================================================
# 2. Character n-gram model
# ===========================================================================
#
# Same family of model as the verifier's `CharacterTrigramModel` (its Step 7), but
# trained on fantasy PREFIXES rather than whole names, with a configurable order and
# a proper context back-off so short or novel contexts degrade gracefully instead of
# refusing to generate. Independently implemented -- see the module docstring.

class CharNGramModel:
    """Add-k smoothed character n-gram model with weighted sampling and context back-off.

        lm = CharNGramModel(prefixes, order=3, seed=7)
        lm.sample(rng)                 # a novel fantasy prefix
        lm.mean_logprob("zumar")       # how "in-distribution" a string is
        lm.typicality("zumar")         # percentile against the training corpus
    """

    BOUNDARY = "^"
    END = "$"

    def __init__(self, tokens: Sequence[str], order: int = 3, k: float = 0.35):
        self.order = max(1, order)
        self.k = k
        clean = [normalise(t) for t in tokens]
        clean = [t for t in clean if t]
        self.n_training = len(clean)

        # context (tuple of chars, length up to `order`) -> Counter(next_char)
        self.successors: Dict[Tuple[str, ...], Counter] = {}
        self.start_contexts: List[Tuple[str, ...]] = []
        self._char_freq: Counter = Counter()

        for w in clean:
            s = (self.BOUNDARY * self.order) + w + self.END
            self.start_contexts.append(tuple(s[: self.order]))
            for i in range(len(s) - self.order):
                ctx = tuple(s[i : i + self.order])
                nxt = s[i + self.order]
                self.successors.setdefault(ctx, Counter())[nxt] += 1
                if nxt not in (self.BOUNDARY, self.END):
                    self._char_freq[nxt] += 1

        self.vocab: List[str] = sorted(self._char_freq) or list("aeioubcdfghjklmnpqrstvwxyz")
        self._reference_logprobs = sorted(self.mean_logprob(w) for w in clean) or [0.0]

    # -- sampling --------------------------------------------------------
    def _weighted_choice(self, rng: random.Random, counter: Counter,
                          temperature: float, avoid_bigrams: Optional[Counter] = None,
                          prev_char: Optional[str] = None) -> str:
        chars = list(counter.keys())
        weights = [float(counter[c]) for c in chars]
        if temperature != 1.0:
            weights = [max(1e-6, w) ** (1.0 / max(0.05, temperature)) for w in weights]
        if avoid_bigrams and prev_char:
            # Down-weight (never zero out) transitions that formed part of a bigram
            # the refinement loop has seen inside a REJECTED candidate's colliding
            # region -- this is the "steer away from that region of the space" half
            # of the rl_refined strategy; see Generator.generate_and_refine.
            weights = [w / (1.0 + avoid_bigrams.get(prev_char + c, 0)) for w, c in zip(weights, chars)]
        return rng.choices(chars, weights=weights, k=1)[0]

    def _next_char(self, rng: random.Random, ctx: Tuple[str, ...], temperature: float,
                    avoid_bigrams: Optional[Counter] = None,
                    prev_char: Optional[str] = None) -> str:
        """Back off to a shorter context (then to global unigram frequency) if the
        exact context was never observed -- the reason short or unusual candidate
        stubs still produce something instead of raising or looping forever.
        """
        for L in range(len(ctx), -1, -1):
            sub = ctx[len(ctx) - L :]
            counter = self.successors.get(sub)
            if counter:
                return self._weighted_choice(rng, counter, temperature, avoid_bigrams, prev_char)
        # total fallback: global unigram distribution
        chars = list(self._char_freq.keys()) or self.vocab
        weights = [self._char_freq[c] for c in chars] or [1] * len(chars)
        return rng.choices(chars, weights=weights, k=1)[0]

    def sample(self, rng: random.Random, min_len: int = 3, max_len: int = 10,
               temperature: Optional[float] = None,
               avoid_bigrams: Optional[Counter] = None) -> str:
        """Sample one novel string. Retries internally (bounded) so a run of unlucky
        early terminations doesn't return something below `min_len`.

        `avoid_bigrams`: optional `Counter` mapping two-character substrings to a
        penalty count; sampling is down-weighted (not forbidden) away from producing
        them. This is how `rl_refined` biases future draws away from whatever kept
        getting rejected earlier in the same run -- see `Generator.generate_and_refine`.
        """
        temp = self.temperature_or(temperature)
        for _ in range(40):
            ctx = rng.choice(self.start_contexts) if self.start_contexts else \
                tuple(self.BOUNDARY * self.order)
            out: List[str] = []
            for _ in range(max_len + 4):
                prev_char = out[-1] if out else None
                ch = self._next_char(rng, ctx, temp, avoid_bigrams, prev_char)
                if ch == self.END:
                    if len(out) >= min_len:
                        break
                    ctx = ctx[1:] + (self.BOUNDARY,) if len(ctx) > 1 else ctx
                    continue
                out.append(ch)
                ctx = ctx[1:] + (ch,)
                if len(out) >= max_len:
                    break
            s = "".join(out)
            if min_len <= len(s) <= max_len:
                return s
        return "".join(out)[:max_len] if out else "xyl"

    def temperature_or(self, temperature: Optional[float]) -> float:
        return 1.0 if temperature is None else temperature

    # -- scoring -----------------------------------------------------------
    def mean_logprob(self, s: str) -> float:
        w = normalise(s)
        if not w:
            return -99.0
        padded = (self.BOUNDARY * self.order) + w + self.END
        total, n = 0.0, 0
        v = max(1, len(self.vocab) + 1)  # +1 for END symbol
        for i in range(len(padded) - self.order):
            ctx = tuple(padded[i : i + self.order])
            nxt = padded[i + self.order]
            counter = self.successors.get(ctx, Counter())
            num = counter.get(nxt, 0) + self.k
            den = sum(counter.values()) + self.k * v
            total += math.log(num / den)
            n += 1
        return total / max(1, n)

    def typicality(self, s: str) -> float:
        """Fraction of the training corpus this string is at least as likely as."""
        import bisect
        lp = self.mean_logprob(s)
        return bisect.bisect_left(self._reference_logprobs, lp) / len(self._reference_logprobs)


# ===========================================================================
# 3. Constrained syllable grammar (orthographic, not phonemic)
# ===========================================================================
#
# The verifier's V4 check works from a phonemic transcription (its own rule-based
# G2P). The generator has no need to duplicate that machinery: building candidates
# from a small legal onset/nucleus/coda grammar at the LETTER level is enough to make
# most output pronounceable by construction, which is the whole point of this
# strategy -- trade some corpus-typicality (what `rejection_sampling` optimises for)
# for a much higher first-pass rate on V4 specifically.

_ONSETS_SIMPLE = ["b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z"]
_ONSETS_CLUSTER = ["br", "cr", "dr", "fr", "gr", "pr", "tr", "bl", "cl", "fl", "gl", "pl",
                    "sl", "sc", "sk", "sm", "sn", "sp", "st", "sw", "th", "ch", "sh", "ph"]
_NUCLEI = ["a", "e", "i", "o", "u", "ai", "ea", "ie", "oa", "ou", "ei"]
_CODAS_SIMPLE = ["n", "r", "l", "s", "t", "d", "m"]
_CODAS_CLUSTER = ["x", "nd", "st", "rt", "lt", "ns"]


def _sample_onset(rng: random.Random, allow_empty: bool, allow_cluster: bool) -> str:
    """Onset-maximisation, simplified: a consonant cluster onset is only offered when
    the previous syllable did NOT already close on a consonant (see caller), which is
    what keeps three- and four-consonant pileups at syllable seams from happening.
    """
    pool = ([""] * 3 if allow_empty else []) + _ONSETS_SIMPLE * 3
    if allow_cluster:
        pool = pool + _ONSETS_CLUSTER
    return rng.choice(pool)


def _sample_coda(rng: random.Random, allow_cluster: bool) -> str:
    pool = [""] * 4 + _CODAS_SIMPLE * 2
    if allow_cluster:
        pool = pool + _CODAS_CLUSTER
    return rng.choice(pool)


def constrained_syllable_string(rng: random.Random, n_syllables: int) -> str:
    """Build a pronounceable-by-construction string from `n_syllables` legal syllables,
    applying onset maximisation at each seam: if the previous syllable closed with a
    consonant, the next syllable's onset is biased toward empty/simple rather than a
    second cluster, and complex (2-3 letter) codas are reserved for the final syllable,
    where English tolerates them much more readily than mid-word.
    """
    parts: List[str] = []
    prev_coda = ""
    for i in range(n_syllables):
        is_last = i == n_syllables - 1
        onset = _sample_onset(rng, allow_empty=(i == 0 or bool(prev_coda)),
                               allow_cluster=not bool(prev_coda))
        nucleus = rng.choice(_NUCLEI)
        coda = _sample_coda(rng, allow_cluster=is_last)
        parts.append(onset + nucleus + coda)
        prev_coda = coda
    return "".join(parts)


# ===========================================================================
# 4. Generation strategies -- raw candidates only, no verification
# ===========================================================================

def strategy_rejection_sampling(rng: random.Random, lm: CharNGramModel,
                                 min_len: int, max_len: int,
                                 temperature: float = 1.0,
                                 banned: Optional[set] = None,
                                 avoid_bigrams: Optional[Counter] = None,
                                 max_tries: int = 30) -> str:
    """Draw a fragment from a trained character n-gram model, retrying (bounded)
    against a ban-list so a guided/repeated run doesn't keep re-proposing the same
    already-rejected fragment. `avoid_bigrams` (used by the rl_refined strategy)
    additionally down-weights sampling away from substrings seen in prior rejections.
    """
    banned = banned or set()
    for _ in range(max_tries):
        s = lm.sample(rng, min_len=min_len, max_len=max_len, temperature=temperature,
                       avoid_bigrams=avoid_bigrams)
        if s not in banned:
            return s
    return s  # exhausted retries; caller's refine loop will still try to fix it


def strategy_constrained_decoding(rng: random.Random, min_len: int, max_len: int,
                                   max_tries: int = 30) -> str:
    for _ in range(max_tries):
        n_syl = rng.choice([2, 2, 2, 3])
        s = constrained_syllable_string(rng, n_syl)
        if min_len <= len(s) <= max_len:
            return s
    return s[:max_len] if len(s) > max_len else s + rng.choice("aeiou")


def strategy_llm_baseline(target_class: Optional[str], target_type: TargetType,
                           target_stem: Optional[str], n: int,
                           config: GeneratorConfig,
                           avoid_names: Optional[Sequence[str]] = None,
                           api_key: Optional[str] = None,
                           client: Optional[Any] = None) -> List[str]:
    """Ask an LLM (Claude) to propose whole candidate names directly, given the
    regulatory constraints in plain language. This is the strategy that makes NOMINA
    a *generative-AI* naming tool rather than a pure statistical-model one -- the
    other three strategies are useful baselines and refinement fallbacks, but this is
    the one that can reason about semantics (e.g. avoiding words that sound like
    unrelated real drugs, or that read as marketing claims) the way a human namer would.

    Requires the `anthropic` package and an API key (see the notebook's Step for how
    to supply one via `getpass`, matching the project's existing convention for
    secrets). Raises a plain `RuntimeError` with a clear message if neither is
    available, rather than silently falling back, so a misconfigured run is loud.
    """
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "llm_baseline requires the 'openai' package for OpenRouter. Install it with "
                "`pip install openai`."
            ) from e
        
        import os
        key_to_use = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        
        # We default to OpenRouter base URL
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key_to_use
        )

    if target_type == TargetType.GENERIC:
        stem = (target_stem or "").lstrip("-")
        constraint = (
            f"Every candidate MUST end in the literal letters '{stem}' (the required "
            f"USAN/INN stem for this pharmacological class). The part before the stem "
            f"is a 'fantasy prefix' -- it must be pronounceable and must NOT itself "
            f"resemble any real drug name."
        )
    else:
        constraint = (
            "This is a PROPRIETARY (brand) name. It must NOT end in any recognised "
            "USAN/INN stem, and must NOT contain words implying safety, efficacy, or "
            "superiority (e.g. 'cure', 'best', 'safe', 'miracle')."
        )

    avoid_txt = ""
    if avoid_names:
        sample = list(avoid_names)[: config.llm_avoid_list_size]
        avoid_txt = (
            "\nCandidates that are look-alike/sound-alike close to any of these real "
            f"existing names will be REJECTED -- steer away from them: {', '.join(sample)}."
        )

    prompt = (
        f"You are naming a new pharmaceutical {target_type.value} product.\n"
        f"Pharmacological class: {target_class or '(unspecified)'}\n"
        f"{constraint}{avoid_txt}\n\n"
        f"Produce exactly {n} distinct candidate names. Rules for the output:\n"
        f"- one candidate per line, lowercase letters only (a-z), no hyphens or spaces\n"
        f"- 4 to 20 characters each\n"
        f"- no numbering, bullets, or commentary -- just the {n} names, one per line\n"
    )

    resp = client.chat.completions.create(
        model=config.llm_model,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content
    names = []
    for line in text.splitlines():
        cand = normalise(line)
        if cand:
            names.append(cand)
    return names[:n]


# ===========================================================================
# 5. Refinement engine
# ===========================================================================
#
# This is the contract's central design rule in practice: branch on `signal.code`
# and read `signal.payload`, never parse `signal.human_readable`. Each function below
# handles exactly one FailureCode and returns an edited candidate string, or None if
# the right move is to discard the lineage and draw a fresh candidate instead (e.g. a
# malformed candidate has nothing worth salvaging).

_VOWELS = set("aeiou")
_CONSONANTS = set("bcdfghjklmnpqrstvwxyz")


def _mutate_region(name: str, start: int, end: int, rng: random.Random) -> str:
    """Replace one character inside [start, end) with a different, same-class
    (vowel<->vowel, consonant<->consonant) character. Falls back to inserting a
    character if the region is empty. This is the general-purpose "make it different
    right here" move used by every similarity-family fix below.
    """
    start = max(0, min(start, len(name)))
    end = max(start, min(end, len(name)))
    if end <= start:
        pos = rng.randrange(0, len(name) + 1)
        ch = rng.choice(list(_VOWELS if pos == 0 or name[pos - 1] not in _CONSONANTS else _CONSONANTS))
        return name[:pos] + ch + name[pos:]
    pos = rng.randrange(start, end)
    cur = name[pos]
    pool = list((_VOWELS if cur in _VOWELS else _CONSONANTS) - {cur})
    new_ch = rng.choice(pool)
    return name[:pos] + new_ch + name[pos + 1 :]


def _break_consonant_run(name: str, rng: random.Random) -> str:
    """Find the longest run of consonants and insert a vowel in the middle of it --
    the direct fix for NO_VOWEL_NUCLEUS / UNPRONOUNCEABLE / ILLEGAL_ONSET_CLUSTER."""
    best_start, best_len = 0, 0
    run_start, run_len = 0, 0
    for i, ch in enumerate(name):
        if ch in _CONSONANTS or ch not in _VOWELS:
            if run_len == 0:
                run_start = i
            run_len += 1
        else:
            run_len = 0
        if run_len > best_len:
            best_len, best_start = run_len, run_start
    if best_len == 0:
        return name + rng.choice(list(_VOWELS))
    insert_at = best_start + best_len // 2 + 1
    return name[:insert_at] + rng.choice(list(_VOWELS)) + name[insert_at:]


def _strip_suffix_if_present(name: str, suffix: str) -> str:
    bare = suffix.lstrip("-").lower()
    if bare and name.endswith(bare) and len(name) > len(bare):
        return name[: len(name) - len(bare)]
    return name


def _fix_similarity(name: str, signal, rng: random.Random,
                     mutable_end: Optional[int] = None) -> str:
    """SIMILARITY_TOO_HIGH / SIMILARITY_MODERATE / EXACT_NAME_COLLISION / TRADEMARK_HIT /
    INTRA_STEM_TOO_CLOSE all reduce to the same move: mutate inside the region the
    candidate shares with the colliding name, using the shared-prefix/suffix lengths
    the verifier reports so the edit lands where it actually helps.

    `mutable_end` caps how far into the string an edit may land -- for a generic
    candidate this is set to "before the mandated stem" by the caller, so a
    similarity fix never rewrites the letters a STEM check will then immediately
    complain about again. Without this, the two fixes can fight each other forever:
    similarity mutates into the stem, stem-missing re-appends it, similarity mutates
    it again.
    """
    cap = len(name) if mutable_end is None else max(1, min(mutable_end, len(name)))
    payload = signal.payload
    spl = payload.get("shared_prefix_len")
    ssl = payload.get("shared_suffix_len")
    if spl is not None or ssl is not None:
        spl, ssl = spl or 0, ssl or 0
        start = max(0, min(spl - 1, cap - 1))
        end = min(cap, len(name) - max(0, ssl - 1))
        if end <= start:
            end = min(cap, start + 1)
        return _mutate_region(name[:cap], start, end, rng) + name[cap:]
    # No prefix/suffix decomposition available (trademark / intra-stem / exact match):
    # mutate the middle third of the mutable region.
    third = max(1, cap // 3)
    return _mutate_region(name[:cap], third, cap - third, rng) + name[cap:]


def _fix_stem(name: str, signal, target_type: TargetType, rng: random.Random) -> str:
    code = signal.code
    payload = signal.payload
    if code == FailureCode.STEM_MISSING:
        return name + payload.get("expected_stem", "").lstrip("-")
    if code == FailureCode.STEM_MISMATCH:
        stripped = _strip_suffix_if_present(name, payload.get("detected_stem") or "")
        return stripped + payload.get("expected_stem", "").lstrip("-")
    if code == FailureCode.STEM_PREFIX_TOO_SHORT:
        extra = rng.choice(_ONSETS_SIMPLE) + rng.choice(_NUCLEI)
        return extra + name
    if code in (FailureCode.STEM_MISUSE_IN_BRAND, FailureCode.STEM_EMBEDDED_IN_BRAND):
        stripped = _strip_suffix_if_present(name, payload.get("stem", ""))
        # Neutral, non-stem-looking ending -- deliberately NOT drawn from the pharma
        # suffix table so it doesn't reintroduce a real stem by accident.
        return stripped + rng.choice(["ex", "ix", "or", "ara", "eo", "yn"])
    return name


def _fix_crosslingual(name: str, signal, rng: random.Random) -> str:
    hits = signal.payload.get("hits") or []
    term = hits[0]["term"] if hits else None
    if term and term in name:
        idx = name.find(term)
        return _mutate_region(name, idx, idx + len(term), rng)
    return _mutate_region(name, 0, len(name), rng)


def _fix_implied_claim(name: str, signal, rng: random.Random) -> str:
    terms = signal.payload.get("terms") or []
    for t in terms:
        if t in name:
            idx = name.find(t)
            name = name[:idx] + name[idx + len(t) :]
    return name if len(name) >= 3 else name + rng.choice(list(_VOWELS))


# Priority order: fix the most structurally fundamental problem first, since a single
# refinement round only applies ONE edit and re-verifies -- fixing well-formedness
# before similarity is pointless work if the name is about to be discarded anyway.
_FIX_PRIORITY: List[FailureCode] = [
    FailureCode.MALFORMED_CANDIDATE,
    FailureCode.NON_ALPHABETIC,
    FailureCode.LENGTH_OUT_OF_RANGE,
    FailureCode.NO_VOWEL_NUCLEUS,
    FailureCode.UNPRONOUNCEABLE,
    FailureCode.ILLEGAL_ONSET_CLUSTER,
    FailureCode.STEM_MISSING,
    FailureCode.STEM_MISMATCH,
    FailureCode.STEM_PREFIX_TOO_SHORT,
    FailureCode.STEM_MISUSE_IN_BRAND,
    FailureCode.EXACT_NAME_COLLISION,
    FailureCode.SIMILARITY_TOO_HIGH,
    FailureCode.INTRA_STEM_TOO_CLOSE,
    FailureCode.TRADEMARK_HIT,
    FailureCode.SIMILARITY_MODERATE,
    FailureCode.CROSSLINGUAL_ADVERSE_MEANING,
    FailureCode.IMPLIED_CLAIM,
    FailureCode.STEM_EMBEDDED_IN_BRAND,
]

# Codes that mean "this lineage is unsalvageable; draw a fresh candidate instead".
_REGENERATE_CODES = {FailureCode.MALFORMED_CANDIDATE, FailureCode.NON_ALPHABETIC,
                      FailureCode.LENGTH_OUT_OF_RANGE}


def refine_candidate(name: str, response: VerifierResponse, target_type: TargetType,
                      rng: random.Random) -> Optional[str]:
    """Apply one edit addressing the highest-priority FAIL signal in `response`.

    Returns the edited candidate, or None if the response calls for discarding this
    lineage and drawing a fresh candidate from scratch.
    """
    fails = {s.code: s for s in response.refinement_feedback if s.severity.value == "fail"}
    if not fails:
        return name  # nothing to fix (shouldn't normally be called in this state)

    # Cap similarity-driven mutations to land before the mandated stem, for generic
    # candidates, so a similarity fix can't undo a stem fix (see _fix_similarity).
    mutable_end = None
    if target_type == TargetType.GENERIC:
        expected = response.checks.stem_conflict.expected_stem
        if expected:
            bare = expected.lstrip("-")
            if name.endswith(bare):
                mutable_end = len(name) - len(bare)

    for code in _FIX_PRIORITY:
        if code not in fails:
            continue
        if code in _REGENERATE_CODES:
            return None
        signal = fails[code]
        if code in (FailureCode.NO_VOWEL_NUCLEUS, FailureCode.UNPRONOUNCEABLE,
                    FailureCode.ILLEGAL_ONSET_CLUSTER):
            return _break_consonant_run(name, rng)
        if code in (FailureCode.STEM_MISSING, FailureCode.STEM_MISMATCH,
                    FailureCode.STEM_PREFIX_TOO_SHORT, FailureCode.STEM_MISUSE_IN_BRAND):
            return _fix_stem(name, signal, target_type, rng)
        if code in (FailureCode.EXACT_NAME_COLLISION, FailureCode.SIMILARITY_TOO_HIGH,
                    FailureCode.INTRA_STEM_TOO_CLOSE, FailureCode.TRADEMARK_HIT,
                    FailureCode.SIMILARITY_MODERATE):
            return _fix_similarity(name, signal, rng, mutable_end=mutable_end)
        if code == FailureCode.CROSSLINGUAL_ADVERSE_MEANING:
            return _fix_crosslingual(name, signal, rng)
        if code == FailureCode.IMPLIED_CLAIM:
            return _fix_implied_claim(name, signal, rng)
    # Only warning-severity feedback remains (e.g. STEM_EMBEDDED_IN_BRAND) but
    # overall_pass was already False for some FAIL not covered above -- shouldn't
    # normally happen given _FIX_PRIORITY covers every FailureCode; regenerate as a
    # safe fallback rather than looping on an unfixable state.
    return None


# ===========================================================================
# 6. Orchestration: the Generator class
# ===========================================================================

@dataclass
class GenerationResult:
    """One completed lineage: either accepted, or abandoned after the attempt budget."""
    candidate_name: str
    accepted: bool
    strategy: str
    rounds: int
    lineage: List[str]                          # every string this candidate passed through
    final_response: Optional[VerifierResponse] = None

    def to_row(self) -> Dict[str, Any]:
        r = self.final_response
        return {
            "candidate_name": self.candidate_name,
            "accepted": self.accepted,
            "strategy": self.strategy,
            "rounds": self.rounds,
            "lineage_length": len(self.lineage),
            "composite_risk_score": r.composite_risk_score if r else None,
            "distinctiveness_margin": r.checks.similarity.distinctiveness_margin if r else None,
            "nearest_match": r.checks.similarity.nearest_match if r else None,
            "pronounceability": r.checks.pronounceability.score if r else None,
        }


class Generator:
    """The NOMINA generator: proposes candidates and drives them through the
    verifier's refinement loop.

        gen = Generator.from_data_layer(data_layer)
        results, stats = gen.generate_and_refine(
            verifier, n_accepted=20, target_type="generic",
            target_class="beta-blocker", target_stem="-olol",
            strategy="rejection_sampling")
    """

    def __init__(self, names_df, stems_df, config: Optional[GeneratorConfig] = None,
                 seed: Optional[int] = None):
        self.config = config or GeneratorConfig()
        self.rng = random.Random(self.config.seed if seed is None else seed)
        self.stems_df = stems_df
        self.prefix_corpus = build_prefix_corpus(names_df, stems_df)
        self.brand_names = build_brand_corpus(names_df, min_len=self.config.min_brand_length,
                                               max_len=self.config.max_brand_length)
        self.lm_prefix = CharNGramModel(self.prefix_corpus["prefixes"],
                                         order=self.config.ngram_order,
                                         k=self.config.ngram_smoothing_k)
        self.lm_brand = CharNGramModel(self.brand_names, order=self.config.ngram_order,
                                        k=self.config.ngram_smoothing_k)
        self.all_real_names = sorted(set(self.prefix_corpus["generic_tokens"]) | set(self.brand_names))

    @classmethod
    def from_data_layer(cls, data_layer_module, config: Optional[GeneratorConfig] = None,
                         seed: Optional[int] = None) -> "Generator":
        """Build straight from the shared data layer both halves of the project import."""
        return cls(data_layer_module.load_existing_names(), data_layer_module.load_usan_stems(),
                    config=config, seed=seed)

    # -- raw candidate proposal -------------------------------------------------
    def _propose_raw(self, strategy: str, target_type: TargetType,
                      target_stem: Optional[str], target_class: Optional[str],
                      temperature: float, banned: Optional[set] = None,
                      avoid_bigrams: Optional[Counter] = None) -> str:
        cfg = self.config
        if target_type == TargetType.GENERIC:
            bare_stem = (target_stem or "").lstrip("-")
            if strategy in ("rejection_sampling", "rl_refined"):
                prefix = strategy_rejection_sampling(
                    self.rng, self.lm_prefix, cfg.min_fantasy_prefix, cfg.max_fantasy_prefix,
                    temperature=temperature, banned=banned, avoid_bigrams=avoid_bigrams)
                return prefix + bare_stem
            if strategy == "constrained_decoding":
                prefix = strategy_constrained_decoding(
                    self.rng, cfg.min_fantasy_prefix, cfg.max_fantasy_prefix)
                return prefix + bare_stem
            raise ValueError(f"raw proposal not defined for strategy={strategy!r}")
        else:  # brand
            if strategy in ("rejection_sampling", "rl_refined"):
                return strategy_rejection_sampling(
                    self.rng, self.lm_brand, cfg.min_brand_length, cfg.max_brand_length,
                    temperature=temperature, banned=banned, avoid_bigrams=avoid_bigrams)
            if strategy == "constrained_decoding":
                return strategy_constrained_decoding(
                    self.rng, cfg.min_brand_length, cfg.max_brand_length)
            raise ValueError(f"raw proposal not defined for strategy={strategy!r}")

    def generate_raw_batch(self, n: int, target_type: str = "generic",
                            target_stem: Optional[str] = None,
                            target_class: Optional[str] = None,
                            strategy: str = "rejection_sampling") -> List[str]:
        """Unverified candidates only -- for eyeballing what each strategy proposes
        before spending verifier calls on it."""
        ttype = TargetType(target_type)
        if strategy == "llm_baseline":
            avoid = self._nearest_real_names_for_stem(target_stem) if ttype == TargetType.GENERIC \
                else self.rng.sample(self.brand_names, min(self.config.llm_avoid_list_size,
                                                             len(self.brand_names)))
            return strategy_llm_baseline(target_class, ttype, target_stem, n, self.config,
                                          avoid_names=avoid)
        out, seen = [], set()
        while len(out) < n:
            c = self._propose_raw(strategy, ttype, target_stem, target_class, self.config.temperature)
            if 4 <= len(c) <= self.config.max_length and c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def _nearest_real_names_for_stem(self, target_stem: Optional[str]) -> List[str]:
        if not target_stem:
            return self.rng.sample(self.all_real_names,
                                    min(self.config.llm_avoid_list_size, len(self.all_real_names)))
        bare = target_stem.lstrip("-")
        siblings = [n for n in self.prefix_corpus["generic_tokens"] if n.endswith(bare)]
        return siblings[: self.config.llm_avoid_list_size] or self.rng.sample(
            self.all_real_names, min(self.config.llm_avoid_list_size, len(self.all_real_names)))

    # -- the refinement loop ------------------------------------------------
    def generate_and_refine(self, verifier: Any, n_accepted: int,
                             target_type: str = "generic",
                             target_class: Optional[str] = None,
                             target_stem: Optional[str] = None,
                             strategy: str = "rejection_sampling",
                             max_total_attempts: Optional[int] = None,
                             llm_batch: Optional[List[str]] = None
                             ) -> Tuple[List[GenerationResult], Dict[str, Any]]:
        """Generate candidates with `strategy`, verify each one, and refine rejects
        for up to `config.max_refinement_rounds` edits before abandoning that lineage
        and drawing a fresh candidate. Stops once `n_accepted` candidates are
        accepted or the total attempt budget is exhausted.

        `verifier` is anything exposing `.verify(name, target_type=..., target_class=...,
        target_stem=...)` -> VerifierResponse -- the real `Verifier` from verifier.py,
        or `contracts.MockVerifier` for a dependency-free dev loop.

        `llm_batch`: for strategy='llm_baseline', an already-fetched batch of raw
        candidate strings (from `generate_raw_batch`) can be supplied directly, since
        each LLM call produces several candidates at once and re-calling the API
        inside the attempt loop one name at a time would be wasteful and slow.
        """
        cfg = self.config
        ttype = TargetType(target_type)
        max_total = max_total_attempts or (n_accepted * cfg.max_candidates_per_accept)

        results: List[GenerationResult] = []
        total_attempts = 0
        rejected_penalty_ngrams: Counter = Counter()   # rl_refined session-level state
        banned_prefixes: set = set()                    # rl_refined session-level state
        llm_pool = list(llm_batch) if llm_batch else []
        n_ok = 0

        while n_ok < n_accepted and total_attempts < max_total:
            # -- draw a fresh starting candidate -----------------------------
            if strategy == "llm_baseline":
                if not llm_pool:
                    avoid = self._nearest_real_names_for_stem(target_stem)
                    llm_pool = strategy_llm_baseline(
                        target_class, ttype, target_stem, cfg.llm_batch_size, cfg,
                        avoid_names=avoid)
                    if not llm_pool:
                        break  # LLM returned nothing usable; stop rather than spin
                name = llm_pool.pop(0)
            elif strategy == "rl_refined":
                temp = cfg.temperature * (1.0 + 0.05 * min(10, len(rejected_penalty_ngrams)))
                name = self._propose_raw(strategy, ttype, target_stem, target_class,
                                          temperature=temp, banned=banned_prefixes,
                                          avoid_bigrams=rejected_penalty_ngrams)
            else:
                name = self._propose_raw(strategy, ttype, target_stem, target_class,
                                          temperature=cfg.temperature)

            lineage = [name]
            response = None
            rounds = 0
            accepted = False

            for rounds in range(1, cfg.max_refinement_rounds + 1):
                total_attempts += 1
                # Always go through a CandidateRequest rather than the real Verifier's
                # bare-string-plus-kwargs shortcut: MockVerifier only accepts the
                # request object, and building one explicitly keeps this loop working
                # unchanged against either verifier, which is the entire point of
                # MockVerifier existing.
                request = CandidateRequest(candidate_name=name, target_type=ttype,
                                            target_class=target_class, target_stem=target_stem,
                                            generation_strategy=strategy)
                response = verifier.verify(request)
                if response.overall_pass:
                    accepted = True
                    break
                if strategy == "rl_refined":
                    banned_prefixes.add(lineage[0])
                    for sig in response.refinement_feedback:
                        for key in ("nearest_match", "sibling"):
                            v = sig.payload.get(key)
                            if v:
                                rejected_penalty_ngrams.update(v[i:i + 2] for i in range(len(v) - 1))
                if total_attempts >= max_total:
                    break
                nxt = refine_candidate(name, response, ttype, self.rng)
                if nxt is None or nxt == name:
                    break  # unsalvageable or stuck; abandon this lineage
                name = nxt
                lineage.append(name)

            results.append(GenerationResult(
                candidate_name=name, accepted=accepted, strategy=strategy,
                rounds=rounds, lineage=lineage, final_response=response))
            if accepted:
                n_ok += 1

        stats = {
            "strategy": strategy, "target_type": target_type, "target_class": target_class,
            "target_stem": target_stem, "requested": n_accepted, "accepted": n_ok,
            "lineages_started": len(results), "total_verifier_calls": total_attempts,
            "pass_rate": round(n_ok / len(results), 4) if results else 0.0,
            "candidates_per_accept": round(total_attempts / max(1, n_ok), 2),
            "mean_rounds_to_accept": round(
                sum(r.rounds for r in results if r.accepted) / max(1, n_ok), 2),
        }
        return results, stats

    # -- comparison across strategies ---------------------------------------
    def compare_strategies(self, verifier: Any, target_type: str, target_class: str,
                            target_stem: Optional[str], strategies: Sequence[str],
                            n_accepted_each: int = 15) -> List[Dict[str, Any]]:
        """Run `generate_and_refine` once per strategy under identical conditions and
        return the comparison rows the write-up needs: pass_rate and
        candidates-per-accepted-name, per strategy.
        """
        rows = []
        for strat in strategies:
            t0 = time.perf_counter()
            _, stats = self.generate_and_refine(
                verifier, n_accepted_each, target_type=target_type, target_class=target_class,
                target_stem=target_stem, strategy=strat)
            stats["wall_seconds"] = round(time.perf_counter() - t0, 2)
            rows.append(stats)
        return rows


def export_generation_report(results: Sequence[GenerationResult], path_prefix: str = "generator_results") -> Tuple[str, str]:
    """Write the accepted shortlist and the full attempt log as CSVs (no pandas
    dependency here -- kept to stdlib `csv` so this module's only hard dependency
    stays `contracts.py`)."""
    import csv
    accepted_path = f"{path_prefix}_accepted.csv"
    full_path = f"{path_prefix}_all_attempts.csv"
    fieldnames = ["candidate_name", "accepted", "strategy", "rounds", "lineage_length",
                  "composite_risk_score", "distinctiveness_margin", "nearest_match",
                  "pronounceability"]
    with open(full_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r.to_row())
    with open(accepted_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            if r.accepted:
                w.writerow(r.to_row())
    return accepted_path, full_path


if __name__ == "__main__":
    import argparse
    try:
        import data_layer
    except ImportError:
        print("Error: data_layer.py not found.")
        exit(1)
        
    parser = argparse.ArgumentParser(description="Run NOMINA Generator")
    parser.add_argument("--type", type=str, choices=["generic", "brand"], default="generic")
    parser.add_argument("--class-keyword", type=str, default="beta-blocker")
    parser.add_argument("--stem", type=str, default="-olol")
    parser.add_argument("--strategy", type=str, default="rejection_sampling",
                        choices=["llm_baseline", "rejection_sampling", "constrained_decoding", "rl_refined"])
    parser.add_argument("--n-accepted", type=int, default=5)
    parser.add_argument("--mock-verifier", action="store_true", help="Use MockVerifier instead of real Verifier")
    args = parser.parse_args()

    print(f"Loading data layer...")
    gen = Generator.from_data_layer(data_layer)
    
    if args.mock_verifier:
        from contracts import MockVerifier
        verifier = MockVerifier()
        print("Using MockVerifier.")
    else:
        try:
            from verifier import Verifier
            verifier = Verifier.from_data_layer(data_layer)
            print("Using real Verifier.")
        except ImportError:
            print("WARNING: verifier.py not found. Falling back to MockVerifier.")
            print("(Ensure you have merged the Verifier branch to get verifier.py and contracts.py)")
            from contracts import MockVerifier
            verifier = MockVerifier()

    print(f"Running generation: target={args.type}, class={args.class_keyword}, stem={args.stem}, strategy={args.strategy}")
    results, stats = gen.generate_and_refine(
        verifier=verifier,
        n_accepted=args.n_accepted,
        target_type=args.type,
        target_class=args.class_keyword,
        target_stem=args.stem,
        strategy=args.strategy
    )
    
    print("\n=== Generation Stats ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
        
    print("\n=== Accepted Candidates ===")
    for r in results:
        if r.accepted:
            print(f"- {r.candidate_name} (Rounds: {r.rounds}, Lineage: {' -> '.join(r.lineage)})")
            
    accepted_csv, all_csv = export_generation_report(results)
    print(f"\nSaved reports to {accepted_csv} and {all_csv}")
