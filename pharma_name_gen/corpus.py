"""
Corpus construction.

One snapshot, two derived corpora, one place that documents the difference.

In v1 these were built independently on each side of the project, with different
tokenisers and different stopword lists, and the divergence was invisible: the verifier
screened against 1,918 generic names while the generator trained on 420, because the
generator silently dropped every multi-word entry. Same source file, two different
pictures of reality, no note anywhere saying so. That is now a deliberate, recorded
split with a stated reason for each filter:

    ScreeningCorpus  — the comparison UNIVERSE. Recall matters more than precision:
                       a junk token in here costs a slightly conservative margin, a
                       missing real name costs a false clearance, which is the failure
                       mode that actually harms patients. Filters are permissive.

    TrainingCorpus   — what the statistical proposer imitates. Precision matters more:
                       every junk token in here is something the model may learn to
                       emit. Filters are strict, and multi-word entries are SPLIT rather
                       than dropped, which is the single change that took the fantasy
                       prefix pool from 86 strings to a usable size.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

def fold(name: Optional[str]) -> str:
    """ASCII-fold to lowercase letters only. The single normalisation both corpora and
    every candidate string pass through, so comparisons are never accidentally made
    between differently-folded forms."""
    if name is None:
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", s.strip().lower())


_SPLIT = re.compile(r"[\s\-/,;()\[\]+.]+")


def tokenise(entry: Optional[str]) -> List[str]:
    """Split a corpus entry into candidate name tokens."""
    if not isinstance(entry, str) or not entry.strip():
        return []
    return [t for t in (fold(p) for p in _SPLIT.split(entry.strip().lower())) if t]


# --------------------------------------------------------------------------
# Stopwords
# --------------------------------------------------------------------------

DOSAGE_FORM_WORDS = {
    "tablet", "tablets", "capsule", "capsules", "solution", "suspension", "syrup",
    "injection", "injectable", "cream", "ointment", "gel", "lotion", "spray", "patch",
    "drops", "drop", "powder", "suppository", "lozenge", "inhaler", "inhalation",
    "topical", "oral", "sterile", "kit", "film", "coated", "extended", "release",
    "delayed", "chewable", "granules", "emulsion", "foam", "shampoo", "paste", "elixir",
    "concentrate", "pellets", "implant", "aerosol", "nebulizer", "intravenous",
}

SALT_AND_MOIETY_WORDS = {
    "hydrochloride", "hcl", "sodium", "potassium", "calcium", "magnesium", "sulfate",
    "sulphate", "phosphate", "citrate", "tartrate", "maleate", "acetate", "mesylate",
    "besylate", "fumarate", "succinate", "chloride", "nitrate", "bromide", "iodide",
    "monohydrate", "dihydrate", "anhydrous", "hydrate", "bitartrate", "gluconate",
    "lactate", "carbonate", "bicarbonate", "oxide", "hydroxide", "stearate", "palmitate",
    "propionate", "valerate", "dipropionate", "furoate", "xinafoate", "trihydrate",
    "disodium", "dihydrochloride", "hemihydrate", "tosylate", "pamoate", "decanoate",
}

MARKETING_WORDS = {
    "extra", "strength", "maximum", "original", "advanced", "daily", "care", "clear",
    "night", "cold", "flu", "cough", "pain", "relief", "allergy", "sinus", "baby",
    "kids", "adult", "children", "childrens", "junior", "sun", "sunscreen", "spf",
    "moisturizing", "free", "plus", "and", "with", "for", "the", "value", "family",
    "size", "count", "pack", "assortment", "formula", "brand", "store", "premium",
    "natural", "organic", "gentle", "regular", "acne", "anti", "up", "usp", "nf",
    "new", "improved", "fast", "acting", "long", "lasting", "sugar", "alcohol",
    "compound", "complex", "mixture", "combination", "generic", "professional",
}

CLINICAL_NOISE_WORDS = {
    "human", "prescription", "drug", "otc", "vaccine", "allergenic", "extract",
    "unapproved", "homeopathic", "cellular", "therapy", "plasma", "derivative", "blood",
    "water", "purified", "dextrose", "saline", "sodiumchloride", "acid", "vitamin",
}

# Ordinary English vocabulary that appears in OTC product descriptions. These are not
# distinctive marks, and treating them as collision targets produces false positives
# without protecting anyone: a screen that reports `yellow` or `crayola` as a
# candidate's nearest neighbour is reporting noise, and a reviewer who sees that once
# stops trusting the nearest-neighbour column entirely.
COMMON_ENGLISH_WORDS = {
    "yellow", "white", "black", "green", "blue", "brown", "pink", "purple", "orange",
    "silver", "gold", "grey", "gray", "violet", "amber", "cherry", "grape", "lemon",
    "lime", "mint", "berry", "vanilla", "honey", "cocoa", "coconut", "almond", "olive",
    "lavender", "rose", "citrus", "apple", "peach", "melon", "ginger", "garlic",
    "hand", "face", "body", "skin", "hair", "foot", "feet", "eye", "eyes", "ear",
    "nose", "lip", "lips", "tooth", "teeth", "mouth", "nail", "scalp", "throat",
    "sanitizer", "wipes", "wipe", "swab", "swabs", "pads", "pad", "wash", "soap",
    "cleanser", "toner", "serum", "mask", "balm", "stick", "roll", "spray", "mist",
    "clean", "fresh", "pure", "soft", "smooth", "bright", "glow", "repair", "renew",
    "protect", "shield", "guard", "defense", "boost", "active", "sport", "cool",
    "warm", "deep", "light", "heavy", "dry", "wet", "oil", "oils", "butter", "cream",
    "milk", "water", "juice", "tea", "coffee", "sugar", "salt", "corn", "rice",
    "wheat", "soy", "egg", "fish", "beef", "pork", "chicken", "cattle", "bovine",
    "porcine", "equine", "canine", "feline", "grass", "weed", "tree", "pollen",
    "dust", "mite", "mold", "yeast", "bacteria", "virus", "vaccine", "typhoid",
    "cold", "flu", "cough", "fever", "headache", "nausea", "stress", "sleep", "energy",
    "immune", "digestion", "detox", "cleanse", "remover", "removal", "treatment",
    "wart", "acne", "rash", "burn", "cut", "wound", "first", "aid", "care", "plus",
    "total", "complete", "ultra", "super", "mega", "max", "mini", "large", "small",
    "double", "triple", "twin", "multi", "all", "day", "night", "morning", "evening",
    "smart", "simple", "basic", "essential", "select", "choice", "signature", "classic",
}

# Latin binomial and homeopathic nomenclature. Homeopathic OTC entries list ingredients
# as Latin species and materia medica names (`avena sativa`, `kalmia latifolia`,
# `cysteinum`), which enter the corpus as tokens and then surface as nearest neighbours
# for unrelated candidates. They are ingredient nomenclature, not marketed names.
HOMEOPATHIC_LATIN = {
    "sativa", "sativus", "vulgaris", "vulgare", "officinalis", "officinale",
    "latifolia", "latifolium", "canadensis", "americana", "americanum", "communis",
    "nigra", "nigrum", "alba", "album", "rubra", "rubrum", "crudum", "crude",
    "carbonicum", "muriaticum", "sulphuricum", "phosphoricum", "aceticum",
    "nitricum", "arsenicosum", "metallicum", "cysteinum", "histaminum", "hepar",
    "glandula", "suprarenalis", "berberis", "avena", "kalmia", "beta", "foeniculum",
    "fucus", "vesiculosus", "arnica", "belladonna", "bryonia", "calendula",
    "chamomilla", "gelsemium", "ignatia", "lycopodium", "nux", "pulsatilla",
    "rhus", "sepia", "silicea", "sulphur", "thuja", "urtica", "aconitum", "aconite",
    "apis", "argentum", "aurum", "baptisia", "cantharis", "carbo", "causticum",
    "cimicifuga", "cina", "cocculus", "colocynthis", "conium", "crataegus", "cuprum",
    "digitalis", "drosera", "dulcamara", "echinacea", "euphrasia", "ferrum",
    "graphites", "hamamelis", "hydrastis", "hypericum", "iodium", "ipecacuanha",
    "kali", "lachesis", "ledum", "magnesia", "mercurius", "natrum", "petroleum",
    "phosphorus", "phytolacca", "plantago", "podophyllum", "ruta", "sabadilla",
    "sanguinaria", "sarsaparilla", "senega", "spongia", "staphysagria", "stramonium",
    "symphytum", "tabacum", "veratrum", "viburnum", "zincum", "oxalacetate",
}

STOPWORDS: Set[str] = (DOSAGE_FORM_WORDS | SALT_AND_MOIETY_WORDS
                       | MARKETING_WORDS | CLINICAL_NOISE_WORDS
                       | COMMON_ENGLISH_WORDS | HOMEOPATHIC_LATIN)


# --------------------------------------------------------------------------
# Corpora
# --------------------------------------------------------------------------

@dataclass
class ScreeningCorpus:
    """The universe the verifier measures distance from."""
    generic: List[str]
    brand: List[str]
    all: List[str]
    source: Dict[str, str]
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingCorpus:
    """What the statistical proposer learns to imitate."""
    prefixes: List[str]                  # fantasy prefixes: real generic minus its stem
    prefixes_by_stem: Dict[str, List[str]]
    brand_names: List[str]
    generic_tokens: List[str]
    stem_index: Dict[str, str]           # bare stem -> meaning, longest-first order kept
    morpheme_blocklist: Set[str]         # class-signalling fragments a new name must not reuse
    stats: Dict[str, Any] = field(default_factory=dict)


# Latin-nomenclature endings. Applied ONLY to tokens drawn from multi-ingredient
# homeopathic entries, never to single-ingredient prescription names, because plenty of
# real INN names end in -a or -um (`levodopa`, `sodium`-free forms) and a blanket rule
# would delete them from the screening universe.
_LATIN_ENDINGS = ("icum", "osum", "ium", "eum", "aris", "alis", "ensis", "issima")


def _acceptable(tok: str, min_len: int, max_len: int, drop_stopwords: bool = True,
                latin_context: bool = False) -> bool:
    if not tok.isalpha() or not (min_len <= len(tok) <= max_len):
        return False
    if drop_stopwords and tok in STOPWORDS:
        return False
    if latin_context and tok.endswith(_LATIN_ENDINGS):
        return False
    return True


def _is_homeopathic_list(entry: Optional[str]) -> bool:
    """A long comma-separated ingredient list of Latin materia medica.

    Deliberately narrow. Ordinary combination drugs are also comma-separated
    ("acetaminophen, dextromethorphan hydrobromide, guaifenesin") and must stay in the
    corpus, so the test requires BOTH a long list AND at least two recognised
    homeopathic Latin terms in it.
    """
    if not isinstance(entry, str):
        return False
    parts = [p.strip().lower() for p in entry.split(",")]
    if len(parts) < 3:
        return False
    hits = sum(1 for p in parts
               for w in p.split()
               if re.sub(r"[^a-z]", "", w) in HOMEOPATHIC_LATIN)
    return hits >= 2


def build_screening_corpus(names_df, min_len: int = 4, max_len: int = 25) -> ScreeningCorpus:
    """Permissive: split every entry, keep every plausible token.

    Recall-first by design. A token that is not really a marketed name adds at most a
    spurious near-neighbour, which makes the screen slightly conservative. A real name
    that never enters the universe produces a candidate cleared against a name it
    actually collides with, which is the error that reaches a pharmacy shelf.
    """
    generic: Set[str] = set()
    brand: Set[str] = set()

    dropped = 0
    for entry in names_df.get("generic_name", []):
        latin = _is_homeopathic_list(entry)
        for t in tokenise(entry):
            if _acceptable(t, min_len, max_len, latin_context=latin):
                generic.add(t)
            else:
                dropped += 1
    for entry in names_df.get("brand_name", []):
        latin = _is_homeopathic_list(entry)
        # A multi-token brand entry is a product description ("tylenol extra strength"),
        # so keep only tokens that are not obvious descriptors rather than the whole run.
        for t in tokenise(entry):
            if _acceptable(t, min_len, max_len, latin_context=latin):
                brand.add(t)
            else:
                dropped += 1

    source: Dict[str, str] = {n: "generic" for n in generic}
    for n in brand:
        source[n] = "both" if n in generic else "brand"

    return ScreeningCorpus(
        generic=sorted(generic), brand=sorted(brand),
        all=sorted(generic | brand), source=source,
        stats={"kept_generic": len(generic), "kept_brand": len(brand),
               "kept_total_unique": len(generic | brand),
               "tokens_filtered": dropped,
               "raw_rows": int(len(names_df))},
    )


def build_training_corpus(names_df, stems_df,
                          min_prefix: int = 3, max_prefix: int = 9,
                          min_brand: int = 5, max_brand: int = 12) -> TrainingCorpus:
    """Strict: only clean, single-morpheme tokens the model should imitate.

    Two changes from v1 do most of the work here. Multi-word generic entries are split
    instead of discarded, which multiplies the token pool; and the stem table is the
    expanded INN/USAN one, which multiplies the fraction of those tokens whose stem can
    actually be identified and stripped. Between them the fantasy-prefix pool goes from
    86 strings, small enough that an order-3 character model simply memorised them, to
    a pool large enough to interpolate rather than recite.
    """
    stem_rows = [(str(r.stem).strip().lower(), str(r.meaning))
                 for r in stems_df.itertuples(index=False)]
    bare_stems = sorted(
        {(s.strip("-"), m) for s, m in stem_rows if s.strip("-")},
        key=lambda x: len(x[0]), reverse=True)
    stem_index = {s: m for s, m in bare_stems}
    suffix_stems = [s for s, _ in bare_stems]

    generic_tokens: Set[str] = set()
    for entry in names_df.get("generic_name", []):
        for t in tokenise(entry):
            if _acceptable(t, 4, 25):
                generic_tokens.add(t)

    prefixes: List[str] = []
    by_stem: Dict[str, List[str]] = {}
    matched = 0
    for name in sorted(generic_tokens):
        for st in suffix_stems:
            if name.endswith(st) and len(name) - len(st) >= min_prefix:
                pre = name[: len(name) - len(st)]
                if min_prefix <= len(pre) <= max_prefix:
                    prefixes.append(pre)
                    by_stem.setdefault(st, []).append(pre)
                matched += 1
                break

    brand_names: Set[str] = set()
    for entry in names_df.get("brand_name", []):
        toks = tokenise(entry)
        if len(toks) == 1 and _acceptable(toks[0], min_brand, max_brand):
            brand_names.add(toks[0])

    # Morphemes that signal a *class*. A new name reusing one of these reads to a
    # clinician as a member of that class, which is exactly the misidentification the
    # INN stem system exists to prevent. Built from two places: the stem table itself,
    # and the recurring leading fragments of real generic names (so `erythro`, `amoxi`
    # and `acyclo` are caught even though none of them is a registered stem).
    lead_counts: Counter = Counter()
    for name in generic_tokens:
        for n in (4, 5, 6):
            if len(name) > n + 2:
                lead_counts[name[:n]] += 1
    blocklist = {s for s in suffix_stems if len(s) >= 4}
    blocklist |= {frag for frag, c in lead_counts.items() if c >= 3 and len(frag) >= 5}

    return TrainingCorpus(
        prefixes=prefixes, prefixes_by_stem=by_stem,
        brand_names=sorted(brand_names), generic_tokens=sorted(generic_tokens),
        stem_index=stem_index, morpheme_blocklist=blocklist,
        stats={"generic_tokens": len(generic_tokens),
               "prefixes_extracted": len(prefixes),
               "unique_prefixes": len(set(prefixes)),
               "stems_matched": matched,
               "stem_coverage": round(matched / max(1, len(generic_tokens)), 3),
               "stems_available": len(suffix_stems),
               "brand_names": len(brand_names),
               "blocklisted_morphemes": len(blocklist)},
    )


def siblings_for_stem(corpus: TrainingCorpus, stem: str) -> List[str]:
    """Every real generic name already using this stem. The sibling list seeds proposal
    intra-class distinctiveness term read from here."""
    bare = stem.strip().lstrip("-").lower()
    return [n for n in corpus.generic_tokens if n.endswith(bare)]
