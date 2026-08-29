# Reference data provenance

A screening result is only as defensible as the universe it was screened against. This
file records where every corpus came from, how it is filtered, and what its known limits
are. Any figure quoted from a run of this system should be readable alongside this document.

## Sources

| File | Source | Publisher | Licence | Notes |
|---|---|---|---|---|
| `existing_drug_names.csv` | openFDA NDC directory | US FDA | Public domain (US Gov) | Committed snapshot, 2,168 rows. Refresh with `scripts/fetch_reference_data.py`. |
| `inn_usan_stems.csv` | INN stem book + USAN stem list | WHO / USAN Council | Curated transcription | 277 stems with meaning and position. |
| `usan_stems_seed.csv` | Original project seed table | — | — | 34 stems. Retained only as a last-resort fallback and for the v1/v2 comparison. |

Live sources reached at run time when `live=True`:

- **openFDA NDC** — `https://api.fda.gov/drug/ndc.json`. Unauthenticated `skip` is
  capped at 25,000, so the default limit stays inside that ceiling; exceeding it 400s
  mid-pagination and leaves a half-built corpus behind.
- **RxNorm IN + BN** — `https://rxnav.nlm.nih.gov/REST/allconcepts.json`. Contributes the
  *normalised* ingredient vocabulary that NDC buries inside product descriptions.
- **EMA medicines register** — the published `.xlsx` export. This is what makes the
  screen non-US: a name free in the US NDC universe can still collide with a centrally
  authorised EU product.

No aggregator or scraped sources are used at any point. A screening universe assembled
from a third-party site is not citable.

## Derived corpora

One snapshot, two corpora, filtered differently on purpose.

**Screening universe** (`build_screening_corpus`) — what the verifier measures distance
from. Recall-first. A junk token here costs a slightly conservative margin; a *missing*
real name costs a false clearance, which is the error that reaches a pharmacy shelf.

**Training corpus** (`build_training_corpus`) — what the statistical proposer imitates.
Precision-first. Every junk token here is something the model may learn to emit.
Multi-word entries are **split**, not dropped; dropping them was what held the v1
fantasy-prefix pool at 86 strings, small enough that an order-3 character model recited
rather than generalised.

## Filtering, and why each rule exists

| Rule | Removes | Rationale |
|---|---|---|
| Dosage-form words | tablet, capsule, solution … | Product descriptors, not names |
| Salt and moiety words | hydrochloride, sodium, mesylate … | Chemical modifiers shared across thousands of products |
| Marketing words | extra, strength, maximum … | Packaging copy |
| Common English words | yellow, hand, sanitizer … | Not distinctive marks. A screen reporting `yellow` as a nearest neighbour is reporting noise, and a reviewer who sees that once stops trusting the column. |
| Homeopathic Latin | sativus, latifolia, cysteinum … | Ingredient nomenclature from homeopathic combination products, not marketed names |
| Latin endings, in Latin context only | `-icum`, `-osum`, `-aris` … | Applied **only** to tokens from multi-ingredient homeopathic entries. A blanket rule would delete real INN names ending in `-a` or `-um`. |

The narrowness of the homeopathic test is deliberate: ordinary combination drugs are also
comma-separated (`acetaminophen, dextromethorphan hydrobromide, guaifenesin`) and must
stay in the corpus, so the test requires **both** a long ingredient list **and** at least
two recognised homeopathic Latin terms in it.

Filtering is verified in both directions. `tests/test_pharma_name_gen.py` asserts that the junk
tokens are gone **and** that `metoprolol`, `tylenol`, `advil` and `ibuprofen` survive —
a filter aggressive enough to delete what you screen against is worse than no filter.

## Known limitations

1. **The committed snapshot is small** (2,168 rows). It is a starting point, not the
   screening universe a real submission would use. Run
   `scripts/fetch_reference_data.py` for a full pull and commit the result with its
   `PROVENANCE.json`.
2. **Stem coverage is partial.** 277 stems is a large expansion on 34 but not the
   complete INN stem book, and coverage of the corpus sits near 19%. Coverage is
   reported in every manifest rather than being left implicit.
3. **`crayola` and `salmonella` remain in the corpus.** Both are legitimate: Crayola is a
   registered mark appearing on an FDA-listed product, and *Salmonella* names a vaccine
   antigen. They look odd as nearest neighbours but removing them would be filtering on
   aesthetics rather than on principle.
4. **No trademark register.** The trademark check is a marketed-name collision proxy.
   A genuine clearance search is a legal instrument, not a string comparison.
