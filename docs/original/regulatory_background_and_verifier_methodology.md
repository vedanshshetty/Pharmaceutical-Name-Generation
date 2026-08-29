# Regulatory Background and Verifier Methodology

*Person B sections. Written to drop into the report with light editing; every claim about
regulatory practice is sourced, and every number about our own system comes from the
executed verifier notebook.*

---

## 1. Scope boundary (put this in the introduction, not a footnote)

The verifier automates the **computationally checkable first pass** of pharmaceutical
name review. It does not replicate, replace, or approximate the whole of regulatory review.

What real review contains that we do not model:

- **Prescription-simulation and name-simulation studies.** FDA's guidance asks sponsors to
  test proposed names under conditions that mirror real practice, varying handwriting,
  background noise, verbal orders, and the prescribing/transcribing/dispensing/administering
  chain. That is human-subjects work, not string processing.
- **Full likelihood-of-confusion analysis.** Trademark law weighs goods and services class,
  channels of trade, commercial impression, and market context. Our V3 is a string-and-sound
  screen against a name corpus, which is a *proxy* for one input to that analysis.
- **Committee judgement.** The USAN Council is a five-member volunteer body that votes on
  names by ballot, weighing whether a name reflects drug action, how well it translates
  internationally, and how easily clinicians can say it. The WHO INN Expert Group then reviews
  for international conflicts and connotations in languages other than English.

Framed honestly, this is a **decision-support and candidate-screening tool**. That framing is
not a hedge: it makes the evaluation cleaner, because constraint satisfaction is falsifiable in
a way that "would a committee like this name" is not.

---

## 2. The two naming systems

A drug carries at least two names, produced by different bodies under different rules. The
distinction drives the entire architecture, because the two are almost opposite generation
problems.

### 2.1 Nonproprietary names — INN and USAN

The nonproprietary (generic) name is **not creative writing**. It is assembled from a formal,
published system of **stems**: a word ending (occasionally a prefix or infix) that signals
pharmacological class. Drugs in a class share a stem — *-olol* for beta-blockers, *-statin* for
HMG-CoA reductase inhibitors, *-mab* for monoclonal antibodies — combined with a "fantasy"
prefix that is meaningless by design and must distinguish the drug from its siblings.

The process, per the AMA's published procedure:

1. A firm files an application once the substance is in clinical trials, supplying an IND
   number, literature and chemical searches, evidence the proposed names are free of trademark
   and generic-name conflicts, and a **rationale for the requested stem**.
2. The **USAN Council** reviews and ballots, judging whether the name reflects drug action,
   translates internationally, and is easy to pronounce.
3. The **WHO INN Expert Group** then checks international conflicts and cross-language
   connotations.
4. Names **too similar to existing generic or trade names are rejected**.
5. Applications typically arrive during Phase 2; adoption is published roughly 60 days after
   approval.

The stem inventory itself is public: WHO maintains the INN stem book (*Use of stems in the
selection of INN for pharmaceutical substances*, 2024, plus periodic addenda), and the AMA
publishes the USAN stem list.

**What this gives the project:** a genuine formal grammar with unambiguous ground truth. That
is why the plan sequences generic naming first — the pass/fail signal is real, not a matter of
taste.

### 2.2 Proprietary names — the brand layer

Proprietary names are the commercially important, discretionary, expensive layer. A proprietary
name must simultaneously avoid look-alike/sound-alike confusion with every marketed name, avoid
carrying a USAN stem it is not entitled to, clear trademark, avoid implying claims, and still be
memorable and sayable.

FDA's *Best Practices in Developing Proprietary Names for Human Prescription Drug Products*
sets out the rules our V2, V3 and V5 encode:

- **Stems.** Sponsors should avoid proprietary names that incorporate a USAN stem **in the
  position USAN designates for it**, because doing so implies pharmacological properties the
  drug may not have. The guidance explicitly exempts **two-letter stems**, which "are often not
  distinct enough to be recognized as USAN stems." Sponsors are told to screen proposed names
  against the official USAN Council stem list.
- **Claims.** A name must not mislead about safety or efficacy. A fanciful name "may misbrand a
  product by suggesting that it has some unique effectiveness or composition when it is
  actually a common substance," and names suggesting a cure for a chronic condition are called
  out specifically. FDA assesses this using "common morphological and semantic associations"
  alongside phonetics.
- **Other disqualifiers.** Embedded medical abbreviations, collision with a foreign drug name
  carrying a *different* active ingredient, brand-name extension across different actives, and
  attribute-laden names that will date.
- **Process.** FDA makes a tentative acceptance decision early and finalises at approval, and
  the sponsor carries an ongoing post-market obligation.

---

## 3. POCA: what is public, and what our reimplementation is based on

### 3.1 What FDA itself discloses

FDA's Phonetic and Orthographic Computer Analysis (POCA) program page states that the tool
"uses an advanced algorithm to determine the orthographic and phonetic similarity between two
drug names," and that it screens a candidate against four regularly-updated sources:

| Source | Update cadence |
|---|---|
| Drugs@FDA | monthly |
| RxNorm | monthly |
| Suffixes in approved biological product names | monthly |
| United States Adopted Names | twice yearly |

A public web version is available at `poca-public.fda.gov`, and a 2009 Federal Register notice
announced availability of the software program for review of proprietary drug and biologic
names.

**Important for honesty in the write-up:** the program page does **not** publish the internal
formulas. Do not write "FDA states POCA uses BI-SIM and ALINE," because that page does not.

### 3.2 What the literature supplies

The algorithm family is identified in the academic literature, principally the peer-reviewed
work adapting POCA for national look-alike/sound-alike medicine-name screening outside the US
(Emmerton et al., *International Journal of Medical Informatics*, 2020), which independently
reimplemented the components — a Levenshtein-edit-distance-based orthographic measure, the
bigram measure **BI-SIM**, and the feature-based phonetic alignment algorithm **ALINE**. BI-SIM
and ALINE both originate with Kondrak: ALINE from his work on phonetic alignment, and BI-SIM
from Kondrak & Dorr's work on identifying confusable drug names.

That prior reimplementation is our **methodological precedent**, and it is what makes the claim
"the verifier is a faithful reimplementation of a real regulatory algorithm" defensible rather
than aspirational. Cite it in related work.

### 3.3 Thresholds — confirmed from primary sources

FDA's own published proprietary name reviews (the `NameR` documents in Drugs@FDA) state the
operating points explicitly:

| Band | Combined match percentage |
|---|---|
| Highly similar | **≥ 70%** |
| Moderately similar | **55% – 69%** |
| Low similarity | **≤ 54%** |

The reviews also record that the search itself is run **at a 55% threshold in POCA**, with
everything above that retrieved for human analysis. This matters for how our results should be
read: **55 is the screen, 70 is the high-risk flag.** They are two different operating points
serving two different purposes, and Section 6 shows our reimplementation behaves the same way.

---

## 4. What the verifier implements

Six checks. V1 is the POCA reimplementation; V2–V5 encode the further regulatory constraints
described above.

### V0 — well-formedness

Length band (4–20 characters), alphabet, normalisation to a comparison form (ASCII fold,
lowercase, letters only). Cheap, but it stops malformed generator output from silently
polluting downstream scores.

### V1 — similarity (the POCA reimplementation)

**Orthographic component.** Two measures, averaged:

- *Normalised Levenshtein.* Classic Wagner–Fischer edit distance with unit-cost insert, delete
  and substitute, rescaled by the longer string:
  `LED_sim(a,b) = 1 − d(a,b) / max(|a|,|b|)`.
- *BI-SIM.* Bigram-level similarity computed by dynamic programming. Rather than treating
  bigrams as a *set* — which discards order — it aligns the two bigram sequences with an
  LCS-style recurrence in which a bigram pair may match **partially**, scoring the fraction of
  identical corresponding letters (0, ½, or 1):

  `S(i,j) = max{ S(i−1,j−1) + s(A_i, B_j),  S(i−1,j),  S(i,j−1) }`
  `s((a₁,a₂),(b₁,b₂)) = ( [a₁=b₁] + [a₂=b₂] ) / 2`
  `BI-SIM(A,B) = S(n−1, m−1) / max(n−1, m−1)`

  That partial-credit alignment is why BI-SIM catches transposition-style confusables that raw
  edit distance under-penalises.

**Phonetic component.** Two stages.

*Grapheme-to-phoneme.* POCA compares phonemic representations, and a dictionary lookup is
useless here because every candidate is by construction a non-word. Our transcriber is
therefore rule-based and deterministic, encoding (a) standard English grapheme rules, (b) the
Greek/Latinate readings that dominate pharmaceutical orthography — `ch` = /k/ before a liquid,
`ph` = /f/, initial `ps-`/`pn-`/`gn-` reduction — and (c) a table of conventional readings for
the productive pharmaceutical endings (`-azine`, `-prazole`, `-cillin`, `-olol`, …).

The methodological argument for accepting an imperfect transcriber, which belongs in the
methods section rather than a footnote: **the candidate and every corpus name pass through the
same transcriber, so systematic transcription bias largely cancels in the comparison.** What a
similarity screen requires is that names which sound alike receive similar phone strings — a
property robust to a consistent convention, and one that does not require the convention to be
phonetically perfect.

*ALINE.* Two phone strings are scored by the alignment that maximises total phonetic
similarity, where two individual phones are compared by a **salience-weighted difference of
their articulatory features** rather than a binary match. That is what lets it know /p/ and /b/
are near-identical while /p/ and /l/ are not — the exact distinction LASA screening turns on.

- Features and salience weights are Kondrak's published values: manner 50, place 40, voice 10,
  nasal 10, lateral 10, retroflex 10, syllabic 5, high 5, back 5, round 5, aspirated 5, long 1.
- Operation constants: `C_skip = −10`, `C_sub = 35`, `C_exp = 45`, `C_vwl = 10`.
- `σ_sub(p,q) = C_sub − δ(p,q) − V(p) − V(q)`, where `δ` is the salience-weighted feature
  distance and `V` applies the vowel down-weight. Expansion/compression lets one phone align
  against two.
- Alignment runs in **local** mode (as published) or **global** mode. We report both rather
  than asserting one; see Section 6.
- Normalisation: `2·s(x,y) / (s(x,x) + s(y,y))`, which is 1.0 exactly when the phone strings
  are identical.

**Composite.** `composite = 100 · (w_o · orthographic + w_p · phonetic)`, with `w_o = w_p = 0.5`
by default, reproducing POCA's equal weighting. Every weight is a constructor parameter, so a
weighted variant can be run without touching algorithm code.

### V2 — USAN/INN stem grammar

- *Generic candidates* must end in the required class stem. Longer stems that end in the
  expected one are treated as legal specialisations (`-zumab` satisfies a `-mab` target,
  because a humanized monoclonal antibody is a monoclonal antibody). A different class's stem
  raises `STEM_MISMATCH`. A fantasy prefix shorter than two characters raises
  `STEM_PREFIX_TOO_SHORT`.
- *Intra-class distinctiveness* is checked separately: a generic candidate scoring ≥ 75 against
  a sibling that shares its stem raises `INTRA_STEM_TOO_CLOSE`, which is the computational form
  of the USAN requirement that the name be distinguishable from others in its class.
- *Brand candidates* carrying a real stem **in stem position** are rejected outright
  (`STEM_MISUSE_IN_BRAND`); a stem appearing elsewhere in the name is a warning, not a
  rejection. This mirrors the guidance's positional rule, and its exemption for very short
  stems is respected by the minimum-length filter on embedded matches.

### V3 — trademark collision

**Default path is offline** — a corpus of marketed brand names plus a curated Nice Class 5
sample, screened with the same composite scorer at a 70 cutoff. This is deliberate: it needs no
API key, has no rate limit, and makes the reported results reproducible indefinitely.

An optional live cell demonstrates a real lookup (openFDA marketed brand names, free and
keyless; USPTO's Open Data Portal, which needs a free key and whose endpoints have moved more
than once). Live lookups are demonstrations, not part of a results run — a run that depends on
a third-party endpoint's availability is not reproducible.

**Say plainly in the report:** this is a screening proxy, not legal clearance.

### V4 — pronounceability

Two components, weighted 0.6 / 0.4:

- *Phonotactic well-formedness.* Syllabification by the maximal-onset principle, then penalties
  for illegal onset clusters, over-long codas, consonant runs beyond three, vowel-hiatus chains,
  and monosyllabic or excessively polysyllabic shapes.
- *Corpus typicality.* An add-k smoothed character trigram model, reported as an **empirical
  percentile** against the corpus, so 0.5 means "as typical as the median marketed name."

One implementation detail that mattered more than expected: generic and proprietary names
occupy different orthographic distributions (*metoprolol* versus *Xanax*). Scoring a brand
candidate against a generic reference makes every plausible brand name look bizarre — *Zyprexa*
scored in the bottom 0.1% against the generic corpus and the 51st percentile against the brand
corpus. The verifier therefore keeps **one model per register** and selects by `target_type`.

### V5 — cross-lingual meaning and implied claims

- *Cross-lingual.* A curated lexicon of adverse-meaning terms (death, poison, pain, illness,
  blood, and a restrained profanity component) across eight market languages — Spanish, French,
  German, Italian, Portuguese, Japanese, Chinese, Hindi — matched both as substrings and
  phonetically via our own ALINE at a 0.88 cutoff.
- *Implied claims.* Terms that imply efficacy, superiority, potency, speed of onset, or
  permanence, which the FDA guidance treats as misbranding risk. Default severity is a
  **warning**, configurable to a hard failure.

**Coverage is partial by construction.** This is a curated lexicon over eight languages, not a
translation engine. State that limit; do not claim solved cross-cultural screening.

---

## 5. Corpus construction, and why it lives on the verifier's side

The shared data layer pulls names from openFDA's NDC directory, which mixes true drug names
with OTC **product descriptions**: `target up and up morning facial moisturizing with spf`,
`meijer lidocaine pain relief patch assortment`, `456 acne relief cream`. Left in, these
dominate the nearest-match field and distort every threshold calibrated against them.

The filtering rule, stated so it can be defended in one sentence: **a corpus entry must be a
single alphabetic token of 4–25 characters that is not a dosage form, a salt or counter-ion, or
a marketing modifier.** Multi-ingredient generic names are additionally split, because each
component of *acetaminophen dextromethorphan guaifenesin* is itself a real active-ingredient
name and belongs in the comparison universe.

| | Raw | After filtering |
|---|---|---|
| Unique generic names | 1,518 | 1,918 (combinations split into components) |
| Unique brand names | 1,699 | 408 (multi-word product descriptions removed) |
| **Screening universe** | — | **2,075 unique names** |
| Trademark screening corpus | — | 596 marks |

This lives in `verifier.py`, **not** in the shared `data_layer.py`. That module is jointly
owned and Person A has already built against it; changing it unilaterally is precisely the
silent-divergence failure the collaboration plan warns about.

**Residual noise, stated honestly:** homeopathic and botanical entries (Latin plant names)
survive the filter, because they genuinely are single-token marketed names. POCA screens
against everything marketed too, so this is arguably correct rather than a defect — but it
should be acknowledged rather than hidden.

---

## 6. Validation methodology and results

The verifier is validated **independently of the generator**, which is the "definition of done"
the collaboration plan assigns to Person B.

### 6.1 Candidate-level: does each defect trip the right code?

Twenty engineered candidates, each violating one specific rule, plus clean controls that must
clear every check. **20/20 behaved as specified.**

### 6.2 Pair-level: is the scoring function actually discriminative?

Treating the composite score as a binary classifier of "would a pharmacist confuse these?":

- **Positives (68 pairs)** — name pairs documented as confused in practice, drawn from
  published confused-drug-name lists and the LASA literature.
- **Negatives (438 pairs)** — 40 deliberately adversarial curated pairs (same class, same stem,
  or same initial letter, so the task is not trivially separable) plus 400 seeded random corpus
  pairs.

| Configuration | AUC | mean (confusable) | mean (distinct) | P @70 | R @70 | F1 @70 | best F1 |
|---|---|---|---|---|---|---|---|
| **ALINE (local)** | 0.9667 | 62.6 | 27.4 | 0.767 | 0.338 | 0.469 | 0.771 @ 48 |
| **ALINE (global)** | **0.9672** | 61.0 | 20.5 | 0.793 | 0.338 | 0.474 | 0.772 @ 48 |
| Metaphone | 0.9608 | 59.6 | 19.5 | 0.900 | 0.265 | 0.409 | 0.794 @ 49 |
| Orthographic only | 0.9647 | 59.3 | 20.2 | 0.875 | 0.309 | 0.457 | 0.785 @ 41 |
| Phonetic only (ALINE) | 0.9511 | 65.8 | 34.5 | 0.732 | 0.441 | 0.551 | 0.718 @ 54 |

### 6.3 Three findings worth arguing in the paper

**(a) The reimplementation reproduces POCA's operational logic, not just its arithmetic.** Read
the threshold sweep at the two published operating points:

| Cutoff | Precision | Recall | F1 | Specificity |
|---|---|---|---|---|
| 55 (POCA's retrieval threshold) | 0.746 | 0.735 | **0.741** | 0.961 |
| 70 (the "highly similar" flag) | 0.767 | 0.338 | 0.469 | **0.984** |

That is exactly the division of labour FDA's reviews describe: 55 is a **high-recall screen**
that surfaces candidates for human analysis, and 70 is a **high-specificity flag** for
candidates at genuine LASA risk. Our reimplementation lands on the same trade-off without being
tuned to. Report the published cutoffs as the primary operating points, with the swept optimum
(F1 0.771 at 48) alongside. **Do not quietly retune the threshold to flatter the result.**

**(b) There is a real ceiling, and it is not our bug.** Recall at 70 is 0.34. That is not a
defect in the reimplementation — it reflects the fact that names get confused for reasons an
orthographic/phonetic screen cannot see: packaging and label design, adjacent shelf position,
overlapping indications, similar dosing. This is the honest limit of computational screening
and is exactly why FDA pairs POCA with prescription-simulation studies. Saying so strengthens
the paper.

**(c) The feature-based phonetic model earns its complexity, modestly.** ALINE beats Metaphone
on AUC (0.967 vs 0.961) and materially on recall at the published cutoff (0.338 vs 0.265).
Metaphone buys precision (0.900) by being conservative — it collapses names to a coarse
consonant skeleton, so it only fires when they are very close. Metaphone also mis-reads
pharmaceutical orthography in a systematic way (`ch` in *chlorpromazine* as /tʃ/ rather than
/k/), which is itself a small illustration of why a domain-tuned G2P plus a feature-based
alignment is the right pairing here.

### 6.4 The stem-inflation problem

Worth a paragraph of its own, because it is a genuine design tension the architecture has to
answer rather than a bug.

Generic names are **required** to carry their class stem. Every `-umab` name therefore shares
four letters with every other `-umab` name before the generator has made a single choice. Plain
POCA scoring reads that mandated overlap as similarity and rejects well-formed antibody names:

| Candidate | Stem | Plain POCA | Verdict | Stem-aware | Verdict |
|---|---|---|---|---|---|
| nadrelumab | -umab | 73.2 | REJECT | 53.1 | pass |
| velituzumab | -zumab | 77.2 | REJECT | 60.9 | REJECT |
| narvasartan | -sartan | 69.7 | pass | 57.6 | pass |
| drovastatin | -statin | 85.4 | REJECT | 66.5 | REJECT |

The `stem_aware_similarity` option discounts the required stem from both strings before scoring,
so the screen measures the distinctiveness of the **fantasy prefix** — the part the candidate
actually controls, and the part USAN actually judges. Note that it is not a blanket loosening:
*velituzumab* and *drovastatin* are still rejected, because their prefixes really are too close
to *polatuzumab* and *pravastatin*.

The report must state which mode the headline numbers use, and why. Plain POCA is the faithful
baseline; stem-aware is the arguably-more-correct screen for stem-governed generic names.

### 6.5 Throughput

75 ms per candidate against a 2,075-name universe on a Colab CPU, with three-stage blocking
(bigram prefilter → orthographic scoring → ALINE on the top 150). Both pool sizes are
configurable and can be set to zero to reproduce exhaustive comparison.

---

## 7. Threats to validity

State these; a grader reads an unqualified results section as naïveté.

1. **Ground-truth labels are partly judgement.** Confused-drug-name lists are revised, and some
   pairs are arguably confusable but undocumented. Two pairs moved columns during development
   for exactly this reason. Verify every pair against the current ISMP list before citing it.
2. **Negative sampling inflates AUC.** 400 of 438 negatives are random corpus pairs, which are
   trivially separable. The curated adversarial 40 are the hard part of the negative set; AUC
   would fall on an all-hard-negative set.
3. **The corpus is a ~2,000-name snapshot** of openFDA, not the full universe POCA screens
   (Drugs@FDA plus RxNorm plus biologic suffixes plus USAN). Absolute scores are comparable
   within our experiments, not directly against published POCA output.
4. **G2P is rule-based, not learned or validated against a pronunciation dictionary.** The
   cancellation argument in Section 4 is the defence, and it is an argument, not a measurement.
5. **The trademark corpus is a proxy.** It is not the register.
6. **Cross-lingual coverage is a curated eight-language lexicon.**
7. **Blocking is an approximation.** Exhaustive comparison is available but the reported runs
   use pooled scoring; the pools are generous enough that the top-k is unchanged in practice,
   which is an empirical claim we assert rather than prove.

---

## 8. Citation list for these sections

**Primary regulatory sources**

- U.S. Food and Drug Administration. *Phonetic and Orthographic Computer Analysis (POCA)
  Program.* https://www.fda.gov/drugs/resources-drugs/phonetic-and-orthographic-computer-analysis-poca-program
- U.S. Food and Drug Administration. *POCA public tool.* https://poca-public.fda.gov/
- Federal Register, 17 Feb 2009. *Phonetic Orthographic Computer Analysis Software Program for
  Review of Proprietary Drug and Biologic Names; Availability.*
  https://www.federalregister.gov/documents/2009/02/17/E9-3170/
- FDA. *Best Practices in Developing Proprietary Names for Human Prescription Drug Products —
  Guidance for Industry.* https://www.fda.gov/media/88496/download
- FDA. *Best Practices in Developing Proprietary Names for Human Nonprescription Drug Products
  (draft).* https://www.fda.gov/media/144257/download
- FDA Drugs@FDA proprietary name reviews (`NameR` documents) — the primary source for the
  70% / 55% / 54% bands and for the 55% POCA retrieval threshold. Example:
  https://www.accessdata.fda.gov/drugsatfda_docs/nda/2024/215430Orig1s000NameR.pdf
- World Health Organization. *Use of stems in the selection of International Nonproprietary
  Names (INN) for pharmaceutical substances, 2024.*
  https://www.who.int/publications/i/item/9789240099388
- American Medical Association. *Procedure for USAN name selection.*
  https://www.ama-assn.org/about/united-states-adopted-names-usan/procedure-usan-name-selection
- American Medical Association. *Protection of USAN & INN stems.*
  https://www.ama-assn.org/about/united-states-adopted-names-usan/protection-usan-inn-stems

**Algorithmic sources**

- Kondrak, G. *A New Algorithm for the Alignment of Phonetic Sequences* (ALINE), NAACL 2000.
- Kondrak, G. & Dorr, B. *Identification of confusable drug names: a new approach and
  evaluation methodology* (BI-SIM), COLING 2004; and *Automatic identification of confusable
  drug names*, Artificial Intelligence in Medicine, 2006.
- Emmerton, L. et al. *Development and exploratory analysis of software to detect look-alike,
  sound-alike medicine names*, International Journal of Medical Informatics, 2020.
  https://www.sciencedirect.com/science/article/abs/pii/S1386505619312651
  — the methodological precedent for reimplementing the POCA component algorithms.
- Wagner, R. & Fischer, M. *The String-to-String Correction Problem*, JACM 1974.
- Philips, L. *Hanging on the Metaphone*, Computer Language, 1990.

**Note on citation hygiene:** the FDA program page describes POCA's *function*, not its
formulas. Attribute the specific algorithms (Levenshtein / BI-SIM / ALINE) to the academic
literature that identified and reimplemented them, not to FDA. Getting this right costs one
sentence and is exactly the kind of thing a careful reader checks.
