# Generative-Verifier Architecture for Regulation-Compliant Pharmaceutical Name Generation

**Regulation-aware generation and screening of pharmaceutical names.**

This system proposes candidate drug names and screens them against the constraints that
actually govern pharmaceutical nomenclature: INN/USAN stem conformance, look-alike and
sound-alike collision with marketed products, trademark conflict, phonotactic
well-formedness, and adverse cross-lingual meaning. It runs two separate pipelines,
because generic and proprietary names have opposed objectives.

Everything is reproducible: content-addressed artifacts, seeded runs, and a manifest on
every result recording the corpus fingerprint, the data mode, and the git SHA. There is
no metered proposer and nothing in a run depends on a remote service, an API key, or a
rate limit.

**The idea in one paragraph.** Generate many, discard nothing unseen, screen everything, then
rank the survivors. Every request runs two proposers in parallel — a syllable grammar induced
from the corpus, and an order-3 n-gram model that each rejected candidate re-guides — and pools
their output. The verifier then applies five separate regulatory checks to *every* member of the
pool (INN/USAN stem conformance, orthographic and phonetic look-alike screening against the
assembled regulator corpus, trademark collision, phonotactic well-formedness, adverse
cross-lingual meaning) and admits a name only if it clears all five. Quality is a separate,
downstream objective with distinct generic and brand most-similarity profiles; it ranks what
survived and never rescues a rejection. Because verification and ranking are separated, the
regulatory claim is testable on its own, and the results below — discrimination, per-class
acceptance, proposer contribution — are its evidence.

---

## Quickstart

```bash
git clone https://github.com/vedanshshetty/Pharmaceutical-Name-Generation.git
cd Pharmaceutical-Name-Generation
git checkout main
cd Pharmaceutical-Name-Generation
pip install -r requirements.txt
```

**Notebook (primary interface):**

```bash
jupyter lab main.ipynb
```

**Web UI (secondary):**

```bash
streamlit run app/streamlit_app.py
```

**Library:**

```python
from pharma_name_gen import build_system
from pharma_name_gen.verifier import VerifierConfig

system = build_system(live=True, verifier_config=VerifierConfig(stem_aware_similarity=True))
report = system.generic.generate(n_shortlist=10,
                                 target_class="beta-blocker",
                                 target_stem="-olol")

for c in report.shortlist:
    print(f"{c.name:14s} quality={c.quality.total:5.1f}  band={c.response.risk_band.value}")

report.to_frame().to_csv("all_attempts.csv", index=False)   # every candidate, not just winners
```

Runs fully offline with no API keys. `PHARMA_NAME_GEN_OFFLINE=1` forces the committed snapshot.

---

## Architecture

```
      request(target_type, class, stem, N)
                    |
        +-----------+-----------+
        |                       |            <- run in PARALLEL, every request
   GrammarProposer         NGramProposer          (nothing is discarded unseen)
   (induced syllable        (order-3 chars,
    grammar)                guided by this
        |                    run's rejections)
        +-----------+-----------+
                    |
              CANDIDATE POOL
                    |
              Verifier.verify()   <- the WHOLE pool, not the first success
                    |
        +-----------+-----------+
        |                       |
   rejected                 admissible
        |                       |
  structured feedback           |          <- payloads, not prose
  -> bigram penalties           |
  -> temperature ramp           |
  -> refine top-k               |
        +-----------+-----------+
                    |
            QualityScorer.score()
                    |
            SELECT top N by quality
                    |
               shortlist
```

| Module | Responsibility |
|---|---|
| `pharma_name_gen/contracts.py` | Pydantic API boundary between generation and screening (schema 1.1.0) |
| `pharma_name_gen/data_layer.py` | Live-first, static-fallback reference data with provenance |
| `pharma_name_gen/corpus.py` | Screening universe and training corpus, filtered differently on purpose |
| `pharma_name_gen/phonotactics.py` | Syllable grammar induced from the corpus |
| `pharma_name_gen/verifier.py` | The five regulatory checks and the composite risk score |
| `pharma_name_gen/quality.py` | The NameQuality objective, separate generic and brand profiles |
| `pharma_name_gen/orchestrator.py` | Pool-and-select pipeline |
| `pharma_name_gen/artifacts.py` | Content-addressed persistence for corpora and trained models |
| `pharma_name_gen/system.py` | Single builder used by every entry point |
| `pharma_name_gen/sweep.py` | Multi-class harness logging every attempt |
| `pharma_name_gen/evaluation.py` | Discrimination, ablations, architecture comparison |
| `pharma_name_gen/introspect.py` | `model.summary()`-style reporting for the notebook |

---

## Data sources

All primary-source, openly licensed, citable. No aggregator sites.

| Source | Publisher | Contributes |
|---|---|---|
| openFDA NDC directory | US FDA | Marketed US generic and proprietary names |
| RxNorm (IN + BN) | US NLM | Normalised ingredient and brand vocabulary |
| Medicines register | EMA (EU) | Centrally authorised EU products |
| INN/USAN stem table | WHO INN / USAN, curated | 277 stems with meanings and positions |

Live is the primary path, the committed snapshot is the fallback, and
`DataSnapshot.mode` always reports which one ran. That tag is load-bearing: silently
degrading from a live multi-regulator universe to the small committed snapshot would
quietly inflate every distinctiveness margin in the run.

---

## Design decisions

Fourteen are documented in full — decision, rejected alternative, reasoning, evidence — in
`pharma_name_gen/introspect.py::DESIGN_DECISIONS`, and rendered in section 3 of the notebook.
The ones that shaped the architecture most:

**Pool-and-select, not a cascade.** An early-exit cascade optimises for cost, not
quality: if the first proposer clears the bar with margin 1.04 it stops, and never learns
that another proposer had margin 15 on the same call. Pooling verifies everything before
choosing, so a shortlist can never be a local optimum of whichever proposer ran first.

**`rl_refined` is not a strategy.** It used the same n-gram model as `rejection_sampling`
plus a bigram penalty and a temperature ramp. On a single fresh call with no prior
rejections it is byte-identical to plain sampling. It is the batch-drawing policy for the
statistical proposer, not a peer mechanism, and it no longer appears as a
`generation_strategy` value.

**The grammar is induced, not written.** A hand-specified inventory encoded English
orthography and produced `skemkultolol`. Induced from real fantasy prefixes it produces
`lelma`, `veldo`, `monite`. Recombining at the syllable level also makes whole-morpheme
copying structurally impossible rather than merely penalised.

**Quality is orthogonal to and downstream of the verifier.** The verifier's claim is
regulatory: it answers whether a name *may* exist. Quality answers whether anyone would
*want* it. Quality ranks what survives and never rescues a rejection, because folding
preference into the regulatory decision would make the regulatory claim indefensible.

---

## Results

Every number is regenerable in one command (`python paper/reproduce.py`, or per-seed
then `--aggregate` for the sweep) and lands in `paper/results/` beside the manuscript.

**Verifier discrimination** against FDA/ISMP documented confusable pairs:

| Metric | Value |
|---|---|
| ROC AUC | 0.997 |
| Mean score, confusable pairs | 64.0 |
| Mean score, random pairs | 24.0 |
| Separation | 39.9 |

**Brand vs generic across seeds {1,2,3}** — 20 default targets, 10 shortlist each, per
run. Brand marks are out of the stem regime entirely, which is the point of having two
pipelines: brand acceptance is more than double generic, generic best-quality is higher
(stem conformance pays for distinctiveness in a way brand never does).

| Mode | Accept rate (mean±std) | Mean quality | Best quality |
|---|---|---|---|
| Generic | 0.241 ± 0.003 | 62.3 ± 0.2 | 75.4 ± 0.2 |
| Brand | 0.501 ± 0.051 | 60.7 ± 0.4 | 68.2 ± 1.0 |

**Acceptance by class difficulty** (merged three seeds) — the stratification predicts,
which is why it is reported rather than assumed:

| Difficulty | Candidates | Accept rate | Mean quality |
|---|---|---|---|
| brand | 1152 | 50.0% | 60.7 |
| roomy (`-pril`, `-sartan`, `-vaptan`, `-dipine`) | 1356 | 30.8% | 63.7 |
| crowded (`-olol`, `-prazole`, `-caine`, `-statin`, `-cycline`) | 2972 | 22.5% | 62.4 |
| saturated (`-tinib`, `-mab`, `-gliptin`, `-gliflozin`, `-parib`) | 2601 | 22.5% | 61.5 |

**Proposer contribution** — the comparison v1's `compare_strategies` was reaching for,
now conducted over logged pool data rather than forced on the user as a mode switch.
The two proposers have near-identical accept rates but very different shortlist shares
(grammar 69%, guided n-gram 31%), and the n-gram produces the highest-quality candidate.
That split is the empirical case for pooling instead of picking one.

**Two findings worth reporting as results, not footnotes:**

1. At the configured reject cutoff of 70, specificity against random pairs is already
   1.00 while sensitivity to documented confusable pairs is much lower than at the 55
   review line. The 55-70 band is doing real safety work, which is the direct
   justification for reporting it explicitly rather than passing it silently.

2. In stem-governed classes, almost every plausible novel name lands in the review band,
   because the mandated stem itself forces high orthographic similarity to siblings. This
   is a property of the problem rather than a defect of the screen; stem-aware similarity
   scoring is how it is handled.

---

## Paper

`paper/manuscript.pdf` is the full paper — IEEE conference format, nine pages, no external
services: **"Generative–Verifier Architecture for Regulation-Compliant Pharmaceutical Name
Generation."** It states the five checks, the two opposed pipelines, and the three-seed evidence
of Section V: discrimination, per-class acceptance, the proposer-contribution split, and the
quality-component ablation. Everything the paper reports is regenerable from committed artifacts:

```bash
python paper/reproduce.py             # full pipeline; or `python paper/reproduce.py SEED` per seed
python paper/reproduce.py --aggregate
python paper/make_figures.py          # paper/figures/fig_corpus.png, fig_strata.png
cd paper && tectonic manuscript.tex   # rebuild manuscript.pdf
```

---

## Reproducibility

- Every run emits a manifest: corpus fingerprint, data mode, source records, all
  thresholds and weights, git SHA.
- Artifacts are content-addressed on the corpus fingerprint plus config hash, so
  changing the stem table produces a different key and stale models are not found.
- Seeded runs are byte-identical; there is a test asserting it.
- CI runs the suite on Python 3.10, 3.11 and 3.12 with `PHARMA_NAME_GEN_OFFLINE=1`, so a red
  build always means the code broke, never that a regulator's API had a bad afternoon.

```bash
pytest                                          # 35 tests
python scripts/fetch_reference_data.py          # refresh the committed snapshot
python scripts/train_artifacts.py --live        # rebuild and stage artifacts
```

---

## Limitations

1. **This is a first-pass screen.** No human-factors prescription simulation, no legal
   likelihood-of-confusion analysis, no committee judgement. A `low` band means *worth
   taking to review*, never *cleared*.
2. **The universe bounds the claim.** Distinctiveness is measured against the assembled
   corpus; anything absent cannot be collided with, so margins are upper bounds.
3. **Phonetics are orthographic and Anglocentric.** The grapheme-to-phoneme converter is
   rule-based and English-oriented; cross-lingual checking is a lexicon lookup, not a
   phonological model of each language.
4. **The trademark check is a proxy** for marketed-name collision, not a registered
   trademark class search, which is a legal instrument.

## Licence

MIT. See `LICENSE`.