# NOMINA

**Regulation-aware generation and screening of pharmaceutical names.**

NOMINA proposes candidate drug names and screens them against the constraints that
actually govern pharmaceutical nomenclature: INN/USAN stem conformance, look-alike and
sound-alike collision with marketed products, trademark conflict, phonotactic
well-formedness, and adverse cross-lingual meaning. It runs two separate pipelines,
because generic and proprietary names have opposed objectives.

Everything is reproducible: content-addressed artifacts, seeded runs, and a manifest on
every result recording the corpus fingerprint, the data mode, and the git SHA.

---

## Quickstart

```bash
git clone -b production https://github.com/vedanshshetty/Pharmaceutical-Name-Generation.git
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
from nomina import build_system
from nomina.verifier import VerifierConfig

system = build_system(live=True, verifier_config=VerifierConfig(stem_aware_similarity=True))
report = system.generic.generate(n_shortlist=10,
                                 target_class="beta-blocker",
                                 target_stem="-olol")

for c in report.shortlist:
    print(f"{c.name:14s} quality={c.quality.total:5.1f}  band={c.response.risk_band.value}")

report.to_frame().to_csv("all_attempts.csv", index=False)   # every candidate, not just winners
```

Runs fully offline with no API keys. `NOMINA_OFFLINE=1` forces the committed snapshot.

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
         best quality < threshold?  --yes--> LLMProposer (metered, escalation only)
                    |
            SELECT top N by quality
                    |
               shortlist
```

| Module | Responsibility |
|---|---|
| `nomina/contracts.py` | Pydantic API boundary between generation and screening (schema 1.1.0) |
| `nomina/data_layer.py` | Live-first, static-fallback reference data with provenance |
| `nomina/corpus.py` | Screening universe and training corpus, filtered differently on purpose |
| `nomina/phonotactics.py` | Syllable grammar induced from the corpus |
| `nomina/verifier.py` | The five regulatory checks and the composite risk score |
| `nomina/quality.py` | The NameQuality objective, separate generic and brand profiles |
| `nomina/orchestrator.py` | Pool-and-select pipeline |
| `nomina/llm.py` | OpenRouter free tier with runtime model resolution |
| `nomina/artifacts.py` | Content-addressed persistence for corpora and trained models |
| `nomina/system.py` | Single builder used by every entry point |
| `nomina/sweep.py` | Multi-class harness logging every attempt |
| `nomina/evaluation.py` | Discrimination, ablations, architecture comparison |
| `nomina/introspect.py` | `model.summary()`-style reporting for the notebook |

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

Sixteen are documented in full — decision, rejected alternative, reasoning, evidence — in
`nomina/introspect.py::DESIGN_DECISIONS`, and rendered in section 3 of the notebook. The
ones that shaped the architecture most:

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

**The LLM escalates, it does not join the free pool.** A unit mismatch: the CPU proposers
cost microseconds, an LLM call costs seconds and a rate-limit slot. It is invoked when
the free pool's best quality is thin, which is the one case where semantic reasoning is
worth paying for.

---

## Results

From `results/sweep_all_attempts.csv` — sixteen targets spanning the difficulty range,
one row per candidate ever evaluated.

**Verifier discrimination** against FDA/ISMP documented confusable pairs:

| Metric | Value |
|---|---|
| ROC AUC | 0.994 |
| Mean score, confusable pairs | 64.0 |
| Mean score, random pairs | 25.1 |
| Separation | 38.9 |

**Acceptance by class difficulty** — the stratification predicts, which is why it is
reported rather than assumed:

| Difficulty | Candidates | Accept rate | Mean quality |
|---|---|---|---|
| roomy (`-pril`, `-sartan`) | 322 | 41.3% | 65.0 |
| saturated (`-tinib`, `-mab`) | 631 | 33.9% | 62.2 |
| crowded (`-olol`, `-prazole`) | 647 | 28.9% | 63.5 |

**Proposer contribution** — the comparison v1's `compare_strategies` was reaching for,
now conducted over logged pool data rather than forced on the user as a mode switch.
The two proposers have near-identical accept rates but very different shortlist shares,
and the n-gram produces the single highest-quality candidate. That split is the
empirical case for pooling instead of picking one.

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

## Reproducibility

- Every run emits a manifest: corpus fingerprint, data mode, source records, all
  thresholds and weights, git SHA.
- Artifacts are content-addressed on the corpus fingerprint plus config hash, so
  changing the stem table produces a different key and stale models are not found.
- Seeded runs are byte-identical; there is a test asserting it.
- CI runs the suite on Python 3.10, 3.11 and 3.12 with `NOMINA_OFFLINE=1`, so a red
  build always means the code broke, never that a regulator's API had a bad afternoon.

```bash
pytest                                          # 36 tests
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
5. **Free-tier LLM availability rotates.** The model chain resolves at run time for
   exactly this reason, but the proposer can still return nothing, and the system is
   built to proceed without it.

## Licence

MIT. See `LICENSE`.
