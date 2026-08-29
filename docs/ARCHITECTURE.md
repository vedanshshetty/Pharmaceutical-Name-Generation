# Architecture notes

Companion to section 2 of `main.ipynb`. The notebook prints the *live* configuration;
this file explains the reasoning that does not change between runs.

## The contract

`pharma_name_gen/contracts.py` is the API boundary. Generation and screening were built by
different people and neither imports the other's internals; both validate against these
Pydantic models. Schema version 1.1.0.

The most important property is that **rejection feedback is structured data**. Each
`RefinementSignal` carries a machine-readable `payload` alongside its `human_readable`
string, and the generator reads only the payload. It never parses prose and never makes
a second model call to work out why something failed. `_learn_from_rejection` pulls
`nearest_match`, `sibling`, `fragment`, `conflicts` and `foreign_stems` straight out of
the payload and converts them into bigram sampling penalties.

## Why two corpora

See `data/PROVENANCE.md`. The short version: the screening universe is recall-first
because a missing name causes a false clearance, and the training corpus is
precision-first because a junk token becomes something the model emits. In v1 these were
built independently on each side of the project with different tokenisers and diverged
silently (1,918 names against 420) with nothing recording the difference.

## Why the grammar is induced

A hand-specified onset/nucleus/coda inventory encodes whichever language the author was
thinking in. v1's had nuclei `{ai, ea, ie, oa, ou, ei}` and codas `{nd, st, rt, lt, ns}`,
which is English, and it produced `skemkultolol` and `jeimheistolol` — legal strings, and
nothing like an INN name.

Inducing the inventory from real fantasy prefixes fixes the distributional problem, and
carries a second benefit that is arguably larger: because generation recombines at the
**syllable** level, reproducing a whole morpheme such as `erythro` would require `e`,
`ry` and `thro` to be independently resampled in that order. The memorisation failure
mode is excluded by the representation rather than caught by a penalty afterwards.

Onset legality is itself bootstrapped: the parser's set of legal onsets is the set of
consonant clusters actually observed word-initially in the corpus, so the syllabifier and
the generator agree by construction.

## Why quality and risk are separate

The verifier's claim is regulatory. It answers whether a name may exist, and every part
of that answer traces to a published rule or a documented convention. Quality answers
whether anyone would want the name, which is a preference judgement.

Folding the two together would make the regulatory claim a preference model, and a
reviewer would be right to reject it. So `QualityReport` is attached to the response and
is never consulted by `overall_pass`. Quality ranks what survives; it cannot rescue a
rejection and cannot cause one.

The one place quality enters generation is `cheap_reward`, a verifier-free approximation
used *inside* the sampling loop where a full 50 ms verify per draw would be prohibitive.
Every survivor is scored properly afterwards, so the approximation can waste a draw but
can never let a bad name onto a shortlist. A test asserts the two correlate.

## Why the review band is not a rejection band

Treating 55-70 as failure was implemented and reverted. It dropped the admissible rate to
0.5%, because in a stem-governed class the mandated stem forces high orthographic
similarity to every sibling, so almost everything plausible lands in that band.

The resolution is to report rather than to hide. `risk_band` is on every response, both
margins are carried, and the quality objective's distinctiveness term scores headroom
against the **55** line, so grey-band names rank below clear ones automatically. v1 both
accepted these silently *and* reported a margin measured against the wrong cutoff, so the
output read as comfortable when it was not.

`PipelineConfig.treat_moderate_as_failure` exposes the strict policy for anyone who wants
it, and the evaluation section reports both.

## Cost model

| Proposer | Unit cost | When it runs |
|---|---|---|
| Grammar | microseconds | Every request |
| N-gram | microseconds | Every request |

Both proposers are free CPU and run on every request. There is deliberately no metered
proposer: every mechanism in the pool is an interpreter-free, statistical or rule-based
component that runs offline and deterministically. Nothing in a run depends on a remote
service, a key, or a rate limit, so results are reproducible without a budget line.
