"""Builds main.ipynb. Kept as a script so the notebook is reviewable in diffs."""
import json
from pathlib import Path

C = []


def md(src): C.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)})
def code(src): C.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


# ---------------------------------------------------------------- 0. Title
md(r"""
<div style="background:linear-gradient(135deg,#0b3d5c 0%,#1a7a8c 100%);color:#fff;padding:28px 32px;border-radius:12px">
<h1 style="margin:0;font-size:2.1em;letter-spacing:.5px">Generative-Verifier Architecture</h1>
<p style="margin:6px 0 0;font-size:1.08em;opacity:.93">
Regulation-aware generation and screening of pharmaceutical names
</p>
<p style="margin:14px 0 0;font-size:.92em;opacity:.85">
Generic (INN/USAN) and proprietary (brand) pipelines &middot; live FDA / EMA / RxNorm reference data
&middot; parallel pool-and-select generation &middot; explicit quality objective
</p>
</div>

### What this notebook is

The single entry point. All logic lives in the `pharma_name_gen` package; this notebook
**orchestrates and explains**, it does not implement. Run it top to bottom.

| Section | What it covers |
|---|---|
| 1 | Environment and reference data (live, with fallback) |
| 2 | **Architecture** — printed from the live system, not from documentation |
| 3 | **Design decisions** and the reasoning behind each |
| 4 | Interactive generation (dropdowns and sliders, no code editing) |
| 5 | Results, with the full audit trail for any candidate |
| 6 | Evaluation — the numbers for the paper |
| 7 | Full multi-class sweep |
| 8 | Export and run manifest |

### Scope boundary

This screens the **computationally checkable first pass**. It is not FDA or USAN review,
which additionally involve prescription-simulation human-factors studies, full legal
likelihood-of-confusion analysis, and committee judgement. Nothing here should be read
as regulatory clearance.
""")

# ---------------------------------------------------------------- 1. Setup
md(r"""
---
## 1. Environment and reference data

Two things happen here. Dependencies are installed, and the reference corpus is
assembled from the primary regulators.

The data layer is **live-first with a static fallback**, and it always reports which one
ran. That matters more than it sounds: silently degrading from a live multi-regulator
universe to the small committed snapshot would quietly inflate every distinctiveness
margin in the run, and the results would look better precisely because the screen got
weaker.
""")

code(r"""
# Install. Safe to re-run; skipped when already satisfied.
import importlib, subprocess, sys

REQUIRED = ["pandas", "pydantic", "requests", "openpyxl", "ipywidgets", "matplotlib"]
missing = [p for p in REQUIRED if importlib.util.find_spec(p.split("[")[0]) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=False)
print("dependencies ready")
""")

code(r"""
# Get the code. In Colab this clones; locally it uses the checkout you are already in.
import os, subprocess, sys
from pathlib import Path

REPO = "https://github.com/vedanshshetty/Pharmaceutical-Name-Generation.git"
BRANCH = "production"          # pinned; never the default branch

if Path("pharma_name_gen").is_dir():
    ROOT = Path.cwd()
else:
    target = Path("Pharmaceutical-Name-Generation")
    if not target.exists():
        # -b is the fix for the v1 bug: cloning without it fetched the default branch,
        # which contains no verifier.py, so the next import raised ImportError on any
        # fresh runtime.
        subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, str(target)],
                       check=True)
    ROOT = target.resolve()

sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
print(f"working directory: {ROOT}")
""")

code(r"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 200)

from pharma_name_gen import build_system, PipelineConfig, TargetType
from pharma_name_gen.verifier import VerifierConfig
from pharma_name_gen import introspect as ins

# ---- knobs you may want to change before the first build --------------------
USE_LIVE_DATA = True      # False forces the committed snapshot (fully offline)
USE_ARTIFACTS = True      # reuse cached corpora and trained models where valid
# -----------------------------------------------------------------------------

system = build_system(
    live=USE_LIVE_DATA,
    use_artifacts=USE_ARTIFACTS,
    # stem_aware_similarity strips the MANDATED stem before scoring, so distinctiveness
    # measures the part the namer actually controls. Without it, every name in a
    # stem-governed class scores as similar to its siblings purely for complying with
    # the regulation that forced the stem on it.
    verifier_config=VerifierConfig(stem_aware_similarity=True),
    progress=print,
)
print()
print(system.summary())
""")

md(r"""
> **Reading the `[LIVE]` / `[STATIC]` tag above.** `LIVE` means at least one regulator
> feed was reached and merged. `STATIC` means all of them failed or were disabled and
> the run is using the committed snapshot only. The tag propagates into the run manifest
> in section 8, so any exported result carries the provenance of the universe it was
> screened against.
""")

# ---------------------------------------------------------------- 2. Architecture
md(r"""
---
## 2. Architecture

Everything below is **printed from the live system object**. If a number here is wrong,
it is wrong because the system is wired that way, not because documentation drifted.
""")

code(r"""
print(ins.architecture_summary(system))
""")

md(r"""
### 2.1 Request path

Read the diagram for the two claims it encodes. First, both free proposers run **in
parallel for every request** and the whole pool is verified before anything is selected,
so a shortlist can never be a local optimum of whichever proposer happened to go first.
Second, rejection feedback is **structured data**, not prose: the generator reads
`signal.payload`, so it never has to parse an error message or make a second model call
to work out why something failed.
""")

code(r"""
print(ins.data_flow())
""")

md(r"""
### 2.2 Reference data

The two derived corpora serve opposite purposes and are filtered differently on purpose.

* **Screening universe** — what the verifier measures distance from. Recall-first: a
  junk token here costs a slightly conservative margin, whereas a *missing* real name
  costs a false clearance, which is the error that reaches a pharmacy shelf.
* **Training corpus** — what the statistical proposer imitates. Precision-first: every
  junk token here is something the model might learn to emit.

In v1 these were built independently on each side of the project and diverged silently
(1,918 names against 420) with nothing recording the difference.
""")

code(r"""
display(ins.corpus_table(system))
print("\nSources:")
display(pd.DataFrame(system.snapshot.manifest()["sources"]))
""")

md(r"""
### 2.3 The screening checks

Five checks, each returning structured failure codes and machine-readable payloads.
""")

code(r"""
display(ins.threshold_table(system))
""")

md(r"""
### 2.4 The quality objective

The verifier answers *may this name exist*. This answers *is this a name anyone would
want*, which nothing in v1 measured. The two are deliberately kept orthogonal: quality
ranks what survives screening and never rescues a rejection, because folding preference
into the regulatory decision would make the regulatory claim indefensible.

Note the two weight columns. Generic and brand have **opposed** objectives — a generic
name must carry its class stem and be systematically unmemorable, a brand name must
carry no stem at all and be memorable — so they are separate pipelines with separate
weights rather than one pipeline with a branch.
""")

code(r"""
display(ins.quality_weights_table(system))
""")

md(r"""
### 2.5 The two mechanisms that carry the argument

Prose about a design is not evidence the design is real. These are the actual
implementations, displayed from source.

**The induced syllable grammar.** v1 sampled from a hand-written inventory that encoded
English orthography and produced `skemkultolol`. Inducing the inventory from real
fantasy prefixes fixes both the plausibility problem and, because it recombines at the
*syllable* level, makes whole-morpheme copying (`erythro` + `olol`) structurally
impossible rather than merely penalised.
""")

code(r"""
from pharma_name_gen.phonotactics import InducedGrammar
grammar = InducedGrammar.induce(system.training.prefixes)
print(grammar.summary())

import random
rng = random.Random(11)
print("\nSampled fantasy prefixes (novel, corpus-plausible, consonant-final):")
print(" ", ", ".join(grammar.sample(rng, 3, 8, final_coda=True) for _ in range(16)))
""")

md(r"""
**Learning from rejections.** This is what v1 called `rl_refined`. It is not a separate
strategy and it is not a trained policy: it is a batch-drawing discipline that reads the
verifier's structured payloads and turns them into sampling pressure.
""")

code(r"""
from pharma_name_gen.orchestrator import NominaPipeline
print(ins.source_code(NominaPipeline._learn_from_rejection))
""")

# ---------------------------------------------------------------- 3. Decisions
md(r"""
---
## 3. Design decisions

Every non-obvious choice, the alternative that was rejected, and why. Several of these
were reached by trying the alternative first and measuring it.
""")

code(r"""
decisions = ins.design_decisions()
for area in decisions["area"].unique():
    print(f"\n{'=' * 100}\n{area.upper()}\n{'=' * 100}")
    for _, r in decisions[decisions.area == area].iterrows():
        print(f"\n  DECISION    {r['decision']}")
        print(f"  INSTEAD OF  {r['alternative']}")
        print(f"  BECAUSE     {r['reasoning']}")
        print(f"  EVIDENCE    {r['evidence']}")
""")

# ---------------------------------------------------------------- 4. Interactive
md(r"""
---
## 4. Generate

Use the controls. No code editing required.
""")

code(r"""
import ipywidgets as W
from IPython.display import display, HTML, clear_output

stems_df = system.snapshot.stems
generic_options = sorted(
    [(f"{r.meaning}  ({r.stem})", r.stem) for r in stems_df.itertuples(index=False)
     if str(r.position) == "suffix"],
    key=lambda x: x[0])

w_type = W.ToggleButtons(options=[("Generic (INN/USAN)", "generic"),
                                  ("Brand (proprietary)", "brand")],
                         value="generic", description="Name type:",
                         style={"description_width": "120px"})
w_class = W.Combobox(placeholder="type to filter, e.g. kinase",
                     options=[o[0] for o in generic_options],
                     value="beta-blocker (beta-adrenoceptor antagonist)  (-olol)",
                     description="Class:", ensure_option=False, layout=W.Layout(width="640px"),
                     style={"description_width": "120px"})
w_brand_class = W.Text(value="cardiovascular, once-daily oral",
                       description="Context:", layout=W.Layout(width="640px"),
                       style={"description_width": "120px"})
w_n = W.IntSlider(value=10, min=3, max=25, step=1, description="Shortlist:",
                  continuous_update=False, style={"description_width": "120px"})
w_pool = W.IntSlider(value=24, min=8, max=64, step=4, description="Pool/proposer:",
                     continuous_update=False, style={"description_width": "120px"})
w_rounds = W.IntSlider(value=4, min=1, max=8, description="Max rounds:",
                       continuous_update=False, style={"description_width": "120px"})
w_strict = W.Checkbox(value=False, description="Strict: reject the 55-70 review band",
                      indent=False)
w_seed = W.IntText(value=20260826, description="Seed:", style={"description_width": "120px"})
w_go = W.Button(description="Generate", button_style="primary", icon="play",
                layout=W.Layout(width="180px", height="40px"))
out = W.Output()

def _toggle(change=None):
    is_generic = w_type.value == "generic"
    w_class.layout.display = "" if is_generic else "none"
    w_brand_class.layout.display = "none" if is_generic else ""
w_type.observe(_toggle, "value"); _toggle()

controls = W.VBox([
    W.HTML("<h3 style='margin-bottom:4px'>Target</h3>"), w_type, w_class, w_brand_class,
    W.HTML("<h3 style='margin-bottom:4px'>Search effort</h3>"), w_n, w_pool, w_rounds,
    W.HTML("<h3 style='margin-bottom:4px'>Policy</h3>"), w_strict, w_seed,
    W.HTML("<br>"), w_go,
])
display(controls, out)
""")

code(r"""
LAST_REPORT = None

def _band_pill(band):
    colour = {"low": "#1a7f4b", "moderate": "#b06e00", "high": "#a11"}.get(band, "#555")
    return f"<span style='background:{colour};color:#fff;padding:2px 9px;border-radius:10px;font-size:.82em'>{band}</span>"

def _quality_bar(q):
    pct = max(0, min(100, q))
    colour = "#1a7f4b" if q >= 68 else ("#b06e00" if q >= 58 else "#a11")
    return (f"<div style='background:#eee;border-radius:4px;width:120px;height:14px;display:inline-block;vertical-align:middle'>"
            f"<div style='background:{colour};width:{pct * 1.2:.0f}px;height:14px;border-radius:4px'></div></div>"
            f" <b>{q:.1f}</b>")

def render(report):
    s = report.stats
    head = (f"<div style='background:#f4f7f9;border-left:4px solid #1a7a8c;padding:12px 16px;border-radius:6px'>"
            f"<b>{s['returned']}</b> names returned from <b>{s['candidates_evaluated']}</b> evaluated "
            f"&middot; {s['admissible']} admissible ({s['band_low']} clear, {s['band_moderate']} review-band) "
            f"&middot; best quality <b>{s['best_quality']}</b> "
            f"&middot; {s['verifier_calls']} screens "
            f"&middot; {s['wall_seconds']}s</div>")
    rows = []
    for i, c in enumerate(report.shortlist, 1):
        q = {x.name: x.score for x in c.quality.components}
        sim = c.response.checks.similarity
        rows.append(
            f"<tr>"
            f"<td style='padding:7px 10px;color:#888'>{i}</td>"
            f"<td style='padding:7px 10px'><b style='font-size:1.15em;letter-spacing:.4px'>{c.name}</b></td>"
            f"<td style='padding:7px 10px'>{_quality_bar(c.quality.total)}</td>"
            f"<td style='padding:7px 10px'>{_band_pill(c.response.risk_band.value)}</td>"
            f"<td style='padding:7px 10px;text-align:right'>{c.risk:.1f}</td>"
            f"<td style='padding:7px 10px;text-align:right'>{sim.distinctiveness_margin_moderate:+.1f}</td>"
            f"<td style='padding:7px 10px;color:#666;font-size:.9em'>{sim.nearest_match}</td>"
            f"<td style='padding:7px 10px;font-size:.85em'>"
            f"nov {q.get('novelty', 0):.2f} &middot; say {q.get('pronounceability', 0):.2f} "
            f"&middot; shape {q.get('shape', 0):.2f}</td>"
            f"<td style='padding:7px 10px;color:#777;font-size:.85em'>{c.proposer}</td>"
            f"</tr>")
    table = (
        "<table style='border-collapse:collapse;width:100%;margin-top:14px;font-family:system-ui'>"
        "<thead><tr style='background:#0b3d5c;color:#fff;text-align:left'>"
        + "".join(f"<th style='padding:8px 10px'>{h}</th>" for h in
                  ["#", "Name", "Quality", "Band", "Risk", "Margin to review",
                   "Nearest existing", "Components", "Proposer"])
        + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")
    display(HTML(head + table))

def on_go(_):
    global LAST_REPORT
    with out:
        clear_output(wait=True)
        target_type = w_type.value
        if target_type == "generic":
            label = w_class.value
            stem = dict(generic_options).get(label, "-olol")
            target_class = label.split("  (")[0]
        else:
            stem, target_class = None, w_brand_class.value

        pipeline = system.pipeline(target_type)
        pipeline.config = PipelineConfig(
            pool_per_proposer=w_pool.value, max_rounds=w_rounds.value,
            shortlist_size=w_n.value,
            treat_moderate_as_failure=w_strict.value, seed=w_seed.value)
        for p in pipeline.proposers:
            if hasattr(p, "config"):
                p.config = pipeline.config

        print(f"generating {w_n.value} {target_type} names for '{target_class}'"
              + (f" with stem {stem}" if stem else "") + " ...")
        LAST_REPORT = pipeline.generate(n_shortlist=w_n.value, target_class=target_class,
                                        target_stem=stem)
        clear_output(wait=True)
        render(LAST_REPORT)

w_go.on_click(on_go)
print("Press Generate above.")
""")

# ---------------------------------------------------------------- 5. Audit
md(r"""
---
## 5. Audit trail

Any single candidate, fully decomposed. This is what makes the shortlist defensible:
every name can answer *why did you pick me* and *what did you check*.
""")

code(r"""
if LAST_REPORT is None:
    print("Run section 4 first.")
else:
    picker = W.Dropdown(options=[c.name for c in LAST_REPORT.shortlist],
                        description="Inspect:", style={"description_width": "80px"},
                        layout=W.Layout(width="380px"))
    detail = W.Output()

    def show(change=None):
        with detail:
            clear_output(wait=True)
            c = next(x for x in LAST_REPORT.shortlist if x.name == picker.value)
            r = c.response
            print(f"{c.name.upper()}\n{'=' * 62}")
            print(f"proposer          {c.proposer}   round {c.round}")
            if c.lineage and len(c.lineage) > 1:
                print(f"refinement path   {' -> '.join(c.lineage)}")
            print(f"admissible        {c.accepted}   band={r.risk_band.value}   "
                  f"risk={r.composite_risk_score:.1f}")
            print(f"\nQUALITY {c.quality.total:.1f}/100")
            for x in sorted(c.quality.components, key=lambda y: -y.weight):
                bar = "#" * int(x.score * 22)
                print(f"  {x.name:<18}{x.score:5.2f}  w={x.weight:.2f}  {bar}")
            print(f"\nSCREENING")
            for name, chk in [("similarity", r.checks.similarity),
                              ("stem", r.checks.stem_conflict),
                              ("trademark", r.checks.trademark_collision),
                              ("pronounceability", r.checks.pronounceability),
                              ("crosslingual", r.checks.cross_lingual)]:
                mark = "PASS" if getattr(chk, "passed", True) else "FAIL"
                print(f"  [{mark}] {name}")
            sim = r.checks.similarity
            print(f"\n  nearest existing name : {sim.nearest_match} ({sim.nearest_match_score:.1f})")
            print(f"  margin to review (55) : {sim.distinctiveness_margin_moderate:+.2f}")
            print(f"  margin to reject (70) : {sim.distinctiveness_margin:+.2f}")
            print(f"  phonemes              : {' '.join(r.checks.pronounceability.phonemes)}")
            print(f"  syllables             : {r.checks.pronounceability.syllables}")
            if r.warning_codes:
                print(f"\n  WARNINGS: {[w.value for w in r.warning_codes]}")

    picker.observe(show, "value"); show()
    display(picker, detail)
""")

code(r"""
# The full machine-readable contract for the top candidate. This object is the API
# boundary between the two halves of the system; both sides validate against it.
if LAST_REPORT:
    import json
    print(json.dumps(LAST_REPORT.shortlist[0].response.model_dump(mode="json"),
                     indent=2)[:2600] + "\n...")
""")

# ---------------------------------------------------------------- 6. Evaluation
md(r"""
---
## 6. Evaluation

### 6.1 Does the verifier discriminate?

Known-confusable name pairs (FDA Name Differentiation Project / ISMP confused drug names
— pairs with *documented dispensing errors*, not pairs that merely look similar) against
random pairs from the same corpus. If those two distributions overlapped, the composite
score would not be measuring confusability and no threshold could rescue it.
""")

code(r"""
from pharma_name_gen import evaluation as ev
verif = ev.evaluate_verifier(system, n_negative=400)
print(f"confusable pairs   n={verif['n_confusable_pairs']:<4} mean score {verif['mean_confusable_score']}")
print(f"random pairs       n={verif['n_random_pairs']:<4} mean score {verif['mean_random_score']}")
print(f"separation         {verif['separation']}")
print(f"ROC AUC            {verif['roc_auc']}")
print(f"\nconfigured cutoffs : review {verif['configured_moderate_cutoff']}, "
      f"reject {verif['configured_high_cutoff']}")
print(f"Youden-optimal     : {verif['optimal_threshold_youden']}")
display(verif["threshold_sweep"])
""")

md(r"""
> **This is a finding, not a formality.** The AUC is high, but read the sensitivity
> column at the configured cutoff. At **70**, the screen flags only a minority of pairs
> that have caused real dispensing errors, while specificity is already 1.00 — it is
> refusing nothing it should keep. Sensitivity rises substantially at **55** at no
> measurable cost in false positives.
>
> Two things follow. First, the 55-70 review band is doing real safety work and should
> not be treated as a formality, which is the direct justification for reporting it
> explicitly rather than passing it silently as v1 did. Second, the hard cutoff inherited
> from the POCA literature is conservative for *this* corpus, and a paper making claims
> about screening performance should say so rather than adopt the round number.
""")

code(r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
ax[0].hist(verif["negative_scores"], bins=30, alpha=.7, label="random pairs", color="#1a7a8c")
ax[0].hist(verif["positive_scores"], bins=20, alpha=.8, label="known confusable", color="#c0392b")
for cut, style, lbl in [(verif["configured_moderate_cutoff"], "--", "review 55"),
                        (verif["configured_high_cutoff"], "-", "reject 70")]:
    ax[0].axvline(cut, color="#333", linestyle=style, linewidth=1.4, label=lbl)
ax[0].set_xlabel("composite similarity"); ax[0].set_ylabel("count")
ax[0].set_title("Discrimination"); ax[0].legend(fontsize=8)

sw = verif["threshold_sweep"]
ax[1].plot(sw.threshold, sw.sensitivity, marker="o", ms=3, label="sensitivity", color="#c0392b")
ax[1].plot(sw.threshold, sw.specificity, marker="s", ms=3, label="specificity", color="#1a7a8c")
ax[1].axvline(70, color="#333", linewidth=1.4, label="configured reject")
ax[1].set_xlabel("threshold"); ax[1].set_title("Operating characteristics"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

md(r"""
### 6.2 Which similarity algorithm is doing the work?

Zero out each algorithm's weight in turn and re-measure AUC. An algorithm whose removal
changes nothing should not be described in the methodology as if it contributes.
""")

code(r"""
display(ev.ablate_phonetic_algorithms(system, n_negative=250))
""")

md(r"""
### 6.3 v1 against v2, same corpus, same verifier, same seed

The comparison that matters. v1's actual output is scored by the **same** quality
function as v2's — it had no quality function of its own, so measuring it now is the only
way to make the improvement a number rather than an opinion.
""")

code(r"""
comp = ev.compare_architectures(system, n=10)
display(comp["summary"])
print("\nPer-component means:")
display(comp["by_component"])
print("\nWhat each architecture actually produced:")
display(comp["candidates"][["architecture", "candidate_name", "quality_total",
                            "composite_risk_score", "risk_band", "proposer"]])
""")

md(r"""
### 6.4 Reproducibility
""")

code(r"""
det = ev.determinism_check(system, runs=3)
print("identical across runs:", det["identical"])
for i, r in enumerate(det["runs"], 1):
    print(f"  run {i}: {r}")
""")

# ---------------------------------------------------------------- 7. Sweep
md(r"""
---
## 7. Full sweep

Sixteen targets spanning the difficulty range — roomy classes, crowded classes,
saturated classes, and brand mode — writing **one row per candidate ever evaluated**,
accepted or not, with the `accepted` flag, failure codes and full quality decomposition.

A winners-only table cannot answer the question a reviewer actually asks, which is not
"what did it produce" but "what did it produce relative to what it tried".
""")

code(r"""
from pharma_name_gen.sweep import (run_sweep, acceptance_by_difficulty, proposer_contribution,
                          failure_profile, quality_component_correlations,
                          DEFAULT_TARGETS)

RUN_SWEEP = True          # ~3 minutes offline
N_PER_CLASS = 10

if RUN_SWEEP:
    sweep_df, sweep_summary = run_sweep(system, DEFAULT_TARGETS, n_per_class=N_PER_CLASS,
                                        out_csv="results/sweep_all_attempts.csv",
                                        progress=print)
    print(f"\n{sweep_summary['total_candidates']} candidates, "
          f"{sweep_summary['total_accepted']} accepted "
          f"({sweep_summary['overall_accept_rate']:.1%}) in {sweep_summary['wall_seconds']}s")
else:
    sweep_df = pd.read_csv("results/sweep_all_attempts.csv")
""")

code(r"""
print("Acceptance by class difficulty")
display(acceptance_by_difficulty(sweep_df))

print("\nProposer contribution — the comparison v1's `compare_strategies` was reaching for.")
print("Note the split between accept rate and shortlist share: they are different questions.")
display(proposer_contribution(sweep_df))

print("\nWhich checks actually reject")
display(failure_profile(sweep_df))

print("\nDoes each quality term move the composite?")
display(quality_component_correlations(sweep_df))
""")

code(r"""
display(ev.ablate_quality_components(system, sweep_df))
""")

code(r"""
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for d in sweep_df["difficulty"].unique():
    ax[0].hist(sweep_df[sweep_df.difficulty == d]["quality_total"].dropna(),
               bins=25, alpha=.55, label=d)
ax[0].set_xlabel("quality"); ax[0].set_title("Quality by class difficulty"); ax[0].legend(fontsize=8)

acc = sweep_df.groupby("sweep_target")["accepted"].mean().sort_values()
ax[1].barh(range(len(acc)), acc.values, color="#1a7a8c")
ax[1].set_yticks(range(len(acc))); ax[1].set_yticklabels(acc.index, fontsize=7)
ax[1].set_xlabel("acceptance rate"); ax[1].set_title("Difficulty by class")

ax[2].scatter(sweep_df["composite_risk_score"], sweep_df["quality_total"],
              s=6, alpha=.25, c=sweep_df["accepted"].map({True: "#1a7f4b", False: "#c0392b"}))
ax[2].axvline(55, ls="--", c="#333", lw=1); ax[2].axvline(70, c="#333", lw=1)
ax[2].set_xlabel("risk"); ax[2].set_ylabel("quality"); ax[2].set_title("Risk against quality")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 8. Export
md(r"""
---
## 8. Export and run manifest

The manifest is what makes a result citable. Corpus fingerprint, source records, git SHA,
every threshold and weight. Two runs are comparable if and only if their manifests match.
""")

code(r"""
import json
from pathlib import Path
from datetime import datetime, timezone

Path("results").mkdir(exist_ok=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
manifest = system.manifest()

if LAST_REPORT:
    LAST_REPORT.to_frame().to_csv(f"results/run_{stamp}_all_candidates.csv", index=False)
    pd.DataFrame([c.to_row() for c in LAST_REPORT.shortlist]).to_csv(
        f"results/run_{stamp}_shortlist.csv", index=False)
    manifest["last_run"] = {"request": LAST_REPORT.request, "stats": LAST_REPORT.stats}

Path(f"results/run_{stamp}_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
print(json.dumps({k: manifest[k] for k in ("system_version", "git_sha", "built_at")},
                 indent=2))
print(f"\ndata: {manifest['data']['mode']}  fingerprint={manifest['data']['fingerprint']}")
print(f"wrote results/run_{stamp}_*.csv and _manifest.json")
""")

code(r"""
# Publish trained artifacts so a fresh clone starts warm. This stages files with
# `git add` and deliberately stops there: a tool that commits and pushes as a side
# effect of a training run will eventually push something you did not want.
staged = system.store.publish()
print(f"staged {len(staged)} artifacts for commit:")
for p in staged:
    print("  ", p.name)
print("\nCommit them with:  git commit -m 'Update trained artifacts'")
""")

md(r"""
---
### Limitations

Stated plainly, because a paper that omits these invites the reviewer to find them.

1. **This is a first-pass screen.** No human-factors prescription simulation, no legal
   likelihood-of-confusion analysis, no committee judgement. A `low` band means *worth
   taking to review*, never *cleared*.
2. **The universe bounds the claim.** Distinctiveness is measured against the assembled
   corpus. Anything absent from it cannot be collided with, so margins are upper bounds
   and the `[LIVE]`/`[STATIC]` tag is load-bearing.
3. **Phonetics are orthographic and Anglocentric.** The grapheme-to-phoneme converter is
   rule-based and English-oriented; cross-lingual checking is a lexicon lookup, not a
   phonological model of each language.
4. **In stem-governed classes almost everything lands in the review band.** The stem
   itself forces high orthographic similarity to siblings. This is a property of the
   problem rather than a defect of the screen, and stem-aware scoring is how it is
   handled — but it means the low/moderate split should be read with the class's
   saturation in mind.
5. **The trademark check is a proxy.** It screens marketed-name collision, not registered
   trademark classes; a genuine clearance search is a legal instrument.
""")

nb = {"cells": C,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

Path("main.ipynb").write_text(json.dumps(nb, indent=1))
print(f"main.ipynb written: {len(C)} cells "
      f"({sum(1 for c in C if c['cell_type']=='code')} code, "
      f"{sum(1 for c in C if c['cell_type']=='markdown')} markdown)")
