"""
Streamlit interface for the generative-verifier pharmaceutical name system.

Run with:  streamlit run app/streamlit_app.py

Secondary to the notebook by design. The notebook is the artifact that explains the
system; this is the one you hand to someone who only wants to use it. Same package, same
config objects, same manifest, so a name produced here and a name produced in the
notebook are produced by identical code — there is no second implementation to drift.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pharma_name_gen import PipelineConfig, build_system                      # noqa: E402
from pharma_name_gen import introspect as ins                                  # noqa: E402
from pharma_name_gen.verifier import VerifierConfig                            # noqa: E402

st.set_page_config(page_title="Pharmaceutical Name Generation",
                   page_icon="⚗", layout="wide")

BAND_COLOUR = {"low": "#1a7f4b", "moderate": "#b06e00", "high": "#a11"}

st.markdown("""
<style>
  .hero {background:linear-gradient(135deg,#0b3d5c 0%,#1a7a8c 100%);color:#fff;
         padding:22px 28px;border-radius:12px;margin-bottom:18px}
  .hero h1 {margin:0;font-size:2em;letter-spacing:.5px}
  .hero p  {margin:6px 0 0;opacity:.92}
  .namecard {border:1px solid #e3e8ec;border-radius:10px;padding:14px 18px;
             margin-bottom:10px;background:#fff}
  .namecard b.n {font-size:1.45em;letter-spacing:.6px;color:#0b3d5c}
  .pill {color:#fff;padding:2px 10px;border-radius:11px;font-size:.78em}
  .muted {color:#7a8792;font-size:.86em}
</style>
<div class="hero">
  <h1>Pharmaceutical Name Generation</h1>
  <p>Regulation-aware generation and screening of pharmaceutical names</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Assembling reference data and building the system…")
def get_system(live: bool, use_llm: bool, stem_aware: bool):
    return build_system(live=live, use_llm=use_llm, use_artifacts=True,
                        verifier_config=VerifierConfig(stem_aware_similarity=stem_aware))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("System")
    live = st.toggle("Live regulator data", value=True,
                     help="openFDA, RxNorm and the EMA register. Falls back to the "
                          "committed snapshot on any failure, and says so.")
    use_llm = st.toggle("LLM proposer", value=True,
                        help="OpenRouter free tier. Escalated to only when the free "
                             "pool's best result is thin. Needs OPENROUTER_API_KEY.")
    stem_aware = st.toggle("Stem-aware similarity", value=True,
                           help="Strip the mandated stem before scoring, so "
                                "distinctiveness measures the part the namer controls.")

    system = get_system(live, use_llm, stem_aware)
    snap = system.snapshot.manifest()
    st.caption(f"**{snap['mode'].upper()}** · {snap['unique_generic']} generic / "
               f"{snap['unique_brand']} brand names · {snap['stem_rows']} stems")
    st.caption(f"fingerprint `{snap['fingerprint']}`")
    if not system.generic.llm and use_llm:
        st.warning("No OPENROUTER_API_KEY found — running on the free CPU pool only.")

    st.divider()
    st.header("Search effort")
    pool = st.slider("Pool per proposer", 8, 64, 24, 4)
    rounds = st.slider("Max rounds", 1, 8, 4)
    n = st.slider("Shortlist size", 3, 25, 10)
    st.divider()
    strict = st.toggle("Strict mode", value=False,
                       help="Reject the 55-70 review band outright. In stem-governed "
                            "classes this removes most candidates, because the stem "
                            "itself forces similarity to siblings.")
    seed = st.number_input("Seed", value=20260826, step=1)


# ---------------------------------------------------------------- tabs
tab_gen, tab_arch, tab_eval = st.tabs(["Generate", "Architecture", "Evaluation"])

with tab_gen:
    mode = st.radio("Name type", ["Generic (INN/USAN)", "Brand (proprietary)"],
                    horizontal=True, label_visibility="collapsed")
    is_generic = mode.startswith("Generic")

    stems = system.snapshot.stems
    options = {f"{r.meaning}  ({r.stem})": r.stem
               for r in stems.itertuples(index=False) if str(r.position) == "suffix"}

    if is_generic:
        label = st.selectbox("Pharmacological class", sorted(options),
                             index=sorted(options).index(
                                 next(k for k in sorted(options) if "(-olol)" in k)))
        stem, target_class = options[label], label.split("  (")[0]
        siblings = [x for x in system.training.generic_tokens
                    if x.endswith(stem.lstrip("-"))]
        st.caption(f"Stem `{stem}` · {len(siblings)} existing names in this class"
                   + (f": {', '.join(siblings[:10])}" if siblings else ""))
    else:
        stem = None
        target_class = st.text_input("Therapeutic context",
                                     "cardiovascular, once-daily oral")
        st.caption("Brand mode inverts the objective: no stem, short, memorable, "
                   "no implied claim.")

    if st.button("Generate", type="primary", use_container_width=True):
        pipeline = system.pipeline("generic" if is_generic else "brand")
        pipeline.config = PipelineConfig(pool_per_proposer=pool, max_rounds=rounds,
                                         shortlist_size=n, use_llm=use_llm,
                                         treat_moderate_as_failure=strict, seed=int(seed))
        for p in pipeline.proposers:
            if hasattr(p, "config"):
                p.config = pipeline.config
        with st.spinner("Proposing in parallel, screening the pool, ranking…"):
            st.session_state.report = pipeline.generate(
                n_shortlist=n, target_class=target_class, target_stem=stem)

    report = st.session_state.get("report")
    if report:
        s = report.stats
        c = st.columns(5)
        c[0].metric("Returned", s["returned"])
        c[1].metric("Evaluated", s["candidates_evaluated"])
        c[2].metric("Best quality", s["best_quality"])
        c[3].metric("Clear band", f"{s['band_low']} / {s['admissible']}")
        c[4].metric("Time", f"{s['wall_seconds']}s")

        st.subheader("Shortlist")
        for i, cand in enumerate(report.shortlist, 1):
            q = {x.name: x.score for x in cand.quality.components}
            sim = cand.response.checks.similarity
            band = cand.response.risk_band.value
            left, right = st.columns([3, 2])
            with left:
                st.markdown(
                    f"<div class='namecard'><b class='n'>{i}. {cand.name}</b> "
                    f"<span class='pill' style='background:{BAND_COLOUR[band]}'>{band}</span>"
                    f"<br><span class='muted'>quality <b>{cand.quality.total:.1f}</b> · "
                    f"risk {cand.risk:.1f} · margin to review "
                    f"{sim.distinctiveness_margin_moderate:+.1f} · "
                    f"nearest <i>{sim.nearest_match}</i> · via {cand.proposer}</span></div>",
                    unsafe_allow_html=True)
            with right:
                st.bar_chart(pd.DataFrame({"score": q}), height=130)

        with st.expander("Every candidate evaluated, accepted or not"):
            st.dataframe(report.to_frame(), use_container_width=True, height=380)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        d1, d2 = st.columns(2)
        d1.download_button("Download all attempts (CSV)",
                           report.to_frame().to_csv(index=False),
                           f"pharma_name_gen_{stamp}_all.csv", use_container_width=True)
        d2.download_button("Download run manifest (JSON)",
                           json.dumps({**system.manifest(),
                                       "request": report.request,
                                       "stats": report.stats}, indent=2, default=str),
                           f"pharma_name_gen_{stamp}_manifest.json", use_container_width=True)

with tab_arch:
    st.code(ins.architecture_summary(system), language="text")
    st.code(ins.data_flow(), language="text")
    c1, c2 = st.columns(2)
    c1.subheader("Reference data"); c1.dataframe(ins.corpus_table(system),
                                                 use_container_width=True, hide_index=True)
    c2.subheader("Quality objective"); c2.dataframe(ins.quality_weights_table(system),
                                                    use_container_width=True, hide_index=True)
    st.subheader("Design decisions")
    for _, r in ins.design_decisions().iterrows():
        with st.expander(f"[{r['area']}]  {r['decision']}"):
            st.markdown(f"**Instead of:** {r['alternative']}\n\n"
                        f"**Because:** {r['reasoning']}\n\n"
                        f"**Evidence:** {r['evidence']}")

with tab_eval:
    st.caption("Known-confusable pairs (FDA / ISMP documented dispensing errors) against "
               "random pairs from the same corpus.")
    if st.button("Run verifier evaluation"):
        from pharma_name_gen import evaluation as ev
        with st.spinner("Scoring pairs…"):
            r = ev.evaluate_verifier(system, n_negative=300)
        c = st.columns(4)
        c[0].metric("ROC AUC", r["roc_auc"])
        c[1].metric("Separation", r["separation"])
        c[2].metric("Confusable mean", r["mean_confusable_score"])
        c[3].metric("Random mean", r["mean_random_score"])
        st.dataframe(r["threshold_sweep"], use_container_width=True, hide_index=True)
        st.info(
            f"At the configured reject cutoff of {r['configured_high_cutoff']}, "
            f"specificity is already 1.00 while sensitivity to documented confusable "
            f"pairs is much lower than at the {r['configured_moderate_cutoff']} review "
            f"line. The review band is doing real safety work, which is why it is "
            f"reported explicitly rather than passed silently.")