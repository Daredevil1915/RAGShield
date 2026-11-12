"""RAG Security Research Mode — Streamlit view.

A thin UI over app/research.py (which itself reuses the untouched research code).
Flow: pick a valid target -> run clean RAG -> pick BBO/GCG -> generate/inject a
blocker into the ISOLATED experimental collection -> run poisoned RAG -> compare
-> verdict (retrieval reported separately from jamming) -> always clean up -> export.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@st.cache_resource(show_spinner=False)
def get_service():
    from app.research import ResearchService
    if os.getenv("APP_RESEARCH_DEMO") == "1":
        from app.model_backend import embedding_namespace
        namespace = embedding_namespace()
        return ResearchService(source_db=f"data/demo_research_source_{namespace}", source_collection="demo_corpus",
                               exp_db=f"data/research_chroma_{namespace}")
    return ResearchService()


def _metric_badge(result) -> None:
    if not result.valid_target:
        st.warning(f"⚪ {result.verdict_label}", icon="⚪")
    elif result.jammed:
        st.error(f"🔴 {result.verdict_label}", icon="🔴")
    else:
        st.success(f"🟢 {result.verdict_label}", icon="🟢")


def render() -> None:
    from app.research import GCG_STYLES, METHOD_BBO, METHOD_GCG

    ss = st.session_state
    ss.setdefault("rs_clean", {})      # query -> clean_pass result
    ss.setdefault("rs_result", None)   # last ExperimentResult

    st.title("🛡️ RAG Security Research Mode")
    st.caption(
        "Reproduces the availability (jamming) attack from the Summer Training Report: a single blocker "
        "document is injected into an **isolated copy** of the MS-MARCO corpus, and the clean vs poisoned "
        "answers are compared. **Retrieval success is reported separately from jamming success** — a blocker "
        "can be retrieved and the query still not be jammed. Blockers are always removed after each run."
    )

    svc = get_service()
    demo = os.getenv("APP_RESEARCH_DEMO") == "1"
    if demo:
        st.info("Demo corpus: a small MS-MARCO sample with gold passages and 40 distractors. "
                "Results from this sample do not reproduce the full research benchmark.")
        if st.button("Prepare demo corpus"):
            bar = st.progress(0.0, "Connecting to models")
            try:
                svc.prepare_demo_corpus(progress=lambda f, m: bar.progress(f, m))
                st.rerun()
            except Exception as exc:
                st.error(f"Could not prepare the demo corpus: {exc}")

    # ── Backend + experimental-collection readiness ────────────────────────────
    try:
        status = svc.experiment_status()
    except Exception as exc:
        st.error(f"Cannot reach the research corpus or models: {exc}")
        return

    if status.get("error"):
        if demo:
            st.info("Select Prepare demo corpus to set up the research demonstration.")
        else:
            st.error(status["error"])
        return

    with st.sidebar:
        st.subheader("Experiment setup")
        if status["ready"]:
            st.success(f"Isolated collection ready · {status['copied']:,} passages", icon="✅")
        else:
            st.warning(
                f"One-time setup: copy {status['source']:,} MS-MARCO vectors into an isolated collection "
                "(the research DB is never modified).", icon="⚙️",
            )
            if st.button("Build experimental collection", type="primary", width="stretch"):
                bar = st.progress(0.0, "Starting…")
                try:
                    svc.ensure_experiment_collection(progress=lambda f, m: bar.progress(min(f, 1.0), m))
                    bar.progress(1.0, "Done")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Build failed: {exc}")
            st.info("Build the collection to enable experiments.")

    # ── Step 1: target query + clean pass ──────────────────────────────────────
    st.subheader("1 · Target query")
    queries = svc.dataset()
    labels = {q.query: q.query for q in queries}
    query = st.selectbox("Pick a valid target (clean RAG answers these):", list(labels), format_func=labels.get)

    run_clean = st.button("Run clean RAG", disabled=not status["ready"])
    if run_clean:
        try:
            with st.spinner("Running the clean RAG pipeline…"):
                ss.rs_clean[query] = svc.clean_pass(query)
            ss.rs_result = None
        except Exception as exc:
            st.error(f"Clean pass failed. Check the configured model APIs: {exc}")

    clean = ss.rs_clean.get(query)
    if clean:
        st.markdown(f"**Clean answer** {'🚫 (clean system already refuses — not a valid target)' if clean['refuses'] else ''}")
        st.info(clean["answer"])
        with st.expander(f"Top-{len(clean['passages'])} retrieved passages (clean KB)", expanded=False):
            for p in clean["passages"]:
                tag = " · 🧪 BLOCKER" if p.is_blocker else ""
                st.markdown(f"**#{p.rank}** · similarity-distance {p.distance:.3f}{tag}")
                st.caption(p.text + ("…" if len(p.text) >= 600 else ""))

    # ── Step 2: attack + experiment ────────────────────────────────────────────
    st.subheader("2 · Attack method")
    c1, c2 = st.columns(2)
    method = c1.radio("Method", [METHOD_GCG, METHOD_BBO],
                      help="GCG = gradient-optimized adversarial suffix. BBO = black-box blocker optimization.")

    if method == METHOD_GCG:
        prebuilt = svc.available_prebuilt_styles(query)
        # Offer the standard styles plus whatever pre-made styles exist for this query.
        style_opts = sorted(set(GCG_STYLES) | set(prebuilt))
        default_idx = style_opts.index(prebuilt[0]) if prebuilt else 0
        style = c2.selectbox("Jam style", style_opts, index=default_idx,
                             help="'nosuffix' is the report's control (on-topic prose, no suffix). "
                                  f"Pre-made for this query: {', '.join(prebuilt) or 'none'}.")
        premade_ok = style in prebuilt
        source = c2.radio("Blocker source", ["Pre-made (instant)", "Generate fresh"],
                          index=0 if premade_ok else 1,
                          help="Pre-made reproduces the report exactly; fresh re-runs your generator (~30-60s).")
        quick = True
    else:
        style = "bbo"
        c2.caption("BBO optimizes against real retrieved context (black-box, query-only). "
                   "No pre-made blockers exist for these 18 queries, so BBO generates fresh.")
        source = "Generate fresh"
        quick = c2.toggle("Quick run (fewer iterations)", value=True)

    use_judge = st.toggle("Grade with LLM judge (report methodology; slower)", value=True)

    can_run = status["ready"] and clean is not None and not clean["refuses"]
    if clean is not None and clean["refuses"]:
        st.warning("This query's clean answer is a refusal, so it is not a valid target. Pick another query.")

    if st.button("Generate blocker & run experiment", type="primary", disabled=not can_run):
        try:
            use_premade = method == METHOD_GCG and source == "Pre-made (instant)"
            with st.spinner("Preparing the blocker…" if use_premade else "Generating blocker (reusing your attack code)…"):
                blk = svc.get_prebuilt_blocker(query, style) if use_premade else None
                if blk is None:
                    blk = svc.generate_blocker(query, method, style=style, quick=quick)
            with st.spinner("Injecting into the isolated collection and running the poisoned pass… (clean-up is automatic)"):
                ss.rs_result = svc.run_experiment(
                    query, blk["full_blocker"], method=method, style=style,
                    blocker_origin=blk["origin"], use_llm_judge=use_judge,
                )
        except Exception as exc:
            st.error(f"Experiment failed: {exc}")

    # ── Step 3: results ────────────────────────────────────────────────────────
    result = ss.rs_result
    if result and result.query == query:
        st.subheader("3 · Clean vs poisoned")
        _metric_badge(result)

        a, b = st.columns(2)
        a.markdown("**Clean answer**")
        a.info(result.clean_answer)
        b.markdown("**Poisoned answer**")
        (b.error if result.jammed else b.info)(result.poisoned_answer)

        m = st.columns(4)
        m[0].metric("Blocker sim-rank", result.blocker_sim_rank if result.blocker_sim_rank > 0 else "—")
        m[1].metric("In-context pos", result.blocker_ctx_pos if result.blocker_ctx_pos > 0 else "not retrieved")
        m[2].metric("Outcome", result.outcome)
        m[3].metric("Jammed?", "YES" if result.jammed else "no")

        st.markdown(
            f"> **Retrieval ≠ Jamming.** {result.retrieval_vs_jamming}. "
            + ("The blocker reached the model but the correct answer survived — exactly the dilution effect "
               "documented in the report." if result.retrieved and not result.jammed else
               "The blocker both reached the model and suppressed the answer." if result.jammed else
               "The blocker did not even make it into the context.")
        )

        with st.expander("Blocker document + run details"):
            st.caption(f"Source: {result.blocker_origin} · method {result.method}/{result.style} · "
                       f"k={result.k} · judge={'LLM' if result.used_llm_judge else 'keywords'} · "
                       f"{result.elapsed_s}s")
            st.code(result.full_blocker)
            st.caption(f"✅ Cleanup verified: {result.blockers_cleaned} tagged blockers and "
                       f"{result.ids_restored} stray docs remained after the run (both should be 0 — "
                       "the isolated collection is back to the clean corpus).")

        if st.button("Export results"):
            paths = svc.export(result)
            st.success(f"Saved {paths['json']}")
            jf = Path(paths["json"]); cf = Path(paths["csv"])
            d1, d2 = st.columns(2)
            d1.download_button("Download run JSON", jf.read_bytes(), file_name=jf.name, mime="application/json")
            d2.download_button("Download runs CSV", cf.read_bytes(), file_name=cf.name, mime="text/csv")
