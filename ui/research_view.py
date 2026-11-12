"""Interactive workspace for controlled RAG availability experiments."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import streamlit as st

from ui.theme import hero, step_card
from app.model_backend import public_demo_mode, demo_live_generation_enabled


@st.cache_resource(show_spinner=False)
def get_service():
    from app.research import ResearchService
    if os.getenv("APP_RESEARCH_DEMO") == "1":
        from app.model_backend import embedding_namespace
        namespace = embedding_namespace()
        state_dir = Path(os.getenv("APP_DB_DIR", "data/app_chroma")).parent
        service = ResearchService(source_db=str(state_dir / f"demo_research_source_{namespace}"), source_collection="demo_corpus",
                                  exp_db=str(state_dir / f"research_chroma_{namespace}"),
                                  export_dir=str(state_dir / "research_runs"))
        from app.model_backend import public_demo_mode
        if public_demo_mode() and not service.experiment_status().get("ready"):
            service.prepare_demo_corpus()
            service.ensure_experiment_collection()
        return service
    return ResearchService()


def render_retrieval_demo(svc):
    st.info("LIVE RETRIEVAL EXPERIMENT | Runs on Render without model APIs or billing.")
    st.caption("This measures retrieval poisoning. Model refusal and answer jamming are not evaluated in this mode.")
    queries = [item.query for item in svc.dataset()]
    default = next((index for index, query in enumerate(queries) if "huitzilopochtli" in query.lower()), 0)
    with st.container(border=True):
        st.subheader("Compare clean and poisoned retrieval")
        query = st.selectbox("Target question", queries, index=default, key="retrieval_target")
        styles = svc.available_prebuilt_styles(query)
        style = st.selectbox("Saved blocker style", styles) if styles else None
        saved = svc.get_prebuilt_blocker(query, style) if style else None
        text = saved["full_blocker"] if saved else f"Accounts of {query} are conflicting and unverified."
        blocker = st.text_area("Blocker document", value=text, max_chars=5000, height=140,
                               key=f"blocker_{query}_{style}", help="Use the saved document or edit it before injection.")
        depth = st.select_slider("Context passages", options=[1, 3, 5, 10], value=5)
        if st.button("Run retrieval experiment", type="primary", icon=":material/science:", disabled=not blocker.strip()):
            try:
                with st.spinner("Retrieving the clean context, injecting the blocker, and restoring the collection..."):
                    st.session_state.rs_retrieval = svc.run_retrieval_experiment(query, blocker, top_k=depth)
            except Exception as exc:
                st.error(f"The retrieval experiment could not finish: {exc}")
    result = st.session_state.get("rs_retrieval")
    if result and result["query"] == query:
        metrics = st.columns(4)
        metrics[0].metric("Blocker retrieved", "Yes" if result["blocker_retrieved"] else "No")
        metrics[1].metric("Blocker rank", result["blocker_rank"] if result["blocker_rank"] > 0 else "Outside top 50")
        metrics[2].metric("Original context retained", f"{result['original_passages_retained']}/{result['context_size']}")
        metrics[3].metric("Collection restored", "Yes" if result["collection_restored"] else "No")
        st.success("Live experiment complete. The injected blocker was removed and the original collection was restored.")
        st.caption("These are actual vector-search results. A retrieved blocker does not establish language-model jamming.")
        for column, title, key in zip(st.columns(2), ["Clean context", "Poisoned context"], ["clean_context", "poisoned_context"]):
            with column:
                st.subheader(title)
                for passage in result[key]:
                    label = "Injected blocker" if passage["is_blocker"] else "Corpus passage"
                    with st.expander(f"#{passage['rank']} | {label}", expanded=passage["is_blocker"]):
                        st.caption(f"Distance {passage['distance']:.4f}")
                        st.write(passage["text"])
        st.download_button("Download experiment JSON", json.dumps(result, indent=2),
                           file_name="ragshield-retrieval-experiment.json", mime="application/json")


def render():
    from app.research import GCG_STYLES, METHOD_BBO, METHOD_GCG

    ss = st.session_state
    ss.setdefault("rs_clean", {})
    ss.setdefault("rs_result", None)
    ss.setdefault("rs_exports", None)
    hero("SECURITY RESEARCH", "Put your RAG pipeline to the test.",
         "Compare clean and poisoned answers. See whether a blocker reaches the model and whether the answer survives.")
    with st.expander("Explore a recorded project experiment", icon=":material/history:"):
        record = json.loads((Path(__file__).resolve().parents[1] / "examples/recorded_experiment.json").read_text(encoding="utf-8"))
        result = record["result"]
        st.caption("RECORDED PROJECT RUN | Not a live inference or a current-demo benchmark result")
        st.markdown(f"**Target question:** {result['query']}")
        left, right = st.columns(2)
        left.metric("Recorded retrieval", "Retrieved" if result["retrieved"] else "Not retrieved")
        right.metric("Recorded jamming", "Jammed" if result["jammed"] else "Answer survived")
        st.markdown("**Clean response**")
        st.write(result["clean_answer"])
        st.markdown("**Poisoned response**")
        st.write(result["poisoned_answer"])
        st.caption("In this recorded run the blocker ranked first, but the answer survived. Retrieval success alone did not establish jamming.")
        st.code(result["full_blocker"], language=None)
        st.caption(record["provenance"]["note"])
    demo = os.getenv("APP_RESEARCH_DEMO") == "1"
    try:
        svc = get_service()
        status = svc.experiment_status()
    except Exception as exc:
        st.error(f"The research workspace is unavailable: {exc}")
        return

    with st.sidebar:
        st.subheader("Research workspace")
        if status.get("ready"):
            st.success("Experiment collection ready")
            st.caption(f"{status['copied']:,} passages | isolated experiment storage")
        else:
            st.caption("Prepare the collection to start your first experiment.")
        st.divider()
        st.markdown("**How to read the results**")
        st.caption("Retrieved: the blocker reached the context.\n\nJammed: a previously answerable question lost its answer.")
        st.caption("RAGShield | Shared demo workspace")

    if not status.get("ready"):
        with st.container(border=True):
            st.subheader("Set up your research workspace")
            st.write("Prepare a small MS-MARCO sample and an isolated experiment collection." if demo else
                     "Create an isolated experiment collection from the configured research corpus.")
            if demo:
                st.caption("64 passages | 24 gold passages and 40 distractors | demonstration corpus")
            if status.get("error") and not demo:
                st.warning("The research corpus has not been configured. Use the demo corpus or follow the research setup guide.")
            elif st.button("Prepare research workspace", type="primary", icon=":material/science:"):
                setup = None
                try:
                    with st.status("Preparing the research workspace...", expanded=True) as setup:
                        if demo:
                            st.write("Embedding the demonstration corpus")
                            svc.prepare_demo_corpus()
                        st.write("Building the isolated experiment collection")
                        svc.ensure_experiment_collection()
                        setup.update(label="Research workspace ready", state="complete")
                    st.rerun()
                except Exception as exc:
                    if setup is not None:
                        setup.update(label="Research setup needs attention", state="error", expanded=True)
                    st.error(f"Setup could not finish: {exc}")
        columns = st.columns(3)
        for column, args in zip(columns, [
            ("01", "Establish a baseline", "Check that the clean pipeline answers the target question."),
            ("02", "Introduce a blocker", "Inject a controlled document into the isolated collection."),
            ("03", "Compare the outcome", "Inspect retrieval, answer changes, and automatic cleanup."),
        ]):
            with column:
                step_card(*args)
        return

    queries = svc.dataset()
    if not queries:
        st.info("No target questions are available in the configured dataset.")
        return
    metrics = st.columns(3)
    metrics[0].metric("Corpus passages", status["copied"])
    metrics[1].metric("Target questions", len(queries))
    metrics[2].metric("Retrieval depth", svc.k)
    if demo:
        st.caption("Demo corpus | Results illustrate this sample and model configuration, not the original benchmark.")

    if public_demo_mode() and not demo_live_generation_enabled():
        render_retrieval_demo(svc)
        return

    st.write("")
    with st.container(border=True):
        st.subheader("01 | Establish a baseline")
        st.caption("A question must be answerable before an availability attack can be evaluated.")
        query = st.selectbox("Target question", [item.query for item in queries])
        if st.button("Run clean baseline", icon=":material/play_arrow:"):
            try:
                with st.spinner("Retrieving context and generating the clean answer..."):
                    ss.rs_clean[query] = svc.clean_pass(query)
                ss.rs_result = None
                ss.rs_exports = None
            except Exception as exc:
                st.error(f"The baseline could not finish: {exc}")
                st.info("Live research requires cloud generation. The recorded experiment above remains available; document Q&A also supports source-only answers.")
        clean = ss.rs_clean.get(query)
        if clean:
            if clean["refuses"]:
                st.warning("The clean pipeline could not answer this question. Choose another target.")
            st.markdown("**Clean answer**")
            st.info(clean["answer"])
            with st.expander("Inspect baseline evidence", icon=":material/format_quote:"):
                for passage in clean["passages"]:
                    st.markdown(f"**Passage {passage.rank}**")
                    st.caption(f"Retrieval distance {passage.distance:.3f}")
                    st.write(passage.text)

    with st.container(border=True):
        st.subheader("02 | Configure the experiment")
        left, right = st.columns(2)
        method = left.radio("Attack method", [METHOD_GCG, METHOD_BBO], horizontal=True,
                            help="GCG uses adversarial suffixes. BBO optimizes blockers using black-box model calls.")
        fresh_available = bool(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"))
        if method == METHOD_GCG:
            prebuilt = svc.available_prebuilt_styles(query)
            styles = sorted(set(GCG_STYLES) | set(prebuilt))
            style = right.selectbox("Blocker style", styles, index=styles.index(prebuilt[0]) if prebuilt else 0)
            options = ["Pre-made (instant)", "Generate fresh"] if style in prebuilt else ["Generate fresh"]
            source = left.radio("Blocker source", options, horizontal=True)
            quick = True
            generation_ready = source == "Pre-made (instant)" or fresh_available
            if not generation_ready:
                st.caption("Fresh suffix generation requires the full research environment. Choose an available pre-made style for this demo.")
        else:
            style, source, generation_ready = "bbo", "Generate fresh", True
            right.caption("BBO generates a blocker against the retrieved context using the configured cloud APIs.")
            quick = left.toggle("Quick experiment", value=True, help="Use fewer optimization iterations.")
        use_judge = st.toggle("Use an LLM judge", value=not public_demo_mode(),
                              help="Grade answer preservation with the model. This makes additional API calls.")
        can_run = clean is not None and not clean["refuses"] and generation_ready
        if not clean:
            st.caption("Run the clean baseline to enable the experiment.")
        if st.button("Run poisoning experiment", type="primary", icon=":material/science:", disabled=not can_run):
            ss.rs_exports = None
            try:
                use_premade = method == METHOD_GCG and source == "Pre-made (instant)"
                with st.status("Running the experiment...", expanded=True) as experiment:
                    st.write("Preparing the blocker document")
                    blocker = svc.get_prebuilt_blocker(query, style) if use_premade else None
                    if blocker is None:
                        blocker = svc.generate_blocker(query, method, style=style, quick=quick)
                    st.write("Comparing clean and poisoned answers")
                    ss.rs_result = svc.run_experiment(query, blocker["full_blocker"], method=method, style=style,
                                                      blocker_origin=blocker["origin"], use_llm_judge=use_judge)
                    experiment.update(label="Experiment complete", state="complete")
            except Exception as exc:
                st.error(f"The experiment could not finish: {exc}")

    result = ss.rs_result
    if result and result.query == query:
        st.subheader("03 | Compare the results")
        if not result.valid_target:
            st.warning(result.verdict_label)
        elif result.jammed:
            st.error(result.verdict_label)
        else:
            st.success(result.verdict_label)
        metrics = st.columns(4)
        metrics[0].metric("Blocker retrieved", "Yes" if result.retrieved else "No")
        metrics[1].metric("Answer jammed", "Yes" if result.jammed else "No")
        metrics[2].metric("Retrieval rank", result.blocker_sim_rank if result.blocker_sim_rank > 0 else "Outside top 50")
        metrics[3].metric("Run time", f"{result.elapsed_s:.1f}s")
        left, right = st.columns(2)
        with left.container(border=True):
            st.markdown("**Clean response**")
            st.write(result.clean_answer)
        with right.container(border=True):
            st.markdown("**Poisoned response**")
            st.write(result.poisoned_answer)
        st.caption("Retrieval and jamming are separate outcomes. A blocker can be retrieved while the answer remains intact.")
        with st.expander("Inspect experiment evidence", icon=":material/description:"):
            st.caption(f"{result.method} | {result.style} | context position {result.blocker_ctx_pos} | outcome {result.outcome}")
            st.code(result.full_blocker, language=None)
            st.caption(f"Cleanup sweep removed {result.blockers_cleaned} tagged blockers and {result.ids_restored} stray documents.")
        if st.button("Prepare result downloads", icon=":material/download:"):
            ss.rs_exports = {"query": query, **svc.export(result)}
        if ss.rs_exports and ss.rs_exports["query"] == query:
            left, right = st.columns(2)
            json_file, csv_file = Path(ss.rs_exports["json"]), Path(ss.rs_exports["csv"])
            left.download_button("Download run JSON", json_file.read_bytes(), file_name=json_file.name, mime="application/json")
            right.download_button("Download results CSV", csv_file.read_bytes(), file_name=csv_file.name, mime="text/csv")
