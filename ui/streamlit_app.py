"""Public document workspace. The API is accessed server-side over HTTP."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
import streamlit as st

# Support both Streamlit's script runner and headless AppTest.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ui.theme import apply_theme, brand, hero, step_card
from app.model_backend import public_demo_mode

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
HISTORY_TO_SEND = 10

st.set_page_config(page_title="RAGShield | Document workspace", page_icon="\U0001f6e1\ufe0f", layout="wide")
apply_theme()
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("uploader_key", 0)
ss.setdefault("upload_notes", [])

with st.sidebar:
    brand()
    MODE = st.radio("Workspace", ["Document Q&A", "Security Research"], key="app_mode")
    st.divider()

if MODE == "Security Research":
    from ui import research_view
    research_view.render()
    st.stop()


def api(method: str, path: str, **kwargs):
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=kwargs.pop("timeout", 30), **kwargs)
    except requests.RequestException:
        return None, "The document service is temporarily unavailable. Please try again shortly."
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        if isinstance(detail, list):
            detail = "; ".join(item.get("msg", str(item)) for item in detail)
        return response, str(detail)
    return response, None


@st.cache_data(ttl=10, show_spinner=False)
def get_health():
    response, error = api("GET", "/health", timeout=10)
    return (response.json() if response is not None and not error else None), error


def render_sources(message: dict):
    sources = message.get("sources") or []
    if message.get("status") != "answered" or not sources:
        return
    refs = " | ".join(f"[{source['id']}] {source['filename']}, p. {source['page']}" for source in sources)
    label = "Sources" if message.get("cited") else "Retrieved passages"
    st.caption(f"{label}: {refs}")
    with st.expander("Explore the evidence", icon=":material/format_quote:"):
        if message.get("standalone") and message["standalone"] != message.get("question"):
            st.caption(f"Searched for: {message['standalone']}")
        if message.get("grounding") is not None:
            st.caption(f"Term overlap with retrieved passages: {message['grounding']:.0%}")
        for source in sources:
            with st.container(border=True):
                st.markdown(f"**[{source['id']}] {source['filename']} | page {source['page']}**")
                st.caption(f"Relevance {source['score']:.2f}")
                st.write(source["text"])
    if message.get("timings_ms"):
        seconds = sum(message["timings_ms"].values()) / 1000
        st.caption(f"Response time | {seconds:.1f}s")


with st.sidebar:
    st.subheader("Your library")
    st.caption("Add PDFs to give your conversation a source.")
    if public_demo_mode() and st.button("Load sample PDF", width="stretch", icon=":material/description:"):
        _, error = api("POST", "/demo/document")
        ss.upload_notes = [("error", error)] if error else [("success", "The RAGShield guide is ready. Try a starter question.")]
        get_health.clear()
        st.rerun()
    files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True,
                             key=f"uploader_{ss.uploader_key}", help="Text-based PDFs, up to 25 MB each.")
    if files and st.button("Add to library", type="primary", width="stretch", icon=":material/add:"):
        notes = []
        for file in files:
            with st.spinner(f"Preparing {file.name}..."):
                response, error = api("POST", "/documents",
                                      files={"file": (file.name, file.getvalue(), "application/pdf")}, timeout=900)
            if error:
                notes.append(("error", f"{file.name}: {error}"))
            else:
                result = response.json()
                if result["duplicate"]:
                    notes.append(("info", f"{file.name} is already in your library."))
                else:
                    notes.append(("success", f"{file.name} is ready | {result['num_pages']} pages."))
        ss.upload_notes = notes
        ss.uploader_key += 1
        get_health.clear()
        st.rerun()

    for kind, text in ss.upload_notes:
        getattr(st, kind)(text)

    response, docs_error = api("GET", "/documents")
    documents = response.json() if response is not None and not docs_error else []
    selected_ids = None
    if docs_error:
        st.caption("The document library is currently unavailable.")
    elif documents:
        names = {document["doc_id"]: document["filename"] for document in documents}
        chosen = st.multiselect("Search scope", options=list(names), format_func=names.get,
                                placeholder="All documents")
        selected_ids = chosen or None
        for document in documents:
            with st.container(border=True):
                st.markdown(f"**{document['filename']}**")
                st.caption(f"{document['pages_with_text']} pages | {document['chunks']} passages")
                if st.button("Remove", key=f"del_{document['doc_id']}", icon=":material/delete:"):
                    _, error = api("DELETE", f"/documents/{document['doc_id']}")
                    ss.upload_notes = [("error", error)] if error else [("info", f"Removed {document['filename']}.")]
                    get_health.clear()
                    st.rerun()
    else:
        st.caption("Your library is empty. Add your first PDF above.")

    st.divider()
    with st.expander("Service status", icon=":material/cloud:"):
        health, health_error = get_health()
        if health_error:
            st.error(health_error)
        elif health["status"] == "configured":
            backend = health.get("backend") or {}
            if backend.get("retrieval") == "keyword":
                st.info("Demo search ready on Render")
                st.caption("Word-feature retrieval needs no embedding API credits. Cloud generation is optional; source-only answers remain available.")
            else:
                st.info("Cloud APIs configured")
                st.caption("Access is verified when you upload or ask a question.")
        elif health["status"] == "ok":
            st.success("Backend ready")
        else:
            backend = health.get("backend") or {}
            if backend.get("provider") not in (None, "ollama"):
                st.warning(backend.get("error", "Check the cloud API configuration."))
            else:
                st.warning("The model service needs attention. Check its connection and available models.")
        if st.button("Refresh status", icon=":material/refresh:"):
            get_health.clear()
            st.rerun()
    st.caption("RAGShield | Shared demo workspace")

hero("DOCUMENT WORKSPACE", "Your documents. Clearer answers.",
     "Explore your PDFs through conversation, with source passages and page references a click away.")
with st.container(key="workspace_metrics"):
    metrics = st.columns(3)
    metrics[0].metric("Documents", len(documents))
    metrics[1].metric("Pages", sum(document["pages_with_text"] for document in documents))
    metrics[2].metric("Questions", sum(message["role"] == "user" for message in ss.messages))
st.write("")
heading, actions = st.columns([3, 1])
heading.subheader("Conversation")
if actions.button("New conversation", width="stretch", icon=":material/add_comment:"):
    ss.messages = []
    st.rerun()

suggested_question = None
if not ss.messages:
    if not documents:
        st.markdown("#### A little context goes a long way.")
        st.caption("Start with a report, research paper, or any PDF with selectable text.")
        columns = st.columns(3)
        for column, args in zip(columns, [
            ("01", "Add your documents", "Upload a PDF using the library panel."),
            ("02", "Ask a question", "Explore findings, methods, or details in your own words."),
            ("03", "Follow the evidence", "Open the original passages and page references behind an answer."),
        ]):
            with column:
                step_card(*args)
    else:
        st.markdown("#### Where would you like to start?")
        st.caption("Choose a starting point, or write your own question below.")
        sample = any(document["filename"] == "RAGShield-Guide.pdf" for document in documents)
        prompts = (["How does RAGShield answer questions?", "What is jamming success?", "What are the demo limitations?"] if sample else
                   ["What are the main findings?", "What limitations are mentioned?", "What methods are described?"])
        for index, (column, prompt) in enumerate(zip(st.columns(3), prompts)):
            if column.button(prompt, key=f"suggestion_{index}", width="stretch"):
                suggested_question = prompt

for message in ss.messages:
    avatar = ":material/person:" if message["role"] == "user" else ":material/shield:"
    with st.chat_message(message["role"], avatar=avatar):
        if message.get("generation_mode") == "extractive":
            st.caption("SOURCE-ONLY ANSWER | Retrieved sentences with citations; this is not a generated LLM response.")
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message)

with st.container():
    question = st.chat_input("Ask a question about your documents...", disabled=not documents) or suggested_question
if question:
    history = [{"role": message["role"], "content": message["content"]}
               for message in ss.messages if message.get("status") != "error"][-HISTORY_TO_SEND:]
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)
    with st.chat_message("assistant", avatar=":material/shield:"):
        with st.spinner("Finding the relevant passages..."):
            response, error = api("POST", "/chat", json={"question": question, "history": history,
                                                          "doc_ids": selected_ids}, timeout=900)
        if error:
            message = {"role": "assistant", "content": error, "status": "error"}
        else:
            data = response.json()
            message = {"role": "assistant", "content": data["answer"], "status": data["status"],
                       "sources": data["sources"], "cited": data["cited"], "grounding": data.get("grounding"),
                       "standalone": data["standalone_question"], "question": question,
                       "generation_mode": data.get("generation_mode", "llm"),
                       "timings_ms": data.get("timings_ms", {})}
        st.markdown(message["content"])
        render_sources(message)
    ss.messages.append(message)
    st.rerun()
