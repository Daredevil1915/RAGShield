"""Streamlit frontend. It only talks to the FastAPI backend over HTTP.

Run:  streamlit run ui/streamlit_app.py      (API_URL defaults to http://localhost:8000)

The UI holds the chat history in st.session_state and sends it with every
question, so follow-up questions work while the API stays stateless.
"""
from __future__ import annotations

import hmac
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
HISTORY_TO_SEND = 10  # messages; the API uses the most recent ones to rewrite follow-ups
# Set DEMO_PASSWORD when the UI is reachable from the internet (e.g. via a tunnel).
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "")

st.set_page_config(page_title="RAGShield", page_icon="🛡️", layout="wide")

ss = st.session_state

if DEMO_PASSWORD and not ss.get("authenticated"):
    st.title("🛡️ RAGShield")
    st.caption("This demo is password-protected.")
    with st.form("login"):
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Enter", type="primary"):
            if hmac.compare_digest(pw.encode(), DEMO_PASSWORD.encode()):
                ss.authenticated = True
                st.rerun()
            st.error("Wrong password.")
    st.stop()

ss.setdefault("messages", [])        # {"role", "content", "status", "sources", "standalone"}
ss.setdefault("uploader_key", 0)
ss.setdefault("upload_notes", [])    # results of the last upload, shown after the rerun

# Two modes in one app. Document Q&A (below) is unchanged; Security Research is a
# separate page that reuses the research attack/eval code (see ui/research_view.py).
MODE = st.sidebar.radio("Mode", ["📄 Document Q&A", "🛡️ Security Research"], key="app_mode")
if MODE.endswith("Security Research"):
    import research_view
    research_view.render()
    st.stop()
st.sidebar.divider()


def api(method: str, path: str, **kwargs):
    """Call the backend; return (response or None, error message or None)."""
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=kwargs.pop("timeout", 30), **kwargs)
    except requests.RequestException:
        return None, f"Cannot reach the API at {API_URL}. Is the backend running?"
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        if isinstance(detail, list):  # FastAPI validation errors
            detail = "; ".join(d.get("msg", str(d)) for d in detail)
        return r, str(detail)
    return r, None


@st.cache_data(ttl=10, show_spinner=False)
def get_health():
    r, err = api("GET", "/health", timeout=10)
    return (r.json() if r is not None and not err else None), err


def render_sources(msg: dict) -> None:
    sources = msg.get("sources") or []
    if msg.get("status") != "answered" or not sources:
        return
    # Always visible: which document and page each [n] refers to.
    refs = " · ".join(f"[{s['id']}] {s['filename']}, p. {s['page']}" for s in sources)
    st.caption(f"📎 {'Sources' if msg.get('cited') else 'Retrieved from (no inline citations)'}: {refs}")

    with st.expander("Show source passages", expanded=False):
        details = []
        if msg.get("standalone") and msg["standalone"] != msg.get("question"):
            details.append(f"Searched for: *{msg['standalone']}*")
        if msg.get("grounding") is not None:
            details.append(f"Grounding: {msg['grounding']:.0%} of the answer's key terms appear in the sources")
        if details:
            st.caption("  \n".join(details))
        for s in sources:
            st.markdown(f"**[{s['id']}] {s['filename']} — page {s['page']}**  ·  relevance {s['score']:.2f}")
            text = s["text"]
            st.caption(text[:700] + ("…" if len(text) > 700 else ""))


# ── Sidebar: status, upload, documents ─────────────────────────────────────
with st.sidebar:
    st.header("📄 Documents")

    health, health_err = get_health()
    if health_err:
        st.error(health_err)
    elif health["status"] == "ok":
        st.success(f"Backend ready · {health.get('documents') or 0} chunks indexed", icon="✅")
    else:
        o = health["ollama"]
        if not o["reachable"]:
            st.warning(f"Ollama is not reachable at {o['base_url']}. Start Ollama, then refresh.", icon="⚠️")
        else:
            missing = [k for k, ok in o["models"].items() if not ok]
            st.warning(f"Ollama is up, but these models are missing: {', '.join(missing)}. "
                       "Run `ollama pull mistral` and `ollama pull nomic-embed-text`.", icon="⚠️")

    files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True, key=f"uploader_{ss.uploader_key}"
    )
    if files and st.button("Index documents", type="primary", width="stretch"):
        notes = []
        for f in files:
            with st.spinner(f"Reading, chunking and embedding {f.name}…"):
                r, err = api("POST", "/documents", files={"file": (f.name, f.getvalue(), "application/pdf")},
                             timeout=900)
            if err:
                notes.append(("error", f"{f.name}: {err}"))
            elif r.json()["duplicate"]:
                notes.append(("info", f"{f.name}: already indexed."))
            else:
                d = r.json()
                notes.append(("success", f"{f.name}: {d['num_pages']} pages → {d['chunks']} chunks."))
        ss.upload_notes = notes
        ss.uploader_key += 1          # clears the file picker
        get_health.clear()
        st.rerun()

    for kind, text in ss.upload_notes:
        getattr(st, kind)(text)

    r, docs_err = api("GET", "/documents")
    docs = r.json() if r is not None and not docs_err else []
    if docs_err and not health_err:
        st.caption(f"Could not list documents: {docs_err}")

    selected_ids = None
    if docs:
        st.subheader("Indexed")
        names = {d["doc_id"]: d["filename"] for d in docs}
        chosen = st.multiselect(
            "Search only in (empty = all documents)", options=list(names), format_func=names.get
        )
        selected_ids = chosen or None
        for d in docs:
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{d['filename']}**  \n{d['pages_with_text']} pages · {d['chunks']} chunks")
            if c2.button("🗑", key=f"del_{d['doc_id']}", help="Remove from the knowledge base"):
                api("DELETE", f"/documents/{d['doc_id']}")
                ss.upload_notes = [("info", f"Removed {d['filename']}.")]
                get_health.clear()
                st.rerun()
    elif not docs_err:
        st.caption("No documents yet. Upload a PDF to get started.")

    st.divider()
    if st.button("Clear chat", width="stretch"):
        ss.messages = []
        st.rerun()


# ── Main: chat ─────────────────────────────────────────────────────────────
st.title("Ask your PDFs")
st.caption("Answers come only from your uploaded documents, with document and page citations. "
           "If the documents don't contain the answer, the assistant says so instead of guessing.")

for msg in ss.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg)

question = st.chat_input("Ask a question about your documents…")
if question:
    history = [{"role": m["role"], "content": m["content"]}
               for m in ss.messages if m.get("status") != "error"][-HISTORY_TO_SEND:]
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents and writing an answer…"):
            r, err = api("POST", "/chat", json={"question": question, "history": history, "doc_ids": selected_ids},
                         timeout=900)
        if err:
            msg = {"role": "assistant", "content": f"⚠️ {err}", "status": "error"}
        else:
            data = r.json()
            content = data["answer"] if data["status"] == "answered" else f"🔎 {data['answer']}"
            msg = {
                "role": "assistant", "content": content, "status": data["status"], "sources": data["sources"],
                "cited": data["cited"], "grounding": data.get("grounding"),
                "standalone": data["standalone_question"], "question": question,
            }
        st.markdown(msg["content"])
        render_sources(msg)
    ss.messages.append(msg)
