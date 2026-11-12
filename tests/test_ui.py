"""The Streamlit UI, driven headlessly with streamlit.testing.AppTest.

The UI's HTTP calls (requests.request) are routed into the FastAPI app via
TestClient, so this exercises UI -> API -> pipeline in one process, no Ollama.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from app import api as api_module
from app.rag import RAGPipeline
from app.services import Services, index_pdf
from tests.conftest import FakeLLM
from app.research import ExperimentResult

UI_SCRIPT = str(Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py")


@pytest.fixture
def ui(settings, store, physics_pdf, monkeypatch):
    svc = Services(settings=settings, store=store, pipeline=RAGPipeline(store, FakeLLM(), settings))
    index_pdf(store, settings, "physics.pdf", physics_pdf)
    api_module.app.dependency_overrides[api_module.services] = lambda: svc
    monkeypatch.setattr(api_module, "get_services", lambda: svc)
    monkeypatch.setattr(
        api_module, "ollama_status",
        lambda: {"reachable": True, "base_url": "http://ollama", "models": {"llm": True, "embeddings": True}},
    )
    client = TestClient(api_module.app)
    monkeypatch.setattr(requests, "request", lambda method, url, timeout=None, **kw: client.request(method, url, **kw))
    monkeypatch.delenv("DEMO_PASSWORD", raising=False)
    st.cache_data.clear()
    yield AppTest.from_file(UI_SCRIPT, default_timeout=30)
    api_module.app.dependency_overrides.clear()


def last_reply(at):
    msg = at.chat_message[-1]
    return " ".join(m.value for m in msg.markdown), [c.value for c in msg.caption]


def test_ui_shows_documents_and_answers_with_page_citation(ui):
    ui.run()
    assert not ui.exception
    assert "Backend ready" in ui.sidebar.success[0].value
    assert any("physics.pdf" in m.value for m in ui.sidebar.markdown)

    ui.chat_input[0].set_value("At what temperature does water boil?").run()
    text, captions = last_reply(ui)
    assert "100 degrees" in text
    assert any("physics.pdf, p. 1" in c for c in captions)
    assert any(metric.label == "Questions" and metric.value == "1" for metric in ui.metric)


def test_ui_off_topic_question_shows_not_found(ui):
    ui.run()
    ui.chat_input[0].set_value("Who won the football world cup final?").run()
    text, captions = last_reply(ui)
    assert "couldn't find" in text
    assert not any("Sources" in c for c in captions)


def test_ui_opens_without_login_even_with_existing_render_password(ui, monkeypatch):
    monkeypatch.setenv("DEMO_PASSWORD", "test-pass")
    ui.run()
    assert not ui.exception
    assert ui.chat_input and not ui.chat_input[0].disabled
    assert not ui.text_input
    assert not any("password" in item.value.lower() for item in ui.caption)


def test_ui_starter_question_runs_chat(ui):
    ui.run()
    next(button for button in ui.button if button.key == "suggestion_0").click().run()
    assert not ui.exception
    assert len(ui.chat_message) == 2


def test_ui_new_conversation_clears_messages(ui):
    ui.run()
    ui.chat_input[0].set_value("At what temperature does water boil?").run()
    next(button for button in ui.button if button.label == "New conversation").click().run()
    assert not ui.chat_message


def test_ui_empty_library_disables_chat(ui, store):
    for document in store.list_documents():
        store.delete_document(document["doc_id"])
    ui.run()
    assert not ui.exception
    assert ui.chat_input[0].disabled
    assert any("first PDF" in item.value for item in ui.sidebar.caption)


class ResearchUIService:
    k = 5

    def __init__(self, ready=True):
        self.ready = ready
        self.prepared = False

    def experiment_status(self):
        return {"ready": self.ready, "copied": 64 if self.ready else 0, "source": 64}

    def dataset(self):
        return [SimpleNamespace(query="who is huitzilopochtli")]

    def prepare_demo_corpus(self):
        self.prepared = True

    def ensure_experiment_collection(self):
        self.ready = True

    def clean_pass(self, query):
        return {"answer": "An Aztec deity of war and the sun.", "refuses": False, "passages": []}

    def available_prebuilt_styles(self, query):
        return ["distraction_nosuffix"]

    def get_prebuilt_blocker(self, query, style):
        return {"full_blocker": "A controlled blocker", "origin": "test fixture"}

    def run_experiment(self, query, full_blocker, **kwargs):
        return ExperimentResult(
            query=query, method=kwargs["method"], style=kwargs["style"], blocker_origin="test fixture",
            full_blocker=full_blocker, clean_answer="An Aztec deity of war and the sun.",
            poisoned_answer="I don't know based on the provided context.",
            clean_refuses=False, poisoned_refuses=True, blocker_sim_rank=1, blocker_ctx_pos=1,
            retrieved=True, num_context=5, outcome="refusal", valid_target=True, jammed=True,
            k=5, used_llm_judge=kwargs["use_llm_judge"], blockers_cleaned=0, ids_restored=0,
            elapsed_s=1.2, timestamp="test",
        )


def research_workspace(ui, monkeypatch, ready=True):
    from ui import research_view
    svc = ResearchUIService(ready=ready)
    monkeypatch.setattr(research_view, "get_service", lambda: svc)
    monkeypatch.setenv("APP_RESEARCH_DEMO", "1")
    ui.run()
    ui.sidebar.radio[0].set_value("Security Research").run()
    return svc


def test_ui_research_setup_is_one_step(ui, monkeypatch):
    svc = research_workspace(ui, monkeypatch, ready=False)
    assert not ui.exception
    next(button for button in ui.button if button.label == "Prepare research workspace").click().run()
    assert not ui.exception
    assert svc.prepared and svc.ready
    assert any(button.label == "Run clean baseline" for button in ui.button)


def test_ui_research_setup_failure_is_visible_and_retryable(ui, monkeypatch):
    svc = research_workspace(ui, monkeypatch, ready=False)
    def fail():
        raise RuntimeError("API credits are exhausted. Add credits before retrying.")
    monkeypatch.setattr(svc, "prepare_demo_corpus", fail)
    next(button for button in ui.button if button.label == "Prepare research workspace").click().run()
    assert not ui.exception
    assert any("API credits are exhausted" in item.value for item in ui.error)
    assert any(button.label == "Prepare research workspace" for button in ui.button)


def test_ui_research_runs_baseline_and_compares_results(ui, monkeypatch):
    research_workspace(ui, monkeypatch)
    experiment = next(button for button in ui.button if button.label == "Run poisoning experiment")
    assert experiment.disabled
    next(button for button in ui.button if button.label == "Run clean baseline").click().run()
    assert not ui.exception
    next(button for button in ui.button if button.label == "Run poisoning experiment").click().run()
    assert not ui.exception
    assert any(metric.label == "Answer jammed" and metric.value == "Yes" for metric in ui.metric)


def test_ui_failed_document_delete_does_not_report_success(ui, monkeypatch):
    original = requests.request
    monkeypatch.setattr(requests, "request", lambda method, url, **kwargs:
                        SimpleNamespace(status_code=503, text="unavailable", json=lambda: {"detail": "Please retry"})
                        if method == "DELETE" else original(method, url, **kwargs))
    ui.run()
    next(button for button in ui.button if button.label == "Remove").click().run()
    assert any("Please retry" in error.value for error in ui.sidebar.error)
    assert any("physics.pdf" in item.value for item in ui.sidebar.markdown)


def test_ui_cloud_configuration_status(ui, monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama_cloud")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OLLAMA_API_KEY", "test-chat-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-embedding-key")
    ui.run()
    assert not ui.exception
    assert any("Cloud APIs configured" in m.value for m in ui.sidebar.info)
    assert not ui.sidebar.warning


def test_ui_cloud_missing_key_shows_setup_error(ui, monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama_cloud")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    ui.run()
    assert not ui.exception
    assert any("OLLAMA_API_KEY" in m.value for m in ui.sidebar.warning)
    assert ui.chat_input
