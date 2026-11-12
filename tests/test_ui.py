"""The Streamlit UI, driven headlessly with streamlit.testing.AppTest.

The UI's HTTP calls (requests.request) are routed into the FastAPI app via
TestClient, so this exercises UI -> API -> pipeline in one process, no Ollama.
"""
from pathlib import Path

import pytest
import requests
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from app import api as api_module
from app.rag import RAGPipeline
from app.services import Services, index_pdf
from tests.conftest import FakeLLM

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


def test_ui_off_topic_question_shows_not_found(ui):
    ui.run()
    ui.chat_input[0].set_value("Who won the football world cup final?").run()
    text, captions = last_reply(ui)
    assert "couldn't find" in text
    assert not any("Sources" in c for c in captions)


def test_ui_password_gate(ui, monkeypatch):
    monkeypatch.setenv("DEMO_PASSWORD", "test-pass")
    ui.run()
    assert not ui.chat_input  # nothing usable before login
    ui.text_input[0].input("nope")
    ui.button[0].click().run()
    assert ui.error[0].value == "Wrong password."
    ui.text_input[0].input("test-pass")
    ui.button[0].click().run()
    assert ui.title[0].value == "Ask your PDFs"


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
