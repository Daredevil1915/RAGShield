from pathlib import Path

import requests
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from app import api as api_module
from app.services import build_services


def test_public_visitor_can_open_guide_and_get_labelled_answer(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_PUBLIC_DEMO", "1")
    monkeypatch.setenv("APP_DEMO_LIVE_LLM", "0")
    monkeypatch.setenv("MODEL_PROVIDER", "ollama_cloud")
    monkeypatch.setenv("APP_DB_DIR", str(tmp_path / "documents"))
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("APP_MIN_RELEVANCE", raising=False)
    svc = build_services()
    api_module.app.dependency_overrides[api_module.services] = lambda: svc
    monkeypatch.setattr(api_module, "get_services", lambda: svc)
    client = TestClient(api_module.app)
    monkeypatch.setattr(requests, "request", lambda method, url, timeout=None, **kwargs: client.request(method, url, **kwargs))
    st.cache_data.clear()
    try:
        ui = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "ui/streamlit_app.py"), default_timeout=30).run()
        assert not ui.exception
        assert any("RAGShield-Guide.pdf" in item.value for item in ui.sidebar.markdown)
        next(button for button in ui.button if button.key == "suggestion_0").click().run()
        assert not ui.exception
        assert any("SOURCE-ONLY ANSWER" in item.value for item in ui.caption)
        assert any("RAGShield-Guide.pdf, p." in item.value for item in ui.caption)
        from app.research import ResearchService
        from ui import research_view
        research = ResearchService(source_db=str(tmp_path / "source"), exp_db=str(tmp_path / "experiment"))
        research.prepare_demo_corpus()
        research.ensure_experiment_collection()
        monkeypatch.setenv("APP_RESEARCH_DEMO", "1")
        monkeypatch.setattr(research_view, "get_service", lambda: research)
        ui.sidebar.radio[0].set_value("Security Research").run()
        next(button for button in ui.button if button.label == "Run retrieval experiment").click().run()
        assert not ui.exception
        assert any(metric.label == "Collection restored" and metric.value == "Yes" for metric in ui.metric)
        assert not any(button.label == "Run clean baseline" for button in ui.button)
    finally:
        api_module.app.dependency_overrides.clear()
