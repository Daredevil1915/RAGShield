from pathlib import Path

import requests
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from app import api as api_module
from app.services import build_services


def test_public_visitor_can_open_guide_and_get_labelled_answer(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_PUBLIC_DEMO", "1")
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
    finally:
        api_module.app.dependency_overrides.clear()
