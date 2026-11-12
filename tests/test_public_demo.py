"""The public interview workspace must run without embedding credentials."""
import math

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from app import api as api_module
from app.config import get_settings
from app.demo import DEMO_FILENAME
from app.model_backend import DemoEmbeddings, cloud_status, get_embedding_provider
from app.research import ResearchService
from app.services import build_services, index_pdf


@pytest.fixture
def public_demo(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama_cloud")
    monkeypatch.setenv("APP_PUBLIC_DEMO", "1")
    monkeypatch.setenv("APP_RESEARCH_DEMO", "1")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")  # Existing Render setting must not break the public demo.
    monkeypatch.setenv("APP_DB_DIR", str(tmp_path / "documents"))
    monkeypatch.delenv("APP_MIN_RELEVANCE", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: pytest.fail("No cloud request is needed without a configured key"))


def test_word_feature_vectors_are_deterministic_and_normalized():
    embed = DemoEmbeddings()
    vector = embed.embed_query("uploaded documents and citations")
    assert vector == embed.embed_documents(["uploaded documents and citations"])[0]
    assert len(vector) == 2048
    assert math.isclose(sum(value * value for value in vector), 1)


def test_public_workspace_has_a_real_sample_pdf_without_api_calls(public_demo):
    assert get_embedding_provider() == "hashing"
    status = cloud_status()
    assert status["configured"] and status["retrieval"] == "keyword"
    svc = build_services()
    documents = svc.store.list_documents()
    assert len(documents) == 1 and documents[0]["filename"] == DEMO_FILENAME
    assert documents[0]["pages_with_text"] == 3
    assert get_settings().min_relevance == 0.08


def test_missing_cloud_key_still_returns_cited_source_sentences(public_demo):
    result = build_services().pipeline.answer("What is jamming success?")
    assert result.status == "answered" and result.cited
    assert result.generation_mode == "extractive"
    assert "Jamming success means" in result.answer
    assert any(source.page == 2 for source in result.sources)


def test_public_demo_still_refuses_irrelevant_questions(public_demo):
    result = build_services().pipeline.answer("Which football team won the World Cup in 2022?")
    assert result.status == "not_found"


def test_visitors_can_upload_search_and_remove_their_pdf(public_demo, physics_pdf):
    svc = build_services()
    result = index_pdf(svc.store, svc.settings, "physics.pdf", physics_pdf)
    answer = svc.pipeline.answer("At what temperature does water boil?", doc_ids=[result["doc_id"]])
    assert answer.status == "answered" and answer.generation_mode == "extractive"
    assert "100 degrees" in answer.answer
    assert all(source.filename == "physics.pdf" for source in answer.sources)
    assert svc.store.delete_document(result["doc_id"]) > 0


def test_public_research_corpus_prepares_without_an_embedding_api(public_demo, tmp_path):
    svc = ResearchService(source_db=str(tmp_path / "source"), exp_db=str(tmp_path / "experiment"))
    svc.prepare_demo_corpus()
    assert svc.ensure_experiment_collection() == {"ready": True, "source": 64, "copied": 64}


def test_cloud_generation_still_runs_when_available(public_demo, monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-ollama-key")
    calls = []
    def post(url, **kwargs):
        calls.append(url)
        assert url == "https://ollama.com/api/chat"
        return httpx.Response(200, json={"message": {"content":
            "RAGShield retrieves relevant passages and answers with document and page citations [1]."}},
            request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", post)
    result = build_services().pipeline.answer("How does RAGShield answer questions?")
    assert result.status == "answered" and result.generation_mode == "llm"
    assert len(calls) == 1


def test_cloud_quota_failure_keeps_demo_usable_and_avoids_repeated_calls(public_demo, monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-cloud-key")
    calls = []
    def post(url, **kwargs):
        calls.append(url)
        return httpx.Response(429, json={"error": {"code": "credit_balance_exhausted"}},
                              request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", post)
    svc = build_services()
    for _ in range(2):
        result = svc.pipeline.answer("What is jamming success?")
        assert result.status == "answered" and result.generation_mode == "extractive"
        assert result.cited
    assert calls == ["https://ollama.com/api/chat"]


def test_api_sample_restore_and_source_only_chat(public_demo, monkeypatch):
    svc = build_services()
    api_module.app.dependency_overrides[api_module.services] = lambda: svc
    monkeypatch.setattr(api_module, "get_services", lambda: svc)
    try:
        with TestClient(api_module.app) as client:
            assert client.get("/health").json()["backend"]["public_demo"]
            assert client.post("/demo/document").json()["duplicate"]
            response = client.post("/chat", json={"question": "What is jamming success?"})
            assert response.status_code == 200
            assert response.json()["generation_mode"] == "extractive"
    finally:
        api_module.app.dependency_overrides.clear()
