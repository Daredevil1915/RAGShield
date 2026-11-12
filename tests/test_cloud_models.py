"""Cloud protocol, API flow and research checks; never call a live provider."""
import httpx
import pytest
from fastapi.testclient import TestClient

from app import api as api_module
from app.model_backend import CloudEmbeddings, CloudLLM, ModelBackendError, cloud_status, embedding_namespace
from app.research import ResearchService
from app.services import build_services
from tests.conftest import FakeEmbeddings


@pytest.fixture
def cloud(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama_cloud")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OLLAMA_API_KEY", "test-ollama-cloud")
    monkeypatch.setenv("OPENAI_API_KEY", "test-embedding-cloud")
    monkeypatch.setenv("OLLAMA_CLOUD_BASE_URL", "https://chat.example")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://embedding.example/v1")
    for name in ["OPENAI_EMBED_BASE_URL", "OPENAI_EMBED_API_KEY"]:
        monkeypatch.delenv(name, raising=False)
    calls = []
    vectors = FakeEmbeddings()

    def post(url, *, headers, json, timeout):
        calls.append((url, headers, json))
        request = httpx.Request("POST", url)
        if url == "https://embedding.example/v1/embeddings":
            assert headers["Authorization"] == "Bearer test-embedding-cloud"
            assert json["encoding_format"] == "float"
            assert all(not t.startswith("search_") for t in json["input"])
            data = [{"index": i, "embedding": vec} for i, vec in enumerate(vectors.embed_documents(json["input"]))]
            return httpx.Response(200, json={"data": list(reversed(data))}, request=request)
        assert url == "https://chat.example/api/chat"  # No localhost or Ollama daemon calls.
        assert headers["Authorization"] == "Bearer test-ollama-cloud"
        assert json["stream"] is False and json["think"] is False
        prompt = json["messages"][0]["content"]
        answer = ("I don't know based on the provided context." if "DEMO_JAM_BLOCKER" in prompt else
                  "Huitzilopochtli is an Aztec deity of war and the sun." if "huitzilopochtli" in prompt.lower() else
                  "Water boils at 100 degrees Celsius at sea level [1].")
        return httpx.Response(200, json={"message": {"content": answer, "thinking": "never display reasoning"}}, request=request)

    monkeypatch.setattr(httpx, "post", post)
    return calls


def test_cloud_upload_chat_delete_via_real_api(cloud, settings, physics_pdf, monkeypatch):
    svc = build_services(settings)
    assert svc.store.db._collection.name.endswith(embedding_namespace())
    api_module.app.dependency_overrides[api_module.services] = lambda: svc
    monkeypatch.setattr(api_module, "get_services", lambda: svc)
    try:
        with TestClient(api_module.app) as client:
            health = client.get("/health").json()
            assert health["status"] == "configured"
            assert health["backend"]["provider"] == "ollama_cloud"
            assert cloud == []  # Health requests never incur inference charges.
            uploaded = client.post("/documents", files={"file": ("physics.pdf", physics_pdf, "application/pdf")})
            assert uploaded.status_code == 201, uploaded.text
            answer = client.post("/chat", json={"question": "At what temperature does water boil?"}).json()
            assert answer["status"] == "answered"
            assert "100 degrees" in answer["answer"]
            assert answer["sources"][0]["page"] == 1
            assert "reasoning" not in answer["answer"]
            assert client.delete("/documents/" + uploaded.json()["doc_id"]).status_code == 204
    finally:
        api_module.app.dependency_overrides.clear()


def test_cloud_research_clean_poisoned_and_cleanup(cloud, tmp_path):
    svc = ResearchService(source_db=str(tmp_path / "source"), exp_db=str(tmp_path / "experiment"))
    svc.prepare_demo_corpus()
    svc.ensure_experiment_collection()
    result = svc.run_experiment(
        "who is huitzilopochtli", "DEMO_JAM_BLOCKER " + "huitzilopochtli " * 20,
        method="GCG", style="test", blocker_origin="test fixture", use_llm_judge=False,
    )
    assert result.valid_target and result.retrieved and result.jammed
    assert svc._exp_collection_obj().count() == 64
    assert svc.sweep_cleanup() == 0
    assert any(url.endswith("/api/chat") for url, *_ in cloud)


def test_single_openai_compatible_provider(cloud, monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openai")

    def post(url, **kwargs):
        assert url == "https://embedding.example/v1/chat/completions"
        assert kwargs["headers"]["Authorization"] == "Bearer test-embedding-cloud"
        return httpx.Response(200, json={"choices": [{"message": {"content": "Hosted answer"}}]},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    assert CloudLLM().invoke("Test") == "Hosted answer"


def test_missing_key_is_configuration_error_without_network(cloud, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    assert cloud_status()["configured"] is False
    with pytest.raises(ModelBackendError, match="OPENAI_API_KEY"):
        CloudEmbeddings()
    assert cloud == []


@pytest.mark.parametrize("code", [401, 403, 404, 429, 503])
def test_provider_errors_are_safe_and_become_503(cloud, code, settings, physics_pdf, monkeypatch):
    svc = build_services(settings)

    def fail(url, **kwargs):
        return httpx.Response(code, json={"error": "secret provider response test-embedding-cloud"},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fail)
    api_module.app.dependency_overrides[api_module.services] = lambda: svc
    try:
        with TestClient(api_module.app) as client:
            response = client.post("/documents", files={"file": ("physics.pdf", physics_pdf, "application/pdf")})
        assert response.status_code == 503
        assert str(code) in response.text
        assert "test-embedding-cloud" not in response.text
        assert "secret provider" not in response.text
    finally:
        api_module.app.dependency_overrides.clear()


def test_embedding_model_changes_storage_namespace(cloud, monkeypatch, settings):
    first = build_services(settings)
    monkeypatch.setenv("OPENAI_EMBED_MODEL", "another-embedding-model")
    second = build_services(settings)
    assert first.store.db._collection.name != second.store.db._collection.name


def test_separate_embedding_service_uses_its_own_key(cloud, monkeypatch):
    monkeypatch.setenv("OPENAI_EMBED_BASE_URL", "https://separate.example/v1")
    monkeypatch.setenv("OPENAI_EMBED_API_KEY", "test-separate-key")

    def post(url, **kwargs):
        assert url == "https://separate.example/v1/embeddings"
        assert kwargs["headers"]["Authorization"] == "Bearer test-separate-key"
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    assert CloudEmbeddings().embed_query("Test") == [0.1, 0.2]


def test_cloud_mode_rejects_local_embeddings(cloud, monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    assert not cloud_status()["configured"]
    assert "local Ollama is not used" in cloud_status()["error"]


@pytest.mark.parametrize("body", [[], {"message": None}, {"message": {"content": ""}}])
def test_invalid_chat_response_is_actionable(cloud, monkeypatch, body):
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: httpx.Response(
        200, json=body, request=httpx.Request("POST", url)))
    with pytest.raises(ModelBackendError):
        CloudLLM().invoke("Test")
