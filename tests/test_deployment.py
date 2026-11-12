"""Exercise the hosted Ollama protocol and deployment additions without model downloads."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.research import ResearchService
from app.services import build_services, index_pdf, ollama_status
from tests.conftest import FakeEmbeddings


@pytest.fixture
def hosted_ollama(monkeypatch):
    vectors = FakeEmbeddings()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send_json(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            if self.headers.get("Authorization") != "Bearer test-hosted-token":
                self.send_json({"error": "unauthorized"}, 401)
                return False
            return True

        def do_GET(self):
            if self.authorized():
                self.send_json({"models": [{"name": "mistral:latest"}, {"name": "nomic-embed-text:latest"}]})

        def do_POST(self):
            if not self.authorized():
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/api/embed":
                texts = body["input"]
                texts = [texts] if isinstance(texts, str) else texts
                self.send_json({"embeddings": vectors.embed_documents(texts)})
            elif self.path == "/api/generate":
                self.send_json({"model": "mistral", "response": "Water boils at 100 degrees Celsius at sea level [1].",
                                "done": True})
            else:
                self.send_json({"error": "unexpected endpoint"}, 404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OLLAMA_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("OLLAMA_API_KEY", "test-hosted-token")
    monkeypatch.setenv("OLLAMA_LLM_MODEL", "mistral")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    yield
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_hosted_models_authenticate_for_health_embedding_and_generation(hosted_ollama, settings, physics_pdf):
    status = ollama_status()
    assert status["reachable"] and all(status["models"].values())
    svc = build_services(settings)
    index_pdf(svc.store, settings, "physics.pdf", physics_pdf)
    result = svc.pipeline.answer("At what temperature does water boil?")
    assert result.status == "answered"
    assert "100 degrees" in result.answer
    assert result.sources[0].page == 1


def test_demo_corpus_embeds_and_copies_without_original_database(hosted_ollama, tmp_path):
    svc = ResearchService(source_db=str(tmp_path / "source"), exp_db=str(tmp_path / "experiment"))
    svc.prepare_demo_corpus()
    svc.prepare_demo_corpus()  # idempotent
    status = svc.ensure_experiment_collection()
    assert status == {"ready": True, "source": 64, "copied": 64}


def test_liveness_does_not_require_model_backend(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:1")
    with TestClient(app) as client:
        assert client.get("/livez").json() == {"status": "ok"}
