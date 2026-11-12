"""FastAPI backend.

    GET    /health               Ollama reachable? models pulled? how many chunks indexed?
    POST   /documents            upload one PDF (multipart form field "file")
    GET    /documents            list indexed documents
    DELETE /documents/{doc_id}   remove a document and all its chunks
    POST   /chat                 ask a question, with optional history and document filter

Run:   uvicorn app.api:app --reload        (from the RAG-TEST folder)
Docs:  http://localhost:8000/docs          (interactive, generated from schemas.py)
"""
from __future__ import annotations

import logging

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile, status

from .documents import InvalidDocumentError
from .model_backend import ModelBackendError, cloud_status, get_model_provider
from .schemas import ChatRequest, ChatResponse, DocumentOut, HealthResponse, UploadResponse
from .services import ServiceUnavailable, Services, get_services, index_pdf, ollama_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

app = FastAPI(
    title="RAGShield API",
    description="Upload PDFs and ask questions about them. Answers cite document and page. "
    "Configurable cloud chat and embedding APIs with ChromaDB retrieval.",
    version="1.0.0",
)

# Errors that mean "the model server is down or a model is missing", not a bug.
try:
    from ollama import ResponseError as _OllamaResponseError
except ImportError:  # pragma: no cover
    _OllamaResponseError = RuntimeError
MODEL_ERRORS = (ServiceUnavailable, ModelBackendError, ConnectionError, httpx.HTTPError, _OllamaResponseError)


def services() -> Services:
    """FastAPI dependency. Tests replace it with fake models via app.dependency_overrides."""
    try:
        return get_services()
    except ServiceUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))


def _model_unavailable(exc: Exception) -> HTTPException:
    log.warning("model backend error: %r", exc)
    hint = str(exc)
    if "not found" in hint.lower():
        hint += " (pull the model first, e.g. `ollama pull mistral` / `ollama pull nomic-embed-text`)"
    return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Model server unavailable: {hint}")


@app.get("/livez")
def livez():
    """Process liveness, independent of model availability."""
    return {"status": "ok"}


@app.get("/health", response_model=HealthResponse)
def health():
    if get_model_provider() != "ollama":
        backend = cloud_status()
        documents = None
        if backend["configured"]:
            try:
                documents = get_services().store.count()
            except Exception:
                backend.update(configured=False, error="Could not open the document index.")
        return HealthResponse(status="configured" if backend["configured"] else "degraded",
                              backend=backend, documents=documents)
    ollama = ollama_status()
    documents = None
    ok = ollama["reachable"] and all(ollama["models"].values())
    if ollama["reachable"]:
        try:
            documents = get_services().store.count()
        except Exception:
            ok = False
    return HealthResponse(status="ok" if ok else "degraded", ollama=ollama, backend={"provider": "ollama", **ollama}, documents=documents)


@app.post("/documents", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
def upload_document(file: UploadFile = File(...), svc: Services = Depends(services)):
    # Read at most limit+1 bytes: enough to detect "too large" without reading a huge upload into memory.
    data = file.file.read(svc.settings.max_upload_mb * 1024 * 1024 + 1)
    try:
        result = index_pdf(svc.store, svc.settings, file.filename or "", data)
    except InvalidDocumentError as exc:
        log.info("rejected upload %r: %s", file.filename, exc)
        raise HTTPException(exc.status_code, detail=str(exc))
    except MODEL_ERRORS as exc:
        raise _model_unavailable(exc)
    return UploadResponse(**result)


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(svc: Services = Depends(services)):
    return [DocumentOut(**d) for d in svc.store.list_documents()]


@app.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: str, svc: Services = Depends(services)):
    if svc.store.delete_document(doc_id) == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, svc: Services = Depends(services)):
    try:
        result = svc.pipeline.answer(
            req.question,
            history=[m.model_dump() for m in req.history],
            doc_ids=req.doc_ids or None,
        )
    except MODEL_ERRORS as exc:
        raise _model_unavailable(exc)

    top = max((s.score for s in result.retrieved), default=None)
    log.info(
        "chat status=%s top_score=%s cited=%s grounding=%s timings=%s q=%r",
        result.status, top, result.cited, result.grounding, result.timings_ms, req.question[:80],
    )
    return ChatResponse(
        answer=result.answer,
        status=result.status,
        standalone_question=result.standalone_question,
        sources=[s.__dict__ for s in result.sources],
        cited=result.cited,
        grounding=result.grounding,
        timings_ms=result.timings_ms,
    )
