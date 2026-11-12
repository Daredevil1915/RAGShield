"""Wires the real models into the pipeline, plus the indexing use case.

Kept apart from api.py so the HTTP layer stays thin, and so tests and the
evaluation script can build the same objects with fake or real models.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.request import Request, urlopen

from rag_utils import create_embeddings, create_llm, get_embed_model, get_llm_model, get_ollama_base_url, get_ollama_headers

from .config import Settings, get_settings
from .documents import chunk_document, parse_pdf
from .embeddings import NomicPrefixedEmbeddings
from .model_backend import embedding_namespace, get_embedding_provider, public_demo_mode
from .rag import RAGPipeline
from .vector_store import VectorStore

log = logging.getLogger("app")


class ServiceUnavailable(RuntimeError):
    """A model backend is down or misconfigured. The API returns HTTP 503."""


@dataclass
class Services:
    settings: Settings
    store: VectorStore
    pipeline: RAGPipeline


_services: Optional[Services] = None
_lock = threading.Lock()


def build_services(settings: Optional[Settings] = None) -> Services:
    settings = settings or get_settings()
    try:
        # Shared factories route both app and research to the selected provider.
        embeddings = create_embeddings()
        if get_embedding_provider() == "ollama":
            embeddings = NomicPrefixedEmbeddings(embeddings)
        llm = create_llm()
    except RuntimeError as exc:  # rag_utils.ensure_ollama_server raises this
        raise ServiceUnavailable(str(exc).strip()) from exc
    collection = settings.collection
    if get_embedding_provider() != "ollama":
        collection += "_" + embedding_namespace()
    store = VectorStore(embeddings, settings.db_dir, collection)
    result = Services(settings=settings, store=store, pipeline=RAGPipeline(store, llm, settings))
    if public_demo_mode() and store.count() == 0:
        from .demo import load_demo_document
        load_demo_document(result)
    return result


def get_services() -> Services:
    """Build once, on first use (not at import), so the API can start before Ollama is up."""
    global _services
    if _services is None:
        with _lock:
            if _services is None:
                _services = build_services()
    return _services


def index_pdf(store: VectorStore, settings: Settings, filename: str, data: bytes) -> Dict:
    """Validate -> extract -> chunk -> embed -> store. Raises InvalidDocumentError for bad files."""
    parsed = parse_pdf(filename, data, max_bytes=settings.max_upload_mb * 1024 * 1024, max_pages=settings.max_pages)
    result = {
        "doc_id": parsed.doc_id,
        "filename": parsed.filename,
        "num_pages": parsed.num_pages,
        "pages_with_text": len(parsed.pages),
    }
    if store.has_document(parsed.doc_id):
        # Same bytes as an existing upload: no duplicate chunks (they would crowd the top-k).
        return {**result, "chunks": 0, "duplicate": True}
    chunks, ids = chunk_document(parsed, settings.chunk_size, settings.chunk_overlap)
    store.add_chunks(chunks, ids)
    log.info("indexed %s doc_id=%s pages=%d chunks=%d", filename, parsed.doc_id, parsed.num_pages, len(chunks))
    return {**result, "chunks": len(chunks), "duplicate": False}


def ollama_status(timeout: float = 3.0) -> Dict:
    """Is Ollama reachable, and are both models pulled? Used by /health."""
    base = get_ollama_base_url()
    wanted = {"llm": get_llm_model(), "embeddings": get_embed_model()}
    try:
        with urlopen(Request(f"{base}/api/tags", headers=get_ollama_headers()), timeout=timeout) as resp:
            names = [m.get("name", "") for m in json.load(resp).get("models", [])]
    except Exception:
        return {"reachable": False, "base_url": base, "models": {k: False for k in wanted}}

    def present(model: str) -> bool:
        return any(n == model or n.split(":")[0] == model or n == f"{model}:latest" for n in names)

    return {"reachable": True, "base_url": base, "models": {k: present(v) for k, v in wanted.items()}}
