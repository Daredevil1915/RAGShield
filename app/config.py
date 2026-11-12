"""Application settings.

Everything is overridable with an environment variable (or a line in .env), so
the same code runs locally, in tests and in Docker. The APP_ prefix keeps these
separate from the research settings (DB_DIR, COLLECTION_NAME) in the existing
.env, so the app never touches the research ChromaDB databases in db/.

Ollama settings (OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_EMBED_MODEL, ...)
are read by the existing rag_utils.py, exactly as in the research code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# Greedy decoding for the app: the same question gives the same answer, which
# suits factual Q&A and makes evaluation runs reproducible. The research code
# keeps its 0.1. rag_utils.create_llm() reads this variable; an explicit
# OLLAMA_TEMPERATURE in the environment still wins.
os.environ.setdefault("OLLAMA_TEMPERATURE", "0")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # Storage: a separate ChromaDB folder and collection from the research DBs.
    db_dir: str = "data/app_chroma"
    collection: str = "pdf_documents"

    # Chunking, in tiktoken tokens (same splitter as the research code).
    # 500 / 50 is what wiki_ingest.py used: small enough that a citation points
    # at a specific passage, big enough to keep a paragraph together.
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval. top_k chunks go to the LLM; chunks whose cosine similarity is
    # below min_relevance are discarded. If nothing survives, the app says
    # "not found" WITHOUT calling the LLM, so it has nothing to make up an
    # answer from. The default was calibrated with evaluation/run_eval.py.
    top_k: int = 4
    min_relevance: float = 0.55

    # Input limits (basic reliability).
    max_upload_mb: int = 25
    max_pages: int = 500
    max_question_chars: int = 1000

    # Follow-up questions: how many past messages are used to rewrite them.
    history_messages: int = 6


def get_settings() -> Settings:
    d = Settings()
    return Settings(
        db_dir=os.getenv("APP_DB_DIR", d.db_dir),
        collection=os.getenv("APP_COLLECTION", d.collection),
        chunk_size=_int("APP_CHUNK_SIZE", d.chunk_size),
        chunk_overlap=_int("APP_CHUNK_OVERLAP", d.chunk_overlap),
        top_k=_int("APP_TOP_K", d.top_k),
        min_relevance=_float("APP_MIN_RELEVANCE", d.min_relevance),
        max_upload_mb=_int("APP_MAX_UPLOAD_MB", d.max_upload_mb),
        max_pages=_int("APP_MAX_PAGES", d.max_pages),
        max_question_chars=_int("APP_MAX_QUESTION_CHARS", d.max_question_chars),
        history_messages=_int("APP_HISTORY_MESSAGES", d.history_messages),
    )
