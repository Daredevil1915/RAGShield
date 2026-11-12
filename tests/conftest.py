"""Shared test fixtures.

The tests run without Ollama. They swap in two deterministic fakes:
  * FakeEmbeddings: bag-of-words hashing, so texts that share words are similar;
  * FakeLLM: scripted replies, and it records every prompt it receives.
Everything else is real: pypdf parsing, the tiktoken splitter, ChromaDB on a
temp folder, the RAG pipeline and the FastAPI app.
"""
from __future__ import annotations

import hashlib
import math
import re
import sys
from pathlib import Path
from typing import Callable, List, Optional

import pytest
from fpdf import FPDF
from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `import rag_utils` and `import app` work

from app.config import Settings  # noqa: E402
from app.embeddings import NomicPrefixedEmbeddings  # noqa: E402
from app.rag import RAGPipeline  # noqa: E402
from app.vector_store import VectorStore  # noqa: E402

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "and", "or", "what", "who",
    "how", "does", "do", "did", "it", "its", "for", "at", "by", "with", "about", "which", "when",
    "this", "that", "be", "as", "from", "did",
    # the nomic task prefixes ("search_document: ", "search_query: ") split into these words
    "search", "document", "query",
}


class FakeEmbeddings(Embeddings):
    dim = 256

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            if w in STOPWORDS:
                continue
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    """reply(prompt) -> str decides the answer; every prompt is recorded."""

    def __init__(self, reply: Optional[Callable[[str], str]] = None):
        self.prompts: List[str] = []
        self.reply = reply or (lambda p: "Water boils at 100 degrees Celsius at sea level [1].")

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply(prompt)

    @property
    def answer_prompts(self):
        return [p for p in self.prompts if "<sources>" in p]

    @property
    def rewrite_prompts(self):
        return [p for p in self.prompts if "Standalone question:" in p]


def make_pdf(pages: List[str], password: Optional[str] = None) -> bytes:
    pdf = FPDF()
    pdf.set_font("Helvetica", size=11)
    for text in pages:
        pdf.add_page()
        if text:
            pdf.multi_cell(0, 6, text)
    if password:
        pdf.set_encryption(owner_password="owner-" + password, user_password=password)
    return bytes(pdf.output())


PHYSICS_PAGES = [
    "Chapter 1. Water boils at 100 degrees Celsius at sea level. At high altitude water boils at a lower temperature.",
    "Chapter 2. Photosynthesis converts sunlight, carbon dioxide and water into glucose and oxygen inside chloroplasts.",
    "Chapter 3. The speed of light in vacuum is about 299,792 kilometres per second.",
]
HISTORY_PAGES = ["The Battle of Hastings took place in 1066 between Normans and English forces."]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(db_dir=str(tmp_path / "chroma"), collection="test", top_k=3, min_relevance=0.2)


@pytest.fixture
def store(settings) -> VectorStore:
    return VectorStore(NomicPrefixedEmbeddings(FakeEmbeddings()), settings.db_dir, settings.collection)


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def pipeline(store, llm, settings) -> RAGPipeline:
    return RAGPipeline(store, llm, settings)


@pytest.fixture
def physics_pdf() -> bytes:
    return make_pdf(PHYSICS_PAGES)


@pytest.fixture
def history_pdf() -> bytes:
    return make_pdf(HISTORY_PAGES)
