"""nomic-embed-text v1.5 with its task prefixes.

nomic-embed-text was trained with a task instruction at the start of every
input: "search_document: " for passages that get stored and "search_query: "
for the user's question. Ollama does not add these, and the research pipeline
(rag_utils.create_embeddings) embedded raw text. The GPU backend
(gpu_backend.GPUEmbeddings) already used the prefixes; this wrapper does the
same for the Ollama embeddings, so questions and passages land in the
embedding space the model was trained on.

Both sides must use the same scheme. That is one reason the app keeps its own
collection instead of reusing the research databases.
"""
from __future__ import annotations

from typing import List

from langchain_core.embeddings import Embeddings

DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class NomicPrefixedEmbeddings(Embeddings):
    """Wraps any LangChain Embeddings object and adds the nomic task prefixes."""

    def __init__(self, base: Embeddings):
        self.base = base

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.base.embed_documents([DOCUMENT_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> List[float]:
        return self.base.embed_query(QUERY_PREFIX + text)
