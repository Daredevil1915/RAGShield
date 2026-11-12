"""ChromaDB wrapper: the only module that talks to the vector database.

Same stack as the research code (chromadb.PersistentClient + langchain_chroma),
with two changes:
  * cosine distance (hnsw:space=cosine), so a relevance score is just the cosine
    similarity in [-1, 1] and one threshold means the same thing for any query;
  * every chunk carries doc_id / source / page metadata, which enables
    per-document filtering, deletion and citations.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

ADD_BATCH_SIZE = 32  # embed in batches, as ingestion_pipeline.py does


class VectorStore:
    def __init__(self, embeddings: Embeddings, persist_dir: str, collection_name: str):
        client = chromadb.PersistentClient(
            path=persist_dir, settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.db = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.db._collection.count()

    def has_document(self, doc_id: str) -> bool:
        return bool(self.db.get(where={"doc_id": doc_id}, limit=1)["ids"])

    def add_chunks(self, chunks: List[Document], ids: List[str]) -> None:
        for i in range(0, len(chunks), ADD_BATCH_SIZE):
            self.db.add_documents(chunks[i : i + ADD_BATCH_SIZE], ids=ids[i : i + ADD_BATCH_SIZE])

    def delete_document(self, doc_id: str) -> int:
        ids = self.db.get(where={"doc_id": doc_id})["ids"]
        if ids:
            self.db.delete(ids=ids)
        return len(ids)

    def list_documents(self) -> List[Dict]:
        """One entry per uploaded document, built from chunk metadata.

        Reading all metadata is fine at demo scale (thousands of chunks). A
        larger system would keep a separate documents table.
        """
        docs: Dict[str, Dict] = {}
        for meta in self.db.get(include=["metadatas"])["metadatas"]:
            d = docs.setdefault(
                meta["doc_id"],
                {"doc_id": meta["doc_id"], "filename": meta.get("source", "?"), "chunks": 0, "pages": set()},
            )
            d["chunks"] += 1
            d["pages"].add(meta.get("page"))
        out = []
        for d in docs.values():
            d["pages_with_text"] = len(d.pop("pages"))
            out.append(d)
        return sorted(out, key=lambda d: d["filename"].lower())

    def search(self, query: str, k: int, doc_ids: Optional[List[str]] = None) -> List[Tuple[Document, float]]:
        """Top-k chunks by cosine similarity, best first, as (chunk, score) pairs."""
        flt = None
        if doc_ids:
            flt = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": list(doc_ids)}}
        return self.db.similarity_search_with_relevance_scores(query, k=k, filter=flt)
