"""
suffix_attack/injector.py
=========================
ChromaDB inject / remove / rank utilities for DASA blockers.
"""

from __future__ import annotations

import uuid
from typing import Dict, Any, List, Optional

import chromadb
from langchain_chroma import Chroma

from .config import SuffixAttackConfig


class BlockerInjector:
    """
    Injects DASA blocker documents into ChromaDB and measures retrieval rank.

    Parameters
    ----------
    embedding_model : OllamaEmbeddings
        Same embedder used by the RAG pipeline.
    db_dir : str
        Path to ChromaDB persistent storage.
    collection_name : str
        Name of the ChromaDB collection.
    config : SuffixAttackConfig
        Config for verbose flag.
    """

    def __init__(
        self,
        embedding_model,
        db_dir:          str = "db/chroma_db",
        collection_name: str = "my_collection",
        config:          Optional[SuffixAttackConfig] = None,
    ) -> None:
        self.embedding = embedding_model
        self.config    = config or SuffixAttackConfig()

        client   = chromadb.PersistentClient(path=db_dir)
        self.db  = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_model,
        )

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    def inject(self, result: Dict[str, Any]) -> str:
        """
        Inject the full blocker document from a generator result dict.

        Parameters
        ----------
        result : Dict returned by SuffixAttackGenerator.generate()

        Returns
        -------
        doc_id : str — unique ID of the injected document (use to remove later)
        """
        full_blocker = result["full_blocker"]
        doc_id       = str(uuid.uuid4())

        self.db.add_texts(
            texts=[full_blocker],
            ids=[doc_id],
            metadatas=[{
                "source":        "dasa_attack",
                "query":         result.get("query", ""),
                "attack_type":   "suffix",
                "suffix":        result.get("suffix", ""),
            }],
        )

        if self.config.verbose:
            print(f"\n[Injector] Injected blocker (id={doc_id[:8]}...)")
            print(f"           Full blocker: {full_blocker[:100]}...")

        return doc_id

    def remove(self, doc_id: str) -> None:
        """Remove a previously injected blocker document."""
        self.db._collection.delete(ids=[doc_id])
        if self.config.verbose:
            print(f"\n[Injector] Removed blocker (id={doc_id[:8]}...)")

    def get_retrieval_rank(self, query: str, doc_id: str, top_k: int = 20) -> int:
        """
        Query ChromaDB and return the rank of the injected blocker.

        Returns
        -------
        rank : int
            1-based rank. Returns -1 if doc_id not found in top_k.
        """
        results = self.db._collection.query(
            query_embeddings=[self.embedding.embed_query(query)],
            n_results=top_k,
            include=["documents", "metadatas"],
        )

        ids = results.get("ids", [[]])[0]
        try:
            return ids.index(doc_id) + 1   # 1-based
        except ValueError:
            return -1   # not in top_k

    def get_top_k_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Return top-k retrieved documents for inspection."""
        results = self.db._collection.query(
            query_embeddings=[self.embedding.embed_query(query)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        docs = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            docs.append({
                "rank":     i + 1,
                "document": doc[:150],
                "source":   meta.get("source", "unknown"),
                "distance": round(dist, 4),
            })
        return docs
