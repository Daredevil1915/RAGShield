"""
suffix_attack/evaluator.py
==========================
Batch evaluation of DASA attack across multiple queries.

Metrics computed per query:
  - Cosine Similarity : embed_ollama(full_blocker) vs embed_ollama(query)
  - Retrieval Rank    : what rank does the blocker get in ChromaDB?
  - Attack Success    : rank == 1? (blocker retrieved before real documents)
  - Elapsed Time      : seconds to generate the blocker

Usage:
  python -m suffix_attack.evaluator --queries queries.txt
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from .config import SuffixAttackConfig
from .generator import SuffixAttackGenerator
from .injector import BlockerInjector


class SuffixAttackEvaluator:
    """
    Runs DASA attack on a list of queries and reports RETRIEVAL metrics only.

    NOTE: this class measures whether the blocker gets RETRIEVED (rank), NOT
    whether it JAMS. Retrieval success != jamming success — a blocker can be
    rank 1 and still fail to jam because the other k-1 chunks answer the query.
    For the real attack objective (clean answers -> poisoned refuses), use
    `suffix_attack/jam_eval.py`.
    """

    def __init__(
        self,
        embedding_model,
        config: Optional[SuffixAttackConfig] = None,
        db_dir: str = "db/chroma_db",
        collection_name: str = "my_collection",
    ) -> None:
        self.config    = config or SuffixAttackConfig()
        self.generator = SuffixAttackGenerator(embedding_model, self.config)
        self.injector  = BlockerInjector(embedding_model, db_dir, collection_name, self.config)

    def evaluate(self, queries: List[str]) -> Dict[str, Any]:
        """
        Run attack on each query, inject blocker, measure rank, remove blocker.

        Parameters
        ----------
        queries : List of query strings to attack

        Returns
        -------
        Summary dict with per-query results and aggregate metrics
        """
        results  = []
        total_asr = 0

        print(f"\n{'='*62}")
        print(f"  DASA Evaluation  |  {len(queries)} queries")
        print(f"{'='*62}\n")

        for i, query in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] {query}")

            # Generate blocker
            result = self.generator.generate(query)

            # Inject into ChromaDB
            doc_id = self.injector.inject(result)

            # Measure retrieval rank
            rank = self.injector.get_retrieval_rank(query, doc_id)

            # Retrieval success: blocker retrieved at rank 1 (NOT jamming!)
            retrieved_rank1 = (rank == 1)
            if retrieved_rank1:
                total_asr += 1

            # Remove blocker (clean up)
            self.injector.remove(doc_id)

            row = {
                "query":           query,
                "full_blocker":    result["full_blocker"],
                "nomic_sim":       result["nomic_sim"],
                "rank":            rank,
                "retrieved_rank1": retrieved_rank1,
                "elapsed_s":       result["elapsed_s"],
                "suffix":          result["suffix"],
            }
            results.append(row)

            status = "RANK-1 ✓" if retrieved_rank1 else f"rank {rank}"
            print(f"  -> retrieval={status}  |  sim={result['nomic_sim']:.4f}  |  {result['elapsed_s']}s\n")

        # Aggregate  (RETRIEVAL metrics only — see class docstring)
        n = len(queries)
        rank1_pct = round(total_asr / n * 100, 1) if n else 0
        avg_sim   = round(sum(r["nomic_sim"] for r in results) / n, 4) if n else 0
        avg_rank  = round(sum(r["rank"] if r["rank"] > 0 else 20 for r in results) / n, 2) if n else 0
        avg_time  = round(sum(r["elapsed_s"] for r in results) / n, 2) if n else 0

        summary = {
            "total_queries":     n,
            "retrieved_rank1":   total_asr,
            "rank1_percent":     rank1_pct,
            "avg_similarity":    avg_sim,
            "avg_rank":          avg_rank,
            "avg_elapsed_s":     avg_time,
            "results":           results,
            "timestamp":         datetime.now().isoformat(),
            "config": {
                "suffix_length": self.config.suffix_length,
                "pgd_steps":     self.config.pgd_steps,
                "epsilon":       self.config.epsilon,
            },
        }

        print(f"\n{'='*62}")
        print(f"  SUMMARY  (RETRIEVAL ONLY — run jam_eval.py for jamming ASR)")
        print(f"  Rank-1 rate : {rank1_pct}%  ({total_asr}/{n} queries)")
        print(f"  Avg Sim     : {avg_sim}")
        print(f"  Avg Rank    : {avg_rank}")
        print(f"  Avg Time    : {avg_time}s")
        print(f"{'='*62}")

        self._save_summary(summary)
        return summary

    def _save_summary(self, summary: Dict[str, Any]) -> None:
        os.makedirs(self.config.results_dir, exist_ok=True)
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"eval_summary_{ts}.json"
        fpath = os.path.join(self.config.results_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n  [Summary saved -> {fpath}]")


# ------------------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from rag_utils import create_embeddings

    parser = argparse.ArgumentParser(description="DASA Evaluator")
    parser.add_argument("--queries", required=True, help="Path to queries text file (one per line)")
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--steps",   type=int,   default=20)
    parser.add_argument("--suffix-length", type=int, default=8)
    args = parser.parse_args()

    with open(args.queries, encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    emb = create_embeddings()
    cfg = SuffixAttackConfig(
        epsilon=args.epsilon,
        pgd_steps=args.steps,
        suffix_length=args.suffix_length,
    )
    evaluator = SuffixAttackEvaluator(emb, cfg)
    evaluator.evaluate(queries)
