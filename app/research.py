"""RAG Security Research Mode — orchestration layer.

This module adds NO new attack science. It is a thin orchestrator that wires the
existing, untouched research code into the Streamlit app:

    suffix_attack/jam_eval.py        JamEvaluator  — clean vs poisoned, verdict, cleanup
    suffix_attack/generator.py       SuffixAttackGenerator (GCG/suffix blocker)
    bbo_attack.py                    BBOBlockerGenerator  (BBO blocker)
    blocker_tool.py / injector.py    blocker inject/remove by tag

It preserves the exact experimental definitions from the Summer Training Report:

  * A query is a VALID TARGET only if the clean system answers it. Attacks get no
    credit for a query the clean system already failed.
  * A query is JAMMED iff the clean system answered it AND the poisoned system no
    longer conveys the correct answer (soft_fail or refusal), graded by LLM judge.
  * RETRIEVAL SUCCESS IS NOT JAMMING SUCCESS. Retrieval rank / in-context position
    are reported separately from the jam verdict. A blocker can be retrieved at
    rank 1 and the query still not be jammed.
  * Injected blockers are always removed, even on error.

Isolation (requirement 5): all injection happens in a SEPARATE experimental
collection that is a vector-level copy of the research MS-MARCO corpus. The
original research DB (db/chroma_db_msmarco) is never written to. On top of that,
every experiment snapshots the collection's id-set and restores it in a finally
block, so the experimental collection always returns to the exact clean corpus
regardless of what happens mid-run.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

# Load suffix_attack/jam_eval.py as a STANDALONE module (not via the suffix_attack
# package). The package __init__ eagerly imports the GCG generator, which needs
# torch + scikit-learn; jam_eval.py itself needs only chromadb (+ Ollama, lazily).
# This lets the core research mode — clean/poisoned evaluation with PRE-MADE
# blockers — run without the heavy ML stack. Fresh GCG generation imports the
# generator lazily (and only then needs torch/sklearn).
_JAM = None


def _jam():
    global _JAM
    if _JAM is None:
        path = ROOT / "suffix_attack" / "jam_eval.py"
        spec = importlib.util.spec_from_file_location("rag_sec_jam_eval", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _JAM = mod
    return _JAM


# The blocker metadata tag and refusal detector, taken from the research code.
BLOCKER_TAG = _jam().BLOCKER_TAG
looks_like_refusal = _jam().looks_like_refusal

# Attack methods exposed in the UI.
METHOD_GCG = "GCG"   # gradient-optimized adversarial suffix (suffix_attack/generator.py)
METHOD_BBO = "BBO"   # black-box blocker optimization (bbo_attack.py)

# GCG jam styles (suffix_attack/config.py). "nosuffix" is the report's control
# condition (on-topic prose, no suffix) that jammed the most — the headline of
# the negative result.
GCG_STYLES = ["distraction", "category", "passive", "nosuffix"]

COPY_BATCH = 2000  # vectors per batch when building the experimental collection


@dataclass
class DatasetQuery:
    query_id: str
    query: str
    gold_corpus_id: str = ""


@dataclass
class Passage:
    rank: int
    text: str
    distance: float
    is_blocker: bool = False


@dataclass
class ExperimentResult:
    """One clean-vs-poisoned run. Mirrors JamEvaluator.evaluate() plus run metadata."""
    query: str
    method: str
    style: str
    blocker_origin: str          # "pre-made (<file>)" or "freshly generated (...)"
    full_blocker: str
    # clean vs poisoned
    clean_answer: str
    poisoned_answer: str
    clean_refuses: bool
    poisoned_refuses: bool
    # retrieval (reported SEPARATELY from jamming)
    blocker_sim_rank: int        # 1-based similarity rank of the blocker; -1 = outside top-50
    blocker_ctx_pos: int         # 1-based position among chunks the LLM saw; -1 = not retrieved
    retrieved: bool
    num_context: int
    # jamming verdict
    outcome: str                 # correct | soft_fail | refusal | clean_refused
    valid_target: bool
    jammed: bool
    # run metadata
    k: int
    used_llm_judge: bool
    blockers_cleaned: int        # tagged blockers remaining after cleanup (should be 0)
    ids_restored: int            # stray docs removed by snapshot/restore (should be 0)
    elapsed_s: float
    timestamp: str

    @property
    def verdict_label(self) -> str:
        if not self.valid_target:
            return "Not a valid target — the clean system already failed this query"
        if self.jammed:
            return f"JAMMED ({self.outcome})"
        if self.retrieved:
            return f"NOT jammed ({self.outcome}) — blocker WAS retrieved but the answer survived"
        return f"NOT jammed ({self.outcome}) — blocker was not even retrieved"

    @property
    def retrieval_vs_jamming(self) -> str:
        r = "retrieved" if self.retrieved else "not retrieved"
        j = "jammed" if self.jammed else "not jammed"
        return f"Retrieval: {r} (sim-rank {self.blocker_sim_rank}, context pos {self.blocker_ctx_pos}) | Jamming: {j}"

    def to_row(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class ResearchService:
    """Stateful service: build once (loads models + experimental collection), reuse per run."""

    def __init__(
        self,
        root: Path = ROOT,
        source_db: str = "db/chroma_db_msmarco",
        source_collection: str = "my_collection",
        exp_db: str = "data/research_chroma",
        exp_collection: str = "msmarco_experiment",
        k: int = 5,
        export_dir: str = "evaluation/research_runs",
    ):
        from .model_backend import embedding_namespace, get_embedding_provider
        if get_embedding_provider() != "ollama":
            if source_db == "db/chroma_db_msmarco":
                source_db += "_" + embedding_namespace()
            if exp_db == "data/research_chroma":
                exp_db += "_" + embedding_namespace()
        self.root = Path(root)
        self.source_db = str(self.root / source_db)
        self.source_collection = source_collection
        self.exp_db = str(self.root / exp_db)
        self.exp_collection = exp_collection
        self.k = k
        self.export_dir = self.root / export_dir
        self._evaluator = None            # lazy JamEvaluator (loads Ollama models)
        self._blocker_index: Optional[Dict] = None

    # ── dataset ───────────────────────────────────────────────────────────────
    def dataset(self) -> List[DatasetQuery]:
        """The 18 valid attack targets from the report (msmarco_queries_valid18.txt + meta)."""
        valid_file = self.root / "msmarco_queries_valid18.txt"
        meta_file = self.root / "msmarco_query_meta.json"
        meta_by_q = {}
        if meta_file.exists():
            for m in json.loads(meta_file.read_text(encoding="utf-8")):
                meta_by_q[m["query"].strip().lower()] = m
        out = []
        for line in valid_file.read_text(encoding="utf-8").splitlines():
            q = line.strip()
            if not q:
                continue
            m = meta_by_q.get(q.lower(), {})
            out.append(DatasetQuery(query_id=m.get("query_id", ""), query=q, gold_corpus_id=m.get("gold_corpus_id", "")))
        return out

    # ── experimental collection (isolated copy of the research corpus) ─────────
    def prepare_demo_corpus(self, progress=None):
        """Embed the small shipped demo corpus on explicit request."""
        import chromadb
        from rag_utils import create_embeddings

        data = json.loads((self.root / "examples/research_corpus.json").read_text(encoding="utf-8"))
        client = chromadb.PersistentClient(path=self.source_db)
        col = client.get_or_create_collection(self.source_collection)
        if col.count() == len(data["ids"]):
            return
        embedding = create_embeddings()
        for start in range(0, len(data["ids"]), 16):
            docs = data["documents"][start:start + 16]
            col.upsert(ids=data["ids"][start:start + 16], documents=docs,
                       embeddings=embedding.embed_documents(docs))
            if progress:
                progress(min((start + len(docs)) / len(data["ids"]), 1.0), "Embedding demo passages")

    def _source_count(self):
        import chromadb
        c = chromadb.PersistentClient(path=self.source_db)
        return c.get_collection(self.source_collection).count()

    def experiment_status(self) -> Dict[str, Any]:
        import chromadb
        try:
            src = self._source_count()
        except Exception as exc:
            return {"ready": False, "error": f"research corpus not found: {exc}", "source": 0, "copied": 0}
        copied = 0
        try:
            c = chromadb.PersistentClient(path=self.exp_db)
            names = [col.name for col in c.list_collections()]
            if self.exp_collection in names:
                copied = c.get_collection(self.exp_collection).count()
        except Exception:
            copied = 0
        return {"ready": copied == src and src > 0, "source": src, "copied": copied}

    def ensure_experiment_collection(self, progress: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        """Copy the research corpus vectors into the isolated experimental collection, once.

        Embeddings are copied directly (no re-embedding), so retrieval in the
        experimental collection is identical to the research DB and the report.
        """
        import chromadb

        status = self.experiment_status()
        if status.get("error"):
            raise RuntimeError(status["error"])
        if status["ready"]:
            return status

        src_client = chromadb.PersistentClient(path=self.source_db)
        src = src_client.get_collection(self.source_collection)
        total = src.count()

        os.makedirs(self.exp_db, exist_ok=True)
        dst_client = chromadb.PersistentClient(path=self.exp_db)
        # Start clean so a half-finished earlier copy cannot leave a partial collection.
        if self.exp_collection in [c.name for c in dst_client.list_collections()]:
            dst_client.delete_collection(self.exp_collection)
        dst = dst_client.create_collection(self.exp_collection, metadata={"hnsw:space": "cosine"})

        done = 0
        for offset in range(0, total, COPY_BATCH):
            batch = src.get(limit=COPY_BATCH, offset=offset, include=["embeddings", "documents", "metadatas"])
            ids = batch["ids"]
            if not ids:
                break
            # The research corpus carries no metadata; chromadb rejects empty dicts,
            # so pass metadatas only when a batch actually has some.
            metas = batch.get("metadatas") or []
            has_meta = any(m for m in metas)
            dst.add(
                ids=ids,
                embeddings=batch["embeddings"],
                documents=batch["documents"],
                metadatas=([m or {"origin": "corpus"} for m in metas] if has_meta else None),
            )
            done += len(ids)
            if progress:
                progress(done / total, f"Copied {done:,}/{total:,} vectors")
        return {"ready": True, "source": total, "copied": done}

    # ── evaluator (reuses the research pipeline + prompt) ──────────────────────
    def evaluator(self, use_llm_judge: bool = True):
        JamEvaluator = _jam().JamEvaluator
        if self._evaluator is None or self._evaluator.use_llm_judge != use_llm_judge:
            self._evaluator = JamEvaluator(
                db_dir=self.exp_db,
                collection_name=self.exp_collection,
                k=self.k,
                use_llm_judge=use_llm_judge,
                soft_failure=True,
                verbose=False,
                backend="ollama",
            )
        return self._evaluator

    # ── clean pass (requirement 2) ─────────────────────────────────────────────
    def clean_pass(self, query: str) -> Dict[str, Any]:
        """Run the clean RAG pipeline; return its answer and the top-k retrieved passages."""
        from rag_utils import safe_display
        ev = self.evaluator()
        answer, _docs = ev._rag_answer(query)
        res = ev.collection.query(
            query_embeddings=[ev.embedding.embed_query(query)],
            n_results=self.k,
            include=["documents", "metadatas", "distances"],
        )
        passages = []
        for i, (doc, meta, dist) in enumerate(
            zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1
        ):
            passages.append(Passage(
                rank=i,
                text=safe_display(doc, repair=True)[:600],
                distance=round(float(dist), 4),
                is_blocker=(meta or {}).get("type") == BLOCKER_TAG,
            ))
        return {"query": query, "answer": answer, "refuses": looks_like_refusal(answer), "passages": passages}

    # ── blockers: pre-made (default) or fresh (reuse generators) ───────────────
    def _build_blocker_index(self) -> Dict:
        """Index every saved blocker that carries its text, keyed by (style, query) and query."""
        index: Dict[str, Dict[str, Dict]] = {"by_style_query": {}, "by_query": {}}
        res_dir = self.root / "suffix_attack_results"
        for path in glob.glob(str(res_dir / "batch_blockers_*.json")):
            name = os.path.basename(path)
            m = re.match(r"batch_blockers_(.+?)(?:_valid18)?(?:_\d{8}_\d{6})?\.json$", name)
            style = m.group(1) if m else "unknown"
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:
                continue
            for row in data.get("blockers", []):
                q, fb = row.get("query", "").strip(), row.get("full_blocker")
                if q and fb:
                    index["by_style_query"][(style, q.lower())] = {"full_blocker": fb, "file": name}
        # Per-query single-result files (query + full_blocker)
        for path in glob.glob(str(res_dir / "suffix_*.json")):
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:
                continue
            q, fb = str(data.get("query", "")).strip(), data.get("full_blocker")
            if q and fb:
                index["by_query"].setdefault(q.lower(), {"full_blocker": fb, "file": os.path.basename(path)})
        return index

    def available_prebuilt_styles(self, query: str) -> List[str]:
        if self._blocker_index is None:
            self._blocker_index = self._build_blocker_index()
        ql = query.strip().lower()
        styles = sorted({st for (st, q) in self._blocker_index["by_style_query"] if q == ql})
        if ql in self._blocker_index["by_query"]:
            styles.append("saved")
        return styles

    def get_prebuilt_blocker(self, query: str, style: str) -> Optional[Dict[str, str]]:
        if self._blocker_index is None:
            self._blocker_index = self._build_blocker_index()
        ql = query.strip().lower()
        hit = self._blocker_index["by_style_query"].get((style, ql))
        if not hit and style == "saved":
            hit = self._blocker_index["by_query"].get(ql)
        if not hit:
            return None
        return {"full_blocker": hit["full_blocker"], "origin": f"pre-made ({hit['file']})"}

    def generate_blocker(self, query: str, method: str, style: str = "distraction", quick: bool = True) -> Dict[str, str]:
        """Freshly generate a blocker by REUSING the research generators (no algorithm changes)."""
        if method == METHOD_GCG:
            try:
                from suffix_attack.config import SuffixAttackConfig
                from suffix_attack.generator import SuffixAttackGenerator
            except ImportError as exc:
                raise RuntimeError(
                    "Fresh GCG generation needs the ML stack (torch, transformers, scikit-learn), "
                    f"which is not installed in this environment ({exc}). Use a pre-made blocker instead, "
                    "or install those packages to generate GCG suffixes live."
                ) from exc
            # Styles saved in result files can be compound, e.g. "distraction_nosuffix".
            no_suffix = "nosuffix" in style
            base = style.replace("_nosuffix", "").replace("nosuffix", "")
            jam_style = base if base in ("distraction", "category", "passive", "instruction") else "passive"
            cfg = SuffixAttackConfig(jam_mode="R1", jam_style=jam_style)
            if no_suffix:
                cfg.suffix_length = 0
            gen = SuffixAttackGenerator(embedding_model=None, config=cfg)
            res = gen.generate(query)
            tag = f"GCG/{jam_style}" + ("/no-suffix" if no_suffix else "")
            return {"full_blocker": res["full_blocker"], "origin": f"freshly generated ({tag})"}

        if method == METHOD_BBO:
            # BBO optimizes against REAL retrieved context, so it injects candidate
            # documents into the collection during the search. We hand it the
            # experimental collection and guarantee cleanup via snapshot/restore
            # around the whole experiment (see run_experiment).
            try:
                from bbo_attack import BBOBlockerGenerator
            except ImportError as exc:
                raise RuntimeError(
                    f"Fresh BBO generation needs scikit-learn, which is not installed ({exc}). "
                    "Install it (pip install scikit-learn) or use a pre-made blocker."
                ) from exc
            from rag_utils import create_embeddings, create_llm
            import chromadb
            client = chromadb.PersistentClient(path=self.exp_db)
            collection = client.get_collection(self.exp_collection)
            gen = BBOBlockerGenerator(
                llm=create_llm(), embedding_model=create_embeddings(),
                collection=collection, top_k=self.k, mode="refusal", verbose=False,
                results_dir=str(self.export_dir / "bbo_generated"),  # keep out of research attack_results/
            )
            (self.export_dir / "bbo_generated").mkdir(parents=True, exist_ok=True)
            kw = dict(max_iterations=3, batch_size=4, population_size=4, max_restarts=0) if quick else {}
            res = gen.generate_blocker(query, **kw)
            return {"full_blocker": res["blocker"], "origin": f"freshly generated (BBO, {'quick' if quick else 'full'})"}

        raise ValueError(f"Unknown method: {method}")

    # ── guaranteed cleanup helpers ─────────────────────────────────────────────
    def _exp_collection_obj(self):
        import chromadb
        return chromadb.PersistentClient(path=self.exp_db).get_collection(self.exp_collection)

    def _snapshot_ids(self) -> set:
        try:
            return set(self._exp_collection_obj().get(include=[])["ids"])
        except Exception:
            return set()

    def sweep_cleanup(self) -> int:
        """Delete any blocker-tagged docs left in the experimental collection. Returns count."""
        try:
            col = self._exp_collection_obj()
            ids = col.get(where={"type": {"$eq": BLOCKER_TAG}}).get("ids", [])
            if ids:
                col.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0

    def _restore_to_snapshot(self, before: set) -> int:
        """Remove any doc id that did not exist before the run (belt-and-suspenders)."""
        if not before:
            return 0
        try:
            col = self._exp_collection_obj()
            now = set(col.get(include=[])["ids"])
            stray = list(now - before)
            if stray:
                col.delete(ids=stray)
            return len(stray)
        except Exception:
            return 0

    # ── the experiment (requirements 6-9) ──────────────────────────────────────
    def run_experiment(
        self, query: str, full_blocker: str, method: str, style: str, blocker_origin: str,
        use_llm_judge: bool = True, evaluator=None,
    ) -> ExperimentResult:
        """Inject the blocker into the isolated collection, run clean-vs-poisoned, always clean up."""
        ev = evaluator or self.evaluator(use_llm_judge=use_llm_judge)
        t0 = time.time()
        before = self._snapshot_ids()
        try:
            # JamEvaluator.evaluate() does clean pass -> inject -> poisoned pass ->
            # verdict, and removes the blocker in its own finally block.
            r = ev.evaluate(query, full_blocker)
        finally:
            cleaned = self.sweep_cleanup()
            restored = self._restore_to_snapshot(before)
        return ExperimentResult(
            query=query, method=method, style=style, blocker_origin=blocker_origin, full_blocker=full_blocker,
            clean_answer=r["clean_answer"], poisoned_answer=r["poisoned_answer"],
            clean_refuses=r["clean_refuses"], poisoned_refuses=r["poisoned_refuses"],
            blocker_sim_rank=r["blocker_sim_rank"], blocker_ctx_pos=r["blocker_ctx_pos"],
            retrieved=r["retrieved"], num_context=r["num_context"],
            outcome=r["outcome"], valid_target=r["valid_target"], jammed=r["jammed"],
            k=self.k, used_llm_judge=use_llm_judge, blockers_cleaned=cleaned, ids_restored=restored,
            elapsed_s=round(time.time() - t0, 2), timestamp=datetime.now().isoformat(),
        )

    # ── export (requirement 10) ────────────────────────────────────────────────
    def export(self, result: ExperimentResult) -> Dict[str, str]:
        self.export_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_q = re.sub(r"[^\w]+", "_", result.query)[:40].strip("_")
        json_path = self.export_dir / f"run_{safe_q}_{ts}.json"
        json_path.write_text(json.dumps(result.to_row(), indent=2, ensure_ascii=False), encoding="utf-8")

        csv_path = self.export_dir / "research_runs.csv"
        cols = ["timestamp", "query", "method", "style", "blocker_origin", "k", "used_llm_judge",
                "valid_target", "retrieved", "blocker_sim_rank", "blocker_ctx_pos", "outcome", "jammed",
                "blockers_cleaned", "ids_restored", "elapsed_s", "clean_answer", "poisoned_answer"]
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            if write_header:
                w.writeheader()
            row = result.to_row()
            row["clean_answer"] = row["clean_answer"][:300]
            row["poisoned_answer"] = row["poisoned_answer"][:300]
            w.writerow(row)
        return {"json": str(json_path), "csv": str(csv_path)}
