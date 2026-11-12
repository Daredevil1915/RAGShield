"""End-to-end evaluation of the PDF RAG app against the real models.

Indexes a PDF into a throwaway ChromaDB folder (your app database and the
research databases are not touched), runs a golden question set through the
exact pipeline the API uses, and reports:

  retrieval hit rate   an expected page is among the retrieved chunks (answerable questions)
  answer accuracy      answered, and the answer contains an expected keyword
  citation accuracy    a cited page is one of the expected pages
  false refusal rate   answerable questions that came back "not found"
  refusal accuracy     unanswerable questions that correctly came back "not found"
  latency              median and max seconds per question

Usage (from the RAG-TEST folder, with Ollama running):
    python -m evaluation.run_eval                          # full evaluation
    python -m evaluation.run_eval --retrieval-only         # scores only, no LLM: for threshold calibration
    python -m evaluation.run_eval --min-relevance 0.5      # try a different threshold
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.services import build_services, index_pdf  # noqa: E402

HERE = Path(__file__).resolve().parent


def keyword_hit(answer: str, groups) -> bool:
    """Every group must match; a group matches if the answer contains any of its alternatives."""
    a = answer.lower()
    return all(any(alt.lower() in a for alt in group) for group in groups)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=str(ROOT / "usenixsecurity25-shafran.pdf"))
    ap.add_argument("--golden", default=str(HERE / "golden_set.json"))
    ap.add_argument("--min-relevance", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--chunk-overlap", type=int, default=None)
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    tmp_db = tempfile.mkdtemp(prefix="rag_eval_")
    settings = replace(get_settings(), db_dir=tmp_db, collection="eval")
    if args.min_relevance is not None:
        settings = replace(settings, min_relevance=args.min_relevance)
    if args.top_k is not None:
        settings = replace(settings, top_k=args.top_k)
    if args.chunk_size is not None:
        settings = replace(settings, chunk_size=args.chunk_size)
    if args.chunk_overlap is not None:
        settings = replace(settings, chunk_overlap=args.chunk_overlap)

    svc = build_services(settings)
    t0 = time.perf_counter()
    info = index_pdf(svc.store, settings, Path(args.pdf).name, Path(args.pdf).read_bytes())
    print(f"Indexed {info['filename']}: {info['num_pages']} pages -> {info['chunks']} chunks "
          f"in {time.perf_counter() - t0:.1f}s  (top_k={settings.top_k}, min_relevance={settings.min_relevance})\n")

    cases = (
        [{**c, "kind": "answerable"} for c in golden["answerable"]]
        + [{**c, "kind": "follow_up"} for c in golden.get("follow_ups", [])]
        + [{**c, "kind": "unanswerable"} for c in golden["unanswerable"]]
    )

    if args.retrieval_only:
        print(f"{'id':<4} {'kind':<13} {'top score':>9}  pages   question")
        tops = {"answerable": [], "unanswerable": []}
        for c in cases:
            hits = svc.pipeline.retrieve(c["question"])
            top = hits[0].score if hits else 0.0
            if c["kind"] != "follow_up":  # follow-ups need rewriting first, so their raw score means little
                tops[c["kind"]].append(top)
            print(f"{c['id']:<4} {c['kind']:<13} {top:9.3f}  {[h.page for h in hits]}  {c['question'][:60]}")
        print(f"\nlowest answerable top score : {min(tops['answerable']):.3f}")
        print(f"highest unanswerable score  : {max(tops['unanswerable']):.3f}")
        print("A good threshold sits between these two numbers.")
        return

    rows = []
    for c in cases:
        t = time.perf_counter()
        res = svc.pipeline.answer(c["question"], history=c.get("history"))
        secs = time.perf_counter() - t
        retrieved_pages = sorted({s.page for s in res.retrieved})
        cited_pages = sorted({s.page for s in res.sources}) if res.status == "answered" else []
        row = {
            "id": c["id"], "kind": c["kind"], "question": c["question"], "status": res.status,
            "standalone_question": res.standalone_question, "answer": res.answer,
            "top_score": max((s.score for s in res.retrieved), default=0.0),
            "retrieved_pages": retrieved_pages, "cited_pages": cited_pages, "cited": res.cited,
            "grounding": res.grounding,
            "seconds": round(secs, 2),
        }
        if c["kind"] == "unanswerable":
            row["correct"] = res.status == "not_found"
        else:
            row["retrieval_hit"] = bool(set(retrieved_pages) & set(c["pages"]))
            row["answer_correct"] = res.status == "answered" and keyword_hit(res.answer, c["keywords"])
            row["citation_correct"] = bool(set(cited_pages) & set(c["pages"]))
            row["correct"] = row["answer_correct"]
        rows.append(row)
        mark = "PASS" if row["correct"] else "FAIL"
        print(f"[{mark}] {c['id']:<4} {res.status:<10} {secs:5.1f}s  score={row['top_score']:.2f}  {c['question'][:55]}")
        print(f"        -> {res.answer[:160].replace(chr(10), ' ')}")
        if c["kind"] == "follow_up":
            print(f"        rewritten as: {res.standalone_question}")

    ans = [r for r in rows if r["kind"] != "unanswerable"]
    una = [r for r in rows if r["kind"] == "unanswerable"]
    pct = lambda xs: 100.0 * sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    summary = {
        "retrieval_hit_rate": pct([r["retrieval_hit"] for r in ans]),
        "answer_accuracy": pct([r["answer_correct"] for r in ans]),
        "citation_accuracy": pct([r["citation_correct"] for r in ans]),
        "false_refusal_rate": pct([r["status"] == "not_found" for r in ans]),
        "refusal_accuracy": pct([r["correct"] for r in una]),
        "median_latency_s": round(statistics.median(r["seconds"] for r in rows), 2),
        "max_latency_s": round(max(r["seconds"] for r in rows), 2),
        "n_answerable": len(ans), "n_unanswerable": len(una),
        "settings": {"top_k": settings.top_k, "min_relevance": settings.min_relevance,
                     "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap},
    }
    print("\n=== Summary ===")
    for k, v in summary.items():
        if k != "settings":
            print(f"  {k:<20} {v:.1f}{'%' if k.endswith(('rate', 'accuracy')) else ''}" if isinstance(v, float) else f"  {k:<20} {v}")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
