"""
suffix_attack/perplexity_check.py
==================================
Perplexity-based detectability check for blocker documents.

Mirrors Shafran et al. (USENIX Sec'25) Section 8.1: compute perplexity of
generated blockers vs. clean corpus documents using a causal LM, report the
separation (mean/median, and detection rate at a threshold set from clean
docs). This produces the honest "limitations" evidence for the paper: the
jamming payload (natural prose) should look normal, but the GCG suffix
(random-token search) is expected to be high-perplexity and detectable.

Uses GPT-2 as the perplexity model (standard choice in this literature,
e.g. Alon & Kamfonas 2023, cited by Shafran §8.1). Runs on CPU — a handful
of short forward passes, not sustained load.

Usage:
    python -m suffix_attack.perplexity_check --batch suffix_attack_results/batch_blockers_distraction_*.json
    python -m suffix_attack.perplexity_check --result suffix_attack_results/suffix_....json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from typing import Dict, List

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ------------------------------------------------------------------------------
# Perplexity model (GPT-2, CPU, loaded once)
# ------------------------------------------------------------------------------
class PerplexityScorer:
    def __init__(self, model_name: str = "gpt2"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"[ppl] Loading {model_name} (perplexity scorer)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        print("[ppl] Ready.")

    @torch.no_grad()
    def perplexity(self, text: str) -> float:
        """Standard causal-LM perplexity: exp(mean token cross-entropy)."""
        if not text or not text.strip():
            return float("nan")
        ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).input_ids
        if ids.shape[1] < 2:
            return float("nan")
        out = self.model(ids, labels=ids)
        return float(torch.exp(out.loss).item())


# ------------------------------------------------------------------------------
# Clean-document sample (real KB chunks, for the "trusted" baseline)
# ------------------------------------------------------------------------------
def sample_clean_documents(n: int = 60, db_dir: str = "db/chroma_db",
                           collection: str = "my_collection") -> List[str]:
    import chromadb
    client = chromadb.PersistentClient(path=db_dir)
    col = client.get_collection(collection)
    total = col.count()
    if total == 0:
        return []
    # Pull a random-ish slice: chromadb .get() has no native random sample,
    # so fetch a batch of ids via offset sampling.
    k = min(n, total)
    offsets = sorted(random.sample(range(total), k))
    docs: List[str] = []
    # simplest robust approach: get() supports limit/offset
    for off in offsets:
        r = col.get(limit=1, offset=off, include=["documents"])
        if r["documents"]:
            docs.append(r["documents"][0])
    return docs


# ------------------------------------------------------------------------------
# Blocker loading
# ------------------------------------------------------------------------------
def load_blockers_from_batch(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("blockers", data) if isinstance(data, dict) else data
    return [{"query": r["query"], "full_blocker": r["full_blocker"]} for r in rows]


def load_blocker_from_result(path: str) -> Dict[str, str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {"query": data["query"], "full_blocker": data["full_blocker"],
            "base_blocker": data.get("base_blocker"), "suffix": data.get("suffix")}


# ------------------------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------------------------
def run(blockers: List[Dict[str, str]], n_clean: int = 60) -> Dict:
    scorer = PerplexityScorer()

    print(f"[ppl] Sampling {n_clean} clean KB documents for the trusted baseline...")
    clean_docs = sample_clean_documents(n_clean)
    clean_ppls = [scorer.perplexity(d) for d in clean_docs]
    clean_ppls = [p for p in clean_ppls if p == p]  # drop NaN

    print(f"[ppl] Scoring {len(blockers)} blocker(s)...")
    rows = []
    for b in blockers:
        full = b["full_blocker"]
        ppl_full = scorer.perplexity(full)
        row = {"query": b["query"], "ppl_full_blocker": round(ppl_full, 2)}
        if b.get("base_blocker"):
            row["ppl_base_only"] = round(scorer.perplexity(b["base_blocker"]), 2)
        if b.get("suffix"):
            row["ppl_suffix_only"] = round(scorer.perplexity(b["suffix"]), 2)
        rows.append(row)
        print(f"  {b['query'][:50]:50s}  full_ppl={row['ppl_full_blocker']:.1f}")

    def stats(xs):
        xs = sorted(xs)
        n = len(xs)
        return {
            "n": n,
            "mean": round(sum(xs) / n, 2) if n else None,
            "median": round(xs[n // 2], 2) if n else None,
            "min": round(xs[0], 2) if n else None,
            "max": round(xs[-1], 2) if n else None,
        }

    clean_stats = stats(clean_ppls)
    blocker_ppls = [r["ppl_full_blocker"] for r in rows]
    blocker_stats = stats(blocker_ppls)

    # Detection rate: threshold = 95th percentile of CLEAN docs (a simple,
    # standard perplexity-filter design point). What fraction of blockers
    # exceed it (i.e. would be flagged as adversarial)?
    threshold = None
    detect_rate = None
    if clean_ppls:
        sorted_clean = sorted(clean_ppls)
        idx = min(int(0.95 * len(sorted_clean)), len(sorted_clean) - 1)
        threshold = sorted_clean[idx]
        flagged = sum(1 for p in blocker_ppls if p > threshold)
        detect_rate = round(flagged / len(blocker_ppls) * 100, 1) if blocker_ppls else None

    summary = {
        "clean_docs": clean_stats,
        "blockers": blocker_stats,
        "detection_threshold_p95_clean": round(threshold, 2) if threshold else None,
        "detection_rate_pct": detect_rate,
        "per_blocker": rows,
        "clean_ppls_raw": [round(p, 2) for p in clean_ppls],
        "blocker_ppls_raw": [round(p, 2) for p in blocker_ppls],
    }
    return summary


def print_summary(summary: Dict) -> None:
    c, b = summary["clean_docs"], summary["blockers"]
    print(f"\n{'='*64}")
    print("  PERPLEXITY SUMMARY  (mirrors Shafran et al. §8.1)")
    print(f"{'='*64}")
    print(f"  Clean KB docs   (n={c['n']:3d}): mean={c['mean']:.1f}  median={c['median']:.1f}  "
          f"range=[{c['min']:.1f}, {c['max']:.1f}]")
    print(f"  Blockers        (n={b['n']:3d}): mean={b['mean']:.1f}  median={b['median']:.1f}  "
          f"range=[{b['min']:.1f}, {b['max']:.1f}]")
    if summary["detection_threshold_p95_clean"] is not None:
        print(f"\n  Perplexity-filter threshold (p95 of clean) = {summary['detection_threshold_p95_clean']:.1f}")
        print(f"  Blockers flagged as adversarial            = {summary['detection_rate_pct']}%")
    print(f"{'='*64}")


def save(summary: Dict, out_dir: str = "suffix_attack_results") -> str:
    os.makedirs(out_dir, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"perplexity_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  [saved -> {path}]")
    return path


def plot(summary: Dict, out_path: str = "perplexity_comparison.png") -> None:
    """Histogram figure in the same visual style as plot_ablation.py."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean = summary["clean_ppls_raw"]
    blockers = summary["blocker_ppls_raw"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    bins = 30
    ax.hist(clean, bins=bins, alpha=0.65, color="#4fc3f7", label=f"Clean KB docs (n={len(clean)})")
    ax.hist(blockers, bins=bins, alpha=0.65, color="#ff6b6b", label=f"Blockers (n={len(blockers)})")

    if summary.get("detection_threshold_p95_clean") is not None:
        ax.axvline(summary["detection_threshold_p95_clean"], color="#44ff88",
                   linewidth=1.6, linestyle="--")
        ax.text(summary["detection_threshold_p95_clean"], ax.get_ylim()[1] * 0.9,
               "  p95 clean\n  (filter threshold)", color="#44ff88", fontsize=8.5, va="top")

    ax.set_xlabel("Perplexity (GPT-2)", color="#cccccc", fontsize=12, labelpad=10)
    ax.set_ylabel("Count", color="#cccccc", fontsize=12, labelpad=10)
    ax.set_title("Blocker vs. Clean-Document Perplexity\n(lower = more natural-looking)",
                color="#ffffff", fontsize=12, fontweight="bold", pad=14)
    ax.tick_params(colors="#aaaaaa", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.grid(True, linestyle="--", alpha=0.2, color="#aaaaaa")
    ax.legend(loc="upper right", fontsize=9, facecolor="#1a1d27",
             edgecolor="#444455", labelcolor="#cccccc")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    print(f"  [figure -> {out_path}]")


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Perplexity-based detectability check for blockers")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--batch", help="Path to a batch_blockers_*.json (or glob)")
    src.add_argument("--result", help="Path to a single generator result JSON")
    parser.add_argument("--n-clean", type=int, default=60, help="Number of clean KB docs to sample")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.batch:
        paths = glob.glob(args.batch) if any(c in args.batch for c in "*?[") else [args.batch]
        blockers: List[Dict[str, str]] = []
        for p in paths:
            blockers.extend(load_blockers_from_batch(p))
    else:
        blockers = [load_blocker_from_result(args.result)]

    summary = run(blockers, n_clean=args.n_clean)
    print_summary(summary)
    save(summary)
    if not args.no_plot:
        plot(summary)


if __name__ == "__main__":
    main()
