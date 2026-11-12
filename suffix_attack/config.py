"""
suffix_attack/config.py
=======================
Central configuration for the Dual-Objective Adversarial Suffix Attack (DASA).
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field


def _auto_device() -> str:
    """Auto-detect GPU. Falls back to CPU silently."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _auto_steps() -> int:
    """500 steps on GPU (GCG converges fast), 30 on CPU (quick verify)."""
    return 500 if torch.cuda.is_available() else 30


def _auto_gcg_batch() -> int:
    """Candidate swaps verified per step.

    Every candidate is a forward pass through the nomic surrogate. 256 is
    trivial on an L40S and painful on a laptop CPU, so the CPU path verifies
    32 -- still a real verified-descent step, just a narrower search.
    """
    return 256 if torch.cuda.is_available() else 32


def _auto_gcg_topk() -> int:
    """Top-k candidate tokens per slot taken from the gradient."""
    return 256 if torch.cuda.is_available() else 64


@dataclass
class SuffixAttackConfig:
    """
    All hyperparameters for DASA in one place.

    GPU vs CPU behaviour (auto-detected)
    -------------------------------------
    On GPU : 500 GCG steps — GCG verifies every swap, converges fast.
             Each step takes ~0.08s on a V100/A100 = ~40s total per query.
    On CPU : 30  steps — fast enough to verify flow end-to-end.
    """

    # Surrogate model
    model_name: str = "facebook/bart-base"
    device: str     = field(default_factory=_auto_device)

    # Suffix length: 15 tokens is the sweet spot (ablation study validated)
    # < 6  -> too few degrees of freedom
    # > 20 -> diminishing returns, document looks unnatural
    suffix_length: int = 15

    # Suffix optimiser
    #   "gcg"     : Greedy Coordinate Gradient (Zou et al. 2023) — gradient proposes
    #               top-k candidate swaps across ALL slots, then a batch is VERIFIED
    #               by true loss and the best is kept. Monotonic descent guaranteed.
    #   "hotflip" : Legacy single-random-slot HotFlip (kept for ablation only).
    optimizer: str = "gcg"
    # 256/256 on GPU, 32/64 on CPU -- see _auto_gcg_batch/_auto_gcg_topk
    gcg_batch: int = field(default_factory=_auto_gcg_batch)
    gcg_topk:  int = field(default_factory=_auto_gcg_topk)

    # Optimisation steps
    pgd_steps: int = field(default_factory=_auto_steps)  # shared loop counter

    # BART decoding (used by surrogate helpers, not by PGD itself)
    num_beams:      int = 4
    max_new_tokens: int = 20

    # -----------------------------------------------------------------
    # Jamming configuration (the part that actually blocks the answer)
    # -----------------------------------------------------------------
    # The suffix handles RETRIEVAL. These control the base blocker that
    # handles JAMMING — i.e. what the LLM sees and must be forced to refuse.
    #
    # jam_mode  : which refusal to induce (Shafran et al., USENIX Sec '25, §5.1)
    #   "R1" insufficient info | "R2" safety | "R3" correctness
    # jam_style : how hard the jammer pushes
    #   "instruction" active prompt-injection (strong, survives 9-vs-1 dilution)
    #   "passive"     weak declarative statement (kept only for ablation)
    jam_mode:  str = "R1"
    # jam_style:
    #   "passive"     original core idea: fixed "context has no info" statement
    #                 (R1/R2/R3/R4 text), never the query itself. (default)
    #   "distraction" on-topic, non-answering, uncertainty-saturated passage that
    #                 hijacks attention (contextual distraction + entrainment).
    #                 NOTE: contains the query's own topic words, so part of its
    #                 retrievability is attributable to the base text itself, not
    #                 the suffix (measured base-alone similarity ~0.75).
    #   "category"    query routed (via embedding similarity to anchor phrases,
    #                 never literal query words) to one of several jamming
    #                 templates (practical / factual / sensitive / entertainment
    #                 / generic). Combines "distraction"-level relevance with
    #                 "passive"-level query-independence: the template text
    #                 never contains the query itself, only the *choice* of
    #                 template depends on the query, indirectly.
    #   "instruction" active prompt injection (ablation).
    jam_style: str = "passive"

    # Minimum query↔anchor cosine similarity required to route to a specific
    # category in jam_style="category"; below this, falls back to "generic".
    jam_category_threshold: float = 0.45

    # Text inserted between the base blocker and the GCG suffix, both during
    # optimisation (baked into the frozen "base" the optimiser scores) and in
    # the final injected string -- so the two stay consistent.
    #
    # Default " " reproduces the original bare tail. The GCG suffix is
    # necessarily high-perplexity token salad (see perplexity_check.py); glued
    # on raw, it reads as an obvious tonal break that may cue the LLM the whole
    # chunk is low-quality/untrustworthy and worth discounting. Framing it as
    # something an LLM already expects to see as non-prose -- a reference code,
    # document ID, citation key -- may let it pass as normal metadata instead.
    # e.g. suffix_frame = "\n\nReference ID: "
    suffix_frame: str = " "

    # Output
    results_dir: str = "suffix_attack_results"
    verbose: bool    = True

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def summary(self) -> str:
        gpu_tag = "GPU ✓" if self.device == "cuda" else "CPU"
        return (
            f"SuffixAttackConfig("
            f"model={self.model_name!r}, "
            f"device={self.device!r} [{gpu_tag}], "
            f"optimizer={self.optimizer!r}, "
            f"suffix_length={self.suffix_length}, "
            f"steps={self.pgd_steps}, "
            f"gcg_batch={self.gcg_batch}, "
            f"gcg_topk={self.gcg_topk}"
            f")"
        )
