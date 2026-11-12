"""
suffix_attack/optimizer.py
==========================
Core PGD suffix optimisation loop.

Algorithm
---------
Given:
  base_text     -- fixed jamming sentence (base blocker, not optimised)
  suffix_ids    -- initial suffix token IDs (randomly sampled from BART vocab)
  query_embed   -- target embedding (Ollama/nomic, fixed, no gradient)

The loop runs pgd_steps times:

  1. Get base embeddings (frozen -- no gradient)
  2. Apply accumulated delta to suffix embeddings
  3. Concatenate base + perturbed suffix embeddings
  4. Run Nomic encoder -> hidden states H
  5. Mean pool H -> h_pooled (1, 768)
  6. Loss = 1 - cosine_sim(h_pooled, query_embed)
  7. Backpropagate -> gradient flows only through suffix embeddings
  8. PGD update: delta -= alpha * sign(grad)
  9. Clip delta to [-epsilon, +epsilon]
  10. Project updated embeddings to nearest clean vocab tokens
  11. Track BEST projected suffix (lowest loss seen so far)

Why track best and not just use the final step?
  PGD in continuous space may oscillate near the boundary.
  The last step is not guaranteed to be the best.
  We always return the suffix that achieved the lowest loss.

Returns best suffix token IDs, decoded text, and per-step loss history.

Key insight:
  The WHOLE document (base + suffix) is embedded together.
  Base is frozen. Gradient only updates suffix via delta.
  BART self-attention means suffix words influence base representation
  and vice versa -- richer than embedding suffix alone.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F

from .config import SuffixAttackConfig
from .surrogate import NomicSurrogate


class SuffixOptimizer:
    """
    PGD-based adversarial suffix optimiser.

    Runs on whatever device the surrogate model is on (CPU or CUDA).
    All tensors are created on self.device automatically.
    """

    def __init__(self, surrogate: NomicSurrogate, config: SuffixAttackConfig) -> None:
        self.surrogate = surrogate
        self.config    = config
        self.device    = config.torch_device

    def optimize(self, base_text, suffix_ids, query_embed):
        """Dispatch to GCG (default, recommended) or the legacy HotFlip search."""
        method = getattr(self.config, "optimizer", "gcg")
        if method == "gcg":
            return self._optimize_gcg(base_text, suffix_ids, query_embed)
        return self._optimize_hotflip(base_text, suffix_ids, query_embed)

    # ------------------------------------------------------------------
    # GCG — Greedy Coordinate Gradient (Zou et al. 2023)
    # ------------------------------------------------------------------
    def _optimize_gcg(
        self,
        base_text:   str,
        suffix_ids:  torch.Tensor,
        query_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, str, List[dict]]:
        """
        Gradient proposes top-k candidate swaps across ALL slots; a batch of
        one-token swaps is then VERIFIED by true loss and the best is kept.

        vs HotFlip: HotFlip commits to the gradient's first guess for one random
        slot (no verification -> oscillation). GCG measures before committing ->
        monotonic descent -> lower final loss -> higher query similarity.
        """
        cfg  = self.config
        emb  = self.surrogate.get_embedding_layer()

        base_ids     = self.surrogate.tokenize(base_text)
        base_embeds  = emb(base_ids).detach()                                  # (1, Lb, 768)
        vocab_tensor = torch.tensor(self.surrogate._clean_vocab_ids, device=self.device)
        vocab_embeds = emb(vocab_tensor).detach()                              # (V, 768)

        L = int(suffix_ids.shape[1])
        B = int(getattr(cfg, "gcg_batch", 256))
        K = min(int(getattr(cfg, "gcg_topk", 256)), int(vocab_tensor.numel()))

        current_ids = suffix_ids.clone()
        best_loss   = float("inf")
        best_ids    = current_ids.clone()
        history: List[dict] = []
        print_every = 1 if self.device.type == "cpu" else 25

        for step in range(cfg.pgd_steps):
            # 1) gradient of loss wrt the current suffix embeddings
            suffix_embeds = emb(current_ids).detach().requires_grad_(True)     # (1, L, 768)
            full = torch.cat([base_embeds, suffix_embeds], dim=1)
            H = self.surrogate.encode_with_embeds(full)
            h = F.normalize(H.mean(dim=1), dim=-1)
            loss = 1.0 - (h * query_embed).sum(dim=-1).squeeze()
            loss.backward()
            grad = suffix_embeds.grad[0]                                       # (L, 768)

            with torch.no_grad():
                # 2) gradient -> top-K candidate words per slot
                #    delta_L ~ grad . (v_w - e_p); minimise => maximise (-grad . v_w)
                token_scores = -(grad @ vocab_embeds.T)                        # (L, V) higher=better
                topk_vocab   = token_scores.topk(K, dim=1).indices            # (L, K) -> vocab idx

                # 3) sample B one-token swaps (random slot + random top-K word)
                pos    = torch.randint(0, L, (B,), device=self.device)
                choice = torch.randint(0, K, (B,), device=self.device)
                new_tok = vocab_tensor[topk_vocab[pos, choice]]               # (B,)
                cand = current_ids.repeat(B, 1)                               # (B, L)
                cand[torch.arange(B, device=self.device), pos] = new_tok

                # 4) VERIFY: true loss of every candidate, batched
                cand_embeds = emb(cand)                                        # (B, L, 768)
                base_b = base_embeds.expand(B, -1, -1)                         # (B, Lb, 768)
                full_b = torch.cat([base_b, cand_embeds], dim=1)
                H_b = self.surrogate.encode_with_embeds(full_b)
                h_b = F.normalize(H_b.mean(dim=1), dim=-1)
                losses = 1.0 - (h_b * query_embed).sum(dim=-1)                # (B,)

                # 5) keep the best candidate
                b_star = int(torch.argmin(losses).item())
                current_ids = cand[b_star:b_star + 1].clone()
                cur_loss = float(losses[b_star].item())

            sim_val = 1.0 - cur_loss
            history.append({"step": step + 1, "loss": round(cur_loss, 6), "sim": round(sim_val, 4)})
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_ids  = current_ids.clone()

            if cfg.verbose and (step + 1) % print_every == 0:
                txt = self.surrogate.decode_suffix_ids(current_ids)
                print(f"    [GCG] step {step + 1:04d}/{cfg.pgd_steps}  "
                      f"loss={cur_loss:.4f}  sim={sim_val:.4f}  best={1 - best_loss:.4f}")
                print(f"          suffix: {txt}")

        return best_ids, self.surrogate.decode_suffix_ids(best_ids), history

    # ------------------------------------------------------------------
    # Legacy HotFlip (single random slot, no verification) — kept for ablation
    # ------------------------------------------------------------------
    def _optimize_hotflip(
        self,
        base_text:    str,
        suffix_ids:   torch.Tensor,
        query_embed:  torch.Tensor,
    ) -> Tuple[torch.Tensor, str, List[dict]]:
        """
        Run PGD to find the best adversarial suffix.

        Parameters
        ----------
        base_text    : Fixed jamming sentence. Never modified.
        suffix_ids   : Initial suffix token IDs, shape (1, suffix_len).
                       Randomly sampled from BART vocabulary.
        query_embed  : Target embedding, shape (1, 768).
                       This is from Ollama (nomic space) — NOT BART.
                       It is fixed. No gradient flows through it.

        Returns
        -------
        best_suffix_ids  : (1, suffix_len) tensor of best found token IDs
        best_suffix_text : Decoded English text for best suffix
        history          : List of dicts with step, loss, bart_sim
        """
        import random
        emb_layer = self.surrogate.get_embedding_layer()

        # ------------------------------------------------------------------
        # Base embeddings: frozen.
        # ------------------------------------------------------------------
        base_ids    = self.surrogate.tokenize(base_text)
        base_embeds = emb_layer(base_ids).detach()          # (1, L_base, 768)

        # Pre-compute vocab embeddings for HotFlip projection
        vocab_tensor = torch.tensor(self.surrogate._clean_vocab_ids, device=self.device)
        vocab_embeds = emb_layer(vocab_tensor).detach()     # (vocab_size, 768)

        history   = []
        best_loss = float("inf")
        best_ids  = suffix_ids.clone()
        current_ids = suffix_ids.clone()

        print_every = 1 if self.device.type == "cpu" else 50

        if self.config.verbose:
            print()

        for step in range(self.config.pgd_steps):

            # Get embeddings for current discrete tokens and attach gradient
            suffix_embeds = emb_layer(current_ids).detach().requires_grad_(True)

            # Full document
            full_embeds = torch.cat([base_embeds, suffix_embeds], dim=1)

            # Nomic encoder forward pass
            H = self.surrogate.encode_with_embeds(full_embeds)
            h_pooled = F.normalize(H.mean(dim=1), dim=-1)

            # Cosine similarity vs query embedding
            cos_sim = (h_pooled * query_embed).sum(dim=-1).squeeze()

            # Loss to minimise
            loss = 1.0 - cos_sim
            loss.backward()

            loss_val = float(loss.item())
            sim_val  = float(cos_sim.item())

            history.append({
                "step":     step + 1,
                "loss":     round(loss_val, 6),
                "bart_sim": round(sim_val,  4),
            })

            # Track the best DISCRETE suffix seen so far
            if loss_val < best_loss:
                best_loss = loss_val
                best_ids = current_ids.clone()

            # Print progress
            if self.config.verbose and (step + 1) % print_every == 0:
                current_suffix_text = self.surrogate.decode_suffix_ids(current_ids)
                full_sentence = base_text + " " + current_suffix_text
                print(f"    Step {step+1:03d}/{self.config.pgd_steps}"
                      f"  loss={loss_val:.4f}"
                      f"  sim={sim_val:.4f}")
                print(f"    Full: {full_sentence}")
                print()

            # ------------------------------------------------------------------
            # Gradient-Guided Token Substitution (HotFlip / UAT)
            # ------------------------------------------------------------------
            with torch.no_grad():
                grad = suffix_embeds.grad  # (1, suffix_len, 768)

                # First-order Taylor approximation of loss change for swapping tokens.
                # We want to minimize delta_L = grad * (new_embed - old_embed)
                # To minimize this, we maximize -grad * new_embed
                scores = torch.matmul(-grad, vocab_embeds.T)  # (1, suffix_len, vocab_size)

                # To avoid massive oscillation, update ONE random position per step
                pos = random.randint(0, current_ids.shape[1] - 1)

                # Pick the token that minimizes the loss approximation for this position
                best_vocab_idx = torch.argmax(scores[0, pos]).item()
                new_token_id = vocab_tensor[best_vocab_idx]

                current_ids[0, pos] = new_token_id

        best_text = self.surrogate.decode_suffix_ids(best_ids)
        return best_ids, best_text, history
