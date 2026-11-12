"""
suffix_attack/surrogate.py
==========================
Nomic white-box surrogate.

Key design:
  - Clean vocab list built ONCE at init time
  - Used for BOTH initial random noise AND final PGD projection
  - Guarantees start AND end suffix are clean natural English words
"""

from __future__ import annotations

import random
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from .config import SuffixAttackConfig


class NomicSurrogate:
    """
    Wraps nomic-ai/nomic-embed-text-v1.5 for white-box gradient access.

    The model is loaded in eval mode. No training happens here.
    Gradients are computed only during PGD steps in the optimizer.
    """

    def __init__(self, config: SuffixAttackConfig) -> None:
        self.config = config
        self.device = config.torch_device

        model_name = "nomic-ai/nomic-embed-text-v1.5"
        if config.verbose:
            print(f"\n[DASA] Loading Nomic surrogate: {model_name} ...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.model.to(self.device)

        # Build clean vocab list ONCE at init.
        self._clean_vocab_ids: List[int] = self._build_clean_vocab()

        if config.verbose:
            n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
            print(f"[DASA] Nomic loaded. {n_params:.0f}M params on {self.device}. "
                  f"Clean vocab: {len(self._clean_vocab_ids)} words.")

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def get_embedding_layer(self) -> torch.nn.Embedding:
        """Return Nomic's shared token embedding table."""
        return self.model.embeddings.word_embeddings

    def tokenize(self, text: str) -> torch.Tensor:
        """Text -> Nomic token IDs. Returns shape (1, seq_len) on self.device."""
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=2048,
        ).input_ids.to(self.device)

    def encode_with_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run Nomic encoder on pre-computed token embeddings.
        Allows gradient to flow through suffix embeddings only.

        Parameters
        ----------
        inputs_embeds : (1, L, 768)
        attention_mask: (1, L) — auto-created if None

        Returns
        -------
        H : (1, L, 768) — encoder hidden states
        """
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2],
                dtype=torch.long,
                device=self.device,
            )

        # Nomic uses standard BERT forward signature
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state

    def model_embed(self, text: str, prefix="search_query: ") -> torch.Tensor:
        """Mean-pooled, L2-normalized embedding of text. No gradients. Returns (1, 768)."""
        with torch.no_grad():
            full_text = prefix + text if prefix else text
            ids = self.tokenize(full_text)
            embeds = self.get_embedding_layer()(ids)
            H = self.encode_with_embeds(embeds)

            # Mean pooling and normalization (Nomic standard)
            pooled = H.mean(dim=1)
            return F.normalize(pooled, p=2, dim=-1)

    # ------------------------------------------------------------------
    # Vocabulary utilities
    # ------------------------------------------------------------------

    def _build_clean_vocab(self) -> List[int]:
        """
        Build a filtered list of clean English token IDs from Nomic vocab.
        Nomic uses WordPiece. Full words do NOT start with ##.
        """
        _blocklist = {
            "fuckin", "fucked", "fucking", "nigger", "nigga", "bitch", "bastard",
            "lightsaber", "caliphate", "glyphosate", "sociop", "incarcer",
        }

        special_ids = set(self.tokenizer.all_special_ids)
        vocab = self.tokenizer.get_vocab()

        valid = []
        for word, token_id in vocab.items():
            if token_id in special_ids:
                continue
            if word.startswith("##"):
                continue
            if not word.isalpha():
                continue
            if len(word) < 3 or len(word) > 12:
                continue
            if word.lower() in _blocklist:
                continue
            valid.append(token_id)

        return valid

    def sample_from_vocab(self, length: int) -> torch.Tensor:
        """
        Randomly sample `length` token IDs from the clean vocab list.

        The suffix ALWAYS starts as random noise — it is never seeded from the
        query. Retrieval must be earned by optimising this random start toward
        the query embedding; deriving the start from the query would let the
        document retrieve trivially and defeat the whole suffix mechanism.
        """
        sampled = random.sample(self._clean_vocab_ids, length)
        return torch.tensor([sampled], dtype=torch.long, device=self.device)

    def project_to_vocab(self, perturbed_embeds: torch.Tensor) -> torch.Tensor:
        """
        Project continuous embeddings to the nearest clean vocab tokens using Cosine Similarity.
        """
        vocab_tensor = torch.tensor(self._clean_vocab_ids, device=self.device)
        vocab_embeds = self.get_embedding_layer()(vocab_tensor)

        vocab_norm = F.normalize(vocab_embeds, p=2, dim=-1)
        pert_norm = F.normalize(perturbed_embeds, p=2, dim=-1)

        sims = torch.matmul(pert_norm, vocab_norm.T)
        best_idx = torch.argmax(sims, dim=-1)

        projected_ids = vocab_tensor[best_idx]
        return projected_ids.squeeze(0)

    def decode_suffix_ids(self, ids: torch.Tensor) -> str:
        """Decode token IDs back to a string."""
        if ids.dim() == 2:
            ids = ids.squeeze(0)
        return self.tokenizer.decode(ids.tolist(), skip_special_tokens=True).strip()
