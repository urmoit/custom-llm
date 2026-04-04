"""Fully custom GPT-style language model — trained from scratch, no pre-trained weights.

Architecture
------------
* Token + learnable positional embeddings
* N × TransformerBlock (pre-norm):
    - MultiHeadSelfAttention  (causal / masked, built from raw tensor ops)
    - Position-wise FeedForward  (two linear layers + GELU)
* Final LayerNorm + linear projection (weight-tied with token embedding)

All components are implemented using basic PyTorch tensor operations.
No ``transformers``, no ``sentence-transformers``, and no pre-trained weights
are used anywhere in this file.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Minimum temperature to avoid division by zero during sampling
_MIN_TEMPERATURE = 1e-8


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """Masked (causal) multi-head self-attention built from first principles.

    The attention score is computed as::

        Attention(Q, K, V) = softmax( Q K^T / sqrt(d_head) ) V

    using plain matrix multiplications — no ``nn.MultiheadAttention``.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # Scale factor sqrt(head_dim) prevents dot-products from growing large
        # in high dimensions, which would push softmax into near-zero gradient regions.
        self.scale = math.sqrt(self.head_dim)

        # Separate projections (no bias — following GPT-2 style)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        # Project to Q / K / V and split into heads: (B, H, T, D)
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Scaled dot-product attention: (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply causal (upper-triangular) mask — prevents attending to future tokens
        if causal_mask is not None:
            scores = scores + causal_mask

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        # Weighted sum of values: (B, H, T, D) -> (B, T, C)
        out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


class FeedForward(nn.Module):
    """Position-wise two-layer feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block using pre-layer-norm (GPT-2 variant).

    Residual connections wrap both the attention sub-layer and the FFN
    sub-layer::

        x = x + Attn(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal_mask=causal_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CustomLanguageModel(nn.Module):
    """Custom GPT-style autoregressive language model — all weights trained from scratch.

    Parameters
    ----------
    vocab_size:
        Size of the token vocabulary.
    d_model:
        Embedding / hidden dimension.
    n_heads:
        Number of attention heads.
    n_layers:
        Number of transformer blocks.
    context_length:
        Maximum sequence length the model can process.
    d_ff:
        Inner dimension of the position-wise feed-forward network.
    dropout:
        Dropout probability used throughout.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        context_length: int = 256,
        d_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.context_length = context_length

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(context_length, d_model)
        self.emb_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output head
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: output projection shares weights with token embedding
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Initialise weights with a small normal distribution (GPT-2 style)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return an additive upper-triangular mask (−∞ above diagonal)."""
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute logits over the vocabulary for every position.

        Parameters
        ----------
        token_ids:
            Integer tensor of shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """
        B, T = token_ids.shape
        device = token_ids.device

        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        hidden = self.emb_drop(self.token_emb(token_ids) + self.pos_emb(positions))

        causal_mask = self._causal_mask(T, device)
        for block in self.blocks:
            hidden = block(hidden, causal_mask=causal_mask)

        hidden = self.norm(hidden)
        return self.lm_head(hidden)  # (B, T, vocab_size)

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_id: int = 3,
        device: str = "cpu",
    ) -> List[int]:
        """Autoregressively generate tokens given a prompt.

        Parameters
        ----------
        prompt_ids:
            List of starting token IDs (the prompt).
        max_new_tokens:
            Maximum number of tokens to generate.
        temperature:
            Sampling temperature — lower values are more conservative.
        top_k:
            Restrict sampling to the top-*k* most likely next tokens.
        eos_id:
            Token ID that signals end-of-sequence; generation stops here.
        device:
            PyTorch device string.

        Returns
        -------
        List[int]
            Full sequence including the original prompt IDs.
        """
        self.eval()
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            # Crop to the context window
            idx_ctx = idx[:, -self.context_length:]
            logits = self.forward(idx_ctx)
            next_logits = logits[:, -1, :] / max(temperature, _MIN_TEMPERATURE)

            # Top-k filtering
            if top_k > 0:
                top_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < top_vals[:, [-1]]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

            if next_id.item() == eos_id:
                break

        return idx[0].tolist()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self) -> dict:
        """Return a JSON-serialisable dictionary of hyperparameters."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "context_length": self.context_length,
            "n_layers": len(self.blocks),
            "n_heads": self.blocks[0].attn.n_heads if self.blocks else 0,
            "d_ff": self.blocks[0].ffn.fc1.out_features if self.blocks else 0,
            "num_parameters": self.num_parameters,
        }
