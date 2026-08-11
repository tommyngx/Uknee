# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Independent cross/self-attention blocks for OA26 anatomical regions."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class OA26RegionTransformerLayer(nn.Module):
    """Cross-attend landmarks to one ROI, then self-attend only within that region instance."""

    def __init__(self, d_model: int = 128, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        """Initialize pre-normalized attention and feed-forward sublayers."""
        super().__init__()
        self.cross_norm = nn.LayerNorm(d_model)
        self.image_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_model), nn.Dropout(dropout)
        )

    def forward(
        self, queries: torch.Tensor, image_tokens: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Update each region independently; the leading M dimension is never flattened or mixed."""
        query = self.cross_norm(queries)
        image = self.image_norm(image_tokens)
        queries = queries + self.cross_attention(query, image, image, need_weights=False)[0]
        queries = queries.masked_fill(~valid_mask.unsqueeze(-1), 0)
        query = self.self_norm(queries)
        queries = queries + self.self_attention(
            query, query, query, key_padding_mask=~valid_mask, need_weights=False
        )[0]
        queries = queries.masked_fill(~valid_mask.unsqueeze(-1), 0)
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries.masked_fill(~valid_mask.unsqueeze(-1), 0)


class OA26RegionTransformer(nn.Module):
    """Stack cross/self-attention blocks without any inter-region attention path."""

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        """Initialize the requested number of independent region layers."""
        super().__init__()
        self.layers = nn.ModuleList(
            OA26RegionTransformerLayer(d_model, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.gradient_checkpointing = bool(gradient_checkpointing)

    def forward(
        self, queries: torch.Tensor, image_tokens: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return refined landmark tokens for each independent region instance."""
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                queries = checkpoint(layer, queries, image_tokens, valid_mask, use_reentrant=False)
            else:
                queries = layer(queries, image_tokens, valid_mask)
        return self.norm(queries).masked_fill(~valid_mask.unsqueeze(-1), 0)
