from __future__ import annotations

from torch import nn


class LandmarkQueryTransformer(nn.Module):
    """Self-attention over identity-preserving landmark queries, not full DETR."""

    def __init__(
        self,
        query_dim: int,
        attention_heads: int,
        feedforward_dim: int,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        block = nn.TransformerEncoderLayer(
            d_model=query_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            block, num_layers=layers, enable_nested_tensor=False
        )

    def forward(self, queries):
        return self.encoder(queries)
