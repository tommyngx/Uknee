"""ViTPose++ full-frame heatmap models with task-aware mixture-of-experts FFNs."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class StochasticDepth(nn.Module):
    """Drop a residual branch per sample during training."""

    def __init__(self, probability: float = 0.0):
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("Stochastic-depth probability must be in [0, 1)")
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep_probability = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_probability)
        return x * mask / keep_probability


class MixtureOfExpertsMLP(nn.Module):
    """ViTPose++ FFN with shared channels and task-specific expert channels.

    The reference model chooses an expert from its source dataset. Uknee can
    supply the same zero-based expert index, while standalone calls use a
    learned image-level router so the model remains usable on one knee dataset.
    """

    def __init__(
        self,
        channels: int,
        num_experts: int = 6,
        part_features: int = 192,
        ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        if not 0 < part_features < channels:
            raise ValueError("part_features must be between 0 and channels")
        hidden = channels * ratio
        self.num_experts = num_experts
        self.part_features = part_features
        self.input_projection = nn.Linear(channels, hidden)
        self.activation = nn.GELU()
        self.shared_projection = nn.Linear(hidden, channels - part_features)
        self.experts = nn.ModuleList(nn.Linear(hidden, part_features) for _ in range(num_experts))
        self.router = nn.Linear(channels, num_experts)
        self.dropout = nn.Dropout(dropout)

    def _routing_weights(self, x: torch.Tensor, expert_index: torch.Tensor | None) -> torch.Tensor:
        if expert_index is None:
            return self.router(x.mean(dim=1)).softmax(dim=-1)
        index = expert_index.to(device=x.device, dtype=torch.long).reshape(-1)
        if index.numel() == 1 and x.shape[0] != 1:
            index = index.expand(x.shape[0])
        if index.numel() != x.shape[0]:
            raise ValueError("expert_index must contain one value per image")
        if torch.any((index < 0) | (index >= self.num_experts)):
            raise ValueError(f"expert_index values must be in [0, {self.num_experts})")
        return F.one_hot(index, num_classes=self.num_experts).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor, expert_index: torch.Tensor | None = None) -> torch.Tensor:
        routing = self._routing_weights(x, expert_index)
        hidden = self.activation(self.input_projection(x))
        shared = self.shared_projection(hidden)
        expert_outputs = torch.stack([expert(hidden) for expert in self.experts], dim=2)
        selected = (expert_outputs * routing[:, None, :, None]).sum(dim=2)
        return self.dropout(torch.cat((shared, selected), dim=-1))


class MoETransformerBlock(nn.Module):
    """Transformer block whose FFN follows the ViTPose++ partial-MoE design."""

    def __init__(
        self,
        channels: int,
        heads: int,
        num_experts: int = 6,
        part_features: int = 192,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = MixtureOfExpertsMLP(
            channels,
            num_experts=num_experts,
            part_features=part_features,
            ratio=mlp_ratio,
            dropout=dropout,
        )
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x: torch.Tensor, expert_index: torch.Tensor | None = None) -> torch.Tensor:
        normalized = self.norm1(x)
        x = x + self.drop_path(self.attention(normalized, normalized, normalized, need_weights=False)[0])
        return x + self.drop_path(self.mlp(self.norm2(x), expert_index))


class ViTPosePlusPlusS(nn.Module):
    """Small ViTPose++ encoder with partial-MoE FFNs and a heatmap head."""

    def __init__(
        self,
        input_channels: int = 3,
        num_landmarks: int = 129,
        image_size: int = 640,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        attention_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        num_experts: int = 6,
        part_features: int = 192,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(input_channels, embed_dim, patch_size, patch_size)
        grid = max(image_size // patch_size, 1)
        self.position = nn.Parameter(torch.zeros(1, embed_dim, grid, grid))
        path_rates = torch.linspace(0.0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            MoETransformerBlock(
                embed_dim,
                attention_heads,
                num_experts=num_experts,
                part_features=part_features,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=path_rates[index],
            )
            for index in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_landmarks, 1),
        )
        nn.init.trunc_normal_(self.position, std=0.02)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor, expert_index: torch.Tensor | None = None) -> torch.Tensor:
        features = self.patch_embed(image)
        batch, channels, height, width = features.shape
        position = F.interpolate(self.position, (height, width), mode="bicubic", align_corners=False)
        tokens = (features + position).flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens, expert_index)
        features = self.norm(tokens).transpose(1, 2).reshape(batch, channels, height, width)
        return self.heatmap_head(features)


class ViTPosePlusPlusB(ViTPosePlusPlusS):
    """Base ViTPose++ encoder with partial-MoE FFNs and a heatmap head."""

    def __init__(
        self,
        input_channels: int = 3,
        num_landmarks: int = 129,
        image_size: int = 640,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        attention_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        drop_path_rate: float = 0.3,
        num_experts: int = 6,
        part_features: int = 192,
    ):
        super().__init__(
            input_channels=input_channels,
            num_landmarks=num_landmarks,
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            attention_heads=attention_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            num_experts=num_experts,
            part_features=part_features,
        )


__all__ = [
    "MixtureOfExpertsMLP",
    "MoETransformerBlock",
    "ViTPosePlusPlusS",
    "ViTPosePlusPlusB",
]
