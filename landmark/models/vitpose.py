from __future__ import annotations

from torch import nn

from .heatmap_baseline import decode_global_heatmaps


class ViTPoseLandmarkBaseline(nn.Module):
    """Compact ViTPose-style baseline with a transformer and deconvolution head."""

    def __init__(
        self,
        input_channels: int = 1,
        num_landmarks: int = 129,
        embed_dim: int = 192,
        patch_size: int = 16,
        depth: int = 6,
        attention_heads: int = 6,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            input_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        layer = nn.TransformerEncoderLayer(
            embed_dim,
            attention_heads,
            embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, num_landmarks, 1),
        )

    def forward(self, image, **_):
        features = self.patch_embed(image)
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.norm(self.transformer(tokens))
        features = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return decode_global_heatmaps(self.heatmap_head(features))
