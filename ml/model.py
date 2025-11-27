"""
1D ResNet + Attention Model for ECG Arrhythmia Classification

Architecture:
- 1D Convolutional layers with residual connections
- Multi-head self-attention for focusing on important regions
- Global pooling and dense layers for classification

Uses PyTorch for better stability and performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """
    Residual block with two conv layers and skip connection.

    Architecture:
    x -> Conv1D -> BN -> ReLU -> Conv1D -> BN -> Add(skip) -> ReLU
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for 1D sequences."""
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.permute(0, 2, 1)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        # Back to (batch, channels, seq_len)
        return x.permute(0, 2, 1)


class ECGResNetAttention(nn.Module):
    """
    1D ResNet + Attention model for ECG classification.

    Args:
        input_channels: Number of input channels (2 for ECG)
        num_classes: Number of output classes (3: AF, VTACH, PAUSE)
        filters: List of filter sizes for each ResNet stage
        num_res_blocks: Number of residual blocks per stage
        num_attention_heads: Number of attention heads
        dropout_rate: Dropout rate for regularization
    """
    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 3,
        filters: list = [64, 128, 256],
        num_res_blocks: int = 2,
        num_attention_heads: int = 4,
        dropout_rate: float = 0.5
    ):
        super().__init__()

        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv1d(input_channels, 32, 7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1)
        )

        # ResNet stages
        self.stages = nn.ModuleList()
        in_channels = 32
        for i, out_channels in enumerate(filters):
            stage = nn.Sequential()
            for j in range(num_res_blocks):
                stride = 2 if (i > 0 and j == 0) else 1
                stage.add_module(
                    f'block_{j}',
                    ResidualBlock(in_channels if j == 0 else out_channels, out_channels, stride=stride)
                )
            self.stages.append(stage)
            in_channels = out_channels

        # Attention
        self.attention = MultiHeadAttention(filters[-1], num_attention_heads)

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(filters[-1], 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.permute(0, 2, 1)

        # Initial conv
        x = self.initial(x)

        # ResNet stages
        for stage in self.stages:
            x = stage(x)

        # Attention
        x = self.attention(x)

        # Global pooling
        x = self.global_pool(x).squeeze(-1)

        # Classification
        x = self.classifier(x)

        return x


class SimpleCNN(nn.Module):
    """Simple CNN baseline model for comparison."""
    def __init__(self, input_channels: int = 2, num_classes: int = 3, dropout_rate: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.ReLU(),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, seq, ch) -> (batch, ch, seq)
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.classifier(x)
        return x


def build_resnet_attention_model(
    input_shape: Tuple[int, int] = (18000, 2),
    num_classes: int = 3,
    filters: list = [64, 128, 256],
    num_res_blocks: int = 2,
    num_attention_heads: int = 4,
    dropout_rate: float = 0.5,
    **kwargs
) -> ECGResNetAttention:
    """Build 1D ResNet + Attention model for ECG classification."""
    return ECGResNetAttention(
        input_channels=input_shape[1],
        num_classes=num_classes,
        filters=filters,
        num_res_blocks=num_res_blocks,
        num_attention_heads=num_attention_heads,
        dropout_rate=dropout_rate
    )


def build_simple_cnn(
    input_shape: Tuple[int, int] = (18000, 2),
    num_classes: int = 3,
    dropout_rate: float = 0.5
) -> SimpleCNN:
    """Simple CNN baseline model for comparison."""
    return SimpleCNN(
        input_channels=input_shape[1],
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )

