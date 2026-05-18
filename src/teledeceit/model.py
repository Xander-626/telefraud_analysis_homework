"""Fusion classifier for cached audio and text embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ClassifierOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class AudioTextFusionClassifier(nn.Module):
    """Lightweight classifier that fuses one audio embedding and one text embedding."""

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        input_dim = audio_dim + text_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> ClassifierOutput:
        features = torch.cat([audio_features.float(), text_features.float()], dim=-1)
        logits = self.classifier(features)
        loss = F.cross_entropy(logits, labels.long()) if labels is not None else None
        return ClassifierOutput(logits=logits, loss=loss)
