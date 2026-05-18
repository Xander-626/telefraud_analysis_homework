"""Feature cache dataset utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class CachedFeatureDataset(Dataset[dict[str, Any]]):
    """Torch dataset backed by cached audio/text embeddings."""

    def __init__(self, cache_path: str | Path) -> None:
        payload = torch.load(Path(cache_path), map_location="cpu", weights_only=False)
        self.sample_id = list(payload["sample_id"])
        self.audio_features = _as_float_tensor(payload["audio_features"])
        self.text_features = _as_float_tensor(payload["text_features"])
        self.labels = torch.as_tensor(payload["labels"], dtype=torch.long)
        self._validate_lengths()

    def __len__(self) -> int:
        return len(self.sample_id)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id[index],
            "audio_features": self.audio_features[index],
            "text_features": self.text_features[index],
            "label": self.labels[index],
        }

    @property
    def audio_dim(self) -> int:
        return int(self.audio_features.shape[-1])

    @property
    def text_dim(self) -> int:
        return int(self.text_features.shape[-1])

    def _validate_lengths(self) -> None:
        lengths = {
            len(self.sample_id),
            int(self.audio_features.shape[0]),
            int(self.text_features.shape[0]),
            int(self.labels.shape[0]),
        }
        if len(lengths) != 1:
            raise ValueError("Feature cache fields do not share the same row count")


def _as_float_tensor(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.dim() != 2:
        raise ValueError(f"Expected a 2D feature tensor, got shape {tuple(tensor.shape)}")
    return tensor
