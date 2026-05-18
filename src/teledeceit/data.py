"""Dataset loading helpers for TeleAntiFraud binary classification."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import torch


LABEL_TO_ID = {
    "normal": 0,
    "fraud": 1,
}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class BinarySample:
    sample_id: str
    audio_path: Path
    text: str
    label: int
    answer: str


def load_binary_samples(json_path: str | Path, data_root: str | Path) -> list[BinarySample]:
    """Load binary fraud samples from TeleAntiFraud prompt-style JSON."""
    json_path = Path(json_path)
    data_root = Path(data_root)
    records = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a list in {json_path}, got {type(records).__name__}")

    samples: list[BinarySample] = []
    split_name = json_path.stem
    for index, record in enumerate(records):
        answer = str(record.get("answer", "")).strip().lower()
        if answer not in LABEL_TO_ID:
            raise ValueError(f"Unsupported label {answer!r} in {json_path} at index {index}")

        audio_url = _first_audio_url(record.get("prompt", []))
        audio_path = data_root / audio_url
        text = "\n".join(_iter_prompt_text(record.get("prompt", []))).strip()
        samples.append(
            BinarySample(
                sample_id=f"{split_name}-{index:06d}",
                audio_path=audio_path,
                text=text,
                label=LABEL_TO_ID[answer],
                answer=answer,
            )
        )
    return samples


def collate_text_only(samples: list[BinarySample]) -> dict[str, Any]:
    """Collate metadata and labels; model-specific tokenization happens later."""
    return {
        "sample_id": [sample.sample_id for sample in samples],
        "audio_path": [sample.audio_path for sample in samples],
        "text": [sample.text for sample in samples],
        "labels": torch.tensor([sample.label for sample in samples], dtype=torch.long),
    }


def _first_audio_url(prompt: Iterable[dict[str, Any]]) -> str:
    for message in prompt:
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "audio":
                    audio_url = item.get("audio_url") or item.get("audio")
                    if audio_url:
                        return str(audio_url)
    raise ValueError("Sample does not contain an audio reference")


def _iter_prompt_text(prompt: Iterable[dict[str, Any]]) -> Iterable[str]:
    for message in prompt:
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            yield content.strip()
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        yield text.strip()
