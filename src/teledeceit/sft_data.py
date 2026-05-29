"""SFT JSONL data loader for TeleAntiFraud multi-turn audio instruction data."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


LABEL_TO_ID = {"normal": 0, "fraud": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class SftSample:
    sample_id: str
    task_pattern: str          # "SCENE_ONLY" | "FRAUD_BINARY" | "FRAUD_TYPE"
    audio_path: Path
    messages: list[dict]        # full multi-turn conversation
    answers: str                # original answer string
    binary_label: int | None    # 0=normal, 1=fraud, None for SCENE_ONLY


def load_sft_samples(jsonl_path: str | Path, data_root: str | Path) -> list[SftSample]:
    """Parse multi-turn JSONL and classify by task pattern."""
    jsonl_path = Path(jsonl_path)
    data_root = Path(data_root)
    split_name = jsonl_path.stem

    samples: list[SftSample] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            answers = str(record.get("answers", "")).strip()
            audios = record.get("audios", [])

            if not audios:
                raise ValueError(f"No audio in {jsonl_path} line {idx}")
            audio_path = data_root / str(audios[0])

            num_messages = len(messages)
            if num_messages <= 2:
                task_pattern = "SCENE_ONLY"
                binary_label = None
            elif num_messages <= 4:
                task_pattern = "FRAUD_BINARY"
                ans_lower = answers.lower()
                binary_label = LABEL_TO_ID.get(ans_lower)
                if binary_label is None:
                    raise ValueError(
                        f"Expected fraud/normal, got {answers!r} in {jsonl_path} line {idx}"
                    )
            else:
                task_pattern = "FRAUD_TYPE"
                binary_label = 1  # all 3-turn samples are fraud cases

            samples.append(
                SftSample(
                    sample_id=f"{split_name}-{idx:06d}",
                    task_pattern=task_pattern,
                    audio_path=audio_path,
                    messages=messages,
                    answers=answers,
                    binary_label=binary_label,
                )
            )

    return samples


def filter_binary_samples(samples: list[SftSample]) -> list[SftSample]:
    """Keep only samples with a binary fraud label (exclude SCENE_ONLY)."""
    return [s for s in samples if s.binary_label is not None]
