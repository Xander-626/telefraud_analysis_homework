"""Cache audio and text embeddings for TeleAntiFraud binary classification.

This script intentionally freezes large encoders and writes compact .pt caches.
The downstream classifier can then train quickly on an 8 GB GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from teledeceit.data import BinarySample, load_binary_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-path", required=True, type=Path)
    parser.add_argument("--data-root", default=Path("data"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audio-model", default="openai/whisper-small")
    parser.add_argument("--text-model", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--max-audio-seconds", default=30.0, type=float)
    parser.add_argument("--max-text-length", default=256, type=int)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_binary_samples(args.json_path, args.data_root)
    if args.limit is not None:
        samples = samples[: args.limit]

    device = torch.device(args.device)
    audio_encoder = WhisperAudioEncoder(
        model_name=args.audio_model,
        device=device,
        max_seconds=args.max_audio_seconds,
    )
    text_encoder = TransformerTextEncoder(
        model_name=args.text_model,
        device=device,
        max_length=args.max_text_length,
    )

    sample_ids: list[str] = []
    audio_features: list[torch.Tensor] = []
    text_features: list[torch.Tensor] = []
    labels: list[int] = []
    audio_paths: list[str] = []

    for batch in tqdm(list(_batched(samples, args.batch_size)), desc="Caching features"):
        sample_ids.extend(sample.sample_id for sample in batch)
        audio_paths.extend(str(sample.audio_path) for sample in batch)
        labels.extend(sample.label for sample in batch)
        audio_features.append(audio_encoder.encode([sample.audio_path for sample in batch]).cpu())
        text_features.append(text_encoder.encode([sample.text for sample in batch]).cpu())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sample_id": sample_ids,
            "audio_path": audio_paths,
            "audio_features": torch.cat(audio_features, dim=0),
            "text_features": torch.cat(text_features, dim=0),
            "labels": torch.tensor(labels, dtype=torch.long),
            "audio_model": args.audio_model,
            "text_model": args.text_model,
        },
        args.output,
    )
    print(f"Saved {len(sample_ids)} rows to {args.output}")


class WhisperAudioEncoder:
    def __init__(self, model_name: str, device: torch.device, max_seconds: float) -> None:
        import torchaudio
        from transformers import WhisperModel, WhisperProcessor

        self.torchaudio = torchaudio
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name).encoder.to(device).eval()
        self.device = device
        self.max_seconds = max_seconds
        self.sample_rate = int(self.processor.feature_extractor.sampling_rate)

    @torch.no_grad()
    def encode(self, paths: list[Path]) -> torch.Tensor:
        waves = [self._load_audio(path) for path in paths]
        inputs = self.processor(
            waves,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs.input_features.to(self.device)
        hidden = self.model(input_features=input_features).last_hidden_state
        return hidden.mean(dim=1)

    def _load_audio(self, path: Path) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        waveform, sample_rate = self.torchaudio.load(str(path))
        waveform = waveform.mean(dim=0)
        if sample_rate != self.sample_rate:
            waveform = self.torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        max_samples = int(self.max_seconds * self.sample_rate)
        return waveform[:max_samples].numpy()


class TransformerTextEncoder:
    def __init__(self, model_name: str, device: torch.device, max_length: int) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output = self.model(**inputs)
        hidden = output.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _batched(samples: list[BinarySample], batch_size: int):
    for start in range(0, len(samples), batch_size):
        yield samples[start : start + batch_size]


if __name__ == "__main__":
    main()
