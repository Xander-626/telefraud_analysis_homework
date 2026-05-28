"""End-to-end fusion classifier with partial Whisper unfreezing.

Unlike the pure cache-based approach, this script loads audio on-the-fly through
a partially unfrozen Whisper encoder. Text embeddings come from pre-cached ASR
features to avoid loading RoBERTa during training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from teledeceit.data import BinarySample
from teledeceit.metrics import compute_binary_metrics
from teledeceit.model import E2EFusionClassifier

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class E2EFeatureDataset(Dataset[dict[str, object]]):
    """Loads audio paths + pre-cached text embeddings from a .pt cache file."""

    def __init__(self, cache_path: Path, limit: int | None = None) -> None:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.sample_ids = list(payload["sample_id"])
        self.audio_paths = [Path(p) for p in payload["audio_path"]]
        self.text_features = torch.as_tensor(payload["text_features"], dtype=torch.float32)
        self.labels = torch.as_tensor(payload["labels"], dtype=torch.long)

        if limit is not None:
            self.sample_ids = self.sample_ids[:limit]
            self.audio_paths = self.audio_paths[:limit]
            self.text_features = self.text_features[:limit]
            self.labels = self.labels[:limit]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "sample_id": self.sample_ids[index],
            "audio_path": self.audio_paths[index],
            "text_features": self.text_features[index],
            "label": self.labels[index],
        }

    @property
    def text_dim(self) -> int:
        return int(self.text_features.shape[-1])


# ---------------------------------------------------------------------------
# Audio preprocessing (reused from cache_multimodal_features.py)
# ---------------------------------------------------------------------------


class AudioProcessor:
    """Load audio files and convert to Whisper mel features."""

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        max_seconds: float = 30.0,
        device: torch.device | None = None,
    ) -> None:
        import torchaudio
        from transformers import WhisperProcessor

        self.torchaudio = torchaudio
        self.processor = WhisperProcessor.from_pretrained(model_name, local_files_only=True)
        self.sample_rate = int(self.processor.feature_extractor.sampling_rate)
        self.max_samples = int(max_seconds * self.sample_rate)
        self.max_seconds = max_seconds
        self._device = device

    def load_mel(self, path: Path) -> torch.Tensor:
        """Load a single audio file and return mel spectrogram features."""
        waveform = self._load_audio(path)
        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.squeeze(0)
        expected_len = 3000
        cur_len = input_features.shape[-1]
        if cur_len < expected_len:
            import torch.nn.functional as F
            input_features = F.pad(input_features, (0, expected_len - cur_len))
        elif cur_len > expected_len:
            input_features = input_features[..., :expected_len]
        if self._device is not None:
            input_features = input_features.to(self._device)
        return input_features

    def _load_audio(self, path: Path) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        waveform = self._load_audio_torchaudio(path)
        return waveform[:self.max_samples]

    def _load_audio_torchaudio(self, path: Path) -> torch.Tensor:
        try:
            waveform, sample_rate = self.torchaudio.load(str(path))
        except RuntimeError:
            waveform, sample_rate = self._load_audio_ffmpeg(path)
        waveform = waveform.mean(dim=0) if waveform.dim() > 1 else waveform
        if sample_rate != self.sample_rate:
            waveform = self.torchaudio.functional.resample(
                waveform, sample_rate, self.sample_rate
            )
        return waveform

    def _load_audio_ffmpeg(self, path: Path):
        import subprocess
        import numpy as np

        ffmpeg_exe = self._find_ffmpeg()
        cmd = [
            ffmpeg_exe, "-i", str(path),
            "-f", "s16le", "-ac", "1", "-ar", str(self.sample_rate),
            "-", "-loglevel", "error",
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
        audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        return torch.from_numpy(audio).float(), self.sample_rate

    @staticmethod
    def _find_ffmpeg() -> str:
        import shutil, os
        which = shutil.which("ffmpeg")
        if which:
            return which
        winget_root = os.environ.get("LOCALAPPDATA", "")
        candidate = os.path.join(
            winget_root,
            "Microsoft", "WinGet", "Packages",
            "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe",
            "ffmpeg-8.1.1-full_build", "bin", "ffmpeg.exe",
        )
        if os.path.isfile(candidate):
            return candidate
        raise RuntimeError("ffmpeg not found; install it or add it to PATH")


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def e2e_collate_fn(
    batch: list[dict[str, object]],
    audio_processor: AudioProcessor,
) -> dict[str, object]:
    mels = [audio_processor.load_mel(item["audio_path"]) for item in batch]  # type: ignore[arg-type]
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "audio_features": torch.stack(mels),
        "text_features": torch.stack([item["text_features"] for item in batch]),  # type: ignore[arg-type]
        "labels": torch.stack([item["label"] for item in batch]),  # type: ignore[arg-type]
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--test-cache", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("runs/binary_e2e_asr"), type=Path)
    parser.add_argument("--audio-model", default="openai/whisper-small")
    parser.add_argument("--unfreeze-layers", default=2, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--grad-accum", default=8, type=int)
    parser.add_argument("--hidden-dim", default=256, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--max-audio-seconds", default=30.0, type=float)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _freeze_except_last_layers(encoder: nn.Module, num_unfreeze: int) -> None:
    layers = list(encoder.layers)
    for layer in layers[:-num_unfreeze]:
        for p in layer.parameters():
            p.requires_grad_(False)
    # Also freeze embedding layers (positional + conv)
    for name, param in encoder.named_parameters():
        if not name.startswith("layers."):
            param.requires_grad_(False)


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict[str, float | int]:
    model.eval()
    preds: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch(batch, device)
            with torch.amp.autocast("cuda", enabled=True):
                output = model(
                    audio_features=batch["audio_features"],
                    text_features=batch["text_features"],
                    labels=batch["labels"],
                )
            batch_preds = output.logits.argmax(dim=-1)
            preds.append(batch_preds.cpu())
            labels.append(batch["labels"].cpu())  # type: ignore[arg-type]

            if output.loss is not None:
                bs = int(batch["labels"].shape[0])  # type: ignore[union-attr]
                total_loss += float(output.loss.cpu().item()) * bs
                total_items += bs

    metrics = compute_binary_metrics(torch.cat(preds), torch.cat(labels))
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_amp = torch.cuda.is_available() and not args.no_fp16

    # ---- 1. Datasets ----
    train_dataset = E2EFeatureDataset(args.train_cache, limit=args.limit)
    test_dataset = E2EFeatureDataset(args.test_cache, limit=args.limit)

    audio_processor = AudioProcessor(
        model_name=args.audio_model,
        max_seconds=args.max_audio_seconds,
        device=None,  # mel features stay on CPU until collate moves them
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: e2e_collate_fn(batch, audio_processor),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: e2e_collate_fn(batch, audio_processor),
    )

    # ---- 2. Audio encoder with partial unfreezing ----
    from transformers import WhisperModel

    whisper = WhisperModel.from_pretrained(args.audio_model).to(device)
    encoder = whisper.encoder
    _freeze_except_last_layers(encoder, args.unfreeze_layers)

    audio_dim = encoder.config.d_model  # Whisper-small: 768

    # ---- 3. Model ----
    model = E2EFusionClassifier(
        audio_encoder=encoder,
        text_dim=train_dataset.text_dim,
        audio_dim=audio_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # Separate param groups: encoder layers use lower LR
    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr * 0.1},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- 4. Training ----
    best_f1 = -1.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar):
            batch = _move_batch(batch, device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(
                    audio_features=batch["audio_features"],
                    text_features=batch["text_features"],
                    labels=batch["labels"],
                )
                assert output.loss is not None
                loss = output.loss / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            bs = int(batch["labels"].shape[0])  # type: ignore[union-attr]
            total_loss += float(output.loss.detach().cpu().item()) * bs * args.grad_accum
            total_items += bs
            pbar.set_postfix({"loss": f"{float(output.loss.detach().cpu().item()):.4f}"})

        train_loss = total_loss / max(total_items, 1)
        eval_metrics = evaluate(model, test_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **eval_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))

        if float(eval_metrics["f1"]) > best_f1:
            best_f1 = float(eval_metrics["f1"])
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "audio_dim": audio_dim,
                    "text_dim": train_dataset.text_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "best_metrics": eval_metrics,
                },
                args.output_dir / "best_model.pt",
            )

    (args.output_dir / "metrics.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
