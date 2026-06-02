"""Cache audio and text embeddings for TeleAntiFraud binary classification.

This script intentionally freezes large encoders and writes compact .pt caches.
The downstream classifier can then train quickly on an 8 GB GPU.
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
import sys

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

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
    parser.add_argument(
        "--use-asr",
        action="store_true",
        help="Transcribe audio with Whisper and use ASR text as RoBERTa input instead of prompt text",
    )
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
        load_for_asr=args.use_asr,
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
    asr_texts: list[str] = []

    desc = "Caching features (ASR mode)" if args.use_asr else "Caching features"
    for batch in tqdm(list(_batched(samples, args.batch_size)), desc=desc):
        batch_audio_paths = [sample.audio_path for sample in batch]
        sample_ids.extend(sample.sample_id for sample in batch)
        audio_paths.extend(str(p) for p in batch_audio_paths)
        labels.extend(sample.label for sample in batch)
        audio_features.append(audio_encoder.encode(batch_audio_paths).cpu())

        if args.use_asr:
            transcriptions = audio_encoder.transcribe(batch_audio_paths)
            asr_texts.extend(transcriptions)
            text_features.append(text_encoder.encode(transcriptions).cpu())
        else:
            text_features.append(text_encoder.encode([sample.text for sample in batch]).cpu())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache_data: dict[str, object] = {
        "sample_id": sample_ids,
        "audio_path": audio_paths,
        "audio_features": torch.cat(audio_features, dim=0),
        "text_features": torch.cat(text_features, dim=0),
        "labels": torch.tensor(labels, dtype=torch.long),
        "audio_model": args.audio_model,
        "text_model": args.text_model,
        "use_asr": args.use_asr,
    }
    if args.use_asr:
        cache_data["asr_texts"] = asr_texts
    torch.save(cache_data, args.output)
    print(f"Saved {len(sample_ids)} rows to {args.output}")


class WhisperAudioEncoder:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        max_seconds: float,
        load_for_asr: bool = False,
    ) -> None:
        import torchaudio
        from transformers import WhisperModel, WhisperProcessor

        self.torchaudio = torchaudio
        self.processor = WhisperProcessor.from_pretrained(model_name, local_files_only=True)
        self.device = device
        self.max_seconds = max_seconds
        self.sample_rate = int(self.processor.feature_extractor.sampling_rate)
        self._asr_model: object = None

        if load_for_asr:
            from transformers import WhisperForConditionalGeneration

            asr_model = WhisperForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16, local_files_only=True
            ).to(device).eval()
            # Rebuild generation config from tokenizer since cached version is missing/outdated
            from transformers import GenerationConfig

            tokenizer = self.processor.tokenizer
            lang_to_id = {}
            for tid in range(50259, 50358):
                tok = tokenizer.decode([tid])
                if tok.startswith("<|") and tok.endswith("|>"):
                    lang_to_id[tok.strip("<>|")] = tid
            gen_config = GenerationConfig(
                lang_to_id=lang_to_id,
                task_to_id={
                    "transcribe": int(tokenizer.convert_tokens_to_ids("<|transcribe|>")),
                    "translate": int(tokenizer.convert_tokens_to_ids("<|translate|>")),
                },
                language="zh",
                task="transcribe",
                max_length=256,
                decoder_start_token_id=int(tokenizer.convert_tokens_to_ids("<|startoftranscript|>")),
                bos_token_id=int(tokenizer.convert_tokens_to_ids("<|startoftranscript|>")),
                eos_token_id=int(tokenizer.convert_tokens_to_ids("<|endoftext|>")),
                pad_token_id=int(tokenizer.convert_tokens_to_ids("<|endoftext|>")),
            )
            asr_model.generation_config = gen_config
            self._asr_model = asr_model
            self.model = asr_model.model.encoder
        else:
            self.model = WhisperModel.from_pretrained(model_name).encoder.to(device).eval()

    @torch.no_grad()
    def encode(self, paths: list[Path]) -> torch.Tensor:
        import torch.nn.functional as F

        waves = [self._load_audio(path) for path in paths]
        inputs = self.processor(
            waves,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs.input_features.to(
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )
        # Pad to the fixed 3000-frame length Whisper expects
        expected_len = 3000
        cur_len = input_features.shape[-1]
        if cur_len < expected_len:
            pad_amount = expected_len - cur_len
            input_features = F.pad(input_features, (0, pad_amount))
        elif cur_len > expected_len:
            input_features = input_features[..., :expected_len]
        hidden = self.model(input_features=input_features).last_hidden_state
        return hidden.mean(dim=1)

    @torch.no_grad()
    def transcribe(self, paths: list[Path]) -> list[str]:
        """Transcribe audio to Chinese text using the full Whisper model."""
        if self._asr_model is None:
            raise RuntimeError("ASR model not loaded; pass load_for_asr=True")

        waves = [self._load_audio(path) for path in paths]
        inputs = self.processor(
            waves,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs.input_features.to(
            device=self.device,
            dtype=next(self._asr_model.parameters()).dtype,
        )

        generated_ids = self._asr_model.generate(
            input_features,
            max_length=256,
        )
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

    def _load_audio(self, path: Path) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        waveform = self._load_audio_torchaudio(path)
        max_samples = int(self.max_seconds * self.sample_rate)
        return waveform[:max_samples].numpy()

    def _load_audio_torchaudio(self, path: Path):
        try:
            waveform, sample_rate = self.torchaudio.load(str(path))
        except RuntimeError:
            waveform, sample_rate = self._load_audio_ffmpeg(path)
        waveform = waveform.mean(dim=0) if waveform.dim() > 1 else waveform
        if sample_rate != self.sample_rate:
            waveform = self.torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
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
        import shutil
        which = shutil.which("ffmpeg")
        if which:
            return which
        import os
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
