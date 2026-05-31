"""Real-model inference pipeline: Whisper ASR + RoBERTa + MLP classifier."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperModel,
    WhisperProcessor,
)


# ---------------------------------------------------------------------------
# Audio loading (FFmpeg fallback)
# ---------------------------------------------------------------------------

def _find_ffmpeg() -> str:
    import shutil
    winget = shutil.which("ffmpeg")
    if winget:
        return winget
    for c in [r"C:\Program Files\ffmpeg\bin\ffmpeg.exe", r"C:\ffmpeg\bin\ffmpeg.exe"]:
        if Path(c).exists():
            return c
    return "ffmpeg"


_FFMPEG_BIN = _find_ffmpeg()


def _load_audio(path: Path, sr: int = 16000) -> np.ndarray:
    try:
        import torchaudio
        waveform, sample_rate = torchaudio.load(str(path))
        if sample_rate != sr:
            waveform = torchaudio.transforms.Resample(sample_rate, sr)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0).numpy().astype(np.float32)
    except RuntimeError:
        pass
    cmd = [_FFMPEG_BIN, "-i", str(path), "-f", "s16le",
           "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1", "-"]
    raw = subprocess.run(cmd, capture_output=True, timeout=30).stdout
    return (np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

class FraudDetector:
    """Complete inference pipeline: audio → ASR text + embedding → prediction."""

    def __init__(
        self,
        whisper_model: str = "openai/whisper-small",
        text_model: str = "hfl/chinese-roberta-wwm-ext",
        checkpoint: str = "runs/binary_fusion_whisper_small_asr_roberta/best_model.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device = torch.device(device)
        print(f"[FraudDetector] Loading models on {self.device}...")

        # ---- Whisper processor (mel extraction) + encoder-only model ----
        print("[FraudDetector]   Whisper-small...")
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model)
        # WhisperModel.encoder gives us encoder-only (no decoder needed for embedding)
        self.whisper_encoder = WhisperModel.from_pretrained(
            whisper_model,
        ).encoder.to(self.device).eval()

        # ---- Full Whisper model for ASR (lazy-loaded) ----
        self._whisper_model_name = whisper_model
        self._whisper_asr = None

        # ---- RoBERTa ----
        print("[FraudDetector]   Chinese RoBERTa...")
        self.roberta = AutoModel.from_pretrained(text_model).to(self.device).eval()
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model)

        # ---- MLP classifier ----
        print("[FraudDetector]   MLP classifier...")
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        from teledeceit.model import AudioTextFusionClassifier
        self.classifier = AudioTextFusionClassifier(
            audio_dim=ckpt["audio_dim"],
            text_dim=ckpt["text_dim"],
            hidden_dim=ckpt["hidden_dim"],
            dropout=ckpt["dropout"],
        ).to(self.device)
        self.classifier.load_state_dict(ckpt["model_state"])
        self.classifier.eval()

        self._best_metrics = ckpt.get("best_metrics", {})
        print(f"[FraudDetector] Ready. Best F1={self._best_metrics.get('f1', 'N/A')}")

    # ---- helpers ----

    def _get_whisper_asr(self):
        """Lazy-load the full Whisper model for ASR transcription."""
        if self._whisper_asr is None:
            print("[FraudDetector]   Loading Whisper ASR model...")
            self._whisper_asr = WhisperForConditionalGeneration.from_pretrained(
                self._whisper_model_name, torch_dtype=torch.float16,
            ).to(self.device).eval()
        return self._whisper_asr

    def _extract_mel(self, audio_path: Path) -> torch.Tensor:
        """Load audio → WhisperProcessor → mel spectrogram → pad/trunc to 3000."""
        waveform = _load_audio(audio_path)
        inputs = self.whisper_processor(
            waveform, sampling_rate=16000, return_tensors="pt",
        )
        mel = inputs.input_features.squeeze(0)  # (80, frames)
        if mel.shape[-1] < 3000:
            mel = torch.nn.functional.pad(mel, (0, 3000 - mel.shape[-1]))
        else:
            mel = mel[:, :3000]
        return mel.unsqueeze(0).to(self.device)  # (1, 80, 3000)

    @torch.no_grad()
    def _encode_audio(self, audio_path: Path) -> torch.Tensor:
        """Whisper encoder: audio → (768,) embedding."""
        mel = self._extract_mel(audio_path)
        # self.whisper_encoder is encoder-only — no decoder inputs needed
        hidden = self.whisper_encoder(mel).last_hidden_state  # (1, 1500, 768)
        return hidden.mean(dim=1).squeeze(0)  # (768,)

    @torch.no_grad()
    def _transcribe(self, audio_path: Path) -> str:
        """Whisper ASR: audio → Chinese text."""
        asr_model = self._get_whisper_asr()
        mel = self._extract_mel(audio_path)
        # ASR model is fp16 — match input dtype
        mel = mel.to(asr_model.dtype)
        generated = asr_model.generate(
            mel, max_length=448, language="zh",
            task="transcribe", do_sample=False,
        )
        return self.whisper_processor.tokenizer.decode(
            generated[0], skip_special_tokens=True,
        )

    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        """RoBERTa: Chinese text → (768,) embedding."""
        tokens = self.text_tokenizer(
            text, return_tensors="pt", max_length=256, truncation=True,
        ).to(self.device)
        hidden = self.roberta(**tokens).last_hidden_state
        return hidden[:, 0, :].squeeze(0)  # [CLS] token

    # ---- main API ----

    @torch.no_grad()
    def predict(self, audio_path: Path) -> dict:
        """Run full pipeline on a single audio file.

        Returns a dict with prediction, fraud_probability, risk_level,
        asr_text, evidence, and inference_time_seconds.
        """
        import time
        t0 = time.perf_counter()

        # Step 1: ASR transcription
        asr_text = self._transcribe(audio_path)
        t1 = time.perf_counter()

        # Step 2: Audio embedding
        audio_emb = self._encode_audio(audio_path)

        # Step 3: Text embedding
        text_emb = self._encode_text(asr_text)

        # Step 4: MLP classification
        logits = self.classifier(
            audio_features=audio_emb.unsqueeze(0),
            text_features=text_emb.unsqueeze(0),
        ).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        fraud_prob = float(probs[1].cpu())
        pred_idx = int(logits.argmax(dim=-1).cpu().item())
        t2 = time.perf_counter()

        # Evidence extraction
        evidence = []
        fraud_keywords = [
            "转账", "安全账户", "验证码", "银行卡", "冻结", "涉嫌",
            "公检法", "贷款", "中奖", "资金", "密码", "账户",
        ]
        matched = [kw for kw in fraud_keywords if kw in asr_text]
        if matched:
            evidence.append(f"ASR 文本命中电诈关键词: {', '.join(matched)}")
        else:
            evidence.append("ASR 文本未命中典型电诈关键词")

        prediction = "fraud" if pred_idx == 1 else "normal"
        if fraud_prob >= 0.75:
            risk_level = "high"
        elif fraud_prob >= 0.45:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "prediction": prediction,
            "fraud_probability": round(fraud_prob, 4),
            "risk_level": risk_level,
            "asr_text": asr_text,
            "evidence": evidence,
            "model": f"Whisper-small ASR + Chinese RoBERTa + MLP (F1={self._best_metrics.get('f1', 'N/A')})",
            "mode": "real_inference",
            "asr_time_s": round(t1 - t0, 2),
            "total_time_s": round(t2 - t0, 2),
        }


# ---- Singleton ----

_detector: FraudDetector | None = None


def get_detector(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> FraudDetector:
    """Get or create the global FraudDetector singleton."""
    global _detector
    if _detector is None:
        _detector = FraudDetector(device=device)
    return _detector
