"""Screen SFT binary samples for hard cases using the existing frozen-encoder classifier.

Samples where the classifier is uncertain (low confidence) or makes mistakes
are retained as "hard cases" for SFT/LoRA training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from teledeceit.sft_data import SftSample, filter_binary_samples, load_sft_samples
from teledeceit.model import AudioTextFusionClassifier

# Reuse existing encoder classes
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_multimodal_features import TransformerTextEncoder, WhisperAudioEncoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl-path", default="data/sft/sft/train.jsonl", type=Path)
    p.add_argument("--data-root", default="data", type=Path)
    p.add_argument("--checkpoint", default="runs/binary_fusion_whisper_small_asr_roberta/best_model.pt", type=Path)
    p.add_argument("--output", default="data/sft/screened/hard_cases_train.jsonl", type=Path)
    p.add_argument("--audio-model", default="openai/whisper-small")
    p.add_argument("--text-model", default="hfl/chinese-roberta-wwm-ext")
    p.add_argument("--confidence-threshold", default=0.9, type=float)
    p.add_argument("--keep-misclassified", action="store_true", default=True)
    p.add_argument("--max-samples", default=None, type=int)
    p.add_argument("--limit", default=None, type=int)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_classifier(checkpoint_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = AudioTextFusionClassifier(
        audio_dim=ckpt["audio_dim"],
        text_dim=ckpt["text_dim"],
        hidden_dim=ckpt.get("hidden_dim", 256),
        dropout=ckpt.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load SFT binary samples
    all_samples = load_sft_samples(args.jsonl_path, args.data_root)
    binary_samples = filter_binary_samples(all_samples)
    if args.limit is not None:
        binary_samples = binary_samples[: args.limit]

    print(f"Loaded {len(binary_samples)} binary samples from {args.jsonl_path}")

    # Load encoders and classifier
    audio_enc = WhisperAudioEncoder(
        model_name=args.audio_model, device=device,
        max_seconds=30.0, load_for_asr=True,
    )
    text_enc = TransformerTextEncoder(
        model_name=args.text_model, device=device, max_length=256,
    )
    classifier = load_classifier(args.checkpoint, device)

    hard_cases: list[dict] = []
    stats = {"total": 0, "hard": 0, "misclassified": 0, "low_confidence": 0}

    paths = [s.audio_path for s in binary_samples]
    labels = [s.binary_label for s in binary_samples]

    batch_size = 2
    for i in tqdm(range(0, len(binary_samples), batch_size), desc="Screening"):
        batch_paths = paths[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]
        batch_samples = binary_samples[i : i + batch_size]

        with torch.no_grad():
            transcriptions = audio_enc.transcribe(batch_paths)
            text_feats = text_enc.encode(transcriptions)
            audio_feats = audio_enc.encode(batch_paths)
            output = classifier(
                audio_features=audio_feats.to(device),
                text_features=text_feats.to(device),
            )
            probs = torch.softmax(output.logits, dim=-1)
            fraud_probs = probs[:, 1].cpu()
            preds = output.logits.argmax(dim=-1).cpu()

        for j, sample in enumerate(batch_samples):
            stats["total"] += 1
            pred = int(preds[j].item())
            prob = float(fraud_probs[j].item())
            label = batch_labels[j]
            confidence = prob if pred == 1 else 1.0 - prob
            is_hard = False
            reason = ""

            if args.keep_misclassified and pred != label:
                is_hard = True
                stats["misclassified"] += 1
                reason = f"misclassified: pred={pred}, label={label}"
            elif confidence < args.confidence_threshold:
                is_hard = True
                stats["low_confidence"] += 1
                reason = f"low_confidence: {confidence:.3f}"

            if is_hard:
                stats["hard"] += 1
                hard_cases.append({
                    "sample_id": sample.sample_id,
                    "audio_path": str(sample.audio_path),
                    "label": label,
                    "prediction": pred,
                    "fraud_probability": prob,
                    "reason": reason,
                    "transcription": transcriptions[j],
                })

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        for case in hard_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"Screened {stats['total']} samples -> {stats['hard']} hard cases "
          f"({stats['misclassified']} misclassified, {stats['low_confidence']} low confidence)")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
