"""Run predictions from a trained fusion classifier over a feature cache."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from teledeceit.data import ID_TO_LABEL
from teledeceit.features import CachedFeatureDataset
from teledeceit.model import AudioTextFusionClassifier
from teledeceit.training import feature_collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    dataset = CachedFeatureDataset(args.cache)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=feature_collate_fn)

    model = AudioTextFusionClassifier(
        audio_dim=int(checkpoint["audio_dim"]),
        text_dim=int(checkpoint["text_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "prediction", "fraud_probability"])
        writer.writeheader()
        for batch in loader:
            logits = model(
                audio_features=batch["audio_features"].to(device),
                text_features=batch["text_features"].to(device),
            ).logits
            probs = logits.softmax(dim=-1).cpu()
            preds = probs.argmax(dim=-1)
            for sample_id, pred, prob in zip(batch["sample_id"], preds.tolist(), probs[:, 1].tolist()):
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "prediction": ID_TO_LABEL[int(pred)],
                        "fraud_probability": f"{prob:.6f}",
                    }
                )


if __name__ == "__main__":
    main()
