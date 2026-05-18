"""Train the lightweight fusion classifier on cached embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from teledeceit.features import CachedFeatureDataset
from teledeceit.model import AudioTextFusionClassifier
from teledeceit.training import evaluate, feature_collate_fn, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--test-cache", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("runs/binary_fusion"), type=Path)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--hidden-dim", default=256, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_dataset = CachedFeatureDataset(args.train_cache)
    test_dataset = CachedFeatureDataset(args.test_cache)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=feature_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=feature_collate_fn,
    )

    model = AudioTextFusionClassifier(
        audio_dim=train_dataset.audio_dim,
        text_dim=train_dataset.text_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, device)
        eval_stats = evaluate(model, test_loader, device)
        row = {"epoch": epoch, "train_loss": train_stats["loss"], **eval_stats}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))

        if float(eval_stats["f1"]) > best_f1:
            best_f1 = float(eval_stats["f1"])
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "audio_dim": train_dataset.audio_dim,
                    "text_dim": train_dataset.text_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "best_metrics": eval_stats,
                },
                args.output_dir / "best_model.pt",
            )

    (args.output_dir / "metrics.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
