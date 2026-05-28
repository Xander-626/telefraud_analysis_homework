"""Training and evaluation loops for cached multimodal features."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from teledeceit.metrics import compute_binary_metrics


def feature_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "audio_features": torch.stack([item["audio_features"] for item in batch]),
        "text_features": torch.stack([item["text_features"] for item in batch]),
        "labels": torch.stack([item["label"] for item in batch]),
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
    grad_accum: int = 1,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_items = 0
    use_amp = scaler is not None

    for step, batch in enumerate(dataloader):
        batch = _move_batch(batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = model(
                audio_features=batch["audio_features"],
                text_features=batch["text_features"],
                labels=batch["labels"],
            )
            assert output.loss is not None
            loss = output.loss / grad_accum

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = int(batch["labels"].shape[0])
        total_loss += float(output.loss.detach().cpu().item()) * batch_size * grad_accum
        total_items += batch_size

    return {"loss": total_loss / max(total_items, 1)}


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict[str, float | int]:
    model.eval()
    preds: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    total_loss = 0.0
    total_items = 0

    for batch in dataloader:
        batch = _move_batch(batch, device)
        output = model(
            audio_features=batch["audio_features"],
            text_features=batch["text_features"],
            labels=batch["labels"],
        )
        batch_preds = output.logits.argmax(dim=-1)
        preds.append(batch_preds.cpu())
        labels.append(batch["labels"].cpu())

        if output.loss is not None:
            batch_size = int(batch["labels"].shape[0])
            total_loss += float(output.loss.cpu().item()) * batch_size
            total_items += batch_size

    metrics = compute_binary_metrics(torch.cat(preds), torch.cat(labels))
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
