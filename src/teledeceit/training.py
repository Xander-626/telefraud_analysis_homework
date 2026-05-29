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


@torch.no_grad()
def evaluate_sft_binary(
    model,
    tokenizer,
    dataloader: DataLoader,
    device: torch.device,
    max_new_tokens: int = 200,
) -> dict[str, float | int]:
    """Evaluate SFT model by generating responses and parsing binary fraud labels.

    Uses the prompt_templates.parse_fraud_binary_response() to extract
    is_fraud from generated text, then computes standard binary metrics.
    """
    from teledeceit.metrics import compute_binary_metrics
    from teledeceit.prompt_templates import parse_fraud_binary_response

    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    parse_failures = 0

    for batch in dataloader:
        batch = _move_batch(batch, device)
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        binary_labels = batch.get("binary_label")

        if input_ids is None:
            continue

        # Generate responses
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        # Decode only the new tokens
        prompt_len = input_ids.shape[-1]
        for i in range(generated.shape[0]):
            response_ids = generated[i][prompt_len:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            parsed = parse_fraud_binary_response(response_text)

            if parsed is not None:
                preds.append(parsed)
                if binary_labels is not None:
                    labels.append(int(binary_labels[i].item()))
            else:
                parse_failures += 1

    metrics: dict[str, float | int] = {"parse_failures": parse_failures}
    if preds:
        preds_t = torch.tensor(preds)
        labels_t = torch.tensor(labels)
        metrics.update(compute_binary_metrics(preds_t, labels_t))
    return metrics
