"""Metrics for binary fraud detection."""

from __future__ import annotations

import torch


def compute_binary_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict[str, float | int]:
    preds = preds.detach().cpu().long()
    labels = labels.detach().cpu().long()

    tp = int(((preds == 1) & (labels == 1)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())

    total = max(tp + tn + fp + fn, 1)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
