import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from teledeceit.metrics import compute_binary_metrics
from teledeceit.model import AudioTextFusionClassifier


class ModelAndMetricsTest(unittest.TestCase):
    def test_fusion_classifier_returns_logits_and_loss(self) -> None:
        model = AudioTextFusionClassifier(audio_dim=4, text_dim=3, hidden_dim=5, dropout=0.0)
        audio = torch.ones(2, 4)
        text = torch.ones(2, 3)
        labels = torch.tensor([0, 1])

        output = model(audio_features=audio, text_features=text, labels=labels)

        self.assertEqual(tuple(output.logits.shape), (2, 2))
        self.assertIsNotNone(output.loss)
        self.assertEqual(output.loss.dim(), 0)

    def test_compute_binary_metrics_reports_core_scores(self) -> None:
        preds = torch.tensor([0, 1, 1, 0])
        labels = torch.tensor([0, 1, 0, 1])

        metrics = compute_binary_metrics(preds, labels)

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)


if __name__ == "__main__":
    unittest.main()
