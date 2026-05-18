import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from teledeceit.features import CachedFeatureDataset
from teledeceit.model import AudioTextFusionClassifier
from teledeceit.training import evaluate, feature_collate_fn, train_one_epoch


class TrainingLoopTest(unittest.TestCase):
    def test_train_and_evaluate_on_cached_features(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            cache_path = tmp_path / "features.pt"
            torch.save(
                {
                    "sample_id": ["a", "b", "c", "d"],
                    "audio_features": torch.randn(4, 4),
                    "text_features": torch.randn(4, 3),
                    "labels": torch.tensor([0, 1, 0, 1]),
                },
                cache_path,
            )
            dataset = CachedFeatureDataset(cache_path)
            loader = DataLoader(dataset, batch_size=2, collate_fn=feature_collate_fn)
            model = AudioTextFusionClassifier(4, 3, hidden_dim=6, dropout=0.0)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            train_stats = train_one_epoch(model, loader, optimizer, torch.device("cpu"))
            eval_stats = evaluate(model, loader, torch.device("cpu"))

            self.assertGreaterEqual(train_stats["loss"], 0.0)
            self.assertIn("accuracy", eval_stats)
            self.assertIn("f1", eval_stats)


class TemporaryWorkspace:
    def __enter__(self) -> Path:
        import tempfile

        self._tempdir = tempfile.TemporaryDirectory()
        return Path(self._tempdir.name)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._tempdir.cleanup()


if __name__ == "__main__":
    unittest.main()
