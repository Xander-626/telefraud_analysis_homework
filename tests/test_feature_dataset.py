import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from teledeceit.features import CachedFeatureDataset


class CachedFeatureDatasetTest(unittest.TestCase):
    def test_cached_feature_dataset_loads_rows(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            cache_path = tmp_path / "features.pt"
            torch.save(
                {
                    "sample_id": ["a", "b"],
                    "audio_features": torch.ones(2, 4),
                    "text_features": torch.zeros(2, 3),
                    "labels": torch.tensor([0, 1]),
                },
                cache_path,
            )

            dataset = CachedFeatureDataset(cache_path)
            row = dataset[1]

            self.assertEqual(len(dataset), 2)
            self.assertEqual(row["sample_id"], "b")
            self.assertEqual(tuple(row["audio_features"].shape), (4,))
            self.assertEqual(tuple(row["text_features"].shape), (3,))
            self.assertEqual(row["label"].item(), 1)


class TemporaryWorkspace:
    def __enter__(self) -> Path:
        import tempfile

        self._tempdir = tempfile.TemporaryDirectory()
        return Path(self._tempdir.name)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._tempdir.cleanup()


if __name__ == "__main__":
    unittest.main()
