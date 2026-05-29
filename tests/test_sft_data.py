"""Tests for SFT JSONL data loader."""

from __future__ import annotations

import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from teledeceit.sft_data import (
    LABEL_TO_ID,
    SftSample,
    filter_binary_samples,
    load_sft_samples,
)

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
TRAIN_JSONL = DATA_ROOT / "sft" / "sft" / "train.jsonl"
TEST_JSONL = DATA_ROOT / "sft" / "sft" / "test.jsonl"


class TestSftData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = load_sft_samples(TRAIN_JSONL, DATA_ROOT)
        cls.test = load_sft_samples(TEST_JSONL, DATA_ROOT)

    def test_parses_all_samples(self):
        self.assertEqual(len(self.train), 27146)
        self.assertEqual(len(self.test), 6807)

    def test_task_pattern_distribution(self):
        from collections import Counter
        counts = Counter(s.task_pattern for s in self.train)
        self.assertEqual(counts["SCENE_ONLY"], 10711)
        self.assertEqual(counts["FRAUD_BINARY"], 10711)
        self.assertEqual(counts["FRAUD_TYPE"], 5724)

    def test_filter_binary_samples(self):
        binary = filter_binary_samples(self.train)
        self.assertEqual(len(binary), 16435)
        for s in binary:
            self.assertIn(s.binary_label, (0, 1))
            self.assertIsNotNone(s.binary_label)

    def test_binary_labels_correct(self):
        binary = filter_binary_samples(self.train)
        for s in binary:
            if s.task_pattern == "FRAUD_BINARY":
                ans = s.answers.lower()
                self.assertIn(ans, LABEL_TO_ID)
                self.assertEqual(s.binary_label, LABEL_TO_ID[ans])
            elif s.task_pattern == "FRAUD_TYPE":
                self.assertEqual(s.binary_label, 1)

    def test_audio_paths_exist(self):
        for s in self.train[:50]:
            self.assertTrue(
                s.audio_path.exists(),
                f"Missing: {s.audio_path}",
            )

    def test_samples_have_unique_ids(self):
        ids = [s.sample_id for s in self.train]
        self.assertEqual(len(ids), len(set(ids)))

    def test_test_split_binary(self):
        test_binary = filter_binary_samples(self.test)
        self.assertGreater(len(test_binary), 0)
        fraud_count = sum(1 for s in test_binary if s.binary_label == 1)
        normal_count = sum(1 for s in test_binary if s.binary_label == 0)
        self.assertGreater(fraud_count, 0)
        self.assertGreater(normal_count, 0)


if __name__ == "__main__":
    unittest.main()
