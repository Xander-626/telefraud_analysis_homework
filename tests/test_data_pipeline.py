import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from teledeceit.data import BinarySample, collate_text_only, load_binary_samples


class DataPipelineTest(unittest.TestCase):
    def test_load_binary_samples_extracts_label_text_and_audio_path(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            data_root = tmp_path / "data"
            audio_file = data_root / "audio" / "call1.mp3"
            audio_file.parent.mkdir(parents=True)
            audio_file.write_bytes(b"fake")

            payload = [
                {
                    "prompt": [
                        {"role": "system", "content": "system prompt"},
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio", "audio_url": "audio/call1.mp3"},
                                {"type": "text", "text": "请判断是否诈骗"},
                            ],
                        },
                    ],
                    "answer": "fraud",
                }
            ]
            json_path = tmp_path / "train.json"
            json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            samples = load_binary_samples(json_path, data_root)

            self.assertEqual(
                samples,
                [
                    BinarySample(
                        sample_id="train-000000",
                        audio_path=audio_file,
                        text="system prompt\n请判断是否诈骗",
                        label=1,
                        answer="fraud",
                    )
                ],
            )

    def test_load_binary_samples_rejects_unknown_labels(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            json_path = tmp_path / "train.json"
            json_path.write_text(json.dumps([{"prompt": [], "answer": "maybe"}]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported label"):
                load_binary_samples(json_path, tmp_path)

    def test_collate_text_only_keeps_batch_order(self) -> None:
        samples = [
            BinarySample("a", Path("a.mp3"), "第一条", 0, "normal"),
            BinarySample("b", Path("b.mp3"), "第二条", 1, "fraud"),
        ]

        batch = collate_text_only(samples)

        self.assertEqual(batch["sample_id"], ["a", "b"])
        self.assertEqual(batch["text"], ["第一条", "第二条"])
        self.assertEqual(batch["labels"].tolist(), [0, 1])


class TemporaryWorkspace:
    def __enter__(self) -> Path:
        import tempfile

        self._tempdir = tempfile.TemporaryDirectory()
        return Path(self._tempdir.name)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._tempdir.cleanup()


if __name__ == "__main__":
    unittest.main()
