import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.serve_demo import build_api_response
from teledeceit.demo_backend import (
    DEMO_MODEL_NAME,
    build_text_prediction,
    list_demo_samples,
    predict_demo_sample,
)


class DemoBackendTest(unittest.TestCase):
    def test_lists_normal_and_fraud_samples_for_frontend(self):
        samples = list_demo_samples()
        labels = {sample["expected_label"] for sample in samples}

        self.assertGreaterEqual(len(samples), 3)
        self.assertIn("normal", labels)
        self.assertIn("fraud", labels)
        self.assertTrue(all("sample_id" in sample for sample in samples))
        self.assertTrue(all("audio_url" in sample for sample in samples))

    def test_predict_demo_sample_returns_stable_detection_payload(self):
        result = predict_demo_sample("fraud_example_1")

        self.assertEqual(result["sample_id"], "fraud_example_1")
        self.assertEqual(result["prediction"], "fraud")
        self.assertEqual(result["risk_level"], "high")
        self.assertGreater(result["fraud_probability"], 0.9)
        self.assertEqual(result["model"], DEMO_MODEL_NAME)
        self.assertGreaterEqual(len(result["evidence"]), 2)

    def test_unknown_demo_sample_raises_key_error(self):
        with self.assertRaises(KeyError):
            predict_demo_sample("missing")

    def test_text_prediction_flags_fraud_language(self):
        result = build_text_prediction("请马上转账到安全账户，否则你的银行卡会被冻结")

        self.assertEqual(result["prediction"], "fraud")
        self.assertEqual(result["risk_level"], "high")
        self.assertGreater(result["fraud_probability"], 0.75)
        self.assertIn("安全账户", result["evidence"])

    def test_text_prediction_handles_normal_language(self):
        result = build_text_prediction("您好，这里是快递员，您的包裹已经放到门口")

        self.assertEqual(result["prediction"], "normal")
        self.assertEqual(result["risk_level"], "low")
        self.assertLess(result["fraud_probability"], 0.5)

    def test_empty_text_prediction_rejects_input(self):
        with self.assertRaises(ValueError):
            build_text_prediction("   ")


class DemoServerHelperTest(unittest.TestCase):
    def test_build_api_response_lists_samples(self):
        status, payload = build_api_response("GET", "/api/demo/samples", b"")

        self.assertEqual(status, 200)
        self.assertIn("samples", payload)
        self.assertGreaterEqual(len(payload["samples"]), 3)

    def test_build_api_response_predicts_sample(self):
        status, payload = build_api_response(
            "POST",
            "/api/demo/predict-sample",
            b'{"sample_id": "normal_example"}',
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["prediction"], "normal")

    def test_build_api_response_rejects_missing_text(self):
        status, payload = build_api_response("POST", "/api/predict", b'{"text": ""}')

        self.assertEqual(status, 400)
        self.assertIn("error", payload)


if __name__ == "__main__":
    unittest.main()
