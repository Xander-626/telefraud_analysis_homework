from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DemoFrontendStructureTest(unittest.TestCase):
    def test_index_contains_required_regions(self):
        html = (ROOT / "web" / "demo" / "index.html").read_text(encoding="utf-8")

        self.assertIn("TeleAntiFraud 电诈检测演示工作台", html)
        self.assertIn('data-tab="samples"', html)
        self.assertIn('data-tab="upload"', html)
        self.assertIn('data-tab="text"', html)
        self.assertIn("result-panel", html)
        self.assertIn("Whisper-small", html)

    def test_javascript_calls_expected_api_routes(self):
        script = (ROOT / "web" / "demo" / "app.js").read_text(encoding="utf-8")

        self.assertIn("/api/demo/samples", script)
        self.assertIn("/api/demo/predict-sample", script)
        self.assertIn("/api/predict", script)
        self.assertIn("renderResult", script)

    def test_styles_include_projection_friendly_layout(self):
        css = (ROOT / "web" / "demo" / "styles.css").read_text(encoding="utf-8")

        self.assertIn(".workspace", css)
        self.assertIn(".risk-high", css)
        self.assertIn("@media", css)


if __name__ == "__main__":
    unittest.main()
