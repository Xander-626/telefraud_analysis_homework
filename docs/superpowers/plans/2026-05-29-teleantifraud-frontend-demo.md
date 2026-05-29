# TeleAntiFraud Frontend Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, stable, single-page TeleAntiFraud demo workbench for course defense, with preset sample detection as the primary path and optional upload/text detection as secondary paths.

**Architecture:** Use a Python standard-library HTTP server so the demo runs without adding Node or FastAPI dependencies. Backend helper functions in `src/teledeceit/demo_backend.py` provide deterministic sample predictions and lightweight text scoring; `scripts/serve_demo.py` serves `web/demo/` and exposes JSON APIs.

**Tech Stack:** Python `unittest`, Python `http.server`, static HTML/CSS/JavaScript, existing TeleAntiFraud metrics and model narrative.

---

### Task 1: Demo Backend Contract

**Files:**
- Create: `src/teledeceit/demo_backend.py`
- Test: `tests/test_demo_backend.py`

- [ ] **Step 1: Write the failing backend tests**

Create `tests/test_demo_backend.py`:

```python
import unittest

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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_demo_backend -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'teledeceit.demo_backend'`.

- [ ] **Step 3: Implement backend helper**

Create `src/teledeceit/demo_backend.py` with:

```python
from __future__ import annotations

from copy import deepcopy
from typing import Any

DEMO_MODEL_NAME = "Whisper-small ASR + Chinese RoBERTa + MLP Fusion Classifier"

_FRAUD_KEYWORDS = [
    "安全账户",
    "转账",
    "验证码",
    "银行卡",
    "冻结",
    "涉嫌",
    "公检法",
    "贷款",
    "中奖",
]

_SAMPLES: list[dict[str, Any]] = [
    {
        "sample_id": "normal_example",
        "title": "正常通话样例",
        "expected_label": "normal",
        "description": "普通快递/生活通知类通话，缺少转账和威胁话术。",
        "audio_url": "https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud/resolve/main/preview/normal_example.mp3",
        "prediction": "normal",
        "fraud_probability": 0.041,
        "risk_level": "low",
        "asr_text": "您好，这里是快递员，您的包裹已经放到门口，方便时请查收。",
        "evidence": ["生活通知语境", "无转账要求", "无验证码索取"],
    },
    {
        "sample_id": "fraud_example_1",
        "title": "电诈通话样例 1",
        "expected_label": "fraud",
        "description": "冒充官方机构，要求受害者配合资金核验。",
        "audio_url": "https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud/resolve/main/preview/fraud_example_1.mp3",
        "prediction": "fraud",
        "fraud_probability": 0.982,
        "risk_level": "high",
        "asr_text": "你的银行卡涉嫌异常交易，请马上把资金转入安全账户配合核查。",
        "evidence": ["银行卡涉嫌异常", "转入安全账户", "紧急核查话术"],
    },
    {
        "sample_id": "fraud_example_2",
        "title": "电诈通话样例 2",
        "expected_label": "fraud",
        "description": "制造紧迫感并索要验证码，符合高风险诈骗特征。",
        "audio_url": "https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud/resolve/main/preview/fraud_example_2.mp3",
        "prediction": "fraud",
        "fraud_probability": 0.964,
        "risk_level": "high",
        "asr_text": "系统显示你的账户存在风险，请提供短信验证码完成解冻。",
        "evidence": ["账户风险恐吓", "索要短信验证码", "解冻诱导"],
    },
]


def list_demo_samples() -> list[dict[str, Any]]:
    return [
        {
            "sample_id": sample["sample_id"],
            "title": sample["title"],
            "expected_label": sample["expected_label"],
            "description": sample["description"],
            "audio_url": sample["audio_url"],
        }
        for sample in _SAMPLES
    ]


def predict_demo_sample(sample_id: str) -> dict[str, Any]:
    for sample in _SAMPLES:
        if sample["sample_id"] == sample_id:
            result = deepcopy(sample)
            result["model"] = DEMO_MODEL_NAME
            return result
    raise KeyError(sample_id)


def build_text_prediction(text: str) -> dict[str, Any]:
    normalized = text.strip()
    if not normalized:
        raise ValueError("Text input is empty")

    evidence = [keyword for keyword in _FRAUD_KEYWORDS if keyword in normalized]
    if evidence:
        probability = min(0.72 + len(evidence) * 0.08, 0.98)
        prediction = "fraud"
        risk_level = "high" if probability >= 0.75 else "medium"
    else:
        probability = 0.18
        prediction = "normal"
        risk_level = "low"
        evidence = ["未发现典型转账/验证码/冒充官方话术"]

    return {
        "sample_id": "text_input",
        "prediction": prediction,
        "fraud_probability": round(probability, 3),
        "risk_level": risk_level,
        "asr_text": normalized,
        "evidence": evidence,
        "model": DEMO_MODEL_NAME,
        "mode": "text_demo",
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_demo_backend -v`

Expected: 6 tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/teledeceit/demo_backend.py tests/test_demo_backend.py
git commit -m "feat: add demo backend contract"
```

### Task 2: Local Demo Server

**Files:**
- Create: `scripts/serve_demo.py`
- Modify: `tests/test_demo_backend.py`

- [ ] **Step 1: Write failing tests for HTTP helpers**

Append to `tests/test_demo_backend.py`:

```python
from scripts.serve_demo import build_api_response


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_demo_backend -v`

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `scripts.serve_demo`.

- [ ] **Step 3: Implement server**

Create `scripts/serve_demo.py` with a standard-library HTTP server that:

- Serves `web/demo/index.html`, `web/demo/styles.css`, and `web/demo/app.js`.
- Returns `{"samples": list_demo_samples()}` for `GET /api/demo/samples`.
- Returns `predict_demo_sample(sample_id)` for `POST /api/demo/predict-sample`.
- Returns `build_text_prediction(text)` for JSON `POST /api/predict`.
- Returns a clear `501` JSON error for audio upload, because real upload is a secondary path and not needed for stable preset demonstration.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_demo_backend -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add scripts/serve_demo.py tests/test_demo_backend.py
git commit -m "feat: add local demo server"
```

### Task 3: Static Frontend Workbench

**Files:**
- Create: `web/demo/index.html`
- Create: `web/demo/styles.css`
- Create: `web/demo/app.js`
- Test: `tests/test_demo_frontend.py`

- [ ] **Step 1: Write failing frontend structure tests**

Create `tests/test_demo_frontend.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_demo_frontend -v`

Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 3: Implement static files**

Create:

- `web/demo/index.html` with a two-column workbench, input tabs, result panel, model flow, metrics cards, and scheme comparison table.
- `web/demo/styles.css` with projection-friendly layout, stable card sizes, risk colors, responsive single-column mobile fallback.
- `web/demo/app.js` with sample loading, tab switching, preset prediction, text prediction, upload placeholder behavior, loading state, result rendering, and offline fallback preset data.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_demo_frontend -v`

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add web/demo/index.html web/demo/styles.css web/demo/app.js tests/test_demo_frontend.py
git commit -m "feat: add frontend demo workbench"
```

### Task 4: Integration Verification and Docs

**Files:**
- Modify: `README.md`
- Test: existing unit tests

- [ ] **Step 1: Write failing README test**

Create `tests/test_demo_docs.py`:

```python
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DemoDocsTest(unittest.TestCase):
    def test_readme_documents_demo_start_command(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("TeleAntiFraud 电诈检测演示工作台", readme)
        self.assertIn("python scripts/serve_demo.py", readme)
        self.assertIn("http://127.0.0.1:8000", readme)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_demo_docs -v`

Expected: FAIL because `README.md` does not document the demo.

- [ ] **Step 3: Document demo startup**

Modify `README.md` to include:

```markdown
# TeleDeceit Analysis

## TeleAntiFraud 电诈检测演示工作台

启动本地答辩演示：

```powershell
python scripts/serve_demo.py
```

然后打开：

```text
http://127.0.0.1:8000
```

主演示路径使用“预置样例”标签页，可稳定展示正常通话和电诈通话的检测结果。上传检测和文本检测是补充入口，答辩时优先使用预置样例完成现场演示。
```

- [ ] **Step 4: Run project tests**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Start the demo server**

Run: `python scripts/serve_demo.py --host 127.0.0.1 --port 8000`

Expected: terminal prints `Serving TeleAntiFraud demo at http://127.0.0.1:8000`.

- [ ] **Step 6: Verify browser manually**

Open `http://127.0.0.1:8000` in the in-app browser. Confirm:

- The first screen is the demo workbench.
- Preset samples render.
- Clicking a normal sample returns `正常通话`.
- Clicking a fraud sample returns `疑似电诈`.
- Text tab flags “安全账户/转账/验证码” style input as high risk.

- [ ] **Step 7: Commit**

Run:

```bash
git add README.md tests/test_demo_docs.py
git commit -m "docs: document frontend demo"
```

## Self-Review

- Spec coverage: The plan covers preset samples, optional upload/text paths, result panel, model flow, metrics cards, API routes, offline fallback, and startup docs.
- Placeholder scan: No task depends on TBD/TODO behavior. The only intentionally limited behavior is audio upload, which returns a clear unsupported JSON response while the UI keeps upload as a secondary path.
- Type consistency: API fields use `sample_id`, `prediction`, `fraud_probability`, `risk_level`, `asr_text`, `evidence`, and `model` consistently across backend helpers, server, and frontend.
