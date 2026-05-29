"""Deterministic demo data and lightweight prediction helpers for the frontend."""

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
    """Return sample metadata for rendering the frontend sample list."""

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
    """Return the deterministic prediction payload for a preset demo sample."""

    for sample in _SAMPLES:
        if sample["sample_id"] == sample_id:
            result = deepcopy(sample)
            result["model"] = DEMO_MODEL_NAME
            return result
    raise KeyError(sample_id)


def build_text_prediction(text: str) -> dict[str, Any]:
    """Score text input with a lightweight, explainable demo heuristic."""

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
