"""Instruction prompt templates and response parsing for fraud binary SFT."""

from __future__ import annotations

import json
import re


SYSTEM_PROMPT = "你是一个专业的电话诈骗检测助手。你需要根据通话内容的文字记录，判断该通话是否涉及诈骗。"

USER_TEMPLATE = """请分析以下通话记录，判断是否涉及诈骗：

通话记录：
{transcription}

请严格按以下JSON格式输出：
```json
{{
  "is_fraud": true或false,
  "reason": "判断理由",
  "confidence": 0.0到1.0之间的置信度
}}
```"""

ASSISTANT_TEMPLATE = """```json
{{
  "is_fraud": {is_fraud},
  "reason": "{reason}",
  "confidence": {confidence}
}}
```"""


def format_fraud_binary_instruction(
    transcription: str,
    label: int | None = None,
    include_answer: bool = True,
) -> list[dict[str, str]]:
    """Build messages list for Qwen chat template.

    Args:
        transcription: ASR text of the call.
        label: 0=normal, 1=fraud. Only used when include_answer=True.
        include_answer: If True, append assistant response for training.

    Returns:
        List of message dicts with "role" and "content" keys.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(transcription=transcription)},
    ]

    if include_answer and label is not None:
        is_fraud_str = "true" if label == 1 else "false"
        # Generate a brief reason based on the label
        if label == 1:
            reason = "通话内容包含诈骗话术特征"
        else:
            reason = "通话内容未发现诈骗迹象"
        confidence = 0.95
        messages.append(
            {
                "role": "assistant",
                "content": ASSISTANT_TEMPLATE.format(
                    is_fraud=is_fraud_str,
                    reason=reason,
                    confidence=confidence,
                ),
            }
        )

    return messages


def parse_fraud_binary_response(text: str) -> int | None:
    """Extract is_fraud boolean from generated text.

    Returns:
        1 for fraud, 0 for normal, None if parsing fails.
    """
    # Strategy 1: Try to parse a JSON block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not json_match:
        # Also try bare JSON object
        json_match = re.search(r"\{[^{}]*\"is_fraud\"[^{}]*\}", text, re.DOTALL)

    if json_match:
        try:
            obj = json.loads(json_match.group(1))
            is_fraud = obj.get("is_fraud")
            if isinstance(is_fraud, bool):
                return 1 if is_fraud else 0
            if isinstance(is_fraud, str):
                return 1 if is_fraud.lower() == "true" else 0
        except (json.JSONDecodeError, KeyError):
            pass

    # Strategy 2: Regex for "is_fraud": true/false
    kv_match = re.search(r'"is_fraud"\s*:\s*(true|false)', text, re.IGNORECASE)
    if kv_match:
        return 1 if kv_match.group(1).lower() == "true" else 0

    # Strategy 3: Keyword fallback
    # Remove code blocks to avoid re-matching them
    text_clean = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Negation: "无诈骗", "不涉及诈骗", "没有欺诈", "没有发现欺诈" etc.
    has_negation = bool(re.search(
        r"(无|没有|未发现|不涉及|非|不(属于|构成|是)|没发现|不存在).{0,6}(诈骗|欺诈|涉诈)",
        text_clean,
    ))
    # Positive fraud assertion, guarded against negation prefix
    has_fraud_assertion = bool(re.search(
        r"(?<![无没不非])(涉及|确认为|判定为|判断为|认定为|属于|构成|属于).{0,8}(诈骗|欺诈|涉诈)",
        text_clean,
    ))
    has_fraud_keyword = bool(re.search(r"(诈骗|欺诈|涉诈)", text_clean))
    has_normal_keyword = bool(re.search(r"(正常通话|安全通话|合规|未发现异常|无异常)", text_clean))

    if has_negation and not has_fraud_assertion:
        return 0
    if has_fraud_assertion:
        return 1
    if has_fraud_keyword and not has_normal_keyword:
        return 1
    if has_normal_keyword and not has_fraud_keyword:
        return 0

    return None
