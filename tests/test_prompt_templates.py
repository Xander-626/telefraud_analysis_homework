"""Tests for instruction prompt templates and response parsing."""

from __future__ import annotations

import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from teledeceit.prompt_templates import (
    format_fraud_binary_instruction,
    parse_fraud_binary_response,
)


class TestPromptTemplates(unittest.TestCase):
    def test_format_with_answer(self):
        msgs = format_fraud_binary_instruction(
            "测试通话内容", label=1, include_answer=True,
        )
        self.assertEqual(len(msgs), 3)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(msgs[2]["role"], "assistant")
        self.assertIn("is_fraud", msgs[2]["content"])

    def test_format_without_answer(self):
        msgs = format_fraud_binary_instruction(
            "测试通话", label=None, include_answer=False,
        )
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")

    def test_format_fraud_label(self):
        msgs = format_fraud_binary_instruction("x", label=1, include_answer=True)
        self.assertIn('"is_fraud": true', msgs[2]["content"])

    def test_format_normal_label(self):
        msgs = format_fraud_binary_instruction("x", label=0, include_answer=True)
        self.assertIn('"is_fraud": false', msgs[2]["content"])

    def test_parse_json_block_true(self):
        text = '```json\n{"is_fraud": true, "reason": "test", "confidence": 0.9}\n```'
        self.assertEqual(parse_fraud_binary_response(text), 1)

    def test_parse_json_block_false(self):
        text = '```json\n{"is_fraud": false, "reason": "ok", "confidence": 0.8}\n```'
        self.assertEqual(parse_fraud_binary_response(text), 0)

    def test_parse_keyvalue_true(self):
        self.assertEqual(parse_fraud_binary_response('"is_fraud": true'), 1)

    def test_parse_keyvalue_false(self):
        self.assertEqual(parse_fraud_binary_response('"is_fraud": false'), 0)

    def test_parse_fraud_keyword(self):
        self.assertEqual(parse_fraud_binary_response("该通话涉及诈骗"), 1)

    def test_parse_negation_normal(self):
        self.assertEqual(parse_fraud_binary_response("无诈骗迹象，通话正常"), 0)
        self.assertEqual(parse_fraud_binary_response("该通话不涉及诈骗"), 0)
        self.assertEqual(parse_fraud_binary_response("没有发现欺诈行为"), 0)

    def test_parse_unknown(self):
        self.assertIsNone(parse_fraud_binary_response("今天天气很好"))

    def test_parse_fraud_assertion(self):
        self.assertEqual(parse_fraud_binary_response("确认为诈骗电话"), 1)
        self.assertEqual(parse_fraud_binary_response("该通话存在明显欺诈行为"), 1)


if __name__ == "__main__":
    unittest.main()
