"""Evaluate SFT LoRA adapter on SFT test split (binary fraud detection)."""
from __future__ import annotations

import json, sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from teledeceit.sft_data import filter_binary_samples, load_sft_samples
from teledeceit.prompt_templates import format_fraud_binary_instruction, parse_fraud_binary_response
from teledeceit.metrics import compute_binary_metrics

ADAPTER = "runs/sft_lora_fraud_binary/adapter"
CACHE = "artifacts/sft_transcriptions.pt"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TEST_JSONL = "data/sft/sft/test.jsonl"
DATA_ROOT = "data"
MAX_SAMPLES = 200
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")
print(f"Loading model: {MODEL}")

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
    device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()
print("Model loaded.")

samples = filter_binary_samples(load_sft_samples(TEST_JSONL, DATA_ROOT))[:MAX_SAMPLES]
cache = json.loads(Path(CACHE).read_text(encoding="utf-8"))
print(f"Evaluating {len(samples)} samples...")

preds, labels, parse_fail = [], [], 0
for i, s in enumerate(samples):
    trans = cache.get(s.sample_id, "")
    if not trans:
        parse_fail += 1
        continue

    msgs = format_fraud_binary_instruction(trans, label=None, include_answer=False)
    input_ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                              return_tensors="pt")
    # apply_chat_template returns a dict-like BatchEncoding when return_tensors="pt"
    if hasattr(input_ids, "input_ids"):
        input_ids = input_ids["input_ids"]
    input_ids = input_ids.to(device)
    gen = model.generate(input_ids, max_new_tokens=200, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    resp = tokenizer.decode(gen[0][input_ids.shape[-1]:], skip_special_tokens=True)
    pred = parse_fraud_binary_response(resp)

    if pred is None:
        parse_fail += 1
    else:
        preds.append(pred)
        labels.append(s.binary_label)

    label = s.binary_label
    status = "Y" if pred == label else ("N" if pred is not None else "?")
    print(f"[{i:03d}] label={label} pred={pred} {status} | {resp[:80]}")

metrics = compute_binary_metrics(torch.tensor(preds), torch.tensor(labels))
metrics["n_samples"] = len(samples)
metrics["parse_failures"] = parse_fail
print()
for k, v in metrics.items():
    print(f"  {k}: {v}")
