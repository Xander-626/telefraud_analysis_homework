"""SFT LoRA fine-tuning for audio fraud detection using Qwen2.5 + Whisper ASR.

Cascaded pipeline:
  1. Whisper-small transcribes audio to Chinese text (offline, cached)
  2. Text + instruction template -> Qwen2.5-1.5B-Instruct (4-bit QLoRA)
  3. Generate responses -> parse is_fraud -> compute binary metrics

Designed for 6 GB VRAM (laptop RTX 3060).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from teledeceit.metrics import compute_binary_metrics
from teledeceit.prompt_templates import (
    format_fraud_binary_instruction,
    parse_fraud_binary_response,
)
from teledeceit.sft_data import filter_binary_samples, load_sft_samples

# Whisper encoder from existing cache script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_multimodal_features import WhisperAudioEncoder

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SftInstructionDataset(Dataset[dict[str, object]]):
    """Tokenizes instruction-formatted messages for causal LM training."""

    def __init__(
        self,
        samples: list,
        transcriptions: dict[str, str],
        tokenizer,
        max_length: int,
    ) -> None:
        self.input_ids: list[torch.Tensor] = []
        self.attention_masks: list[torch.Tensor] = []
        self.labels_list: list[torch.Tensor] = []
        self.sample_ids: list[str] = []
        self.binary_labels: list[int] = []

        for sample in tqdm(samples, desc="Tokenizing"):
            sid = sample.sample_id
            transcription = transcriptions.get(sid, "")
            if not transcription:
                continue

            messages = format_fraud_binary_instruction(
                transcription, label=sample.binary_label, include_answer=True
            )

            tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                max_length=max_length,
                truncation=True,
            )

            input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)
            labels = input_ids.clone()
            # Mask system+user tokens (loss only on assistant response)
            user_end = _find_assistant_start(messages, tokenizer, input_ids)
            if user_end > 0:
                labels[:user_end] = -100

            self.input_ids.append(input_ids)
            self.attention_masks.append(attention_mask)
            self.labels_list.append(labels)
            self.sample_ids.append(sid)
            self.binary_labels.append(sample.binary_label)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels_list[idx],
            "sample_id": self.sample_ids[idx],
            "binary_label": self.binary_labels[idx],
        }


def _find_assistant_start(messages: list[dict], tokenizer, input_ids: torch.Tensor) -> int:
    """Find the token position where the assistant response begins."""
    # Build assistant-only prompt to find its start
    assistant_msg = [m for m in messages if m["role"] == "assistant"]
    if not assistant_msg:
        return -1
    user_msgs = [m for m in messages if m["role"] != "assistant"]
    prompt_ids = tokenizer.apply_chat_template(
        user_msgs, tokenize=True, add_generation_prompt=True,
    )
    return len(prompt_ids)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def sft_collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    """Pad sequences to max length in batch."""
    max_len = max(int(b["input_ids"].shape[0]) for b in batch)  # type: ignore[union-attr]

    input_ids_list, attention_list, labels_list = [], [], []
    sample_ids = []
    binary_labels = []

    for b in batch:
        cur_len = int(b["input_ids"].shape[0])  # type: ignore[union-attr]
        pad_len = max_len - cur_len
        pad_token_id = 0  # Qwen pad token

        input_ids_list.append(torch.cat([
            b["input_ids"],  # type: ignore[arg-type]
            torch.full((pad_len,), pad_token_id, dtype=torch.long),
        ]))
        attention_list.append(torch.cat([
            b["attention_mask"],  # type: ignore[arg-type]
            torch.zeros(pad_len, dtype=torch.long),
        ]))
        labels_list.append(torch.cat([
            b["labels"],  # type: ignore[arg-type]
            torch.full((pad_len,), -100, dtype=torch.long),
        ]))
        sample_ids.append(b["sample_id"])
        binary_labels.append(b["binary_label"])

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_list),
        "labels": torch.stack(labels_list),
        "sample_id": sample_ids,
        "binary_label": torch.tensor(binary_labels, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Data
    p.add_argument("--train-jsonl", default="data/sft/sft/train.jsonl", type=Path)
    p.add_argument("--test-jsonl", default="data/sft/sft/test.jsonl", type=Path)
    p.add_argument("--data-root", default="data", type=Path)
    p.add_argument("--transcription-cache", default="artifacts/sft_transcriptions.pt", type=Path)
    # Model
    p.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--asr-model", default="openai/whisper-small")
    # LoRA
    p.add_argument("--lora-rank", default=8, type=int)
    p.add_argument("--lora-alpha", default=16, type=int)
    p.add_argument("--lora-dropout", default=0.05, type=float)
    # Training
    p.add_argument("--epochs", default=3, type=int)
    p.add_argument("--batch-size", default=1, type=int)
    p.add_argument("--grad-accum", default=16, type=int)
    p.add_argument("--lr", default=2e-4, type=float)
    p.add_argument("--weight-decay", default=0.01, type=float)
    p.add_argument("--max-grad-norm", default=1.0, type=float)
    p.add_argument("--max-seq-length", default=2048, type=int)
    p.add_argument("--warmup-ratio", default=0.03, type=float)
    # Generation
    p.add_argument("--max-new-tokens", default=200, type=int)
    p.add_argument("--eval-limit", default=200, type=int,
                   help="Max test samples to evaluate per epoch (generation is slow)")
    # Output
    p.add_argument("--output-dir", default=Path("runs/sft_lora_fraud_binary"), type=Path)
    p.add_argument("--limit", default=None, type=int)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--skip-transcription", action="store_true")
    return p.parse_args()


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


@torch.no_grad()
def generate_responses(
    model, tokenizer, dataset: SftInstructionDataset, device: torch.device,
    max_new_tokens: int, eval_limit: int,
) -> list[dict[str, object]]:
    """Generate responses for a subset of evaluation samples."""
    model.eval()
    results: list[dict[str, object]] = []
    indices = range(min(len(dataset), eval_limit))

    for idx in tqdm(indices, desc="Generating"):
        sample = dataset[idx]
        sid = sample["sample_id"]
        binary_label = sample["binary_label"]

        # Extract prompt (non-masked tokens) from the training input
        input_ids = sample["input_ids"]  # type: ignore[arg-type]
        labels_tensor = sample["labels"]  # type: ignore[arg-type]
        prompt_mask = labels_tensor == -100
        if prompt_mask.any():
            first_masked = int(prompt_mask.nonzero(as_tuple=True)[0][0].item())
        else:
            first_masked = len(input_ids)
        prompt_ids = input_ids[:first_masked].unsqueeze(0).to(device)

        with torch.amp.autocast("cuda", enabled=True):
            generated = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        response_ids = generated[0][prompt_ids.shape[-1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        parsed = parse_fraud_binary_response(response_text)

        results.append({
            "sample_id": sid,
            "binary_label": int(binary_label),  # type: ignore[arg-type]
            "prediction": parsed,
            "response_text": response_text,
        })

    return results


def run_transcriptions(args, samples: list, device: torch.device) -> dict[str, str]:
    """Transcribe all audio files and cache to disk."""
    cache_path = Path(args.transcription_cache)
    if cache_path.exists() and not args.limit:
        print(f"Loading cached transcriptions from {cache_path}")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    print("Transcribing audio with Whisper-small...")
    encoder = WhisperAudioEncoder(
        model_name=args.asr_model, device=device,
        max_seconds=30.0, load_for_asr=True,
    )

    transcriptions: dict[str, str] = {}
    batch_size = 2
    for i in tqdm(range(0, len(samples), batch_size), desc="ASR"):
        batch = samples[i : i + batch_size]
        paths = [s.audio_path for s in batch]
        try:
            texts = encoder.transcribe(paths)
            for s, t in zip(batch, texts):
                transcriptions[s.sample_id] = t
        except Exception as e:
            print(f"ASR error at batch {i}: {e}")
            for s in batch:
                transcriptions[s.sample_id] = ""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(transcriptions, ensure_ascii=False), encoding="utf-8")
    print(f"Cached {len(transcriptions)} transcriptions to {cache_path}")

    del encoder
    torch.cuda.empty_cache()
    return transcriptions


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_amp = torch.cuda.is_available() and not args.no_fp16

    # ---- 1. Load SFT data ----
    train_all = load_sft_samples(args.train_jsonl, args.data_root)
    test_all = load_sft_samples(args.test_jsonl, args.data_root)
    train_samples = filter_binary_samples(train_all)
    test_samples = filter_binary_samples(test_all)

    if args.limit is not None:
        train_samples = train_samples[: args.limit]
        test_samples = test_samples[: max(1, args.limit // 4)]

    print(f"Train: {len(train_samples)} binary samples")
    print(f"Test:  {len(test_samples)} binary samples")
    print(f"  Train labels: fraud={sum(1 for s in train_samples if s.binary_label==1)}, "
          f"normal={sum(1 for s in train_samples if s.binary_label==0)}")

    # ---- 2. Transcribe audio (Whisper-small) ----
    all_binary = train_samples + test_samples
    if not args.skip_transcription:
        transcriptions = run_transcriptions(args, all_binary, device)
    else:
        transcriptions = {}

    # ---- 3. Load model and tokenizer ----
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {args.model_name} with 4-bit quantization...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    except torch.cuda.OutOfMemoryError:
        print("OOM with 1.5B model. Falling back to Qwen2.5-0.5B-Instruct...")
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable / 1e6:.2f}M / {total / 1e9:.2f}B ({100*trainable/total:.1f}%)")

    # ---- 4. Datasets and dataloaders ----
    train_dataset = SftInstructionDataset(
        train_samples, transcriptions, tokenizer, args.max_seq_length,
    )
    test_dataset = SftInstructionDataset(
        test_samples, transcriptions, tokenizer, args.max_seq_length,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=sft_collate_fn,
    )

    # ---- 5. Optimizer ----
    import bitsandbytes as bnb

    optimizer = bnb.optim.PagedAdamW8bit(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps,
    )

    # ---- 6. Training loop ----
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_f1 = -1.0
    history: list[dict] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar):
            batch = _move_batch(batch, device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = output.loss / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step <= warmup_steps:
                    lr_scale = global_step / max(warmup_steps, 1)
                    for pg in optimizer.param_groups:
                        pg["lr"] = args.lr * lr_scale
                else:
                    scheduler.step()

            bs = int(batch["labels"].shape[0])
            total_loss += float(output.loss.detach().cpu().item()) * bs
            total_items += bs
            pbar.set_postfix({"loss": f"{float(output.loss.detach().cpu().item()):.4f}"})

        train_loss = total_loss / max(total_items, 1)
        torch.cuda.empty_cache()

        # ---- Evaluation ----
        results = generate_responses(
            model, tokenizer, test_dataset, device, args.max_new_tokens, args.eval_limit,
        )
        valid = [r for r in results if r["prediction"] is not None]
        parse_failures = len(results) - len(valid)

        eval_metrics: dict[str, float | int] = {"parse_failures": parse_failures}
        if valid:
            preds_t = torch.tensor([r["prediction"] for r in valid])
            labels_t = torch.tensor([r["binary_label"] for r in valid])
            eval_metrics.update(compute_binary_metrics(preds_t, labels_t))

        row = {"epoch": epoch, "train_loss": round(train_loss, 6), **eval_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        f1 = float(eval_metrics.get("f1", 0))
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(args.output_dir / "adapter")
            tokenizer.save_pretrained(args.output_dir / "adapter")

            # Use safetensors or torch.save for the adapter
            (args.output_dir / "best_metrics.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8",
            )

        # Print sample predictions
        for r in results[:3]:
            print(f"  [{r['sample_id']}] label={r['binary_label']} pred={r['prediction']} "
                  f"→ {r['response_text'][:80]}")

    # ---- Save history ----
    (args.output_dir / "metrics.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Done. Best F1: {best_f1:.4f}. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
