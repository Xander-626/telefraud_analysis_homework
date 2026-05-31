"""Generate dataset analysis figures for the TeleAntiFraud paper.

Outputs:
  fig_ds1_label_distribution.png   – binary label distribution
  fig_ds2_audio_duration.png       – audio duration histogram by label
  fig_ds3_text_length.png          – prompt text length distribution
  fig_ds4_asr_text_length.png      – ASR transcription length distribution
  fig_ds5_fraud_keywords.png       – top fraud-related keywords
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "report_figures"
TRAIN_JSON = BASE / "data/binary_classification/binary_classification/train.json"
ASR_CACHE = BASE / "artifacts/features/binary_train_whisper_small_asr_roberta.pt"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

C_FRAUD = "#d62728"
C_NORMAL = "#1f77b4"
C_BOTH = "#7f7f7f"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_audio_url(record: dict) -> str:
    for msg in record.get("prompt", []):
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "audio":
                    return str(item.get("audio_url", ""))
    return ""


def _get_prompt_text(record: dict) -> str:
    parts = []
    for msg in record.get("prompt", []):
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    t = item.get("text", "")
                    if isinstance(t, str) and t.strip():
                        parts.append(t.strip())
    return "\n".join(parts)


def _audio_duration_ffprobe(path: Path) -> float | None:
    """Get duration in seconds via ffprobe."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            stderr=subprocess.DEVNULL, timeout=10,
        )
        return float(out.decode().strip())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading training data...")
with open(TRAIN_JSON, encoding="utf-8") as f:
    records = json.load(f)

labels = [r["answer"].strip().lower() for r in records]
fraud_records = [r for r, l in zip(records, labels) if l == "fraud"]
normal_records = [r for r, l in zip(records, labels) if l == "normal"]
print(f"  Fraud: {len(fraud_records)}, Normal: {len(normal_records)}")

# Audio durations
print("Measuring audio durations (ffprobe)...")
fraud_durations = []
normal_durations = []
all_durations = []

for r in records:
    url = _get_audio_url(r)
    if url:
        p = BASE / "data" / url
        if p.exists():
            dur = _audio_duration_ffprobe(p)
            if dur is not None and dur > 0:
                all_durations.append(dur)
                if r["answer"].strip().lower() == "fraud":
                    fraud_durations.append(dur)
                else:
                    normal_durations.append(dur)

print(f"  Measured {len(all_durations)} audio durations")
print(f"  Fraud: {len(fraud_durations)}, Normal: {len(normal_durations)}")

# Text lengths
fraud_text_lens = [len(_get_prompt_text(r)) for r in fraud_records]
normal_text_lens = [len(_get_prompt_text(r)) for r in normal_records]

# ASR transcription lengths (from cache if available)
fraud_asr_lens = []
normal_asr_lens = []
asr_texts_fraud = []
asr_texts_normal = []

if ASR_CACHE.exists():
    print("Loading ASR transcriptions...")
    import torch
    cache = torch.load(ASR_CACHE, map_location="cpu", weights_only=False)
    asr_texts = cache.get("asr_texts", {})
    if asr_texts:
        for r in records:
            # Build sample_id matching the cache script convention
            for msg in r.get("prompt", []):
                content = msg.get("content")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "audio":
                            sid = item.get("audio_url", "")
                            break
            # The cache keys are just the audio_url path
            pass
        # Try to match by index
        if isinstance(asr_texts, list):
            for i, r in enumerate(records):
                if i < len(asr_texts):
                    txt = asr_texts[i]
                    lbl = r["answer"].strip().lower()
                    if lbl == "fraud":
                        fraud_asr_lens.append(len(txt))
                        asr_texts_fraud.append(txt)
                    else:
                        normal_asr_lens.append(len(txt))
                        asr_texts_normal.append(txt)
    print(f"  ASR fraud texts: {len(fraud_asr_lens)}, normal: {len(normal_asr_lens)}")


# ===================================================================
# Fig DS1 – Label Distribution
# ===================================================================
def fig_ds1_label_distribution():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    counts = [normal_records.__len__(), fraud_records.__len__()]
    names = [f"Normal\n({counts[0]})", f"Fraud\n({counts[1]})"]
    colors = [C_NORMAL, C_FRAUD]
    bars = ax.bar(names, counts, color=colors, alpha=0.85, edgecolor="black", lw=1.2)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                str(count), ha="center", fontweight="bold", fontsize=13)
    ax.set_ylabel("Sample Count")
    ax.set_title("Binary Classification Label Distribution (Train Set)", fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.12)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ds1_label_distribution.png")
    plt.close(fig)
    print("  [DS1] Label distribution")


# ===================================================================
# Fig DS2 – Audio Duration Distribution
# ===================================================================
def fig_ds2_audio_duration():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Combined histogram
    bins = np.linspace(0, 40, 41)
    ax1.hist(normal_durations, bins=bins, alpha=0.7, color=C_NORMAL, label=f"Normal (n={len(normal_durations)})", edgecolor="white")
    ax1.hist(fraud_durations, bins=bins, alpha=0.7, color=C_FRAUD, label=f"Fraud (n={len(fraud_durations)})", edgecolor="white")
    ax1.set_xlabel("Duration (seconds)")
    ax1.set_ylabel("Count")
    ax1.set_title("Audio Duration Distribution", fontweight="bold")
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Boxplot
    bp = ax2.boxplot([normal_durations, fraud_durations], labels=["Normal", "Fraud"],
                     patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor(C_NORMAL)
    bp["boxes"][1].set_facecolor(C_FRAUD)
    for box in bp["boxes"]:
        box.set_alpha(0.7)
    ax2.set_ylabel("Duration (seconds)")
    ax2.set_title("Audio Duration Boxplot by Label", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    # Stats annotation
    for label, durations, color in [("Normal", normal_durations, C_NORMAL),
                                     ("Fraud", fraud_durations, C_FRAUD)]:
        if durations:
            print(f"  {label}: mean={np.mean(durations):.1f}s, median={np.median(durations):.1f}s, "
                  f"min={np.min(durations):.1f}s, max={np.max(durations):.1f}s")

    fig.suptitle("Training Audio Duration Analysis", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ds2_audio_duration.png")
    plt.close(fig)
    print("  [DS2] Audio duration")


# ===================================================================
# Fig DS3 – Text Length Distribution
# ===================================================================
def fig_ds3_text_length():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Histogram
    bins = np.linspace(0, 800, 41)
    ax1.hist(normal_text_lens, bins=bins, alpha=0.7, color=C_NORMAL,
             label=f"Normal (mean={np.mean(normal_text_lens):.0f})", edgecolor="white")
    ax1.hist(fraud_text_lens, bins=bins, alpha=0.7, color=C_FRAUD,
             label=f"Fraud (mean={np.mean(fraud_text_lens):.0f})", edgecolor="white")
    ax1.set_xlabel("Character Count")
    ax1.set_ylabel("Count")
    ax1.set_title("Prompt Text Length Distribution", fontweight="bold")
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Boxplot
    bp = ax2.boxplot([normal_text_lens, fraud_text_lens], labels=["Normal", "Fraud"],
                     patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor(C_NORMAL)
    bp["boxes"][1].set_facecolor(C_FRAUD)
    for box in bp["boxes"]:
        box.set_alpha(0.7)
    ax2.set_ylabel("Character Count")
    ax2.set_title("Prompt Text Length Boxplot", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Prompt Text Length Analysis", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ds3_text_length.png")
    plt.close(fig)
    print("  [DS3] Text length")


# ===================================================================
# Fig DS4 – ASR Transcription Length
# ===================================================================
def fig_ds4_asr_text_length():
    if not fraud_asr_lens or not normal_asr_lens:
        print("  [DS4] Skipped – no ASR transcriptions available")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    bins = np.linspace(0, 400, 41)
    ax1.hist(normal_asr_lens, bins=bins, alpha=0.7, color=C_NORMAL,
             label=f"Normal (mean={np.mean(normal_asr_lens):.0f})", edgecolor="white")
    ax1.hist(fraud_asr_lens, bins=bins, alpha=0.7, color=C_FRAUD,
             label=f"Fraud (mean={np.mean(fraud_asr_lens):.0f})", edgecolor="white")
    ax1.set_xlabel("Character Count")
    ax1.set_ylabel("Count")
    ax1.set_title("ASR Transcription Length Distribution", fontweight="bold")
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    bp = ax2.boxplot([normal_asr_lens, fraud_asr_lens], labels=["Normal", "Fraud"],
                     patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor(C_NORMAL)
    bp["boxes"][1].set_facecolor(C_FRAUD)
    for box in bp["boxes"]:
        box.set_alpha(0.7)
    ax2.set_ylabel("Character Count")
    ax2.set_title("ASR Transcription Length Boxplot", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("ASR Transcription Length Analysis", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ds4_asr_text_length.png")
    plt.close(fig)
    print("  [DS4] ASR text length")


# ===================================================================
# Fig DS5 – Fraud Keyword Analysis
# ===================================================================
FRAUD_KEYWORDS = [
    "转账", "安全账户", "验证码", "银行卡", "冻结", "涉嫌", "公检法",
    "贷款", "中奖", "资金", "密码", "账户", "风险", "异常", "解冻",
    "配合", "核查", "保证金", "手续费", "退款", "赔付", "起诉", "逮捕",
    "洗钱", "涉案", "警官", "检察院", "法院", "信用", "征信",
]
NORMAL_KEYWORDS = [
    "快递", "包裹", "您好", "请问", "方便", "谢谢", "再见",
    "通知", "取件", "送餐", "外卖", "预约", "确认", "安排",
]


def fig_ds5_keyword_analysis():
    if not asr_texts_fraud or not asr_texts_normal:
        texts_fraud = [_get_prompt_text(r) for r in fraud_records]
        texts_normal = [_get_prompt_text(r) for r in normal_records]
        print("  [DS5] Using prompt text (no ASR cache available)")
    else:
        texts_fraud = asr_texts_fraud
        texts_normal = asr_texts_normal

    # Count keyword occurrences
    fraud_kw_counts = {}
    normal_kw_counts = {}
    all_fraud_text = " ".join(texts_fraud)
    all_normal_text = " ".join(texts_normal)

    for kw in FRAUD_KEYWORDS:
        fraud_kw_counts[kw] = all_fraud_text.count(kw)
        normal_kw_counts[kw] = all_normal_text.count(kw)

    # Sort by fraud count
    sorted_kws = sorted(FRAUD_KEYWORDS, key=lambda k: fraud_kw_counts[k], reverse=True)
    top_kws = sorted_kws[:15]
    fraud_vals = [fraud_kw_counts[k] for k in top_kws]
    normal_vals = [normal_kw_counts[k] for k in top_kws]

    # Use Chinese-capable font
    plt.rcParams["font.family"] = "SimHei"

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(top_kws))
    width = 0.35
    bars1 = ax.bar(x - width / 2, fraud_vals, width, color=C_FRAUD, alpha=0.85,
                   label="Fraud (n=2000)", edgecolor="white")
    bars2 = ax.bar(x + width / 2, normal_vals, width, color=C_NORMAL, alpha=0.85,
                   label="Normal (n=2000)", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(top_kws, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Occurrence Count")
    ax.set_title("电诈关键词在 ASR 转写文本中的出现频率（按类别分组）", fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ds5_keyword_analysis.png")
    plt.close(fig)

    # Restore default font
    plt.rcParams["font.family"] = "sans-serif"
    print("  [DS5] Keyword analysis")


# ===================================================================
# Main
# ===================================================================
def main():
    print("Generating dataset analysis figures...")
    fig_ds1_label_distribution()
    fig_ds2_audio_duration()
    fig_ds3_text_length()
    fig_ds4_asr_text_length()
    fig_ds5_keyword_analysis()
    print(f"\nDone. Figures saved to: {OUT.resolve()}")
    for f in sorted(OUT.glob("fig_ds*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
