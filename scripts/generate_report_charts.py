"""Generate 8 publication-quality charts for the TeleAntiFraud project report."""

from __future__ import annotations

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "report_figures"
OUT.mkdir(exist_ok=True)

# ---- matplotlib global config ----
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

# ---- colour palette ----
C0 = "#1f77b4"   # blue     Frozen MLP (prompt)
C1 = "#ff7f0e"   # orange   Frozen MLP (ASR)
C2 = "#2ca02c"   # green    E2E Unfreeze
C3 = "#d62728"   # red      SFT LoRA
GREY = "#7f7f7f"

# =====================================================================
#  Hard-coded metrics  (read from the real JSON files at runtime)
# =====================================================================

def _load(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

PROMPT = _load("runs/binary_fusion_whisper_small_roberta/metrics.json")
ASR    = _load("runs/binary_fusion_whisper_small_asr_roberta/metrics.json")
E2E    = _load("runs/binary_e2e_asr/metrics.json")
SFT    = _load("results/runs/sft_lora_fraud_binary/metrics.json")

BEST = {
    "prompt": {"f1": 1.0, "acc": 1.0, "prec": 1.0, "rec": 1.0, "fn": 0, "fp": 0,
               "tp": 200, "tn": 200, "size": 1.6, "time": "~5 min", "epoch": 16},
    "asr":    {"f1": 1.0, "acc": 1.0, "prec": 1.0, "rec": 1.0, "fn": 0, "fp": 0,
               "tp": 200, "tn": 200, "size": 1.6, "time": "~5 min", "epoch": 11},
    "e2e":    {"f1": 0.995, "acc": 0.995, "prec": 1.0, "rec": 0.99, "fn": 2, "fp": 0,
               "tp": 198, "tn": 200, "size": 338, "time": "~3 h", "epoch": 4},
    "sft":    {"f1": 0.993, "acc": 0.99, "prec": 1.0, "rec": 0.986, "fn": 2, "fp": 0,
               "tp": 141, "tn": 57, "size": 35, "time": "~6.5 h", "epoch": 3},
}


# =====================================================================
#  Fig 1  –  System architecture pipeline
# =====================================================================
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("System Architecture: Whisper ASR + Qwen2.5 QLoRA Cascaded Pipeline",
                 fontweight="bold", pad=12)

    boxes = [
        (0.3, "Audio\n(.mp3)", C0),
        (2.8, "Whisper-small\nASR Transcription", C1),
        (5.3, "Chinese\nCall Transcript", GREY),
        (7.8, "Qwen2.5-1.5B-Instruct\n4-bit QLoRA (rank=8)", C3),
        (10.3, "JSON Output\n{is_fraud, reason,\nconfidence}", C2),
    ]
    for x, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, 1.5), 1.8, 1.2, facecolor=color, alpha=0.15,
                                    edgecolor=color, lw=2, zorder=2))
        ax.text(x + 0.9, 2.1, label, ha="center", va="center", fontsize=9,
                fontweight="bold", zorder=3)

    # arrows
    for x in [2.1, 4.6, 7.1, 9.6]:
        ax.annotate("", xy=(x + 0.4, 2.1), xytext=(x, 2.1),
                    arrowprops=dict(arrowstyle="->", color="black", lw=2))

    # annotation below
    ax.text(6, 0.4, "Training config: batch_size=1, grad_accum=8, lr=2e-4, cosine schedule, paged_adamw_8bit",
            ha="center", fontsize=8, style="italic", color=GREY)
    fig.savefig(OUT / "fig1_architecture.png")
    plt.close(fig)


# =====================================================================
#  Fig 2  –  Test loss convergence
# =====================================================================
def fig2_loss_convergence():
    fig, ax = plt.subplots(figsize=(9, 5))

    def plot_exp(data, label, color, marker):
        epochs = [d["epoch"] for d in data]
        loss = [d["loss"] for d in data]
        ax.plot(epochs, loss, color=color, marker=marker, lw=2, ms=5, label=label)

    plot_exp(PROMPT, "Frozen MLP (prompt text)", C0, "o")
    plot_exp(ASR,    "Frozen MLP (ASR text)", C1, "s")
    plot_exp(E2E,    "E2E Unfreeze 2 layers", C2, "^")

    ax.set_xlabel("Epoch"); ax.set_ylabel("Test Loss")
    ax.set_title("Test Loss Convergence Across Training Approaches", fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 20.5)
    fig.savefig(OUT / "fig2_loss_convergence.png")
    plt.close(fig)


# =====================================================================
#  Fig 3  –  F1 evolution
# =====================================================================
def fig3_f1_evolution():
    fig, ax = plt.subplots(figsize=(9, 5))

    def plot_f1(data, label, color, marker, best_epoch):
        epochs = [d["epoch"] for d in data]
        f1 = [d["f1"] for d in data]
        ax.plot(epochs, f1, color=color, marker=marker, lw=2, ms=5, label=label)
        # mark best
        idx = next(i for i, d in enumerate(data) if d["epoch"] == best_epoch)
        ax.scatter([best_epoch], [f1[idx]], color=color, s=120, zorder=5, edgecolors="black", lw=1)
        ax.annotate(f"Epoch {best_epoch}\nF1={f1[idx]:.3f}",
                    (best_epoch, f1[idx]), textcoords="offset points", xytext=(10, -15),
                    fontsize=8, color=color, fontweight="bold")

    plot_f1(PROMPT, "Frozen MLP (prompt)", C0, "o", 16)
    plot_f1(ASR,    "Frozen MLP (ASR)", C1, "s", 11)
    plot_f1(E2E,    "E2E Unfreeze", C2, "^", 4)

    ax.set_xlabel("Epoch"); ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score Evolution (Test Set)", fontweight="bold")
    ax.legend(framealpha=0.9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.78, 1.02)
    ax.set_xlim(0.5, 20.5)
    fig.savefig(OUT / "fig3_f1_evolution.png")
    plt.close(fig)


# =====================================================================
#  Fig 4  –  Four-approach metrics comparison
# =====================================================================
def fig4_approach_comparison():
    labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    approaches = ["Frozen MLP\n(prompt)", "Frozen MLP\n(ASR)", "E2E\nUnfreeze", "SFT LoRA\n(Qwen2.5)"]
    colors = [C0, C1, C2, C3]

    data = {
        "Frozen MLP\n(prompt)": [BEST["prompt"]["acc"], BEST["prompt"]["prec"],
                                  BEST["prompt"]["rec"], BEST["prompt"]["f1"]],
        "Frozen MLP\n(ASR)":    [BEST["asr"]["acc"], BEST["asr"]["prec"],
                                  BEST["asr"]["rec"], BEST["asr"]["f1"]],
        "E2E\nUnfreeze":        [BEST["e2e"]["acc"], BEST["e2e"]["prec"],
                                  BEST["e2e"]["rec"], BEST["e2e"]["f1"]],
        "SFT LoRA\n(Qwen2.5)":  [BEST["sft"]["acc"], BEST["sft"]["prec"],
                                  BEST["sft"]["rec"], BEST["sft"]["f1"]],
    }

    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, (app, vals) in enumerate(data.items()):
        ax.bar(x + i * width, vals, width, color=colors[i], alpha=0.85, label=app)

    ax.set_ylabel("Score")
    ax.set_title("Best Metrics by Approach", fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels)
    ax.legend(framealpha=0.9, fontsize=8)
    ax.set_ylim(0.94, 1.005)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(OUT / "fig4_approach_comparison.png")
    plt.close(fig)


# =====================================================================
#  Fig 5  –  Model size vs performance
# =====================================================================
def fig5_size_performance():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    items = [
        ("Frozen MLP\n(prompt)", 1.6, 1.0, 5/60),
        ("Frozen MLP\n(ASR)", 1.6, 1.0, 5/60),
        ("E2E Unfreeze\n2 layers", 338, 0.995, 3),
        ("SFT LoRA\n(Qwen2.5)", 35, 0.993, 6.5),
    ]
    sizes = [s for _, s, _, _ in items]
    f1s = [f for _, _, f, _ in items]
    times = [t for _, _, _, t in items]
    labels = [l for l, _, _, _ in items]
    cs = [C0, C1, C2, C3]

    for i in range(len(items)):
        ax.scatter(sizes[i], f1s[i], s=times[i]*350, color=cs[i], alpha=0.7,
                   edgecolors="black", lw=1, zorder=5)
        offset = 15 if i != 3 else 15
        ax.annotate(labels[i], (sizes[i], f1s[i]),
                    textcoords="offset points", xytext=(offset, -5 if i < 2 else 8),
                    fontsize=8, fontweight="bold", color=cs[i])

    ax.set_xscale("log")
    ax.set_xlabel("Model Size (MB, log scale)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Model Size vs. Performance (bubble size = training time)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.985, 1.003)
    # legend for bubble size
    for h, label in [(5/60*350, "5 min"), (3*350, "3 h"), (6.5*350, "6.5 h")]:
        ax.scatter([], [], s=h, color=GREY, alpha=0.4, edgecolors="black",
                   lw=0.5, label=label)
    ax.legend(title="Training Time", fontsize=8, title_fontsize=9)
    fig.savefig(OUT / "fig5_size_performance.png")
    plt.close(fig)


# =====================================================================
#  Fig 6  –  Confusion matrices (4 subplots)
# =====================================================================
def fig6_confusion_matrices():
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes = axes.flatten()

    cm_data = [
        ("Frozen MLP (prompt)", BEST["prompt"]),
        ("Frozen MLP (ASR)", BEST["asr"]),
        ("E2E Unfreeze", BEST["e2e"]),
        ("SFT LoRA (Qwen2.5)", BEST["sft"]),
    ]

    for ax, (title, b) in zip(axes, cm_data):
        cm = np.array([[b["tn"], b["fp"]],
                        [b["fn"], b["tp"]]], dtype=float)
        im = ax.imshow(cm, cmap="YlOrRd", vmin=0, vmax=200)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Normal", "Pred Fraud"])
        ax.set_yticklabels(["True Normal", "True Fraud"])
        for i in range(2):
            for j in range(2):
                val = int(cm[i, j])
                color = "white" if cm[i, j] > 100 else "black"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=16,
                        fontweight="bold", color=color)
        # ticks already set above

    fig.suptitle("Confusion Matrices (Test Set)", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_confusion_matrices.png")
    plt.close(fig)


# =====================================================================
#  Fig 7  –  SFT data distribution
# =====================================================================
def fig7_sft_data():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Task pattern
    patterns = ["SCENE_ONLY", "FRAUD_BINARY", "FRAUD_TYPE"]
    p_counts = [10711, 10711, 5724]
    p_colors = ["#aec7e8", "#ffbb78", "#98df8a"]
    ax1.pie(p_counts, labels=patterns, autopct="%1.1f%%", colors=p_colors,
            explode=(0, 0, 0.05), startangle=90)
    ax1.set_title("Task Pattern Distribution\n(Train Set, 27,146 samples)", fontweight="bold")

    # Binary labels
    labels_bin = ["Fraud", "Normal"]
    b_counts = [11448, 4987]
    b_colors = ["#d62728", "#1f77b4"]
    ax2.pie(b_counts, labels=labels_bin, autopct="%1.1f%%", colors=b_colors,
            explode=(0.03, 0), startangle=90)
    ax2.set_title("Binary Fraud Label Distribution\n(Train Set, 16,435 binary samples)",
                  fontweight="bold")

    fig.suptitle("SFT Dataset Composition", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_sft_data_distribution.png")
    plt.close(fig)


# =====================================================================
#  Fig 8  –  SFT LoRA training detail (dual axis loss + F1)
# =====================================================================
def fig8_sft_training():
    fig, ax1 = plt.subplots(figsize=(8, 5))

    epochs = [1, 2, 3]
    losses = [0.8966, 0.6855, 0.5525]
    f1_vals = [None, None, 0.993]  # only epoch 3 has eval data

    color_loss = "#d62728"
    color_f1 = "#1f77b4"

    ax1.plot(epochs, losses, color=color_loss, marker="o", lw=2.5, ms=8, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color=color_loss)
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.set_xticks(epochs)

    ax2 = ax1.twinx()
    ax2.plot([3], [0.993], color=color_f1, marker="D", ms=12, zorder=5, label="F1 Score (Epoch 3)")
    ax2.set_ylabel("F1 Score", color=color_f1)
    ax2.tick_params(axis="y", labelcolor=color_f1)
    ax2.set_ylim(0.0, 1.05)

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)

    ax1.set_title("SFT LoRA Training Progress (Qwen2.5-1.5B, RTX 3080)", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # add text annotations
    for i, (ep, loss) in enumerate(zip(epochs, losses)):
        ax1.annotate(f"{loss:.4f}", (ep, loss), textcoords="offset points",
                     xytext=(0, -15), fontsize=8, color=color_loss, ha="center")
    ax2.annotate(f"F1={0.993:.3f}\nAcc=99.0%, FP=0", (3, 0.993),
                textcoords="offset points", xytext=(30, -5),
                fontsize=8, color=color_f1, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "fig8_sft_training.png")
    plt.close(fig)


# =====================================================================
#  Main
# =====================================================================
def main():
    print("Generating report charts...")
    fig1_architecture()
    print("  [1/8] Architecture pipeline")
    fig2_loss_convergence()
    print("  [2/8] Loss convergence")
    fig3_f1_evolution()
    print("  [3/8] F1 evolution")
    fig4_approach_comparison()
    print("  [4/8] Approach comparison")
    fig5_size_performance()
    print("  [5/8] Size vs performance")
    fig6_confusion_matrices()
    print("  [6/8] Confusion matrices")
    fig7_sft_data()
    print("  [7/8] SFT data distribution")
    fig8_sft_training()
    print("  [8/8] SFT LoRA training detail")
    print(f"\nDone. Charts saved to: {OUT.resolve()}")
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
