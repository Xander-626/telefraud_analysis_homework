# TeleAntiFraud — 多模态电诈音频检测系统

基于 **Whisper ASR + Chinese RoBERTa + Qwen2.5 QLoRA** 级联架构的中文电话录音欺诈检测系统。四条技术路线在 4000 条训练/400 条测试样本上系统对比，推荐部署方案以 **1.6 MB** 模型体积、**5 分钟**训练时间取得 **F1=1.000**（零假阳性、零假阴性）。

## 核心结果

| 方案 | F1 | FP | FN | 模型大小 | 训练时间 |
|------|-----|-----|-----|----------|----------|
| 提示词 + 冻结编码器 + MLP | 1.000 | 0 | 0 | 1.6 MB | ~5 min |
| **ASR 转写 + 冻结编码器 + MLP** ★ | **1.000** | **0** | **0** | **1.6 MB** | **~5 min** |
| E2E 部分解冻 (Whisper 2 layers) | 0.995 | 0 | 2 | 338 MB | ~3 h |
| SFT LoRA (Qwen2.5-1.5B, 4-bit) | 0.993 | 0 | 2 | 35 MB | ~6.5 h |

> ★ 推荐部署方案

## 项目架构

```
Audio (.mp3)
  │
  ├─→ Whisper-small Encoder ──→ Audio Embedding (768d)
  │
  └─→ Whisper ASR Decode ──→ Chinese Transcript
                                  │
                                  ├─→ Chinese RoBERTa ──→ Text Embedding (768d)
                                  │
                                  └─→ Qwen2.5-1.5B (4-bit QLoRA) ──→ JSON {is_fraud, reason, confidence}
                                          │
                     Concat (1536d) ←────────┘
                          │
                     MLP Classifier ──→ normal / fraud
```

**四条技术路线：**

1. **提示词 + 冻结编码器 + MLP** — 基线：冻结 Whisper + RoBERTa，离线缓存特征，训练 0.4M 参数 MLP
2. **ASR 转写 + 冻结编码器 + MLP** — 推荐方案：ASR 文本替换提示词，通话语义加速收敛
3. **E2E 部分解冻** — 探索性方案：解冻 Whisper 最后 2 层，梯度累积 + AMP 混合精度适配 6 GB 显存
4. **SFT LoRA 指令微调** — 可解释方案：Qwen2.5-1.5B + 4-bit QLoRA (rank=8)，生成检测理由

## 目录结构

```
teledeceit_analysis/
├── src/teledeceit/          # 核心库
│   ├── data.py              # 数据加载与解析
│   ├── features.py          # 缓存特征读取
│   ├── model.py             # AudioTextFusionClassifier / E2EFusionClassifier
│   ├── training.py          # 训练循环（支持梯度累积 + AMP）
│   ├── metrics.py           # 二分类指标计算
│   ├── sft_data.py          # SFT JSONL 数据加载
│   ├── prompt_templates.py  # 指令模板与响应解析
│   └── demo_backend.py      # 演示后端逻辑
├── scripts/                 # 训练与工具脚本
│   ├── cache_multimodal_features.py  # 离线缓存 Whisper + RoBERTa 特征
│   ├── train_fusion_classifier.py    # 训练 MLP 融合分类器
│   ├── train_e2e_fusion.py           # 端到端部分解冻训练
│   ├── train_sft_lora.py             # SFT LoRA 指令微调
│   ├── eval_sft_adapter.py           # SFT adapter 评估
│   ├── predict_binary.py             # 导出测试集预测
│   ├── serve_demo.py                 # 启动 Web 演示服务器
│   ├── generate_report_charts.py     # 生成论文图表（8 张）
│   ├── visualize_dataset.py          # 生成数据集分析图表（5 张）
│   └── screen_hard_cases.py          # 筛查难例
├── web/demo/                # 前端 SPA（纯 HTML/CSS/JS）
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── data/                    # 数据集（音频 + JSON/JSONL）
├── artifacts/features/      # 离线缓存的特征文件 (.pt)
├── runs/                    # 训练输出（checkpoint + metrics.json）
├── results/                 # SFT 训练结果
├── report_figures/          # 论文与答辩图表 (300dpi PNG)
├── paper/                   # 项目论文 (LaTeX, 22pp) + 答辩 PPT
├── configs/                 # 训练配置参考
├── tests/                   # 单元测试
├── docs/                    # 训练路线文档
└── TRAINING_LOG.md          # 完整训练日志
```

## 环境配置

### 硬件要求

- **GPU**: NVIDIA GPU with ≥6 GB VRAM（RTX 3060 Laptop 测试通过）
- **存储**: ~5 GB（模型下载 + 特征缓存 + 数据集）

### 软件依赖

```bash
# Python 3.10+
pip install torch torchaudio transformers tqdm scikit-learn

# Windows 用户需额外安装 FFmpeg（torchaudio MP3 解码回退）
winget install ffmpeg

# 可选：论文编译
# 安装 TeX Live 并确保 xelatex 可用
```

### 模型下载

脚本首次运行时会自动从 HuggingFace Hub 下载：
- `openai/whisper-small` (~370 MB)
- `hfl/chinese-roberta-wwm-ext` (~400 MB)
- `Qwen/Qwen2.5-1.5B-Instruct` (~3 GB，仅 SFT LoRA 方案需要)

## 快速开始

### 1. 运行单元测试

```bash
python -m unittest discover -s tests -v
```

### 2. 缓存训练/测试特征（方案一/二）

```bash
# 训练集特征（提示词方案）
python scripts/cache_multimodal_features.py \
  --json-path data/binary_classification/binary_classification/train.json \
  --data-root data \
  --output artifacts/features/binary_train_whisper_small_roberta.pt \
  --audio-model openai/whisper-small \
  --text-model hfl/chinese-roberta-wwm-ext \
  --batch-size 2 --max-audio-seconds 30 --device cuda

# 训练集特征（ASR 方案，含转写文本）
python scripts/cache_multimodal_features.py \
  --json-path data/binary_classification/binary_classification/train.json \
  --data-root data \
  --output artifacts/features/binary_train_whisper_small_asr_roberta.pt \
  --audio-model openai/whisper-small \
  --text-model hfl/chinese-roberta-wwm-ext \
  --use-asr --batch-size 1 --max-audio-seconds 30 --device cuda

# 测试集同理，替换 --json-path 和 --output
```

### 3. 训练 MLP 融合分类器（方案一/二）

```bash
python scripts/train_fusion_classifier.py \
  --train-cache artifacts/features/binary_train_whisper_small_asr_roberta.pt \
  --test-cache artifacts/features/binary_test_whisper_small_asr_roberta.pt \
  --output-dir runs/binary_fusion_whisper_small_asr_roberta \
  --epochs 20 --batch-size 64 --hidden-dim 256 --lr 0.001 --device cuda
```

输出：`runs/.../best_model.pt` + `metrics.json`

### 4. 端到端部分解冻训练（方案三）

```bash
python scripts/train_e2e_fusion.py \
  --train-cache artifacts/features/binary_train_whisper_small_asr_roberta.pt \
  --test-cache artifacts/features/binary_test_whisper_small_asr_roberta.pt \
  --output-dir runs/binary_e2e_asr \
  --unfreeze-layers 2 --epochs 10 --lr 5e-5 --grad-accum 8 --device cuda
```

### 5. SFT LoRA 指令微调（方案四）

```bash
# 步骤 1：离线缓存 SFT 音频的 ASR 转写
python scripts/cache_multimodal_features.py \
  --json-path data/sft/sft/train.jsonl \
  --data-root data \
  --output artifacts/sft_transcriptions.pt \
  --use-asr --asr-only --device cuda

# 步骤 2：训练 QLoRA adapter
python scripts/train_sft_lora.py \
  --train-jsonl data/sft/sft/train.jsonl \
  --test-jsonl data/sft/sft/test.jsonl \
  --transcription-cache artifacts/sft_transcriptions.pt \
  --output-dir runs/sft_lora_fraud_binary \
  --lora-r 8 --lora-alpha 16 --epochs 3 --lr 2e-4 --grad-accum 8

# 步骤 3：评估
python scripts/eval_sft_adapter.py
```

### 6. 启动 Web 演示

```bash
python scripts/serve_demo.py
# 打开 http://127.0.0.1:8000
```

演示页面包含三种检测模式：预置样例（答辩主路径）、上传检测、文本检测。

## 论文与答辩

| 文件 | 说明 |
|------|------|
| `paper/teleantifraud_paper.tex` | 项目论文 LaTeX 源文件（22 页） |
| `paper/teleantifraud_paper.pdf` | 编译后的论文 PDF |
| `paper/defense.pptx` | 答辩演示文稿（15 页，12-15 分钟） |
| `report_figures/fig1-8_*.png` | 论文图表（8 张，300dpi） |
| `report_figures/fig_ds1-5_*.png` | 数据集分析图表（5 张，300dpi） |

论文编译：

```bash
cd paper && xelatex teleantifraud_paper.tex && xelatex teleantifraud_paper.tex
```

## 评估指标

所有方案在 400 条测试样本上统一评估：

- **F1 Score** — 精确率与召回率的调和平均（核心指标）
- **False Negative (FN)** — 诈骗被漏判的数量（最关注：漏检代价远高于误报）
- **False Positive (FP)** — 正常通话被误判为诈骗的数量
- **Accuracy / Precision / Recall**

## 关键发现

1. **ASR 转写是最大的单因素提升** — 将文本输入从提示词替换为 ASR 转写，收敛速度提升 33%（提前 5 个 epoch 达 F1=1.0），满分 epoch 复现率从 5% 提升至 30%
2. **所有方案 FP=0** — 四种方案在 400 条测试集上均未出现假阳性，模型对正常通话的误判风险极低
3. **端到端训练的规模不经济** — E2E 方案以 211 倍模型体积和 36 倍训练时间，换取了 F1 下降 0.005 的结果
4. **SFT LoRA 的可解释价值** — 虽 F1 略低 (0.993)，但能输出结构化 JSON 检测报告（含判断理由 + 置信度），在监管合规与人工复核场景具有独特价值
5. **推荐两级部署架构** — 方案二（ASR+MLP）作为第一级快速筛选器 + 方案四（SFT LoRA）作为低置信度样本的二次可解释检测

## 硬件适配说明

本项目所有实验均在 **NVIDIA GeForce RTX 3060 Laptop GPU (6 GB VRAM)** 上完成：

- 方案一/二（冻结编码器 + MLP）：显存 < 1 GB，训练 ~5 min
- 方案三（E2E 部分解冻）：显存 ~370 MB（混合架构），训练 ~3 h
- 方案四（SFT LoRA）：显存 ~4.5 GB（4-bit QLoRA），训练 ~6.5 h

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 音频编码 | Whisper-small (OpenAI, 88.2M) |
| 文本编码 | Chinese RoBERTa-wwm-ext (HFL, 102M) |
| 大语言模型 | Qwen2.5-1.5B-Instruct |
| 微调方法 | LoRA (rank=8) + 4-bit QLoRA (NF4) |
| 融合分类器 | MLP (1536→256→2, GELU, Dropout 0.2) |
| 前端 | HTML/CSS/JS SPA (零依赖) |
| 后端 | Python FastAPI + uvicorn |
| 可视化 | Matplotlib (300dpi, 中文字体) |
| 论文排版 | LaTeX (ctexart, xelatex) |

## License

This project is for educational and research purposes as part of a deep learning course project.

## 相关文档

- `docs/TELEANTIFRAUD_MULTIMODAL_TRAINING.md` — 训练路线图与完整操作指南
- `TRAINING_LOG.md` — 完整训练日志（含所有实验的逐 epoch 指标、环境修复记录和结论）
- `paper/teleantifraud_paper.pdf` — 项目论文
- `paper/defense.pptx` — 答辩 PPT
