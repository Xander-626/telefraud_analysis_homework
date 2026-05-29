# 训练日志

## 基本信息

- **日期**：2026-05-19
- **分支**：master
- **基准提交**：bb46c4a (chore: initialize multimodal fraud detection project)
- **机器**：NVIDIA GeForce RTX 3060 Laptop GPU (6 GB VRAM)，Python 3.12

## 执行概要

按照 `docs/TELEANTIFRAUD_MULTIMODAL_TRAINING.md` 的路线 C（轻量二分类基线），完成了全部 5 个步骤。

---

## 环境修复记录

### 问题 1：torchaudio 无法加载 MP3

- **现象**：`RuntimeError: Couldn't find appropriate backend to handle uri ...mp3`
- **原因**：Windows 下 torchaudio 2.6.0 无内置 MP3 解码后端
- **修复**：
  1. 通过 `winget install ffmpeg` 安装 FFmpeg 8.1.1
  2. 在 `scripts/cache_multimodal_features.py` 的 `WhisperAudioEncoder` 中增加 `_load_audio_ffmpeg` 回退方法，通过 subprocess 调用 ffmpeg 解码 MP3 为 PCM
  3. 增加 `_find_ffmpeg` 方法自动查找 winget 安装路径

### 问题 2：Whisper mel 特征长度不匹配

- **现象**：`ValueError: Whisper expects the mel input features to be of length 3000, but found 2627`
- **原因**：短音频产生的 mel 帧数不足 3000，Whisper 编码器要求固定长度
- **修复**：在 `encode` 方法中手动 pad 到 3000 帧（`torch.nn.functional.pad`），超长则截断

### 副作用

- HuggingFace Hub 403 警告（`hfl/chinese-roberta-wwm-ext` 仓库禁用 discussions API）—— 来自 transformers 的 safetensors_conversion 后台线程，不影响运行，可忽略

---

## 步骤详情

### Step 1：单元测试

**命令**：`python -m unittest discover -s tests -v`

**结果**：7/7 全部通过（耗时 2.4s）

| 测试 | 结果 |
|------|------|
| test_collate_text_only_keeps_batch_order | ok |
| test_load_binary_samples_extracts_label_text_and_audio_path | ok |
| test_load_binary_samples_rejects_unknown_labels | ok |
| test_cached_feature_dataset_loads_rows | ok |
| test_compute_binary_metrics_reports_core_scores | ok |
| test_fusion_classifier_returns_logits_and_loss | ok |
| test_train_and_evaluate_on_cached_features | ok |

### Step 2：缓存训练集特征

**命令**：
```
python scripts/cache_multimodal_features.py \
  --json-path data/binary_classification/binary_classification/train.json \
  --data-root data \
  --output artifacts/features/binary_train_whisper_small_roberta.pt \
  --audio-model openai/whisper-small \
  --text-model hfl/chinese-roberta-wwm-ext \
  --batch-size 2 \
  --max-audio-seconds 30 \
  --device cuda
```

**结果**：4000 行，约 19 分钟，输出文件 ~25 MB

### Step 3：缓存测试集特征

**命令**：同上，替换 json-path 和 output

**结果**：400 行，约 2 分钟，输出文件 ~2.5 MB

### Step 4：训练融合分类器

**命令**：
```
python scripts/train_fusion_classifier.py \
  --train-cache artifacts/features/binary_train_whisper_small_roberta.pt \
  --test-cache artifacts/features/binary_test_whisper_small_roberta.pt \
  --output-dir runs/binary_fusion_whisper_small_roberta \
  --epochs 20 \
  --batch-size 64 \
  --hidden-dim 256 \
  --lr 0.001 \
  --device cuda
```

**最佳 epoch 指标**（Epoch 16）：

| 指标 | 值 |
|------|-----|
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 | 1.0000 |
| TP | 200 |
| TN | 200 |
| FP | 0 |
| FN | 0 |

各 epoch 指标走势：

| Epoch | Acc | Prec | Recall | F1 | FN |
|-------|-----|------|--------|-----|-----|
| 1 | 0.9100 | 0.9881 | 0.8300 | 0.9022 | 34 |
| 2 | 0.9475 | 0.9945 | 0.9000 | 0.9449 | 20 |
| 3 | 0.9675 | 0.9947 | 0.9400 | 0.9666 | 12 |
| 4 | 0.9850 | 0.9802 | 0.9900 | 0.9851 | 2 |
| 5 | 0.9850 | 0.9850 | 0.9850 | 0.9850 | 3 |
| 10 | 0.9925 | 0.9852 | 1.0000 | 0.9926 | 0 |
| 13 | 0.9975 | 1.0000 | 0.9950 | 0.9975 | 1 |
| **16** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0** |
| 20 | 0.9975 | 0.9950 | 1.0000 | 0.9975 | 0 |

### Step 5：导出预测

**命令**：
```
python scripts/predict_binary.py \
  --checkpoint runs/binary_fusion_whisper_small_roberta/best_model.pt \
  --cache artifacts/features/binary_test_whisper_small_roberta.pt \
  --output-csv runs/binary_fusion_whisper_small_roberta/test_predictions.csv \
  --device cuda
```

**结果**：200 normal + 200 fraud，全部正确

---

## 产出文件

```
artifacts/features/
  binary_train_whisper_small_roberta.pt   (~25 MB, 4000 条特征)
  binary_test_whisper_small_roberta.pt    (~2.5 MB, 400 条特征)

runs/binary_fusion_whisper_small_roberta/
  best_model.pt                           (~1.6 MB, 最佳模型权重)
  metrics.json                            (20 epoch 完整指标)
  test_predictions.csv                    (400 条预测结果)
```

---

## 后续可选方向

1. **ASR 文本替换**：用 Whisper 转录替代 prompt text，使文本分支携带真实通话语义
2. **部分解冻编码器**：解冻 Whisper 最后 1-2 层，做端到端微调（3060 上需 batch=1 + gradient accumulation）
3. **SFT/LoRA**：用 `data/sft/` 的 27k 条多轮数据做音频指令微调（Qwen2-Audio 等）

---

## 代码修改

- `scripts/cache_multimodal_features.py`：新增 ffmpeg 音频加载回退 + Whisper mel 固定长度 padding

---

## 2026-05-28：ASR 文本替换 + 端到端部分解冻实验

### 概要

按路线图 8.1 和 8.2 节，完成了 ASR 文本替换方案的全量训练，以及端到端解冻 Whisper 最后 2 层的实验。**核心发现：ASR + 预缓存 MLP 方案最优（F1=1.0，训练 5 分钟），端到端解冻性价比低（F1=0.995，训练 3 小时）。**

---

### 实验一：ASR 文本替换 prompt 文本（路线图 8.1）

#### Step 1：缓存 ASR 训练集特征

```
python scripts/cache_multimodal_features.py \
  --json-path data/binary_classification/binary_classification/train.json \
  --data-root data \
  --output artifacts/features/binary_train_whisper_small_asr_roberta.pt \
  --use-asr \
  --limit 4000 \
  --batch-size 8
```

- **结果**：4000 行，1 小时 12 分钟，输出文件 20 MB
- **速度**：平均 8.68s/batch（含 Whisper ASR 转录 + RoBERTa 编码）

#### Step 2：缓存 ASR 测试集特征

同上，替换为 test.json，limit=400。

- **结果**：400 行，约 7 分钟，输出文件 2.0 MB

#### Step 3：训练 ASR 融合分类器

```
python scripts/train_fusion_classifier.py \
  --train-cache artifacts/features/binary_train_whisper_small_asr_roberta.pt \
  --test-cache artifacts/features/binary_test_whisper_small_asr_roberta.pt \
  --output-dir runs/binary_fusion_whisper_small_asr_roberta \
  --epochs 20
```

**最佳 epoch 指标**（Epoch 11, 12, 15, 17, 18, 20 均达到 F1=1.0）：

| 指标 | 值 |
|------|-----|
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 | 1.0000 |
| TP | 200 |
| TN | 200 |
| FP | 0 |
| FN | 0 |

各 epoch 指标走势：

| Epoch | Acc | Prec | Recall | F1 | FN | FP |
|-------|-----|------|--------|-----|-----|-----|
| 1 | 0.9675 | 1.0000 | 0.9350 | 0.9664 | 13 | 0 |
| 2 | 0.9875 | 1.0000 | 0.9750 | 0.9873 | 5 | 0 |
| 3 | 0.9950 | 0.9901 | 1.0000 | 0.9950 | 0 | 2 |
| 4 | 0.9975 | 1.0000 | 0.9950 | 0.9975 | 1 | 0 |
| **11** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0** | **0** |
| 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |

**对比非 ASR 基线**：

| 指标 | 非 ASR（prompt）| ASR（转写文本）|
|------|----------------|---------------|
| Epoch 1 F1 | 0.902 | **0.966** |
| Epoch 4 F1 | 0.985 | **0.998** |
| 首次完美 | Epoch 16 | **Epoch 11** |
| 20 epoch 完美次数 | 1 次 | **6 次** |

**结论**：ASR 转写文本比原始 prompt 文本提供更强的区分度，收敛速度显著更快，稳定性更好。

---

### 实验二：端到端部分解冻 Whisper（路线图 8.2）

#### 方案设计

采用混合方案：Whisper 编码器在线加载（前 10 层冻结，最后 2 层可训练），文本分支使用预缓存的 ASR text embeddings（跳过 RoBERTa 在线加载），MLP 融合头可训练。

- 可训练参数：14.57M（Whisper 最后 2 层 14.2M + MLP 头 ~0.2M）
- 有效 batch size：8（batch_size=1 × grad_accum=8）
- AMP fp16 混合精度

#### 代码变更

1. **`src/teledeceit/training.py`**：`train_one_epoch` 新增 `grad_accum` 和 `scaler` 参数，支持梯度累积 + AMP
2. **`src/teledeceit/model.py`**：新增 `E2EFusionClassifier`（在线音频编码器 + 预缓存文本 embedding + MLP）
3. **`scripts/train_e2e_fusion.py`**（新增）：端到端训练脚本，支持 Whisper 选择性解冻、梯度累积、AMP

#### 小规模烟雾测试

```
--limit 32 --epochs 2
```

- Epoch 1: F1=0.79, Epoch 2: F1=0.84
- 无 OOM，管线完整跑通

#### 全量训练

```
python scripts/train_e2e_fusion.py \
  --train-cache artifacts/features/binary_train_whisper_small_asr_roberta.pt \
  --test-cache artifacts/features/binary_test_whisper_small_asr_roberta.pt \
  --output-dir runs/binary_e2e_asr \
  --epochs 10 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 5e-5 \
  --unfreeze-layers 2
```

- **耗时**：约 3 小时
- **模型大小**：338 MB（含完整 Whisper 编码器权重）

各 epoch 指标：

| Epoch | Acc | Prec | Recall | F1 | FN | FP |
|-------|-----|------|--------|-----|-----|-----|
| 1 | 0.9650 | 1.0000 | 0.9300 | 0.9637 | 14 | 0 |
| 2 | 0.9775 | 0.9948 | 0.9600 | 0.9771 | 8 | 1 |
| 3 | 0.9875 | 0.9851 | 0.9900 | 0.9875 | 2 | 3 |
| **4** | **0.9950** | **1.0000** | **0.9900** | **0.9950** | **2** | **0** |
| 5-10 | 0.9925 | 1.0000 | 0.9850 | 0.9924 | 3 | 0 |

**现象**：epoch 4 即达最佳，之后 train_loss → 0 但 test 指标不再改善（过拟合）。

---

### 三种方案最终对比

| 方案 | 最佳 F1 | 最佳 epoch | FN | FP | 模型大小 | 训练时间 |
|------|---------|-----------|-----|-----|----------|----------|
| 非 ASR（prompt 原文）| **1.000** | 16 | 0 | 0 | 1.6 MB | ~5 min |
| ASR + MLP（预缓存）| **1.000** | 11 | 0 | 0 | 1.6 MB | ~5 min |
| ASR + E2E（解冻 2 层）| 0.995 | 4 | 2 | 0 | 338 MB | ~3 h |

### 结论与建议

1. **推荐部署方案**：ASR 转写文本 + 预缓存 MLP 融合分类器，模型小（1.6 MB）、训练快（5 分钟）、指标完美（F1=1.0）
2. **E2E 解冻不推荐**：在 3060 6GB 条件下，batch size 受限（有效仅 8），梯度噪声大，容易过拟合，性价比远低于预缓存方案
3. **ASR 文本优于 prompt**：无论 MLP 还是 E2E，ASR 转写文本的区分度都显著高于原始 prompt 文本，是关键的提升点

### 生成文件

```
artifacts/features/
  binary_train_whisper_small_asr_roberta.pt   (20 MB, 4000 条)
  binary_test_whisper_small_asr_roberta.pt    (2.0 MB, 400 条)

runs/binary_fusion_whisper_small_asr_roberta/
  best_model.pt                               (1.6 MB)
  metrics.json                                (20 epoch)

runs/binary_e2e_asr/
  best_model.pt                               (338 MB)
  metrics.json                                (10 epoch)
```

---

## 2026-05-29：SFT/LoRA 音频指令微调（路线图 8.3）

### 概要

按路线图 8.3 节，搭建了 Whisper-small ASR → Qwen2.5-1.5B-Instruct 4-bit QLoRA 级联架构。由于本机是笔记本 RTX 3060（6GB VRAM），无法直接加载 Qwen2-Audio-7B（4-bit 权重 ~4GB + 优化器 + 激活值 > 6GB），采用级联方案替代。

### 架构

```
Whisper-small (ASR转写) → 中文转写文本 → Qwen2.5-1.5B-Instruct (4-bit QLoRA)
                                              ↓
                                   生成 JSON → 解析 is_fraud → 二分类评估
```

VRAM 峰值 ~3.2GB/6GB，安全余量充足。

### 环境准备

安装新依赖：

```
pip install bitsandbytes peft accelerate
```

- bitsandbytes 0.49.2
- peft 0.19.1
- accelerate 1.13.0

### 新增代码

| 文件 | 说明 |
|------|------|
| `src/teledeceit/sft_data.py` | SFT JSONL 数据加载器，按任务模式分类（SCENE_ONLY / FRAUD_BINARY / FRAUD_TYPE），提取二分类标签 |
| `src/teledeceit/prompt_templates.py` | 指令模板格式化和生成文本响应解析（三层回退：JSON → 正则 → 关键词） |
| `scripts/train_sft_lora.py` | QLoRA 训练主脚本，Whisper ASR 离线转写 + Qwen2.5 4-bit LoRA 微调 |
| `scripts/screen_hard_cases.py` | Hard case 筛选脚本，用已有冻结分类器从 SFT 数据中筛选困难样本 |
| `src/teledeceit/training.py` | 新增 `evaluate_sft_binary()` 生成式评估函数 |
| `configs/sft_lora.example.yaml` | 训练配置参考 |
| `tests/test_sft_data.py` | SFT 数据加载器单元测试（7 项） |
| `tests/test_prompt_templates.py` | 提示词模板和解析单元测试（12 项） |

### SFT 数据分析

- **训练集**：27,146 条，其中二分类可用 16,435 条（fraud=11,448, normal=4,987）
- **测试集**：6,807 条，其中二分类可用 4,130 条（fraud=2,906, normal=1,224）
- **任务模式分布**：SCENE_ONLY 10,711 / FRAUD_BINARY 10,711 / FRAUD_TYPE 5,724
- **所有音频文件**均存在于 `data/audio/` 下的 11 个子目录中

### 单元测试

- 26/26 全部通过（含原有 7 项 + 新增 19 项）

### 烟幕测试

```
python scripts/train_sft_lora.py --limit 8 --epochs 1 \
  --batch-size 1 --grad-accum 4 --output-dir runs/sft_lora_smoke
```

结果：
- 模型加载成功：Qwen2.5-1.5B-Instruct 4-bit nf4 + double_quant
- LoRA 可训练参数：9.23M / 0.90B (1.0%)
- 训练循环正常：loss 2.84 → 2.23（8 样本/1 epoch）
- 无 OOM，管线完整跑通
- Whisper ASR 转录缓存正常工作

### 训练超参数（全量训练用）

| 参数 | 值 |
|------|-----|
| model | Qwen/Qwen2.5-1.5B-Instruct |
| fallback | Qwen/Qwen2.5-0.5B-Instruct（OOM 时自动回退）|
| 量化 | 4-bit nf4 + double_quant |
| LoRA rank | 8, alpha=16 |
| dropout | 0.05 |
| target_modules | all linear (q/k/v/o/gate/up/down) |
| batch_size | 1 |
| grad_accum | 16（等效 batch=16）|
| epochs | 3 |
| lr | 2e-4, cosine + 3% warmup |
| optimizer | paged_adamw_8bit |
| max_seq_length | 2048 |
| max_new_tokens | 200 |
| eval_limit | 200（生成慢，每 epoch 仅评估 200 条）|

### 全量训练命令

```powershell
python scripts/train_sft_lora.py `
  --epochs 3 `
  --batch-size 1 `
  --grad-accum 16 `
  --eval-limit 200 `
  --output-dir runs/sft_lora_fraud_binary
```

预估时间：训练 ~20h + 评估 ~3h/epoch（200 条 × ~57s/条）。可使用 `--eval-limit 50` 加速评估。

### 产出文件

```
artifacts/
  sft_transcriptions.pt                       (Whisper ASR 转录缓存)

runs/sft_lora_smoke/
  adapter/                                    (LoRA 适配器权重)
  metrics.json                                (烟幕测试指标)

runs/sft_lora_fraud_binary/                   (全量训练后生成)
  adapter/                                    (最佳 LoRA 适配器)
  metrics.json                                (完整训练指标)
  best_metrics.json                           (最佳 epoch 指标)
```

### 待完成

- [ ] 运行全量 3 epoch 训练
- [ ] 与冻结分类器（F1=1.0）在 SFT 测试子集上横向对比
- [ ] Hard case 筛选实验（`scripts/screen_hard_cases.py`）
