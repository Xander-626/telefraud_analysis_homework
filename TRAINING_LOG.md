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
