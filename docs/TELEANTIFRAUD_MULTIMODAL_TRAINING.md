# TeleAntiFraud 音频+文本电诈检测建模与训练说明

本文档面向后续 agent。目标是在本机 3060Ti 上先训练一个可复现的音频+文本二分类电诈检测模型，再预留 SFT/LoRA 扩展路线。

## 1. 数据集结论

本地数据来自 `data/README.md`：

- 二分类数据：`data/binary_classification/binary_classification/train.json`，4000 条。
- 二分类测试：`data/binary_classification/binary_classification/test.json`，400 条。
- SFT 数据：`data/sft/sft/train.jsonl`，27146 条；`data/sft/sft/test.jsonl`，6807 条。
- 音频文件：`data/audio/.../*.mp3`。
- 二分类标签来自 JSON 的 `answer` 字段：`normal -> 0`，`fraud -> 1`。

注意：二分类 JSON 里的 text 主要是任务提示，不一定是人工转写文本。第一版框架仍保留 text encoder，方便后续替换为 ASR 转写文本或 SFT 中的多轮文本。

## 2. 推荐路线

选择 C：先训练轻量二分类基线，再保留大模型 SFT 扩展。

推荐架构：

1. 冻结 Whisper audio encoder，提取每段通话的音频 embedding。
2. 冻结中文 RoBERTa/BERT text encoder，提取 prompt 或后续 ASR 文本 embedding。
3. 将 audio embedding 与 text embedding 拼接。
4. 训练轻量 MLP 分类头输出 `normal/fraud`。
5. 在 400 条 test split 上报告 accuracy、precision、recall、F1、混淆矩阵。

这样把最费显存的大模型编码阶段改为离线缓存，实际分类训练只训练一个小 MLP，适合 3060Ti。

## 3. 当前代码结构

- `src/teledeceit/data.py`：解析二分类 JSON，抽取音频路径、文本和标签。
- `src/teledeceit/features.py`：读取缓存的 audio/text embedding。
- `src/teledeceit/model.py`：音频+文本融合 MLP 分类器。
- `src/teledeceit/training.py`：训练和评估循环。
- `src/teledeceit/metrics.py`：二分类指标。
- `scripts/cache_multimodal_features.py`：冻结 Whisper + 中文 RoBERTa，缓存特征。
- `scripts/train_fusion_classifier.py`：训练融合分类头。
- `scripts/predict_binary.py`：用训练好的 checkpoint 导出预测 CSV。
- `configs/binary_fusion.example.yaml`：推荐参数记录。
- `tests/`：轻量单元测试。

## 4. 环境准备

建议使用 Python 3.10+，本机已有 Python 3.12 也可运行轻量测试。

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

如果需要 CUDA 版 PyTorch，请按本机 CUDA 版本安装对应 wheel。3060Ti 推荐先确认：

```powershell
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## 5. 训练步骤

### Step 1：运行单元测试

```powershell
python -m unittest discover -s tests -v
```

### Step 2：缓存训练集特征

首次运行会加载或下载 `openai/whisper-small` 和 `hfl/chinese-roberta-wwm-ext`。如显存不足，把 `--batch-size` 改为 1 或把 `--audio-model` 改成 `openai/whisper-base`。

```powershell
python scripts/cache_multimodal_features.py `
  --json-path data/binary_classification/binary_classification/train.json `
  --data-root data `
  --output artifacts/features/binary_train_whisper_small_roberta.pt `
  --audio-model openai/whisper-small `
  --text-model hfl/chinese-roberta-wwm-ext `
  --batch-size 4 `
  --max-audio-seconds 30 `
  --device cuda
```

### Step 3：缓存测试集特征

```powershell
python scripts/cache_multimodal_features.py `
  --json-path data/binary_classification/binary_classification/test.json `
  --data-root data `
  --output artifacts/features/binary_test_whisper_small_roberta.pt `
  --audio-model openai/whisper-small `
  --text-model hfl/chinese-roberta-wwm-ext `
  --batch-size 4 `
  --max-audio-seconds 30 `
  --device cuda
```

### Step 4：训练融合分类器

```powershell
python scripts/train_fusion_classifier.py `
  --train-cache artifacts/features/binary_train_whisper_small_roberta.pt `
  --test-cache artifacts/features/binary_test_whisper_small_roberta.pt `
  --output-dir runs/binary_fusion_whisper_small_roberta `
  --epochs 20 `
  --batch-size 64 `
  --hidden-dim 256 `
  --lr 0.001 `
  --device cuda
```

输出：

- `runs/binary_fusion_whisper_small_roberta/best_model.pt`
- `runs/binary_fusion_whisper_small_roberta/metrics.json`

### Step 5：导出测试集预测

```powershell
python scripts/predict_binary.py `
  --checkpoint runs/binary_fusion_whisper_small_roberta/best_model.pt `
  --cache artifacts/features/binary_test_whisper_small_roberta.pt `
  --output-csv runs/binary_fusion_whisper_small_roberta/test_predictions.csv `
  --device cuda
```

## 6. 3060Ti 参数建议

优先使用以下设置：

- `cache_multimodal_features.py --batch-size 2 或 4`
- `--max-audio-seconds 30`
- `--audio-model openai/whisper-small`
- `train_fusion_classifier.py --batch-size 64`
- `epochs 20`

如果 OOM：

1. 特征缓存阶段先把 `--batch-size` 降到 1。
2. 把 `--audio-model` 从 `openai/whisper-small` 换成 `openai/whisper-base`。
3. 把 `--max-audio-seconds` 降到 20。
4. 缓存阶段使用 `--device cpu`，速度慢但省显存。

预估时间：

- 特征缓存：约 1 到 3 小时，取决于音频总时长、模型大小和是否首次下载模型。
- 分类头训练：通常几分钟到半小时。
- 端到端解冻训练：不建议作为第一版，3060Ti 容易 OOM。

## 7. 评估标准

主要看 fraud 类的召回率和 F1：

- `recall` 高：少漏检电诈。
- `precision` 高：少误伤正常通话。
- `f1`：第一版综合指标。
- `fp/fn`：必须查看，尤其关注 `fn`，即诈骗被判为正常。

若业务更重视少漏检，可以后续在预测阶段调低 fraud 阈值，而不是只用 argmax。

## 8. 后续增强路线

### 8.1 ASR 文本替换 prompt 文本

当前 text encoder 读取的是 prompt text。更强的多模态方案是：

1. 用 Whisper 对每条音频生成中文转写。
2. 将转写文本作为 text encoder 输入。
3. 重新运行特征缓存和分类训练。

这样文本分支会真正携带通话语义。

### 8.2 部分解冻音频编码器

如果缓存特征基线不够强，可尝试：

1. 先用当前框架找到稳定参数。
2. 新增端到端训练脚本。
3. 冻结 Whisper 大部分层，只解冻最后 1 到 2 层。
4. 使用 batch size 1、gradient accumulation 8、fp16/bf16。

3060Ti 上这一步风险较高，建议在基线指标明确之后再做。

### 8.3 SFT/LoRA 扩展

SFT 数据在 `data/sft/sft/*.jsonl`，适合做音频理解问答和解释生成，不适合作为第一版二分类训练入口。

建议路线：

1. 先用二分类模型筛出 hard cases：高置信错判、低置信样本。
2. 从 SFT 中抽取 fraud judgment 相关任务。
3. 使用 Qwen2-Audio、Whisper+LLM 或类似音频指令模型做 LoRA。
4. 量化加载主模型，LoRA rank 8 或 16，batch size 1，gradient accumulation 16。
5. 指标仍回到二分类 test split 上评估，避免只看生成文本质量。

预计 3060Ti 上 SFT/LoRA 会从十几小时到数天不等，且对依赖版本、显存碎片和音频长度非常敏感。

## 9. 给后续 agent 的执行原则

1. 不要先做全量音频大模型微调。
2. 先跑 `--limit 32` 验证缓存脚本能完成，再跑全量。
3. 每次训练保存 `metrics.json`，不要只看终端输出。
4. 如果改数据解析，先跑 `python -m unittest discover -s tests -v`。
5. 如果要引入 ASR 转写，新增独立缓存文件，不覆盖现有特征缓存。
