# TeleAntiFraud 机器学习入门学习文档

> **面向读者**：具备 Python 基础，但尚未系统学习深度学习/机器学习的开发者。
> **学习目标**：理解本项目涉及的模型原理、掌握代码结构、能够复现实验并进行二次开发。

---

## 目录

1. [前置知识速览](#1-前置知识速览)
2. [核心模型与原理](#2-核心模型与原理)
3. [项目使用的框架与工具](#3-项目使用的框架与工具)
4. [环境配置详解](#4-环境配置详解)
5. [项目文件详解（逐文件走读）](#5-项目文件详解)
6. [数据流全流程追踪](#6-数据流全流程追踪)
7. [复现指南](#7-复现指南)
8. [学习路线建议](#8-学习路线建议)
9. [常见问题排查](#9-常见问题排查)

---

## 1. 前置知识速览

在深入本项目之前，建议按以下顺序补全基础知识。每个主题列出了核心概念和学习时间估算。

### 1.1 机器学习基础（约 4-6 小时）

| 概念 | 一句话解释 | 本项目中的应用 |
|------|-----------|---------------|
| **监督学习** | 给模型"题目+答案"，让它学会从题目推出答案 | 我们有 4000 条通话音频，每条标注了 fraud/normal |
| **二分类** | 输出只有两种可能（是/否） | 判断通话是否为诈骗 |
| **训练集/测试集** | 训练集用来学习，测试集用来检验学习效果 | 4000 条训练 / 400 条测试 |
| **损失函数 (Loss)** | 衡量模型输出与正确答案之间的"差距" | 使用交叉熵损失 (Cross-Entropy Loss) |
| **梯度下降** | 沿着"下山"的方向调整参数，让 loss 越来越小 | AdamW 优化器自动完成 |
| **过拟合** | 模型死记硬背训练集，但测试集表现差 | E2E 方案在 epoch 4 后出现过拟合 |
| **Batch / Epoch** | batch=每次喂多少数据，epoch=完整遍历数据集一次 | MLP 方案 batch=64, epoch=20 |

**推荐学习资源**：
- 吴恩达《Machine Learning》课程前 5 周
- 3Blue1Brown《Neural Networks》YouTube 系列（可视化讲解）

### 1.2 深度学习核心概念（约 6-8 小时）

| 概念 | 一句话解释 | 本项目中的应用 |
|------|-----------|---------------|
| **神经网络** | 多层"神经元"堆叠，每层做简单的数学变换 | MLP 分类头：1536→256→2 |
| **激活函数** | 引入非线性，让网络能学复杂模式 | GELU（Whisper/RoBERTa 使用的同类函数） |
| **Transformer** | 当前最主流的深度学习架构，基于"自注意力"机制 | Whisper、RoBERTa、Qwen2.5 都是 Transformer |
| **Embedding（嵌入）** | 用一串数字（向量）表示一段文字或音频的含义 | 768 维向量 = 用 768 个数字描述一段通话 |
| **预训练 + 微调** | 先用大数据训练基础能力，再用小数据适配特定任务 | Whisper 在 68 万小时语料预训练，我们在 4000 条数据上微调 |
| **Dropout** | 训练时随机"关闭"一些神经元，防止过拟合 | MLP 中使用 dropout=0.2 |
| **Layer Normalization** | 让每层输出的数值范围保持稳定，加速训练 | MLP 第一层操作 |

**推荐学习资源**：
- Jay Alammar《The Illustrated Transformer》（图文并茂）
- 李沐《动手学深度学习》第 11 章（Transformer）

### 1.3 本项目特有技术（约 4-6 小时）

| 概念 | 一句话解释 |
|------|-----------|
| **ASR（语音识别）** | 将音频波形 → 中文文字 |
| **LoRA（低秩适配）** | 不修改原模型，只训练很小的"插件"来实现新功能 |
| **QLoRA** | LoRA + 4-bit 量化，让大模型能在消费级显卡上微调 |
| **梯度累积** | 显存不够时，多步累积梯度再更新，模拟更大的 batch |
| **AMP（混合精度）** | 用 fp16（半精度）计算来节省显存和加速 |

---

## 2. 核心模型与原理

本项目使用了三个预训练模型，形成一个级联流水线。

### 2.1 Whisper-small —— 音频处理

**基本信息**：
- 开发者：OpenAI
- 参数量：88.2M（8800 万个参数）
- 架构：Encoder-Decoder Transformer
- 训练数据：68 万小时多语言语料
- 模型文件大小：~370 MB (fp16)

**工作原理**（简化版）：

```
原始音频（MP3）
    │
    ▼  重采样 16kHz 单声道
PCM 波形数据 (1D 数组，每秒 16000 个数值)
    │
    ▼  80 维 log-Mel 滤波器组
Mel 频谱图 (2D 矩阵，80 × 3000)
    │
    ▼  Whisper Encoder（12 层 Transformer）
音频 Embedding (768 维向量)
```

**两个工作模式**：

| 模式 | 做了什么 | 输出 | 本项目用途 |
|------|---------|------|-----------|
| **Encoder 模式** | 只跑 Encoder 部分，提取音频特征 | (B, 3000, 768) hidden states | 方案一/二/三的音频分支 |
| **ASR 转写模式** | Encoder + Decoder 完整推理 | 中文文本字符串 | 方案二/三/四的文本输入 |

**为什么选 Whisper-small？**
- 88M 参数的版本在 RTX 3060 (6GB) 上稳定运行，不会 OOM
- 中文转写质量足够区分正常/诈骗通话的关键语义
- 有更小的 tiny(39M) 和 base(74M) 可作为降级选项

### 2.2 Chinese RoBERTa —— 文本理解

**基本信息**：
- 开发者：哈工大-讯飞联合实验室 (HFL)
- 参数量：102M
- 架构：12 层 Transformer Encoder
- 模型文件大小：~400 MB
- 输入：中文文本（最长 512 tokens）
- 输出：768 维文本 Embedding

**工作原理**：

```
输入文本："你的银行卡涉嫌异常交易，请马上把资金转入安全账户"
    │
    ▼  Tokenizer（分词器）
Token 序列：[CLS] 你 的 银行 卡 涉嫌 异常 交易 ... [SEP]
    │
    ▼  RoBERTa Encoder（12 层 Transformer）
每个 Token 的 768 维向量
    │
    ▼  取 [CLS] 位置的向量（代表整句话的语义）
文本 Embedding (768 维)
```

**[CLS] token** 是 BERT/RoBERTa 体系中一个特殊设计——模型被训练为将整句话的语义"压缩"到 [CLS] 位置的向量中，因此这个向量可以作为整句话的语义表示直接用于下游分类。

### 2.3 MLP 融合分类器 —— 本项目核心创新

**结构**：

```
音频 Embedding (768)  +  文本 Embedding (768)
              │
              ▼  拼接 (Concatenate)
         融合向量 (1536)
              │
              ▼  LayerNorm ── 归一化，让两个模态的数值在同一"尺度"
        归一化向量 (1536)
              │
              ▼  Linear(1536 → 256) ── 降维，提取关键特征
        隐藏向量 (256)
              │
              ▼  GELU ── 非线性激活
        激活向量 (256)
              │
              ▼  Dropout(0.2) ── 随机丢弃 20% 的神经元，防止过拟合
        正则化向量 (256)
              │
              ▼  Linear(256 → 2) ── 输出两个分数
        Logits: [score_normal, score_fraud]
              │
              ▼  argmax ── 取分数高的作为预测
        Prediction: normal 或 fraud
```

**为什么这样设计？**

| 设计选择 | 原因 |
|----------|------|
| 1536 → 256 的压缩 | 保留足够信息的同时大幅减少参数（1536×256 vs 1536→1536 更少参数） |
| GELU 而非 ReLU | GELU 更平滑，与 Transformer 体系风格一致 |
| Dropout 0.2 | 4000 条样本不算多，需要正则化防止过拟合 |
| 仅 0.4M 可训练参数 | 极少参数 = 极快训练 + 极小模型文件（1.6 MB） |

### 2.4 Qwen2.5-1.5B-Instruct —— LLM 方案

**基本信息**：
- 开发者：阿里云通义千问团队
- 参数量：1.5B（15 亿）
- 架构：Decoder-only Transformer
- 模型文件大小：~3 GB (fp16) / ~1 GB (4-bit)

**工作原理**：

Qwen2.5 是一个"对话型"大语言模型。给定一个对话（system prompt + user message），
它以自回归（autoregressive）方式逐 token 生成回复。

```
输入对话：
  System: 你是电诈检测助手...
  User: 请分析以下通话...{transcription}
  Assistant: 
              │
              ▼  Qwen2.5-1.5B（逐 token 生成）
  Assistant: ```json
             {
               "is_fraud": true,
               "reason": "通话包含...",
               "confidence": 0.95
             }
             ```
```

**QLoRA 如何让 1.5B 模型在 6GB 显卡上微调？**

全量微调 1.5B 模型需要约 6GB（仅模型参数，不含优化器状态）。
QLoRA 通过三重技术将显存降至 ~4.5 GB：

1. **4-bit NormalFloat4 (NF4)**：每个参数从 16-bit (2 bytes) 压缩到 4-bit (0.5 bytes)，模型主体仅占 ~0.75 GB
2. **双重量化 (Double Quantization)**：量化常数本身也被量化，再省 ~0.4 GB
3. **LoRA (Low-Rank Adaptation)**：不修改原始参数，在旁路插入可训练的"低秩矩阵"（rank=8 的小矩阵），仅训练这 ~35M (2.3%) 参数

```
原始层：  W (冻结, 4-bit 量化)
          +
旁路：   A (可训练) @ B (可训练)   ← 仅训练这部分
         (d × 8)     (8 × d)
```

---

## 3. 项目使用的框架与工具

### 3.1 PyTorch

深度学习框架，提供：
- `torch.Tensor`：多维数组（类似 NumPy 但支持 GPU 加速）
- `torch.nn.Module`：神经网络的基类
- `torch.utils.data.DataLoader`：批量数据加载器
- `torch.optim`：优化器（AdamW 等）
- `torch.amp.autocast` + `GradScaler`：自动混合精度训练

### 3.2 HuggingFace Transformers

预训练模型库，提供：
- `AutoModel.from_pretrained("模型名")`：一行代码下载并加载预训练模型
- `AutoTokenizer.from_pretrained("模型名")`：加载对应的分词器
- `BitsAndBytesConfig`：4-bit/8-bit 量化配置
- `PeftModel`：加载 LoRA adapter

**关键 API 调用示例**：

```python
# 加载 Whisper 模型
from transformers import AutoModel
whisper = AutoModel.from_pretrained("openai/whisper-small")

# 加载 Qwen2.5 分词器并应用对话模板
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
messages = [
    {"role": "system", "content": "你是电诈检测助手"},
    {"role": "user", "content": "请分析这段通话..."}
]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True)
```

### 3.3 PEFT (Parameter-Efficient Fine-Tuning)

HuggingFace 的微调工具库，提供 LoRA/QLoRA 实现：

```python
from peft import LoraConfig, get_peft_model, PeftModel

# 配置 LoRA
lora_config = LoraConfig(
    r=8,                    # rank
    lora_alpha=16,          # scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# 注入 LoRA adapter
model = get_peft_model(base_model, lora_config)

# 保存/加载（仅保存 adapter 权重，~35MB）
model.save_pretrained("runs/sft_lora/adapter")
model = PeftModel.from_pretrained(base_model, "runs/sft_lora/adapter")
```

### 3.4 其他工具

| 工具 | 用途 |
|------|------|
| `tqdm` | 进度条显示（"Caching features: 100%\|████\| 4000/4000"） |
| `ffmpeg` / `ffprobe` | 音频解码、格式转换、时长分析 |
| `matplotlib` | 生成论文中的全部图表 |
| `scikit-learn` | （未直接使用，但常与 PyTorch 配合做指标计算） |
| `fastapi` + `uvicorn` | （备选后端方案，当前使用标准库 http.server） |

---

## 4. 环境配置详解

### 4.1 硬件最低要求

| 方案 | 最低 GPU 显存 | 推荐配置 |
|------|-------------|----------|
| 方案一/二 (MLP) | 无需 GPU（CPU 也可） | 任意 NVIDIA GPU ≥ 2GB |
| 特征缓存 | ≥ 4 GB | RTX 3060 (6GB) |
| 方案三 (E2E) | ≥ 4 GB | RTX 3060 (6GB) |
| 方案四 (SFT LoRA) | ≥ 6 GB | RTX 3060 (6GB) |

### 4.2 软件安装（Windows）

```powershell
# 1. Python 3.10+（建议 3.12）
python --version  # 确认版本

# 2. 创建虚拟环境（强烈推荐，避免依赖冲突）
python -m venv venv
venv\Scripts\activate

# 3. 安装 PyTorch（CUDA 12.4 版本）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. 安装其他依赖
pip install transformers tqdm scikit-learn peft accelerate bitsandbytes

# 5. 安装 FFmpeg（Windows 特有需求）
winget install ffmpeg
# 验证安装：
ffmpeg -version

# 6. 验证 CUDA 可用
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# 预期输出：True \n NVIDIA GeForce RTX 3060 Laptop GPU
```

### 4.3 常见安装问题

**Q: `pip install torch` 后 `torch.cuda.is_available()` 返回 False？**
A: 说明安装的是 CPU 版 PyTorch。去 [pytorch.org](https://pytorch.org) 选择 CUDA 版本重新安装。

**Q: `bitsandbytes` 安装失败？**
A: Windows 上 bitsandbytes 需要特定版本。可以尝试 `pip install bitsandbytes-windows`。

**Q: FFmpeg 命令找不到？**
A: 使用 `winget install ffmpeg` 安装后，需要**重启终端**让 PATH 生效。

---

## 5. 项目文件详解

本章逐文件解释代码结构和关键实现。

### 5.1 核心库 (`src/teledeceit/`)

#### 5.1.1 `data.py` — 数据加载

**作用**：解析二分类 JSON 文件，提取音频路径、文本和标签。

**关键代码走读**：

```python
# 定义数据样本的结构（dataclass = 自动生成 __init__ 的类）
@dataclass(frozen=True)  # frozen=True 表示不可变（immutable）
class BinarySample:
    sample_id: str       # 样本唯一 ID
    audio_path: Path     # 音频文件路径
    text: str            # 文本内容（提示词或转写文本）
    label: int           # 0=normal, 1=fraud
    answer: str          # 原始标签字符串 "normal" / "fraud"

def load_binary_samples(json_path, data_root) -> list[BinarySample]:
    records = json.loads(json_path.read_text(encoding="utf-8"))
    # 遍历每条记录，提取字段，构建 BinarySample 列表
    for index, record in enumerate(records):
        answer = record.get("answer", "").strip().lower()
        # answer 必须是 "normal" 或 "fraud"
        if answer not in LABEL_TO_ID:  # LABEL_TO_ID = {"normal": 0, "fraud": 1}
            raise ValueError(...)
        audio_url = _first_audio_url(record.get("prompt", []))
        audio_path = data_root / audio_url
        ...
```

**设计要点**：
- `_first_audio_url()` 从 prompt 消息的嵌套结构中提取音频 URL——因为 JSON 格式中音频 URL 藏在 `content[].audio_url` 路径下
- `_iter_prompt_text()` 提取所有 text 类型的 content 拼接为完整文本
- `LABEL_TO_ID` 字典做字符串→数字映射（PyTorch 需要数字标签）

#### 5.1.2 `features.py` — 缓存特征数据集

**作用**：将 `.pt` 缓存文件封装为 PyTorch Dataset，供 DataLoader 使用。

**关键代码走读**：

```python
class CachedFeatureDataset(Dataset):
    def __init__(self, cache_path):
        # torch.load 加载 .pt 文件到内存
        payload = torch.load(cache_path, map_location="cpu")
        self.audio_features = payload["audio_features"]  # (N, 768)
        self.text_features = payload["text_features"]     # (N, 768)
        self.labels = payload["labels"]                   # (N,)
        self._validate_lengths()  # 确保所有字段行数一致

    def __getitem__(self, index):
        # DataLoader 会调用这个方法获取第 index 条数据
        return {
            "audio_features": self.audio_features[index],
            "text_features": self.text_features[index],
            "label": self.labels[index],
        }
```

**PyTorch Dataset 概念**：
- 继承 `torch.utils.data.Dataset`
- 必须实现 `__len__`（返回总样本数）和 `__getitem__`（返回第 i 条样本）
- DataLoader 会自动处理批量打包（batch）、打乱（shuffle）、多进程加载（num_workers）

#### 5.1.3 `model.py` — 模型定义

**作用**：定义两个分类器模型。

**`AudioTextFusionClassifier`（方案一/二使用）**：

```python
class AudioTextFusionClassifier(nn.Module):
    def __init__(self, audio_dim, text_dim, hidden_dim=256, dropout=0.2):
        super().__init__()  # 必须调用父类 __init__
        input_dim = audio_dim + text_dim  # 768 + 768 = 1536
        # nn.Sequential 将多个层串联成一个"流水线"
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),      # 归一化
            nn.Linear(input_dim, hidden_dim),  # 1536 → 256
            nn.GELU(),                    # 激活函数
            nn.Dropout(dropout),          # 正则化
            nn.Linear(hidden_dim, 2),     # 256 → 2 (normal/fraud)
        )

    def forward(self, audio_features, text_features, labels=None):
        # 沿最后一维拼接： (B,768) + (B,768) → (B,1536)
        features = torch.cat([audio_features, text_features], dim=-1)
        logits = self.classifier(features)  # 前向传播
        # 如果提供了 labels，计算损失
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return ClassifierOutput(logits=logits, loss=loss)
```

**`E2EFusionClassifier`（方案三使用）**：

与上面结构相同，差别在于 `audio_encoder` 作为模块的一部分（而非外部独立）传入，
使得梯度可以反向传播到 encoder 的解冻层。

**`forward()` 方法详解**：
- `forward` 是 PyTorch 的特殊方法——当调用 `model(input)` 时，实际执行的是 `model.forward(input)`
- `dim=-1` 表示沿最后一个维度操作（对于 2D 张量 (B, D)，dim=-1 即 dim=1）
- `F.cross_entropy(logits, labels)` 内部做 softmax + 负对数似然，是最常用的分类损失函数

#### 5.1.4 `training.py` — 训练与评估

**作用**：训练循环和评估循环。

**`train_one_epoch()` 核心流程**：

```python
def train_one_epoch(model, dataloader, optimizer, device,
                    max_grad_norm=1.0, grad_accum=1, scaler=None):
    model.train()  # 切换到训练模式（启用 dropout）
    for step, batch in enumerate(dataloader):
        # 1. 数据移到 GPU
        batch = _move_batch(batch, device)

        # 2. 前向传播（在 autocast 上下文中自动混合精度）
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = model(...)
            loss = output.loss / grad_accum  # 除以累积步数

        # 3. 反向传播
        if use_amp:
            scaler.scale(loss).backward()  # AMP：缩放 loss 防止下溢
        else:
            loss.backward()  # 标准反向传播

        # 4. 每 grad_accum 步更新一次参数
        if (step + 1) % grad_accum == 0:
            if use_amp:
                scaler.unscale_(optimizer)  # 还原梯度尺度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if use_amp:
                scaler.step(optimizer)      # 更新参数 + 更新缩放因子
                scaler.update()
            else:
                optimizer.step()            # 标准参数更新
            optimizer.zero_grad()           # 清空梯度
```

**新手理解梯度累积**：

```
标准训练（batch=8）：          梯度累积（batch=1, grad_accum=8）：
  8 条数据一起算 loss              1 条数据算 loss → backward（累计梯度）
  → backward（计算梯度）           1 条数据算 loss → backward（累计梯度）
  → optimizer.step()（更新参数）   ...重复 8 次...
                                  → optimizer.step()（更新参数）
```

梯度的累加效果等价于 batch=8，但显存峰值降低了 8 倍。

**`evaluate()` 函数**：

```python
@torch.no_grad()  # 禁用梯度计算（省显存 + 加速）
def evaluate(model, dataloader, device):
    model.eval()  # 切换到评估模式（禁用 dropout）
    for batch in dataloader:
        output = model(...)
        batch_preds = output.logits.argmax(dim=-1)  # 取分数大的类别
        preds.append(batch_preds.cpu())
    # 汇总所有预测，计算 Accuracy/Precision/Recall/F1
    metrics = compute_binary_metrics(torch.cat(preds), torch.cat(labels))
    return metrics
```

#### 5.1.5 `metrics.py` — 指标计算

**作用**：计算二分类的 TP/TN/FP/FN 及衍生指标。

**核心逻辑**：

```python
def compute_binary_metrics(preds, labels):
    # 逐元素比较，统计四种结果
    tp = ((preds == 1) & (labels == 1)).sum()  # 预测=fraud, 真实=fraud ✓
    tn = ((preds == 0) & (labels == 0)).sum()  # 预测=normal, 真实=normal ✓
    fp = ((preds == 1) & (labels == 0)).sum()  # 预测=fraud, 真实=normal ✗ (误报!)
    fn = ((preds == 0) & (labels == 1)).sum()  # 预测=normal, 真实=fraud ✗ (漏检!)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)   # 判为诈骗的里面，有多少真是诈骗
    recall = tp / (tp + fn)      # 真正的诈骗中，有多少被检出
    f1 = 2 * precision * recall / (precision + recall)  # 调和平均
```

**新手理解混淆矩阵**：

```
              预测 Normal    预测 Fraud
真 Normal        TN            FP
真 Fraud         FN            TP
```

- TN (True Negative)：正常通话被正确放行 → 好的
- TP (True Positive)：诈骗通话被正确拦截 → 好的
- FP (False Positive)：正常通话被误判为诈骗 → 用户烦恼
- FN (False Negative)：诈骗通话被漏判为正常 → **用户被骗！最危险！**

#### 5.1.6 `prompt_templates.py` — 指令模板

**作用**：构建 SFT 训练的对话消息列表，以及解析模型生成的 JSON 响应。

**模板设计**：

```python
SYSTEM_PROMPT = "你是一个专业的电话诈骗检测助手..."
USER_TEMPLATE = """请分析以下通话记录...通话记录：{transcription}..."""
```

训练时 `format_fraud_binary_instruction(transcription, label=1, include_answer=True)` 生成：
```python
[
    {"role": "system", "content": "你是一个专业的..."},
    {"role": "user", "content": "请分析以下通话...通话记录：<ASR文本>"},
    {"role": "assistant", "content": '{"is_fraud": true, "reason": "...", "confidence": 0.95}'},
]
```

推理时 `include_answer=False` 仅生成前两条消息，模型需要补全 assistant 的回复。

**响应解析的三层回退策略**（`parse_fraud_binary_response`）：

```
第一层：尝试解析 JSON 代码块 (```json {...} ```)
   ↓ 失败
第二层：正则匹配 "is_fraud": true/false 键值对
   ↓ 失败
第三层：关键词匹配（"诈骗"/"正常通话"/"无诈骗" 等）
   ↓ 失败
返回 None（计入 parse_failures）
```

#### 5.1.7 `sft_data.py` — SFT 数据加载

**作用**：解析 JSONL 格式的多轮对话 SFT 数据。

**任务模式分类逻辑**：

```python
if num_messages <= 2:
    task_pattern = "SCENE_ONLY"       # 仅场景描述，无标签
elif num_messages <= 4:
    task_pattern = "FRAUD_BINARY"     # 二分类，有 fraud/normal 标签
else:
    task_pattern = "FRAUD_TYPE"       # 多分类，识别诈骗类型
```

`filter_binary_samples()` 函数过滤出仅有二分类标签的样本（排除 SCENE_ONLY）。

#### 5.1.8 `demo_backend.py` — 演示后端逻辑

**作用**：提供确定性的演示数据，使答辩展示不受模型推理波动影响。

**设计要点**：
- 3 个预置样例的检测结果是**硬编码**的（不是实时推理）——确保答辩演示 100% 可预期
- 文本检测模式使用**启发式关键词评分**（不依赖 GPU）——"安全账户""转账""验证码"等关键词线性叠加概率
- 关键词评分公式：`probability = min(0.72 + len(evidence) × 0.08, 0.98)`

### 5.2 训练脚本 (`scripts/`)

#### 5.2.1 `cache_multimodal_features.py` — 特征缓存

**这是整个流程的第一步，也是最重要的脚本。**

**流程图**：

```
输入: JSON (4000条) + 音频文件 (.mp3)
  │
  ├─→ WhisperAudioEncoder.encode()
  │     └─→ ffmpeg 解码 MP3 → 16kHz PCM
  │     └─→ log-Mel 频谱图 → 填充至 3000 帧
  │     └─→ Whisper encoder 前向传播
  │     └─→ 均值池化 → (768,) 音频 embedding
  │
  ├─→ [可选: ASR 模式] WhisperAudioEncoder.transcribe()
  │     └─→ Whisper decoder 自回归生成中文
  │
  └─→ TransformerTextEncoder.encode()
        └─→ Tokenizer 分词
        └─→ RoBERTa encoder 前向传播
        └─→ [CLS] token → (768,) 文本 embedding

输出: .pt 文件 (sample_id, audio_features, text_features, labels, [asr_texts])
```

**WhisperAudioEncoder 类的关键功能**：

```python
class WhisperAudioEncoder:
    def encode(self, audio_paths):
        """提取音频 embedding（encoder 模式）"""
        # 1. 加载音频 → 转换为 mel 频谱图 → pad 到 3000 帧
        mel = self._load_and_extract_mel(paths)  # (B, 80, 3000)
        # 2. 冻结的 encoder 推理
        with torch.no_grad():
            hidden = self.encoder(mel).last_hidden_state  # (B, 3000, 768)
        # 3. 时间维度均值池化
        return hidden.mean(dim=1)  # (B, 768)

    def transcribe(self, audio_paths):
        """ASR 转写（完整 encoder-decoder）"""
        mel = self._load_and_extract_mel(paths)
        with torch.no_grad():
            tokens = self.model.generate(mel, language="zh", task="transcribe")
        return self.tokenizer.decode(tokens)
```

#### 5.2.2 `train_fusion_classifier.py` — MLP 训练

**流程极简**（仅 96 行）：
1. `parse_args()` — 读取命令行参数
2. 加载训练/测试缓存 → 构建 DataLoader
3. 创建 `AudioTextFusionClassifier` 模型
4. 每个 epoch：`train_one_epoch()` → `evaluate()` → 打印指标 → 保存最佳模型
5. 将完整训练历史写入 `metrics.json`

**命令行参数速查**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train-cache` | 必填 | 训练集缓存 .pt 路径 |
| `--test-cache` | 必填 | 测试集缓存 .pt 路径 |
| `--epochs` | 20 | 训练轮数 |
| `--batch-size` | 64 | 每批样本数 |
| `--hidden-dim` | 256 | MLP 隐藏层维度 |
| `--dropout` | 0.2 | Dropout 率 |
| `--lr` | 1e-3 | 学习率 |
| `--device` | cuda | 设备 |

#### 5.2.3 `train_e2e_fusion.py` — 端到端训练

**与 MLP 训练的核心区别**：

| 维度 | MLP 训练 | E2E 训练 |
|------|---------|----------|
| 数据加载 | CachedFeatureDataset | E2EFeatureDataset（在线处理音频） |
| 模型 | AudioTextFusionClassifier | E2EFusionClassifier（含 Whisper encoder） |
| 文本特征 | 从缓存读取 | 从缓存读取（跳过 RoBERTa 在线加载） |
| 音频特征 | 从缓存读取 | Whisper encoder 在线计算 |
| 梯度累积 | 不需要 | 需要（batch=1, grad_accum=8） |
| 混合精度 | 不需要 | 需要 AMP fp16 |

**关键函数 `_freeze_except_last_layers`**：

```python
def _freeze_except_last_layers(encoder, num_unfreeze):
    layers = list(encoder.layers)  # Whisper-small 有 12 层
    for layer in layers[:-num_unfreeze]:  # 冻结前 10 层
        for p in layer.parameters():
            p.requires_grad_(False)  # 不需要梯度 = 不更新
    # 后 2 层保持 requires_grad=True，参与训练
```

#### 5.2.4 `train_sft_lora.py` — SFT LoRA 微调

**这是本项目最复杂的脚本（~500 行），核心流程**：

```
1. 加载 SFT JSONL 数据 → filter_binary_samples → SftInstructionDataset
2. 加载 Qwen2.5-1.5B-Instruct (4-bit) → apply LoRA adapter
3. 每个 epoch:
   a. 遍历 SftInstructionDataset
   b. 每个样本生成完整的对话 input_ids
   c. 对 assistant 回复部分计算 loss（其他部分 mask 掉）
   d. 梯度累积 + optimizer.step
4. 保存 LoRA adapter（仅 35MB）
5. 用 evaluate_sft_binary() 评估：生成 → 解析 JSON → 计算指标
```

**损失掩码（Loss Masking）详解**：

```python
# 假设对话 token 序列为：
# [System: 你是电诈...] [User: 请分析...] [Assistant: {"is_fraud": true...}]
#        ↑                       ↑                        ↑
#     labels=-100            labels=-100              labels=真实token

# 训练时：
# - System/User 部分的 loss 被忽略（labels=-100 是 ignore_index）
# - 仅 Assistant 生成的 JSON 部分计算 loss
# 这样模型学习的是"如何给通话文本生成检测结论"
# 而不是"如何复述 system prompt 和 user message"
```

#### 5.2.5 `serve_demo.py` — Web 演示服务器

**极简 HTTP 服务器**（仅 133 行，零框架依赖）：
- 使用 Python 标准库 `http.server.ThreadingHTTPServer`
- GET 请求 → 返回静态文件（HTML/CSS/JS）或 API JSON
- POST 请求 → 调用 `demo_backend.py` 中的检测逻辑
- 支持 CORS（跨域请求），方便前端开发调试

#### 5.2.6 其他脚本

| 文件 | 用途 |
|------|------|
| `predict_binary.py` | 加载 best_model.pt，对测试集批量推理，导出 CSV |
| `eval_sft_adapter.py` | 加载 SFT LoRA adapter，在测试集上评估生成式指标 |
| `screen_hard_cases.py` | 查找高置信度错判/低置信度样本，用于模型改进 |
| `generate_report_charts.py` | 生成论文 8 张图表（架构图、训练曲线、对比柱状图、混淆矩阵、SFT 分析） |
| `visualize_dataset.py` | 生成数据集 5 张分析图表（标签分布、音频时长、文本长度、关键词频率） |

### 5.3 前端 (`web/demo/`)

- `index.html` — 单页应用结构（三个 Tab：预置样例/上传检测/文本检测）
- `styles.css` — 响应式布局样式
- `app.js` — 前端交互逻辑（Tab 切换、API 调用、结果渲染）

### 5.4 其他文件

| 文件 | 用途 |
|------|------|
| `TRAINING_LOG.md` | 完整训练日志（每次实验的精确命令、指标、环境修复记录） |
| `docs/TELEANTIFRAUD_MULTIMODAL_TRAINING.md` | 训练路线图（后续开发者/agent 的操作手册） |
| `paper/teleantifraud_paper.tex` | 项目论文 LaTeX 源文件（22 页） |
| `paper/defense.pptx` | 答辩 PPT（15 页） |
| `configs/binary_fusion.example.yaml` | 推荐训练参数记录 |
| `tests/` | 7 个单元测试 |
| `.gitignore` | Git 忽略规则 |

---

## 6. 数据流全流程追踪

以下从"原始数据"到"检测结论"逐步骤追踪完整数据流。

### 6.1 特征缓存阶段（离线）

```
步骤 1: data.py::load_binary_samples()
  JSON 文件 → 4000 个 BinarySample 对象
  每个对象: {sample_id, audio_path, text, label(0/1)}

步骤 2: WhisperAudioEncoder.encode()
  audio_path → ffmpeg 解码 → 16kHz PCM → Mel 频谱图(80×3000)
  → Whisper encoder → (3000, 768) → mean(dim=0) → (768,)

步骤 3: [ASR 模式] WhisperAudioEncoder.transcribe()
  同上的 Mel 频谱图 → Whisper decoder → "你的银行卡涉嫌..."
  → TransformerTextEncoder.encode() → tokenizer → RoBERTa → (768,)

步骤 4: 保存缓存
  {sample_id: [...], audio_features: (4000,768),
   text_features: (4000,768), labels: (4000,), asr_texts: [...]}
  → binary_train_whisper_small_asr_roberta.pt (~25 MB)
```

### 6.2 MLP 训练阶段

```
步骤 1: CachedFeatureDataset.__init__()
  .pt 文件 → self.audio_features (4000,768)
           → self.text_features (4000,768)
           → self.labels (4000,)

步骤 2: DataLoader.__iter__()
  每次 yield 一个 batch: {audio_features: (64,768),
                         text_features: (64,768),
                         labels: (64,)}

步骤 3: AudioTextFusionClassifier.forward()
  cat([audio, text], dim=-1) → (64, 1536)
  → LayerNorm → Linear(256) → GELU → Dropout → Linear(2)
  → logits: (64, 2)

步骤 4: F.cross_entropy(logits, labels)
  → loss 标量 → loss.backward() → optimizer.step()
  (重复 N 个 batch × 20 个 epoch)
```

### 6.3 推理阶段

```
输入音频
  → ffmpeg 解码 → Mel 频谱图
  → Whisper encoder → audio_emb (768,)
  → Whisper decoder → ASR 中文文本
  → RoBERTa encoder → text_emb (768,)
  → cat → (1536,) → MLP → logits (2,)
  → argmax → "fraud" / "normal"
```

---

## 7. 复现指南

### 7.1 最小复现（仅方案二，~30 分钟）

```bash
# 前提：已安装所有依赖，数据已就位

# Step 1: 运行测试（确认环境正常）
python -m unittest discover -s tests -v

# Step 2: 缓存测试集特征（先用小数据验证流程）
python scripts/cache_multimodal_features.py \
  --json-path data/binary_classification/binary_classification/test.json \
  --data-root data \
  --output artifacts/features/binary_test_asr_check.pt \
  --audio-model openai/whisper-small \
  --text-model hfl/chinese-roberta-wwm-ext \
  --use-asr --batch-size 1 --limit 20 --device cuda

# Step 3: 如果 Step 2 成功，缓存完整训练+测试集
# 训练集（~20 分钟）
python scripts/cache_multimodal_features.py \
  --json-path data/binary_classification/binary_classification/train.json \
  --data-root data \
  --output artifacts/features/binary_train_asr.pt \
  --use-asr --batch-size 1 --device cuda

# 测试集（~2 分钟）
python scripts/cache_multimodal_features.py \
  --json-path data/binary_classification/binary_classification/test.json \
  --data-root data \
  --output artifacts/features/binary_test_asr.pt \
  --use-asr --batch-size 1 --device cuda

# Step 4: 训练 MLP 分类器（~5 分钟）
python scripts/train_fusion_classifier.py \
  --train-cache artifacts/features/binary_train_asr.pt \
  --test-cache artifacts/features/binary_test_asr.pt \
  --output-dir runs/my_reproduction \
  --epochs 20 --batch-size 64 --device cuda

# Step 5: 查看结果
cat runs/my_reproduction/metrics.json | python -m json.tool | tail -20
```

### 7.2 完整复现（四条路线，~10 小时）

参见 `docs/TELEANTIFRAUD_MULTIMODAL_TRAINING.md` 中的详细步骤。

### 7.3 复现检查清单

| 检查项 | 验证方法 | 预期 |
|--------|---------|------|
| CUDA 可用 | `python -c "import torch; print(torch.cuda.is_available())"` | True |
| Whisper 可加载 | 运行 cache_multimodal_features.py 不报 404 | 自动下载模型 |
| 特征缓存完成 | `ls -lh artifacts/features/*.pt` | 文件存在，~25MB |
| MLP 训练收敛 | Loss 从 ~0.06 降至 ~0.002 | F1 ≥ 0.99 |
| FFmpeg 工作 | `ffmpeg -version` | 显示版本号 |

---

## 8. 学习路线建议

### 8.1 新手入门路径（2-3 周，每天 2-3 小时）

**第 1 周：理论基础**
1. 阅读本文档第 1 节"前置知识速览"中列出的资源
2. 用 PyTorch 写一个 MNIST 手写数字分类器（10 行代码级别的入门项目）
3. 理解 forward / backward / optimizer.step 的含义

**第 2 周：代码走读**
1. 从 `data.py` → `features.py` → `model.py` → `training.py` 顺序阅读
2. 在 `train_fusion_classifier.py` 中加 `print` 语句，观察每步的数据形状
3. 运行单元测试，在测试代码中打断点理解数据流

**第 3 周：动手实验**
1. 复现方案二（见第 7.1 节）
2. 尝试修改超参数（hidden_dim=128/512、lr=1e-4/5e-4）观察指标变化
3. 用 `visualize_dataset.py` 生成数据图表，理解数据特征

### 8.2 进阶路线（4-6 周）

1. 理解方案三的梯度累积和 AMP 原理，尝试在 CIFAR-10 上实现
2. 学习 LoRA 原理（阅读原论文 Introduction + Method 部分），用 `peft` 库在 GPT-2 上做微调实验
3. 阅读 Whisper 论文的 Encoder-Decoder 架构描述
4. 尝试为项目添加新功能（如新的数据增强方法、新的评估指标）

### 8.3 核心概念速查表

| 你想要... | 学习关键词 | 推荐资源 |
|-----------|-----------|----------|
| 理解 Transformer | Self-Attention, Multi-Head Attention, Positional Encoding | Jay Alammar 博客 |
| 理解 BERT/RoBERTa | Masked Language Model, [CLS] token, Fine-tuning | BERT 原论文 |
| 理解 Whisper | Encoder-Decoder, Log-Mel Spectrogram, Seq2Seq | Whisper 论文 + 博客 |
| 理解 LoRA | Low-Rank Matrix, Adapter, Rank-Decomposition | LoRA 论文 Section 3 |
| 理解 QLoRA | NF4 Quantization, Double Quantization, Paged Optimizer | QLoRA 论文 Section 3-4 |
| 理解 AMP | fp16, Loss Scaling, Gradient Underflow | PyTorch AMP Tutorial |
| 提高模型效果 | Data Augmentation, Hyperparameter Tuning, Early Stopping | 李沐《动手学深度学习》 |
| 看懂 PyTorch 报错 | Tensor Shape Mismatch, Device Mismatch, grad vs no_grad | PyTorch 官方 Debugging Guide |

---

## 9. 常见问题排查

### 9.1 环境问题

| 错误信息 | 原因 | 解决方案 |
|----------|------|----------|
| `torch.cuda.is_available() == False` | 安装的是 CPU 版 PyTorch | 去 pytorch.org 选 CUDA 版本重装 |
| `Couldn't find appropriate backend to handle uri ...mp3` | Windows 缺少 MP3 解码器 | `winget install ffmpeg` |
| `CUDA out of memory` | 显存不足 | 降 batch-size 为 1，或加 `--device cpu` |
| `No module named 'teledeceit'` | 未将 src/ 加入 Python 路径 | 项目脚本自动处理，直接运行 |
| `Whisper expects mel input features to be of length 3000, but found 2627` | 音频太短 | 已自动 pad，如仍报错说明 pad 逻辑未触发 |

### 9.2 训练问题

| 现象 | 可能原因 | 排查方法 |
|------|---------|----------|
| Loss 不下降 | 学习率太大/太小 | 尝试 lr=1e-4, 5e-4, 1e-3 |
| F1 始终在 0.5 左右 | 模型在"猜"，没有学到特征 | 检查特征缓存是否正确生成 |
| 训练很快但测试很差 | 过拟合 | 增大 dropout、减小 hidden_dim、加早停 |
| E2E 训练 loss 变成 NaN | AMP 梯度溢出 | 调低 lr、增大 grad_accum |

### 9.3 概念问题

**Q: "768 维向量"到底长什么样？**
A: 就是一个包含 768 个浮点数的列表（list）。例如 `[0.023, -0.451, 0.892, ...]`。用 768 个数字来"编码"一段音频或文本的语义信息。相似的音频/文本，它们的 768 维向量在空间中距离较近。

**Q: 模型文件为什么这么小（1.6 MB）？**
A: 因为只保存了 MLP 分类头的权重（0.4M 参数），没有保存 Whisper（88M）和 RoBERTa（102M）的参数——这些大模型在推理时从 HuggingFace 重新加载。

**Q: 为什么不直接端到端训练所有参数？**
A: 全量训练 Whisper (88M) + RoBERTa (102M) + MLP 需要远超 6 GB 的显存。即使用梯度累积，batch=1 下 190M 参数的 AdamW 状态（momentum + variance）本身就接近 2 GB，加上 fp16 模型参数 ~380 MB 和中间激活，6 GB 显存不足以容纳。

**Q: F1 为什么不直接用 Accuracy？**
A: 如果数据集中 95% 是 normal，模型只需永远说"normal"就能拿 95% accuracy，但它一个诈骗也检测不出来。F1 同时考虑 precision 和 recall，不会被类别比例影响。

---

## 附录：技术术语中英对照

| 中文 | English | 缩写 |
|------|---------|------|
| 自动语音识别 | Automatic Speech Recognition | ASR |
| 自然语言处理 | Natural Language Processing | NLP |
| 大语言模型 | Large Language Model | LLM |
| 低秩适配 | Low-Rank Adaptation | LoRA |
| 参数高效微调 | Parameter-Efficient Fine-Tuning | PEFT |
| 混合精度训练 | Automatic Mixed Precision | AMP |
| 梯度累积 | Gradient Accumulation | — |
| 假阳性 / 误报 | False Positive | FP |
| 假阴性 / 漏检 | False Negative | FN |
| 交叉熵损失 | Cross-Entropy Loss | CE |
| 梅尔频谱图 | Mel Spectrogram | — |

---

> **最后更新**：2026-05-31
> **适用版本**：commit `eb21749` 及之后
