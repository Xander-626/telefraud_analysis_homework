---
language:
- zh
license: apache-2.0
task_categories:
- text-classification
- automatic-speech-recognition
pretty_name: TeleAntiFraud
size_categories:
- 10K<n<100K
tags:
- audio-text
- fraud-detection
- chinese
- llm
- sft
configs:
- config_name: default
  data_files:
  - split: train
    path: viewer/train.parquet
  - split: test
    path: viewer/test.parquet
---

# TeleAntiFraud

Sanitized public release of the **TeleAntiFraud** audio-text fraud detection dataset.

This repository contains public metadata splits, audio archives, and a small preview set for quick inspection on the dataset page.

## License

Copyright 2025 Zhiming Ma. All rights reserved.

Licensed under the Apache License, Version 2.0.

## Overview

TeleAntiFraud is a Chinese audio-text fraud detection dataset designed for:

- binary fraud detection from call audio
- multi-turn audio-text instruction tuning
- speech understanding and fraud-risk reasoning

The public release removes machine-specific paths from the original research environment and normalizes audio references to relative paths.

## Contents

- `binary_classification.zip`
  - `train.json`: 4,000 binary fraud classification samples
  - `test.json`: 400 binary fraud classification samples
- `sft.zip`
  - `train.jsonl`: 27,146 multi-turn SFT samples
  - `test.jsonl`: 6,807 multi-turn SFT samples
- `audio.zip`
  - referenced audio files normalized under `audio/...`
- `dataset_manifest.json`
- `preview/`
  - a few small MP3 examples for quick listening on the Hub page
- `viewer/`
  - lightweight parquet files used by the Hugging Face dataset viewer

## Splits

| Package | File | Samples | Description |
| --- | --- | ---: | --- |
| `binary_classification.zip` | `train.json` | 4,000 | binary call-level fraud classification |
| `binary_classification.zip` | `test.json` | 400 | binary call-level fraud classification |
| `sft.zip` | `train.jsonl` | 27,146 | multi-turn SFT data with audio-grounded prompts |
| `sft.zip` | `test.jsonl` | 6,807 | multi-turn SFT data with audio-grounded prompts |

## Schema Summary

### Binary classification

Each sample keeps a prompt-style structure and a label:

```json
{
  "prompt": [
    {
      "role": "system",
      "content": "..."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "audio",
          "audio_url": "audio/..."
        },
        {
          "type": "text",
          "text": "..."
        }
      ]
    }
  ],
  "answer": "fraud"
}
```

### SFT

Each line in `train.jsonl` or `test.jsonl` is a JSON object containing multi-turn messages and audio-grounded prompts for scene understanding, fraud judgment, and related reasoning tasks.

## Preview

Small preview files are provided for direct listening without downloading the full `audio.zip`.

| Example | Label | Audio | Notes |
| --- | --- | --- | --- |
| `normal_example.mp3` | `normal` | [link](https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud/resolve/main/preview/normal_example.mp3) | binary classification sample |
| `fraud_example_1.mp3` | `fraud` | [link](https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud/resolve/main/preview/fraud_example_1.mp3) | binary classification sample |
| `fraud_example_2.mp3` | `fraud` | [link](https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud/resolve/main/preview/fraud_example_2.mp3) | binary classification sample |

Preview metadata is also available in `preview/preview_samples.json`.

## Viewer Support

The Hugging Face dataset viewer is configured with lightweight parquet files in `viewer/train.parquet` and `viewer/test.parquet`. These files expose a stable preview table with:

- `id`
- `task`
- `audio_path`
- `instruction`
- `label`

## Sanitization

- Absolute local paths from the original research environment were removed.
- Audio references were normalized to relative paths under `audio/`.
- The original field structure was kept whenever possible to avoid breaking downstream scripts.

## Usage Notes

- This release is packaged as zip archives to make distribution of the audio assets more manageable.
- Audio references inside JSON / JSONL files are relative paths, not absolute local paths.
- If you unpack `audio.zip`, the metadata files can be used directly with the normalized `audio/...` paths.
- For project code and evaluation scripts, see the GitHub repository below.

## Related Resources

- GitHub: https://github.com/JimmyMa99/TeleAntiFraud
- Evaluation scripts: https://github.com/JimmyMa99/TeleAntiFraud/tree/main/evaluation
- ModelScope: https://www.modelscope.cn/datasets/JimmyMa99/TeleAntiFraud-28k
- SAFE-QAQ (ACL 2026): https://arxiv.org/abs/2601.01392
