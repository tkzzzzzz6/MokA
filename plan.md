## 🔍 MokA AudioVisualText 最小可运行测试流程（快速验通）

如果只想先验证代码能跑通，可按下面步骤执行。

---

### 🔑 前置准备

```bash
# 1. 登录 Hugging Face
huggingface-cli login --token YOUR_HF_TOKEN

# 2. 设置 huggingface 镜像加速（国内网络推荐）
export HF_ENDPOINT=https://hf-mirror.com
```

---

### 📥 第一步：下载所有必需的预训练权重

> 所有命令都从 `AudioVisualText/` 根目录执行

| 权重 | 大小 | 下载命令 | 存放位置 |
|------|-----:|----------|----------|
| LLaMA-2-7B-Chat-HF | ~13GB | `huggingface-cli download daryl149/llama-2-7b-chat-hf --local-dir models/Llama-2-7b-chat-hf` | `models/Llama-2-7b-chat-hf/` |
| CLIP-ViT-Large-Patch14 | ~1.7GB | `huggingface-cli download openai/clip-vit-large-patch14 --local-dir models/clip-vit-large-patch14` | `models/clip-vit-large-patch14/` |
| BERT-base-uncased | ~440MB | `huggingface-cli download google-bert/bert-base-uncased --local-dir models/google-bert-base-uncased` | `models/google-bert-base-uncased/` |
| BEATs 预训练权重 | ~347MB | `huggingface-cli download WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2 BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --local-dir models` | `models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt` |

**一键完整下载：**
```bash
cd AudioVisualText
mkdir -p models

# Llama-2-7B-Chat-HF
mkdir -p models/Llama-2-7b-chat-hf
huggingface-cli download daryl149/llama-2-7b-chat-hf --local-dir models/Llama-2-7b-chat-hf

# CLIP-ViT-Large-Patch14
mkdir -p models/clip-vit-large-patch14
huggingface-cli download openai/clip-vit-large-patch14 --local-dir models/clip-vit-large-patch14

# BERT-base-uncased
mkdir -p models/google-bert-base-uncased
huggingface-cli download google-bert/bert-base-uncased --local-dir models/google-bert-base-uncased

# BEATs 预训练权重
huggingface-cli download WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2 BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --local-dir models
```

---

### ⚡ 第二步（推荐）：下载作者预训练 projectors

作者已提供预训练好的音频/视觉投影器，可以**跳过 Stage 1 预训练**，直接进入 Stage 2 微调或推理测试，节省大量时间与算力。

```bash
# 音频 projector (~19MB)
huggingface-cli download ahsgdxhs/Crab audio_pretrain.bin --local-dir models
# 视觉 projector (~19MB)
huggingface-cli download ahsgdxhs/Crab visual_pretrain.bin --local-dir models
```

---

### 🛠️ 第三步：修改代码中的路径

**修改 `models/multimodal_encoder.py` 中的路径：**
```python
# 第34行，CLIP 路径
model_name_or_path = '/root/autodl-tmp/MokA/AudioVisualText/models/clip-vit-large-patch14'
# 第87行，BERT 路径
bert_ckpt_path = '/root/autodl-tmp/MokA/AudioVisualText/models/google-bert-base-uncased'
# 第173行，BEATs 权重路径
ckpt_path = '/root/autodl-tmp/MokA/AudioVisualText/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
# 第208行，BERT 路径
bert_ckpt_path = '/root/autodl-tmp/MokA/AudioVisualText/models/google-bert-base-uncased'
```

---

### 🍱 第四步：准备数据集

#### 选项 A：快速验证 - 准备 AudioCaps 小子集（预训练用）
```bash
cd AudioVisualText
bash scripts/pretrain/prepare_audiocaps_500_100.sh
```
**脚本会完成：**
- 从 Hugging Face 获取 AudioCaps 元信息
- 从 YouTube 下载音频片段：训练 500 + 验证 100（每段 10 秒）
- 总大小约：`600 × 1.5MB ≈ 900MB`
- 输出目录：`AudioVisualText/AudioCaps/`

#### 选项 B：smoke test 数据（已准备好）
项目已经自带 2 个 AVE 测试样本在 `smoke_test_data/AVE_data/`，用于快速验证推理流程。

#### 选项 C：完整测试集
- **AVE 测试集**：标注 JSON 在 `huggingface.co/yake0409/MokA_AudioVisualText`，原始视频来自 `AVE-ECCV18`
- **MUSIC-AVQA 测试集**：标注 JSON 在 `huggingface.co/yake0409/MokA_AudioVisualText`，原始视频来自 `MUSIC-AVQA`

---

### ✅ 第五步：运行 smoke test 验证

修改 `scripts/smoke_test/infer_ave_smoke.sh` 中的路径：
```bash
llama_ckpt_path=/root/autodl-tmp/MokA/AudioVisualText/models/Llama-2-7b-chat-hf
ckpt_dir=/path/to/your/fine-tuned/checkpoint  # 微调后填写
vit_ckpt_path=/root/autodl-tmp/MokA/AudioVisualText/models/clip-vit-large-patch14
BEATs_ckpt_path=/root/autodl-tmp/MokA/AudioVisualText/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
```

运行：
```bash
bash scripts/smoke_test/infer_ave_smoke.sh
```

这会用 2 个测试样本验证：数据加载 → 模型加载 → 推理 整个流程。

---

### 🔥 第六步：完整微调（使用预训练 projector）

如果使用作者预训练的 projectors，修改 `scripts/finetune/ft_ave.sh`：
```bash
llama_ckpt_path=/root/autodl-tmp/MokA/AudioVisualText/models/Llama-2-7b-chat-hf
vit_ckpt_path=/root/autodl-tmp/MokA/AudioVisualText/models/clip-vit-large-patch14
BEATs_ckpt_path=/root/autodl-tmp/MokA/AudioVisualText/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
```

在 `scripts/finetune/finetune.py` 中设置：
```python
audio_pretrain_ckpt=/root/autodl-tmp/MokA/AudioVisualText/models/audio_pretrain.bin
visual_pretrain_ckpt=/root/autodl-tmp/MokA/AudioVisualText/models/visual_pretrain.bin
```

运行微调：
```bash
# AVE 数据集
bash scripts/finetune/ft_ave.sh

# MUSIC-AVQA 数据集
bash scripts/finetune/ft_musicavqa.sh
```

---

### 📦 推荐：直接使用作者微调好的模型推理

作者已经在 Hugging Face 发布了微调好的检查点，可以直接下载推理，不需要自己训练。

#### 仓库说明

| 仓库 | 模态 | 任务 |
|------|------|------|
| `yake0409/MokA_VisualText` | 👁️ 仅视觉+文本 | 图像/视频问答 |
| `yake0409/MokA_AudioVisualText` | 👁️🔊 音频+视觉+文本 | 音视频问答（当前项目） |

在 `yake0409/MokA_AudioVisualText` 中：
- `AVE_checkpoint/` - AVE 数据集微调完成的检查点
- `AVQA_checkpoint/` - MUSIC-AVQA 数据集微调完成的检查点

#### 下载命令

```bash
export HF_ENDPOINT=https://hf-mirror.com
cd /root/autodl-tmp/MokA

# 下载整个 AudioVisualText 检查点仓库
mkdir -p MokA_AudioVisualText
huggingface-cli download yake0409/MokA_AudioVisualText --local-dir MokA_AudioVisualText

# 或者，只下载 AVE 检查点
huggingface-cli download yake0409/MokA_AudioVisualText --include "AVE_checkpoint/*" --local-dir MokA_AudioVisualText

# 或者，只下载 MUSIC-AVQA 检查点
huggingface-cli download yake0409/MokA_AudioVisualText --include "AVQA_checkpoint/*" --local-dir MokA_AudioVisualText
```

#### 推理使用

在推理脚本 `scripts/finetune/infer_ave.sh` 或 `infer_avqa.sh` 中设置：
```bash
ckpt_dir=/root/autodl-tmp/MokA/MokA_AudioVisualText/AVE_checkpoint
# 或 ckpt_dir=/root/autodl-tmp/MokA/MokA_AudioVisualText/AVQA_checkpoint
```

然后运行推理：
```bash
# AVE 推理
bash scripts/finetune/infer_ave.sh

# MUSIC-AVQA 推理
bash scripts/finetune/infer_avqa.sh
```

结果会保存在 `$ckpt_dir/inference_results/` 下。

---

### 📊 下载总量估算

| 项目 | 大小 |
|------|-----:|
| AudioCaps（600 个音频） | ~0.9GB |
| LLaMA-2-7B | 13GB |
| CLIP-ViT | 1.7GB |
| BEATs | 0.4GB |
| BERT | 0.4GB |
| 作者预训练 projector | 0.04GB |
| **总计** | **~16.5GB** |

> 结论：若只做"流程验通"，完成上述步骤后即可开始推理测试。

---
