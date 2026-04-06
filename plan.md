## 🔍 最小可运行测试流程（快速验通）

如果只想先验证代码能跑通，可按下面步骤执行。

### 1. 数据准备（你正在执行）
```bash
cd AudioVisualText
bash scripts/pretrain/prepare_audiocaps_500_100.sh
```

**脚本会完成：**
- 从 Hugging Face 获取 AudioCaps 元信息
- 从 YouTube 下载音频片段：训练 500 + 验证 100（每段 10 秒）
- 总大小约：`600 × 1.5MB ≈ 900MB`

**输出目录：**
- `AudioVisualText/AudioCaps/`

---

### 2. 必须下载的预训练权重

| 权重 | 大小 | 下载来源 | 存放位置 / 需修改项 |
|---|---:|---|---|
| LLaMA-2-7B-Chat-HF | ~13GB | Hugging Face | 自定义路径（脚本中修改 `llama_ckpt_path`） |
| CLIP-ViT-Large-Patch14 | ~1.7GB | Hugging Face | 自定义路径（修改 `vit_ckpt_path`） |
| BEATs 音频编码器 | ~400MB | 微软 repo | `AudioVisualText/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt` |
| BERT-base-uncased | ~440MB | Hugging Face | 自定义路径（代码中硬编码，需手动修改） |

---

### 3. 推荐：跳过 Stage 1，自带 projector 直接测试

作者已提供预训练 projector，可直接下载：
- 音频 projector：`audio_pretrain.bin`（~19MB）
- 视觉 projector：`visual_pretrain.bin`（~19MB）

这样可跳过 Stage 1 预训练，直接进入 **Stage 2 微调** 或 **推理测试**，节省大量时间与算力。

---

### 4. 若需完整训练 / 推理，还需要以下数据

#### AVE 测试集
- 标注 JSON：`huggingface.co/yake0409/MokA_AudioVisualText`
- 原始视频：`AVE-ECCV18`

#### MUSIC-AVQA 测试集
- 标注 JSON：`huggingface.co/yake0409/MokA_AudioVisualText`
- 原始视频：`MUSIC-AVQA`

---

## 📊 简单测试下载总量估算

| 项目 | 大小 |
|---|---:|
| AudioCaps（600 个音频） | ~0.9GB |
| LLaMA-2-7B | 13GB |
| CLIP-ViT | 1.7GB |
| BEATs | 0.4GB |
| BERT | 0.4GB |
| 作者预训练 projector | 0.04GB |
| **总计** | **~16.5GB** |

> 结论：若只做“流程验通”，完成 `prepare_audiocaps` 并下载上述预训练权重后，即可开始推理测试。


```bash
# 需要先登录：你的 Hugging Face access token
huggingface-cli login --token YOUR_HF_TOKEN

# 设置 huggingface 镜像加速
export HF_ENDPOINT=https://hf-mirror.com

# ========== 1. 创建目录 ==========
cd AudioVisualText
mkdir -p models

# ========== 2. 下载 Llama-2-7B-Chat-HF ==========
mkdir -p models/Llama-2-7b-chat-hf
huggingface-cli download daryl149/llama-2-7b-chat-hf --local-dir models/Llama-2-7b-chat-hf

# ========== 3. 下载 CLIP-ViT-Large-Patch14 ==========
mkdir -p models/clip-vit-large-patch14
huggingface-cli download openai/clip-vit-large-patch14 --local-dir models/clip-vit-large-patch14

# ========== 4. 下载 BERT-base-uncased ==========
mkdir -p models/google-bert-base-uncased
huggingface-cli download google-bert/bert-base-uncased --local-dir models/google-bert-base-uncased

# ========== 5. 下载 BEATs 预训练权重 ==========
# 放到 models 根目录
huggingface-cli download WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2 BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --local-dir models

# ========== 6. (推荐) 下载作者预训练好的 projectors ==========
# 跳过 Stage 1 预训练，直接进入微调
huggingface-cli download ahsgdxhs/Crab audio_pretrain.bin --local-dir models
huggingface-cli download ahsgdxhs/Crab visual_pretrain.bin --local-dir models
```

---

### 3. 修改代码中的路径

**修改 `models/multimodal_encoder.py` 中的路径：**
```python
# 第34行，CLIP 路径
model_name_or_path = '/path/to/AudioVisualText/models/clip-vit-large-patch14'
# 第87行，BERT 路径
bert_ckpt_path = '/path/to/AudioVisualText/models/google-bert-base-uncased'
# 第173行，BEATs 权重路径
ckpt_path = '/path/to/AudioVisualText/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
# 第208行，BERT 路径
bert_ckpt_path = '/path/to/AudioVisualText/models/google-bert-base-uncased'
```

### 4. 运行 smoke test 验证 (AVE 2个样本)

修改 `scripts/smoke_test/infer_ave_smoke.sh` 中的路径：
```bash
llama_ckpt_path=/path/to/AudioVisualText/models/Llama-2-7b-chat-hf
ckpt_dir=/path/to/your/fine-tuned/checkpoint  # 微调后填写
vit_ckpt_path=/path/to/AudioVisualText/models/clip-vit-large-patch14
BEATs_ckpt_path=/path/to/AudioVisualText/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
```

运行：
```bash
bash scripts/smoke_test/infer_ave_smoke.sh
```

### 5. 完整微调 (使用预训练 projector)

如果使用作者预训练的 projectors，修改 `scripts/finetune/ft_ave.sh`：
```bash
llama_ckpt_path=/path/to/AudioVisualText/models/Llama-2-7b-chat-hf
vit_ckpt_path=/path/to/AudioVisualText/models/clip-vit-large-patch14
BEATs_ckpt_path=/path/to/AudioVisualText/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
# 在 scripts/finetune/finetune.py 中设置：
# audio_pretrain_ckpt=/path/to/AudioVisualText/models/audio_pretrain.bin
# visual_pretrain_ckpt=/path/to/AudioVisualText/models/visual_pretrain.bin
```

运行微调：
```bash
bash scripts/finetune/ft_ave.sh
```


