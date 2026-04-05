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

# 然后下载
mkdir -p models/Llama-2-7b-chat-hf

export HF_ENDPOINT=https://hf-mirror.com

# hf download meta-llama/Llama-2-7b-chat-hf --local-dir models/Llama-2-7b-chat-hf

hf download daryl149/llama-2-7b-chat-hf --local-dir models/Llama-2-7b-chat-hf
```

