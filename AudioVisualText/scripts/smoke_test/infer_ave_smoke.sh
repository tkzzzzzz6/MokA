#!/bin/bash

# Smoke test inference for AVE - minimal test without full dataset
# Usage: bash scripts/smoke_test/infer_ave_smoke.sh

# Environment Variables - use only 1 GPU for smoke test
WORLD_SIZE=1
NPROC_PER_NODE=1
MASTER_PORT=6688
RANK=0

# ========== PLEASE MODIFY THESE PATHS ==========
llama_ckpt_path=/root/autodl-tmp/MokA/AudioVisualText/models/Llama-2-7b-chat-hf
ckpt_dir=YOUR_CHECKPOINT_PATH  # Path to your fine-tuned checkpoint (after training)
vit_ckpt_path=/root/autodl-tmp/MokA/AudioVisualText/models/clip-vit-large-patch14
BEATs_ckpt_path=/root/autodl-tmp/MokA/AudioVisualText/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
# ==============================================

# Local batch size - 1 for smoke test
LOCAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1

export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM='true'
export ASCEND_LAUNCH_BLOCKING='1'

# Enable Hugging Face mirror if needed
# export HF_ENDPOINT=https://hf-mirror.com

echo "Starting AVE smoke test inference with 1 GPU..."
echo "Using test data: smoke_test_data/AVE_data (2 samples)"

torchrun --nproc_per_node $NPROC_PER_NODE \
    --master_port $MASTER_PORT \
    scripts/smoke_test/inference_cut_smoke.py \
    --llm_name llama \
    --reserved_modality None \
    --loramethod test \
    --model_name_or_path $llama_ckpt_path \
    --freeze_backbone True \
    --lora_enable True \
    --bits 32 \
    --lora_r 444 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --blc_weight 1 \
    --blc_alpha 1 \
    --bf16 False \
    --tf32 False \
    --fp16 False \
    --ckpt_dir $ckpt_dir \
    --avqa_task False \
    --ave_task True \
    --visual_branch True \
    --video_frame_nums 8 \
    --vit_ckpt_path $vit_ckpt_path \
    --image_size 224 \
    --patch_size 14 \
    --visual_query_token_nums 32 \
    --audio_branch True \
    --BEATs_ckpt_path $BEATs_ckpt_path \
    --audio_query_token_nums 32 \
    --output_dir output/smoke_test

echo "Smoke test completed. Check output/smoke_test for results."
