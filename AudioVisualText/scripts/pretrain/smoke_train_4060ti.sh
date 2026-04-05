#!/bin/bash

# Smoke test for audio pretraining on 4060Ti 8G
# Quick validation of the training pipeline with minimal resources

echo "=== MokA Audio Pretraining Smoke Test (4060Ti 8G) ==="

# Must run from AudioVisualText root
if [[ ! -f "scripts/pretrain/pretrain_audio.sh" ]]; then
  echo "[error] Please run from AudioVisualText root directory."
  exit 1
fi

# Environment Variables for single GPU
WORLD_SIZE=1
NPROC_PER_NODE=1
MASTER_PORT=6666
RANK=0

llama_ckpt_path=llama2-7b-chat-hf

# Training Arguments for 8GB GPU
LOCAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
# Effective global batch size: 1 * 1 * 1 * 4 = 4

# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=pretrain_smoke_test
RUN_NAME=audio_pretrain_4060ti
OUTP_DIR=results
export CUDA_VISIBLE_DEVICES='0'
export TOKENIZERS_PARALLELISM='true'
export ASCEND_LAUNCH_BLOCKING='1'

echo "[config] NPROC_PER_NODE=$NPROC_PER_NODE"
echo "[config] LOCAL_BATCH_SIZE=$LOCAL_BATCH_SIZE"
echo "[config] GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"
echo "[config] Max steps: 10 (smoke test)"
echo "[config] GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

torchrun --nproc_per_node $NPROC_PER_NODE \
    --master_port $MASTER_PORT \
    scripts/pretrain/pretrain.py \
    --deepspeed deepspeed/stage3-offload.json \
    --llm_name llama \
    --model_name_or_path $llama_ckpt_path \
    --freeze_backbone True \
    --lora_enable False \
    --bits 32 \
    --bf16 True \
    --tf32 False \
    --fp16 False \
    --visual_branch False \
    --image_caption_task False \
    --video_caption_task False \
    --video_frame_nums 8 \
    --vit_ckpt_path clip-vit-large-patch14 \
    --select_feature patch \
    --image_size 224 \
    --patch_size 14 \
    --visual_query_token_nums 32 \
    --audio_branch True \
    --audio_caption_task True \
    --BEATs_ckpt_path BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --audio_query_token_nums 32 \
    --output_dir $OUTP_DIR/$WANDB_PROJECT/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size $LOCAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --ddp_find_unused_parameters True \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --half_precision_backend "auto" \
    --dataloader_num_workers 0 \
    --report_to tensorboard \
    --max_steps 10

echo "[done] Smoke test completed."
