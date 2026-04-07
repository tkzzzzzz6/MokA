Param(
    [int]$WorldSize = 1,
    [int]$NprocPerNode = 4,
    [int]$MasterPort = 6666,
    [int]$Rank = 0,
    [string]$LlamaCkptPath = "llama2-7b-chat-hf",
    [int]$LocalBatchSize = 2,
    [int]$GradientAccumulationSteps = 1,
    [string]$RunName = "audio_pretrain",
    [string]$OutputDir = "results",
    [string]$CudaVisibleDevices = "0,1,2,3,4,5,6,7"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path "scripts/pretrain/pretrain.py")) {
    Write-Error "Please run this script from AudioVisualText root directory."
}

if (-not (Get-Command torchrun -ErrorAction SilentlyContinue)) {
    Write-Error "torchrun not found. Activate your training environment first."
}

$GlobalBatchSize = $WorldSize * $NprocPerNode * $LocalBatchSize * $GradientAccumulationSteps

$env:TRANSFORMERS_OFFLINE = "1"
$env:WANDB_PROJECT = "pretrain"
$env:CUDA_VISIBLE_DEVICES = $CudaVisibleDevices
$env:TOKENIZERS_PARALLELISM = "true"
$env:ASCEND_LAUNCH_BLOCKING = "1"

Write-Host "pretrain"
Write-Host "[config] global_batch_size=$GlobalBatchSize"

& torchrun `
    --nproc_per_node $NprocPerNode `
    --master_port $MasterPort `
    scripts/pretrain/pretrain.py `
    --deepspeed deepspeed/stage2-offload.json `
    --llm_name llama `
    --model_name_or_path $LlamaCkptPath `
    --freeze_backbone True `
    --lora_enable False `
    --bits 32 `
    --bf16 False `
    --tf32 False `
    --fp16 False `
    --visual_branch False `
    --image_caption_task False `
    --video_caption_task False `
    --video_frame_nums 8 `
    --vit_ckpt_path clip-vit-large-patch14 `
    --select_feature patch `
    --image_size 224 `
    --patch_size 14 `
    --visual_query_token_nums 32 `
    --audio_branch True `
    --audio_caption_task True `
    --BEATs_ckpt_path BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt `
    --audio_query_token_nums 32 `
    --output_dir "$OutputDir/$($env:WANDB_PROJECT)/$RunName" `
    --num_train_epochs 1 `
    --per_device_train_batch_size $LocalBatchSize `
    --per_device_eval_batch_size $LocalBatchSize `
    --gradient_accumulation_steps $GradientAccumulationSteps `
    --ddp_find_unused_parameters True `
    --evaluation_strategy no `
    --save_strategy steps `
    --save_steps 0.33 `
    --save_total_limit 5 `
    --learning_rate 1e-4 `
    --weight_decay 0. `
    --warmup_ratio 0.03 `
    --lr_scheduler_type cosine `
    --logging_steps 1 `
    --gradient_checkpointing True `
    --half_precision_backend auto `
    --dataloader_num_workers 4 `
    --report_to tensorboard
