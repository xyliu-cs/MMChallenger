# steps in the progress bar = training samples * num_train_epochs / (per_device_train_batch_size * gradient_accumulation_steps)
HF_ENDPOINT=https://hf-mirror.com python train_llava.py \
    --model_name_or_path /120040051/MLLM_Repos/llava-1.5-13b-hf \
    --output_dir "/120040051/MLLM_Repos/llava-1.5-13b-sft" \
    --max_seq_length 512 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --logging_steps 10 \
    --num_train_epochs 3 \
    --dataset_name xiaoyuanliu/conflict_vis \
    --gradient_checkpointing \
    --bf16 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_target_modules=all-linear \
    --dataloader_num_workers 4 \