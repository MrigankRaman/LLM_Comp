CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --data_path /home/ubuntu/neurips/LLM_Comp/data/final_data_with_lima.jsonl \
    --bf16 True \
    --output_dir /home/ubuntu/neurips/LLM_Comp/src/ours_platypus_13B_2048 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --eval_steps 15000 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 20 \
    --learning_rate 2e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --report_to none \
    --optim "paged_adamw_32bit" \
    --max_grad_norm 0.3 \
    --lora_dropout 0.05 \
    --lora_r 256 \
    --lora_alpha 16 \
    --bits 4 \
    --use_lora True \
    --ddp_find_unused_parameters False \
    --double_quant \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    # --fsdp "full_shard auto_wrap" \
    # --deepspeed /home/mrigankraman/llm-distillation/Llama-X/src/configs/ds_zero2.json \
    # --optim adamw_bnb_8bit \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    # --fsdp "full_shard auto_wrap" \
    # --deepspeed /home/mrigankraman/llm-distillation/Llama-X/src/configs/deepspeed_config.json \
