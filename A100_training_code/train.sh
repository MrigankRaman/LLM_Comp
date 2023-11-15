/home/mrigankr/Apps/anaconda3/envs/neurips/bin/python train_old.py \
    --model_name_or_path Qwen/Qwen-14B \
    --data_path filtered_data_qwen_2000.jsonl \
    --bf16 True \
    --output_dir qwen_ours_3e-5_new \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --eval_steps 15000 \
    --save_strategy "steps" \
    --save_steps 260 \
    --save_total_limit 30 \
    --learning_rate 3e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --adam_beta2 0.95 \
    --model_max_length 1600 \
    --gradient_checkpointing True \
    --report_to none \
    --max_grad_norm 0.3 \
    --lora_dropout 0.05 \
    --lora_r 16 \
    --lora_alpha 32 \
    --bits 16 \
    --double_quant \
    --use_lora True \
    --ddp_find_unused_parameters False \
    --optim paged_adamw_32bit \
    # --deepspeed /home/mrigankr/llm_comp/LLM_Comp/src/configs/deepspeed_config.json \
    # --optim "adamw_torch" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    # --fsdp "full_shard auto_wrap" \
    # --deepspeed /home/mrigankraman/llm-distillation/Llama-X/src/configs/ds_zero2.json \
    # --optim adamw_bnb_8bit \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    # --fsdp "full_shard auto_wrap" \
    # --deepspeed /home/mrigankraman/llm-distillation/Llama-X/src/configs/deepspeed_config.json \
