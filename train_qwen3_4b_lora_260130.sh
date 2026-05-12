# 2 * 21GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model /host/ssddata/disk2/zhanglu/swift/model/Qwen/Qwen3-VL-4B-Instruct \
    --dataset /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/train_data/p1/p1_train_zhengyangben_v3.jsonl \
	 /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/train_data/p1/p1_train_fuyangben_v3.jsonl \
	 /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/train_data/p0/train_zhengyangben_v2.jsonl \
         /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/train_data/p0/train_fuyangben_v2.jsonl \
    --load_from_cache_file true \
    --val_dataset  /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/train_data/p1/p1_val_zhengyangben_v3.jsonl \
    	/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/train_data/p1/p1_val_fuyangben_v3.jsonl \
	 /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/train_data/p0/val_zhengyangben_v2.jsonl \
        /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/train_data/p0/val_fuyangben_v2.jsonl \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --attn_impl flash_attn \
    --padding_free true \
    --learning_rate 3e-5 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --packing true \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --gradient_accumulation_steps 8 \
    --eval_steps 150 \
    --save_steps 150 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output/train_260210p0v2p1v3_epochs20 \
    --warmup_ratio 0.05 \
    --deepspeed zero2 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4
