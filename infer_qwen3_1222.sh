PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
swift infer \
    --adapters /host/ssddata/disk2/zhanglu/swift/4cls/output/train_1219_longprompt/v0-20251222-142122/checkpoint-100 \
    --stream false \
    --max_new_tokens 2048 \
    --val_dataset /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/test_jintao/test_single_longprompt.jsonl \
    --result_path result/train_1219_longprompt/result_1222_jintao_ck100.json \
    --load_data_args false
