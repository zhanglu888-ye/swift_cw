VLLM_USE_V1=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
IMAGE_MAX_TOKEN_NUM=1024 \
swift infer \
    --model /host/ssddata/disk2/zhanglu/swift/4cls/output/train_260210p0v2p1v3_epochs20/v0-20260210-181859/model-9900 \
    --infer_backend vllm \
    --tensor_parallel_size 8 \
    --gpu_memory_utilization 0.85 \
    --val_dataset  /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/待标注_longprompt.jsonl \
    --result_path result/train_260210p0v2p1v3_epochs20/result_0227_biaozhu_ck9900.json \
    --max_new_tokens 2048 \
    --load_data_args false
