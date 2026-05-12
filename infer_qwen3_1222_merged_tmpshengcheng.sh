VLLM_USE_V1=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
IMAGE_MAX_TOKEN_NUM=1024 \
swift infer \
    --model /host/ssddata/disk2/zhanglu/swift/4cls/output/train_1219_longprompt_epochs20/v0-20251231-183707/model-2560 \
    --infer_backend vllm \
    --tensor_parallel_size 8 \
    --gpu_memory_utilization 0.85 \
    --val_dataset  /host/ssddata/disk3/zhanglu/dataset/changwei/20260113_生成/生成_results_v3_0123_p0_测试集.jsonl \
    --result_path /host/ssddata/disk3/zhanglu/dataset/changwei/20260113_生成/生成_results_v3_0123_p0_测试集_finetone_md2560.jsonl \
    --max_new_tokens 2048 \
    --load_data_args false
