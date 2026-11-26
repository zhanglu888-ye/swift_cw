VLLM_USE_V1=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
IMAGE_MAX_TOKEN_NUM=1024 \
swift infer \
    --model /host/ssddata/disk2/zhanglu/swift/model/Qwen/Qwen3-VL-4B-Instruct \
    --infer_backend vllm \
    --tensor_parallel_size 8 \
    --gpu_memory_utilization 0.85 \
    --val_dataset  /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/测试数据集_3cls/算法_3cls_清洗后.jsonl \
    --result_path result/result_0309_3cls_suanfa_ck1800.json \
    --max_new_tokens 2048 \
    --load_data_args false
