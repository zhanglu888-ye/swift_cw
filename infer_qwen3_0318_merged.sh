VLLM_USE_V1=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
IMAGE_MAX_TOKEN_NUM=1024 \
swift infer \
    --model /host/ssddata/disk2/zhanglu/swift/4cls/output/megatron_output/Qwen3-VL-4B-SAPO/v33-20260409-190528-gsponum4/checkpoint-2000-merged \
    --infer_backend vllm \
    --tensor_parallel_size 8 \
    --gpu_memory_utilization 0.85 \
    --val_dataset  /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/测试数据集_3cls/算法_3cls_清洗后.jsonl \
    --result_path result/train_260306cls3v1_epochs20/v33-20260409-190528-gsponum4/result_0410_3cls_suanfa_sftgspo_ck2000.json \
    --max_new_tokens 2048 \
    --load_data_args false
