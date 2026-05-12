# 1. 设置可见的GPU
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 2. 运行脚本 (请将 /path/to/swift/ 替换为你的实际路径)
python /usr/local/lib/python3.10/site-packages/swift/examples/pytorch/llm/llm_eval.py \
    --model output/4cls_0923_7b_full_unnorm/v0-20250923-180828/checkpoint-945/ \
    --dataset /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类_split_dataset/test/test_data_caption_simplified_unnormalized_0923.json \
    --max_new_tokens 8192 \
    --batch_size 8
