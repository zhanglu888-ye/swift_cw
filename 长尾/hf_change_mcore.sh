# #!/bin/bash

# # 指定 8 张显卡用于转换
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# swift export \
#     --model "/host/ssddata/disk2/zhanglu/swift/model/Qwen/Qwen3-VL-4B-Instruct" \
#     --to_mcore true \
#     --torch_dtype bfloat16 \
#     --output_dir "/host/ssddata/disk2/zhanglu/swift/model/Qwen/Qwen3-VL-4B-mcore"


#!/bin/bash

# 指定 8 张显卡用于转换
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift export \
    --model "/host/ssddata/disk2/zhanglu/swift/4cls/output/train_260306cls3v1_epochs20/v0-20260306-165019/model-1800" \
    --to_mcore true \
    --model_type qwen3_vl \
    --torch_dtype bfloat16 \
    --output_dir "/host/ssddata/disk2/zhanglu/swift/model/Qwen/Qwen3-VL-4B-mcore-sft20260306-165019-model1800"