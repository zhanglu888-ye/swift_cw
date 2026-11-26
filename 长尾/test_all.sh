#!/bin/bash

# 设置-e选项，任何命令失败（返回非0退出码）则立即退出脚本
set -e

echo "--- $(date): Starting script train_4cls_3b_lora_unnorm.sh ---"
#./infer.sh
echo "--- $(date): 1.sh finished successfully. ---"

echo "--- $(date): Starting script 2.sh ---"
./infer_1.sh
echo "--- $(date): 2.sh finished successfully. ---"
#./infer_2.sh
echo "--- $(date): Starting script 3.sh ---"
./infer_3.sh
echo "--- $(date): 3.sh finished successfully. ---"
./infer_4.sh
./infer_md.sh
