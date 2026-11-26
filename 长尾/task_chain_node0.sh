#!/bin/bash
echo "Starting Task 0 at $(date)"
bash train_qwen3_4b_sft_grpo_260409megatron_ddp.sh

echo "Task 1 finished. Starting Task 2 at $(date)"
# 建议在两个任务之间休息 30 秒，确保端口完全释放，缓存清理干净
sleep 30
bash train_qwen3_4b_sft_dpo_260413megatron_ddp.sh

echo "All tasks finished at $(date)"
