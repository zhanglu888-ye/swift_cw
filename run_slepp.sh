#!/bin/bash

# 获取当前目录的绝对路径，确保at命令能找到文件
CURRENT_DIR=$(pwd)

# 提交任务：6小时后执行
# 逻辑：先cd到当前目录 -> 运行merge -> 成功后运行infer
echo "cd $CURRENT_DIR && ./merge.sh && ./infer_qwen3_1222_merged.sh" | at now + 7 hours

echo "已设定定时任务：6小时后将依次运行 merge 和 infer 脚本。"
