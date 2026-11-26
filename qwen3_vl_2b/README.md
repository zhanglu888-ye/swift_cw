# Qwen3-VL-2B 人体小图危险物品微调

延续 `长尾/` 的 ms-swift、LoRA 和自定义 GRPO 奖励方法，提供三条独立实验：

| 实验 | 初始权重 | 训练入口 |
| --- | --- | --- |
| SFT | Qwen/Qwen3-VL-2B-Instruct | `02_sft.sh` |
| RL（GRPO） | 同一原始 Instruct 模型 | `03_rl.sh` |
| SFT+RL | SFT LoRA 合并后的完整模型 | `04_sft_rl.sh` |

这里的“纯 RL”指不做本任务 SFT，基础模型本身已经是 Instruct。默认冻结视觉编码器和连接层，训练语言侧 LoRA；没有进行本地 GPU 训练。推荐先在支持 BF16 的 NVIDIA GPU（例如 24GB 及以上）上单卡试跑，GRPO 显存取决于图片、序列和生成组大小，不能保证固定显存可运行。默认不用 vLLM、FlashAttention 或 DeepSpeed，多卡用 DDP，每张卡需要容纳模型副本。

## 1. 同步到服务器

同步本仓库，以及**原始人体小图图片目录**。标注目录中的 JSON 只有图片路径，不包含图片。服务器示例：

```text
/srv/swift_cw/                         # 仓库
  qwen3_vl_2b/
  人体小图危险物品qwen3-27B标注/
/srv/datasets/人体小图/
  positive/*.jpg
  negative/*.jpg
```

训练使用已有 `train.json`（1352 条）和 `val.json`（425 条），不会自行把 `review.json`、`failures.json` 或全部 `annotations.json` 混入训练。标注是教师模型输出，需关注误标和极少的“不确定”样本。保留现有划分便于比较；转换时检查图片内容哈希，发现同图重复或跨划分重复直接报错。相同视频/人物的不同裁剪可能仍然相关，正式评测建议另建按来源分组且人工复核的独立测试集，验证集只用于调参。

## 2. 使用服务器现有镜像

直接进入服务器已经配置好的镜像环境，不构建镜像、不安装或升级依赖。将仓库和图片目录挂载到现有容器即可。以下以容器内仓库路径 `/workspace`、图片路径 `/datasets/人体小图` 为例，按实际挂载路径修改：

```bash
cd /workspace
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export IMAGE_ROOT=/datasets/人体小图
```

以下从下载模型开始执行。脚本沿用现有镜像中的 ms-swift 和 Transformers；奖励插件兼容 `swift.rewards` 和 `swift.plugin` 两种注册入口。训练参数曾按 ms-swift 3.12.0 的公开定义核对，尚未在你的服务器镜像中实测。

## 3. 下载模型、转换数据

```bash
bash qwen3_vl_2b/scripts/00_download.sh
# 或 Hugging Face；可加 --revision <commit-id> 固定模型快照
# bash qwen3_vl_2b/scripts/00_download.sh --provider huggingface
bash qwen3_vl_2b/scripts/01_prepare.sh
```

默认模型保存到 `/workspace/models/Qwen3-VL-2B-Instruct`。可提前 `export MODEL_DIR=/其他目录`。默认标注目录是仓库内的中文目录，可用 `ANNOTATION_DIR` 覆盖。图片从原来的 Mac 路径映射到 `$IMAGE_ROOT/positive或negative/文件名`，缺失则立即失败。转换生成：

- `data/sft_{train,val}.jsonl`：user 图文提问 + assistant 标注 JSON。
- `data/grpo_{train,val}.jsonl`：仅 user 提问 + 独立 `solution`，答案不进入生成提示。
- `data/stats.json`：划分样本数及每类“有/无/不确定”计数。

`--skip-image-check` 仅供无图片环境下检查标注结构；正式训练必须不带此参数重新转换，以验证图片存在及内容重复。

## 4. 分别运行三种实验

先小步检查 SFT 和 RL，建议用临时输出根目录，避免和正式实验混淆：

```bash
OUTPUT_ROOT=/workspace/qwen3_vl_2b/outputs/smoke \
  bash qwen3_vl_2b/scripts/02_sft.sh --max_steps 2
OUTPUT_ROOT=/workspace/qwen3_vl_2b/outputs/smoke \
  bash qwen3_vl_2b/scripts/03_rl.sh --max_steps 2
```

正式 SFT（默认 3 epochs、学习率 1e-4）：

```bash
bash qwen3_vl_2b/scripts/02_sft.sh
```

纯 RL（默认 1 epoch、学习率 5e-6），保持 `MODEL_DIR` 为原始模型：

```bash
bash qwen3_vl_2b/scripts/03_rl.sh
```

SFT+RL：从 SFT 日志选择实际 `checkpoint-*` 目录，显式传入，避免自动选错历史实验。脚本先合并 SFT LoRA，再在合并模型上创建新的 RL LoRA：

```bash
SFT_CHECKPOINT=/workspace/qwen3_vl_2b/outputs/sft/v0-实际时间/checkpoint-实际步数 \
  bash qwen3_vl_2b/scripts/04_sft_rl.sh
```

合并目录默认 `outputs/sft_merged`，已存在会报错，请设置新的 `MERGED_DIR`，防止误用旧权重。三个训练输出分别在 `outputs/sft`、`outputs/rl`、`outputs/sft_rl`；swift 通常会额外创建版本子目录，以日志为准。更换实验可设置新的 `OUTPUT_ROOT`。

常用覆盖变量：`SFT_EPOCHS`、`SFT_LR`、`RL_EPOCHS`、`RL_LR`、`MAX_PIXELS`（默认 262144）、`GRAD_ACCUM`（默认 8）。图片较小，增大像素上限不等于恢复细节。其余 swift 参数可追加在脚本末尾，例如 `--resume_from_checkpoint /实际/checkpoint`。SFT+RL 中断后可直接设置 `MODEL_DIR` 为合并 SFT 权重，调用 `03_rl.sh` 恢复，不必重新合并。

多卡示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
  bash qwen3_vl_2b/scripts/03_rl.sh
```

GRPO 每个 prompt 生成 4 个答案，generation batch 显式设为 `全局卡数 × 4`，支持默认单卡 batch=1；验证每条生成 1 个答案。不要单独覆盖 batch size 或生成数量而不重算 generation batch 的整除关系。多机需各节点同时启动并设置一致的 `NNODES`、`MASTER_ADDR`、`MASTER_PORT`，各节点分别指定 `NODE_RANK`，模型、数据和输出目录需同路径可访问。先完成单机验证再扩展。

## 5. 奖励设计

`rewards.py` 注册两个 ms-swift ORM：

- `danger_format`（权重 0.2）：完整且严格的七字段 JSON 得 1，否则 0；描述非空，六类值限定为“有/无/不确定”。不接受 Markdown 包裹或额外字段。
- `danger_accuracy`（权重 1.0）：分别计算 GT 非“无”字段和“无”字段的精确匹配率，再对非空组平均并映射到 [-1,1]。错误 JSON 得 -1；“不确定”必须精确匹配。正样本全填“无”不会因为其余五类正确而拿到高准确奖励。

不奖励描述长度、推理文本或教师描述的逐字相似度。描述内容主要依赖 SFT，RL 的可验证奖励针对六类判断；不代表描述事实准确性已被自动验证。奖励仍可能受负样本占比和教师误标影响，应检查每类召回而非只看平均 reward。GRPO 相同 prompt 的所有候选得分一致时没有有效组内优势，需观察生成多样性和 reward 方差。

## 6. 导出和公平评测

LoRA 训练结果需要合并成完整模型再供本项目评测脚本使用。SFT 和纯 RL 的 `MODEL_DIR` 为原始模型：

```bash
CHECKPOINT=/实际/sft或rl/checkpoint-xxx \
EXPORT_DIR=/workspace/qwen3_vl_2b/outputs/export_sft \
  bash qwen3_vl_2b/scripts/05_export.sh
```

SFT+RL 的 RL adapter 基座是合并后的 SFT 模型，导出时必须指定它：

```bash
MODEL_DIR=/workspace/qwen3_vl_2b/outputs/sft_merged \
CHECKPOINT=/实际/sft_rl/checkpoint-xxx \
EXPORT_DIR=/workspace/qwen3_vl_2b/outputs/export_sft_rl \
  bash qwen3_vl_2b/scripts/05_export.sh
```

对原始模型及三种完整训练模型分别运行，用不同输出文件：

```bash
MODEL_DIR=/workspace/models/Qwen3-VL-2B-Instruct \
EVAL_OUTPUT=/workspace/qwen3_vl_2b/outputs/eval/base.jsonl \
  bash qwen3_vl_2b/scripts/06_evaluate.sh
MODEL_DIR=/workspace/qwen3_vl_2b/outputs/export_sft \
EVAL_OUTPUT=/workspace/qwen3_vl_2b/outputs/eval/sft.jsonl \
  bash qwen3_vl_2b/scripts/06_evaluate.sh
```

评测默认单进程、逐张图、贪心生成，输出 prediction JSONL 和同名 `.metrics.json`：严格 JSON 合法率、六类同时正确率、各类三种状态的 precision/recall/F1、混淆矩阵，以及六类“有”的 macro F1。错误格式计为各类错误。不要用高占比“无”的总体准确率代替目标类别召回。评测前确认同一图片上限、提示词和生成长度，比较 base/SFT/RL/SFT+RL；另留独立测试集作最终报告。

## 本地验证

不安装训练依赖也可以执行：

```bash
python3 -m unittest discover -s qwen3_vl_2b/tests -v
```

覆盖奖励打分与非法格式、答案对齐、路径映射、GRPO 不泄漏答案、图片缺失及哈希重复拦截、非法输出计入评测错误。已对当前全部 1777 条 train/val 样本完成无图片模式的转换检查，并通过 Python 编译及 Bash 语法检查。原始图片不在当前工程内，图片内容检查及完整训练未在本机执行。
