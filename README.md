# SFT Blueprint Scaffold

这个仓库提供了一套最小化的 Supervised Fine-Tuning（SFT）训练管线，专为挂载到 Magnus 平台作为任务蓝图而设计。

## 项目结构

```text
.
├── train_sft.py              # 训练入口
├── blueprints/
│   └── sft_blueprint_template.py   # Magnus 蓝图模板
├── configs/sft/
│   └── base.yaml             # 默认超参配置
├── data/qwen_small/          # 示例数据（小规模中文对话）
├── docs/
│   ├── blueprint_usage_guide.md
│   └── blueprint_parameter_design.md
├── src/sft/                  # 训练核心模块
│   ├── args.py
│   ├── cli.py
│   ├── data.py
│   ├── eval.py
│   ├── formatting.py
│   ├── model.py
│   ├── trainer.py
│   └── utils.py
├── tests/
├── pyproject.toml
└── requirements.txt
```

## 快速开始

**安装依赖：**

```bash
pip install -r requirements.txt
```

或使用 uv：

```bash
uv sync
```

**查看完整参数列表：**

```bash
python train_sft.py --help
```

**仅验证配置和数据，不启动训练（dry run）：**

```bash
python train_sft.py \
  --experiment_name demo-run \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --train_path data/qwen_small/train_messages.jsonl \
  --val_path data/qwen_small/val_messages.jsonl \
  --output_dir /tmp/sft-demo \
  --use_lora \
  --dry_run
```

**最小 LoRA 训练（单卡）：**

```bash
python train_sft.py \
  --experiment_name qwen05b-lora-v1 \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --train_path data/qwen_small/train_messages.jsonl \
  --val_path data/qwen_small/val_messages.jsonl \
  --output_dir /tmp/sft-output \
  --use_lora \
  --bf16
```

**大模型训练（开启 gradient checkpointing 节省显存）：**

```bash
python train_sft.py \
  --experiment_name qwen7b-lora-v1 \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --train_path /data/train_messages.jsonl \
  --val_path /data/val_messages.jsonl \
  --output_dir /data/outputs/qwen7b-lora-v1 \
  --use_lora \
  --bf16 \
  --gradient_checkpointing \
  --max_seq_length 2048
```

**使用 YAML 配置文件（CLI 参数优先级更高）：**

```bash
python train_sft.py \
  --config configs/sft/base.yaml \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --train_path data/qwen_small/train_messages.jsonl \
  --output_dir /tmp/sft-output
```

**内置 smoke 测试（自动下载极小测试数据）：**

```bash
python train_sft.py \
  --test_mode \
  --output_dir /tmp/sft-test \
  --use_lora
```

`test_mode` 会自动从 Hugging Face 下载测试数据、使用内置 tiny 模型并走完完整训练流程，适合验证环境依赖和训练入口是否正常。

## 数据格式

训练数据必须是 JSONL 格式，每行一个样本，包含 `messages` 字段：

```json
{"messages":[
  {"role":"system","content":"你是一个有帮助的助手。"},
  {"role":"user","content":"请简单解释一下 LoRA。"},
  {"role":"assistant","content":"LoRA 通过训练少量额外参数来实现高效微调。"}
]}
```

**格式要求：**

- 每行必须是合法 JSON
- `messages` 必须是非空列表
- 每条 message 必须包含字符串类型的 `role` 和 `content`
- 支持的 role：`system`、`user`、`assistant`、`tool`
- 最后一条消息必须来自 `assistant`

## 完整 CLI 参数说明

### 基本参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--output_dir` | Path | **必填** | 训练产物保存目录 |
| `--experiment_name` | str | None | 实验名称，用于日志和摘要追踪 |
| `--model_name_or_path` | str | None | HuggingFace 模型 ID 或本地路径 |
| `--train_path` | Path | None | 训练集 JSONL 文件路径 |
| `--val_path` | Path | None | 验证集 JSONL 文件路径（可选） |
| `--config` | Path | None | YAML 配置文件路径（可选） |
| `--dry_run` | flag | False | 仅验证配置和数据，不启动训练 |

### 训练超参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_train_epochs` | float | 3.0 | 训练轮数 |
| `--learning_rate` | float | 2e-5 | 优化器学习率 |
| `--per_device_train_batch_size` | int | 1 | 每张 GPU 的训练 batch size |
| `--per_device_eval_batch_size` | int | 1 | 每张 GPU 的评估 batch size |
| `--gradient_accumulation_steps` | int | 16 | 梯度累积步数 |
| `--max_seq_length` | int | 4096 | 最大序列长度 |
| `--logging_steps` | int | 10 | 日志打印间隔步数 |
| `--save_steps` | int | 200 | checkpoint 保存间隔步数 |
| `--eval_steps` | int | 200 | 验证评估间隔步数（需提供 val_path）|
| `--save_total_limit` | int | 3 | 最多保留的 checkpoint 数量 |

### 精度与优化

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--bf16` | flag | False | 使用 bfloat16 混合精度（推荐，需硬件支持）|
| `--fp16` | flag | False | 使用 float16 混合精度 |
| `--gradient_checkpointing` | flag | False | 开启梯度检查点，显著降低显存占用，适合大模型 |
| `--seed` | int | 42 | 随机种子 |

> `--bf16` 和 `--fp16` 不能同时使用。

### LoRA 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_lora` | flag | False | 启用 LoRA 微调 |
| `--lora_r` | int | 8 | LoRA 秩（rank） |
| `--lora_alpha` | int | 16 | LoRA 缩放系数 |
| `--lora_dropout` | float | 0.05 | LoRA dropout |
| `--lora_target_modules` | str | None | 逗号分隔的目标模块，例如 `q_proj,v_proj`；不填则由 PEFT 自动推断 |

### 其他参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset_format` | str | messages | 数据集格式（当前仅支持 `messages`）|
| `--report_to` | str | none | 日志上报后端，例如 `wandb`、`tensorboard` |
| `--trust_remote_code` | flag | False | 允许加载 HuggingFace 模型的自定义代码 |
| `--test_mode` | flag | False | 使用内置 smoke 测试数据集自动运行 |

## 配置文件

可以通过 `--config` 指定 YAML 文件提供默认参数，CLI 参数在非默认时会覆盖 YAML 设置：

```yaml
# configs/sft/base.yaml
dataset_format: messages
max_seq_length: 4096
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 2e-5
num_train_epochs: 3
use_lora: true
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
gradient_checkpointing: false
save_total_limit: 3
seed: 42
```

## 训练产物

训练完成后，`output_dir` 下会包含以下文件：

```text
{output_dir}/
├── adapter_config.json       # LoRA 配置（LoRA 模式）
├── adapter_model.safetensors # LoRA 权重（LoRA 模式）
├── config.json               # 模型配置
├── tokenizer.json            # Tokenizer 词表
├── special_tokens_map.json
├── checkpoint-{step}/        # 中间 checkpoint（最多保留 save_total_limit 个）
├── resolved_config.json      # 本次运行的完整参数快照
├── run_summary.json          # 实验摘要（样本数、模型、格式等）
├── train_metrics.json        # 训练指标
└── eval_metrics.json         # 评估指标（提供 val_path 时）
```

> 提供 `val_path` 时，训练会自动在结束时加载验证集上 `eval_loss` 最低的 checkpoint。

## 多卡训练

通过 `accelerate launch` 支持多卡数据并行：

```bash
uv run accelerate launch --num_processes 4 train_sft.py \
  --experiment_name qwen7b-lora-4gpu \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --train_path /data/train_messages.jsonl \
  --val_path /data/val_messages.jsonl \
  --output_dir /data/outputs/qwen7b-lora-4gpu \
  --use_lora \
  --bf16 \
  --gradient_checkpointing
```

## Magnus 蓝图集成

**相关文件：**

- [blueprints/sft_blueprint_template.py](blueprints/sft_blueprint_template.py)：蓝图模板
- [docs/blueprint_usage_guide.md](docs/blueprint_usage_guide.md)：用户使用说明
- [docs/blueprint_parameter_design.md](docs/blueprint_parameter_design.md)：蓝图参数设计决策

**使用前替换以下模板常量：**

```python
REPO_NAME = "your-repo-name"
NAMESPACE  = "your-github-org"
WORKDIR    = "/magnus/workspace/repository"
CONDA_SH   = "/opt/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV  = "magnus_shared"
```

**Magnus 标准启动命令：**

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate magnus_shared
cd /magnus/workspace/repository
uv sync --quiet
uv run accelerate launch --num_processes {gpu_count} train_sft.py \
  --experiment_name {experiment_name} \
  --model_name_or_path {base_model} \
  --train_path {train_path} \
  --val_path {val_path} \
  --output_dir {output_dir} \
  --use_lora \
  --bf16 \
  --gradient_checkpointing \
  --save_total_limit 3
```

## 运行测试

```bash
pytest tests/
```
