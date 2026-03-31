# Blueprint 使用说明

本文档面向需要通过 Magnus 平台提交 SFT 训练任务的用户，说明提交前需要准备什么、各参数的含义，以及常见场景的参数组合建议。

## 提交前的准备工作

Blueprint 只是任务提交模板，不会替你生成数据、补全路径或检测环境。提交前请先确认以下内容已就绪。

### 1. 替换仓库常量

蓝图文件 `blueprints/sft_blueprint_template.py` 的顶部有五个常量，首次接入时必须替换为真实值：

```python
REPO_NAME = "your-repo-name"
NAMESPACE  = "your-github-org"
WORKDIR    = "/magnus/workspace/repository"
CONDA_SH   = "/opt/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV  = "magnus_shared"
```

如果不替换，蓝图能通过 Magnus 的语法检查，但任务提交后会找不到仓库或工作目录。

### 2. 基础模型

需要一个可被 `transformers.AutoModelForCausalLM` 加载的因果语言模型：

- HuggingFace 模型 ID，例如 `Qwen/Qwen2.5-0.5B-Instruct`
- 集群本地路径，例如 `/models/Qwen2.5-7B-Instruct`

要求：训练容器内可以访问该模型，tokenizer 也能正常加载。

### 3. 训练数据

提前将数据文件放到集群存储的可访问位置，格式为 JSONL `messages`：

```json
{"messages":[
  {"role":"system","content":"你是一个有帮助的助手。"},
  {"role":"user","content":"请简单解释一下 LoRA。"},
  {"role":"assistant","content":"LoRA 通过训练少量额外参数来实现高效微调。"}
]}
```

格式要求：
- 每行一条合法 JSON
- `messages` 非空列表
- 每条 message 包含字符串 `role` 和 `content`
- 最后一条消息必须来自 `assistant`

### 4. 验证数据（可选）

如需训练过程中做周期性评估，额外准备格式相同的验证集 JSONL。

提供 `val_path` 后，训练会：
- 每隔 `eval_steps` 步评估一次
- 训练结束时自动加载 `eval_loss` 最低的 checkpoint（而非最后一个）

不提供时，评估步骤跳过，仍然正常训练。

### 5. 输出目录

准备一个集群上可写的根目录。Blueprint 会自动将最终输出组织为：

```text
{output_root}/{experiment_name}/
├── adapter_config.json        # LoRA 配置
├── adapter_model.safetensors  # LoRA 权重
├── config.json
├── tokenizer.json
├── checkpoint-{step}/         # 中间 checkpoint
├── resolved_config.json       # 完整参数快照
├── run_summary.json           # 实验摘要
├── train_metrics.json
└── eval_metrics.json
```

### 6. 算力资源

提交前决定好：GPU 数量、GPU 型号、任务优先级。这些是 Magnus 调度层参数，不属于训练代码本身。

---

## Blueprint 参数说明

### `runner`

提交这个任务的集群用户名。用于权限归属和任务审计追踪。

---

### `experiment_name`

本次实验的唯一名称，建议短且稳定。

- 作为 Magnus 任务展示名称的一部分
- 直接拼接到 `output_root` 后构成输出目录
- 方便后续日志、指标和产物追踪

好的命名示例：`qwen25-7b-sft-v1`、`customer-support-lora-001`

---

### `train_path`

训练集 JSONL 文件的绝对路径。这是训练脚本真正读取的主训练数据。

`test_mode` 启用时，此字段在运行时不会被使用，但保留它可以维持 blueprint 表单契约的稳定性。

---

### `val_path`

验证集 JSONL 文件的绝对路径，可选。

提供后会启用训练中评估和训练结束时的 best model 加载，建议在正式训练中提供。

---

### `output_root`

训练输出的根目录。Blueprint 会将其与 `experiment_name` 拼接成最终输出目录，比直接填完整路径更稳定，适合作为团队共享的规范化输出组织方式。

---

### `base_model`

本次微调的基础模型，支持 HuggingFace 模型 ID 或集群本地路径。

它决定了：
- 加载哪套模型权重和 tokenizer
- 是否使用模型自带的 chat template（无 template 时使用项目内置的 fallback 格式）
- 在 `test_mode` 下，会作为 `--test_model_name_or_path` 传入训练脚本

---

### `finetune_method`

训练方式选择：

- `lora`：参数高效微调，只训练少量额外参数，显存占用低，**推荐默认选项**
- `full`：全参数训练，适合有足够算力的场景

---

### `max_seq_len`

训练时使用的最大序列长度（token 数）。超出此长度的样本会被截断。

- 值越大，显存占用越高，训练速度越慢
- 根据你的数据分布和 GPU 资源合理选择，不必默认填最大值

---

### `epochs`

训练集被完整遍历的轮数。

- 数据量较少时可适当增加 epoch，但要注意过拟合风险
- 通常 3～5 epoch 是一个合理起点

---

### `learning_rate`

优化器学习率，对训练结果影响最大的超参数之一。

- 太高可能导致训练不稳定或发散
- 太低可能几乎学不到有效更新
- LoRA 微调的常用范围：`1e-5` ～ `5e-4`；全参数微调通常更低

---

### `per_device_batch_size`

每张 GPU 上的 batch size。

与 `gradient_accumulation_steps` 和 `gpu_count` 共同决定有效 batch size：

```
有效 batch size ≈ per_device_batch_size × gradient_accumulation_steps × gpu_count
```

显存不足时，优先减小此值、增大 `gradient_accumulation_steps`。

---

### `gradient_accumulation_steps`

梯度累积步数。在显存有限时模拟更大 batch size 的方法：每累积 N 步才做一次参数更新。

---

### `gradient_checkpointing`

是否启用梯度检查点。

- 以增加约 20~30% 计算时间为代价，显著降低显存峰值（通常可降低 30~50%）
- **7B 及以上模型强烈建议开启**
- LoRA 和全参数训练均支持

---

### `save_total_limit`

磁盘上最多保留的 checkpoint 数量，默认 3。

训练过程中每隔 `save_steps` 步保存一个 checkpoint，超出限制时自动删除最旧的。建议不要设太大，防止存储爆满。

---

### `gpu_count`

Magnus 为此任务分配的 GPU 数量。同时也是 `accelerate launch --num_processes` 的值。

---

### `gpu_type`

申请的 GPU 型号，使用集群实际支持的字符串，例如 `rtx4090`、`rtx5090`、`a100`。

---

### `priority`

Magnus 调度优先级：

- `A1`：最高优先级
- `A2`：高优先级（一般首选）
- `B1`：可抢占
- `B2`：最低优先级

除非任务确实紧急，不要默认使用 `A1`。

---

### `test_mode`

是否启用内置 smoke 测试流程。启用后：

- 自动从 HuggingFace 下载极小测试数据集
- 使用内置 tiny 模型
- 走完完整训练流程

适合在以下情况使用：
- 新仓库首次接入 Magnus 时验证环境
- 验证依赖安装、网络访问、训练入口是否正常
- 不适合用于验证模型效果或性能

---

### `notes`

本次实验的可选备注，记录在任务描述中。

适合填写：为什么启动这个实验、预期目标、需要告知团队的背景说明。

---

### `extra_args`

附加到 `train_sft.py` 末尾的原始 CLI 参数，可选。这是给高级用户保留的逃生口。

- 参数会被原样拼接进 shell 命令，仅限可信用户使用
- 不建议日常大量依赖，过度使用会削弱 blueprint 模板的稳定性

示例：`--save_steps 100 --eval_steps 100`

---

## 场景参数建议

### 第一次跑通（验证环境）

```
test_mode = true
finetune_method = lora
gpu_count = 1
priority = B1
```

### 小模型 LoRA 试验（0.5B～3B）

```
finetune_method = lora
epochs = 3
learning_rate = 2e-5
per_device_batch_size = 2
gradient_accumulation_steps = 8
max_seq_len = 2048
gradient_checkpointing = false
save_total_limit = 3
gpu_count = 1
```

### 大模型 LoRA 训练（7B～14B）

```
finetune_method = lora
epochs = 3
learning_rate = 2e-5
per_device_batch_size = 1
gradient_accumulation_steps = 16
max_seq_len = 2048
gradient_checkpointing = true    ← 务必开启
save_total_limit = 3
gpu_count = 2 ～ 4
```

### 全参数训练

```
finetune_method = full
gradient_checkpointing = true    ← 务必开启
per_device_batch_size = 1
gradient_accumulation_steps = 16
gpu_count = 4+
```

---

## 提交前检查清单

- [ ] `REPO_NAME`、`NAMESPACE`、`WORKDIR`、`CONDA_SH`、`CONDA_ENV` 已替换为真实值
- [ ] 基础模型 ID 或路径正确，训练容器内可访问
- [ ] `train_path` 文件存在且可读，格式符合 JSONL `messages` 规范
- [ ] `val_path` 如提供，格式正确
- [ ] `output_root` 目录可写
- [ ] 数据最后一条 message 来自 `assistant`
- [ ] 所选 GPU 型号在 Magnus 中可用
- [ ] `gpu_count` 与 `batch_size`、`gradient_accumulation_steps` 的组合符合显存预算
- [ ] 7B 以上模型已开启 `gradient_checkpointing`
- [ ] `experiment_name` 清晰，不与历史任务混淆
- [ ] 正式训练前已用 `test_mode` 验证过环境

---

## Blueprint 与训练脚本的关系

Blueprint 只做三件事：

1. 给用户暴露一个清晰的参数表单
2. 把表单参数转换成 shell 命令
3. 调用 Magnus 的 `submit_job()`

真正的训练逻辑在 `train_sft.py` 和 `src/sft/` 中。如果需要新增训练能力，应先修改训练代码，再判断该能力是否值得作为 blueprint 表单字段暴露出来。
