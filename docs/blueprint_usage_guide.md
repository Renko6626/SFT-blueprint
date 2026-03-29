# Blueprint 使用说明

本文档面向需要调用本项目 SFT blueprint 的用户，说明在提交任务前需要准备哪些内容，以及 blueprint 暴露出来的各个参数分别代表什么。

## 调用 Blueprint 前需要准备什么

blueprint 本质上只是任务提交模板，不会替你生成数据、补全缺失路径，也不会在运行时帮你猜测模型或配置。因此，在调用之前，请先确认下面这些内容已经准备好，并且 Magnus 运行环境能够访问。

另外，模板维护者还需要先把 blueprint 文件中的几个仓库常量替换为真实值：

- `REPO_NAME`
- `NAMESPACE`
- `WORKDIR`

如果这些值不替换，blueprint 虽然能通过 Magnus 的语法与类型检查，但任务提交后无法正确定位到你的仓库和工作目录。

### 1. 基础模型

你需要准备一个可被 `transformers` 加载的因果语言模型。

常见形式：

- Hugging Face 模型 ID，例如 `Qwen/Qwen2.5-0.5B-Instruct`
- 集群本地模型路径，例如 `/models/Qwen2.5-7B-Instruct`

要求：

- 训练容器内可以访问这个模型
- 模型兼容 `AutoModelForCausalLM`
- 对应 tokenizer 也能被正常加载

### 2. 训练数据

你需要提前把训练数据放到集群可访问的存储位置。

当前要求的数据格式：

- `JSONL`
- 每行一条样本
- 每条样本包含一个 `messages` 字段

示例：

```json
{"messages":[
  {"role":"system","content":"你是一个有帮助的助手。"},
  {"role":"user","content":"请简单解释一下 LoRA。"},
  {"role":"assistant","content":"LoRA 通过训练少量额外参数来实现高效微调。"}
]}
```

要求：

- 每一行必须是合法 JSON
- `messages` 必须是非空列表
- 每条 message 必须包含字符串类型的 `role` 和 `content`
- 最后一条消息必须来自 `assistant`

### 3. 可选的验证数据

如果你希望训练过程中做周期性评估，需要额外准备一份验证集 JSONL，格式与训练集一致。

如果不提供 `val_path`：

- 训练仍然可以正常执行
- 训练过程中的评估步骤会被跳过

### 4. 输出目录

你需要准备一个可写的输出根目录，供训练任务保存产物。

blueprint 通常会把最终输出目录组织成：

```text
{output_root}/{experiment_name}
```

这个目录通常会包含：

- LoRA adapter 或完整模型权重
- tokenizer 文件
- checkpoint
- `resolved_config.json`
- `run_summary.json`
- `train_metrics.json`
- `eval_metrics.json`

### 5. 算力资源选择

提交任务前，你需要决定：

- 用多少张 GPU
- 使用哪种 GPU 类型
- 任务优先级是什么

这些属于 Magnus 调度层参数，不属于训练逻辑本身。

### 6. 仓库可执行性

Magnus 运行环境必须能够拉取本仓库，并成功执行下面这类命令：

```bash
python -m pip install -r requirements.txt
accelerate launch ... train_sft.py ...
```

因此 blueprint 的维护者还需要保证以下内容稳定：

- 仓库名
- namespace / 组织名
- 容器内的工作目录
- 依赖安装步骤

## Blueprint 参数说明

### `runner`

表示这个 Magnus 任务以谁的身份运行。

它主要用于：

- 标识集群用户
- 做权限归属和审计追踪

### `experiment_name`

表示本次实验的名称，建议短、稳定、可读。

它主要用于：

- 在 Magnus 中标识这次任务
- 构造最终输出目录名
- 方便日志、指标和产物追踪

好的命名示例：

- `qwen25-05b-sft-v1`
- `customer-support-lora-001`

### `train_path`

训练集 JSONL 文件的绝对路径。

这是 `train_sft.py` 真正读取的主训练数据。

### `val_path`

验证集 JSONL 文件的绝对路径，可选。

适合在这些场景提供：

- 需要训练中定期评估
- 希望生成更有意义的 `eval_metrics.json`

### `output_root`

训练输出的根目录。

blueprint 会把它和 `experiment_name` 拼接成最终输出目录。相比每次手工填写完整输出路径，这种方式更适合作为稳定模板。

### `base_model`

本次微调使用的基础模型。

它决定了：

- 加载哪套模型权重
- 使用哪个 tokenizer
- 使用模型自带 chat template 还是项目中的 fallback 格式化逻辑

### `finetune_method`

blueprint 暴露出来的训练方式。

当前建议值：

- `lora`：参数高效微调，第一版推荐默认方案
- `full`：全参数训练，适合更高级的场景

对于当前项目，第一版默认建议使用 `lora`。

### `max_seq_len`

训练时使用的最大序列长度。

它会影响：

- 显存占用
- 截断行为
- 训练吞吐

值越大并不一定越好，通常也意味着更高的资源成本。

### `epochs`

训练集被完整遍历的轮数。

更高的 epoch 在小数据集上可能更容易拟合，但也会增加训练时长和过拟合风险。

### `learning_rate`

优化器学习率。

这是最敏感的训练参数之一：

- 太高可能训练不稳定
- 太低可能几乎学不到东西

### `per_device_batch_size`

每张 GPU 上的 batch size。

它会影响：

- 显存占用
- 训练吞吐
- 与梯度累积共同决定的有效 batch size

### `gradient_accumulation_steps`

梯度累积步数。

它的作用是：在显存有限时，用更小的单步 batch，累积多步后再做一次参数更新，从而提升有效 batch size。

近似可理解为：

```text
有效 batch size ≈ per_device_batch_size × gradient_accumulation_steps × gpu_count
```

### `gpu_count`

Magnus 为这次任务分配的 GPU 数量。

它同时影响：

- 调度资源申请
- blueprint 中 `accelerate launch --num_processes` 的值

### `gpu_type`

申请的 GPU 型号。

具体取值取决于你的 Magnus 集群，例如：

- `rtx4090`
- `rtx5090`
- `a100`

应使用集群实际支持的字符串。

### `priority`

Magnus 任务优先级。

常见含义一般是：

- `A1`：最高优先级
- `A2`：高优先级
- `B1`：可抢占
- `B2`：最低优先级

除非任务确实紧急，否则不要默认使用最高优先级。

### `notes`

本次实验的可选备注信息。

适合记录：

- 为什么要启动这个实验
- 这次实验的预期目标
- 需要给团队同步的背景说明

### `extra_args`

附加到 `train_sft.py` 末尾的原始 CLI 参数，可选。

它是给高级用户保留的逃生口，不建议日常大量依赖，否则会削弱 blueprint 模板本身的稳定性。由于它会被原样拼接进 shell 命令，因此只适合可信用户场景。

例如：

```text
--save_steps 100 --eval_steps 100
```

## 一次典型提交前的检查清单

在真正提交前，建议逐项确认：

- 基础模型 ID 或路径是否正确
- `train_path` 是否存在且可读
- `val_path` 如果提供，格式是否正确
- `output_root` 是否可写
- 数据是否符合 JSONL `messages` 格式
- 所选 GPU 类型是否在 Magnus 中可用
- `gpu_count`、batch size、梯度累积是否和模型规模匹配
- `experiment_name` 是否清晰且不会和历史任务混淆

## 推荐的起始默认值

如果你是在做第一次 LoRA 试跑，可以先从下面这组默认值开始：

- `finetune_method = lora`
- `epochs = 3`
- `learning_rate = 2e-5`
- `per_device_batch_size = 1`
- `gradient_accumulation_steps = 16`
- `max_seq_len = 4096`

先确认整条流水线能跑通，再去逐步调参。

## Blueprint 与训练脚本的关系

blueprint 只做三件事：

1. 给用户暴露一个清晰的表单
2. 把表单参数转换成 shell 命令
3. 调用 Magnus 的 `submit_job()`

真正的训练逻辑仍然在：

- `train_sft.py`
- `src/sft/`

如果你需要新增训练能力，应该先修改训练代码，再判断这个能力是否值得暴露成 blueprint 表单字段。
