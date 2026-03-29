# Blueprint 参数设计说明

本文档用于说明：对于当前这套 SFT 工程，Magnus blueprint 层应该暴露哪些参数，哪些参数应当保留在训练实现内部，不建议直接暴露给使用者。

## 设计目标

blueprint 的职责是做一个可复用、可维护的任务提交模板，而不是再做一层训练系统。

因此 blueprint 层应当只暴露：

- 用户经常会修改的参数
- 对实验结果或资源配置影响明显的参数
- 能够长期稳定作为平台表单契约存在的参数

而不应当暴露过多训练实现细节，否则 blueprint 会很快失去模板化价值，变成难维护的大表单。

## 建议暴露的参数

### Basic

- `runner`
- `experiment_name`
- `notes`

为什么暴露：

- 这些字段用于标识任务归属和实验上下文
- 有助于在 Magnus 中做追踪、审计和实验回看

### Data

- `train_path`
- `val_path`
- `output_root`

为什么暴露：

- 这是启动任务所必须的数据和输出路径信息
- `output_root` 比直接暴露完整 `output_dir` 更稳定，因为 blueprint 可以统一按 `experiment_name` 组织输出目录

### Model

- `base_model`
- `finetune_method`
- `max_seq_len`

为什么暴露：

- 这些是模型侧最常调整的关键决策
- `finetune_method` 第一版只建议开放 `lora` 和 `full`，保持表单简单

### Optimization

- `epochs`
- `learning_rate`
- `per_device_batch_size`
- `gradient_accumulation_steps`

为什么暴露：

- 这些是最核心、最高频的优化超参数
- 直接映射到训练 CLI，语义清晰，便于长期维护

### Cluster

- `gpu_count`
- `gpu_type`
- `priority`

为什么暴露：

- 这些参数属于 Magnus 调度层，而不是训练代码内部
- 提交者必须能够明确控制资源申请

### Misc

- `test_mode`
- `extra_args`

为什么暴露：

- `test_mode` 适合 Magnus 首次接入、云端环境校验和 smoke test
- 它作为高级用户的逃生口存在
- 能在不污染主表单的前提下，保留少量扩展能力
- 但它应被明确视为“可信用户专用”，因为它本质上是原始命令拼接口

## 第一版不建议暴露的参数

下面这些参数不建议在第一版 blueprint 中直接开放：

- `dataset_format`
- `use_lora`
- `lora_r`
- `lora_alpha`
- `lora_dropout`
- `bf16`
- `fp16`
- `logging_steps`
- `save_steps`
- `eval_steps`
- `report_to`
- `trust_remote_code`

为什么不暴露：

- 它们要么属于实现细节
- 要么是低频参数
- 要么一旦放到表单里，就会明显增加使用复杂度和维护成本

第一版 blueprint 的重点应该是“稳定可提交”，而不是“把所有训练选项都摊给用户”。

## 推荐的默认策略

对于当前项目，建议 blueprint 内部默认采用以下策略：

- 数据格式固定为 `messages`
- 默认微调方式固定为 `lora`
- 默认通过 `source conda.sh -> conda activate -> uv sync --quiet -> uv run accelerate launch` 启动
- 所有进入 shell 命令的自由字符串都必须做安全引用
- `REPO_NAME`、`NAMESPACE`、`WORKDIR`、`CONDA_SH`、`CONDA_ENV` 作为模板常量，由维护者在实际接入时替换成真实值

这些策略的目的是：

- 降低表单复杂度
- 提高模板复用性
- 减少用户误操作

## Blueprint 层与训练实现的边界

blueprint 只负责：

1. 定义用户可填写的参数
2. 把这些参数拼接成 `entry_command`
3. 调用 Magnus 的 `submit_job()`

训练实现负责：

- 参数解析
- 数据读取
- 数据格式转换
- tokenizer / model / LoRA 构造
- trainer 创建
- train / eval / save

也就是说：

- 如果某个能力只是训练实现细节，不应当直接暴露到 blueprint
- 如果某个能力已经成为稳定、高频、平台级的使用需求，才值得上升为 blueprint 参数

## 当前文件映射关系

- 训练入口：`train_sft.py`
- blueprint 模板：`blueprints/sft_blueprint_template.py`
- blueprint 使用说明：`docs/blueprint_usage_guide.md`
- blueprint 参数设计说明：`docs/blueprint_parameter_design.md`
