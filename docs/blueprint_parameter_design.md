# Blueprint 参数设计说明

本文档说明哪些参数应该在 Magnus blueprint 层暴露给用户，哪些应当留在训练实现内部，以及背后的设计依据。

## 设计原则

Blueprint 的职责是做一个可复用、可维护的任务提交模板，而不是再造一层训练系统。

因此 blueprint 层只应暴露：

- 用户实际需要按实验改动的参数
- 对结果或资源配置影响显著的参数
- 能长期稳定作为表单契约存在的参数

不应暴露过多训练实现细节，否则 blueprint 会很快失去模板价值，变成难以维护的大表单。

---

## 建议暴露的参数

### Basic

| 参数 | 理由 |
|------|------|
| `runner` | 任务归属和审计追踪必需 |
| `experiment_name` | 用于任务标识和输出目录构造 |
| `notes` | 实验上下文记录，不影响训练逻辑 |

### Data

| 参数 | 理由 |
|------|------|
| `train_path` | 启动训练的必要输入 |
| `val_path` | 可选，影响评估行为和 best model 加载 |
| `output_root` | 比直接暴露完整路径更稳定，blueprint 可统一按 experiment_name 组织输出 |

### Model

| 参数 | 理由 |
|------|------|
| `base_model` | 最核心的模型决策，每次实验都可能不同 |
| `finetune_method` | 决定训练方式（LoRA vs 全参数），影响显存和效果 |
| `max_seq_len` | 影响截断行为、显存占用和训练吞吐，用户需要明确控制 |

### Optimization

| 参数 | 理由 |
|------|------|
| `epochs` | 最常调整的训练超参数之一 |
| `learning_rate` | 对训练效果影响最大的参数 |
| `per_device_batch_size` | 影响显存和有效 batch size |
| `gradient_accumulation_steps` | 与 batch size 共同决定有效 batch size |
| `gradient_checkpointing` | 大模型训练的关键显存开关，用户需要明确控制；7B 以上强烈建议暴露 |
| `save_total_limit` | 直接影响磁盘占用，长时间训练时容易被忽略导致存储爆满 |

### Cluster

| 参数 | 理由 |
|------|------|
| `gpu_count` | Magnus 调度层参数，提交者必须能控制资源申请 |
| `gpu_type` | 同上 |
| `priority` | 同上 |

### Misc

| 参数 | 理由 |
|------|------|
| `test_mode` | 环境验证和 smoke test 的标准入口，接入新仓库时必需 |
| `extra_args` | 高级用户的逃生口，避免每次加新低频参数都需要改 blueprint |

---

## 不建议暴露的参数

以下参数留在训练实现内部，不在第一版 blueprint 中开放：

| 参数 | 理由 |
|------|------|
| `dataset_format` | 当前只支持 `messages`，无选择空间，暴露无意义 |
| `use_lora` | 由 `finetune_method` 间接控制，不需要单独暴露 |
| `lora_r` / `lora_alpha` / `lora_dropout` | LoRA 内部调参，属于低频高级参数，放 `extra_args` 处理 |
| `lora_target_modules` | 仅在特定模型架构出现问题时才需要手动指定，放 `extra_args` 处理 |
| `bf16` / `fp16` | 精度选择通常由集群环境决定，可在 blueprint 内部根据 GPU 类型硬编码，或通过 `extra_args` 传入 |
| `logging_steps` / `save_steps` / `eval_steps` | 实现细节，在 blueprint 内部固定为合理默认值（10/200/200） |
| `report_to` | 日志上报后端，通常由基础设施层统一配置，不需要用户每次填写 |
| `trust_remote_code` | 安全敏感参数，不应作为表单暴露给普通用户 |
| `load_best_model_at_end` / `metric_for_best_model` | 提供 `val_path` 时自动开启，不需要用户手动控制 |
| `seed` | 低频参数，需要固定种子时通过 `extra_args` 传入 |

---

## 参数边界决策记录

### `gradient_checkpointing` 为何暴露

早期版本将 `gradient_checkpointing` 硬编码为 `False`。实际使用中，7B 及以上模型不开此项容易 OOM，而用户在 blueprint 层没有办法控制。因此将其提升为 blueprint 表单参数，默认关闭，用户可按模型规模自行开启。

### `save_total_limit` 为何暴露

`save_total_limit` 不设上限时，checkpoint 数量随训练步数线性增长。长时间训练或 `save_steps` 较小时，容易无声地撑爆存储。暴露此参数并设默认值 3，可以明确地让用户意识到存储管理问题。

### `load_best_model_at_end` 为何不暴露

提供 `val_path` 时自动开启，行为确定、无歧义，不需要额外的表单字段。用户只需要知道"提供验证集就能得到最优 checkpoint"即可。

### `lora_target_modules` 为何不暴露

绝大多数主流模型（Qwen、LLaMA、Mistral 等）都能被 PEFT 正确自动推断 target modules。只有极少数自定义架构才需要手动指定，属于低频边缘情况，通过 `extra_args` 传入足够。

---

## 文件映射关系

| 文件 | 职责 |
|------|------|
| `train_sft.py` | 训练入口，委托给 `src/sft/cli.py` |
| `src/sft/` | 全部训练实现逻辑 |
| `blueprints/sft_blueprint_template.py` | Magnus 蓝图模板 |
| `configs/sft/base.yaml` | 训练超参默认值 |
| `docs/blueprint_usage_guide.md` | 面向用户的使用说明 |
| `docs/blueprint_parameter_design.md` | 面向维护者的设计决策记录（本文档） |
