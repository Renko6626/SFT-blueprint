# DeepSpeed ZeRO：大模型分布式训练的显存优化

## 问题背景：训练一个大模型需要多少显存？

以 8B 参数的模型为例，用 AdamW + BF16 做全参数训练，每张 GPU 需要存储：

| 内容 | 大小 | 说明 |
|------|------|------|
| 模型权重（BF16）| 16 GB | 8B × 2 bytes |
| 梯度（BF16）| 16 GB | 与权重等大 |
| Adam 一阶矩（FP32）| 32 GB | 8B × 4 bytes |
| Adam 二阶矩（FP32）| 32 GB | 8B × 4 bytes |
| **合计** | **~96 GB** | |

A100 80G 单卡存不下。即使是 LoRA（冻结大部分权重，只训练 adapter），激活值（activations）和梯度也可能随 batch size 和序列长度迅速增长。

朴素的 **DDP（Data Parallel）** 解决不了这个问题——DDP 只在每张卡上复制完整的模型和优化器状态，多卡并行只增加吞吐量，不减少单卡显存。

---

## ZeRO 的核心思路：分片

ZeRO（Zero Redundancy Optimizer，Rajbhandari et al., 2020）的思路是：把原本每张 GPU 都重复存储的内容**分片存储**，每张卡只保存 `1/N` 份，需要时通过 AllGather 操作拼回来。

---

## ZeRO 三个阶段

### Stage 1：分片优化器状态

每张 GPU 只保存 `1/N` 的 Adam 一阶矩和二阶矩。权重和梯度每张卡仍然完整保留。

```
6 卡，8B 模型：
  优化器状态：(32+32) GB / 6 ≈ 10.7 GB per GPU
  权重：16 GB（完整）
  梯度：16 GB（完整）
  单卡约：~43 GB
```

### Stage 2：分片优化器状态 + 梯度

在 Stage 1 基础上，梯度也分片。权重每张卡仍然完整保留。

```
6 卡，8B 模型：
  优化器状态：10.7 GB per GPU
  梯度：16 GB / 6 ≈ 2.7 GB per GPU
  权重：16 GB（完整）
  单卡约：~29 GB  ← 6×A100 完全够用
```

**ZeRO-2 是全参数训练 8B 模型的实用起点。**

### Stage 3：分片权重 + 梯度 + 优化器状态

所有内容都分片。每张卡只保存 `1/N` 的权重，前向/反向传播时通过 AllGather 临时获取其他卡的权重。

```
6 卡，8B 模型：
  优化器状态：10.7 GB per GPU
  梯度：2.7 GB per GPU
  权重：16 GB / 6 ≈ 2.7 GB per GPU
  单卡约：~16 GB  ← 非常宽裕
```

代价：AllGather 的通信量增大，训练速度比 Stage 2 慢约 10～30%（取决于网络带宽）。

---

## 各阶段对比

| | DDP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|--|-----|--------|--------|--------|
| 权重分片 | ❌ | ❌ | ❌ | ✅ |
| 梯度分片 | ❌ | ❌ | ✅ | ✅ |
| 优化器分片 | ❌ | ✅ | ✅ | ✅ |
| 单卡显存（8B，6卡）| ~96 GB | ~43 GB | ~29 GB | ~16 GB |
| 通信开销 | 低 | 低 | 中 | 高 |
| 适用场景 | LoRA | LoRA | 全参数 | 超大模型 |

---

## 本项目提供的配置文件

### `configs/accelerate/deepspeed_zero2.yaml`

适合：6×A100 + 8B 模型**全参数**训练

```yaml
distributed_type: DEEPSPEED
deepspeed_config:
  zero_stage: 2
  offload_optimizer_device: none   # 不 offload 到 CPU，保持训练速度
  offload_param_device: none
  gradient_accumulation_steps: auto
  gradient_clipping: auto
mixed_precision: bf16
num_processes: 6
```

### `configs/accelerate/deepspeed_zero3.yaml`

适合：显存更紧张的场景（更大模型 / 更长序列 / 全参数训练时 ZeRO-2 仍不够）

```yaml
distributed_type: DEEPSPEED
deepspeed_config:
  zero_stage: 3
  zero3_init_flag: true             # 模型初始化时就分片，避免加载完整模型
  zero3_save_16bit_model: true      # 保存时合并分片为完整 BF16 模型
  offload_optimizer_device: none
  offload_param_device: none
mixed_precision: bf16
num_processes: 6
```

---

## 使用方法

### 命令行

```bash
# DDP（默认，无 DeepSpeed）
uv run accelerate launch --num_processes 6 train_sft.py ...

# ZeRO-2
uv run accelerate launch \
  --config_file configs/accelerate/deepspeed_zero2.yaml \
  --num_processes 6 \
  train_sft.py ...

# ZeRO-3
uv run accelerate launch \
  --config_file configs/accelerate/deepspeed_zero3.yaml \
  --num_processes 6 \
  train_sft.py ...
```

`--num_processes` 会覆盖配置文件中的 `num_processes` 字段，保持命令行参数为准。

### Blueprint 参数

在 Magnus 蓝图中，通过 `accelerate_preset` 选择分布式策略：

- `ddp`：朴素数据并行，适合 LoRA
- `zero2`：推荐全参数训练使用
- `zero3`：显存极度紧张时使用

---

## 安装 DeepSpeed

DeepSpeed 不在项目标准依赖中，需要单独安装：

```bash
pip install deepspeed
```

在 Magnus 集群中，如果 `magnus_shared` conda 环境已预装，则无需额外操作。

---

## 重要注意事项

### ZeRO-3 与 LoRA 的兼容性

ZeRO-3 会在模型初始化阶段（`zero3_init_flag: true`）就对权重分片。PEFT 的 LoRA 在这之后再注入 adapter，两者的初始化顺序可能产生冲突。

建议：
- LoRA 训练优先使用 `zero2`（显存通常够用）
- 如果确实需要 ZeRO-3 + LoRA，参考 PEFT 文档中关于 `prepare_model_for_kbit_training` 和 ZeRO-3 的说明

### ZeRO-3 的模型保存

ZeRO-3 下模型权重是分片存储的。在 `build_trainer` 中使用 `trainer.save_model()` 时，会自动触发 AllGather 将权重合并后保存。配置文件中的 `zero3_save_16bit_model: true` 确保保存的是合并后的 BF16 完整模型，而非分片格式。

### CPU Offload 选项

配置文件中 `offload_optimizer_device: none` 表示不将优化器状态 offload 到 CPU。

如果 GPU 显存仍然不足，可以改为 `cpu`：

```yaml
offload_optimizer_device: cpu   # 把 Adam 状态放到 CPU 内存
offload_param_device: cpu       # (ZeRO-3) 把权重也放到 CPU
```

代价：训练速度会显著下降（CPU ↔ GPU 数据传输成为瓶颈），通常只作为最后手段。

---

## 场景选择建议

| 场景 | 推荐策略 |
|------|----------|
| 8B LoRA，6×A100 80G | DDP 或 ZeRO-2 均可 |
| 8B 全参数，6×A100 80G | ZeRO-2 |
| 8B 全参数 + 长序列（≥4096），6×A100 80G | ZeRO-3 |
| 70B LoRA，6×A100 80G | ZeRO-3 |
| 70B 全参数 | ZeRO-3 + CPU Offload，或更多 GPU |
