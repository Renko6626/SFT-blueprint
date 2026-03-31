# Flash Attention 2：高效注意力计算

## 标准注意力的瓶颈

Transformer 的 Self-Attention 计算公式为：

```
Attention(Q, K, V) = softmax(QK^T / √d) · V
```

对于序列长度 `N`，这需要计算和存储一个 `N × N` 的注意力矩阵，导致：

- **时间复杂度**：O(N²)
- **显存复杂度**：O(N²)

当序列长度为 2048 时，注意力矩阵就有 ~16M 个元素；4096 时是 ~64M；8192 时是 ~268M。这既占用大量 HBM（显存），又需要大量数据在 HBM 和 SRAM（计算核心缓存）之间来回搬运。

### GPU 内存层级

现代 GPU 的内存结构分两层：

```
SRAM (共享内存/L1缓存)
 ├── 容量：几十 MB
 └── 带宽：~19 TB/s（A100）

HBM (高带宽显存)
 ├── 容量：40 / 80 GB（A100）
 └── 带宽：~2 TB/s（A100）
```

SRAM 比 HBM 快约 10 倍，但容量小得多。标准注意力的问题在于：注意力矩阵太大，无法放进 SRAM，每次计算都要反复读写 HBM，大量时间花在了"等数据传输"上，而不是实际计算。

---

## Flash Attention 的核心思路

Flash Attention（Dao et al., 2022）通过**分块（tiling）计算**解决了这个问题：

1. 将 Q、K、V 分成小块（tile），每次只把一个 tile 加载进 SRAM
2. 在 SRAM 中完成这个 tile 的全部计算（包括 softmax 的分块递推）
3. 只将最终结果写回 HBM，**不存储中间的 N×N 注意力矩阵**

```
标准注意力：          Flash Attention：
Q,K → HBM → SRAM    Q_tile,K_tile → SRAM
QK^T → SRAM → HBM   在 SRAM 内计算并累积
softmax → SRAM       只有最终 output 写回 HBM
× V → SRAM → HBM
```

效果：
- HBM 访问次数从 O(N²) 降至 O(N)
- 不存储 N×N 矩阵，显存从 O(N²) 降至 O(N)
- 数值结果与标准注意力**完全等价**（不是近似）

Flash Attention 2 在此基础上进一步优化了并行度和线程块调度，在 A100 上的实测速度通常是标准注意力的 **2～4 倍**。

---

## A100 上的实际收益

A100 80G 的具体特性使其特别适合 Flash Attention 2：

| 特性 | A100 数据 | 对 Flash Attention 的意义 |
|------|-----------|--------------------------|
| HBM 带宽 | 2 TB/s | 减少 HBM 访问的收益更大 |
| SRAM 大小 | 192 KB / SM | 支持较大的 tile size |
| TF32 / BF16 Tensor Core | 支持 | Flash Attention 的 BF16 路径完全利用 |

在 8B 模型、序列长度 2048、bf16 精度下，开启 Flash Attention 2 后：
- 单步训练时间约降低 30～50%
- 显存占用降低约 20～30%（不存储 N×N 矩阵）

---

## 安装

Flash Attention 不在项目的标准依赖中，需要单独安装：

```bash
# 需要 CUDA 11.8+ 和对应的 PyTorch
pip install flash-attn --no-build-isolation
```

在 Magnus 集群中，如果 `magnus_shared` conda 环境已预装 flash-attn，则无需额外安装。若未预装，可在 blueprint 的 entry_command 中的 `uv sync` 之后手动安装，或请集群管理员预装。

---

## 使用方法

### CLI 开关

```bash
# 开启 Flash Attention 2（推荐 A100 及以上）
python train_sft.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --train_path /data/train.jsonl \
  --output_dir /data/outputs/run1 \
  --use_lora --bf16 \
  --flash_attention

# 关闭（对比基线，或在不支持的硬件上）
python train_sft.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --train_path /data/train.jsonl \
  --output_dir /data/outputs/run1 \
  --use_lora --bf16
  # 不加 --flash_attention，使用默认 Eager 或 SDPA 注意力
```

### Blueprint 参数

在 Magnus 蓝图中，`flash_attention` 默认开启（`True`）。在 A100 环境中建议保持默认。若在不支持的 GPU 上运行，将其关闭即可。

---

## 兼容性说明

| 条件 | 是否支持 Flash Attention 2 |
|------|--------------------------|
| GPU 架构 Ampere (A100) 及以上 | ✅ 完全支持 |
| GPU 架构 Turing (T4, RTX20xx) | ❌ 不支持 |
| PyTorch ≥ 2.0 | ✅ |
| BF16 或 FP16 精度 | ✅ 必须使用，FP32 不支持 |
| 模型支持（transformers 实现） | 需要模型实现了 `attn_implementation` 接口 |

主流支持的模型：Qwen2/2.5、LLaMA-2/3、Mistral、Gemma、Falcon 等。

若模型不支持，`from_pretrained` 会抛出明确的错误，可关闭此选项回退到默认注意力实现。

---

## 实现细节

本项目在 `src/sft/model.py` 的 `load_model_and_tokenizer` 中处理：

```python
extra_kwargs = {"trust_remote_code": args.trust_remote_code, "torch_dtype": torch_dtype}
if args.flash_attention:
    extra_kwargs["attn_implementation"] = "flash_attention_2"
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **extra_kwargs)
```

`attn_implementation` 参数告诉 transformers 在构建模型时将所有注意力层替换为 Flash Attention 2 实现。这一替换在 `from_pretrained` 阶段完成，对训练代码其余部分完全透明。

---

## 对比实验建议

```bash
# 实验 A：开启 Flash Attention 2
python train_sft.py --flash_attention --output_dir outputs/with_fa2 ...

# 实验 B：不开启（默认实现）
python train_sft.py --output_dir outputs/without_fa2 ...
```

观测指标：
- 每步训练耗时（`train_runtime` / `train_steps_per_second`）
- 显存峰值（`nvidia-smi` 或 `torch.cuda.max_memory_allocated()`）
- 两组的最终 `eval_loss` 应当相同（计算等价）
