# Loss Mask：只对 Assistant 回复计算损失

## 问题背景

在 SFT 指令微调中，每条训练样本通常包含三个部分：

```
[System Prompt] + [User 输入] + [Assistant 回复]
```

标准的语言模型损失函数会对序列中的**每一个 token** 计算交叉熵：

```
Loss = -Σ log P(token_t | token_1, ..., token_{t-1})
```

这意味着模型不仅要学习"如何回复"，还要学习"system prompt 长什么样"和"user 是怎么提问的"。

### 为什么这有问题

**问题一：浪费梯度信号**

System prompt 和 user 输入通常是固定模板，没有信息量。让模型去拟合这部分 token 的预测损失，相当于把宝贵的梯度更新花在了"背诵问题"上，而不是"学习回答"。

**问题二：干扰学习目标**

指令微调的核心目的是让模型学会"根据指令生成高质量回复"。如果损失函数同时也惩罚"没有预测好 user 的提问方式"，学习信号是混乱的。

**问题三：在长 System Prompt 场景下尤为严重**

当 system prompt 很长（如详细的角色设定、任务描述）时，prompt 部分的 token 数远大于 assistant 回复，大部分梯度信号都来自"预测固定模板"，而非学习任务本身。

---

## 解决方案：Loss Mask

Loss Mask 的思路很简单：**只对 assistant 回复部分的 token 计算损失，将 system/user 部分的损失屏蔽为 0**。

```
输入:  [System][User][Assistant 回复]
Loss:   ✗      ✗     ✓✓✓✓✓✓✓✓✓✓
```

这样梯度信号完全来自于"模型是否生成了正确的 assistant 回复"。

---

## 实现原理

TRL 提供了 `DataCollatorForCompletionOnlyLM` 来实现这一功能。

### 核心机制

1. **格式化阶段**：对话被序列化为带 chat template 的完整文本，例如：
   ```
   <|im_start|>system
   你是一个助手<|im_end|>
   <|im_start|>user
   解释一下 LoRA<|im_end|>
   <|im_start|>assistant
   LoRA 是一种参数高效微调方法...<|im_end|>
   ```

2. **Tokenization 阶段**：整条文本被 tokenizer 转换为 token ID 序列。

3. **Mask 阶段**：DataCollator 在 token 序列中搜索 `response_template` 对应的 token ID 子序列（如 `<|im_start|>assistant\n`），将其**之前**所有 token 对应的 `labels` 设为 `-100`。

4. **Loss 计算阶段**：PyTorch 的 `CrossEntropyLoss` 在计算时自动忽略 `label == -100` 的位置，等效于屏蔽这些 token 的损失。

### Response Template

Response template 是区分"prompt 结束 / assistant 回复开始"的关键字符串，每种 chat template 格式不同：

| 模型系列 | Chat Template 格式 | Response Template |
|----------|-------------------|-------------------|
| Qwen2, Yi, InternLM2 | ChatML (`<\|im_start\|>`) | `<\|im_start\|>assistant\n` |
| LLaMA-3 / 3.x | LLaMA-3 header | `<\|start_header_id\|>assistant<\|end_header_id\|>\n\n` |
| Mistral / LLaMA-2 | `[/INST]` | `[/INST]` |
| Gemma | `<start_of_turn>model` | `<start_of_turn>model\n` |

本项目在 `formatting.py` 的 `detect_response_template()` 函数中实现了自动推断，通过检测 tokenizer 的 `chat_template` 字段识别格式。

---

## 使用方法

### CLI 开关

```bash
# 开启（推荐，指令微调标配）
python train_sft.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --train_path /data/train.jsonl \
  --output_dir /data/outputs/run1 \
  --use_lora --bf16 \
  --response_only

# 关闭（对比基线）
python train_sft.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --train_path /data/train.jsonl \
  --output_dir /data/outputs/run1 \
  --use_lora --bf16
  # 不加 --response_only
```

### 手动指定 Response Template

当 tokenizer 的 chat_template 无法被自动识别时，手动传入：

```bash
python train_sft.py \
  --response_only \
  --response_template '<|im_start|>assistant\n' \
  ...
```

### Blueprint 参数

在 Magnus 蓝图中，`response_only` 默认开启（`True`），对于指令微调任务通常无需改动。

---

## 注意事项

**`response_template` 必须出现在格式化后的文本中**

DataCollator 通过在 tokenized 序列中搜索 response_template 的 token IDs 来定位 mask 边界。如果搜索失败（template 不在序列中），该样本的整条 labels 会被置为 `-100`，等于这条样本对训练没有任何贡献。

常见原因：
- Template 字符串与实际 chat template 的格式不完全一致（如多了/少了空格、换行）
- 使用了 fallback 格式化路径（tokenizer 没有 chat_template）而 response_template 仍按 ChatML 格式指定

**与 Data Packing 同时使用时需注意**

Data packing（多条短样本拼接到一条序列）和 `DataCollatorForCompletionOnlyLM` 同时使用时，需要确保每个 packing 边界对 loss mask 的处理是正确的。TRL 在较新版本中已支持这一组合，但建议确认 TRL 版本 ≥ 0.12。

---

## 对比实验建议

如果要做 response_only 效果的对比实验，建议控制其他变量不变，只切换此开关：

```bash
# 实验 A：开启 response_only
python train_sft.py --response_only --output_dir outputs/with_mask ...

# 实验 B：关闭（全文本 loss）
python train_sft.py --output_dir outputs/without_mask ...
```

观测指标：
- `eval_loss`（两组应在同一数量级，但趋势可能不同）
- 生成质量（更直观，建议用固定 prompt 集做推理对比）
- 训练过程中的 loss 下降速度
