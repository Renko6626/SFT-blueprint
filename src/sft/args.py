from __future__ import annotations

import argparse
from pathlib import Path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an SFT model.")
    parser.add_argument("--experiment_name", default=None, help="Optional experiment name for tracking.")
    parser.add_argument("--model_name_or_path", required=True, help="HF model name or local path.")
    parser.add_argument("--train_path", required=True, type=Path, help="Training dataset path.")
    parser.add_argument("--val_path", type=Path, default=None, help="Optional validation dataset path.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--dataset_format", default="messages", choices=["messages"], help="Dataset format.")
    parser.add_argument("--max_seq_length", type=positive_int, default=4096, help="Maximum sequence length.")
    parser.add_argument("--per_device_train_batch_size", type=positive_int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=positive_int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=positive_int, default=16)
    parser.add_argument("--learning_rate", type=non_negative_float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=non_negative_float, default=3.0)
    parser.add_argument("--logging_steps", type=positive_int, default=10)
    parser.add_argument("--save_steps", type=positive_int, default=200)
    parser.add_argument("--eval_steps", type=positive_int, default=200)
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning.")
    parser.add_argument("--lora_r", type=positive_int, default=8)
    parser.add_argument("--lora_alpha", type=positive_int, default=16)
    parser.add_argument("--lora_dropout", type=non_negative_float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config path.")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--report_to", default="none", help="Trainer reporting backend.")
    parser.add_argument("--dry_run", action="store_true", help="Validate config and datasets without training.")
    return parser
