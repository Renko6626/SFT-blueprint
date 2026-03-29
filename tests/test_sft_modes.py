from pathlib import Path
from types import SimpleNamespace

from src.sft.model import build_peft_config
from src.sft.trainer import build_training_args


def _make_args(**overrides: object) -> SimpleNamespace:
    base = {
        "output_dir": Path("outputs/test"),
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 2e-5,
        "num_train_epochs": 3.0,
        "logging_steps": 10,
        "save_steps": 200,
        "eval_steps": 200,
        "val_path": Path("tests/fixtures/val_messages.jsonl"),
        "bf16": False,
        "fp16": False,
        "report_to": "none",
        "seed": 42,
        "max_seq_length": 4096,
        "use_lora": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_peft_config_returns_lora_config_when_enabled() -> None:
    args = _make_args(use_lora=True, lora_r=4, lora_alpha=8, lora_dropout=0.1)
    config = build_peft_config(args)
    assert config is not None
    assert config.r == 4
    assert config.lora_alpha == 8
    assert config.lora_dropout == 0.1


def test_build_peft_config_returns_none_when_lora_disabled() -> None:
    args = _make_args(use_lora=False)
    assert build_peft_config(args) is None


def test_build_training_args_uses_eval_steps_when_val_path_exists() -> None:
    args = _make_args(val_path=Path("tests/fixtures/val_messages.jsonl"), report_to="wandb")
    training_args = build_training_args(args)
    assert training_args.eval_strategy == "steps"
    assert training_args.report_to == ["wandb"]
    assert training_args.max_length == 4096


def test_build_training_args_disables_eval_without_val_path() -> None:
    args = _make_args(val_path=None)
    training_args = build_training_args(args)
    assert training_args.eval_strategy == "no"
    assert training_args.report_to == []
