from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from src.sft.args import build_parser
from src.sft.data import load_datasets, prepare_test_mode_data
from src.sft.eval import evaluate_if_available
from src.sft.formatting import detect_response_template, format_messages
from src.sft.callbacks import GpuMemoryCallback
from src.sft.model import build_peft_config, load_model_and_tokenizer
from src.sft.trainer import build_trainer
from src.sft.utils import LOGGER, configure_logging, dump_json, ensure_dir, set_seed, to_serializable


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config file must be a mapping.")
    return payload


def _merge_args(cli_args: Any, config: dict[str, Any]) -> Any:
    parser = build_parser()
    merged = vars(cli_args).copy()
    for key, value in config.items():
        if key not in merged:
            raise ValueError(f"Unsupported config key: {key}")
        default_value = parser.get_default(key)
        if merged[key] == default_value and value is not None:
            merged[key] = value
    return SimpleNamespace(**merged)


def _validate_args(args: Any) -> None:
    if args.bf16 and args.fp16:
        raise ValueError("Choose at most one mixed precision mode: bf16 or fp16.")
    if args.packing and args.response_only:
        raise ValueError(
            "--packing and --response_only cannot be used together: "
            "sequence packing breaks the token-level loss mask applied by DataCollatorForCompletionOnlyLM."
        )
    if args.model_name_or_path is None:
        raise ValueError("model_name_or_path is required unless test_mode provides a default model.")
    if args.train_path is None:
        raise ValueError("train_path is required unless test_mode prepares test data automatically.")
    if args.val_path is None and args.eval_steps != build_parser().get_default("eval_steps"):
        LOGGER.warning("Ignoring eval_steps=%s because no validation dataset was provided.", args.eval_steps)
    if args.use_lora and not 0 <= args.lora_dropout < 1:
        raise ValueError("LoRA dropout must be in the range [0, 1).")
    if args.num_train_epochs <= 0:
        raise ValueError("num_train_epochs must be greater than zero.")
    if not args.train_path.exists():
        raise FileNotFoundError(f"Training dataset does not exist: {args.train_path}")
    if args.val_path is not None and not args.val_path.exists():
        raise FileNotFoundError(f"Validation dataset does not exist: {args.val_path}")


def _resolve_test_mode_args(args: Any) -> Any:
    if not args.test_mode:
        return args

    resolved = vars(args).copy()
    if resolved["model_name_or_path"] is None:
        resolved["model_name_or_path"] = resolved["test_model_name_or_path"]
    if resolved["train_path"] is None:
        train_path, val_path = prepare_test_mode_data(
            output_dir=resolved["output_dir"],
            dataset_name=resolved["test_dataset"],
            split=resolved["test_dataset_split"],
            train_samples=resolved["test_train_samples"],
            val_samples=resolved["test_val_samples"],
        )
        resolved["train_path"] = train_path
        if resolved["val_path"] is None:
            resolved["val_path"] = val_path
    return SimpleNamespace(**resolved)


def _save_run_metadata(args: Any, train_records: list[dict[str, Any]], val_records: list[dict[str, Any]] | None) -> None:
    resolved_config = to_serializable(vars(args))
    dump_json(args.output_dir / "resolved_config.json", resolved_config)
    summary = {
        "experiment_name": args.experiment_name,
        "test_mode": args.test_mode,
        "train_samples": len(train_records),
        "eval_samples": len(val_records) if val_records else 0,
        "dataset_format": args.dataset_format,
        "use_lora": args.use_lora,
        "model_name_or_path": args.model_name_or_path,
    }
    dump_json(args.output_dir / "run_summary.json", summary)


def _build_data_collator(args: Any, tokenizer: Any) -> Any | None:
    if not args.response_only:
        return None
    from trl import DataCollatorForCompletionOnlyLM

    template = args.response_template or detect_response_template(tokenizer)
    if template is None:
        raise ValueError(
            "--response_only is enabled but the response template could not be auto-detected "
            "from the tokenizer's chat_template. Pass --response_template explicitly, "
            "e.g. --response_template '<|im_start|>assistant\\n'."
        )
    if tokenizer.padding_side != "right":
        LOGGER.warning(
            "DataCollatorForCompletionOnlyLM requires right-padding; "
            "overriding tokenizer.padding_side from %r to 'right'.",
            tokenizer.padding_side,
        )
        tokenizer.padding_side = "right"
    LOGGER.info("Response-only loss masking enabled with template: %r", template)
    return DataCollatorForCompletionOnlyLM(response_template=template, tokenizer=tokenizer)


def _to_hf_dataset(records: list[dict[str, Any]], tokenizer: Any) -> Any:
    from datasets import Dataset

    rows = [{"text": format_messages(record, tokenizer)} for record in records]
    return Dataset.from_list(rows)


def main() -> int:
    configure_logging()
    parser = build_parser()
    cli_args = parser.parse_args()
    args = _merge_args(cli_args, _load_config(cli_args.config))
    ensure_dir(args.output_dir)
    args = _resolve_test_mode_args(args)
    _validate_args(args)
    set_seed(args.seed)
    LOGGER.info("Loading datasets from %s", args.train_path)
    train_records, val_records = load_datasets(args.train_path, args.val_path)
    from accelerate import PartialState
    if PartialState().is_main_process:
        _save_run_metadata(args, train_records, val_records)
    if args.dry_run:
        LOGGER.info("Dry run finished successfully. Configuration and datasets are valid.")
        return 0

    LOGGER.info("Loading model and tokenizer from %s", args.model_name_or_path)
    model, tokenizer = load_model_and_tokenizer(args)
    train_dataset = _to_hf_dataset(train_records, tokenizer)
    eval_dataset = _to_hf_dataset(val_records, tokenizer) if val_records else None
    peft_config = build_peft_config(args)
    data_collator = _build_data_collator(args, tokenizer)
    callbacks = [GpuMemoryCallback()] if args.log_gpu_memory else None
    trainer = build_trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        formatting_func=None,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    LOGGER.info("Starting training")
    train_result = trainer.train()
    train_metrics = {str(key): float(value) for key, value in train_result.metrics.items() if isinstance(value, (int, float))}
    eval_metrics = evaluate_if_available(trainer, eval_dataset)

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    if trainer.is_world_process_zero():
        dump_json(args.output_dir / "train_metrics.json", train_metrics)
        dump_json(args.output_dir / "eval_metrics.json", eval_metrics)
    LOGGER.info("Training complete. Outputs saved to %s", args.output_dir)
    return 0
