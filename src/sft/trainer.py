from __future__ import annotations

import logging
import os
from typing import Any

LOGGER = logging.getLogger("sft")


def _is_zero3() -> bool:
    """Return True when the current process runs under DeepSpeed ZeRO-3."""
    try:
        from accelerate.state import AcceleratorState
        state = AcceleratorState()
        ds_plugin = getattr(state, "deepspeed_plugin", None)
        if ds_plugin is None:
            return False
        return getattr(ds_plugin, "zero_stage", 0) == 3
    except Exception:
        return False


def build_training_args(args: Any) -> Any:
    import torch
    from trl import SFTConfig

    has_val = bool(args.val_path)

    # ZeRO-3 shards model weights across GPUs; loading the best checkpoint at the
    # end requires gathering all shards onto one process, which OOMs for 8B models.
    zero3 = _is_zero3()
    load_best = has_val and not zero3
    if has_val and zero3:
        LOGGER.warning(
            "load_best_model_at_end is disabled because DeepSpeed ZeRO-3 is active. "
            "The final checkpoint (not necessarily the best) will be saved."
        )

    return SFTConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if has_val else "no",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss" if load_best else None,
        greater_is_better=False if load_best else None,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[] if args.report_to == "none" else [args.report_to],
        seed=args.seed,
        max_length=args.max_seq_length,
        lr_scheduler_type=args.lr_scheduler_type,
        use_cpu=not torch.cuda.is_available(),
        # adamw_torch_fused uses CUDA fused kernels on A100, ~10-15% faster than adamw_torch.
        # DeepSpeed ZeRO wraps the optimizer itself, so this only has effect under DDP.
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        packing=args.packing,
        dataset_num_proc=min(os.cpu_count() or 1, 16),
        dataset_text_field="text",
    )


def build_trainer(
    *,
    args: Any,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any | None,
    peft_config: Any | None,
    formatting_func: Any,
    data_collator: Any | None = None,
    callbacks: list[Any] | None = None,
) -> Any:
    from trl import SFTTrainer

    training_args = build_training_args(args)
    return SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        peft_config=peft_config,
        formatting_func=formatting_func,
        data_collator=data_collator,
        callbacks=callbacks,
    )
