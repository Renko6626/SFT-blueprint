from __future__ import annotations

from typing import Any


def build_training_args(args: Any) -> Any:
    import torch
    from trl import SFTConfig

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
        eval_strategy="steps" if args.val_path else "no",
        save_strategy="steps",
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[] if args.report_to == "none" else [args.report_to],
        seed=args.seed,
        max_length=args.max_seq_length,
        use_cpu=not torch.cuda.is_available(),
        optim="adamw_torch",
        gradient_checkpointing=False,
        dataset_num_proc=1,
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
    )
