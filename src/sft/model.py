from __future__ import annotations

from typing import Any


def load_model_and_tokenizer(args: Any) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else "auto")
    extra_kwargs: dict = {"trust_remote_code": args.trust_remote_code, "torch_dtype": torch_dtype}
    if args.flash_attention:
        extra_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **extra_kwargs)
    return model, tokenizer


def build_peft_config(args: Any) -> Any | None:
    if not args.use_lora:
        return None

    from peft import LoraConfig, TaskType

    target_modules = [m.strip() for m in args.lora_target_modules.split(",")] if args.lora_target_modules else None
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
