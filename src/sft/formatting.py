from __future__ import annotations

from typing import Any


def format_messages(example: dict[str, Any], tokenizer: Any) -> str:
    messages = example["messages"]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def detect_response_template(tokenizer: Any) -> str | None:
    """Infer the response template string from the tokenizer's chat_template.

    The response template is the literal token string that immediately precedes
    the assistant's reply in the formatted text. DataCollatorForCompletionOnlyLM
    uses it to locate and mask all non-assistant tokens.

    Returns None when the chat template is not recognized.
    """
    chat_template = getattr(tokenizer, "chat_template", None) or ""
    # ChatML — Qwen2, Yi, InternLM2, etc.
    if "<|im_start|>" in chat_template:
        return "<|im_start|>assistant\n"
    # LLaMA-3 / LLaMA-3.x
    if "<|start_header_id|>" in chat_template:
        return "<|start_header_id|>assistant<|end_header_id|>\n\n"
    # Mistral / LLaMA-2
    if "[/INST]" in chat_template:
        return "[/INST]"
    # Gemma
    if "<start_of_turn>model" in chat_template:
        return "<start_of_turn>model\n"
    return None
