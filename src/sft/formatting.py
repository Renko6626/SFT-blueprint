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
