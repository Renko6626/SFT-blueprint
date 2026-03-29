from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VALID_ROLES = {"system", "user", "assistant", "tool"}


def _validate_messages(messages: Any, source: Path, line_no: int) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{source}:{line_no} must contain a non-empty 'messages' list.")

    validated: list[dict[str, str]] = []
    for index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise ValueError(f"{source}:{line_no} message #{index} is not an object.")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError(f"{source}:{line_no} message #{index} missing valid 'role'.")
        if role not in VALID_ROLES:
            raise ValueError(f"{source}:{line_no} message #{index} has unsupported role '{role}'.")
        if not isinstance(content, str):
            raise ValueError(f"{source}:{line_no} message #{index} missing valid 'content'.")
        validated.append({"role": role, "content": content})
    if validated[-1]["role"] != "assistant":
        raise ValueError(f"{source}:{line_no} last message must be from 'assistant'.")
    return validated


def load_messages_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Dataset path is not a file: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
            if "messages" not in item:
                raise ValueError(f"{path}:{line_no} missing 'messages'.")
            records.append({"messages": _validate_messages(item["messages"], path, line_no)})
    if not records:
        raise ValueError(f"{path} contains no valid samples.")
    return records


def load_datasets(train_path: Path, val_path: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    train_records = load_messages_jsonl(train_path)
    val_records = load_messages_jsonl(val_path) if val_path else None
    return train_records, val_records


def normalize_messages_record(item: dict[str, Any], source: Path, line_no: int = 1) -> dict[str, Any]:
    if "messages" not in item:
        raise ValueError(f"{source}:{line_no} missing 'messages'.")
    return {"messages": _validate_messages(item["messages"], source, line_no)}


def write_messages_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_test_mode_data(
    *,
    output_dir: Path,
    dataset_name: str,
    split: str,
    train_samples: int,
    val_samples: int,
) -> tuple[Path, Path]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "test_mode requires the 'datasets' package, but it is not installed in the current runtime. "
            "Make sure the Magnus job installs requirements.txt before running train_sft.py."
        ) from exc

    cache_dir = output_dir / "test_mode_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        dataset = load_dataset(dataset_name, split=split, cache_dir=str(cache_dir))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download test dataset '{dataset_name}' in test_mode. "
            "Check Hugging Face network access and writable cache paths."
        ) from exc
    required = train_samples + val_samples
    if len(dataset) < required:
        raise ValueError(f"Dataset split has {len(dataset)} rows, fewer than required {required}.")

    records = [
        normalize_messages_record(dict(item), Path(f"{dataset_name}:{split}"), index + 1)
        for index, item in enumerate(dataset.select(range(required)))
    ]
    train_rows = records[:train_samples]
    val_rows = records[train_samples:required]

    target_dir = output_dir / "test_mode_data"
    train_path = target_dir / "train_messages.jsonl"
    val_path = target_dir / "val_messages.jsonl"
    write_messages_jsonl(train_path, train_rows)
    write_messages_jsonl(val_path, val_rows)
    return train_path, val_path
