import json
from pathlib import Path

from src.sft.args import build_parser
from src.sft.data import load_messages_jsonl
from src.sft.cli import _merge_args, _save_run_metadata, _validate_args
from src.sft.formatting import format_messages


def test_parser_accepts_required_args() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--experiment_name",
            "demo-run",
            "--model_name_or_path",
            "/models/Qwen2.5-7B-Instruct",
            "--train_path",
            "tests/fixtures/train_messages.jsonl",
            "--output_dir",
            "outputs/test",
        ]
    )
    assert args.experiment_name == "demo-run"
    assert args.model_name_or_path == "/models/Qwen2.5-7B-Instruct"


def test_load_messages_jsonl() -> None:
    records = load_messages_jsonl(Path("tests/fixtures/train_messages.jsonl"))
    assert len(records) == 2
    assert records[0]["messages"][-1]["role"] == "assistant"


def test_load_messages_jsonl_rejects_non_assistant_last_message(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text('{"messages":[{"role":"user","content":"hello"}]}\n', encoding="utf-8")
    try:
        load_messages_jsonl(bad_path)
    except ValueError as exc:
        assert "last message must be from 'assistant'" in str(exc)
    else:
        raise AssertionError("Expected invalid dataset to raise ValueError")


def test_merge_args_rejects_unknown_config_keys() -> None:
    parser = build_parser()
    cli_args = parser.parse_args(
        [
            "--model_name_or_path",
            "/models/Qwen2.5-7B-Instruct",
            "--train_path",
            "tests/fixtures/train_messages.jsonl",
            "--output_dir",
            "outputs/test",
        ]
    )
    try:
        _merge_args(cli_args, {"unknown_key": 1})
    except ValueError as exc:
        assert "Unsupported config key" in str(exc)
    else:
        raise AssertionError("Expected unsupported config key to raise ValueError")


def test_validate_args_rejects_bf16_and_fp16() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--model_name_or_path",
            "/models/Qwen2.5-7B-Instruct",
            "--train_path",
            "tests/fixtures/train_messages.jsonl",
            "--output_dir",
            "outputs/test",
            "--bf16",
            "--fp16",
        ]
    )
    try:
        _validate_args(args)
    except ValueError as exc:
        assert "Choose at most one mixed precision mode" in str(exc)
    else:
        raise AssertionError("Expected invalid precision combination to raise ValueError")


def test_save_run_metadata_writes_config_and_summary(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--model_name_or_path",
            "/models/Qwen2.5-7B-Instruct",
            "--train_path",
            "tests/fixtures/train_messages.jsonl",
            "--output_dir",
            str(tmp_path),
            "--experiment_name",
            "metadata-test",
            "--dry_run",
        ]
    )
    records = load_messages_jsonl(Path("tests/fixtures/train_messages.jsonl"))
    _save_run_metadata(args, records, None)
    resolved = json.loads((tmp_path / "resolved_config.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert resolved["output_dir"] == str(tmp_path)
    assert summary["experiment_name"] == "metadata-test"
    assert summary["train_samples"] == 2


def test_format_messages_falls_back_without_chat_template() -> None:
    class DummyTokenizer:
        chat_template = None

    text = format_messages(
        {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]
        },
        DummyTokenizer(),
    )
    assert "user: hello" in text
    assert "assistant: world" in text
