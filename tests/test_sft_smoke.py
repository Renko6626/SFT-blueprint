import json
from pathlib import Path
from types import SimpleNamespace

from src.sft.args import build_parser
from src.sft.data import load_messages_jsonl
from src.sft.cli import _merge_args, _resolve_test_mode_args, _save_run_metadata, _validate_args
from src.sft.formatting import format_messages


def test_parser_accepts_required_args() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--output_dir",
            "outputs/test",
            "--experiment_name",
            "demo-run",
            "--model_name_or_path",
            "/models/Qwen2.5-7B-Instruct",
            "--train_path",
            "tests/fixtures/train_messages.jsonl",
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


def test_validate_args_requires_train_path_without_test_mode() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--model_name_or_path",
            "/models/Qwen2.5-7B-Instruct",
            "--output_dir",
            "outputs/test",
        ]
    )
    try:
        _validate_args(args)
    except ValueError as exc:
        assert "train_path is required" in str(exc)
    else:
        raise AssertionError("Expected missing train_path to raise ValueError")


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


def test_resolve_test_mode_args_uses_default_test_model_and_paths(tmp_path: Path, monkeypatch) -> None:
    class FakeDataset:
        def __init__(self, rows: list[dict[str, object]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def select(self, indices: range) -> list[dict[str, object]]:
            return [self.rows[index] for index in indices]

    fake_rows = [
        {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}
        for i in range(10)
    ]

    def fake_load_dataset(dataset_name: str, split: str) -> FakeDataset:
        assert dataset_name == "Butanium/femto-ultrachat"
        assert split == "train"
        return FakeDataset(fake_rows)

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    args = SimpleNamespace(
        experiment_name="test-mode",
        model_name_or_path=None,
        train_path=None,
        val_path=None,
        output_dir=tmp_path,
        test_mode=True,
        test_model_name_or_path="trl-internal-testing/tiny-GPT2LMHeadModel",
        test_dataset="Butanium/femto-ultrachat",
        test_dataset_split="train",
        test_train_samples=8,
        test_val_samples=2,
    )
    resolved = _resolve_test_mode_args(args)
    assert resolved.model_name_or_path == "trl-internal-testing/tiny-GPT2LMHeadModel"
    assert resolved.train_path.exists()
    assert resolved.val_path.exists()
