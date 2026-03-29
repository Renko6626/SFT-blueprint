from pathlib import Path
from typing import Annotated, Literal, Optional


def _load_blueprint_namespace() -> tuple[dict[str, object], list[dict[str, object]]]:
    captured_jobs: list[dict[str, object]] = []

    class DummyJobType:
        A1 = "A1"
        A2 = "A2"
        B1 = "B1"
        B2 = "B2"

    namespace: dict[str, object] = {
        "Annotated": Annotated,
        "Literal": Literal,
        "Optional": Optional,
        "JobType": DummyJobType,
        "submit_job": lambda **kwargs: captured_jobs.append(kwargs),
    }
    blueprint_code = Path("blueprints/sft_blueprint_template.py").read_text(encoding="utf-8")
    exec(blueprint_code, namespace)
    return namespace, captured_jobs


def test_blueprint_lora_mode_includes_use_lora_and_val_path() -> None:
    namespace, captured_jobs = _load_blueprint_namespace()
    namespace["blueprint"](
        runner="alice",
        experiment_name="demo-lora",
        train_path="/data/train.jsonl",
        val_path="/data/val.jsonl",
        output_root="/data/outputs",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        finetune_method="lora",
        extra_args="--save_steps 50",
    )
    job = captured_jobs[0]
    entry_command = job["entry_command"]
    description = job["description"]

    assert "--use_lora" in entry_command
    assert "--val_path '/data/val.jsonl'" in entry_command
    assert "--save_steps 50" in entry_command
    assert "demo-lora" in description
    assert job["task_name"] == "SFT-demo-lora"


def test_blueprint_full_mode_omits_lora_flag_and_optional_fields() -> None:
    namespace, captured_jobs = _load_blueprint_namespace()
    namespace["blueprint"](
        runner="bob",
        experiment_name="demo-full",
        train_path="/data/train.jsonl",
        output_root="/data/outputs",
        finetune_method="full",
        gpu_count=2,
        gpu_type="a100",
        priority="B1",
    )
    job = captured_jobs[0]
    entry_command = job["entry_command"]
    description = job["description"]

    assert "--use_lora" not in entry_command
    assert "--val_path" not in entry_command
    assert "--num_processes 2" in entry_command
    assert "Val Path: None" in description
    assert job["gpu_type"] == "a100"
    assert job["job_type"] == "B1"
