REPO_NAME = "your-repo-name"
NAMESPACE = "your-github-org"
WORKDIR = "/workspace/your-repo"


Runner = Annotated[str, {
    "label": "Runner",
    "description": "Cluster user who runs this job.",
    "scope": "Basic",
    "placeholder": "your_user_name",
    "allow_empty": False,
}]

ExperimentName = Annotated[str, {
    "label": "Experiment Name",
    "description": "Short stable name for this training run.",
    "scope": "Basic",
    "placeholder": "qwen25-05b-sft-v1",
    "allow_empty": False,
}]

TrainPath = Annotated[str, {
    "label": "Train Dataset Path",
    "description": "Absolute path to the training JSONL file on cluster storage.",
    "scope": "Data",
    "placeholder": "/data/project/train_messages.jsonl",
    "allow_empty": False,
}]

ValPath = Annotated[Optional[str], {
    "label": "Validation Dataset Path",
    "description": "Optional validation JSONL path on cluster storage.",
    "scope": "Data",
    "placeholder": "/data/project/val_messages.jsonl",
    "allow_empty": False,
}]

OutputRoot = Annotated[str, {
    "label": "Output Root",
    "description": "Parent directory where this run's output directory will be created.",
    "scope": "Data",
    "placeholder": "/data/outputs/sft",
    "allow_empty": False,
}]

BaseModel = Annotated[str, {
    "label": "Base Model",
    "description": "HF model id or local model path.",
    "scope": "Model",
    "placeholder": "Qwen/Qwen2.5-0.5B-Instruct",
    "allow_empty": False,
}]

FinetuneMethod = Annotated[Literal["lora", "full"], {
    "label": "Finetune Method",
    "description": "Training strategy exposed by this template.",
    "scope": "Model",
    "options": {
        "lora": {"label": "LoRA", "description": "Recommended low-cost fine-tuning path."},
        "full": {"label": "Full", "description": "Full-parameter training for advanced cases."},
    },
}]

MaxSeqLen = Annotated[int, {
    "label": "Max Sequence Length",
    "description": "Maximum sequence length passed to train_sft.py.",
    "scope": "Model",
    "min": 128,
    "max": 32768,
}]

Epochs = Annotated[int, {
    "label": "Epochs",
    "description": "Number of training epochs.",
    "scope": "Optimization",
    "min": 1,
    "max": 50,
}]

LearningRate = Annotated[float, {
    "label": "Learning Rate",
    "description": "Optimizer learning rate.",
    "scope": "Optimization",
    "min": 0.0,
    "max": 1.0,
    "placeholder": "2e-5",
}]

PerDeviceBatchSize = Annotated[int, {
    "label": "Per Device Batch Size",
    "description": "Train batch size per device.",
    "scope": "Optimization",
    "min": 1,
    "max": 512,
}]

GradAccumSteps = Annotated[int, {
    "label": "Gradient Accumulation",
    "description": "Gradient accumulation steps.",
    "scope": "Optimization",
    "min": 1,
    "max": 4096,
}]

GpuCount = Annotated[int, {
    "label": "GPU Count",
    "description": "Number of GPUs requested for this job.",
    "scope": "Cluster",
    "min": 1,
    "max": 16,
}]

GpuType = Annotated[str, {
    "label": "GPU Type",
    "description": "Exact GPU type string supported by the Magnus cluster.",
    "scope": "Cluster",
    "placeholder": "rtx5090",
    "allow_empty": False,
}]

Priority = Annotated[Literal["A1", "A2", "B1", "B2"], {
    "label": "Priority",
    "description": "Scheduling priority for the job.",
    "scope": "Cluster",
    "options": {
        "A1": {"label": "A1", "description": "Highest priority."},
        "A2": {"label": "A2", "description": "High priority."},
        "B1": {"label": "B1", "description": "Preemptible."},
        "B2": {"label": "B2", "description": "Lowest priority."},
    },
}]

Notes = Annotated[Optional[str], {
    "label": "Notes",
    "description": "Optional experiment notes stored in Magnus.",
    "scope": "Misc",
    "multi_line": True,
    "min_lines": 4,
    "placeholder": "Why this run exists and what you expect from it.",
}]

ExtraArgs = Annotated[Optional[str], {
    "label": "Extra CLI Args",
    "description": "Optional raw extra CLI arguments appended to train_sft.py. Trusted users only.",
    "scope": "Misc",
    "multi_line": True,
    "min_lines": 3,
    "placeholder": "--save_steps 100 --eval_steps 100",
}]


def blueprint(
    runner: Runner,
    experiment_name: ExperimentName,
    train_path: TrainPath,
    output_root: OutputRoot,
    val_path: ValPath = None,
    base_model: BaseModel = "Qwen/Qwen2.5-0.5B-Instruct",
    finetune_method: FinetuneMethod = "lora",
    max_seq_len: MaxSeqLen = 4096,
    epochs: Epochs = 3,
    learning_rate: LearningRate = 2e-5,
    per_device_batch_size: PerDeviceBatchSize = 1,
    gradient_accumulation_steps: GradAccumSteps = 16,
    gpu_count: GpuCount = 1,
    gpu_type: GpuType = "rtx5090",
    priority: Priority = "A2",
    notes: Notes = None,
    extra_args: ExtraArgs = None,
):
    def shq(value):
        return "'" + str(value).replace("'", "'\"'\"'") + "'"

    output_dir = f"{output_root.rstrip('/')}/{experiment_name}"

    train_cmd = (
        "accelerate launch"
        + f" --num_processes {gpu_count}"
        + " train_sft.py"
        + f" --experiment_name {shq(experiment_name)}"
        + f" --model_name_or_path {shq(base_model)}"
        + f" --train_path {shq(train_path)}"
        + f" --output_dir {shq(output_dir)}"
        + " --dataset_format messages"
        + f" --max_seq_length {max_seq_len}"
        + f" --per_device_train_batch_size {per_device_batch_size}"
        + f" --per_device_eval_batch_size {per_device_batch_size}"
        + f" --gradient_accumulation_steps {gradient_accumulation_steps}"
        + f" --learning_rate {learning_rate}"
        + f" --num_train_epochs {epochs}"
        + " --logging_steps 10"
        + " --save_steps 200"
        + " --eval_steps 200"
    )

    if val_path is not None:
        train_cmd += f" --val_path {shq(val_path)}"

    if finetune_method == "lora":
        train_cmd += " --use_lora"

    if extra_args:
        train_cmd += f" {extra_args}"

    entry_command = "\n".join([
        f"cd {shq(WORKDIR)}",
        "python -m pip install -r requirements.txt",
        train_cmd,
    ])

    description = f"""## SFT Run

- Experiment: {experiment_name}
- Runner: {runner}
- Train Path: {train_path}
- Val Path: {val_path or 'None'}
- Output Dir: {output_dir}
- Base Model: {base_model}
- Finetune Method: {finetune_method}
- Max Seq Len: {max_seq_len}
- Epochs: {epochs}
- Learning Rate: {learning_rate}
- Batch Size / Device: {per_device_batch_size}
- Grad Accumulation: {gradient_accumulation_steps}
- GPU: {gpu_count} x {gpu_type}
- Priority: {priority}
"""

    if notes:
        description += f"\n### Notes\n{notes}\n"

    submit_job(
        task_name=f"SFT-{experiment_name}",
        entry_command=entry_command,
        repo_name=REPO_NAME,
        namespace=NAMESPACE,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        job_type=getattr(JobType, priority),
        runner=runner,
        description=description,
    )
