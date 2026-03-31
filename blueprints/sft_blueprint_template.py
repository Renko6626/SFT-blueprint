REPO_NAME = "your-repo-name"
NAMESPACE = "your-github-org"
WORKDIR = "/magnus/workspace/repository"
CONDA_SH = "/opt/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV = "magnus_shared"

ACCELERATE_CONFIGS = {
    "ddp":   None,
    "zero2": "configs/accelerate/deepspeed_zero2.yaml",
    "zero3": "configs/accelerate/deepspeed_zero3.yaml",
}


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

WarmupRatio = Annotated[float, {
    "label": "Warmup Ratio",
    "description": "Fraction of total training steps used for LR warmup. Prevents loss spikes at the start of training.",
    "scope": "Optimization",
    "min": 0.0,
    "max": 0.5,
}]

WeightDecay = Annotated[float, {
    "label": "Weight Decay",
    "description": "AdamW weight decay coefficient.",
    "scope": "Optimization",
    "min": 0.0,
    "max": 0.1,
}]

LrSchedulerType = Annotated[Literal["cosine", "linear"], {
    "label": "LR Scheduler",
    "description": "Learning rate decay schedule.",
    "scope": "Optimization",
    "options": {
        "cosine": {"label": "Cosine", "description": "Smooth cosine decay. Recommended for most SFT runs."},
        "linear": {"label": "Linear", "description": "Linear decay to zero."},
    },
}]

Packing = Annotated[bool, {
    "label": "Sequence Packing",
    "description": "Pack multiple short sequences into one max_seq_length sequence to maximize GPU utilization. Incompatible with Response-Only Loss.",
    "scope": "Optimization",
}]

GradientCheckpointing = Annotated[bool, {
    "label": "Gradient Checkpointing",
    "description": "Enable gradient checkpointing to reduce GPU memory usage at the cost of slower training.",
    "scope": "Optimization",
}]

SaveTotalLimit = Annotated[int, {
    "label": "Save Total Limit",
    "description": "Maximum number of checkpoints to keep on disk.",
    "scope": "Optimization",
    "min": 1,
    "max": 20,
}]

ResponseOnly = Annotated[bool, {
    "label": "Response-Only Loss",
    "description": "Only compute loss on assistant responses, masking system/user prompt tokens. Recommended for instruction fine-tuning.",
    "scope": "Model",
}]

FlashAttention = Annotated[bool, {
    "label": "Flash Attention 2",
    "description": "Use Flash Attention 2 for faster attention computation. Requires flash-attn package. Recommended for A100+.",
    "scope": "Model",
}]

AcceleratePreset = Annotated[Literal["ddp", "zero2", "zero3"], {
    "label": "Distributed Strategy",
    "description": "Distributed training strategy for multi-GPU jobs.",
    "scope": "Cluster",
    "options": {
        "ddp":   {"label": "DDP",    "description": "Data parallel, no sharding. Simplest but highest per-GPU memory."},
        "zero2": {"label": "ZeRO-2", "description": "Shards optimizer states across GPUs. Recommended for full fine-tuning."},
        "zero3": {"label": "ZeRO-3", "description": "Shards weights + gradients + optimizer states. Maximum memory efficiency."},
    },
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

TestMode = Annotated[bool, {
    "label": "Test Mode",
    "description": "Run the built-in HF smoke workflow instead of normal dataset-driven training.",
    "scope": "Misc",
}]


def blueprint(
    runner: Runner,
    experiment_name: ExperimentName,
    train_path: TrainPath,
    output_root: OutputRoot,
    val_path: ValPath = None,
    base_model: BaseModel = "Qwen/Qwen2.5-0.5B-Instruct",
    finetune_method: FinetuneMethod = "lora",
    response_only: ResponseOnly = True,
    flash_attention: FlashAttention = True,
    max_seq_len: MaxSeqLen = 4096,
    epochs: Epochs = 3,
    learning_rate: LearningRate = 2e-5,
    warmup_ratio: WarmupRatio = 0.03,
    weight_decay: WeightDecay = 0.01,
    lr_scheduler_type: LrSchedulerType = "cosine",
    per_device_batch_size: PerDeviceBatchSize = 1,
    gradient_accumulation_steps: GradAccumSteps = 16,
    packing: Packing = False,
    gradient_checkpointing: GradientCheckpointing = False,
    save_total_limit: SaveTotalLimit = 3,
    accelerate_preset: AcceleratePreset = "ddp",
    gpu_count: GpuCount = 1,
    gpu_type: GpuType = "rtx5090",
    priority: Priority = "A2",
    test_mode: TestMode = False,
    notes: Notes = None,
    extra_args: ExtraArgs = None,
):
    def shq(value):
        return "'" + str(value).replace("'", "'\"'\"'") + "'"

    output_dir = f"{output_root.rstrip('/')}/{experiment_name}"

    accel_config = ACCELERATE_CONFIGS[accelerate_preset]
    launch_prefix = "uv run accelerate launch"
    if accel_config:
        launch_prefix += f" --config_file {shq(accel_config)}"
    launch_prefix += f" --num_processes {gpu_count}"

    train_cmd = (
        launch_prefix
        + " train_sft.py"
        + f" --experiment_name {shq(experiment_name)}"
        + f" --output_dir {shq(output_dir)}"
        + " --dataset_format messages"
        + f" --max_seq_length {max_seq_len}"
        + f" --per_device_train_batch_size {per_device_batch_size}"
        + f" --per_device_eval_batch_size {per_device_batch_size}"
        + f" --gradient_accumulation_steps {gradient_accumulation_steps}"
        + f" --learning_rate {learning_rate}"
        + f" --warmup_ratio {warmup_ratio}"
        + f" --weight_decay {weight_decay}"
        + f" --lr_scheduler_type {lr_scheduler_type}"
        + f" --num_train_epochs {epochs}"
        + f" --save_total_limit {save_total_limit}"
        + " --logging_steps 10"
        + " --save_steps 200"
        + " --eval_steps 200"
    )

    if packing:
        train_cmd += " --packing"
    if gradient_checkpointing:
        train_cmd += " --gradient_checkpointing"
    if response_only:
        train_cmd += " --response_only"
    if flash_attention:
        train_cmd += " --flash_attention"

    if test_mode:
        train_cmd += " --test_mode"
        train_cmd += f" --test_model_name_or_path {shq(base_model)}"
    else:
        train_cmd += f" --model_name_or_path {shq(base_model)}"
        train_cmd += f" --train_path {shq(train_path)}"
        if val_path is not None:
            train_cmd += f" --val_path {shq(val_path)}"

    if finetune_method == "lora":
        train_cmd += " --use_lora"

    if extra_args:
        train_cmd += f" {extra_args}"

    entry_command = "\n".join([
        f"source {shq(CONDA_SH)}",
        f"conda activate {shq(CONDA_ENV)}",
        f"cd {shq(WORKDIR)}",
        "uv sync --quiet",
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
- Test Mode: {test_mode}
- Max Seq Len: {max_seq_len}
- Epochs: {epochs}
- Learning Rate: {learning_rate}
- Warmup Ratio: {warmup_ratio}
- Weight Decay: {weight_decay}
- LR Scheduler: {lr_scheduler_type}
- Batch Size / Device: {per_device_batch_size}
- Grad Accumulation: {gradient_accumulation_steps}
- Packing: {packing}
- Gradient Checkpointing: {gradient_checkpointing}
- Save Total Limit: {save_total_limit}
- Response-Only Loss: {response_only}
- Flash Attention 2: {flash_attention}
- Distributed Strategy: {accelerate_preset}
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
