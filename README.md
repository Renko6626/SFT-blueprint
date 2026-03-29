# SFT Blueprint Scaffold

This repository provides a minimal supervised fine-tuning pipeline designed to
be called by a Magnus blueprint.

## What It Includes

- a stable CLI entrypoint: `train_sft.py`
- JSONL `messages` dataset validation and loading
- model and tokenizer setup based on `transformers`
- LoRA support through `peft`
- SFT training orchestration through `trl`
- a reusable Magnus blueprint template

## Intended Use

This project is meant to serve as a thin, maintainable backend for SFT jobs.
The blueprint layer collects parameters and submits jobs. The actual training
logic stays in the repository code.

## Quick Start

Install dependencies:

```bash
conda activate renko
python -m pip install -r requirements.txt
```

Inspect the training CLI:

```bash
python train_sft.py --help
```

Validate a dataset and config without starting training:

```bash
python train_sft.py \
  --experiment_name demo-run \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --train_path data/qwen_small/train_messages.jsonl \
  --val_path data/qwen_small/val_messages.jsonl \
  --output_dir /tmp/sft-demo \
  --use_lora \
  --dry_run
```

Run the built-in cloud smoke workflow:

```bash
python train_sft.py \
  --test_mode \
  --output_dir /tmp/sft-test \
  --use_lora \
  --dry_run
```

In `test_mode`, the script will:

- use a tiny default test model if no model is provided
- download a tiny Hugging Face test dataset automatically
- write temporary train/val JSONL files under `output_dir/test_mode_data/`
- continue through the normal validation or training flow

## Dataset Format

The current training pipeline expects JSONL data in `messages` format:

```json
{"messages":[
  {"role":"system","content":"You are a helpful assistant."},
  {"role":"user","content":"Explain LoRA simply."},
  {"role":"assistant","content":"LoRA fine-tunes a model by training a small number of extra parameters."}
]}
```

Rules:

- one JSON object per line
- each sample must contain a non-empty `messages` list
- every message must have `role` and `content`
- the last message must be from `assistant`

## Blueprint Integration

Relevant files:

- [blueprints/sft_blueprint_template.py](blueprints/sft_blueprint_template.py)
- [docs/blueprint_usage_guide.md](docs/blueprint_usage_guide.md)
- [docs/blueprint_parameter_design.md](docs/blueprint_parameter_design.md)

Before using the blueprint template, replace these constants with your real
deployment values:

- `REPO_NAME`
- `NAMESPACE`
- `WORKDIR`
- `CONDA_SH`
- `CONDA_ENV`

## Recommended Magnus Entry Command

The recommended Magnus runtime entry sequence for this repository is:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate magnus_shared
cd /magnus/workspace/repository
uv sync --quiet
uv run accelerate launch --num_processes 1 train_sft.py \
  --test_mode \
  --output_dir /tmp/sft-test \
  --use_lora \
  --dry_run
```

For a normal dataset-driven run, the recommended command shape is:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate magnus_shared
cd /magnus/workspace/repository
uv sync --quiet
uv run accelerate launch --num_processes 1 train_sft.py \
  --experiment_name qwen25-05b-sft-v1 \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --train_path /data/project/train_messages.jsonl \
  --val_path /data/project/val_messages.jsonl \
  --output_dir /data/outputs/sft/qwen25-05b-sft-v1 \
  --dataset_format messages \
  --use_lora
```

This repository includes [pyproject.toml](pyproject.toml) so `uv sync --quiet`
can resolve and install the required training dependencies in Magnus.

## Repository Layout

```text
.
├─ train_sft.py
├─ blueprints/
├─ configs/sft/
├─ data/qwen_small/
├─ docs/
├─ src/sft/
└─ tests/
```
