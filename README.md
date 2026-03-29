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
