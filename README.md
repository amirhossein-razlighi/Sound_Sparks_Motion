# Sound Sparks Motion

Code for the paper **"Sound Sparks Motion: Audio and Text Tuning for Video Editing"** (SIGGRAPH Asia 2026).

We show that jointly optimizing the text and audio conditioning vectors of a pretrained audio-video diffusion model — supervised by a vision-language model (Qwen2.5-VL) — reliably induces desired motion edits without any fine-tuning of the diffusion model itself.

<body>
<video controls autoplay loop muted playsinline>
  <source src="static/supplementary.mp4" type="video/mp4">
</video>
</body>

---

## Overview

Given a source video and an edit prompt, our method:

1. **Freezes** the LTX-2 video generation model.
2. **Optimizes** text and/or audio conditioning embeddings via gradient descent, using Qwen2.5-VL as a differentiable motion alignment signal.
3. **Applies** the optimized embeddings inside the [Retake](packages/ltx-pipelines/src/ltx_pipelines/retake.py) pipeline to regenerate a target time window of the source video.

The result is a video that exhibits the described motion while preserving the appearance and content of the original.

---

## Repository Structure

```
.
├── editing/                     ← Main experiment code
│   ├── configs/                 ← YAML experiment configs (start here)
│   ├── scripts/
│   │   ├── run.sh               ← Main entry point: run from a YAML config
│   │   ├── sweep.sh             ← Hyperparameter sweep (sequential)
│   │   ├── transfer.sh          ← Transfer optimized latents to a new video
│   │   ├── benchmark_from_config.sh  ← Timing benchmark replay
│   │   ├── setup/               ← Model download helpers
│   │   │   ├── download_models.sh
│   │   │   ├── download_qwen.sh
│   │   │   └── download_clip.sh
│   │   ├── ablations/           ← Ablation study scripts
│   │   │   ├── run_ablation.sh
│   │   │   ├── run_benchmark.sh
│   │   │   ├── run_regularizer_ablation.sh
│   │   │   └── run_scorer_ablation.sh
│   │   ├── utils/               ← Internal helpers (YAML parser, etc.)
│   │   └── legacy/              ← Deprecated scripts (kept for reference)
│   ├── optimize_qwen_vl.py      ← Main optimization entry point
│   ├── optimize_qwen_vl_benchmark.py  ← Timing-only version
│   ├── transfer_optimized.py    ← Transfer latents to a new target video
│   ├── experiments/             ← Ablation/observation Python scripts
│   └── src/
│       └── motion_opt/          ← Core Python library
├── packages/
│   ├── ltx-core/                ← LTX-2 model implementation
│   ├── ltx-pipelines/           ← High-level pipelines (Retake, A2V, etc.)
│   └── ltx-trainer/             ← LoRA / fine-tuning tools
├── visualization/               ← Web-based result viewer
└── gpt_as_a_judge_prompt.txt    ← GPT-based evaluation prompt
```

---

## Setup

### 1. Install dependencies

```bash
git clone <this-repo>
cd <repo-root>
uv sync --frozen
source .venv/bin/activate
pip install pyyaml   # required for YAML config parsing
```

### 2. Download model weights

You need three models: **LTX-2.3**, **Qwen2.5-VL-7B**, and **Gemma-3-12b**.

```bash
# Set destination paths
export CKPT_ROOT=~/checkpoints
export HF_MODELS_ROOT=~/models

bash editing/scripts/setup/download_models.sh
```

Or download each model individually:

```bash
# Qwen2.5-VL (supervision model, ~14 GB bf16)
QWEN_DEST=~/models/Qwen2.5-VL-7B-Instruct \
bash editing/scripts/setup/download_qwen.sh

# CLIP models (for diagnostic similarity scores)
bash editing/scripts/setup/download_clip.sh
```

> **On HPC clusters:** run downloads on the login node (internet access). Compute jobs run with `HF_HUB_OFFLINE=1` once weights are cached.

### 3. Set environment variables

```bash
export CKPT_ROOT=/path/to/checkpoints       # contains ltx-2.3-22b-dev.safetensors
export QWEN_ROOT=/path/to/Qwen2.5-VL-7B-Instruct
export GEMMA_ROOT=/path/to/gemma-3-12b-it-qat-q4_0-unquantized
```

---

## Running Experiments

### Single experiment from a YAML config

```bash
bash editing/scripts/run.sh editing/configs/example_hummingbird.yaml
```

Edit the config to point to your own video and prompts:

```yaml
src_video: "input_videos/my_video.mp4"
edit_prompt: "The bird opens its wings."
static_prompt: "A bird perched on a branch."
opt_mode: "both"        # text | audio | both
experiment_name: "bird_wings"
```

See [editing/configs/](editing/configs/) for annotated examples covering all three modes (`text`, `audio`, `both`).

### Dry-run (no GPU required)

```bash
DRY_RUN=1 bash editing/scripts/run.sh editing/configs/example_hummingbird.yaml
```

### Hyperparameter sweep

```bash
# Edit SRC_VIDEO, EDIT_PROMPT, STATIC_PROMPT in sweep.sh, then:
bash editing/scripts/sweep.sh

# Dry-run to preview all combinations:
DRY_RUN=1 bash editing/scripts/sweep.sh
```

### Transfer optimized latents to a new video

After optimizing on a source video, apply the learned conditioning to a different target:

```bash
TARGET_VIDEO=/path/to/target.mp4 \
OPT_DIR=/path/to/results/QwenVL/my_prompt/my_exp/mode_both \
TRANSFER_MODE=both \
EDIT_PROMPT="A cat yawning" \
STATIC_PROMPT="A cat sitting still" \
bash editing/scripts/transfer.sh
```

### Benchmark timing

Replay an existing run in timing mode to measure per-iteration latency:

```bash
bash editing/scripts/benchmark_from_config.sh \
    /path/to/results/QwenVL/my_prompt/my_exp/run_config.json
```

---

## Configuration Reference

All experiment parameters are documented inline in the sample configs:

| Config | Mode | Description |
|--------|------|-------------|
| [`example_hummingbird.yaml`](editing/configs/example_hummingbird.yaml) | `both` | Joint text+audio optimization |
| [`example_turtle_neck.yaml`](editing/configs/example_turtle_neck.yaml) | `audio` | Audio-only optimization |
| [`example_car_door.yaml`](editing/configs/example_car_door.yaml) | `text` | Text-only optimization |

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `opt_mode` | `both` | What to optimize: `text`, `audio`, or `both` |
| `iterations` | 30 | Optimization steps |
| `lr` | 0.005 | Learning rate |
| `retake_start_frames` | 5 | Conditioning window for Retake pipeline |
| `qwen_max_frames` | 8 | Frames sampled for Qwen supervision |
| `qwen_grad_accum_steps` | 3 | Qwen forward passes averaged per optimizer step |
| `lpips_weight` | 0.0 | LPIPS perceptual regularizer (0 = disabled) |
| `temporal_weight` | 0.0 | Temporal consistency regularizer (0 = disabled) |
| `quantization` | `fp8-cast` | Model quantization (required for low-memory GPU) |

---

## Ablation Studies

Scripts for reproducing ablations from the paper are in `editing/scripts/ablations/`:

```bash
# Regularizer ablation (no_reg / lpips_only / temporal_only / all_reg)
SRC_VIDEO=input_videos/my_video.mp4 \
EDIT_PROMPT="A rose blooming" \
STATIC_PROMPT="A rose bud." \
bash editing/scripts/ablations/run_regularizer_ablation.sh

# Scorer ablation (Qwen vs CLIP vs X-CLIP)
bash editing/scripts/ablations/run_scorer_ablation.sh
```

---

## Hardware Requirements

- **GPU:** NVIDIA H100 or H200 80 GB (required for LTX-22B fp8 + Qwen2.5-VL-7B)
- **VRAM breakdown:** LTX-22B fp8 ~22 GB · Qwen2.5-VL-7B bf16 ~14 GB · activations ~30 GB ≈ 66 GB total
- **Time per run:** ~30 min for 30 iterations (without early stopping, otherwise it will take on average nearly 10~15 mins based on complexity of the task) with `opt_mode=both`

If memory is tight: use `qwen_max_frames: 8` (not higher), reduce `qwen_grad_accum_steps` to 1, or switch to `Qwen/Qwen2.5-VL-3B-Instruct` (~6 GB).

---

## Base Model

This project builds on [LTX-2](https://huggingface.co/Lightricks/LTX-2.3) by Lightricks — a DiT-based audio-video foundation model. The base model code is in `packages/` (unchanged from the original repository).

---

## License

See [LICENSE](LICENSE).
