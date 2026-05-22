<div align="center">

<h1>✨ Sound Sparks Motion ✨</h1>
<h3>Audio and Text Tuning for Video Editing</h3>

<div>
  <a href="https://amirhossein-razlighi.github.io" target="_blank">AmirHossein Naghi Razlighi</a><sup>1</sup>&emsp;
  <a href="https://aryanmikaeili.github.io" target="_blank">Aryan Mikaeili</a><sup>2</sup>&emsp;
  <a href="https://www.sfu.ca/~amahdavi" target="_blank">Ali Mahdavi-Amiri</a><sup>2</sup>&emsp;
  <a href="https://danielcohenor.com" target="_blank">Daniel Cohen-Or</a><sup>3</sup>&emsp;
  <a href="https://www.ucy.ac.cy/dir/en/component/comprofiler/userprofile/yiorgos" target="_blank">Yiorgos Chrysanthou</a><sup>1,4</sup>
</div>

<div>
  <sup>1</sup><b>University of Cyprus</b>&emsp;
  <sup>2</sup><b>Simon Fraser University</b>&emsp;
  <sup>3</sup><b>Tel Aviv University</b>&emsp;
  <sup>4</sup><b>CYENS Centre of Excellence</b>
</div>

<br/>

<div>
  <a href="https://arxiv.org/abs/2605.15307"><img src="https://img.shields.io/badge/arXiv-2605.15307-b31b1b?style=flat&logo=arxiv" alt="arXiv"/></a>
  &nbsp;
  <a href="https://amirhossein-razlighi.github.io/Sound_Sparks_Motion/"><img src="https://img.shields.io/badge/Project-Page-blue?style=flat&logo=github" alt="Project Page"/></a>
  &nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat" alt="License"/></a>
  &nbsp;
  <img src="https://img.shields.io/badge/Status-Under%20Review-orange?style=flat" alt="Status"/>
</div>

</div>

---

**Sound Sparks Motion** is a training-free framework for motion-driven video editing. Given a source video and a text prompt, it jointly optimizes the *audio* and *text* conditioning vectors of a frozen audio-video diffusion model (LTX-2), supervised by a vision-language model (Qwen2.5-VL), to produce realistic motion edits — no fine-tuning of the diffusion model required.

<div align="center">

https://github.com/user-attachments/assets/00e68e53-48ab-4bb4-b0c0-e269766d0106

</div>

---

##  Highlights

- **Training-free** — the diffusion model weights are never updated
- **Dual conditioning** — optimizes both text *and* audio latents for richer motion control
- **VLM supervision** — uses Qwen2.5-VL as a differentiable motion alignment signal
- **Transfer** — optimized conditioning can be applied to unseen videos zero-shot

---

##  Overview

![Method Overview](static/teaser.png)

Given a source video and an edit prompt, our method:

1. **Freezes** the LTX-2 video generation model entirely
2. **Optimizes** text and/or audio conditioning embeddings via gradient descent, using Qwen2.5-VL as a reward signal for motion alignment
3. **Applies** the optimized embeddings inside the [Retake](packages/ltx-pipelines/src/ltx_pipelines/retake.py) pipeline to regenerate the target segment of the video

---

##  Repository Structure

```
.
├── editing/
│   ├── configs/                      ← YAML experiment configs (start here)
│   │   └── transfer/                 ← Transfer experiment configs
│   ├── scripts/
│   │   ├── run.sh                    ← Main entry point: run from a YAML config
│   │   ├── transfer.sh               ← Apply optimized latents to a new video
│   │   ├── sweep.sh                  ← Sequential hyperparameter sweep
│   │   └── setup/                    ← Model download helpers
│   │       ├── download_models.sh
│   │       ├── download_qwen.sh
│   │       └── download_clip.sh
│   ├── optimize_qwen_vl.py           ← Main optimization script
│   ├── transfer_optimized.py         ← Transfer script
│   └── src/motion_opt/               ← Core Python library
├── packages/
│   ├── ltx-core/                     ← LTX-2 model implementation
│   ├── ltx-pipelines/                ← High-level pipelines (Retake, A2V, etc.)
│   └── ltx-trainer/                  ← LoRA / fine-tuning tools
└── gpt_as_a_judge_prompt.txt         ← GPT-based evaluation prompt
```

---

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/amirhossein-razlighi/Sound_Sparks_Motion.git
cd Sound_Sparks_Motion
uv sync --frozen
source .venv/bin/activate
pip install pyyaml
```

### 2. Download Model Weights

> [!NOTE]
> You need three models: **LTX-2.3** (~22B fp8), **Qwen2.5-VL-7B**, and **Gemma-3-12b**.
> On HPC clusters, run downloads on the login node (internet access) and set `HF_HUB_OFFLINE=1` on compute nodes.

```bash
export CKPT_ROOT=~/checkpoints
export HF_MODELS_ROOT=~/models

bash editing/scripts/setup/download_models.sh
```

Or download models individually:

```bash
# Qwen2.5-VL supervision model (~14 GB bf16)
QWEN_DEST=~/models/Qwen2.5-VL-7B-Instruct \
bash editing/scripts/setup/download_qwen.sh

# CLIP models (diagnostic similarity scores)
bash editing/scripts/setup/download_clip.sh
```

### 3. Set Environment Variables

```bash
export CKPT_ROOT=/path/to/checkpoints          # contains ltx-2.3-22b-dev.safetensors
export QWEN_ROOT=/path/to/Qwen2.5-VL-7B-Instruct
export GEMMA_ROOT=/path/to/gemma-3-12b-it-qat-q4_0-unquantized
```

---

##  Usage

### Single Experiment from a YAML Config

```bash
bash editing/scripts/run.sh editing/configs/dog_yawning.yaml
```

Or use this config template for your own videos and prompts:
```bash
bash editing/scripts/run.sh editing/configs/yours.yaml
```

Edit the config to point to your own video and prompts. for example:

```yaml
src_video: "input_videos/dog_video.mp4"
edit_prompt: "A dog yawning"
static_prompt: "A dog sitting on a chair"
opt_mode: "both"
experiment_name: "dog_yawning"
```

Also you can customize the tuning hyperparameters (learning rate, number of iterations, regularization weights, etc.) in the same YAML file.

See [`editing/configs/`](editing/configs/) for annotated examples covering all experiments.

### Transfer Optimized Latents to a New Video

![Transfer Overview](static/transfer_with_extra2.jpg)

After optimizing on a source video, apply the learned conditioning to a different target. Use the provided config template:

```bash
bash editing/scripts/transfer.sh editing/configs/transfer/yours.yaml
```

Fill in `yours.yaml` with your paths and prompts:

```yaml
target_video: "input_videos/my_new_video.mp4"
opt_dir: "results/QwenVL/a_dog_yawning/.../mode_both"
mode: "both"              # text | audio | both — must match what was optimized

edit_prompt: "A cat yawning."
static_prompt: "A cat sitting on a chair."
experiment_name: "dog_to_cat_yawning"
```

See [`editing/configs/transfer/`](editing/configs/transfer/) for ready-made examples.

> [!TIP]
> You can also pass env vars directly for quick one-off transfers:
> ```bash
> TARGET_VIDEO=/path/to/target.mp4 \
> OPT_DIR=/path/to/results/.../mode_both \
> TRANSFER_MODE=both \
> EDIT_PROMPT="A cat yawning" \
> STATIC_PROMPT="A cat sitting still" \
> bash editing/scripts/transfer.sh
> ```

### Hyperparameter Sweep

```bash
# Edit SRC_VIDEO, EDIT_PROMPT, STATIC_PROMPT inside sweep.sh, then:
bash editing/scripts/sweep.sh

# Preview all combinations without running:
DRY_RUN=1 bash editing/scripts/sweep.sh
```

---

##  Configuration Reference

Key parameters:

| Parameter | Example Value | Description |
|-----------|---------|-------------|
| `opt_mode` | `both` | What to optimize: `text`, `audio`, or `both` |
| `iterations` | `30` | Optimization steps |
| `lr` | `0.005` | Learning rate |
| `retake_start_frames` | `10` | Conditioning window for the Retake pipeline |
| `qwen_max_frames` | `8` | Frames sampled per Qwen supervision call |
| `qwen_grad_accum_steps` | `3` | Qwen forward passes averaged per optimizer step |
| `lpips_weight` | `0.0` | LPIPS perceptual regularizer (0 = disabled) |
| `temporal_weight` | `0.0` | Temporal consistency regularizer (0 = disabled) |
| `quantization` | `fp8-cast` | Model quantization |

---

##  Hardware Requirements

**Runtime:** ~10-20 min for 30 iterations with `opt_mode=both`with early stopping activated.

> [!TIP]
> If VRAM is tight: set `qwen_max_frames: 8`, reduce `qwen_grad_accum_steps` to `1`, or switch to `Qwen/Qwen2.5-VL-3B-Instruct` (~6 GB). Note that these may reduce the final quality of motion edits.

---

##  Base Model

This project builds on [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) by Lightricks — a DiT-based audio-video foundation model. The base model code lives in `packages/` and is unchanged from the original repository.

---

## TODO

- [x] Release code
- [x] arXiv preprint
- [x] Release project page
- [ ] Release pretrained optimized latents for paper examples
- [ ] Add Gradio demo

---

##  Citation

If you find this work useful, please cite:

```bibtex
@article{razlighi2026sound,
  title={Sound Sparks Motion: Audio and Text Tuning for Video Editing},
  author={Razlighi, AmirHossein Naghi and Mikaeili, Aryan and Mahdavi-Amiri, Ali and Cohen-Or, Daniel and Chrysanthou, Yiorgos},
  journal={arXiv preprint arXiv:2605.15307},
  year={2026}
}
```

---

## License

See [LICENSE](LICENSE).
