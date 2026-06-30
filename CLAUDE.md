# CLAUDE.md

Guidance for working in this repository.

## What this is

**Sound Sparks Motion** — a *training-free* framework for motion-driven video
editing. Given a source video + an edit prompt, it freezes the LTX-2 joint
audio-video diffusion model and optimizes the **text** and/or **audio**
conditioning latents (via gradient descent) so the regenerated clip matches a
target motion. **Qwen2.5-VL** is the differentiable alignment critic:

```
loss = -log P("yes" | video_frames, "Does this video show: {edit_prompt}?")
```

Gradients flow Qwen → differentiable video decoder → the conditioning latent
being tuned. The diffusion weights are never updated.

Paper status: under review at SIGGRAPH Asia 2026 (rebuttal phase). See
[REBUTTAL.md](REBUTTAL.md) for the reviewer-response plan and
[editing/scripts/rebuttal/README.md](editing/scripts/rebuttal/README.md) for the
audio-init ablation built to answer the central reviewer concern.

## Layout

```
editing/
  optimize_qwen_vl.py            # main entry point (argparse → run loop)
  transfer_optimized.py          # apply optimized latents to a new video
  configs/*.yaml                 # one experiment per file (YAML → CLI args)
  configs/rebuttal/*.yaml        # rebuttal audio-init ablation scenarios
  configs/transfer/*.yaml        # transfer experiments
  scripts/
    run.sh                       # run one config (main entry script)
    transfer.sh, sweep.sh
    utils/parse_config.py        # YAML → null-delimited CLI args for run.sh
    eval_metrics.py              # standalone critic-independent metrics → metrics.json
    rebuttal/                    # audio-init ablation runner + SLURM jobs + aggregator
    setup/                       # model download helpers
  src/motion_opt/                # core library
    multimodal_loop_qwen.py      # THE optimization loop (text/audio/both)
    core.py                      # latent encoding, RAFT flow, video IO, audio IO
    qwen_loss.py                 # Qwen2.5-VL yes/no rubric loss (motion/entities/overall)
    clip_loss.py                 # CLIP diagnostics (per-frame static/edit similarity)
    perceptual_loss.py           # LPIPS + temporal consistency regularizers
    models.py, runtime.py, attn_vis.py, data.py, helpers.py
packages/                        # vendored LTX-2 (ltx-core, ltx-pipelines, ltx-trainer)
                                 #   — base model code, treat as unchanged upstream
```

## Running an experiment

Needs a GPU (this is a ~22B fp8 model). Login nodes can only do `DRY_RUN=1`.

```bash
# model roots (the only ones the code reads): CKPT_ROOT must contain ltx-2.3-22b-dev.safetensors
export CKPT_ROOT=/path/to/ltx/checkpoints
export QWEN_ROOT=/path/to/Qwen2.5-VL-7B-Instruct
export GEMMA_ROOT=/path/to/gemma-3-12b-it-qat-q4_0-unquantized

bash editing/scripts/run.sh editing/configs/dog_yawning.yaml      # real run (GPU)
DRY_RUN=1 bash editing/scripts/run.sh editing/configs/dog_yawning.yaml   # print cmd only
```

Config → CLI flow: `run.sh` → `parse_config.py` emits null-delimited args →
`optimize_qwen_vl.py`. Output dir defaults to
`results/QwenVL/<prompt_slug>/<experiment_name>/`, with `baseline_video.mp4`,
`mode_<mode>/best_optimized_video_<mode>.mp4`, CSV logs, saved latents, and
`run_config.json` (full provenance — every argparse value).

Key knobs (config keys = CLI flags): `opt_mode` (text/audio/both), `iterations`,
`lr`, `latent_reg_weight`, `text_reg_weight`, `retake_start_frames`,
`qwen_max_frames`, `qwen_grad_accum_steps`, `lpips_weight`, `temporal_weight`,
`quantization` (`fp8-cast`).

## How the optimization loop works (multimodal_loop_qwen.py)

- `delta_v` — additive residual on the Gemma text embedding (text mode).
- `audio_latent` — the audio conditioning latent, optimized **directly**.
- Both anchored by L2 reg toward the source (text residual ≈ anchored to prompt;
  audio reg ≈ anchored to source latent). Reg can ramp over iterations
  (`reg_schedule`) and LR can cosine-anneal — both to fight late-stage
  adversarial drift of the VLM critic.
- `audio_opt_last_steps` makes only the last N denoising steps differentiable
  (memory). `qwen_grad_accum_steps` averages N Qwen passes per step to cut
  gradient variance from cuBLAS nondeterminism.
- Best checkpoint = lowest total loss; early stopping on no-improvement.

## Rebuttal audio-init ablation (additive feature)

Answers "does the gain come from the audio pathway/prior or just extra
capacity?" via three flags on `optimize_qwen_vl.py` (**defaults reproduce the
original behaviour exactly**):

- `--audio-init {source,zero,random}` — init the audio latent from the encoded
  source audio (default/ours), zeros, or standard-normal noise (no source info).
- `--audio-reg-anchor {source,init,none}` — anchor for the L2 audio reg; `none`
  drops the audio reg term entirely.
- `--audio-init-seed N` — seed for `random` init (defaults to `--seed`); lets the
  init vary while the diffusion seed stays fixed so baselines stay comparable.

Implemented in `multimodal_loop_qwen.py` (init selection + switchable reg
anchor); `parse_config.py` emits these keys **only when present** in a config, so
existing configs are byte-identical.

A second, separate capacity control trains a **LoRA on the frozen DiT** with the
same Qwen critic/losses but no learnable text/audio latents — entry point
`editing/optimize_lora_critic.py` (+ `src/motion_opt/lora_critic.py`), runner
`scripts/rebuttal/run_lora_variant.sh` / `lora.sbatch`. It reuses all
setup/render/loss code and monkey-patches `model_ledger.transformer` to inject
PEFT LoRA — main code untouched.

Run the ablation (8 scenarios × {source, zero, random×3 seeds}):

```bash
source editing/scripts/rebuttal/env.sh          # sets CKPT_ROOT/QWEN_ROOT/GEMMA_ROOT
sbatch editing/scripts/rebuttal/smoke.sbatch     # fast 2-cell plumbing test FIRST
sbatch editing/scripts/rebuttal/ablation.sbatch  # full 40-cell array
# or sequentially in an salloc shell:
bash editing/scripts/rebuttal/run_all_local.sh
# aggregate (login node) → aggregate_long.csv + aggregate_summary.md:
python3 editing/scripts/rebuttal/aggregate_metrics.py --results-root results/rebuttal
```

Each cell also writes `metrics.json` (critic-independent: RAFT motion + weight-free
frame-diff proxy, source-preservation LPIPS, temporal flicker, CLIP edit-alignment
gain; plus the in-loop Qwen yes-prob, labelled as critic, not evidence).

## Conventions / gotchas

- **Don't modify `packages/`** — vendored LTX-2 base model.
- **Keep main code/logic untouched for rebuttal work.** New ablation features are
  additive and default-off; verify originals are unaffected
  (`parse_config.py` on an original config must emit no `--audio-*` flags).
- Configs are the unit of an experiment — prefer a new YAML over editing code.
- `quantization: fp8-cast` is the norm; full bf16 won't fit typical VRAM.
- `qwen_max_frames` must be even; `qwen_img_size` divisible by 28.
- Determinism: seeds set at `run()` start; RAFT/random-init use dedicated
  generators. `Date.now()`-style nondeterminism avoided.
- On offline compute nodes set `HF_HUB_OFFLINE=1`; the CLIP diagnostic
  (`openai/clip-vit-base-patch32`) and RAFT weights must be cached/provided.
- Only three model roots are read by the code: `CKPT_ROOT`, `QWEN_ROOT`,
  `GEMMA_ROOT`. The audio encoder/decoder/vocoder ship inside the LTX checkpoint.
```
