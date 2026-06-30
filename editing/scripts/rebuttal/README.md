# Rebuttal Priority 1 — audio-init ablation

Answers the cross-reviewer question: **does the gain come from the audio
pathway/prior, or just from extra trainable capacity?** We compare the
source-init method (ours) against **zero** and **random** audio inits. Per the
plan, the audio L2 reg is **dropped** for zero/random (`audio_reg_anchor=none`)
so the control is clean and not silently pulled back toward the source latent.

The diffusion `--seed` stays **42** for every variant, so the baseline render is
identical across variants and they are directly comparable. For `random` we vary
only `--audio-init-seed` (42/1/2), keeping the diffusion noise fixed.

## Variants (per scenario)

| variant      | opt-mode | audio-init | audio-reg-anchor | audio-init-seed | run here? |
|--------------|----------|------------|------------------|-----------------|-----------|
| `source`     | both     | source     | source (reg ON)  | 42              | **no** — = ours, already have it |
| `zero`       | both     | zero       | none (reg OFF)   | —               | yes |
| `random_s42` | both     | random     | none (reg OFF)   | 42              | yes |
| `random_s1`  | both     | random     | none (reg OFF)   | 1               | yes |
| `random_s2`  | both     | random     | none (reg OFF)   | 2               | yes |

`source` is **ours** (source-audio init, reg on, diffusion seed 42) — identical
to the existing main-experiment runs, so it is NOT recomputed. The drivers run
only the 4 zero/random controls (8 scenarios × 4 = **32 cells**). For the `ours`
row in the metrics table, run `eval_metrics.py` on the existing `ours` output
dirs (no re-optimization). To recompute `source` anyway, add it to `VARIANTS`.

Scenarios (8): bugatti_lights_flash, dog_jumping, dog_yawning,
falcon_bird_opening_wings, groom_raising_hand, man_pets_dog, red_car_door_opens,
red_rose_blooming. Configs live in `editing/configs/rebuttal/`.

## How to run (your choice)

You are on a login node — do **not** run these here. Use one of:

**A) SLURM array (parallel across GPUs):**
```bash
# edit the #SBATCH account/partition/time first, ensure CKPT_ROOT/QWEN_ROOT/GEMMA_ROOT
sbatch editing/scripts/rebuttal/ablation.sbatch                 # all 32 control cells
sbatch --array=0-3 editing/scripts/rebuttal/ablation.sbatch     # just the first scenario
```

**B) Sequential, inside an `salloc` GPU shell:**
```bash
bash editing/scripts/rebuttal/run_all_local.sh
# subset:
SCENARIOS="dog_yawning red_rose_blooming" VARIANTS="zero random_s42" \
  bash editing/scripts/rebuttal/run_all_local.sh
```

**C) A single cell (debug / smoke test):**
```bash
bash editing/scripts/rebuttal/run_variant.sh \
    editing/configs/rebuttal/dog_yawning.yaml zero
# preview the exact command without a GPU:
DRY_RUN=1 CKPT_ROOT=/x QWEN_ROOT=/y GEMMA_ROOT=/z \
  bash editing/scripts/rebuttal/run_variant.sh editing/configs/rebuttal/dog_yawning.yaml random_s1
```

## Outputs (per cell)

```
results/rebuttal/<scenario>/<variant>/
  ├── baseline_video.mp4
  ├── mode_both/best_optimized_video_both.mp4   (+ logs, latents, best_params)
  ├── run_config.json                            (full provenance incl. audio_init)
  └── metrics.json                               (critic-independent metrics)
```

`metrics.json` reports motion (RAFT flow + weight-free frame-diff proxy),
source-preservation LPIPS (full + static prefix), temporal flicker, CLIP
edit-alignment gain, and the final Qwen yes-prob (the in-loop critic — reported
for context only, **not** independent evidence).

> Offline nodes: set `HF_HUB_OFFLINE=1` and, for the RAFT motion metric, point
> `RAFT_WEIGHTS=/path/to/raft_large.pth` (else that metric degrades to null and
> the weight-free motion proxy is still reported). Set `RUN_EVAL=0` to skip
> metrics entirely.

## Aggregate (login node, after runs finish)

```bash
python3 editing/scripts/rebuttal/aggregate_metrics.py --results-root results/rebuttal
```
Writes `aggregate_long.csv` (every metric per cell) and `aggregate_summary.md`
(per-group means: source vs zero vs random vs lora — random pools its seeds and
lora pools its ranks, mean±std) — the table to translate into the rebuttal
response. Only groups actually present in the results appear.

## Second ablation — LoRA capacity control

Answers "is the gain just extra trainable parameters?" Trains a **LoRA on the
frozen LTX DiT** with the **same Qwen motion critic, iterations, LR** (Qwen loss
ONLY — no LPIPS/temporal/L2 reg), and **no learnable text/audio latents** — the
only free params are the LoRA weights. If it can't reproduce the edit, raw
capacity is not the explanation. Entry point: `editing/optimize_lora_critic.py`
(additive; reuses all setup/render/loss code, injects PEFT LoRA and monkey-patches
the transformer getter — main code untouched). Gradients reach the LoRA via the
same last-`audio_opt_last_steps` differentiable window (keep
`audio_opt_last_steps < num_inference_steps`).

**Where the LoRA goes — `LORA_PRESET`** (output dir `lora_<preset>_r<rank>/`):
- `audio` (**default, fair control**): LoRA on the audio self/cross + **audio→video**
  attention — free params on exactly the audio-conditioning pathway. Strongest
  apples-to-apples test vs optimizing the audio latent.
- `all`: every attention (video+audio+a2v+v2a) in every block — a **max-capacity
  upper bound** (hundreds of millions of params; changes the whole denoiser).
- `video`: video self + video cross-to-text attention only (`attn1`,`attn2`) — the
  **audio path is untouched**. Control for "free params purely on the video branch".
- `a2v`: only the audio→video attention — the tightest scope.

```bash
# SLURM array (8 scenarios x PRESETS x RANKS; default PRESETS=(audio), RANKS=(64) -> 8 cells)
sbatch editing/scripts/rebuttal/lora.sbatch
sbatch --array=0-0 editing/scripts/rebuttal/lora.sbatch          # first cell / smoke

# one cell directly (salloc GPU shell): the fair audio-pathway control, rank 64
LORA_PRESET=audio bash editing/scripts/rebuttal/run_lora_variant.sh editing/configs/rebuttal/dog_yawning.yaml 64
# the max-capacity upper bound:
LORA_PRESET=all   bash editing/scripts/rebuttal/run_lora_variant.sh editing/configs/rebuttal/dog_yawning.yaml 64
```
To run both presets in the array: `PRESETS=(audio all)` in `lora.sbatch` and
`--array=0-15`. Add ranks via `RANKS=(16 64 128)`. Override modules directly with
`LORA_TARGETS="..."` (labelled `custom`). Knobs: `LORA_ALPHA`, `LORA_DROPOUT`.
Resumable + `metrics.json` per cell. Each preset appears as its own column
(`lora_audio`, `lora_all`, …) in `aggregate_summary.md`, pooling ranks (mean±std).

## Third ablation — video-latent residual (z_vid)

A different control *space*: instead of audio/text, optimize a learnable residual
`delta_z` added to the source **video VAE latent** (`z_vid_used = z_vid + delta_z`,
`delta_z` init 0), updated by the **same Qwen motion critic**, with text+audio
frozen. Tests whether tuning the video latent directly — the most direct handle on
the output — matches/beats tuning the audio-conditioning pathway. Entry point:
`editing/optimize_zvid_residual.py` (additive; reuses all setup/render/loss code,
injects `z_vid + delta_z` as the video latent — main code untouched). Because the
video latent is a grad-carrying *input* to the frozen DiT, gradients reach
`delta_z` through the existing differentiable window with no LoRA-style tricks.

```bash
# SLURM array (8 scenarios -> array 0-7)
sbatch editing/scripts/rebuttal/zvid.sbatch
sbatch --array=0-0 editing/scripts/rebuttal/zvid.sbatch          # first scenario / smoke

# one cell directly (salloc GPU shell):
bash editing/scripts/rebuttal/run_zvid_variant.sh editing/configs/rebuttal/dog_yawning.yaml
# anchored variant (L2 on delta_z; match the audio method's latent_reg_weight):
ZVID_REG=0.01 bash editing/scripts/rebuttal/run_zvid_variant.sh editing/configs/rebuttal/dog_yawning.yaml
```
Output `results/rebuttal/<scenario>/zvid/` with `best_delta_z_zvid.pt`, the rendered
video, and `metrics.json`. `ZVID_REG` (default 0) adds an L2 anchor on `delta_z`
(toward 0 = video latent toward source). Resumable; appears as a `zvid` column in
`aggregate_summary.md`.

## What changed in the main code (additive only)

`--audio-init {source,zero,random}`, `--audio-reg-anchor {source,init,none}`,
`--audio-init-seed N` in `optimize_qwen_vl.py`, implemented in
`multimodal_loop_qwen.py`, emitted by `parse_config.py` only when present in a
config. **Defaults are `source`/`source`/`None` → byte-identical to the original
behaviour, so existing configs and main experiments are unaffected.**
