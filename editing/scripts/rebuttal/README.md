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
(per-group means: source vs zero vs random, with random pooled mean±std over the
3 seeds) — the table to translate into the rebuttal response.

## What changed in the main code (additive only)

`--audio-init {source,zero,random}`, `--audio-reg-anchor {source,init,none}`,
`--audio-init-seed N` in `optimize_qwen_vl.py`, implemented in
`multimodal_loop_qwen.py`, emitted by `parse_config.py` only when present in a
config. **Defaults are `source`/`source`/`None` → byte-identical to the original
behaviour, so existing configs and main experiments are unaffected.**
