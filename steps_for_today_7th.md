# Steps for Today — April 7th

## Overview

We're close to the SigAsia deadline. All three tasks below push toward a stronger, cleaner paper.
The QWEN-based loss/optimization is our best-performing setup and is the focus for all experiments.

---

## Task 1: Visualize Attention Maps (Before & During Optimization)

**Goal:** Show *where in the video* the Qwen2.5-VL model is attending while it scores our
edits. This is a scientifically compelling visualization and directly supports the paper's
core claim that our optimization is grounded in semantically meaningful video regions.

### What to implement

1. **Hook into Qwen's cross-attention layers** during the `compute_qwen_video_loss()` call
   (in `editing/src/audio_latent_opt/qwen_loss.py`).
   - Register forward hooks on `Qwen2VLDecoderLayer` attention modules (specifically the
     cross-attention or self-attention layers attending over video token positions).
   - Aggregate attention weights for video tokens vs. text tokens.

2. **Capture at two stages:**
   - **Before optimization** (iteration 0, on the baseline video) — where is Qwen looking
     in the unedited clip?
   - **During optimization** (every `visualize_every_iters` steps, on the current best video)
     — how does the attended region shift as we push `P(yes)` higher?

3. **Visualization format:**
   - Reshape attention weights back to spatial (H×W) and temporal (T) dimensions using the
     `video_grid_thw` grid shape (from `build_qwen_inputs()`).
   - Overlay the attention heatmap on the decoded video frames as a semi-transparent colormap
     (e.g., `plasma` or `inferno`).
   - Save as an MP4 side-by-side: `[original frame | attention overlay]`.

4. **Log to W&B:**
   - Log the before-optimization attention video as `"attn_baseline"`.
   - Log per-iteration snapshots as `"attn_iter_{i}"` (or a W&B media panel with steps).
   - Also log the attention heatmap as a static image at the peak `yes_prob` step for
     easy visual comparison in the W&B dashboard.

### Why this matters for the paper

Reviewers will ask "what is the model actually optimizing?" — showing that attention
concentrates on the correct semantic region (the dog's mouth when editing "yawning",
the car door when editing "door opens") makes the method legible and trustworthy.

---

## Task 2: Find More "Perfect" Cases

**Goal:** Build a curated gallery of experiments with high `yes_prob` (> 0.95), minimum
visual artifacts, and visually obvious edits. These are the hero examples for the paper.

### Already confirmed "perfect" cases

| Experiment | Source prompt | Edit prompt | Best yes_prob | Notes |
|---|---|---|---|---|
| a_dog_yawning | A dog | A dog yawning | **0.9987** | Best result so far; near-perfect |
| a_dog_yawns | A dog | A dog yawns | ~0.867 | Slightly different source |
| a_guitarist_spins_360 | Guitarist playing | Guitarist spins 360 | — | Ours slightly better than baseline |
| a_red_cars_door_opens | Red car | A red car's door opens | — | Clean example per earlier runs |

### New candidates to run (high probability of success)

These share the properties of our best cases: single subject, clear isolated motion,
high temporal contrast, unambiguous semantic change from source to target.

| Priority | Source video prompt (generate/find) | Edit prompt for Qwen | Why it should work |
|---|---|---|---|
| ★★★ | A cat sitting still | A cat yawning | Same motion type as dog yawning; cross-species transfer test |
| ★★★ | A dog sitting | A dog shaking its head | Clear head motion, high temporal contrast |
| ★★★ | A person standing | A person sneezing | Clear, fast, isolated head-body motion |
| ★★★ | A balloon floating | A balloon pops | Instantaneous, unambiguous event |
| ★★★ | A wine bottle on a table | A wine bottle drops and shatters | Already in results — verify best hparams |
| ★★ | A bird perched | A bird spreads its wings | Clear wing extension motion |
| ★★ | A person sitting | A person stands up | Pose change with clear trajectory |
| ★★ | A flower bud | A flower blooms | Gradual clear outward motion |
| ★★ | A person's hand at rest | A person claps their hands | Symmetric, fast, distinct motion |
| ★ | A dog lying down | A dog jumps up | Larger motion, may cause artifacts |

### Selection criteria for "perfect" label

A case is "perfect" if:
- `best_yes_prob > 0.95` in the Qwen log
- The optimized video is visually plausible (no major hallucinations or warping)
- The edited region is correct (e.g., the dog's mouth moves, not background)
- Attention maps (Task 1) concentrate on the semantically correct region

### W&B logging for comparison

Log each experiment to the same W&B project (`ltx-qwen-opt`) with tags:
- `"perfect-case"` if it meets the criteria above
- `"hero-example"` for the top 3–4 paper figures
- Log baseline video, best optimized video, and attention map side-by-side
- Use W&B `Table` to log a comparison table: `[experiment, mode, yes_prob, video_url]`

---

## Task 3: Can We Edit Geometry?

**Goal:** Test whether our optimization can handle *geometric / structural* changes —
not just texture or action changes, but changes to the shape or topology of objects in the scene.

### What "geometry editing" means

- Changing an object's shape or pose: tie → bow tie, open hand → fist, sitting → standing
- Structural deformation: balloon deflated → balloon inflated
- Articulated pose change: arm down → arm raised

This is a harder test than appearance/action editing because it requires the model to
hallucinate new geometry rather than amplify existing motion patterns.

### Candidate experiments

| Priority | Source prompt | Edit prompt | Geometry change type |
|---|---|---|---|
| ★★★ | A man in a suit adjusting his tie | A man in a suit with a bow tie | Object shape replacement (tie topology) |
| ★★★ | A person sitting in a chair | A person standing up from a chair | Full-body pose change |
| ★★★ | A person with a clenched fist | A person with an open palm raised | Hand geometry + arm pose |
| ★★★ | A closed door | A door swinging open | Already close to our red car door case |
| ★★ | A cat with its tail down | A cat with its tail raised high | Appendage pose change |
| ★★ | A person with arms at their sides | A person with arms stretched wide | Symmetric limb geometry |
| ★★ | A deflated balloon | A fully inflated balloon floating | Volume/topology change |
| ★ | A person with straight hair | A person with curly hair | Fine-grained texture+shape |

### What to measure and report

Beyond `yes_prob`, for geometry editing we should also check:
- **Structure preservation** (do background and non-edited regions stay consistent?)
- **Geometric plausibility** (does the new shape look physically realistic?)
- **Compare audio-only vs text-only vs both** — geometry may be harder for audio-only

### Hypothesis to test

Geometry edits (e.g., tie → bow tie) likely require larger text perturbations and may be
harder for the audio branch alone. If `text-only` or `both` significantly outperforms
`audio-only` on geometry tasks, that is a meaningful scientific finding worth including.

### W&B tags for this track

Tag these runs `"geometry-edit"` in W&B and create a separate panel group for side-by-side
comparison. Log a baseline frame, the best-optimized frame, and the attention map.

---

## W&B Logging Improvements (Applies to All Tasks)

### What to add to `optimize_qwen_vl.py` and `multimodal_loop_qwen.py`

1. **Attention map videos** — as described in Task 1
2. **Intermediate video frames** at every visualization step (not just final video):
   - Log as `wandb.Video` with step number so the W&B media panel shows the evolution
3. **Per-iteration comparison table** (W&B `Table`):
   - Columns: `iter, yes_prob, audio_reg, text_reg, total_loss, is_best`
   - Allows sorting/filtering in the W&B UI directly
4. **Loss curve with dual axes**: plot `yes_prob` (0–1) and `total_loss` on the same chart
5. **Baseline vs best video side-by-side** in W&B Summary:
   - `wandb.log({"baseline_video": wandb.Video(...), "best_video": wandb.Video(...)}, step=0)`
   - Confirmed working in existing code — just ensure it runs even when `SAVE_FINAL_VIDEOS=0`
6. **Config tags for easy filtering**: add `geometry-edit`, `perfect-case`, `audio-only` etc.
   as W&B run tags so we can group experiments in the dashboard

---

## Hyperparameter Search

See `editing/scripts/submit_hparam_sweep_qwen.sh` for a SLURM array job that sweeps
the most impactful hyperparameters on the `a_dog_yawning` experiment (our best case):

- `LR` ∈ {0.001, 0.003, 0.005, 0.01}
- `LATENT_REG_WEIGHT` ∈ {0.001, 0.01, 0.1}
- `AUD_OPT_LAST_STEPS` ∈ {4, 8, 12, 16}

Total: 48 array jobs, each ~2h on H100. All log to `ltx-qwen-opt` W&B project with tag
`"hparam-sweep"` so results are easy to compare in the Parallel Coordinates plot.

Once we find optimal hparams on dog yawning, re-run top-3 configs on the other priority
experiments from Tasks 2 and 3.
