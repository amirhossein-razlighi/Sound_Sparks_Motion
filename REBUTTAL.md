# REBUTTAL — Sound Sparks Motion (SIGGRAPH Asia 2026)

**Status:** 3 borderline-reject, 1 borderline-accept. The idea is liked; the
experiments are not yet seen to *isolate the role of audio conditioning*, the
silent demos confuse people, and runtime / video discoverability / single-backbone
need clarifying. This file is the master action list: what to **run**, **ablate**,
**measure**, and **write**. SIGGRAPH Asia rebuttal is **text-only** — no images,
videos, or links. Everything below must collapse into numbers + prose we can paste
into the response box, plus camera-ready promises.

---

## 0. The one message everything serves

> In a **frozen** joint audio-video generator (LTX-2 Retake), the **audio-conditioning
> branch provides a temporally structured latent control space** that can be optimized
> for **visual motion editing**. Audio is an *internal control variable*, not an output.
> Our contribution is this control mechanism — not audio-semantic editing and not
> audio-video output editing.

Two wording rules for the whole rebuttal:
- **Never** say "the audio latent captures audio semantics."
- **Prefer** "the audio-conditioning pathway provides a temporally structured optimization space."

If the new controls (Section 1) show source-init beats random/zero → claim the
temporal *prior* helps. If they're close → claim the *pathway/capacity* is what helps
and revise the wording. **Either outcome is publishable if framed honestly.** Decide
the wording only after the numbers land.

---

## 1. PRIORITY 1 — Audio-specificity / capacity control (THE decisive experiment)

This single experiment is requested by **all four reviewers** (R1 capacity-matched
control, R2 random/zero init, R4 audio-specificity). It is the difference between
accept and reject. Do this first.

> **STATUS: implemented and ready to run (GPU).** Code, configs, runner, SLURM
> array, eval, and aggregator are all built. See
> [`editing/scripts/rebuttal/README.md`](editing/scripts/rebuttal/README.md).
> Decision taken (per user): for zero/random init the **audio L2 reg is dropped**
> (`--audio-reg-anchor none`); the **diffusion seed stays 42** for every variant so
> baselines are identical; `random` varies only `--audio-init-seed` (42/1/2).
> Artifacts:
> - Code (additive, defaults reproduce original): `--audio-init`,
>   `--audio-reg-anchor`, `--audio-init-seed` in
>   [`optimize_qwen_vl.py`](editing/optimize_qwen_vl.py),
>   [`multimodal_loop_qwen.py`](editing/src/motion_opt/multimodal_loop_qwen.py),
>   [`parse_config.py`](editing/scripts/utils/parse_config.py).
> - Configs: [`editing/configs/rebuttal/`](editing/configs/rebuttal/) (8 scenarios).
> - Run: [`run_variant.sh`](editing/scripts/rebuttal/run_variant.sh) /
>   [`run_all_local.sh`](editing/scripts/rebuttal/run_all_local.sh) (salloc) /
>   [`ablation.sbatch`](editing/scripts/rebuttal/ablation.sbatch) (SLURM array, 40 cells).
> - Metrics: [`eval_metrics.py`](editing/scripts/eval_metrics.py) →
>   per-cell `metrics.json`; aggregate with
>   [`aggregate_metrics.py`](editing/scripts/rebuttal/aggregate_metrics.py).
> - Scenarios run here (user-selected): bugatti_lights_flash, dog_jumping,
>   dog_yawning, falcon_bird_opening_wings, groom_raising_hand, man_pets_dog,
>   red_car_door_opens, red_rose_blooming. Variants: source / zero / random×3 seeds.

### 1.1 Why code changes are needed

Audio init is currently hardcoded to the source-encoded audio latent:

- [`editing/src/motion_opt/multimodal_loop_qwen.py:125-128`](editing/src/motion_opt/multimodal_loop_qwen.py#L125-L128)
  ```python
  if optimize_audio:
      audio_latent = torch.nn.Parameter(base_audio_latent_fp32.clone())
  ```
- The L2 regularizer anchors to the **source** latent
  ([`multimodal_loop_qwen.py:390-393`](editing/src/motion_opt/multimodal_loop_qwen.py#L390-L393)):
  ```python
  audio_reg_t = audio_reg_w * torch.mean((audio_latent - base_audio_latent_fp32) ** 2)
  ```

A zero/random init that is *still regularized toward the source latent* is not a
clean control — the reg term would silently drag it back to the source, confounding
"does the audio prior help" with "does the reg pull it home." We must make both the
**init** and the **reg anchor** selectable.

### 1.2 Exact code change (add two flags)

**(a)** `editing/optimize_qwen_vl.py` — add to `build_parser()`:
```python
p.add_argument("--audio-init", default="source",
               choices=["source", "zero", "random"],
               help="Init for the optimized audio latent. source=encoded source "
                    "audio (default/ours); zero/random = capacity-matched controls.")
p.add_argument("--audio-reg-anchor", default="source",
               choices=["source", "init", "none"],
               help="Anchor for the L2 audio reg. 'source'=ours; 'init'=anchor to "
                    "the init point (symmetric control); 'none'=disable.")
```

**(b)** `editing/src/motion_opt/multimodal_loop_qwen.py` — replace the init block
(lines 125-128):
```python
if optimize_audio:
    audio_init = getattr(args, "audio_init", "source")
    if audio_init == "source":
        init_tensor = base_audio_latent_fp32.clone()
    elif audio_init == "zero":
        init_tensor = torch.zeros_like(base_audio_latent_fp32)
    elif audio_init == "random":
        # Standard-normal N(0,1), same shape as source. Uses NO source info
        # (not even its scale) — a fully source-free control.
        init_tensor = torch.randn_like(base_audio_latent_fp32)
    else:
        raise ValueError(f"Unknown audio_init: {audio_init!r}")
    audio_latent = torch.nn.Parameter(init_tensor)
    # Anchor used by the L2 regularizer below.
    audio_reg_anchor = base_audio_latent_fp32 if getattr(args, "audio_reg_anchor", "source") == "source" \
        else init_tensor.detach().clone()
    params.append(audio_latent)
    log.info("[%s] Audio latent init=%s reg_anchor=%s shape=%s, %.1fK params",
             mode, audio_init, getattr(args, "audio_reg_anchor", "source"),
             tuple(audio_latent.shape), audio_latent.numel() / 1e3)
```
And update the reg term (lines 390-393) to use the anchor + honor `none`:
```python
if optimize_audio and audio_latent is not None and audio_reg_w > 0 \
        and getattr(args, "audio_reg_anchor", "source") != "none":
    audio_reg_t = audio_reg_w * torch.mean((audio_latent - audio_reg_anchor) ** 2)
```

**(c)** `editing/scripts/utils/parse_config.py` — emit the new keys (near the
optimisation block, ~line 147):
```python
e("--audio-init",        "audio_init",        "source")
e("--audio-reg-anchor",  "audio_reg_anchor",  "source")
```

> **Important:** the seed is fixed (`--seed 42`, `torch.manual_seed`), so the
> `random` init is reproducible. Run random init with **3 seeds** (42, 1, 2) per
> scenario and report mean±std so a reviewer can't dismiss it as one lucky/unlucky draw.

### 1.3 The variant grid (per scenario)

Hold everything identical (optimizer, LR, iterations, losses, frames, reg weight)
and change only init + anchor:

| # | Variant | `opt_mode` | `audio_init` | `audio_reg_anchor` | Purpose |
|---|---------|-----------|--------------|--------------------|---------|
| 1 | Text only | `text` | — | — | text-only floor |
| 2 | Audio only (ours) | `audio` | `source` | `source` | isolate audio pathway |
| 3 | Both, regularized (ours, main) | `both` | `source` | `source` | full method |
| 4 | Both, **unregularized** | `both` | `source` | `none` | role of reg |
| 5 | **Zero-init** control | `both` | `zero` | `init` | capacity, no audio prior |
| 6 | **Random-init** control ×3 seeds | `both` | `random` | `init` | capacity+scale, no prior |
| 7 | (optional) Random-init, reg→source | `both` | `random` | `source` | shows reg-pull effect |

Variant 7 is the "is it just the reg dragging it home?" check — keep it optional/time-permitting.

### 1.4 Scenarios (run on 8; report all, spotlight 5 in text)

Pick a spread across motion types so a reviewer can't say "only easy cases." Configs
already exist in [`editing/configs/`](editing/configs/):

- **Articulated/organic:** `dog_yawning`, `falcon_bird_opening_wings`, `turtle_neck`, `monkey_reaching_fruit`
- **Human:** `groom_laughing` (or `groom_surprised`), `man_climbs_rocks`
- **Rigid/mechanical:** `red_car_door_opens` (or `ferrari_door_opens`), `robot_waving`
- **Hard / known-artifact:** `goldfish_jumping` (R2 explicitly cited the goldfish artifact — include it and own it)

8 scenarios × 7 variants (random counted once for the grid, ×3 seeds adds 2 extra
runs/scenario) ≈ **~70 runs**. At ~10–20 min/run that's the bulk of the compute
budget — start immediately and run in parallel across GPUs.

### 1.5 Driver script (to add: `editing/scripts/ablation_audio_init.sh`)

Loop over scenarios × variants by overriding env/config. Each variant writes to a
distinct `OUTPUT_DIR` so nothing collides, e.g.
`results/ablation/<scenario>/<variant>/`. Use `DRY_RUN=1` first to verify the
expanded commands. Pattern:
```bash
for cfg in dog_yawning falcon_bird_opening_wings turtle_neck monkey_reaching_fruit \
           groom_laughing man_climbs_rocks red_car_door_opens goldfish_jumping; do
  # variant 3 (ours) — uses config defaults
  OUTPUT_DIR=results/ablation/$cfg/both_source \
    bash editing/scripts/run.sh editing/configs/$cfg.yaml
  # variant 5 (zero) and 6 (random) via a tiny override config that sets
  # opt_mode: both, audio_init: zero|random, audio_reg_anchor: init
  ...
done
```
(Simplest: create `editing/configs/ablation/<scenario>_{zero,random}.yaml` that
`include` the base values and override the three keys; or pass overrides through
env. Keep `seed`, `iterations`, `lr`, `latent_reg_weight` identical to the main config.)

---

## 2. PRIORITY 2 — Metrics that DON'T rely on the Qwen critic (kills "VLM bias")

R4 (and implicitly R2) worry that optimizing with Qwen and evaluating with a VLM is
circular. The repo already has **objective, critic-independent** signals — use them
as the ablation table's spine so no number in the decisive table comes from Qwen.

### 2.1 Objective metric battery (build `editing/scripts/eval_metrics.py`)

A standalone script that, given `baseline_video.mp4` and each variant's
`best_optimized_video_<mode>.mp4`, computes:

| Metric | How (already in repo) | Maps to reviewer term |
|--------|----------------------|-----------------------|
| **Motion strength** | RAFT flow magnitude, mean per-frame ‖flow‖ — `compute_raft_flows`, `load_raft_components` ([core.py:272-329](editing/src/motion_opt/core.py#L272-L329)) | motion strength |
| **Motion naturalness / flicker** | Temporal warp/consistency error — `compute_perceptual_quality_loss` temporal term ([perceptual_loss.py](editing/src/motion_opt/perceptual_loss.py)) | motion naturalness |
| **Source preservation** | LPIPS(optimized, baseline) on the *static prefix* + background — `compute_perceptual_quality_loss` LPIPS term | source preservation |
| **Edit alignment** | CLIP per-frame Δ = sim(edit) − sim(static), already computed as a diagnostic — `compute_clip_dual_prompt_frame_similarities` ([clip_loss.py](editing/src/motion_opt/clip_loss.py)) | edit alignment |
| **Edit locality / no magical insertion** | LPIPS restricted to non-motion regions (low-flow mask from RAFT) vs baseline | edit locality |
| **Visual quality** | Optional no-reference (e.g. CLIP-IQA or a small NR-IQA); or fold into human study | visual quality |

These are diffusion-/critic-agnostic. The **expected story**: ours (source init) and
random/zero may reach *similar edit-alignment* (the optimizer can always satisfy
Qwen), but **source init should win on source preservation, locality, and
naturalness at equal alignment** — i.e., the audio prior buys a better-conditioned
starting point, not just capacity. If even those tie, we honestly pivot to "the
pathway/capacity is the mechanism" (Section 0).

### 2.2 GPT eval = auxiliary, Human study = primary (state explicitly)

- **Qwen2.5-VL** = optimization critic (in the loop).
- **GPT-4o** rubric = *secondary, automatic* evaluation — **different model, different
  rubric** than the optimizer. Disclose it as auxiliary, not primary.
- **Human study** = *primary independent* evidence. In the rebuttal, **lead with the
  human-study numbers**, then GPT as corroboration.

Rebuttal sentence (R4):
> "Qwen is the optimization critic; the human study is our primary independent
> evaluation; GPT-based scoring is auxiliary and uses a different model and rubric.
> Attention maps are motivation, not proof — the modality ablation and the new
> capacity-matched / random-init control are the load-bearing evidence."

### 2.3 The one table to build (paste as ASCII into the rebuttal)

Rows = the 7 variants (Section 1.3); columns = edit alignment, motion strength,
motion naturalness, source preservation, edit locality, (human) overall. Report
mean over 8 scenarios; random-init as mean±std over 3 seeds. Bold the winner per
column. This single table answers R1, R2(3), R4(5) at once.

---

## 3. PRIORITY 3 — Clarify the silent videos (R2's blocker, also R1/R4)

No experiment — a **scope clarification** + a camera-ready paragraph. Do NOT argue
"silent is fine" without explanation.

Rebuttal text:
> "We agree our wording can make the task look like final audio-video editing. The
> task is **visual motion editing**: the audio-conditioning latent is used **internally**
> as a temporal control variable, and the benchmark/metrics target visual motion.
> Final audio quality is not part of the evaluated objective, which is why outputs are
> presented silently. We will (a) add a supplementary subsection *'Clarification: Audio
> as Internal Control, Not Output Audio Editing'*, and (b) revise the title/abstract/
> intro to remove any implication of audio-output editing."

**Optional supporting datapoint (R2's "is the latent meaningful audio?"):** the code
*already decodes the optimized audio latent to a WAV* via `_save_preview_audio`
([multimodal_loop_qwen.py:733-790](editing/src/motion_opt/multimodal_loop_qwen.py#L733-L790)).
We can state: "the optimized latent remains decodable audio and stays close to the
source (bounded by the L2 reg); we do not claim its *semantics* change meaningfully —
consistent with framing it as a control pathway." Do **not** overclaim audio semantics.

Camera-ready task: add the supplementary subsection; reword title/abstract/intro.

---

## 4. PRIORITY 4 — Runtime / cost analysis (R2, R3)

Move runtime into the main text. Numbers to report (H100 80GB):
- 15-iteration run: **589.6 s** total.
- Post-warmup average: **38.87 s/iter**.
- Default ≤30 iters with early stopping (`early_stopping: 15`) + one final render.
- Slower than prompt-only Retake — **state it as a limitation.**

**Verify these numbers freshly** while the ablation runs — every run already logs
per-iter timing implicitly; wall-clock the 15-iter case once on the target H100 to
confirm 589.6 s / 38.87 s/iter still hold, and grab Retake's prompt-only render time
on the same clip for the comparison row.

Rebuttal sentence:
> "Optimization is test-time and costs ~10–20 min (≤30 iters, early stopping). This is
> a limitation: our contribution is a **test-time controllability mechanism**, not an
> interactive editor. Per-iter cost is ~38.9 s post-warmup on one H100; a 15-iter run
> is ~590 s."

Camera-ready task: runtime table (ours vs Retake, iters, s/iter, total, GPU) in main paper.

---

## 5. PRIORITY 5 — Supplementary video discoverability (R3, persuadable!)

R3 is the borderline-accept and explicitly said the comparison videos were hard to
find. This is pure organization — cheap, high leverage. No upload in rebuttal, but
**promise + describe** the index precisely.

Build an index (camera-ready / supplementary HTML or table) with columns:
`scenario | source clip | edit prompt | ours | LTX-Retake baseline | other baselines | notes`.
Generate it from the result tree (each run already stores `prompt.txt`,
`baseline_video.mp4`, `mode_*/best_optimized_video_*.mp4`).

Rebuttal sentence:
> "We apologize the side-by-side comparisons were hard to locate. We will reorganize
> the supplement **by scenario**, and add an index table linking each source+prompt to
> its ours-vs-baseline comparison; side-by-side grids will be included for every case."

Implementation task: small script to walk `results/` and emit `index.md`/`index.html`.

---

## 6. PRIORITY 6 — Single backbone (R3)

Rebuttal text (frame as scope, not weakness):
> "LTX-2 Retake is, to our knowledge, the only **publicly available joint audio-video
> model exposing an editable retake/inpainting interface**, which our method requires.
> We therefore frame this as a **first study** of the audio-conditioning control
> phenomenon and note single-backbone dependence as an explicit limitation. The
> mechanism is architecture-agnostic in principle and should transfer to future joint
> A/V editors."

Optional (only if time after Priority 1–2): probe whether the phenomenon survives
with a *vanilla* (non-Retake) base — but this is **not** worth delaying the decisive
ablation. List as future work if not run.

Camera-ready task: add the single-backbone limitation sentence.

---

## 7. PRIORITY 7 — Text-residual vs direct-audio parameterization (R4)

No new experiment strictly required — a **principled explanation** + optional cheap check.

Rebuttal text:
> "Text is optimized as a **residual** on the Gemma embedding because the text encoding
> is prompt-conditioned and semantically variable; a residual keeps it anchored to the
> prompt's meaning. The audio latent is a **fixed clip-/duration-specific continuous
> conditioning variable**, so we optimize it **directly** with an L2-to-source
> regularizer that provides the same anchoring role the residual gives text."

This matches the code: text uses `delta_v` added to `base_pos_context.video_encoding`
([multimodal_loop_qwen.py:242-247](editing/src/motion_opt/multimodal_loop_qwen.py#L242-L247));
audio is optimized directly with reg-to-source ([L390-393](editing/src/motion_opt/multimodal_loop_qwen.py#L390-L393)).

Optional cheap check (1–2 scenarios): run text as **direct** (parameterize the full
embedding, reg-to-source) and report it's worse/unstable — but a clear explanation
alone likely satisfies R4. Add a short "Why direct audio and residual text?" paragraph
to the method/supp.

---

## 8. PRIORITY 8 — Ethics / broader impact (R1)

Add a short paragraph (camera-ready) + acknowledge in rebuttal:
> "We will add a broader-impact statement: the method could be misused for deceptive
> video manipulation. Mitigations: it is **slow (not real-time), leaves visible
> artifacts**, and outputs should be disclosed/watermarked; use must respect consent
> and provenance."

---

## 9. PRIORITY 9 — Failure analysis (R2) & benchmark release (R4)

- **Failure cases (R2):** write a short paragraph with named examples — the **goldfish**
  (R2 cited it), over-regeneration, and hard/complex multi-part motions. Use frames
  from the goldfish ablation run. Own the artifacts; show we understand when/why it fails.
- **Benchmark/code release (R4):** state we will release **source prompts, task list,
  per-scenario configs, evaluation scripts, and outputs**. (Configs already live in
  [`editing/configs/`](editing/configs/); the README already lists release TODOs.)

Rebuttal sentence (R4):
> "We will release the benchmark prompts, task list, configs, evaluation scripts, and
> generated outputs for independent reproducibility."

---

## 10. PRIORITY 10 — Benchmark size (R1, R2)

The benchmark (25 self-curated + 12 full-baseline) is called small/curated. We won't
build a new large benchmark in the rebuttal window. Strategy:
- Reframe as a **first study** with a **curated but diverse** set spanning organic /
  human / rigid / hard motions (point to the 8-scenario spread in Section 1.4 as
  evidence of diversity).
- Promise release (Section 9) so others can extend it.
- If any cheap additional scenarios can be added from the 58 input videos in
  `input_videos/`, add 3–5 to nudge the count — but only after Priority 1–2.

---

## Execution order & ownership

1. **Day 0 (now):** make the 3 code edits in §1.2; `DRY_RUN=1` verify; smoke-test
   one zero-init + one random-init run end-to-end on `dog_yawning`.
2. **Day 0–2:** launch the full §1.3 grid over §1.4's 8 scenarios in parallel
   across GPUs (~70 runs). This is the long pole — start before writing prose.
3. **Day 1–2 (parallel):** build `eval_metrics.py` (§2.1); freshly time the 15-iter
   run + Retake baseline (§4); build the supplementary index script (§5).
4. **Day 2–3:** assemble the §2.3 master table; decide the §0 wording from the numbers.
5. **Day 3:** write all rebuttal prose (§3,4,5,6,7,8,9,10) + per-reviewer responses.

---

## Pre-rebuttal checklist (mirrors reviewer asks)

- [ ] Code: `--audio-init {source,zero,random}` + `--audio-reg-anchor {source,init,none}` added & tested
- [ ] Ablation grid (7 variants × 8 scenarios, random ×3 seeds) run to completion
- [ ] Master table: modality + capacity/random/zero control, objective + human metrics
- [ ] Objective metric script (RAFT motion / LPIPS preservation / temporal / CLIP align / locality)
- [ ] Goldfish + failure-case paragraph
- [ ] Runtime numbers re-verified on H100; ours-vs-Retake row
- [ ] Supplementary index (scenario/source/prompt/ours/baselines) script + plan
- [ ] Clarification paragraph: audio as internal control, silent outputs intentional
- [ ] "Why direct audio vs residual text?" paragraph
- [ ] Human study framed as primary; GPT auxiliary; Qwen=critic disclosed
- [ ] Single-backbone limitation sentence
- [ ] Ethics / deepfake paragraph
- [ ] Benchmark/config/eval-script/output release statement
- [ ] Title/abstract/intro reworded to drop "audio-output / audio-semantic editing" implication

---

## Per-reviewer response skeleton (fill numbers after runs)

**R1 (borderline reject):** Thank → capacity-matched control added (zero + random init,
same optimizer/iters/losses, ×3 seeds): [result]. Audio-specific gain is in
[preservation/locality/naturalness] / OR we revise the claim to pathway-capacity. →
ethics paragraph added → benchmark diverse + will be released.

**R2 (hardest):** Lead with scope clarification (visual motion editing; audio internal;
silent intentional) → random/zero ablation: [result] → runtime table + limitation →
goldfish/failure analysis added → benchmark release. Avoid any "audio semantics" claim.

**R3 (borderline accept — give ammunition):** Apologize for video discoverability → index
+ by-scenario reorg promised → runtime analysis added → single-backbone explained
(LTX Retake only public joint A/V editor) → reaffirm generalizability/transfer results.

**R4 (constructive reject):** Attention maps = motivation not proof; load-bearing evidence
= modality ablation + new control → residual-text vs direct-audio rationale → human study
primary, GPT auxiliary, Qwen critic → benchmark/config/eval release.
</content>
</invoke>
