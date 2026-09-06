# H3 scenario hunt for the user study (living log)

Goal: 10-15 good-to-perfect A/B pairs (A = H3 text-editing baseline, B = ours) where H3's own editing fails
to perform the motion or does it unnaturally, and our full method (both mode, best recipe) activates or
improves it. Same source video, prompt, noise and step count in A and B.

## Recipe (from the goldfish study, `H3_FINDINGS.md` Sec. 11-14)

- prompt: the generic structured `[video editing]` grammar with the plain edit sentence (`baseline_sweep.grammar_prompt`)
- method: both mode (audio conditioning latent + text residual), NGD relative step 3 %, momentum 0.3, K=2 differentiable steps,
  16 iterations, early stop 10, LPIPS 1.0 / temporal 0.3, no L2 anchors, previews every iteration
- critic: Qwen2.5-VL, fp32 yes/no head, 224 px, a scenario-specific "does X happen in any frame?" question, noisy-OR
  any-window objective and selection (brief events), checkpoint = best any-window score + perceptual
- deliverable per scenario: baseline and optimized rendered with identical noise at 16 steps, plus a 32-step re-render pair
  for the study when the 16-step one is convincing
- budget: baseline screening on 1 GPU (~4 min per candidate, batched); optimization ~1 h on 2 GPUs per scenario; run <= 4
  optimization jobs concurrently; cancel early on degenerate trajectories

## Pipeline

1. `h3_probe/scenarios/candidates_r1.json` - candidates (slug, src, wav, edit, scene, question)
2. `h3_probe/scenarios/screen.py` (+ `screen.sbatch`, 1 GPU) - baseline render + critic scores + contact sheet per candidate
3. visual triage here (fail / unnatural / success) -> pick the fails
4. `h3_full_method.py` with `SCEN_JSON=...` on the picks (2 GPUs each, D recipe)
5. visual verification of the previews; 32-step re-render (`--phase render`) of the winners; A/B package under
   `h3_probe/results/user_study/<slug>/{A_baseline,B_ours}_av.mp4`

## Inputs

- existing LTX retake inputs (512x320, 3.6 s, with audio): results/videos/*/retake_input_prepared.mp4 (19 scenarios)
- `input_videos/` (project storage) was unreachable on 2026-09-06 - the original main-scenario sources are only
  available through the captured H3 states; new sources are generated with H3 text-to-video instead
- generated sources (H3 t2va, 448x768, 5.2 s, with audio): `/scratch/amirrz/H3_exp/inputs/gen/<slug>__t2va.mp4`
  (+ `inputs/sweep/<slug>_h3sound.wav`), prompts in `inputs/gen/gen_inputs_r1.json`

## Log

### 2026-09-06 - round 1 setup
- generated-input job submitted (8 clips: jeep in jungle, woman on bench, horse in field, kid on swing, cat on windowsill,
  man with coffee mug, drummer, frog on lily pad)
- candidate list r1: 32 candidates (5 known fails, 19 new edits on existing inputs, 8 on generated inputs) - `h3_probe/scenarios/candidates_r1.json`
- screening jobs: existing-input candidates 20360024 (19), generated-input candidates 20360025 (8, after gen job 20359619)
- optimization batch 1 (known fails, both, D recipe, LPIPS 1.0/temporal 0.3): turtle_extends_neck 20360026 cat_yawns 20360027 boy_crouches 20360028 red_bird_opens_wings 20360029 

### 2026-09-06 - screening results, batch 2
- generated-input screening OOMed after its first candidate: diffusers' auto-offload hook only evicts other models when a
  component still has to be moved, so after one full-size (448x768, 124 f) render the transformer and text encoder both stay
  resident and the next render has no room for activations. Fix in `screen.py`: offload every component between candidates
  (retake-size inputs never hit it). Rerun of the 7 missing generated candidates: job 20363955.
- triage of the first existing-input results (contact sheets): child_jumps = clear fail (child stays seated, critic 0.0007);
  man_covers_face = success; boy_splashes = unnatural (boy stands up out of the pool instead of splashing);
  man_claps = partial (hands meet once, then drop); eagle_head_turn = eagle too small to judge; gen_jeep_drives = jeep
  creeps forward (critic 0.04/0.11), weak candidate
- batch 2 queued with dependencies so at most 4 optimization jobs run at once: child_jumps 20363983 (after turtle),
  man_shouts 20363984 (after cat_yawns)

### 2026-09-06 - batch 1 outcome, screening complete, batch 3
- existing-input screening done (18/19; dog_runs_off OOMed before the offload fix, re-screened with round 2).
  Successes (skip): man_covers_face, car_hood_opens, robot_both_arms, cartoon_boy_jumps (jump happens early).
  Fails/unnatural: child_jumps, monkey_jumps_branch, dog_stands_up, car_drives_out (moves only in the last frames),
  rose_sways, turtle_walks, man_celebrates (arms up only in the last 3 frames), bird_hops, man_nods, dog_tilts_head,
  cat_stretches, boy_splashes (stands up instead), man_claps (partial), eagle_head_turn (subject too small).
- batch 1: turtle_extends_neck - no visible change (critic any 0.08 -> 0.20 at best iter 3, oscillates after), the head is
  already out in the source and the edit is too subtle for the critic -> dropped. cat_yawns - baseline already yawns in
  the last frames (critic any 0.78); the critic flips between 0.99 and 0.07 across iterations on near-identical frames,
  best iter is visually identical to the baseline -> dropped (a "make it earlier" objective would be needed).
  boy_crouches - with the scenario question the baseline already scores 0.76 any, so the best checkpoint stayed at the
  baseline (dz=0); the earlier run `fullmethod_boy_crouches_both_ngd` (default question, lin objective, 0.22 -> 0.68,
  boy looks down and touches the water) is used for the study pair instead.
- batch 3 (both, D recipe) chained in two tiers of four so at most 4 run at once:
  tier 1 dog_stands_up 20367783, monkey_jumps_branch 20367784, car_drives_out 20367785, cat_stretches 20367786;
  tier 2 man_celebrates 20367787, rose_sways 20367788, turtle_walks 20367789, bird_hops 20367790
- helper: `scenarios/ab_sheet.sh` (multi-row frame sheets for A/B checks), `scenarios/render_ab.sbatch`,
  `scenarios/package_ab.py`

### 2026-09-06 - round 2 sources
- 11 new H3 t2va sources with one sound-linked edit each (dog shakes off water, woman opens door, basketball bounces,
  pigeon takes off, man plays piano, candle blown out, woman waves, rowboat rocks, umbrella opens, bell swings, glass tips
  over): prompts `scenarios/gen_inputs_r2.json`, candidates `scenarios/candidates_r2.json` (group generated_r2).
  Generation job 20368202 (1 GPU), screening chained 20368203 (also re-screens dog_runs_off) -> outputs/screen_r2.
  `screen.py` now takes ':'-separated candidate files and OR-ed ONLY/GROUP filters.

### 2026-09-06 - generated-input screening (round 1)
- successes (skip): gen_woman_stands (stands up mid-clip; critic any-score 0.03 is a critic miss), gen_cat_jumps_down (late but
  jumps), gen_man_drinks, gen_drummer_plays, gen_kid_swings (child too small to judge, critic 0.99).
- fails: gen_horse_rears (walks a little, never rears, 0.0002), gen_frog_jumps (frog leaves the frame only in the last 2 frames,
  0.016), gen_jeep_drives (creeps forward). Tier 3 queued: gen_horse_rears 20368483, gen_frog_jumps 20368484 (first full-size
  448x768 x 124 f sources through Phase B - memory check).

## Scenario table (updated as results land)

| slug | source | edit | baseline (screen) | ours | status |
|---|---|---|---|---|---|
| goldfish | retake input | jumps out of the tank | fail (0.004) | clean leap (0.72 any) | DONE (D) |
| turtle_extends_neck | retake input | extends neck out of shell | fail (known, 0.08 any) | no visible change (0.20 any) | dropped |
| cat_yawns | retake input | yawns widely | late yawn (0.78 any) | identical to baseline | dropped |
| boy_crouches | retake input | crouches, touches water | fail (0.22 default-q) | earlier run: looks down, touches water (0.68) | render |
| red_bird_opens_wings | retake input | opens wings | fail (known) | running 20360029 | opt |
| man_shouts | retake input | shouts loudly | fail (known) | running 20363984 | opt |
| child_jumps | retake input | jumps up and down | fail (0.0007, static) | running 20363983 | opt |
| man_covers_face | retake input | covers face with both hands | success (0.996) | - | skip |
| car_hood_opens | retake input | hood opens | success (0.81/0.99) | - | skip |
| robot_both_arms | retake input | raises both arms | success (0.97/1.0) | - | skip |
| cartoon_boy_jumps | retake input | jumps into the air | success (jumps early, 0.12/0.83) | - | skip |
| boy_splashes | retake input | splashes water with both hands | unnatural (stands up, 0.38/0.63) | - | maybe |
| man_claps | retake input | claps hands | partial (0.14/0.56) | - | maybe |
| eagle_head_turn | retake input | turns head to camera | unclear, subject tiny (0.28/0.63) | - | skip |
| man_nods | retake input | nods head | not visible (0.22/0.67) | - | maybe |
| dog_tilts_head | retake input | tilts head | static (0.46/0.51) | - | maybe |
| dog_stands_up | retake input | stands up on all fours | fail (0.001, stays seated) | queued 20367783 | opt |
| monkey_jumps_branch | retake input | jumps to another branch | fail (0.0004, only head turn) | queued 20367784 | opt |
| car_drives_out | retake input | drives out of the garage | late/weak (0.006, moves in last frames) | queued 20367785 | opt |
| cat_stretches | retake input | stretches front legs | fail (static) | queued 20367786 | opt |
| man_celebrates | retake input | raises both arms | late (last 3 frames, 0.21/0.06) | queued 20367787 | opt |
| rose_sways | retake input | sways in the wind | fail (0.016, static) | queued 20367788 | opt |
| turtle_walks | retake input | walks forward | fail (0.07, static) | queued 20367789 | opt |
| bird_hops | retake input | hops along the branch | fail (0.14/0.27, wing flutter only) | queued 20367790 | opt |
| dog_runs_off | retake input | gets up and runs off | screening OOMed, rescreen r2 | - | pending |
| gen_jeep_drives | H3 t2va | drives forward | weak motion (0.04/0.11) | - | maybe |
| gen_horse_rears | H3 t2va | rears up on hind legs | fail (0.0002, only walks) | queued 20368483 | opt |
| gen_frog_jumps | H3 t2va | jumps off the lily pad | late (leaves frame in last 2 frames, 0.016) | queued 20368484 | opt |
| gen_woman_stands | H3 t2va | stands up from bench | success (stands mid-clip) | - | skip |
| gen_cat_jumps_down | H3 t2va | jumps down from windowsill | success (late but jumps) | - | skip |
| gen_man_drinks | H3 t2va | drinks from mug | success (0.93/0.98) | - | skip |
| gen_drummer_plays | H3 t2va | plays drums | success (0.73/0.99) | - | skip |
| gen_kid_swings | H3 t2va | starts swinging | unclear, subject tiny (0.55/0.99) | - | skip |
