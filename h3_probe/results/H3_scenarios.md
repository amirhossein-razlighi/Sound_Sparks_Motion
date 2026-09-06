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

## What we have learned (updated as we go)

Where H3 fails and ours saves it (the 3 confirmed pairs: goldfish leap, cat yawn, man shout):
- single, centred subject in a medium or close shot; the motion is a **brief event** with a clear visual signature and a
  natural sound (splash, yawn, shout); H3's baseline either does nothing or does it late/weakly.
- the any-window critic objective + NGD 3 % + LPIPS 1.0/temporal 0.3 + drift guard; the showcase iteration is picked by eye
  from the per-iteration previews (often iter 2-4, before drift).

Where it does not work (drop quickly):
- **flat critic** (baseline ~0, no precursor) on whole-body pose changes from a static pose (child jumps, dog stands up,
  monkey jumps, horse rears): nothing moves for 10 iterations, then adversarial drift. Exception: goldfish (strong prior).
- **small motions** (neck, nod, head tilt, hop, sway): critic cannot guide, and too subtle for a user study anyway.
- **edit that does not fit the source** (crouch and touch water when already chest-deep) or **tiny subject** (eagle, kid on swing).
- **sustained/pose edits with the linspace objective** (celebrate, wings, walk): at best a partial pose change without the
  expression that makes it convincing (man_celebrates rejected by the user).
- H3 already succeeds on common actions with clear text semantics (stand up, take off, open umbrella, play piano, drink,
  wave, cover face, raise arms) -> no room to save anything.

Consequences for finding scenarios: prefer animal vocalisations and sudden events (bark, roar, neigh, sneeze, yawn,
laugh, leap, splash, blow-out) on single centred subjects; generate sources at retake size (320x512x89) so runs fit on 2 GPUs.

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

### 2026-09-06 - batch 2 first results: objective must match the failure type
- red_bird_opens_wings: with the scenario question the baseline already scores 0.97 any (the bird half-opens its wings in
  the last frames), so the any-window objective is saturated and the best checkpoint stays at the baseline (dz=0) - same
  pattern as boy_crouches/cat_yawns. Late/partial cases need the linspace objective (rewards the motion across the whole
  clip; baseline lin yes 0.11): resubmitted as `us_red_bird_opens_wings_lin` (20368915, CRITIC_OBJ=lin SELECT_BY=lin, reusing
  the capture).
- child_jumps: critic flat at 0.002 for 10 iterations while the latents move (perceptual 0.05) - H3 does not start a jump
  from this seated pose; man_shouts hovers at 0.15-0.18. Waiting for tier 1 before deciding on the other "flat" cases.

### 2026-09-06 - batch 2 done: drift guard, queue re-planned
- child_jumps: critic flat (0.002) for 10 iterations, then at iter 11-13 the latents drift into caption-like white text
  panels that the critic scores 0.14 - adversarial drift, the total loss (nll + LPIPS 0.5) still preferred it. Unusable.
  Added `--perc-max` / PERC_MAX to h3_full_method.py (default off): checkpoints whose perceptual term exceeds it are never
  selected. All new runs use PERC_MAX=0.25.
- man_shouts: best iter 2 (any 0.73 -> 0.99, small change): hands come up a little earlier; the shout itself is not
  clearly stronger -> weak pair, keep as reserve.
- re-plan: "flat" cases (critic ~0 and no motion precursor: monkey_jumps_branch, gen_horse_rears) and critic-saturated
  cat_stretches are cancelled. Late/partial/sustained cases are resubmitted with the linspace objective (rewards the motion
  across the whole clip instead of one window) as `us_<slug>_lin`, reusing captures where Phase A already ran:
  car_drives_out 20370116, man_celebrates 20370117, bird_hops 20370118, gen_frog_jumps 20370119, rose_sways 20370120, turtle_walks 20370121.
  dog_stands_up (any objective, running) stays as the second data point for flat cases.

### 2026-09-06 - visual inspection of every iteration (rule change)
- The critic's best checkpoint is not the showcase: for each run all iteration previews are now put on one sheet
  (`ab_sheet.sh`, one row per iteration, CROP to zoom on the subject) and inspected.  Findings:
  cat_yawns iter 3 = wide early yawn (baseline yawns only in the last frame) -> usable (alt iter 9); iters 6/7/10 show
  overlay artifacts (drift); man_shouts iter 3 = hands up from mid-clip + open-mouth shout in natural framing (iter 4 is a
  dramatic shout but the shot changes to a face close-up); turtle_extends_neck = no change in any iteration.
- picks are recorded in `scenarios/picks.json` (slug -> run dir + iteration); `package_ab.py` builds
  `results/user_study/<slug>/{A_baseline,B_ours}_av.mp4` from them (16 steps, same noise).  Per-iteration latents are now
  saved by the method script (`iter_XX_latents.pt`) so future picks can be re-rendered at 32 steps.
- packaged so far: goldfish, boy_crouches, cat_yawns (iter 3), man_shouts (iter 3).

### 2026-09-06 - dog_stands_up cancelled; flat-critic rule
- dog_stands_up: 8 iterations, critic flat at 0.005, every preview identical to the baseline (dog stays seated) -> cancelled
  at iter 8. With child_jumps this makes the rule: if the baseline critic score is ~0 and the baseline shows no precursor of
  the motion, the gradient carries no usable direction and the run either stays put or drifts. Such cells are skipped.
- round-2 screening so far: gen_pigeon_rail and gen_pianist succeed at baseline (skip); gen_dog_wet never shakes (0.09/0.23,
  the dog only turns) and gen_woman_door opens the door only in the last frames (0.32/0.28) -> both queued next with the
  linspace objective; gen_basketball is ambiguous (ball hovers in one early frame).

### 2026-09-06 - boy_crouches removed from the study set
- frame-by-frame check of every boy_crouches run (user flagged it): the ngd run only makes the boy look down and bring a
  hand to his face; the any-objective run produces jump cuts (boy standing with hands at the surface in the first frames,
  then the baseline pose) or a different shot (crouching on the pool edge); noreg/acc3 runs equal the baseline. No water
  touch anywhere -> removed from picks. The edit does not fit the source (the boy is already chest-deep in the pool).
  The splash edit on the same source (boy_splashes, baseline stands up out of the pool instead) is queued with the
  linspace objective (20372392).

### 2026-09-06 - first linspace results
- red_bird_opens_wings (lin): critic 0.11 -> 0.42 at iter 8 (iters 9-16 drift, perceptual > 0.58, excluded by the guard).
  Automated frame analysis (silhouette width / colour area / CLIP wing probe, run because the image viewer was blocked)
  finds no wing spread in any iteration: iters 1-6 re-time frames 3-5, iter 7 has a transient glitch, iter 8 changes texture
  only. Visual confirmation pending.
- car_drives_out (lin): flat at 0.005 for 16 iterations - the linspace sampling never sees the late movement -> failed.
- man_celebrates (lin): 0.24 -> 0.95 at iter 4 (perceptual 0.24, within the guard), iters 4-7 all > 0.65 -> promising,
  visual check pending.

### 2026-09-06 (evening) - man_celebrates confirmed, full-size inputs OOM on 2 GPUs
- man_celebrates (lin) verified on the iteration sheet: iter 4 raises both arms fully above the head from mid-clip and lowers
  them again (iter 5/6 similar, iter 6 earlier); the baseline only lifts the hands slightly in the last 3 frames -> PACKAGED
  (4 pairs now: goldfish, cat_yawns, man_shouts, man_celebrates).
- gen_frog_jumps (lin) OOMed in Phase B on 2 GPUs: the H3-generated sources (448x768, 124 frames) give a longer reference
  sequence than the retake inputs. All generated-source runs move to the 3-GPU sbatch.

- red_bird_opens_wings and bird_hops (lin) verified on the sheets: every iteration equals the baseline (late flutter only),
  no wing spread, no hop -> both failed. The parrot source is exhausted.
- generated-source runs resubmitted on 3 GPUs (`full_method_3gpu.sbatch`, two chains so at most two 3-GPU jobs run):
  gen_frog_jumps 20379834 (capture reused) -> gen_woman_door 20379837 -> gen_basketball 20379840; gen_dog_wet 20379836 -> gen_glass_table 20379839.

### 2026-09-06 (evening) - man_celebrates rejected, round 3 = face/mouth sound events
- user verdict on man_celebrates: arms go up but no smile/laugh, not convincing -> removed from the study set (3 pairs).
- what has worked so far are brief, unambiguous sound-linked events (leap, yawn, shout) with the any-window objective;
  sustained/pose edits with the linspace objective have not produced a convincing pair yet. Round 3 therefore targets
  face/mouth events on sources we already have: man_laughs (groom), man_sneezes (grey room), man_gasps (surprised man),
  child_laughs, dog_barks, monkey_screams, cat_meows, man_yawns (clapping man), cartoon_boy_laughs, gen_horse_neighs,
  gen_woman_sneezes - `scenarios/candidates_r3.json`. Screening: 20380263 (retake sources), 20380264 (generated) -> outputs/screen_r3.
  Fails go to optimization with CRITIC_OBJ=any / SELECT_BY=any (the goldfish/cat/shout recipe) + PERC_MAX.

### 2026-09-06 (night) - round 4: designed-to-fail event sources at retake size
- 13 new t2va sources generated at 320x512x89 (GEN_H/GEN_W/GEN_FRAMES overrides in t2va_sounds.py, default unchanged):
  dog on rug (barks), lion (roars), rooster (crows), wolf (howls), cow (moos), duck (flaps+splashes), dolphin (leaps),
  koi (jumps), man on couch (sneezes), woman at desk (yawns), girl with cake (blows out candles), sea lion (barks),
  goat (bleats) - `scenarios/gen_inputs_r4.json`, `scenarios/candidates_r4.json`. gen 20380356 -> screening 20380357 (outputs/screen_r4).

## Scenario table (updated as results land)

| slug | source | edit | baseline (screen) | ours | status |
|---|---|---|---|---|---|
| goldfish | retake input | jumps out of the tank | fail (0.004) | clean leap (0.72 any) | PACKAGED |
| turtle_extends_neck | retake input | extends neck out of shell | fail (known, 0.08 any) | no visible change (0.20 any) | dropped |
| cat_yawns | retake input | yawns widely | late yawn (last frame, 0.78 any) | iter 3: wide early yawn | PACKAGED |
| boy_crouches | retake input | crouches, touches water | fail (0.22 default-q) | no run touches the water (looks down / jump cuts) | dropped |
| red_bird_opens_wings | retake input | opens wings | partial, late (0.11 lin / 0.97 any) | lin: no wing spread in any iteration | failed |
| man_shouts | retake input | shouts loudly | partial, late hands (0.73 any) | iter 3: hands up + open-mouth shout | PACKAGED |
| child_jumps | retake input | jumps up and down | fail (0.0007, static) | no activation, adversarial drift at iter 11+ | failed |
| man_covers_face | retake input | covers face with both hands | success (0.996) | - | skip |
| car_hood_opens | retake input | hood opens | success (0.81/0.99) | - | skip |
| robot_both_arms | retake input | raises both arms | success (0.97/1.0) | - | skip |
| cartoon_boy_jumps | retake input | jumps into the air | success (jumps early, 0.12/0.83) | - | skip |
| boy_splashes | retake input | splashes water with both hands | unnatural (stands up, 0.38/0.63) | lin run 20372392 | opt |
| man_claps | retake input | claps hands | partial (hands meet once, 0.14/0.56) | lin run 20379582 | opt |
| eagle_head_turn | retake input | turns head to camera | unclear, subject tiny (0.28/0.63) | - | skip |
| man_nods | retake input | nods head | not visible (0.22/0.67) | - | maybe |
| dog_tilts_head | retake input | tilts head | static (0.46/0.51) | - | maybe |
| dog_stands_up | retake input | stands up on all fours | fail (0.001, stays seated) | flat critic, no change in 8 iters | failed |
| monkey_jumps_branch | retake input | jumps to another branch | fail (0.0004, only head turn) | cancelled (flat critic) | skip |
| car_drives_out | retake input | drives out of the garage | late/weak (0.006, moves in last frames) | lin: flat, no change | failed |
| cat_stretches | retake input | stretches front legs | static but critic says 0.77 lin | cancelled (critic saturated) | skip |
| man_celebrates | retake input | raises both arms | late (last 3 frames, 0.21/0.06) | iter 4: arms up but no expression - user: not convincing | rejected |
| rose_sways | retake input | sways in the wind | fail (0.016, static) | lin run 20370120 | opt |
| turtle_walks | retake input | walks forward | fail (0.07, static) | lin run 20370121 | opt |
| bird_hops | retake input | hops along the branch | fail (0.14/0.27, wing flutter only) | lin: identical to baseline | failed |
| dog_runs_off | retake input | gets up and runs off | fail (0.04/0.06), dog small in cluttered scene | - | skip |
| gen_jeep_drives | H3 t2va | drives forward | weak motion (0.04/0.11) | - | maybe |
| gen_horse_rears | H3 t2va | rears up on hind legs | fail (0.0002, only walks) | cancelled (flat critic) | skip |
| gen_frog_jumps | H3 t2va | jumps off the lily pad | late (leaves frame in last 2 frames, 0.016) | 2-GPU OOM; 3-GPU run 20379834 | opt |
| gen_woman_stands | H3 t2va | stands up from bench | success (stands mid-clip) | - | skip |
| gen_cat_jumps_down | H3 t2va | jumps down from windowsill | success (late but jumps) | - | skip |
| gen_man_drinks | H3 t2va | drinks from mug | success (0.93/0.98) | - | skip |
| gen_drummer_plays | H3 t2va | plays drums | success (0.73/0.99) | - | skip |
| gen_kid_swings | H3 t2va | starts swinging | unclear, subject tiny (0.55/0.99) | - | skip |
| gen_pigeon_rail | H3 t2va (r2) | takes off and flies | success (0.98/1.0) | - | skip |
| gen_pianist | H3 t2va (r2) | plays the piano | success (0.96/1.0) | - | skip |
| gen_dog_wet | H3 t2va (r2) | shakes water off | fail (only turns, 0.09/0.23) | 3-GPU run 20379836 | opt |
| gen_woman_door | H3 t2va (r2) | opens door, walks in | late (door opens in last frames, 0.32/0.28) | 3-GPU run 20379837 | opt |
| gen_basketball | H3 t2va (r2) | bounces | partial (hovers in one frame, 0.17/0.23) | 3-GPU run 20379840 | opt |
| gen_woman_beach | H3 t2va (r2) | waves at camera | success (0.98/1.0) | - | skip |
| gen_umbrella | H3 t2va (r2) | opens umbrella | success (0.98/0.99) | - | skip |
| gen_candle | H3 t2va (r2) | flame blown out | fail (0.0009, steady flame) | held (flat critic) | hold |
| gen_rowboat | H3 t2va (r2) | rocks side to side | fail (0.01, static) | held (flat critic) | hold |
| gen_bell | H3 t2va (r2) | swings and rings | fail (0.06/0.10, barely moves) | held | hold |
| gen_glass_table | H3 t2va (r2) | tips over, spills | late + distorted (glass morphs in last frames) | 3-GPU run 20379839 | opt |
