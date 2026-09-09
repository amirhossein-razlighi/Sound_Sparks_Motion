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
  monkey jumps, horse rears): nothing moves for 10 iterations, then adversarial drift. Exceptions: goldfish and koi (the
  critic stayed flat on the koi although the jump is visible) - always look at the previews before dropping a run.
- **small motions** (neck, nod, head tilt, hop, sway): critic cannot guide, and too subtle for a user study anyway.
- **edit that does not fit the source** (crouch and touch water when already chest-deep) or **tiny subject** (eagle, kid on swing).
- **sustained/pose edits with the linspace objective** (celebrate, wings, walk): at best a partial pose change without the
  expression that makes it convincing (man_celebrates rejected by the user).
- H3 already succeeds on common actions with clear text semantics (stand up, take off, open umbrella, play piano, drink,
  wave, cover face, raise arms) -> no room to save anything. The same holds for dramatic whole-object destruction (the
  whole car shatters into debris): a drastic geometry change is a strong text prior for H3, not a weakness.

Consequences for finding scenarios: prefer animal vocalisations and sudden events (bark, roar, neigh, sneeze, yawn,
laugh, leap, splash, blow-out) on single centred subjects; generate sources at reduced size (320x512, 124 frames = H3's 5 s minimum) so runs fit on 2 GPUs.

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
- 13 new t2va sources generated at 320x512 (124 frames) (GEN_H/GEN_W/GEN_FRAMES overrides in t2va_sounds.py, default unchanged):
  dog on rug (barks), lion (roars), rooster (crows), wolf (howls), cow (moos), duck (flaps+splashes), dolphin (leaps),
  koi (jumps), man on couch (sneezes), woman at desk (yawns), girl with cake (blows out candles), sea lion (barks),
  goat (bleats) - `scenarios/gen_inputs_r4.json`, `scenarios/candidates_r4.json`. gen 20380356 -> screening 20380357 (outputs/screen_r4).

### 2026-09-06 (night) - round 5: variety beyond mouth events
- user: vary the motion types (cars, doors, dances, falling, shattering, breaking), not only mouths. Round 5 = 14 generated
  sources at retake size + 2 edits on existing car sources: glass falls and shatters, vase topples, dominoes topple, block
  tower collapses, balloon pops, ice cracks, door slams, elevator doors open, ceiling fan spins, woman dances/spins, man
  dances, stone splashes into pond, car door swings open (generated + the red-car retake source), windmill turns, car
  headlights flash + horn - `scenarios/gen_inputs_r5.json`, `scenarios/candidates_r5.json`. gen 20380390 (after r4 gen) ->
  screening 20380391 (outputs/screen_r5).

### Conditional plan: Qwen3-VL critic (only if adversarial critic-fooling turns out to be the dominant failure)
- user: if, after the current runs, many "successes" are adversarial answers that fool the Qwen2.5-VL critic, consider a
  Qwen3-VL critic - only if memory and the rest of the pipeline allow it. Feasibility checked 2026-09-06 evening: the venv has
  transformers 5.16.1 (Qwen3VLForConditionalGeneration available), torch 2.14; the login node has internet and /project has
  ~326 GB free, so Qwen3-VL-8B-Instruct (~17 GB bf16, about the same footprint as Qwen2.5-VL-7B) can be downloaded to
  /project/def-amahdavi/amirrz/HF/models/. Work needed: an additive critic module for the Qwen3-VL vision path (patch 16,
  frame timestamps in the text stream) mirroring `compute_qwen_video_loss`, then a calibration pass like Sec. 11. Not started;
  evidence so far: adversarial drift appeared in child_jumps (iters 11+), cat_yawns (iters 6/7/10 overlays), red_bird (iters
  9-16) - always late and always caught by the perceptual guard, so the guard + early picks have been sufficient.

- round 4/5 generation at 89 frames failed: H3 generates 5-15 s only (num_frames rounded to 17n+5 must be in 120..360).
  Resubmitted at 320x512x124 (still ~half the reference tokens of the 448x768 sources, should fit 2 GPUs):
  r4 gen 20380816 -> screen 20380817; r5 gen 20380818 -> screen 20380819.

- rose_sways and turtle_walks (lin): critic decreases from iter 1 onwards and every preview equals the baseline -> failed.
  Confirms the rule: static baseline + sustained small motion is not recoverable with this critic.

### 2026-09-06 (night) - round 3 screening
- H3 handles laugh/gasp/yawn on people well: man_laughs 0.95, man_gasps 0.82, child_laughs 0.75, man_yawns 0.94,
  cartoon_boy_laughs 0.97, monkey_screams (mouth wide open mid-clip) -> all skip. So human facial events are not where H3
  fails; animal vocalisations and less common events are.
- fails/late: dog_barks (lin 0.008 / any 0.90: head turn + mouth only in the last frames -> "late" like the cat yawn),
  man_sneezes (0.06/0.04: raises hands instead), cat_meows (brief early meow, 0.23/0.20 - partial), gen_horse_neighs
  (0.02/0.13, only walks), gen_woman_sneezes (0.006/0.23, hand to face late). Queued with the event recipe (any objective):
  dog_barks 20383015, man_sneezes 20383016, cat_meows 20383017 (2 GPUs). Horse/woman (full-size sources, 3 GPUs) held for now.

### 2026-09-06 (night) - round 4 screening; 3-GPU memory fix; boy_splashes promising
- gen_frog_jumps OOMed even on 3 GPUs: the default 3-GPU split puts all transformer blocks on GPUs 0/1 (31+31 GiB, 69 GB
  peak each after the render) and only the critic/VAE on GPU 2 (26 GB). Resubmitted all full-size runs with
  GPU_MEM_SPLIT=24,24,14: frog 20384601 -> woman_door 20384602 -> basketball 20384603; dog_wet 20384604 -> glass 20384605.
- round-4 screening (320x512x124 sources): H3 succeeds on lion roar (0.95), duck flap (0.84), dog on rug barks (0.59/0.98),
  girl blows out candles (0.71) -> skip. Fails: koi (swims, never jumps, 0.001), dolphin (fin only, 0.001), wolf (static,
  0.02), woman at desk (no yawn, 0.04), man on couch (dark, small, 0.008), cow (0.12/0.38), goat (0.23/0.36),
  rooster (silhouette, 0.25/0.65 partial), sea lion (mouth opens only at the end, 0.09/0.87 -> late).
  Queued with the event recipe (any objective, 2 GPUs) in two chains behind the running 2-GPU jobs:
  A: gen_woman_desk 20384606 -> gen_wolf_hill 20384607 -> gen_dolphin_sea 20384608 -> gen_rooster 20384609;
  B: gen_koi_pond 20384610 -> gen_sealion 20384611 -> gen_cow_field 20384612 -> gen_goat_field 20384613. man_couch skipped (too dark/small).
- boy_splashes (lin): critic 0.33 -> 0.91 at iters 9-10; sheets confirm a real two-handed splash with spray in iters 7-10
  (iter 9 clearest) while the baseline only stands up -> PACKAGED (4 pairs: goldfish, cat_yawns, man_shouts, boy_splashes).

### 2026-09-06 (night) - round 5 screening (variety)
- H3 succeeds: vase topples (0.87), ceiling fan spins (0.92), generated car door opens (0.96), dominoes topple early
  (critic missed it), man at bus stop dances (critic missed it), stone splash partial (0.39/0.81) -> skip.
- fails: glass at edge tips but vanishes instead of shattering (0.16/0.14), block tower static (0.04), balloon never pops
  (0.003), ice barely cracks (0.006, too subtle), hallway door stays open (0.02/0.12), elevator doors stay shut (0.03),
  woman in studio static and small (0.003), windmill blades static (0.14/0.63), workshop car door stays shut (0.04/0.24),
  headlights never flash (0.005).
- queued (2 GPUs) appended to the round-4 chains: A: gen_balloon_kid 20392350 -> gen_door_hall 20392351 -> gen_elevator 20392352 ->
  gen_blocks_tower 20392353 (any); B: car_door_opens 20392354 -> car_lights_flash 20392355 -> gen_glass_edge 20392356 (any) -> gen_windmill 20392357 (lin,
  sustained rotation). Skipped: ice (too subtle), woman_studio (subject too small).

- man_claps (lin): iters 2-3 reach 0.89/0.80 but with perceptual 0.52-0.56: H3 zooms out to a wider shot in which the man
  claps continuously - a real clap, same man and studio, but a framing change (the tight face shot cannot show hands);
  iter 4 swaps the person (identity drift, then collapse). Packaged as RESERVE with the caveat in the note; user decides.

### 2026-09-07 (early) - 320x512x124 sources still OOM on 2 GPUs -> trimmed to 89 frames
- gen_woman_desk OOMed on 2 GPUs (78 GB on GPU 0): 124 reference frames are 1.4x the retake inputs' 89. All round-4/5 generated
  sources are now trimmed to 89 frames / 3.7 s (`*_89f.mp4`, `*_89f.wav`, originals kept), i.e. exactly the retake geometry
  that fits; candidates_r4/r5 point to the trimmed files (screening used the 124-frame clips, so the run's own Phase-A
  baseline is the reference for each pair). Chains resubmitted (REUSE_CAPTURE=0 so no stale capture is reused):
  A: woman_desk 20400034 -> wolf 20400035 -> dolphin 20400036 -> rooster 20400037 -> balloon 20400038 -> door 20400039 -> elevator 20400040;
  B: koi 20400041 -> sealion 20400042 -> cow 20400043 -> goat 20400044 -> blocks 20400045 -> glass_edge 20400046 -> windmill 20400047 (lin).
  car_door_opens / car_lights_flash (retake sources) keep their jobs.

### 2026-09-07 (early) - full-size sources OOM even on 3 GPUs -> downscaled copies
- gen_frog_jumps with GPU_MEM_SPLIT=24,24,14 still OOMed on GPU 0 (77.8 GB): the 448x768x124 reference simply does not fit
  the K=2 differentiable render. The five round-2 generated sources are converted to 320x512x89 copies (`*_89f_small.mp4`,
  originals kept) and run on 2 GPUs as `us_<slug>_small` in one chain behind cat_meows: frog 20403374 -> woman_door 20403375 ->
  glass 20403376 -> basketball 20403377 (any) -> dog_wet 20403378 (lin). 3-GPU chain cancelled. Lesson: keep all sources at the retake
  geometry (320x512, 89 f).

- root cause of the 3-GPU OOMs found: `--export=ALL,GPU_MEM_SPLIT=24,24,14` is split on the commas by sbatch, so the script
  saw budget [24] and put all 50 blocks on GPU 0 (61.7 GB weights). Fixed: the flag now also accepts ";" separators. One
  full-size 3-GPU test resubmitted with `GPU_MEM_SPLIT=24;24;14` (frog, 20403410, `us_gen_frog_jumps_full3`) to learn whether
  448x768x124 sources are viable at all; the small-source 2-GPU chain stays the main path.

### 2026-09-07 (early) - car_door_opens saved
- car_door_opens (any): critic 0.02 -> 0.95 at iter 4 (perceptual 0.12); the driver door of the car on the lift swings open
  mid-clip in iters 2-5 with the rest of the scene intact; from iter 6 extra people/tyres appear (drift) -> iter 4 PACKAGED.
  First non-body-motion pair (object/sound event). 5 pairs + 1 reserve.

- gen_woman_desk (trimmed source): the run's own Phase-A baseline already yawns (any 0.91, lin 1.0) - H3's behaviour differs
  between the 124-frame and the 89-frame reference. Nothing to save -> skip. (Screening verdicts on 124-frame clips are only
  indicative for the trimmed runs; each run's own baseline decides.)

### 2026-09-07 03:00 - user review of the overnight runs
- user picks from the iteration previews: gen_woman_desk iter 2, gen_sealion iter 2 (iter 3 candidate), gen_koi_pond best
  (iters 4/6 candidates), gen_dolphin_sea iter 2 (run still in progress). Packaged; alternates are stored as
  \`B_candidate_iterNN_av.mp4\` next to \`B_ours_av.mp4\` (package_ab.py \`alts\`). Study set: 9 pairs + 1 reserve.
- lesson: the critic can also miss real motion (koi: flat 0.003 while the jump is visible to a human), so "flat critic"
  runs still deserve a look at the previews before being called failed - the drift guard keeps them cheap to keep.

- gen_dolphin_sea finished: iter 2 is a full dolphin leap with splash (perceptual 0.29, just above the guard so the script
  never called it "best"); iters 3-4 replace the dolphin with a bird / a dog leaping out of the sea (identity drift, a new
  failure mode to watch for), iter 9 a weaker leap. Only iter 9 kept as candidate. The perceptual guard at 0.25 was slightly
  too strict here - worth remembering when a large motion legitimately changes many pixels.

- 3-GPU full-size test (gen_frog_jumps_full3, GPU_MEM_SPLIT=24;24;14): runs to completion - so 448x768x124 sources are
  viable on 3 GPUs once the split is parsed correctly (memory finding for the paper). The frog itself: critic flat
  (0.03 at iter 2, then 0.001), frames barely change (diff 0.017); the final re-render scores 0.41 (render nondeterminism).
  Sheet kept (\`us_gen_frog_jumps_full3/iters_sheet_a.jpg\`).

### 2026-09-07 05:30 - mode ablation for the packaged pairs
- user: show that text-only and audio-only are not as good as both. \`scenarios/ablate.py\` rebuilds each winning run's
  exact environment from its run_config.json (same objective, question, recipe, seed), reuses its Phase-A capture and
  submits OPT_MODE=text and OPT_MODE=audio to \`outputs/ablation/<slug>/{text_only,audio_only}\` (3 chained lines of 2-GPU
  jobs; frog full-size on 3 GPUs). Manifest: \`results/ablation/manifest.json\`. Packaging into
  \`results/ablation/<slug>/{both,text_only,audio_only}\` with a comparison sheet follows when the runs finish.

### 2026-09-07 07:00 - glass at the table edge saved
- gen_glass_edge (any, trimmed source): iter 3 makes the glass tip off the edge and shatter into fragments on the floor, where the
  baseline just makes it vanish -> PACKAGED (14 pairs + 1 reserve); ablation runs added for it.

### 2026-09-07 12:00 - user picks (windmill, wolf); study folders now carry the model input and prompts
- user picks: gen_windmill iter 3 (iter 14 candidate), gen_wolf_hill iter 3; gen_glass_edge iter 3 confirmed. 16 pairs + 1 reserve.
  Ablation runs added for windmill and wolf.
- every \`results/user_study/<slug>/\` now also contains \`source_input_av.mp4\` (the raw video given to the model, with its
  audio), \`source_audio.wav\`, and \`prompts.txt\` (edit sentence, scene, the full H3 prompt shared by A and B, the critic
  question, mode/objective/picked iteration). package_ab.py builds these from each run's run_config.json.

### 2026-09-08 - ablation nearly complete
- 31/32 ablation runs finished and packaged (\`results/ablation/README.md\`, per-folder \`critic_scores.txt\`). The one failure,
  gen_woman_door audio-only, died on rg21702 ("CUDA driver initialization failed") - node added to the exclude list and the
  run resubmitted (ablate.py now takes MODES=audio|text to resubmit a single mode).
- pattern across the 15 complete triples: audio-only never changes the frames by more than ~2 % (inert on H3, even where the
  critic reports a high score, e.g. man_shouts 0.86 / windmill 0.92 with <1 % frame change - critic fooled); text-only either
  stays at the baseline or drifts, reaching a competitive critic score only late (dolphin iter 10, car door iter 11,
  windmill iter 4); both mode is the only one that produced the clean motion in each user-study pair.

### 2026-09-08 - round 6 (user-selected, no water): 6 sources
- user cancelled my 14-source list and chose 6: champagne cork pops, toaster pops the toast up, beer poured until the glass
  overflows, woman screams in fright, man slams his fist on the desk, books fall off the shelf. Plan agreed: generate ->
  trim to 89 f -> screen (baseline); optimize (both mode, event recipe) ONLY where the baseline leaves room.
  `scenarios/gen_inputs_r6.json`, `scenarios/candidates_r6.json`. gen job 20506624. Plus a yoga scenario (user: does a complex multi-step
  body motion - arms overhead, deep forward fold, rise - show where we shine?): gen job 20506630, screened with ONLY=gen_yoga.

- gen_yoga screened: H3's baseline performs the whole flow (arms overhead -> deep forward fold -> rise) cleanly (critic 0.81 lin /
  0.89 any, visually complete). Complex multi-step body motion is NOT where H3 fails -> stop, per the agreed rule.

- 2026-09-08 15:10: user suggestion - interactive allocation. \`salloc -p gpubase_interac --gres=gpu:h100:2 --time=3:00:00 --no-shell\`
  was granted within a minute (job 20518172, rg32102) while batch jobs had been waiting for hours. Steps are run inside it with
  \`srun --jobid=20518172 --overlap ...\`; the batch screening 20507917 was cancelled in favour of it.

- round-6 screening (interactive node rg31802; rg32102 hung twice on Lustre I/O and is excluded): gen_champagne baseline is
  completely static (cork never pops, 0.08/0.15) -> room -> both-mode run 20524811 submitted (event recipe).

### 2026-09-08 evening - overnight coordination (user: I stay on as coordinator/judge all night)
- source audit: beer_pour and toaster sources already contained the target motion (pouring; toast up from frame 2) -> invalid,
  regenerated as gen_beer_pour2 / gen_toaster2 with before-state prompts and GEN_SEED=7. All 17 packaged study sources audited:
  none contains its motion (sheets in outputs/source_check/). Rule from now on: inspect every generated source before screening.
- screening verdicts (interactive node rg31802): champagne static -> room (ours running in allocation 20523721, both/any);
  woman_scream late (room); man_desk late/partial (room); books screening OOMed because a regeneration step landed on the same
  GPU (lesson: one GPU step at a time in an allocation).
- plan: champagne opt -> regenerate beer2/toaster2 (sequential) -> inspect -> screen books/toaster2/beer2 -> new allocation ->
  ours on scream, desk, and whichever of the three has room -> judge sheets -> package to results/user_study_candidates/.
  If < 5 clear wins: bank of new no-water impulsive events (piano lid slams, cat knocks vase off table, bike tips over,
  ladder slides and falls, bowling strike, firecracker, soda can crushed, popcorn pops out of the pan, gorilla beats chest,
  elephant raises trunk and trumpets, horse neighs, kettle whistles with steam).

- 19:40 champagne (both/any) finished: critic 0.04-0.08 for 11 iterations, previews identical to the baseline -> failed; the cork is
  a few pixels in this framing. Queued a close-up variant (gen_champagne_close: neck fills the frame, edit = cork shoots out + foam
  sprays). Candidate package with sheets kept in results/user_study_candidates/gen_champagne/ for review.

- 20:15 interactive-node loading stall diagnosed: processes sit in Lustre `cl_sync_io_wait` while memory-mapping the 100 GB
  checkpoint (15-25 min per process; the t2va step needed 24 min for one clip). Fix: the runner now copies the H3 checkpoint
  and Qwen to the node-local NVMe once per allocation (~2 min) and every step loads from there (H3_CKPT/H3_QWEN overrides).
- beer source attempt 2 also pours in the first third -> gen_beer_pour3 with nobody in the source; toaster2 regenerates with
  the champagne close-up in the next GEN step.

- 21:35 sources verified (sheets): gen_champagne_close (neck fills frame, cork in, still), gen_toaster2 (empty slots, lever down),
  gen_beer_pour3 (full glass + closed bottle, nobody) -> all valid; screening. Local-NVMe staging + index-path rewrite brought the
  per-clip generation back to ~4 min (was 24 min).

- 22:16 screening: champagne close-up baseline already pops/foams (0.95/0.98) -> no room; toaster2 0.37/0.42 -> room, ours queued;
  beer3 not reached (the 3-candidate screening step hit its 43-min timeout: ~14 min per candidate on this node even from local
  disk) -> re-queued first. Allocation rolled over automatically (20568376 on rg32201, staging).

- 23:30 gen_woman_scream (both/any): iter 2 = a clear wide-open scream mid-clip (critic 0.95, perceptual 0.04) vs a baseline that
  opens the mouth only in the last frame -> candidate win (package in user_study_candidates/gen_woman_scream/, B_best_iter02).
  Iter 3 drifts into the woman running down the corridor (perceptual 0.34, excluded); the optimization then collapses back to
  the baseline for iters 4-12 (critic 0.01) - the usual early-peak pattern.

- 23:35 gen_beer_pour3 baseline: H3 adds a hand, pours and overflows the glass on its own (0.95/0.94) -> no room. Beer is done
  (3 source attempts; the valid one is handled by H3). Ours running on gen_man_desk; books and toaster2 follow.

- 00:40 gen_man_desk (both/any): critic 0.16 -> 0.23; previews show only a raised fist (earlier in iters 3-4) but never the slam
  on the desk; iters 6-7 drift to a different person -> failed. Package kept in user_study_candidates/gen_man_desk/.

- 01:35 gen_books_shelf (both/any): the critic is saturated (baseline 0.90), best-by-critic = baseline; visually iter 2 makes the
  collapse start ~2 frames earlier with more books falling (perceptual 0.11) - a modest "sooner and fuller" case, not a
  striking win. Packaged with iter 2 as a candidate.

- 02:20 gen_toaster2 (both/any): critic flat 0.01-0.04; previews: the toast disappears in iters 2-5, iters 6+ zoom-drift -> failed.
  Round-6 tally: woman_scream = candidate win; books = modest improvement; champagne (x2), toaster (x2), beer (x3), fist slam,
  yoga = no room or failed. Bank feed started: piano lid, cat+vase, bicycle, ladder (generating).

- 02:30 bank sources verified: piano (lid open, still), cat + vase (the vase stands on the floor next to the cat, not on a
  table -> edit changed to "pushes the vase over and it shatters on the floor"), bicycle (leaning on a tree, still),
  ladder (against a wall, still) -> all valid; screening on allocation 20625821 (rg31607).

- 02:55 bank screening (rg31607, fast node: 4 candidates in 20 min): piano lid closes slowly at baseline (critic missed it) -> ours
  cancelled; cat pushes the vase over at baseline -> skip; ladder slides and falls at baseline -> auto-OPT removed; bicycle
  falls late (frames 9-12) -> ours queued. Lesson: H3 handles "object falls/tips" edits; the critic under-scores them.

- 03:00 bank extended and reordered for the next feeds (learning from tonight: plain object falls/tips are handled by H3;
  shatters, bursts and face bursts are where it fails): firecracker, kettle steam, plate slides off counter and shatters,
  car window shatters, girl's gum bubble pops, man's spit-take, picture frame falls, light bulb bursts, then gorilla, horse
  neigh, popcorn, soda can, elephant, bowling.

- 03:45 gen_bike_tips (both/any): the optimization removes the late fall (iters 2-4 stay upright, critic 0.003) and drifts into
  texture garbage from iter 5 -> failed. Bank feed #2 generating: firecracker, kettle, plate, car window.

- 03:50 bank feed #2 sources: firecracker already sparks and smokes in the source (only the bang is missing -> weak scenario);
  kettle (still), plate (close-up on the counter, still), car window (the sedan drives slowly along the street, window
  intact) -> valid. Screening.

- 04:10 bank feed #2 screening: kettle static (room); plate slides off and shatters late (modest room); car window shatters at
  baseline (critic missed it) -> OPT removed; firecracker baseline = smoky source, no bang -> ours running but the scenario is
  weak (A already smokes). New allocation 20630380 (rg31607).

### 2026-09-09 04:20 - user review: none of the overnight candidates is good enough
- user: run ours on car window shatter, cat + vase, firecracker, ladder, in that order, with more early-stop patience
  (EARLY 10 -> 14). Firecracker run interrupted and re-queued third; bank auto-feed paused (no_bank) so this order holds.

- 04:35 user idea: same car source, but the edit asks for the WHOLE car to shatter into glass shards (a drastic geometry change
  H3 is unlikely to do on its own). Added as gen_car_shatter (bank), queued right after the car-window run: screen, then ours.
  Auto-queuing disabled (no_auto) - I decide manually now.

- 05:30 gen_car_window (both/any, patience 14): iter 3 = the windows blow out and the car's side bursts into glass shards from frame
  ~60 (critic 0.96, perceptual 0.13) vs a baseline that only shows a faint crack -> candidate win, our most violent event so far.
  Iters 4-6 similar; iter 7 (perceptual 0.43, over the guard) makes the whole car explode into debris - kept as an extra
  candidate file; iters 8-16 hallucinate people (drift). This also previews the whole-car-shatter idea: screening it now.

- 05:40 user correction: in the car-window run the whole car and its doors shatter, not the window -> an over-edit for that
  prompt, not a win. The whole-car-shatter scenario (gen_car_shatter, same source) is the right home for that behaviour;
  its screening + ours are running now.

- 05:35 gen_car_shatter screened: H3's baseline ALREADY does the whole-car shatter on its own - frames 1-3 intact, frame 4 the
  body cracks, frames 5-8 the car bursts into dark chunks, frames 9-16 a debris field with the wheels left standing
  (critic 0.86/0.60; output kept in outputs/overnight/screen/gen_car_shatter_av.mp4). Rule "stop when the baseline already does
  it well" -> our run cancelled 3 min in (rc 137, no results), skip. Lesson: a drastic geometry change is not where H3 fails -
  whole-object destruction is a strong text prior for it (the car-window over-edit was that prior taking over). Queue continues
  with cat + vase, firecracker, ladder (new allocation 20634712 on rg31702).

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
| boy_splashes | retake input | splashes water with both hands | unnatural (stands up, 0.38/0.63) | iter 9: clear two-handed splash mid-clip (alt 7/8/10) | PACKAGED |
| man_claps | retake input | claps hands | partial (hands meet once, 0.14/0.56) | iter 2: sustained clapping but shot zooms out (perc 0.52); iter 4 changes identity | RESERVE |
| eagle_head_turn | retake input | turns head to camera | unclear, subject tiny (0.28/0.63) | - | skip |
| man_nods | retake input | nods head | not visible (0.22/0.67) | - | maybe |
| dog_tilts_head | retake input | tilts head | static (0.46/0.51) | - | maybe |
| dog_stands_up | retake input | stands up on all fours | fail (0.001, stays seated) | flat critic, no change in 8 iters | failed |
| monkey_jumps_branch | retake input | jumps to another branch | fail (0.0004, only head turn) | cancelled (flat critic) | skip |
| car_drives_out | retake input | drives out of the garage | late/weak (0.006, moves in last frames) | lin: flat, no change | failed |
| cat_stretches | retake input | stretches front legs | static but critic says 0.77 lin | cancelled (critic saturated) | skip |
| man_celebrates | retake input | raises both arms | late (last 3 frames, 0.21/0.06) | iter 4: arms up but no expression - user: not convincing | rejected |
| rose_sways | retake input | sways in the wind | fail (0.016, static) | lin: critic falls monotonically, no motion in any iteration | failed |
| turtle_walks | retake input | walks forward | fail (0.07, static) | lin: no motion in any iteration (slight camera drift only) | failed |
| bird_hops | retake input | hops along the branch | fail (0.14/0.27, wing flutter only) | lin: identical to baseline | failed |
| man_laughs | retake input (groom) | laughs out loud | success (0.95/1.0) | - | skip |
| man_gasps | retake input | gasps, mouth open | success (0.82/1.0) | - | skip |
| child_laughs | retake input | laughs out loud | success (0.75/0.98) | - | skip |
| man_yawns | retake input | yawns widely | success (0.94/1.0) | - | skip |
| cartoon_boy_laughs | retake input | laughs out loud | success (0.97/1.0) | - | skip |
| monkey_screams | retake input | opens mouth and screams | success (mid-clip scream) | - | skip |
| dog_barks | retake input | barks loudly | late (mouth only in last frames, 0.008/0.90) | any: baseline already 0.89 any, iters 1-3 identical, then collapse | failed |
| man_sneezes | retake input | sneezes | fail (raises hands instead, 0.06/0.04) | any: flat 0.03-0.05, frames unchanged (diff 0.014) | failed |
| cat_meows | retake input | meows loudly | partial (brief early meow, 0.23/0.20) | any: best 0.31 at iter 3, frames unchanged (diff 0.01), then decline | failed |
| gen_horse_neighs | H3 t2va | neighs, head raised | fail (only walks, 0.02/0.13) | held (3 GPUs) | hold |
| gen_woman_sneezes | H3 t2va | sneezes | late/partial (hand to face, 0.006/0.23) | held (3 GPUs) | hold |
| gen_lion_rock | H3 t2va (r4) | roars | success (0.95) | - | skip |
| gen_duck_pond | H3 t2va (r4) | flaps and splashes | success (0.84) | - | skip |
| gen_dog_rug | H3 t2va (r4) | barks | success (0.59/0.98) | - | skip |
| gen_girl_cake | H3 t2va (r4) | blows out candles | success (0.71) | - | skip |
| gen_koi_pond | H3 t2va (r4) | jumps out of water | fail (swims only, 0.001) | user: best (iter 6 latents) is good, iters 4/6 candidates - critic stayed flat (critic miss) | PACKAGED |
| gen_dolphin_sea | H3 t2va (r4) | leaps out of water | fail (fin only, 0.001) | user: iter 2 = full leap with splash; iter 9 weaker candidate; iters 3-4 drift to a bird / a dog (wrong animal) | PACKAGED |
| gen_wolf_hill | H3 t2va (r4) | howls, head up | user: iter 3 is good (trimmed-source baseline also howls; judge the videos) | PACKAGED |
| gen_woman_desk | H3 t2va (r4) | yawns widely | fail with the 124-f clip (0.04); trimmed-source baseline scored 0.91 | user: iter 2 is good | PACKAGED |
| gen_sealion | H3 t2va (r4) | barks, head raised | late (mouth opens at the end, 0.09/0.87) | user: iter 2 best, iter 3 candidate; run finished: critic best iter 4 (0.94) added as candidate | PACKAGED |
| gen_cow_field | H3 t2va (r4) | moos | fail with the 124-f clip (0.12/0.38); trimmed-source baseline scores 0.88 (moos) | user: iters 3, 4, 7 ok/good (iter 3 primary) | PACKAGED |
| gen_goat_field | H3 t2va (r4) | bleats | fail (static, 0.23/0.36) | any: drift from iter 3 (perceptual 0.5-0.66), critic rises only inside the drift; sheet kept | failed |
| gen_rooster | H3 t2va (r4) | crows, head back | partial (silhouette, 0.25/0.65) | any: best iter 2 (0.63) but frames ~unchanged (diff 0.009); drift after iter 7; sheet kept for review | weak |
| gen_man_couch | H3 t2va (r4) | sneezes | fail (0.008) but dark and small | - | skip |
| gen_vase_shelf | H3 t2va (r5) | topples and breaks | success (0.87) | - | skip |
| gen_ceiling_fan | H3 t2va (r5) | starts spinning | success (0.92) | - | skip |
| gen_car_street | H3 t2va (r5) | car door swings open | success (0.96) | - | skip |
| gen_dominoes | H3 t2va (r5) | topple | success (topple early; critic 0.03 missed it) | - | skip |
| gen_man_busstop | H3 t2va (r5) | starts dancing | success (moves/dances; critic 0.015 missed it) | - | skip |
| gen_pond_stone | H3 t2va (r5) | stone splashes | partial success (0.39/0.81) | - | skip |
| gen_glass_edge | H3 t2va (r5) | falls and shatters | partial/unnatural (tips, then vanishes, 0.16/0.14) | iter 3: falls and visibly shatters into fragments (alt iter 8, earlier but duplicated glasses); iters 5/7 drift | PACKAGED |
| gen_blocks_tower | H3 t2va (r5) | tower collapses | fail with the 124-f clip (0.04); trimmed-source baseline scores 0.78/0.92 (collapses) | baseline succeeds; sheet kept | skip |
| gen_balloon_kid | H3 t2va (r5) | balloon pops | fail (0.003) | any: flat 0.01-0.02, one drift spike at iter 8; sheet kept | failed |
| gen_ice_lake | H3 t2va (r5) | ice cracks | fail (0.006) but too subtle | - | skip |
| gen_door_hall | H3 t2va (r5) | door slams shut | fail (stays open, 0.02/0.12) | trimmed source: baseline 0.37, no improvement, drift from iter 4; sheet kept | failed |
| gen_elevator | H3 t2va (r5) | elevator doors open | fail (stay shut, 0.03) | any: flat 0.03-0.11, no change; sheet kept | failed |
| gen_woman_studio | H3 t2va (r5) | dances, spins | fail (0.003) but subject small | - | skip |
| gen_windmill | H3 t2va (r5) | blades start turning | fail (static, 0.14/0.63) | user: iters 3 and 14 ok (iter 3 primary) | PACKAGED |
| car_door_opens | retake input (workshop) | car door swings open | fail (0.04/0.24) | iter 4: door swings open mid-clip, clean (alt 3/5) | PACKAGED |
| car_lights_flash | retake input (garage) | headlights flash, horn | fail (0.005) | any: flat 0.03, drift after iter 10 | failed |
| gen_champagne | H3 t2va (r6) | cork pops out | fail (completely static, 0.08/0.15) | ours: every iteration identical to the baseline (cork ~5 px, no gradient) | failed |
| gen_champagne_close | H3 t2va (r6) | cork shoots out, foam sprays (close-up) | baseline 0.95/0.98 (H3 does it at this framing) | - | skip (verify sheet) |
| gen_toaster2 | H3 t2va (r6) | toast pops up | baseline shows toast sitting up from frame 1, no pop (0.37/0.42) | ours: toast vanishes (iters 2-5), then zoom drift; no pop motion | failed |
| gen_beer_pour3 | H3 t2va (r6) | pours until overflow (no person in source) | baseline: a hand appears, pours, glass overflows with foam in the last third (0.95/0.94) | - | skip (H3 does it) |
| gen_toaster | H3 t2va (r6) | toast pops up | INVALID SOURCE (toast already up) -> gen_toaster2 | - | redo |
| gen_beer_pour | H3 t2va (r6) | pours until overflow | INVALID SOURCE x2 (pouring in the clip even with a 'not pouring' prompt) -> gen_beer_pour3 (no person in the source) | - | redo |
| gen_woman_scream | H3 t2va (r6) | screams in fright | late (mouth opens only in the last frame, 0.52/0.42) | iter 2: wide-open scream mid-clip for ~1 s (0.95); iter 3 drifts (she runs away), iters 4+ collapse to baseline | CANDIDATE WIN |
| gen_man_desk | H3 t2va (r6) | slams fist on desk | late/partial (fist up at frame ~11, no clear slam, 0.08/0.17) | ours: only a raised fist at various times (iters 2-5), never a slam; iters 6-7 swap the person | failed |
| gen_books_shelf | H3 t2va (r6) | books fall off shelf | late (topple only in the last 2 frames, 0.93/0.91) | iter 2: cascade starts at frame ~70 and more books fall (earlier, fuller); iters 3/5 fling a single book; iter 7 zoom drift | modest improvement |
| gen_yoga | H3 t2va (r6) | yoga flow: arms overhead, forward fold, rise | success at baseline (full flow, 0.81/0.89) | - | skip (no room) |
| gen_piano_lid | bank | lid slams shut | baseline closes the lid slowly but completely (frames 4-7); critic missed it (0.12/0.02) | ours cancelled (no room) | skip |
| gen_cat_vase | bank | pushes the vase over, it shatters | baseline does it (0.97/0.99) | - | skip |
| gen_bike_tips | bank | bicycle tips over | baseline falls late (frames 9-12, 0.78/0.40) | ours: iters 2-4 remove the fall, iters 5+ drift | failed |
| gen_ladder | bank | ladder slides and falls | baseline does it mid-clip (0.67/0.20, critic miss) | auto-OPT removed | skip |
| gen_firecracker | bank | explodes with a bang and smoke | source already sparks/smokes; baseline = source, no bang (0.07/0.49) | ours running (weak scenario) | opt |
| gen_kettle | bank | whistles, steam shoots out | baseline static, no steam (0.02/0.10) | ours queued | opt |
| gen_plate_counter | bank | plate slides off and shatters | baseline does it late (slides at frame ~9, shatters; 0.66/0.13) | ours queued (late case) | opt |
| gen_car_window | bank | side window shatters | baseline: only a faint crack pattern (0.10/0.06) | iters 3-7 shatter the whole car and doors, not the window (over-edit) | not a win (user) |
| gen_car_shatter | bank (car_window source) | the whole car shatters into glass shards | baseline does it, early and complete (cracks at frame 4, debris field by frame 9; 0.86/0.60) | ours cancelled (stop rule) | skip |
| dog_runs_off | retake input | gets up and runs off | fail (0.04/0.06), dog small in cluttered scene | - | skip |
| gen_jeep_drives | H3 t2va | drives forward | weak motion (0.04/0.11) | - | maybe |
| gen_horse_rears | H3 t2va | rears up on hind legs | fail (0.0002, only walks) | cancelled (flat critic) | skip |
| gen_frog_jumps | H3 t2va | jumps off the lily pad | late (leaves frame in last 2 frames, 0.016) | user: full-size 3-GPU run iter 3 is good (critic flat = critic miss); small-source run flat | PACKAGED |
| gen_woman_stands | H3 t2va | stands up from bench | success (stands mid-clip) | - | skip |
| gen_cat_jumps_down | H3 t2va | jumps down from windowsill | success (late but jumps) | - | skip |
| gen_man_drinks | H3 t2va | drinks from mug | success (0.93/0.98) | - | skip |
| gen_drummer_plays | H3 t2va | plays drums | success (0.73/0.99) | - | skip |
| gen_kid_swings | H3 t2va | starts swinging | unclear, subject tiny (0.55/0.99) | - | skip |
| gen_pigeon_rail | H3 t2va (r2) | takes off and flies | success (0.98/1.0) | - | skip |
| gen_pianist | H3 t2va (r2) | plays the piano | success (0.96/1.0) | - | skip |
| gen_dog_wet | H3 t2va (r2) | shakes water off | fail (only turns, 0.09/0.23) | small (lin): critic flat 0.02-0.07, frames barely change; sheet kept for review | likely failed |
| gen_woman_door | H3 t2va (r2) | opens door, walks in | late (door opens in last frames, 0.32/0.28) | user: iter 4 ok-ish (door opens wider); iter 6+ add extra people (drift) | PACKAGED (ok-ish) |
| gen_basketball | H3 t2va (r2) | bounces | partial (hovers in one frame, 0.17/0.23) | small: critic 0.13 -> 0.47 at iter 7 but frames ~unchanged (diff 0.008); sheet kept | weak |
| gen_woman_beach | H3 t2va (r2) | waves at camera | success (0.98/1.0) | - | skip |
| gen_umbrella | H3 t2va (r2) | opens umbrella | success (0.98/0.99) | - | skip |
| gen_candle | H3 t2va (r2) | flame blown out | fail (0.0009, steady flame) | held (flat critic) | hold |
| gen_rowboat | H3 t2va (r2) | rocks side to side | fail (0.01, static) | held (flat critic) | hold |
| gen_bell | H3 t2va (r2) | swings and rings | fail (0.06/0.10, barely moves) | held | hold |
| gen_glass_table | H3 t2va (r2) | tips over, spills | late + distorted (glass morphs in last frames) | user: iter 4 is good (critic flat = critic miss); later iterations drift | PACKAGED |
