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

## Scenario table (updated as results land)

| slug | source | edit | baseline (screen) | ours | status |
|---|---|---|---|---|---|
| goldfish | retake input | jumps out of the tank | fail (0.004) | clean leap (0.72 any) | DONE (D) |
| turtle_extends_neck | retake input | extends neck out of shell | fail (known) | running 20360026 | opt |
| cat_yawns | retake input | yawns widely | late/weak yawn (known) | running 20360027 | opt |
| boy_crouches | retake input | crouches, touches water | fail (known) | running 20360028 | opt |
| red_bird_opens_wings | retake input | opens wings | fail (known) | queued 20360029 | opt |
| man_shouts | retake input | shouts loudly | fail (known) | queued 20363984 | opt |
| child_jumps | retake input | jumps up and down | fail (0.0007, static) | queued 20363983 | opt |
| man_covers_face | retake input | covers face with both hands | success (0.996) | - | skip |
| boy_splashes | retake input | splashes water with both hands | unnatural (stands up, 0.38/0.63) | - | maybe |
| man_claps | retake input | claps hands | partial (0.14/0.56) | - | maybe |
| eagle_head_turn | retake input | turns head to camera | unclear, subject tiny (0.28/0.63) | - | skip |
| gen_jeep_drives | H3 t2va | drives forward | weak motion (0.04/0.11) | - | maybe |
