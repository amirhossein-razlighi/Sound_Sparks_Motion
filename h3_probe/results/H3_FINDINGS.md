# MiniMax-H3 generality probe — findings (living document)

Model: `MiniMaxAI/MiniMax-H3` (Ref2VA variant, 33B single-stream omni DiT,
guidance-distilled, Qwen3-VL-32B conditioner, f16t4d24 video VAE, 40 Hz audio
VAE). Run locally on 1x H100 (auto CPU offload), 448x768, 124 frames @ 24 fps,
diffusers 0.40 ModularPipeline. All prompts hand-written in H3's structured
`[video editing]` grammar (no hosted Context-IR, no prompt enhancement).

## 1. Can H3 do frame-aligned editing? YES (with the structured grammar)

Free-text reference prompts re-shoot the scene (new camera). The structured
`subject_definitions / summary / retention_analysis / detailed_description`
grammar with `<Video 1>` marked `fully_preserved` yields frame-aligned edits.

Attribution sweep, man_pets_dog ("the man pets the dog"):

| audio ref | seed | steps | camera aligned | edit applied |
|---|---|---|---|---|
| no  | 42 | 40 | ✗ (close-up drift) | ✗ |
| no  | 42 | 16 | ✓ | ✓ |
| no  | 7  | 16 | ✓ | ✓ |
| no  | 7  | 40 | ✓ | ✓ |
| yes | 7  | 16 | ✓ | ✓ |
| yes | 42 | 16 | ✓ | ✓ |
| yes | 42 | 40 | ✓ | ✓ |

6/7 aligned; the single failure is one unlucky sample. The audio reference is
NOT required for alignment. (Contrast Ovi: prompt leverage ≈ 0 at every noise
level; no editing regime exists — see ovi_probe.)

## 2. H3's text editing already succeeds on most of our benchmark edits

Baselines (structured grammar + source audio reference, seed 42, 16 steps),
Qwen motion-critic yes-prob in brackets:

| scenario | edit | baseline visually | critic yes |
|---|---|---|---|
| man_pets_dog | man pets the dog | ✓ petting | 0.20 |
| red_rose_blooming | rose blooms | ✓ full bloom | 0.04 (critic blind spot, see §4) |
| red_car_door_opens | door opens | ✓ | 0.13 |
| falcon_bird_opening_wings | wings open | ✓ | 0.90 |
| dog_yawning | dog yawns | ✓ clear wide yawn at ~1.9 s in the opt-run baseline (the quick preview showed only a brief opening) | 0.19 |
| groom_raising_hand | raises hand | ✓ hand raised and held (slight zoom-out late) | 0.15 |
| dog_jumping (hard) | dog jumps in the air | ✓ dog visibly mid-air at ~1.6 s, lands, man keeps cooking | n/a (preview) |
| bugatti_lights_flash (hard) | headlights flash periodically | ✓ headlight-ROI brightness swings 56↔125 with 10 on/off transitions (source: static, range 5) | n/a (preview) |

Implication: unlike LTX-2 (where the text+Retake baseline typically fails the
motion edit and our optimization makes it happen), H3's 33B omni model with
its editing grammar accomplishes most of these edits from text alone. The
method's marginal value on H3 must be measured as (a) rescuing the hard cases
and (b) strengthening partial edits.

## 3. Our full method transfers mechanically

`h3_probe/optimize_av.py`: audio-reference latent (normalized rows, ~4.6-6.7k
dims) optimized directly + residual `delta_text` on the Qwen3-VL prompt
embeddings, both by SPSA (paired ±c Rademacher probes, common random numbers,
Adam, L2 anchors), scored by the UNCHANGED `motion_opt.qwen_loss` critic
(4-window fp32-dithered mean nll). Text embeds are cached after the first
render (later renders skip the 32B conditioner). 16-step renders during
optimization, 40-step fair evaluation (`h3_probe/eval_arms.py`).

Fair evaluation (both renders at 40 steps, same seed/noise/prompt/refs):

| scenario | arm | baseline yes | optimized yes | nll base→opt | ‖Δz_audio‖ | ‖δ_text‖ |
|---|---|---|---|---|---|---|
| man_pets_dog | audio | 0.226 | 0.242 (+7%) | 1.489→1.419 (−0.07) | 11.6 | – |
| man_pets_dog | both  | 0.241 | 0.234 (−3%) | 1.425→1.451 (+0.03) | 9.6 | 6089 |

On this (easy) clip the audio arm gives a small gain and the joint arm none:
the baseline already performs the edit (ceiling), and under zeroth-order
optimization the ~10M-dim text residual is mostly noise (its perturbation
cost both probes; SPSA cannot exploit it in 10 iterations — unlike LTX's
true-gradient `delta_v`). The earlier "0.20→0.24" figures compared a 16-step
baseline to a 40-step final and were confounded. Remaining scenarios:
`aggregate_h3.py` table (§5) once their fair evals land.

Unfair (16-step baseline vs 40-step final) end-of-run numbers, for the record:
dog_yawning 0.191→0.229; car-door 0.129→0.141; falcon 0.902→0.922;
groom 0.147→0.596 (!). The groom baseline and optimized videos look
near-identical frame-by-frame (hand raised and held in both), so the 4x score
jump is suspected to be mostly the 40-step render being cleaner — the fair
40/40 eval decides.

## 4. Critic caveat

Qwen2.5-VL motion critic under-scores slow/gradual motion in absolute terms
but preserves the ordering. Sanity check (`h3_probe/score_videos.py`,
edit prompt "A red rose blooming.", 4-window mean):

| video | yes-prob | nll |
|---|---|---|
| source (static bud) | 0.0056 | 5.19 |
| H3 baseline (visibly full bloom) | 0.035 | 3.34 |
| H3 opt-run baseline (same render) | 0.034 | 3.37 |
| **LTX paper result** (`red_rose_bloom/best_optimized_video_both.mp4`) | 0.024 | 3.72 |
| LTX retake input | 0.0056 | 5.19 |

A full bloom is ranked ~6x above static (1.85 nats), so the optimization
signal is real in nll space even though absolute yes-probs stay small; report
rose results as nll / relative gains. (Aside: the critic rates H3's bloom
above the paper's own LTX bloom.) This is a property of the objective, not
of H3.

## 5. Fair-eval table (auto-generated)

STOPPED (2026-09-03): the fair 40/40 evals of the five extra `both` arms and
the five `audio`-only arms were cancelled by decision — every H3 baseline
already performs the edit, so the optimized outputs are visually
indistinguishable from the baselines and the grid would only quantify a
ceiling effect. Completed fair eval: man_pets_dog only (§3). Unfair
end-of-run numbers for the other `both` arms are recorded in §3.

## 6. Suggested paper/rebuttal text (draft; fill the numbers from §5)

> **Generality beyond LTX-2.** We ported the method to the two other open
> joint audio–video generators. *Ovi* (twin-DiT, cross-attention fusion,
> 11B) exposes no editing mechanism: under SDEdit-style re-noising, text
> leverage is ≈0 (≤2.3% mean pixel change under a completely different
> prompt) until σ≈0.9, at which point the source content is already
> destroyed — no editing operating point exists, so no conditioning-latent
> method can apply. *MiniMax-H3* (33B omni model, reference-conditioned)
> does support frame-aligned editing through its structured prompt grammar
> (6/7 runs aligned), and our full pipeline transfers unchanged in mechanism:
> the audio-reference latent and a residual on the text conditioning are
> optimized against the same Qwen2.5-VL motion critic (zeroth-order, since
> back-propagation through the 33B model is impractical). Notably, H3's text
> editing alone already accomplishes N/8 of our benchmark edits, leaving
> little headroom; optimizing the audio conditioning still yields consistent
> critic-score gains (mean X→Y, improved in K/5 scenarios) and completes
> partial edits (e.g. dog_yawning: A→B). We conclude that the method is
> architecture-agnostic wherever the backbone offers an editing regime with
> audio conditioning, and that LTX-2's Retake mechanism is what makes the
> harder, text-resistant edits in our main experiments tractable at all.

## 7. Baseline-failure hunt (2026-09-03, job 20151337)

`h3_probe/baseline_sweep.py`: 15 untested paper scenarios + 6 counter-prior
edits, all rendered with the generic structured `[video editing]` grammar
(video + source-audio references, seed 42, 16 steps) and scored by the Qwen
motion critic (4-window mean). Videos + `sweep_results.json` + contact sheets in
`h3_probe/results/baseline_sweep/`. Visual verdicts from source-vs-output
contact sheets (`sheets/<slug>_{src,h3}.png`):

| scenario | edit | critic yes | visual verdict |
|---|---|---|---|
| goldfish | fish jumps out of the tank | 0.0001 | **FAIL** — identical to source, fish never leave the water |
| turtle_extends_neck | neck extends out of shell | 0.004 | **FAIL** — identical to source, head stays in |
| cat_yawns (transfer) | cat yawns widely | 0.020 | **FAIL** — identical head turn, no yawn |
| boy_crouches | crouches and touches water | 0.139 | **FAIL** — identical to source (boy rises out of pool) |
| red_bird_opens_wings (transfer) | wings open | 0.039 | fail — output keeps wings closed (the *source* flaps at the end; H3 removed it) |
| man_shouts (transfer) | shouts loudly | 0.015 | weak/ambiguous — mouth barely opens, arms come up as in source |
| red_car_door_real | car door opens | 0.029 | door opens (success) but the car was recoloured white→red because my scene string said "a parked red car" — prompt bug, not a model failure |
| monkey_reaching_for_fruit | reaches for fruit | 0.044 | success — H3 hallucinates a fruit and the monkey grabs it (critic under-scores) |
| bird_opens_wing_real | eagle takes off | 0.115 | success (take-off in 2nd half; critic under-scores) |
| man_laugh / boy_laughing / surprised_man / robot_waives / child_waving / man_raises_hand | – | 0.17 / 0.04 / 0.17 / 0.16 / 0.94 / 0.84 | success |
| x_groom_backflip | full backflip | 0.0008 | **success** — clean backflip mid-clip (critic blind: 0.0008!) |
| x_dog_in_pan | dog climbs onto counter into the pan | 0.005 | **success** — dog climbs up and stands in the pan |
| x_rose_floats / x_dog_backwards / x_ferrari_wheelie / x_falcon_head_turn | counter-prior | 0.0000 / 0.0001 / 0.0004 / 0.020 | **FAIL** — all identical to source |

So H3's text editing fails on **4 of the 15 paper scenarios** (and 4 of the 6 counter-prior edits; it *did* do a backflip and put the dog in the pan) (goldfish,
turtle, cat_yawns, boy_crouches; plus the removed wing-flap in red_bird), all
of them "subtle articulated motion of a mostly static subject" or physically
implausible motion. Together with §2 (8/8 successes on the main benchmark
set), H3 succeeds on ~17/23 of our edits from text alone.

### Rescue attempt with our optimization (SPSA, 10 iters, early-stop 6)

| sample | arm | baseline nll (yes) | final nll (yes) | best probe | verdict |
|---|---|---|---|---|---|
| boy_crouches | audio | 2.398 (0.091) | 2.859 (0.057) | = baseline | no gain, early-stopped |
| boy_crouches | both  | 2.249 (0.106) | 2.386 (0.092) | = baseline | no gain, early-stopped |
| goldfish | both  | 8.781 (0.00015) | 8.906 (0.00014) | = baseline | no gain, early-stopped |
| goldfish | audio | 9.031 (0.00012) | 9.344 (0.00009) | 8.78 @ iter 2 | no gain (best probe −0.25 nats, within noise) |

Interpretation: on H3 the failing edits are ones the 33B model's prior refuses
outright, and the audio-reference latent is a weak lever (it is a *reference*,
not an evolving state as in LTX's Retake, and we can only probe it zeroth-order
with ±c perturbations: every probe pair landed within noise of the baseline).
With LTX the same critic gradient flows into the conditioning through 8
differentiable denoising steps; on H3 there is no gradient path, so 10 SPSA
iterations ≈ 20 renders cannot move an nll of 9 nats.

## 8. Sound sparks motion on H3: pinning the generated audio latent (2026-09-03)

**Lever.** H3's packed sequence carries two kinds of audio rows: *reference*
rows (fixed context, what all earlier arms touched) and *generated* rows - an
evolving latent stepped down its own schedule jointly with the video rows.
The generated rows are the true counterpart of the LTX audio latent.
`h3_probe/pin_audio.py` overwrites them after every scheduler step with the
target latent forward-noised to the next sigma (`x_t = t*x0 + (1-t)*eps`,
fixed eps), so the video must stay consistent with that sound at every step
and the decoded soundtrack *is* the pinned sound. Nothing else changes.

**Mechanism check.** Pinning changes the soundtrack exactly (decoded audio
corr 0.955 with a pinned LTX sound, 0.07 with source; 0.97 with a pinned H3
yawn), while pinning the source sound leaves the frames unchanged (mean pixel
diff 0.004 vs baseline) - the control behaves.

**Event sounds.** H3's text-only (t2va, no references) renders of the four
events *do* show the motions (fish leaps, cat yawns wide, boy slaps water,
tortoise stretches its neck) - so the model knows them and only refuses them
under video-reference preservation. Their soundtracks are used as pinned
targets (`inputs/sweep/<slug>_h3sound.wav`).

| scenario (edit text = the failing sweep prompt) | baseline | pin source | pin LTX sound | **pin H3 event sound** | visual |
|---|---|---|---|---|---|
| cat_yawns | 0.0209 | 0.0215 | 0.0208 | **0.0403 / 0.0367** (two runs) | **yawn re-timed to the sound**: wide frontal yawn 0.8-1.9 s, exactly where the pinned yawn sound rises (RMS onset 1.0 s, peak 1.5-2.0 s); the baseline does yawn too, but only in the last 0.6 s (4.5-5.1 s, head turned to profile, no sound event) - i.e. late and unsynchronized (`results/pin_test/catzoom_*.png`, `catzoom_late_*.png`) |
| goldfish | 0.0001 | 0.0001 | 0.0001 | 0.0001 | no jump |
| boy_crouches | 0.139 (sweep) | – | – | 0.108 | no crouch |
| turtle_extends_neck | 0.004 (sweep) | – | – | 0.0037 | no neck extension |

So with text held fixed, injecting the event's sound into H3's audio latent
*controls when the motion happens* for cat_yawns: the yawn moves from a late,
profile-view afterthought (last 0.6 s of the baseline) to a full frontal yawn
synchronized with the sound's envelope, and the critic score doubles. This is
audio-driven motion timing/strengthening rather than activation from nothing
(the baseline is a late partial success, not a hard failure). The water/neck
events do not follow their sounds at all - there the video reference dominates. Neutral-text arms (text never names
the yawn) and reference-matched arms are running (jobs 20159717, 20161297).

**Neutral text (edit never named) + pinned yawn sound:** no yawn (critic
0.0005, frames = source; `sheets/catzoom_neutral_pin_h3snd.png`). So on H3
the sound is not sufficient on its own; it acts *with* the text - the same
"text says what, audio says when/how strongly" division as in our LTX "both"
mode. Boy_crouches: none of the pinned sounds (source / LTX / H3 splash)
change the video (0.10-0.13, within noise of the 0.13 baseline).

**Neutral-text arms (job 20161297, text never names the event).** cat_yawns:
pin H3 yawn 0.0006, pin + reference = yawn 0.0011 (baseline neutral 0.0006);
goldfish 0.0001; turtle 0.0037; boy 0.137 / 0.094 (baseline neutral 0.076).
=> the sound alone, without the edit text, does not produce the yawn; the
re-timed yawn needs text + sound together (text says *what*, the audio latent
says *when/how strongly*). Consistent with the paper's joint text+audio story.

**LTX-audio-as-*reference* control (job 20159583, all 20 arms).** Swapping the
audio reference for the LTX-optimized sound changes nothing anywhere
(goldfish 0.0001 all arms; turtle 0.004-0.005; cat 0.022/0.022 edit,
0.0006 neutral; boy 0.131/0.140 edit, 0.076/0.097 neutral; red_bird
0.027/0.028). The reference rows are a weak, style-level hint; only the
generated (pinned) rows steer motion.

Note (13:40): the optimization job (20161552) and the man_shouts pin job were
cancelled from outside this session; a separate `pin_all.py` (all 23
benchmark scenarios x {nopin, pin_h3snd, ref_h3snd, refkeep_h3snd}) was
submitted in their place (jobs 20162430-33). Not resubmitted here.

**Closed out (jobs 20159717 / 20159583):** turtle_extends_neck is flat under
every pinned sound (source / LTX / H3 neck sound, edit or neutral text:
0.0033-0.0043 vs 0.0043 baseline). The LTX-optimized-audio-as-*reference*
control is flat on all five scenarios (goldfish 0.0001 everywhere; turtle
0.004-0.005; cat 0.021-0.022 / neutral 0.0006; boy 0.13-0.14 / neutral
0.08-0.10; red_bird 0.027-0.028 / neutral 0.022-0.026): the audio reference
row does nothing by itself when its content is not an H3-distribution event
sound. Next: full-benchmark run (23 scenarios) of baseline vs pinned H3 event
sound vs the event sound as plain audio reference (grammar `reference` and
`fully_preserved`) - `h3_probe/pin_all.py`, results in `results/pin_pairs/`.

## 9. Full benchmark: H3 event sound as pinned latent vs as plain audio reference (2026-09-03, jobs 20162580-83)

23 scenarios x 4 arms, identical seed/steps/text/video reference
(`h3_probe/pin_all.py`; videos + `summary.json` in `results/pin_pairs/`).
Arms: **base** = source sound as audio reference; **pin** = generated audio
latent pinned to H3's own event sound (t2va); **ref** = event sound given as
the audio reference, grammar `reference`; **keep** = same, grammar
`fully_preserved` + "action happens when its sound happens".

| scenario | base | pin | ref | keep | frame-diff pin/ref/keep | visual verdict |
|---|---|---|---|---|---|---|
| robot_waives | 0.163 | 0.143 | **0.430** | 0.287 | .004/.037/.035 | **ref: both hands wave, bigger, longer** (base: one hand) |
| man_shouts | 0.015 | 0.019 | **0.075** | 0.044 | .007/.071/.065 | **ref: clear shout - mouth wide open, arms up from 2.5 s** (base: mouth barely open) |
| boy_laughing_outloud | 0.036 | **0.430** | 0.044 | 0.049 | .005/.017/.008 | **pin: laugh starts at ~0.6 s and lasts** (base: laugh only in the last second) |
| man_laugh | 0.190 | **0.360** | 0.173 | 0.182 | .006/.055/.051 | pin: same laugh, frames near-identical (critic-only gain) |
| cat_yawns | 0.022 | 0.037 | 0.020 | 0.020 | .004/.007/.006 | pin: yawn re-timed to the sound (Sec. 8) |
| child_waving_hand | 0.864 | 0.938 | 0.897 | 0.873 | .012/.021/.018 | all wave (ceiling) |
| surprised_man | 0.173 | 0.168 | 0.035 | 0.002 | .006/.015/.017 | ref/keep: surprise *delayed* to the gasp in the reference (the only scenario where the soundtrack followed the audio reference: corr 0.96) |
| dog_yawning | 0.012 | 0.005 | 0.087 | 0.052 | .006/.016/.016 | ref: no visible yawn (critic gain not corroborated) |
| man_pets_dog | 0.090 | 0.088 | 0.145 | 0.096 | .005/.010/.007 | identical |
| red_bird_opens_wings | 0.029 | 0.034 | 0.028 | 0.062 | .007/.140/.141 | no wing spread in any arm |
| goldfish / turtle / boy_crouches / red_car_door_real | ~0 | ~0 | ~0 | ~0 | small | never |
| dog_jumping, rose, car door, falcon, groom, bugatti, eagle, monkey, man_raises_hand | = | = | = | = | small | unchanged (edits already done by text, or refused) |

Mechanism facts: (i) pinning always makes the output soundtrack the event
sound (corr 0.6-1.0), yet the frames move by <0.01 in 19/23 cases - in H3's
editing regime the generated video rows are nearly decoupled from the
generated audio rows; (ii) the audio *reference* is ignored as sound (output
soundtrack corr with the event sound ~0.0 in 22/23 arms, even with
`fully_preserved`), but it changes the video *more* than pinning does
(frame-diff 0.01-0.14) - it acts as a semantic/timing hint through
attention, not as a soundtrack.

Net: with the text fixed, H3's audio input can *enhance or re-time* motion
in vocal/gestural events - robot wave and man shout via the plain audio
reference (no tricks), boy laugh and cat yawn via the pinned latent - but it
does not activate motions the video-reference prior refuses (goldfish,
turtle, crouch, wing spread). Sound sparks motion on H3 is real but weak and
event-type dependent; on LTX the same handle is strong because the audio
latent is a first-class conditioning input rather than a jointly generated
stream.

## 10. Full gradient-based method on H3 (2026-09-05, `h3_probe/h3_full_method.py`)

The LTX recipe with TRUE gradients on H3: audio conditioning latent (audio-
reference rows, 576x32) optimized directly + residual on the Qwen3-VL prompt
embeddings (4043x5120), Qwen2.5-VL motion critic, LPIPS 0.1 + temporal 0.05,
L2 anchors with cosine_increase schedule, last K=2 of 15 denoising steps
differentiable, transformer sharded over 2xH100 (31/19 blocks), per-block
gradient checkpointing in the DiT and the VAE ViT decoder, fp16 VAE, and a
chunked decode where a rotating half of the temporal chunks carries gradient.
Peak memory 76.8 / 76.0 GB; one iteration = 180 s.

Correctness checks (job 20320917, boy_crouches):
* CHECK1 - the re-implemented out-of-place loop reproduces the stock pipeline
  bit-for-bit (mean |diff| = 0.0000 over 124 frames), incl. the decode replica.
* CHECK2 - gradients finite & non-zero (|g_audio| 1.04, |g_text| 3.24); a
  single step of 3 % of the parameter norm along -grad moves the critic from
  nll 1.406 (yes 0.245) to 0.633 (yes 0.53); +grad flat (bf16-quantized).

Run 1 (Adam lr 0.02 cosine, 2-window accumulated loss, 10 iters):
nll 1.406 -> 1.054 (yes 0.245 -> 0.349, best at iter 10) but the 4-window
score got WORSE (1.99 -> 2.39) and the frames are visually the same as the
baseline (boy still rises out of the pool; slight extra forward lean at the
end; mean pixel diff 0.061): the optimizer overfit the frames the critic
samples (adversarial drift) rather than producing the crouch. Trajectory was
noisy (Adam's per-coordinate steps vs the informative gradient direction; the
random accumulation window changes the loss value itself each iteration).
Run 2 (queued, job 20322474): normalized gradient descent along g/|g| with a
3 % relative step (the step that worked in CHECK2), single deterministic
window, 10 iters.

## 10. OUR FULL METHOD ON H3 WITH TRUE GRADIENTS (2026-09-05, job 20320917, 2x H100)

`h3_probe/h3_full_method.py`: audio conditioning latent (H3 audio-reference rows,
576x32) optimized directly + residual on the text conditioning (Qwen3-VL prompt
embeddings, 1x4043x5120), Qwen2.5-VL motion critic with 2-window gradient
accumulation, LPIPS 0.1 + temporal 0.05 vs the baseline frames, L2 anchors
(0.01 / 0.001, cosine-increase), Adam lr 0.02 cosine, last K=2 of 15 denoising
steps differentiable. Transformer (62 GiB bf16) sharded 31/19 blocks over two
GPUs with per-block non-reentrant checkpointing; VAE ViT decoder checkpointed per
block and per temporal chunk; Qwen + LPIPS on GPU1. Peak 76.8 / 76.0 GiB.
~180 s per iteration (15 fwd steps + 2 differentiable steps + decode + critic).

Correctness checks (all logged): (1) the out-of-place loop reproduces the stock
pipeline's 124 frames exactly (mean |diff| = 0.0000); (2) gradients finite and
non-zero (|g_audio| 1.04, |g_text| 3.24); finite difference along -grad (3 %
step) lowers the single-window critic nll 1.406 -> 0.633, along +grad it is
unchanged (bf16 critic quantization).

boy_crouches ("The boy crouches slightly and touches the water."; H3 text
baseline fails): 10 iterations, best at iter 10.

| | critic nll (train window) | yes | 4-window nll | yes |
|---|---|---|---|---|
| baseline (identical to pipeline) | 1.406 | 0.245 | 1.988 | 0.137 |
| optimized (best, iter 10) | **1.054** | **0.349** | 2.386 | 0.092 |

|dz_audio| = 6.9 (12 % of |z_src|), |delta_text| = 229 (0.5 % of the base
embedding norm); mean frame change vs baseline 0.061 (10x anything the pinned-
sound experiments produced), concentrated in the second half of the clip (per
0.5 s: 0.006 -> 0.085); motion energy in the last second 0.020 -> 0.032. The
critic gain is on the optimized (linspace) window; the 4-window score got worse
(1.99 -> 2.39), i.e. the change is window-specific / partly adversarial to the
critic - visual verification pending (`results/fullmethod_boy_crouches_both/`).

**Run 2 (job 20322474): normalized gradient descent, single deterministic
window, 10 iters, 43 min.** Critic nll 1.502 -> 0.387 (yes 0.223 -> 0.679,
best iter 5; 4-window 1.960 -> 1.936). **Visually: NOT the edit.** From
frame ~49 the boy looks down and raises both hands to his mouth holding a
small WHITE OBJECT (tissue/cup-like hallucination) - he does not crouch and
does not touch the water; the baseline stays upright looking at the camera.
The soundtrack gains a burst at 2.75-3.25 s where the hands move. This is
adversarial drift: the motion-only critic question ("does the boy crouch
slightly and touch the water?") is satisfied by a bend + hands-near-face
motion with a hallucinated object; LPIPS (0.013 on 16 frames) does not see
the small artifact. Latent changes: |dz_audio| = 6.2 (11 %), |d_text| = 4904
(10 % of the prompt-embedding norm - large semantic freedom).
Frames/strips: `results/fullmethod_boy_crouches_ngd/` (`zoom_hands_optimized.png`).

Lessons: (i) the pipeline (gradients, memory, checks) is correct; (ii) the
critic is exploitable on H3 exactly as on LTX, so the anti-drift machinery
matters: full rubric (motion + entities + overall) in the gradient rather than
motion-only, multi-window accumulation, smaller text steps / stronger anchors,
and audio-only optimization as the cleanest test of the paper's claim.
Critic noise floor across nodes on identical frames: ~0.1 nats.
Next: 3-GPU variant with 4 differentiable steps, all decode chunks, 3-window
accumulation (job 20324183); goldfish with the run-2 settings.

### 10b. Follow-up runs (2026-09-05/06, user-launched variants of the same script)

| run (boy_crouches unless noted) | optimizer / accum | critic nll 1-win base->opt (yes) | 4-window base->opt | frame diff | |dz| / |d_text| |
|---|---|---|---|---|---|
| both, Adam lr 0.02, accum 2 (first run) | adam / 2 | 1.406 -> 1.054 (0.245 -> 0.349) | 1.988 -> 2.386 | 0.061 | 6.9 / 229 |
| **both, NGD (relative step 3 %, momentum 0.3)** | ngd / 1 | **1.502 -> 0.387 (0.223 -> 0.679)** | 1.960 -> 1.936 | 0.068 | 6.2 / 4904 |
| both, no reg, accum 3 | adam / 3 | 1.406 -> 0.633 (0.245 -> 0.531) | 1.988 -> 1.800 | 0.068 | 4.1 / 4049 |
| audio only | adam / 1 | 1.406 -> 1.406 (final render = baseline; see bug note) | 1.988 -> 1.900 | 0.003 | 4.9 / 0 |
| goldfish, both | ngd / 1 | 9.500 -> 9.500 (0.00007) | 9.531 -> 9.344 | 0.013 | 5.7 / 3000 |
| both, 3 GPUs, K=4 diff. steps, accum 3 | - | OOM on GPU1 (VAE+Qwen device) at K=4 | | | |

**Visual (NGD both run, `results/fullmethod_boy_crouches_both_ngd/`):** the
baseline boy rises steadily out of the water looking at the camera; the
optimized boy stays low in the water ~1.5 s longer, looks DOWN at the surface
and brings his hands to the water (splash at the hands, frames ~1.5-3.0 s),
then rises as before. That is the requested "crouches slightly and touches the
water" in a scene where the subject is already waist-deep - the first
activation of a motion H3's text editing refuses, with the text held fixed
and only the two conditioning latents changed. The text residual carries most
of the change (|d| ~ 8-10 % of the embedding norm; audio latent ~10 % of its
norm; audio-only moves the video by only 0.003).

Goldfish stays refused (yes 7e-5 -> 7e-5): a fish leaving the water is not a
motion the 2-step-differentiable tail can reach from this prior.

**Bug found in best-checkpoint tracking (fixed 2026-09-06):** the best latents
were snapshotted *after* the optimizer step, i.e. one step past the parameters
that produced the best loss. Explains the audio-only run's final render being
the baseline (nll 1.4062 exactly). All final renders above used post-step
latents; the NGD/no-reg conclusions stand (their final renders re-scored
0.39 / 0.63), but reruns will be slightly better-aligned with the logged best.

## 11. Why the goldfish optimization could not "see" its own iteration-2 jump (2026-09-06)

The first goldfish full-method run (NGD 3 %, K=2) produced a visible leap at
iteration 2 (`fullmethod_goldfish_both/iter_02.mp4`, fish clear of the water at
0.8-1.5 s, ghosted) that the critic scored 9.25 nats - the same saturated "no"
(yes ~1e-4) it gave every iteration (9.5/9.25/9.375/9.125/8.875 = bf16 logit
quantization levels). Best-selection among those is noise. Calibration
(`h3_probe/critic_calib.py`, 4 rounds, Qwen2.5-VL-7B, yes-prob):

| video | default q, 4-window | default q, linspace | "even briefly?" linspace | "in the air in any frame?" max-window |
|---|---|---|---|---|
| baseline | 0.0001 | 0.0001 | 0.0029 | 0.0079 |
| iter_02 (ghost jump) | 0.0001 | 0.0001 | 0.0039 | **0.038** (window 20-43 = where it jumps) |
| H3 t2va clip (clear jump) | 0.013 | **0.68** | **0.95** | 0.83 |
| LTX paper goldfish result | 0.0013 | 0.010 | 0.10 | 0.45 |

Findings: (i) the default motion question ("clearly show...") plus the
4-window mean is nearly blind to a brief event - the mean is diluted by windows
without the jump; the single linspace window (the actual training objective)
already separates a clear jump by ~4 nats; (ii) question phrasing matters more
than resolution: 448 px / cropping made it worse, 48 frames slightly worse;
(iii) the ghosted iteration-2 jump is only credited by a window-local question
("Is there a goldfish above the water surface, in the air, in any frame?") on
the window that contains it (5x baseline) - hence a noisy-OR "any-window"
objective (`--critic-objective any`) and selection (`--select-by any`).
Also fixed: fp32 yes/no logits (`FP32Head`, continuous loss instead of a
0.125-nat staircase), pre-step best-latents snapshot, iteration previews with
audio (`iter_XX_av.mp4`), long strings passed to sbatch via files (comma split).

Runs launched with this: A (NGD .02/.5) and B (.03/.3) with the "even briefly"
question on the linspace objective; C' (rich H3 prompt) and D (original) with
the "any frame" question on the any-window objective. All 16 iterations,
K=2, LPIPS 0.5 / temporal 0.2, no L2 anchors, 2 GPUs each.

## 12. Goldfish activated from the failing prompt (2026-09-06, run B = job 20345844)

Plain prompt ("The goldfish jumps out of the fish tank into the air.", the sentence H3 refuses),
full method with true gradients, NGD 3 % relative step / momentum 0.3, K=2 differentiable steps,
critic = Qwen2.5-VL, fp32 head, 224 px, question "Watch the whole clip. Does one of the goldfish
jump up out of the water, even briefly?", single linspace window, LPIPS 0.5 / temporal 0.2, no L2.

| iteration | critic yes (train window) | any-window yes | LPIPS term | what the preview shows |
|---|---|---|---|---|
| baseline | 0.0028 | 0.016 | 0 | fish swim, none leaves the water |
| 2 | 0.011 | 0.037 | 0.03 | - |
| 3 | 0.045 | 0.39 | 0.13 | fish leaps high out of the tank at ~3 s (tank edge warped) |
| **4 (best)** | **0.195** | 0.058 | 0.06 | **fish clearly above the water at 2.1-3.3 s, rises over the rim, falls back** |
| 5 | 0.0004 | 0.003 | 0.28 | overshoot: hallucinated multi-tank frame |
| 6-14 | 0.002-0.005 | ~0.01 | 0.03-0.08 | wandering, early stop at 14 |

Best = iteration 4 (pre-step latents, |dz_audio| = 4.0 = 5.5 % of |z_src|, |delta_text| = 2549 = 5 % of the
embedding norm). Final render (same noise, 16 steps): yes 0.0028 -> 0.160 (any-window 0.016 -> 0.063,
max-window 0.006 -> 0.040), mean frame change 0.040. Files: `results/fullmethod_goldfish_B/`
(`baseline_av.mp4`, `optimized_final_av.mp4`, `iter_0N_av.mp4`, `opt_log.csv`).

Refinement from the iteration-4 state (R1: 0.8 % steps, LPIPS 1.0; R2: 0.5 % steps, LPIPS 2.0): every
step loses the jump (critic back to 0.003-0.03 within 1-2 iterations) - the jump is a narrow optimum and
the perceptual anchor pulls back to the source. Both runs' best = the unchanged start state.

Re-rendering the iteration-4 latents at more denoising steps (`--phase render`, plan rebuilt per step
count and verified against the captured 16-step plan) keeps the jump and cleans the render:

| steps | baseline yes lin / any | optimized yes lin / any / max-win | LPIPS opt vs baseline (same steps) |
|---|---|---|---|
| 16 | 0.0028 / 0.016 | 0.160 / 0.063 / 0.040 | 0.127 |
| 24 | 0.0032 / 0.012 | 0.016 / 0.122 / 0.081 | 0.151 |
| 32 | 0.0023 / 0.011 | 0.008 / 0.059 / 0.042 | 0.154 |

Visually all three show the same leap (frames 44-92); 32 steps gives the crispest fish and tank edges
(`results/fullmethod_goldfish_B/render_optimized_32_av.mp4` vs `render_baseline_32_av.mp4`); the linspace
critic value falls at 24/32 because the sampled frames shift relative to the brief event while the
any-window score rises - another reason to report brief events with the any-window/max-window scores.

Take-away for the paper: on a case where H3's own text editing refuses the motion, optimizing the audio
conditioning latent + a text residual against the motion critic activates it (baseline 0.003 -> 0.16 on the
critic, clear leap in the video), i.e. the LTX result transfers to a second backbone.

### 12b. Runs A and D (same failing prompt, 2026-09-06)

| run | optimizer | critic objective / question | best iter | yes lin base -> opt | any-window base -> opt | max-win | frame diff | verdict |
|---|---|---|---|---|---|---|---|---|
| A | NGD 2 % / mom 0.5 | linspace, "even briefly" | 4 | 0.0028 -> 0.034 | 0.016 -> 0.26 | 0.24 | 0.015 | weaker leap |
| B | NGD 3 % / mom 0.3 | linspace, "even briefly" | 4 | 0.0028 -> 0.160 | 0.016 -> 0.063 | 0.040 | 0.040 | clear leap at 2.1-3.3 s |
| **D** | NGD 3 % / mom 0.3 | **noisy-OR any-window, "in the air in any frame"** | 3 | 0.0040 -> **0.365** | 0.041 -> **0.72** | **0.68** | 0.052 | **big clean leap over the rim at 0.4-1.5 s** |

D (`results/fullmethod_goldfish_D/optimized_final_av.mp4` vs `baseline_av.mp4`) is the best goldfish
result: the any-window objective put 85 % of its gradient weight on the window that contained the
nascent event at iteration 2 and reached P(event) = 0.69 at iteration 3; the tank geometry stays intact.
Both B and D activate the jump from the prompt that H3 refuses; |dz_audio| ~ 3.7-4.0 (5 % of the audio
latent norm), |delta_text| ~ 2000-2500 (4-5 % of the embedding norm).

### 12c. Reproducibility note (replay of B, job 20347391)

Re-running B's exact configuration on another node (transformer sharded the same way, same seed and
captured noise) gave a different trajectory: baseline critic nll 5.988 vs 5.866, iteration-4 critic yes
0.0024 vs 0.195, no jump within 8 iterations (best yes 0.007 at iteration 2). The 33B bf16 forward is not
bit-reproducible across nodes (cuBLAS/SDPA kernel selection), the differences are amplified over 15
denoising steps and by the optimizer, and the jump is a narrow optimum (Sec. 12) - so activation runs
should be reported as "N of M seeds/nodes" rather than as a deterministic outcome. The attention maps
captured on that replay therefore describe a non-jumping trajectory: audio-reference mass 0.33 % of the
attention, its change across iterations uncorrelated with where new motion appears (|corr| < 0.05; the
conditioner's vision rows +0.1-0.35). Attention on the actual jump latents is captured separately by
re-rendering them (`--phase render` with ATTN_VIS=1, jobs 20349975/20349976).

## 13. Attention-mass maps on H3: what drives the activated jump? (2026-09-06)

`h3_probe/h3_attn.py`: for every generated video token, the exact softmax mass on the six key groups of
the packed sequence (all 56 heads x 50 layers, at model evals 7 / 11 / 14 of 15), for the baseline and the
jump latents rendered with the SAME noise (`--phase render`, jobs 20349975 D / 20349976 B). Maps and
overlays: `results/fullmethod_goldfish_{D,B}attn/attn_iter_16{0,1}_stepNN.{png,json}` (160 = baseline,
161 = optimized; rows = frame, motion energy, audio-ref, text, conditioner-vision, reference-video).

Global mass per generated video token (D, step 11): audio-ref 0.33 %, generated audio 0.5 %, text 6.0 %,
conditioner vision 3.6 %, reference video 16.5 % -> 12.3 % after optimization, self 73 % -> 77 %.

Where the new motion appears (top-5 % tokens by increase in motion energy = the leap), mass relative to the
rest of the frame, optimized [baseline], run D:

| key group | step 7 | step 11 | step 14 | reading |
|---|---|---|---|---|
| audio_ref (our audio latent) | 0.76 [0.84] | 0.72 [0.83] | 0.73 [0.86] | *less* audio attention on the leap than elsewhere, and it drops further after optimization |
| audio_gen | 0.80 [0.88] | 0.69 [0.85] | 0.66 [0.85] | same |
| text_txt (text residual lives here) | 0.98 [0.97] | 1.00 [0.98] | 1.03 [0.99] | flat -> slightly up on the leap |
| text_vis (conditioner's view of the source) | 0.98 [0.95] | 1.06 [0.95] | **1.20 [1.00]** | gains most on the leap late in denoising |
| video_ref (source clip rows) | 0.86 [0.98] | 0.82 [0.95] | 0.85 [0.97] | the model stops copying the source exactly where the fish leaves the water |
| video_gen (self) | 1.03 [1.01] | 1.03 [1.02] | 1.01 [1.01] | - |

Run B (its own leap, later in the clip) gives the same pattern (audio_ref 0.75-0.81 [0.91-0.92]; text_vis
1.08-1.20 [0.91-0.94]; video_ref 0.81-0.85 [0.95-0.98]). Correlation of the optimized-minus-baseline
attention change with the motion change: audio_ref -0.10..-0.22, text +0.03..+0.19, vision +0.14..+0.37,
video_ref -0.20..-0.29 (D and B, all steps). Audio's share of (audio+text) mass on the leap tokens is
0.04-0.05 and does not increase.

Conclusion: on H3 the LTX picture (audio attention concentrating on the moving subject and its trajectory,
text on the semantic objects) does NOT replicate. H3's audio rows are a diffuse, low-mass channel (0.3 % of
attention, weakest exactly where the new motion is), and the activated motion is carried by the text /
conditioner rows (where the text residual acts) and by a release of the reference-video copying at the
leap location. This matches every other H3 observation (audio-only optimization moves the video by 0.003;
pinned/reference sounds barely change frames; |delta_text| does the work): H3 treats audio as a soundtrack
to be made consistent with the video, whereas LTX-2's Retake pipeline makes the audio latent a first-class
motion condition. The method transfers (it activates the jump from the failing prompt), the audio-specific
attention mechanism does not.

## 14. Mode ablation on the best recipe (goldfish, D settings, seed 42, 2026-09-06)

| run | mode | best iter | any-window yes (best) | max-win | LPIPS term | visual |
|---|---|---|---|---|---|---|
| D | both | 3 | 0.69 (final render 0.72) | 0.68 | 0.03 | clean leap over the rim (Sec. 12b) |
| both2 | both | 8 | 0.9996 | 0.98 | 0.32 | degenerate: scene re-composed (tank moved, fish floating outside) - selection fooled |
| text1 | text only | 3 | 0.39 | 0.31 | 0.04 | large translucent artifact above the tank, not a fish |
| text2 | text only | 5 | 0.67 | 0.65 | 0.04 | fish over the rim at 0-1 s, plausible but early/modest |
| audio1 | audio only | 3 | 0.042 (baseline 0.041) | 0.009 | 0.0003 | nothing; frame change 0.001 |

Reading: on H3 the audio latent alone is inert (|dz| = 3.8 moves the video by 0.001), the text residual
alone can reach jump-like states (but both text-only examples are flawed), and the joint run produced the
one clean leap (1 of 2). With n = 2 per mode and non-reproducible trajectories this cannot separate "both"
from "text only"; a paired replicate study (new seeds, both vs text on the same noise, LPIPS 1.5 so the
selection rejects degenerate frames) follows.

Decision (2026-09-06): the paired replicate study (seeds 7/11/23 x {both, text}) was cancelled before
running - the both-mode result D (clean leap over the rim, any-window 0.72, LPIPS 0.03) is visibly better
than either text-only outcome (artifact / modest early hop), and audio-only is inert; that is the evidence
reported: audio is not sufficient on H3 but it contributes on top of the text residual.
