# Steps for Today — April 8th

## Overview

Today is about turning the current QWEN optimization idea into stronger evidence:
understand *when* the frame-level semantics transition from the source prompt to the
edit prompt, reduce artifacts, and curate clean examples where our method clearly beats LTX.

The main working hypothesis is that QWEN can act as a temporally aware semantic guide:
early frames should remain close to the non-edited source prompt, while later frames should
move toward the edit prompt without introducing visual artifacts.

---

## Task 1: Check Frame-to-Frame CLIP Similarity

**Goal:** Measure whether frame CLIP embeddings interpolate between the source-video prompt
and the edit prompt across time.

### What to measure

For each optimized video, compute frame-wise CLIP similarity against:
- **Source prompt:** the prompt used to generate the original/source video
- **Edit prompt:** the prompt used for editing the video

Evaluate this at multiple temporal regions:
- First frames
- Middle frames
- End frames
- Optional: all frames as a continuous curve

### Questions to answer

- Do early frames stay closer to the source prompt?
- Do later frames become closer to the edit prompt?
- Is the transition smooth, or does it jump abruptly?
- Are artifact-heavy frames associated with abnormal CLIP scores or unstable similarity curves?

### Output

Save plots for each video:
- `clip_sim_source_vs_edit_over_time.png`
- Optional: a CSV with columns `[frame_idx, sim_source, sim_edit, sim_delta]`

Use these plots to decide whether our QWEN optimization is creating a real semantic
temporal transition or simply pushing isolated frames toward the edit prompt.

---

## Task 2: Sample 30 Consecutive Frames for QWEN

**Goal:** Test whether QWEN loss improves when it receives a temporally continuous frame
window instead of sparse or uneven samples.

### Experiment

Sample **30 consecutive frames** from the middle of the video as the default QWEN input.
Also test other windows if time allows:
- First 30 frames
- Middle 30 frames
- Last 30 frames
- A sliding 30-frame window

### Why this matters

Continuous frames may help QWEN judge temporal consistency and artifacts better than sparse
sampling. This is especially important for edits where the event unfolds over time, such as:
- Drummer motion
- Yawning
- Door opening
- Object dropping or shattering

### What to compare

For the same experiment and seed, compare:
- Sparse frame sampling
- 30 consecutive middle frames
- 30 consecutive late frames
- Sliding or randomly selected 30-frame windows, if easy to implement

Track:
- `yes_prob`
- visual artifacts
- edit correctness
- temporal smoothness

---

## Task 3: Redesign QWEN Loss as a Temporal Source-to-Edit Objective

**Goal:** Make the QWEN loss explicitly encourage a source-to-edit transition over time.

### Candidate loss idea

Early frames should be similar to the source prompt:

> "Does this video match the original/source prompt?"

Later frames should be similar to the edit prompt:

> "Does this video match the edit prompt?"

Possible implementation:
- Compute QWEN score for the early window against the source prompt
- Compute QWEN score for the late window against the edit prompt
- Combine them with a temporal weighting schedule

Example:
- Frames 0–30%: maximize `P(yes | source prompt)`
- Frames 30–70%: allow interpolation / lower weight
- Frames 70–100%: maximize `P(yes | edit prompt)`

### Key question

Does this reduce unwanted global drift while still producing the desired edit?

### Success criteria

- Early frames remain visually close to the source video
- Late frames clearly express the edit prompt
- Fewer artifacts than the current single edit-prompt QWEN loss
- Higher or comparable `yes_prob` for the edit region

---

## Task 4: Use QWEN to Minimize Artifacts

**Goal:** Test whether QWEN can directly help suppress artifacts, not only encourage semantic
edit success.

### Candidate prompts

Use QWEN as a binary judge for video quality:
- "Is this video realistic and artifact-free?"
- "Does this video contain visual artifacts, distortions, or unrealistic warping?"
- "Is the motion temporally smooth and physically plausible?"

Potential scoring:
- Maximize `P(yes | artifact-free prompt)`
- Minimize `P(yes | artifact-present prompt)`
- Combine with the edit-success QWEN objective

### Experiments

Start with artifact-prone scenarios:
- Drummer
- Fast motion examples
- Cases where background or body shape warps

Compare:
- Edit-only QWEN loss
- Artifact-only QWEN loss
- Edit + artifact QWEN loss

### Risk

The artifact prompt may push the video toward being conservative and reduce the edit strength.
If that happens, try a lower artifact-loss weight or apply the artifact loss only after the
edit starts to appear.

---

## Task 5: Use DINO to Minimize Artifacts

**Goal:** Test whether DINO features can regularize the video and reduce structural artifacts.

### Candidate ideas

- Use DINO frame-to-frame feature consistency as a temporal smoothness term
- Use DINO similarity between source and optimized frames to preserve non-edited regions
- Compare DINO features in background regions if masks or attention maps are available

### What to test first

Start simple:
- Compute DINO feature distance between consecutive optimized frames
- Penalize unusually large jumps
- Compare artifact level with and without this penalty

### Why DINO may help

DINO features may capture object and scene structure better than pixel losses, so they could
discourage shape-breaking artifacts while still allowing semantic edits.

---

## Task 6: Find 10–15 Strong Examples Where Ours Beats LTX

**Goal:** Build a clean comparison set for the paper: examples where our method is clearly
better than LTX and artifact-free.

### Selection criteria

Each example should satisfy:
- Ours has stronger edit correctness than LTX
- Ours has no obvious artifacts
- The comparison is visually obvious without over-explaining
- The source prompt and edit prompt are unambiguous
- QWEN/CLIP scores support the qualitative result

### Priority examples

Start with known or promising cases:
- Dog yawning
- Dog yawns
- Red car door opens
- Guitarist spins 360
- Drummer case, if artifacts can be reduced
- Bottle drops/shatters, if clean
- Cat yawning
- Person sneezing
- Bird spreads wings
- Person stands up

### Output

Create a candidate table:

| Example | Source prompt | Edit prompt | Ours quality | LTX quality | Artifact-free? | Keep? |
|---|---|---|---|---|---|---|

Target: **10–15 keepers** for the main paper or supplementary gallery.

---

## Task 7: Adaptive Focus on Weak Frames

**Goal:** Spend optimization effort on frames that QWEN thinks are not good enough yet.

### Candidate idea

As iterations go forward:
- Compute per-window or per-frame `P(yes)`
- Focus more weight on windows where `P(yes)` is below a threshold
- Reduce weight on windows that are already good enough

Example:
- If `P(yes) < 0.7`, use high loss weight
- If `0.7 <= P(yes) < 0.9`, use medium loss weight
- If `P(yes) >= 0.9`, use low or zero loss weight

### Why this matters

The current optimization may over-edit already successful frames while failing to fix the
harder frames. Adaptive weighting could improve temporal consistency and reduce artifacts.

### First implementation path

Start with window-level scoring rather than per-frame scoring:
- Early window
- Middle window
- Late window

Then optimize the weakest window more heavily.

---

## Task 8: Artifact Reduction Focus Case — Drummer

**Goal:** Use the drummer example as the main artifact-reduction debugging case.

### What to inspect

- Which frames show artifacts?
- Do artifacts align with low QWEN `yes_prob`?
- Do artifacts align with abrupt CLIP or DINO feature changes?
- Does the edit prompt create too much motion pressure?

### Experiments to try

- 30 consecutive middle-frame QWEN sampling
- QWEN edit + artifact-free loss
- DINO temporal consistency loss
- Lower learning rate
- Higher latent regularization
- Adaptive weak-window weighting

### Success criteria

The drummer edit should remain semantically correct while reducing:
- body warping
- background deformation
- flicker
- unrealistic limb motion

---

## Task 9: Novelty Check for QWEN Loss

**Goal:** Determine whether the QWEN loss can be framed as a paper contribution, or whether
similar approaches already exist.

### Questions to answer

- Are there papers using QWEN/VLM `P(yes)` as a differentiable optimization loss for video editing?
- Are there papers using VLM-based binary question answering as an optimization objective?
- Are there papers using CLIP/VLM reward models for test-time video optimization?
- Is our novelty specifically the temporal source-to-edit formulation?
- Is our novelty the use of QWEN for artifact-aware video editing?

### Search terms

Use combinations of:
- "VLM loss video editing"
- "Qwen2.5-VL loss optimization video"
- "visual language model reward video editing"
- "binary question answering loss video optimization"
- "test-time optimization video editing VLM"
- "CLIP loss video editing"
- "VQA reward video generation"

### Expected framing if novel enough

Potential contribution statement:

> We introduce a QWEN-based temporal semantic objective that scores source-to-edit consistency
> over video windows and uses those scores to guide test-time video editing optimization.

If prior work exists, frame ours more specifically:
- QWEN-based instead of CLIP-only
- temporal source-to-edit scheduling
- artifact-aware VLM regularization
- adaptive weak-window optimization

---

## Priority Order for Today

1. Run CLIP source-vs-edit frame similarity analysis on current best examples.
2. Test 30 consecutive middle-frame QWEN sampling.
3. Try the temporal source-to-edit QWEN loss on one strong case and one artifact-prone case.
4. Use drummer as the artifact-reduction debugging target.
5. Start the 10–15 example comparison table against LTX.
6. Do the novelty/literature check for QWEN loss framing.

