# Next Steps

## Immediate fixes

1. Fix the reported Qwen metric.
   - The implementation currently optimizes `-log P(yes)` but logs `qwen_score = 1 - qwen_loss`.
   - Replace the reported score with the true `yes_prob = exp(-nll)` or log both `nll` and `yes_prob`.
   - Re-rank checkpoints using the true metric rather than `1 - nll`.

2. Replace the single yes/no objective with a harder-to-game loss.
   - The current setup uses one frozen VLM, one binary question, one prompt, and one yes/no token decision.
   - Move to multiple prompt paraphrases, explicit counterfactual negatives, and a margin or pairwise ranking loss.
   - Example:
     - Positive: "dog yawning"
     - Negatives: "dog barking", "dog sitting still", "mouth already open"

3. Separate claims by branch.
   - Present `audio-only`, `text-only`, and `both` as distinct methods rather than just modes.
   - If `both` wins mainly due to the text-delta capacity, state that directly.
   - Avoid over-claiming audio-driven control when the dense text branch may be dominating.

## High-impact research improvements

4. Constrain the text optimization much more.
   - The current dense text delta spans the full Gemma video encoding and has far more capacity than the audio latent.
   - Try one of:
     - Low-rank text delta
     - Sparse token subset updates
     - Prompt tuning with a few virtual tokens
     - Direct discrete token optimization if that is the intended paper claim

5. Add preservation losses.
   - The current regularization appears too weak relative to the main objective.
   - Add source-faithfulness losses such as:
     - Identity / appearance preservation
     - Background preservation
     - Motion locality
     - Flow consistency outside the edited region
     - DINO / LPIPS consistency for non-target content

6. Make the audio branch stronger and fairer.
   - The audio latent is scientifically interesting but may be too weak an actuator in the current setup.
   - Try:
     - More late denoising steps
     - A step schedule over optimization depth
     - A richer audio control subspace
     - Better balancing so audio is not compared unfairly against a much larger text parameterization

## Evaluation improvements

7. Use multiple evaluators.
   - It is fine to optimize with Qwen, but report with separate frozen judges too.
   - Add:
     - CLIP
     - Another VLM
     - Human evaluation if possible
   - This helps address evaluator overfitting concerns.

8. Tighten preprocessing correctness.
   - Make sure the processor resolution matches `--qwen-img-size`.
   - Avoid cached `video_grid_thw` mismatches.
   - Verify the actual generated token IDs for `" yes"` and `" no"` rather than assuming `"yes"` and `"no"` are sufficient.

## Core scientific caveat

The current setup is promising, but it risks reading as surrogate-loss hacking with a powerful hidden prompt perturbation rather than a clean demonstration of audio-driven motion editing.

If only one thing gets fixed immediately:
- Fix the evaluator objective and metric reporting.

If two things get fixed immediately:
- Fix the evaluator objective and metric reporting.
- Constrain the text branch so the causal story remains believable.
