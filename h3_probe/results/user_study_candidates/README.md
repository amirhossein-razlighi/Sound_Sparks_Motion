# Overnight candidates for the user study (review folder)

One folder per scenario processed by the overnight runner (`h3_probe/scenarios/night_run.sh`, log in `../overnight/night.log`).

Each folder:
- `source_input_av.mp4` / `source_audio.wav` - the raw input given to the model (already checked to NOT contain the target motion)
- `prompts.txt` - edit sentence, scene, full H3 prompt (identical for A and B), critic question, mode/objective
- `A_baseline_av.mp4` - H3 as is (16 steps)
- `B_best_iterNN_av.mp4` - our best iteration by the critic within the perceptual guard; `B_cand_iterNN_av.mp4` - next best iterations
- `critic_scores.txt` - every iteration's critic yes-probabilities and perceptual term (which ones were ranked)
- `iters_sheet_a.jpg` / `iters_sheet_b.jpg` - one row per iteration (top row = baseline), 12 frames each
- `verdict.json` - automatic verdict from the critic (`strong_win` / `improved` / `no_gain`); my visual notes are in `../H3_scenarios.md`

The critic is only a guide: it has missed real motion (koi, frog, glass table) and has been fooled by texture drift.
Judge the videos. Scenarios that were rejected at the source-check or baseline stage are listed in `../H3_scenarios.md`
(section "overnight") with the reason; their baseline sheets are in `/scratch/amirrz/H3_exp/outputs/overnight/screen/`.
