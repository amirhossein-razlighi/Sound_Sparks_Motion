# Demo-video clips

Clips where ours looks good enough for the paper's demo video but the pair is NOT a clean user-study sample (the baseline
half-does the edit, the gain is timing-only, or the edit was chosen for looks rather than for an H3 failure).
Kept apart from `../user_study/` on purpose. Picks live in `scenarios/demo_picks.json`; package with

    PICKS_FILE=h3_probe/scenarios/demo_picks.json DST_DIR=h3_probe/results/demo INDEX_TITLE="Demo-video clips" python3 h3_probe/scenarios/package_ab.py

Same layout as the study folders: `source_input_av.mp4` + `source_audio.wav` (model input), `prompts.txt`, `A_baseline_av.mp4`,
`B_ours_av.mp4` (+ `B_candidate_iterNN_av.mp4` alternates), `meta.json`. See `INDEX.md` for the list.
