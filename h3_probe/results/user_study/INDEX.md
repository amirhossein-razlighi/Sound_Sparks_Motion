# User-study A/B pairs (A = H3 baseline, B = ours; same source, prompt, noise and 16 steps)

Files per scenario: `source_input_av.mp4` (the model input) + `source_audio.wav`, `prompts.txt` (edit sentence, full H3 prompt, critic question), `A_baseline_av.mp4`, `B_ours_av.mp4` (+ `B_candidate_iterNN_av.mp4` alternates), `meta.json`.
B is the iteration picked by visual inspection of all previews (`scenarios/picks.json`).

| slug | edit | picked iter | what changes |
|---|---|---|---|
| gen_plate_counter2 | The plate slides off the edge of the counter and drops onto the floor without breaking. | 3 | user (09-10 10:40): ok-ish, not bad. Ours tips the plate off at frames 2-3 and it lies intact on the floor from frame 4; baseline slides it off only in the last frames (10-12). Critic missed it (0.05); perceptual 0.32 (over the guard - fall edits move many pixels). |
