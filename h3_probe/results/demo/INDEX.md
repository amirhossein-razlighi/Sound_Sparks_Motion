# Demo-video clips (not part of the user study) (A = H3 baseline, B = ours; same source, prompt, noise and 16 steps)

Files per scenario: `source_input_av.mp4` (the model input) + `source_audio.wav`, `prompts.txt` (edit sentence, full H3 prompt, critic question), `A_baseline_av.mp4`, `B_ours_av.mp4` (+ `B_candidate_iterNN_av.mp4` alternates), `meta.json`.
B is the iteration picked by visual inspection of all previews (`scenarios/picks.json`).

| slug | edit | picked iter | what changes |
|---|---|---|---|
| real_gatsby_splash | The champagne splashes out of the glass in his hand, spraying up into the air. | 3 (alt: 2) | user (09-10 08:35): iter 3 is not bad, demo only. Foam bubbles up and overflows the rim at frames 7-9; iter 2 alternate. Baseline has a brief spray of its own at frames 7-9, so no study pair. |
