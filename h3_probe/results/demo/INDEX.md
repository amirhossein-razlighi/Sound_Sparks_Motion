# Demo-video clips (not part of the user study) (A = H3 baseline, B = ours; same source, prompt, noise and 16 steps)

Files per scenario: `source_input_av.mp4` (the model input) + `source_audio.wav`, `prompts.txt` (edit sentence, full H3 prompt, critic question), `A_baseline_av.mp4`, `B_ours_av.mp4` (+ `B_candidate_iterNN_av.mp4` alternates), `meta.json`.
B is the iteration picked by visual inspection of all previews (`scenarios/picks.json`); `B_criticbest_iterNN_av.mp4` is the iteration the critic itself ranked best (when it differs from the pick), for reference.

| slug | edit | picked iter | what changes |
|---|---|---|---|
| real_gatsby_splash | The champagne splashes out of the glass in his hand, spraying up into the air. | 3 (alt: 2) [critic best: 1] | user (09-10 08:35): iter 3 is not bad, demo only. Foam bubbles up and overflows the rim at frames 7-9; iter 2 alternate. Baseline has a brief spray of its own at frames 7-9, so no study pair. |
| real_tom_walk2 | The cat trips and falls flat on his face with a crash, and the yellow pillow he is holding and the red picnic basket tumble out of his hands. | 3 (alt: 5) [critic best: 1] | user (09-10 15:40): iter 3 is cool for the demo, not bad. Tom trips at frames 8-9 with the yellow pillow and the basket tumbling and stays down; iter 5 alternate. Demo only: the baseline with this anchored prompt trips him at the same moment, so no study pair. |
