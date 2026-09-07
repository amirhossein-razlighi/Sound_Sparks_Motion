# User-study A/B pairs (A = H3 baseline, B = ours; same source, prompt, noise and 16 steps)

Files per scenario: `A_baseline_av.mp4`, `B_ours_av.mp4` (+ `B_candidate_iterNN_av.mp4` alternates, `_32steps` re-renders when available), `meta.json`.
B is the iteration picked by visual inspection of all previews (`scenarios/picks.json`).

| slug | edit | picked iter | what changes |
|---|---|---|---|
| boy_splashes | The boy splashes the water with both hands. | 9 (alt: 7, 8, 10) | two-handed splash with water spraying up mid-clip (frames ~40-65); baseline only stands up out of the water. alt: iters 7, 8, 10 |
| car_door_opens | The car door swings open. | 4 (alt: 3, 5) | the driver door of the car on the lift swings open mid-clip (from frame ~45) and stays open, scene otherwise unchanged; baseline door never opens. alt: iters 3, 5. iters 6+ add extra people/objects (drift). |
| cat_yawns | The cat yawns widely. | 3 (alt: 9) | wide early yawn (frames ~30-50); baseline yawns only in the last frame. alt: iter 9 |
| gen_dolphin_sea | The dolphin leaps out of the water. | 2 (alt: 3, 9) | user pick: iter 2 (really good; perceptual 0.29 so the guard skipped it as 'best'); candidates iter 3 and iter 9 (critic best, 0.27 with perceptual 0.13). |
| gen_koi_pond | The koi jumps out of the water. | final (alt: 4, 6) | user pick: best (final = iter 6 latents) with iters 4 and 6 as candidates; critic stayed flat at 0.003 - a critic miss, the jump is visible to a human. |
| gen_sealion | The sea lion barks with its head raised. | 2 (alt: 3, 4) | user pick: iter 2 best, iter 3 candidate; iter 4 added as candidate (critic best, 0.94); baseline opens the mouth only at the very end. |
| gen_woman_desk | The woman yawns widely. | 2 | user pick: iter 2 (wide yawn). Note: the run's own baseline also scored 0.91 on the critic. |
| goldfish | The goldfish jumps out of the fish tank into the air. | final | clean leap out of the tank; baseline never jumps |
| man_claps | The man claps his hands. | 2 (alt: 3) | RESERVE - clear sustained clapping (frames ~20-110) but H3 zooms out to a wider shot to show the hands (baseline is a tight face shot where hands meet once for ~3 frames). alt: iter 3. Framing change = weaker preservation. |
| man_shouts | The man shouts loudly. | 3 | hands come up from mid-clip and an open-mouth shout; baseline raises hands only in the last frames |
