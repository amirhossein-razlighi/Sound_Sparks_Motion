# User-study A/B pairs (A = H3 baseline, B = ours; same source, prompt, noise and 16 steps)

Files per scenario: `A_baseline_av.mp4`, `B_ours_av.mp4` (+ `_32steps` re-renders when available), `meta.json`.
B is the iteration picked by visual inspection of all previews (`scenarios/picks.json`).

| slug | edit | picked iter | what changes |
|---|---|---|---|
| boy_crouches | The boy crouches slightly and touches the water. | final | boy looks down and touches the water; baseline stays upright |
| cat_yawns | The cat yawns widely. | 3 | wide early yawn (frames ~30-50); baseline yawns only in the last frame. alt: iter 9 |
| goldfish | The goldfish jumps out of the fish tank into the air. | final | clean leap out of the tank; baseline never jumps |
| man_shouts | The man shouts loudly. | 3 | hands come up from mid-clip and an open-mouth shout; baseline raises hands only in the last frames |
