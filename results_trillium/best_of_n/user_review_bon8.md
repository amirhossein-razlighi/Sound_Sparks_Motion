# User visual review of best-of-8 prompt-only Retake vs paper OURS (2026-11-06)

Grids viewed: INPUT | PAPER OURS | PAPER BASELINE (s42) | Retake seeds 2-6, labelled with in-loop critic yes_prob
(videos: `trillium:/scratch/amirrz/iclr_rebuttal/bon8_compare/<scenario>.mp4`). Comments are the user's, condensed.

| scenario | critic: best seed / ours | user verdict on the critic-best seed vs ours |
|---|---|---|
| red_car_door_opens | 0.687 / 0.024 | ours much more natural and better; best seed = abrupt motion (the door does open) |
| robot_waving | 0.984 / 0.090 | best seed abrupt motion; ours natural |
| turtle_neck | 0.953 / 0.378 | best seed completely artifact-y, even funny |
| monkey_reaching_fruit | 0.622 / 0.005 | best seed: fruit suddenly appears in the monkey's hand |
| man_pets_dog | 0.977 / 0.905 | best seed: dog identity collapses in the last frames |
| falcon_bird_opening_wings | 0.971 / 0.984 | seeds: abrupt motion and collapse |
| groom_raising_hand | 0.989 / 0.990 | seeds: abrupt, weird, unnatural motion |
| goldfish_jumping | 0.378 / 0.469 | seeds: a fish that should jump OUT of the tank is instead INSERTED into the tank |
| bugatti_lights_flash | 0.016 / 0.005 | seeds: lights flash only once (prompt asks for periodic flashing) |
| red_rose_blooming | 0.018 / 0.026 | seeds fail (no bloom) |
| dog_yawning | 0.990 / 0.967 | (no comment) |
| man_climbs_rocks | 0.725 / 0.092 | (no comment) |

Takeaway (user): the critic rewards an abrupt, obvious occurrence of the action and does not penalise unnatural motion,
artifacts, identity collapse or object insertion; on visual inspection ours is better than the critic-best prompt-only seed.
