# Mode ablation on the user-study pairs (same source, prompt, noise, recipe, capture; only OPT_MODE differs)

Per scenario: `both/` (A baseline + the packaged B), `text_only/`, `audio_only/` (critic-best checkpoint and the same iteration as the both pick), `compare_sheet.jpg` (rows: baseline, both, text-only, audio-only).
Critic yes = in-loop Qwen yes-probability at the selected checkpoint (indicative only; judge the videos).

| slug | edit | both (iter) | baseline yes | both best yes | text-only best yes (iter) | audio-only best yes (iter) |
|---|---|---|---|---|---|---|
| goldfish | The goldfish jumps out of the fish tank into the air. | final | 0.004 | 0.365 | pending | pending |
| cat_yawns | None | 3 | 0.936 | 0.953 | pending | pending |
| man_shouts | None | 3 | 0.388 | 0.546 | pending | pending |
| boy_splashes | The boy splashes the water with both hands. | 9 | 0.327 | 0.915 | pending | pending |
| car_door_opens | The car door swings open. | 4 | 0.024 | 0.9 | pending | pending |
| gen_dolphin_sea | The dolphin leaps out of the water. | 2 | 0.001 | 0.013 | pending | pending |
| gen_koi_pond | The koi jumps out of the water. | final | 0.001 | 0.003 | pending | pending |
| gen_sealion | The sea lion barks with its head raised. | 2 | 0.077 | 0.78 | pending | pending |
| gen_woman_desk | The woman yawns widely. | 2 | 0.911 | 0.911 | pending | pending |
| gen_cow_field | The cow moos loudly. | 3 | 0.884 | 0.884 | pending | pending |
| gen_woman_door | The woman opens the door and walks inside. | 4 | 0.098 | 0.24 | pending | pending |
| gen_frog_jumps | The frog jumps off the lily pad into the water. | 3 | 0.01 | 0.414 | pending | pending |
| gen_glass_table | The glass tips over and spills the water. | 4 | 0.096 | 0.023 | pending | pending |
