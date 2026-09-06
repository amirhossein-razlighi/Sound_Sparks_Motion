# H3 event sound: baseline vs pinned vs plain audio-reference (edit text identical)

Per folder: `baseline_av.mp4` (source sound as audio reference, H3 makes its own soundtrack), `pinned_h3sound_av.mp4` (generated audio latent pinned to H3's own event sound), `audioref_h3sound_av.mp4` (event sound as the audio reference, grammar `reference` - no tricks), `audioref_keep_h3sound_av.mp4` (same, grammar `fully_preserved` + sync instruction), `event_sound_source_t2va_av.mp4` (text-only clip the sound came from). Contact sheets of the interesting pairs in `_sheets/`. Verdicts: H3_FINDINGS.md Sec. 9.

| scenario | base | pinned | audio-ref | audio-ref keep | frame-diff vs base (pin / ref / keep) |
|---|---|---|---|---|---|
| bird_opens_wing_real | 0.113 | 0.110 | 0.112 | 0.115 | 0.008 / 0.015 / 0.014 |
| boy_crouches | 0.117 | 0.108 | 0.118 | 0.093 | 0.055 / 0.061 / 0.061 |
| boy_laughing_outloud | 0.036 | 0.430 | 0.044 | 0.049 | 0.005 / 0.017 / 0.008 |
| bugatti_lights_flash | 0.013 | 0.011 | 0.016 | 0.015 | 0.003 / 0.006 / 0.005 |
| cat_yawns | 0.022 | 0.037 | 0.020 | 0.020 | 0.004 / 0.007 / 0.006 |
| child_waving_hand | 0.864 | 0.938 | 0.897 | 0.873 | 0.012 / 0.021 / 0.018 |
| dog_jumping | 0.016 | 0.001 | 0.007 | 0.005 | 0.003 / 0.013 / 0.010 |
| dog_yawning | 0.012 | 0.005 | 0.087 | 0.052 | 0.006 / 0.016 / 0.016 |
| falcon_bird_opening_wings | 0.129 | 0.133 | 0.127 | 0.140 | 0.027 / 0.048 / 0.047 |
| goldfish | 0.000 | 0.000 | 0.000 | 0.000 | 0.006 / 0.016 / 0.016 |
| groom_raising_hand | 0.138 | 0.145 | 0.139 | 0.136 | 0.008 / 0.024 / 0.022 |
| man_laugh | 0.190 | 0.360 | 0.173 | 0.182 | 0.006 / 0.055 / 0.051 |
| man_pets_dog | 0.090 | 0.088 | 0.145 | 0.096 | 0.005 / 0.010 / 0.007 |
| man_raises_hand_real | 0.838 | 0.837 | 0.857 | 0.845 | 0.009 / 0.044 / 0.036 |
| man_shouts | 0.015 | 0.019 | 0.075 | 0.044 | 0.007 / 0.071 / 0.065 |
| monkey_reaching_for_fruit | 0.090 | 0.071 | 0.033 | 0.035 | 0.038 / 0.117 / 0.122 |
| red_bird_opens_wings | 0.029 | 0.034 | 0.028 | 0.062 | 0.007 / 0.140 / 0.141 |
| red_car_door_opens | 0.129 | 0.110 | 0.032 | 0.034 | 0.003 / 0.016 / 0.015 |
| red_car_door_real | 0.001 | 0.001 | 0.002 | 0.001 | 0.003 / 0.010 / 0.008 |
| red_rose_blooming | 0.024 | 0.026 | 0.010 | 0.004 | 0.006 / 0.035 / 0.035 |
| robot_waives | 0.163 | 0.143 | 0.430 | 0.287 | 0.004 / 0.037 / 0.035 |
| surprised_man | 0.173 | 0.168 | 0.035 | 0.002 | 0.006 / 0.015 / 0.017 |
| turtle_extends_neck | 0.004 | 0.004 | 0.006 | 0.006 | 0.004 / 0.026 / 0.026 |
