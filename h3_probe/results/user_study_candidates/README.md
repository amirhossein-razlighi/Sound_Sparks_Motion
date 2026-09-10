# Overnight candidates for the user study (review folder)

One folder per scenario processed by the overnight runner (`h3_probe/scenarios/night_run.sh`, log in `../overnight/night.log`).

Each folder:
- `source_input_av.mp4` / `source_audio.wav` - the raw input given to the model (already checked to NOT contain the target motion)
- `prompts.txt` - edit sentence, scene, full H3 prompt (identical for A and B), critic question, mode/objective
- `A_baseline_av.mp4` - H3 as is (16 steps)
- `B_best_iterNN_av.mp4` - our best iteration by the critic within the perceptual guard; `B_cand_iterNN_av.mp4` - next best iterations (plus any iteration flagged by eye via `FORCE_ITERS`, marked "flagged by eye" in `critic_scores.txt`)
- `critic_scores.txt` - every iteration's critic yes-probabilities and perceptual term (which ones were ranked)
- `iters_sheet_a.jpg` / `iters_sheet_b.jpg` - one row per iteration (top row = baseline), 12 frames each
- `verdict.json` - automatic verdict from the critic (`strong_win` / `improved` / `no_gain`); my visual notes are in `../H3_scenarios.md`

The critic is only a guide: it has missed real motion (koi, frog, glass table) and has been fooled by texture drift.
Judge the videos. Scenarios that were rejected at the source-check or baseline stage are listed in `../H3_scenarios.md`
(section "overnight") with the reason; their baseline sheets are in `/scratch/amirrz/H3_exp/outputs/overnight/screen/`.

## Status (2026-09-10, updated 18:35)

| folder | baseline (A) | ours (B) | verdict |
|---|---|---|---|
| gen_woman_scream | no scream | iter 2: screams mid-clip | rejected (user 04:20: none of the overnight samples good enough) |
| gen_books_shelf | topple in the last 2 frames | iter 2: earlier, fuller collapse | rejected (user 04:20) |
| gen_champagne, gen_toaster2, gen_man_desk, gen_bike_tips | see `../H3_scenarios.md` | - | rejected (user 04:20: none good enough) |
| gen_car_window | faint crack only | iters 3-7 shatter the whole car and doors, not the window | not a win (user) - over-edit |
| gen_cat_vase | knocks the vase over mid-clip (0.99) | iter 5: the cat paws at the vase from frame 2 and pushes it over (`B_cand_iter05`, critic miss) | user: iter 5 good -> in `../user_study/gen_cat_vase` (B_ours = iter 5) |
| gen_firecracker | smoke puff at frame 29, no bang (0.55) | iters 7-11: flash + bang at frame 25 (iter 8 = fireball + dense smoke); iter 9 critic best (`B_best_iter09`) | user: iters 7 and 8 good -> in `../user_study/gen_firecracker` (B_ours = iter 8, candidate = iter 7) |
| gen_ladder | ladder falls late (critic miss 0.04) | every iteration after the first drifts (perceptual 0.4-0.7) | failed |
| real_chaplin_sneeze | fork fiddling, no sneeze (0.12/0.30) | no sneeze in any of 16 iterations, critic flat 0.16-0.30, best = untouched | failed (user 16:45, judged from the videos): wide dark shot, face too small for a facial event |
| gen_tyre | static wheel (0.001/0.009) | static in all 16 iterations, critic flat 0.009-0.016, iters 4/6 layout drift | failed (auto, 18:35): static-object dead end |

Screened but not optimized: gen_car_shatter (H3 already shatters the whole car), gen_piano_lid (baseline closes the lid),
gen_kettle and gen_plate_counter (baseline fails / late) are being optimized in round 7 (started 11:50), together with new sources - see `../H3_scenarios.md`.
