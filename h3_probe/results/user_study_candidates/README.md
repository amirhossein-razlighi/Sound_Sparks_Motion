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

## Status (2026-09-10, updated 03:30)

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
| gen_lightbulb2 | glowing bulb, static (0.001/0.005) | static in all iterations, critic flat 0.005; iter 7 zoom drift, iter 9 garbage frames | failed (auto, 19:40): static-object dead end |
| real_michael_cry | talking, frown/cry only in the last ~20 frames (0.40/0.61) | iters 4-7: full open-mouthed sob from ~frame 55 with the head dropping (`B_cand_iter06`/`07` biggest, `B_best_iter04` critic pick); iters 10-15 lose the cry | CANDIDATE (auto, 20:55): demo-grade; study pair only if the late baseline frown does not count as crying |
| real_michael_hiccup | talking, static (0.002/0.012) | no hiccup; iters 11-16 swap the mug for a transparent bottle and insert a second person (critic bump at iter 12 = prop swap) | failed (auto, 21:55): no hiccup prior; the liquid clause is served by object substitution |
| real_tony_shout | stares off, mouth opens slightly at frames ~50-60 (0.06/0.24) | iters 8-10: full open-mouthed yell at frames ~45-67, head turning to the camera (`B_best_iter10` critic best, `B_cand_iter09`/`08`); iters 11-16 back to the baseline look | CANDIDATE (auto, 23:00): strong; demo + likely study pair |
| real_leo_couch | sips from the can, then sits (0.004/0.043) | no spit-take, no laugh in any iteration; critic flat 0.02-0.04, best = untouched | failed (auto, 23:58): small figure in a dark wide shot, face too small for a mouth event |
| real_casablanca | laughs and throws her head back on its own (0.71/0.95) | iters 2-4 laugh slightly earlier with a changed framing (perc 0.27-0.35, guard-rejected); iters 8-16 drift to a crowd scene | rejected (auto, 01:00): no room, saturated critic + drift |
| real_casablanca_hat | H3 grabs and throws the hat at ~1-2 s, slight smile after (critic miss: 0.004) | iters 4-5: laugh with the head back in the last third, hat vanishes, brighter re-framed shot (perc 0.41); iters 8-16 drift | user (03:25): iter 4 is good -> in `../user_study/real_casablanca_hat` (iter 5 alternate) |
| real_steamboat_wheel | turns the wheel moderately on its own (0.26/0.55) | every iteration scores lower than the baseline, same mild motion (proxy), best = untouched | failed (auto, 03:00, numbers + proxy; eye recheck pending): Mickey source closed, 0 for 4 |
| real_leo_clap | nearly static, dark (0.02/0.25 critic misread) | nearly static in every iteration (proxy), critic 0.01-0.05, best = untouched | failed (auto, 04:00, numbers + proxy; eye recheck pending): Leo couch source closed, 0 for 2 |
| real_steamboat_wheel | wheel spins wildly and flings Mickey by itself (0.26/0.55) | every iteration dampens the spin (critic 0.14-0.29), best = untouched | rejected (auto, 03:15): no gain, Mickey source closed |

Screened but not optimized: gen_car_shatter (H3 already shatters the whole car), gen_piano_lid (baseline closes the lid),
gen_kettle and gen_plate_counter (baseline fails / late) are being optimized in round 7 (started 11:50), together with new sources - see `../H3_scenarios.md`.
