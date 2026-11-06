#!/usr/bin/env python3
"""Build scoring manifests (CSV: id,group,scenario,method,video,edit_prompt,critic_from).

critic_from = run_config.json whose Qwen critic settings define the *loop* critic
for that video (used by loop_rescore.py; held-out scoring ignores it).

  comparisons : 12 scenarios x {input, ours, ltx2_retake, lora_edit, runway_aleph,
                kiwi_edit, uniedit, vace_wan, vace_ltx}   (paper comparison videos)
  controls    : 8 rebuttal scenarios x {zero, random_s42, random_s1, lora_audio_r64,
                lora_all_r64, zvid, ctrl_baseline} + paper ours / paper baseline
  best_of_n   : selected best-of-N picks (after P1 + select_bon.py)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from common import COMPARISONS, OUT_ROOT, PAPER_RESULTS, REPO_ROOT, SCENARIOS, TWELVE, paper_run_dir

CMP_FILES = {
    "input": "input_video.mp4", "ours": "ours.mp4", "ltx2_retake": "baseline_ltx2.mp4",
    "lora_edit": "lora_edit.mp4", "runway_aleph": "runway_aleph.mp4", "kiwi_edit": "kiwi_edit.mp4",
    "uniedit": "uniedit.gif", "vace_wan": "vace_wan1.3.mp4", "vace_ltx": "vace_ltx_video.mp4",
}
CONTROL_SCEN = ["bugatti_lights_flash", "dog_jumping", "dog_yawning", "falcon_bird_opening_wings",
                "groom_raising_hand", "man_pets_dog", "red_car_door_opens", "red_rose_blooming"]
# paper ours run for the 8 control scenarios (dog_jumping is outside the 12; matched by md5 of
# results/videos/dog_jumps/best_optimized_video_both.mp4)
EXTRA_PAPER = {"dog_jumping": "the_dog_jumps_up_into/sweep_exp/rsf5_accum1_modenormal_frames8_lr0.005_lpipsOFF"}
CTRL_VIDEO = {
    "zero": "mode_both/best_optimized_video_both.mp4", "random_s42": "mode_both/best_optimized_video_both.mp4",
    "random_s1": "mode_both/best_optimized_video_both.mp4", "lora_audio_r64": "mode_lora/best_optimized_video_lora.mp4",
    "lora_all_r64": "mode_lora/best_optimized_video_lora.mp4", "zvid": "mode_zvid/best_optimized_video_zvid.mp4",
}
FIELDS = ["id", "group", "scenario", "method", "video", "edit_prompt", "critic_from"]


def _prompt(run_dir: Path) -> str:
    return json.loads((run_dir / "run_config.json").read_text())["args"]["edit_prompt"]


def comparisons():
    rows = []
    for s in TWELVE:
        rd = paper_run_dir(s)
        p = _prompt(rd)
        for m, fn in CMP_FILES.items():
            v = COMPARISONS / SCENARIOS[s][1] / fn
            rows.append(dict(id=f"cmp/{s}/{m}", group="comparisons", scenario=s, method=m, video=str(v),
                             edit_prompt=p, critic_from=str(rd / "run_config.json")))
    return rows


def controls():
    rows = []
    R = REPO_ROOT / "results" / "rebuttal"
    for s in CONTROL_SCEN:
        prd = PAPER_RESULTS / EXTRA_PAPER[s] if s in EXTRA_PAPER else paper_run_dir(s)
        for m, rel in CTRL_VIDEO.items():
            d = R / s / m
            rows.append(dict(id=f"ctrl/{s}/{m}", group="controls", scenario=s, method=m, video=str(d / rel),
                             edit_prompt=_prompt(d), critic_from=str(d / "run_config.json")))
        d = R / s / "zero"  # the controls' own prompt-only baseline (identical across control cells)
        rows.append(dict(id=f"ctrl/{s}/ctrl_baseline", group="controls", scenario=s, method="ctrl_baseline",
                         video=str(d / "baseline_video.mp4"), edit_prompt=_prompt(d), critic_from=str(d / "run_config.json")))
        rows.append(dict(id=f"ctrl/{s}/paper_ours", group="controls", scenario=s, method="paper_ours",
                         video=str(prd / "mode_both/best_optimized_video_both.mp4"), edit_prompt=_prompt(prd),
                         critic_from=str(prd / "run_config.json")))
        rows.append(dict(id=f"ctrl/{s}/paper_baseline", group="controls", scenario=s, method="paper_baseline",
                         video=str(prd / "baseline_video.mp4"), edit_prompt=_prompt(prd), critic_from=str(prd / "run_config.json")))
    return rows


def best_of_n():
    rows = []
    sel = OUT_ROOT / "best_of_n" / "selected"
    for s in TWELVE:
        rd = paper_run_dir(s)
        for f in sorted((sel / s).glob("bon*.mp4")):
            rows.append(dict(id=f"bon/{s}/{f.stem}", group="best_of_n", scenario=s, method=f.stem, video=str(f),
                             edit_prompt=_prompt(rd), critic_from=str(rd / "run_config.json")))
    return rows


def transfer():
    rows = []
    for t in csv.DictReader(open(OUT_ROOT / "transfer" / "targets.csv")):
        o = Path(t["out_dir"])
        for m, v in (("transfer", o / "transfer_both.mp4"), ("baseline", o / "baseline_video.mp4")):
            rows.append(dict(id=f"xfer/{t['latent']}/{t['target']}/{m}", group="transfer", scenario=f"{t['latent']}__{t['target']}",
                             method=m, video=str(v), edit_prompt=t["edit_prompt"], critic_from=""))
    return rows


def main():
    which = sys.argv[1:] or ["comparisons", "controls"]
    outd = OUT_ROOT / "manifests"
    outd.mkdir(parents=True, exist_ok=True)
    for w in which:
        rows = globals()[w]()
        missing = [r["video"] for r in rows if not Path(r["video"]).exists()]
        with open(outd / f"{w}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows(rows)
        print(f"{w}: {len(rows)} rows, {len(missing)} missing", *missing[:10], sep="\n  ")


if __name__ == "__main__":
    main()
