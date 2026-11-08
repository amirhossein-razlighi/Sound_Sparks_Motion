#!/usr/bin/env python3
"""Manifest of every video of the multi-critic proof of concept, for heldout_score.py (qwen25 / qwen3vl / internvl3,
16 uniform frames, held-out protocol) and the fidelity/motion metrics.

Per scenario: input (the prepared input the methods saw), retake (prompt-only LTX Retake = the run's baseline_video,
same seed), paper_ours (the paper's video), A_final / B_final (best checkpoint, final render), A_prev_iterNNN /
B_prev_iterNNN (the loop's best-so-far previews, every 5 iterations), A_it_NNN / B_it_NNN (the video the critics scored at
each iteration). Writes $ROOT/manifest.csv (columns id,group,scenario,method,video,edit_prompt).
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SCENARIOS, COMPARISONS, load_paper_args, paper_run_dir  # noqa: E402

ROOT = Path("/scratch/amirrz/iclr_rebuttal/multicritic")
ARMS = {"A": "A_qwen25_only", "B": "B_qwen25+internvl3"}


def arm_key(name):
    """A_qwen25_only -> A ; B_qwen25+internvl3_r2 -> B_r2"""
    import re
    m = re.search(r"_r(\d+)$", name)
    return name[0] + (f"_r{m.group(1)}" if m else "")


def arm_dirs(sdir):
    return sorted(d for d in sdir.glob("[AB]_*") if d.is_dir() and ".old" not in d.name)


IR = Path("/scratch/amirrz/iclr_rebuttal/ivebench")
IVE = {Path(x["src_video_name"]).stem: x for x in __import__("json").load(open(IR / "subset_subject_motion_short.json"))}
rows = []
dirs = sys.argv[1].split(",") if len(sys.argv) > 1 else sorted(d.name for d in ROOT.iterdir() if d.is_dir() and (d.name in SCENARIOS or d.name.startswith("ive_")))
for s in dirs:
    if not (ROOT / s).exists():
        continue
    if s.startswith("ive_"):   # ive_<item>_<cfg>, e.g. ive_short_0251_E
        item, cfg = s[4:].rsplit("_", 1)
        p = IVE[item]["target_prompt"].strip() if cfg != "G" else IVE[item]["target_phrase"].strip()
        phrase = IVE[item]["target_phrase"].strip()
        src, ref = IR / "inputs" / f"{item}.mp4", IR / "runs" / item / cfg / "best_optimized_video_both.mp4"
    else:
        ns = load_paper_args(s, {"output_dir": "/tmp/unused"})
        p, phrase = ns.edit_prompt, None
        src, ref = paper_run_dir(s) / "retake_input_prepared.mp4", COMPARISONS / SCENARIOS[s][1] / "ours.mp4"

    def add(m, v, g="final", s=s, p=p, phrase=phrase):
        rows.append(dict(id=f"mc/{s}/{m}", group=g, scenario=s, method=m, video=str(v), edit_prompt=p))
        if phrase and g == "final":   # IVEBench: also the short target phrase (the long prompt is satisfied by the source)
            rows.append(dict(id=f"mc/{s}/{m}@phrase", group="final_phrase", scenario=s, method=m, video=str(v), edit_prompt=phrase))
    add("input", src)
    add("paper_ours", ref)
    for r in arm_dirs(ROOT / s):
        a = arm_key(r.name)
        if not (r / "mode_both" / "best_optimized_video_both.mp4").exists():
            continue
        has_rt = any(x["method"] == "retake" and x["scenario"] == s for x in rows)
        add("retake" if not has_rt else f"retake_{a}", r / "baseline_video.mp4")
        add(f"{a}_final", r / "mode_both" / "best_optimized_video_both.mp4")
        for v in sorted((r / "mode_both" / "visualizations").glob("iter_*/best_optimized_video_both.mp4")):
            add(f"{a}_prev_{v.parent.name}", v, "preview")
        for v in sorted((r / "multicritic" / "iter_frames").glob("iter_*.mp4")):
            add(f"{a}_it_{v.stem.split('_')[1]}", v, "iterate")
with open(ROOT / "manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "rows ->", ROOT / "manifest.csv")
