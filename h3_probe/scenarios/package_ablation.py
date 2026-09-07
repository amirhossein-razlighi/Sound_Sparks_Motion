#!/usr/bin/env python3
"""Package the mode ablation: h3_probe/results/ablation/<slug>/{both,text_only,audio_only}/ + README.md table.

both/       A_baseline_av.mp4, B_both_av.mp4 (the packaged pick)
text_only/  B_text_only_best_av.mp4 (critic-best checkpoint), B_text_only_iterNN_av.mp4 (same iteration as the both pick)
audio_only/ same for audio
<slug>/compare_sheet.jpg : rows = baseline, both (pick), text-only (same iter), audio-only (same iter)
Videos are git-ignored; README.md, meta.json are tracked.  Run on the login node; skips modes whose run has not finished.
"""
import json
import os
import shutil
import subprocess

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DST = os.path.join(REPO, "h3_probe/results/ablation")
SHEET = os.path.join(REPO, "h3_probe/scenarios/ab_sheet.sh")
MAN = json.load(open(os.path.join(DST, "manifest.json")))


def vid(run, it):
    return os.path.join(run, "optimized_final_av.mp4" if it == "final" else f"iter_{int(it):02d}_av.mp4")


def res(run):
    p = os.path.join(run, "results.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    rows = []
    for s, m in MAN.items():
        it = m["both_iter"]; both = m["both_run"]; d = os.path.join(DST, s)
        os.makedirs(os.path.join(d, "both"), exist_ok=True)
        shutil.copy2(os.path.join(both, "baseline_av.mp4"), os.path.join(d, "both", "A_baseline_av.mp4"))
        shutil.copy2(vid(both, it), os.path.join(d, "both", "B_both_av.mp4"))
        rb = res(both) or {}
        line = {"slug": s, "edit": m.get("edit"), "both_iter": it, "baseline_yes": rb.get("baseline_yes"), "both_best_yes": rb.get("final_yes")}
        sheet = [os.path.join(both, "baseline.mp4"), vid(both, it).replace("_av.mp4", ".mp4")]
        for mode in ("text_only", "audio_only"):
            run = m[mode]["out"]; r = res(run)
            line[mode] = None
            if r is None:
                continue
            os.makedirs(os.path.join(d, mode), exist_ok=True)
            shutil.copy2(os.path.join(run, "optimized_final_av.mp4"), os.path.join(d, mode, f"B_{mode}_best_av.mp4"))
            same = vid(run, it) if it != "final" else None
            if same and os.path.exists(same):
                shutil.copy2(same, os.path.join(d, mode, f"B_{mode}_iter{int(it):02d}_av.mp4"))
                sheet.append(same.replace("_av.mp4", ".mp4"))
            else:
                sheet.append(os.path.join(run, "optimized_final.mp4"))
            line[mode] = {"best_iter": r.get("best_iter"), "best_yes": r.get("final_yes"), "frame_diff": r.get("frame_diff_vs_baseline"),
                          "dz_audio": r.get("dz_audio"), "d_text": r.get("d_text")}
        if all(os.path.exists(x) for x in sheet):
            subprocess.run(["bash", SHEET, os.path.join(d, "compare_sheet.jpg")] + sheet, env=dict(os.environ, W="200", N="12"), check=False)
        json.dump(line, open(os.path.join(d, "meta.json"), "w"), indent=1)
        rows.append(line)
    with open(os.path.join(DST, "README.md"), "w") as f:
        f.write("# Mode ablation on the user-study pairs (same source, prompt, noise, recipe, capture; only OPT_MODE differs)\n\n"
                "Per scenario: `both/` (A baseline + the packaged B), `text_only/`, `audio_only/` (critic-best checkpoint and the same "
                "iteration as the both pick), `compare_sheet.jpg` (rows: baseline, both, text-only, audio-only).\n"
                "Critic yes = in-loop Qwen yes-probability at the selected checkpoint (indicative only; judge the videos).\n\n"
                "| slug | edit | both (iter) | baseline yes | both best yes | text-only best yes (iter) | audio-only best yes (iter) |\n|---|---|---|---|---|---|---|\n")
        for l in rows:
            t = l.get("text_only"); a = l.get("audio_only")
            fmt = lambda r: f"{r['best_yes']:.3f} ({r['best_iter']})" if r else "pending"
            f.write(f"| {l['slug']} | {l['edit']} | {l['both_iter']} | {l['baseline_yes'] if l['baseline_yes'] is None else round(l['baseline_yes'],3)} | "
                    f"{l['both_best_yes'] if l['both_best_yes'] is None else round(l['both_best_yes'],3)} | {fmt(t)} | {fmt(a)} |\n")
    print("wrote", os.path.join(DST, "README.md"))


if __name__ == "__main__":
    main()
