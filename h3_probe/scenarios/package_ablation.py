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


def log_rows(run):
    """opt_log.csv -> {iter: {yes_lin, yes_any, yes_4win, perceptual}} (in-loop critic scores of the preview at that iteration)."""
    p = os.path.join(run, "opt_log.csv"); rows = {}
    if not os.path.exists(p):
        return rows
    import csv
    for r in csv.DictReader(open(p)):
        try:
            rows[int(r["iter"])] = {"yes_lin": float(r["yes_prob"]), "yes_any": float(r["yes_any"]), "yes_4win": float(r["yes_4win"]),
                                    "perceptual": float(r.get("perceptual") or 0.0)}
        except (KeyError, ValueError):
            pass
    return rows


def score_line(name, run, it, r):
    """One line for critic_scores.txt: file, iteration, yes(linspace), yes(any-window), yes(4-window mean), perceptual."""
    rows = log_rows(run); rr = r or {}
    if it == "final":
        bi = rr.get("best_iter"); row = rows.get(bi, {}) if bi else {}
        extra = f"  [best checkpoint = iter {bi}; final re-render yes(lin)={rr.get('final_yes', float('nan')):.4f} yes(any)={rr.get('final_yes_any', float('nan')):.4f}]"
        it_s = f"final(best={bi})"
    elif it == "baseline":
        row = {"yes_lin": rr.get("baseline_yes", float("nan")), "yes_any": rr.get("baseline_yes_any", float("nan")), "yes_4win": float("nan"), "perceptual": 0.0}
        extra = ""; it_s = "baseline"
    else:
        row = rows.get(int(it), {}); extra = ""; it_s = f"iter {int(it):02d}"
    g = lambda k: row.get(k, float("nan"))
    return f"{name:34s} {it_s:18s} yes_lin={g('yes_lin'):.4f}  yes_any={g('yes_any'):.4f}  yes_4win={g('yes_4win'):.4f}  perceptual={g('perceptual'):.3f}{extra}\n"


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
        scores = {"both": [score_line("A_baseline_av.mp4", both, "baseline", rb), score_line("B_both_av.mp4", both, it, rb)]}
        line = {"slug": s, "edit": m.get("edit"), "both_iter": it, "baseline_yes": rb.get("baseline_yes"), "both_best_yes": rb.get("final_yes")}
        sheet = [os.path.join(both, "baseline.mp4"), vid(both, it).replace("_av.mp4", ".mp4")]
        for mode in ("text_only", "audio_only"):
            run = m[mode]["out"]; r = res(run)
            line[mode] = None
            if r is None:
                continue
            os.makedirs(os.path.join(d, mode), exist_ok=True)
            shutil.copy2(os.path.join(run, "optimized_final_av.mp4"), os.path.join(d, mode, f"B_{mode}_best_av.mp4"))
            scores[mode] = [score_line(f"B_{mode}_best_av.mp4", run, "final", r)]
            same = vid(run, it) if it != "final" else None
            if same and os.path.exists(same):
                shutil.copy2(same, os.path.join(d, mode, f"B_{mode}_iter{int(it):02d}_av.mp4"))
                sheet.append(same.replace("_av.mp4", ".mp4"))
                scores[mode].append(score_line(f"B_{mode}_iter{int(it):02d}_av.mp4", run, it, r))
            else:
                sheet.append(os.path.join(run, "optimized_final.mp4"))
            line[mode] = {"best_iter": r.get("best_iter"), "best_yes": r.get("final_yes"), "frame_diff": r.get("frame_diff_vs_baseline"),
                          "dz_audio": r.get("dz_audio"), "d_text": r.get("d_text")}
        if all(os.path.exists(x) for x in sheet):
            subprocess.run(["bash", SHEET, os.path.join(d, "compare_sheet.jpg")] + sheet, env=dict(os.environ, W="200", N="12"), check=False)
        hdr = ("Qwen2.5-VL critic yes-probability of each stored video (in-loop score of that iteration's preview; "
               "yes_lin = linspace 24-frame window, yes_any = noisy-OR over 6 contiguous windows, yes_4win = mean over windows; "
               "perceptual = LPIPS+temporal term vs the baseline). Indicative only - the critic can be fooled and can miss motion.\n"
               f"selection objective of these runs: {rb.get('select_by', '?')}\n\n")
        with open(os.path.join(d, "critic_scores.txt"), "w") as f:
            f.write(hdr)
            for mode in ("both", "text_only", "audio_only"):
                if mode in scores:
                    f.write(f"[{mode}]\n" + "".join(scores[mode]) + "\n")
        for mode, lines in scores.items():
            with open(os.path.join(d, mode, "critic_scores.txt"), "w") as f:
                f.write(hdr + "".join(lines))
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
