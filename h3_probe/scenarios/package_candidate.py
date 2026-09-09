#!/usr/bin/env python3
"""Package one overnight scenario for the morning review (login node):
  h3_probe/results/user_study_candidates/<slug>/
    source_input_av.mp4, source_audio.wav, prompts.txt, A_baseline_av.mp4,
    B_best_iterNN_av.mp4 (critic-best within the perceptual guard) + B_cand_iterNN_av.mp4 (next best iterations),
    critic_scores.txt (every iteration: yes_lin, yes_any, perceptual), iters_sheet_a.jpg / iters_sheet_b.jpg
Usage: package_candidate.py <slug> <run_dir> [n_candidates=3]"""
import csv, json, os, re, shutil, subprocess, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SHEET = os.path.join(REPO, "h3_probe/scenarios/ab_sheet.sh")
slug, run = sys.argv[1], sys.argv[2]; ncand = int(sys.argv[3]) if len(sys.argv) > 3 else 3
d = os.path.join(REPO, "h3_probe/results/user_study_candidates", slug); os.makedirs(d, exist_ok=True)
cfg = json.load(open(os.path.join(run, "run_config.json"))); sc = cfg.get("scenario", {}); perc_max = float(cfg.get("perc_max", 0.25) or 0.25)
# source + prompts (same logic as package_ab)
src = sc.get("src", ""); src = src if os.path.isabs(src) else os.path.join(REPO, src); wav = sc.get("wav", "")
def has_audio(p):
    return "audio" in subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", p], capture_output=True, text=True).stdout
if os.path.exists(src) and not os.path.exists(os.path.join(d, "source_input_av.mp4")):
    if has_audio(src) or not os.path.exists(wav):
        shutil.copy2(src, os.path.join(d, "source_input_av.mp4"))
    else:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-threads", "1", "-i", src, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-shortest", os.path.join(d, "source_input_av.mp4")], check=False)
if os.path.exists(wav):
    shutil.copy2(wav, os.path.join(d, "source_audio.wav"))
m = re.search(r"^def grammar_prompt\(.*?(?=^def |^[A-Z_]+ = |\Z)", open(os.path.join(REPO, "h3_probe/baseline_sweep.py")).read(), re.S | re.M); ns = {}; exec(m.group(0), ns)
with open(os.path.join(d, "prompts.txt"), "w") as f:
    f.write(f"scenario: {slug}\nedit sentence: {sc.get('edit')}\nscene: {sc.get('scene')}\ncritic question: {sc.get('question')}\n"
            f"mode: {cfg.get('opt_mode')}  objective: {cfg.get('critic_objective')}  select_by: {cfg.get('select_by')}  perc_max: {perc_max}\n"
            f"source video: {src}\nsource audio: {wav}\nrun dir: {run}\n\nfull H3 prompt (A and B):\n{ns['grammar_prompt'](sc.get('edit',''), sc.get('scene',''))}\n")
shutil.copy2(os.path.join(run, "baseline_av.mp4"), os.path.join(d, "A_baseline_av.mp4"))
# iterations
rows = []
for r in csv.DictReader(open(os.path.join(run, "opt_log.csv"))):
    try:
        rows.append({"iter": int(r["iter"]), "yes_lin": float(r["yes_prob"]), "yes_any": float(r["yes_any"]), "perc": float(r.get("perceptual") or 0.0)})
    except (KeyError, ValueError):
        pass
res = json.load(open(os.path.join(run, "results.json"))) if os.path.exists(os.path.join(run, "results.json")) else {}
key = "yes_any" if cfg.get("select_by", "any") == "any" else "yes_lin"
ok = sorted([r for r in rows if r["perc"] <= perc_max and os.path.exists(os.path.join(run, f"iter_{r['iter']:02d}_av.mp4"))], key=lambda r: -r[key])
for f in os.listdir(d):
    if f.startswith("B_"):
        os.remove(os.path.join(d, f))
for i, r in enumerate(ok[:ncand + 1]):
    name = f"B_best_iter{r['iter']:02d}_av.mp4" if i == 0 else f"B_cand_iter{r['iter']:02d}_av.mp4"
    shutil.copy2(os.path.join(run, f"iter_{r['iter']:02d}_av.mp4"), os.path.join(d, name))
# FORCE_ITERS=5,7 : iterations flagged by eye (critic misses) are always packaged as B_cand files, guard or not
forced = [int(x) for x in os.environ.get("FORCE_ITERS", "").replace(";", ",").split(",") if x.strip()]
for it in forced:
    src = os.path.join(run, f"iter_{it:02d}_av.mp4")
    if os.path.exists(src) and not any(f.endswith(f"iter{it:02d}_av.mp4") for f in os.listdir(d)):
        shutil.copy2(src, os.path.join(d, f"B_cand_iter{it:02d}_av.mp4"))
with open(os.path.join(d, "critic_scores.txt"), "w") as f:
    f.write(f"baseline: yes_lin={res.get('baseline_yes', float('nan')):.4f} yes_any={res.get('baseline_yes_any', float('nan')):.4f}\n")
    f.write(f"ranking key: {key} (perceptual <= {perc_max} only); B_best = rank 1, B_cand = next {ncand}\n\n")
    for r in rows:
        tag = "  <- B_best" if ok and r is ok[0] else ("  <- cand" if r in ok[1:ncand + 1] else ("  <- cand (flagged by eye)" if r["iter"] in forced else ("  (over perceptual guard)" if r["perc"] > perc_max else "")))
        f.write(f"iter {r['iter']:02d}  yes_lin={r['yes_lin']:.4f}  yes_any={r['yes_any']:.4f}  perceptual={r['perc']:.3f}{tag}\n")
# sheets (baseline + iters 1-8, baseline + iters 9-16)
its = sorted(r["iter"] for r in rows); env = dict(os.environ, W="200", N="12")
for tag, sel in (("a", [i for i in its if i <= 8]), ("b", [i for i in its if i > 8])):
    vids = [os.path.join(run, "baseline.mp4")] + [os.path.join(run, f"iter_{i:02d}.mp4") for i in sel if os.path.exists(os.path.join(run, f"iter_{i:02d}.mp4"))]
    if len(vids) > 1:
        subprocess.run(["bash", SHEET, os.path.join(d, f"iters_sheet_{tag}.jpg")] + vids, env=env, check=False)
b_any = res.get("baseline_yes_any"); best_any = ok[0]["yes_any"] if ok else None
if b_any is None or best_any is None: verdict = "unknown"
elif b_any < 0.35 and best_any > 0.7: verdict = "strong_win"
elif best_any > max(0.5, (b_any or 0) + 0.2): verdict = "improved"
else: verdict = "no_gain"
json.dump({"slug": slug, "baseline_any": b_any, "baseline_lin": res.get("baseline_yes"), "best_iter": ok[0]["iter"] if ok else None, "best_any": best_any,
           "best_lin": ok[0]["yes_lin"] if ok else None, "best_perc": ok[0]["perc"] if ok else None, "auto_verdict": verdict,
           "note": "auto verdict from the in-loop critic; visual review decides"}, open(os.path.join(d, "verdict.json"), "w"), indent=1)
print("packaged", d, "best", ok[0] if ok else None, "auto_verdict", verdict)
