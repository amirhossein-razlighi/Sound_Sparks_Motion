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
with open(os.path.join(d, "critic_scores.txt"), "w") as f:
    f.write(f"baseline: yes_lin={res.get('baseline_yes', float('nan')):.4f} yes_any={res.get('baseline_yes_any', float('nan')):.4f}\n")
    f.write(f"ranking key: {key} (perceptual <= {perc_max} only); B_best = rank 1, B_cand = next {ncand}\n\n")
    for r in rows:
        tag = "  <- B_best" if ok and r is ok[0] else ("  <- cand" if r in ok[1:ncand + 1] else ("  (over perceptual guard)" if r["perc"] > perc_max else ""))
        f.write(f"iter {r['iter']:02d}  yes_lin={r['yes_lin']:.4f}  yes_any={r['yes_any']:.4f}  perceptual={r['perc']:.3f}{tag}\n")
# sheets (baseline + iters 1-8, baseline + iters 9-16)
its = sorted(r["iter"] for r in rows); env = dict(os.environ, W="200", N="12")
for tag, sel in (("a", [i for i in its if i <= 8]), ("b", [i for i in its if i > 8])):
    vids = [os.path.join(run, "baseline.mp4")] + [os.path.join(run, f"iter_{i:02d}.mp4") for i in sel if os.path.exists(os.path.join(run, f"iter_{i:02d}.mp4"))]
    if len(vids) > 1:
        subprocess.run(["bash", SHEET, os.path.join(d, f"iters_sheet_{tag}.jpg")] + vids, env=env, check=False)
print("packaged", d, "best", ok[0] if ok else None)
