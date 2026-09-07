#!/usr/bin/env python3
"""Package user-study A/B pairs from scenarios/picks.json (login node, no GPU).

A = <run>/baseline_av.mp4 (H3 baseline, 16 steps), B = <run>/iter_<iter>_av.mp4 or optimized_final_av.mp4 (ours, same
noise and step count).  32-step re-renders (render_*_32_av.mp4, from render_ab.sbatch) are copied too when present.
Output: h3_probe/results/user_study/<slug>/{A_baseline,B_ours}_av.mp4 (+ _32steps), meta.json, and INDEX.md.
Videos are git-ignored; picks.json, meta.json and INDEX.md are tracked.
"""
import json
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DST = os.path.join(REPO, "h3_probe/results/user_study")
PICKS = {k: v for k, v in json.load(open(os.path.join(REPO, "h3_probe/scenarios/picks.json"))).items() if not k.startswith("_")}
CANDS = {}
for f in ("candidates_r1.json", "candidates_r2.json"):
    p = os.path.join(REPO, "h3_probe/scenarios", f)
    if os.path.exists(p):
        CANDS.update({k: v for k, v in json.load(open(p)).items() if not k.startswith("_")})


def _grammar_prompt():
    # exec only the grammar_prompt function from baseline_sweep.py (importing the module would pull in torch)
    src = open(os.path.join(REPO, "h3_probe/baseline_sweep.py")).read()
    m = re.search(r"^def grammar_prompt\(.*?(?=^def |^[A-Z_]+ = |\Z)", src, re.S | re.M)
    ns = {}; exec(m.group(0), ns); return ns["grammar_prompt"]


GRAMMAR = _grammar_prompt()


def has_audio(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout
    return "audio" in out


def add_source_and_prompts(d, run, pk, it):
    """Copy the model input (source video with its audio) and write prompts.txt into the scenario folder."""
    cfg_p = os.path.join(run, "run_config.json")
    if not os.path.exists(cfg_p):
        return
    cfg = json.load(open(cfg_p)); sc = cfg.get("scenario") or {}
    src = sc.get("src", ""); src = src if os.path.isabs(src) else os.path.join(REPO, src); wav = sc.get("wav", "")
    dst = os.path.join(d, "source_input_av.mp4")
    if os.path.exists(src) and not os.path.exists(dst):
        if has_audio(src) or not os.path.exists(wav):
            shutil.copy2(src, dst)
        else:
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-threads", "1", "-i", src, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-shortest", dst], check=False)
    if os.path.exists(wav) and not os.path.exists(os.path.join(d, "source_audio.wav")):
        shutil.copy2(wav, os.path.join(d, "source_audio.wav"))
    edit = pk.get("edit") or sc.get("edit", ""); scene = sc.get("scene", "")
    with open(os.path.join(d, "prompts.txt"), "w") as f:
        f.write(f"scenario: {pk.get('slug', '')}\n")
        f.write(f"edit sentence (the only thing the user asks for): {edit}\n")
        f.write(f"scene description used in the prompt: {scene}\n")
        f.write(f"critic question (Qwen2.5-VL, our method only): {sc.get('question', cfg.get('motion_question', ''))}\n")
        f.write(f"mode: {cfg.get('opt_mode')}   critic objective: {cfg.get('critic_objective')}   picked iteration: {it}\n")
        f.write(f"source video: {src}\nsource audio: {wav}\n\n")
        f.write("full H3 prompt (identical for A and B; A = H3 with this prompt, B = ours with this prompt + optimized audio/text latents):\n")
        f.write(GRAMMAR(edit, scene) if scene else cfg.get("prompt_edit", "") or "(see run_config.json)")
        f.write("\n")


def main():
    slugs = sys.argv[1:] or sorted(PICKS)
    rows = []
    for s in slugs:
        pk = PICKS[s]
        run, it = pk["run"], pk["iter"]
        b = os.path.join(run, "optimized_final_av.mp4" if it == "final" else f"iter_{int(it):02d}_av.mp4")
        a = os.path.join(run, "baseline_av.mp4")
        if not (os.path.exists(a) and os.path.exists(b)):
            print(f"[skip] {s}: missing {a if not os.path.exists(a) else b}")
            continue
        d = os.path.join(DST, s)
        os.makedirs(d, exist_ok=True)
        shutil.copy2(a, os.path.join(d, "A_baseline_av.mp4"))
        shutil.copy2(b, os.path.join(d, "B_ours_av.mp4"))
        extra = []
        for alt in pk.get("alts", []):   # alternates the user may prefer: B_candidate_iterNN_av.mp4 (or _final)
            fa = os.path.join(run, "optimized_final_av.mp4" if alt == "final" else f"iter_{int(alt):02d}_av.mp4")
            if os.path.exists(fa):
                na = "B_candidate_final" if alt == "final" else f"B_candidate_iter{int(alt):02d}"
                shutil.copy2(fa, os.path.join(d, na + "_av.mp4")); extra.append(na)
        for tag, name in (("baseline", "A_baseline"), ("optimized", "B_ours")):
            f = next((x for x in (os.path.join(run, f"render_{tag}_32_av.mp4"), os.path.join(run, "render", f"render_{tag}_32_av.mp4"))
                      if os.path.exists(x)), None)
            if f:
                shutil.copy2(f, os.path.join(d, f"{name}_32steps_av.mp4"))
                extra.append(name + "_32steps")
        add_source_and_prompts(d, run, dict(pk, slug=s), it)
        res = json.load(open(os.path.join(run, "results.json"))) if os.path.exists(os.path.join(run, "results.json")) else {}
        edit = pk.get("edit") or CANDS.get(s, {}).get("edit") or res.get("edit") or ""
        meta = {"slug": s, "edit": edit, "run": run, "iter": it, "note": pk.get("note", ""), "steps": 16, "extra": extra,
                "baseline_yes": res.get("baseline_yes"), "baseline_yes_any": res.get("baseline_yes_any"),
                "best_iter_by_critic": res.get("best_iter")}
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
        alts = ", ".join(str(a) for a in pk.get("alts", []))
        rows.append(f"| {s} | {edit} | {it}{' (alt: ' + alts + ')' if alts else ''} | {pk.get('note', '')} |")
        print(rows[-1])
    with open(os.path.join(DST, "INDEX.md"), "w") as f:
        f.write("# User-study A/B pairs (A = H3 baseline, B = ours; same source, prompt, noise and 16 steps)\n\n"
                "Files per scenario: `source_input_av.mp4` (the model input) + `source_audio.wav`, `prompts.txt` (edit sentence, full H3 prompt, critic question), `A_baseline_av.mp4`, `B_ours_av.mp4` (+ `B_candidate_iterNN_av.mp4` alternates), `meta.json`.\n"
                "B is the iteration picked by visual inspection of all previews (`scenarios/picks.json`).\n\n"
                "| slug | edit | picked iter | what changes |\n|---|---|---|---|\n")
        for r in rows:
            f.write(r + "\n")
    print("wrote", os.path.join(DST, "INDEX.md"))


if __name__ == "__main__":
    main()
