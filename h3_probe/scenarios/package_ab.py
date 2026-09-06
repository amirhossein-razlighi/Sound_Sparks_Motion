#!/usr/bin/env python3
"""Package user-study A/B pairs from scenarios/picks.json (login node, no GPU).

A = <run>/baseline_av.mp4 (H3 baseline, 16 steps), B = <run>/iter_<iter>_av.mp4 or optimized_final_av.mp4 (ours, same
noise and step count).  32-step re-renders (render_*_32_av.mp4, from render_ab.sbatch) are copied too when present.
Output: h3_probe/results/user_study/<slug>/{A_baseline,B_ours}_av.mp4 (+ _32steps), meta.json, and INDEX.md.
Videos are git-ignored; picks.json, meta.json and INDEX.md are tracked.
"""
import json
import os
import shutil
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DST = os.path.join(REPO, "h3_probe/results/user_study")
PICKS = {k: v for k, v in json.load(open(os.path.join(REPO, "h3_probe/scenarios/picks.json"))).items() if not k.startswith("_")}
CANDS = {}
for f in ("candidates_r1.json", "candidates_r2.json"):
    p = os.path.join(REPO, "h3_probe/scenarios", f)
    if os.path.exists(p):
        CANDS.update({k: v for k, v in json.load(open(p)).items() if not k.startswith("_")})


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
        for tag, name in (("baseline", "A_baseline"), ("optimized", "B_ours")):
            f = next((x for x in (os.path.join(run, f"render_{tag}_32_av.mp4"), os.path.join(run, "render", f"render_{tag}_32_av.mp4"))
                      if os.path.exists(x)), None)
            if f:
                shutil.copy2(f, os.path.join(d, f"{name}_32steps_av.mp4"))
                extra.append(name + "_32steps")
        res = json.load(open(os.path.join(run, "results.json"))) if os.path.exists(os.path.join(run, "results.json")) else {}
        edit = pk.get("edit") or CANDS.get(s, {}).get("edit") or res.get("edit") or ""
        meta = {"slug": s, "edit": edit, "run": run, "iter": it, "note": pk.get("note", ""), "steps": 16, "extra": extra,
                "baseline_yes": res.get("baseline_yes"), "baseline_yes_any": res.get("baseline_yes_any"),
                "best_iter_by_critic": res.get("best_iter")}
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
        rows.append(f"| {s} | {edit} | {it} | {pk.get('note', '')} |")
        print(rows[-1])
    with open(os.path.join(DST, "INDEX.md"), "w") as f:
        f.write("# User-study A/B pairs (A = H3 baseline, B = ours; same source, prompt, noise and 16 steps)\n\n"
                "Files per scenario: `A_baseline_av.mp4`, `B_ours_av.mp4` (+ `_32steps` re-renders when available), `meta.json`.\n"
                "B is the iteration picked by visual inspection of all previews (`scenarios/picks.json`).\n\n"
                "| slug | edit | picked iter | what changes |\n|---|---|---|---|\n")
        for r in rows:
            f.write(r + "\n")
    print("wrote", os.path.join(DST, "INDEX.md"))


if __name__ == "__main__":
    main()
