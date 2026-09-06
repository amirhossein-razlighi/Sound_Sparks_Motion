#!/usr/bin/env python3
"""Package finished user-study scenarios as A/B pairs (login node, no GPU).

For each slug (argv, or all us_* with render_results.json): copy the 16- and 32-step renders of the
baseline (A) and ours (B) into h3_probe/results/user_study/<slug>/ together with the scores and the
scenario text, and (re)write user_study/INDEX.md.  Videos are git-ignored; the index and json are tracked.
"""
import json
import os
import shutil
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUTS = "/scratch/amirrz/H3_exp/outputs"
DST = os.path.join(REPO, "h3_probe/results/user_study")
CANDS = {}
for f in ("candidates_r1.json", "candidates_r2.json"):
    p = os.path.join(REPO, "h3_probe/scenarios", f)
    if os.path.exists(p):
        CANDS.update({k: v for k, v in json.load(open(p)).items() if not k.startswith("_")})


def main():
    slugs = sys.argv[1:] or sorted(d[3:] for d in os.listdir(OUTS) if d.startswith("us_") and
                                   os.path.exists(os.path.join(OUTS, d, "render_results.json")))
    rows = []
    for s in slugs:
        src = os.path.join(OUTS, "us_" + s)
        rr = json.load(open(os.path.join(src, "render_results.json")))
        res = json.load(open(os.path.join(src, "results.json"))) if os.path.exists(os.path.join(src, "results.json")) else {}
        d = os.path.join(DST, s)
        os.makedirs(d, exist_ok=True)
        for steps in sorted({v["steps"] for v in rr.values()}):
            for tag, name in (("baseline", "A_baseline"), ("optimized", "B_ours")):
                f = os.path.join(src, f"render_{tag}_{steps}_av.mp4")
                if os.path.exists(f):
                    shutil.copy2(f, os.path.join(d, f"{name}_{steps}steps_av.mp4"))
        sc = CANDS.get(s, {})
        meta = {"slug": s, "edit": sc.get("edit"), "question": sc.get("question"), "src": sc.get("src"), "render": rr,
                "best_iter": res.get("best_iter"), "baseline_yes_any": res.get("baseline", {}).get("yes_any"),
                "optimized_yes_any": res.get("optimized", {}).get("yes_any")}
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
        b16, o16 = rr.get("baseline_16", {}), rr.get("optimized_16", {})
        b32, o32 = rr.get("baseline_32", {}), rr.get("optimized_32", {})
        rows.append(f"| {s} | {sc.get('edit', '')} | {b16.get('yes_any', float('nan')):.3f} -> {o16.get('yes_any', float('nan')):.3f} "
                    f"| {b32.get('yes_any', float('nan')):.3f} -> {o32.get('yes_any', float('nan')):.3f} | {o32.get('lpips_vs_baseline_same_steps', float('nan')):.3f} |")
        print(rows[-1])
    idx = os.path.join(DST, "INDEX.md")
    old = {}
    if os.path.exists(idx):
        for line in open(idx):
            if line.startswith("| ") and not line.startswith("| slug") and not line.startswith("|---"):
                old[line.split("|")[1].strip()] = line.rstrip("\n")
    for r in rows:
        old[r.split("|")[1].strip()] = r
    with open(idx, "w") as f:
        f.write("# User-study A/B pairs (A = H3 baseline, B = ours; same source, prompt, noise, steps)\n\n"
                "Files per scenario: `A_baseline_<steps>steps_av.mp4`, `B_ours_<steps>steps_av.mp4`, `meta.json`.\n"
                "Critic yes-any = noisy-OR over 6 windows of the scenario question (in-loop critic, not evidence).\n\n"
                "| slug | edit | yes-any 16 steps (A -> B) | yes-any 32 steps (A -> B) | LPIPS(B,A) 32 |\n|---|---|---|---|---|\n")
        for k in sorted(old):
            f.write(old[k] + "\n")
    print("wrote", idx)


if __name__ == "__main__":
    main()
