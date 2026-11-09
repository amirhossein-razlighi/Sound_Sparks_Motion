#!/usr/bin/env python3
"""Final best-of-N table: paper ours, LTX-2 Retake, prompt-only N=3/4/8, ours + N=3/4/8 on every metric. CPU only.
Writes best_of_n_ours/final_per_scenario.csv and prints markdown means (mean over 12 clips; held-out: mean and clips with P>=.5)."""
import pandas as pd
R = "/scratch/amirrz/iclr_rebuttal/"; T = "/home/amirrz/my_codes/Sound_Sparks_Motion/results_trillium/best_of_n_ours/"
cs = pd.read_csv(R + "best_of_n_ours/critic_summary.csv").set_index("scenario")
J = ("qwen25", "qwen3vl", "internvl3")
rows = {}
def put(method, metric, ser):
    for s, v in ser.items():
        rows.setdefault((s, method), {})[metric] = v
for m in J:
    h = pd.read_csv(R + f"heldout/{m}.csv"); h = h[h.group == "comparisons"].drop_duplicates(["scenario", "method"]).set_index(["scenario", "method"]).yes_prob
    for me in ("ours", "ltx2_retake"):
        put(me, m, h.xs(me, level="method"))
    put("prompt_bo3", m, pd.read_csv(R + f"heldout_bon/{m}.csv").set_index("scenario").yes_prob)
pb = pd.read_csv(R + "best_of_n_ours/prompt_bo4_metrics.csv", index_col=0)
for m in J:
    put("prompt_bo4", m, pb[m]); put("prompt_bo8", m, pb[m + "_bo8"])
picks = pd.read_csv(R + "best_of_n_ours/picks_all_metrics.csv")
for st, me in (("seed42", "ours_seed42"), ("bo3", "ours_bo3"), ("bo4", "ours_bo4"), ("bo8", "ours_bo8")):
    p = picks[picks.set == st].set_index("scenario")
    for c in list(J) + ["flow_mag", "warp_err", "first_frame_fidelity", "aligned_fidelity"]:
        put(me, c, p[c])
def tab(path, col_method="method"):
    return pd.read_csv(path)
f = tab(R + "p7/flow_warp.csv"); f8 = tab(R + "p7_bon8/flow_warp.csv"); fs = tab(T + "metrics_prompt_seed2/flow_warp.csv")
d = tab(R + "p7/source_fidelity_dinov2.csv"); d3 = tab(R + "p7_bono/source_fidelity_bon3.csv"); d8 = tab(R + "p7_bon8/source_fidelity_dinov2.csv"); ds = tab(T + "metrics_prompt_seed2/source_fidelity.csv")
y = tab(R + "p7/vbench_dyn_continuous.csv"); y8 = tab(R + "p7_bon8/vbench_dyn_continuous.csv"); yo = tab(R + "p7_bono/vbench_dyn_continuous.csv"); ys = tab(R + "p7_bono_s2/vbench_dyn_continuous.csv")
def s(df, method, cols):
    x = df[df.method == method].drop_duplicates("scenario").set_index("scenario")
    return {c: x[c] for c in cols}
FC, DC, YC = ["flow_mag", "warp_err"], ["first_frame_fidelity", "aligned_fidelity"], ["dyn_mean", "frac_above"]
def vb(p):
    v = pd.read_csv(p); v["score"] = v.score.replace({"True": 1, "False": 0}).astype(float); v["m"] = v.method.str.split("__").str[-1]
    return v.pivot_table(index=["m", "scenario"], columns="dim", values="score")
V = {"p7": pd.concat([vb(R + "p7/vbench_per_video.csv"), ], axis=0), "p7q": vb(R + "p7/vbench_per_video_quality.csv"),
     "b8": vb(R + "p7_bon8/vbench_per_video.csv"), "b8q": vb(R + "p7_bon8/vbench_per_video_quality.csv"),
     "bo": vb(R + "p7_bono/vbench_per_video.csv"), "boq": vb(R + "p7_bono/vbench_per_video_quality.csv"),
     "s2": vb(R + "p7_bono_s2/vbench_per_video.csv"), "s2q": vb(R + "p7_bono_s2/vbench_per_video_quality.csv")}
VD = ["subject_consistency", "background_consistency", "motion_smoothness", "temporal_flickering"]; QD = ["imaging_quality", "aesthetic_quality", "dynamic_degree"]
def putv(me, a, b, key):
    for src, dims in ((a, VD), (b, QD)):
        if key in src.index.get_level_values(0):
            x = src.xs(key, level=0)
            for c in dims: put(me, c, x[c])
for me, tag in (("ours", "ours"), ("ltx2_retake", "ltx2_retake")):
    for k, v in s(f, tag, FC).items(): put(me, k, v)
    for k, v in s(d, tag, DC).items(): put(me, k, v)
    for k, v in s(y, tag, YC).items(): put(me, k, v)
    putv(me, V["p7"], V["p7q"], tag)
for k, v in {**s(f, "bon3", FC), **s(d3, "bon3", DC), **s(y, "bon3", YC)}.items(): put("prompt_bo3", k, v)
putv("prompt_bo3", V["p7"], V["p7q"], "bon3")
for k, v in {**s(f8, "bon8", FC), **s(d8, "bon8", DC), **s(y8, "bon8", YC)}.items(): put("prompt_bo8", k, v)
putv("prompt_bo8", V["b8"], V["b8q"], "bon8")
for me in ("ours_bo3", "ours_bo4", "ours_bo8"):
    for k, v in s(yo, me, YC).items(): put(me, k, v)
    putv(me, V["bo"], V["boq"], me)
# prompt-only best-of-4: reuse the bo3 / bo8 row with the same pick, otherwise the seed-2 render
s2 = {}
for k, v in {**s(fs, "prompt_s02", FC), **s(ds, "prompt_s02", DC), **s(ys, "prompt_s02", YC)}.items(): s2[k] = v
for c in VD:
    if "prompt_s02" in V["s2"].index.get_level_values(0): s2[c] = V["s2"].xs("prompt_s02", level=0)[c]
for c in QD:
    if "prompt_s02" in V["s2q"].index.get_level_values(0): s2[c] = V["s2q"].xs("prompt_s02", level=0)[c]
for sc in cs.index:
    p3, p4, p8 = (cs.loc[sc, f"prompt_bo{n}_pick"] for n in (3, 4, 8))
    for c in FC + DC + YC + VD + QD:
        if p4 == p3: v = rows.get((sc, "prompt_bo3"), {}).get(c)
        elif p4 == p8: v = rows.get((sc, "prompt_bo8"), {}).get(c)
        else: v = s2[c].get(sc) if c in s2 else None
        rows.setdefault((sc, "prompt_bo4"), {})[c] = v
df = pd.DataFrame.from_dict(rows, orient="index"); df.index.names = ["scenario", "method"]
df.to_csv(R + "best_of_n_ours/final_per_scenario.csv")
order = ["ltx2_retake", "ours", "prompt_bo3", "prompt_bo4", "prompt_bo8", "ours_seed42", "ours_bo3", "ours_bo4", "ours_bo8"]
scen = list(cs.index)
print("| set | n | Qwen2.5 | Qwen3-VL | InternVL3 | flow | warp | DINO first | DINO aligned | dyn_mean | frac_above | subject | background | smooth | flicker | MUSIQ | LAION | DD |")
print("|" + "---|" * 18)
for me in order:
    x = df.xs(me, level="method").reindex(scen)
    def g(c, n=3): return "-" if x[c].notna().sum() == 0 or c not in x else f"{x[c].mean():.{n}f}"
    def hj(c): return f"{x[c].mean():.3f} ({int((x[c] >= .5).sum())})"
    n = int(x.flow_mag.notna().sum())
    print(f"| {me} | {n} | {hj('qwen25')} | {hj('qwen3vl')} | {hj('internvl3')} | {g('flow_mag')} | {g('warp_err',4)} | {g('first_frame_fidelity',4)} | {g('aligned_fidelity')} | " +
          " | ".join((g(c) if c in x else "-") for c in ("dyn_mean", "frac_above", "subject_consistency", "background_consistency", "motion_smoothness", "temporal_flickering")) + " | " +
          " | ".join((g(c, 2 if c == "imaging_quality" else 3) if c in x else "-") for c in ("imaging_quality", "aesthetic_quality", "dynamic_degree")) + " |")
print(df.xs("prompt_bo4", level="method").isna().sum().to_dict())
x = df.xs("ours_bo3", level="method"); print("ours_bo3 flow excl dog", round(x.drop("dog_yawning").flow_mag.mean(), 3))
