#!/usr/bin/env python3
"""Merge held-out VLMs, flow, DINOv2 fidelity and VBench for the noise-optimization finals; write all_metrics.csv and markdown tables. CPU only."""
import json, pandas as pd
R = "/scratch/amirrz/iclr_rebuttal/noiseopt/"; O = R + "metrics/"
d = None
for m in ("qwen25", "qwen3vl", "internvl3"):
    x = pd.read_csv(O + m + ".csv")[["scenario", "method", "yes_prob"]].rename(columns={"yes_prob": m})
    d = x if d is None else d.merge(x, on=["scenario", "method"], how="outer")
d = d.merge(pd.read_csv(O + "flow_warp.csv")[["scenario", "method", "flow_mag", "warp_err"]], on=["scenario", "method"], how="left")
d = d.merge(pd.read_csv(O + "source_fidelity_dinov2.csv"), on=["scenario", "method"], how="left")
d = d.merge(pd.read_csv(O + "vbench_dyn_continuous.csv")[["scenario", "method", "dyn_mean", "frac_above"]], on=["scenario", "method"], how="left")
for n in ("vbench_per_video.csv", "vbench_per_video_quality.csv"):
    v = pd.read_csv(O + n); v["method"] = v.method.str.split("__").str[-1]; v["score"] = v.score.replace({"True": 1, "False": 0}).astype(float)
    d = d.merge(v.pivot_table(index=["scenario", "method"], columns="dim", values="score").reset_index(), on=["scenario", "method"], how="left")
V = {("turtle_neck", "0.05"): "collapse: white/yellow frames, then black", ("turtle_neck", "0.02"): "scene cut to white background with a red/white tongue-like shape; no neck extension",
     ("turtle_neck", "0.01"): "collapse: fire-like glow, then black", ("turtle_neck", "0.005"): "collapse: fades to black", ("turtle_neck", "0.0025"): "clean; small white horn-like shape at the end; no neck extension",
     ("dog_yawning", "0.02"): "final = baseline (best iterate is iter 1); panting, no yawn", ("dog_yawning", "0.01"): "final = baseline (best iterate is iter 1); panting, no yawn",
     ("dog_yawning", "0.005"): "tongue out, head turns away; no yawn", ("dog_yawning", "0.0025"): "SUCCESS: wide open yawn-like mouth, scene kept, close to ours",
     ("red_car_door_opens", "0.02"): "door opens late; camera swings top-down", ("red_car_door_opens", "0.01"): "both doors open at the end; camera rises to a high angle, car look changes",
     ("red_car_door_opens", "0.005"): "zooms out; no door", ("red_car_door_opens", "0.0025"): "close to baseline; no door",
     ("man_pets_dog", "0.02"): "no petting; camera drifts", ("man_pets_dog", "0.01"): "no petting; camera drifts, table scene changes", ("man_pets_dog", "0.005"): "no petting; camera drifts, table scene changes", ("man_pets_dog", "0.0025"): "no petting; camera drifts"}
P = {}
for (s, lr) in V:
    t = json.load(open(f"{R}{s}/lr{lr}/timing.json")); P[(s, lr)] = (t["best_p"], t["best_iter"], t["mean_iter_s_post_warmup"], t["peak_mem_gb_loop"])
d["lr"] = d.method.str.replace("noise_lr", "", regex=False).where(d.method.str.startswith("noise_"))
d["frame_verdict"] = [V.get((s, l), "") for s, l in zip(d.scenario, d.lr)]
d["inloop_best_p"] = [P[(s, l)][0] if (s, l) in P else None for s, l in zip(d.scenario, d.lr)]
d["best_iter"] = [P[(s, l)][1] if (s, l) in P else None for s, l in zip(d.scenario, d.lr)]
d.to_csv(O + "all_metrics.csv", index=False)
o = {"input": 0, "ltx2_retake": 1, "ours": 2}
d["o"] = [o.get(m, 3) for m in d.method]
d["lrv"] = pd.to_numeric(d.lr, errors="coerce")
d = d.sort_values(["scenario", "o", "lrv"], ascending=[True, True, False])
f = lambda x, n=3: "-" if pd.isna(x) else f"{x:.{n}f}"
print("| clip | run | in-loop P (best iter) | Qwen2.5 | Qwen3-VL | InternVL3 | flow | warp | DINO aligned | subject | background | smooth | flicker | MUSIQ | LAION | dyn_mean | frame check of the final video |")
print("|" + "---|" * 17)
for r in d.itertuples():
    ip = "-" if pd.isna(r.inloop_best_p) else f"{r.inloop_best_p:.3f} ({int(r.best_iter)})"
    print(f"| {r.scenario} | {r.method.replace('noise_','noise ')} | {ip} | {f(r.qwen25)} | {f(r.qwen3vl)} | {f(r.internvl3)} | {f(r.flow_mag)} | {f(r.warp_err,4)} | {f(r.aligned_fidelity)} | {f(r.subject_consistency)} | {f(r.background_consistency)} | {f(r.motion_smoothness)} | {f(r.temporal_flickering)} | {f(r.imaging_quality,1)} | {f(r.aesthetic_quality)} | {f(r.dyn_mean)} | {r.frame_verdict} |")
print()
print("| set (4 clips) | Qwen2.5 mean (P>=.5) | Qwen3-VL | InternVL3 | DINO aligned | flow |")
print("|---|---|---|---|---|---|")
def row(name, x):
    print(f"| {name} | " + " | ".join(f"{x[m].mean():.3f} ({int((x[m]>=.5).sum())})" for m in ("qwen25", "qwen3vl", "internvl3")) + f" | {x.aligned_fidelity.mean():.3f} | {x.flow_mag.mean():.3f} |")
for m in ("ours", "ltx2_retake"):
    row(m, d[d.method == m])
for lr in ("0.02", "0.01", "0.005", "0.0025"):
    row(f"noise lr {lr} (fixed)", d[d.lr == lr])
n = d[d.lr.notna()]
row("noise, lr chosen per clip by highest in-loop P", n.loc[n.groupby("scenario").inloop_best_p.idxmax()])
row("noise, lr chosen per clip by highest Qwen2.5 held-out (oracle)", n.loc[n.groupby("scenario").qwen25.idxmax()])
print()
print("Compute:", {f"{s} {l}": (round(v[2], 1), round(v[3], 1)) for (s, l), v in P.items() if l == "0.0025"})
