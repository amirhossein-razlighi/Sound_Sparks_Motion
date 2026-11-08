#!/usr/bin/env python3
"""Report for the multi-critic proof of concept (CPU + ffmpeg; run on a compute node).

Runs per scenario: A = Qwen2.5-VL only (paper setup), B = Qwen2.5-VL + InternVL3 in the loop, repeats _r2, _r3 ...
Reads multicritic/passes.csv, mode_both/optimization_log_qwen_both.csv, timing.json of every run,
$ROOT/heldout/{qwen25,internvl3,qwen3vl}.csv (16-frame held-out protocol on every final / preview / iterate) and
$ROOT/metrics.csv (DINOv2 source fidelity, motion, MAD).
Writes $ROOT/report/: summary.md, finals.csv, per_iteration.csv, curves.png, sheets/<scenario>_iters.jpg,
side_by_side/<scenario>.mp4 (input | Retake | every run), and $ROOT/multicritic_package.zip (report + all videos).
"""
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/scratch/amirrz/iclr_rebuttal/multicritic")
REP = ROOT / "report"
FONT = "/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
LABEL = {"A": "A: Qwen2.5-VL only (paper setup)", "B": "B: Qwen2.5-VL + InternVL3 (multi-critic)"}
COL = {"A": "#2a78d6", "B": "#eb6834"}
LS = {"": "-", "_r2": "--", "_r3": ":", "_r4": "-."}


def arm_key(name):
    m = re.search(r"_r(\d+)$", name)
    return name[0] + (f"_r{m.group(1)}" if m else "")


def arm_dirs(sdir):
    return sorted(d for d in sdir.glob("[AB]_*") if d.is_dir() and ".old" not in d.name
                  and (d / "mode_both" / "best_optimized_video_both.mp4").exists())


def lab(a):
    return LABEL[a[0]] + (f", repeat {a[3:]}" if "_r" in a else "")


if REP.exists():
    shutil.move(str(REP), f"{REP}.old_{int(REP.stat().st_mtime)}")
(REP / "sheets").mkdir(parents=True); (REP / "side_by_side").mkdir()

man = pd.read_csv(ROOT / "manifest.csv").set_index("id")
H = {}
for mdl in ("qwen25", "internvl3", "qwen3vl"):
    f = ROOT / "heldout" / f"{mdl}.csv"
    if f.exists():
        H[mdl] = pd.read_csv(f).drop_duplicates("id", keep="last").set_index("id")["yes_prob"]
MET = pd.read_csv(ROOT / "metrics.csv").set_index("id") if (ROOT / "metrics.csv").exists() else pd.DataFrame()
scen = list(dict.fromkeys(man.scenario))
RUNS = {s: {arm_key(d.name): d for d in arm_dirs(ROOT / s)} for s in scen}


def hv(mdl, i):
    return float(H[mdl].get(i, np.nan)) if mdl in H else np.nan


# ---------------- per-iteration ----------------
it_rows, best = [], {}
for s in scen:
    for a, r in RUNS[s].items():
        P = pd.read_csv(r / "multicritic" / "passes.csv")
        P = P[(P.grad == 1) & (P.iter >= 1)]
        L = pd.read_csv(r / "mode_both" / "optimization_log_qwen_both.csv").set_index("iter")
        best[(s, a)] = int(L[L.is_best == 1].index.max()) if (L.is_best == 1).any() else None
        for it, grp in P.groupby("iter"):
            p0 = grp[grp["pass"] == 0].iloc[0]
            i = f"mc/{s}/{a}_it_{int(it):03d}"
            it_rows.append(dict(scenario=s, arm=a, iter=int(it), n_passes=len(grp),
                                inloop_qwen_p_pass0=p0.qwen_yes_prob, inloop_internvl_p_pass0=p0.internvl_yes_prob,
                                inloop_qwen_p_mean_nll=float(np.exp(-grp.qwen_nll.mean())),
                                inloop_internvl_p_mean_nll=float(np.exp(-grp.internvl_nll.mean())) if a[0] == "B" else np.nan,
                                combined_loss=float(grp.combined_loss.mean()),
                                loop_total_loss=float(L.loc[it, "total_loss"]) if it in L.index else np.nan,
                                is_best=int(L.loc[it, "is_best"]) if it in L.index else 0,
                                heldout_qwen25=hv("qwen25", i), heldout_internvl3=hv("internvl3", i), heldout_qwen3vl=hv("qwen3vl", i),
                                qwen_s=float(grp.qwen_s.sum()), internvl_s=float(grp.internvl_s.sum())))
IT = pd.DataFrame(it_rows)
IT.to_csv(REP / "per_iteration.csv", index=False, float_format="%.5f")

# ---------------- finals ----------------
fin = []
for s in scen:
    for m in ["input", "retake", "paper_ours"] + [f"{a}_final" for a in RUNS[s]]:
        i = f"mc/{s}/{m}"
        if i not in man.index:
            continue
        a = m[:-6] if m.endswith("_final") else None
        o = dict(scenario=s, video=m, heldout_qwen25=hv("qwen25", i), heldout_internvl3=hv("internvl3", i),
                 heldout_qwen3vl=hv("qwen3vl", i))
        if f"{i}@phrase" in man.index:   # IVEBench items: same videos scored with the short target phrase
            o.update(heldout_qwen25_phrase=hv("qwen25", f"{i}@phrase"), heldout_internvl3_phrase=hv("internvl3", f"{i}@phrase"),
                     heldout_qwen3vl_phrase=hv("qwen3vl", f"{i}@phrase"))
        if i in MET.index:
            o.update({k: MET.loc[i, k] for k in ("dino_first", "dino_aligned", "motion", "mad_vs_input", "mad_vs_retake")})
        if a:
            bi = best.get((s, a)); o["best_iter"] = bi
            t = json.loads((RUNS[s][a] / "timing.json").read_text())
            o["n_iter_run"] = t.get("n_iterations"); o["s_per_iter"] = t.get("mean_iter_s_post_warmup")
            o["peak_gb_gpu0"] = t.get("peak_mem_gb_total"); o["peak_gb_internvl_gpu"] = t.get("peak_mem_gb_internvl")
            row = IT[(IT.scenario == s) & (IT.arm == a) & (IT.iter == bi)]
            if len(row):
                o["inloop_qwen_p_pass0_at_best"] = float(row.inloop_qwen_p_pass0.iloc[0])
                o["inloop_internvl_p_pass0_at_best"] = float(row.inloop_internvl_p_pass0.iloc[0])
                o["inloop_qwen_p_allpasses_at_best"] = float(row.inloop_qwen_p_mean_nll.iloc[0])
        fin.append(o)
F = pd.DataFrame(fin)
F.to_csv(REP / "finals.csv", index=False, float_format="%.4f")


def md(df):
    f = lambda v: "-" if (v is None or (isinstance(v, float) and np.isnan(v))) else (f"{v:.3f}" if isinstance(v, float) else str(v))
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)] +
                     ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])


# ---------------- curves ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
panels = [("inloop_qwen_p_pass0", "Qwen2.5-VL in-loop, full-clip pass\n(the paper's critic; both arms optimise it)"),
          ("heldout_internvl3", "InternVL3, 16-frame protocol\n(in the loop for B only)"),
          ("heldout_qwen3vl", "Qwen3-VL, 16-frame protocol\n(held out for both arms)")]
fig, ax = plt.subplots(len(scen), 3, figsize=(13, 2.7 * len(scen)), squeeze=False, sharex=True, sharey=True)
for r_, s in enumerate(scen):
    for c, (k, t) in enumerate(panels):
        x = ax[r_, c]
        for a in RUNS[s]:
            d = IT[(IT.scenario == s) & (IT.arm == a)]
            if len(d) and d[k].notna().any():
                x.plot(d.iter, d[k], color=COL[a[0]], lw=2 if "_r" not in a else 1.3, ls=LS.get(a[1:], "-"), label=lab(a))
                if best.get((s, a)):
                    x.plot([best[(s, a)]], [d.set_index("iter")[k].get(best[(s, a)], np.nan)], "o", ms=8, mfc="white",
                           mec=COL[a[0]], mew=2)
        x.set_ylim(-0.02, 1.02); x.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            x.spines[sp].set_visible(False)
        if r_ == 0:
            x.set_title(t, fontsize=9.5)
        if c == 0:
            x.set_ylabel(f"{s}\nP(yes)", fontsize=9)
            x.legend(fontsize=7, loc="best", frameon=False)
        if r_ == len(scen) - 1:
            x.set_xlabel("iteration (rendered iterate; circle = selected best iteration)", fontsize=8.5)
fig.tight_layout(); fig.savefig(REP / "curves.png", dpi=130); plt.close(fig)


# ---------------- sheets: every 5th iterate, 4 frames per run ----------------
def frames(p):
    c = av.open(str(p)); f = [x.to_ndarray(format="rgb24") for x in c.decode(video=0)]; c.close(); return f


fnt = ImageFont.truetype(FONT, 13)
TW, TH, NF = 176, 110, 4
for s in scen:
    runs = list(RUNS[s].items())
    its = [1, 5, 10, 15, 20, 25, 30]
    W, Hh = len(runs) * (NF * TW + 16) + 80, len(its) * (TH + 4) + 34
    im = Image.new("RGB", (W, Hh), "white"); dr = ImageDraw.Draw(im)
    for j, (a, r) in enumerate(runs):
        x0 = 80 + j * (NF * TW + 16)
        dr.text((x0, 4), lab(a) + (f"  best iter {best.get((s, a))}"), fill="black", font=fnt)
        for k, it in enumerate(its):
            p = r / "multicritic" / "iter_frames" / f"iter_{it:03d}.mp4"
            if not p.exists():
                continue
            f = frames(p); idx = np.linspace(0, len(f) - 1, NF).round().astype(int)
            im.paste(Image.fromarray(np.concatenate([np.asarray(Image.fromarray(f[i]).resize((TW, TH))) for i in idx], 1)),
                     (x0, 30 + k * (TH + 4)))
    for k, it in enumerate(its):
        dr.text((6, 30 + k * (TH + 4) + TH // 2 - 7), f"iter {it}", fill="black", font=fnt)
    im.save(REP / "sheets" / f"{s}_iters.jpg", quality=88)

# ---------------- side by side ----------------
esc = lambda s: s.replace("'", "").replace(":", " ").replace("%", " pct").replace(",", " ")


def sbs(panels, title, dst):
    a_, fl = [], []
    for i, (v, lb) in enumerate(panels):
        a_ += ["-i", str(v)]
        fl.append(f"[{i}:v]fps=25,scale=448:280:force_original_aspect_ratio=decrease,pad=448:280:(ow-iw)/2:(oh-ih)/2,setsar=1,"
                  f"drawtext=fontfile={FONT}:text='{esc(lb)}':x=5:y=5:fontsize=12:fontcolor=white:box=1:boxcolor=black@0.6[v{i}]")
    k = len(panels)
    fl.append("".join(f"[v{i}]" for i in range(k)) + f"hstack=inputs={k},pad=iw:ih+28:0:28:black,"
              f"drawtext=fontfile={FONT}:text='{esc(title)}':x=8:y=6:fontsize=15:fontcolor=yellow[o]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", *a_, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", "-shortest", str(dst)], check=True)


short = {"A": "A single critic", "B": "B multi-critic"}
for s in scen:
    pan = []
    for m, lb in [("input", "input"), ("retake", "prompt-only Retake")] + \
                 [(f"{a}_final", short[a[0]] + (f" rep{a[3:]}" if "_r" in a else "")) for a in RUNS[s]]:
        i = f"mc/{s}/{m}"
        if i in man.index:
            sc = " ".join(f"{n} {hv(k, i):.2f}" for k, n in (("qwen25", "Q2.5"), ("internvl3", "IVL3"), ("qwen3vl", "Q3")) if k in H)
            pan.append((man.loc[i, "video"], f"{lb}  {sc}"))
    sbs(pan, f"{s}: {man[man.scenario == s].edit_prompt.iloc[0]}   (held-out 16-frame P(yes) per panel)",
        REP / "side_by_side" / f"{s}.mp4")

# ---------------- summary.md ----------------
cols = ["scenario", "video", "heldout_qwen25", "heldout_internvl3", "heldout_qwen3vl", "heldout_qwen3vl_phrase", "dino_aligned", "motion",
        "mad_vs_input", "best_iter", "inloop_qwen_p_pass0_at_best", "inloop_internvl_p_pass0_at_best", "s_per_iter"]
L = ["# Multi-critic proof of concept: Qwen2.5-VL + InternVL3 in the loop", "",
     "A = paper setup (Qwen2.5-VL-7B critic only); B = loss 0.5 NLL(Qwen2.5-VL-7B) + 0.5 NLL(InternVL3-8B), same question, "
     "same frames per pass (InternVL3 gets the Qwen pass's frames thinned to <= 16), everything else identical (paper "
     "run_config, seed 42, regularisers, best-checkpoint rule). _rN = repeat of the same arm (runs are not "
     "bit-reproducible). Qwen3-VL-8B is never in any loop.", "",
     "## Final videos (held-out columns: 16 uniform frames, held-out question wording; DINO: source fidelity)", "",
     md(F[[c for c in cols if c in F.columns]]), "", "Full columns: finals.csv. Per iteration: per_iteration.csv.", "",
     "![curves](curves.png)"]
(REP / "summary.md").write_text("\n".join(L) + "\n")

# ---------------- package ----------------
z = ROOT / "multicritic_package.zip"
if z.exists():
    z.rename(f"{z}.old_{int(z.stat().st_mtime)}")
with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
    for p in REP.rglob("*"):
        if p.is_file():
            zf.write(p, f"multicritic/report/{p.relative_to(REP)}")
    for s in scen:
        for m in ("input", "retake", "paper_ours"):
            i = f"mc/{s}/{m}"
            if i in man.index:
                zf.write(man.loc[i, "video"], f"multicritic/{s}/0_{m}.mp4")
        for a, r in RUNS[s].items():
            zf.write(r / "mode_both" / "best_optimized_video_both.mp4", f"multicritic/{s}/{r.name}/final_best.mp4")
            for f in ("multicritic/passes.csv", "multicritic/config.json", "mode_both/optimization_log_qwen_both.csv",
                      "timing.json", "run.log", "run_config.json"):
                if (r / f).exists():
                    zf.write(r / f, f"multicritic/{s}/{r.name}/{Path(f).name}")
            for v in sorted((r / "mode_both" / "visualizations").glob("iter_*/best_optimized_video_both.mp4")):
                zf.write(v, f"multicritic/{s}/{r.name}/previews_bestsofar/{v.parent.name}.mp4")
            for v in sorted((r / "multicritic" / "iter_frames").glob("*.mp4")):
                zf.write(v, f"multicritic/{s}/{r.name}/every_iteration/{v.name}")
print("report ->", REP, "| package ->", z)
