#!/usr/bin/env python3
"""Report for the multi-critic proof of concept (CPU + ffmpeg; run on a compute node).

Reads, per scenario and arm (A = Qwen2.5-VL only, B = Qwen2.5-VL + InternVL3 in the loop):
  multicritic/passes.csv, mode_both/optimization_log_qwen_both.csv, timing.json,
  $ROOT/heldout/{qwen25,internvl3,qwen3vl}.csv (16-frame held-out protocol on every final / preview / iterate),
  $ROOT/metrics.csv (DINOv2 source fidelity, motion, MAD).
Writes $ROOT/report/: summary.md, finals.csv, per_iteration.csv, curves.png, sheets/<scenario>_iters.jpg,
side_by_side/<scenario>.mp4 (input | Retake | A | B), and $ROOT/multicritic_package.zip (report + all videos).
"""
import csv
import json
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
ARMS = {"A": "A_qwen25_only", "B": "B_qwen25+internvl3"}
LABEL = {"A": "A: Qwen2.5-VL only (paper setup)", "B": "B: Qwen2.5-VL + InternVL3 (multi-critic)"}
COL = {"A": "#2a78d6", "B": "#eb6834"}
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
scen = [s for s in man.scenario.unique()]


def hv(mdl, i):
    return float(H[mdl].get(i, np.nan)) if mdl in H else np.nan


# ---------------- per-iteration ----------------
it_rows, best = [], {}
for s in scen:
    for a, d in ARMS.items():
        r = ROOT / s / d
        if not (r / "multicritic" / "passes.csv").exists():
            continue
        P = pd.read_csv(r / "multicritic" / "passes.csv")
        P = P[(P.grad == 1) & (P.iter >= 1)]
        g = P.groupby("iter")
        L = pd.read_csv(r / "mode_both" / "optimization_log_qwen_both.csv").set_index("iter")
        bi = int(L[L.is_best == 1].index.max()) if (L.is_best == 1).any() else None
        best[(s, a)] = bi
        for it, grp in g:
            p0 = grp[grp["pass"] == 0].iloc[0]
            i = f"mc/{s}/{a}_it_{int(it):03d}"
            it_rows.append(dict(scenario=s, arm=a, iter=int(it), n_passes=len(grp),
                                inloop_qwen_p_pass0=p0.qwen_yes_prob, inloop_internvl_p_pass0=p0.internvl_yes_prob,
                                inloop_qwen_p_mean_nll=float(np.exp(-grp.qwen_nll.mean())),
                                inloop_internvl_p_mean_nll=float(np.exp(-grp.internvl_nll.mean())) if a == "B" else np.nan,
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
    for m in ("input", "retake", "paper_ours", "A_final", "B_final"):
        i = f"mc/{s}/{m}"
        if i not in man.index:
            continue
        a = m[0] if m in ("A_final", "B_final") else None
        o = dict(scenario=s, video=m, heldout_qwen25=hv("qwen25", i), heldout_internvl3=hv("internvl3", i),
                 heldout_qwen3vl=hv("qwen3vl", i))
        if i in MET.index:
            o.update({k: MET.loc[i, k] for k in ("dino_first", "dino_aligned", "motion", "mad_vs_input", "mad_vs_retake")})
        if a:
            bi = best.get((s, a)); o["best_iter"] = bi
            t = json.loads((ROOT / s / ARMS[a] / "timing.json").read_text())
            o["s_per_iter"] = t.get("mean_iter_s_post_warmup"); o["peak_gb_gpu0"] = t.get("peak_mem_gb_total")
            o["peak_gb_internvl_gpu"] = t.get("peak_mem_gb_internvl")
            row = IT[(IT.scenario == s) & (IT.arm == a) & (IT.iter == bi)]
            if len(row):
                o["inloop_qwen_p_at_best"] = float(row.inloop_qwen_p_mean_nll.iloc[0])
                o["inloop_internvl_p_at_best"] = float(row.inloop_internvl_p_mean_nll.iloc[0])
        fin.append(o)
F = pd.DataFrame(fin)
F.to_csv(REP / "finals.csv", index=False, float_format="%.4f")


def md(df):
    f = lambda v: "-" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.3f}" if isinstance(v, float) else str(v))
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)] +
                     ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])


# ---------------- curves ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
panels = [("inloop_qwen_p_mean_nll", "Qwen2.5-VL, in-loop protocol\n(the paper's critic; both arms optimise it)"),
          ("heldout_internvl3", "InternVL3, 16-frame protocol\n(in the loop for B only)"),
          ("heldout_qwen3vl", "Qwen3-VL, 16-frame protocol\n(held out for both arms)")]
fig, ax = plt.subplots(len(scen), 3, figsize=(13, 2.7 * len(scen)), squeeze=False, sharex=True, sharey=True)
for r, s in enumerate(scen):
    for c, (k, t) in enumerate(panels):
        x = ax[r, c]
        for a in ARMS:
            d = IT[(IT.scenario == s) & (IT.arm == a)]
            if len(d):
                x.plot(d.iter, d[k], color=COL[a], lw=2, label=LABEL[a])
                if best.get((s, a)):
                    x.axvline(best[(s, a)], color=COL[a], lw=1, ls=":")
        x.set_ylim(-0.02, 1.02); x.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            x.spines[sp].set_visible(False)
        if r == 0:
            x.set_title(t, fontsize=9.5)
        if c == 0:
            x.set_ylabel(f"{s}\nP(yes)", fontsize=9)
        if r == len(scen) - 1:
            x.set_xlabel("iteration (rendered iterate; dotted = selected best iteration)", fontsize=8.5)
ax[0, 0].legend(fontsize=8, loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig(REP / "curves.png", dpi=130); plt.close(fig)


# ---------------- sheets: every 5th iterate, 6 frames each, A | B ----------------
def frames(p):
    c = av.open(str(p)); f = [x.to_ndarray(format="rgb24") for x in c.decode(video=0)]; c.close(); return f


fnt = ImageFont.truetype(FONT, 14)
for s in scen:
    its = sorted({1, *range(5, 31, 5)})
    tiles = []
    for it in its:
        row = []
        for a, d in ARMS.items():
            p = ROOT / s / d / "multicritic" / "iter_frames" / f"iter_{it:03d}.mp4"
            if not p.exists():
                row.append(None); continue
            f = frames(p); idx = np.linspace(0, len(f) - 1, 6).round().astype(int)
            row.append(np.concatenate([np.asarray(Image.fromarray(f[i]).resize((192, 108))) for i in idx], 1))
        tiles.append((it, row))
    W, Hh = 2 * 6 * 192 + 20 + 90, len(tiles) * 112 + 30
    im = Image.new("RGB", (W, Hh), "white"); dr = ImageDraw.Draw(im)
    dr.text((90, 5), LABEL["A"], fill="black", font=fnt); dr.text((90 + 6 * 192 + 20, 5), LABEL["B"], fill="black", font=fnt)
    for k, (it, row) in enumerate(tiles):
        y = 30 + k * 112
        dr.text((5, y + 45), f"iter {it}", fill="black", font=fnt)
        for j, t in enumerate(row):
            if t is not None:
                im.paste(Image.fromarray(t), (90 + j * (6 * 192 + 20), y))
    im.save(REP / "sheets" / f"{s}_iters.jpg", quality=88)


# ---------------- side by side ----------------
esc = lambda s: s.replace("'", "").replace(":", " ").replace("%", " pct").replace(",", " ")


def sbs(panels, title, dst):
    a_, fl = [], []
    for i, (v, lab) in enumerate(panels):
        a_ += ["-i", str(v)]
        fl.append(f"[{i}:v]fps=25,scale=448:252:force_original_aspect_ratio=decrease,pad=448:252:(ow-iw)/2:(oh-ih)/2,setsar=1,"
                  f"drawtext=fontfile={FONT}:text='{esc(lab)}':x=5:y=5:fontsize=13:fontcolor=white:box=1:boxcolor=black@0.6[v{i}]")
    k = len(panels)
    fl.append("".join(f"[v{i}]" for i in range(k)) + f"hstack=inputs={k},pad=iw:ih+28:0:28:black,"
              f"drawtext=fontfile={FONT}:text='{esc(title)}':x=8:y=6:fontsize=15:fontcolor=yellow[o]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", *a_, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", "-shortest", str(dst)], check=True)


for s in scen:
    pan = []
    for m, lab in (("input", "input"), ("retake", "prompt-only Retake"), ("A_final", "A single critic"), ("B_final", "B multi-critic")):
        i = f"mc/{s}/{m}"
        if i in man.index:
            sc = " ".join(f"{n} {hv(k, i):.2f}" for k, n in (("qwen25", "Q2.5"), ("internvl3", "IVL3"), ("qwen3vl", "Q3")) if k in H)
            pan.append((man.loc[i, "video"], f"{lab}  {sc}"))
    sbs(pan, f"{s}: {man[man.scenario == s].edit_prompt.iloc[0]}   (held-out 16-frame P(yes) per panel)", REP / "side_by_side" / f"{s}.mp4")

# ---------------- summary.md ----------------
L = ["# Multi-critic proof of concept: Qwen2.5-VL + InternVL3 in the loop", "",
     "Arms: A = paper setup (Qwen2.5-VL-7B critic only); B = loss 0.5 NLL(Qwen2.5-VL-7B) + 0.5 NLL(InternVL3-8B), same "
     "question, same frames per pass (InternVL3 gets the Qwen pass's frames thinned to <= 16), everything else identical "
     "(paper run_config, seed, regularisers, best-checkpoint rule). Qwen3-VL-8B is never in any loop.", "",
     "## Final videos (held-out columns = 16 uniform frames, held-out question wording; DINO = source fidelity)", "",
     md(F), "", "![curves](curves.png)", "", "## Per-iteration", "", "See per_iteration.csv (every iteration, both arms)."]
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
        for a, d in ARMS.items():
            r = ROOT / s / d
            if not r.exists():
                continue
            zf.write(r / "mode_both" / "best_optimized_video_both.mp4", f"multicritic/{s}/{d}/final_best.mp4")
            for f in ("multicritic/passes.csv", "multicritic/config.json", "mode_both/optimization_log_qwen_both.csv",
                      "timing.json", "run.log", "run_config.json"):
                if (r / f).exists():
                    zf.write(r / f, f"multicritic/{s}/{d}/{Path(f).name}")
            for v in sorted((r / "mode_both" / "visualizations").glob("iter_*/best_optimized_video_both.mp4")):
                zf.write(v, f"multicritic/{s}/{d}/previews_bestsofar/{v.parent.name}.mp4")
            for v in sorted((r / "multicritic" / "iter_frames").glob("*.mp4")):
                zf.write(v, f"multicritic/{s}/{d}/every_iteration/{v.name}")
print("report ->", REP, "| package ->", z)
