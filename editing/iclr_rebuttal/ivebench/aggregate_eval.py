#!/usr/bin/env python3
"""Aggregate official IVEBench per-video results for selected ours vs selected Retake (CPU).

Averages replicate IVEBench get_average_score.py exactly (that script is broken: parse_args returns 2 values, main unpacks 5):
values <= -1 -> NaN (motion_fidelity is skipped by design for subject_motion_editing), mean per metric, min-max normalisation of
instruction_satisfaction (1..5), content_fidelity (1..5), vtss (-0.05..0.1), dimension = weighted mean (IS 3, CF 3, VTSS 4, others 1),
total = mean of quality / compliance / fidelity.
Adds (not in IVEBench): per-item paired comparison ours vs Retake per metric (wins/ties/losses, mean diff, Wilcoxon signed-rank p),
per-video total computed with the same formula, and subsets by frame-check verdict.
Usage: aggregate_eval.py [pairs e.g. ours_critic:retake_critic ours_eye:retake_eye]  -> $R/eval/summary_*.csv/.md
"""
import glob, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
R = "/scratch/amirrz/iclr_rebuttal/ivebench"
CAT = {"subject_consistency_score": "quality", "background_consistency_score": "quality", "temporal_flickering_score": "quality",
       "motion_smoothness_score": "quality", "vtss_score": "quality", "overall_semantic_consistency_score": "compliance",
       "phrase_semantic_consistency_score": "compliance", "instruction_satisfaction_score": "compliance",
       "semantic_fidelity_score": "fidelity", "motion_fidelity_score": "fidelity", "content_fidelity_score": "fidelity"}
W = {"instruction_satisfaction_score": 3, "content_fidelity_score": 3, "vtss_score": 4}
RNG = {"instruction_satisfaction_score": (1, 5), "content_fidelity_score": (1, 5), "vtss_score": (-0.05, 0.1)}
M = list(CAT)

def load(tag):
    f = sorted(glob.glob(f"{R}/eval/{tag}/{tag}_*_eval_results.csv"))[-1]
    d = pd.read_csv(f); d["item"] = d["video_name"].str.replace(".mp4", "", regex=False)
    d[M] = d[M].apply(pd.to_numeric, errors="coerce").mask(lambda x: x <= -1)
    return d.set_index("item")

def total(means):
    n = means.copy()
    for c, (a, b) in RNG.items():
        n[c] = (n[c] - a) / (b - a)
    dims = {}
    for dim in ("quality", "compliance", "fidelity"):
        cs = [c for c in M if CAT[c] == dim and not pd.isna(n.get(c, np.nan))]
        dims[dim] = sum(n[c] * W.get(c, 1) for c in cs) / sum(W.get(c, 1) for c in cs) if cs else np.nan
    dims["total_score"] = np.nanmean([dims[k] for k in ("quality", "compliance", "fidelity")])
    return dims

def per_video_total(d):
    return pd.Series({i: total(r[M])["total_score"] for i, r in d.iterrows()})

def md_table(df):
    f = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)] +
                     ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])

sel = pd.read_csv(f"{R}/selection.csv")
pairs = [a.split(":") for a in sys.argv[1:]] or [("ours_critic", "retake_critic"), ("ours_eye", "retake_eye")]
md = []
for a, b in pairs:
    try:
        A, B = load(a), load(b)
    except IndexError:
        print("missing", a, b); continue
    items = sorted(set(A.index) & set(B.index))
    A, B = A.loc[items], B.loc[items]
    A["total_pv"], B["total_pv"] = per_video_total(A), per_video_total(B)
    side = "eye" if "eye" in a else "critic"
    vo = sel[(sel.side == "ours") & (sel[f"pick_{side}"] == 1)].set_index("item")["verdict"]
    vr = sel[(sel.side == "retake") & (sel[f"pick_{side}"] == 1)].set_index("item")["verdict"]
    subsets = {"all": items,
               "edit visible in ours or retake (verdict >= partial)": [i for i in items if {vo.get(i), vr.get(i)} & {"partial", "partial+", "ok", "ok+", "good", "good+"}],
               "no visible problem in either pick": [i for i in items if "problem" not in (vo.get(i), vr.get(i))]}
    try:   # objective subsets from the UNEDITED SOURCE scored by the same official judge (independent of both methods)
        S = load("source_noedit").reindex(items)["instruction_satisfaction_score"]
        subsets["source not already satisfying the edit (source IS <= 2)"] = [i for i in items if S.get(i, 9) <= 2]
        subsets["source IS <= 3"] = [i for i in items if S.get(i, 9) <= 3]
        subsets["source IS = 5 (judge says the unedited input already does the edit)"] = [i for i in items if S.get(i, 0) == 5]
        A["src_IS"] = S; B["src_IS"] = S
    except IndexError:
        pass
    rows = []
    for sname, its in subsets.items():
        ta, tb = total(A.loc[its, M].mean()), total(B.loc[its, M].mean())
        for k in ("total_score", "quality", "compliance", "fidelity"):
            rows.append(dict(subset=sname, n=len(its), metric=k, ours=ta[k], retake=tb[k], diff=ta[k] - tb[k]))
        for c in M + ["total_pv"]:
            x, y = A.loc[its, c], B.loc[its, c]; ok = x.notna() & y.notna()
            if ok.sum() == 0:
                continue
            dx = (x - y)[ok]
            try:
                p = wilcoxon(x[ok], y[ok]).pvalue if (dx != 0).any() else 1.0
            except ValueError:
                p = np.nan
            rows.append(dict(subset=sname, n=int(ok.sum()), metric=c, ours=x[ok].mean(), retake=y[ok].mean(), diff=dx.mean(),
                             wins=int((dx > 1e-9).sum()), ties=int((dx.abs() <= 1e-9).sum()), losses=int((dx < -1e-9).sum()), wilcoxon_p=p))
    out = pd.DataFrame(rows); out.to_csv(f"{R}/eval/summary_{a}_vs_{b}.csv", index=False, float_format="%.4f")
    pv = pd.DataFrame({"ours_verdict": vo.reindex(items), "retake_verdict": vr.reindex(items)})
    for c in ["instruction_satisfaction_score", "content_fidelity_score", "vtss_score", "total_pv"]:
        pv[f"ours_{c}"], pv[f"retake_{c}"] = A[c], B[c]
    pv.to_csv(f"{R}/eval/per_item_{a}_vs_{b}.csv", float_format="%.4f")
    md.append(f"## {a} vs {b}\n\n" + md_table(out) + "\n")
    print(f"== {a} vs {b}"); print(out[out.subset == "all"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
open(f"{R}/eval/summary.md", "w").write("\n".join(md))
