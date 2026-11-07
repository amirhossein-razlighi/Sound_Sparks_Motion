#!/usr/bin/env python3
"""Per-item markdown table for FINDINGS section 15 (CPU): edit, configs run, picks (critic / eye) for ours and Retake with
frame-check verdicts, qwen25 held-out phrase + prompt scores, official IVEBench judge scores (instruction satisfaction IS,
content fidelity CF, per-video total) for ours / Retake / unedited source, and all frame-check notes verbatim.
Writes $R/eval/per_item_table.md."""
import csv, glob, json
from collections import defaultdict
from pathlib import Path
import pandas as pd
R = Path("/scratch/amirrz/iclr_rebuttal/ivebench"); G = Path("/home/amirrz/my_codes/Sound_Sparks_Motion/results_trillium/ivebench")
meta = {Path(x["src_video_name"]).stem: x for x in json.load(open(R / "subset_subject_motion_short.json"))}
sel = pd.read_csv(R / "selection.csv")
def ev(tag):
    f = sorted(glob.glob(str(R / "eval" / tag / f"{tag}_*_eval_results.csv")))
    if not f:
        return None
    d = pd.read_csv(f[-1]); d["item"] = d["video_name"].str.replace(".mp4", "", regex=False); return d.set_index("item")
E = {t: ev(t) for t in ("ours_critic", "retake_critic", "ours_eye", "retake_eye", "source_noedit")}
P = {}
for t in ("critic", "eye"):
    f = R / "eval" / f"per_item_ours_{t}_vs_retake_{t}.csv"
    if f.exists():
        P[t] = pd.read_csv(f, index_col=0)
notes = defaultdict(list)
for r in csv.DictReader(open(R / "notes.csv")):
    notes[r["name"]].append(f"[R {r['cfg']}] retake={r['verdict_retake']} ours={r['verdict_ours']}: {r['notes']}")
for k, v in json.load(open(G / "notes.json")).items():
    n, c = k.split(":"); notes[n].append(f"[T {c}] {v}")
def pick(n, side, rule):
    s = sel[(sel.item == n) & (sel.side == side) & (sel[f"pick_{rule}"] == 1)]
    if s.empty:
        return "", None
    r = s.iloc[0]; it = "" if r["iter"] == "final" else "/" + r["iter"]
    return f"{r['cfg']}{it} {r['verdict']} q={r['q_phrase']:.3f}", r
def js(tag, n, col):
    d = E.get(tag)
    if d is None or n not in d.index:
        return "-"
    v = d.loc[n, col]; return f"{v:.0f}" if col != "total" else f"{v:.3f}"
def tot(rule, side, n):
    p = P.get(rule)
    return f"{p.loc[n, f'{side}_total_pv']:.3f}" if p is not None and n in p.index else "-"
L = ["| item | edit (IVEBench instruction) | n | configs | ours critic pick | Retake critic pick | ours eye pick | Retake eye pick | "
     "IS ours/Retake/source (critic picks) | CF ours/Retake/source | total ours/Retake (critic) | IS ours/Retake (eye) | total ours/Retake (eye) |",
     "|" + "---|" * 13]
for n in sorted(meta):
    cfgs = " ".join(sorted(set(sel[sel.item == n]["cfg"])))
    nf = json.load(open(R / "inputs" / f"{n}.json"))["n"]
    IS, CF = "instruction_satisfaction_score", "content_fidelity_score"
    L.append(f"| {n} | {meta[n]['edit_prompt'][:80]} | {nf} | {cfgs} | {pick(n,'ours','critic')[0]} | {pick(n,'retake','critic')[0]} | "
             f"{pick(n,'ours','eye')[0]} | {pick(n,'retake','eye')[0]} | "
             f"{js('ours_critic',n,IS)}/{js('retake_critic',n,IS)}/{js('source_noedit',n,IS)} | "
             f"{js('ours_critic',n,CF)}/{js('retake_critic',n,CF)}/{js('source_noedit',n,CF)} | {tot('critic','ours',n)}/{tot('critic','retake',n)} | "
             f"{js('ours_eye',n,IS)}/{js('retake_eye',n,IS)} | {tot('eye','ours',n)}/{tot('eye','retake',n)} |")
L += ["", "### Frame-check notes, verbatim (R = Rorqual notes.csv, T = Trillium notes.json)", ""]
for n in sorted(meta):
    L.append(f"- **{n}** ({meta[n]['edit_prompt']}): " + " // ".join(x.replace("|", "/") for x in notes[n]))
(R / "eval" / "per_item_table.md").write_text("\n".join(L) + "\n")
print("wrote", R / "eval" / "per_item_table.md", len(meta), "items")
