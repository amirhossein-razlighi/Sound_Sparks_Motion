#!/usr/bin/env python3
"""Per-item selection of OURS and RETAKE for the official IVEBench eval (CPU, reads only existing CSVs).

Candidates per item = every config run (ours video, Retake = that config's baseline video) + eye-saved ours iterations.
Visible-problem runs (frame checks: Rorqual notes.csv + Trillium notes.json, see VERDICT_T below) are excluded per side.

Two rules, applied IDENTICALLY to ours and to Retake:
  critic : highest held-out qwen25 yes-prob with the target_PHRASE question (honest edit question; '@phrase' rows),
           tie-break target_prompt score. If every candidate of a side has a visible problem, the best one is kept and flagged.
  eye    : best frame-check verdict (GOOD > ok+ > ok > partial+ > partial > weak+ > weak > fail > problem), tie-break = critic score.
Outputs: $R/selection.csv (all candidates + both picks), $R/eval_lists/{ours,retake}_{critic,eye}.csv (name,video).
"""
import csv, json
from collections import defaultdict
from pathlib import Path
R = Path("/scratch/amirrz/iclr_rebuttal/ivebench"); G = Path("/home/amirrz/my_codes/Sound_Sparks_Motion/results_trillium/ivebench")
items = [Path(x["src_video_name"]).stem for x in json.load(open(R / "subset_subject_motion_short.json"))]
RANK = {"good+": 9, "good": 8, "ok+": 7, "ok": 6, "partial+": 5, "partial": 4, "weak+": 3, "weak": 2, "fail": 1, "unknown": 1, "problem": 0}
# Trillium frame-check verdicts (retake, ours), transcribed from results_trillium/ivebench/notes.json (text kept there).
VERDICT_T = {
 "short_0027:A": ("ok", "weak"), "short_0027:B": ("ok", "weak+"), "short_0027:C": ("ok", "problem"), "short_0027:E": ("ok", "weak"),
 "short_0042:E": ("ok", "ok"), "short_0057:A": ("fail", "fail"), "short_0057:B": ("fail", "partial"),
 "short_0066:E": ("problem", "problem"), "short_0066:F": ("problem", "problem"), "short_0083:E": ("unknown", "unknown"),
 "short_0095:E": ("problem", "problem"), "short_0095:F": ("problem", "partial"), "short_0106:B": ("weak", "weak"),
 "short_0106:E": ("weak", "weak"), "short_0113:E": ("good", "good"), "short_0127:E": ("good", "good+"),
 "short_0136:E": ("unknown", "unknown"), "short_0158:E": ("fail", "fail"), "short_0158:G": ("fail", "fail"),
 "short_0170:E": ("problem", "problem"), "short_0176:E": ("partial", "partial"), "short_0190:E": ("fail", "fail"),
 "short_0202:E": ("fail", "fail"), "short_0202:G": ("fail", "fail"), "short_0221:E": ("good+", "good"),
 "short_0228:E": ("problem", "problem"), "short_0246:E": ("partial", "partial"), "short_0260:E": ("fail", "fail"),
 "short_0262:E": ("good", "good"), "short_0266:E": ("good", "good"), "short_0273:E": ("weak", "weak"),
 "short_0289:E": ("partial", "partial+"), "short_0305:A": ("problem", "problem"),
}
# 0095:F retake = 'blurred reframe' (heavy blur; E baseline note says reframe with heavy motion blur) -> treated as problem like E.
# 0228:E 'background degradation' (blur + pillar shapes) -> problem. 0057:B ours inserts a plate (kept: partial, flagged in notes).
def norm(v):
    v = v.strip().lower()
    for k in ("problem", "good", "ok+", "ok", "partial+", "partial", "weak+", "weak", "fail"):
        if v.startswith(k):
            return "ok" if k == "ok" and v.startswith("ok-ish") else k
    return {"scene redrawn": "weak", "fail-ish": "fail"}.get(v, "unknown")
V = {}
for k, (rv, ov) in VERDICT_T.items():
    V[(k, "retake")], V[(k, "ours")] = rv, ov
for r in csv.DictReader(open(R / "notes.csv")):        # Rorqual frame checks (later rows override earlier ones)
    k = f"{r['name']}:{r['cfg']}"; V[(k, "retake")], V[(k, "ours")] = norm(r["verdict_retake"]), norm(r["verdict_ours"])
    if "problem in both" in r["notes"]:
        V[(k, "retake")] = V[(k, "ours")] = "problem"
# final-video overrides where the notes say the FINAL ours has a visible problem but an earlier saved iteration does not
V[("short_0174:E", "ours")] = "problem"   # final: tram swings into the camera, large drift (iter_010 = clean take-off)
V[("short_0045:A", "ours")] = "problem"   # iter_020+ distort the last frames (iter_010 = cleaner)
# eye-saved iterations (Rorqual): ours candidate with its own verdict
ITER_V = {("short_0174", "E", "iter_010"): "good", ("short_0045", "A", "iter_010"): "partial"}
S = defaultdict(dict)   # S[id_without_phrase]["prompt"/"phrase"] = yes_prob ; meta
for r in csv.DictReader(open(R / "heldout" / "qwen25.csv")):
    if r.get("yes_prob") in ("", None):
        continue
    base, ph = (r["id"][:-7], "phrase") if r["id"].endswith("@phrase") else (r["id"], "prompt")
    S[base][ph] = float(r["yes_prob"]); S[base]["video"] = r["video"]; S[base]["scenario"] = r["scenario"]
rows, picks = [], defaultdict(dict)
for n in items:
    cands = []
    for base, s in S.items():
        p = base.split("/")
        if len(p) != 4 or p[1] != n or "phrase" not in s:
            continue
        cfg, kind = p[2], p[3]
        side = "retake" if kind == "retake" else "ours"
        if kind.startswith("ours-iter"):
            it = kind.split("-", 1)[1]; v = ITER_V.get((n, cfg, it), V.get((f"{n}:{cfg}", "ours"), "unknown"))
        elif kind in ("ours", "retake"):
            it = "final"; v = V.get((f"{n}:{cfg}", side), "unknown")
        else:
            continue
        cands.append(dict(item=n, side=side, cfg=cfg, iter=it, verdict=v, q_phrase=s["phrase"], q_prompt=s.get("prompt", ""),
                          video=s["video"], id=base))
    for side in ("ours", "retake"):
        cs = [c for c in cands if c["side"] == side]
        if not cs:
            continue
        ok = [c for c in cs if c["verdict"] != "problem"]
        crit = max(ok or cs, key=lambda c: (c["q_phrase"], c["q_prompt"] or 0))
        eye = max(cs, key=lambda c: (RANK[c["verdict"]], c["q_phrase"]))
        picks[n][side] = dict(critic=crit, eye=eye, all_problem=not ok)
        for c in cs:
            c["pick_critic"] = int(c is crit); c["pick_eye"] = int(c is eye); c["side_all_problem"] = int(not ok)
    rows += cands
cols = ["item", "side", "cfg", "iter", "verdict", "q_phrase", "q_prompt", "pick_critic", "pick_eye", "side_all_problem", "id", "video"]
with open(R / "selection.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
(R / "eval_lists").mkdir(exist_ok=True)
for side in ("ours", "retake"):
    for rule in ("critic", "eye"):
        with open(R / "eval_lists" / f"{side}_{rule}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["name", "video", "cfg", "iter", "verdict", "q_phrase"])
            for n in items:
                if side in picks[n]:
                    c = picks[n][side][rule]; w.writerow([n, c["video"], c["cfg"], c["iter"], c["verdict"], f"{c['q_phrase']:.4f}"])
missing = [n for n in items if "ours" not in picks[n] or "retake" not in picks[n]]
print("items", len(items), "with both picks", len(items) - len(missing), "missing", missing)
print("candidates", len(rows), "| verdict unknown:", sorted({f"{c['item']}:{c['cfg']}" for c in rows if c["verdict"] == "unknown"}))
for rule in ("critic", "eye"):
    d = sum(picks[n]["ours"][rule]["cfg"] != picks[n]["retake"][rule]["cfg"] for n in items if n not in missing)
    print(rule, "items where ours and retake picks come from different configs:", d)
