#!/usr/bin/env python3
"""Download package for the IVEBench study (run on a compute node: ffmpeg encodes 64 side-by-sides).
$R/package/ivebench_package/<group>/<item>__<edit>/ :
  0_source.mp4 (prepared input the methods saw), 1_ours_criticpick.mp4, 2_retake_criticpick.mp4,
  3_ours_eyepick.mp4 / 4_retake_eyepick.mp4 (only when the eye pick differs from the critic pick),
  side_by_side.mp4 (SOURCE | RETAKE | OURS [critic picks] + eye-pick panels when different, labelled with verdict + official judge
  scores), info.txt (edit, every config run, picks, verdicts, qwen25 scores, official IVEBench scores, frame-check notes verbatim).
Groups (by the eye picks' frame-check verdicts): 1_ours_better, 2_both_work_equal, 3_retake_better, 4_both_fail, 5_problem_in_both,
6_not_verifiable. Also INDEX.csv, README.txt, FINDINGS_section15.md, eval summaries. Zips: ivebench_package.zip (all) and
ivebench_side_by_side.zip (side-by-sides only, flat, prefixed by group)."""
import csv, glob, json, shutil, subprocess, sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
R = Path("/scratch/amirrz/iclr_rebuttal/ivebench"); G = Path("/home/amirrz/my_codes/Sound_Sparks_Motion/results_trillium/ivebench")
FONT = "/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
OUT = R / "package" / "ivebench_package"; SBS = R / "package" / "ivebench_side_by_side"
for d in (OUT, SBS):
    if d.exists():
        shutil.move(str(d), str(d) + ".old_" + str(int(d.stat().st_mtime)))
    d.mkdir(parents=True)
meta = {Path(x["src_video_name"]).stem: x for x in json.load(open(R / "subset_subject_motion_short.json"))}
sel = pd.read_csv(R / "selection.csv")
L = {t: {r["name"]: r for r in csv.DictReader(open(R / "eval_lists" / f"{t}.csv"))} for t in ("ours_critic", "retake_critic", "ours_eye", "retake_eye")}
sys.path.insert(0, str(Path(__file__).parent))
import aggregate_eval as AE  # noqa: E402  (reuses its patch-aware loader)
EV = {t: AE.load(t) for t in ("ours_critic", "retake_critic", "ours_eye", "retake_eye", "source_noedit")}
notes = defaultdict(list)
for r in csv.DictReader(open(R / "notes.csv")):
    notes[r["name"]].append(f"[Rorqual {r['cfg']}] retake={r['verdict_retake']} ours={r['verdict_ours']}: {r['notes']}")
for k, v in json.load(open(G / "notes.json")).items():
    n, c = k.split(":"); notes[n].append(f"[Trillium {c}] {v}")
RANK = {"good+": 9, "good": 8, "ok+": 7, "ok": 6, "partial+": 5, "partial": 4, "weak+": 3, "weak": 2, "fail": 1, "problem": 0}
def group(vo, vr):
    if "unknown" in (vo, vr): return "6_not_verifiable"
    if vo == "problem" and vr == "problem": return "5_problem_in_both"
    ro, rr = RANK[vo], RANK[vr]
    if ro > rr: return "1_ours_better"
    if ro < rr: return "3_retake_better"
    return "2_both_work_equal" if ro >= RANK["partial"] else "4_both_fail"
def sc(tag, n):
    d = EV[tag]
    if n not in d.index: return "IS - CF -"
    return f"IS {d.loc[n,'instruction_satisfaction_score']:.0f} CF {d.loc[n,'content_fidelity_score']:.0f}"
esc = lambda s: s.replace("'", "").replace(":", " ").replace("%", " pct").replace(",", " ")
def sbs(panels, title, dst):
    a, fl = [], []
    for i, (v, lab) in enumerate(panels):
        a += ["-i", str(v)]
        fl.append(f"[{i}:v]fps=25,scale=448:252:force_original_aspect_ratio=decrease,pad=448:252:(ow-iw)/2:(oh-ih)/2,setsar=1,"
                  f"drawtext=fontfile={FONT}:text='{esc(lab)}':x=5:y=5:fontsize=13:fontcolor=white:box=1:boxcolor=black@0.6[v{i}]")
    k = len(panels)
    fl.append("".join(f"[v{i}]" for i in range(k)) + f"hstack=inputs={k},pad=iw:ih+28:0:28:black,"
              f"drawtext=fontfile={FONT}:text='{esc(title)}':x=8:y=6:fontsize=15:fontcolor=yellow[o]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", *a, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", "-crf", "23", "-shortest", str(dst)], check=True)
idx = []
for n in sorted(meta):
    oc, rc, oe, re_ = (L[t][n] for t in ("ours_critic", "retake_critic", "ours_eye", "retake_eye"))
    g = group(oe["verdict"], re_["verdict"])
    vis = bool({oc["verdict"], rc["verdict"]} & {"partial", "partial+", "ok", "ok+", "good", "good+"})
    top = "A_edit_visible_in_ours_or_retake" if vis else "B_rest_edit_not_visible_or_problem"
    edit = meta[n]["edit_prompt"]
    slug = "".join(ch if ch.isalnum() else "_" for ch in edit.lower())[:50].strip("_")
    d = OUT / top / f"{g}__{n}__{slug}"; d.mkdir(parents=True)
    src = R / "inputs" / f"{n}.mp4"
    shutil.copy(src, d / "0_source.mp4")
    tagf = lambda r: f"{r['cfg']}{'' if r['iter']=='final' else '_' + r['iter']}"
    shutil.copy(oc["video"], d / f"1_ours_criticpick_{tagf(oc)}.mp4")
    shutil.copy(rc["video"], d / f"2_retake_criticpick_{tagf(rc)}.mp4")
    panels = [(src, f"SOURCE  {sc('source_noedit', n)}"),
              (rc["video"], f"RETAKE {tagf(rc)} [{rc['verdict']}]  {sc('retake_critic', n)}"),
              (oc["video"], f"OURS {tagf(oc)} [{oc['verdict']}]  {sc('ours_critic', n)}")]
    if oe["video"] != oc["video"]:
        shutil.copy(oe["video"], d / f"3_ours_eyepick_{tagf(oe)}.mp4"); panels.append((oe["video"], f"OURS eye pick {tagf(oe)} [{oe['verdict']}]  {sc('ours_eye', n)}"))
    if re_["video"] != rc["video"]:
        shutil.copy(re_["video"], d / f"4_retake_eyepick_{tagf(re_)}.mp4"); panels.append((re_["video"], f"RETAKE eye pick {tagf(re_)} [{re_['verdict']}]  {sc('retake_eye', n)}"))
    sbs(panels, f"{n}  {edit}", d / "side_by_side.mp4")
    shutil.copy(d / "side_by_side.mp4", SBS / f"{top[0]}_{g}__{n}.mp4")
    # ALL runs of the item: every config's final ours + its Retake render, and every saved iteration preview (Rorqual runs keep
    # them in run_full.tar; Trillium runs shipped finals only) + eye-saved iterations
    ad = d / "all_runs"; ad.mkdir()
    import tarfile
    for rd in sorted((R / "runs" / n).iterdir()):
        if not (rd / "best_optimized_video_both.mp4").exists():
            continue
        c = rd.name
        shutil.copy(rd / "best_optimized_video_both.mp4", ad / f"{c}_ours_final.mp4")
        shutil.copy(rd / "baseline_video.mp4", ad / f"{c}_retake.mp4")
        for it in rd.glob("iter_*.mp4"):
            shutil.copy(it, ad / f"{c}_ours_{it.stem}_eyesaved.mp4")
        if (rd / "run_full.tar").exists():
            with tarfile.open(rd / "run_full.tar") as tf:
                for m in tf.getmembers():
                    if m.name.endswith("best_optimized_video_both.mp4") and "/visualizations/iter_" in m.name:
                        itn = m.name.split("/visualizations/")[1].split("/")[0]
                        (ad / f"{c}_ours_{itn}_bestsofar.mp4").write_bytes(tf.extractfile(m).read())
    cands = sel[sel.item == n].sort_values(["side", "cfg", "iter"])
    info = [f"item: {n}   group: {g}", f"edit instruction (IVEBench edit_prompt): {edit}",
            f"target_phrase (critic question for selection): {meta[n]['target_phrase']}", f"target_prompt (LTX prompt): {meta[n]['target_prompt']}",
            f"frames: {json.load(open(R / 'inputs' / f'{n}.json'))['n']}", "",
            "PICKS (same rule for ours and Retake; critic = no visible problem, then max qwen25 phrase score; eye = best frame-check verdict)",
            f"  ours   critic pick: {tagf(oc)} verdict={oc['verdict']} q_phrase={oc['q_phrase']}   official: {sc('ours_critic', n)}",
            f"  retake critic pick: {tagf(rc)} verdict={rc['verdict']} q_phrase={rc['q_phrase']}   official: {sc('retake_critic', n)}",
            f"  ours   eye pick:    {tagf(oe)} verdict={oe['verdict']} q_phrase={oe['q_phrase']}   official: {sc('ours_eye', n)}",
            f"  retake eye pick:    {tagf(re_)} verdict={re_['verdict']} q_phrase={re_['q_phrase']}   official: {sc('retake_eye', n)}",
            f"  unedited source:    official: {sc('source_noedit', n)}", "",
            "ALL CANDIDATES (side cfg iter verdict q_phrase q_prompt):"]
    info += [f"  {r.side:6s} {r.cfg} {r['iter']:9s} {r.verdict:8s} {r.q_phrase:.4f} {r.q_prompt:.4f}" for _, r in cands.iterrows()]
    info += ["", "FRAME-CHECK NOTES (verbatim):"] + [f"  {x}" for x in notes[n]]
    (d / "info.txt").write_text("\n".join(info) + "\n")
    idx.append(dict(item=n, group=g, edit=edit, ours_critic=tagf(oc), ours_verdict=oc["verdict"], retake_critic=tagf(rc), retake_verdict=rc["verdict"],
                    ours_eye=tagf(oe), ours_eye_verdict=oe["verdict"], retake_eye=tagf(re_), retake_eye_verdict=re_["verdict"],
                    ours_official=sc("ours_critic", n), retake_official=sc("retake_critic", n), source_official=sc("source_noedit", n), subset31=int(vis), folder=str(d.relative_to(OUT))))
    print(n, g, flush=True)
pd.DataFrame(idx).to_csv(OUT / "INDEX.csv", index=False)
ev = OUT / "eval_tables"; ev.mkdir()
for f in glob.glob(str(R / "eval" / "summary*")) + glob.glob(str(R / "eval" / "per_item*")):
    shutil.copy(f, ev)
shutil.copy(R / "selection.csv", ev); shutil.copy(R / "notes.csv", ev / "notes_rorqual.csv"); shutil.copy(G / "notes.json", ev / "notes_trillium.json")
F = open("/scratch/amirrz/iclr_rebuttal/FINDINGS_for_rebuttal.md").read()
i = F.index("## 15. IVEBench"); j = F.index("\n## 13.", i)
(OUT / "FINDINGS_section15.md").write_text(F[i:j])
cnt = pd.DataFrame(idx).group.value_counts().sort_index()
(OUT / "README.txt").write_text(
    "IVEBench subject_motion_editing (64 items): ours (test-time tuned audio latent + text residual) vs prompt-only LTX Retake.\n"
    "TOP LEVEL: A_edit_visible_in_ours_or_retake = the 31-item subset (frame check: ours OR Retake critic pick at least partly does\n"
    "the edit; symmetric but post-hoc), B_rest = the other 33. Item folders are prefixed by the outcome of the EYE picks: " + ", ".join(f"{k} ({v})" for k, v in cnt.items()) + ".\n"
    "Each item: 0_source, 1_ours_criticpick_<cfg>, 2_retake_criticpick_<cfg>, (3_ours_eyepick / 4_retake_eyepick when different),\n"
    "side_by_side.mp4 = SOURCE | RETAKE | OURS (+ eye picks), labels: [frame-check verdict] and official IVEBench judge scores\n"
    "(IS = instruction satisfaction 1-5, CF = content fidelity 1-5, Qwen2.5-VL-72B). info.txt = everything about the item.\n"
    "Critic pick = the automatic rule (no visible problem, best held-out Qwen2.5-VL-7B score on the short target phrase); eye pick =\n"
    "best frame-check verdict. Same rule for ours and Retake. Configs: A K=5/8 frames, B K=5/24 frames, C K=1, E K=25, F E+LPIPS/temporal,\n"
    "G E with the phrase critic, H A with the phrase critic; iter_010 = an earlier saved iteration (final had a visible problem).\n"
    "all_runs/ = EVERY run of the item: <cfg>_ours_final, <cfg>_retake (prompt-only render of that config), <cfg>_ours_iter_NNN_bestsofar\n"
    "(best-so-far preview saved every 5 iterations; Rorqual runs only), <cfg>_ours_iter_NNN_eyesaved (iterations picked by eye).\n"
    "Tables: eval_tables/ (official per-set summaries, per-item CSVs, selection.csv, notes). Full write-up: FINDINGS_section15.md.\n")
for src, z in ((OUT, "ivebench_package"), (SBS, "ivebench_side_by_side")):
    zp = R / "package" / f"{z}.zip"
    if zp.exists():
        shutil.move(str(zp), str(zp)[:-4] + f".old_{int(zp.stat().st_mtime)}.zip")
    shutil.make_archive(str(R / "package" / z), "zip", root_dir=src.parent, base_dir=src.name)
print("done", cnt.to_dict())
