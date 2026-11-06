#!/bin/bash
# TRILLIUM: pack every finished IVEBench run (rc.txt present) not yet packed into
#   results_trillium/ivebench/<name>__<cfg>.tgz.b64  (baseline_video.mp4, best_optimized_video_both.mp4, run_config.json,
#   timing.json, optimization_log_qwen_both.csv, rc.txt) + results_trillium/ivebench/runs.csv (one row per run).
# Prints the names packed in this call. No mp4 is committed directly (LFS budget exceeded).
R=${IVE_ROOT:-/scratch/amirrz/iclr_rebuttal/ivebench}
G=/home/amirrz/my_codes/Sound_Sparks_Motion/results_trillium/ivebench; mkdir -p "$G"
for rcf in "$R"/runs/*/*/rc.txt; do
  [[ -f $rcf ]] || continue
  d=$(dirname "$rcf"); cfg=$(basename "$d"); n=$(basename "$(dirname "$d")"); out=$G/${n}__${cfg}.tgz.b64
  [[ -f $out ]] && continue
  files=(); for f in baseline_video.mp4 best_optimized_video_both.mp4 run_config.json timing.json optimization_log_qwen_both.csv rc.txt OOM.json; do
    [[ -f $d/$f ]] && files+=("$f"); done
  tar -C "$d" -czf - "${files[@]}" | base64 > "$out"
  echo "$n $cfg $(cat "$rcf")"
done
python3 - "$R" "$G" <<'PY'
import csv, json, sys
from pathlib import Path
R, G = Path(sys.argv[1]), Path(sys.argv[2]); rows = []
notes = json.loads((G / "notes.json").read_text()) if (G / "notes.json").exists() else {}
for rc in sorted(R.glob("runs/*/*/rc.txt")):
    d = rc.parent; t = json.loads((d / "timing.json").read_text()) if (d / "timing.json").exists() else {}
    best = ("", "")
    lg = d / "optimization_log_qwen_both.csv"
    if lg.exists():
        b = [r for r in csv.DictReader(lg.open()) if str(r.get("is_best", "")).strip() in ("1", "True", "true")]
        if b: best = (b[-1]["iter"], b[-1]["qwen_yes_prob"])
    rows.append(dict(name=d.parent.name, cfg=d.name, rc=rc.read_text().strip(), oom=(d / "OOM.json").exists(),
                     n_iters=t.get("n_iterations", ""), mean_iter_s=t.get("mean_iter_s_post_warmup", ""),
                     peak_gb=t.get("peak_mem_gb_loop", ""), wall_s=t.get("wall_total_s", ""), best_iter=best[0],
                     inloop_best_yes=best[1], gpu=t.get("gpu", ""), output_dir=f"trillium:{d}",
                     notes=notes.get(f"{d.parent.name}:{d.name}", "")))
if rows:
    with open(G / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
PY
