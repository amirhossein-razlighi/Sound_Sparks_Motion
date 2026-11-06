#!/bin/bash
# Official IVEBench metrics on a set of outputs. Needs >= 3 H100 visible (Qwen2.5-VL-72B judges, device_map=auto).
#   bash ivebench/eval_official.sh <tag> <list.csv: name,video>      (source = our prepared input, exactly what the method saw)
# Frames go to node-local disk; only CSVs/logs come back to $R/eval/<tag>/.
set -o pipefail
TAG=$1; LIST=$2
R=${IVE_ROOT:-/scratch/amirrz/iclr_rebuttal/ivebench}; C=$R/code/metrics; OUT=$R/eval/$TAG; mkdir -p $OUT
T=${SLURM_TMPDIR:-/localscratch/amirrz.${SLURM_JOB_ID}.0}/ive_eval/$TAG; mkdir -p $T/src $T/tgt
source /home/amirrz/my_codes/Sound_Sparks_Motion/.venv/bin/activate
export PYTHONPATH=$R/pydeps${PYTHONPATH:+:$PYTHONPATH} TORCH_HOME=$R/torch_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python - "$LIST" "$R" "$T" <<'PY'
import csv, json, subprocess, sys
from pathlib import Path
lst, R, T = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
rows = list(csv.DictReader(open(lst)))
items = {Path(x["src_video_name"]).stem: x for x in json.load(open(R / "subset_subject_motion_short.json"))}
info = []
for r in rows:
    n = r["name"]
    for kind, v in (("src", R / "inputs" / f"{n}.mp4"), ("tgt", Path(r["video"]))):
        d = T / kind / n; d.mkdir(parents=True, exist_ok=True)
        if not any(d.iterdir()):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(v), "-start_number", "0", str(d / "%05d.png")], check=True)
    info.append(items[n])
json.dump(info, open(T / "info.json", "w"), indent=1)
print(len(info), "videos framed")
PY
cd $C
python evaluate.py --output_path $OUT --source_videos_path $T/src --target_videos_path $T/tgt --info_json_path $T/info.json --name $TAG \
  --metric subject_consistency temporal_flickering background_consistency motion_smoothness vtss overall_semantic_consistency \
           instruction_satisfaction phrase_semantic_consistency semantic_fidelity motion_fidelity content_fidelity 2>&1 | tail -200 > $OUT/evaluate_tail.log
rc=$?; cp $C/*_ivebench.log $OUT/ 2>/dev/null
for f in $OUT/*.csv; do [[ $f == *average* ]] && continue; python get_average_score.py -i "$f" -o "${f%.csv}_average.csv" || true; done
echo "rc=$rc" > $OUT/rc.txt; ls $OUT
