#!/bin/bash
# TRILLIUM variant of eval_official.sh (same metric list, same frame preparation): IVEBench clone + its own venv on /project,
# checkpoints in /project/def-amahdavi/amirrz/IVEBench_ckpt (path.yml edited in the clone; original kept as path.yml.orig).
#   cd /scratch/amirrz/iclr_rebuttal && sbatch --gpus-per-node=2 <repo>/editing/iclr_rebuttal/ivebench/trillium_eval_official.sbatch <tag> <list.csv: name,video>
# Source frames = our prepared input mp4 (exactly what the method saw); target = the method output. quantity_accuracy omitted.
#SBATCH --job-name=ssm-iveval
#SBATCH --account=def-amahdavi
#SBATCH --partition=debug
#SBATCH --gpus-per-node=2
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/amirrz/iclr_rebuttal/logs/iveval_%j.out
set -o pipefail
TAG=${1:?tag}; LIST=${2:?list.csv}
R=/scratch/amirrz/iclr_rebuttal/ivebench; C=/project/def-amahdavi/amirrz/IVEBench/metrics; OUT=$R/eval/$TAG; mkdir -p $OUT
T=${SLURM_TMPDIR:-/dev/shm/amirrz_${SLURM_JOB_ID}}/ive_eval/$TAG; mkdir -p $T/src $T/tgt
source /project/def-amahdavi/amirrz/ivebench_venv/bin/activate
export TORCH_HOME=/scratch/amirrz/torch HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/scratch/amirrz/huggingface
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; echo "host=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is)"
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
           instruction_satisfaction phrase_semantic_consistency semantic_fidelity motion_fidelity content_fidelity > $OUT/evaluate.log 2>&1
rc=$?; cp $C/*_ivebench.log $OUT/ 2>/dev/null
echo "rc=$rc" > $OUT/rc.txt; tail -40 $OUT/evaluate.log; ls $OUT
