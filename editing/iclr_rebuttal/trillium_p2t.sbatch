#!/bin/bash
# TRILLIUM P2-T: held-out scoring of Trillium outputs (same protocol as Rorqual P2). Resumable (scripts skip done ids).
#   cd /scratch/amirrz/iclr_rebuttal && sbatch <repo>/editing/iclr_rebuttal/trillium_p2t.sbatch <manifest.csv> <out_dir>
#SBATCH --job-name=ssm-p2t
#SBATCH --account=def-amahdavi
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/amirrz/iclr_rebuttal/logs/p2t_%j.out
set -uo pipefail
M=${1:?manifest}; OUT=${2:?out dir}
source /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/job_env_trillium.sh
mkdir -p "$OUT"
python loop_rescore.py --manifest "$M" --out "$OUT" || echo "loop_rescore rc=$?"
python heldout_score.py --manifest "$M" --out "$OUT" --models qwen25,qwen3vl,internvl3; echo "heldout_score rc=$?"
echo "end=$(date -Is)"
