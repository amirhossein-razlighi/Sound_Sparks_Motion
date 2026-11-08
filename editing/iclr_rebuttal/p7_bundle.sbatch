#!/bin/bash
# Non-VLM metric bundle of section 5 for one manifest: RAFT flow + warp error, DINOv2 source fidelity, VBench consistency /
# smoothness / flicker, VBench quality + Dynamic Degree, continuous Dynamic Degree. Env: MANIFEST (method rows), MANIFEST_SRC (same + input rows), OUT.
#SBATCH --job-name=ssm-p7
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --account=def-amahdavi
#SBATCH --output=/scratch/amirrz/iclr_rebuttal/p7_bon8/logs/p7_%j.out
set -x
source /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/job_env.sh
python p7_flow.py --manifest $MANIFEST --out $OUT; echo "FLOW rc=$?"
python source_fidelity.py $MANIFEST_SRC $OUT/source_fidelity_dinov2.csv; echo "FID rc=$?"
export VBENCH_CACHE_DIR=/scratch/amirrz/iclr_rebuttal/vbench_cache; PYV=/scratch/amirrz/iclr_rebuttal/venv_vbench/bin/python
env -u PYTHONPATH $PYV p7_vbench.py --manifest $MANIFEST --out $OUT; echo "VB rc=$?"
env -u PYTHONPATH $PYV p7_vbench.py --manifest $MANIFEST --out $OUT --dims imaging_quality,aesthetic_quality,dynamic_degree --tag quality; echo "VQ rc=$?"
env -u PYTHONPATH $PYV p7_dyn_cont.py --manifest $MANIFEST --out $OUT/vbench_dyn_continuous.csv; echo "DYN rc=$?"
