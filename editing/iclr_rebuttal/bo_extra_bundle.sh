#!/bin/bash
# Continuous Dynamic Degree on the ours+best-of-N picks and VBench + continuous DD on the two prompt-only seed-2 picks
# (the only prompt-only best-of-4 picks not already covered by the best-of-3 / best-of-8 runs).
set -x
source /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/job_env.sh
R=/scratch/amirrz/iclr_rebuttal; M=$R/manifests
export VBENCH_CACHE_DIR=$R/vbench_cache
env -u PYTHONPATH $R/venv_vbench/bin/python p7_dyn_cont.py --manifest $M/bon_ours_picks.csv --out $R/p7_bono/vbench_dyn_continuous.csv; echo "DYN1 rc=$?"
env -u PYTHONPATH $R/venv_vbench/bin/python p7_dyn_cont.py --manifest $M/prompt_s2.csv --out $R/p7_bono_s2/vbench_dyn_continuous.csv; echo "DYN2 rc=$?"
deactivate; unset PYTHONPATH
MANIFEST=$M/prompt_s2.csv OUT=$R/p7_bono_s2 bash /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/p7_vbench_only.sh
