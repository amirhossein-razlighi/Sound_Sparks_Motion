#!/bin/bash
# DINOv2 fidelity for prompt-only best-of-3 (was missing from section 5), then VBench on ours + best-of-N picks.
source /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/job_env.sh
R=/scratch/amirrz/iclr_rebuttal
python source_fidelity.py $R/manifests/bon3_with_input.csv $R/p7_bono/source_fidelity_bon3.csv; echo "FID rc=$?"
deactivate 2>/dev/null; unset PYTHONPATH
export MANIFEST=$R/manifests/bon_ours_picks.csv OUT=$R/p7_bono
bash /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/p7_vbench_only.sh
