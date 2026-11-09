#!/bin/bash
# Held-out VLMs, RAFT flow, DINOv2 source fidelity, continuous Dynamic Degree and VBench on the noise-optimization finals.
set -x
source /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/job_env.sh
R=/scratch/amirrz/iclr_rebuttal; M=$R/manifests; O=$R/noiseopt/metrics
python heldout_score.py --manifest $M/noiseopt_with_input.csv --out $O --models qwen25,qwen3vl,internvl3; echo "HELD rc=$?"
python p7_flow.py --manifest $M/noiseopt.csv --out $O; echo "FLOW rc=$?"
python source_fidelity.py $M/noiseopt_with_input.csv $O/source_fidelity_dinov2.csv; echo "FID rc=$?"
export VBENCH_CACHE_DIR=$R/vbench_cache
env -u PYTHONPATH $R/venv_vbench/bin/python p7_dyn_cont.py --manifest $M/noiseopt.csv --out $O/vbench_dyn_continuous.csv; echo "DYN rc=$?"
deactivate; unset PYTHONPATH
MANIFEST=$M/noiseopt.csv OUT=$O bash /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/p7_vbench_only.sh
