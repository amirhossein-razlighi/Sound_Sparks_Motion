#!/bin/bash
# Body of the interactive smoke test (salloc ... srun bash smoke_body.sh); must live on shared storage.
source /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/job_env.sh
python noiseopt/run_noiseopt.py --scenario dog_yawning --output-dir /scratch/amirrz/iclr_rebuttal/noiseopt/smoke_dog_interac_$SLURM_JOB_ID --iter-every 1 --set iterations=3
echo "SMOKE rc=$?"
