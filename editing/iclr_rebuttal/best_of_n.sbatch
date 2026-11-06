#!/bin/bash
# P1 best-of-N: one task per scenario (12). Usage: sbatch --array=0-11%3 best_of_n.sbatch
#SBATCH --job-name=ssm-bon
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --account=def-amahdavi
#SBATCH --exclude=rg32102,rg31801
#SBATCH --output=/scratch/amirrz/iclr_rebuttal/logs/bon_%A_%a.out
set -euo pipefail
source /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/job_env.sh
SCEN=(bugatti_lights_flash red_car_door_opens groom_raising_hand dog_yawning falcon_bird_opening_wings goldfish_jumping man_climbs_rocks man_pets_dog monkey_reaching_fruit red_rose_blooming robot_waving turtle_neck)
S=${SCEN[$SLURM_ARRAY_TASK_ID]}
python best_of_n.py --scenario "$S" ${BON_SEEDS:+--seeds "$BON_SEEDS"} ${BON_OUT:+--out-root "$BON_OUT"}
echo "end=$(date -Is)"
