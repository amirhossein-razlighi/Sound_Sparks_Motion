#!/bin/bash
# Start the lane on an interactive allocation: stage models to node-local disk, then N workers (1 GPU each).
#   bash launch_lane.sh <ALLOC_JOBID> [n_workers=4]
ALLOC=$1; N=${2:-4}; D=/home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal; LG=/scratch/amirrz/iclr_rebuttal/logs
echo "$(date -Is) staging on $ALLOC" >> /scratch/amirrz/iclr_rebuttal/lane/events.txt
srun --jobid=$ALLOC --gres=none --cpus-per-task=8 --mem=16G -n1 bash $D/stage.sh > $LG/stage_$ALLOC.out 2>&1
echo "$(date -Is) staged rc=$?" >> /scratch/amirrz/iclr_rebuttal/lane/events.txt
for w in $(seq 0 $((N-1))); do
  setsid nohup srun --jobid=$ALLOC --gres=gpu:h100:1 --cpus-per-task=6 --mem=60G --exact -n1 bash $D/worker.sh $w > $LG/worker_${ALLOC}_$w.out 2>&1 < /dev/null &
  sleep 2
done
echo "$(date -Is) started $N workers" >> /scratch/amirrz/iclr_rebuttal/lane/events.txt
