#!/bin/bash
# 1-GPU job: bring the multi-critic report up to date (held-out scoring of runs not yet scored, encoding check, metrics,
# report, curves, package), then run single-GPU A tasks from q1.txt until the queue is empty.
#SBATCH --job-name=ssm-mcu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --account=def-amahdavi
#SBATCH --exclude=rg32102,rg31801,rg21702
#SBATCH --output=/scratch/amirrz/iclr_rebuttal/multicritic/logs/mcu_%j.out
set -x
ROOT=/scratch/amirrz/iclr_rebuttal/multicritic
H=/home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/multicritic
source /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/job_env.sh
echo "MC_UPDATE start job=$SLURM_JOB_ID" >> $ROOT/events.txt
python $H/mc_manifest.py
TAG=extra_$(date +%m%d%H%M); D=$ROOT/heldout_split/$TAG; mkdir -p $D
python - "$ROOT" "$D" <<'PY'
import csv, sys
from pathlib import Path
root, d = Path(sys.argv[1]), Path(sys.argv[2])
rows = list(csv.DictReader(open(root / "manifest.csv")))
done = {m: ({r["id"] for r in csv.DictReader(open(root / "heldout" / f"{m}.csv"))} if (root / "heldout" / f"{m}.csv").exists() else set())
        for m in ("qwen25", "qwen3vl", "internvl3")}
main = [r for r in rows if r["group"] != "iterate" and not all(r["id"] in done[m] for m in done)]
it = [r for r in rows if r["group"] == "iterate" and r["id"] not in done["internvl3"]]
for name, rr in (("main", main), ("iter", it)):
    with open(d / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rr)
    print(name, len(rr))
PY
python heldout_score.py --manifest $D/main.csv --out $D --models internvl3,qwen25,qwen3vl > $D/log_main.txt 2>&1
python heldout_score.py --manifest $D/iter.csv --out $D/iter --models internvl3 > $D/log_iter.txt 2>&1
python $H/mc_merge_heldout.py
[[ -e $ROOT/encoding_check.csv ]] && mv $ROOT/encoding_check.csv $ROOT/encoding_check.bak_$(date +%m%d%H%M).csv
python $H/check_encoding.py > $ROOT/logs/check_encoding_$SLURM_JOB_ID.log 2>&1
OMP_NUM_THREADS=6 python $H/mc_metrics.py > $ROOT/logs/metrics_$SLURM_JOB_ID.log 2>&1
python $H/mc_report.py && /scratch/amirrz/iclr_rebuttal/venv_vbench/bin/python $H/mc_curves.py && \
  /scratch/amirrz/iclr_rebuttal/venv_vbench/bin/python -c "
import zipfile; z=zipfile.ZipFile('$ROOT/multicritic_package.zip','a'); z.write('$ROOT/report/curves.png','multicritic/report/curves.png'); z.close()"
echo "MC_UPDATE report rc=$? job=$SLURM_JOB_ID" >> $ROOT/events.txt
[[ -z $NO_WORKER ]] && ONLY_Q1=1 IDLE_MAX=4 bash $H/mc_worker.sh u$SLURM_JOB_ID
echo "MC_UPDATE done job=$SLURM_JOB_ID" >> $ROOT/events.txt
