#!/bin/bash
# Held-out VLM scoring of the multi-critic outputs, one slice per GPU (run inside a 1-GPU step).
#   bash mc_score.sh <slice k> <n slices>
# Slice k of: (a) final + preview + final_phrase rows -> qwen25, qwen3vl, internvl3 (16 frames, held-out protocol);
#             (b) iterate rows (in-loop render of every iteration) -> internvl3 only (fast; qwen3vl ~3 videos/min).
# Writes $ROOT/heldout_split/<k>/{qwen25,qwen3vl,internvl3}.csv ; merge with mc_merge_heldout.py.
K=$1; NS=$2
ROOT=/scratch/amirrz/iclr_rebuttal/multicritic
source /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/job_env.sh
D=$ROOT/heldout_split/$K; mkdir -p $D
python - "$ROOT/manifest.csv" "$D" "$K" "$NS" <<'PY'
import csv, sys
man, d, k, ns = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rows = list(csv.DictReader(open(man)))
main = [r for r in rows if r["group"] != "iterate"]; it = [r for r in rows if r["group"] == "iterate"]
for name, rr in (("main", main), ("iter", it)):
    sl = rr[k::ns]
    with open(f"{d}/{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(sl)
    print(name, len(sl))
PY
python heldout_score.py --manifest $D/main.csv --out $D --models internvl3,qwen25,qwen3vl > $D/log_main.txt 2>&1
python heldout_score.py --manifest $D/iter.csv --out $D/iter --models internvl3 > $D/log_iter.txt 2>&1
echo "MC_SCORED slice $K rc=$?" >> $ROOT/events.txt
