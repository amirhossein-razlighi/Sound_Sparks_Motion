#!/bin/bash
# Wait for all non-reused transfer outputs (<= 40 min), then build the transfer manifest, score, analyse.
R=/scratch/amirrz/iclr_rebuttal
for i in $(seq 1 80); do
  miss=$(python3 -c "
import csv,os
print(sum(1 for t in csv.DictReader(open('$R/transfer/targets.csv')) if not os.path.exists(os.path.join(t['out_dir'],'transfer_both.mp4'))))")
  echo "missing transfers: $miss"; [ "$miss" = "0" ] && break; sleep 30
done
python build_manifests.py transfer
python heldout_score.py --manifest $R/manifests/transfer.csv --out $R/heldout --models qwen25,internvl3
python p5_analyze.py
python heldout_score.py --manifest $R/manifests/transfer.csv --out $R/heldout --models qwen3vl
python p5_analyze.py
