#!/bin/bash
# Score every IVEBench output that exists (ours per config, eye-picked iterations runs/<n>/<cfg>/iter_NNN.mp4, Retake baselines,
# source) with the held-out critics (16 uniform frames, task question with IVEBench target_prompt). Lane CMD task.
R=${IVE_ROOT:-/scratch/amirrz/iclr_rebuttal/ivebench}
bash ivebench/import_trillium.sh
python ivebench/build_manifest.py --root $R --out $R/manifest_ivebench.csv
python heldout_score.py --manifest $R/manifest_ivebench.csv --out $R/heldout --models ${MODELS:-qwen25,qwen3vl,internvl3}
