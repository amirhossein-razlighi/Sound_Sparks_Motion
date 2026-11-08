#!/usr/bin/env python3
"""Merge heldout_split/<k>/{model}.csv and heldout_split/<k>/iter/{model}.csv into $ROOT/heldout/{model}.csv
(existing merged files are renamed .bak_<time>, never overwritten)."""
import time
from pathlib import Path

import pandas as pd

ROOT = Path("/scratch/amirrz/iclr_rebuttal/multicritic")
out = ROOT / "heldout"; out.mkdir(exist_ok=True)
for m in ("qwen25", "qwen3vl", "internvl3"):
    fs = sorted(ROOT.glob(f"heldout_split/*/{m}.csv")) + sorted(ROOT.glob(f"heldout_split/*/iter/{m}.csv"))
    if not fs:
        continue
    d = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates("id", keep="last")
    dst = out / f"{m}.csv"
    if dst.exists():
        dst.rename(f"{dst}.bak_{time.strftime('%H%M%S')}")
    d.to_csv(dst, index=False)
    print(m, len(d), "rows,", int(d["error"].notna().sum() if "error" in d else 0), "with error text")
