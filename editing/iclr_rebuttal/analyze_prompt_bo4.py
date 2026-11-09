#!/usr/bin/env python3
"""Prompt-only best-of-4 held-out and flow numbers, assembled from existing best-of-3, best-of-8 and seed-2 rows. CPU only."""
import pandas as pd
R = "/scratch/amirrz/iclr_rebuttal/"; T = "/home/amirrz/my_codes/Sound_Sparks_Motion/results_trillium/"
cs = pd.read_csv(R + "best_of_n_ours/critic_summary.csv").set_index("scenario")
S2 = T + "best_of_n_ours/metrics_prompt_seed2/"
def src(s, n):
    p = cs.loc[s, f"prompt_bo{n}_pick"]
    return p
out = {}
for m in ("qwen25", "qwen3vl", "internvl3"):
    b3 = pd.read_csv(R + f"heldout_bon/{m}.csv").set_index("scenario").yes_prob
    sx = pd.read_csv(T + f"best_of_n/heldout/{m}.csv").set_index(["scenario", "method"]).yes_prob  # seeds 2-6
    def val(s, p): return b3[s] if int(p[1:]) in (42, 0, 1) else sx[(s, "seed_%02d" % int(p[1:]))]
    out[m] = pd.Series({s: val(s, cs.loc[s, "prompt_bo4_pick"]) for s in cs.index})
    out[m + "_bo8"] = pd.Series({s: val(s, cs.loc[s, "prompt_bo8_pick"]) for s in cs.index})
f3 = pd.read_csv(R + "p7/flow_warp.csv"); f3 = f3[f3.method == "bon3"].set_index("scenario")
f8 = pd.read_csv(R + "p7_bon8/flow_warp.csv").set_index("scenario")
fs = pd.read_csv(S2 + "flow_warp.csv"); fs = fs[fs.method == "prompt_s02"].set_index("scenario")
for c in ("flow_mag", "warp_err"):
    v = {}
    for s in cs.index:
        p4, p3, p8 = (cs.loc[s, f"prompt_bo{n}_pick"] for n in (4, 3, 8))
        v[s] = f3.loc[s, c] if p4 == p3 else (f8.loc[s, c] if p4 == p8 else fs.loc[s, c])
    out[c] = pd.Series(v)
d = pd.DataFrame(out); d.to_csv(R + "best_of_n_ours/prompt_bo4_metrics.csv")
print(d.round(3).to_string()); print(d.mean().round(4).to_string())
print({m: int((d[m] >= .5).sum()) for m in d.columns if "qwen" in m or "intern" in m})
print(cs[[c for c in cs if c.startswith("prompt_bo4")]].to_string())
