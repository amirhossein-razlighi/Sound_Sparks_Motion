#!/usr/bin/env python3
"""Mode ablation (text-only / audio-only vs the packaged 'both' runs).

For every scenario in picks.json (except reserves) rebuild the exact environment of the winning run from its
run_config.json, reuse its Phase-A capture, and submit OPT_MODE=text and OPT_MODE=audio runs to
/scratch/amirrz/H3_exp/outputs/ablation/<slug>/{text_only,audio_only}.  Jobs are chained in a few lines so at most
LINES ablation jobs run at once.  Writes h3_probe/results/ablation/manifest.json.
"""
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SCN = os.path.join(REPO, "h3_probe/scenarios")
ABL = "/scratch/amirrz/H3_exp/outputs/ablation"
ENV = {  # run_config key -> env var read by h3_full_method.py
    "steps": "STEPS", "final_steps": "FINAL_STEPS", "grad_steps": "GRAD_STEPS", "iterations": "ITERS", "early_stopping": "EARLY",
    "optimizer": "OPTIM", "ngd_eta": "NGD_ETA", "ngd_momentum": "NGD_MOM", "lr": "LR", "text_lr_mult": "TEXT_LR_MULT",
    "lr_schedule": "LR_SCHEDULE", "grad_clip": "GRAD_CLIP", "latent_reg_weight": "AUDIO_REG", "text_reg_weight": "TEXT_REG",
    "reg_schedule": "REG_SCHEDULE", "lpips_weight": "LPIPS_W", "temporal_weight": "TEMPORAL_W", "qwen_max_frames": "QWEN_FRAMES",
    "qwen_img_size": "QWEN_IMG", "qwen_grad_accum_steps": "QWEN_ACCUM", "critic_objective": "CRITIC_OBJ",
    "qwen_gradient_rubric": "QWEN_RUBRIC", "text_eta_mult": "TEXT_ETA_MULT", "save_iter_previews": "SAVE_PREVIEWS",
    "decode_grad_frac": "DECODE_GRAD_FRAC", "vae_dtype": "VAE_DTYPE", "critic_fp32_head": "FP32_HEAD", "qwen_crop": "QWEN_CROP",
    "select_by": "SELECT_BY", "attn_steps": "ATTN_STEPS", "perc_max": "PERC_MAX",
}
SCEN_JSON = ":".join(os.path.join(SCN, f) for f in ("candidates_r1.json", "candidates_r2.json", "candidates_r3.json",
                                                      "candidates_r4.json", "candidates_r5.json", "candidates_fullsize_test.json"))
EXCL = "rg31701,rg13401,rg21803,rg21802,rg31502,rg32202"
LINES = int(os.environ.get("LINES", "3"))
ORDER = ["goldfish", "cat_yawns", "man_shouts", "boy_splashes", "car_door_opens", "gen_dolphin_sea", "gen_koi_pond", "gen_sealion",
         "gen_woman_desk", "gen_cow_field", "gen_woman_door", "gen_frog_jumps"]


def main():
    picks = {k: v for k, v in json.load(open(os.path.join(SCN, "picks.json"))).items() if not k.startswith("_")}
    slugs = [s for s in ORDER if s in picks] + [s for s in picks if s not in ORDER and "RESERVE" not in picks[s].get("note", "")]
    only = [x for x in os.environ.get("ONLY", "").replace(";", ",").split(",") if x]   # ONLY=slug1,slug2 -> submit just these
    if only:
        slugs = [s for s in slugs if s in only]
    mp = os.path.join(REPO, "h3_probe/results/ablation/manifest.json")
    manifest = json.load(open(mp)) if os.path.exists(mp) else {}
    dep = {}   # line index -> last job id
    n = 0
    for s in slugs:
        run = picks[s]["run"]
        cfg = json.load(open(os.path.join(run, "run_config.json")))
        env = {k: str(cfg[key]) for key, k in ENV.items() if key in cfg and cfg[key] not in ("", None)}
        env["ATTN_VIS"] = "0"; env["GRAD_CHECK"] = "0"; env["SCEN_JSON"] = SCEN_JSON; env["SLUG"] = cfg["slug"]
        if cfg.get("gpu_mem_split"):
            env["GPU_MEM_SPLIT"] = cfg["gpu_mem_split"].replace(",", ";")
        three = bool(cfg.get("gpu_mem_split")) and "full" in run
        for mode in ("text", "audio"):
            out = os.path.join(ABL, s, f"{mode}_only")
            os.makedirs(out, exist_ok=True)
            cap = os.path.join(out, "capture.pt")
            if not os.path.exists(cap):
                os.symlink(os.path.join(run, "capture.pt"), cap)
            e = dict(env, OPT_MODE=mode, OUT_DIR=out)
            for key, var in (("motion_question", "MOTION_Q_FILE"), ("prompt_edit", "PROMPT_EDIT_FILE")):
                if cfg.get(key):
                    fp = os.path.join(out, var.lower() + ".txt"); open(fp, "w").write(cfg[key]); e[var] = fp
            assert not any("," in v for v in e.values()), e
            line = 99 if three else n % LINES
            cmd = ["sbatch", "--parsable", f"--exclude={EXCL}"]
            if line in dep:
                cmd.append(f"--dependency=afterany:{dep[line]}")
            cmd += ["--export=ALL," + ",".join(f"{k}={v}" for k, v in e.items()),
                    os.path.join(REPO, "h3_probe", "full_method_3gpu.sbatch" if three else "full_method.sbatch")]
            if os.environ.get("DRY_RUN"):
                print(" ".join(cmd)[:300]); jid = f"dry{n}"
            else:
                jid = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
            dep[line] = jid; n += 1
            manifest.setdefault(s, {"both_run": run, "both_iter": picks[s]["iter"], "edit": picks[s].get("edit")})[f"{mode}_only"] = {"out": out, "job": jid}
            print(f"{s:18s} {mode}-only -> {jid} ({'3 GPUs' if three else '2 GPUs'}, line {line})")
    json.dump(manifest, open(mp, "w"), indent=1)
    print("manifest:", mp)


if __name__ == "__main__":
    main()
