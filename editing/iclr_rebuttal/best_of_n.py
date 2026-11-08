#!/usr/bin/env python3
"""P1 - best-of-N prompt-only Retake at matched compute (forward only, no tuning).

For one scenario: load the LTX-2 Retake pipeline + Qwen2.5-VL critic once, then
for every seed render the *prompt-only* Retake output (no tuned latents; same
source, prompt, K prefix, resolution, steps and full-quality final guidance as
the paper's baseline_video.mp4) and score it with the exact critic call of the
paper run (see critic.py). Seed 42 should reproduce the paper's LTX-2 Retake
baseline; we log its pixel difference to the paper file as a sanity check.

Also scores the paper's ours.mp4 and baseline_ltx2.mp4 with the same critic.

--ours (added 2026-11-08, default off = behaviour above unchanged): best-of-N ON TOP OF OURS. The paper run's learned
latents (mode_both/best_audio_latent_both.pt + best_text_delta_both.pt) replace the source audio latent and are added
to the prompt's video text encoding (exactly as render_final_video does); everything else identical. Seed 42 is
compared with the paper's ours.mp4 (sanity). Default out root best_of_n_ours/.

Outputs (results/iclr_rebuttal/best_of_n/<scenario>/):
  candidates/seed_XX.mp4, scores.json (incremental, resumable), setup.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from common import OUT_ROOT, critic_spec, load_paper_args, paper_baseline_video, paper_ours_video

log = logging.getLogger("best_of_n")

DEFAULT_SEEDS = [42] + list(range(23))  # 24 seeds; order defines best-of-n prefixes


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--ours", action="store_true", help="render the paper run's learned latents (best-of-N on ours)")
    a = ap.parse_args()
    if a.out_root is None:
        a.out_root = str(OUT_ROOT / ("best_of_n_ours" if a.ours else "best_of_n"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    from motion_opt.core import build_cached_source_latents, build_guiders_for_mode, compute_target_shape, _parse_loras
    from motion_opt.models import build_retake_pipeline, resolve_quantization_policy
    from motion_opt.multimodal_loop import pre_encode_base_contexts, render_baseline_video
    from motion_opt.runtime import build_retake_kwargs, prepare_retake_input_video
    from ltx_pipelines.utils.constants import detect_params
    from critic import LoopCritic, load_video_chw

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    ns = load_paper_args(a.scenario)
    out = Path(a.out_root) / a.scenario
    (out / "candidates").mkdir(parents=True, exist_ok=True)
    scores_path = out / "scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {"seeds": {}, "reference": {}}

    torch.manual_seed(ns.seed)
    torch.cuda.manual_seed_all(ns.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    t_setup = time.time()

    height, width, num_frames, frame_rate = compute_target_shape(ns.src_video, ns.height, ns.width, ns.num_frames, ns.frame_rate)
    duration = num_frames / frame_rate
    quant = resolve_quantization_policy(ns.retake_quantization if ns.retake_quantization is not None else ns.quantization)
    retake_input = prepare_retake_input_video(args=ns, is_main=True, output_dir=out, height=height, width=width,
                                              num_frames=num_frames, frame_rate=frame_rate)
    params = detect_params(ns.checkpoint_path)
    vg, ag = build_guiders_for_mode(args=ns, params=params, use_low_memory_guidance=ns.low_memory_guidance)
    pipeline = build_retake_pipeline(checkpoint_path=ns.checkpoint_path, gemma_root=ns.gemma_root,
                                     loras=_parse_loras(ns.loras), device=device, quant_policy=quant,
                                     gradient_checkpointing=ns.gradient_checkpointing)
    cached_video_latent, base_audio_latent, wave_sr = build_cached_source_latents(
        pipeline=pipeline, src_video=str(retake_input), height=height, width=width,
        num_frames=num_frames, audio_sr=ns.audio_sr, device=device)
    pos_ctx, neg_ctx = pre_encode_base_contexts(pipeline=pipeline, pos_prompt=ns.edit_prompt,
                                                neg_prompt=ns.negative_prompt, device=device)
    if a.ours:  # learned latents of the paper run (same objects render_final_video uses)
        from common import paper_run_dir
        md = paper_run_dir(a.scenario) / "mode_both"
        lat_a = torch.load(md / "best_audio_latent_both.pt", map_location="cpu")
        lat_t = torch.load(md / "best_text_delta_both.pt", map_location="cpu")
        assert tuple(lat_a.shape) == tuple(base_audio_latent.shape), (lat_a.shape, base_audio_latent.shape)
        assert tuple(lat_t.shape) == tuple(pos_ctx.video_encoding.shape), (lat_t.shape, pos_ctx.video_encoding.shape)
        base_audio_latent = lat_a.to(device=base_audio_latent.device, dtype=base_audio_latent.dtype)
        pos_ctx = type(pos_ctx)(video_encoding=pos_ctx.video_encoding + lat_t.to(device=pos_ctx.video_encoding.device,
                                                                                  dtype=pos_ctx.video_encoding.dtype),
                                audio_encoding=pos_ctx.audio_encoding, attention_mask=pos_ctx.attention_mask)
        log.info("[ours] loaded learned latents from %s", md)
    critic = LoopCritic(ns, device)

    retake_kwargs = build_retake_kwargs(args=ns, frame_rate=frame_rate, duration=duration,
                                        video_guider_params=vg, audio_guider_params=ag)
    final_kwargs = dict(retake_kwargs)  # == optimize_qwen_vl.run(): final render settings
    if ns.final_retake_num_inference_steps is not None:
        final_kwargs["num_inference_steps"] = ns.final_retake_num_inference_steps
    fvg, fag = build_guiders_for_mode(args=ns, params=params, use_low_memory_guidance=False)
    final_kwargs["video_guider_params"], final_kwargs["audio_guider_params"] = fvg, fag
    _sync()
    setup_s = time.time() - t_setup

    (out / "setup.json").write_text(json.dumps({
        "scenario": a.scenario, "seeds": seeds, "setup_seconds": setup_s,
        "critic": critic_spec(ns), "shape": [height, width, num_frames, frame_rate],
        "retake_start_frames": ns.retake_start_frames, "num_inference_steps": final_kwargs["num_inference_steps"],
        "src_video": ns.src_video, "edit_prompt": ns.edit_prompt, "gpu": torch.cuda.get_device_name(0),
        "note": ("OURS learned latents (paper run mode_both), " if a.ours else "prompt-only Retake, ") +
                "full-quality final guidance (same as paper baseline_video.mp4 / final render); "
                "critic = exact loop call of the paper run (critic.py)",
        "ours": bool(a.ours),
    }, indent=2, default=str))

    def save():
        scores_path.write_text(json.dumps(scores, indent=2))

    # reference videos scored with the identical critic
    for name, path in (("paper_ours", paper_ours_video(a.scenario)), ("paper_baseline", paper_baseline_video(a.scenario))):
        if name not in scores["reference"]:
            _sync(); t = time.time()
            r = critic.score_video(path)
            _sync(); r["score_s"] = time.time() - t
            r["path"] = str(path)
            scores["reference"][name] = r
            save()
            log.info("[ref] %s yes=%.4f", name, r["yes_prob"])

    for s in seeds:
        key = str(s)
        mp4 = out / "candidates" / f"seed_{s:02d}.mp4"
        if key in scores["seeds"] and mp4.exists():
            continue
        kw = dict(final_kwargs)
        kw["seed"] = s
        torch.manual_seed(s)
        torch.cuda.reset_peak_memory_stats()
        _sync(); t0 = time.time()
        render_baseline_video(pipeline=pipeline, src_video=str(retake_input), cached_video_latent=cached_video_latent,
                              base_audio_latent=base_audio_latent, base_pos_context=pos_ctx, base_neg_context=neg_ctx,
                              retake_kwargs=kw, output_path=mp4, num_frames=num_frames, frame_rate=frame_rate,
                              audio_sr=wave_sr)
        _sync(); t1 = time.time()
        r = critic.score_video(mp4)
        _sync(); t2 = time.time()
        r.update(render_s=t1 - t0, score_s=t2 - t1, peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9)
        if s == ns.seed:  # sanity: seed 42 vs the paper baseline video (or the paper ours video with --ours)
            ref = paper_ours_video(a.scenario) if a.ours else paper_baseline_video(a.scenario)
            a_ = load_video_chw(mp4, device)
            b_ = load_video_chw(ref, device)
            n = min(len(a_), len(b_))
            r["mean_abs_diff_vs_paper_ours" if a.ours else "mean_abs_diff_vs_paper_baseline"] = \
                float((a_[:n] - b_[:n]).abs().mean().item())
            del a_, b_
        scores["seeds"][key] = r
        save()
        log.info("[seed %d] yes=%.4f render=%.1fs score=%.1fs", s, r["yes_prob"], r["render_s"], r["score_s"])

    log.info("done: %s", out)


if __name__ == "__main__":
    main()
