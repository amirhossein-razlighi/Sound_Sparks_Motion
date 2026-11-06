#!/usr/bin/env python3
"""Diagnostic: which prompt-only render reproduces the paper's baseline_video.mp4 (seed 42)?

Variants (all seed 42, paper run args, full-quality final guidance):
  injected_raw   : current render_baseline_video (raw Gemma context injected, no prompt enhancement)
  retake_enhance : Retake encodes the prompt itself with enhance_prompt=True (paper run_config value)
  retake_plain   : Retake encodes the prompt itself with enhance_prompt=False
Reports mean |diff| (0-1 scale) of each vs the paper baseline video. Output: diag/<scenario>/*.mp4, diag.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import OUT_ROOT, load_paper_args, paper_baseline_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="dog_yawning")
    a = ap.parse_args()
    from motion_opt.core import build_cached_source_latents, build_guiders_for_mode, compute_target_shape, _parse_loras
    from motion_opt.models import build_retake_pipeline, resolve_quantization_policy
    from motion_opt.multimodal_loop import pre_encode_base_contexts, render_baseline_video
    from motion_opt.runtime import build_retake_kwargs, prepare_retake_input_video
    from ltx_pipelines.utils.constants import detect_params
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.utils.media_io import encode_video
    import ltx_pipelines.retake as R
    from critic import load_video_chw

    ns = load_paper_args(a.scenario)
    out = OUT_ROOT / "diag" / a.scenario
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(ns.seed)
    dev = torch.device("cuda")
    h, w, nf, fr = compute_target_shape(ns.src_video, ns.height, ns.width, ns.num_frames, ns.frame_rate)
    src = prepare_retake_input_video(args=ns, is_main=True, output_dir=out, height=h, width=w, num_frames=nf, frame_rate=fr)
    params = detect_params(ns.checkpoint_path)
    pipe = build_retake_pipeline(checkpoint_path=ns.checkpoint_path, gemma_root=ns.gemma_root, loras=_parse_loras(ns.loras),
                                 device=dev, quant_policy=resolve_quantization_policy(ns.quantization),
                                 gradient_checkpointing=ns.gradient_checkpointing)
    cvl, bal, sr = build_cached_source_latents(pipeline=pipe, src_video=str(src), height=h, width=w, num_frames=nf,
                                               audio_sr=ns.audio_sr, device=dev)
    pos, neg = pre_encode_base_contexts(pipeline=pipe, pos_prompt=ns.edit_prompt, neg_prompt=ns.negative_prompt, device=dev)
    vg, ag = build_guiders_for_mode(args=ns, params=params, use_low_memory_guidance=ns.low_memory_guidance)
    kw = build_retake_kwargs(args=ns, frame_rate=fr, duration=nf / fr, video_guider_params=vg, audio_guider_params=ag)
    if ns.final_retake_num_inference_steps is not None:
        kw["num_inference_steps"] = ns.final_retake_num_inference_steps
    kw["video_guider_params"], kw["audio_guider_params"] = build_guiders_for_mode(args=ns, params=params, use_low_memory_guidance=False)
    ref = load_video_chw(paper_baseline_video(a.scenario), dev)
    res = {}

    def mad(p):
        x = load_video_chw(p, dev); n = min(len(x), len(ref))
        return float((x[:n] - ref[:n]).abs().mean())

    p = out / "injected_raw.mp4"
    torch.manual_seed(ns.seed)
    render_baseline_video(pipeline=pipe, src_video=str(src), cached_video_latent=cvl, base_audio_latent=bal, base_pos_context=pos,
                          base_neg_context=neg, retake_kwargs=kw, output_path=p, num_frames=nf, frame_rate=fr, audio_sr=sr)
    res["injected_raw"] = mad(p)
    print(res, flush=True)
    # same, but with the TRAINING-time low-memory guidance (cfg 1, no STG) used inside the optimisation loop
    klow = dict(kw)
    klow["video_guider_params"], klow["audio_guider_params"] = vg, ag
    print("full guidance:", kw["video_guider_params"], "| low-mem:", vg, flush=True)
    p = out / "injected_raw_lowmem.mp4"
    torch.manual_seed(ns.seed)
    render_baseline_video(pipeline=pipe, src_video=str(src), cached_video_latent=cvl, base_audio_latent=bal, base_pos_context=pos,
                          base_neg_context=neg, retake_kwargs=klow, output_path=p, num_frames=nf, frame_rate=fr, audio_sr=sr)
    res["injected_raw_lowmem"] = mad(p)
    x, y = load_video_chw(out / "injected_raw.mp4", dev), load_video_chw(p, dev)
    res["full_vs_lowmem"] = float((x - y).abs().mean())
    print(res, flush=True)
    # Retake encodes the prompt itself (only video/audio latents injected, exactly like the current audio-only path)
    oe, oa = R._encode_video_for_retake, R._encode_audio_for_retake
    R._encode_video_for_retake = lambda *x, **y: cvl
    R._encode_audio_for_retake = lambda *x, **y: bal
    try:
        for name, enh in ():
            k2 = dict(kw); k2["enhance_prompt"] = enh
            torch.manual_seed(ns.seed)
            with torch.no_grad():
                vid, _ = pipe(video_path=str(src), **k2)
            p = out / f"{name}.mp4"
            encode_video(video=vid, fps=int(round(fr)), audio=None, output_path=str(p),
                         video_chunks_number=get_video_chunks_number(nf, TilingConfig.default()))
            res[name] = mad(p)
            print(res, flush=True)
    finally:
        R._encode_video_for_retake, R._encode_audio_for_retake = oe, oa
    (out / "diag.json").write_text(json.dumps(res, indent=2))
    print("DONE", res)


if __name__ == "__main__":
    main()
