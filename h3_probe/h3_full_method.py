#!/usr/bin/env python3
"""OUR FULL METHOD (Sound Sparks Motion) ON MINIMAX-H3, WITH TRUE GRADIENTS.

Exactly the LTX recipe, ported to H3's single-stream omni DiT:
  * optimized variables  : the AUDIO conditioning latent (H3's audio-reference
                           rows, normalized latent space, optimized directly) and
                           a RESIDUAL on the TEXT conditioning (delta on the
                           Qwen3-VL prompt embeddings)          -> opt_mode both|audio|text
  * critic               : the unchanged Qwen2.5-VL motion rubric loss
                           (motion_opt.qwen_loss), gradient-accumulated over
                           frame windows exactly as in multimodal_loop_qwen.py
  * regularizers         : L2 anchor of the audio latent to the source latent,
                           L2 on the text residual, LPIPS source preservation and
                           temporal-consistency (motion_opt.perceptual_loss),
                           reg schedule (cosine_increase) and cosine LR
  * gradient path        : critic -> differentiable video-VAE decode -> the last
                           K denoising steps of the 33B transformer -> the two
                           conditioning latents (all K steps see the current
                           latents; only the last K are differentiable, like
                           LTX's audio_opt_last_steps)
  * memory               : transformer sharded over the visible GPUs (accelerate
                           device_map), per-block gradient checkpointing in the
                           DiT and in the VAE's ViT decoder, bf16 weights, fp32
                           latents, frozen weights (no optimizer state for them)

Phase A runs the stock diffusers pipeline once (baseline video, and it captures
the packed-sequence state right before the denoise loop: noise, layout, prompt
embeddings, reference rows, timestep plan). Phase B re-implements the loop
out-of-place with gradients and verifies it against Phase A (pixel diff), then
verifies the gradient (finite-difference along -grad / +grad), then optimizes.

    sbatch h3_probe/full_method.sbatch   (env: SLUG=boy_crouches ...)
"""
from __future__ import annotations

import argparse
import csv
import functools
import gc
import importlib.util
import json
import logging
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(_HERE + "/..")
sys.path.insert(0, os.path.join(_REPO, "editing/src"))
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("h3full")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_bs = _load("baseline_sweep")
CKPT, QWEN = _bs.CKPT, _bs.QWEN
SCEN = {k: v for k, v in json.load(open(os.path.join(_HERE, "pin_all_scenarios.json"))).items() if not k.startswith("_")}
for _extra in [p for p in os.environ.get("SCEN_JSON", "").split(":") if p]:   # extra scenario files (user-study candidates)
    SCEN.update({k: v for k, v in json.load(open(_extra)).items() if not k.startswith("_")})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="all", choices=["all", "a", "b", "render"],
                   help="a: stock pipeline baseline + state capture -> capture.pt; b: gradient method from capture.pt")
    p.add_argument("--slug", default=os.environ.get("SLUG", "boy_crouches"))
    p.add_argument("--out-dir", default=os.environ.get("OUT_DIR", ""))
    p.add_argument("--opt-mode", default=os.environ.get("OPT_MODE", "both"), choices=["both", "audio", "text"])
    p.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", "16")))
    p.add_argument("--final-steps", type=int, default=int(os.environ.get("FINAL_STEPS", "16")))
    p.add_argument("--grad-steps", type=int, default=int(os.environ.get("GRAD_STEPS", "2")),
                   help="last K denoising steps are differentiable (LTX: audio_opt_last_steps)")
    p.add_argument("--iterations", type=int, default=int(os.environ.get("ITERS", "10")))
    p.add_argument("--early-stopping", type=int, default=int(os.environ.get("EARLY", "6")))
    p.add_argument("--optimizer", default=os.environ.get("OPTIM", "ngd"), choices=["ngd", "adam"],
                   help="ngd: normalized gradient descent, step = eta * ||theta_ref|| * g/||g|| per parameter (eta cosine-decayed); "
                        "adam: torch Adam with --lr")
    p.add_argument("--ngd-eta", type=float, default=float(os.environ.get("NGD_ETA", "0.03")),
                   help="initial relative step (fraction of the reference norm) for ngd")
    p.add_argument("--ngd-momentum", type=float, default=float(os.environ.get("NGD_MOM", "0.3")))
    p.add_argument("--lr", type=float, default=float(os.environ.get("LR", "0.02")))
    p.add_argument("--text-lr-mult", type=float, default=float(os.environ.get("TEXT_LR_MULT", "1.0")))
    p.add_argument("--lr-schedule", default=os.environ.get("LR_SCHEDULE", "cosine"), choices=["cosine", "constant"])
    p.add_argument("--grad-clip", type=float, default=float(os.environ.get("GRAD_CLIP", "0.0")))
    p.add_argument("--latent-reg-weight", type=float, default=float(os.environ.get("AUDIO_REG", "0.01")))
    p.add_argument("--text-reg-weight", type=float, default=float(os.environ.get("TEXT_REG", "0.001")))
    p.add_argument("--reg-schedule", default=os.environ.get("REG_SCHEDULE", "cosine_increase"))
    p.add_argument("--lpips-weight", type=float, default=float(os.environ.get("LPIPS_W", "0.1")))
    p.add_argument("--temporal-weight", type=float, default=float(os.environ.get("TEMPORAL_W", "0.05")))
    p.add_argument("--lpips-backbone", default="alex")
    p.add_argument("--qwen-max-frames", type=int, default=int(os.environ.get("QWEN_FRAMES", "24")))
    p.add_argument("--qwen-img-size", type=int, default=int(os.environ.get("QWEN_IMG", "224")))
    p.add_argument("--qwen-grad-accum-steps", type=int, default=int(os.environ.get("QWEN_ACCUM", "1")),
                   help="1 = single deterministic linspace window (loss exact & comparable to the baseline); >1 adds random windows")
    p.add_argument("--qwen-sample-mode", default="linspace")
    p.add_argument("--critic-objective", default=os.environ.get("CRITIC_OBJ", "lin"), choices=["lin", "any"],
                   help="lin: single linspace window (LTX default); any: noisy-OR over contiguous 24-frame windows "
                        "(starts 0/20/40/60/80/100) = P(event happens in some window) - for brief events")
    p.add_argument("--qwen-gradient-rubric", default=os.environ.get("QWEN_RUBRIC", "motion"), choices=["motion", "full"],
                   help="full = motion 0.7 + entities 0.2 + overall 0.1 (the 'overall' question penalizes unrelated artifacts)")
    p.add_argument("--text-eta-mult", type=float, default=float(os.environ.get("TEXT_ETA_MULT", "1.0")),
                   help="ngd relative step for the text residual = ngd_eta * this (limits semantic drift)")
    p.add_argument("--save-iter-previews", type=int, default=int(os.environ.get("SAVE_PREVIEWS", "1")),
                   help="save the frames of every gradient render as iter_XX.mp4 (cheap: they already exist)")
    p.add_argument("--decode-grad-frac", type=float, default=float(os.environ.get("DECODE_GRAD_FRAC", "0.5")),
                   help="fraction of the VAE's temporal decode chunks that carry gradient per iteration (rotating); 1.0 = all")
    p.add_argument("--vae-dtype", default=os.environ.get("VAE_DTYPE", "float16"), choices=["float16", "float32"])
    p.add_argument("--critic-fp32-head", type=int, default=int(os.environ.get("FP32_HEAD", "1")),
                   help="compute the critic's yes/no logits in fp32 (bf16 logits quantize the nll to ~0.1-nat steps)")
    p.add_argument("--qwen-crop", default=os.environ.get("QWEN_CROP", ""),
                   help='crop fed to the critic as fractions "x0;y0;x1;y1" (or comma-separated); empty = full frame')
    p.add_argument("--select-by", default=os.environ.get("SELECT_BY", "lin"), choices=["lin", "any", "4win", "train"],
                   help="best-checkpoint criterion (+perceptual): lin = training linspace window nll; any = -log P(event in any "
                        "contiguous window) (brief events); 4win = mean over 4 windows (dilutes brief events); train = training total")
    def _env_or_file(var):
        """sbatch --export splits on commas, so long strings are passed as files: VAR_FILE=path takes precedence."""
        f = os.environ.get(var + "_FILE", "")
        return open(f).read().strip() if f else os.environ.get(var, "")

    p.add_argument("--motion-question", default=_env_or_file("MOTION_Q"),
                   help="override the critic's motion question template (may contain {edit_prompt}); empty = library default")
    p.add_argument("--prompt-edit", default=_env_or_file("PROMPT_EDIT"),
                   help="override the edit sentence used in the H3 grammar prompt (critic question stays the scenario's)")
    p.add_argument("--attn-vis", type=int, default=int(os.environ.get("ATTN_VIS", "1")),
                   help="per-iteration attention-mass maps (audio-ref / text / vision / reference-video -> generated video)")
    p.add_argument("--attn-steps", default=os.environ.get("ATTN_STEPS", "14"),
                   help="denoising-step indices (0-based model evals) at which the attention mass is captured")
    p.add_argument("--perc-max", type=float, default=float(os.environ.get("PERC_MAX", "inf")),
                   help="checkpoints whose perceptual term (LPIPS+temporal) exceeds this are never selected as best "
                        "(guards against late adversarial drift, e.g. caption-like textures); default: no limit")
    p.add_argument("--render-steps", default=os.environ.get("RENDER_STEPS", "16"),
                   help="phase=render: step counts at which baseline and --init-latents are re-rendered and scored")
    p.add_argument("--init-latents", default=os.environ.get("INIT_LATENTS", ""),
                   help="latents.pt from a previous run: start z_audio/delta_text from its best_z/best_d (refinement run); "
                        "the L2 anchors and LPIPS source stay the ORIGINAL source/baseline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad-check", type=int, default=int(os.environ.get("GRAD_CHECK", "1")))
    p.add_argument("--gpu-mem-split", default=os.environ.get("GPU_MEM_SPLIT", ""),
                   help='per-GPU GiB budget for the transformer, e.g. "40,26" (default: balanced with headroom)')
    return p.parse_args()


# ----------------------------------------------------------------------------- helpers
def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().to("cpu").clone()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


def to_dev(obj, dev):
    if torch.is_tensor(obj):
        return obj.to(dev)
    if isinstance(obj, dict):
        return {k: to_dev(v, dev) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_dev(v, dev) for v in obj)
    return obj


def _init_all_cuda():
    """The allocator stats API raises 'Invalid device argument' on a device that has not been touched yet."""
    for i in range(torch.cuda.device_count()):
        torch.empty(0, device=f"cuda:{i}")


def _init_devices():
    for i in range(torch.cuda.device_count()):
        torch.zeros(1, device=f"cuda:{i}")  # create the CUDA context so memory-stat calls are valid


def gpu_mem_str():
    _init_devices()
    return " ".join(f"g{i}:{torch.cuda.max_memory_allocated(i) / 2**30:.1f}G" for i in range(torch.cuda.device_count()))


def reset_peaks():
    _init_devices()
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def save_av(frames_np, wav, sr, path):
    """frames_np [F,H,W,3] uint8/float in [0,1]; wav [samples, ch] or None."""
    from diffusers.utils import export_to_video
    import soundfile as sf
    export_to_video([np.asarray(f) for f in frames_np], path, fps=24)
    if wav is None:
        return
    stem = os.path.splitext(path)[0]
    sf.write(stem + ".wav", wav, sr)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-threads", "2", "-i", path, "-i", stem + ".wav",
                    "-c:v", "copy", "-c:a", "aac", "-shortest", stem + "_av.mp4"], check=False)


class FP32Head(torch.nn.Module):
    """Critic lm_head replacement: only the last position is used by the loss, computed in fp32."""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, x):
        w, b = self.head.weight, self.head.bias
        return torch.nn.functional.linear(x[:, -1:].float(), w.float(), None if b is None else b.float())


def parse_crop(spec: str):
    if not spec:
        return None
    v = [float(t) for t in spec.replace(";", ",").split(",")]
    assert len(v) == 4 and 0 <= v[0] < v[2] <= 1 and 0 <= v[1] < v[3] <= 1, spec
    return tuple(v)


def crop_frames(frames, crop):
    """frames [F,3,H,W] -> cropped view (differentiable)."""
    if crop is None:
        return frames
    x0, y0, x1, y1 = crop
    H_, W_ = frames.shape[-2:]
    return frames[:, :, int(y0 * H_):int(y1 * H_), int(x0 * W_):int(x1 * W_)]


# ----------------------------------------------------------------------------- phase A
def phase_a_pipeline(args, sc, out_dir):
    """Stock pipeline: baseline render + capture of the loop-start state."""
    from diffusers import ModularPipeline
    from diffusers.modular_pipelines import ComponentsManager
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3AudioReference, MiniMaxH3VideoReference
    import diffusers.modular_pipelines.minimax_h3.denoise as dn

    cap: dict = {}
    cls = dn.MiniMaxH3Ref2VADenoiseStep          # the ref2va denoise-loop wrapper: receives the full PipelineState
    orig = cls.__call__

    def capture(self, components, state):
        if "state" not in cap:
            vals = state.values if hasattr(state, "values") else vars(state)
            fields = {}
            for k, v in dict(vals).items():
                if k in ("generator", "references", "normalized_references"):
                    continue
                try:
                    fields[k] = to_cpu(v)
                except Exception:
                    pass
            cap["state"] = fields
            cap["vae_latents_mean"] = list(components.vae.config.latents_mean)
            cap["vae_latents_std"] = list(components.vae.config.latents_std)
            cap["audio_latents_mean"] = list(components.audio_vae.config.latents_mean)
            cap["audio_latents_std"] = list(components.audio_vae.config.latents_std)
            cap["pixel_mean"], cap["pixel_std"] = list(components.pixel_mean), list(components.pixel_std)
            cap["patch_size"] = list(components.patch_size)
            cap["audio_channels"] = int(components.audio_channels)
            cap["audio_sr"] = int(components.audio_sampling_rate)
            cap["vae_latent_channels"] = int(components.vae_latent_channels)
            log.info("[A] captured loop-start state: %s", sorted(fields.keys()))
        return orig(self, components, state)

    cls.__call__ = capture
    try:
        cm = ComponentsManager()
        cm.enable_auto_cpu_offload(device="cuda")
        pipe = ModularPipeline.from_pretrained(CKPT, workflow="ref2va", components_manager=cm)
        pipe.load_components(torch_dtype=torch.bfloat16)
        src = _bs.__dict__.get("resolve_src", None)
        src_path = sc["src"] if os.path.isabs(sc["src"]) else os.path.join(_REPO, sc["src"])
        if not os.path.exists(src_path) and src_path.endswith("_main.mp4") and os.path.exists(src_path[:-9] + ".mp4"):
            src_path = src_path[:-9] + ".mp4"
        refs = [MiniMaxH3VideoReference.from_file(src_path), MiniMaxH3AudioReference.from_file(sc["wav"])]
        prompt = _bs.grammar_prompt(args.prompt_edit or sc["edit"], sc["scene"])
        log.info("[A] H3 edit sentence: %r (critic question: %r)", args.prompt_edit or sc["edit"], sc["edit"])
        t0 = time.time()
        state = pipe(prompt=prompt, references=refs, num_frames=_bs.NUM_FRAMES, height=_bs.HEIGHT, width=_bs.WIDTH,
                     num_inference_steps=args.steps, generator=torch.Generator("cuda").manual_seed(args.seed),
                     output_type="np")
        log.info("[A] pipeline baseline rendered in %.0fs", time.time() - t0)
        vids = state.values.get("videos")
        vid = np.asarray(vids[0] if isinstance(vids, (list, tuple)) else vids)
        if vid.ndim == 5:
            vid = vid[0]
        aud = state.values.get("audio")
        wav = None
        if aud is not None:
            aud = aud[0] if isinstance(aud, (list, tuple)) else aud
            wav = np.squeeze(np.asarray(torch.as_tensor(aud).float().cpu()))
            if wav.ndim == 2 and wav.shape[0] in (1, 2):
                wav = wav.T
        save_av(vid, wav, cap.get("audio_sr", 32000), os.path.join(out_dir, "pipeline_baseline.mp4"))
        cap["pipeline_frames"] = torch.from_numpy(np.ascontiguousarray(vid)).float()  # [F,H,W,3] in [0,1]
        cap["prompt"] = prompt
        cap["src_path"] = src_path
        assert "state" in cap, "loop-start state was not captured"
        del pipe, cm, state
    finally:
        cls.__call__ = orig
    gc.collect()
    torch.cuda.empty_cache()
    return cap


# ----------------------------------------------------------------------------- phase B
class H3Grad:
    """Out-of-place re-implementation of the ref2va denoise loop + decode, with gradients."""

    def __init__(self, args, cap):
        from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio, MiniMaxH3Transformer3DModel
        from diffusers import MiniMaxH3Scheduler

        self.args = args
        self.cap = cap
        n_gpu = torch.cuda.device_count()
        assert n_gpu >= 1
        st = cap["state"]
        # ---- devices / sharding plan ----
        if args.gpu_mem_split:
            budget = [float(x) for x in args.gpu_mem_split.replace(";", ",").split(",") if x.strip()]   # ";" survives sbatch --export
        elif n_gpu == 1:
            budget = [70.0]
        elif n_gpu == 2:
            budget = [40.0, 30.0]       # weights only (bf16 ~62 GiB total): ~40 GiB on GPU0, rest on GPU1 (which also hosts VAE+Qwen)
        else:
            budget = [31.0, 31.0] + [0.0] * (n_gpu - 2)  # transformer balanced on GPU0/1; last GPU: VAE + Qwen + LPIPS only
        self.aux_dev = torch.device(f"cuda:{n_gpu - 1}")  # VAE decode + Qwen + LPIPS
        log.info("[B] %d GPUs; transformer weight budgets (GiB)=%s; aux device=%s", n_gpu, budget, self.aux_dev)

        t0 = time.time()
        device_map = self._build_device_map(MiniMaxH3Transformer3DModel, budget)
        self.tr = MiniMaxH3Transformer3DModel.from_pretrained(
            CKPT, subfolder="transformer_ref", torch_dtype=torch.bfloat16, device_map=device_map)
        self.tr.eval().requires_grad_(False)
        ckpt_fn = functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
        self.tr.enable_gradient_checkpointing(gradient_checkpointing_func=ckpt_fn)
        dm = getattr(self.tr, "hf_device_map", None)
        devs = sorted({str(v) for v in dm.values()}) if dm else ["?"]
        log.info("[B] transformer loaded in %.0fs, devices=%s (%d entries)", time.time() - t0, devs, len(dm or {}))
        assert dm and all("cuda" in str(v) or isinstance(v, int) for v in dm.values()), f"transformer not fully on GPUs: {dm}"
        self.dev0 = self.tr.proj_in.weight.device
        n_blocks_per_dev = {}
        for k, v in (dm or {}).items():
            if "transformer_blocks" in k:
                n_blocks_per_dev[str(v)] = n_blocks_per_dev.get(str(v), 0) + 1
        log.info("[B] transformer blocks per device: %s", n_blocks_per_dev)

        vae_dtype = torch.float16 if args.vae_dtype == "float16" else torch.float32
        self.vae = AutoencoderKLMiniMaxH3.from_pretrained(CKPT, subfolder="vae", torch_dtype=vae_dtype).to(self.aux_dev)
        self.vae.eval().requires_grad_(False)
        self.vae.enable_gradient_checkpointing(gradient_checkpointing_func=ckpt_fn)
        # Chunk-level checkpointing of the decode: `_decode` loops over temporal chunks, each decoding a grid of
        # spatial tiles through the ViT decoder. Without this, the graphs of all ~70 tile decodes stay alive at
        # once (OOM on GPU1); with it only one chunk's graph exists at a time and chunks are recomputed in backward.
        _orig_clip = self.vae._decode_clip

        def _ckpt_clip(z):
            if torch.is_grad_enabled() and z.requires_grad:
                return torch.utils.checkpoint.checkpoint(_orig_clip, z, use_reentrant=False)
            return _orig_clip(z)

        self.vae._decode_clip = _ckpt_clip
        self.audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(CKPT, subfolder="audio_vae", torch_dtype=torch.float32).to(self.aux_dev)
        self.audio_vae.eval().requires_grad_(False)
        self.sched = MiniMaxH3Scheduler.from_pretrained(CKPT, subfolder="scheduler")
        self.asched = MiniMaxH3Scheduler.from_pretrained(CKPT, subfolder="audio_scheduler")

        # ---- captured state -> dev0 ----
        self.latents0 = st["latents"].to(self.dev0)               # [N_vid_rows, C*p]
        self.audio0 = st["audio_latents"].to(self.dev0)           # [N_aud_rows, 32]
        self.prompt_embeds = st["prompt_embeds"].to(self.dev0)    # [1, n_text, 5120] bf16
        self.layout = {k: st[k].to(self.dev0) for k in ("token_tags", "position_ids", "video_indices", "audio_indices", "text_indices")}
        self.plan = [tuple(x.to(self.dev0) for x in p) for p in st["row_timestep_plan"]]
        self.timesteps = st["timesteps"].to(self.dev0)
        self.audio_timesteps = st["audio_timesteps"].to(self.dev0)
        self.ncv = int(st["num_condition_video_rows"])
        self.nca = int(st["num_condition_audio_rows"])
        self.nlf, self.lh, self.lw = int(st["num_latent_frames"]), int(st["latent_height"]), int(st["latent_width"])
        self.n_aud = int(st["num_audio_latents"])
        pt_, ph_, pw_ = cap["patch_size"]
        n_gen_rows = self.latents0.shape[0] - self.ncv
        assert n_gen_rows == (self.nlf // pt_) * (self.lh // ph_) * (self.lw // pw_), (n_gen_rows, self.nlf, self.lh, self.lw)
        assert self.audio0.shape[0] - self.nca == self.n_aud * cap["audio_channels"], (self.audio0.shape, self.n_aud)
        self.attention_kwargs = st.get("attention_kwargs", None)
        self.text_token_tags = st.get("text_token_tags", None)
        self.attn_steps: set = set()
        self.last_attn: dict = {}
        self.patch = tuple(cap["patch_size"])
        self.C = cap["vae_latent_channels"]
        # scheduler consistency with the pipeline
        self.sched.set_timesteps(args.steps, device=self.dev0)
        self.asched.set_timesteps(args.steps, device=self.dev0)
        assert torch.allclose(self.sched.timesteps.float(), self.timesteps.float(), atol=1e-6), "video timesteps mismatch"
        assert torch.allclose(self.asched.timesteps.float(), self.audio_timesteps.float(), atol=1e-6), "audio timesteps mismatch"
        assert len(self.plan) == len(self.timesteps) == args.steps - 1 or len(self.plan) == len(self.timesteps), \
            (len(self.plan), len(self.timesteps))
        log.info("[B] state: video rows %s (cond %d), audio rows %s (cond %d), text %s, %d model evals, latent %dx%dx%d",
                 tuple(self.latents0.shape), self.ncv, tuple(self.audio0.shape), self.nca, tuple(self.prompt_embeds.shape),
                 len(self.timesteps), self.nlf, self.lh, self.lw)

        self.vae_mean = torch.tensor(cap["vae_latents_mean"], device=self.aux_dev).view(1, -1, 1, 1, 1)
        self.vae_std = torch.tensor(cap["vae_latents_std"], device=self.aux_dev).view(1, -1, 1, 1, 1)
        self.pix_mean = torch.tensor(cap["pixel_mean"], device=self.aux_dev).view(1, -1, 1, 1, 1)
        self.pix_std = torch.tensor(cap["pixel_std"], device=self.aux_dev).view(1, -1, 1, 1, 1)
        self.aud_mean = torch.tensor(cap["audio_latents_mean"], device=self.aux_dev).view(1, -1, 1)
        self.aud_std = torch.tensor(cap["audio_latents_std"], device=self.aux_dev).view(1, -1, 1)

    @staticmethod
    def _build_device_map(cls_model, budget):
        """Explicit layer-wise map: fill GPU i with transformer blocks up to its GiB budget (weights sized in the
        checkpoint dtype from a meta-device instance); every non-block module goes to GPU 0."""
        from accelerate import init_empty_weights
        cfg = cls_model.load_config(CKPT, subfolder="transformer_ref")
        with init_empty_weights():
            meta = cls_model.from_config(cfg)
        fp32 = set(getattr(cls_model, "_keep_in_fp32_modules", []) or [])

        def size_gib(mod, name):
            bpp = 4 if any(name.startswith(m) or ("." + m) in ("." + name) for m in fp32) else 2
            return sum(p.numel() for p in mod.parameters()) * bpp / 2**30

        gpus = [i for i, b in enumerate(budget) if b > 0]
        dmap, used = {}, {g: 0.0 for g in gpus}
        cur = 0
        for name, child in meta.named_children():
            if name == "transformer_blocks":
                for bi, blk in enumerate(child):
                    sz = size_gib(blk, f"{name}.{bi}")
                    while cur < len(gpus) - 1 and used[gpus[cur]] + sz > budget[gpus[cur]]:
                        cur += 1
                    dmap[f"{name}.{bi}"] = gpus[cur]
                    used[gpus[cur]] += sz
            else:
                dmap[name] = gpus[0]
                used[gpus[0]] += size_gib(child, name)
        log.info("[B] explicit device map: weights per GPU %s GiB; blocks: %s",
                 {g: round(u, 1) for g, u in used.items()},
                 {g: sum(1 for k, v in dmap.items() if k.startswith("transformer_blocks") and v == g) for g in gpus})
        del meta
        return dmap

    # -- one transformer evaluation --------------------------------------------------------
    def _eval(self, latents, audio_latents, enc, i, plan=None):
        uts, tidx = (plan or self.plan)[i]
        out = self.tr(hidden_states=latents[None], audio_hidden_states=audio_latents[None], encoder_hidden_states=enc,
                      timestep=uts, timestep_indices=tidx, attention_kwargs=self.attention_kwargs, return_dict=False,
                      **self.layout)
        return out[0], out[1]

    KEYFRAME_NOISE_AUG = 0.999   # MiniMaxH3ModularPipeline.keyframe_noise_aug (verified against the captured plan)

    def _plan_for(self, steps: int):
        """(row_timestep_plan, timesteps, audio_timesteps) for an arbitrary step count, rebuilt exactly as the
        pipeline's SetTimesteps block does; the captured plan is used (and the rebuild verified) for --steps."""
        from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep as S
        self.sched.set_timesteps(steps, device=self.dev0)
        self.asched.set_timesteps(steps, device=self.dev0)
        ts, ats = self.sched.timesteps, self.asched.timesteps
        vi, ai = self.layout["video_indices"].cpu(), self.layout["audio_indices"].cpu()
        n_text = int(self.layout["text_indices"].numel())
        plan = [tuple(x.to(self.dev0) for x in S.build_row_timesteps(vi, ai, self.ncv, self.nca, n_text, float(t), float(at),
                                                                      max(float(t), self.KEYFRAME_NOISE_AUG), 1.0))
                for t, at in zip(ts, ats)]
        if steps == self.args.steps:
            ok = len(plan) == len(self.plan) and all(torch.allclose(a[0].float(), b[0].float()) and torch.equal(a[1], b[1])
                                                     for a, b in zip(plan, self.plan))
            assert ok, "rebuilt row-timestep plan differs from the captured one"
            return self.plan, self.timesteps, self.audio_timesteps
        return plan, ts, ats

    def render(self, z_audio, delta_text, grad_steps: int, steps: int | None = None):
        """Returns (video_rows [N_gen_rows, C*p] on dev0, audio_rows [N_gen_aud, 32] detached).
        The last `grad_steps` model evaluations are differentiable w.r.t. z_audio / delta_text."""
        steps = steps or self.args.steps
        plan, timesteps, audio_timesteps = self._plan_for(steps)
        n_eval = len(timesteps)
        latents = self.latents0
        aud_gen = self.audio0[self.nca:]
        self.last_attn = {}
        for i in range(n_eval):
            grad_on = i >= n_eval - grad_steps
            ctx = torch.enable_grad() if grad_on else torch.no_grad()
            with ctx:
                z = z_audio if grad_on else z_audio.detach()
                d = delta_text if grad_on else delta_text.detach()
                audio_latents = torch.cat([z.to(aud_gen.dtype), aud_gen], dim=0)
                enc = self.prompt_embeds.to(torch.float32) + d
                cap_attn = i in self.attn_steps
                if cap_attn:
                    import h3_attn
                    h3_attn.begin()
                v_pred, a_pred = self._eval(latents, audio_latents, enc, i, plan)
                if cap_attn:
                    res = h3_attn.end()
                    if res is not None:
                        self.last_attn[i] = res
                new_vid = self.sched.step(v_pred[0, self.ncv:].float(), timesteps[i], latents[self.ncv:], return_dict=False)[0]
                new_aud = self.asched.step(a_pred[0, self.nca:].float(), audio_timesteps[i], aud_gen, return_dict=False)[0]
                latents = torch.cat([latents[:self.ncv], new_vid], dim=0)
                aud_gen = new_aud
            if not grad_on:
                latents = latents.detach()
                aud_gen = aud_gen.detach()
        return latents[self.ncv:], aud_gen

    def _decode_latents(self, z, grad_chunks=None):
        """Replica of AutoencoderKLMiniMaxH3.decode/_decode (temporal chunks, overlap cross-fade, spatial tiling
        via _decode_clip) where only the chunks in `grad_chunks` keep an autograd graph. grad_chunks=None -> all."""
        from contextlib import nullcontext
        vae = self.vae
        z = z.to(next(vae.decoder.parameters()).dtype)
        tcs, td, tr = vae.tokens_chunk_size, vae.config.token_drop, vae.temporal_compression_ratio
        cnf = tcs * tr
        num_tokens = z.shape[2] + td
        pad_tokens = (-num_tokens) % tcs
        num_chunks = (num_tokens + pad_tokens) // tcs - int(td > 0)
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
        decoded, overlap = [], None
        for i in range(num_chunks):
            start = i * tcs
            zi = z[:, :, start:start + tcs + vae.token_overlap]
            with_grad = grad_chunks is None or i in grad_chunks
            with (nullcontext() if with_grad else torch.no_grad()):
                clip = vae._decode_clip(zi if with_grad else zi.detach())
            for j in range(int(td > 0) + 1):
                fs = j * cnf
                chunk = clip[:, :, fs:fs + cnf][:, :, vae.frame_pre_padding:]
                if j == 0:
                    if overlap is not None:
                        chunk = vae._blend(overlap, chunk, vae.frame_overlap, dim=-3)
                    decoded.append(chunk)
                else:
                    overlap = chunk
        if overlap is not None:
            decoded.append(overlap)
        dec = torch.cat(decoded, dim=2)
        if pad_tokens > 0:
            intra_tail = vae.config.clip_length % tr
            n_before = z.shape[2] - pad_tokens
            pad_frames = sum(intra_tail if intra_tail and (n_before + k) % tcs == 0 else tr for k in range(pad_tokens))
            dec = dec[:, :, :-pad_frames]
        self._num_decode_chunks = num_chunks
        return dec

    def decode_video(self, rows, grad_chunks=None):
        """rows [N_gen_rows, C*p] -> frames [F, 3, H, W] float in [0,1] on aux_dev.
        grad_chunks: None = every temporal chunk differentiable; else the set of chunk indices that carry gradient."""
        pt, ph, pw = self.patch
        x = rows.reshape(-1, self.nlf // pt, self.lh // ph, self.lw // pw, self.C, pt, ph, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(-1, self.C, self.nlf, self.lh, self.lw)
        x = x.to(self.aux_dev) * self.vae_std + self.vae_mean
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            video = self._decode_latents(x, grad_chunks)
        video = (video.float() * self.pix_std + self.pix_mean).clamp(0, 1)   # [1,3,F,H,W]
        return video[0].permute(1, 0, 2, 3)

    @torch.no_grad()
    def decode_audio(self, aud_rows):
        """generated audio rows [ch*T, 32] (normalized, channel-major) -> wav [samples, ch] float32."""
        ch = self.cap["audio_channels"]
        lat = aud_rows.reshape(ch, self.n_aud, -1).permute(0, 2, 1).to(self.aux_dev) * self.aud_std + self.aud_mean
        audio = self.audio_vae.decode(lat.float(), return_dict=False)[0]      # [ch, 1, samples]
        return audio.float().squeeze(1).permute(1, 0).cpu().numpy()


def main():
    args = parse_args()
    sc = SCEN[args.slug]
    out_dir = args.out_dir or f"/scratch/amirrz/H3_exp/outputs/fullmethod_{args.slug}_{args.opt_mode}"
    os.makedirs(out_dir, exist_ok=True)
    json.dump({**vars(args), "scenario": sc, "num_frames": _bs.NUM_FRAMES, "height": _bs.HEIGHT, "width": _bs.WIDTH},
              open(os.path.join(out_dir, "run_config.json"), "w"), indent=2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    # ================= Phase A (own process: frees the ~180 GB of offloaded pipeline weights before Phase B) ====
    cap_path = os.path.join(out_dir, "capture.pt")
    if args.phase in ("all", "a"):
        cap = None
        if os.path.exists(cap_path) and os.environ.get("REUSE_CAPTURE", "1") == "1":
            old = torch.load(cap_path, map_location="cpu", weights_only=False)
            ra = old.get("run_args", {})
            if (ra.get("steps") == args.steps and ra.get("seed") == args.seed and ra.get("slug") == args.slug
                    and (ra.get("prompt_edit") or "") == (args.prompt_edit or "")):
                log.info("[A] reusing existing capture %s (same slug/steps/seed)", cap_path)
                cap = old
            del old
        if cap is None:
            cap = phase_a_pipeline(args, sc, out_dir)
            cap["run_args"] = {"steps": args.steps, "seed": args.seed, "slug": args.slug, "prompt_edit": args.prompt_edit or ""}
            log.info("[A] captured state keys: %s", sorted(cap["state"].keys()))
            torch.save(cap, cap_path)
            log.info("[A] capture saved to %s (%.1f GB)", cap_path, os.path.getsize(cap_path) / 1e9)
        if args.phase == "a":
            return
    cap = torch.load(cap_path, map_location="cpu", weights_only=False)
    log.info("[B] capture loaded: %s", sorted(cap["state"].keys()))

    # ================= Phase B =================
    reset_peaks()
    H = H3Grad(args, cap)
    if args.attn_vis and H.text_token_tags is not None:
        sys.path.insert(0, _HERE)
        import h3_attn
        h3_attn.install({k: v.cpu() for k, v in H.layout.items()}, H.text_token_tags, H.ncv, H.nca)
        H.attn_steps = {int(x) for x in args.attn_steps.replace(";", ",").split(",") if x.strip()}
        log.info("[B] attention-mass capture at model evals %s (groups: %s)", sorted(H.attn_steps), h3_attn.GROUPS)
    from motion_opt.qwen_loss import build_qwen_model, build_qwen_rubric_inputs, compute_qwen_video_loss
    from motion_opt.perceptual_loss import adaptive_reg_weight, compute_perceptual_quality_loss
    qwen, proc = build_qwen_model(QWEN, device=H.aux_dev, gradient_checkpointing=True)
    qwen.requires_grad_(False)
    if args.qwen_img_size != 224:  # build_qwen_model pins the processor to 224^2 pixels; rebuild it for the requested size
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(QWEN, min_pixels=args.qwen_img_size ** 2, max_pixels=args.qwen_img_size ** 2)
        log.info("[B] critic processor rebuilt for img_size=%d", args.qwen_img_size)
    if args.critic_fp32_head:
        qwen.lm_head = FP32Head(qwen.lm_head)
    CROP = parse_crop(args.qwen_crop)
    log.info("[B] critic: img_size=%d crop=%s fp32_head=%s select_by=%s", args.qwen_img_size, CROP, bool(args.critic_fp32_head), args.select_by)
    ci, yes_id, no_id = build_qwen_rubric_inputs(processor=proc, edit_prompt=sc["edit"], num_frames=args.qwen_max_frames,
                                                 img_size=args.qwen_img_size, device=H.aux_dev,
                                                 motion_question=(args.motion_question or sc.get("question") or None), gradient_rubric="motion")
    log.info("[B] critic motion question: %s", args.motion_question or sc.get("question") or "<library default>")
    ov = {"motion": 1.0, "entities": 0.0, "overall": 0.0} if args.qwen_gradient_rubric == "motion" else None  # None -> rubric defaults
    ov_report = {"motion": 1.0, "entities": 0.0, "overall": 0.0}   # reporting stays motion-only (comparable to all H3 tables)
    log.info("[B] models ready; gradient rubric=%s; peak mem %s", args.qwen_gradient_rubric, gpu_mem_str())

    @torch.no_grad()
    def score_windows(frames):
        """No-grad critic scores: linspace window (training objective), mean of 4 windows (legacy tables),
        and noisy-OR / max over 6 contiguous 24-frame windows (brief events)."""
        fr = crop_frames(frames.detach().to(H.aux_dev), CROP)
        wins = [("linspace", 0)] + [("contiguous", st) for st in (0, 20, 40, 50, 60, 80, 100)]
        nll = {}
        for mode, start in wins:
            loss, _ = compute_qwen_video_loss(frames_chw=fr, qwen_model=qwen, cached_inputs=ci, yes_token_id=yes_id,
                                              no_token_id=no_id, max_frames=args.qwen_max_frames, img_size=args.qwen_img_size,
                                              backward=False, return_details=True, sample_mode=mode, contiguous_start_frame=start,
                                              rubric_weight_overrides=ov_report)
            nll[(mode, start)] = float(loss)
        lin = nll[("linspace", 0)]
        four = float(np.mean([lin, nll[("contiguous", 0)], nll[("contiguous", 50)], nll[("contiguous", 100)]]))
        contig = [nll[("contiguous", st)] for st in (0, 20, 40, 60, 80, 100)]
        p_any = 1.0 - float(np.prod([1.0 - math.exp(-v) for v in contig]))
        return {"lin": lin, "4win": four, "any": -math.log(max(p_any, 1e-8)), "max": min(contig), "contig": contig}

    @torch.no_grad()
    def score_frames(frames):
        """4-window dithered Qwen score (reporting; same as all H3 tables) + single-window nll (training objective)."""
        fr = crop_frames(frames.detach().to(H.aux_dev), CROP)
        wins = [("linspace", 0), ("contiguous", 0), ("contiguous", 50), ("contiguous", 100)]
        vals = []
        for mode, start in wins:
            loss, _ = compute_qwen_video_loss(frames_chw=fr, qwen_model=qwen, cached_inputs=ci, yes_token_id=yes_id,
                                              no_token_id=no_id, max_frames=args.qwen_max_frames, img_size=args.qwen_img_size,
                                              backward=False, return_details=True, sample_mode=mode, contiguous_start_frame=start,
                                              rubric_weight_overrides=ov_report)
            vals.append(float(loss))
        return float(np.mean(vals)), vals[0]

    # ---- parameters (LTX: audio latent optimized directly, anchored to source; text residual anchored to 0) ----
    z_src = H.audio0[:H.nca].detach().clone().float()
    z_audio = torch.nn.Parameter(z_src.clone())
    delta_text = torch.nn.Parameter(torch.zeros(H.prompt_embeds.shape, dtype=torch.float32, device=H.dev0))
    if args.init_latents:
        init = torch.load(args.init_latents, map_location="cpu")
        zi, di = init["best_z"].to(H.dev0).float(), init["best_d"].to(H.dev0).float()
        assert zi.shape == z_audio.shape and di.shape == delta_text.shape, (zi.shape, di.shape)
        with torch.no_grad():
            z_audio.copy_(zi)
            delta_text.copy_(di)
        log.info("[B] initialized from %s (best_iter=%s): |z-z_src|=%.2f |d|=%.1f", args.init_latents, init.get("best_iter"),
                 float((zi - z_src).norm()), float(di.norm()))
    opt_audio = args.opt_mode in ("both", "audio")
    opt_text = args.opt_mode in ("both", "text")
    z_audio.requires_grad_(opt_audio)
    delta_text.requires_grad_(opt_text)
    groups = []
    if opt_audio:
        groups.append({"params": [z_audio], "lr": args.lr})
    if opt_text:
        groups.append({"params": [delta_text], "lr": args.lr * args.text_lr_mult})
    optimizer = torch.optim.Adam(groups)   # used for adam; for ngd only its zero_grad/param bookkeeping is used
    lr_sched = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iterations, eta_min=args.lr * 0.01)
                if args.lr_schedule == "cosine" and args.iterations > 1 and args.optimizer == "adam" else None)
    ref_norm = {"audio": float(z_src.norm()), "text": float(H.prompt_embeds.float().norm())}
    ngd_buf = {"audio": None, "text": None}

    def ngd_step(it: int):
        """theta <- theta - eta(it) * ||theta_ref|| * m/||m||, m = momentum-averaged gradient (per parameter)."""
        frac = (it - 1) / max(1, args.iterations - 1)
        eta = args.ngd_eta * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac))) if args.lr_schedule == "cosine" else args.ngd_eta
        with torch.no_grad():
            for name, prm in (("audio", z_audio), ("text", delta_text)):
                if prm.grad is None:
                    continue
                g = prm.grad
                ngd_buf[name] = g.clone() if ngd_buf[name] is None else args.ngd_momentum * ngd_buf[name] + g
                m = ngd_buf[name]
                step = eta * (args.text_eta_mult if name == "text" else 1.0)
                prm.add_(-step * ref_norm[name] * m / (m.norm() + 1e-12))
        return eta
    text_std = float(H.prompt_embeds.float().std())
    log.info("[B] params: audio latent %s (|z|=%.1f, std %.3f), text residual %s (base std %.3f); mode=%s",
             tuple(z_audio.shape), float(z_src.norm()), float(z_src.std()), tuple(delta_text.shape), text_std, args.opt_mode)

    def attn_report(frames, tag, it):
        """Render + log the attention-mass maps captured during the last H.render (never raises)."""
        if not H.last_attn:
            return
        try:
            import h3_attn
            for step_i, res in sorted(H.last_attn.items()):
                png = os.path.join(out_dir, f"attn_iter_{it:02d}_step{step_i:02d}.png")
                st = h3_attn.render(res, frames, H.nlf, H.lh, H.lw, H.patch, png, tag=f"{tag} step {step_i}")
                maps = h3_attn.maps_from_mass(res["mass"], H.nlf, H.lh, H.lw, H.patch)
                np.savez_compressed(png[:-4] + ".npz", maps=maps.astype(np.float16), groups=np.array(h3_attn.GROUPS))
                json.dump(st, open(png[:-4] + ".json", "w"), indent=1)
                mm, cr = st["mean_mass"], st["corr_with_motion"]
                log.info("   attn@%d %s: mass audio_ref=%.4f audio_gen=%.4f text=%.4f vision=%.4f video_ref=%.3f self=%.3f | "
                         "corr(motion): audio_ref=%+.2f text=%+.2f vision=%+.2f video_ref=%+.2f%s", step_i, tag,
                         mm["audio_ref"], mm["audio_gen"], mm["text_txt"], mm["text_vis"], mm["video_ref"], mm["video_gen"],
                         cr["audio_ref"], cr["text_txt"], cr["text_vis"], cr["video_ref"],
                         f"  (capture error: {st['err']})" if st.get("err") else "")
        except Exception as e:
            log.warning("attention report failed: %s", e)
        H.last_attn = {}

    if args.phase == "render":
        # re-render baseline (source conditioning) and the --init-latents at several step counts; score + save
        assert args.init_latents, "phase=render needs --init-latents"
        out = {}
        for steps in [int(x) for x in args.render_steps.replace(";", ",").split(",") if x.strip()]:
            for tag, zz, dd in (("baseline", z_src, torch.zeros_like(delta_text)), ("optimized", z_audio.detach(), delta_text.detach())):
                t0 = time.time()
                with torch.no_grad():
                    rows, aud = H.render(zz, dd, grad_steps=0, steps=steps)
                    fr = H.decode_video(rows)
                    wav = H.decode_audio(aud)
                sw = score_windows(fr)
                key = f"{tag}_{steps}"
                save_av(fr.permute(0, 2, 3, 1).cpu().numpy(), wav, cap["audio_sr"], os.path.join(out_dir, f"render_{key}.mp4"))
                attn_report(fr, key, {"baseline": 0, "optimized": 1}[tag] + 10 * steps)   # attn_iter_<10*steps+{0,1}>_stepNN.png
                if tag == "baseline":
                    base_fr = fr.detach()
                    lp = 0.0
                else:
                    lp = float(compute_perceptual_quality_loss(gen_frames=fr, src_frames=base_fr, lpips_weight=1.0, temporal_weight=0.0,
                                                               backbone=args.lpips_backbone, max_lpips_frames=24, max_temporal_pairs=1,
                                                               backward=False)[0])
                out[key] = {"steps": steps, "nll_lin": sw["lin"], "yes_lin": math.exp(-sw["lin"]), "yes_any": math.exp(-sw["any"]),
                            "yes_maxwin": math.exp(-sw["max"]), "yes_4win": math.exp(-sw["4win"]), "lpips_vs_baseline_same_steps": lp,
                            "sec": round(time.time() - t0)}
                log.info("[render] %-16s yes lin=%.4f any=%.4f max-win=%.4f 4win=%.4f lpips=%.4f (%.0fs)", key, out[key]["yes_lin"],
                         out[key]["yes_any"], out[key]["yes_maxwin"], out[key]["yes_4win"], lp, out[key]["sec"])
                json.dump(out, open(os.path.join(out_dir, "render_results.json"), "w"), indent=1)
        log.info("RENDER PHASE DONE")
        return

    # ---- CHECK 1: our loop (no grad) reproduces the pipeline ----
    t0 = time.time()
    rows_b, aud_b = H.render(z_src, torch.zeros_like(delta_text), grad_steps=0)   # always the SOURCE conditioning
    with torch.no_grad():
        frames_b = H.decode_video(rows_b)                       # [F,3,H,W] on aux
    pf = cap["pipeline_frames"].permute(0, 3, 1, 2).to(H.aux_dev)  # [F,3,H,W]
    n = min(pf.shape[0], frames_b.shape[0])
    diff = (frames_b[:n] - pf[:n]).abs()
    log.info("[CHECK1] loop-vs-pipeline: mean|diff|=%.4f  max=%.3f  (frames %d vs %d, %.0fs)  -> %s",
             float(diff.mean()), float(diff.max()), frames_b.shape[0], pf.shape[0], time.time() - t0,
             "OK" if float(diff.mean()) < 0.02 else "MISMATCH - investigate")
    base_dither, base_nll = score_frames(frames_b)
    log.info("[B] baseline (our loop): qwen nll=%.4f yes=%.5f | 4-window nll=%.4f yes=%.5f", base_nll, math.exp(-base_nll),
             base_dither, math.exp(-base_dither))
    wav_b = H.decode_audio(aud_b)
    save_av(frames_b.permute(0, 2, 3, 1).cpu().numpy(), wav_b, cap["audio_sr"], os.path.join(out_dir, "baseline.mp4"))
    cached_src_frames = frames_b.detach()
    attn_report(frames_b, "baseline", 0)
    del rows_b, diff, pf
    torch.cuda.empty_cache()

    # ---- one full differentiable step: render -> decode -> losses -> backward ----
    def loss_and_backward(it: int, num_iters: int):
        reset_peaks()
        t_r = time.time()
        rows, aud = H.render(z_audio, delta_text, grad_steps=args.grad_steps)
        mem_after_render = gpu_mem_str()
        n_ch = getattr(H, "_num_decode_chunks", None)
        if args.decode_grad_frac >= 1.0 or n_ch is None:
            gchunks = None
        else:
            stride = max(1, int(round(1.0 / args.decode_grad_frac)))
            gchunks = set(range(it % stride, n_ch, stride))       # rotates over iterations -> every chunk gets gradient
        frames = H.decode_video(rows, gchunks)                  # graph -> last K steps -> params
        t_render = time.time() - t_r
        if args.save_iter_previews and it > 0:
            try:
                save_av(frames.detach().permute(0, 2, 3, 1).cpu().numpy(), H.decode_audio(aud.detach()), cap["audio_sr"],
                        os.path.join(out_dir, f"iter_{it:02d}.mp4"))
            except Exception as e:  # never let a preview kill the run
                log.warning("preview save failed: %s", e)
        log.info("   render done (%s) ; decode grad chunks=%s/%s ; peak after decode %s", mem_after_render,
                 "all" if gchunks is None else sorted(gchunks), n_ch, gpu_mem_str())
        if it > 0:
            attn_report(frames.detach(), f"iter {it}", it)
        else:
            H.last_attn = {}
        assert frames.requires_grad, "decoded frames carry no graph - gradient path broken"
        metrics = {}
        # perceptual (LTX order: first, retain graph)
        need_perc = args.lpips_weight > 0 or args.temporal_weight > 0
        if need_perc:
            perc, det = compute_perceptual_quality_loss(gen_frames=frames, src_frames=cached_src_frames,
                                                        lpips_weight=args.lpips_weight, temporal_weight=args.temporal_weight,
                                                        backbone=args.lpips_backbone, max_lpips_frames=min(24, frames.shape[0]),
                                                        max_temporal_pairs=min(12, frames.shape[0] - 1), backward=False)
            if perc.requires_grad:
                perc.backward(retain_graph=True)
            metrics["perceptual"] = float(perc.detach())
            metrics.update({f"perc_{k}": float(v) for k, v in det.items() if isinstance(v, (int, float))})
        # selection scores on this render (no grad) - what the iter-XX previews are judged by
        sw = score_windows(frames)
        metrics.update(nll_4win=sw["4win"], nll_any=sw["any"], nll_max=sw["max"], nll_lin_sel=sw["lin"])
        # Qwen critic with gradient accumulation over frame windows (as in multimodal_loop_qwen.py)
        accum = max(1, args.qwen_grad_accum_steps)
        frames_c = crop_frames(frames, CROP)
        if args.critic_objective == "any":
            # noisy-OR over contiguous windows: L = -log(1 - prod_w (1 - p_w)). Pass 1 (no grad) gives the p_w and the
            # per-window coefficients dL/dnll_w = S p_w / ((1-S)(1-p_w)); pass 2 backpropagates each window separately
            # (one Qwen graph alive at a time), the gradient concentrating on the windows that contain the event.
            starts = (0, 20, 40, 60, 80, 100)
            gf = frames_c.detach().requires_grad_(True)
            with torch.no_grad():
                pw = []
                for st in starts:
                    nll_w = compute_qwen_video_loss(frames_chw=gf, qwen_model=qwen, cached_inputs=ci, yes_token_id=yes_id,
                                                    no_token_id=no_id, max_frames=args.qwen_max_frames, img_size=args.qwen_img_size,
                                                    backward=False, return_details=False, sample_mode="contiguous",
                                                    contiguous_start_frame=st, rubric_weight_overrides=ov)
                    pw.append(min(max(math.exp(-float(nll_w)), 1e-12), 1 - 1e-6))
            S = float(np.prod([1.0 - x for x in pw]))
            L_any = -math.log(max(1.0 - S, 1e-12))
            coefs = [S * x / ((1.0 - S) * (1.0 - x)) for x in pw]
            for st, c in zip(starts, coefs):
                if c < 1e-4 * max(coefs):
                    continue
                nll_w = compute_qwen_video_loss(frames_chw=gf, qwen_model=qwen, cached_inputs=ci, yes_token_id=yes_id,
                                                no_token_id=no_id, max_frames=args.qwen_max_frames, img_size=args.qwen_img_size,
                                                backward=False, return_details=False, sample_mode="contiguous",
                                                contiguous_start_frame=st, rubric_weight_overrides=ov)
                (c * nll_w).backward()
            frames_c.backward(gf.grad)
            qwen_nll = L_any
            metrics["p_windows"] = " ".join(f"{x:.4f}" for x in pw)
            log.info("   any-window objective: P(event)=%.4f  per-window yes=%s  coefs=%s", 1.0 - S, metrics["p_windows"],
                     " ".join(f"{c:.2f}" for c in coefs))
        elif accum == 1:
            qloss, _ = compute_qwen_video_loss(frames_chw=frames_c, qwen_model=qwen, cached_inputs=ci, yes_token_id=yes_id,
                                               no_token_id=no_id, max_frames=args.qwen_max_frames, img_size=args.qwen_img_size,
                                               backward=True, return_details=True, sample_mode=args.qwen_sample_mode,
                                               contiguous_start_frame=0, rubric_weight_overrides=ov)
            qwen_nll = float(qloss)
        else:
            gf = frames_c.detach().requires_grad_(True)
            acc = 0.0
            nll_lin = None
            for k in range(accum):
                mode = args.qwen_sample_mode if k == 0 else "contiguous_random"
                res = compute_qwen_video_loss(frames_chw=gf, qwen_model=qwen, cached_inputs=ci, yes_token_id=yes_id,
                                              no_token_id=no_id, max_frames=args.qwen_max_frames, img_size=args.qwen_img_size,
                                              backward=False, return_details=(k == 0), sample_mode=mode,
                                              contiguous_start_frame=0, rubric_weight_overrides=ov)
                li = res[0] if k == 0 else res
                (li / accum).backward()
                acc += float(li) / accum
                if k == 0:
                    nll_lin = float(li)
            frames_c.backward(gf.grad)    # chain the averaged critic gradient through decode + K steps
            qwen_nll = nll_lin            # tracked/reported: the deterministic window (comparable to the baseline)
            metrics["qwen_nll_train_avg"] = acc
        # regularizers (LTX forms)
        a_w = adaptive_reg_weight(args.latent_reg_weight, it, num_iters, schedule=args.reg_schedule)
        t_w = adaptive_reg_weight(args.text_reg_weight, it, num_iters, schedule=args.reg_schedule)
        reg = torch.zeros((), device=H.dev0)
        if opt_audio and a_w > 0:
            reg = reg + a_w * torch.mean((z_audio - z_src) ** 2)
        if opt_text and t_w > 0:
            reg = reg + t_w * torch.mean(delta_text ** 2)
        if reg.requires_grad:
            reg.backward()
        metrics.update(qwen_nll=qwen_nll, yes_prob=math.exp(-qwen_nll), reg=float(reg), total=qwen_nll + float(reg),
                       t_render=t_render, t_total=time.time() - t_r, peak_mem=gpu_mem_str())
        perc_v = metrics.get("perceptual", 0.0)
        metrics["sel"] = {"lin": metrics["nll_lin_sel"] + perc_v, "any": metrics["nll_any"] + perc_v,
                          "4win": metrics["nll_4win"] + perc_v, "train": metrics["total"]}[args.select_by]
        return metrics

    # ---- CHECK 2: finite-difference gradient check (one grad pass + two no-grad renders) ----
    if args.grad_check:
        optimizer.zero_grad(set_to_none=True)
        m0 = loss_and_backward(0, args.iterations)
        g_a = z_audio.grad.detach().clone() if opt_audio else None
        g_t = delta_text.grad.detach().clone() if opt_text else None
        for name, g in (("audio", g_a), ("text", g_t)):
            if g is not None:
                assert torch.isfinite(g).all(), f"{name} grad has non-finite values"
                assert float(g.norm()) > 0, f"{name} grad is identically zero - gradient path broken"
        log.info("[CHECK2] grads: |g_audio|=%s |g_text|=%s  (nll=%.4f, %s, %.0fs)",
                 f"{float(g_a.norm()):.3e}" if g_a is not None else "-", f"{float(g_t.norm()):.3e}" if g_t is not None else "-",
                 m0["qwen_nll"], m0["peak_mem"], m0["t_total"])
        # compare single-window (linspace) scores throughout: the training value m0["qwen_nll"] averages accum windows
        nll0 = base_nll
        eps_a = 0.03 * float(z_src.norm()) if g_a is not None else 0.0
        eps_t = 0.03 * float(H.prompt_embeds.float().norm()) if g_t is not None else 0.0
        fd = {}
        for sign in (-1.0, +1.0):
            with torch.no_grad():
                za = z_audio.detach().clone()
                dt = delta_text.detach().clone()
                if g_a is not None:
                    za = za + sign * eps_a * g_a / g_a.norm()
                if g_t is not None:
                    dt = dt + sign * eps_t * g_t / g_t.norm()
                rows, _ = H.render(za, dt, grad_steps=0)
                fr = H.decode_video(rows)
                _, nll = score_frames(fr)
            fd[sign] = nll
            log.info("[CHECK2] step %s grad: qwen nll (single window) %.4f -> %.4f (delta %+.4f)",
                     "along -" if sign < 0 else "along +", nll0, nll, nll - nll0)
        verdict = "OK: loss decreases along -grad and increases along +grad" if fd[-1.0] < nll0 < fd[1.0] else \
            ("OK (one-sided): descent direction lowers loss, +grad flat/quantized" if fd[-1.0] < nll0 - 0.01
             else "FAIL: -grad does not lower loss (noise or bug)")
        log.info("[CHECK2] verdict: %s", verdict)
        json.dump({"nll0_single_window": nll0, "nll0_train_avg": m0["qwen_nll"], "nll_minus": fd[-1.0], "nll_plus": fd[1.0], "eps_audio": eps_a, "eps_text": eps_t,
                   "g_audio": float(g_a.norm()) if g_a is not None else None, "g_text": float(g_t.norm()) if g_t is not None else None,
                   "verdict": verdict}, open(os.path.join(out_dir, "grad_check.json"), "w"), indent=2)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    # ---- optimization ----
    # Baseline "total" for best-tracking must be measured the same way as the iterations (accumulation mean over the
    # Qwen frame windows), not the single-window nll: reuse the grad-check pass (params unchanged = baseline) when it
    # ran, otherwise start from +inf so the first iteration becomes best.
    base_total = m0["total"] if args.grad_check else float("inf")
    base_sel = m0["sel"] if args.grad_check else float("inf")
    log.info("[B] best-tracking baseline: total=%.4f sel(%s)=%.4f (%s)", base_total, args.select_by, base_sel,
             "grad-check pass" if args.grad_check else "none")
    best = {"total": base_total, "sel": base_sel, "qwen_nll": base_nll, "iter": 0, "z": z_src.clone(), "d": torch.zeros_like(delta_text)}
    csvf = open(os.path.join(out_dir, "opt_log.csv"), "w", newline="")
    cw = csv.writer(csvf)
    cw.writerow(["iter", "qwen_nll", "yes_prob", "nll_4win", "yes_4win", "nll_any", "yes_any", "nll_max", "perceptual", "reg", "total",
                 "sel", "g_audio", "g_text", "dz_audio", "d_text", "lr", "is_best", "t_total", "peak_mem"])
    no_improve = 0
    for it in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        m = loss_and_backward(it, args.iterations)
        # snapshot the parameters that PRODUCED this loss, before the optimizer moves them (best-checkpoint fix)
        z_pre, d_pre = z_audio.detach().clone(), delta_text.detach().clone()
        # per-iteration latents (the state that produced iter_XX.mp4) so any iteration can be re-rendered with --init-latents
        torch.save({"best_z": z_pre.cpu(), "best_d": d_pre.cpu(), "best_iter": it}, os.path.join(out_dir, f"iter_{it:02d}_latents.pt"))
        ga = float(z_audio.grad.norm()) if (opt_audio and z_audio.grad is not None) else 0.0
        gt = float(delta_text.grad.norm()) if (opt_text and delta_text.grad is not None) else 0.0
        if args.optimizer == "ngd":
            cur_lr = ngd_step(it)
        else:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for p in (z_audio, delta_text) if p.grad is not None], args.grad_clip)
            optimizer.step()
            if lr_sched is not None:
                lr_sched.step()
            cur_lr = optimizer.param_groups[0]["lr"]
        is_best = m["sel"] < best["sel"] and m.get("perceptual", 0.0) <= args.perc_max
        if is_best:
            best = {"total": m["total"], "sel": m["sel"], "qwen_nll": m["qwen_nll"], "iter": it, "z": z_pre, "d": d_pre}
            no_improve = 0
        else:
            no_improve += 1
        dz = float((z_audio.detach() - z_src).norm())
        dd = float(delta_text.detach().norm())
        cw.writerow([it, m["qwen_nll"], m["yes_prob"], m["nll_4win"], math.exp(-m["nll_4win"]), m["nll_any"], math.exp(-m["nll_any"]),
                     m["nll_max"], m.get("perceptual", 0.0), m["reg"], m["total"], m["sel"], ga, gt, dz, dd, cur_lr, int(is_best),
                     round(m["t_total"]), m["peak_mem"]])
        csvf.flush()
        log.info("iter %2d/%d  qwen_nll=%.4f yes=%.4f | any yes=%.4f max-win yes=%.4f 4win yes=%.4f | perc=%.4f reg=%.4f sel=%.4f | "
                 "g_a=%.2e g_t=%.2e |dz|=%.2f |d|=%.2f lr=%.4f best_sel=%.4f@%d %s (%.0fs, %s)", it, args.iterations, m["qwen_nll"],
                 m["yes_prob"], math.exp(-m["nll_any"]), math.exp(-m["nll_max"]), math.exp(-m["nll_4win"]), m.get("perceptual", 0.0),
                 m["reg"], m["sel"], ga, gt, dz, dd, cur_lr, best["sel"], best["iter"], "*" if is_best else "", m["t_total"], m["peak_mem"])
        torch.save({"z_audio": z_audio.detach().cpu(), "delta_text": delta_text.detach().cpu(), "z_src": z_src.cpu(),
                    "best_z": best["z"].cpu(), "best_d": best["d"].cpu(), "best_iter": best["iter"]},
                   os.path.join(out_dir, "latents.pt"))
        if args.early_stopping > 0 and no_improve >= args.early_stopping:
            log.info("early stopping at iter %d (no improvement for %d iters)", it, no_improve)
            break
    csvf.close()

    # ---- final render with the best latents (same schedule, same noise as the baseline) ----
    with torch.no_grad():
        rows, aud = H.render(best["z"], best["d"], grad_steps=0)
        frames_o = H.decode_video(rows)
        wav_o = H.decode_audio(aud)
    fin_dither, fin_nll = score_frames(frames_o)
    sw_fin, sw_base = score_windows(frames_o), score_windows(cached_src_frames)
    save_av(frames_o.permute(0, 2, 3, 1).cpu().numpy(), wav_o, cap["audio_sr"], os.path.join(out_dir, "optimized_final.mp4"))
    fd_pix = float((frames_o - cached_src_frames).abs().mean())
    res = {"slug": args.slug, "opt_mode": args.opt_mode, "baseline_nll": base_nll, "baseline_yes": math.exp(-base_nll),
           "baseline_nll_4win": base_dither, "final_nll": fin_nll, "final_yes": math.exp(-fin_nll), "final_nll_4win": fin_dither,
           "best_iter": best["iter"], "best_total": best["total"], "best_sel": best["sel"], "select_by": args.select_by,
           "critic": {"img_size": args.qwen_img_size, "crop": args.qwen_crop, "fp32_head": bool(args.critic_fp32_head),
                      "motion_question": args.motion_question or "default"},
           "frame_diff_vs_baseline": fd_pix, "baseline_yes_any": math.exp(-sw_base["any"]), "final_yes_any": math.exp(-sw_fin["any"]),
           "baseline_yes_maxwin": math.exp(-sw_base["max"]), "final_yes_maxwin": math.exp(-sw_fin["max"]),
           "dz_audio": float((best["z"] - z_src).norm()), "d_text": float(best["d"].norm()), "grad_steps": args.grad_steps}
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=2)
    log.info("DONE. baseline nll=%.4f (yes=%.5f) -> optimized nll=%.4f (yes=%.5f) [4-win: %.4f -> %.4f]; best iter %d; "
             "|dz|=%.2f |d|=%.2f; mean frame diff vs baseline %.4f", base_nll, math.exp(-base_nll), fin_nll, math.exp(-fin_nll),
             base_dither, fin_dither, best["iter"], res["dz_audio"], res["d_text"], fd_pix)


if __name__ == "__main__":
    main()
