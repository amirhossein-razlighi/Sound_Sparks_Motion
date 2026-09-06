#!/usr/bin/env python3
"""Baseline screening for user-study candidates (1 GPU).

For each candidate in the JSON (env CANDS; optional ONLY / GROUP filters): render H3's text-editing
baseline with the generic [video editing] grammar (seed 42, 16 steps, same settings as the method's
baseline), save mp4 + wav + _av.mp4 + a contact sheet, and score it with the critic in the method's
configuration (fp32 head, 224 px): the scenario's own "any frame" question (single window + noisy-OR
over 6 contiguous windows) and the library default question. Writes <OUT>/screen_results.json.
"""
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
_PROBE = os.path.abspath(_HERE + "/..")
_REPO = os.path.abspath(_PROBE + "/..")
sys.path.insert(0, os.path.join(_REPO, "editing/src"))
sys.path.insert(0, _PROBE)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("screen")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_bs = _load("baseline_sweep", os.path.join(_PROBE, "baseline_sweep.py"))
_fm = _load("h3_full_method", os.path.join(_PROBE, "h3_full_method.py"))
CKPT, QWEN = _bs.CKPT, _bs.QWEN
STEPS, SEED, NUM_FRAMES, HEIGHT, WIDTH = _bs.STEPS, _bs.SEED, _bs.NUM_FRAMES, _bs.HEIGHT, _bs.WIDTH
CANDS = os.environ["CANDS"]
OUT = os.environ.get("OUT_DIR", "/scratch/amirrz/H3_exp/outputs/screen_r1")
ONLY = [s for s in os.environ.get("ONLY", "").replace(";", ",").split(",") if s]
GROUP = os.environ.get("GROUP", "")


def main():
    os.makedirs(OUT, exist_ok=True)
    from diffusers import ModularPipeline
    from diffusers.modular_pipelines import ComponentsManager
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3AudioReference, MiniMaxH3VideoReference
    from motion_opt.qwen_loss import build_qwen_model, build_qwen_rubric_inputs, compute_qwen_video_loss

    cands = {k: v for k, v in json.load(open(CANDS)).items() if not k.startswith("_")}
    todo = [k for k, v in cands.items() if (not ONLY or k in ONLY) and (not GROUP or v.get("group") == GROUP)]
    log.info("screening %d candidates: %s", len(todo), todo)

    cm = ComponentsManager()
    cm.enable_auto_cpu_offload(device="cuda")
    pipe = ModularPipeline.from_pretrained(CKPT, workflow="ref2va", components_manager=cm)
    pipe.load_components(torch_dtype=torch.bfloat16)
    dev = torch.device("cuda")
    qwen, proc = build_qwen_model(QWEN, device=dev, gradient_checkpointing=False)
    qwen.lm_head = _fm.FP32Head(qwen.lm_head)
    qwen.to("cpu")
    torch.cuda.empty_cache()
    ov = {"motion": 1.0, "entities": 0.0, "overall": 0.0}

    rp = os.path.join(OUT, "screen_results.json")
    results = json.load(open(rp)) if os.path.exists(rp) else {}

    def score(vid, edit, question):
        fr = torch.from_numpy(np.ascontiguousarray(vid)).permute(0, 3, 1, 2).float().clamp(0, 1).to(dev)
        qwen.to(dev)
        out = {}
        with torch.no_grad():
            for qname, q in (("q_scenario", question), ("q_default", None)):
                ci, yes_id, no_id = build_qwen_rubric_inputs(processor=proc, edit_prompt=edit, num_frames=24, img_size=224,
                                                             device=dev, motion_question=q, gradient_rubric="motion")
                wins = [("linspace", 0)] + [("contiguous", st) for st in (0, 20, 40, 60, 80, 100)]
                p = []
                for mode, start in wins:
                    loss, _ = compute_qwen_video_loss(frames_chw=fr, qwen_model=qwen, cached_inputs=ci, yes_token_id=yes_id,
                                                      no_token_id=no_id, max_frames=24, img_size=224, backward=False,
                                                      return_details=True, sample_mode=mode, contiguous_start_frame=start,
                                                      rubric_weight_overrides=ov)
                    p.append(math.exp(-float(loss)))
                out[qname] = {"yes_lin": p[0], "yes_any": 1 - float(np.prod([1 - x for x in p[1:]])), "yes_max": max(p[1:])}
        qwen.to("cpu")
        torch.cuda.empty_cache()
        return out

    for slug in todo:
        sc = cands[slug]
        path = os.path.join(OUT, f"{slug}.mp4")
        if slug in results and os.path.exists(path):
            log.info("[skip] %s", slug)
            continue
        src = sc["src"] if os.path.isabs(sc["src"]) else os.path.join(_REPO, sc["src"])
        if not (os.path.exists(src) and os.path.exists(sc["wav"])):
            log.warning("missing input for %s (%s / %s) - skipped", slug, os.path.exists(src), os.path.exists(sc["wav"]))
            results[slug] = {"error": "missing input"}
            continue
        t0 = time.time()
        try:
            refs = [MiniMaxH3VideoReference.from_file(src), MiniMaxH3AudioReference.from_file(sc["wav"])]
            prompt = _bs.grammar_prompt(sc["edit"], sc["scene"])
            state = pipe(prompt=prompt, references=refs, num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH,
                         num_inference_steps=STEPS, generator=torch.Generator("cuda").manual_seed(SEED), output_type="np")
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
            _fm.save_av(vid, wav, int(state.values.get("sampling_rate", 32000) or 32000), path)
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-threads", "2", "-i", path, "-vf",
                            "select='not(mod(n\\,10))',scale=224:-1,tile=13x1", "-frames:v", "1", path[:-4] + "_sheet.png"], check=False)
            sc_out = score(vid, sc["edit"], sc["question"])
            results[slug] = {**sc_out, "edit": sc["edit"], "group": sc.get("group"), "src": src, "sec": round(time.time() - t0)}
            log.info("RESULT %-24s scenario-q: lin=%.4f any=%.4f max=%.4f | default-q: lin=%.4f any=%.4f  (%.0fs)  %s", slug,
                     sc_out["q_scenario"]["yes_lin"], sc_out["q_scenario"]["yes_any"], sc_out["q_scenario"]["yes_max"],
                     sc_out["q_default"]["yes_lin"], sc_out["q_default"]["yes_any"], time.time() - t0, sc["edit"])
        except Exception as e:
            log.exception("FAILED %s: %s", slug, e)
            results[slug] = {"error": str(e)[:300], "edit": sc["edit"]}
        json.dump(results, open(rp, "w"), indent=1)
    log.info("SCREEN DONE")


if __name__ == "__main__":
    main()
