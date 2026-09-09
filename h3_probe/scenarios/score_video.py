#!/usr/bin/env python3
"""Score one or more videos with the Qwen2.5-VL motion critic (1 GPU, no H3).  Used by the overnight driver to
check that a generated SOURCE clip does not already contain the target motion.
env: VIDEOS (";"-separated), QUESTION, EDIT, OUT (json).  Writes {path: {yes_lin, yes_any, yes_max}}."""
import importlib.util, json, logging, math, os, sys
import numpy as np, torch
_HERE = os.path.dirname(os.path.abspath(__file__)); _PROBE = os.path.abspath(_HERE + "/.."); _REPO = os.path.abspath(_PROBE + "/..")
sys.path.insert(0, os.path.join(_REPO, "editing/src")); sys.path.insert(0, _PROBE)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"); log = logging.getLogger("score")
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
_fm = _load("h3_full_method", os.path.join(_PROBE, "h3_full_method.py"))
_cc = _load("critic_calib", os.path.join(_PROBE, "critic_calib.py"))
QWEN = os.environ.get("H3_QWEN", "/project/def-amahdavi/amirrz/HF/models/Qwen2.5-VL-7B-Instruct")
def main():
    from motion_opt.qwen_loss import build_qwen_model, build_qwen_rubric_inputs, compute_qwen_video_loss
    dev = torch.device("cuda"); qwen, proc = build_qwen_model(QWEN, device=dev, gradient_checkpointing=False); qwen.lm_head = _fm.FP32Head(qwen.lm_head)
    ov = {"motion": 1.0, "entities": 0.0, "overall": 0.0}; q = os.environ["QUESTION"]; edit = os.environ.get("EDIT", "")
    ci, yes_id, no_id = build_qwen_rubric_inputs(processor=proc, edit_prompt=edit, num_frames=24, img_size=224, device=dev, motion_question=q, gradient_rubric="motion")
    out = {}
    for vp in [v for v in os.environ["VIDEOS"].split(";") if v]:
        fr = _cc.load_frames(vp).to(dev); p = []
        with torch.no_grad():
            for mode, start in [("linspace", 0)] + [("contiguous", st) for st in (0, 20, 40, 60, 80, 100)]:
                if mode == "contiguous" and start + 24 > fr.shape[0]:
                    continue
                loss, _ = compute_qwen_video_loss(frames_chw=fr, qwen_model=qwen, cached_inputs=ci, yes_token_id=yes_id, no_token_id=no_id, max_frames=24,
                                                  img_size=224, backward=False, return_details=True, sample_mode=mode, contiguous_start_frame=start, rubric_weight_overrides=ov)
                p.append(math.exp(-float(loss)))
        out[vp] = {"yes_lin": p[0], "yes_any": 1 - float(np.prod([1 - x for x in p[1:]])) if len(p) > 1 else p[0], "yes_max": max(p)}
        log.info("SCORE %s lin=%.4f any=%.4f max=%.4f", os.path.basename(vp), out[vp]["yes_lin"], out[vp]["yes_any"], out[vp]["yes_max"])
    json.dump(out, open(os.environ["OUT"], "w"), indent=1); log.info("SCORE DONE")
if __name__ == "__main__":
    main()
