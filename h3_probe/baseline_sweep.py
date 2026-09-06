#!/usr/bin/env python3
"""Hunt for edits H3's text baseline FAILS on.

Loads the H3 ref2va pipeline + our Qwen critic once, then for every candidate
(the untested paper-benchmark scenarios + a set of deliberately counter-prior
edits on known sources) renders the grammar-aligned [video editing] baseline
(16 steps, seed 42, video + source-audio references) and scores it with the
motion critic. Writes <OUT>/<slug>.mp4 and <OUT>/sweep_results.json (sorted by
critic yes-prob, lowest = most likely failures) for visual verification.

    /scratch/amirrz/H3_exp/venv/bin/python h3_probe/baseline_sweep.py
"""
import json
import logging
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(_HERE + "/../editing/src"))
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("sweep")

CKPT = "/scratch/amirrz/H3_exp/ckpt"
OUT = "/scratch/amirrz/H3_exp/outputs/baseline_sweep"
QWEN = "/project/def-amahdavi/amirrz/HF/models/Qwen2.5-VL-7B-Instruct"
RV = "/home/amirrz/my_codes/Sound_Sparks_Motion/results/videos"
IV = "/home/amirrz/my_codes/Sound_Sparks_Motion/input_videos"
SW = "/scratch/amirrz/H3_exp/inputs/sweep"
I = "/scratch/amirrz/H3_exp/inputs"
STEPS, SEED, NUM_FRAMES, HEIGHT, WIDTH = 16, 42, 124, 448, 768

# (slug, source video, source wav, edit sentence, one-line scene description)
PAPER = [
    ("bird_opens_wing_real", f"{RV}/bird_opens_wing_real/retake_input_prepared.mp4", f"{SW}/bird_opens_wing_real.wav",
     "An eagle opens its wings and takes off flying.", "an eagle perched, seen against the sky"),
    ("boy_crouches", f"{RV}/boy_crouches/retake_input_prepared.mp4", f"{SW}/boy_crouches.wav",
     "The boy crouches slightly and touches the water.", "a boy standing near water"),
    ("child_waving_hand", f"{RV}/child_waving_hand/retake_input_prepared.mp4", f"{SW}/child_waving_hand.wav",
     "The child waves his hand.", "a child facing the camera"),
    ("goldfish", f"{RV}/goldfish/retake_input_prepared.mp4", f"{SW}/goldfish.wav",
     "The goldfish jumps out of the fish tank into the air.", "a goldfish in a fish tank"),
    ("man_laugh", f"{RV}/man_laugh/retake_input_prepared.mp4", f"{SW}/man_laugh.wav",
     "The man laughs out loud.", "a man facing the camera"),
    ("man_raises_hand_real", f"{RV}/man_raises_hand_real/retake_input_prepared.mp4", f"{SW}/man_raises_hand_real.wav",
     "The man raises his hand up and holds it there.", "a man standing, facing the camera"),
    ("monkey_reaching_for_fruit", f"{RV}/monkey_reaching_for_fruit/retake_input_prepared.mp4", f"{SW}/monkey_reaching_for_fruit.wav",
     "The monkey reaches out for a fruit.", "a monkey near fruit"),
    ("red_car_door_real", f"{RV}/red_car_door_real/retake_input_prepared.mp4", f"{SW}/red_car_door_real.wav",
     "The car door opens.", "a parked red car"),
    ("robot_waives", f"{RV}/robot_waives/retake_input_prepared.mp4", f"{SW}/robot_waives.wav",
     "The robot waves its hand.", "a humanoid robot facing the camera"),
    ("surprised_man", f"{RV}/surprised_man/retake_input_prepared.mp4", f"{SW}/surprised_man.wav",
     "The man's face becomes surprised: eyes widen, mouth opens.", "a man facing the camera"),
    ("turtle_extends_neck", f"{RV}/turtle_extends_neck/retake_input_prepared.mp4", f"{SW}/turtle_extends_neck.wav",
     "The turtle extends its neck forward out of its shell.", "a turtle"),
    ("boy_laughing_outloud", f"{RV}/transfers/boy_laughing_outloud/retake_input_prepared.mp4", f"{SW}/boy_laughing_outloud.wav",
     "The boy laughs out loud.", "a boy facing the camera"),
    ("cat_yawns", f"{RV}/transfers/cat_yawns/retake_input_prepared.mp4", f"{SW}/cat_yawns.wav",
     "The cat yawns widely.", "a cat sitting"),
    ("man_shouts", f"{RV}/transfers/man_shouts/retake_input_prepared.mp4", f"{SW}/man_shouts.wav",
     "The man shouts loudly.", "a man facing the camera"),
    ("red_bird_opens_wings", f"{RV}/transfers/red_bird_opens_wings/retake_input_prepared.mp4", f"{SW}/red_bird_opens_wings.wav",
     "The red bird opens its wings.", "a red bird perched"),
]
# Deliberately counter-prior / precise / compositional edits on known sources.
EXOTIC = [
    ("x_ferrari_wheelie", f"{IV}/a_red_ferrari_standing_still_in_the_main.mp4", f"{I}/cardoor_source.wav",
     "The red car lifts its front wheels off the ground, rearing up onto its rear wheels, then drops back down.",
     "a glossy red Ferrari parked front-on in a white studio"),
    ("x_rose_floats", f"{IV}/a_red_rose_bud_in_a_green_main.mp4", f"{I}/rose_source.wav",
     "The rose bud detaches from its stem and floats straight up into the air.",
     "a closed red rose bud on a stem in a grass field"),
    ("x_dog_backwards", f"{IV}/dog_sitting_on_chair_main.mp4", f"{I}/dog_source.wav",
     "The dog stands up on its hind legs and walks backwards off the chair.",
     "a golden retriever in sunglasses sitting on a wooden garden chair"),
    ("x_falcon_head_turn", f"{IV}/a_falcon_bird_sitting_on_a_tree_main.mp4", f"{I}/falcon_source.wav",
     "The bird turns its head fully around to look directly behind itself, then turns it back.",
     "a brown-and-white raptor perched on a tree branch"),
    ("x_groom_backflip", f"{IV}/a_groom_standing_in_the_middle_of_main.mp4", f"{I}/groom_source.wav",
     "The man does a full backflip and lands back on his feet.",
     "a smiling groom in a black suit under a white pergola"),
    ("x_dog_in_pan", f"{IV}/cook_spinach_main.mp4", f"{I}/mpd_source.wav",
     "The white dog climbs up onto the counter and sits down inside the frying pan.",
     "a man in an apron cooking at a kitchen counter with a white dog sitting on the left"),
]


def grammar_prompt(edit: str, scene: str) -> str:
    e = edit.rstrip(".")
    return f"""subject_definitions:
<Video 1> is the source video to edit: a fixed-camera shot of {scene}.
<Subject 1> is the main subject of <Video 1>: {scene}.
<Audio 2> is the soundtrack accompanying the target video.

summary:
[video editing] The target video is an edited version of <Video 1>. The camera position, framing, scene, lighting, and pacing of <Video 1> are kept identical. The only change: {e}.

retention_analysis:
<Audio 2>: reference - the target video's sound follows the content and timing of <Audio 2>; on-screen actions stay synchronized with it.
<Video 1> (camera, framing, scene layout, lighting, and temporal structure): fully_preserved - the fixed camera, background, lighting, and overall pacing are kept identical to the source.
<Subject 1> (appears throughout): partially_preserved - appearance and position are identical to <Video 1>; only its action changes: {e}.

detailed_description:
[Shot 1] Fixed camera, framing identical to <Video 1>. The scene of <Video 1>: {scene}. Within the first second, {e}; the motion is clear and fully visible, and continues naturally through the shot. Everything else - the background, the lighting, the framing - remains exactly as in <Video 1>.

overall_soundscape: The soundscape follows <Audio 2>, with the natural sounds of the described action.

non_diegetic_music: None."""


def main():
    os.makedirs(OUT, exist_ok=True)
    from diffusers import ModularPipeline
    from diffusers.modular_pipelines import ComponentsManager
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3AudioReference, MiniMaxH3VideoReference
    from diffusers.utils import export_to_video
    from motion_opt.qwen_loss import build_qwen_model, build_qwen_rubric_inputs, compute_qwen_video_loss

    cm = ComponentsManager()
    cm.enable_auto_cpu_offload(device="cuda")
    pipe = ModularPipeline.from_pretrained(CKPT, workflow="ref2va", components_manager=cm)
    pipe.load_components(torch_dtype=torch.bfloat16)
    dev = torch.device("cuda")
    qwen, proc = build_qwen_model(QWEN, device=dev, gradient_checkpointing=False)
    qwen.to("cpu")
    torch.cuda.empty_cache()
    windows = [("linspace", 0), ("contiguous", 0), ("contiguous", 50), ("contiguous", 100)]
    ov = {"motion": 1.0, "entities": 0.0, "overall": 0.0}

    results_path = os.path.join(OUT, "sweep_results.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}

    for slug, src, wav, edit, scene in PAPER + EXOTIC:
        if slug in results and os.path.exists(os.path.join(OUT, f"{slug}.mp4")):
            log.info("[skip] %s", slug)
            continue
        t0 = time.time()
        try:
            refs = [MiniMaxH3VideoReference.from_file(src), MiniMaxH3AudioReference.from_file(wav)]
            state = pipe(prompt=grammar_prompt(edit, scene), references=refs, num_frames=NUM_FRAMES,
                         height=HEIGHT, width=WIDTH, num_inference_steps=STEPS,
                         generator=torch.Generator("cuda").manual_seed(SEED), output_type="np")
            vids = state.values.get("videos")
            vid = np.asarray(vids[0] if isinstance(vids, (list, tuple)) else vids)
            if vid.ndim == 5:
                vid = vid[0]
            export_to_video([np.asarray(f) for f in vid], os.path.join(OUT, f"{slug}.mp4"), fps=24)

            ci, yes_id, no_id = build_qwen_rubric_inputs(processor=proc, edit_prompt=edit, num_frames=24,
                                                         img_size=224, device=dev, motion_question=None,
                                                         gradient_rubric="motion")
            fr = torch.from_numpy(np.ascontiguousarray(vid)).permute(0, 3, 1, 2).float().clamp(0, 1).to(dev)
            qwen.to(dev)
            acc = 0.0
            with torch.no_grad():
                for mode, start in windows:
                    loss, _ = compute_qwen_video_loss(frames_chw=fr, qwen_model=qwen, cached_inputs=ci,
                                                      yes_token_id=yes_id, no_token_id=no_id, max_frames=24,
                                                      img_size=224, backward=False, return_details=True,
                                                      sample_mode=mode, contiguous_start_frame=start,
                                                      rubric_weight_overrides=ov)
                    acc += float(loss)
            qwen.to("cpu")
            torch.cuda.empty_cache()
            nll = acc / len(windows)
            results[slug] = {"edit": edit, "src": src, "wav": wav, "scene": scene,
                             "nll": nll, "yes": math.exp(-nll), "sec": round(time.time() - t0)}
            log.info("RESULT %-28s yes=%.4f nll=%.3f (%.0fs)  %s", slug, math.exp(-nll), nll, time.time() - t0, edit)
        except Exception as e:  # keep sweeping
            log.exception("FAILED %s: %s", slug, e)
            results[slug] = {"edit": edit, "error": str(e)[:300]}
        json.dump(results, open(results_path, "w"), indent=2)

    ranked = sorted((v.get("yes", 9), k) for k, v in results.items())
    log.info("=== ranked by critic yes (lowest first) ===")
    for y, k in ranked:
        log.info("  %.4f  %s", y, k)
    log.info("SWEEP DONE")


if __name__ == "__main__":
    main()
