"""Shared helpers for the rebuttal experiments (additive; no main-code changes).

Key design rule: every experiment is parameterised from the *paper run's own*
``run_config.json`` (the run whose ``best_optimized_video_both.mp4`` is the
paper's "ours" video, matched by checksum), NOT from ``editing/configs/*.yaml``.
For 6/12 scenarios the YAML differs from the run that produced the paper video
(critic frames / sampling / K / passes), so the run_config is the only setting
that is guaranteed to match the paper's ours + baseline videos.

Args missing from an old run_config (flags added later, e.g.
``qwen_grad_accum_steps``) take the *current parser default*, which reproduces
the behaviour of the code that did not have the flag.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EDITING = REPO_ROOT / "editing"
sys.path.insert(0, str(EDITING / "src"))
sys.path.insert(0, str(EDITING))

import os as _os

# Overridable for other clusters (e.g. Trillium, from trillium_bundle.tar.gz); defaults = Rorqual.
PAPER_RESULTS = Path(_os.environ.get("SSM_PAPER_RESULTS", "/project/def-amahdavi/amirrz/results"))
COMPARISONS = Path(_os.environ.get("SSM_COMPARISONS", str(REPO_ROOT / "results" / "videos" / "comparisons")))
OUT_ROOT = Path(_os.environ.get("SSM_OUT_ROOT", str(REPO_ROOT / "results" / "iclr_rebuttal")))

# scenario (config name) -> (paper run dir relative to PAPER_RESULTS, comparisons/ folder)
# Matched by md5(comparisons/<cmp>/ours.mp4) == md5(<run>/mode_both/best_optimized_video_both.mp4)
# and md5(comparisons/<cmp>/baseline_ltx2.mp4) == md5(<run>/baseline_video.mp4).
SCENARIOS: dict[str, tuple[str, str]] = {
    "bugatti_lights_flash": ("the_car_front_lights_turn/sweep_exp/rsf5_accum3_modenormal_frames8_lr0.005_lpipsOFF", "car_lights"),
    "red_car_door_opens": ("a_red_car_door_opens./multi_grad/qwen_frames_8/linspace_wo_lpips", "red_car_door"),
    "groom_raising_hand": ("a_man_raising_his_right/both", "man_raises_hand"),
    "dog_yawning": ("a_dog_yawning/linspace_w_lpips/both", "dog_yawn"),
    "falcon_bird_opening_wings": ("a_bird_openning_its_wings./linspace_w_lpips/both", "bird_opens_wing"),
    "goldfish_jumping": ("a_goldfish_jumps_out_of/sweep_exp/rsf1_accum1_modenormal_frames24_lr0.005_lpips0.1_temp0.05", "goldfish"),
    "man_climbs_rocks": ("the_man_climbs_up_the/sweep_exp/rsf5_accum3_modelinspace_frames24_lr0.005_lpips0.1_temp0.05", "man_climbs_rock"),
    "man_pets_dog": ("the_man_pets_the_dog./linspace_w_lpips/both", "man_pets_dog"),
    "monkey_reaching_fruit": ("monkey_reaching_for_a_fruit./sweep_exp/rsf25_accum1_modenormal_frames24_lr0.005_lpips0.1_temp0.05", "monkey_reaching_for_fruit"),
    "red_rose_blooming": ("a_red_rose_blooming./qwen_frames_8/normal_w_lpips/both", "red_rose_bloom"),
    "robot_waving": ("the_robot_waving_its_hand./sweep_exp/rsf3_accum3_modelinspace_frames8_lr0.005_lpips0.1_temp0.05", "robot_wave"),
    "turtle_neck": ("turtle_extending_its_neck./sweep_exp/rsf25_accum1_modelinspace_frames24_lr0.005_lpips0.1_temp0.05", "turtle_neck_extend"),
}
TWELVE = list(SCENARIOS)
FIVE = ["dog_yawning", "red_car_door_opens", "turtle_neck", "man_pets_dog", "falcon_bird_opening_wings"]

# Keys that are machine/run specific and must never be copied from an old run.
_DROP_KEYS = {"output_dir", "resume", "wandb_entity", "wandb_tags", "wandb_project"}


def paper_run_dir(scenario: str) -> Path:
    return PAPER_RESULTS / SCENARIOS[scenario][0]


def paper_ours_video(scenario: str) -> Path:
    return COMPARISONS / SCENARIOS[scenario][1] / "ours.mp4"


def paper_baseline_video(scenario: str) -> Path:
    return COMPARISONS / SCENARIOS[scenario][1] / "baseline_ltx2.mp4"


def load_paper_args(scenario: str, overrides: dict | None = None) -> argparse.Namespace:
    """Namespace = current parser defaults, overwritten by the paper run's args, then overrides.

    Also applies the same post-processing as optimize_qwen_vl.main().
    """
    from optimize_qwen_vl import build_parser

    parser = build_parser()
    # Required args get placeholders; they are overwritten from the run_config.
    ns = parser.parse_args(["--src-video", "x", "--edit-prompt", "x", "--output-dir", "x"])
    old = json.loads((paper_run_dir(scenario) / "run_config.json").read_text())["args"]
    valid = set(vars(ns))
    for k, v in old.items():
        if k in valid and k not in _DROP_KEYS:
            setattr(ns, k, v)
    # Source: the paper run's own retake_input_prepared.mp4 (exactly what that
    # run consumed; the old repo at /home/amirrz/my_codes/LTX-2 no longer exists).
    # Fallback: same basename under this repo's input_videos/.
    ns.paper_src_video = ns.src_video
    prepared = paper_run_dir(scenario) / "retake_input_prepared.mp4"
    ns.src_video = str(prepared if prepared.exists() else REPO_ROOT / "input_videos" / Path(ns.src_video).name)
    apply_model_path_env(ns)
    for k, v in (overrides or {}).items():
        setattr(ns, k, v)
    # == optimize_qwen_vl.main() post-processing ==
    if ns.qwen_max_frames % 2:
        ns.qwen_max_frames += 1
    if ns.qwen_img_size % 28:
        ns.qwen_img_size = (ns.qwen_img_size // 28 + 1) * 28
    if ns.retake_num_inference_steps is None:
        ns.retake_num_inference_steps = ns.num_inference_steps
    ns.ti2v_num_inference_steps = ns.num_inference_steps
    ns.clip_max_frames = ns.qwen_max_frames
    return ns


def apply_model_path_env(ns) -> None:
    """Optional node-local copies of the (unchanged) model weights: SSM_CKPT_PATH,
    SSM_GEMMA_ROOT, SSM_QWEN_ROOT. Unset -> paper paths are kept."""
    import os
    if os.environ.get("SSM_CKPT_PATH"):
        ns.checkpoint_path = os.environ["SSM_CKPT_PATH"]
    if os.environ.get("SSM_GEMMA_ROOT"):
        ns.gemma_root = os.environ["SSM_GEMMA_ROOT"]
    if os.environ.get("SSM_QWEN_ROOT"):
        ns.qwen_model = os.environ["SSM_QWEN_ROOT"]


def args_to_cli(ns: argparse.Namespace) -> list[str]:
    """Serialise a Namespace back into optimize_qwen_vl.py CLI tokens (round-trip safe)."""
    from optimize_qwen_vl import build_parser

    parser = build_parser()
    toks: list[str] = []
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        val = getattr(ns, action.dest, None)
        flag = max((o for o in action.option_strings if not o.startswith("--no-")), key=len)
        if isinstance(action, argparse.BooleanOptionalAction):
            toks.append(flag if val else "--no-" + flag[2:])
        elif isinstance(action, argparse._StoreTrueAction):
            if val:
                toks.append(flag)
        elif action.dest == "loras":
            for pair in val or []:
                toks += [flag, *map(str, pair)]
        elif val is None:
            continue
        else:
            toks += [flag, str(val)]
    return toks


def critic_spec(ns: argparse.Namespace) -> dict:
    """The exact Qwen critic call used in the optimisation loop for this run."""
    return {
        "question": ns.qwen_motion_question,
        "edit_prompt": ns.edit_prompt,
        "max_frames": ns.qwen_max_frames,
        "img_size": ns.qwen_img_size,
        "sample_mode": ns.qwen_sample_mode,
        "contiguous_start_frame": ns.qwen_contiguous_start_frame,
        "gradient_rubric": ns.qwen_gradient_rubric,
        "passes": max(1, int(getattr(ns, "qwen_grad_accum_steps", 1) or 1)),
    }
