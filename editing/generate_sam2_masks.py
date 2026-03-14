#!/usr/bin/env python3
"""
Generate object mask videos for source/target clips using SAM2 + a detector seed.

This script detects an object class (e.g., "dog") per frame with a COCO detector,
then refines each box into a segmentation mask with SAM2 image predictor.
It writes two binary mask videos:
- source mask video
- target mask video

Outputs are intended for ROI-masked flow optimization in optimize_audio_embedding.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor

log = logging.getLogger(__name__)


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int


def decode_rgb_frames(video_path: str, max_frames: int | None, frame_stride: int) -> tuple[list[np.ndarray], VideoMeta]:
    frames: list[np.ndarray] = []
    src = av.open(video_path)
    try:
        vs = next(s for s in src.streams if s.type == "video")
        fps = float(vs.average_rate) if vs.average_rate is not None else 24.0
        width = int(vs.width)
        height = int(vs.height)

        frame_idx = 0
        for frame in src.decode(vs):
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue
            rgb = np.asarray(frame.to_image().convert("RGB"), dtype=np.uint8)
            frames.append(rgb)
            frame_idx += 1
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        src.close()

    return frames, VideoMeta(fps=fps, width=width, height=height)


def write_mask_video(mask_frames: list[np.ndarray], out_path: str, fps: float, width: int, height: int) -> None:
    dst = av.open(out_path, mode="w")
    try:
        vs_out = dst.add_stream("libx264", rate=int(round(fps)))
        vs_out.width = width
        vs_out.height = height
        vs_out.pix_fmt = "yuv420p"
        vs_out.options = {"crf": "18", "preset": "veryfast"}

        from fractions import Fraction as _Fraction

        time_base = _Fraction(1, int(round(fps)))

        for idx, mask in enumerate(mask_frames):
            gray = (mask.astype(np.uint8) * 255)
            rgb = np.stack([gray, gray, gray], axis=-1)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame = frame.reformat(width=width, height=height, format="yuv420p")
            frame.pts = idx
            frame.time_base = time_base
            for pkt in vs_out.encode(frame):
                dst.mux(pkt)

        for pkt in vs_out.encode():
            dst.mux(pkt)
    finally:
        dst.close()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "mask"


def select_prompt_category(prompt: str, categories: list[str]) -> tuple[int, str]:
    p = prompt.strip().lower()

    alias_map = {
        "person": ["person", "man", "woman", "human"],
        "dog": ["dog", "puppy"],
        "cat": ["cat", "kitten"],
        "horse": ["horse"],
        "bear": ["bear"],
        "bird": ["bird"],
        "car": ["car", "automobile"],
        "bicycle": ["bicycle", "bike"],
        "motorcycle": ["motorcycle", "motorbike"],
        "bus": ["bus"],
        "truck": ["truck"],
    }

    for canonical, aliases in alias_map.items():
        if p in aliases and canonical in categories:
            return categories.index(canonical), canonical

    if p in categories:
        return categories.index(p), p

    for i, name in enumerate(categories):
        if p in name or name in p:
            return i, name

    raise ValueError(
        f"Could not map object prompt '{prompt}' to a COCO class. "
        f"Try one of: {', '.join(categories[:20])}, ..."
    )


def build_detector(device: torch.device):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()
    categories = list(weights.meta["categories"])
    return model, categories


def build_sam2_predictor(config_path: str, checkpoint_path: str, device: torch.device):
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise RuntimeError(
            "SAM2 is not installed. Install facebookresearch/sam2 and its deps in this environment."
        ) from exc

    sam2_model = build_sam2(config_path, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor


def detect_box_for_class(
    frame_rgb: np.ndarray,
    detector: torch.nn.Module,
    class_idx: int,
    score_thresh: float,
    device: torch.device,
    prev_box: np.ndarray | None,
) -> np.ndarray | None:
    inp = to_tensor(frame_rgb).to(device)
    with torch.no_grad():
        out = detector([inp])[0]

    labels = out["labels"].detach().cpu().numpy()
    scores = out["scores"].detach().cpu().numpy()
    boxes = out["boxes"].detach().cpu().numpy()

    valid = np.where((labels == class_idx) & (scores >= score_thresh))[0]
    if len(valid) == 0:
        return prev_box

    best = valid[np.argmax(scores[valid])]
    return boxes[best]


def predict_mask_from_box(frame_rgb: np.ndarray, box_xyxy: np.ndarray, predictor) -> np.ndarray:
    predictor.set_image(frame_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box_xyxy[None, :],
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(np.uint8)


def segment_video(
    video_path: str,
    object_prompt: str,
    detector,
    categories: list[str],
    predictor,
    score_thresh: float,
    frame_stride: int,
    max_frames: int | None,
    device: torch.device,
) -> tuple[list[np.ndarray], VideoMeta]:
    frames, meta = decode_rgb_frames(video_path, max_frames=max_frames, frame_stride=frame_stride)
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    class_idx, class_name = select_prompt_category(object_prompt, categories)
    log.info("Object prompt '%s' mapped to detector class '%s'", object_prompt, class_name)

    masks: list[np.ndarray] = []
    prev_box = None
    for i, frame in enumerate(frames):
        box = detect_box_for_class(
            frame_rgb=frame,
            detector=detector,
            class_idx=class_idx,
            score_thresh=score_thresh,
            device=device,
            prev_box=prev_box,
        )
        if box is None:
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        else:
            mask = predict_mask_from_box(frame, box, predictor)
            prev_box = box
        masks.append(mask)

        if (i + 1) % 20 == 0:
            cov = float(np.mean(mask) * 100.0)
            log.info("%s frame %d/%d, last mask coverage %.2f%%", video_path, i + 1, len(frames), cov)

    return masks, meta


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-video", required=True)
    p.add_argument("--target-video", required=True)
    p.add_argument("--object-prompt", required=True, help="Object class to segment (e.g., dog)")
    p.add_argument("--sam2-config", required=True, help="SAM2 model config path")
    p.add_argument("--sam2-checkpoint", required=True, help="SAM2 checkpoint path")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--name-tag",
        type=str,
        default=None,
        help=(
            "Optional deterministic tag used in output filenames. "
            "If omitted, a tag is built from object prompt + src/target stems + frame stride."
        ),
    )
    p.add_argument("--det-score-threshold", type=float, default=0.35)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    args = build_parser().parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    src_video = str(Path(args.src_video).expanduser().resolve())
    target_video = str(Path(args.target_video).expanduser().resolve())
    sam2_cfg = str(Path(args.sam2_config).expanduser().resolve())
    sam2_ckpt = str(Path(args.sam2_checkpoint).expanduser().resolve())

    if not Path(sam2_cfg).exists():
        raise FileNotFoundError(f"SAM2 config not found: {sam2_cfg}")
    if not Path(sam2_ckpt).exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {sam2_ckpt}")

    src_stem = Path(src_video).stem
    tgt_stem = Path(target_video).stem
    auto_tag = f"{args.object_prompt}_{src_stem}_to_{tgt_stem}_fs{args.frame_stride}"
    name_tag = _slugify(args.name_tag if args.name_tag is not None else auto_tag)

    detector, categories = build_detector(device)
    predictor = build_sam2_predictor(sam2_cfg, sam2_ckpt, device)

    src_masks, src_meta = segment_video(
        video_path=src_video,
        object_prompt=args.object_prompt,
        detector=detector,
        categories=categories,
        predictor=predictor,
        score_thresh=args.det_score_threshold,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        device=device,
    )
    target_masks, target_meta = segment_video(
        video_path=target_video,
        object_prompt=args.object_prompt,
        detector=detector,
        categories=categories,
        predictor=predictor,
        score_thresh=args.det_score_threshold,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        device=device,
    )

    src_out = output_dir / f"sam2_mask_src__{name_tag}.mp4"
    tgt_out = output_dir / f"sam2_mask_target__{name_tag}.mp4"
    manifest_path = output_dir / f"sam2_mask_manifest__{name_tag}.json"

    write_mask_video(src_masks, str(src_out), fps=src_meta.fps, width=src_meta.width, height=src_meta.height)
    write_mask_video(target_masks, str(tgt_out), fps=target_meta.fps, width=target_meta.width, height=target_meta.height)

    manifest = {
        "name_tag": name_tag,
        "object_prompt": args.object_prompt,
        "source_video": src_video,
        "target_video": target_video,
        "sam2_config": sam2_cfg,
        "sam2_checkpoint": sam2_ckpt,
        "device": str(device),
        "frame_stride": args.frame_stride,
        "max_frames": args.max_frames,
        "det_score_threshold": args.det_score_threshold,
        "source_mask_video": str(src_out),
        "target_mask_video": str(tgt_out),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log.info("Saved source mask video: %s", src_out)
    log.info("Saved target mask video: %s", tgt_out)
    log.info("Saved mask manifest: %s", manifest_path)
    print(f"MASK_NAME_TAG={name_tag}")
    print(f"SRC_MASK_VIDEO={src_out}")
    print(f"TARGET_MASK_VIDEO={tgt_out}")
    print(f"MASK_MANIFEST={manifest_path}")


if __name__ == "__main__":
    main()
