#!/usr/bin/env python3
"""P5 setup: target list -> transfer YAMLs (same template as editing/configs/transfer/*.yaml)
plus audio muxing for silent targets. CPU only; writes under results/iclr_rebuttal/transfer/.

Silent targets: real_* film clips get their own extracted wav; the other silent clips get a
duration-matched silent track (as H3 does for silent clips). This only affects the prompt-only
baseline: the transfer replaces the audio latent entirely with the learned one.
"""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import yaml

from common import OUT_ROOT, PAPER_RESULTS, REPO_ROOT

IV = REPO_ROOT / "input_videos"
H3 = Path("/scratch/amirrz/H3_exp/inputs")
LATENTS = {  # learned latent -> paper run (mode_both holds best_audio_latent_both.pt / best_text_delta_both.pt)
    "dog_yawning": "a_dog_yawning/linspace_w_lpips/both",
    "falcon_bird_opening_wings": "a_bird_openning_its_wings./linspace_w_lpips/both",
    "groom_angry_shouting": "a_angry_man_is_shouting./normal_wo_lpips/both",  # = latent of the paper's man-shouting transfer
    "groom_laughing": "a_man_is_laughing_outloud/normal_wo_lpips/both",
    "man_pets_dog": "the_man_pets_the_dog./linspace_w_lpips/both",
}
# Paper transfers that already exist (finalized; reused, NOT rerun). transfer_both.mp4 + baseline_video.mp4 in each dir.
PAPER_XFER = {
    ("dog_yawning", "cat_sunglasses"): PAPER_RESULTS / "transfer/a_cat_yawning./linspace_w_lpips",
    ("falcon_bird_opening_wings", "parrot"): PAPER_RESULTS / "transfer/a_red_bird_openning_its/linspace_w_lpips",
    ("groom_angry_shouting", "person_room"): PAPER_RESULTS / "transfer/a_man_shouting./normal_wo_lpips",
    ("man_pets_dog", "man_dog_park"): PAPER_RESULTS / "transfer/the_man_pets_the_dog./linspace_w_lpips",
}
# (latent, target_id, video, edit_prompt, static_prompt, subject_category_match, style, note)
T = [
    ("dog_yawning", "cat_sunglasses", IV / "a_cat_with_sunglasses_sitting_on_a.mp4", "A cat yawning.", "A cat is sitting on a chair.", "animal-other", "gen", "paper transfer"),
    ("dog_yawning", "cat_ground", IV / "a_cat_sitting_on_the_ground_near.mp4", "A cat yawning.", "A cat sitting on the floor.", "animal-other", "gen", ""),
    ("dog_yawning", "dog_real", IV / "dog_sitting_real.mp4", "A dog yawning.", "A small dog sitting on the floor.", "same", "real", ""),
    ("dog_yawning", "dog_rug_h3", H3 / "gen/gen_dog_rug__t2va_av.mp4", "A dog yawning.", "A dog lying on a rug.", "same", "gen-H3", ""),
    ("dog_yawning", "koala", IV / "a_koala_sitting_on_a_tree_trunk.mp4", "A koala yawning.", "A koala sitting on a tree trunk.", "animal-other", "gen", ""),
    ("dog_yawning", "lion_h3", H3 / "gen/gen_lion_rock__t2va_av.mp4", "A lion yawning.", "A lion lying on a rock.", "animal-other", "gen-H3", ""),
    ("dog_yawning", "old_man", IV / "an_old_man_sitting_on_a_bench.mp4", "An old man yawning.", "An old man sitting on a bench.", "human", "gen", "cross-category"),
    ("falcon_bird_opening_wings", "parrot", IV / "a_red_parrot_sitting_on_a_tree.mp4", "A red bird opening its wings.", "A red bird sitting on a branch.", "animal-other", "gen", "paper transfer"),
    ("falcon_bird_opening_wings", "eagle_cliff", IV / "an_eagle_perched_on_a_cliff_edge.mp4", "An eagle opening its wings.", "An eagle perched on a cliff.", "same", "gen", ""),
    ("falcon_bird_opening_wings", "eagle_real", IV / "an_eagle_real.mp4", "An eagle opening its wings.", "An eagle perched on a branch.", "same", "real", "silent->silence"),
    ("falcon_bird_opening_wings", "rooster_h3", H3 / "gen/gen_rooster__t2va_av.mp4", "A rooster opening its wings.", "A rooster standing on a fence.", "animal-other", "gen-H3", ""),
    ("falcon_bird_opening_wings", "duck_h3", H3 / "gen/gen_duck_pond__t2va_av.mp4", "A duck opening its wings.", "A duck floating on a pond.", "animal-other", "gen-H3", ""),
    ("falcon_bird_opening_wings", "monkey", IV / "a_monkey_sitting_on_a_tree_branch.mp4", "A monkey opening its arms wide.", "A monkey sitting on a tree branch.", "mismatch", "gen", "cross-category"),
    ("groom_angry_shouting", "person_room", IV / "a_person_standing_up_in_front_of.mp4", "A man shouting.", "A man is standing.", "same", "gen", "paper transfer"),
    ("groom_angry_shouting", "man_real", IV / "man_looking_at_camera_real.mp4", "A man shouting.", "A man looking at the camera.", "same", "real", "silent->silence"),
    ("groom_angry_shouting", "michael_film", H3 / "real/real_michael_mug__real_89f.mp4", "A man shouting.", "A man holding a mug in an office.", "same", "real-film", "own wav"),
    ("groom_angry_shouting", "man_desk_h3", H3 / "gen/gen_man_desk__t2va_av.mp4", "A man shouting.", "A man sitting at a desk.", "same", "gen-H3", ""),
    ("groom_angry_shouting", "cartoon_boy", IV / "a_cartoonish_video_of_a_man_young.mp4", "The boy is shouting.", "A boy standing in front of a building.", "same", "cartoon", ""),
    ("groom_angry_shouting", "woman_desk_h3", H3 / "gen/gen_woman_desk__t2va_av.mp4", "A woman shouting.", "A woman sitting at a desk with a laptop.", "human-other", "gen-H3", ""),
    ("groom_angry_shouting", "lion_h3", H3 / "gen/gen_lion_rock__t2va_av.mp4", "A lion roaring.", "A lion lying on a rock.", "mismatch", "gen-H3", "cross-category"),
    ("groom_laughing", "cartoon_boy", IV / "a_cartoonish_video_of_a_man_young.mp4", "The boy is laughing out loud.", "A boy standing in front of a building.", "same", "cartoon", "paper transfer"),
    ("groom_laughing", "cartoon_boy2", IV / "cartoonish_2.mp4", "The boy is laughing out loud.", "A boy standing in front of a house.", "same", "cartoon", "silent->silence"),
    ("groom_laughing", "man_real", IV / "man_looking_at_camera_real.mp4", "A man is laughing out loud.", "A man looking at the camera.", "same", "real", "silent->silence"),
    ("groom_laughing", "person_room", IV / "a_person_standing_up_in_front_of.mp4", "A man is laughing out loud.", "A man is standing.", "same", "gen", ""),
    ("groom_laughing", "old_man", IV / "an_old_man_sitting_on_a_bench.mp4", "An old man is laughing out loud.", "An old man sitting on a bench.", "same", "gen", ""),
    ("groom_laughing", "child_real", IV / "a_child_real.mp4", "A child is laughing out loud.", "A child looking to the side.", "human-other", "real", "silent->silence"),
    ("groom_laughing", "keaton_film", H3 / "real/real_keaton__real_89f.mp4", "A man is laughing out loud.", "A man in a hat looking down.", "same", "real-film-bw", "own wav"),
    ("man_pets_dog", "man_dog_park", IV / "a_man_and_a_dog_in_front.mp4", "The man pets the dog.", "A man and a dog in front of a bench.", "same", "gen", "paper transfer"),
    ("man_pets_dog", "dog_chair", IV / "dog_sitting_on_chair.mp4", "A man pets the dog.", "A dog sitting on a chair.", "partial (no person)", "gen", ""),
    ("man_pets_dog", "dog_rug_h3", H3 / "gen/gen_dog_rug__t2va_av.mp4", "A man pets the dog.", "A dog lying on a rug.", "partial (no person)", "gen-H3", ""),
    ("man_pets_dog", "cat_sunglasses", IV / "a_cat_with_sunglasses_sitting_on_a.mp4", "A man pets the cat.", "A cat is sitting on a chair.", "partial (no person)", "gen", ""),
    ("man_pets_dog", "dog_real", IV / "dog_sitting_real.mp4", "A man pets the dog.", "A small dog sitting on the floor.", "partial (no person)", "real", ""),
    ("man_pets_dog", "starwars", IV / "starwars.mp4", "The man gently strokes the woman's hair.", "A man and a woman sitting in a meadow.", "mismatch", "real-film", "cross-category"),
]


def has_audio(p: Path) -> bool:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True).stdout
    return "audio" in out


def main():
    root = OUT_ROOT / "transfer"
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    tmpl = yaml.safe_load(open(REPO_ROOT / "editing/configs/transfer/dog_to_cat_yawning.yaml"))
    rows = []
    for lat, tid, vid, ep, sp, cat, style, note in T:
        assert vid.exists(), vid
        src = vid
        if not has_audio(vid):
            src = root / "inputs" / f"{vid.stem}_av.mp4"
            if not src.exists():
                wav = vid.parent / vid.name.replace("__real_89f.mp4", "_audio_89f.wav")
                if wav.exists():
                    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(vid), "-i", str(wav), "-c:v", "copy", "-c:a", "aac", "-shortest", str(src)]
                else:
                    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(vid), "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                           "-c:v", "copy", "-c:a", "aac", "-shortest", str(src)]
                subprocess.run(cmd, check=True)
        cfg = dict(tmpl)
        cfg.update(target_video=str(src), opt_dir=str(PAPER_RESULTS / LATENTS[lat] / "mode_both"), edit_prompt=ep,
                   static_prompt=sp, experiment_name=f"{lat}__{tid}")
        cp = root / "configs" / f"{lat}__{tid}.yaml"
        yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
        src_prompt = __import__("json").loads((PAPER_RESULTS / LATENTS[lat] / "run_config.json").read_text())["args"]["edit_prompt"]
        rows.append(dict(latent=lat, target=tid, config=str(cp), target_video=str(src), orig_video=str(vid),
                         source_prompt=src_prompt, edit_prompt=ep, static_prompt=sp, subject_category_match=cat,
                         style=style, note=note,
                         out_dir=str(PAPER_XFER.get((lat, tid), root / "runs" / lat / tid)),
                         reused_paper_output=int((lat, tid) in PAPER_XFER)))
    with open(root / "targets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(len(rows), "targets ->", root / "targets.csv")


if __name__ == "__main__":
    main()
