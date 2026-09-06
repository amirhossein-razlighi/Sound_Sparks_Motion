#!/bin/bash
# Mirror nopin-vs-pin_h3snd pairs (with sound) into h3_probe/results/pin_pairs/<slug>/
cd /home/amirrz/my_codes/Sound_Sparks_Motion
D=h3_probe/results/pin_pairs; mkdir -p $D
for O in /scratch/amirrz/H3_exp/outputs/pin_test /scratch/amirrz/H3_exp/outputs/pin_test_h3snd /scratch/amirrz/H3_exp/outputs/pin_all /scratch/amirrz/H3_exp/outputs/pin_all_b /scratch/amirrz/H3_exp/outputs/pin_all_ref /scratch/amirrz/H3_exp/outputs/pin_all_ref_b; do
  for f in $O/*__edit__nopin_av.mp4; do [ -f "$f" ] || continue; s=$(basename $f __edit__nopin_av.mp4); mkdir -p $D/$s; cp -u $f $D/$s/baseline_av.mp4; done
  for f in $O/*__edit__pin_h3snd_av.mp4; do [ -f "$f" ] || continue; s=$(basename $f __edit__pin_h3snd_av.mp4); mkdir -p $D/$s; cp -u $f $D/$s/pinned_h3sound_av.mp4; done
  for f in $O/*__edit__ref_h3snd_av.mp4; do [ -f "$f" ] || continue; s=$(basename $f __edit__ref_h3snd_av.mp4); mkdir -p $D/$s; cp -u $f $D/$s/audioref_h3sound_av.mp4; done
  for f in $O/*__edit__refkeep_h3snd_av.mp4; do [ -f "$f" ] || continue; s=$(basename $f __edit__refkeep_h3snd_av.mp4); mkdir -p $D/$s; cp -u $f $D/$s/audioref_keep_h3sound_av.mp4; done
done
for O in /scratch/amirrz/H3_exp/outputs/t2va_sounds /scratch/amirrz/H3_exp/outputs/t2va_all; do
  for f in $O/*__t2va_av.mp4; do [ -f "$f" ] || continue; s=$(basename $f __t2va_av.mp4); [ -d $D/$s ] && cp -u $f $D/$s/event_sound_source_t2va_av.mp4; done
done
python3 - <<'PY'
import json, os
D="h3_probe/results/pin_pairs"; R={}
for O in ["pin_test","pin_test_h3snd","pin_all","pin_all_b","pin_all_ref","pin_all_ref_b"]:
    p=f"/scratch/amirrz/H3_exp/outputs/{O}/results.json"
    if os.path.exists(p):
        for k,v in json.load(open(p)).items():
            if "yes" in v and v["arm"] in ("edit__nopin","edit__pin_h3snd","edit__ref_h3snd","edit__refkeep_h3snd"): R.setdefault(v["slug"],{})[v["arm"]]=v["yes"]
with open(f"{D}/INDEX.md","w") as f:
    f.write("# H3 event sound: baseline vs pinned vs plain audio-reference (edit text identical)\n\nPer folder: `baseline_av.mp4` (H3 generates its own soundtrack, source audio as reference), `pinned_h3sound_av.mp4` (generated audio latent pinned to H3's own sound of the event), `audioref_h3sound_av.mp4` (NO tricks: the event sound is simply the audio reference, grammar 'reference'), `audioref_keep_h3sound_av.mp4` (same, grammar 'fully_preserved' + sync instruction), `event_sound_source_t2va_av.mp4` (the text-only clip the sound came from).\n\n| scenario | baseline | pinned | audio-ref | audio-ref keep |\n|---|---|---|---|---|\n")
    for s in sorted(os.listdir(D)):
        if os.path.isdir(f"{D}/{s}"):
            a=R.get(s,{}); g=lambda k: a.get(k,float('nan'))
            f.write(f"| {s} | {g('edit__nopin'):.4f} | {g('edit__pin_h3snd'):.4f} | {g('edit__ref_h3snd'):.4f} | {g('edit__refkeep_h3snd'):.4f} |\n")
print("pairs:", len([s for s in os.listdir(D) if os.path.isdir(f'{D}/{s}')]))
PY
