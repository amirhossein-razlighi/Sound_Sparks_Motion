#!/bin/bash
# ab_sheet.sh <out.png> <video1.mp4> [<video2.mp4> ...]: one row per video, N frames evenly spaced (login-node safe).
# env N (frames per row, default 10), W (frame width, default 288), START/END frame range (default whole clip),
# CROP=w:h:x:y (ffmpeg crop applied before scaling, e.g. to zoom on a face).
out=$1; shift; N=${N:-10}; W=${W:-288}
args=(); filt=""; i=0
for v in "$@"; do
  nf=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$v")
  s=${START:-0}; e=${END:-$((nf-1))}; step=$(( (e - s) / (N - 1) )); [ $step -lt 1 ] && step=1
  cropf=""; [ -n "${CROP:-}" ] && cropf="crop=$CROP,"
  args+=(-i "$v"); filt="$filt[$i:v]select='between(n\,$s\,$e)*not(mod(n-$s\,$step))',${cropf}scale=$W:-2,tile=${N}x1[r$i];"; i=$((i+1))
done
rows=""; for ((k=0;k<i;k++)); do rows="$rows[r$k]"; done
if [ $i -eq 1 ]; then stack="[r0]copy"; else stack="${rows}vstack=inputs=$i"; fi   # vstack needs >= 2 inputs
ffmpeg -y -v error -threads 1 "${args[@]}" -filter_complex "$filt$stack" -frames:v 1 -q:v 4 -threads 1 "$out"   # .png or .jpg (jpg is ~10x smaller)
