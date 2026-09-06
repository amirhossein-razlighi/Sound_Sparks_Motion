#!/bin/bash
# Tar exactly the user-approved dirs (data preserved as .tar, originals removed
# as archived via --remove-files, freeing scratch FILE-COUNT quota):
#   seeing_sound_baselines/av_hubert  -> seeing_sound_baselines/av_hubert.tar
#   seeing_sound_baselines/vallr      -> seeing_sound_baselines/vallr.tar
#   video_time_manip/venv             -> video_time_manip/venv.tar
# PREREQ: the approved deletes (comfy-ltx/harness_results, comfy-ltx/demos)
# must have freed some quota headroom first, else tar creation fails.
#
#   sbatch h3_probe/tar_approved.sbatch
#SBATCH --job-name=tar-approved
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/tar_approved_%j.out
#SBATCH --error=slurm_logs/tar_approved_%j.err
#SBATCH --account=def-amahdavi
set -uo pipefail

tar_one() {
    local parent="$1" name="$2"
    local out="${parent}/${name}.tar"
    if [[ ! -d "${parent}/${name}" ]]; then echo "[skip] ${parent}/${name} missing"; return; fi
    if [[ -f "${out}" ]]; then echo "[skip] ${out} exists"; return; fi
    echo "[tar] ${parent}/${name} ..."
    tar -cf "${out}" --remove-files -C "${parent}" "${name}" \
        && echo "[done] ${out}" \
        || echo "[WARN] tar errors for ${name} (may be partially archived)"
}

tar_one /scratch/amirrz/seeing_sound_baselines av_hubert
tar_one /scratch/amirrz/seeing_sound_baselines vallr
tar_one /scratch/amirrz/video_time_manip venv

echo "TAR DONE"
lfs quota -u amirrz /scratch | sed -n 3p
