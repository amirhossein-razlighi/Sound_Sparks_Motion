#!/bin/bash
# Free scratch FILE-COUNT quota by tarring /scratch/amirrz/seeing_sound
# (1.03M files -> a handful of .tar archives; DATA IS PRESERVED, block space
# is unaffected). Uses tar --remove-files so the file count decreases as it
# archives. CPU-only job; metadata-heavy on Lustre, hence the long time limit.
#
#   PREREQ: a little quota headroom must exist first (delete something small,
#   e.g. /scratch/amirrz/tmp), else the first tar file cannot be created.
#
#   sbatch h3_probe/tar_seeing_sound.sbatch
#SBATCH --job-name=tar-seeing-sound
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/tar_ss_%j.out
#SBATCH --error=slurm_logs/tar_ss_%j.err
#SBATCH --account=def-amahdavi
set -uo pipefail

SRC="/scratch/amirrz/seeing_sound"
DST="/scratch/amirrz/seeing_sound_tars"
mkdir -p "${DST}"

for d in "${SRC}"/*/; do
    name="$(basename "${d}")"
    out="${DST}/${name}.tar"
    if [[ -f "${out}" ]]; then echo "[skip] ${out} exists"; continue; fi
    echo "[tar] ${name} ..."
    tar -cf "${out}" --remove-files -C "${SRC}" "${name}" \
        && echo "[done] ${name}" \
        || echo "[WARN] tar reported errors for ${name} (originals may be partially removed)"
done
# loose files at the top level, if any
if compgen -G "${SRC}/*" >/dev/null && ! compgen -G "${SRC}/*/" >/dev/null; then
    tar -cf "${DST}/_toplevel_files.tar" --remove-files -C "${SRC}" . || true
fi
echo "TAR DONE"; lfs quota -u amirrz /scratch | head -3
