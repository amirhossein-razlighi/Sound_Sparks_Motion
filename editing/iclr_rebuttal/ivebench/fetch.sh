#!/bin/bash
# Download the 64 IVEBench short subject-motion source videos + metadata into $IVE_ROOT (login node; needs internet).
R=${IVE_ROOT:?set IVE_ROOT}; D=$(cd "$(dirname "$0")" && pwd); mkdir -p $R/src $R/meta
cp $D/subset_subject_motion_short.json $D/subset_files.txt $R/
for f in ivebenchdb_prompt_short.json ivebenchdb_name_mapping.csv; do curl -sSfL -o $R/meta/$f https://huggingface.co/datasets/Coraxor/IVEBench/resolve/main/$f; done
while read f; do [[ -s $R/src/$f ]] || curl -sSfL -o $R/src/$f "https://huggingface.co/datasets/Coraxor/IVEBench/resolve/main/ivebenchdb_mp4_short/$f" || echo "FAIL $f"; done < $R/subset_files.txt
ls $R/src | wc -l
