#!/bin/bash
#SBATCH --job-name=ltx_T2V
#SBATCH --account=def-amahdavi
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load opencv cuda/12.9

source .venv/bin/activate

python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path /project/def-amahdavi/amirrz/LTX-2/checkpoints/ltx-2.3-22b-dev.safetensors \
    --distilled-lora /project/def-amahdavi/amirrz/LTX-2/checkpoints/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
    --spatial-upsampler-path /project/def-amahdavi/amirrz/LTX-2/checkpoints/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
    --gemma-root /project/def-amahdavi/amirrz/HF/models/gemma-3-12b-it-qat-q4_0-unquantized/ \
    --prompt "A person playing guitar on stage" \
    --output-path output_guitar_playing.mp4
