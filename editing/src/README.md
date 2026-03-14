# Clean Modular Version (editing/src)

This folder contains a refactored, multi-file version of the audio-latent
optimization flow from editing/optimize_audio_embedding.py.

## Goals

- Keep the original script unchanged.
- Improve readability by separating responsibilities.
- Preserve behavior by reusing proven utility functions from the original script.

## Structure

- editing/src/audio_latent_opt/config.py: structured configs/dataclasses
- editing/src/audio_latent_opt/distributed.py: distributed setup/shutdown and barriers
- editing/src/audio_latent_opt/helpers.py: general helper functions
- editing/src/audio_latent_opt/metrics.py: optical-flow/RAFT metric utilities
- editing/src/audio_latent_opt/losses.py: objective and regularization composition
- editing/src/audio_latent_opt/models.py: guider and pipeline builder helpers
- editing/src/audio_latent_opt/data.py: runtime data containers
- editing/src/audio_latent_opt/runtime.py: preprocessing + target/flow preparation
- editing/src/audio_latent_opt/loop.py: optimization loop
- editing/src/audio_latent_opt/main.py: orchestration entrypoint
- editing/src/optimize_audio_embedding_clean.py: runnable script

## Run

From repo root:

python editing/src/optimize_audio_embedding_clean.py --help

Use the same CLI options as editing/optimize_audio_embedding.py.
