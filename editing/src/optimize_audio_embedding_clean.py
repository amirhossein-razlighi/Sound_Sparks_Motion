#!/usr/bin/env python3
"""Clean modular entrypoint for audio-latent optimization.

This leaves `editing/optimize_audio_embedding.py` unchanged and provides
an equivalent multi-file structure under `editing/src/`.
"""

from audio_latent_opt.main import main


if __name__ == "__main__":
    main()
