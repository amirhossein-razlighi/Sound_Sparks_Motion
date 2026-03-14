from __future__ import annotations

# Reuse battle-tested helpers from the legacy script.
from editing.optimize_audio_embedding import (  # noqa: F401
    align_waveform_length,
    compute_target_shape,
    extract_first_frame_png,
    save_audio_wav,
    write_temp_video_with_audio,
)
