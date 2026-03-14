from __future__ import annotations

# Reuse battle-tested metric functions from the legacy script.
from editing.optimize_audio_embedding import (  # noqa: F401
    compute_raft_flows,
    decode_video_frames_rgb,
    flatten_video_chunks,
    frames_rgb_uint8_to_chw_float,
    load_raft_components,
)
