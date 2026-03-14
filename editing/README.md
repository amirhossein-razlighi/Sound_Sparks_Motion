# Audio-Edit → Video Experiment

Research experiment probing how **audio modifications affect LTX-2's
audio-conditioned video generation** (`A2VidPipelineTwoStage`).

## Concept

LTX-2 supports bidirectional audio-video synchronisation:

```
video changes → audio adapts   (V2A)
audio changes → video adapts   (A2V)
```

This experiment exploits the **A2V** pathway: take a source video,
extract its audio, apply a battery of perturbations, then re-generate
the video with each modified audio while anchoring to the first frame.

Comparing outputs reveals **which audio features drive visual content**:
rhythm, spectral shape, phase coherence, temporal order, pitch, etc.

---

## Files

```
editing/
  audio_perturbations.py   — 25 perturbation functions + registry
  experiment.py            — Main script (loads pipeline once)
  scripts/
    submit_all.sh          — Single SLURM job, all perturbations sequentially
    submit_array.sh        — SLURM job array, one job per perturbation
    launch_array.sh        — Helper: compute array range and call sbatch
```

---

## Quick Start

### Audio-Latent Motion Optimization (new)
Optimize only the audio latent (all LTX weights frozen) so generated video motion
matches a target motion pattern via optical-flow loss.

```bash
python editing/optimize_audio_embedding.py \
  --src-video /path/to/source.mp4 \
  --edit-prompt "A dog in the scene" \
  --target-prompt "The dog jumps energetically" \
  --output-dir ./audio_latent_opt \
  --iterations 8 \
  --lr 0.05
```

If you already have a target video, replace `--target-prompt ...` with
`--target-video /path/to/target_motion.mp4`.

If whole-frame motion loss is too noisy, you can restrict the objective to the
edited object using a binary mask video aligned with the target video:

```bash
python editing/optimize_audio_embedding.py \
  --src-video /path/to/source.mp4 \
  --edit-prompt "A dog in the scene" \
  --target-video /path/to/target_motion.mp4 \
  --roi-mask-video /path/to/dog_mask_video.mp4 \
  --output-dir ./audio_latent_opt
```

The mask video should have white pixels on the edited object and black elsewhere.
This is a good place to use SAM2 or any external tracker/segmenter; the optimizer
will then apply the RAFT flow and motion-magnitude losses only inside that region.

### Auto-generate masks with SAM2

You can generate both source and target mask videos directly with:

```bash
python editing/generate_sam2_masks.py \
  --src-video /path/to/source.mp4 \
  --target-video /path/to/target_motion.mp4 \
  --object-prompt dog \
  --sam2-config /path/to/sam2_config.yaml \
  --sam2-checkpoint /path/to/sam2_checkpoint.pt \
  --output-dir ./audio_latent_opt
```

This writes:
- `src_mask_dog.mp4`
- `target_mask_dog.mp4`

The Slurm script can run this automatically before optimization and wire the
generated target mask into `--roi-mask-video`:

```bash
export GENERATE_SAM2_MASKS=1
export OBJECT_PROMPT=dog
export SAM2_CONFIG=/path/to/sam2_config.yaml
export SAM2_CHECKPOINT=/path/to/sam2_checkpoint.pt
sbatch editing/scripts/submit_optimize_jump_dog.sh /path/to/source.mp4
```

Useful output files:
- `best_optimized_video.mp4`: best video found during optimization
- `baseline_unoptimized_video.mp4`: same setup with original audio latent
- `optimization_log.csv`: per-candidate losses
- `best_latent_params.pt` and `best_audio_latent.pt`: saved optimized latent state

### 1. List available perturbations
```bash
source .venv/bin/activate
python editing/experiment.py --list-perturbations
```

### 2. Run locally (small subset for testing)
```bash
python editing/experiment.py \
    --src-video /path/to/source.mp4 \
    --prompt "A person playing guitar on stage" \
    --perturbations identity silence noise_snr10 phase_randomize time_reverse \
    --output-dir ./editing_results
```

### 3. Submit all perturbations as a SLURM job array (recommended)
```bash
bash editing/scripts/launch_array.sh \
    /path/to/source.mp4 \
    "A person playing guitar on stage"
```

### 4. Submit all perturbations as a single long SLURM job
```bash
sbatch editing/scripts/submit_all.sh \
    /path/to/source.mp4 \
    "A person playing guitar on stage"
```

---

## Perturbation catalogue

| Name | Description |
|------|-------------|
| `identity` | No change (baseline) |
| `noise_snr20` | Add Gaussian noise at SNR = 20 dB |
| `noise_snr10` | Add Gaussian noise at SNR = 10 dB |
| `noise_snr5` | Add Gaussian noise at SNR = 5 dB |
| `noise_snr0` | Add Gaussian noise at SNR = 0 dB (50% noise) |
| `silence` | Complete silence (zero signal) |
| `white_noise` | Replace with white noise (same RMS) |
| `lowpass_1000hz` | Low-pass filter at 1 kHz |
| `lowpass_500hz` | Low-pass filter at 500 Hz |
| `highpass_4000hz` | High-pass filter at 4 kHz |
| `bandpass_speech` | 300–3400 Hz (telephone band) |
| `phase_randomize` | Randomize phase, keep magnitude (destroys temporal structure) |
| `time_reverse` | Audio played backwards |
| `time_shuffle_1s` | Shuffle 1-second chunks in random order |
| `partial_mute_50` | Mute a contiguous random 50% of the audio |
| `partial_mute_80` | Mute a contiguous random 80% of the audio |
| `pitch_up_4` | Pitch shift up 4 semitones |
| `pitch_up_12` | Pitch shift up one octave |
| `pitch_down_4` | Pitch shift down 4 semitones |
| `pitch_down_12` | Pitch shift down one octave |
| `echo` | Add echo (300 ms delay, 50% decay) |
| `speed_up` | Speed up 1.5× (pitch also rises) |
| `slow_down` | Slow down 0.67× (pitch also drops) |

---

## CLI reference

```
experiment.py [options]

Required:
  --src-video PATH          Source video with audio track
  --prompt TEXT             Text description of the video

Shape (auto-detected if omitted):
  --height INT              Output height (multiple of 64)
  --width INT               Output width (multiple of 64)
  --num-frames INT          Frame count (8k + 1)
  --frame-rate FLOAT        Frames per second

Perturbations:
  --perturbations NAME ...  Subset to run (default: all)
  --list-perturbations      Print names and exit

Image conditioning:
  --no-image-cond           Disable first-frame anchoring (more audio freedom)
  --image-cond-strength F   Conditioning strength [0–1], default 1.0

Checkpoints (pre-configured for the cluster):
  --checkpoint-path PATH
  --distilled-lora PATH
  --distilled-lora-strength FLOAT   default 0.8
  --spatial-upsampler-path PATH
  --gemma-root PATH
  --quantization {fp8-cast,fp8-scaled-mm}

Diffusion:
  --num-inference-steps INT   default 30
  --seed INT                  default 42
  --cfg-scale FLOAT           CFG scale, default 3.0
  --a2v-scale FLOAT           Audio-to-video guidance scale, default 2.5
```

---

## Outputs

For each perturbation `<name>`, two files are written:

| File | Contents |
|------|----------|
| `<name>.mp4` | Re-generated video with modified audio muxed in |
| `<name>_audio.wav` | Modified audio standalone (for listening) |

The baseline (`identity`) output is the reference: same audio as original.

---

## Design notes

* **Pipeline loaded once** — `experiment.py` loads the 22 B model a single time and
  loops through all perturbations. This avoids the ~3 min load penalty per perturbation.
* **First-frame conditioning** — by default the first frame of the source video is used
  as a visual anchor (`--image PATH 0 1.0`). Disable via `--no-image-cond` to give the
  model more freedom to diverge visually.
* **Random seed is fixed** across all perturbations (default `--seed 42`) so that
  differences between outputs are purely due to the audio change, not sampling variance.
  Change `--seed` to explore different random trajectories.
* **Audio is muxed back** into the output video so that you can watch+listen to each
  perturbation without hunting for the corresponding `.wav` file.
