#!/usr/bin/env python3
"""Turn a real web clip (meme / film scene) into an H3 source in the exact format of our generated sources:
512x320 (cover-crop), 24 fps, 89 frames = 3.7083 s, plus a 32 kHz mono wav of the same window (silence if no audio).
Login-node light (yt-dlp + ffmpeg -threads 1).  Nothing is deleted; every download is kept in inputs/real/raw/.

Usage:
  prep_real_source.py <slug> <youtube-id|url|local file> <start-seconds> [--crop left|center|right|x=<0..1>] [--sheet]
  --crop : where to take the 16:10 window from when the clip is wider/taller (default center); x= is the centre of the
           crop as a fraction of the width.
  --sheet: also write <slug>__src_sheet.jpg (12 frames) for inspection.
Outputs: /scratch/amirrz/H3_exp/inputs/real/<slug>__real_89f.mp4, <slug>_audio_89f.wav, <slug>__raw.mp4 (kept)."""
import os, subprocess, sys, json, shutil

RAW = "/scratch/amirrz/H3_exp/inputs/real/raw"; OUT = "/scratch/amirrz/H3_exp/inputs/real"
YTDLP = os.path.expanduser("~/.local/bin/yt-dlp"); W, H, FPS, N = 512, 320, 24, 89; DUR = N / FPS
R = os.path.dirname(os.path.abspath(__file__))

def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)

def fetch(slug, src):
    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(src):
        dst = os.path.join(RAW, f"{slug}__raw{os.path.splitext(src)[1]}")
        if os.path.abspath(src) != os.path.abspath(dst): shutil.copy2(src, dst)
        return dst
    url = src if src.startswith("http") else f"https://www.youtube.com/watch?v={src}"
    dst = os.path.join(RAW, f"{slug}__raw.%(ext)s")
    run([YTDLP, "--no-warnings", "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b", "--merge-output-format", "mp4",
         "-o", dst, url])
    files = [f for f in os.listdir(RAW) if f.startswith(f"{slug}__raw.")]
    assert files, "download failed"; return os.path.join(RAW, sorted(files)[-1])

def probe(path):
    v = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", path]).stdout.strip().split(",")
    a = run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", path]).stdout.strip()
    return int(v[0]), int(v[1]), bool(a)

def main():
    slug, src, start = sys.argv[1], sys.argv[2], float(sys.argv[3]); crop = "center"; sheet = False
    for a in sys.argv[4:]:
        if a.startswith("--crop"): crop = a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
        if a == "--sheet": sheet = True
    os.makedirs(OUT, exist_ok=True)
    raw = fetch(slug, src); w, h, has_audio = probe(raw)
    # cover-scale to at least 512x320 then crop a 512x320 window
    if w / h >= W / H:  # wider than 16:10 -> scale by height, crop width
        sc = f"scale=-2:{H}"; frac = {"left": 0.0, "center": 0.5, "right": 1.0}.get(crop, float(crop.split("=")[-1]) if "x=" in crop else 0.5)
        cropf = f"crop={W}:{H}:(iw-{W})*{frac}:0"
    else:               # taller -> scale by width, crop height (centre)
        sc = f"scale={W}:-2"; cropf = f"crop={W}:{H}:0:(ih-{H})/2"
    vf = f"{sc},{cropf},fps={FPS},format=yuv420p"
    mp4 = os.path.join(OUT, f"{slug}__real_89f.mp4"); wav = os.path.join(OUT, f"{slug}_audio_89f.wav")
    run(["ffmpeg", "-y", "-v", "error", "-threads", "1", "-ss", f"{start:.3f}", "-i", raw, "-t", f"{DUR:.4f}", "-vf", vf,
         "-frames:v", str(N), "-an", "-c:v", "libx264", "-crf", "16", "-preset", "medium", mp4])
    if has_audio:
        run(["ffmpeg", "-y", "-v", "error", "-threads", "1", "-ss", f"{start:.3f}", "-i", raw, "-t", f"{DUR:.4f}", "-vn", "-ac", "1", "-ar", "32000",
             "-c:a", "pcm_s16le", wav])
    else:
        run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=32000:cl=mono", "-t", f"{DUR:.4f}", "-c:a", "pcm_s16le", wav])
    n = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", mp4]).stdout.strip()
    print(json.dumps({"slug": slug, "raw": raw, "raw_size": [w, h], "audio": has_audio, "src": mp4, "wav": wav, "frames": int(n), "start": start}))
    if sheet:
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", W="220", N="12")
        subprocess.run(["bash", os.path.join(R, "ab_sheet.sh"), os.path.join(OUT, f"{slug}__src_sheet.jpg"), mp4], env=env, capture_output=True)

if __name__ == "__main__":
    main()
