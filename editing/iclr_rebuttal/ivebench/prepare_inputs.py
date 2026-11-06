#!/usr/bin/env python3
"""IVEBench subject-motion (short) -> LTX inputs. Same geometry/sampling rule as IVEBench data_process/resize_batch.py
(scale = max(W/w, H/h), Lanczos, centre crop; uniform np.linspace(.., dtype=int) frame sampling), with the frame count
set to the largest 8k+1 <= min(n_src, 89) (LTX constraint). Writes inputs/<name>.mp4 (crf 10, source fps, source audio
trimmed to the clip duration or a silent track) and inputs/<name>.json (indices, fps, had_audio)."""
import json, subprocess, sys
from pathlib import Path
import av, cv2, numpy as np
import os
R = Path(os.environ.get("IVE_ROOT", "/scratch/amirrz/iclr_rebuttal/ivebench")); W, H, NMAX = 512, 288, 89
out = R / "inputs"; out.mkdir(exist_ok=True)
items = json.load(open(R / "subset_subject_motion_short.json"))
for it in items:
    name = Path(it["src_video_name"]).stem; dst = out / f"{name}.mp4"
    if dst.exists() and (out / f"{name}.json").exists():
        continue
    src = R / "src" / it["src_video_name"]
    c = av.open(str(src)); vs = c.streams.video[0]; fps = float(vs.average_rate)
    frames = [f.to_ndarray(format="bgr24") for f in c.decode(vs)]; c.close()
    T = len(frames); n = ((min(T, NMAX) - 1) // 8) * 8 + 1
    idx = np.linspace(0, T - 1, num=n, dtype=int) if T > n else np.arange(T)
    def cc(img):
        h, w = img.shape[:2]; s = max(W / w, H / h); nw, nh = round(w * s), round(h * s)
        r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4); x, y = (nw - W) // 2, (nh - H) // 2
        return r[y:y + H, x:x + W]
    sel = [cc(frames[i]) for i in idx]
    tmp = out / f"{name}.noaudio.mp4"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}",
                          "-i", "-", "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
    for f in sel: p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close(); p.wait()
    dur = n / fps
    has_a = bool(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(src)],
                                capture_output=True, text=True).stdout.strip())
    a_in = ["-i", str(src)] if has_a else ["-f", "lavfi", "-t", f"{dur}", "-i", "anullsrc=r=44100:cl=stereo"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), *a_in, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                    "-ar", "44100", "-t", f"{dur}", str(dst)], check=True)
    tmp.unlink()
    (out / f"{name}.json").write_text(json.dumps(dict(name=name, n_src=T, n=int(n), fps=fps, idx=idx.tolist(), had_audio=has_a,
                                                      src_wh=list(frames[0].shape[1::-1]))))
    print(name, T, "->", n, f"{fps:.3f}fps audio={has_a}", flush=True)
print("done", len(list(out.glob("*.mp4"))))
