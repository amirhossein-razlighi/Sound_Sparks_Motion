# per-video motion: mean |frame_t - frame_t-1| (gray, 256x144, 0-255) over the whole clip and after the preserved prefix K
import csv, json, subprocess, sys, numpy as np
from concurrent.futures import ProcessPoolExecutor
R='/scratch/amirrz/iclr_rebuttal/ivebench'
def frames(v):
    raw=subprocess.run(['ffmpeg','-v','error','-i',v,'-vf','scale=256:144,format=gray','-f','rawvideo','-'],capture_output=True,check=True).stdout
    return np.frombuffer(raw,np.uint8).reshape(-1,144,256).astype(np.float32)
def kof(cfg): return {'A':5,'B':5,'H':5,'D':5,'C':1,'E':25,'F':25,'G':25}.get(cfg,0)
def job(a):
    tag,n,v,cfg=a; f=frames(v); d=np.abs(np.diff(f,axis=0)).mean(axis=(1,2)); k=kof(cfg)
    return dict(set=tag,item=n,cfg=cfg,video=v,motion_all=float(d.mean()),motion_after_k=float(d[max(k-1,0):].mean()) if len(d)>k else float('nan'),n_frames=len(f))
jobs=[]
for t in ['ours_critic','retake_critic','ours_eye','retake_eye']:
    for r in csv.DictReader(open(f'{R}/eval_lists/{t}.csv')): jobs.append((t,r['name'],r['video'],r['cfg']))
for r in csv.DictReader(open(f'{R}/eval_lists/source_noedit.csv')): jobs.append(('source_noedit',r['name'],r['video'],''))
with ProcessPoolExecutor(8) as ex: out=list(ex.map(job,jobs))
w=csv.DictWriter(open(f'{R}/eval/motion_proxy.csv','w',newline=''),fieldnames=list(out[0])); w.writeheader(); w.writerows(out); print(len(out))
