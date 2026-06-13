import csv, io, sys, numpy as np
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,'.')
from dts.features import normalise

W=Path('.'); EX=Path('./data/fallvision_extracted')
MAX_T=150
COCO17={"nose":0,"left eye":1,"right eye":2,"left ear":3,"right ear":4,"left shoulder":5,"right shoulder":6,"left elbow":7,"right elbow":8,"left wrist":9,"right wrist":10,"left hip":11,"right hip":12,"left knee":13,"right knee":14,"left ankle":15,"right ankle":16}

def parse_csv_flexible(text):
    try: rows=list(csv.DictReader(io.StringIO(text)))
    except Exception: return None,None
    if not rows: return None,None
    cm={}
    for col in rows[0]:
        cl=col.strip().lower()
        if cl=='frame': cm['frame']=col
        elif cl=='keypoint': cm['keypoint']=col
        elif cl=='x': cm['x']=col
        elif cl=='y': cm['y']=col
        elif 'conf' in cl: cm['confidence']=col
    if {'frame','keypoint','x','y','confidence'}-set(cm): return None,None
    frames=defaultdict(lambda:{'xy':np.zeros((17,2)),'cf':np.zeros(17),'cnt':np.zeros(17)})
    for row in rows:
        try:
            fr=int(float(row[cm['frame']])); raw=row[cm['keypoint']]
            try: kp=int(float(raw))
            except Exception: kp=COCO17.get(str(raw).strip().lower(),-1)
            if 0<=kp<17:
                frames[fr]['xy'][kp]+=[float(row[cm['x']]),float(row[cm['y']])]
                frames[fr]['cf'][kp]=max(frames[fr]['cf'][kp],float(row[cm['confidence']]))
                frames[fr]['cnt'][kp]+=1
        except Exception: continue
    if not frames: return None,None
    sf=sorted(frames); seq=np.zeros((len(sf),17,2),np.float32); conf=np.zeros((len(sf),17),np.float32)
    for i,fr in enumerate(sf):
        cnt=np.maximum(frames[fr]['cnt'],1)[:,None]
        seq[i]=frames[fr]['xy']/cnt; conf[i]=frames[fr]['cf']
    return seq,conf

def prep(seq,conf):
    n,_=normalise(seq); n=np.asarray(n,np.float32)
    n[np.asarray(conf)<0.10]=0.0
    flat=n.reshape(len(n),-1)
    L=min(len(flat),MAX_T)
    out=np.zeros((MAX_T,34),np.float32); out[:L]=flat[:L]
    return out,L

man=list(csv.DictReader(open(W/'aaai/tables/fallvision_manifest.csv')))
archives=sorted({r['archive'] for r in man})
(W/'seq').mkdir(exist_ok=True)
target=sys.argv[1]
rows=[(i,r) for i,r in enumerate(man) if r['archive']==target]
X=np.zeros((len(rows),MAX_T,34),np.float32); L=np.zeros(len(rows),np.int64); idx=np.zeros(len(rows),np.int64)
nf_mismatch=0
for j,(i,r) in enumerate(rows):
    stem=r['clip_id'].split('__',1)[1]
    p=EX/target/(stem+'.csv')
    if not p.exists():
        cands=list((EX/target).rglob(stem+'.csv'))
        p=cands[0] if cands else None
    seq,conf=parse_csv_flexible(p.read_text(errors='replace'))
    if int(r['n_frames'])!=len(seq): nf_mismatch+=1
    X[j],L[j]=prep(seq,conf); idx[j]=i
np.savez_compressed(W/'seq'/(target+'.npz'), X=X, L=L, idx=idx)
print(target,'rows',len(rows),'n_frames mismatches',nf_mismatch)
