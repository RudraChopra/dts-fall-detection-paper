import sys, json, numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
W=Path('.')
d=np.load(W/'aaai/fallvision_dts128_features.npz',allow_pickle=True)
X,y=d['X_fv'],d['y_fv']; sm=np.load(W/'split_mr.npz')
itr,iva,ite=sm['train'],sm['val'],sm['test']
mask=np.load(W/'test_clean_mask.npy')
FAM=['BBox-W','Hip-Y','Torso-Ang','Hip-Spd','Ctr-Spd','Hip-Acc','Shldr-Y','Head-Y']
def run(mask_cols=None):
    Xm=X.copy()
    if mask_cols is not None: Xm[:,mask_cols]=0.0
    m=HistGradientBoostingClassifier(max_iter=600,learning_rate=0.2,l2_regularization=0.1,random_state=42)
    m.fit(Xm[itr],y[itr])
    pva=m.predict_proba(Xm[iva])[:,1]; pte=m.predict_proba(Xm[ite])[:,1]
    best=(0.,0.5)
    for thr in np.linspace(0.05,0.95,91):
        f=f1_score(y[iva],(pva>=thr).astype(int),zero_division=0)
        if f>best[0]: best=(float(f),float(thr))
    ptc=pte[mask]; ytc=y[ite][mask]
    yp=(ptc>=best[1]).astype(int)
    return {'f1':float(f1_score(ytc,yp)),'auroc':float(roc_auc_score(ytc,ptc))}
out_f=W/'ablation_clean.json'
out=json.load(open(out_f)) if out_f.exists() else {}
todo=sys.argv[1]
if todo=='1':
    if 'Full' not in out: out['Full']=run()
    for i in range(0,4):
        k='-'+FAM[i]
        if k not in out: out[k]=run(list(range(16*i,16*i+16)))
elif todo=='2':
    for i in range(4,8):
        k='-'+FAM[i]
        if k not in out: out[k]=run(list(range(16*i,16*i+16)))
    if 'StaticOnly' not in out: out['StaticOnly']=run([16*i+15 for i in range(8)])
json.dump(out,open(out_f,'w'),indent=1)
for k,v in out.items(): print(k, round(v['f1'],3), round(v['auroc'],4))
