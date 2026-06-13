import sys, json, re, csv, numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, recall_score
W=Path('.')
d=np.load(W/'aaai/fallvision_dts128_features.npz',allow_pickle=True)
X,y,scen=d['X_fv'],d['y_fv'],d['scenarios_fv']
man=list(csv.DictReader(open(W/'aaai/tables/fallvision_manifest.csv')))
sess=np.array([re.search(r'_(\d+)_(?:keypoints|ketpoints)_csv',r['archive']).group(1) for r in man])
def fit_eval(tr,te):
    m=HistGradientBoostingClassifier(max_iter=600,learning_rate=0.2,l2_regularization=0.1,random_state=42)
    m.fit(X[tr],y[tr]); p=m.predict_proba(X[te])[:,1]
    yt=y[te]; yp=(p>=0.5).astype(int)
    res={'n':int(te.sum()),'falls':int(yt.sum()),'f1':float(f1_score(yt,yp)),'auroc':float(roc_auc_score(yt,p)),'rec':float(recall_score(yt,yp))}
    rng=np.random.RandomState(42); ci={'f1':[],'auroc':[],'rec':[]}
    n=len(yt)
    for b in range(2000):
        s=rng.randint(0,n,n); ys,ps=yt[s],p[s]
        if ys.min()==ys.max(): continue
        ypb=(ps>=0.5).astype(int)
        ci['f1'].append(f1_score(ys,ypb,zero_division=0)); ci['auroc'].append(roc_auc_score(ys,ps)); ci['rec'].append(recall_score(ys,ypb,zero_division=0))
    res['ci']={k:[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))] for k,v in ci.items()}
    return res
mode=sys.argv[1]; out_f=W/f'{mode}_ci.json'
out=json.load(open(out_f)) if out_f.exists() else {}
if mode=='lfo':
    for hold in ['Bed','Chair','Stand']:
        if hold in out: continue
        out[hold]=fit_eval(scen!=hold, scen==hold); json.dump(out,open(out_f,'w'),indent=1)
        print(hold, {k:round(v,4) if isinstance(v,float) else v for k,v in out[hold].items() if k!='ci'},flush=True)
else:
    for s in ['1','2','3','4']:
        if s in out: continue
        out[s]=fit_eval(sess!=s, sess==s); json.dump(out,open(out_f,'w'),indent=1)
        print(s, {k:round(v,4) if isinstance(v,float) else v for k,v in out[s].items() if k!='ci'},flush=True)
# also iid stratified reference under same protocol for transfer gap
if mode=='lfo' and 'iid_ref' not in out:
    sm=np.load(W/'split_mr.npz')
    r=fit_eval(sm['train'], sm['test'])
    out['iid_ref']=r; json.dump(out,open(out_f,'w'),indent=1); print('iid_ref',round(r['f1'],4))
