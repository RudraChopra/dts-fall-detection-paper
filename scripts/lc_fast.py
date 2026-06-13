import sys, json, numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
d=np.load('tf_data.npz', allow_pickle=True); X,y,sp=d['X'],d['y'],d['sp']
ite=sp=='test'; yte=y[ite]
subs=np.load('lc_subs.npz')
out_f=Path('lc.json'); out=json.load(open(out_f)) if out_f.exists() else {}
out.setdefault('HGB',{})
for n in ['250','500','1000','2000']:
    if n in out['HGB']: continue
    tr=subs[n]
    m=HistGradientBoostingClassifier(max_iter=600,learning_rate=0.1,l2_regularization=0.1,random_state=20260610)
    m.fit(X[tr],y[tr]); p=m.predict_proba(X[ite])[:,1]
    rng=np.random.RandomState(42); cis=[]
    for b in range(1000):
        s=rng.randint(0,len(yte),len(yte))
        if yte[s].min()==yte[s].max(): continue
        cis.append(roc_auc_score(yte[s],p[s]))
    out['HGB'][n]={'auroc':float(roc_auc_score(yte,p)),'ci':[float(np.percentile(cis,2.5)),float(np.percentile(cis,97.5))]}
    print('HGB',n,round(out['HGB'][n]['auroc'],4),flush=True)
    json.dump(out,open(out_f,'w'))
