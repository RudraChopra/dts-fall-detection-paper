import sys, json, numpy as np
from pathlib import Path
from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import roc_auc_score
d=np.load('lc_data.npz'); S,L,y,sp=d['S'],d['L'],d['y'],d['sp']
Xa=S.transpose(0,2,1).astype(np.float32)
ite=sp=='test'; yte=y[ite]; subs=np.load('lc_subs.npz')
out_f=Path('lc.json'); out=json.load(open(out_f)); out.setdefault('MiniROCKET',{})
n=sys.argv[1]
if n not in out['MiniROCKET']:
    tr=subs[n]
    mr=MiniRocket(n_kernels=10000, random_state=20260610, n_jobs=4)
    Xtr=np.asarray(mr.fit_transform(Xa[tr]),dtype=np.float32)
    Xte=np.asarray(mr.transform(Xa[ite]),dtype=np.float32)
    sc=StandardScaler().fit(Xtr)
    clf=RidgeClassifierCV(alphas=np.logspace(-3,3,10)).fit(sc.transform(Xtr),y[tr])
    p=clf.decision_function(sc.transform(Xte))
    rng=np.random.RandomState(42); cis=[]
    for b in range(1000):
        s=rng.randint(0,len(yte),len(yte))
        if yte[s].min()==yte[s].max(): continue
        cis.append(roc_auc_score(yte[s],p[s]))
    out['MiniROCKET'][n]={'auroc':float(roc_auc_score(yte,p)),'ci':[float(np.percentile(cis,2.5)),float(np.percentile(cis,97.5))]}
    json.dump(out,open(out_f,'w'))
print('MR',n,round(out['MiniROCKET'][n]['auroc'],4))
