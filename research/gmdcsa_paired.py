import json,numpy as np,warnings;warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import roc_auc_score
from aeon.transformations.collection.convolution_based import MultiRocket
g=np.load('results/gmdcsa/scores.npz',allow_pickle=True)
F,seq,y,subj=g['feats'],g['fixed'],g['y'],g['subject']
seqA=seq.transpose(0,2,1).astype(np.float32)
subs=['S1','S2','S3','S4']
oof_h=np.zeros(len(y));oof_m=np.zeros(len(y))
for s in subs:
    tr=subj!=s;te=subj==s
    oof_h[te]=HGB(max_iter=600,learning_rate=0.2,l2_regularization=0.1,random_state=42).fit(F[tr],y[tr]).predict_proba(F[te])[:,1]
    mr=MultiRocket(n_kernels=10000,random_state=42,n_jobs=2)
    Xtr=mr.fit_transform(seqA[tr]);Xte=mr.transform(seqA[te])
    ss=StandardScaler().fit(Xtr)
    oof_m[te]=RidgeClassifierCV(alphas=np.logspace(-3,3,10)).fit(ss.transform(Xtr),y[tr]).decision_function(ss.transform(Xte))
ah=roc_auc_score(y,oof_h);am=roc_auc_score(y,oof_m)
rng=np.random.default_rng(7);dif=[]
for _ in range(2000):
    i=rng.integers(0,len(y),len(y))
    if len(set(y[i]))<2:continue
    dif.append(roc_auc_score(y[i],oof_h[i])-roc_auc_score(y[i],oof_m[i]))
ci=[float(np.percentile(dif,2.5)),float(np.percentile(dif,97.5))]
out=dict(pooled_oof_auroc_hgb=float(ah),pooled_oof_auroc_multirocket=float(am),
         delta=float(ah-am),delta_ci=ci,n=int(len(y)))
json.dump(out,open('research/gmdcsa_paired.json','w'),indent=1)
print('pooled OOF AUROC: DTS+HGB %.4f vs MultiROCKET %.4f  delta %+.4f CI [%.4f, %.4f]'%(ah,am,ah-am,ci[0],ci[1]))
