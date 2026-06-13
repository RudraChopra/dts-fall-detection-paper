"""True subject-held-out on GMDCSA-24 (4 subjects). Compare DTS+HGB, alpha-only+LR,
MiniROCKET, and the modern MultiROCKET-Hydra (2023). Real sequences."""
import json,sys,numpy as np,warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
g=np.load('results/gmdcsa/scores.npz',allow_pickle=True)
F,seq,y,subj=g['feats'],g['fixed'],g['y'],g['subject']
seqA=seq.transpose(0,2,1).astype(np.float32)  # (n,34,150) for aeon
subs=['S1','S2','S3','S4']
alpha_idx=[i*16+15 for i in range(8)]
which=sys.argv[1] if len(sys.argv)>1 else 'all'
res={}
def loso(scorer):
    pers={}
    for s in subs:
        tr=subj!=s; te=subj==s
        sc=scorer(tr,te)
        pers[s]=float(roc_auc_score(y[te],sc))
    pers['mean']=float(np.mean([pers[s] for s in subs]))
    return pers
if which in ('all','hgb'):
    res['DTS+HGB']=loso(lambda tr,te: HGB(max_iter=600,learning_rate=0.2,l2_regularization=0.1,random_state=42).fit(F[tr],y[tr]).predict_proba(F[te])[:,1])
    print('DTS+HGB',res.get('DTS+HGB'),flush=True)
if which in ('all','alpha'):
    def ascore(tr,te):
        sc=StandardScaler().fit(F[tr][:,alpha_idx])
        lr=LogisticRegression(max_iter=1000).fit(sc.transform(F[tr][:,alpha_idx]),y[tr])
        return lr.predict_proba(sc.transform(F[te][:,alpha_idx]))[:,1]
    res['alpha-only+LR']=loso(ascore); print('alpha-only',res['alpha-only+LR'],flush=True)
if which in ('all','mr'):
    from aeon.transformations.collection.convolution_based import MiniRocket
    from sklearn.linear_model import RidgeClassifierCV
    def mrscore(tr,te):
        mr=MiniRocket(n_kernels=10000,random_state=42,n_jobs=2)
        Xtr=mr.fit_transform(seqA[tr]); Xte=mr.transform(seqA[te])
        ss=StandardScaler().fit(Xtr)
        clf=RidgeClassifierCV(alphas=np.logspace(-3,3,10)).fit(ss.transform(Xtr),y[tr])
        return clf.decision_function(ss.transform(Xte))
    res['MiniROCKET']=loso(mrscore); print('MiniROCKET',res['MiniROCKET'],flush=True)
if which in ('all','hydra'):
    from aeon.classification.convolution_based import MultiRocketHydraClassifier
    def hyscore(tr,te):
        clf=MultiRocketHydraClassifier(random_state=42,n_jobs=2).fit(seqA[tr],y[tr])
        try: return clf.predict_proba(seqA[te])[:,1]
        except Exception: return clf.predict(seqA[te]).astype(float)
    res['MultiROCKET-Hydra']=loso(hyscore); print('MultiROCKET-Hydra',res['MultiROCKET-Hydra'],flush=True)
# merge into results file
import os
out='research/gmdcsa_loso.json'
prev=json.load(open(out)) if os.path.exists(out) else {}
prev.update(res); json.dump(prev,open(out,'w'),indent=1)

if which=='multirocket':
    from aeon.transformations.collection.convolution_based import MultiRocket
    from sklearn.linear_model import RidgeClassifierCV
    import os
    def mscore(tr,te):
        mr=MultiRocket(n_kernels=10000,random_state=42,n_jobs=2)
        Xtr=mr.fit_transform(seqA[tr]); Xte=mr.transform(seqA[te])
        ss=StandardScaler().fit(Xtr)
        clf=RidgeClassifierCV(alphas=np.logspace(-3,3,10)).fit(ss.transform(Xtr),y[tr])
        return clf.decision_function(ss.transform(Xte))
    r={'MultiROCKET':loso(mscore)}; print('MultiROCKET',r['MultiROCKET'],flush=True)
    out='research/gmdcsa_loso.json'; prev=json.load(open(out)) if os.path.exists(out) else {}
    prev.update(r); json.dump(prev,open(out,'w'),indent=1)
