"""Controlled temporal-order benchmark: three-phase fall trajectories vs frame-shuffled
counterparts (identical marginals by construction)."""
import sys, json, numpy as np
sys.path.insert(0,'.')
from pathlib import Path
from dts.features import extract_dts128
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
W=Path('.'); rng=np.random.RandomState(42)

def make_skeleton(T, rng):
    """Three-phase hip trajectory: still, drop, floor; 17-joint skeleton around hip."""
    t1=int(T*rng.uniform(0.25,0.4)); t2=t1+int(T*rng.uniform(0.1,0.2))
    hip_y=np.concatenate([np.full(t1,1.0), np.linspace(1.0,0.15,max(1,t2-t1)), np.full(T-t2,0.15)])
    hip_y+=rng.normal(0,0.01,T)
    hip_x=np.cumsum(rng.normal(0,0.005,T))
    base=np.zeros((17,2))
    base[0]=[0,0.75]; base[1]=[ .03,.78]; base[2]=[-.03,.78]; base[3]=[ .06,.76]; base[4]=[-.06,.76]
    base[5]=[ .15,.55]; base[6]=[-.15,.55]; base[7]=[ .22,.3]; base[8]=[-.22,.3]
    base[9]=[ .25,.05]; base[10]=[-.25,.05]; base[11]=[ .1,0]; base[12]=[-.1,0]
    base[13]=[ .12,-.45]; base[14]=[-.12,-.45]; base[15]=[ .13,-.9]; base[16]=[-.13,-.9]
    seq=np.zeros((T,17,2),np.float32)
    for t in range(T):
        # body rotates toward horizontal as hip drops
        frac=1-(hip_y[t]-0.15)/0.85
        ang=frac*(np.pi/2)*0.9
        R=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
        seq[t]=(base@R.T)*1.0+ [hip_x[t],hip_y[t]]
        seq[t]+=rng.normal(0,0.008,(17,2))
    conf=np.ones((T,17),np.float32)
    return seq,conf

def bof_features(seq):
    """Order-invariant bag-of-frames statistics (marginal pose distribution only)."""
    flat=np.sort(seq.astype(np.float64).reshape(len(seq),-1),axis=0)
    return np.concatenate([flat.mean(0),flat.std(0),np.percentile(flat,25,axis=0),np.percentile(flat,75,axis=0)])

N=600
X_dts=[]; X_bof=[]; yl=[]
for i in range(N):
    T=int(rng.uniform(60,150))
    seq,conf=make_skeleton(T,rng)
    perm=rng.permutation(len(seq)); shuf=seq[perm]
    X_dts.append(extract_dts128(seq,conf,tau=0.30)); X_bof.append(bof_features(seq)); yl.append(1)
    X_dts.append(extract_dts128(shuf,conf,tau=0.30)); X_bof.append(bof_features(shuf)); yl.append(0)
X_dts=np.array(X_dts); X_bof=np.array(X_bof); yl=np.array(yl)
alpha_cols=[i*16+15 for i in range(8)]
res={}
# KS test on marginals (hip-y pooled)
from scipy.stats import ks_2samp
# pooled hip-y values across fall vs shuffled are identical multisets by construction -> p=1
res['ks_note']='identical multisets by construction'
# learning curve
curve={}
pairs=np.arange(N)
for n in [100,250,500,1000,2000,3100]:
    aucs={'DTS':[],'BoF':[],'alpha':[]}
    n_pairs=min(n//2, N-100)
    for rep in range(3):
        porder=np.random.RandomState(100+rep).permutation(N)
        trp, tep = porder[:n_pairs], porder[-100:]
        tr=np.concatenate([2*trp,2*trp+1]); te=np.concatenate([2*tep,2*tep+1])
        m=HistGradientBoostingClassifier(max_iter=300,random_state=42).fit(X_dts[tr],yl[tr])
        aucs['DTS'].append(roc_auc_score(yl[te],m.predict_proba(X_dts[te])[:,1]))
        lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(X_bof[tr],yl[tr])
        aucs['BoF'].append(roc_auc_score(yl[te],lr.predict_proba(X_bof[te])[:,1]))
        la=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(X_dts[tr][:,alpha_cols],yl[tr])
        aucs['alpha'].append(roc_auc_score(yl[te],la.predict_proba(X_dts[te][:,alpha_cols])[:,1]))
    curve[n]={k:float(np.mean(v)) for k,v in aucs.items()}
    print(n,curve[n],flush=True)
res['curve']=curve
# 5-fold CV at full N
skf=StratifiedKFold(5,shuffle=True,random_state=42); folds=[]
for tr,te in skf.split(X_dts,yl):
    m=HistGradientBoostingClassifier(max_iter=300,random_state=42).fit(X_dts[tr],yl[tr])
    folds.append(float(roc_auc_score(yl[te],m.predict_proba(X_dts[te])[:,1])))
res['cv5']=folds; print('5-fold CV AUROC',folds)
json.dump(res,open(W/'synth.json','w'),indent=1)
