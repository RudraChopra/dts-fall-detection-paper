import json,sys,numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score,f1_score
c=np.load('../work/old_outputs/dts_one_run/fallvision_dts128_features.npz',allow_pickle=True)
X,y,sc=c['X_fv'].astype(np.float64),c['y_fv'],c['scenarios_fv']
_,idx=np.unique(np.round(X,6),axis=0,return_index=True);idx=np.sort(idx);X,y,sc=X[idx],y[idx],sc[idx]
bed=sc=='Bed';Xtr0,ytr0=X[~bed],y[~bed];Xbed,ybed=X[bed],y[bed]
HP=dict(max_iter=600,learning_rate=0.2,l2_regularization=0.1)
SEEDS=[0,1,2,3,4]
rng0=np.random.default_rng(2026);perm=rng0.permutation(len(Xbed))
pool_idx=perm[:200];test_idx=perm[200:];Xbt,ybt=Xbed[test_idx],ybed[test_idx]
for K in [int(x) for x in sys.argv[1:]]:
    aucs=[];f1s=[]
    for s in SEEDS:
        rng=np.random.default_rng(100+s)
        addsel=rng.choice(pool_idx,size=K,replace=False) if K>0 else np.array([],dtype=int)
        Xtr=np.vstack([Xtr0,Xbed[addsel]]) if K>0 else Xtr0
        ytr=np.concatenate([ytr0,ybed[addsel]]) if K>0 else ytr0
        clf=HGB(random_state=20260610+s,**HP).fit(Xtr,ytr)
        p=clf.predict_proba(Xbt)[:,1]
        aucs.append(roc_auc_score(ybt,p));f1s.append(f1_score(ybt,(p>=0.5).astype(int)))
    a=np.array(aucs)
    rec=dict(auroc=float(a.mean()),auroc_sd=float(a.std()),
             auroc_ci=[float(a.mean()-1.96*a.std()/np.sqrt(len(a))),float(a.mean()+1.96*a.std()/np.sqrt(len(a)))],
             f1=float(np.mean(f1s)),n_test=int(len(ybt)),seeds=len(SEEDS))
    json.dump(rec,open('research/bed_parts/K%d.json'%K,'w'))
    print('K=%3d AUROC %.4f ± %.4f F1 %.3f'%(K,a.mean(),a.std(),np.mean(f1s)),flush=True)
