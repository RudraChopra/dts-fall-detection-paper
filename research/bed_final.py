import json,numpy as np,glob
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score,f1_score
c=np.load('../work/old_outputs/dts_one_run/fallvision_dts128_features.npz',allow_pickle=True)
X,y,sc=c['X_fv'].astype(np.float64),c['y_fv'],c['scenarios_fv']
_,idx=np.unique(np.round(X,6),axis=0,return_index=True);idx=np.sort(idx);X,y,sc=X[idx],y[idx],sc[idx]
bed=sc=='Bed';Xtr0,ytr0=X[~bed],y[~bed];Xbed,ybed=X[bed],y[bed]
HP=dict(max_iter=600,learning_rate=0.2,l2_regularization=0.1)
rng0=np.random.default_rng(2026);perm=rng0.permutation(len(Xbed))
pool_idx=perm[:200];test_idx=perm[200:];Xbt,ybt=Xbed[test_idx],ybed[test_idx]
sweep={}
for f in sorted(glob.glob('research/bed_parts/K*.json'),key=lambda p:int(p.split('K')[-1].split('.')[0])):
    K=int(f.split('K')[-1].split('.')[0]);sweep[str(K)]=json.load(open(f))
clf0=HGB(random_state=20260610,**HP).fit(Xtr0,ytr0);p0=clf0.predict_proba(Xbt)[:,1];auc0=roc_auc_score(ybt,p0)
f1_def=f1_score(ybt,(p0>=0.5).astype(int))
ths=np.linspace(0.05,0.95,181);bt=max(ths,key=lambda t:f1_score(ybt,(p0>=t).astype(int)))
f1_cal=f1_score(ybt,(p0>=bt).astype(int))
calib=dict(auroc=float(auc0),f1_default=float(f1_def),f1_bed_threshold=float(f1_cal),best_threshold=float(bt))
pred=(p0>=0.5).astype(int);fn=(pred==0)&(ybt==1);tp=(pred==1)&(ybt==1)
ahs=Xbt[:,3*16+15];ahy=Xbt[:,1*16+15]
err=dict(n_fn=int(fn.sum()),n_falls=int((ybt==1).sum()),miss_rate=float(fn.sum()/(ybt==1).sum()),
         hipspd_alpha_fn=float(ahs[fn].mean()),hipspd_alpha_tp=float(ahs[tp].mean()),
         hipy_alpha_fn=float(ahy[fn].mean()),hipy_alpha_tp=float(ahy[tp].mean()))
addsel=np.random.default_rng(100).choice(pool_idx,size=200,replace=False)
clfK=HGB(random_state=20260610,**HP).fit(np.vstack([Xtr0,Xbed[addsel]]),np.concatenate([ytr0,ybed[addsel]]))
pK=clfK.predict_proba(Xbt)[:,1]
dist=dict(k0_sep=float(p0[ybt==1].mean()-p0[ybt==0].mean()),k200_sep=float(pK[ybt==1].mean()-pK[ybt==0].mean()),
          k0_fall=float(p0[ybt==1].mean()),k0_non=float(p0[ybt==0].mean()),k200_fall=float(pK[ybt==1].mean()),k200_non=float(pK[ybt==0].mean()))
out=dict(sweep=sweep,calibration=calib,error_analysis=err,score_dist=dist,
         protocol='dedup 5,572; Chair+Stand base; fixed 1,547 Bed test; 5 seeds; HGB(600,0.2,0.1)',
         scores_k0=p0.tolist(),scores_k200=pK.tolist(),y=ybt.astype(int).tolist())
json.dump(out,open('research/bed_deepdive.json','w'),indent=1)
print('K-sweep:',{k:round(v['auroc'],4) for k,v in sweep.items()})
print('Calib: AUROC %.4f  F1 0.5->%.3f, bedthr(%.2f)->%.3f'%(auc0,f1_def,bt,f1_cal))
print('Err: %d/%d missed (%.1f%%); hip-spd alpha FN %.3f vs TP %.3f'%(err['n_fn'],err['n_falls'],100*err['miss_rate'],err['hipspd_alpha_fn'],err['hipspd_alpha_tp']))
print('Dist: separation K0 %.3f -> K200 %.3f'%(dist['k0_sep'],dist['k200_sep']))
