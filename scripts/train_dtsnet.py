import sys, json, numpy as np, torch
sys.path.insert(0,'.')
from pathlib import Path
from dts.models import DTSNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
W=Path('.'); torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)
d=np.load(W/'aaai/fallvision_dts128_features.npz',allow_pickle=True)
X,y=d['X_fv'],d['y_fv']; sm=np.load(W/'split_mr.npz')
sc=StandardScaler().fit(X[sm['train']])
Xtr,ytr=sc.transform(X[sm['train']]),y[sm['train']].astype(np.float32)
Xva,yva=sc.transform(X[sm['val']]),y[sm['val']].astype(np.float32)
Xte,yte=sc.transform(X[sm['test']]),y[sm['test']].astype(np.float32)
model=DTSNet()
opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-4)
Xtr_t=torch.tensor(Xtr,dtype=torch.float32); ytr_t=torch.tensor(ytr)
Xva_t=torch.tensor(Xva,dtype=torch.float32); Xte_t=torch.tensor(Xte,dtype=torch.float32)
best_va=0; best_pva=best_pte=None
for ep in range(80):
    g=torch.Generator().manual_seed(42+ep); perm=torch.randperm(len(ytr),generator=g)
    model.train()
    for i in range(0,len(ytr),64):
        b=perm[i:i+64]
        loss=model.training_loss(Xtr_t[b],ytr_t[b])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    model.eval()
    with torch.no_grad():
        pva=model.predict_proba(Xva_t)[:,1].numpy()
    va=roc_auc_score(yva,pva)
    if va>best_va:
        best_va=va
        with torch.no_grad(): best_pva,best_pte=pva,model.predict_proba(Xte_t)[:,1].numpy()
best=(0.,0.5)
for thr in np.linspace(0.05,0.95,91):
    f=f1_score(yva,(best_pva>=thr).astype(int),zero_division=0)
    if f>best[0]: best=(float(f),float(thr))
thr=best[1]; yp=(best_pte>=thr).astype(int)
tn,fp,fn,tp=confusion_matrix(yte,yp,labels=[0,1]).ravel()
res=dict(auroc=float(roc_auc_score(yte,best_pte)),f1=float(f1_score(yte,yp)),prec=float(precision_score(yte,yp)),
         rec=float(recall_score(yte,yp)),acc=float(accuracy_score(yte,yp)),tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn),thr=thr)
rng=np.random.RandomState(42); ci={k:[] for k in ['auroc','f1','prec','rec','acc']}
for b in range(2000):
    s=rng.randint(0,len(yte),len(yte)); ys,ps=yte[s],best_pte[s]
    if ys.min()==ys.max(): continue
    ypb=(ps>=thr).astype(int)
    ci['auroc'].append(roc_auc_score(ys,ps)); ci['f1'].append(f1_score(ys,ypb,zero_division=0))
    ci['prec'].append(precision_score(ys,ypb,zero_division=0)); ci['rec'].append(recall_score(ys,ypb,zero_division=0)); ci['acc'].append(accuracy_score(ys,ypb))
res_ci={k:[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))] for k,v in ci.items()}
nparams=sum(p.numel() for p in model.parameters())
json.dump({'test':res,'test_ci':res_ci,'best_val_auroc':float(best_va),'params':nparams},open(W/'nn/final_DTS-Net.json','w'),indent=1)
np.save(W/'scores_DTS-Net.npy',best_pte)
# attention weights for interpretability check (5 seeds would take longer; main seed only)
print('DTS-Net params',nparams,'| TEST',res)
