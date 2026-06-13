import sys, time, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
sys.path.insert(0,'.')
from dts.models import FallLSTM, FallGRU, FallTransformer
t0=time.time(); W=Path('.'); (W/'nn').mkdir(exist_ok=True)
import os; torch.set_num_threads(int(os.environ.get('NT','4')))

class SimpleSTGCN(nn.Module):
    """2-layer spatial GCN + 2-layer temporal conv on COCO-17 graph."""
    def __init__(self, c1=48, c2=48, c3=96, k=7, dropout=0.3):
        super().__init__()
        E=[(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
        A=np.eye(17,dtype=np.float32)
        for i,j in E: A[i,j]=A[j,i]=1
        D=A.sum(1); A=A/np.sqrt(D[:,None]*D[None,:])
        self.register_buffer('A', torch.tensor(A))
        self.g1=nn.Linear(2,c1); self.g2=nn.Linear(c1,c2)
        self.bn1=nn.BatchNorm2d(c1); self.bn2=nn.BatchNorm2d(c2)
        self.t1=nn.Conv1d(c2*17//17,0,1) if False else nn.Conv1d(c2,c3,k,padding=k//2)
        self.bn3=nn.BatchNorm1d(c3)
        self.t2=nn.Conv1d(c3,c3,k,padding=k//2)
        self.bn4=nn.BatchNorm1d(c3)
        self.drop=nn.Dropout(dropout); self.fc=nn.Linear(c3,1)
    def forward(self,x,lengths):
        B,T,_=x.shape
        h=x.view(B,T,17,2)
        h=torch.einsum('uv,btvc->btuc', self.A, self.g1(h))
        h=torch.relu(self.bn1(h.permute(0,3,1,2)).permute(0,2,3,1))
        h=torch.einsum('uv,btvc->btuc', self.A, self.g2(h))
        h=torch.relu(self.bn2(h.permute(0,3,1,2)).permute(0,2,3,1))
        h=h.mean(2)                       # (B,T,c2) joint pool
        h=h.permute(0,2,1)                # (B,c2,T)
        h=torch.relu(self.bn3(self.t1(h)))
        h=torch.relu(self.bn4(self.t2(h)))
        mask=(torch.arange(T,device=x.device)[None,:]<lengths[:,None]).float()
        h=(h*mask[:,None,:]).sum(2)/mask.sum(1,keepdim=True).clamp(min=1)
        return self.fc(self.drop(h)).squeeze(-1)

MODELS={'LSTM':FallLSTM,'GRU':FallGRU,'Transformer':FallTransformer,'SimpleST-GCN':SimpleSTGCN}
name=sys.argv[1]; import os as _os; EPOCHS=int(_os.environ.get('EPOCHS','15')); EXT=_os.environ.get('EXT'); PATIENCE=int(_os.environ.get('PAT','5')); BATCH=64
d=np.load(W/'seq_all.npz'); X,L=d['X'],d['L']
sm=np.load(W/'split_mr.npz'); y=np.load(W/'aaai/fallvision_dts128_features.npz',allow_pickle=True)['y_fv']
def tens(idx): return torch.tensor(X[idx]), torch.tensor(L[idx]), torch.tensor(y[idx],dtype=torch.float32)
Xtr,Ltr,ytr=tens(sm['train']); Xva,Lva,yva=tens(sm['val']); Xte,Lte,yte=tens(sm['test'])
ntr=len(ytr); steps=(ntr+BATCH-1)//BATCH
state_f=W/'nn'/f'{name}_v2_state.pt'

torch.manual_seed(42)
model=MODELS[name]()
print(name,'params',sum(p.numel() for p in model.parameters()))
opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
if EXT:
    EPOCHS=25
    for gp in opt.param_groups: gp['lr']=2e-4
    class _NoSched:
        def step(self): pass
        def state_dict(self): return {}
        def load_state_dict(self,d): pass
    sched=_NoSched()
else:
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=5e-3,steps_per_epoch=steps,epochs=EPOCHS)
start=0; best_va=0.0; best_pva=best_pte=None; best_ep=0
if state_f.exists():
    st=torch.load(state_f,weights_only=False)
    model.load_state_dict(st['model']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
    if EXT:
        for gp in opt.param_groups: gp['lr']=2e-4
    start=st['epoch']; best_va=st['best_va']; best_pva=st['best_pva']; best_pte=st['best_pte']; best_ep=st.get('best_ep',0)
    print('resumed at epoch',start,'best_va',round(best_va,5))

def pred(Xs,Ls,bs=128):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(Xs),bs):
            out.append(torch.sigmoid(model(Xs[i:i+bs],Ls[i:i+bs])).numpy())
    return np.concatenate(out)

def save_state(ep,offset=0):
    import os
    if state_f.exists():
        try:
            cur=torch.load(state_f,weights_only=False)
            if (cur['epoch'],cur.get('offset',0))>(ep,offset):
                print('skip save: file ahead',(cur['epoch'],cur.get('offset',0)),'vs',(ep,offset)); return
        except Exception: pass
    tmp=str(state_f)+'.tmp'
    torch.save({'model':model.state_dict(),'opt':opt.state_dict(),'sched':sched.state_dict(),'epoch':ep,'best_va':best_va,'best_pva':best_pva,'best_pte':best_pte,'best_ep':best_ep,'offset':offset},tmp)
    os.replace(tmp,state_f)
bce=nn.BCEWithLogitsLoss()
ep=start
offset = st.get('offset',0) if 'st' in globals() else 0
print('DEBUG resume: start',start,'offset',offset,flush=True)
BUD=float(os.environ.get('BUDGET','27'))
stop=False
while ep<EPOCHS and not stop:
    g=torch.Generator().manual_seed(42+ep)
    perm=torch.randperm(ntr,generator=g)
    model.train()
    i=offset
    while i<ntr:
        if time.time()-t0>BUD:
            save_state(ep,offset=i); print('mid-epoch save ep',ep,'offset',i,'(%.1fs)'%(time.time()-t0),flush=True); stop=True; break
        b=perm[i:i+BATCH]
        loss=bce(model(Xtr[b],Ltr[b]),ytr[b])
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        try: sched.step()
        except ValueError: pass
        i+=BATCH
    if stop: break
    offset=0
    pva=pred(Xva,Lva); va=roc_auc_score(yva.numpy(),pva)
    if va>best_va:
        best_va=va; best_pva=pva; best_pte=pred(Xte,Lte); best_ep=ep+1
    ep+=1
    if ep-best_ep>=PATIENCE:
        print('early stop at',ep,'best epoch',best_ep); ep=EPOCHS
    save_state(ep)
    print('epoch',ep,'val_auroc %.5f best %.5f (%.1fs)'%(va,best_va,time.time()-t0),flush=True)

save_state(ep)
if ep>=EPOCHS:
    yv,yt=yva.numpy(),yte.numpy()
    best=(0.0,0.5)
    for thr in np.linspace(0.05,0.95,91):
        f=f1_score(yv,(best_pva>=thr).astype(int),zero_division=0)
        if f>best[0]: best=(float(f),float(thr))
    thr=best[1]; yp=(best_pte>=thr).astype(int)
    tn,fp,fn,tp=confusion_matrix(yt,yp,labels=[0,1]).ravel()
    res=dict(auroc=float(roc_auc_score(yt,best_pte)),f1=float(f1_score(yt,yp)),prec=float(precision_score(yt,yp)),
             rec=float(recall_score(yt,yp)),acc=float(accuracy_score(yt,yp)),tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn),thr=thr)
    rng=np.random.RandomState(42); ci={k:[] for k in ['auroc','f1','prec','rec','acc']}
    for b in range(2000):
        s=rng.randint(0,len(yt),len(yt)); ys,ps=yt[s],best_pte[s]
        if ys.min()==ys.max(): continue
        ypb=(ps>=thr).astype(int)
        ci['auroc'].append(roc_auc_score(ys,ps)); ci['f1'].append(f1_score(ys,ypb,zero_division=0))
        ci['prec'].append(precision_score(ys,ypb,zero_division=0)); ci['rec'].append(recall_score(ys,ypb,zero_division=0)); ci['acc'].append(accuracy_score(ys,ypb))
    res_ci={k:[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))] for k,v in ci.items()}
    json.dump({'test':res,'test_ci':res_ci,'best_val_auroc':best_va,'params':sum(p.numel() for p in model.parameters())},open(W/'nn'/f'final_{name}.json','w'),indent=1)
    np.save(W/f'scores_{name}.npy',best_pte)
    print('FINAL',name,res)
print('exit at epoch',ep)
