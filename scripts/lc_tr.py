import sys, os, time, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score
t0=time.time(); torch.set_num_threads(4)
sys.path.insert(0,'/sessions/compassionate-stoic-wozniak/mnt/files-mentioned-by-the-user-paper/dts-fall-detection')
class FallTransformer(nn.Module):
    def __init__(self, input_dim=34, d_model=64, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        self.proj=nn.Linear(input_dim,d_model)
        layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=128,dropout=dropout,batch_first=True)
        self.enc=nn.TransformerEncoder(layer,num_layers=layers)
        self.fc=nn.Sequential(nn.Dropout(dropout),nn.Linear(d_model,1))
    def forward(self,x,lengths):
        b,t,_=x.shape
        h=self.proj(x)+ (torch.arange(t,dtype=torch.float32)[None,:,None]/100.0)
        mask=torch.arange(t)[None,:]>=lengths[:,None]
        out=self.enc(h,src_key_padding_mask=mask)
        valid=(~mask).float().unsqueeze(-1)
        return self.fc((out*valid).sum(1)/valid.sum(1).clamp(min=1)).squeeze(-1)
d=np.load('lc_data.npz'); S,L,y,sp=d['S'],d['L'],d['y'],d['sp']
iva,ite=sp=='val',sp=='test'
Xva,Lva,yva=torch.tensor(S[iva]),torch.tensor(L[iva]),y[iva]
Xte,Lte=torch.tensor(S[ite]),torch.tensor(L[ite]); yte=y[ite]
subs=np.load('lc_subs.npz'); n=sys.argv[1]; tr=subs[n]
Xtr,Ltr,ytr=torch.tensor(S[tr]),torch.tensor(L[tr]),torch.tensor(y[tr].astype(np.float32))
EP=20; B=64; sf=Path(f'lc_tr_{n}.pt')
torch.manual_seed(20260610)
model=FallTransformer(); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
ep=0; best=-1; best_pte=None
if sf.exists():
    st=torch.load(sf,weights_only=False); model.load_state_dict(st['m']); opt.load_state_dict(st['o'])
    ep=st['ep']; best=st['best']; best_pte=st['pte']
def pred(X,Ln,bs=256):
    model.eval(); o=[]
    with torch.no_grad():
        for i in range(0,len(X),bs): o.append(torch.sigmoid(model(X[i:i+bs],Ln[i:i+bs])).numpy())
    return np.concatenate(o)
bce=nn.BCEWithLogitsLoss()
while ep<EP and time.time()-t0<28:
    g=torch.Generator().manual_seed(20260610+ep); perm=torch.randperm(len(ytr),generator=g)
    model.train()
    for i in range(0,len(ytr),B):
        b=perm[i:i+B]
        loss=bce(model(Xtr[b],Ltr[b]),ytr[b])
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    pva=pred(Xva,Lva); a=roc_auc_score(yva,pva)
    if a>best: best=a; best_pte=pred(Xte,Lte)
    ep+=1
    tmp=str(sf)+'.tmp'; torch.save({'m':model.state_dict(),'o':opt.state_dict(),'ep':ep,'best':best,'pte':best_pte},tmp); os.replace(tmp,sf)
    print(f'n={n} ep{ep} val {a:.4f} best {best:.4f} ({time.time()-t0:.0f}s)',flush=True)
if ep>=EP:
    out=json.load(open('lc.json')); out.setdefault('Transformer',{})
    rng=np.random.RandomState(42); cis=[]
    for b in range(1000):
        s=rng.randint(0,len(yte),len(yte))
        if yte[s].min()==yte[s].max(): continue
        cis.append(roc_auc_score(yte[s],best_pte[s]))
    out['Transformer'][n]={'auroc':float(roc_auc_score(yte,best_pte)),'ci':[float(np.percentile(cis,2.5)),float(np.percentile(cis,97.5))]}
    json.dump(out,open('lc.json','w'))
    print('FINAL',n,round(out['Transformer'][n]['auroc'],4))
