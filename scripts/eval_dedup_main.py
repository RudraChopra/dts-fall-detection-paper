import json, numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix
W=Path('.')
y=np.load(W/'aaai/fallvision_dts128_features.npz',allow_pickle=True)['y_fv']
te=np.load(W/'split_mr.npz')['test']; mask=np.load(W/'test_clean_mask.npy')
yte=y[te][mask]
files={'LSTM':('nn/final_LSTM.json','scores_LSTM.npy'),'GRU':('nn/final_GRU.json','scores_GRU.npy'),
 'Transformer':('nn/final_Transformer.json','scores_Transformer.npy'),'SimpleST-GCN':('nn/final_SimpleST-GCN.json','scores_SimpleST-GCN.npy'),
 'DTS-Net':('nn/final_DTS-Net.json','scores_DTS-Net.npy'),'DTS+LR':('final_LR.json','scores_LR.npy'),
 'DTS+ET':('final_ET.json','scores_ET.npy'),'DTS+RF':('final_RF.json','scores_RF.npy'),'DTS+HGB':('final_HGB.json','scores_HGB.npy')}
import time; t0=time.time()
outf=W/'dedup_main.json'
out=json.load(open(outf)) if outf.exists() else {}
for name,(jf,sf) in files.items():
    if name in out: continue
    if time.time()-t0>25: print('TIME'); break
    j=json.load(open(W/jf)); thr=j['test']['thr']
    p=np.load(W/sf)[mask]
    yp=(p>=thr).astype(int)
    tn,fp,fn,tp=confusion_matrix(yte,yp,labels=[0,1]).ravel()
    res=dict(auroc=float(roc_auc_score(yte,p)),f1=float(f1_score(yte,yp)),prec=float(precision_score(yte,yp)),
             rec=float(recall_score(yte,yp)),acc=float(accuracy_score(yte,yp)),tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn),thr=thr)
    rng=np.random.RandomState(42); ci={k:[] for k in ['auroc','f1','prec','rec','acc']}
    for b in range(2000):
        s=rng.randint(0,len(yte),len(yte)); ys,ps=yte[s],p[s]
        if ys.min()==ys.max(): continue
        ypb=(ps>=thr).astype(int)
        ci['auroc'].append(roc_auc_score(ys,ps)); ci['f1'].append(f1_score(ys,ypb,zero_division=0))
        ci['prec'].append(precision_score(ys,ypb,zero_division=0)); ci['rec'].append(recall_score(ys,ypb,zero_division=0)); ci['acc'].append(accuracy_score(ys,ypb))
    out[name]={'test':res,'test_ci':{k:[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))] for k,v in ci.items()},
               'params':j.get('params'),'selected':j.get('selected')}
    json.dump(out,open(outf,'w'),indent=1)
    print(name,'auroc %.4f f1 %.3f fn %d'%(res['auroc'],res['f1'],res['fn']),flush=True)
