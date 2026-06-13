import json, csv, re, sys, numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix

W = Path('.'); rng = np.random.RandomState(42)
d = np.load(W/'aaai/fallvision_dts128_features.npz', allow_pickle=True)
X, y, scen = d['X_fv'], d['y_fv'], d['scenarios_fv']
man = list(csv.DictReader(open(W/'aaai/tables/fallvision_manifest.csv')))
clip_ids = [r['clip_id'] for r in man]
sess = np.array([re.search(r'_(\d+)_(?:keypoints|ketpoints)_csv', r['archive']).group(1) for r in man])
sm = np.load(W/'split_mr.npz')
itr,iva,ite = sm['train'], sm['val'], sm['test']
Xtr,ytr,Xva,yva,Xte,yte = X[itr],y[itr],X[iva],y[iva],X[ite],y[ite]

def thr_tune(yv,pv):
    best=(0.0,0.5)
    for thr in np.linspace(0.05,0.95,91):
        f=f1_score(yv,(pv>=thr).astype(int),zero_division=0)
        if f>best[0]: best=(float(f),float(thr))
    return best[1]
def test_eval(yt,pt,thr):
    yp=(pt>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(yt,yp,labels=[0,1]).ravel()
    return dict(auroc=float(roc_auc_score(yt,pt)),f1=float(f1_score(yt,yp)),prec=float(precision_score(yt,yp)),
        rec=float(recall_score(yt,yp)),acc=float(accuracy_score(yt,yp)),tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn),thr=float(thr))
def boot_ci(yt,pt,thr,B=2000):
    n=len(yt); out={k:[] for k in ['auroc','f1','prec','rec','acc']}
    for b in range(B):
        s=rng.randint(0,n,n); ys,ps=yt[s],pt[s]
        if ys.min()==ys.max(): continue
        yp=(ps>=thr).astype(int)
        out['auroc'].append(roc_auc_score(ys,ps)); out['f1'].append(f1_score(ys,yp,zero_division=0))
        out['prec'].append(precision_score(ys,yp,zero_division=0)); out['rec'].append(recall_score(ys,yp,zero_division=0))
        out['acc'].append(accuracy_score(ys,yp))
    return {k:[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))] for k,v in out.items()}
def make_model(name,p):
    if name=='LR': return make_pipeline(StandardScaler(), LogisticRegression(C=p['C'],max_iter=2000,random_state=42))
    if name=='ET': return ExtraTreesClassifier(n_estimators=p['n'],max_features=p['mf'],random_state=42,n_jobs=-1)
    if name=='RF': return RandomForestClassifier(n_estimators=p['n'],max_features=p['mf'],random_state=42,n_jobs=-1)
    if name=='HGB': return HistGradientBoostingClassifier(max_iter=p['it'],learning_rate=p['lr'],l2_regularization=p['l2'],random_state=42)
grids = {
 'LR':[{'C':c} for c in [0.05,0.1,0.5,1.0,2.0,10.0]],
 'ET':[{'n':n,'mf':mf} for n in [100,300,600] for mf in ['sqrt',0.25]],
 'RF':[{'n':n,'mf':mf} for n in [100,300,600] for mf in ['sqrt',0.25]],
 'HGB1':[{'it':it,'lr':lr,'l2':l2} for it in [100,300] for lr in [0.05,0.1,0.2] for l2 in [0.0,0.1]],
 'HGB2':[{'it':600,'lr':lr,'l2':l2} for lr in [0.05,0.1,0.2] for l2 in [0.0,0.1]],
}
stage = sys.argv[1]
if stage in grids:
    name = 'HGB' if stage.startswith('HGB') else stage
    rows=[]
    for p in grids[stage]:
        m=make_model(name,p); m.fit(Xtr,ytr); pva=m.predict_proba(Xva)[:,1]
        rows.append({'params':p,'val_auroc':float(roc_auc_score(yva,pva)),
                     'val_f1':float(f1_score(yva,(pva>=thr_tune(yva,pva)).astype(int)))})
        print(stage,p,rows[-1]['val_auroc'],flush=True)
    json.dump(rows, open(W/f'grid_{stage}.json','w'))
elif stage.startswith('final:'):
    name = stage.split(':')[1]
    rows=[]
    for f in W.glob(f'grid_{name}*.json'): rows += json.load(open(f))
    best=max(rows,key=lambda r:(round(r['val_auroc'],6),round(r['val_f1'],6)))
    m=make_model(name,best['params']); m.fit(Xtr,ytr)
    pva=m.predict_proba(Xva)[:,1]; pte=m.predict_proba(Xte)[:,1]
    thr=thr_tune(yva,pva); te=test_eval(yte,pte,thr); ci=boot_ci(yte,pte,thr)
    json.dump({'grid':rows,'selected':best,'test':te,'test_ci':ci}, open(W/f'final_{name}.json','w'), indent=1)
    np.save(W/f'scores_{name}.npy', pte)
    print('SELECTED',name,best['params']); print('TEST',te); print('CI',ci)
elif stage=='extras':
    hgb_p={'it':300,'lr':0.1,'l2':0.0}
    lfo={}
    for hold in ['Bed','Chair','Stand']:
        m=make_model('HGB',hgb_p); tr=scen!=hold; te_=scen==hold
        m.fit(X[tr],y[tr]); p=m.predict_proba(X[te_])[:,1]; yp=(p>=0.5).astype(int)
        lfo[hold]={'n':int(te_.sum()),'falls':int(y[te_].sum()),'f1':float(f1_score(y[te_],yp)),'auroc':float(roc_auc_score(y[te_],p))}
        print('LFO',hold,lfo[hold],flush=True)
    lso={}
    for s in ['1','2','3','4']:
        m=make_model('HGB',hgb_p); tr=sess!=s; te_=sess==s
        m.fit(X[tr],y[tr]); p=m.predict_proba(X[te_])[:,1]; yp=(p>=0.5).astype(int)
        lso[s]={'n':int(te_.sum()),'falls':int(y[te_].sum()),'f1':float(f1_score(y[te_],yp)),'auroc':float(roc_auc_score(y[te_],p)),'rec':float(recall_score(y[te_],yp))}
        print('LSO',s,lso[s],flush=True)
    # attention correlation check
    import csv as c2
    rows=list(c2.DictReader(open(W/'fv/tables/attention_importance.csv')))
    a=np.array([float(r['attention_mean']) for r in rows]); e=np.array([float(r['et_importance']) for r in rows])
    from scipy.stats import pearsonr
    r,pv=pearsonr(a,e); print('attention vs ET importance r=%.3f p=%.3f'%(r,pv))
    json.dump({'lfo':lfo,'lso':lso,'attn_r':float(r),'attn_p':float(pv)}, open(W/'extras.json','w'), indent=1)
print('OK', stage)
