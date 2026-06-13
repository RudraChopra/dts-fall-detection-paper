import sys, time, json, pickle, numpy as np
from pathlib import Path
W=Path('.'); t0=time.time()
from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import roc_auc_score, f1_score
d=np.load(W/'seq_all.npz'); X,L=d['X'],d['L']
y=np.load(W/'aaai/fallvision_dts128_features.npz',allow_pickle=True)['y_fv']
sm=np.load(W/'split_mr.npz')
Xa = X.transpose(0,2,1).astype(np.float32)  # (n, channels, time) for aeon
stage=sys.argv[1]; nk=int(sys.argv[2]) if len(sys.argv)>2 else 10000
if stage=='fit':
    mr=MiniRocket(n_kernels=nk, random_state=42, n_jobs=4)
    Xtr=mr.fit_transform(Xa[sm['train']])
    print('fit+transform train', Xtr.shape, '%.1fs'%(time.time()-t0))
    pickle.dump(mr, open(W/f'mr_{nk}.pkl','wb'))
    np.save(W/f'mrtr_{nk}.npy', np.asarray(Xtr, dtype=np.float32))
elif stage=='tx':
    mr=pickle.load(open(W/f'mr_{nk}.pkl','rb'))
    for part in ['val','test']:
        Xt=mr.transform(Xa[sm[part]])
        np.save(W/f'mr{part}_{nk}.npy', np.asarray(Xt,dtype=np.float32))
        print('transform', part, '%.1fs'%(time.time()-t0), flush=True)
elif stage=='ridge':
    Xtr=np.load(W/f'mrtr_{nk}.npy'); Xva=np.load(W/f'mrval_{nk}.npy'); Xte=np.load(W/f'mrtest_{nk}.npy')
    sc=StandardScaler().fit(Xtr)
    clf=RidgeClassifierCV(alphas=np.logspace(-3,3,10)).fit(sc.transform(Xtr), y[sm['train']])
    sva=clf.decision_function(sc.transform(Xva)); ste=clf.decision_function(sc.transform(Xte))
    np.save(W/f'mr_sva_{nk}.npy',sva); np.save(W/f'mr_ste_{nk}.npy',ste)
    va_auc=roc_auc_score(y[sm['val']],sva)
    print(nk,'val AUROC %.4f'%va_auc,'%.1fs'%(time.time()-t0))
    j=json.load(open(W/'mr_grid.json')) if (W/'mr_grid.json').exists() else {}
    j[str(nk)]=float(va_auc); json.dump(j,open(W/'mr_grid.json','w'))
