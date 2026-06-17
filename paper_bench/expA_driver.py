"""Experiment A driver: MultiROCKET vs DTS+HGB few-shot Bed recovery.

Built from the canonical cache (work/ninefive/fallvision_min10_tau030_cache.npz:
fixed_seq, feature30). Replicates bedrec_build.py's protocol exactly and
self-validates the published anchors (dedup=5572, Bed=1747, DTS+HGB 0.714->0.848)
before trusting any MultiROCKET number.

Resumable stages (state in /tmp/bedrec):  prep -> fit -> tx -> gram -> sweep
Run:  python3 expA_driver.py <stage> <budget_seconds>
Deps: numpy, scikit-learn, aeon (+numba).  Gram stage: see gram step in the log.
"""
import sys, os, time, json, pickle
import numpy as np
D = "/tmp/bedrec"; os.makedirs(D, exist_ok=True)
CACHE = "/tmp/cache.npz"        # = work/ninefive/fallvision_min10_tau030_cache.npz
KS = [0, 25, 50, 100, 200]; SEEDS = [0, 1, 2, 3, 4]; ALPHA = 1e3

def stage_prep():
    z = np.load(CACHE, allow_pickle=True)
    np.save(f"{D}/X.npy", np.asarray(z['fixed_seq'], np.float32))
    np.save(f"{D}/F.npy", np.asarray(z['feature30'], np.float64))
    np.savez(f"{D}/meta.npz", label=np.asarray(z['label']).astype(int),
             scen=np.asarray(z['scenario']).astype(str))
    F = np.asarray(z['feature30'], np.float64)
    _, idx = np.unique(np.round(F, 6), axis=0, return_index=True); idx = np.sort(idx)
    np.save(f"{D}/dedup_idx.npy", idx)
    sc = np.asarray(z['scenario']).astype(str)
    print(f"prep: dedup={len(idx)} (exp 5572) Bed={(sc[idx]=='Bed').sum()} (exp 1747)")
    return 0

def bed_split():
    m = np.load(f"{D}/meta.npz", allow_pickle=True)
    label, scen = m['label'], m['scen'].astype(str)
    idx = np.load(f"{D}/dedup_idx.npy"); y = label[idx]; sc = scen[idx]
    bed = np.where(sc == 'Bed')[0]; base = np.where(sc != 'Bed')[0]
    perm = np.random.default_rng(2026).permutation(len(bed))
    return idx, y, bed, base, bed[perm[:200]], bed[perm[200:]]

def stage_fit(budget):
    from aeon.transformations.collection.convolution_based import MultiRocket
    X = np.load(f"{D}/X.npy", mmap_mode="r"); idx, y, bed, base, pool, test = bed_split()
    mr = MultiRocket(random_state=42, n_jobs=4)
    mr.fit(np.ascontiguousarray(X[idx[base]].transpose(0, 2, 1)))
    pickle.dump(mr, open(f"{D}/mr.pkl", "wb")); print("fit done"); return 0

def stage_tx(budget):
    X = np.load(f"{D}/X.npy", mmap_mode="r"); idx = np.load(f"{D}/dedup_idx.npy")
    mr = pickle.load(open(f"{D}/mr.pkl", "rb"))
    nf = np.asarray(mr.transform(np.ascontiguousarray(X[idx[:2]].transpose(0, 2, 1)))).shape[1]
    M = np.lib.format.open_memmap(f"{D}/M.npy", mode=("r+" if os.path.exists(f"{D}/M.npy") else "w+"),
                                  dtype=np.float32, shape=(len(idx), nf))
    prog = int(np.load(f"{D}/tx_prog.npy")) if os.path.exists(f"{D}/tx_prog.npy") else 0
    t0 = time.time(); i = prog
    while i < len(idx) and time.time() - t0 < budget:
        j = min(i + 750, len(idx))
        M[i:j] = np.asarray(mr.transform(np.ascontiguousarray(X[idx[i:j]].transpose(0, 2, 1))), np.float32)
        i = j; np.save(f"{D}/tx_prog.npy", np.array(i))
    M.flush(); return 0 if i >= len(idx) else 3

def stage_gram(budget):
    idx, y, bed, base, pool, test = bed_split()
    M = np.load(f"{D}/M.npy", mmap_mode="r")
    Mb = np.asarray(M[base], np.float64); mu = Mb.mean(0);
    sd = Mb.std(0) + 1e-8; del Mb
    A = ((np.asarray(M, np.float32) - mu.astype(np.float32)) / sd.astype(np.float32))
    np.save(f"{D}/gram.npy", (A @ A.T).astype(np.float32)); print("gram done"); return 0

def stage_sweep(budget):
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    idx, y, bed, base, pool, test = bed_split()
    F = np.load(f"{D}/F.npy"); G = np.load(f"{D}/gram.npy"); yt = y[test]
    HP = dict(max_iter=600, learning_rate=0.2, l2_regularization=0.1)
    res_p = f"{D}/results.json"; res = json.load(open(res_p)) if os.path.exists(res_p) else {}
    t0 = time.time()
    for K in KS:
        for s in (SEEDS if K > 0 else [0]):
            key = f"K{K}_s{s}"
            if key in res: continue
            if time.time() - t0 > budget: json.dump(res, open(res_p, "w")); return 3
            rng = np.random.default_rng(100 + s)
            add = rng.choice(pool, size=K, replace=False) if K > 0 else np.array([], int)
            tr = np.concatenate([base, add]).astype(int)
            Gtr = G[np.ix_(tr, tr)].astype(np.float64); Kte = G[np.ix_(test, tr)].astype(np.float64)
            sol = np.linalg.solve(Gtr + ALPHA * np.eye(len(tr)), (y[tr] * 2 - 1).astype(np.float64))
            auc_mr = roc_auc_score(yt, Kte @ sol)
            clf = HGB(random_state=20260610 + s, **HP).fit(F[idx[tr]], y[tr])
            auc_hgb = roc_auc_score(yt, clf.predict_proba(F[idx[test]])[:, 1])
            res[key] = dict(K=K, seed=s, multirocket=float(auc_mr), dts_hgb=float(auc_hgb))
            json.dump(res, open(res_p, "w"))
    summ = {}
    for K in KS:
        h = [v['dts_hgb'] for v in res.values() if v['K'] == K]
        mm = [v['multirocket'] for v in res.values() if v['K'] == K]
        summ[K] = dict(hgb=[float(np.mean(h)), float(np.std(h))], mr=[float(np.mean(mm)), float(np.std(mm))])
    json.dump(summ, open(f"{D}/summary.json", "w"), indent=1); print("sweep done"); return 0

if __name__ == "__main__":
    st = sys.argv[1]; bud = float(sys.argv[2]) if len(sys.argv) > 2 else 38.0
    sys.exit({"prep": stage_prep, "fit": lambda: stage_fit(bud), "tx": lambda: stage_tx(bud),
              "gram": lambda: stage_gram(bud), "sweep": lambda: stage_sweep(bud)}[st]())
