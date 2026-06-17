"""Experiment A extension: add MiniROCKET to the cache-based Bed few-shot
comparison so Figure 2 carries three consistent curves (DTS+HGB, MiniROCKET,
MultiROCKET) on the identical protocol. Mirrors expA_driver.py exactly, with
MiniRocket(10000) in place of MultiRocket.

Resumable stages (state in /tmp/bedrec):  fit -> tx -> gram -> sweep
Run:  python3 expA_minirocket.py <stage> <budget_seconds>
"""
import sys, os, time, json, pickle
import numpy as np
D = "/tmp/bedrec"
KS = [0, 25, 50, 100, 200]; SEEDS = [0, 1, 2, 3, 4]; ALPHA = 1e3

def bed_split():
    m = np.load(f"{D}/meta.npz", allow_pickle=True)
    label, scen = m['label'], m['scen'].astype(str)
    idx = np.load(f"{D}/dedup_idx.npy"); y = label[idx]; sc = scen[idx]
    bed = np.where(sc == 'Bed')[0]; base = np.where(sc != 'Bed')[0]
    perm = np.random.default_rng(2026).permutation(len(bed))
    return idx, y, bed, base, bed[perm[:200]], bed[perm[200:]]

def stage_fit(budget):
    from aeon.transformations.collection.convolution_based import MiniRocket
    X = np.load(f"{D}/X.npy", mmap_mode="r"); idx, y, bed, base, pool, test = bed_split()
    mr = MiniRocket(n_kernels=10000, random_state=42, n_jobs=4)
    mr.fit(np.ascontiguousarray(X[idx[base]].transpose(0, 2, 1)))
    pickle.dump(mr, open(f"{D}/mini.pkl", "wb")); print("mini fit done"); return 0

def stage_tx(budget):
    X = np.load(f"{D}/X.npy", mmap_mode="r"); idx = np.load(f"{D}/dedup_idx.npy")
    mr = pickle.load(open(f"{D}/mini.pkl", "rb"))
    nf = np.asarray(mr.transform(np.ascontiguousarray(X[idx[:2]].transpose(0, 2, 1)))).shape[1]
    M = np.lib.format.open_memmap(f"{D}/Mmini.npy", mode=("r+" if os.path.exists(f"{D}/Mmini.npy") else "w+"),
                                  dtype=np.float32, shape=(len(idx), nf))
    prog = int(np.load(f"{D}/txmini_prog.npy")) if os.path.exists(f"{D}/txmini_prog.npy") else 0
    t0 = time.time(); i = prog
    while i < len(idx) and time.time() - t0 < budget:
        j = min(i + 750, len(idx))
        M[i:j] = np.asarray(mr.transform(np.ascontiguousarray(X[idx[i:j]].transpose(0, 2, 1))), np.float32)
        i = j; np.save(f"{D}/txmini_prog.npy", np.array(i))
    M.flush(); print("mini tx", i); return 0 if i >= len(idx) else 3

def stage_gram(budget):
    idx, y, bed, base, pool, test = bed_split()
    M = np.load(f"{D}/Mmini.npy", mmap_mode="r")
    Mb = np.asarray(M[base], np.float64); mu = Mb.mean(0); sd = Mb.std(0) + 1e-8; del Mb
    A = ((np.asarray(M, np.float32) - mu.astype(np.float32)) / sd.astype(np.float32))
    np.save(f"{D}/gram_mini.npy", (A @ A.T).astype(np.float32)); print("mini gram done"); return 0

def stage_sweep(budget):
    from sklearn.metrics import roc_auc_score
    idx, y, bed, base, pool, test = bed_split()
    G = np.load(f"{D}/gram_mini.npy"); yt = y[test]
    res_p = f"{D}/results_mini.json"; res = json.load(open(res_p)) if os.path.exists(res_p) else {}
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
            res[key] = dict(K=K, seed=s, minirocket=float(roc_auc_score(yt, Kte @ sol)))
            json.dump(res, open(res_p, "w"))
    summ = {str(K): float(np.mean([v['minirocket'] for v in res.values() if v['K'] == K])) for K in KS}
    json.dump(summ, open(f"{D}/summary_mini.json", "w"), indent=1)
    print("mini sweep done", summ); return 0

if __name__ == "__main__":
    st = sys.argv[1]; bud = float(sys.argv[2]) if len(sys.argv) > 2 else 38.0
    sys.exit({"fit": lambda: stage_fit(bud), "tx": lambda: stage_tx(bud),
              "gram": lambda: stage_gram(bud), "sweep": lambda: stage_sweep(bud)}[st]())
