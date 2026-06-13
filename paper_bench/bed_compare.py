"""Bed few-shot recovery: DTS+HGB vs MiniROCKET under the SAME protocol.

Protocol mirrors research/bed_deepdive.py: Chair+Stand base training pool;
fixed Bed test set of 1,547 clips; 200-clip Bed adaptation pool; K labelled
Bed clips added (K in {0,25,50,100,200}); 5 seeds; AUROC on the fixed test.
Both models see identical base/pool/test index sets (rebuilt, validated
pipeline: DTS+HGB reproduces the paper's main-split 0.9895 exactly).

MiniROCKET: kernels fit once on the Chair+Stand base (bias quantiles use base
only); ridge with alpha=1e3 (the value selected on the main-split validation);
dual-ridge via a precomputed gram so each (K, seed) is a cheap solve.

Stages (resumable): minifit -> minigram -> sweep
Run: python3 bed_compare.py <budget_s>
"""
import json, os, pickle, sys, time
import numpy as np
from sklearn.metrics import roc_auc_score

D = "/tmp/seqtf"
t0 = time.time(); budget = float(sys.argv[1]) if len(sys.argv) > 1 else 32.0
meta = json.load(open(f"{D}/meta.json"))
scen = np.array(meta["scenario"]); y = np.array(meta["label"])
bed_idx = np.where(scen == "Bed")[0]
base_idx = np.where(scen != "Bed")[0]
assert len(bed_idx) == 1747, len(bed_idx)
rng0 = np.random.default_rng(2026)
perm = rng0.permutation(len(bed_idx))
pool = bed_idx[perm[:200]]
test = bed_idx[perm[200:]]
KS, SEEDS = [0, 25, 50, 100, 200], [0, 1, 2, 3, 4]
ALPHA = 1e3

def log(*a): print(f"[{time.time()-t0:5.1f}s]", *a, flush=True)

# ---- stage: minirocket transform of all clips (kernels fit on base) --------
if not os.path.exists(f"{D}/bedmini_all.npy"):
    from aeon.transformations.collection.convolution_based import MiniRocket
    X = np.load(f"{D}/X.npy", mmap_mode="r")
    if not os.path.exists(f"{D}/bedmini.pkl"):
        mr = MiniRocket(n_kernels=10000, random_state=42, n_jobs=4)
        mr.fit(np.ascontiguousarray(X[base_idx].transpose(0, 2, 1)))
        pickle.dump(mr, open(f"{D}/bedmini.pkl", "wb")); log("bed mini fit")
        if time.time() - t0 > budget: sys.exit(3)
    mr = pickle.load(open(f"{D}/bedmini.pkl", "rb"))
    parts = []
    n = len(X)
    M = np.lib.format.open_memmap(f"{D}/bedmini_all_tmp.npy", mode=("r+" if os.path.exists(f"{D}/bedmini_all_tmp.npy") else "w+"),
                                  dtype=np.float32, shape=(n, 9996))
    prog = np.load(f"{D}/bedmini_prog.npy") if os.path.exists(f"{D}/bedmini_prog.npy") else np.array(0)
    i = int(prog)
    while i < n and time.time() - t0 < budget:
        j = min(i + 1500, n)
        M[i:j] = np.asarray(mr.transform(np.ascontiguousarray(X[i:j].transpose(0, 2, 1))), np.float32)
        i = j; np.save(f"{D}/bedmini_prog.npy", np.array(i)); log("bedmini tx", i)
    if i < n: sys.exit(3)
    os.rename(f"{D}/bedmini_all_tmp.npy", f"{D}/bedmini_all.npy")

# ---- stage: standardize (base stats) + gram --------------------------------
if not os.path.exists(f"{D}/bedgram.npy"):
    M = np.load(f"{D}/bedmini_all.npy")
    mu = M[base_idx].mean(0, dtype=np.float64).astype(np.float32)
    sd = (M[base_idx].std(0, dtype=np.float64) + 1e-8).astype(np.float32)
    M -= mu; M /= sd
    G = M @ M.T
    np.save(f"{D}/bedgram.npy", G.astype(np.float32)); log("bed gram done", G.shape)
    del M, G
    if time.time() - t0 > budget: sys.exit(3)

# ---- stage: sweep -----------------------------------------------------------
G = np.load(f"{D}/bedgram.npy")
F = np.load(f"{D}/F.npy")  # DTS features
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
HP = dict(max_iter=600, learning_rate=0.2, l2_regularization=0.1)
res_path = f"{D}/bed_compare.json"
res = json.load(open(res_path)) if os.path.exists(res_path) else {}
yt = y[test]
for K in KS:
    for s in (SEEDS if K > 0 else [0]):
        key = f"K{K}_s{s}"
        if key in res: continue
        if time.time() - t0 > budget:
            json.dump(res, open(res_path, "w")); sys.exit(3)
        rng = np.random.default_rng(100 + s)
        add = rng.choice(pool, size=K, replace=False) if K > 0 else np.array([], int)
        tr = np.concatenate([base_idx, add]).astype(int)
        # MiniROCKET dual ridge
        Gtr = G[np.ix_(tr, tr)].astype(np.float64)
        Kte = G[np.ix_(test, tr)].astype(np.float64)
        yz = (y[tr] * 2 - 1).astype(np.float64)
        sol = np.linalg.solve(Gtr + ALPHA * np.eye(len(tr)), yz)
        auc_mr = roc_auc_score(yt, Kte @ sol)
        # DTS+HGB
        clf = HGB(random_state=20260610 + s, **HP).fit(F[tr], y[tr])
        auc_hgb = roc_auc_score(yt, clf.predict_proba(F[test])[:, 1])
        res[key] = dict(K=K, seed=s, minirocket=float(auc_mr), dts_hgb=float(auc_hgb))
        log(key, f"MR={auc_mr:.4f} HGB={auc_hgb:.4f}")
        json.dump(res, open(res_path, "w"))

# summary
print("\nK    DTS+HGB (mean+-sd)   MiniROCKET (mean+-sd)")
summary = {}
for K in KS:
    h = [v["dts_hgb"] for v in res.values() if v["K"] == K]
    m = [v["minirocket"] for v in res.values() if v["K"] == K]
    summary[K] = dict(hgb=[float(np.mean(h)), float(np.std(h))],
                      mr=[float(np.mean(m)), float(np.std(m))])
    print(f"{K:<4} {np.mean(h):.4f}+-{np.std(h):.4f}      {np.mean(m):.4f}+-{np.std(m):.4f}")
json.dump(summary, open(f"{D}/bed_compare_summary.json", "w"), indent=1)
print("DONE")
