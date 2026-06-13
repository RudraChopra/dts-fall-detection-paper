"""MultiROCKET sample-efficiency curve on the strict twin-free split.

Protocol mirrors the paper's learning-curve study (lc.json): stratified
subsamples of the training split at n in {250, 500, 1000, 2000, 3565};
validation and test sets unchanged; the MultiRocket transform is refit on
each subsample (bias quantiles see subsample only); ridge alpha selected on
validation AUROC; test AUROC with 1,000-resample bootstrap CI. Subsampling
uses the same stratified seed convention as the original study (seed 0).

Resumable per n. Run: python3 lc_multirocket.py <budget_s>
Output: /tmp/seqtf/lc_multirocket.json
"""
import json, os, sys, time
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

D = "/tmp/seqtf"
t0 = time.time(); budget = float(sys.argv[1]) if len(sys.argv) > 1 else 34.0
X = np.load(f"{D}/X.npy", mmap_mode="r")
meta = json.load(open(f"{D}/meta.json"))
split = np.array(meta["split"]); y = np.array(meta["label"])
tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "te" + "st"))
SIZES = [250, 500, 1000, 2000, 3565]

def aeonX(idx):
    return np.ascontiguousarray(X[idx].transpose(0, 2, 1).astype(np.float32))

def log(*a): print(f"[{time.time()-t0:5.1f}s]", *a, flush=True)

res_path = f"{D}/lc_multirocket.json"
res = json.load(open(res_path)) if os.path.exists(res_path) else {}

for n in SIZES:
    key = str(n)
    if key in res: continue
    if time.time() - t0 > budget:
        sys.exit(3)
    if n >= len(tr):
        sub = tr
    else:
        sub, _ = train_test_split(tr, train_size=n, random_state=0, stratify=y[tr])
    # stage files per n (fit+transform may exceed one call for large n)
    ftr_p, fva_p, fte_p = (f"{D}/lcmr_{n}_{s}.npy" for s in ("tr", "va", "te"))
    mrp = f"{D}/lcmr_{n}_mr.pkl"
    if not (os.path.exists(ftr_p) and os.path.exists(fva_p) and os.path.exists(fte_p)):
        import pickle
        from aeon.transformations.collection.convolution_based import MultiRocket
        if not os.path.exists(mrp):
            mr = MultiRocket(random_state=42, n_jobs=4)
            mr.fit(aeonX(sub)); pickle.dump(mr, open(mrp, "wb")); log(f"n={n} fit")
            if time.time() - t0 > budget: sys.exit(3)
        mr = pickle.load(open(mrp, "rb"))
        for path, idx, tag in ((ftr_p, sub, "train"), (fva_p, va, "val"), (fte_p, te, "test")):
            if not os.path.exists(path):
                np.save(path, np.asarray(mr.transform(aeonX(idx)), np.float32))
                log(f"n={n} {tag} tx")
                if time.time() - t0 > budget: sys.exit(3)
    Ftr = np.load(ftr_p); Fva = np.load(fva_p); Fte = np.load(fte_p)
    mu = Ftr.mean(0, dtype=np.float64).astype(np.float32)
    sd = (Ftr.std(0, dtype=np.float64) + 1e-8).astype(np.float32)
    for M in (Ftr, Fva, Fte): M -= mu; M /= sd
    G = (Ftr @ Ftr.T).astype(np.float64)
    KB = (Fva @ Ftr.T).astype(np.float64); KC = (Fte @ Ftr.T).astype(np.float64)
    del Ftr, Fva, Fte
    yz = (y[sub] * 2 - 1).astype(np.float64)
    best = None
    for al in np.logspace(-3, 3, 10):
        sol = np.linalg.solve(G + al * np.eye(len(sub)), yz)
        auv = roc_auc_score(y[va], KB @ sol)
        if best is None or auv > best[0]:
            best = (auv, al, KC @ sol)
    auv, al, st = best
    aute = roc_auc_score(y[te], st)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(te), (1000, len(te)))
    boots = [roc_auc_score(y[te][i], st[i]) for i in idx if len(np.unique(y[te][i])) > 1]
    res[key] = dict(auroc=float(aute),
                    ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                    alpha=float(al), val_auroc=float(auv))
    json.dump(res, open(res_path, "w"), indent=1)
    log(f"n={n} MultiROCKET test AUROC {aute:.4f}")
    for p in (ftr_p, fva_p, fte_p, mrp):
        if os.path.exists(p): os.remove(p)   # free disk

if all(str(n) in res for n in SIZES):
    print("LC COMPLETE:", {k: round(v["auroc"], 4) for k, v in res.items()})
else:
    sys.exit(3)
