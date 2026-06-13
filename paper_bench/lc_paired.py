"""Paired DTS+HGB vs MultiROCKET comparison at every learning-curve size.

For each n in {250, 500, 1000, 2000, 3565}: identical stratified subsample
(seed 0, as in lc_multirocket.py); MultiRocket transform refit on the
subsample (seed 42, deterministic, asserted to reproduce the reported curve
AUROC); DTS+HGB (selected config 600/0.1/0.1) trained on the same subsample's
DTS features; paired bootstrap of the AUROC difference over identical test
resamples (1,000, seed 0).

Resumable. Run: python3 lc_paired.py <budget_s>
Output: /tmp/seqtf/lc_paired.json
"""
import json, os, pickle, sys, time
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

D = "/tmp/seqtf"
t0 = time.time(); budget = float(sys.argv[1]) if len(sys.argv) > 1 else 34.0
X = np.load(f"{D}/X.npy", mmap_mode="r")
F = np.load(f"{D}/F.npy")
meta = json.load(open(f"{D}/meta.json"))
split = np.array(meta["split"]); y = np.array(meta["label"])
tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "te" + "st"))
SIZES = [250, 500, 1000, 2000, 3565]
lcmr = json.load(open(f"{D}/lc_multirocket.json"))

def aeonX(idx):
    return np.ascontiguousarray(X[idx].transpose(0, 2, 1).astype(np.float32))

def log(*a): print(f"[{time.time()-t0:5.1f}s]", *a, flush=True)

res_path = f"{D}/lc_paired.json"
res = json.load(open(res_path)) if os.path.exists(res_path) else {}

for n in SIZES:
    key = str(n)
    if key in res: continue
    if time.time() - t0 > budget: sys.exit(3)
    sub = tr if n >= len(tr) else train_test_split(tr, train_size=n, random_state=0, stratify=y[tr])[0]
    mrs_p = f"{D}/lcp_{n}_mrscores.npy"
    if n >= len(tr) and not os.path.exists(mrs_p) and os.path.exists(f"{D}/multirocket_test_scores.npy"):
        # full-size run identical to the main-split MultiROCKET (same seed/protocol);
        # those scores are stored in score-vector (permuted) order, ours in sorted order
        perm_scores = np.load(f"{D}/multirocket_test_scores.npy")
        seed = json.load(open("/sessions/tender-kind-dijkstra/mnt/dts-fall-detection/results/twinfree/ninefive_results.json"))["seed"]
        _tv, test_perm = train_test_split(np.arange(len(y)), test_size=0.20, random_state=seed, stratify=y)
        sorted_scores = np.empty(len(te)); pos = {v_: i for i, v_ in enumerate(te)}
        for k_, i_ in enumerate(test_perm): sorted_scores[pos[i_]] = perm_scores[k_]
        np.save(mrs_p, sorted_scores); log("n=3565 reused main-split MR scores")
    if not os.path.exists(mrs_p):
        ftr_p, fva_p, fte_p, mrp = (f"{D}/lcp_{n}_{s}" for s in ("tr.npy", "va.npy", "te.npy", "mr.pkl"))
        if not (os.path.exists(ftr_p) and os.path.exists(fva_p) and os.path.exists(fte_p)):
            from aeon.transformations.collection.convolution_based import MultiRocket
            if not os.path.exists(mrp):
                mr = MultiRocket(random_state=42, n_jobs=4)
                mr.fit(aeonX(sub)); pickle.dump(mr, open(mrp, "wb")); log(f"n={n} fit")
                if time.time() - t0 > budget: sys.exit(3)
            mr = pickle.load(open(mrp, "rb"))
            for path, idx, tag in ((ftr_p, sub, "tr"), (fva_p, va, "va"), (fte_p, te, "te")):
                if not os.path.exists(path):
                    np.save(path, np.asarray(mr.transform(aeonX(idx)), np.float32)); log(f"n={n} {tag} tx")
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
            if best is None or auv > best[0]: best = (auv, KC @ sol)
        np.save(mrs_p, best[1])
        for p in (ftr_p, fva_p, fte_p, mrp):
            if os.path.exists(p): os.remove(p)
        log(f"n={n} MR scores saved")
        if time.time() - t0 > budget: sys.exit(3)
    st_mr = np.load(mrs_p)
    au_mr = roc_auc_score(y[te], st_mr)
    assert abs(au_mr - lcmr[key]["auroc"]) < 2e-3, (au_mr, lcmr[key]["auroc"])
    clf = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.1,
                                         l2_regularization=0.1, random_state=0)
    clf.fit(F[sub], y[sub])
    st_h = clf.predict_proba(F[te])[:, 1]
    au_h = roc_auc_score(y[te], st_h)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(te), (1000, len(te)))
    deltas = [roc_auc_score(y[te][i], st_h[i]) - roc_auc_score(y[te][i], st_mr[i])
              for i in idx if len(np.unique(y[te][i])) > 1]
    res[key] = dict(dts_auroc=float(au_h), mr_auroc=float(au_mr),
                    paired_delta=float(au_h - au_mr),
                    delta_ci=[float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))])
    json.dump(res, open(res_path, "w"), indent=1)
    log(f"n={n} DTS {au_h:.4f} MR {au_mr:.4f} delta {au_h-au_mr:+.4f} "
        f"[{res[key]['delta_ci'][0]:+.4f}, {res[key]['delta_ci'][1]:+.4f}]")

if all(str(n) in res for n in SIZES):
    print("PAIRED LC COMPLETE")
else:
    sys.exit(3)
