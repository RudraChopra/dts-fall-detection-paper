"""MiniROCKET validation + MultiROCKET main-split run on the rebuilt strict
twin-free sequences. Staged + resumable: each invocation does the next pending
stage within a time budget. State in /tmp/seqtf/.

Stages:
  mini_fit   fit MiniRocket(10k, seed 42) on train, transform train/val/test
  mini_eval  ridge (val-selected alpha), test AUROC vs paper 0.9760
  multi_fit  fit MultiRocket(seed 42) on train
  multi_tx   transform train/val/test (chunked)
  multi_eval dual ridge via gram (val-selected alpha), threshold on val F1,
             full test metrics + bootstrap CIs + paired delta vs DTS+HGB
Run: python3 rocket_main.py <budget_s>
"""
import json, os, pickle, sys, time
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

D = "/tmp/seqtf"
REPO = "/sessions/tender-kind-dijkstra/mnt/dts-fall-detection"
t0 = time.time()
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 34.0

X = np.load(f"{D}/X.npy", mmap_mode="r")
meta = json.load(open(f"{D}/meta.json"))
split = np.array(meta["split"]); y = np.array(meta["label"])
tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))
assert len(tr) == 3565 and len(va) == 892 and len(te) == 1115

def aeonX(idx):
    return np.ascontiguousarray(X[idx].transpose(0, 2, 1).astype(np.float32))

def log(*a): print(f"[{time.time()-t0:5.1f}s]", *a, flush=True)

def dual_ridge_scores(Ftr, ytr, Fva, Fte, alphas):
    """Centered+scaled dual ridge; returns dict alpha -> (val_scores, test_scores)."""
    mu = Ftr.mean(0); sd = Ftr.std(0) + 1e-8
    A = (Ftr - mu) / sd
    B = (Fva - mu) / sd
    C = (Fte - mu) / sd
    yz = (ytr * 2 - 1).astype(np.float64)
    G = A @ A.T
    KB = B @ A.T; KC = C @ A.T
    out = {}
    n = len(A)
    for al in alphas:
        sol = np.linalg.solve(G + al * np.eye(n), yz)
        out[al] = (KB @ sol, KC @ sol)
    return out

def stage_done(name): return os.path.exists(f"{D}/{name}.done")
def mark(name): open(f"{D}/{name}.done", "w").write("1")

# ---------------- MiniROCKET (validation of the rebuilt sequences) ----------
if not stage_done("mini"):
    from aeon.transformations.collection.convolution_based import MiniRocket
    if not os.path.exists(f"{D}/mini.pkl"):
        mr = MiniRocket(n_kernels=10000, random_state=42, n_jobs=4)
        mr.fit(aeonX(tr)); pickle.dump(mr, open(f"{D}/mini.pkl", "wb")); log("mini fit")
        if time.time() - t0 > budget: sys.exit(3)
    mr = pickle.load(open(f"{D}/mini.pkl", "rb"))
    for name, idx in (("tr", tr), ("va", va), ("te", te)):
        if not os.path.exists(f"{D}/mini_{name}.npy"):
            np.save(f"{D}/mini_{name}.npy", np.asarray(mr.transform(aeonX(idx)), np.float32))
            log(f"mini {name} tx")
            if time.time() - t0 > budget: sys.exit(3)
    Ftr = np.load(f"{D}/mini_tr.npy").astype(np.float64)
    Fva = np.load(f"{D}/mini_va.npy"); Fte = np.load(f"{D}/mini_te.npy")
    res = dual_ridge_scores(Ftr, y[tr], Fva, Fte, np.logspace(-3, 3, 10))
    best, sbest = None, None
    for al, (sv, st) in res.items():
        auv = roc_auc_score(y[va], sv)
        if best is None or auv > best[1]: best, sbest = (al, auv), (sv, st)
    aute = roc_auc_score(y[te], sbest[1])
    log(f"MiniROCKET val-alpha={best[0]:.3g} valAUROC={best[1]:.4f} testAUROC={aute:.4f} (paper 0.9760)")
    json.dump(dict(alpha=best[0], val_auroc=best[1], test_auroc=aute),
              open(f"{D}/mini_eval.json", "w"))
    mark("mini")
    if time.time() - t0 > budget: sys.exit(3)

# ---------------- MultiROCKET ------------------------------------------------
if not stage_done("multi_fit"):
    from aeon.transformations.collection.convolution_based import MultiRocket
    mr = MultiRocket(random_state=42, n_jobs=4)
    Xtr = aeonX(tr); log("fitting MultiRocket")
    Ftr = np.asarray(mr.fit_transform(Xtr), np.float32); log("multi train tx", Ftr.shape)
    np.save(f"{D}/multi_tr.npy", Ftr); del Ftr, Xtr
    pickle.dump(mr, open(f"{D}/multi.pkl", "wb"))
    mark("multi_fit")
    if time.time() - t0 > budget: sys.exit(3)

if not stage_done("multi_tx"):
    from aeon.transformations.collection.convolution_based import MultiRocket
    mr = pickle.load(open(f"{D}/multi.pkl", "rb"))
    if not os.path.exists(f"{D}/multi_va.npy"):
        np.save(f"{D}/multi_va.npy", np.asarray(mr.transform(aeonX(va)), np.float32)); log("multi val tx")
        if time.time() - t0 > budget: sys.exit(3)
    np.save(f"{D}/multi_te.npy", np.asarray(mr.transform(aeonX(te)), np.float32)); log("multi test tx")
    mark("multi_tx")
    if time.time() - t0 > budget: sys.exit(3)

if not stage_done("multi_eval"):
    Ftr = np.load(f"{D}/multi_tr.npy")          # float32, ~710 MB
    Fva = np.load(f"{D}/multi_va.npy")
    Fte = np.load(f"{D}/multi_te.npy")
    log("loaded multi features", Ftr.shape)
    mu = Ftr.mean(0, dtype=np.float64).astype(np.float32)
    sd = (Ftr.std(0, dtype=np.float64) + 1e-8).astype(np.float32)
    for M in (Ftr, Fva, Fte):
        M -= mu; M /= sd                         # in-place, no copies
    G  = (Ftr @ Ftr.T).astype(np.float64); log("gram done")
    KB = (Fva @ Ftr.T).astype(np.float64)
    KC = (Fte @ Ftr.T).astype(np.float64)
    del Ftr, Fva, Fte
    yz = (y[tr] * 2 - 1).astype(np.float64)
    res = {}
    for al in np.logspace(-3, 3, 10):
        sol = np.linalg.solve(G + al * np.eye(len(tr)), yz)
        res[al] = (KB @ sol, KC @ sol)
    log("ridge sweep done")
    best, sbest = None, None
    for al, (sv, st) in res.items():
        auv = roc_auc_score(y[va], sv)
        if best is None or auv > best[1]: best, sbest = (al, auv), (sv, st)
    sv, st = sbest
    # probability-like squash for threshold grid comparability
    pva = 1/(1+np.exp(-sv)); pte = 1/(1+np.exp(-st))
    grid = np.linspace(0.01, 0.99, 99)
    thr = grid[int(np.argmax([f1_score(y[va], pva >= g) for g in grid]))]
    pred = (pte >= thr).astype(int)
    tp = int(((pred==1)&(y[te]==1)).sum()); fp = int(((pred==1)&(y[te]==0)).sum())
    fn = int(((pred==0)&(y[te]==1)).sum()); tn = int(((pred==0)&(y[te]==0)).sum())
    prec, rec = tp/max(tp+fp,1), tp/(tp+fn)
    out = dict(alpha=best[0], val_auroc=best[1],
               auroc=roc_auc_score(y[te], st), f1=2*prec*rec/max(prec+rec,1e-9),
               precision=prec, recall=rec, accuracy=(tp+tn)/len(te),
               tp=tp, fp=fp, fn=fn, tn=tn, threshold=float(thr))
    # bootstrap CIs + paired delta vs DTS+HGB.
    # The saved score vectors follow the train_test_split output order; rebuild
    # that exact permutation from the manifest labels and the stored seed.
    z = np.load(f"{REPO}/results/twinfree/ninefive_core_full/score_vectors_twinfree.npz")
    yt = z["y_test"].astype(int)
    seed = json.load(open(f"{REPO}/results/twinfree/ninefive_results.json"))["seed"]
    from sklearn.model_selection import train_test_split
    idx_all = np.arange(len(y))
    _tv, test_perm = train_test_split(idx_all, test_size=0.20, random_state=seed, stratify=y)
    assert (y[test_perm] == yt).all(), "could not reconstruct score-vector order"
    # my arrays are in ascending manifest order over `te`; remap to perm order
    pos = {v: i for i, v in enumerate(te)}
    remap = np.array([pos[i] for i in test_perm])
    st = st[remap]; pte = pte[remap]
    y_te_perm = y[test_perm]
    assert (np.sort(test_perm) == te).all()
    # recompute counts in the permuted frame (identical, order-free)
    hgb = z["DTS+HGB_test"]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(te), (1000, len(te)))
    aucs, deltas = [], []
    for i in idx:
        if len(np.unique(y_te_perm[i])) < 2: continue
        aucs.append(roc_auc_score(y_te_perm[i], st[i]))
        deltas.append(roc_auc_score(y_te_perm[i], hgb[i]) - aucs[-1])
    out["auroc_ci"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
    out["delta_hgb_minus_multirocket"] = float(roc_auc_score(yt, hgb) - out["auroc"])
    out["delta_ci"] = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]
    np.save(f"{D}/multirocket_test_scores.npy", st)
    json.dump(out, open(f"{D}/multirocket_main.json", "w"), indent=1)
    log("MULTIROCKET MAIN:", json.dumps({k: round(v,4) if isinstance(v,float) else v
                                         for k,v in out.items() if not isinstance(v,list)}))
    mark("multi_eval")
log("ALL STAGES DONE" if stage_done("multi_eval") else "more to do")
