"""TCN baseline on the strict twin-free split (JAX implementation, CPU).

Architecture: 4 dilated causal conv blocks (dilations 1,2,4,8, 64 channels,
kernel 3, ReLU, dropout 0.1) + global average pooling + linear head, trained
with Adam (1e-3, weight decay 1e-4), BCE loss, batch 64, grad clip 1.0,
validation-AUROC checkpointing for 20 epochs, fixed seed. Mirrors the paper's
neural-baseline protocol (AdamW, clipping, val checkpointing, 20 epochs).

Resumable: checkpoints params+epoch to /tmp/seqtf/tcn_ckpt.pkl each epoch.
Run: python3 tcn_main.py <budget_s>
"""
import json, os, pickle, sys, time
import numpy as np
import jax, jax.numpy as jnp
from sklearn.metrics import roc_auc_score, f1_score

D = "/tmp/seqtf"
REPO = "/sessions/tender-kind-dijkstra/mnt/dts-fall-detection"
t0 = time.time(); budget = float(sys.argv[1]) if len(sys.argv) > 1 else 32.0
EPOCHS, BATCH, LR, WD, CLIP, CH, SEED = 20, 64, 1e-3, 1e-4, 1.0, 64, 0

X = np.load(f"{D}/X.npy", mmap_mode="r")
meta = json.load(open(f"{D}/meta.json"))
split = np.array(meta["split"]); y = np.array(meta["label"])
tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))

mu = X[tr].mean((0, 1), keepdims=True); sd = X[tr].std((0, 1), keepdims=True) + 1e-8
Xtr = ((X[tr] - mu) / sd).astype(np.float32)
Xva = ((X[va] - mu) / sd).astype(np.float32)
Xte = ((X[te] - mu) / sd).astype(np.float32)
ytr = y[tr].astype(np.float32)

DILS = (1, 2, 4, 8)

def init_params(key):
    p = {}
    cin = 34
    for i, d in enumerate(DILS):
        key, k1 = jax.random.split(key)
        p[f"w{i}"] = jax.random.normal(k1, (3, cin, CH)) * np.sqrt(2.0 / (3 * cin))
        p[f"b{i}"] = jnp.zeros((CH,))
        cin = CH
    key, k1 = jax.random.split(key)
    p["wo"] = jax.random.normal(k1, (CH,)) * 0.01
    p["bo"] = jnp.zeros(())
    return p

def forward(p, x, train, key):
    h = x  # (B, T, C)
    for i, d in enumerate(DILS):
        pad = 2 * d  # causal: left-pad (kernel-1)*dilation
        hp = jnp.pad(h, ((0, 0), (pad, 0), (0, 0)))
        h = jax.lax.conv_general_dilated(
            hp, p[f"w{i}"], window_strides=(1,), padding="VALID",
            rhs_dilation=(d,), dimension_numbers=("NWC", "WIO", "NWC")) + p[f"b{i}"]
        h = jax.nn.relu(h)
        if train:
            key, k1 = jax.random.split(key)
            h = h * jax.random.bernoulli(k1, 0.9, h.shape) / 0.9
    g = h.mean(axis=1)
    return g @ p["wo"] + p["bo"]

def loss_fn(p, x, yb, key):
    logit = forward(p, x, True, key)
    return jnp.mean(jnp.maximum(logit, 0) - logit * yb + jnp.log1p(jnp.exp(-jnp.abs(logit))))

@jax.jit
def adam_step(p, m, v, t, x, yb, key):
    g = jax.grad(loss_fn)(p, x, yb, key)
    gn = jnp.sqrt(sum(jnp.sum(gi ** 2) for gi in jax.tree_util.tree_leaves(g)))
    scale = jnp.minimum(1.0, CLIP / (gn + 1e-9))
    g = jax.tree_util.tree_map(lambda gi: gi * scale, g)
    m = jax.tree_util.tree_map(lambda mi, gi: 0.9 * mi + 0.1 * gi, m, g)
    v = jax.tree_util.tree_map(lambda vi, gi: 0.999 * vi + 0.001 * gi ** 2, v, g)
    def upd(pi, mi, vi):
        mh = mi / (1 - 0.9 ** t); vh = vi / (1 - 0.999 ** t)
        return pi - LR * (mh / (jnp.sqrt(vh) + 1e-8) + WD * pi)
    return jax.tree_util.tree_map(upd, p, m, v), m, v

@jax.jit
def predict(p, x):
    return forward(p, x, False, jax.random.PRNGKey(0))

def scores(p, Xs):
    return np.concatenate([np.asarray(predict(p, jnp.asarray(Xs[i:i+256])))
                           for i in range(0, len(Xs), 256)])

ck = f"{D}/tcn_ckpt.pkl"
if os.path.exists(ck):
    state = pickle.load(open(ck, "rb"))
else:
    key = jax.random.PRNGKey(SEED)
    p = init_params(key)
    zeros = jax.tree_util.tree_map(jnp.zeros_like, p)
    state = dict(p=p, m=zeros, v=zeros, t=0, epoch=0, best=-1.0, best_p=None)

p, m, v, t = state["p"], state["m"], state["v"], state["t"]
rng = np.random.default_rng(SEED + 1)
order = np.arange(len(Xtr))
while state["epoch"] < EPOCHS and time.time() - t0 < budget:
    rng = np.random.default_rng(1000 + state["epoch"])
    rng.shuffle(order)
    key = jax.random.PRNGKey(2000 + state["epoch"])
    for b in range(0, len(order), BATCH):
        j = order[b:b + BATCH]; t += 1
        key, k1 = jax.random.split(key)
        p, m, v = adam_step(p, m, v, t, jnp.asarray(Xtr[j]), jnp.asarray(ytr[j]), k1)
    auv = roc_auc_score(y[va], scores(p, Xva))
    state["epoch"] += 1
    if auv > state["best"]:
        state["best"] = float(auv)
        state["best_p"] = jax.tree_util.tree_map(np.asarray, p)
    print(f"[{time.time()-t0:5.1f}s] epoch {state['epoch']} valAUROC {auv:.4f} best {state['best']:.4f}", flush=True)
    state.update(p=p, m=m, v=v, t=t)
    pickle.dump(state, open(ck, "wb"))

if state["epoch"] >= EPOCHS:
    bp = state["best_p"]
    sv = scores(bp, Xva); st_sorted = scores(bp, Xte)
    pva = 1/(1+np.exp(-sv)); pte = 1/(1+np.exp(-st_sorted))
    grid = np.linspace(0.01, 0.99, 99)
    thr = grid[int(np.argmax([f1_score(y[va], pva >= g) for g in grid]))]
    # remap to score-vector order for paired comparison
    from sklearn.model_selection import train_test_split
    seed = json.load(open(f"{REPO}/results/twinfree/ninefive_results.json"))["seed"]
    _tv, test_perm = train_test_split(np.arange(len(y)), test_size=0.20,
                                      random_state=seed, stratify=y)
    pos = {v_: i for i, v_ in enumerate(te)}
    remap = np.array([pos[i] for i in test_perm])
    st = st_sorted[remap]; pte = pte[remap]; yt = y[test_perm]
    pred = (pte >= thr).astype(int)
    tp = int(((pred==1)&(yt==1)).sum()); fp = int(((pred==1)&(yt==0)).sum())
    fn = int(((pred==0)&(yt==1)).sum()); tn = int(((pred==0)&(yt==0)).sum())
    prec, rec = tp/max(tp+fp,1), tp/(tp+fn)
    z = np.load(f"{REPO}/results/twinfree/ninefive_core_full/score_vectors_twinfree.npz")
    hgb = z["DTS+HGB_test"]
    rngb = np.random.default_rng(0)
    idx = rngb.integers(0, len(yt), (1000, len(yt)))
    aucs, deltas = [], []
    for i in idx:
        if len(np.unique(yt[i])) < 2: continue
        aucs.append(roc_auc_score(yt[i], st[i]))
        deltas.append(roc_auc_score(yt[i], hgb[i]) - aucs[-1])
    out = dict(auroc=roc_auc_score(yt, st), f1=2*prec*rec/max(prec+rec,1e-9),
               precision=prec, recall=rec, accuracy=(tp+tn)/len(yt),
               tp=tp, fp=fp, fn=fn, tn=tn, threshold=float(thr),
               val_best_auroc=state["best"], epochs=EPOCHS,
               params=int(sum(np.size(x) for x in jax.tree_util.tree_leaves(state["best_p"]))),
               auroc_ci=[float(np.percentile(aucs,2.5)), float(np.percentile(aucs,97.5))],
               delta_hgb_minus_tcn=float(roc_auc_score(yt,hgb)-roc_auc_score(yt,st)),
               delta_ci=[float(np.percentile(deltas,2.5)), float(np.percentile(deltas,97.5))])
    np.save(f"{D}/tcn_test_scores.npy", st)
    json.dump(out, open(f"{D}/tcn_main.json", "w"), indent=1)
    print("TCN MAIN:", json.dumps({k:(round(v,4) if isinstance(v,float) else v)
                                   for k,v in out.items() if not isinstance(v,list)}))
else:
    print(f"epochs done {state['epoch']}/{EPOCHS}; rerun")
    sys.exit(3)
