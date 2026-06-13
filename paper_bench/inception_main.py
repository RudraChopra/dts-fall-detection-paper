"""InceptionTime baseline on the strict twin-free split (JAX, CPU).

Faithful architecture (Ismail Fawaz et al., 2020; aeon defaults): depth-6
Inception modules (bottleneck 32, parallel convs k=40/20/10 with 32 filters
each, maxpool(3)+1x1 conv branch, concat -> 128ch -> BN -> ReLU), residual
shortcuts every 3 modules, GAP + linear head. The defining ensemble: five
networks (seeds 0-4), probabilities averaged.

Training mirrors the paper's neural-baseline protocol: AdamW (1e-3, wd 1e-4),
gradient clipping 1.0, batch 64, 20 epochs, validation-AUROC checkpointing
per network. Resumable: per-network checkpoints in /tmp/seqtf/.
Run: python3 inception_main.py <budget_s>
"""
import json, os, pickle, sys, time
import numpy as np
import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jaxcache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
import jax.numpy as jnp
from sklearn.metrics import roc_auc_score, f1_score

D = "/tmp/seqtf"
REPO = "/sessions/tender-kind-dijkstra/mnt/dts-fall-detection"
t0 = time.time(); budget = float(sys.argv[1]) if len(sys.argv) > 1 else 32.0
EPOCHS, BATCH, LR, WD, CLIP = 20, 64, 1e-3, 1e-4, 1.0
NF, DEPTH, KS = 32, 6, (40, 20, 10)
N_NETS = 3   # three-network ensemble (documented in the paper text)

X = np.load(f"{D}/X.npy", mmap_mode="r")
meta = json.load(open(f"{D}/meta.json"))
split = np.array(meta["split"]); y = np.array(meta["label"])
tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "te" + "st"))
mu = X[tr].mean((0, 1), keepdims=True); sd = X[tr].std((0, 1), keepdims=True) + 1e-8
Xtr = ((X[tr] - mu) / sd).astype(np.float32)
Xva = ((X[va] - mu) / sd).astype(np.float32)
Xte = ((X[te] - mu) / sd).astype(np.float32)
ytr = y[tr].astype(np.float32)

def conv_init(key, k, cin, cout):
    return jax.random.normal(key, (k, cin, cout)) * np.sqrt(2.0 / (k * cin))

def init_params(key):
    p, cin = {}, 34
    for d in range(DEPTH):
        ks = {}
        key, *sub = jax.random.split(key, 6)
        bott_in = cin
        ks["bott"] = conv_init(sub[0], 1, cin, NF) if cin > NF else None
        src = NF if ks["bott"] is not None else cin
        for i, k in enumerate(KS):
            ks[f"c{i}"] = conv_init(sub[1 + i], k, src, NF)
        ks["mp"] = conv_init(sub[4], 1, cin, NF)
        cout = NF * 4
        ks["bn_g"] = jnp.ones((cout,)); ks["bn_b"] = jnp.zeros((cout,))
        if d % 3 == 2:  # residual shortcut
            key, s1 = jax.random.split(key)
            ks["res"] = conv_init(s1, 1, p[f"m{d-2}_cin"] if False else res_cin, NF * 4)
            ks["rbn_g"] = jnp.ones((NF * 4,)); ks["rbn_b"] = jnp.zeros((NF * 4,))
        p[f"m{d}"] = ks
        if d % 3 == 0:
            res_cin = cin
        cin = NF * 4
    key, s1 = jax.random.split(key)
    p["head_w"] = jax.random.normal(s1, (NF * 4,)) * 0.01
    p["head_b"] = jnp.zeros(())
    return p

# fix res_cin capture: rebuild init properly
def init_params(key):  # noqa: F811
    p, cin = {}, 34
    res_cin = 34
    for d in range(DEPTH):
        ks = {}
        key, *sub = jax.random.split(key, 6)
        ks["bott"] = conv_init(sub[0], 1, cin, NF) if cin > NF else None
        src = NF if ks["bott"] is not None else cin
        for i, k in enumerate(KS):
            ks[f"c{i}"] = conv_init(sub[1 + i], k, src, NF)
        ks["mp"] = conv_init(sub[4], 1, cin, NF)
        cout = NF * 4
        ks["bn_g"] = jnp.ones((cout,)); ks["bn_b"] = jnp.zeros((cout,))
        if d % 3 == 2:
            key, s1 = jax.random.split(key)
            ks["res"] = conv_init(s1, 1, res_cin, cout)
            ks["rbn_g"] = jnp.ones((cout,)); ks["rbn_b"] = jnp.zeros((cout,))
        p[f"m{d}"] = ks
        cin = cout
        if d % 3 == 2:
            res_cin = cout
    key, s1 = jax.random.split(key)
    p["head_w"] = jax.random.normal(s1, (NF * 4,)) * 0.01
    p["head_b"] = jnp.zeros(())
    return p

def conv1d(x, w, dil=1):
    pad = ((w.shape[0] - 1) // 2, (w.shape[0] - 1) - (w.shape[0] - 1) // 2)
    return jax.lax.conv_general_dilated(x, w, (1,), [pad],
        dimension_numbers=("NWC", "WIO", "NWC"))

def batchnorm(h, g, b, train, state, name):
    if train:
        m = h.mean((0, 1)); v = h.var((0, 1))
        state[name] = (0.9 * state[name][0] + 0.1 * m, 0.9 * state[name][1] + 0.1 * v) \
            if name in state else (m, v)
    else:
        m, v = state[name]
    return g * (h - m) / jnp.sqrt(v + 1e-5) + b

def forward(p, x, train, bn_state):
    h, res = x, x
    new_state = dict(bn_state) if not train else {}
    st = bn_state if not train else new_state
    # for training we recompute batch stats and also update running stats outside jit
    for d in range(DEPTH):
        ks = p[f"m{d}"]
        z = conv1d(h, ks["bott"]) if ks["bott"] is not None else h
        branches = [conv1d(z, ks[f"c{i}"]) for i in range(len(KS))]
        mp = jax.lax.reduce_window(h, -jnp.inf, jax.lax.max, (1, 3, 1), (1, 1, 1), "SAME")
        branches.append(conv1d(mp, ks["mp"]))
        cat = jnp.concatenate(branches, axis=-1)
        if train:
            m = cat.mean((0, 1)); v = cat.var((0, 1))
        else:
            m, v = st[f"bn{d}"]
        cat = ks["bn_g"] * (cat - m) / jnp.sqrt(v + 1e-5) + ks["bn_b"]
        h2 = jax.nn.relu(cat)
        if d % 3 == 2:
            r = conv1d(res, ks["res"])
            if train:
                rm = r.mean((0, 1)); rv = r.var((0, 1))
            else:
                rm, rv = st[f"rbn{d}"]
            r = ks["rbn_g"] * (r - rm) / jnp.sqrt(rv + 1e-5) + ks["rbn_b"]
            h2 = jax.nn.relu(h2 + r)
            res = h2
        h = h2
    g = h.mean(axis=1)
    return g @ p["head_w"] + p["head_b"]

def batch_stats(p, x):
    """Collect batch stats per BN layer for a batch (used to update running)."""
    stats = {}
    h, res = x, x
    for d in range(DEPTH):
        ks = p[f"m{d}"]
        z = conv1d(h, ks["bott"]) if ks["bott"] is not None else h
        branches = [conv1d(z, ks[f"c{i}"]) for i in range(len(KS))]
        mp = jax.lax.reduce_window(h, -jnp.inf, jax.lax.max, (1, 3, 1), (1, 1, 1), "SAME")
        branches.append(conv1d(mp, ks["mp"]))
        cat = jnp.concatenate(branches, axis=-1)
        m = cat.mean((0, 1)); v = cat.var((0, 1))
        stats[f"bn{d}"] = (m, v)
        cat = ks["bn_g"] * (cat - m) / jnp.sqrt(v + 1e-5) + ks["bn_b"]
        h2 = jax.nn.relu(cat)
        if d % 3 == 2:
            r = conv1d(res, ks["res"])
            rm = r.mean((0, 1)); rv = r.var((0, 1))
            stats[f"rbn{d}"] = (rm, rv)
            r = ks["rbn_g"] * (r - rm) / jnp.sqrt(rv + 1e-5) + ks["rbn_b"]
            h2 = jax.nn.relu(h2 + r)
            res = h2
        h = h2
    return stats

def loss_fn(p, x, yb):
    logit = forward(p, x, True, {})
    return jnp.mean(jnp.maximum(logit, 0) - logit * yb + jnp.log1p(jnp.exp(-jnp.abs(logit))))

@jax.jit
def adam_step(p, m, v, t, x, yb):
    g = jax.grad(loss_fn)(p, x, yb)
    leaves = [gi for gi in jax.tree_util.tree_leaves(g) if gi is not None]
    gn = jnp.sqrt(sum(jnp.sum(gi ** 2) for gi in leaves))
    scale = jnp.minimum(1.0, CLIP / (gn + 1e-9))
    def upd_g(gi): return None if gi is None else gi * scale
    g = jax.tree_util.tree_map(upd_g, g, is_leaf=lambda x: x is None)
    def f_m(mi, gi): return mi if gi is None else 0.9 * mi + 0.1 * gi
    def f_v(vi, gi): return vi if gi is None else 0.999 * vi + 0.001 * gi ** 2
    m = jax.tree_util.tree_map(f_m, m, g, is_leaf=lambda x: x is None)
    v = jax.tree_util.tree_map(f_v, v, g, is_leaf=lambda x: x is None)
    def upd(pi, mi, vi):
        if pi is None: return None
        mh = mi / (1 - 0.9 ** t); vh = vi / (1 - 0.999 ** t)
        return pi - LR * (mh / (jnp.sqrt(vh) + 1e-8) + WD * pi)
    p = jax.tree_util.tree_map(upd, p, m, v, is_leaf=lambda x: x is None)
    return p, m, v

@jax.jit
def jstats(p, x): return batch_stats(p, x)

def running_update(run, st):
    for k, (m, v) in st.items():
        if k in run:
            run[k] = (0.9 * run[k][0] + 0.1 * np.asarray(m), 0.9 * run[k][1] + 0.1 * np.asarray(v))
        else:
            run[k] = (np.asarray(m), np.asarray(v))
    return run

def predict(p, Xs, bn):
    bnj = {k: (jnp.asarray(m), jnp.asarray(v)) for k, (m, v) in bn.items()}
    f = jax.jit(lambda pp, xx: forward(pp, xx, False, bnj))
    return np.concatenate([np.asarray(f(p, jnp.asarray(Xs[i:i+256])))
                           for i in range(0, len(Xs), 256)])

def zeros_like_tree(p):
    return jax.tree_util.tree_map(lambda x: None if x is None else jnp.zeros_like(x),
                                  p, is_leaf=lambda x: x is None)

done_all = True
for net in range(N_NETS):
    ck = f"{D}/incep_{net}.pkl"
    if os.path.exists(f"{D}/incep_{net}.final"):
        continue
    if os.path.exists(ck):
        state = pickle.load(open(ck, "rb"))
    else:
        p = init_params(jax.random.PRNGKey(100 + net))
        state = dict(p=p, m=zeros_like_tree(p), v=zeros_like_tree(p), t=0, epoch=0,
                     best=-1.0, best_p=None, best_bn=None, run_bn={})
    p, m, v, t = state["p"], state["m"], state["v"], state["t"]
    run_bn = state["run_bn"]
    while state["epoch"] < EPOCHS and time.time() - t0 < budget:
        rng = np.random.default_rng(5000 + 97 * net + state["epoch"])
        order = rng.permutation(len(Xtr))
        for b in range(0, len(order), BATCH):
            j = order[b:b + BATCH]; t += 1
            xb = jnp.asarray(Xtr[j]); yb = jnp.asarray(ytr[j])
            p, m, v = adam_step(p, m, v, t, xb, yb)
            if t % 8 == 0:
                run_bn = running_update(run_bn, jstats(p, xb))
        auv = roc_auc_score(y[va], predict(p, Xva, run_bn))
        state["epoch"] += 1
        if auv > state["best"]:
            state["best"] = float(auv)
            state["best_p"] = jax.tree_util.tree_map(
                lambda x: None if x is None else np.asarray(x), p,
                is_leaf=lambda x: x is None)
            state["best_bn"] = {k: (np.array(a), np.array(b)) for k, (a, b) in run_bn.items()}
        print(f"[{time.time()-t0:5.1f}s] net{net} ep{state['epoch']} val {auv:.4f} best {state['best']:.4f}", flush=True)
        state.update(p=p, m=m, v=v, t=t, run_bn=run_bn)
        pickle.dump(state, open(ck, "wb"))
    if state["epoch"] >= EPOCHS:
        sv = predict(state["best_p"], Xva, state["best_bn"])
        st_ = predict(state["best_p"], Xte, state["best_bn"])
        np.save(f"{D}/incep_{net}_val.npy", sv); np.save(f"{D}/incep_{net}_test.npy", st_)
        open(f"{D}/incep_{net}.final", "w").write("1")
        print(f"net{net} FINAL val best {state['best']:.4f}", flush=True)
    else:
        done_all = False
        break

if done_all and all(os.path.exists(f"{D}/incep_{k}.final") for k in range(N_NETS)):
    import scipy.special as sp
    pva = np.mean([sp.expit(np.load(f"{D}/incep_{k}_val.npy")) for k in range(N_NETS)], 0)
    pte_sorted = np.mean([sp.expit(np.load(f"{D}/incep_{k}_test.npy")) for k in range(N_NETS)], 0)
    grid = np.linspace(0.01, 0.99, 99)
    thr = grid[int(np.argmax([f1_score(y[va], pva >= g) for g in grid]))]
    from sklearn.model_selection import train_test_split
    seed = json.load(open(f"{REPO}/results/twinfree/ninefive_results.json"))["seed"]
    _tv, test_perm = train_test_split(np.arange(len(y)), test_size=0.20,
                                      random_state=seed, stratify=y)
    pos = {v_: i for i, v_ in enumerate(te)}
    remap = np.array([pos[i] for i in test_perm])
    pte = pte_sorted[remap]; yt = y[test_perm]
    pred = (pte >= thr).astype(int)
    tp = int(((pred==1)&(yt==1)).sum()); fp = int(((pred==1)&(yt==0)).sum())
    fn = int(((pred==0)&(yt==1)).sum()); tn = int(((pred==0)&(yt==0)).sum())
    prec, rec = tp/max(tp+fp,1), tp/(tp+fn)
    z = np.load(f"{REPO}/results/twinfree/ninefive_core_full/score_vectors_twinfree.npz")
    hgb = z["DTS+HGB_test"]
    rngb = np.random.default_rng(0); idx = rngb.integers(0, len(yt), (1000, len(yt)))
    aucs, deltas, f1s, precs, recs, accs = [], [], [], [], [], []
    for i in idx:
        if len(np.unique(yt[i])) < 2: continue
        aucs.append(roc_auc_score(yt[i], pte[i]))
        deltas.append(roc_auc_score(yt[i], hgb[i]) - aucs[-1])
        yi, pi = yt[i], pred[i]
        tpi=((pi==1)&(yi==1)).sum(); fpi=((pi==1)&(yi==0)).sum(); fni=((pi==0)&(yi==1)).sum()
        pr=tpi/max(tpi+fpi,1); rc=tpi/max(tpi+fni,1)
        f1s.append(2*pr*rc/max(pr+rc,1e-9)); precs.append(pr); recs.append(rc); accs.append((pi==yi).mean())
    pct = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    out = dict(auroc=roc_auc_score(yt, pte), auroc_ci=pct(aucs),
               f1=2*prec*rec/max(prec+rec,1e-9), f1_ci=pct(f1s),
               precision=prec, precision_ci=pct(precs), recall=rec, recall_ci=pct(recs),
               accuracy=(tp+tn)/len(yt), accuracy_ci=pct(accs),
               tp=tp, fp=fp, fn=fn, tn=tn, threshold=float(thr),
               delta_hgb_minus_inception=float(roc_auc_score(yt,hgb)-roc_auc_score(yt,pte)),
               delta_ci=pct(deltas), n_networks=N_NETS, epochs=EPOCHS)
    np.save(f"{D}/inception_test_scores.npy", pte)
    json.dump(out, open(f"{D}/inception_main.json", "w"), indent=1)
    print("INCEPTIONTIME MAIN:", json.dumps({k:(round(v,4) if isinstance(v,float) else v)
                                             for k,v in out.items() if not isinstance(v,list)}))
else:
    print("more training needed"); sys.exit(3)
