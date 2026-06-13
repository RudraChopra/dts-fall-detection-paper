"""Expanded controlled temporal-order benchmark for the PGTS/DTS papers.

All results produced by this script are REAL computed numbers (fixed seeds,
saved to JSON). Three temporal-order constructions are studied; in every one,
negatives are frame permutations of positives (or order-permuted variants), so
class-conditional frame marginals are identical multisets by construction and
any order-invariant feature map is provably at chance (Theorem 2).

Constructions
  fall      : three-phase (still, drop, floor) trajectory; negative = uniform
              frame permutation of its paired positive.
  sit2stand : rise-then-plateau trajectory (sit-to-stand-like); negative =
              uniform frame permutation.
  burst     : class 0 = early motion burst, class 1 = late motion burst, with
              the SAME multiset of increments (timing-only difference).

Feature maps
  dts   : 16-operator bank per channel (mirrors the paper's operator list)
  alpha : temporal-asymmetry alpha(z; tau) per channel only
  bof   : bag-of-frames (order-invariant value quantiles per channel)
  raw   : flattened raw sequence (order-sensitive, unstructured)

Classifiers: HistGradientBoosting (HGB), LogisticRegression (LR),
MLP on raw, MiniROCKET+Ridge (aeon), and optionally tiny GRU / TCN (torch).

Usage: python3 synth_bench.py <sweep>   where <sweep> in
  {main, noise, timing, length, duration, distractor, operators, tasks, all}
Outputs: results_<sweep>.json next to this file.
"""
import json, math, os, sys, time
import numpy as np

RESULT_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4]
N_TEST = 500           # test pairs per run
TAU_GRID = np.round(np.arange(0.20, 0.81, 0.05), 2)

# ----------------------------------------------------------------------------
# Generators
# ----------------------------------------------------------------------------
def gen_fall(n, T=150, d_inf=8, d_noise=0, p1=0.5, dur=0.2, snr=8.0, rng=None):
    """Three-phase fall-like positives + permuted negatives.

    Informative channels share the event timing; each channel gets the
    three-phase increment structure with channel-specific scale. snr is the
    ratio of drop-phase increment scale to still-phase increment scale.
    """
    rng = rng or np.random.default_rng(0)
    p2 = min(p1 + dur, 0.95)
    sig_s = 1.0
    sig_d = snr * sig_s
    t = np.arange(T) / T
    X = np.zeros((n, T, d_inf + d_noise), dtype=np.float64)
    for i in range(n):
        jit1 = np.clip(p1 + rng.normal(0, 0.03), 0.05, 0.9)
        jit2 = np.clip(p2 + rng.normal(0, 0.03), jit1 + 0.02, 0.98)
        for c in range(d_inf):
            scale = rng.uniform(0.5, 1.5)
            inc = rng.normal(0, sig_s, T)
            drop_mask = (t >= jit1) & (t < jit2)
            inc[drop_mask] += -sig_d * scale * (1 + 0.2 * rng.normal(0, 1, drop_mask.sum()))
            X[i, :, c] = np.cumsum(inc)
        for c in range(d_inf, d_inf + d_noise):
            X[i, :, c] = np.cumsum(rng.normal(0, sig_s, T))
    Xn = np.empty_like(X)
    for i in range(n):
        perm = rng.permutation(T)
        Xn[i] = X[i, perm, :]
    return X, Xn


def gen_sit2stand(n, T=150, d_inf=8, d_noise=0, p1=0.4, dur=0.3, snr=8.0, rng=None):
    """Rise-then-plateau (sit-to-stand-like) positives + permuted negatives."""
    rng = rng or np.random.default_rng(0)
    p2 = min(p1 + dur, 0.95)
    sig_s = 1.0
    sig_r = snr * sig_s
    t = np.arange(T) / T
    X = np.zeros((n, T, d_inf + d_noise), dtype=np.float64)
    for i in range(n):
        jit1 = np.clip(p1 + rng.normal(0, 0.03), 0.05, 0.9)
        jit2 = np.clip(p2 + rng.normal(0, 0.03), jit1 + 0.02, 0.98)
        for c in range(d_inf):
            scale = rng.uniform(0.5, 1.5)
            inc = rng.normal(0, sig_s, T)
            rise_mask = (t >= jit1) & (t < jit2)
            # smooth ramp profile inside the event window
            k = rise_mask.sum()
            if k > 0:
                prof = np.sin(np.linspace(0, np.pi, k))
                inc[rise_mask] += sig_r * scale * prof
            X[i, :, c] = np.cumsum(inc)
        for c in range(d_inf, d_inf + d_noise):
            X[i, :, c] = np.cumsum(rng.normal(0, sig_s, T))
    Xn = np.empty_like(X)
    for i in range(n):
        perm = rng.permutation(T)
        Xn[i] = X[i, perm, :]
    return X, Xn


def gen_burst(n, T=150, d_inf=8, d_noise=0, snr=8.0, rng=None):
    """Direction-of-time construction: negative = exact frame reversal of its
    paired positive. The per-clip frame-value multiset is therefore IDENTICAL
    across classes (reversal is a permutation), so any order-invariant feature
    is provably at chance (Theorem 2 applies pairwise); only temporal
    direction/timing distinguishes the classes. Positives carry an early
    motion burst, so reversed negatives carry a late one."""
    rng = rng or np.random.default_rng(0)
    sig_s, sig_b = 1.0, snr
    L = max(int(0.15 * T), 3)
    X = np.zeros((n, T, d_inf + d_noise))
    for i in range(n):
        for c in range(d_inf + d_noise):
            inc = rng.normal(0, sig_s, T)
            if c < d_inf:
                s0 = int(0.1 * T)
                inc[s0:s0 + L] += rng.normal(0, sig_b, L)  # early burst
            X[i, :, c] = np.cumsum(inc)
    Xn = X[:, ::-1, :].copy()   # exact time reversal: same value multiset
    return X, Xn

GENS = {"fall": gen_fall, "sit2stand": gen_sit2stand, "burst": gen_burst}

# ----------------------------------------------------------------------------
# Feature maps
# ----------------------------------------------------------------------------
def alpha_stat(Z, tau):
    """Temporal asymmetry per channel. Z: (n, T, d)."""
    dZ = np.abs(np.diff(Z, axis=1))            # (n, T-1, d)
    T1 = dZ.shape[1]
    k = max(int(math.floor(tau * (T1 + 1))) - 1, 1)
    num = dZ[:, :k, :].sum(axis=1)
    den = dZ.sum(axis=1) + 1e-8
    return num / den


def dts_features(Z, tau, ops="full"):
    """16-operator bank per channel, mirroring the paper's Omega(z; tau).

    ops: 'full', 'noalpha', 'alpha', 'order_invariant_only', 'order_only',
         'quantiles', 'slopes', 'endpoints'
    """
    n, T, d = Z.shape
    dZ = np.diff(Z, axis=1)
    adZ = np.abs(dZ)
    d2Z = np.abs(np.diff(Z, n=2, axis=1))
    q25, q50, q75 = np.percentile(Z, [25, 50, 75], axis=1)
    tt = np.arange(T) - (T - 1) / 2.0
    denom_t = (tt ** 2).sum()
    slope = np.einsum("t,ntd->nd", tt, Z - Z.mean(axis=1, keepdims=True)) / denom_t
    half = T // 2
    beta = Z[:, half:, :].mean(axis=1) - Z[:, :half, :].mean(axis=1)
    feats = {
        "mean": Z.mean(axis=1), "std": Z.std(axis=1),
        "min": Z.min(axis=1), "max": Z.max(axis=1),
        "q25": q25, "q50": q50, "q75": q75,
        "range": Z.max(axis=1) - Z.min(axis=1),
        "z1": Z[:, 0, :], "zT": Z[:, -1, :],
        "slope": slope, "beta": beta,
        "tv": adZ.sum(axis=1), "maxinc": adZ.max(axis=1),
        "maxacc": d2Z.max(axis=1),
        "alpha": alpha_stat(Z, tau),
    }
    order_invariant = ["mean", "std", "min", "max", "q25", "q50", "q75", "range"]
    order_sensitive = ["z1", "zT", "slope", "beta", "tv", "maxinc", "maxacc", "alpha"]
    if ops == "full":
        keys = order_invariant + order_sensitive
    elif ops == "noalpha":
        keys = [k for k in order_invariant + order_sensitive if k != "alpha"]
    elif ops == "alpha":
        keys = ["alpha"]
    elif ops == "order_invariant_only":
        keys = order_invariant
    elif ops == "order_only":
        keys = order_sensitive
    elif ops == "quantiles":
        keys = ["q25", "q50", "q75", "min", "max"]
    elif ops == "slopes":
        keys = ["slope", "beta"]
    elif ops == "endpoints":
        keys = ["z1", "zT"]
    else:
        raise ValueError(ops)
    return np.concatenate([feats[k] for k in keys], axis=1)


def bof_features(Z, n_q=16):
    """Order-invariant bag-of-frames: per-channel value quantiles.
    Provably identical in law across classes for permutation negatives."""
    qs = np.linspace(0, 100, n_q)
    return np.concatenate([np.percentile(Z, q, axis=1) for q in qs], axis=1)

# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score


def select_tau(Xtr_pos, Xtr_neg):
    """Train-only tau selection by summed |Welch t| over channels (paper's rule)."""
    best_tau, best_score = TAU_GRID[0], -1
    for tau in TAU_GRID:
        a_p = alpha_stat(Xtr_pos, tau); a_n = alpha_stat(Xtr_neg, tau)
        mp, mn_ = a_p.mean(0), a_n.mean(0)
        vp, vn = a_p.var(0) + 1e-12, a_n.var(0) + 1e-12
        tstat = np.abs(mp - mn_) / np.sqrt(vp / len(a_p) + vn / len(a_n))
        s = tstat.sum()
        if s > best_score:
            best_score, best_tau = s, tau
    return float(best_tau)


def fit_score(model_name, Xp_tr, Xn_tr, Xp_te, Xn_te, tau, seed, ops="full"):
    """Returns test AUROC for one model on one train/test draw."""
    ytr = np.r_[np.ones(len(Xp_tr)), np.zeros(len(Xn_tr))]
    yte = np.r_[np.ones(len(Xp_te)), np.zeros(len(Xn_te))]
    if model_name == "dts_hgb":
        Ftr = np.r_[dts_features(Xp_tr, tau, ops), dts_features(Xn_tr, tau, ops)]
        Fte = np.r_[dts_features(Xp_te, tau, ops), dts_features(Xn_te, tau, ops)]
        clf = HistGradientBoostingClassifier(random_state=seed, max_iter=300)
        clf.fit(Ftr, ytr); s = clf.predict_proba(Fte)[:, 1]
    elif model_name == "dts_lr":
        Ftr = np.r_[dts_features(Xp_tr, tau, ops), dts_features(Xn_tr, tau, ops)]
        Fte = np.r_[dts_features(Xp_te, tau, ops), dts_features(Xn_te, tau, ops)]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(Ftr, ytr); s = clf.predict_proba(Fte)[:, 1]
    elif model_name == "alpha_lr":
        Ftr = np.r_[alpha_stat(Xp_tr, tau), alpha_stat(Xn_tr, tau)]
        Fte = np.r_[alpha_stat(Xp_te, tau), alpha_stat(Xn_te, tau)]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(Ftr, ytr); s = clf.predict_proba(Fte)[:, 1]
    elif model_name == "bof_hgb":
        Ftr = np.r_[bof_features(Xp_tr), bof_features(Xn_tr)]
        Fte = np.r_[bof_features(Xp_te), bof_features(Xn_te)]
        clf = HistGradientBoostingClassifier(random_state=seed, max_iter=300)
        clf.fit(Ftr, ytr); s = clf.predict_proba(Fte)[:, 1]
    elif model_name == "raw_mlp":
        Ftr = np.r_[Xp_tr.reshape(len(Xp_tr), -1), Xn_tr.reshape(len(Xn_tr), -1)]
        Fte = np.r_[Xp_te.reshape(len(Xp_te), -1), Xn_te.reshape(len(Xn_te), -1)]
        clf = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=200, random_state=seed,
            early_stopping=True, n_iter_no_change=10))
        clf.fit(Ftr, ytr); s = clf.predict_proba(Fte)[:, 1]
    elif model_name == "raw_hgb":
        Ftr = np.r_[Xp_tr.reshape(len(Xp_tr), -1), Xn_tr.reshape(len(Xn_tr), -1)]
        Fte = np.r_[Xp_te.reshape(len(Xp_te), -1), Xn_te.reshape(len(Xn_te), -1)]
        clf = HistGradientBoostingClassifier(random_state=seed, max_iter=300)
        clf.fit(Ftr, ytr); s = clf.predict_proba(Fte)[:, 1]
    elif model_name == "minirocket":
        from aeon.transformations.collection.convolution_based import MiniRocket
        tr = np.r_[Xp_tr, Xn_tr].transpose(0, 2, 1)  # aeon wants (n, d, T)
        te = np.r_[Xp_te, Xn_te].transpose(0, 2, 1)
        mr = MiniRocket(random_state=seed)
        Ftr = mr.fit_transform(tr); Fte = mr.transform(te)
        clf = make_pipeline(StandardScaler(with_mean=False), RidgeClassifier(alpha=1.0))
        clf.fit(Ftr, ytr); s = clf.decision_function(Fte)
    elif model_name in ("gru", "tcn"):
        s = _torch_model(model_name, Xp_tr, Xn_tr, Xp_te, Xn_te, seed)
        if s is None:
            return None
    else:
        raise ValueError(model_name)
    return float(roc_auc_score(yte, s))


def _torch_model(kind, Xp_tr, Xn_tr, Xp_te, Xn_te, seed, epochs=15):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None
    torch.manual_seed(seed)
    Xtr = np.r_[Xp_tr, Xn_tr].astype(np.float32)
    ytr = np.r_[np.ones(len(Xp_tr)), np.zeros(len(Xn_tr))].astype(np.float32)
    Xte = np.r_[Xp_te, Xn_te].astype(np.float32)
    mu, sd = Xtr.mean((0, 1), keepdims=True), Xtr.std((0, 1), keepdims=True) + 1e-8
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    d = Xtr.shape[2]
    if kind == "gru":
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.g = nn.GRU(d, 32, num_layers=1, batch_first=True)
                self.f = nn.Linear(32, 1)
            def forward(self, x):
                _, h = self.g(x); return self.f(h[-1]).squeeze(-1)
    else:  # tcn: simple 3-block dilated causal conv net
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                ch = 32
                layers, ci = [], d
                for dil in (1, 2, 4):
                    layers += [nn.Conv1d(ci, ch, 3, padding=dil, dilation=dil), nn.ReLU()]
                    ci = ch
                self.c = nn.Sequential(*layers)
                self.f = nn.Linear(ch, 1)
            def forward(self, x):
                h = self.c(x.transpose(1, 2)).mean(-1)
                return self.f(h).squeeze(-1)
    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr)
    n = len(Xt); idx = np.arange(n)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        rng.shuffle(idx)
        for b in range(0, n, 64):
            j = idx[b:b + 64]
            opt.zero_grad()
            out = net(Xt[j])
            loss = lossf(out, yt[j])
            loss.backward(); opt.step()
    with torch.no_grad():
        s = net(torch.from_numpy(Xte)).numpy()
    return s

# ----------------------------------------------------------------------------
# Sweeps
# ----------------------------------------------------------------------------
def run_one(gen_name, model, n_train, seed, gen_kwargs=None, ops="full",
            tau_mode="selected"):
    gen_kwargs = dict(gen_kwargs or {})
    rng = np.random.default_rng(10_000 * seed + 7)
    gen = GENS[gen_name]
    Xp_tr, Xn_tr = gen(n_train, rng=rng, **gen_kwargs)
    Xp_te, Xn_te = gen(N_TEST, rng=rng, **gen_kwargs)
    if tau_mode == "selected":
        tau = select_tau(Xp_tr, Xn_tr)
    elif tau_mode == "fixed03":
        tau = 0.30
    elif tau_mode == "random":
        tau = float(np.random.default_rng(seed).choice(TAU_GRID))
    else:
        raise ValueError(tau_mode)
    auc = fit_score(model, Xp_tr, Xn_tr, Xp_te, Xn_te, tau, seed, ops)
    return auc, tau


S3 = [0, 1, 2]   # seeds for secondary sweeps

def all_jobs():
    """Flat, deterministic job list across every sweep."""
    jobs = []
    def add(sweep, gen, model, n, seed, kw=None, ops="full", tau_mode="selected", tag=""):
        jobs.append(dict(sweep=sweep, gen=gen, model=model, n_train_pairs=n,
                         seed=seed, kwargs=kw or {}, ops=ops, tau_mode=tau_mode,
                         tag=tag))
    # tasks: three constructions x sizes x models (headline, 5 seeds)
    for gen in ["fall", "sit2stand", "burst"]:
        for n in [100, 250, 1000]:
            for seed in SEEDS:
                for m in ["dts_hgb", "alpha_lr", "bof_hgb", "minirocket", "raw_mlp"]:
                    add("tasks", gen, m, n, seed)
    # main: sample-size sweep on the fall construction (5 seeds)
    for n in [50, 100, 250, 500, 1000, 2000]:
        for seed in SEEDS:
            for m in ["dts_hgb", "alpha_lr", "bof_hgb", "raw_mlp", "minirocket"]:
                add("main", "fall", m, n, seed)
    # operators ablation (3 seeds)
    for ops in ["full", "noalpha", "alpha", "order_invariant_only",
                "order_only", "quantiles", "slopes", "endpoints"]:
        for seed in S3:
            add("operators", "fall", "dts_hgb", 250, seed, ops=ops, tag=f"ops={ops}")
    for tau_mode in ["selected", "fixed03", "random"]:
        for seed in S3:
            add("operators", "fall", "alpha_lr", 250, seed, tau_mode=tau_mode,
                tag=f"taumode={tau_mode}")
    # timing (3 seeds), includes misspecified fixed tau
    for p1 in [0.15, 0.3, 0.5, 0.7]:
        for seed in S3:
            for m in ["dts_hgb", "alpha_lr", "bof_hgb"]:
                add("timing", "fall", m, 250, seed, kw={"p1": p1}, tag=f"p1={p1}")
            add("timing", "fall", "alpha_lr", 250, seed, kw={"p1": p1},
                tau_mode="fixed03", tag=f"p1={p1}|fixedtau")
    # noise (3 seeds)
    for snr in [1.0, 2.0, 4.0, 8.0, 16.0]:
        for seed in S3:
            for m in ["dts_hgb", "alpha_lr", "bof_hgb", "minirocket"]:
                add("noise", "fall", m, 250, seed, kw={"snr": snr}, tag=f"snr={snr}")
    # distractor channels (3 seeds)
    for dn in [0, 8, 32, 120]:
        for seed in S3:
            for m in ["dts_hgb", "alpha_lr", "bof_hgb", "minirocket"]:
                add("distractor", "fall", m, 250, seed, kw={"d_noise": dn},
                    tag=f"dnoise={dn}")
    # event duration (3 seeds)
    for dur in [0.05, 0.1, 0.2, 0.4]:
        for seed in S3:
            for m in ["dts_hgb", "alpha_lr", "bof_hgb"]:
                add("duration", "fall", m, 250, seed, kw={"dur": dur}, tag=f"dur={dur}")
    # sequence length (3 seeds)
    for T in [30, 60, 150, 300]:
        for seed in S3:
            for m in ["dts_hgb", "alpha_lr", "bof_hgb"]:
                add("length", "fall", m, 250, seed, kw={"T": T}, tag=f"T={T}")
    return jobs


def job_key(j):
    kw = ",".join(f"{k}={v}" for k, v in sorted(j["kwargs"].items()))
    return (f'{j["sweep"]}|{j["gen"]}|{j["model"]}|{j["n_train_pairs"]}|'
            f'{j["seed"]}|{j["ops"]}|{j["tau_mode"]}|{kw}')


def resume(budget_s=34.0):
    """Run pending jobs until the time budget is exhausted. Appends each
    completed run to results_all.jsonl immediately (resumable)."""
    path = os.path.join(RESULT_DIR, "results_all.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    done.add(job_key(json.loads(line)))
                except Exception:
                    pass
    jobs = [j for j in all_jobs() if job_key(j) not in done]
    print(f"pending: {len(jobs)}", flush=True)
    t0 = time.time()
    f = open(path, "a")
    ran = 0
    for j in jobs:
        if time.time() - t0 > budget_s:
            break
        auc, tau = run_one(j["gen"], j["model"], j["n_train_pairs"], j["seed"],
                           j["kwargs"], j["ops"], j["tau_mode"])
        if auc is None:
            continue
        rec = dict(j); rec["tau"] = tau; rec["auroc"] = auc
        f.write(json.dumps(rec) + "\n"); f.flush()
        ran += 1
        print(f'[{time.time()-t0:5.1f}s] {j["sweep"]} {j["gen"]} {j["model"]} '
              f'n={j["n_train_pairs"]} seed={j["seed"]} {j["tag"]} AUROC={auc:.4f}',
              flush=True)
    remaining = len(jobs) - ran
    print(f"DONE_THIS_CALL ran={ran} remaining={remaining}", flush=True)
    return remaining


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "resume"
    if which == "resume":
        budget = float(sys.argv[2]) if len(sys.argv) > 2 else 34.0
        sys.exit(0 if resume(budget) == 0 else 3)
    else:
        raise SystemExit("use: python3 synth_bench.py resume [budget_s]")
