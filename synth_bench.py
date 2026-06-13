#!/usr/bin/env python3
"""
Comprehensive controlled-benchmark family.

Constructions:
  fall       3-phase hip trajectory (still → drop → rest) vs permuted negatives
  sit2stand  gradual height ramp (floor → standing) vs permuted negatives
  burst      early motion burst vs time-reversed sequence

Sweeps (each based on 'fall' unless noted, seeds 0-2):
  training_size   n ∈ {50, 100, 250, 500, 1000, 2000}
  snr             drop amplitude ∈ {1.5, 2.0, 3.0, 5.0, 10.0}
  onset_p1        event start fraction ∈ {0.15, 0.25, 0.35, 0.45}
  event_dur       event duration fraction ∈ {0.08, 0.12, 0.18, 0.25}
  seq_len         T ∈ {30, 60, 120, 240}
  distractors     extra uninformative channels ∈ {0, 2, 4, 8}
  operator_bank   feature subsets: full, no_asym, no_diff, stats_only
  tau_select      τ selection: tau_star, fixed_030, fixed_050

Headline runs use seeds 0-4; sweeps use seeds 0-2.
Results are appended to results_all.jsonl and never recomputed (resumable).

Usage:
  python3 synth_bench.py              # full run
  python3 synth_bench.py resume       # skip already-done experiments (same as default)
  python3 synth_bench.py resume N     # N ignored (kept for interface compatibility)
"""
import sys, json, time
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from dts.features import temporal_op16

OUTFILE = Path(__file__).parent / "results" / "results_all.jsonl"
N_CLIPS = 600   # clips per construction (300 pos + 300 neg)

# ── Skeleton generators ───────────────────────────────────────────────────────

def make_fall_skeleton(T, rng, onset_p1=0.30, event_dur=0.15, snr=5.0, n_distractor=0):
    """Three-phase fall (still → drop → floor) vs random permutation."""
    t1 = int(T * onset_p1)
    t2 = t1 + max(1, int(T * event_dur))
    t2 = min(t2, T - 1)
    drop = snr * 0.15  # hip drops by snr * body_unit
    hip_y = np.concatenate([
        np.full(t1, 1.0),
        np.linspace(1.0, 1.0 - drop, max(1, t2 - t1)),
        np.full(T - t2, 1.0 - drop)
    ])
    hip_y = np.clip(hip_y + rng.normal(0, 0.01, T), -0.5, 2.0)
    hip_x = np.cumsum(rng.normal(0, 0.005, T))
    seq = _skeleton_around_hip(T, hip_x, hip_y, rng)
    if n_distractor > 0:
        dist = rng.normal(0, 0.02, (T, n_distractor, 2))
        seq = np.concatenate([seq, dist], axis=1)
    conf = np.ones((T, seq.shape[1]))
    return seq, conf


def make_sit2stand_skeleton(T, rng, onset_p1=0.25, event_dur=0.50, snr=3.0, n_distractor=0):
    """Gradual rise from floor to standing (opposite temporal structure to fall)."""
    t1 = int(T * onset_p1)
    t2 = t1 + max(1, int(T * event_dur))
    t2 = min(t2, T - 1)
    rise = snr * 0.15
    hip_y = np.concatenate([
        np.full(t1, 1.0 - rise),
        np.linspace(1.0 - rise, 1.0, max(1, t2 - t1)),
        np.full(T - t2, 1.0)
    ])
    hip_y = np.clip(hip_y + rng.normal(0, 0.01, T), -0.5, 2.0)
    hip_x = np.cumsum(rng.normal(0, 0.005, T))
    seq = _skeleton_around_hip(T, hip_x, hip_y, rng)
    if n_distractor > 0:
        dist = rng.normal(0, 0.02, (T, n_distractor, 2))
        seq = np.concatenate([seq, dist], axis=1)
    conf = np.ones((T, seq.shape[1]))
    return seq, conf


def make_burst_skeleton(T, rng, snr=4.0, n_distractor=0):
    """Early motion burst — high |Δz| in first 30% of sequence."""
    t_burst = max(1, int(T * 0.30))
    hip_y = np.ones(T)
    hip_y[:t_burst] += np.cumsum(rng.normal(0, snr * 0.03, t_burst))
    hip_y = np.clip(hip_y + rng.normal(0, 0.008, T), -0.5, 2.0)
    hip_x = np.cumsum(rng.normal(0, 0.005, T))
    seq = _skeleton_around_hip(T, hip_x, hip_y, rng)
    if n_distractor > 0:
        dist = rng.normal(0, 0.02, (T, n_distractor, 2))
        seq = np.concatenate([seq, dist], axis=1)
    conf = np.ones((T, seq.shape[1]))
    return seq, conf


def _skeleton_around_hip(T, hip_x, hip_y, rng):
    """Build a 17-joint skeleton centred on hip trajectory."""
    base = np.array([
        [0, 0.75], [.03,.78], [-.03,.78], [.06,.76], [-.06,.76],
        [.15,.55], [-.15,.55], [.22,.3], [-.22,.3],
        [.25,.05], [-.25,.05], [.1, 0], [-.1, 0],
        [.12,-.45], [-.12,-.45], [.13,-.9], [-.13,-.9]
    ], dtype=np.float32)
    seq = np.zeros((T, 17, 2), np.float32)
    for t in range(T):
        frac = max(0, 1 - (hip_y[t] - 0.15) / 0.85)
        ang  = frac * (np.pi / 2) * 0.9
        R    = np.array([[np.cos(ang), -np.sin(ang)],
                         [np.sin(ang),  np.cos(ang)]])
        seq[t] = base @ R.T + [hip_x[t], hip_y[t]]
        seq[t] += rng.normal(0, 0.008, (17, 2)).astype(np.float32)
    return seq


# ── Feature extractors ─────────────────────────────────────────────────────────

def dts_features(seq, conf, tau=0.30, mode="full"):
    """DTS feature extraction with operator-bank ablation support."""
    from dts.features import normalise, EPS
    seq_n, s = normalise(seq[:, :17, :])  # only first 17 joints for DTS
    hip  = (seq[:, 11, :] + seq[:, 12, :]) / 2.0
    sm   = (seq_n[:, 5, :] + seq_n[:, 6, :]) / 2.0
    tv   = seq_n[:, 0, :] - sm

    prims = [
        seq_n[:, :, 0].max(1) - seq_n[:, :, 0].min(1),  # BBox-W
        seq_n[:, 11, 1],                                  # Hip-Y
        np.arctan2(tv[:, 0], tv[:, 1] + EPS),            # Torso-Ang
        _hip_speed(hip, s),                               # Hip-Spd
        _ctr_speed(seq_n),                                # Ctr-Spd
        _hip_acc(hip, s),                                 # Hip-Acc
        sm[:, 1],                                         # Shldr-Y
        seq_n[:, 0, 1],                                   # Head-Y
    ]
    if mode == "full":
        return np.concatenate([temporal_op16(p, tau) for p in prims])
    elif mode == "no_asym":
        return np.concatenate([temporal_op16(p, tau)[:15] for p in prims])
    elif mode == "no_diff":
        # drop elements 10 (delta), 11 (slope), 12 (mean|Δ|), 13 (max|Δ|), 14 (max|Δ²|), 15 (α)
        idx = [0,1,2,3,4,5,6,7,8,9]
        return np.concatenate([temporal_op16(p, tau)[idx] for p in prims])
    elif mode == "stats_only":
        idx = [0,1,2,3,4,5,6,7]  # mean, std, min, max, Q25, Q50, Q75, range
        return np.concatenate([temporal_op16(p, tau)[idx] for p in prims])
    raise ValueError(f"Unknown mode: {mode}")


def _hip_speed(hip, s):
    dhc = np.zeros_like(hip)
    dhc[1:] = hip[1:] / s - hip[:-1] / s
    return np.linalg.norm(dhc, axis=1)

def _ctr_speed(seq_n):
    T = len(seq_n)
    cx = seq_n[:, :, 0].mean(1); cy = seq_n[:, :, 1].mean(1)
    dcxy = np.zeros((T, 2))
    dcxy[1:] = np.column_stack([cx, cy])[1:] - np.column_stack([cx, cy])[:-1]
    return np.linalg.norm(dcxy, axis=1)

def _hip_acc(hip, s):
    spd = np.zeros(len(hip))
    dhc = np.zeros_like(hip)
    dhc[1:] = hip[1:] / s - hip[:-1] / s
    spd = np.linalg.norm(dhc, axis=1)
    acc = np.zeros(len(spd))
    acc[1:] = spd[1:] - spd[:-1]
    return acc


# ── Dataset builders ──────────────────────────────────────────────────────────

def build_dataset(construction, rng, T=90, n_clips=N_CLIPS, **kw):
    """Return (X_dts, y) for a construction."""
    X, y = [], []
    tau = kw.pop("tau", 0.30)
    mode = kw.pop("mode", "full")
    tau_mode = kw.pop("tau_mode", "fixed")  # fixed | data_driven

    gen_pos = {"fall": make_fall_skeleton,
               "sit2stand": make_sit2stand_skeleton,
               "burst": make_burst_skeleton}[construction]

    seqs_pos, confs_pos = [], []
    for _ in range(n_clips):
        Ti = int(rng.integers(max(30, T//2), T + T//2 + 1))
        seq, conf = gen_pos(Ti, rng, **kw)
        seqs_pos.append((seq, conf))

    # τ selection for data-driven mode
    if tau_mode == "data_driven" and construction == "fall":
        from dts.features import find_tau_star
        clips_mock = [(s[:,:17,:], c[:,:17]) for s,c in seqs_pos[:min(50, n_clips)]]
        labels_mock = np.ones(len(clips_mock))
        tau = find_tau_star(clips_mock, labels_mock)

    for seq, conf in seqs_pos:
        feat = dts_features(seq, conf, tau=tau, mode=mode)
        X.append(feat); y.append(1)
        perm = rng.permutation(len(seq))
        feat_neg = dts_features(seq[perm], conf[perm], tau=tau, mode=mode)
        if construction == "burst":
            feat_neg = dts_features(seq[::-1], conf[::-1], tau=tau, mode=mode)
        X.append(feat_neg); y.append(0)

    return np.array(X), np.array(y)


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate_cv(X, y, rng_seed=42):
    """5-fold stratified CV AUROC with HGB."""
    skf = StratifiedKFold(5, shuffle=True, random_state=rng_seed)
    aucs = []
    for tr, te in skf.split(X, y):
        m = HistGradientBoostingClassifier(max_iter=300, random_state=42)
        m.fit(X[tr], y[tr])
        aucs.append(float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])))
    return float(np.mean(aucs))


# ── Experiment registry ───────────────────────────────────────────────────────

def build_experiments():
    """Return list of (key_dict, run_fn) pairs."""
    exps = []
    headline_seeds = list(range(5))
    sweep_seeds    = list(range(3))

    # ─ headline constructions (seeds 0-4) ─────────────────────────────────────
    for construction in ["fall", "sit2stand", "burst"]:
        for seed in headline_seeds:
            exps.append({
                "construction": construction, "sweep": "headline",
                "params": {}, "seed": seed,
            })

    # ─ sweeps (fall construction, seeds 0-2) ──────────────────────────────────
    for seed in sweep_seeds:
        for n_tr in [50, 100, 250, 500, 1000, 2000]:
            exps.append({"construction":"fall","sweep":"training_size",
                         "params":{"n_clips": n_tr},"seed":seed})
        for snr in [1.5, 2.0, 3.0, 5.0, 10.0]:
            exps.append({"construction":"fall","sweep":"snr",
                         "params":{"snr": snr},"seed":seed})
        for p1 in [0.15, 0.25, 0.35, 0.45]:
            exps.append({"construction":"fall","sweep":"onset_p1",
                         "params":{"onset_p1": p1},"seed":seed})
        for dur in [0.08, 0.12, 0.18, 0.25]:
            exps.append({"construction":"fall","sweep":"event_dur",
                         "params":{"event_dur": dur},"seed":seed})
        for T in [30, 60, 120, 240]:
            exps.append({"construction":"fall","sweep":"seq_len",
                         "params":{"T": T},"seed":seed})
        for nd in [0, 2, 4, 8]:
            exps.append({"construction":"fall","sweep":"distractors",
                         "params":{"n_distractor": nd},"seed":seed})
        for mode in ["full", "no_asym", "no_diff", "stats_only"]:
            exps.append({"construction":"fall","sweep":"operator_bank",
                         "params":{"mode": mode},"seed":seed})
        for tm in ["fixed_030", "fixed_050", "data_driven"]:
            tau = {"fixed_030": 0.30, "fixed_050": 0.50, "data_driven": 0.30}[tm]
            exps.append({"construction":"fall","sweep":"tau_select",
                         "params":{"tau": tau, "tau_mode": tm if tm == "data_driven" else "fixed"},
                         "seed":seed})
    return exps


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)

    # Load already-completed experiments
    done = set()
    if OUTFILE.exists():
        for line in OUTFILE.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done.add(json.dumps({
                "construction": r["construction"],
                "sweep": r["sweep"],
                "params": r["params"],
                "seed": r["seed"],
            }, sort_keys=True))

    experiments = build_experiments()
    n_total = len(experiments)
    n_skip  = 0
    n_run   = 0

    print(f"Total experiments: {n_total}  |  Already done: {len(done)}")

    with OUTFILE.open("a") as fh:
        for i, exp in enumerate(experiments):
            key = json.dumps({k: exp[k] for k in ("construction","sweep","params","seed")},
                             sort_keys=True)
            if key in done:
                n_skip += 1
                continue

            construction = exp["construction"]
            seed         = exp["seed"]
            params       = dict(exp["params"])
            T            = params.pop("T", 90)
            n_clips      = params.pop("n_clips", N_CLIPS)

            rng = np.random.default_rng(seed)
            t0  = time.perf_counter()
            X, y = build_dataset(construction, rng, T=T, n_clips=n_clips, **params)
            auroc = evaluate_cv(X, y, rng_seed=seed)
            elapsed = time.perf_counter() - t0

            result = {**exp, "auroc": auroc, "elapsed_s": round(elapsed, 2)}
            fh.write(json.dumps(result) + "\n")
            fh.flush()
            done.add(key)
            n_run += 1
            if n_run % 20 == 1 or n_run <= 5:
                print(f"[{n_run:3d}/{n_total-n_skip-len(done)+n_run:3d}]  "
                      f"{construction:10s} {exp['sweep']:14s} "
                      f"params={params} seed={seed}  AUROC={auroc:.4f}  ({elapsed:.1f}s)")

    print(f"\nDone.  Ran {n_run} new experiments, skipped {n_skip}.")
    print(f"Results in: {OUTFILE}")


if __name__ == "__main__":
    main()
