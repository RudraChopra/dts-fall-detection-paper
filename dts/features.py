"""
DTS-128 Feature Extractor
=========================
Extracts the 128-dimensional Dynamic Trajectory Signature from a skeleton clip.

Input : (seq, conf) where
        seq  : np.ndarray  shape (T, 17, 2)   -- COCO keypoints (x, y)
        conf : np.ndarray  shape (T, 17)       -- per-joint confidence scores

Output: np.ndarray shape (128,)
        8 kinematic families × 16 temporal operators each.
        Element [i*16 + 15] is the temporal asymmetry α for family i.

Family order:
    0  BBox-W       bounding-box width
    1  Hip-Y        normalised hip-centre height
    2  Torso-Ang    torso orientation angle
    3  Hip-Spd      hip-centre frame-to-frame speed
    4  Ctr-Spd      centre-of-mass speed
    5  Hip-Acc      hip-centre acceleration (1st diff of speed)
    6  Shldr-Y      shoulder midpoint height
    7  Head-Y       head (nose) height
"""

import numpy as np

EPS = 1e-8


# ──────────────────────────────────────────────────────────────────────────────
#  16-element temporal operator
# ──────────────────────────────────────────────────────────────────────────────

def temporal_op16(z: np.ndarray, tau: float = 0.30) -> np.ndarray:
    """
    Apply the 16-element operator Ω(z) to a scalar trajectory z.

    Elements:
        0  mean
        1  std
        2  min
        3  max
        4  Q25
        5  Q50
        6  Q75
        7  range (max - min)
        8  first value z[0]
        9  last  value z[-1]
       10  delta  z[-1] - z[0]
       11  OLS slope β
       12  mean |Δz|
       13  max  |Δz|
       14  max  |Δ²z|
       15  temporal asymmetry α(z; τ)
    """
    z = np.asarray(z, dtype=np.float64)
    T = len(z)
    if T < 3:
        return np.zeros(16)

    dz  = np.zeros(T);  dz[1:]  = z[1:]  - z[:-1]
    d2z = np.zeros(T);  d2z[1:] = dz[1:] - dz[:-1]

    sp  = max(1, int(tau * T))
    al  = float(np.abs(dz[1:sp + 1]).sum() / (np.abs(dz[1:]).sum() + EPS))
    b   = float(np.polyfit(np.arange(T, dtype=float), z, 1)[0])

    return np.array([
        z.mean(), z.std() + EPS, z.min(), z.max(),
        np.percentile(z, 25), np.percentile(z, 50), np.percentile(z, 75),
        z.max() - z.min(), z[0], z[-1], z[-1] - z[0], b,
        np.abs(dz).mean(), np.abs(dz).max(), np.abs(d2z).max(), al
    ], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
#  Normalisation
# ──────────────────────────────────────────────────────────────────────────────

def normalise(seq: np.ndarray) -> tuple:
    """
    Hip-centred, torso-length-normalised coordinates.

    Returns (n, s) where
        n  : np.ndarray (T, 17, 2)  normalised coordinates
        s  : float                  median torso length (body scale)
    """
    seq = np.asarray(seq, dtype=np.float64)
    hip_centre = (seq[:, 11, :] + seq[:, 12, :]) / 2.0
    torso_len  = np.linalg.norm(seq[:, 5, :] - hip_centre, axis=1)
    s          = float(np.median(torso_len)) + EPS
    n          = (seq - hip_centre[:, np.newaxis, :]) / s
    return n, s


# ──────────────────────────────────────────────────────────────────────────────
#  Main extractor
# ──────────────────────────────────────────────────────────────────────────────

def extract_dts128(
    seq:  np.ndarray,
    conf: np.ndarray,
    tau:  float = 0.30,
) -> np.ndarray:
    """
    Extract the 128-dimensional DTS feature vector from a single clip.

    Parameters
    ----------
    seq  : (T, 17, 2) array of COCO keypoint coordinates
    conf : (T, 17)    array of per-joint confidence scores
    tau  : split ratio for temporal asymmetry (default τ*=0.30)

    Returns
    -------
    np.ndarray shape (128,)
    """
    seq  = np.asarray(seq,  dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64)
    T    = seq.shape[0]

    # Normalise
    n, s = normalise(seq)
    hip  = (seq[:, 11, :] + seq[:, 12, :]) / 2.0

    # Family 0 – bounding-box width
    xr = n[:, :, 0].max(1) - n[:, :, 0].min(1)

    # Family 1 – hip-centre height (normalised)
    hip_y = n[:, 11, 1]

    # Family 2 – torso angle
    sm     = (n[:, 5, :] + n[:, 6, :]) / 2.0
    tv     = n[:, 0, :] - sm
    torso_angle = np.arctan2(tv[:, 0], tv[:, 1] + EPS)

    # Family 3 – hip speed
    dhc    = np.zeros_like(hip)
    dhc[1:] = hip[1:] / s - hip[:-1] / s
    hip_spd = np.linalg.norm(dhc, axis=1)

    # Family 4 – centre-of-mass speed
    cx = n[:, :, 0].mean(1);  cy = n[:, :, 1].mean(1)
    dcxy    = np.zeros((T, 2))
    dcxy[1:] = np.column_stack([cx, cy])[1:] - np.column_stack([cx, cy])[:-1]
    ctr_spd = np.linalg.norm(dcxy, axis=1)

    # Family 5 – hip acceleration
    hip_acc    = np.zeros(T)
    hip_acc[1:] = hip_spd[1:] - hip_spd[:-1]

    # Family 6 – shoulder height
    shldr_y = sm[:, 1]

    # Family 7 – head (nose) height
    head_y = n[:, 0, 1]

    primitives = [xr, hip_y, torso_angle, hip_spd, ctr_spd, hip_acc, shldr_y, head_y]
    return np.concatenate([temporal_op16(p, tau) for p in primitives])


# ──────────────────────────────────────────────────────────────────────────────
#  Optimal τ* estimation
# ──────────────────────────────────────────────────────────────────────────────

def find_tau_star(
    clips:  list,
    labels: np.ndarray,
    grid:   np.ndarray = None,
) -> float:
    """
    Estimate τ* from labelled training clips by maximising the summed
    Welch t-statistic between fall and non-fall asymmetry distributions
    across all eight primitive trajectories.

    Parameters
    ----------
    clips  : list of (seq, conf) tuples
    labels : (N,) array, 1=fall 0=non-fall
    grid   : optional 1-D array of τ candidates

    Returns
    -------
    float  optimal τ*
    """
    from scipy.stats import ttest_ind

    if grid is None:
        grid = np.linspace(0.20, 0.80, 13)

    best_tau, best_score = 0.30, -1.0

    for tau in grid:
        # Extract asymmetry element (index 15 of each 16-dim block) for each clip
        alpha_f, alpha_n = [], []
        for (seq, conf), lbl in zip(clips, labels):
            feat = extract_dts128(seq, conf, tau=tau)
            # 8 alpha values (one per family)
            alphas = feat[[i * 16 + 15 for i in range(8)]]
            if lbl == 1:
                alpha_f.append(alphas)
            else:
                alpha_n.append(alphas)

        if min(len(alpha_f), len(alpha_n)) < 5:
            continue

        af = np.array(alpha_f)   # (N_f, 8)
        an = np.array(alpha_n)   # (N_n, 8)

        score = sum(abs(ttest_ind(af[:, m], an[:, m]).statistic) for m in range(8))
        if score > best_score:
            best_score, best_tau = score, float(tau)

    return best_tau
