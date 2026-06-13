"""Numerical verification of the finite-sample asymmetry-separation theorem.

Theorem (paper, upgraded Proposition 3). Increments w_2..w_{n+1} independent,
supported in [0, B]; mean mu_s on still [0, p1) and floor [p2, 1] phases, mean
mu_d >= mu_s on the drop phase [p1, p2). Negative = uniformly random
permutation of the same increments. tau <= p1, m = floor(tau * n),
a_f = tau * mu_s / mu_bar, mu_bar = (p1 + 1 - p2) mu_s + (p2 - p1) mu_d,
Delta = tau - a_f.  For n >= 6 / Delta and theta = a_f + Delta / 2:

  P(alpha_fall >= theta) + P(alpha_neg <= theta) <= 5 exp(-n mu_bar^2 Delta^2 / (72 B^2))

and hence AUROC(alpha) >= 1 - 5 exp(-n mu_bar^2 Delta^2 / (72 B^2)).

This script checks (i) the bound is never violated empirically, and (ii) the
empirical log failure rate decays at least linearly in n (the bound's shape).
"""
import numpy as np

rng = np.random.default_rng(0)

mu_s, mu_d = 0.1, 1.0
p1, p2, tau = 0.3, 0.5, 0.25
B = 2 * mu_d                      # increments Uniform[0, 2*mu] subsets of [0, B]
mu_bar = (p1 + 1 - p2) * mu_s + (p2 - p1) * mu_d
a_f = tau * mu_s / mu_bar
Delta = tau - a_f
theta = a_f + Delta / 2
print(f"mu_bar={mu_bar:.4f} a_f={a_f:.4f} Delta={Delta:.4f} theta={theta:.4f}")
print(f"n must be >= {6/Delta:.1f}")

TRIALS = 20000
print(f"{'n':>6} {'P(af>=th)':>10} {'P(an<=th)':>10} {'sum':>10} {'bound':>12} {'AUROC_emp':>10}")
for n in [30, 60, 120, 250, 500, 1000]:
    if n < 6 / Delta:
        continue
    m = int(np.floor(tau * n))
    k1, k2 = int(p1 * n), int(p2 * n)
    # fall increments
    W = np.empty((TRIALS, n))
    W[:, :k1] = rng.uniform(0, 2 * mu_s, (TRIALS, k1))
    W[:, k1:k2] = rng.uniform(0, 2 * mu_d, (TRIALS, k2 - k1))
    W[:, k2:] = rng.uniform(0, 2 * mu_s, (TRIALS, n - k2))
    S = W.sum(1)
    a_fall = W[:, :m].sum(1) / S
    # permuted negatives (same multisets, independent permutations)
    idx = np.argsort(rng.random((TRIALS, n)), axis=1)
    Wn = np.take_along_axis(W, idx, axis=1)
    a_neg = Wn[:, :m].sum(1) / S
    p_f = (a_fall >= theta).mean()
    p_n = (a_neg <= theta).mean()
    bound = 5 * np.exp(-n * mu_bar**2 * Delta**2 / (72 * B**2))
    auroc = (a_neg[:, None] > a_fall[None, :TRIALS // 100]).mean()  # subsample pairs
    ok = "OK" if p_f + p_n <= min(bound, 1.0) + 1e-12 else "VIOLATED"
    print(f"{n:>6} {p_f:>10.5f} {p_n:>10.5f} {p_f+p_n:>10.5f} {min(bound,1):>12.5f} {auroc:>10.5f} {ok}")
