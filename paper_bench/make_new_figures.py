#!/usr/bin/env python3
"""Regenerate the two figures added in the revision, from artifacts only.

  fig_bed_compare.pdf   Bed few-shot recovery, DTS+HGB vs MiniROCKET+ridge
                        (from results/new_baselines/bed_compare_summary.json)
  fig_score_dists.pdf   score distributions: FallVision / GMDCSA-24 / Bed pre-post
                        (from saved score vectors + research/bed_deepdive.json)

Run from repo root: python3 paper_bench/make_new_figures.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "legend.fontsize": 7.2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})

# ---- Bed few-shot comparison ------------------------------------------------
s = json.load(open(R / "results/new_baselines/bed_compare_summary.json"))
Ks = sorted(int(k) for k in s)
fig, ax = plt.subplots(figsize=(3.3, 2.15))
ax.errorbar(Ks, [s[str(k)]["hgb"][0] for k in Ks],
            yerr=1.96 * np.array([s[str(k)]["hgb"][1] for k in Ks]),
            marker="o", ms=3.5, lw=1.4, capsize=2, label="DTS+HGB", color="#c0392b")
ax.errorbar(Ks, [s[str(k)]["mr"][0] for k in Ks],
            yerr=1.96 * np.array([s[str(k)]["mr"][1] for k in Ks]),
            marker="s", ms=3.5, lw=1.4, capsize=2, ls="--",
            label="MiniROCKET+Ridge", color="#7f8c8d")
ax.set_xlabel("labelled Bed clips added ($K$)"); ax.set_ylabel("Bed AUROC")
ax.set_title("Few-shot Bed recovery (same protocol)")
ax.legend(loc="lower right"); ax.grid(alpha=0.25, lw=0.4)
fig.tight_layout(); fig.savefig(OUT / "fig_bed_compare.pdf"); plt.close(fig)
print("wrote fig_bed_compare.pdf")

# ---- Score distributions ----------------------------------------------------
z = np.load(R / "results/twinfree/ninefive_core_full/score_vectors_twinfree.npz")
yfv, sfv = z["y_test"].astype(int), z["DTS+HGB_test"]
g = np.load(R / "results/gmdcsa/scores.npz", allow_pickle=True)
yg, sg = g["y"].astype(int), g["scores"]
bd = json.load(open(R / "research/bed_deepdive.json"))
yb = np.array(bd["y"]); s0 = np.array(bd["scores_k0"]); s2 = np.array(bd["scores_k200"])
fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.9))
bins = np.linspace(0, 1, 33)
def panel(ax, sc, yy, title):
    ax.hist(sc[yy == 1], bins=bins, alpha=0.6, density=True, label="fall", color="#c0392b")
    ax.hist(sc[yy == 0], bins=bins, alpha=0.6, density=True, label="non-fall", color="#2980b9")
    ax.axvline(0.5, ls="--", lw=0.9, c="k"); ax.set_title(title); ax.set_yticks([])
panel(axes[0], sfv, yfv, "FallVision strict test")
panel(axes[1], sg, yg, "GMDCSA-24 zero-shot")
ax = axes[2]
ax.hist(s0[yb == 1], bins=bins, alpha=0.45, density=True, label="fall, $K{=}0$", color="#c0392b")
ax.hist(s0[yb == 0], bins=bins, alpha=0.45, density=True, label="non-fall, $K{=}0$", color="#2980b9")
ax.hist(s2[yb == 1], bins=bins, histtype="step", lw=1.3, density=True, label="fall, $K{=}200$", color="#7b241c")
ax.hist(s2[yb == 0], bins=bins, histtype="step", lw=1.3, density=True, label="non-fall, $K{=}200$", color="#1a5276")
ax.axvline(0.5, ls="--", lw=0.9, c="k"); ax.set_title("Bed fold: before/after few-shot"); ax.set_yticks([])
axes[0].legend(loc="upper center"); axes[2].legend(loc="upper left", fontsize=5.8)
for a in axes: a.set_xlabel("DTS+HGB score")
fig.tight_layout(); fig.savefig(OUT / "fig_score_dists.pdf"); plt.close(fig)
print("wrote fig_score_dists.pdf")
