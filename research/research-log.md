# Autoresearch log

## Reproducibility note (Bed pipeline)
The shipped feature cache `dts_one_run/fallvision_dts128_features.npz` is rank-correlated
with the paper's twinfree extraction (GMDCSA zero-shot reproduces, Spearman 0.87) but is NOT
bit-identical: dedup non-Bed -> all-1747-Bed gives AUROC 0.781 vs the paper's twinfree
lfo_tf Bed 0.7123 (same 1747 test, falls=922). The features differ slightly (conf-zeroing /
extraction seed). DECISION: the paper's reviewed numbers (Table 4 Bed 0.7123; bed_adapt
0.714->0.848) stay authoritative. The Bed few-shot FIGURE is built from the paper's
bed_adapt.json. My 5-seed reconstruction (0.778->0.895, monotone, sd<0.01) is reported as
independent confirmation of the recovery SHAPE only. Mechanism diagnostics (threshold-free
AUROC; missed-fall alpha shift) are structural QDA predictions, pipeline-robust.
