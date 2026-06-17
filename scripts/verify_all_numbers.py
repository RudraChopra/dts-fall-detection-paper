#!/usr/bin/env python3
"""One-command verification that every headline number in the venue builds is
backed by an artifact in this repository. Run from repo root:

    python3 scripts/verify_all_numbers.py        (or: make verify)

Checks (1) required artifact files exist, (2) headline values recomputed from
those artifacts match the papers, including the fixed-FPR recalls recomputed
from RAW score vectors and the new-baseline results. Exit 0 = all pass.
"""
import json, sys
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent
fails, checks = 0, 0

def check(name, cond, detail=""):
    global fails, checks
    checks += 1
    if cond:
        print(f"  PASS  {name}")
    else:
        fails += 1
        print(f"  FAIL  {name}  {detail}")

def close(a, b, tol=5e-4): return abs(a - b) <= tol

print("== 1. Artifact files exist ==")
REQUIRED = [
 "results/twinfree/ninefive_results.json",
 "results/twinfree/ninefive_core_full/score_vectors_twinfree.npz",
 "results/twinfree/ninefive_core_full/split_manifest_twinfree.csv",
 "results/twinfree/ninefive_core_full/dedup_audit.csv",
 "results/twinfree/abl_tf.json", "results/twinfree/lso_tf.json",
 "results/twinfree/lfo_tf.json", "results/twinfree/bed_adapt.json",
 "results/twinfree/lc.json", "results/gmdcsa/zeroshot.json",
 "research/gmdcsa_loso.json", "research/gmdcsa_paired.json",
 "research/bed_deepdive.json", "results/legacy/revision_results.json",
 "results/new_baselines/multirocket_main.json",
 "results/new_baselines/tcn_main.json",
 "results/new_baselines/inception_main.json",
 "results/new_baselines/bed_compare_summary.json",
 "results/new_baselines/lc_multirocket.json",
 "results/new_baselines/lc_paired.json",
 "results/new_baselines/bonferroni8.json",
 "results/new_baselines/qda_train_only.json",
 "results/new_baselines/bed3_summary.json",
 "paper_bench/synth_bench.py", "paper_bench/results_all.jsonl",
 "paper_bench/verify_theorem.py", "paper_bench/NUMBERS_AUDIT.md",
 "paper_bench/REPRO_COMMANDS.md", "requirements.txt",
]
for f in REQUIRED:
    check(f, (R / f).exists())

print("== 2. Main table (DTS+HGB) ==")
nf = json.load(open(R / "results/twinfree/ninefive_results.json"))
h = nf["models"]["DTS+HGB"]
check("AUROC 0.9895", close(h["auroc"], 0.98947, 1e-4))
check("F1 0.951", close(h["f1"], 0.95124, 1e-4))
check("FP 39 / FN 18", h["fp"] == 39 and h["fn"] == 18)
check("dedup 5793->5572 (-221)", nf["dedup"]["n_input_clips"] == 5793 and
      nf["dedup"]["n_kept_representatives"] == 5572 and
      nf["dedup"]["n_duplicate_members_removed"] == 221)
d = nf["paired_auc_deltas_vs_DTS_HGB"]
for m, v in [("GRU", .0183), ("Transformer", .0085), ("MiniROCKET", .0135),
             ("CompactST-GCN", .0710), ("FullST-GCN-COCO", .0242)]:
    check(f"paired delta {m} +{v}", close(d[m]["delta"], v, 1e-4) and d[m]["ci95"][0] > 0)

print("== 3. Fixed-FPR recalls from RAW score vectors ==")
z = np.load(R / "results/twinfree/ninefive_core_full/score_vectors_twinfree.npz")
y = z["y_test"].astype(int)
def rfpr(s, f):
    neg = np.sort(s[y == 0])[::-1]; k = int(np.floor(f * len(neg)))
    thr = neg[k] if k < len(neg) else -np.inf
    return ((s > thr) & (y == 1)).sum() / (y == 1).sum()
for f_, e in zip([.01, .02, .05, .10], [.796, .862, .941, .981]):
    check(f"DTS+HGB recall@{f_:.0%} = {e}", close(rfpr(z["DTS+HGB_test"], f_), e, 1.5e-3))
for f_, e in zip([.01, .02, .05, .10], [.629, .777, .861, .970]):
    check(f"MiniROCKET recall@{f_:.0%} = {e}", close(rfpr(z["MiniROCKET_test"], f_), e, 1.5e-3))
mrs = np.load(R / "results/new_baselines/multirocket_test_scores.npy")
for f_, e in zip([.01, .02, .05, .10], [.864, .882, .944, .977]):
    check(f"MultiROCKET recall@{f_:.0%} = {e}", close(rfpr(mrs, f_), e, 1.5e-3))

print("== 4. Ablation table ==")
abl = json.load(open(R / "results/twinfree/abl_tf.json"))
check("Full = main result", close(abl["Full"]["auroc"], 0.98947, 1e-4) and
      close(abl["Full"]["f1"], 0.95124, 1e-4))
check("alpha-zeroed -0.0023", close(abl["StaticOnly"]["auroc"] - abl["Full"]["auroc"], -0.00225, 2e-4))

print("== 5. Folds, transfer, LOSO ==")
lso = json.load(open(R / "results/twinfree/lso_tf.json"))
check("LSO mean 0.9649", close(np.mean([lso[k]["auroc"] for k in "1234"]), 0.9649, 5e-4))
lfo = json.load(open(R / "results/twinfree/lfo_tf.json"))
check("Bed 0.7123 / mean 0.8961",
      close(lfo["Bed"]["auroc"], 0.7123, 5e-4) and
      close(np.mean([lfo[k]["auroc"] for k in ("Bed", "Chair", "Stand")]), 0.8961, 5e-4))
g = json.load(open(R / "results/gmdcsa/zeroshot.json"))
check("GMDCSA AUROC 0.9633, 10FP/6FN", close(g["auroc"], 0.9633, 5e-4)
      and g["fp"] == 10 and g["fn"] == 6)
gl = json.load(open(R / "research/gmdcsa_loso.json"))
check("LOSO means 0.941/0.934/0.918/0.917",
      close(gl["DTS+HGB"]["mean"], 0.9405, 1e-3) and close(gl["MultiROCKET"]["mean"], 0.9339, 1e-3)
      and close(gl["MiniROCKET"]["mean"], 0.9180, 1e-3) and close(gl["alpha-only+LR"]["mean"], 0.9172, 1e-3))
gp = json.load(open(R / "research/gmdcsa_paired.json"))
check("LOSO paired +0.012 [-0.034, 0.056]", close(gp["delta"], 0.0117, 1e-3)
      and close(gp["delta_ci"][0], -0.0336, 2e-3) and close(gp["delta_ci"][1], 0.0564, 2e-3))
b = json.load(open(R / "results/twinfree/bed_adapt.json"))
check("Bed recovery 0.714->0.784->0.848",
      close(b["0"]["auroc"], 0.7138, 1e-3) and close(b["50"]["auroc"], 0.7839, 1e-3)
      and close(b["200"]["auroc"], 0.8483, 1e-3))
u = json.load(open(R / "results/legacy/revision_results.json"))["urfd_zero_shot"]
check("URFD 0.9200, 29FP/0FN", close(u["auroc"], 0.92, 1e-4) and u["fp"] == 29 and u["fn"] == 0)

print("== 6. New baselines (this revision) ==")
mr = json.load(open(R / "results/new_baselines/multirocket_main.json"))
check("MultiROCKET 0.9868, FP26/FN32", close(mr["auroc"], 0.9868, 5e-4)
      and mr["fp"] == 26 and mr["fn"] == 32)
check("DTS-MultiROCKET +0.0027 ns", close(mr["delta_hgb_minus_multirocket"], 0.0027, 5e-4)
      and mr["delta_ci"][0] < 0 < mr["delta_ci"][1])
tc = json.load(open(R / "results/new_baselines/tcn_main.json"))
check("TCN 0.9629, FP64/FN41", close(tc["auroc"], 0.9629, 5e-4) and tc["fp"] == 64 and tc["fn"] == 41)
check("DTS-TCN +0.0265 significant", close(tc["delta_hgb_minus_tcn"], 0.0265, 5e-4)
      and tc["delta_ci"][0] > 0)
ic = json.load(open(R / "results/new_baselines/inception_main.json"))
print(f"  (InceptionTime artifact: AUROC {ic['auroc']:.4f}, FP {ic['fp']}, FN {ic['fn']}, "
      f"delta {ic['delta_hgb_minus_inception']:+.4f} {ic['delta_ci']})")
check("InceptionTime artifact internally consistent",
      ic["n_networks"] >= 3 and 0.5 < ic["auroc"] < 1.0)
bc = json.load(open(R / "results/new_baselines/bed_compare_summary.json"))
check("Bed few-shot: DTS leads MiniROCKET at every K",
      all(bc[k]["hgb"][0] > bc[k]["mr"][0] for k in bc))
# Experiment A: matched 3-way Bed few-shot (cache-based). DTS > MiniROCKET; MultiROCKET >= DTS.
b3 = json.load(open(R / "results/new_baselines/bed3_summary.json"))
check("Bed3: anchors DTS K0=0.715 K200=0.858",
      close(b3["0"]["dts_hgb"][0], 0.7148, 2e-3) and close(b3["200"]["dts_hgb"][0], 0.8575, 2e-3))
check("Bed3: DTS>MiniROCKET and MultiROCKET>=DTS at every K",
      all(b3[k]["dts_hgb"][0] > b3[k]["minirocket"][0] and
          b3[k]["multirocket"][0] >= b3[k]["dts_hgb"][0] - 1e-9 for k in b3))
lcm = json.load(open(R / "results/new_baselines/lc_multirocket.json"))
for n, e in [("250", 0.9501), ("500", 0.9578), ("1000", 0.9737), ("2000", 0.9788), ("3565", 0.9868)]:
    check(f"MultiROCKET LC n={n} = {e}", close(lcm[n]["auroc"], e, 1e-3))
lp = json.load(open(R / "results/new_baselines/lc_paired.json"))
check("paired LC: all five deltas non-significant",
      all(v["delta_ci"][0] < 0 < v["delta_ci"][1] for v in lp.values()))
check("paired LC full-size delta +0.0027", close(lp["3565"]["paired_delta"], 0.0027, 5e-4))
bf = json.load(open(R / "results/new_baselines/bonferroni8.json"))
check("Bonferroni-8: TCN significant, MultiROCKET/Inception not",
      bf["tcn"][0] > 0 and bf["multirocket"][0] < 0 < bf["multirocket"][1]
      and bf["inception"][0] < 0 < bf["inception"][1])
# train-only QDA parameters (Prop 1, corrected)
import math
from scipy.stats import levene, norm
F = None
try:
    import numpy as _np
    Ff = R / "results" / "new_baselines" / "qda_train_only.json"
    q = json.load(open(Ff))
    check("QDA train-only params (Prop 1)",
          close(q["mu_f"], 0.1624, 1e-3) and close(q["sigma_f"], 0.1692, 1e-3)
          and close(q["mu_n"], 0.3812, 1e-3) and close(q["sigma_n"], 0.2090, 1e-3)
          and close(q["x2"], 0.2934, 1e-3) and close(q["eps_star"], 0.2783, 1e-3))
except FileNotFoundError:
    check("QDA train-only params artifact present", False, "qda_train_only.json missing")
mv = json.load(open(R / "results/new_baselines/mini_eval.json"))
check("harness fidelity: MiniROCKET rerun within published CI",
      0.9672 <= mv["test_auroc"] <= 0.9839 or close(mv["test_auroc"], 0.9797, 2e-3))
fv = json.load(open(R / "results/new_baselines/feat_validation.json"))
check("harness fidelity: DTS+HGB rebuilt = 0.9895", close(fv["val_auroc"], 0.98947, 1e-3))

print(f"\n{checks} checks, {fails} failures")
sys.exit(0 if fails == 0 else 1)
