#!/usr/bin/env python3
"""
Numbers audit: verify every derivable scalar in the paper against saved JSON outputs.
87 checks; expect 82 PASS, 5 documented flags.
Run from repo root: python3 audit_numbers.py
"""
import json, math, sys
from pathlib import Path
import numpy as np
from scipy.special import erf

ROOT = Path(__file__).parent
RES  = ROOT / "results"
TF   = RES / "twinfree"

# ── Load data sources ────────────────────────────────────────────────────────
main  = json.loads((TF  / "ninefive_results.json").read_text())
lso   = json.loads((TF  / "lso_tf.json").read_text())
lfo   = json.loads((TF  / "lfo_tf.json").read_text())
synth = json.loads((RES / "synth.json").read_text())
urfd  = json.loads((RES / "legacy" / "revision_results.json").read_text())["urfd_zero_shot"]

# ── Helpers ───────────────────────────────────────────────────────────────────
results = []

def check(label, got, expected, flag_note=None, tol=5e-5):
    """Register one check. FLAG if flag_note is given (value may still match)."""
    if isinstance(expected, float):
        ok = abs(float(got) - expected) <= tol
    elif isinstance(expected, int):
        ok = int(got) == expected
    else:
        ok = str(got) == str(expected)

    status = "PASS" if (ok and flag_note is None) else ("FLAG" if (ok and flag_note) else "FAIL")
    results.append((status, label, got, expected, flag_note or ""))
    return ok

def f4(x): return round(float(x), 4)
def f3(x): return round(float(x), 3)

# shortcut for model metrics from ninefive_results
models = main["models"]
delta  = main["paired_auc_deltas_vs_DTS_HGB"]
split  = main["split"]
dedup  = main["dedup"]

# ── A: Dataset split sizes (9) ────────────────────────────────────────────────
check("testn",          split["test"]["n"],             1115)
check("testfalls",      split["test"]["falls"],          574)
check("testnonfalls",   split["test"]["nonfalls"],       541)
check("trainn",         split["train"]["n"],            3565)
check("trainfalls",     split["train"]["falls"],        1834)
check("trainnonfalls",  split["train"]["nonfalls"],     1731)
check("valn",           split["val"]["n"],               892)
check("valfalls",       split["val"]["falls"],           459)
check("valnonfalls",    split["val"]["nonfalls"],        433)

# ── B: Dedup statistics (4) ───────────────────────────────────────────────────
check("fvparsed",       dedup["n_input_clips"],          5793)
check("fvunique",       dedup["n_kept_representatives"], 5572)
check("fvdupsremoved",  dedup["n_duplicate_members_removed"], 221)
check("fvambiguous",    dedup["n_ambiguous_groups_dropped"],    0)

# ── C: Theory – QDA bound for hip-speed primitive (9) ────────────────────────
MU_F, SIGMA_F = 0.1706767811976061, 0.1645223306517444
MU_N, SIGMA_N = 0.38148932484420617, 0.2202889746401749

a = 1/SIGMA_F**2 - 1/SIGMA_N**2
b = -2*(MU_F/SIGMA_F**2 - MU_N/SIGMA_N**2)
c = (MU_F**2/SIGMA_F**2 - MU_N**2/SIGMA_N**2) - 2*math.log(SIGMA_N/SIGMA_F)
disc  = b**2 - 4*a*c
x1_exact = (-b - math.sqrt(disc)) / (2*a)
x2_exact = (-b + math.sqrt(disc)) / (2*a)
x1, x2 = min(x1_exact, x2_exact), max(x1_exact, x2_exact)

Phi    = lambda z: 0.5*(1 + erf(z/math.sqrt(2)))
Pf_in  = Phi((x2-MU_F)/SIGMA_F) - Phi((x1-MU_F)/SIGMA_F)
Pn_in  = Phi((x2-MU_N)/SIGMA_N) - Phi((x1-MU_N)/SIGMA_N)
eps    = 0.5*((1-Pf_in) + Pn_in)

check("muf",    f3(MU_F),    0.171)
check("sigf",   f3(SIGMA_F), 0.165)
check("mun",    f3(MU_N),    0.381)
check("sign",   f3(SIGMA_N), 0.220)
check("xone",   f3(x1),     -0.498,
      flag_note="x1 ≈ -0.50 commonly; paper states exact value -0.498")
check("xtwo",   f3(x2),      0.308)
check("pmissf", f3(1-Pf_in), 0.202)
check("pfalsen",f3(Pn_in),   0.369)
check("epsqda", f3(eps),     0.286)

# ── D: Main model AUROCs (11) ─────────────────────────────────────────────────
model_map = {
    "LSTMauroc":       ("LSTM",           "auroc", 0.9558),
    "GRUauroc":        ("GRU",            "auroc", 0.9712),
    "Transformerauroc":("Transformer",    "auroc", 0.9810),
    "SimpleSTGCNauroc":("CompactST-GCN",  "auroc", 0.9185),
    "MiniROCKETauroc": ("MiniROCKET",     "auroc", 0.9760),
    "DTSLRauroc":      ("DTS+LR",         "auroc", 0.9433),
    "DTSETauroc":      ("DTS+ET",         "auroc", 0.9801),
    "DTSRFauroc":      ("DTS+RF",         "auroc", 0.9785),
    "DTSHGBauroc":     ("DTS+HGB",        "auroc", 0.9895),
    "DTSNetauroc":     ("DTS-Net",        "auroc", 0.9775),
    "FullSTGCNauroc":  ("FullST-GCN-COCO","auroc", 0.9652),
}
for label, (model, metric, expected) in model_map.items():
    check(label, f4(models[model][metric]), expected)

# ── E: DTS+HGB secondary metrics (6) ─────────────────────────────────────────
hgb = models["DTS+HGB"]
check("DTSHGBf",    f3(hgb["f1"]),        0.951)
check("DTSHGBprec", f3(hgb["precision"]), 0.934)
check("DTSHGBrec",  f3(hgb["recall"]),    0.969)
check("DTSHGBacc",  f3(hgb["accuracy"]),  0.949)
check("DTSHGBfp",   int(hgb["fp"]),       39)
check("DTSHGBfn",   int(hgb["fn"]),       18)

# ── F: FP/FN counts for three sequence baselines (6) ─────────────────────────
check("LSTMfp",        int(models["LSTM"]["fp"]),        63)
check("LSTMfn",        int(models["LSTM"]["fn"]),        48)
check("GRUfp",         int(models["GRU"]["fp"]),         51)
check("GRUfn",         int(models["GRU"]["fn"]),         42)
check("Transformerfp", int(models["Transformer"]["fp"]), 30)
check("Transformerfn", int(models["Transformer"]["fn"]), 38)

# ── G: Paired bootstrap deltas vs DTS+HGB (10) ───────────────────────────────
pd_map = {
    "hgbminusgruauroc":     ("GRU",            0.0183),
    "hgbminusgcnauroc":     ("CompactST-GCN",  0.0710),
    "hgbminusrocketauroc":  ("MiniROCKET",     0.0135),
    "hgbminustransauroc":   ("Transformer",    0.0085),
    "hgbminusfullgcnauroc": ("FullST-GCN-COCO",0.0242),
    "hgbminusdtsnetauroc":  ("DTS-Net",        0.0120),
}
for label, (model, expected) in pd_map.items():
    check(label, f4(delta[model]["delta"]), expected)

# CI lower bounds for four key comparisons
ci_map = {
    "hgbminusgruaurocCI_lo":     ("GRU",           0.0105),
    "hgbminusgcnaurocCI_lo":     ("CompactST-GCN", 0.0568),
    "hgbminusrocketaurocCI_lo":  ("MiniROCKET",    0.0058),
    "hgbminustransaurocCI_lo":   ("Transformer",   0.0024),
}
for label, (model, expected) in ci_map.items():
    check(label, f4(delta[model]["ci95"][0]), expected)

# ── H: Leave-session-out (9) ──────────────────────────────────────────────────
lso_sessions = {"One":"1","Two":"2","Three":"3","Four":"4"}
auroc_vals = [lso[s]["auroc"] for s in "1234"]
check("lsoOneauroc",   f4(lso["1"]["auroc"]), 0.9382)
check("lsoTwoauroc",   f4(lso["2"]["auroc"]), 0.9926)
check("lsoThreeauroc", f4(lso["3"]["auroc"]), 0.9840)
check("lsoFourauroc",  f4(lso["4"]["auroc"]), 0.9448)
check("lsomeanauroc",  f4(np.mean(auroc_vals)), 0.9649,
      flag_note="derived average of 4 folds; not a raw stored value")
check("lsorangelow",   f4(min(auroc_vals)),  0.9382)
check("lsorangehigh",  f4(max(auroc_vals)),  0.9926)
check("lsoOnen",       int(lso["1"]["n"]),  1513)
check("lsoOnefalls",   int(lso["1"]["falls"]), 679)

# ── I: Leave-fall-origin (10) ─────────────────────────────────────────────────
lfo_origins = ["Bed","Chair","Stand"]
lfo_aurocs  = [lfo[h]["auroc"] for h in lfo_origins]
lfo_f1s     = [lfo[h]["f1"]    for h in lfo_origins]
check("lfoBedauroc",   f4(lfo["Bed"]["auroc"]),   0.7123,
      flag_note="Bed baseline AUROC from lfo_tf.json; K=0 bed adaptation (bed_adapt.json) gives 0.7138")
check("lfoChairauroc", f4(lfo["Chair"]["auroc"]), 0.9844)
check("lfoStandauroc", f4(lfo["Stand"]["auroc"]), 0.9916)
check("lfomeanauroc",  f4(np.mean(lfo_aurocs)),   0.8961)
check("lfoBedfall",    int(lfo["Bed"]["falls"]),   922)
check("lfoChairfall",  int(lfo["Chair"]["falls"]), 961)
check("lfoStandfall",  int(lfo["Stand"]["falls"]), 984)
check("iidreff",       f3(lfo["iid_ref"]["f1"]),  0.950,
      flag_note="Gamma_F1 = 0.951-0.880; iidreff=0.950 (iid_ref.f1) ≠ DTSHGBf=0.951 (main test)")
check("transfergap",   f3(lfo["iid_ref"]["f1"] - np.mean(lfo_f1s)), 0.071)
check("lfomeanf",      f3(np.mean(lfo_f1s)),     0.880)

# ── J: URFD zero-shot transfer (8) ───────────────────────────────────────────
check("urfdauroc",     f4(urfd["auroc"]),     0.9200)
check("urfdrec",       f3(urfd["recall"]),    1.000)
check("urfdprec",      f3(urfd["precision"]), 0.508)
check("urfdf",         f3(urfd["f1"]),        0.674)
check("urfdfp",        int(urfd["fp"]),       29)
check("urfdfn",        int(urfd["fn"]),        0)
check("urfdprecCI_lo", f3(urfd["precision_ci"][0]), 0.379,
      flag_note="paper uses Clopper-Pearson CI [0.379, 0.638]; Wilson CI of 29/59 gives [0.384, 0.632]")
check("urfdaurocCI_lo",f4(urfd["auroc_ci"][0]),     0.8483)

# ── K: Controlled temporal-order synthetic benchmark (5) ─────────────────────
check("synthcv",          f3(np.mean(synth["cv5"])),           1.000)
check("synthdtsathundred",f3(synth["curve"]["100"]["DTS"]),     1.000)
check("synthbof",         f3(synth["curve"]["100"]["BoF"]),     0.500)
check("synthalphamin",    f3(min(v["alpha"] for v in synth["curve"].values())), 0.996)
check("synthdropfour",    f3(synth["multifam"]["drop4"]),       1.000)

# ── Summary ───────────────────────────────────────────────────────────────────
total  = len(results)
n_pass = sum(1 for r in results if r[0] == "PASS")
n_flag = sum(1 for r in results if r[0] == "FLAG")
n_fail = sum(1 for r in results if r[0] == "FAIL")

print(f"\n{'='*70}")
print(f"  AUDIT RESULTS  —  {total} checks")
print(f"{'='*70}")
for status, label, got, expected, note in results:
    marker = {"PASS":"✓","FLAG":"⚑","FAIL":"✗"}[status]
    line   = f"  {marker} {status:4s}  {label:<35s}  got={got}  exp={expected}"
    print(line)
    if note:
        print(f"         NOTE: {note}")

print(f"\n{'='*70}")
print(f"  SUMMARY:  {n_pass} PASS  |  {n_flag} FLAG  |  {n_fail} FAIL  (total {total})")
print(f"  EXPECTED: 82 PASS  |   5 FLAG  |   0 FAIL  (total 87)")
print(f"{'='*70}")

if n_fail > 0:
    print("\n  ATTENTION: unexpected FAILs above — investigate before submission.")
    sys.exit(1)
elif total != 87 or n_pass != 82 or n_flag != 5:
    print(f"\n  NOTE: counts differ from expected (82 PASS / 5 FLAG). Check above.")
    sys.exit(0)
else:
    print("\n  All checks passed (82 PASS + 5 documented flags).")
    sys.exit(0)
