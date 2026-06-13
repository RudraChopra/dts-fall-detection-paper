#!/usr/bin/env python3
"""
Aggregate synth_bench.py output into LaTeX tables.

Reads:  results/results_all.jsonl
Prints: LaTeX tables for headline constructions, training-size curve,
        operator-bank ablation, tau-selection ablation, and all sweeps.

Usage:
  python3 analyze_bench.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

INFILE = Path(__file__).parent / "results" / "results_all.jsonl"


def load(path=INFILE):
    if not path.exists():
        print(f"ERROR: {path} not found. Run synth_bench.py first.", file=sys.stderr)
        sys.exit(1)
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def group_by(records, keys):
    """Return dict keyed by tuple of field values → list of auroc floats."""
    groups = defaultdict(list)
    for r in records:
        key = tuple(r[k] for k in keys)
        groups[key].append(r["auroc"])
    return groups


def fmt(mean, std=None, n=None):
    """Format as 'mean ± std' or 'mean' with 3 dp."""
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} \\pm {std:.3f}"


def mean_std(vals):
    a = np.array(vals)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def latex_table(caption, label, header_cols, rows, note=None):
    cols_fmt = "l" + "c" * (len(header_cols) - 1)
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{label}}}",
        f"\\begin{{tabular}}{{{cols_fmt}}}",
        "\\toprule",
        " & ".join(header_cols) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(c) for c in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines.append(f"\\\\[2pt]\\small {note}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def section(title):
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


# ── Main analysis ─────────────────────────────────────────────────────────────

def main():
    records = load()
    print(f"Loaded {len(records)} records from {INFILE}")

    # ── 1. Headline constructions ─────────────────────────────────────────────
    section("1. Headline: AUROC by construction (seeds 0-4)")
    groups = group_by(
        [r for r in records if r["sweep"] == "headline"],
        ["construction"]
    )
    rows = []
    for construction in ["fall", "sit2stand", "burst"]:
        vals = groups.get((construction,), [])
        if not vals:
            print(f"  WARNING: no data for {construction}")
            continue
        m, s = mean_std(vals)
        rows.append([construction, f"${fmt(m, s)}$", len(vals)])
        print(f"  {construction:12s}  {m:.3f} ± {s:.3f}  (n={len(vals)})")
    table = latex_table(
        "Headline controlled-benchmark: mean CV-AUROC across seeds 0--4",
        "bench_headline",
        ["Construction", "AUROC (mean $\\pm$ std)", "Seeds"],
        rows,
    )
    print("\n--- LaTeX ---")
    print(table)

    # ── 2. Training-size curve ────────────────────────────────────────────────
    section("2. Training-size sweep (fall, seeds 0-2)")
    sweep_recs = [r for r in records if r["sweep"] == "training_size"]
    groups = group_by(sweep_recs, ["params"])
    n_vals = sorted({r["params"].get("n_clips") for r in sweep_recs if "n_clips" in r["params"]})
    rows = []
    for n in n_vals:
        vals = []
        for r in sweep_recs:
            if r["params"].get("n_clips") == n:
                vals.append(r["auroc"])
        if not vals:
            continue
        m, s = mean_std(vals)
        rows.append([f"$n={n}$", f"${fmt(m, s)}$"])
        print(f"  n_clips={n:5d}  {m:.3f} ± {s:.3f}")
    table = latex_table(
        "Effect of training size on synthetic-benchmark AUROC",
        "bench_trainsize",
        ["Training clips (per class)", "AUROC (mean $\\pm$ std)"],
        rows,
    )
    print("\n--- LaTeX ---")
    print(table)

    # ── 3. SNR sweep ──────────────────────────────────────────────────────────
    section("3. SNR sweep (fall, seeds 0-2)")
    sweep_recs = [r for r in records if r["sweep"] == "snr"]
    snr_vals = sorted({r["params"].get("snr") for r in sweep_recs if "snr" in r["params"]})
    rows = []
    for snr in snr_vals:
        vals = [r["auroc"] for r in sweep_recs if r["params"].get("snr") == snr]
        if not vals:
            continue
        m, s = mean_std(vals)
        rows.append([f"${snr}$", f"${fmt(m, s)}$"])
        print(f"  SNR={snr:5.1f}  {m:.3f} ± {s:.3f}")
    table = latex_table(
        "Effect of signal-to-noise ratio on synthetic-benchmark AUROC",
        "bench_snr",
        ["SNR (drop amplitude)", "AUROC (mean $\\pm$ std)"],
        rows,
    )
    print("\n--- LaTeX ---")
    print(table)

    # ── 4. Operator-bank ablation ─────────────────────────────────────────────
    section("4. Operator-bank ablation (fall, seeds 0-2)")
    sweep_recs = [r for r in records if r["sweep"] == "operator_bank"]
    mode_order = ["full", "no_asym", "no_diff", "stats_only"]
    mode_label = {
        "full":       "Full DTS (128-dim)",
        "no_asym":    "No asymmetry $\\alpha$",
        "no_diff":    "No differential ops",
        "stats_only": "Statistics only",
    }
    rows = []
    for mode in mode_order:
        vals = [r["auroc"] for r in sweep_recs if r["params"].get("mode") == mode]
        if not vals:
            continue
        m, s = mean_std(vals)
        rows.append([mode_label.get(mode, mode), f"${fmt(m, s)}$"])
        print(f"  {mode:12s}  {m:.3f} ± {s:.3f}")
    table = latex_table(
        "Operator-bank ablation on synthetic benchmark",
        "bench_opbank",
        ["Feature set", "AUROC (mean $\\pm$ std)"],
        rows,
        note="All variants: fall construction, seeds 0--2."
    )
    print("\n--- LaTeX ---")
    print(table)

    # ── 5. τ-selection ablation ───────────────────────────────────────────────
    section("5. τ-selection ablation (fall, seeds 0-2)")
    sweep_recs = [r for r in records if r["sweep"] == "tau_select"]
    tau_modes = ["fixed_030", "fixed_050", "data_driven"]
    tau_label = {
        "fixed_030":   "$\\tau = 0.30$ (fixed)",
        "fixed_050":   "$\\tau = 0.50$ (fixed)",
        "data_driven": "$\\tau^*$ (data-driven)",
    }
    rows = []
    for tm in tau_modes:
        vals = []
        for r in sweep_recs:
            p = r["params"]
            if p.get("tau_mode") == "data_driven" and tm == "data_driven":
                vals.append(r["auroc"])
            elif p.get("tau_mode") == "fixed" and p.get("tau") == {"fixed_030": 0.30, "fixed_050": 0.50}.get(tm):
                vals.append(r["auroc"])
        if not vals:
            continue
        m, s = mean_std(vals)
        rows.append([tau_label.get(tm, tm), f"${fmt(m, s)}$"])
        print(f"  {tm:14s}  {m:.3f} ± {s:.3f}")
    table = latex_table(
        "$\\tau$ selection strategy ablation on synthetic benchmark",
        "bench_tau",
        ["$\\tau$ strategy", "AUROC (mean $\\pm$ std)"],
        rows,
    )
    print("\n--- LaTeX ---")
    print(table)

    # ── 6. Remaining sweeps summary ───────────────────────────────────────────
    for sweep, param_key, sweep_title in [
        ("onset_p1",   "onset_p1",   "Event onset fraction"),
        ("event_dur",  "event_dur",  "Event duration fraction"),
        ("seq_len",    "T",          "Sequence length T"),
        ("distractors","n_distractor","Number of distractor channels"),
    ]:
        section(f"6. {sweep_title} sweep (fall, seeds 0-2)")
        sweep_recs = [r for r in records if r["sweep"] == sweep]
        vals_seen = sorted({r["params"].get(param_key) for r in sweep_recs
                            if param_key in r["params"]})
        rows = []
        for v in vals_seen:
            vals = [r["auroc"] for r in sweep_recs if r["params"].get(param_key) == v]
            if not vals:
                continue
            m, s = mean_std(vals)
            rows.append([f"${v}$", f"${fmt(m, s)}$"])
            print(f"  {param_key}={v}  {m:.3f} ± {s:.3f}")

    # ── 7. Qualitative assertions ─────────────────────────────────────────────
    section("7. Qualitative assertions from REPRO_COMMANDS.md")

    errors = []

    def assert_claim(desc, condition, got_str):
        mark = "PASS" if condition else "FAIL"
        print(f"  [{mark}] {desc}  ({got_str})")
        if not condition:
            errors.append(desc)

    # BoF stays near 0.500 (not tested here — BoF is a separate baseline in synth.json)
    # For the synthetic benchmark the bag-of-frames ≡ random-permutation negatives baseline
    # is tested inside audit_numbers.py via synth.json; check manually here if data present.

    headline_groups = group_by(
        [r for r in records if r["sweep"] == "headline"],
        ["construction"]
    )
    for construction in ["fall", "sit2stand", "burst"]:
        vals = headline_groups.get((construction,), [])
        if vals:
            m = np.mean(vals)
            assert_claim(
                f"{construction} DTS near 1.0 at default SNR",
                m >= 0.95,
                f"mean AUROC = {m:.3f}"
            )

    op_vals = {
        mode: np.mean([r["auroc"] for r in records
                       if r["sweep"] == "operator_bank" and r["params"].get("mode") == mode])
        for mode in ["full", "no_asym", "stats_only"]
        if any(r["sweep"] == "operator_bank" and r["params"].get("mode") == mode for r in records)
    }
    if "full" in op_vals and "no_asym" in op_vals:
        assert_claim(
            "Removing asymmetry degrades AUROC",
            op_vals["full"] > op_vals["no_asym"],
            f"full={op_vals['full']:.3f} > no_asym={op_vals['no_asym']:.3f}"
        )
    if "stats_only" in op_vals:
        assert_claim(
            "Stats-only (order-invariant) near chance",
            op_vals["stats_only"] < 0.65,
            f"stats_only={op_vals['stats_only']:.3f}"
        )

    # τ: data-driven should match or beat fixed τ
    tau_groups = {
        tm: np.mean([r["auroc"] for r in records
                     if r["sweep"] == "tau_select" and (
                         (r["params"].get("tau_mode") == "data_driven" and tm == "data_driven") or
                         (r["params"].get("tau_mode") == "fixed" and
                          r["params"].get("tau") == {"fixed_030": 0.30, "fixed_050": 0.50}.get(tm))
                     )])
        for tm in ["fixed_030", "fixed_050", "data_driven"]
        if any(r["sweep"] == "tau_select" and (
            (r["params"].get("tau_mode") == "data_driven" and tm == "data_driven") or
            (r["params"].get("tau_mode") == "fixed" and
             r["params"].get("tau") == {"fixed_030": 0.30, "fixed_050": 0.50}.get(tm))
        ) for r in records)
    }
    if "data_driven" in tau_groups and "fixed_050" in tau_groups:
        assert_claim(
            "τ=0.50 (misspecified) lower than data-driven τ*",
            tau_groups["data_driven"] >= tau_groups["fixed_050"],
            f"data_driven={tau_groups.get('data_driven', float('nan')):.3f}  "
            f"fixed_050={tau_groups.get('fixed_050', float('nan')):.3f}"
        )

    print()
    if errors:
        print(f"  {len(errors)} assertion(s) failed:")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  All qualitative assertions passed.")


if __name__ == "__main__":
    main()
