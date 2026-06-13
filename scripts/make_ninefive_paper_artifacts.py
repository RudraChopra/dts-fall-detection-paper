#!/usr/bin/env python3
"""Merge strict twin-free results and refresh paper artifacts.

Inputs:
  outputs/ninefive_core_full/twinfree_results.json
  outputs/ninefive_core_full/score_vectors_twinfree.npz
  outputs/ninefive_neural_full/twinfree_results.json
  outputs/ninefive_neural_full/score_vectors_twinfree.npz

Outputs:
  outputs/ninefive_results.json
  outputs/ninefive_score_vectors.npz
  work/ninefive_paper/numbers.tex
  work/ninefive_paper/fig1_main.pdf
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "outputs" / "ninefive_core_full"
NEURAL = ROOT / "outputs" / "ninefive_neural_full"
PAPER = ROOT / "work" / "ninefive_paper"
OUT = ROOT / "outputs"
OLD_NUMBERS = ROOT / "work" / "final_fix" / "numbers.tex"
RUNNER = ROOT / "work" / "ninefive" / "run_twinfree_benchmark.py"

MODEL_MACROS = {
    "LSTM": "LSTM",
    "GRU": "GRU",
    "Transformer": "Transformer",
    "CompactST-GCN": "SimpleSTGCN",
    "FullST-GCN-COCO": "FullSTGCN",
    "MiniROCKET": "MiniROCKET",
    "DTS+LR": "DTSLR",
    "DTS+ET": "DTSET",
    "DTS+RF": "DTSRF",
    "DTS+HGB": "DTSHGB",
    "DTS-Net": "DTSNet",
}

FIG_ORDER = [
    "LSTM",
    "GRU",
    "Transformer",
    "CompactST-GCN",
    "FullST-GCN-COCO",
    "MiniROCKET",
    "DTS+LR",
    "DTS+ET",
    "DTS+RF",
    "DTS+HGB",
    "DTS-Net",
]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def f4(x: float) -> str:
    return f"{x:.4f}"


def f3(x: float) -> str:
    return f"{x:.3f}"


def nfmt(n: int | float) -> str:
    return f"{int(n):,}".replace(",", "{,}")


def ci_fmt(ci: list[float], digits: int) -> str:
    if digits == 4:
        return f"[{ci[0]:.4f}, {ci[1]:.4f}]"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def load_vectors(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def paired_auc_delta_ci(
    y: np.ndarray,
    primary: np.ndarray,
    comparator: np.ndarray,
    bootstraps: int = 2000,
    seed: int = 20260610,
) -> dict:
    delta = float(roc_auc_score(y, primary) - roc_auc_score(y, comparator))
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(bootstraps):
        ii = rng.integers(0, n, n)
        yy = y[ii]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(float(roc_auc_score(yy, primary[ii]) - roc_auc_score(yy, comparator[ii])))
    return {
        "delta": delta,
        "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
        "bootstraps": bootstraps,
    }


def metric_macro_values(models: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for model_name, macro in MODEL_MACROS.items():
        m = models[model_name]
        ci = m["ci95"]
        out[f"{macro}auroc"] = f4(m["auroc"])
        out[f"{macro}aurocCI"] = ci_fmt(ci["auroc"], 4)
        metric_map = {"f1": "f", "precision": "prec", "recall": "rec", "accuracy": "acc"}
        for src, dest in metric_map.items():
            out[f"{macro}{dest}"] = f3(m[src])
            out[f"{macro}{dest}CI"] = ci_fmt(ci[src], 3)
        out[f"{macro}fp"] = str(int(m["fp"]))
        out[f"{macro}fn"] = str(int(m["fn"]))
    return out


def count_params() -> dict[str, int]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("ninefive_runner", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["ninefive_runner"] = mod
    spec.loader.exec_module(mod)
    compact = mod.STGCN([64, 64, 128], dropout=0.3)
    full = mod.STGCN([64, 64, 64, 128, 128, 256], dropout=0.3)
    lstm = mod.FallRNN("lstm")
    dtsnet = mod.DTSNet()
    return {
        "lstm": sum(p.numel() for p in lstm.parameters()),
        "compact_stgcn": sum(p.numel() for p in compact.parameters()),
        "full_stgcn": sum(p.numel() for p in full.parameters()),
        "dtsnet": sum(p.numel() for p in dtsnet.parameters()),
    }


def selected_macro_values(selected: dict) -> dict[str, str]:
    out = {}
    out["selLR"] = f"C={selected['DTS+LR']['selected_params']['C']}"
    et = selected["DTS+ET"]["selected_params"]
    rf = selected["DTS+RF"]["selected_params"]
    hgb = selected["DTS+HGB"]["selected_params"]
    out["selET"] = f"{et['n_estimators']} trees, max\\_features={et['max_features']}"
    out["selRF"] = f"{rf['n_estimators']} trees, max\\_features={rf['max_features']}"
    out["selHGB"] = (
        f"{hgb['max_iter']} iterations, lr={hgb['learning_rate']}, "
        f"$\\lambda_2$={hgb['l2_regularization']}"
    )
    return out


def rewrite_numbers(overrides: dict[str, str]) -> None:
    pattern = re.compile(r"^\\newcommand\{\\([A-Za-z0-9]+)\}\{(.*)\}$")
    seen: set[str] = set()
    lines = []
    for line in OLD_NUMBERS.read_text().splitlines():
        m = pattern.match(line)
        if not m:
            lines.append(line)
            continue
        name = m.group(1)
        seen.add(name)
        value = overrides.get(name, m.group(2))
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    for name in sorted(set(overrides) - seen):
        lines.append(f"\\newcommand{{\\{name}}}{{{overrides[name]}}}")
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n")


def make_fig1(models: dict) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 6,
            "ytick.labelsize": 7,
            "figure.dpi": 150,
            "pdf.fonttype": 42,
        }
    )
    group = {
        "LSTM": "seq",
        "GRU": "seq",
        "Transformer": "seq",
        "CompactST-GCN": "graph",
        "FullST-GCN-COCO": "graph",
        "MiniROCKET": "conv",
        "DTS+LR": "dts",
        "DTS+ET": "dts",
        "DTS+RF": "dts",
        "DTS+HGB": "dts",
        "DTS-Net": "dtsnet",
    }
    colors = {
        "seq": "#7c9fc9",
        "graph": "#c98f7c",
        "conv": "#b3a16e",
        "dts": "#5d9e75",
        "dtsnet": "#8d7cc9",
    }
    labels = {
        "Transformer": "Transformer",
        "CompactST-GCN": "Compact ST-GCN",
        "FullST-GCN-COCO": "Full ST-GCN",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.05), sharey=True)
    xmin = 0.70
    for ax, metric, title, digits in [
        (axes[0], "auroc", "AUROC", 4),
        (axes[1], "recall", "Recall (fall detection rate)", 3),
    ]:
        vals = [models[k][metric] for k in FIG_ORDER]
        lo = [models[k][metric] - models[k]["ci95"][metric][0] for k in FIG_ORDER]
        hi = [models[k]["ci95"][metric][1] - models[k][metric] for k in FIG_ORDER]
        y = np.arange(len(FIG_ORDER))
        ax.barh(
            y,
            vals,
            color=[colors[group[k]] for k in FIG_ORDER],
            xerr=[lo, hi],
            error_kw={"lw": 0.8, "capsize": 2},
            height=0.68,
        )
        for yi, v in zip(y, vals):
            txt = f"{v:.4f}" if digits == 4 else f"{v:.3f}"
            ax.text(xmin + 0.005, yi, txt, ha="left", va="center", fontsize=5.8)
        ax.set_yticks(y)
        ax.set_yticklabels([labels.get(k, k) for k in FIG_ORDER])
        ax.invert_yaxis()
        ax.set_xlim(xmin, 1.005)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlabel("Score")
    fig.tight_layout()
    fig.savefig(PAPER / "fig1_main.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig4(params: dict[str, int], train_n: int) -> None:
    synth_path = ROOT / "dts-fall-detection" / "results" / "synth.json"
    if not synth_path.exists():
        return
    sy = load_json(synth_path)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    ns = sorted(int(k) for k in sy["curve"])
    dts = [sy["curve"][str(n)]["DTS"] for n in ns]
    bof = [sy["curve"][str(n)]["BoF"] for n in ns]
    alp = [sy["curve"][str(n)]["alpha"] for n in ns]
    axes[0].plot(ns, dts, "o-", color="#5d9e75", label="DTS+HGB (128-dim)")
    axes[0].plot(ns, alp, "s--", color="#e8a33d", label=r"$\alpha$-only + LR (8-dim)")
    axes[0].plot(ns, bof, "^-", color="#7c9fc9", label="BoF + LR (static, chance)")
    axes[0].set_xscale("log")
    axes[0].set_ylim(0.45, 1.03)
    axes[0].set_xlabel("Training samples $n$")
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("Controlled temporal-order benchmark")
    axes[0].legend(loc="center right")
    axes[0].grid(alpha=0.25)

    ns2 = np.logspace(2, 5.2, 200)
    lstm_params = params["lstm"]
    b_dts = np.sqrt(128 / ns2)
    b_lstm = np.sqrt(lstm_params / ns2)
    nstar = lstm_params / math.log(lstm_params)
    axes[1].plot(ns2, b_dts, color="#5d9e75", label=r"DTS bound $\sqrt{128/n}$")
    axes[1].plot(ns2, b_lstm, color="#7c9fc9", label=rf"LSTM bound $\sqrt{{{lstm_params:,}/n}}$".replace(",", "{,}"))
    axes[1].axvline(train_n, color="k", ls="--", lw=0.8)
    axes[1].text(train_n * 1.1, 4.0, rf"$n_{{FV}}$={train_n:,}".replace(",", "{,}"), rotation=90, fontsize=6.5, va="bottom")
    axes[1].axvline(nstar, color="#c0504d", ls=":", lw=1.0)
    axes[1].text(nstar * 1.1, 4.0, f"$n^*$={nstar:,.0f}", rotation=90, fontsize=6.5, va="bottom", color="#c0504d")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Training samples $n$")
    axes[1].set_ylabel(r"$\sqrt{d/n}$ (parallel bounds)")
    axes[1].set_title("Capacity reference scale (heuristic; bounds do not cross)")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(PAPER / "fig4_synth.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    core = load_json(CORE / "twinfree_results.json")
    neural = load_json(NEURAL / "twinfree_results.json")
    core_vec = load_vectors(CORE / "score_vectors_twinfree.npz")
    neural_vec = load_vectors(NEURAL / "score_vectors_twinfree.npz")
    if not np.array_equal(core_vec["y_test"], neural_vec["y_test"]):
        raise ValueError("core/neural y_test vectors differ")
    if not np.array_equal(core_vec["y_val"], neural_vec["y_val"]):
        raise ValueError("core/neural y_val vectors differ")

    models = {**core["models"], **neural["models"]}
    missing = [m for m in MODEL_MACROS if m not in models]
    if missing:
        raise ValueError(f"missing models: {missing}")

    score_vectors: dict[str, np.ndarray] = {}
    for src in (core_vec, neural_vec):
        score_vectors.update(src)
    np.savez_compressed(OUT / "ninefive_score_vectors.npz", **score_vectors)

    y_test = score_vectors["y_test"]
    hgb = score_vectors["DTS+HGB_test"]
    paired = {}
    for model in ["GRU", "Transformer", "MiniROCKET", "CompactST-GCN", "FullST-GCN-COCO", "DTS-Net"]:
        paired[model] = paired_auc_delta_ci(y_test, hgb, score_vectors[f"{model}_test"])

    params = count_params()
    selected = {**core["selected_configs"], **neural["selected_configs"]}
    summary = {
        "protocol": core["protocol"],
        "seed": core["seed"],
        "dedup": core["dedup"],
        "split": core["split"],
        "tau_star": core["tau_star"],
        "models": models,
        "selected_configs": selected,
        "paired_auc_deltas_vs_DTS_HGB": paired,
        "matched_operating_points": core.get("matched_operating_points", {}),
        "parameter_counts": params,
        "artifacts": {
            "source_score_vectors": "ninefive_score_vectors.npz",
            "strict_core": "ninefive_core_full/twinfree_results.json",
            "strict_neural": "ninefive_neural_full/twinfree_results.json",
            "split_manifest": "ninefive_core_full/split_manifest_twinfree.csv",
            "dedup_audit": "ninefive_core_full/dedup_audit.csv",
        },
    }
    (OUT / "ninefive_results.json").write_text(json.dumps(summary, indent=2))

    overrides = metric_macro_values(models)
    overrides.update(selected_macro_values(selected))
    split = core["split"]
    dedup = core["dedup"]
    overrides.update(
        {
            "fvparsed": nfmt(dedup["n_input_clips"]),
            "fvunique": nfmt(dedup["n_kept_representatives"]),
            "fvdupsremoved": nfmt(dedup["n_duplicate_members_removed"]),
            "fvambiguous": nfmt(dedup["n_ambiguous_groups_dropped"]),
            "trainn": nfmt(split["train"]["n"]),
            "trainfalls": nfmt(split["train"]["falls"]),
            "trainnonfalls": nfmt(split["train"]["nonfalls"]),
            "valn": nfmt(split["val"]["n"]),
            "valfalls": nfmt(split["val"]["falls"]),
            "valnonfalls": nfmt(split["val"]["nonfalls"]),
            "testn": nfmt(split["test"]["n"]),
            "testfalls": nfmt(split["test"]["falls"]),
            "testnonfalls": nfmt(split["test"]["nonfalls"]),
            "lstmparams": nfmt(params["lstm"]),
            "gcnparams": nfmt(params["compact_stgcn"]),
            "fullgcnparams": nfmt(params["full_stgcn"]),
            "dtsnetparams": nfmt(params["dtsnet"]),
            "nstar": nfmt(params["lstm"] / math.log(params["lstm"])),
            "nstarover": f"{(params['lstm'] / math.log(params['lstm'])) / split['train']['n']:.1f}",
            "hgbminusrocketauroc": f4(paired["MiniROCKET"]["delta"]),
            "hgbminusrocketaurocCI": ci_fmt(paired["MiniROCKET"]["ci95"], 4),
            "hgbminusgcnauroc": f4(paired["CompactST-GCN"]["delta"]),
            "hgbminusgcnaurocCI": ci_fmt(paired["CompactST-GCN"]["ci95"], 4),
            "hgbminusfullgcnauroc": f4(paired["FullST-GCN-COCO"]["delta"]),
            "hgbminusfullgcnaurocCI": ci_fmt(paired["FullST-GCN-COCO"]["ci95"], 4),
            "hgbminusgruauroc": f4(paired["GRU"]["delta"]),
            "hgbminusgruaurocCI": ci_fmt(paired["GRU"]["ci95"], 4),
            "hgbminustransauroc": f4(paired["Transformer"]["delta"]),
            "hgbminustransaurocCI": ci_fmt(paired["Transformer"]["ci95"], 4),
            "hgbminusdtsnetauroc": f4(paired["DTS-Net"]["delta"]),
            "hgbminusdtsnetaurocCI": ci_fmt(paired["DTS-Net"]["ci95"], 4),
            "hgberr": f3(1 - models["DTS+HGB"]["accuracy"]),
            "gapfactor": f"{0.2856 / (1 - models['DTS+HGB']['accuracy']):.1f}",
        }
    )
    rewrite_numbers(overrides)
    make_fig1(models)
    make_fig4(params, split["train"]["n"])
    print(f"wrote {OUT / 'ninefive_results.json'}")
    print(f"wrote {PAPER / 'numbers.tex'}")
    print(f"wrote {PAPER / 'fig1_main.pdf'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
