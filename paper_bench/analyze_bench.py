"""Aggregate results_all.jsonl into LaTeX tables used by the papers.

Generates (into ../papers/):
  bench_tasks_table.tex   - 3-construction headline table (main text)
  bench_sweeps_table.tex  - noise / timing / distractor / duration / length /
                            operator sweeps (appendix)
Every number is computed from the saved runs; nothing is typed by hand.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPERS = os.path.join(HERE, "..", "papers")
if not os.path.isdir(PAPERS):
    PAPERS = HERE  # in-repo: write tables next to this script

runs = [json.loads(l) for l in open(os.path.join(HERE, "results_all.jsonl"))]

MODEL_NAMES = {
    "dts_hgb": "DTS+HGB (128-d)",
    "alpha_lr": "$\\alpha$-only+LR (8-d)",
    "minirocket": "MiniROCKET+Ridge",
    "raw_mlp": "Raw-seq MLP",
    "bof_hgb": "Bag-of-frames+HGB",
}
GEN_NAMES = {"fall": "Fall", "sit2stand": "Sit-to-stand", "burst": "Reversal"}


def agg(filt):
    vals = [r["auroc"] for r in runs if all(r.get(k) == v for k, v in filt.items())]
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def fmt(m, s, nseeds):
    if m is None:
        return "--"
    if s < 5e-4:
        return f"{m:.3f}"
    return f"{m:.3f}{{\\scriptsize$\\pm${s:.3f}}}"


def tasks_table():
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\setlength{\\tabcolsep}{2.6pt}",
        "\\begin{tabular}{lcccccc}", "\\toprule",
        " & \\multicolumn{2}{c}{Fall} & \\multicolumn{2}{c}{Sit-to-stand} & \\multicolumn{2}{c}{Reversal} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "Model & $n{=}100$ & $n{=}1000$ & $n{=}100$ & $n{=}1000$ & $n{=}100$ & $n{=}1000$ \\\\",
        "\\midrule",
    ]
    order = ["dts_hgb", "alpha_lr", "minirocket", "raw_mlp", "bof_hgb"]
    for m in order:
        cells = []
        for g in ["fall", "sit2stand", "burst"]:
            for n in [100, 1000]:
                mu, sd, k = agg({"sweep": "tasks", "gen": g, "model": m,
                                 "n_train_pairs": n})
                cells.append(fmt(mu, sd, k))
        lines.append(f"{MODEL_NAMES[m]} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}",
        "\\caption{Expanded controlled temporal-order benchmark: held-out AUROC "
        "(mean$\\pm$sd over five seeds, 500 test pairs) on three constructions in "
        "which negatives are frame permutations (Fall, Sit-to-stand) or exact time "
        "reversals (Reversal) of positives, so class-conditional frame multisets are "
        "identical and Theorem~\\ref{thm:impossibility} applies. Order-invariant "
        "bag-of-frames is exactly at chance everywhere; the structured signature and "
        "order-sensitive baselines solve all three constructions.}",
        "\\label{tab:bench}", "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def sweep_block(sweep, key, vals, models, header, keyfmt=str):
    lines = ["\\midrule",
             f"\\multicolumn{{{len(models)+1}}}{{l}}{{\\emph{{{header}}}}} \\\\"]
    for v in vals:
        cells = []
        for m in models:
            filt = {"sweep": sweep, "model": m}
            sel = [r["auroc"] for r in runs
                   if r["sweep"] == sweep and r["model"] == m
                   and r["kwargs"].get(key) == v
                   and r["tau_mode"] == "selected"]
            if sel:
                cells.append(fmt(float(np.mean(sel)), float(np.std(sel)), len(sel)))
            else:
                cells.append("--")
        lines.append(f"{keyfmt(v)} & " + " & ".join(cells) + " \\\\")
    return lines


def sweeps_table():
    models = ["dts_hgb", "alpha_lr", "bof_hgb", "minirocket"]
    head = ["\\begin{table}[h]", "\\centering", "\\small",
            "\\begin{tabular}{lcccc}", "\\toprule",
            "Setting & DTS+HGB & $\\alpha$-only+LR & Bag-of-frames & MiniROCKET \\\\"]
    body = []
    body += sweep_block("noise", "snr", [1.0, 2.0, 4.0, 8.0, 16.0], models,
                        "Drop SNR $\\mu_d/\\sigma_s$ (fall, $n{=}250$)",
                        lambda v: f"SNR {v:g}")
    body += sweep_block("timing", "p1", [0.15, 0.3, 0.5, 0.7], models,
                        "Event onset $p_1$ (fall, $n{=}250$)",
                        lambda v: f"$p_1{{=}}{v:g}$")
    # misspecified tau rows
    body.append("\\midrule")
    body.append(f"\\multicolumn{{{len(models)+1}}}{{l}}{{\\emph{{$\\alpha$-only with "
                f"misspecified fixed $\\tau{{=}}0.30$ vs train-selected $\\tau$}}}} \\\\")
    for p1 in [0.15, 0.3, 0.5, 0.7]:
        sel = [r["auroc"] for r in runs if r["sweep"] == "timing"
               and r["model"] == "alpha_lr" and r["kwargs"].get("p1") == p1
               and r["tau_mode"] == "selected"]
        fix = [r["auroc"] for r in runs if r["sweep"] == "timing"
               and r["model"] == "alpha_lr" and r["kwargs"].get("p1") == p1
               and r["tau_mode"] == "fixed03"]
        c1 = fmt(float(np.mean(sel)), float(np.std(sel)), len(sel)) if sel else "--"
        c2 = fmt(float(np.mean(fix)), float(np.std(fix)), len(fix)) if fix else "--"
        body.append(f"$p_1{{=}}{p1:g}$ & \\multicolumn{{2}}{{c}}{{selected: {c1}}} & "
                    f"\\multicolumn{{2}}{{c}}{{fixed 0.30: {c2}}} \\\\")
    body += sweep_block("distractor", "d_noise", [0, 8, 32, 120], models,
                        "Distractor channels added (fall, $n{=}250$)",
                        lambda v: f"+{v} noise ch.")
    body += sweep_block("duration", "dur", [0.05, 0.1, 0.2, 0.4],
                        ["dts_hgb", "alpha_lr", "bof_hgb"],
                        "Event duration $p_2{-}p_1$ (fall, $n{=}250$)",
                        lambda v: f"dur {v:g}")
    body += sweep_block("length", "T", [30, 60, 150, 300],
                        ["dts_hgb", "alpha_lr", "bof_hgb"],
                        "Sequence length $T$ (fall, $n{=}250$)",
                        lambda v: f"$T{{=}}{v}$")
    # operator ablation
    body.append("\\midrule")
    body.append(f"\\multicolumn{{{len(models)+1}}}{{l}}{{\\emph{{Operator-bank "
                f"ablation, DTS+HGB (fall, $n{{=}}250$)}}}} \\\\")
    OPS = [("full", "full 16-operator bank"), ("noalpha", "$-$ all $\\alpha$"),
           ("alpha", "$\\alpha$ only"), ("order_only", "order-sensitive ops only"),
           ("order_invariant_only", "order-invariant ops only"),
           ("quantiles", "quantiles/min/max only"), ("slopes", "slopes only"),
           ("endpoints", "endpoints only")]
    for ops, label in OPS:
        sel = [r["auroc"] for r in runs if r["sweep"] == "operators"
               and r["model"] == "dts_hgb" and r["ops"] == ops]
        c = fmt(float(np.mean(sel)), float(np.std(sel)), len(sel)) if sel else "--"
        body.append(f"{label} & \\multicolumn{{4}}{{c}}{{{c}}} \\\\")
    tail = ["\\bottomrule", "\\end{tabular}",
            "\\caption{Controlled-benchmark sweeps (held-out AUROC, mean$\\pm$sd over "
            "three seeds, 500 test pairs). Bag-of-frames remains at exactly 0.500 in "
            "every setting, as Theorem~\\ref{thm:impossibility} requires, and "
            "order-invariant operator subsets of the signature itself (quantiles, "
            "min/max) also sit exactly at chance, confirming that the order-sensitive "
            "operators carry all discriminative signal. $\\alpha$-only degrades as "
            "drop SNR approaches 1 and dips mildly under a misspecified fixed "
            "$\\tau$ or very short events, while the full operator bank and "
            "train-selected $\\tau$ remain at ceiling. At these settings every "
            "order-sensitive method solves the constructions, including with 120 "
            "distractor channels: the family measures the necessity of order "
            "sensitivity, not separation among order-sensitive methods.}",
            "\\label{tab:benchsweeps}", "\\end{table}"]
    return "\n".join(head + body + tail) + "\n"


def tasks_table_compact():
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lccc}", "\\toprule",
        "Model & Fall & Sit-to-stand & Reversal \\\\",
        "\\midrule",
    ]
    order = ["dts_hgb", "alpha_lr", "minirocket", "raw_mlp", "bof_hgb"]
    for m in order:
        cells = []
        for g in ["fall", "sit2stand", "burst"]:
            mu, sd, k = agg({"sweep": "tasks", "gen": g, "model": m,
                             "n_train_pairs": 250})
            cells.append(fmt(mu, sd, k))
        lines.append(f"{MODEL_NAMES[m]} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}",
        "\\caption{Expanded controlled temporal-order benchmark: held-out AUROC "
        "(mean$\\pm$sd over five seeds, $n{=}250$ training pairs, 500 test pairs) on "
        "three constructions whose negatives are frame permutations (Fall, "
        "Sit-to-stand) or exact time reversals (Reversal) of positives, so "
        "class-conditional frame multisets are identical and "
        "Theorem~\\ref{thm:impossibility} applies. Order-invariant bag-of-frames is "
        "exactly at chance everywhere; the structured signature and order-sensitive "
        "baselines solve all three constructions. Full $n$-sweeps in the appendix "
        "artefacts.}",
        "\\label{tab:bench}", "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


with open(os.path.join(PAPERS, "bench_tasks_table.tex"), "w") as f:
    f.write(tasks_table())
with open(os.path.join(PAPERS, "bench_tasks_table_compact.tex"), "w") as f:
    f.write(tasks_table_compact())
with open(os.path.join(PAPERS, "bench_sweeps_table.tex"), "w") as f:
    f.write(sweeps_table())
print("tables written;", len(runs), "runs aggregated")
