# Dynamic Trajectory Signatures (DTS) for Data-Scarce Skeleton Fall Detection

Code, results, and paper for **"Dynamic Trajectory Signatures: Strong Temporal Feature Engineering for Data-Scarce Skeleton Fall Detection"** (2026).

DTS is a 128-dimensional representation built from eight kinematic skeleton
primitives, each summarised by 16 temporal operators including *temporal
asymmetry*: the fraction of a primitive's total path variation concentrated in
the first fraction tau of the sequence. With zero sequence-learning parameters,
DTS + gradient-boosted trees attains the best AUROC of eleven evaluated
models on FallVision under a strict twin-free protocol.

## Headline results (strict twin-free split: 1,115 test clips, 574 falls / 541 non-falls)

| Model | AUROC | F1 | FP | FN |
|---|---|---|---|---|
| **DTS+HGB** | **0.9895** [0.9851, 0.9935] | 0.951 | 39 | 18 |
| Transformer | 0.9810 | 0.940 | 30 | 38 |
| DTS+ET | 0.9801 | 0.936 | 57 | 19 |
| DTS+RF | 0.9785 | 0.927 | 66 | 21 |
| DTS-Net | 0.9775 | 0.928 | 42 | 41 |
| MiniROCKET (ridge scores) | 0.9760 | 0.936 | 40 | 34 |
| GRU | 0.9712 | 0.920 | 51 | 42 |
| FullST-GCN-COCO | 0.9652 | 0.907 | 62 | 46 |
| LSTM | 0.9558 | 0.905 | 63 | 48 |
| DTS+LR | 0.9433 | 0.890 | 68 | 59 |
| CompactST-GCN | 0.9185 | 0.849 | 85 | 88 |

Splits are built from 5,572 unique feature groups (221 duplicate members
removed); no clip with an identical twin straddles any split boundary
(audit shipped in results/twinfree/). Paired-bootstrap AUROC deltas vs
DTS+HGB all exclude zero, and remain significant after Bonferroni
correction across the five primary comparisons. Matched operating points:
at MiniROCKET's false-positive count (40), DTS+HGB detects 557/574 falls;
at matched recall (548/574), DTS+HGB needs 34 FP vs MiniROCKET's 47.
Robustness on the deduplicated pool: leave-session-out mean AUROC 0.9649,
leave-fall-origin mean 0.8961 (Bed hardest, 0.7123, as the QDA model predicts).

## Integrity notes (please read before comparing to older drafts)

1. **Leakage correction.** Earlier drafts evaluated on a faulty archive
   extraction in which each clip appeared 3 to 4 times and 728 duplicates
   straddled train/test. Every number from those drafts (e.g. AUROC 0.9998,
   recall 1.000) was inflated and is superseded by this repository.
2. **Twin-free protocol (current).** The main split is built from 5,572 unique
   feature groups; no identical twin straddles any split boundary. Artifacts in
   `results/twinfree/` (score vectors, split manifest, dedup audit). An interim
   draft used post-hoc twin removal from the test set (986 -> 938); superseded.
3. **MiniROCKET scoring.** The ridge classifier has no `predict_proba`; an
   earlier draft accidentally computed its AUROC from binarised predictions
   (0.9402). This repo scores it with ridge decision values (0.9804).

## Repository layout

```
paper/      LaTeX source, figures, compiled PDF (AAAI 2027 two-column style)
dts/        DTS feature extractor (features.py) and model definitions (models.py)
scripts/    full experimental pipeline (see order below)
results/    every result JSON: grids, selected configs, CIs, paired deltas,
            ablations, robustness folds, synthetic benchmark, interpretability
            results/splits/  count-matched split indices + duplicate-twin mask
            results/legacy/  archived pre-correction artefacts kept for provenance
data/       place FallVision keypoint archives here (not distributed; see data/README.md)
```

## Reproducing the pipeline

```bash
pip install -r requirements.txt
# 1. parse raw keypoint CSVs into fixed-length sequences (per archive)
python scripts/parse_all_archives.py
# 2. DTS classifier grid search (validation-only selection)
python scripts/run_grid_search.py LR|ET|RF|HGB1|HGB2, then final:LR etc.
# 3. neural baselines (LSTM, GRU, Transformer, SimpleST-GCN; resumable)
python scripts/train_neural_baselines.py LSTM   # etc.
python scripts/train_dtsnet.py
# 4. MiniROCKET with ridge decision scores
python scripts/run_minirocket.py fit 10000 && python scripts/run_minirocket.py tx 10000 && python scripts/run_minirocket.py ridge 10000
# 5. robustness, ablations, synthetic benchmark
python scripts/run_leave_session_origin.py lfo|lso
python scripts/run_ablation.py 1 && python scripts/run_ablation.py 2
python scripts/run_synthetic_benchmark.py
# 6. cleaned-test evaluation + paired bootstrap
python scripts/eval_dedup_main.py
# 7. regenerate paper numbers and figures
python scripts/gen_paper_numbers.py && python scripts/make_figures.py
```

Scripts were written for a constrained execution environment, so the longer
ones (neural training, grid search) checkpoint after every epoch/stage and can
be re-invoked to resume.

## Data

FallVision: Harvard Dataverse, DOI:10.7910/DVN/75QPKK (keypoint CSV archives).
URFD: Kwolek and Kepski, 2014. Neither dataset is redistributed here; see
`data/README.md`.

## Citation

```bibtex
@misc{chopra2026dts,
  author = {Chopra, Rudra},
  title  = {Dynamic Trajectory Signatures: Strong Temporal Feature Engineering
            for Data-Scarce Skeleton Fall Detection},
  year   = {2026},
  note   = {arXiv preprint}
}
```

## License

MIT (see LICENSE).


## Reproducibility

Two commands, two scopes:

- `make verify` (= `python3 scripts/verify_all_numbers.py`): recomputes every
  headline number in the paper from the released artifacts in under a minute,
  with no raw data needed. Covers Table 1 (including the added MultiROCKET,
  resource-matched InceptionTime, and TCN rows), Table 2 fixed-FPR recalls
  recomputed from raw score vectors, ablations, session/fall-origin folds,
  external transfer, GMDCSA subject-held-out, Bed adaptation (including the
  matched-baseline comparison), and the MultiROCKET learning curve.
- `make reproduce`: retrains the full pipeline from the raw FallVision
  archives (see `data/README.md` for download instructions).

The controlled-benchmark suite, the Theorem 3 simulation, and the revision
figures regenerate from `paper_bench/` (`synth_bench.py`, `verify_theorem.py`,
`analyze_bench.py`, `make_new_figures.py`; all seeded). Protocol-fairness
guarantees for the added baselines are documented in `paper_bench/PROTOCOL.md`.
Venue builds of the paper are in `paper/venue_builds/`.
