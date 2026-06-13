# Change log relative to the two earlier drafts

## Critical correction (found during verification)
The original "strict" FallVision extraction duplicated every clip 3-4x across
archive folders; 728 duplicated clips appeared in BOTH train and test. All
headline numbers in both earlier drafts (e.g. AUROC 0.9998, recall 1.000) were
inflated by this leakage. Every experiment was re-run under a deduplicated,
leakage-controlled protocol; the paper now reports those honest numbers and
discloses the correction (Data section footnote + Limitations erratum).

## New experiments run for this version (all real, all on the actual data)
- Validation-only hyperparameter grid search for DTS+LR/ET/RF/HGB (grids in Table 1).
- LSTM, GRU, Transformer retrained from scratch on the clean split (15-epoch
  one-cycle + plateau extension, best-validation checkpointing).
- Compact ST-GCN (2 GCN + 2 TCN layers, 227K params) implemented and trained.
- DTS-Net retrained; attention/importance correlation recomputed (r=0.650, p=0.081;
  the old r=0.760, p=0.029 came from the leaked-era run and was replaced).
- Bootstrap 95% CIs from saved score vectors for every retrained model; paired
  bootstrap AUROC differences (DTS+HGB significantly ahead of every baseline).
- Leave-session-out and leave-fall-origin re-run with per-fold bootstrap CIs.
- Controlled temporal-order benchmark regenerated with pair-level splits
  (fixes a subtle twin-leakage artifact); BoF now exactly at chance 0.500.
- Single-family + alpha-zero ablations re-run under the clean protocol.
- Residual duplicate sensitivity analysis (48/986 test twins; effect < 0.0011 AUROC).

## Mentor checklist
1. CIs for all results .......................... done (every table + abstract)
2. Hyperparameter tuning ........................ done (real grids, validation-only, Table 1)
3. Undefined epsilon in Eq. 1 ................... defined (eps = 1e-8, after Eq. 1)
4. Figure 1 too small / y-axes .................. full width, both panels same y-range
5. Fig 4 "crossover" with parallel lines ........ reframed as reference scale n* = d/ln d;
   the text and caption now state explicitly the bounds are parallel and do not cross.
   n* recomputed from the actual LSTM parameter count (216,193 -> n* ~ 17,599).
6. Em dashes .................................... all removed
7. "Optimal Split tau" header LaTeX bug ......... fixed
8. Remark 1 ..................................... removed; QDA error recomputed exactly
   from the fitted parameters: eps* = 0.286 (old 0.287/0.294 both replaced)
9. "by 0.077 AUROC ... 0.131 recall" ............ sentence superseded (leaked numbers);
   all comparisons now phrased "in AUROC" with paired-bootstrap CIs
10. 3 vs 4 decimal places ....................... policy: AUROC 4dp, others 3dp (stated in captions)
11. controlled vs synthetic benchmark ........... standardized to "controlled temporal-order benchmark"
12. ROCKET-style baseline ....................... MiniROCKET (aeon) included with kernel grid search

## Other fixes
- Transfer-gap arithmetic corrected (0.992-0.891=0.055 was wrong; now 0.939-0.892=0.047
  with a protocol-consistent iid reference).
- Meta-commentary about "the revised manuscript" removed everywhere.
- x1/x2/P(miss)/P(false alarm) recomputed exactly: x1=-0.498, x2=0.308, 0.202/0.369.
- Multi-family synthetic ablation rows dropped (not reproducible under a faithful
  reconstruction of the benchmark; FallVision ablations retained).
- References extended (ROCKET, MiniRocket, aeon); bibliography compiled.

# Round 2 (mentor follow-up fixes)

1. Residual duplicates: the 48 feature-identical test twins are now REMOVED from
   the main test set. All main-table rows, CIs, paired deltas, ablations, and the
   transfer-gap reference were recomputed on the cleaned 938-clip test set
   (417 falls, 521 non-falls). Full-986 numbers retained as a sensitivity note.
2. MiniROCKET scoring bug found and fixed: aeon's ridge classifier has no
   predict_proba, so the archived AUROC 0.9402 was computed from binarised
   predictions ((0.949+0.932)/2 = 0.9402 exactly). Rerun with proper ridge
   decision scores: MiniROCKET = 0.9804, now the strongest baseline. DTS+HGB
   remains ahead with a small but significant paired delta +0.0075 [+0.0001, +0.0150].
3. Claim strength: every superiority claim is now explicitly about ranking
   quality (AUROC); threshold metrics are framed as operating points. Abstract,
   contributions, results, discussion, limitations, and conclusion updated.
4. Ablation framing: the paper now states plainly that several DTS families are
   individually dispensable on FallVision and the eight-family design is
   redundancy for robustness, not per-family necessity.
5. Baseline scope: expanded justification for excluding pretrained full ST-GCN /
   PoseC3D (pretraining would confound the data-scarce question); compact ST-GCN
   framed as a capacity-appropriate from-scratch graph control; pretrained
   transfer flagged as future work.

# Round 3 (strict twin-free protocol, verified)

1. Main table now reports the STRICT TWIN-FREE protocol: feature-level dedup
   (5,793 -> 5,572 unique groups; 221 duplicate members removed), stratified
   64/16/20 split (3,565/892/1,115; test 574 falls/541 non-falls), zero twins
   across any split boundary (independently re-verified). All 11 models
   retrained: DTS+HGB 0.9895 [0.9851, 0.9935] remains top AUROC.
2. New FullST-GCN-COCO baseline (COCO-17-adapted ST-GCN trained from scratch,
   MPS): AUROC 0.9652. Compact ST-GCN retained as capacity control.
3. Every number in the main table, all five paired deltas, and the matched
   operating-point claims were independently re-verified from the released
   score vectors; the published DTS+HGB row was reproduced exactly by an
   independent retrain (AUROC 0.9895).
4. All five paired comparisons remain significant after Bonferroni correction
   (99% paired-bootstrap intervals exclude zero); statement added to the paper.
5. Leave-session-out and leave-fall-origin RERUN on the deduplicated 5,572-clip
   pool: LSO mean AUROC 0.9649 (was 0.9729 on the duplicated pool), LFO mean
   0.8961 with Bed 0.7123 (was 0.9197/0.7936). Transfer gap 0.071. The harder,
   honest numbers strengthen the predicted Bed-origin effect.
6. Ablation table rerun on the twin-free split (largest single-family AUROC
   drop: hip height, -0.0039).
7. Matched-recall sentence made exact (548 of 574 falls: DTS+HGB 34 FP vs
   MiniROCKET 47). URFD provenance note added.
8. Anonymized submission PDF + named arXiv PDF + Overleaf zip produced; score
   vectors, split manifest, and dedup audit shipped under results/twinfree/.

# Round 4 (sample efficiency, Bed adaptation, page compliance)

1. NEW real-data sample-efficiency study (Figure 5): DTS+HGB vs Transformer vs
   MiniROCKET at n in {250, 500, 1000, 2000, 3565} on the strict twin-free
   split. DTS+HGB leads at every size (e.g. 0.9471 vs 0.9360/0.9296 at n=250),
   directly testing the data-scarcity thesis on real data.
2. NEW Bed-fold few-shot adaptation: adding K labelled Bed clips to training
   (fixed 1,547-clip Bed eval set) raises AUROC 0.714 (K=0) -> 0.784 (K=50)
   -> 0.848 (K=200); the hardest fold is quantified and partially recoverable.
3. LSO/LFO/ablation moved fully onto the twin-free protocol (dedup pool):
   LSO mean 0.9649, LFO mean 0.8961 (Bed 0.7123), gap 0.071; ablation table
   regenerated on the strict split; all stale round-2 numbers purged.
4. Bonferroni statement added (all five primary paired comparisons stay
   significant at the corrected level); matched-recall claim made exact.
5. Page budget enforced: 7 pages content + references only on page 8 (AAAI
   compliant), zero errors/overfull/undefined; anonymized and named PDFs and
   the Overleaf zip rebuilt.
