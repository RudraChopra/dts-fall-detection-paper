# NUMBERS_AUDIT.md

Audit of every quantitative claim in `paper_canonical_named.pdf` (Dynamic Trajectory
Signatures). Each derivable number was recomputed from other stated quantities by
`audit_numbers.py` (this folder). Run `python3 audit_numbers.py` to regenerate.

## Verdict summary

- 87 recomputable checks: 82 PASS, 5 FLAGGED.
- All Table 1 confusion-matrix arithmetic (recall/precision/F1/accuracy from FP/FN and
  574/541 class counts) verifies exactly for all 11 models.
- QDA Bayes error 0.286, x2=0.308, P(miss)=0.202, P(fa)=0.369, n*=17,599, all split and
  dedup arithmetic, all table means, all Table 2 fixed-FPR recalls (integer-count
  consistent), URFD/GMDCSA confusion arithmetic, and most Wilson intervals verify.

## Flagged items and resolutions applied in the rebuilt papers

| # | Item | In PDF | Recomputed | Resolution in rebuilt papers |
|---|------|--------|------------|------------------------------|
| 1 | QDA lower crossover x1 | -0.498 | -0.506 (from rounded mu/sigma) | Report "x1 ~ -0.50 (outside the support of alpha)"; effective rule (single threshold x2=0.308) unchanged. |
| 2 | Transfer gap equation | "0.950 - 0.880 = 0.071" | 0.950-0.880=0.070; 0.951-0.8797=0.0713 | Use Gamma_F1 = 0.951 - 0.880 = 0.071 (stratified reference from Table 1); note Figure 4 panel shows 0.950 due to two-decimal rounding. |
| 3 | LOSO margin DTS-MultiROCKET | "+0.012 [-0.034, 0.056]" | RESOLVED against `research/gmdcsa_paired.json`: paired pooled out-of-fold AUROC margin (0.9401 vs 0.9284; delta 0.0117 [-0.0336, 0.0564]), distinct from the difference of fold means (+0.007) | Papers state it precisely: "paired pooled out-of-fold AUROC +0.012 [-0.034, 0.056], not significant at four subjects," keeping the "matches" framing. |
| 4 | URFD precision CI (Wilson) | [0.379, 0.638] | Wilson = [0.384, 0.632]; the artifact cell is Clopper-Pearson while URFD recall and all GMDCSA count CIs in the artifacts are Wilson | Papers print Wilson [0.384, 0.632] consistent with the stated method; regenerate the lone artifact cell. |
| 5 | Bed zero-shot at K=0 | Fig.2 text "0.714"; Table 7 "0.7123" | RESOLVED against artifacts: two different protocols. LFO full Bed fold (`lfo_tf.json`, n=1747) = 0.7123; recovery-curve start (`bed_adapt.json`, n=1547; part of the fold reserved as adaptation pool) = 0.7138 -> 0.714 | Papers report 0.714 for the recovery start with a one-clause protocol explanation, and 0.7123 for the LFO fold. Both correct; different quantities. |

## Artifact-level verification (against the connected repo, June 2026)

Verified EXACTLY against `dts-fall-detection` saved outputs:

- Table 1 (all DTS+HGB cells incl. bootstrap CIs, threshold 0.45, TP/FP/FN/TN) vs
  `results/twinfree/ninefive_results.json`; dedup counts 5793/5572/221; split
  3565/892/1115 (574/541 test); tau* = 0.30.
- All five paired-bootstrap deltas and CIs vs the same file (2,000 bootstraps):
  GRU +0.01828 [0.01050, 0.02660]; Transformer +0.00847 [0.00244, 0.01506];
  MiniROCKET +0.01348 [0.00584, 0.02201]; Compact +0.07102 [0.05684, 0.08788];
  Full +0.02423 [0.01661, 0.03382].
- Matched operating points (557/574 at FP 40; 548/574 at FP 34 vs MiniROCKET FP 47).
- Table 2 fixed-FPR recalls recomputed from RAW score vectors
  (`score_vectors_twinfree.npz`): DTS+HGB 0.7962/0.8624/0.9408/0.9808 and
  MiniROCKET 0.6289/0.7770/0.8606/0.9704; AUROCs from vectors 0.98947/0.97599.
- Table 5 ablation: every row of `results/twinfree/abl_tf.json` matches, including
  Full = 0.98947/0.95124 (the 0.9879/0.935 numbers seen earlier were a stale legacy
  ablation file, not the twin-free artifact).
- Leave-session-out (`lso_tf.json`) and leave-fall-origin (`lfo_tf.json`) fold
  values and CIs; Bed 0.71234.
- Sample efficiency (`lc.json`): HGB 0.9471 / Transformer 0.9360 / MiniROCKET
  0.9296 at n=250, margins +0.018 to +0.021 for n<=500.
- GMDCSA zero-shot (`results/gmdcsa/zeroshot.json`): AUROC 0.96328 [0.93620,
  0.98492], 79 falls/81 ADL, recall 0.92405 Wilson [0.84405, 0.96473], precision
  0.87952 Wilson [0.79224, 0.93322], FP 10 FN 6, recall@5% 0.8608, @10% 0.9114.
- GMDCSA LOSO (`research/gmdcsa_loso.json`): all 16 per-subject cells and all four
  means match Table 4 (note: `results/gmdcsa/loso.json` is an older, different run).
- Bed recovery: main pipeline (`bed_adapt.json`) 0.7138/0.7839/0.8483 at
  K=0/50/200; independent five-seed curve (`research/bed_parts/K*.json`)
  0.778 -> 0.895, matching Figure 2's second curve.

Remaining artifact-dependent items not yet re-verified: Levene p, QDA parameter
estimates, DTS-Net attention statistics, runtime/memory claims, and the URFD legacy
block (recall/AUROC values match the legacy JSON; the precision CI is the lone
Clopper-Pearson cell, see flag 4).

## New experiments run for the revision (June 2026, this session)

Pipeline rebuilt from raw CSVs via the released manifest; fidelity checks
passed BEFORE any new number was accepted: 5,572/5,572 clips parsed with zero
frame-count mismatches; re-extracted DTS-128 features reproduce DTS+HGB
main-split AUROC exactly (0.9895); MiniROCKET re-run in the new harness gives
0.9797, inside its published CI [0.9672, 0.9839].

| Result | Value | Source |
|---|---|---|
| MultiROCKET main split | AUROC 0.9868 [0.9808, 0.9920], F1 0.949, FP 26, FN 32 | `multirocket_main.json` |
| DTS+HGB - MultiROCKET paired | +0.0027 [-0.0035, 0.0086], not significant | same |
| MultiROCKET recall@1/2/5/10% FPR | 0.864 / 0.882 / 0.944 / 0.977 | `multirocket_test_scores.npy` |
| TCN main split (43,713 params, JAX) | AUROC 0.9629 [0.9523, 0.9734], F1 0.910, FP 64, FN 41 | `tcn_main.json` |
| DTS+HGB - TCN paired | +0.0265 [0.0163, 0.0377], significant | same |
| InceptionTime main split (3-net ensemble, 496,129 params each) | AUROC 0.9834 [0.9774, 0.9889], F1 0.941, FP 22, FN 44 | `inception_main.json` |
| DTS+HGB - InceptionTime paired | +0.0061 [-0.0002, 0.0123], not significant (match) | same |
| Bed few-shot, DTS+HGB | 0.719 / 0.758 / 0.778 / 0.811 / 0.855 at K=0/25/50/100/200 | `bed_compare_summary.json` |
| Bed few-shot, MiniROCKET+ridge | 0.705 / 0.725 / 0.759 / 0.780 / 0.819 (DTS leads at every K) | same |
| MultiROCKET learning curve (transform refit per subsample) | 0.9501 / 0.9578 / 0.9737 / 0.9788 / 0.9868 at n=250/500/1000/2000/3565 | `lc_multirocket.json` |

Learning-curve consequence applied to the papers: DTS+HGB and MultiROCKET
trade the lead along the curve within overlapping intervals (MultiROCKET ahead
at 250 and 1,000; DTS ahead at 500, 2,000, full size); "leads at every
training size" is now scoped to the originally evaluated baselines.

Theorem 3 independent audit (fresh-context reviewer): probabilistic core
verified correct (both Hoeffding applications, Hoeffding-Serfling, constants
4eps/mu=Delta/3 and exponent 72, five failure terms, separation chain, AUROC
step, corollary). Three minor gaps found and fixed in the paper's proof text:
(i) explicit negative-side stabiliser remark (cost <= Delta/11, absorbed by the
Delta/6 slack); (ii) mu-bar defined as the empirical phase average with the
boundary convention t <= p1 n; (iii) one sentence distinguishing the
independent-draw fifth failure term from the conservative paired count.
Second-round audit of the fixed text: stabiliser chain verified
(S >= (11/12) n mu-bar; 12 eps0/(11 n mu-bar) <= 6/(11n) <= Delta/11);
absorption margin 5 Delta/66 > 0 confirmed; eps/mu-bar = Delta/12 < 1/12;
boundary convention sound (m <= tau n <= p1 n); five-term accounting clear;
one clause added so a_f and Delta inherit the empirical mu-bar; the
sub-Gaussian claim softened to "may be derived".

## Statistical-rigour round (this session)

- Paired learning-curve bootstrap (`lc_paired.json`, both models retrained per
  size in the same harness, identical subsamples and test resamples): deltas
  -0.0024 / +0.0039 / -0.0005 / +0.0040 / +0.0027 at n=250/500/1000/2000/3565,
  ALL five 95% CIs contain zero. Papers now state "no significant difference
  anywhere along the curve" instead of "statistically indistinguishable based
  on overlapping intervals"; full table in the appendix (tab:lcpaired).
- Multiple comparisons: added baselines labelled as additional,
  non-prespecified comparisons; Bonferroni across the expanded eight-comparison
  family (`bonferroni8.json`): TCN [+0.0131, +0.0422] still significant;
  MultiROCKET [-0.0048, +0.0109] and InceptionTime [-0.0020, +0.0148] not.
- QDA estimation scope CORRECTED: the published Prop-1 parameters traced to the
  legacy (pre-dedup) artifact. Recomputed on the corrected twin-free TRAINING
  split only (`qda_train_only.json`): mu_f=0.162, sigma_f=0.169, mu_n=0.381,
  sigma_n=0.209, Levene p=5.3e-16, x1=-0.80 (outside support), x2=0.293,
  P(miss)=0.219, P(fa)=0.337, eps*=0.278, factor 5.5. Paper updated everywhere
  (abstract, contribution 1, Prop 1, Figure 4 right panel regenerated from
  these values).
- Proposition 2 (capacity heuristic) demoted entirely to the appendix; the
  empirical learning curve carries the data-efficiency claim.
- 388x phrased as "approximately 388x lower-dimensional (128 vs 49,728
  features; a dimensionality, not memory or runtime, comparison)".

Honesty note now reflected in both papers: with MultiROCKET added, "leads at
every fixed FPR" holds only against MiniROCKET; MultiROCKET leads at 1-2% FPR
(0.864 vs 0.796 at 1%), ties at 5%, and trails at 10% and at the F1 point
(32 vs 18 missed falls), at statistically indistinguishable AUROC. All
operating-point claims were rescoped accordingly.

## Full check log

```
check                                                              stated     computed  verdict  note
------------------------------------------------------------------------------------------------------------------------
T1 LSTM recall                                                      0.916     0.916376     PASS  
T1 LSTM precision                                                   0.893     0.893039     PASS  
T1 LSTM F1                                                          0.905     0.904557     PASS  
T1 LSTM accuracy                                                      0.9     0.900448     PASS  
T1 GRU recall                                                       0.927     0.926829     PASS  
T1 GRU precision                                                    0.913     0.912521     PASS  
T1 GRU F1                                                            0.92      0.91962     PASS  
T1 GRU accuracy                                                     0.917     0.916592     PASS  
T1 Transformer recall                                               0.934     0.933798     PASS  
T1 Transformer precision                                            0.947     0.946996     PASS  
T1 Transformer F1                                                    0.94     0.940351     PASS  
T1 Transformer accuracy                                             0.939     0.939013     PASS  
T1 CompactSTGCN recall                                              0.847      0.84669     PASS  
T1 CompactSTGCN precision                                           0.851     0.851138     PASS  
T1 CompactSTGCN F1                                                  0.849     0.848908     PASS  
T1 CompactSTGCN accuracy                                            0.845     0.844843     PASS  
T1 FullSTGCN recall                                                  0.92     0.919861     PASS  
T1 FullSTGCN precision                                              0.895     0.894915     PASS  
T1 FullSTGCN F1                                                     0.907     0.907216     PASS  
T1 FullSTGCN accuracy                                               0.903     0.903139     PASS  
T1 MiniROCKET recall                                                0.941     0.940767     PASS  
T1 MiniROCKET precision                                             0.931     0.931034     PASS  
T1 MiniROCKET F1                                                    0.936     0.935875     PASS  
T1 MiniROCKET accuracy                                              0.934     0.933632     PASS  
T1 DTS+LR recall                                                    0.897     0.897213     PASS  
T1 DTS+LR precision                                                 0.883     0.883362     PASS  
T1 DTS+LR F1                                                         0.89     0.890233     PASS  
T1 DTS+LR accuracy                                                  0.886     0.886099     PASS  
T1 DTS+ET recall                                                    0.967     0.966899     PASS  
T1 DTS+ET precision                                                 0.907     0.906863     PASS  
T1 DTS+ET F1                                                        0.936     0.935919     PASS  
T1 DTS+ET accuracy                                                  0.932     0.931839     PASS  
T1 DTS+RF recall                                                    0.963     0.963415     PASS  
T1 DTS+RF precision                                                 0.893     0.893376     PASS  
T1 DTS+RF F1                                                        0.927     0.927075     PASS  
T1 DTS+RF accuracy                                                  0.922     0.921973     PASS  
T1 DTS+HGB recall                                                   0.969     0.968641     PASS  
T1 DTS+HGB precision                                                0.934     0.934454     PASS  
T1 DTS+HGB F1                                                       0.951      0.95124     PASS  
T1 DTS+HGB accuracy                                                 0.949     0.948879     PASS  
T1 DTS-Net recall                                                   0.929     0.928571     PASS  
T1 DTS-Net precision                                                0.927     0.926957     PASS  
T1 DTS-Net F1                                                       0.928     0.927763     PASS  
T1 DTS-Net accuracy                                                 0.926     0.925561     PASS  
QDA x1                                                             -0.498    -0.505699     FAIL  
QDA x2                                                              0.308     0.307699     PASS  
QDA P(miss|f)                                                       0.202      0.20372     PASS  
QDA P(fa|n)                                                         0.369     0.369469     PASS  
QDA Bayes error                                                     0.286     0.286595     PASS  
Factor-5.6 reduction (0.286/0.051)                                    5.6     5.607843     PASS  
n* = d/ln d                                                         17599 17599.665271     PASS  
5,793 - 221 = 5,572                                                  5572         5572     PASS  
3,565+892+1,115 = 5,572                                              5572         5572     PASS  
test falls+nonfalls = 1,115                                          1115         1115     PASS  
FallVision parse falls+nonfalls                                      5793         5793     PASS  
Leave-session mean AUROC                                           0.9649       0.9649     PASS  
LFO mean AUROC                                                     0.8961       0.8961     PASS  
LFO mean F1                                                          0.88     0.879667     PASS  
Transfer gap as printed (0.950-0.880=0.071)                         0.071 0.06999999999999995     FAIL  PDF text arithmetic inconsistent; with Table 1 F1=0.951: 0.951-0.8797=0.0713 -> 0.071. Fix text to 0.951-0.880.
Transfer gap from unrounded (0.951-0.8797)                          0.071     0.071333     PASS  
LOSO DTS+HGB mean                                                   0.941      0.94075     PASS  
LOSO MultiROCKET mean                                               0.934        0.934     PASS  
LOSO MiniROCKET mean                                                0.918        0.918     PASS  
LOSO alpha-only mean                                                0.917        0.917     PASS  
LOSO margin DTS-MultiROCKET                                         0.012      0.00675     FAIL  
URFD recall                                                           1.0          1.0     PASS  
URFD precision                                                      0.508     0.508475     PASS  
URFD F1                                                             0.674     0.674157     PASS  
URFD fixed-thr FPR (29/40)                                          0.725        0.725     PASS  
GMDCSA recall                                                       0.924     0.924051     PASS  
GMDCSA precision                                                     0.88     0.879518     PASS  
GMDCSA F1                                                           0.901     0.901235     PASS  
GMDCSA clips                                                          160          160     PASS  
T2 recall@1% = 0.796 -> 457/574 = 0.7962                            0.796       0.7962     PASS  integer count consistency
T2 recall@2% = 0.862 -> 495/574 = 0.8624                            0.862       0.8624     PASS  integer count consistency
T2 recall@5% = 0.941 -> 540/574 = 0.9408                            0.941       0.9408     PASS  integer count consistency
T2 recall@10% = 0.981 -> 563/574 = 0.9808                           0.981       0.9808     PASS  integer count consistency
Matched FP=40: 557/574                                               0.97     0.970383     PASS  recall at matched FP
Matched recall 548/574 = 95.5%                                      0.955     0.954704     PASS  
URFD recall Wilson lo                                               0.886     0.886487     PASS  
GMDCSA recall Wilson lo                                             0.844     0.844046     PASS  
GMDCSA recall Wilson hi                                             0.965     0.964728     PASS  
GMDCSA precision Wilson lo                                          0.792     0.792237     PASS  
GMDCSA precision Wilson hi                                          0.933     0.933223     PASS  
URFD precision Wilson lo                                            0.379     0.384351     FAIL  
URFD precision Wilson hi                                            0.638     0.631562     FAIL  
Bed K=0 0.714 vs Table 7 0.7123                                     0.714       0.7123     PASS  0.7123 rounds to 0.712 not 0.714; flag: harmonize (use 0.712/0.7123 everywhere or explain seed difference)
------------------------------------------------------------------------------------------------------------------------
TOTAL 87 checks, 5 flagged
```
