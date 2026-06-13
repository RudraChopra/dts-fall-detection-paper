# PROTOCOL.md: fairness guarantees for the added baselines

Every baseline added in the revision (MultiROCKET, TCN, resource-matched
InceptionTime, the Bed few-shot comparison, and the MultiROCKET learning
curve) satisfies the following, provable from the released artifacts:

1. **Same split manifest.** All runs read
   `results/twinfree/ninefive_core_full/split_manifest_twinfree.csv` (clip id,
   split, label, scenario, session, source path). No model sees validation or
   test data during training.
2. **Same preprocessing.** Sequences are rebuilt by `paper_bench/parse_seq.py`
   using the repo's own `dts/features.py:normalise` (hip-centred,
   torso-normalised, confidence < 0.10 zeroed, fixed length 150). Fidelity is
   verified before any new number is accepted: 5,572/5,572 clips parse with
   zero frame-count mismatches against the manifest; re-extracted DTS features
   reproduce DTS+HGB's main-split AUROC exactly (0.9895,
   `results/new_baselines/feat_validation.json`); MiniROCKET re-run in the new
   harness scores 0.9797, inside its published CI
   (`results/new_baselines/mini_eval.json`). The residual MiniROCKET
   difference vs the original 0.9760 is attributable to ridge-regularisation
   selection (validation-selected here vs leave-one-out in the original run)
   and library version; the MiniRocket transform is deterministic at seed 42.
3. **Validation-only selection.** Ridge regularisation is selected on
   validation AUROC over logspace(-3, 3, 10); decision thresholds maximise
   validation F1 on the grid [0.01, 0.99]; neural baselines (TCN,
   InceptionTime) use validation-AUROC checkpointing. Test labels are never
   used for any selection step.
4. **Same test ordering.** Saved test scores are remapped to the original
   `train_test_split` output order and asserted equal to `y_test` in
   `score_vectors_twinfree.npz` before any paired comparison
   (`rocket_main.py`, `tcn_main.py`, `inception_main.py`).
5. **No test-based checkpointing.** Checkpoint selection uses validation AUROC
   only; test scores are computed once from the selected checkpoint.
6. **Saved scores.** Test score vectors: `results/new_baselines/*_test_scores.npy`.
   Validation scores for the neural ensemble: `results/new_baselines/val_scores.npz`.
7. **Seeds recorded.** MultiRocket/MiniRocket transforms: seed 42. TCN: seed 0
   (init), epoch-indexed shuffling seeds (1000+epoch). InceptionTime: network
   seeds 100+k, shuffling seeds 5000+97k+epoch. Subsampling for the learning
   curve: stratified, seed 0. Bootstrap: seed 0, 1,000 resamples.
8. **Resource-matched InceptionTime.** Three-network ensemble (496,129
   parameters each, kernels 40/20/10, depth 6, bottleneck 32), trained under
   the same 20-epoch validation-checkpoint budget as the paper's other neural
   baselines; the canonical method ensembles five networks trained to
   convergence, so the paper labels this implementation resource-matched, not
   canonical.
