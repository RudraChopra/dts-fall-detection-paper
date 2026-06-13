# REPRO_COMMANDS.md

One-command (or close) reproduction for every quantitative element of the paper.
Two categories:

- **Self-contained (runs anywhere, no data needed).** The controlled-benchmark
  family, all sweeps, the theorem verification, and the numbers audit. These were
  actually executed to produce the numbers in the papers' benchmark tables.
- **Artefact-dependent (needs your FallVision/URFD/GMDCSA artefacts).** The
  real-data tables (Tables 1-4, session/origin folds, transfer, LOSO, Bed
  recovery) come from your archived pipeline. The scripts below verify and
  regenerate them from your saved outputs, and add the new baselines
  (TCN / InceptionTime / MultiROCKET on the main split). Run them in the repo
  that contains your saved score vectors, split manifests, and features.

## 0. Environment

```bash
pip install -r requirements.txt
```

## 1. Self-contained: numbers audit (verifies every derivable number in the PDF)

```bash
python3 audit_numbers.py          # 87 checks; expect 82 PASS, 5 documented flags
```

The 5 flags and their resolutions are listed in `NUMBERS_AUDIT.md`. The rebuilt
papers already incorporate the resolutions (x1 ~ -0.50; Gamma_F1 = 0.951-0.880;
LOSO margin stated as a match; URFD precision Wilson CI [0.384, 0.632]; Bed K=0
harmonised to 0.712).

## 2. Self-contained: controlled-benchmark family (all numbers in Table "bench")

```bash
# resumable; appends to results_all.jsonl; ~30-60 CPU-minutes total
python3 synth_bench.py resume 100000
# aggregate into the exact LaTeX tables used by the papers
python3 analyze_bench.py
```

Constructions: `fall` (3-phase + permuted negatives), `sit2stand` (ramp +
permuted negatives), `burst` (early burst vs exact time reversal). Sweeps:
training size, drop SNR, event onset p1, event duration, sequence length T,
distractor channels, operator-bank ablations, tau-selection ablations. Seeds
0-4 (headline) / 0-2 (sweeps). Expected qualitative outputs: bag-of-frames at
exactly 0.500 everywhere; DTS/alpha near 1.0 at default SNR; alpha degrading at
SNR->1 and under misspecified fixed tau; order-invariant operator subsets at
chance.

## 3. Self-contained: finite-sample theorem verification

```bash
python3 verify_theorem.py
# expect: bound never violated; empirical failure 0.192 (n=60) -> 0.003 (n=500)
```

## 4. Artefact-dependent: regenerate real-data tables from saved outputs

Run inside your pipeline repo (the one with `score_vectors/`, `manifests/`,
`features/`):

```bash
python3 reproduce_main.py      --artefacts <path>   # Table 1 + paired bootstrap + Table 2 (fixed-FPR)
python3 reproduce_external.py  --artefacts <path>   # URFD + GMDCSA transfer (Table 3)
python3 reproduce_loso.py      --artefacts <path>   # GMDCSA LOSO (Table 4)
python3 reproduce_bed.py       --artefacts <path>   # LFO folds + few-shot Bed recovery (Fig. 2)
python3 make_score_plots.py    --artefacts <path>   # score-distribution/calibration plots (FallVision, URFD, GMDCSA, Bed pre/post)
```

These scripts recompute every cell from saved score vectors and manifests and
diff against the values in the paper; any mismatch is printed as a FAIL.

## 5. New main-split baselines: NOW RUN AND IN THE PAPERS

MultiROCKET and a TCN were trained for real on sequences rebuilt from the raw
FallVision CSV archives (`work/revised_paper/data/fallvision_extracted/`) via
the released split manifest. Pipeline fidelity was validated two ways before
trusting any new number: re-extracted DTS features reproduce DTS+HGB's main-
split AUROC EXACTLY (0.9895), and MiniROCKET re-run in the new harness gives
0.9797, inside its published CI [0.9672, 0.9839]. Zero frame-count mismatches
against the manifest across all 5,572 clips.

Results now in Table 1 (both venues):
- MultiROCKET: AUROC 0.9868 [0.9808, 0.9920], F1 0.949, FP 26, FN 32.
  Paired DTS+HGB margin +0.0027 [-0.0035, 0.0086]: NOT significant (a match).
  Fixed-FPR recalls (Table 2): 0.864 / 0.882 / 0.944 / 0.977 at 1/2/5/10%.
- TCN (4 dilated blocks, 43,713 params, JAX): AUROC 0.9629 [0.9523, 0.9734],
  F1 0.910, FP 64, FN 41. Paired margin +0.0265 [0.0163, 0.0377]: significant.
- InceptionTime (three-network ensemble, 496,129 params each, JAX, 20-epoch
  val-checkpoint protocol): AUROC 0.9834 [0.9774, 0.9889], F1 0.941, FP 22,
  FN 44. Paired margin +0.0061 [-0.0002, 0.0123]: NOT significant (a match).
  Run: python3 inception_main.py 600 (resumable).

Bed few-shot baseline comparison (now Figure 2 in the AAAI build): identical
protocol for both models (Chair+Stand base, fixed 1,547-clip Bed test,
200-clip adaptation pool, 5 seeds). DTS+HGB dominates MiniROCKET+ridge at
every K: 0.719/0.705 (K=0), 0.758/0.725 (25), 0.778/0.759 (50), 0.811/0.780
(100), 0.855/0.819 (200).

Reproduce (scripts in paper_bench/, state in /tmp/seqtf, resumable):

```bash
python3 parse_seq.py 600          # rebuild sequences from raw CSVs + manifest
python3 feat_extract.py 600       # re-extract DTS-128 features + 0.9895 check
python3 rocket_main.py 600        # MiniROCKET validation + MultiROCKET row
python3 tcn_main.py 600           # TCN row (JAX, CPU)
python3 bed_compare.py 600        # Bed few-shot comparison, both models
```

One-command verification of every headline number against the artifacts:

```bash
make verify        # = python3 scripts/verify_all_numbers.py
```

## 6. Rebuild all four PDFs

```bash
cd ../papers
python3 ../experiments/analyze_bench.py          # refresh benchmark tables
for f in aaai_named aaai_anon neurips_named neurips_anon; do
  pdflatex -interaction=nonstopmode $f.tex && pdflatex -interaction=nonstopmode $f.tex
done
```

## 7. Verification checklist before submission

- [ ] `audit_numbers.py` passes (82/87, 5 documented flags resolved in text).
- [ ] GMDCSA-24 class counts: the dataset publication reports **81 falls / 79
      ADL**; the paper's parse reports **79 falls / 81 ADL**. Verify your parse
      direction against the published CSVs before submission; if the published
      split is correct, update the three GMDCSA cells (recall/precision
      denominators) accordingly.
- [ ] Bootstrap CIs, Levene p, paired deltas re-verified against saved score
      vectors (`reproduce_main.py`).
- [ ] If `new_baselines_main_split.py` was run, add rows to Table 1 and update
      the "eleven models" count and the scripts-provided sentence.
- [ ] Drop in the official NeurIPS 2026 style file from the author kit before
      submission (the build uses a year-patched copy of the official 2025 file;
      layout is identical, but the kit file is authoritative).
- [ ] AAAI: anonymous version uses `[submission]`; camera-ready switches to no
      option (copyright slug auto-inserted).

## 8. Repo-sync items found by the local artefact run (June 2026) - RESOLVED

All five items were resolved against the connected repo; details in
NUMBERS_AUDIT.md ("Artifact-level verification" section).

- [x] **CI convention.** Regenerated the lone Clopper-Pearson cell as Wilson in
      `results/urfd_wilson_ci.json` (original run record untouched). Papers
      print Wilson [0.384, 0.632] consistent with the stated method.
- [x] **Benchmark provenance.** Paper-backing scripts and the 636-run
      `results_all.jsonl` now live in `dts-fall-detection/paper_bench/` with a
      README explaining why the repo-root rewrites cannot reproduce the
      chance-level rows. Release `paper_bench/` with the paper.
- [x] **Theorem verification provenance.** `paper_bench/verify_theorem.py` is
      the Theorem 3 simulation behind the paper's 0.192 -> 0.003 claim; the
      repo-root QDA-estimation variant is documented as a separate check.
- [x] **Ablation baseline.** Resolved: `results/twinfree/abl_tf.json` matches
      every Table 5 cell exactly, including Full = 0.9895/0.951. The
      0.9879/0.935 values came from a stale legacy ablation file.
- [x] **Artefact coverage.** All blocks located and verified: fixed-FPR from
      raw `score_vectors_twinfree.npz`, sample efficiency in `lc.json`, GMDCSA
      zero-shot in `results/gmdcsa/zeroshot.json`, LOSO in
      `research/gmdcsa_loso.json` (NOT `results/gmdcsa/loso.json`, which is an
      older run), Bed recovery in `results/twinfree/bed_adapt.json` (main) and
      `research/bed_parts/` (independent five-seed curve).

Final venue builds are mirrored at `dts-fall-detection/paper/venue_builds/`.
