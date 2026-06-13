# paper_bench: the code that backs the papers' controlled-benchmark tables

These are the scripts whose outputs appear in the venue builds
(`paper/venue_builds/*.pdf`). They are NOT the same as the similarly named
scripts in the repo root, which were written later for a different purpose and
do not reproduce the paper tables.

Why the root scripts differ:

- root `synth_bench.py` builds features from pipeline primitives (hip speed,
  hip acceleration). Those are diff-based, so even its "stats_only" subset is
  order-sensitive and everything scores 1.000. It cannot reproduce the paper's
  chance-level rows.
- `paper_bench/synth_bench.py` computes operators on raw frame values; its
  bag-of-frames and order-invariant subsets are provably order-invariant under
  frame permutation (Theorem 2 applies exactly), which is why they sit at
  exactly 0.500 in the paper tables.
- root `verify_theorem.py` tests QDA-parameter-estimation concentration with a
  tolerance calibrated to match expected output. The paper's 0.192 -> 0.003
  decay refers to the Theorem 3 three-phase-increment simulation implemented in
  `paper_bench/verify_theorem.py`.

Reproduce everything in the paper tables:

    python3 paper_bench/synth_bench.py resume 100000   # 645 jobs, resumable
    python3 paper_bench/analyze_bench.py               # regenerates LaTeX tables
    python3 paper_bench/verify_theorem.py              # Theorem 3 bound check

`results_all.jsonl` contains the 636 completed seeded runs used for the tables
in the venue builds. Release THIS folder as the paper's benchmark code.
