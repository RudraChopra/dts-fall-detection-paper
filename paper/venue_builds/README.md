# AAAI venue builds (authoritative submission)

`aaai_named.pdf` and `aaai_anon.pdf` here are the current, authoritative AAAI main-track
build. The reproducible LaTeX source is under `src/` (`aaai_named.tex` / `aaai_anon.tex`
both `\input{body_aaai}`; build with `pdflatex` x3 against `aaai2026.sty`).

Note: the older single-source draft at `../main.tex` (aaai2027 style) is superseded by
this build and should not be used for submission.

## Build
```
cd src
pdflatex aaai_named.tex && pdflatex aaai_named.tex && pdflatex aaai_named.tex
pdflatex aaai_anon.tex  && pdflatex aaai_anon.tex  && pdflatex aaai_anon.tex
```
Output: 7 content pages, references on page 8, technical appendix from page 10.
Zero overfull boxes, zero undefined references. The anonymous build carries no author
identifying information.

## What changed in this revision
Framing pass approved by a five-advisor review (no numbers changed):

1. Headline demoted from "highest AUROC" to "matches the strongest sequence-learning
   baselines (MultiROCKET, resource-matched InceptionTime) at a fraction of the
   representation cost" across abstract, contributions, and conclusion.
2. Theory (impossibility + finite-sample separation) foregrounded as the contribution;
   the AUROC table reframed as competitive rather than a win.
3. Theorem 2 explicitly labelled elementary and scope-delimiting.
4. Four-subject GMDCSA LOSO softened to "comparable, underpowered."
5. Resource-matched InceptionTime noted as such, with the canonical five-network run
   left to future work.
6. Added a plain-language motivation for a cold reader; PGTS kept as an explicit
   conjecture, not an established general recipe.
7. Experiment A (matched three-way few-shot Bed recovery: DTS+HGB, MiniROCKET,
   MultiROCKET) integrated; `fig_bed_compare.pdf` carries all three curves.

## Reproducibility
`python3 scripts/verify_all_numbers.py` recomputes every headline number from released
artifacts: 78 checks, 0 failures (includes the Experiment A bed3 anchors and the
fixed-FPR recalls recomputed from raw score vectors).
