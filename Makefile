# Reproduce the strict twin-free benchmark and paper artifacts.
# Requires: data/fallvision_extracted populated (see data/README.md), pip install -r requirements.txt
reproduce:
	python scripts/run_twinfree_benchmark.py --stage all
	python scripts/make_ninefive_paper_artifacts.py
	cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
verify:
	python3 scripts/verify_all_numbers.py
verify-dedup:
	python scripts/eval_dedup_main.py
.PHONY: reproduce verify verify-dedup
