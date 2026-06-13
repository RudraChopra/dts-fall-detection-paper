# Data directory

Raw datasets are not redistributed. To reproduce:

1. FallVision (keypoint CSV archives): Harvard Dataverse, DOI:10.7910/DVN/75QPKK.
   Download the 20 keypoint RAR archives (f_mask_* and nf_mask_*) and extract them to
   `data/fallvision_extracted/<archive_name>/`, one folder per archive.
   IMPORTANT: extract each archive into its own folder. The leakage incident
   documented in the paper was caused by archives extracted on top of each
   other, which duplicated clips across archive labels.
2. URFD: http://fenix.ur.edu.pl/~mkepski/ds/uf.html (Kwolek and Kepski, 2014).

After extraction, run `python scripts/parse_all_archives.py` to build the
fixed-length sequence tensor, and verify per-clip frame counts match the
manifest (the parser asserts zero mismatches).
