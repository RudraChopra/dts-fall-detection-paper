import csv, subprocess, time, sys
from pathlib import Path
W=Path('.')
man=list(csv.DictReader(open(W/'aaai/tables/fallvision_manifest.csv')))
archives=sorted({r['archive'] for r in man})
t0=time.time()
for a in archives:
    if (W/'seq'/(a+'.npz')).exists(): continue
    if time.time()-t0>30: print('TIME, remaining:',sum(1 for x in archives if not (W/'seq'/(x+'.npz')).exists())); sys.exit(0)
    subprocess.run(['python3','parse_seqs.py',a],cwd=W,check=True)
print('ALL DONE')
