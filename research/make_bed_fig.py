import json,numpy as np,matplotlib
matplotlib.use("Agg");import matplotlib.pyplot as plt
ba=json.load(open('results/twinfree/bed_adapt.json'))
Kp=sorted(int(k) for k in ba); aucp=[ba[str(k)]['auroc'] for k in Kp]
# reconstruction (independent, for shape confirmation) from bed_parts
import glob
rec={}
for f in glob.glob('research/bed_parts/K*.json'):
    K=int(f.split('K')[-1].split('.')[0]); rec[K]=json.load(open(f))
Kr=sorted(rec); aucr=[rec[k]['auroc'] for k in Kr]; sdr=[rec[k]['auroc_sd'] for k in Kr]
fig,ax=plt.subplots(figsize=(3.4,2.5))
ax.plot(Kp,aucp,'o-',color="#c0504d",label="DTS+HGB few-shot (main pipeline)")
ax.errorbar(Kr,aucr,yerr=[1.96*s for s in sdr],fmt='s--',color="#5d9e75",ms=4,capsize=2,label="independent 5-seed reproduction")
ax.axhline(0.7123,ls=':',color='gray',lw=0.8); ax.text(120,0.706,'Table 4 Bed (zero-shot)',fontsize=6,color='gray')
ax.set_xlabel("labelled Bed clips added ($K$)");ax.set_ylabel("Bed AUROC")
ax.set_title("Few-shot Bed-origin recovery",fontsize=9)
ax.legend(fontsize=6,loc='lower right');ax.grid(alpha=0.25)
fig.tight_layout();fig.savefig('paper/fig_bed.pdf' if False else 'research/fig_bed.pdf',bbox_inches='tight')
fig.savefig('/sessions/laughing-charming-gauss/mnt/outputs/fig_bed_check.png',dpi=150,bbox_inches='tight')
print('paper-pipeline curve:',dict(zip(Kp,[round(a,4) for a in aucp])))
print('reconstruction curve:',dict(zip(Kr,[round(a,4) for a in aucr])))
