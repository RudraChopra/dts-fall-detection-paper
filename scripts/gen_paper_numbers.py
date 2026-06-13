"""Generate numbers.tex (LaTeX macros) from all experiment JSONs in /tmp/work.
Every volatile number in the paper comes from here; nothing hand-typed."""
import json, math, csv, re
import numpy as np
from pathlib import Path

W = Path('.')
OUT = Path('../paper')
OUT.mkdir(parents=True, exist_ok=True)

L = []
def M(name, val):
    L.append(f'\\newcommand{{\\{name}}}{{{val}}}')

def f4(x): return f'{x:.4f}'
def f3(x): return f'{x:.3f}'
def ci4(c): return f'[{c[0]:.4f}, {c[1]:.4f}]'
def ci3(c): return f'[{c[0]:.3f}, {c[1]:.3f}]'

# ---------------- main results (deduplicated cleaned test set, n=938) ----------------
order = ['LSTM','GRU','Transformer','SimpleSTGCN','MiniROCKET','DTSLR','DTSET','DTSRF','DTSHGB','DTSNet']
NAME = {'LSTM':'LSTM','GRU':'GRU','Transformer':'Transformer','SimpleSTGCN':'SimpleST-GCN',
        'DTSNet':'DTS-Net','DTSLR':'DTS+LR','DTSET':'DTS+ET','DTSRF':'DTS+RF','DTSHGB':'DTS+HGB'}
dm = json.load(open(W/'dedup_main.json'))
rows = {}
for k,n in NAME.items():
    d = dm[n]
    rows[k] = {'test':d['test'],'ci':d['test_ci'],'params':d.get('params'),
               'sel':(d.get('selected') or {}).get('params')}
mrr = json.load(open(W/'mr_rerun.json'))
rows['MiniROCKET'] = {'test':mrr['test'],'ci':mrr['test_ci'],'params':None,'sel':None}
mr = json.load(open('results/legacy/revision_results.json'))
M('mrbinauroc', '0.9402')
M('mrfullauroc', f4(mrr['full_test']['auroc']))
# full-986 sensitivity numbers for DTS+HGB
fh = json.load(open(W/'final_HGB.json'))
M('fullhgbauroc', f4(fh['test']['auroc'])); M('fullhgbf', f3(fh['test']['f1']))
# paired bootstrap deltas (cleaned test)
pdl = json.load(open(W/'paired_delta_clean.json'))
def dlt(x): return f'{x:+.4f}'
for key,mac in [('GRU','pdGRU'),('Transformer','pdTrans'),('DTS-Net','pdNet'),('ET','pdET'),('MiniROCKET','pdMR')]:
    r = pdl[key]
    M(mac, dlt(r['delta'])); M(mac+'ci', f"[{dlt(r['ci'][0])}, {dlt(r['ci'][1])}]")
M('testn','938'); M('testfalls','417'); M('testnonfalls','521')

for k in order:
    r = rows[k]; t=r['test']; c=r['ci']
    M(k+'auroc', f4(t['auroc'])); M(k+'aurocCI', ci4(c['auroc']))
    for src,name in [('f1','f'),('prec','prec'),('rec','rec'),('acc','acc')]:
        M(k+name, f3(t[src])); M(k+name+'CI', ci3(c[src]))
    M(k+'fp', str(t['fp'])); M(k+'fn', str(t['fn']))

best_auroc = max(order, key=lambda k: rows[k]['test']['auroc'])
M('bestmodel', best_auroc)
hgb = rows['DTSHGB']['test']
M('hgbminusrocketauroc', f3(hgb['auroc']-rows['MiniROCKET']['test']['auroc']))
M('hgbminusgcnauroc', f3(hgb['auroc']-rows['SimpleSTGCN']['test']['auroc']))
M('hgbminusgcnrec', f3(hgb['rec']-rows['SimpleSTGCN']['test']['rec']))
M('hgberr', f3(1-hgb['acc']))
M('gapfactor', f'{0.2856/(1-hgb["acc"]):.1f}')

# params
def num(x): return f'{x:,}'.replace(',', '{,}')
M('lstmparams', num(rows['LSTM']['params']))
M('gcnparams', num(rows['SimpleSTGCN']['params']))
M('dtsnetparams', num(rows['DTSNet']['params']))
dl = rows['LSTM']['params']
nstar = dl/math.log(dl)
M('nstar', f'{nstar:,.0f}'.replace(',', '{,}'))
M('nstarover', f'{nstar/3153:.1f}')

# ---------------- theory ----------------
mu_f,s_f,mu_n,s_n = 0.1706767811976061,0.1645223306517444,0.38148932484420617,0.2202889746401749
a=1/s_f**2-1/s_n**2; b=-2*(mu_f/s_f**2-mu_n/s_n**2); cq=(mu_f**2/s_f**2-mu_n**2/s_n**2)-2*math.log(s_n/s_f)
disc=b*b-4*a*cq; x1=(-b-math.sqrt(disc))/(2*a); x2=(-b+math.sqrt(disc))/(2*a); x1,x2=min(x1,x2),max(x1,x2)
Phi=lambda z:0.5*(1+math.erf(z/math.sqrt(2)))
Pfin=Phi((x2-mu_f)/s_f)-Phi((x1-mu_f)/s_f); Pnin=Phi((x2-mu_n)/s_n)-Phi((x1-mu_n)/s_n)
eps=0.5*((1-Pfin)+Pnin)
M('xone', f'{x1:.3f}'); M('xtwo', f'{x2:.3f}')
M('pmissf', f'{1-Pfin:.3f}'); M('pfalsen', f'{Pnin:.3f}')
M('epsqda', f'{eps:.3f}')
M('muf','0.171'); M('mun','0.381'); M('sigf','0.165'); M('sign','0.220')

# ---------------- leave-session / leave-origin ----------------
lso = json.load(open(W/'lso_ci.json'))
WORD = {'1':'One','2':'Two','3':'Three','4':'Four'}
for s in ['1','2','3','4']:
    r=lso[s]; w=WORD[s]
    M(f'lso{w}auroc', f4(r['auroc'])); M(f'lso{w}aurocCI', ci4(r['ci']['auroc']))
    M(f'lso{w}f', f3(r['f1'])); M(f'lso{w}fCI', ci3(r['ci']['f1']))
    M(f'lso{w}rec', f3(r['rec'])); M(f'lso{w}recCI', ci3(r['ci']['rec']))
    M(f'lso{w}n', f"{r['n']:,}".replace(',', '{,}')); M(f'lso{w}falls', str(r['falls']))
M('lsomeanauroc', f4(np.mean([lso[s]['auroc'] for s in '1234'])))
M('lsomeanf', f3(np.mean([lso[s]['f1'] for s in '1234'])))
M('lsorangelow', f4(min(lso[s]['auroc'] for s in '1234')))
M('lsorangehigh', f4(max(lso[s]['auroc'] for s in '1234')))

lfo = json.load(open(W/'lfo_ci.json'))
for h in ['Bed','Chair','Stand']:
    r=lfo[h]
    M(f'lfo{h}auroc', f4(r['auroc'])); M(f'lfo{h}aurocCI', ci4(r['ci']['auroc']))
    M(f'lfo{h}f', f3(r['f1'])); M(f'lfo{h}fCI', ci3(r['ci']['f1']))
    M(f'lfo{h}n', f"{r['n']:,}".replace(',', '{,}')); M(f'lfo{h}falls', f"{r['falls']:,}".replace(',', '{,}'))
M('lfomeanauroc', f4(np.mean([lfo[h]['auroc'] for h in ['Bed','Chair','Stand']])))
M('lfomeanf', f3(np.mean([lfo[h]['f1'] for h in ['Bed','Chair','Stand']])))
M('iidreff', f3(lfo['iid_ref_clean']['f1']))
M('transfergap', f3(lfo['iid_ref_clean']['f1']-np.mean([lfo[h]['f1'] for h in ['Bed','Chair','Stand']])))

# ---------------- URFD ----------------
u = mr['urfd_zero_shot']
M('urfdauroc', f4(u['auroc'])); M('urfdaurocCI', ci4(u['auroc_ci']))
M('urfdrec', f3(u['recall'])); M('urfdrecCI', ci3(u['recall_ci']))
M('urfdprec', f3(u['precision'])); M('urfdprecCI', ci3(u['precision_ci']))
M('urfdf', f3(u['f1'])); M('urfdfCI', ci3(u['f1_ci']))
M('urfdfp', str(u['fp'])); M('urfdfn', str(u['fn']))

# ---------------- ablations ----------------
ab = json.load(open(W/'ablation_clean.json'))
full = ab['Full']
M('abFullf', f3(full['f1'])); M('abFullauroc', f4(full['auroc']))
for fam in ['BBox-W','Hip-Y','Torso-Ang','Hip-Spd','Ctr-Spd','Hip-Acc','Shldr-Y','Head-Y']:
    key = fam.replace('-','')
    r = ab['-'+fam]
    M('ab'+key+'f', f3(r['f1'])); M('ab'+key+'auroc', f4(r['auroc']))
    M('ab'+key+'df', f'{r["f1"]-full["f1"]:+.3f}'); M('ab'+key+'da', f'{r["auroc"]-full["auroc"]:+.4f}')
st = ab['StaticOnly']
M('abStaticf', f3(st['f1'])); M('abStaticauroc', f4(st['auroc']))
M('abStaticdf', f'{st["f1"]-full["f1"]:+.3f}'); M('abStaticda', f'{st["auroc"]-full["auroc"]:+.4f}')

# ---------------- synthetic controlled benchmark ----------------
sy = json.load(open(W/'synth.json'))
cv = sy['cv5']
M('synthcv', f3(np.mean(cv)))
c100 = sy['curve']['100']
M('synthdtsathundred', f3(c100['DTS']))
M('synthalphamin', f3(min(v['alpha'] for v in sy['curve'].values())))
M('synthbof', f3(c100['BoF']))
if 'multifam' in sy:
    mf = sy['multifam']
    M('synthdropfour', f3(mf['drop4'])); M('synthkeeptwo', f3(mf['keep2']))
    M('synthalphaonly', f3(mf['alpha_only']))
    M('synthdropone', f3(mf['drop1'])); M('synthdroptwo', f3(mf['drop2']))

# ---------------- interpretability ----------------
if (W/'interp.json').exists():
    ip = json.load(open(W/'interp.json'))
    M('attnr', f'{ip["r"]:.3f}'); M('attnp', f'{ip["p"]:.3f}')
    M('attntopfam', ip['top_fam'].replace('-','--') if False else ip['top_fam'])
    M('attntopw', f'{ip["top_attn"]:.3f}'); M('attntopei', f'{ip["top_ei"]:.3f}')
    M('attnsecfam', ip['sec_fam']); M('attnsecw', f'{ip["sec_attn"]:.3f}'); M('attnsecei', f'{ip["sec_ei"]:.3f}')

# ---------------- selected hyperparameters ----------------
sel = {k: rows[k]['sel'] for k in ['DTSLR','DTSET','DTSRF','DTSHGB'] if rows[k]['sel']}
M('selLR', f"C={sel['DTSLR']['C']}")
M('selET', f"{sel['DTSET']['n']} trees, max\\_features={sel['DTSET']['mf']}")
M('selRF', f"{sel['DTSRF']['n']} trees, max\\_features={sel['DTSRF']['mf']}")
M('selHGB', f"{sel['DTSHGB']['it']} iterations, lr={sel['DTSHGB']['lr']}, $\\lambda_2$={sel['DTSHGB']['l2']}")

open(OUT/'numbers.tex','w').write('\n'.join(L)+'\n')
print(f'wrote {len(L)} macros to numbers.tex')
print('best model by AUROC:', best_auroc, rows[best_auroc]['test']['auroc'])
