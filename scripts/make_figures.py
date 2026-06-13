"""Generate the four paper figures as PDFs from the experiment JSONs."""
import json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

W = Path('.')
OUT = Path('../paper')
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                     'legend.fontsize': 6.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'figure.dpi': 150, 'pdf.fonttype': 42})

dm = json.load(open(W/'dedup_main.json'))
rows = {k: {'t':v['test'],'c':v['test_ci']} for k,v in dm.items()}
mrr = json.load(open(W/'mr_rerun.json'))
rows['MiniROCKET'] = {'t':mrr['test'],'c':mrr['test_ci']}

# ---------------- Figure 1: main results, full width, SAME y-range both panels ----
order = ['LSTM','GRU','Transformer','SimpleST-GCN','MiniROCKET','DTS+LR','DTS+ET','DTS+RF','DTS+HGB','DTS-Net']
groups = {'LSTM':'seq','GRU':'seq','Transformer':'seq','SimpleST-GCN':'graph','MiniROCKET':'conv',
          'DTS+LR':'dts','DTS+ET':'dts','DTS+RF':'dts','DTS+HGB':'dts','DTS-Net':'dtsnet'}
cols = {'seq':'#7c9fc9','graph':'#c98f7c','conv':'#b3a16e','dts':'#5d9e75','dtsnet':'#8d7cc9'}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
ymin = 0.70
for ax, met, ttl in [(axes[0],'auroc','AUROC'),(axes[1],'rec','Recall (fall detection rate)')]:
    vals = [rows[k]['t'][met] for k in order]
    lo = [rows[k]['t'][met]-rows[k]['c'][met][0] for k in order]
    hi = [rows[k]['c'][met][1]-rows[k]['t'][met] for k in order]
    cs = [cols[groups[k]] for k in order]
    x = np.arange(len(order))
    ax.bar(x, vals, color=cs, yerr=[lo,hi], error_kw={'lw':0.8,'capsize':2}, width=0.7)
    for xi, v in zip(x, vals):
        ax.text(xi, ymin+0.005, f'{v:.4f}' if met=='auroc' else f'{v:.3f}',
                rotation=90, ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace('SimpleST-GCN','Simple\nST-GCN').replace('MiniROCKET','Mini\nROCKET').replace('Transformer','Trans-\nformer') for k in order], fontsize=6)
    ax.set_ylim(ymin, 1.005)
    ax.set_title(ttl)
    ax.grid(axis='y', alpha=0.25)
axes[0].set_ylabel('Score')
fig.tight_layout()
fig.savefig(OUT/'fig1_main.pdf', bbox_inches='tight'); plt.close(fig)

# ---------------- Figure 2: leave-session-out ----------------
lso = json.load(open(W/'lso_ci.json'))
fig, ax = plt.subplots(figsize=(3.3, 2.3))
x = np.arange(4)
au = [lso[s]['auroc'] for s in '1234']; f1 = [lso[s]['f1'] for s in '1234']
au_e = [[lso[s]['auroc']-lso[s]['ci']['auroc'][0] for s in '1234'],
        [lso[s]['ci']['auroc'][1]-lso[s]['auroc'] for s in '1234']]
f1_e = [[lso[s]['f1']-lso[s]['ci']['f1'][0] for s in '1234'],
        [lso[s]['ci']['f1'][1]-lso[s]['f1'] for s in '1234']]
ax.errorbar(x-0.07, au, yerr=au_e, fmt='o-', label='AUROC', capsize=2, lw=1)
ax.errorbar(x+0.07, f1, yerr=f1_e, fmt='s--', label='F1', capsize=2, lw=1)
ax.axhline(np.mean(au), color='C0', ls=':', lw=0.8)
ax.axhline(np.mean(f1), color='C1', ls=':', lw=0.8)
ax.text(3.45, np.mean(au), f'mean {np.mean(au):.4f}', fontsize=6, va='bottom', ha='right', color='C0')
ax.text(3.45, np.mean(f1), f'mean {np.mean(f1):.3f}', fontsize=6, va='bottom', ha='right', color='C1')
ax.set_xticks(x)
ax.set_xticklabels([f'Session {s}\n(n={lso[s]["n"]:,})' for s in '1234'], fontsize=6.5)
ax.set_ylabel('Score'); ax.set_ylim(0.82, 1.01)
ax.legend(loc='lower left'); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT/'fig2_lso.pdf', bbox_inches='tight'); plt.close(fig)

# ---------------- Figure 3: leave-fall-origin + transfer gap + alpha distributions --
lfo = json.load(open(W/'lfo_ci.json'))
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
holds = ['Bed','Chair','Stand']
x = np.arange(3)
au = [lfo[h]['auroc'] for h in holds]; f1 = [lfo[h]['f1'] for h in holds]
au_e = [[lfo[h]['auroc']-lfo[h]['ci']['auroc'][0] for h in holds],
        [lfo[h]['ci']['auroc'][1]-lfo[h]['auroc'] for h in holds]]
f1_e = [[lfo[h]['f1']-lfo[h]['ci']['f1'][0] for h in holds],
        [lfo[h]['ci']['f1'][1]-lfo[h]['f1'] for h in holds]]
axes[0].bar(x-0.18, au, 0.36, yerr=au_e, label='AUROC', color='#5d9e75', error_kw={'lw':0.8,'capsize':2})
axes[0].bar(x+0.18, f1, 0.36, yerr=f1_e, label='F1', color='#7c9fc9', error_kw={'lw':0.8,'capsize':2})
axes[0].set_xticks(x); axes[0].set_xticklabels(holds)
axes[0].set_ylim(0.5, 1.02); axes[0].legend(loc='lower right'); axes[0].grid(axis='y', alpha=0.25)
axes[0].set_title('Leave-fall-origin (held-out group)')
axes[0].set_ylabel('Score')

iid = lfo['iid_ref']['f1']; mean_f1 = np.mean(f1)
axes[1].bar([0,1], [iid, mean_f1], 0.5, color=['#999999','#7c9fc9'])
axes[1].set_xticks([0,1]); axes[1].set_xticklabels(['Stratified\n(iid)','Leave-fall-\norigin'], fontsize=7)
axes[1].set_ylim(0.80, 1.0)
axes[1].annotate('', xy=(1, mean_f1), xytext=(1, iid), arrowprops=dict(arrowstyle='<->', lw=0.8))
axes[1].text(1.08, (iid+mean_f1)/2, f'$\\Gamma_{{F1}}$ = {iid-mean_f1:.3f}', fontsize=7)
for xi, v in [(0,iid),(1,mean_f1)]:
    axes[1].text(xi, v+0.004, f'{v:.3f}', ha='center', fontsize=7)
axes[1].set_title('Transfer gap (mean F1)')
axes[1].grid(axis='y', alpha=0.25)

mu_f, s_f = 0.1706767811976061, 0.1645223306517444
mu_n, s_n = 0.38148932484420617, 0.2202889746401749
xs = np.linspace(-0.5, 1.0, 400)
pf = np.exp(-(xs-mu_f)**2/(2*s_f**2))/(s_f*np.sqrt(2*np.pi))
pn = np.exp(-(xs-mu_n)**2/(2*s_n**2))/(s_n*np.sqrt(2*np.pi))
axes[2].plot(xs, pf, label='Fall', color='#c0504d')
axes[2].plot(xs, pn, label='Non-fall', color='#4f81bd')
axes[2].fill_between(xs, pf, alpha=0.15, color='#c0504d')
axes[2].fill_between(xs, pn, alpha=0.15, color='#4f81bd')
for xv, lab in [(-0.498,'$x_1$'),(0.308,'$x_2$')]:
    axes[2].axvline(xv, color='k', ls='--', lw=0.8)
    axes[2].text(xv+0.02, 2.3, lab, fontsize=7)
axes[2].set_xlabel(r'Asymmetry $\alpha(z;\tau^*)$'); axes[2].set_ylabel('Density')
axes[2].set_title('Class-conditional distributions')
axes[2].legend(); axes[2].grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT/'fig3_lfo.pdf', bbox_inches='tight'); plt.close(fig)

# ---------------- Figure 4: controlled benchmark + capacity reference scale --------
sy = json.load(open(W/'synth.json'))
lstm_params = json.load(open(W/'nn/final_LSTM.json'))['params']
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
ns = sorted(int(k) for k in sy['curve'])
dts = [sy['curve'][str(n)]['DTS'] for n in ns]
bof = [sy['curve'][str(n)]['BoF'] for n in ns]
alp = [sy['curve'][str(n)]['alpha'] for n in ns]
axes[0].plot(ns, dts, 'o-', color='#5d9e75', label='DTS+HGB (128-dim)')
axes[0].plot(ns, alp, 's--', color='#e8a33d', label=r'$\alpha$-only + LR (8-dim)')
axes[0].plot(ns, bof, '^-', color='#7c9fc9', label='BoF + LR (static, chance)')
axes[0].set_xscale('log'); axes[0].set_ylim(0.45, 1.03)
axes[0].set_xlabel('Training samples $n$'); axes[0].set_ylabel('AUROC')
axes[0].set_title('Controlled temporal-order benchmark')
axes[0].legend(loc='center right'); axes[0].grid(alpha=0.25)

ns2 = np.logspace(2, 5.2, 200)
b_dts = np.sqrt(128/ns2); b_lstm = np.sqrt(lstm_params/ns2)
nstar = lstm_params/math.log(lstm_params)
axes[1].plot(ns2, b_dts, color='#5d9e75', label=r'DTS bound $\sqrt{128/n}$')
axes[1].plot(ns2, b_lstm, color='#7c9fc9', label=rf'LSTM bound $\sqrt{{{lstm_params:,}/n}}$'.replace(',','{,}'))
axes[1].axvline(3153, color='k', ls='--', lw=0.8)
axes[1].text(3153*1.1, 4.0, '$n_{FV}$=3,153', rotation=90, fontsize=6.5, va='bottom')
axes[1].axvline(nstar, color='#c0504d', ls=':', lw=1.0)
axes[1].text(nstar*1.1, 4.0, f'$n^*\\approx${nstar:,.0f}', rotation=90, fontsize=6.5, va='bottom', color='#c0504d')
axes[1].set_xscale('log'); axes[1].set_yscale('log')
axes[1].set_xlabel('Training samples $n$'); axes[1].set_ylabel(r'$\sqrt{d/n}$ (parallel bounds)')
axes[1].set_title('Capacity reference scale (heuristic; bounds do not cross)')
axes[1].legend(loc='lower left'); axes[1].grid(alpha=0.25, which='both')
fig.tight_layout(); fig.savefig(OUT/'fig4_synth.pdf', bbox_inches='tight'); plt.close(fig)
print('figures written')
