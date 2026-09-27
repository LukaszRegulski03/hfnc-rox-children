"""Step 4. Figure 2 and eFigures 2-3, from the 2-hour results of step 2.

    python scripts/4_make_figures.py       # writes results/figures/

Figure 2: ROC curves on each validation dataset (ROX, ROX-HR, histogram gradient
boosting) and SHAP importance for all eight models. eFigure 2: decision curves.
eFigure 3: calibration. Figure 1 (study design) and eFigure 1 (flow diagram)
are drawings, not generated from results.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

import _setup  # noqa: F401
from hfnc_rox.config import SEED
from hfnc_rox.models import NAMES

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--results', default=os.path.join(_setup.ROOT, 'results'))
a = p.parse_args()
OUT = os.path.join(a.results, 'figures')
os.makedirs(OUT, exist_ok=True)

MODELS = [NAMES[k] for k in ('HistGB', 'RandomForest', 'LogReg-L2', 'LogReg-L1', 'SVM-RBF',
                             'ExtraTrees', 'XGBoost', 'TabPFN')]
COLOURS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#E69F00', '#8B4513', '#B279A2']
SPLITS = {'A': dict(dev='Datasets 1+2', ext='Dataset 3'), 'B': dict(dev='Datasets 1+3', ext='Dataset 2')}
LABELS = {'HR_0': 'Heart rate, initiation', 'HR_T': 'Heart rate, 2 h', 'd_HR': 'Δ Heart rate',
          'RR_0': 'Resp. rate, initiation', 'RR_T': 'Resp. rate, 2 h', 'd_RR': 'Δ Resp. rate',
          'SpO2_0': 'SpO₂, initiation', 'SpO2_T': 'SpO₂, 2 h', 'd_SpO2': 'Δ SpO₂',
          'FiO2_0': 'FiO₂, initiation', 'FiO2_T': 'FiO₂, 2 h', 'd_FiO2': 'Δ FiO₂'}
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 6.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 400})


def load(split):
    d = os.path.join(a.results, f'L02_{split}')
    return dict(res=pd.read_csv(os.path.join(d, 'results.csv')).set_index('predictor'),
                z=np.load(os.path.join(d, 'scores.npz')),
                shap=pd.read_csv(os.path.join(d, 'shap.csv')).pivot_table(index='feature', columns='model',
                                                                           values='share'),
                curve=pd.read_csv(os.path.join(d, 'decision_curve.csv')))


def roc_band(y, s, n=2000):
    """Pointwise 95% band from a stratified bootstrap, on a common FPR grid."""
    rng = np.random.default_rng(SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    grid = np.linspace(0, 1, 101)
    curves = []
    for _ in range(n):
        i = np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]
        f, t, _ = roc_curve(y[i], s[i])
        curves.append(np.interp(grid, f, t))
    return grid, *np.percentile(np.vstack(curves), [2.5, 97.5], axis=0)


def roc_panel(ax, D, split, letter):
    y, z, res = D['z']['y_ext'], D['z'], D['res']
    g, lo, hi = roc_band(y, z['score_ext__ROX-HR'])
    ax.fill_between(g, lo, hi, color='k', alpha=.11, lw=0, label='ROX-HR 95% bootstrap band')
    for name, colour, style, width in (('Histogram gradient boosting', '#0072B2', '-', 1.4),
                                       ('ROX-HR', '#000000', '-', 2.0), ('ROX', '#7F7F7F', '--', 1.8)):
        f, t, _ = roc_curve(y, z[f'score_ext__{name}'])
        r = res.loc[name]
        ax.plot(f, t, style, lw=width, color=colour,
                label=f'{name}  {r.ext_auroc:.2f} ({r.ext_auroc_lo:.2f}–{r.ext_auroc_hi:.2f})')
    ax.plot([0, 1], [0, 1], ':', c='0.75', lw=.9)
    ax.set(xlim=(-.01, 1.01), ylim=(-.01, 1.01), xlabel='1 − specificity', ylabel='Sensitivity')
    ax.set_aspect('equal')
    ax.set_title(f"{letter}   Validation {SPLITS[split]['ext']}  (n = {len(y)}, {int(y.sum())} failures)",
                 loc='left', fontweight='bold')
    ax.legend(loc='lower right', facecolor='white', framealpha=.92, edgecolor='none')


def shap_panel(ax, D, split, letter):
    piv = D['shap'][MODELS]
    order = piv.mean(1).sort_values().index
    y = np.arange(len(order))
    ax.barh(y, piv.loc[order].mean(1), color='#BFBFBF', height=.62, label='mean of 8 models')
    for m, colour in zip(MODELS, COLOURS):
        ax.scatter(piv.loc[order, m], y, s=13, color=colour, alpha=.9, zorder=3, label=m)
    ax.set_yticks(y, [LABELS[f] for f in order])
    ax.set_xlim(0, float(piv.max().max()) * 1.06)
    ax.set_xlabel('Share of total attributed importance (SHAP)')
    ax.set_title(f"{letter}   Predictor importance, development {SPLITS[split]['dev']}", loc='left',
                 fontweight='bold')
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    return ax


DATA = {s: load(s) for s in SPLITS}

# ── Figure 2 ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9.6, 8.8))
gs = fig.add_gridspec(2, 2, hspace=.30, wspace=.30)
roc_panel(fig.add_subplot(gs[0, 0]), DATA['A'], 'A', 'A')
axb = shap_panel(fig.add_subplot(gs[0, 1]), DATA['A'], 'A', 'B')
roc_panel(fig.add_subplot(gs[1, 0]), DATA['B'], 'B', 'C')
shap_panel(fig.add_subplot(gs[1, 1]), DATA['B'], 'B', 'D')
fig.legend(*axb.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, .028), ncol=5,
           frameon=False, handlelength=1.0, columnspacing=1.1)
fig.savefig(os.path.join(OUT, 'figure2.png'), bbox_inches='tight')
plt.close(fig)

# ── eFigure 2: decision curves ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))
for ax, (split, D) in zip(axes, DATA.items()):
    c, y = D['curve'], D['z']['y_ext']
    x = c.threshold_probability
    for m, colour in zip(MODELS, COLOURS):
        ax.plot(x, c[m], lw=1.0, alpha=.8, color=colour, label=m)
    ax.plot(x, c['ROX'], ls=(0, (5, 1.4)), lw=2.0, color='#7F7F7F', label='ROX')
    ax.plot(x, c['ROX-HR'], lw=2.6, color='#000000', label='ROX-HR')
    ax.plot(x, c.escalate_all, color='0.62', ls=(0, (1.6, 1.6)), lw=1.5, label='Escalate all')
    ax.axhline(0, color='0.30', ls=(0, (7, 2)), lw=1.2, label='Escalate none')
    ax.set_ylim(-.06, c[MODELS + ['ROX', 'ROX-HR']].max().max() * 1.32)
    ax.set_xlim(x.min(), x.max())
    ax.set(xlabel='Threshold probability', ylabel='Net benefit')
    ax.set_title(f"{'AB'[split == 'B']}   Validation {SPLITS[split]['ext']} (n = {len(y)}, {int(y.sum())} failures)",
                 loc='left', fontweight='bold')
axes[0].legend(frameon=False, ncol=3, fontsize=6.0, loc='upper center', bbox_to_anchor=(1.10, -.20))
fig.subplots_adjust(bottom=.34, wspace=.26)
fig.savefig(os.path.join(OUT, 'efigure2_decision_curve.png'), bbox_inches='tight')
plt.close(fig)

# ── eFigure 3: calibration ───────────────────────────────────────────────────
fig = plt.figure(figsize=(9.4, 7.4))
gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1.0], hspace=.24, wspace=.24)
for j, (split, D) in enumerate(DATA.items()):
    y, P = D['z']['y_ext'], {m: D['z'][f'prob_ext__{m}'] for m in MODELS}
    bins = {}
    for m in MODELS:                       # five equal-sized bins of predicted risk
        q = np.quantile(P[m], np.linspace(0, 1, 6))
        q[-1] += 1e-9
        b = np.digitize(P[m], q[1:-1])
        bins[m] = ([P[m][b == k].mean() for k in range(5) if (b == k).any()],
                   [y[b == k].mean() for k in range(5) if (b == k).any()])
    allv = np.concatenate([np.concatenate(v) for v in bins.values()] + list(P.values()))
    pad = (allv.max() - allv.min()) * .08
    lim = (allv.min() - pad, allv.max() + pad)
    ax = fig.add_subplot(gs[0, j])
    for m, colour in zip(MODELS, COLOURS):
        ax.plot(*bins[m], 'o-', ms=3, lw=1.0, color=colour, alpha=.9, label=m)
    ax.plot(lim, lim, ':', c='0.6', lw=1.0, label='Perfect calibration')
    ax.axhline(y.mean(), color='0.35', ls=(0, (6, 2)), lw=1.0, label='Observed failure rate')
    ax.set(xlim=lim, ylim=lim, xlabel='Predicted probability of failure', ylabel='Observed failure frequency')
    ax.set_aspect('equal')
    ax.set_title(f"{'AB'[j]}   Validation {SPLITS[split]['ext']} (n = {len(y)}, {int(y.sum())} failures)",
                 loc='left', fontweight='bold')
    if j == 0:
        handles = ax.get_legend_handles_labels()
    axd = fig.add_subplot(gs[1, j])
    for m, colour in zip(MODELS, COLOURS):
        axd.plot(np.sort(P[m]), np.linspace(0, 1, len(y)), lw=1.0, color=colour, alpha=.9)
    axd.axvline(y.mean(), color='0.35', ls=(0, (6, 2)), lw=1.0)
    axd.set(xlim=lim, ylim=(0, 1), xlabel='Predicted probability of failure', ylabel='Cumulative\nproportion')
    axd.set_title(f"{'CD'[j]}   Distribution of predicted probabilities", loc='left', fontweight='bold')
# legend below the panels: inside panel A the curves ran through it
fig.legend(*handles, loc='upper center', bbox_to_anchor=(.5, .04), ncol=4, frameon=False, fontsize=6.6)
fig.savefig(os.path.join(OUT, 'efigure3_calibration.png'), bbox_inches='tight')
plt.close(fig)
print(f'figures written to {OUT}')
