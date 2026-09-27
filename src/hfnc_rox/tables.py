"""The paper's tables, from the analysis results.

Each function takes the results directory of one analysis (one landmark, one
split) and returns a DataFrame with the numbers as printed (two decimals).
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

ORDER = ['ROX-HR', 'ROX', 'Histogram gradient boosting', 'Random forest']   # then by balanced accuracy


def f2(x):
    return '—' if x is None or x != x else f'{x:.2f}'


def ci(lo, hi):
    return f'({lo:+.2f} to {hi:+.2f})' if lo == lo else ''


def _ordered(res):
    key = res.predictor.map(lambda n: ORDER.index(n) if n in ORDER else len(ORDER))
    return res.assign(_k=key).sort_values(['_k', 'ext_balanced_accuracy'], ascending=[True, False]).drop(columns='_k')


def performance(res_dir):
    """Table 2 (2 h) and eTables 4 and 5 (6 h, 12 h): every predictor at its
    frozen operating point, on development and validation."""
    res = _ordered(pd.read_csv(os.path.join(res_dir, 'results.csv')))
    rows = []
    for r in res.itertuples():
        name = f'{r.predictor} ≤ {r.cutoff:.2f}' if r.kind == 'index' else r.predictor
        delta = ('—' if r.predictor == 'ROX-HR' else
                 f'{r.d_balacc_vs_roxhr:+.2f} {ci(r.d_balacc_lo, r.d_balacc_hi)}')
        rows.append({'Predictor': name, 'Dev. AUROC': f2(r.dev_auroc), 'Dev. Sens': f2(r.dev_sens),
                     'Dev. Spec': f2(r.dev_spec),
                     'Val. AUROC (95% CI)': f'{r.ext_auroc:.2f} ({r.ext_auroc_lo:.2f}–{r.ext_auroc_hi:.2f})',
                     'Val. Sens': f2(r.ext_sens), 'Val. Spec': f2(r.ext_spec), 'Val. PPV': f2(r.ext_ppv),
                     'Val. NPV': f2(r.ext_npv), 'Val. Bal. Acc.': f2(r.ext_balanced_accuracy),
                     'Δ Bal. Acc. vs ROX-HR (95% CI)': delta})
    return pd.DataFrame(rows)


def auroc_differences(res_dir):
    """eTable 2: paired difference in validation AUROC against ROX-HR (DeLong)."""
    res = pd.read_csv(os.path.join(res_dir, 'results.csv'))
    res = res[res.kind == 'model']
    return pd.DataFrame({'Model': res.predictor, 'AUROC': res.ext_auroc.map(f2),
                         'ΔAUROC vs ROX-HR (95% CI)': [f'{d:+.2f} {ci(lo, hi)}' for d, lo, hi in
                                                        zip(res.d_auroc_vs_roxhr, res.d_auroc_lo, res.d_auroc_hi)],
                         'p': res.delong_p.map(lambda p: f'{p:.3f}')})


def operating_point_transfer(res_dir):
    """eTable 3: each cut-off scored on the development data it was chosen on and
    on the validation data."""
    res = _ordered(pd.read_csv(os.path.join(res_dir, 'results.csv')))
    return pd.DataFrame({'Predictor': res.predictor,
                         'Dev. Sens': res.dev_sens.map(f2), 'Dev. Spec': res.dev_spec.map(f2),
                         'Dev. Bal. Acc.': res.dev_balanced_accuracy.map(f2),
                         'Val. Sens': res.ext_sens.map(f2), 'Val. Spec': res.ext_spec.map(f2),
                         'Val. Bal. Acc.': res.ext_balanced_accuracy.map(f2),
                         'Share flagged, Dev.': (res.dev_n_flagged / (res.dev_tp + res.dev_fp + res.dev_tn + res.dev_fn)).map(f2),
                         'Share flagged, Val.': (res.ext_n_flagged / (res.ext_tp + res.ext_fp + res.ext_tn + res.ext_fn)).map(f2)})


def decision_curve_summary(res_dir):
    """eTable 6: net benefit against ROX-HR across the 0.02-0.60 threshold range."""
    d = pd.read_csv(os.path.join(res_dir, 'decision_curve_vs_roxhr.csv'))
    return pd.DataFrame({'Predictor': d.predictor, 'Mean Δ net benefit': d.mean_difference.map(lambda x: f'{x:+.3f}'),
                         'Max Δ net benefit': d.max_difference.map(lambda x: f'{x:+.3f}'),
                         'Thresholds with CI above zero': d.n_ci_above_zero.astype(str) + ' of ' + d.n_thresholds.astype(str),
                         'Thresholds with CI below zero': d.n_ci_below_zero.astype(str) + ' of ' + d.n_thresholds.astype(str),
                         'Share of resamples ahead': d.share_of_resamples_ahead.map(f2)})


def calibration(res_dir):
    """eTable 7: Brier score and calibration on the validation data."""
    res = pd.read_csv(os.path.join(res_dir, 'results.csv'))
    res = res[res.kind == 'model']
    return pd.DataFrame({'Model': res.predictor,
                         'Brier (95% CI)': [f'{b:.3f} ({lo:.3f}–{hi:.3f})' for b, lo, hi in
                                            zip(res.brier, res.brier_lo, res.brier_hi)],
                         'Calibration slope': res.calibration_slope.map(f2),
                         'Calibration intercept': res.calibration_intercept.map(f2)})


def shap_shares(res_dir):
    """eTable 8: share of total SHAP importance, every model, every predictor."""
    sh = pd.read_csv(os.path.join(res_dir, 'shap.csv'))
    piv = sh.pivot_table(index='feature', columns='model', values='share')
    piv['Mean'] = piv.mean(axis=1)
    return piv.sort_values('Mean', ascending=False).round(3).reset_index()


# ── descriptive tables, from the 2-hour analysis population ─────────────────
TABLE1_VARS = [('age_months', 'Age, months'),
               ('SpO2_0', 'SpO2 at treatment initiation, %'), ('SpO2_T', 'SpO2 at 2 h, %'),
               ('FiO2_0', 'FiO2 at treatment initiation, fraction'), ('FiO2_T', 'FiO2 at 2 h, fraction'),
               ('RR_0', 'Respiratory rate at treatment initiation, /min'), ('RR_T', 'Respiratory rate at 2 h, /min'),
               ('HR_0', 'Heart rate at treatment initiation, /min'), ('HR_T', 'Heart rate at 2 h, /min')]
DATASETS = {'D1': 'Dataset 1 (USA)', 'D2': 'Dataset 2 (Brazil)', 'D3': 'Dataset 3 (Brazil)'}


def p_value(p):
    return '< 0.001' if p < 0.001 else f'{p:.3f}' if p < 0.01 else f'{p:.2f}'


def cohort_characteristics(table):
    """Table 1 (median [IQR]) and eTable 9 (mean ± SD, n). Kruskal-Wallis across
    the three datasets (every variable departs from normality), chi-square for the
    failure rate.

    The baseline FiO2 row includes the values filled with each dataset's median
    (63 of 101 in Dataset 2, 59 of 389 in Dataset 1), as in the published table."""
    df = table.copy()
    rows = [dict(Variable='n', **{DATASETS[d]: str((df.dataset == d).sum()) for d in DATASETS}, p='')]
    fail = {DATASETS[d]: f'{int(g.failure.sum())} ({100 * g.failure.mean():.1f})' for d, g in df.groupby('dataset')}
    rows.append(dict(Variable='HFNC failure, n (%)', **fail,
                     p=p_value(stats.chi2_contingency(pd.crosstab(df.dataset, df.failure))[1])))
    for col, label in TABLE1_VARS:
        dp = 2 if col.startswith('FiO2') else 1
        groups = {d: g[col].dropna() for d, g in df.groupby('dataset')}
        cells = {DATASETS[d]: (f'{x.median():.{dp}f} [{x.quantile(.25):.{dp}f}–{x.quantile(.75):.{dp}f}]'
                               f'; {x.mean():.{dp}f} ± {x.std():.{dp}f}; n = {len(x)}') for d, x in groups.items()}
        p = stats.kruskal(*groups.values())[1]
        rows.append(dict(Variable=label, **cells, p=p_value(p)))
    return pd.DataFrame(rows)


def missingness(table):
    """eTable 1: missing values by dataset and variable, before any imputation."""
    rows = []
    for d, g in table.groupby('dataset'):
        for col, label in TABLE1_VARS:
            n_miss = int(g[col].isna().sum()) + (int(g.fio2_0_imputed.sum()) if col == 'FiO2_0' else 0)
            rows.append({'Dataset': DATASETS[d], 'Variable': label, 'n': len(g), 'Missing, n': n_miss,
                         'Missing, %': round(100 * n_miss / len(g), 1)})
    return pd.DataFrame(rows)
