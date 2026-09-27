"""Tests that need no patient data: the statistics on known cases, and the whole
analysis end to end on synthetic children.

    python -m pytest tests/
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from hfnc_rox import analysis, metrics, tables  # noqa: E402
from hfnc_rox.config import Settings, VITALS  # noqa: E402


def synthetic(n, seed, name):
    """Plausible infant vital signs; failure more likely with low ROX-HR."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({'uid': [f'{name}_{i}' for i in range(n)], 'dataset': name,
                      'hfnc_duration': rng.uniform(3, 72, n), 'age_months': rng.uniform(0, 24, n),
                      'SpO2_0': rng.normal(95, 3, n).clip(80, 100), 'SpO2_T': rng.normal(96, 3, n).clip(80, 100),
                      'FiO2_0': rng.uniform(.21, .8, n), 'FiO2_T': rng.uniform(.21, .8, n),
                      'RR_0': rng.normal(52, 12, n), 'RR_T': rng.normal(50, 12, n),
                      'HR_0': rng.normal(150, 20, n), 'HR_T': rng.normal(145, 20, n),
                      'fio2_0_imputed': 0})
    for v in VITALS:
        d[f'd_{v}'] = d[f'{v}_T'] - d[f'{v}_0']
    rox_hr = d.SpO2_T / (d.FiO2_T * d.RR_T) / d.HR_T * 100
    d['failure'] = (rng.uniform(size=n) < 1 / (1 + np.exp(3 * (rox_hr - rox_hr.median())))).astype(int)
    d.loc[rng.choice(n, n // 20, replace=False), 'HR_0'] = np.nan      # some missing values
    return d


def test_auroc_matches_sklearn():
    rng = np.random.default_rng(0)
    y, s = rng.integers(0, 2, 200), rng.normal(size=200)
    assert metrics.auroc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_threshold_reaches_target_sensitivity():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = y + rng.normal(size=300)
    thr = metrics.threshold_for_sensitivity(y, s, 0.60)
    assert metrics.operating_point(y, s, thr)['sens'] >= 0.60


def test_delong_of_identical_scores_is_zero():
    rng = np.random.default_rng(2)
    y, s = rng.integers(0, 2, 150), rng.normal(size=150)
    _, _, d, _, _, p = metrics.delong(y, s, s)
    assert d == 0 and p == 1.0


def test_whole_analysis_on_synthetic_children(tmp_path):
    dev = pd.concat([synthetic(300, 3, 'D1'), synthetic(100, 4, 'D2')], ignore_index=True)
    ext = synthetic(120, 5, 'D3')
    s = Settings.quick()
    s.n_boot = 50
    res = analysis.run(dev, ext, s, str(tmp_path), models=['LogReg-L2', 'HistGB'], log=lambda *a: None)
    assert set(res.predictor) == {'ROX-HR', 'ROX', 'Logistic regression (L2)', 'Histogram gradient boosting'}
    assert (res.dev_sens >= 0.595).all()               # every cut-off meets the rule on development
    assert res.ext_auroc.between(0, 1).all()
    table = tables.performance(str(tmp_path))
    assert list(table.Predictor)[:2] == [f'ROX-HR ≤ {res.set_index("predictor").cutoff["ROX-HR"]:.2f}',
                                          f'ROX ≤ {res.set_index("predictor").cutoff["ROX"]:.2f}']
