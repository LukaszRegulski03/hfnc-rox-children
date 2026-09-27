"""One analysis: develop on one set of datasets, validate on another, at one landmark.

    results = run(dev, ext, settings, out_dir)

For each of ROX, ROX-HR and the eight tuned models it measures, on development
and on the held-out validation data:

1. discrimination: AUROC (development: mean over the outer folds of nested
   cross-validation; validation: with a stratified bootstrap interval);
2. an operating point: the cut-off reaching >= 0.60 sensitivity on development
   with the highest specificity, frozen and applied unchanged to validation;
3. the comparison with ROX-HR: paired difference in balanced accuracy at the
   frozen operating points (primary) and in AUROC (DeLong);
4. calibration and Brier score, after mapping each score to a probability on
   development data only;
5. decision curves (net benefit against ROX-HR);
6. SHAP importance of each predictor.

Nothing is chosen after seeing the validation data: every cut-off, every
hyperparameter and every probability mapping is fitted on development only.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import metrics as M
from .config import DCA_RANGE, FEATURES
from .models import NAMES, score, tuned_models, untuned_models
from .shap_importance import shap_importance

INDICES = {'ROX-HR': 'rox_hr', 'ROX': 'rox'}


def add_indices(df):
    """ROX = (SpO2 / FiO2) / respiratory rate; ROX-HR = ROX / heart rate x 100, at
    the landmark. Undefined (NaN) where a vital sign is missing; not imputed."""
    df = df.copy()
    df['rox'] = df.SpO2_T / (df.FiO2_T * df.RR_T)
    df['rox_hr'] = df['rox'] / df.HR_T * 100
    return df


def nested_cv_auroc(est, grid, X, y, s):
    """Honest development AUROC of a tuned model: the grid is searched inside
    each outer training fold, and the chosen configuration is scored on the
    outer fold it never saw. Returns the AUROC of every outer fold."""
    out = np.full((s.cv_repeats, s.cv_folds), np.nan)
    for rep in range(s.cv_repeats):
        outer = StratifiedKFold(s.cv_folds, shuffle=True, random_state=s.seed + rep)
        for k, (tr, te) in enumerate(outer.split(X, y)):
            inner = StratifiedKFold(s.inner_folds, shuffle=True, random_state=s.seed + 100 * rep + k)
            search = GridSearchCV(clone(est), grid, scoring=s.tuning_metric, cv=inner, n_jobs=s.jobs,
                                  refit=True, error_score=np.nan).fit(X[tr], y[tr])
            out[rep, k] = M.auroc(y[te], score(search.best_estimator_, X[te]))
    return out


def final_model(est, grid, X, y, s):
    """The configuration chosen on all development data, refitted on it."""
    inner = StratifiedKFold(s.inner_folds, shuffle=True, random_state=s.seed)
    search = GridSearchCV(clone(est), grid, scoring=s.tuning_metric, cv=inner, n_jobs=s.jobs,
                          refit=True, error_score=np.nan).fit(X, y)
    return search.best_estimator_, search.best_params_


def out_of_fold_scores(model, X, y, s):
    """Development scores of the final configuration, each child scored by a
    model that did not see them, averaged over all cross-validation repeats.
    Thresholds and probability mappings are fitted on these. (A cut-off frozen
    on one 5-fold split varies widely with the split; averaging removes that.)"""
    oof = np.zeros((s.cv_repeats, len(y)))
    for rep in range(s.cv_repeats):
        folds = StratifiedKFold(s.cv_folds, shuffle=True, random_state=s.seed + rep)
        for tr, te in folds.split(X, y):
            oof[rep, te] = score(clone(model).fit(X[tr], y[tr]), X[te])
    return oof.mean(0)


def _point(prefix, op):
    return {f'{prefix}_{k}': v for k, v in op.items()}


def fit_models(X_dev, y_dev, X_ext, s, models=None, log=print):
    """Tune, fit and score the eight models. Returns name -> scores and the
    development AUROC from nested cross-validation."""
    pos_weight = float((y_dev == 0).sum() / (y_dev == 1).sum())
    out = {}
    for key, (est, grid) in tuned_models(pos_weight).items():
        if models and key not in models:
            continue
        t = time.time()
        folds = nested_cv_auroc(est, grid, X_dev, y_dev, s)
        model, params = final_model(est, grid, X_dev, y_dev, s)
        out[NAMES[key]] = dict(key=key, oof=out_of_fold_scores(model, X_dev, y_dev, s),
                               ext=score(model, X_ext), dev_auroc=float(np.nanmean(folds)),
                               dev_auroc_fold_sd=float(np.nanstd(folds)),
                               chosen_params=json.dumps({k: str(v) for k, v in params.items()}))
        log(f'  {NAMES[key]:<32} development AUROC {np.nanmean(folds):.3f}   {time.time() - t:.0f}s')
    return out


def evaluate(dev, ext, fitted, s, out_dir):
    """Every reported number, from the models' scores. Writes results.csv,
    decision_curve.csv, decision_curve_vs_roxhr.csv and scores.npz."""
    dev, ext = add_indices(dev), add_indices(ext)
    y_dev, y_ext = dev.failure.values.astype(int), ext.failure.values.astype(int)
    rows, flags, probs, arrays = [], {}, {}, {'y_dev': y_dev, 'y_ext': y_ext}

    # ── the two indices ──────────────────────────────────────────────────────
    for name, col in INDICES.items():
        sd, se = -dev[col].values, -ext[col].values        # low index = high risk
        md, me = np.isfinite(sd), np.isfinite(se)
        thr = M.threshold_for_sensitivity(y_dev[md], sd[md], s.target_sensitivity)
        lo, hi = M.bootstrap_ci(y_ext[me], se[me], s.n_boot)
        rows.append(dict(kind='index', predictor=name, cutoff=-thr, threshold=thr,
                         n_dev_defined=int(md.sum()), n_ext_defined=int(me.sum()),
                         dev_auroc=M.auroc(y_dev[md], sd[md]), ext_auroc=M.auroc(y_ext[me], se[me]),
                         ext_auroc_lo=lo, ext_auroc_hi=hi,
                         **_point('dev', M.operating_point(y_dev[md], sd[md], thr)),
                         **_point('ext', M.operating_point(y_ext[me], se[me], thr))))
        flags[name] = (se >= thr).astype(int)
        probs[name] = M.platt(sd, y_dev, se)
        arrays[f'score_dev__{name}'], arrays[f'score_ext__{name}'] = sd, se

    # ── the models ───────────────────────────────────────────────────────────
    for name, f in fitted.items():
        oof, s_ext = np.asarray(f['oof'], float), np.asarray(f['ext'], float)
        p_ext = M.platt(oof, y_dev, s_ext)
        thr = M.threshold_for_sensitivity(y_dev, oof, s.target_sensitivity)
        lo, hi = M.bootstrap_ci(y_ext, s_ext, s.n_boot)
        blo, bhi = M.bootstrap_ci(y_ext, p_ext, s.n_boot,
                                  fn=lambda yy, pp: brier_score_loss(yy, np.clip(pp, 0, 1)))
        slope, intercept = M.calibration(y_ext, p_ext)
        rows.append(dict(kind='model', predictor=name, model_key=f.get('key'), threshold=thr,
                         chosen_params=f.get('chosen_params'),
                         dev_auroc=f['dev_auroc'], dev_auroc_fold_sd=f['dev_auroc_fold_sd'],
                         ext_auroc=M.auroc(y_ext, s_ext), ext_auroc_lo=lo, ext_auroc_hi=hi,
                         brier=brier_score_loss(y_ext, p_ext), brier_lo=blo, brier_hi=bhi,
                         calibration_slope=slope, calibration_intercept=intercept,
                         **_point('dev', M.operating_point(y_dev, oof, thr)),
                         **_point('ext', M.operating_point(y_ext, s_ext, thr))))
        flags[name] = (s_ext >= thr).astype(int)
        probs[name] = p_ext
        arrays[f'score_dev__{name}'], arrays[f'score_ext__{name}'] = oof, s_ext
        arrays[f'prob_ext__{name}'] = p_ext

    # ── comparisons with ROX-HR, on the same children ────────────────────────
    ref = arrays['score_ext__ROX-HR']
    bal = M.paired_balanced_accuracy(flags, y_ext, 'ROX-HR', s.n_boot, s.seed)
    for r in rows:
        name = r['predictor']
        r['ext_balanced_accuracy'] = M.balanced_accuracy(flags[name], y_ext)
        if name != 'ROX-HR':
            r['d_balacc_vs_roxhr'], r['d_balacc_lo'], r['d_balacc_hi'] = bal[name]
            _, _, r['d_auroc_vs_roxhr'], r['d_auroc_lo'], r['d_auroc_hi'], r['delong_p'] = \
                M.delong(y_ext, arrays[f'score_ext__{name}'], ref)
    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(out_dir, 'results.csv'), index=False)

    # ── decision curves ──────────────────────────────────────────────────────
    grid = np.linspace(*DCA_RANGE, 60)
    pd.DataFrame({'threshold_probability': grid,
                  'escalate_all': M.net_benefit(y_ext, np.ones(len(y_ext)), grid),
                  **{k: M.net_benefit(y_ext, p, grid) for k, p in probs.items()}}
                 ).to_csv(os.path.join(out_dir, 'decision_curve.csv'), index=False)
    pd.DataFrame(M.net_benefit_vs_reference(y_ext, probs, 'ROX-HR', np.linspace(*DCA_RANGE, 40),
                                            s.n_boot, s.seed)
                 ).to_csv(os.path.join(out_dir, 'decision_curve_vs_roxhr.csv'), index=False)
    np.savez_compressed(os.path.join(out_dir, 'scores.npz'), **arrays)
    return results


def run(dev, ext, s, out_dir, models=None, log=print):
    """The whole analysis for one landmark and split."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    json.dump(s.as_dict(), open(os.path.join(out_dir, 'config.json'), 'w'), indent=2)
    X_dev, X_ext = dev[FEATURES].astype(float).values, ext[FEATURES].astype(float).values
    y_dev = dev.failure.values.astype(int)
    log(f'development n={len(dev)} ({y_dev.sum()} failures); '
        f'validation n={len(ext)} ({ext.failure.sum()} failures)')
    fitted = fit_models(X_dev, y_dev, X_ext, s, models, log)
    results = evaluate(dev, ext, fitted, s, out_dir)

    # SHAP, on the a-priori configurations (as published)
    pos_weight = float((y_dev == 0).sum() / (y_dev == 1).sum())
    untuned = {NAMES[k]: clone(m).fit(X_dev, y_dev) for k, m in untuned_models(pos_weight).items()
               if not models or k in models}
    shap_importance(untuned, X_dev, FEATURES, s, log).to_csv(os.path.join(out_dir, 'shap.csv'), index=False)
    log(f'done in {(time.time() - t0) / 60:.1f} min -> {out_dir}')
    return results
