"""Performance measures, intervals and comparisons.

Conventions used throughout:
- a *score* is "higher = higher risk". ROX and ROX-HR are protective (low value =
  high risk), so their score is the negative index.
- every bootstrap is *stratified*: children who failed and children who did not
  are resampled separately, so each resample keeps the observed number of events.
- every interval is a 95% percentile interval.
"""
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

from .config import SEED


def auroc(y, s):
    """Area under the ROC curve (the Mann-Whitney form: ties count as half)."""
    y, s = np.asarray(y), np.asarray(s, float)
    if len(np.unique(y)) < 2 or not np.isfinite(s).all():
        return np.nan
    r = stats.rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def stratified_draws(y, n_boot, seed=SEED):
    """Index arrays for `n_boot` stratified bootstrap resamples."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(np.asarray(y) == 1)[0], np.where(np.asarray(y) == 0)[0]
    return [np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
            for _ in range(n_boot)]


def bootstrap_ci(y, s, n_boot, fn=auroc, seed=SEED):
    """95% interval for fn(y, s) by stratified bootstrap."""
    y, s = np.asarray(y), np.asarray(s, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = []
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        v = fn(y[i], s[i])
        if np.isfinite(v):
            out.append(v)
    return tuple(np.percentile(out, [2.5, 97.5]))


def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong(y, s1, s2):
    """DeLong's test for two correlated ROC curves on the same children.
    Returns (AUROC 1, AUROC 2, difference, 95% CI low, high, p)."""
    y = np.asarray(y)
    order = np.argsort(-y, kind='mergesort')
    s = np.vstack([np.asarray(s1, float), np.asarray(s2, float)])[:, order]
    m = int(y.sum())
    n = len(y) - m
    tx = np.array([_midrank(s[r, :m]) for r in range(2)])
    ty = np.array([_midrank(s[r, m:]) for r in range(2)])
    tz = np.array([_midrank(s[r, :]) for r in range(2)])
    a = (tz[:, :m].sum(1) / m - (m + 1) / 2) / n
    cov = np.cov((tz[:, :m] - tx) / n) / m + np.cov(1 - (tz[:, m:] - ty) / m) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    d = a[0] - a[1]
    if not np.isfinite(var) or var <= 0:
        return a[0], a[1], d, np.nan, np.nan, 1.0
    se = np.sqrt(var)
    return a[0], a[1], d, d - 1.96 * se, d + 1.96 * se, float(2 * stats.norm.sf(abs(d) / se))


def threshold_for_sensitivity(y, s, target):
    """The cut-off with the highest specificity among those reaching at least
    `target` sensitivity. Children at or above it are flagged."""
    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    ok = np.where(tpr >= target)[0]
    return float(thr[ok[np.argmin(fpr[ok])]])


def operating_point(y, s, thr):
    """Counts and rates when children with score >= thr are flagged."""
    y = np.asarray(y)
    flag = (np.asarray(s, float) >= thr).astype(int)
    tp = int(((flag == 1) & (y == 1)).sum()); fn = int(((flag == 0) & (y == 1)).sum())
    tn = int(((flag == 0) & (y == 0)).sum()); fp = int(((flag == 1) & (y == 0)).sum())
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, n_flagged=int(flag.sum()),
                sens=sens, spec=spec, balanced_accuracy=(sens + spec) / 2,
                ppv=tp / (tp + fp) if tp + fp else np.nan,
                npv=tn / (tn + fn) if tn + fn else np.nan)


def balanced_accuracy(flag, y):
    return 0.5 * (flag[y == 1].mean() + (1 - flag[y == 0]).mean())


def paired_balanced_accuracy(flags, y, reference, n_boot, seed=SEED):
    """Difference in balanced accuracy between each predictor and `reference`,
    each flagging children at its own frozen cut-off, on the same children.
    `flags` maps name -> 0/1 array. The same resamples are used for every
    predictor. Returns name -> (difference, low, high)."""
    y = np.asarray(y)
    draws = stratified_draws(y, n_boot, seed)
    ref = flags[reference]
    out = {}
    for name, f in flags.items():
        if name == reference:
            continue
        diffs = np.array([balanced_accuracy(f[i], y[i]) - balanced_accuracy(ref[i], y[i]) for i in draws])
        out[name] = (balanced_accuracy(f, y) - balanced_accuracy(ref, y),
                     float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
    return out


def platt(score_dev, y_dev, score_new):
    """Turn a score into a probability with a logistic fit on development data
    only. Children whose score is undefined get the development failure rate."""
    s, y = np.asarray(score_dev, float), np.asarray(y_dev)
    ok = np.isfinite(s)
    lr = LogisticRegression(max_iter=2000).fit(s[ok].reshape(-1, 1), y[ok])
    new = np.asarray(score_new, float)
    out = np.full(len(new), float(y[ok].mean()))
    fin = np.isfinite(new)
    out[fin] = lr.predict_proba(new[fin].reshape(-1, 1))[:, 1]
    return out


def calibration(y, p):
    """Calibration slope (1 = ideal; > 1 = predictions too bunched together) and
    calibration-in-the-large intercept (0 = ideal)."""
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    slope = float(LogisticRegression(penalty=None, max_iter=2000).fit(logit.reshape(-1, 1), y).coef_[0][0])

    def nll(a):
        z = a + logit
        return float(np.sum(np.logaddexp(0, z) - y * z))
    intercept = float(minimize_scalar(nll, bounds=(-10, 10), method='bounded').x)
    return slope, intercept


def net_benefit(y, p, thresholds):
    """Net benefit = TP/N - FP/N * pt/(1-pt), at each threshold probability pt."""
    y, p = np.asarray(y), np.asarray(p, float)
    n = len(y)
    out = np.empty(len(thresholds))
    for j, pt in enumerate(thresholds):
        flag = p >= pt
        out[j] = (flag & (y == 1)).sum() / n - (flag & (y == 0)).sum() / n * pt / (1 - pt)
    return out


def net_benefit_vs_reference(y, probs, reference, thresholds, n_boot, seed=SEED):
    """Bootstrap the paired difference in net benefit between each predictor and
    `reference` across the threshold range. For each predictor: the mean and
    maximum difference, and at how many thresholds the 95% interval lies wholly
    above or below zero."""
    y = np.asarray(y)
    draws = stratified_draws(y, n_boot, seed)
    point = {k: net_benefit(y, v, thresholds) for k, v in probs.items()}
    boot = {k: np.array([net_benefit(y[i], np.asarray(v)[i], thresholds) for i in draws])
            for k, v in probs.items()}
    rows = []
    for k in probs:
        if k == reference:
            continue
        d = boot[k] - boot[reference]
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        diff = point[k] - point[reference]
        rows.append(dict(predictor=k, mean_difference=float(diff.mean()), max_difference=float(diff.max()),
                         n_thresholds=len(thresholds), n_ci_above_zero=int((lo > 0).sum()),
                         n_ci_below_zero=int((hi < 0).sum()),
                         share_of_resamples_ahead=float((d.mean(axis=1) > 0).mean())))
    return rows
