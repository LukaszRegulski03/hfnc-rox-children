"""How much each predictor contributes to each model's predictions (SHAP).

For tree models SHAP values are exact (TreeExplainer, on the imputed inputs).
For the others (logistic regression, SVM, TabPFN) they are estimated by
KernelSHAP on 80 development children against a 20-point background.

Importance = mean |SHAP value| over children; `share` = that importance divided
by the model's total, so models on different scales can be compared.
"""
import numpy as np
import pandas as pd
import shap

from .models import score


def shap_importance(fitted, X, features, s, log=print):
    background = shap.kmeans(X, 20)
    rows = []
    for name, model in fitted.items():
        clf = model.named_steps['clf']
        Xt = X
        for step in ('imp', 'sc'):
            if step in model.named_steps:
                Xt = model.named_steps[step].transform(Xt)
        try:
            values = shap.TreeExplainer(clf).shap_values(Xt)
            if isinstance(values, list):
                values = values[1]
            if getattr(values, 'ndim', 2) == 3:
                values = values[:, :, 1]
        except Exception:
            # KernelSHAP samples coalitions with numpy's global generator; seed it
            # so the values are reproducible
            np.random.seed(s.seed)
            explainer = shap.KernelExplainer(lambda z, m=model: score(m, z), background)
            sub = X[np.random.default_rng(s.seed).choice(len(X), min(80, len(X)), replace=False)]
            values = explainer.shap_values(sub, nsamples=s.shap_kernel_samples, silent=True)
        importance = np.abs(np.asarray(values)).mean(0)
        total = importance.sum() or 1.0
        rows += [dict(model=name, feature=f, mean_abs_shap=float(v), share=float(v / total))
                 for f, v in zip(features, importance)]
        log(f'  SHAP {name}')
    return pd.DataFrame(rows)
