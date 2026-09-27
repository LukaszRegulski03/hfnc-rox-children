"""The eight model families and their hyperparameter grids.

Every model is a pipeline: median imputation (fitted on the training fold only),
standardisation for the models that need it, then the classifier. Class weights
are balanced because failure is the minority class.

`tuned_models()` are the models the paper reports (Table 2): each grid is searched
exhaustively. `untuned_models()` are fixed a-priori configurations; the published
SHAP importances (Figure 2B/2D, eTable 8) were computed on these.
"""
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier

from .config import SEED

try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
except ImportError:
    DEVICE = 'cpu'

NAMES = {'LogReg-L2': 'Logistic regression (L2)', 'LogReg-L1': 'Logistic regression (L1)',
         'RandomForest': 'Random forest', 'ExtraTrees': 'Extremely randomised trees',
         'SVM-RBF': 'Support vector machine (RBF)', 'HistGB': 'Histogram gradient boosting',
         'XGBoost': 'XGBoost', 'TabPFN': 'TabPFN'}


def pipeline(clf, scale=True):
    steps = [('imp', SimpleImputer(strategy='median'))]
    if scale:
        steps.append(('sc', StandardScaler()))
    steps.append(('clf', clf))
    return Pipeline(steps)


def tuned_models(pos_weight):
    """name -> (pipeline, grid). `pos_weight` is non-failures per failure in the
    development data; XGBoost searches it alongside fixed ratios."""
    med = [SimpleImputer(strategy='median')]
    return {
        'LogReg-L2': (pipeline(LogisticRegression(penalty='l2', max_iter=4000, class_weight='balanced',
                                                  random_state=SEED)),
                      {'imp': med, 'clf__C': [0.01, 0.1, 1.0, 10.0]}),
        'LogReg-L1': (pipeline(LogisticRegression(penalty='l1', solver='liblinear', max_iter=4000,
                                                  class_weight='balanced', random_state=SEED)),
                      {'imp': med, 'clf__C': [0.01, 0.1, 1.0, 10.0]}),
        'RandomForest': (pipeline(RandomForestClassifier(n_estimators=500, class_weight='balanced_subsample',
                                                         n_jobs=1, random_state=SEED), scale=False),
                         {'imp': med, 'clf__max_depth': [2, 3, 4, 6], 'clf__min_samples_leaf': [5, 10, 20]}),
        'ExtraTrees': (pipeline(ExtraTreesClassifier(n_estimators=500, class_weight='balanced', n_jobs=1,
                                                     random_state=SEED), scale=False),
                       {'imp': med, 'clf__max_depth': [3, 5, 8, None], 'clf__min_samples_leaf': [5, 10, 20]}),
        'SVM-RBF': (pipeline(SVC(kernel='rbf', probability=False, class_weight='balanced', random_state=SEED)),
                    {'imp': med, 'clf__C': [0.1, 1.0, 10.0], 'clf__gamma': ['scale', 0.01, 0.1]}),
        'HistGB': (pipeline(HistGradientBoostingClassifier(max_iter=400, random_state=SEED,
                                                           class_weight='balanced'), scale=False),
                   {'imp': med, 'clf__max_depth': [2, 3, 4], 'clf__learning_rate': [0.03, 0.1],
                    'clf__min_samples_leaf': [10, 20]}),
        'XGBoost': (pipeline(XGBClassifier(n_estimators=400, subsample=0.8, colsample_bytree=0.8,
                                           eval_metric='logloss', verbosity=0, nthread=1, random_state=SEED,
                                           tree_method='hist'), scale=False),
                    {'imp': med, 'clf__max_depth': [2, 3, 4], 'clf__learning_rate': [0.03, 0.1],
                     'clf__min_child_weight': [5, 10], 'clf__reg_lambda': [1.0, 10.0],
                     'clf__scale_pos_weight': [1.0, 2.0, 3.0, 5.0, pos_weight]}),
        'TabPFN': (pipeline(TabPFNClassifier(device=DEVICE, ignore_pretraining_limits=True,
                                             balance_probabilities=True, random_state=SEED), scale=False),
                   {'imp': med, 'clf__n_estimators': [4, 8, 16]}),
    }


def untuned_models(pos_weight):
    """name -> pipeline, fixed a priori. Used for the published SHAP importances."""
    return {
        'LogReg-L2': pipeline(LogisticRegression(penalty='l2', C=1.0, max_iter=4000, class_weight='balanced',
                                                 random_state=SEED)),
        'LogReg-L1': pipeline(LogisticRegression(penalty='l1', C=1.0, solver='liblinear', max_iter=4000,
                                                 class_weight='balanced', random_state=SEED)),
        'RandomForest': pipeline(RandomForestClassifier(n_estimators=400, max_depth=4, min_samples_leaf=10,
                                                        max_features='sqrt', class_weight='balanced_subsample',
                                                        n_jobs=1, random_state=SEED), scale=False),
        'ExtraTrees': pipeline(ExtraTreesClassifier(n_estimators=400, max_depth=5, min_samples_leaf=10,
                                                    max_features='sqrt', class_weight='balanced', n_jobs=1,
                                                    random_state=SEED), scale=False),
        'SVM-RBF': pipeline(SVC(kernel='rbf', C=1.0, gamma='scale', probability=False, class_weight='balanced',
                                random_state=SEED)),
        'HistGB': pipeline(HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400,
                                                          min_samples_leaf=10, l2_regularization=1.0,
                                                          class_weight='balanced', random_state=SEED), scale=False),
        'XGBoost': pipeline(XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
                                          colsample_bytree=0.8, min_child_weight=5, reg_alpha=0.1, reg_lambda=2.0,
                                          scale_pos_weight=pos_weight, eval_metric='logloss', verbosity=0,
                                          nthread=1, random_state=SEED, tree_method='hist'), scale=False),
        'TabPFN': pipeline(TabPFNClassifier(n_estimators=8, device=DEVICE, balance_probabilities=True,
                                            ignore_pretraining_limits=True, random_state=SEED), scale=False),
    }


def score(model, X):
    """Risk score, higher = higher risk. The SVM has no probabilities (its
    internal calibration is unstable on 500 children), so its decision function
    is used; `metrics.platt` turns any score into a probability when needed."""
    if hasattr(model, 'predict_proba'):
        try:
            return model.predict_proba(X)[:, 1]
        except (AttributeError, NotImplementedError):
            pass
    return model.decision_function(X)
