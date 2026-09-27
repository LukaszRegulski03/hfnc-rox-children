"""Every setting that changes a number, in one place.

A run writes these to `config.json` next to its results, so two runs can be
compared with `diff`.
"""
from dataclasses import dataclass, asdict

SEED = 42

# The four vital signs. Each is used at treatment initiation (_0), at the
# landmark (_T) and as the change between them (d_), giving twelve predictors.
VITALS = ['SpO2', 'FiO2', 'RR', 'HR']
FEATURES = [f'{v}_T' for v in VITALS] + [f'{v}_0' for v in VITALS] + [f'd_{v}' for v in VITALS]

# Development cohorts and the held-out validation cohort for each split.
# Dataset 1: US paediatric ICU (Webb et al. 2022)
# Dataset 2: Brazilian observational cohort (Nascimento et al. 2024)
# Dataset 3: HFNC arm of a Brazilian randomised trial (Santos et al. 2024)
SPLITS = {'A': (['D1', 'D2'], 'D3'),
          'B': (['D1', 'D3'], 'D2')}
LANDMARKS = (2, 6, 12)     # hours after HFNC initiation; 2 h is the primary analysis

# Risk range over which escalating an infant is a defensible decision
DCA_RANGE = (0.02, 0.60)


@dataclass
class Settings:
    cv_folds: int = 5            # outer and repeated cross-validation
    cv_repeats: int = 200        # repeats of 5-fold CV (nested CV and out-of-fold scores)
    inner_folds: int = 5         # grid search inside each outer training fold
    n_boot: int = 10_000         # bootstrap resamples for every interval
    target_sensitivity: float = 0.60   # threshold rule, applied to every predictor
    tuning_metric: str = 'roc_auc'
    shap_kernel_samples: int = 1000    # KernelSHAP samples for non-tree models
    jobs: int = -1               # parallel grid-search workers
    seed: int = SEED

    @classmethod
    def quick(cls):
        """For smoke tests: the same code path with tiny repeats. Numbers are not
        meaningful."""
        return cls(cv_repeats=3, n_boot=200, shap_kernel_samples=50)

    def as_dict(self):
        return asdict(self)
