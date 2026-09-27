"""Step 2. Run one analysis: one landmark, one development/validation split.

    python scripts/2_run_analysis.py --landmark 2 --split A           # the paper's split A
    python scripts/2_run_analysis.py --landmark 2 --split A --quick   # smoke test, minutes

A full run fits about 600,000 models (200 x 5-fold nested cross-validation with
exhaustive grids) and is meant for a cluster: see cluster/run_all.sbatch.
Results go to results/L{landmark}_{split}/.
"""
import argparse
import os

import pandas as pd

import _setup  # noqa: F401
from hfnc_rox import analysis
from hfnc_rox.config import LANDMARKS, SPLITS, Settings

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--landmark', type=int, default=2, choices=LANDMARKS)
p.add_argument('--split', default='A', choices=list(SPLITS))
p.add_argument('--quick', action='store_true', help='tiny repeats; numbers are not meaningful')
p.add_argument('--models', default='', help='comma-separated subset, e.g. HistGB,RandomForest')
p.add_argument('--jobs', type=int, default=-1)
p.add_argument('--data', default=os.path.join(_setup.ROOT, 'data', 'processed'))
p.add_argument('--out', default='')
a = p.parse_args()

s = Settings.quick() if a.quick else Settings()
s.jobs = a.jobs
tag = f'L{a.landmark:02d}_{a.split}'
# The tables are read back from CSV, as in the original analysis. This matters:
# reading changes the last binary digit of some values, and histogram gradient
# boosting bins on exact values, so it would otherwise give slightly different fits.
dev = pd.read_csv(os.path.join(a.data, f'{tag}_development.csv'))
ext = pd.read_csv(os.path.join(a.data, f'{tag}_validation.csv'))
out = a.out or os.path.join(_setup.ROOT, 'results', tag + ('_quick' if a.quick else ''))
print(f'{tag}: {a.landmark} h landmark, split {a.split} ({"+".join(SPLITS[a.split][0])} -> {SPLITS[a.split][1]})')
analysis.run(dev, ext, s, out, models=[m for m in a.models.split(',') if m])
