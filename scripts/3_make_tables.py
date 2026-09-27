"""Step 3. The paper's tables, from the results of step 2.

    python scripts/3_make_tables.py        # writes results/tables/

Table 2 is the 2-hour analysis; eTables 4 and 5 are the same table at 6 and 12
hours. Each has a panel A (validation Dataset 3) and B (validation Dataset 2).
"""
import argparse
import os

import pandas as pd

import _setup  # noqa: F401
from hfnc_rox import tables as T

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--results', default=os.path.join(_setup.ROOT, 'results'))
p.add_argument('--data', default=os.path.join(_setup.ROOT, 'data', 'processed'))
a = p.parse_args()
out = os.path.join(a.results, 'tables')
os.makedirs(out, exist_ok=True)

md = []


def write(name, title, df):
    df.to_csv(os.path.join(out, f'{name}.csv'), index=False)
    md.append(f'## {title}\n\n{df.to_markdown(index=False, disable_numparse=True)}\n')


table = pd.read_csv(os.path.join(a.data, 'L02_all.csv'))
write('table1', 'Table 1. Characteristics of the three datasets at 2 hours', T.cohort_characteristics(table))
write('etable1', 'eTable 1. Missing values before imputation', T.missingness(table))
write('consort', 'CONSORT counts (eFigure 1)', pd.read_csv(os.path.join(a.data, 'consort.csv')))

labels = {'A': 'A. Development Datasets 1+2, validation Dataset 3',
          'B': 'B. Development Datasets 1+3, validation Dataset 2'}
for L, name in ((2, 'Table 2'), (6, 'eTable 4'), (12, 'eTable 5')):
    for split in 'AB':
        d = os.path.join(a.results, f'L{L:02d}_{split}')
        if os.path.exists(os.path.join(d, 'results.csv')):
            write(f'{name.replace(" ", "").lower()}_{split}', f'{name} ({L} h), {labels[split]}', T.performance(d))

for split in 'AB':
    d = os.path.join(a.results, f'L02_{split}')
    if not os.path.exists(os.path.join(d, 'results.csv')):
        continue
    write(f'etable2_{split}', f'eTable 2. Paired difference in AUROC against ROX-HR, {labels[split]}', T.auroc_differences(d))
    write(f'etable3_{split}', f'eTable 3. Transfer of the operating point, {labels[split]}', T.operating_point_transfer(d))
    write(f'etable6_{split}', f'eTable 6. Net benefit against ROX-HR, {labels[split]}', T.decision_curve_summary(d))
    write(f'etable7_{split}', f'eTable 7. Brier score and calibration, {labels[split]}', T.calibration(d))
    write(f'etable8_{split}', f'eTable 8. SHAP importance (share of total), {labels[split]}', T.shap_shares(d))

open(os.path.join(out, 'tables.md'), 'w').write('\n'.join(md))
print(f'{len(md)} tables written to {out}')
