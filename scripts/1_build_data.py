"""Step 1. Build the modelling tables from the three source datasets.

    python scripts/1_build_data.py            # reads data/raw/, writes data/processed/

Writes, for each landmark (2, 6, 12 h) and split (A, B), a development and a
validation table, plus the 2-hour table for all three datasets (for Table 1)
and the CONSORT counts.
"""
import argparse
import os

import _setup  # noqa: F401
from hfnc_rox import config, data

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--raw', default=os.path.join(_setup.ROOT, 'data', 'raw'))
p.add_argument('--out', default=os.path.join(_setup.ROOT, 'data', 'processed'))
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

tables, flow = data.build_all(a.raw, config.LANDMARKS)
flow.to_csv(os.path.join(a.out, 'consort.csv'), index=False)
tables[2].to_csv(os.path.join(a.out, 'L02_all.csv'), index=False)
for L, table in tables.items():
    for name, (dev_sets, ext_set) in config.SPLITS.items():
        dev, ext = data.split(table, dev_sets, ext_set)
        dev.to_csv(os.path.join(a.out, f'L{L:02d}_{name}_development.csv'), index=False)
        ext.to_csv(os.path.join(a.out, f'L{L:02d}_{name}_validation.csv'), index=False)
        print(f'{L:>2} h, split {name}: development n={len(dev)} ({dev.failure.sum()} failures), '
              f'validation n={len(ext)} ({ext.failure.sum()} failures)')
print(f'written to {a.out}')
