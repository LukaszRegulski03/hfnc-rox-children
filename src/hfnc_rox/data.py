"""From the three source datasets to one modelling table per landmark.

The raw files are not public (see data/README.md). Each loader returns a "long"
table: one row per child per time point, with the four vital signs, the outcome
and the time on HFNC.

    long = load_cohort('D1', raw_dir)
    table, flow = landmark_table(long, 'D1', landmark=2)

A child enters the landmark table if they were still on HFNC after the landmark
and had SpO2 and respiratory rate recorded at it. The outcome is HFNC failure
(escalation to non-invasive or invasive ventilation) at any later time.
"""
import os

import numpy as np
import pandas as pd

from .config import VITALS

# Source files, as supplied by each study team
RAW_FILES = {
    'D1': dict(vitals='ROX Data ReOrganized 2 (1).xlsx',
               duration='ROX Data ReOrganized 2 - Sheet 1.csv',
               age='WEBB_with_age.csv'),
    'D2': dict(workbook='BD_IROX.xlsx'),
    'D3': dict(table='SANTOS__extenral_validation_data.csv'),
}
# Dataset 1 codes 999 as missing and 888 as "no longer on HFNC". 888 occurs only
# in wide ROX columns that are not read here; the per-time-point sheets encode
# "off HFNC" by leaving the row out.
SENTINEL = [999, 888]


def _fio2_as_fraction(df):
    df.loc[df['FiO2'] > 1.05, 'FiO2'] /= 100.0
    return df


def _load_d1(raw):
    """Dataset 1 (US PICU): one sheet per time point."""
    f = RAW_FILES['D1']
    xl = os.path.join(raw, f['vitals'])
    summary = (pd.read_excel(xl, sheet_name='Patient Summary')
               [['Encounter Number', 'HFNC Failure (N=0, Y=1)']]
               .rename(columns={'Encounter Number': 'patient_id',
                                'HFNC Failure (N=0, Y=1)': 'failure'})
               .replace(SENTINEL, np.nan))
    dur = (pd.read_csv(os.path.join(raw, f['duration']))
           [['Encounter Number', 'HFNC duration (HRs)']]
           .rename(columns={'Encounter Number': 'patient_id',
                            'HFNC duration (HRs)': 'hfnc_duration'}))
    dur = dur.apply(pd.to_numeric, errors='coerce').replace(SENTINEL, np.nan)
    summary = summary.merge(dur, on='patient_id', how='left')

    sheets = {'0hr': 0, '1hr': 1, '2hr': 2, '4hr': 4, '6hr': 6, '8hr': 8,
              '10hr': 10, '12hr': 12, '18hr': 18, '24hr': 24}
    frames = []
    for sheet, t in sheets.items():
        d = (pd.read_excel(xl, sheet_name=sheet)
             .rename(columns={'Encounter Number': 'patient_id', 'O2 Sat': 'SpO2'}))
        d['time_hr'] = t
        frames.append(d[[c for c in ['patient_id', 'time_hr'] + VITALS if c in d.columns]])
    long = pd.concat(frames, ignore_index=True).replace(SENTINEL, np.nan)
    for v in VITALS:
        long[v] = pd.to_numeric(long[v], errors='coerce')
    long = _fio2_as_fraction(long)
    long = long[~long.duplicated(['patient_id', 'time_hr'], keep='first')].copy()
    long = long.merge(summary, on='patient_id', how='left')
    long['uid'] = 'D1_' + long['patient_id'].astype(str)

    age = pd.read_csv(os.path.join(raw, f['age']), usecols=['Encounter Number', 'Age (months)'])
    age = age.dropna(subset=['Encounter Number'])
    age['uid'] = 'D1_' + pd.to_numeric(age['Encounter Number'], errors='coerce').astype(int).astype(str)
    age['age_months'] = pd.to_numeric(age['Age (months)'], errors='coerce')
    return long.merge(age[['uid', 'age_months']], on='uid', how='left')


def _load_d2(raw):
    """Dataset 2 (Brazilian observational cohort): one wide row per child.
    The outcome is inverted in the source: desfecho 0 = failure."""
    src = pd.read_excel(os.path.join(raw, RAW_FILES['D2']['workbook']))
    src = src.assign(failure=(src['desfecho'] == 0).astype(int),
                     hfnc_duration=pd.to_numeric(src['tempo_caf'], errors='coerce'),
                     patient_id=range(len(src)),
                     age_months=pd.to_numeric(src['idade'], errors='coerce'))
    suffix = {0: '', 2: '_2h', 6: '_6h', 12: '_12h', 18: '_18h', 24: '_24h'}
    names = dict(SpO2='sat_ox', FiO2='fc_ox_insp', RR='freq_resp', HR='freq_card')
    frames = []
    for t, sfx in suffix.items():
        row = pd.DataFrame({'patient_id': src.patient_id.values, 'time_hr': t,
                            'failure': src.failure.values,
                            'hfnc_duration': src.hfnc_duration.values,
                            'age_months': src.age_months.values})
        for v, col in names.items():
            col = col + sfx
            row[v] = pd.to_numeric(src[col], errors='coerce').values if col in src.columns else np.nan
        frames.append(row)
    long = pd.concat(frames, ignore_index=True).replace(SENTINEL, np.nan)
    long = _fio2_as_fraction(long)
    long['uid'] = 'D2_' + long['patient_id'].astype(str)
    return long


def _load_d3(raw):
    """Dataset 3 (HFNC arm of a Brazilian RCT). Age is recorded only as a band
    ('0-5', '6-10', ...) and is converted to the band's midpoint."""
    src = pd.read_csv(os.path.join(raw, RAW_FILES['D3']['table']))
    src['failure'] = (src['It'] == 'y').astype(int)
    src['hfnc_duration'] = pd.to_numeric(src['TD'], errors='coerce')

    def midpoint(band):
        parts = str(band).split('-')
        return (float(parts[0]) + float(parts[1])) / 2 if len(parts) == 2 and not pd.isna(band) else np.nan
    src['age_months'] = src['Ag'].apply(midpoint)
    frames = []
    for t in [0, 2, 6, 12, 24]:
        row = pd.DataFrame({v: pd.to_numeric(src.get(f'{v}_{t}', np.nan), errors='coerce')
                            for v in ['HR', 'RR', 'SpO2', 'FiO2']})
        row['patient_id'] = src['Plot_ID'].values
        row['time_hr'] = t
        row['failure'] = src['failure'].values
        row['hfnc_duration'] = src['hfnc_duration'].values
        row['age_months'] = src['age_months'].values
        frames.append(row)
    long = pd.concat(frames, ignore_index=True)
    long['FiO2'] /= 100.0                        # recorded as a percentage
    long['uid'] = 'D3_' + long['patient_id'].astype(str)
    return long


LOADERS = {'D1': _load_d1, 'D2': _load_d2, 'D3': _load_d3}


def load_cohort(name, raw_dir):
    return LOADERS[name](raw_dir)


def landmark_table(long, name, landmark):
    """One row per eligible child at `landmark` hours, plus the CONSORT counts.

    Baseline FiO2 is missing for 62% of Dataset 2 (the oxygen device used before
    HFNC did not record it) and 15% of Dataset 1. As in the source studies it is
    filled with that dataset's own median here, before any modelling, and flagged
    in `fio2_0_imputed`. Every other missing value is imputed later, inside each
    training fold.
    """
    first = long.drop_duplicates('uid').set_index('uid')
    n_eligible, f_eligible = len(first), int(first['failure'].sum())

    base = (long[long.time_hr == 0].drop_duplicates('uid').set_index('uid')[VITALS]
            .rename(columns={v: f'{v}_0' for v in VITALS}))
    base['fio2_0_imputed'] = base['FiO2_0'].isna().astype(int)
    base['FiO2_0'] = base['FiO2_0'].fillna(base['FiO2_0'].median())

    at = long[long.time_hr == landmark].drop_duplicates('uid').set_index('uid')
    # children with no row at the landmark: off HFNC by then, or no record
    absent = first.index.difference(at.index)
    off_before = int((first.loc[absent, 'hfnc_duration'] <= landmark).sum())
    n_at = len(at)
    at = at[at.hfnc_duration > landmark]
    n_still_on = len(at)

    df = at.rename(columns={v: f'{v}_T' for v in VITALS}).join(base, how='inner')
    n_with_baseline = len(df)
    for v in VITALS:
        df[f'd_{v}'] = df[f'{v}_T'] - df[f'{v}_0']
    df = df.dropna(subset=['SpO2_T', 'RR_T'])       # both core landmark measurements
    df = df.reset_index()
    df['dataset'] = name
    df['failure'] = df['failure'].astype(int)
    cols = (['uid', 'dataset', 'failure', 'hfnc_duration', 'age_months']
            + [f'{v}_T' for v in VITALS] + [f'{v}_0' for v in VITALS]
            + [f'd_{v}' for v in VITALS] + ['fio2_0_imputed'])
    df = df[cols]

    flow = dict(dataset=name, landmark_h=landmark, n_eligible=n_eligible, n_fail_eligible=f_eligible,
                excluded_no_landmark_record=len(absent) - off_before,
                excluded_off_hfnc_before_landmark=off_before + (n_at - n_still_on),
                excluded_no_baseline_record=n_still_on - n_with_baseline,
                excluded_missing_core_measurements=n_with_baseline - len(df),
                n_analysed=len(df), n_fail_analysed=int(df.failure.sum()))
    return df, flow


def build_all(raw_dir, landmarks):
    """Landmark tables for every dataset, and the CONSORT flow."""
    longs = {n: load_cohort(n, raw_dir) for n in LOADERS}
    tables, flows = {}, []
    for L in landmarks:
        parts = []
        for n in LOADERS:
            t, f = landmark_table(longs[n], n, L)
            parts.append(t)
            flows.append(f)
        tables[L] = pd.concat(parts, ignore_index=True)
    return tables, pd.DataFrame(flows)


def split(table, dev_sets, ext_set):
    """Development and validation rows, keeping the datasets in D1, D2, D3 order."""
    dev = table[table.dataset.isin(dev_sets)].reset_index(drop=True)
    ext = table[table.dataset == ext_set].reset_index(drop=True)
    return dev, ext
