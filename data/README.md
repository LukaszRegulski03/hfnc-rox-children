# Data

**No data are in this repository.** The three datasets belong to the studies
that collected them and are available from their investigators on reasonable
request (see the paper's data availability statement).

| | source | children | failures | role |
|---|---|---|---|---|
| Dataset 1 | US paediatric ICU (Webb et al., *Respir Care* 2022) | 446 | 111 | development in both splits |
| Dataset 2 | Brazilian observational cohort (Nascimento et al., *Sci Rep* 2024) | 102 | 18 | development (split A) / validation (split B) |
| Dataset 3 | HFNC arm of a Brazilian randomised trial (Santos et al., *BMC Pediatr* 2024) | 126 | 29 | validation (split A) / development (split B) |

Counts are before the 2-hour landmark; 614 children (130 failures) are analysed.

## What `scripts/1_build_data.py` expects in `data/raw/`

| file | contents |
|---|---|
| `ROX Data ReOrganized 2 (1).xlsx` | Dataset 1: a `Patient Summary` sheet (`Encounter Number`, `HFNC Failure (N=0, Y=1)`) and one sheet per time point (`0hr`, `1hr`, `2hr`, `4hr`, `6hr`, `8hr`, `10hr`, `12hr`, `18hr`, `24hr`) with `Encounter Number`, `O2 Sat`, `FiO2`, `RR`, `HR` |
| `ROX Data ReOrganized 2 - Sheet 1.csv` | Dataset 1: `Encounter Number`, `HFNC duration (HRs)` |
| `WEBB_with_age.csv` | Dataset 1: `Encounter Number`, `Age (months)` |
| `BD_IROX.xlsx` | Dataset 2, one row per child: `desfecho` (0 = failure), `tempo_caf` (hours on HFNC), `idade` (months), and `sat_ox`, `fc_ox_insp`, `freq_resp`, `freq_card` with suffixes `''`, `_2h`, `_6h`, `_12h`, `_18h`, `_24h` |
| `SANTOS__extenral_validation_data.csv` | Dataset 3, one row per child: `Plot_ID`, `It` (`y` = failure), `TD` (hours on HFNC), `Ag` (age band, months), and `HR_t`, `RR_t`, `SpO2_t`, `FiO2_t` (%) for t = 0, 2, 6, 12, 24 |

## Coding that matters

- Dataset 1 codes `999` as missing and `888` as "no longer on HFNC". `888` occurs
  only in wide ROX columns that are not read; in the per-time-point sheets a
  child off HFNC simply has no row.
- Dataset 2's outcome is inverted: `desfecho` 0 means failure.
- FiO₂ is converted to a fraction (0.21–1.0) wherever it was recorded as a percentage.
- Baseline FiO₂ is missing for 62% of Dataset 2 (the oxygen device used before
  HFNC did not record it) and 15% of Dataset 1. It is filled with that dataset's
  own median before modelling, as in the source studies, and flagged in
  `fio2_0_imputed`. All other missing values are imputed inside each training fold.
- Dataset 3 records age only as a band; the band midpoint is used. Age is not a predictor.
