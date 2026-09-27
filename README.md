# Predicting high-flow nasal cannula failure in children: ROX, ROX-HR and machine learning

Analysis code for *Accuracy and Generalizability of Tools for Early Identification
of High-Flow Nasal Cannula Failure in Children: A Multi-Centre Study* (Regulski L,
Saffaran S, Yu H, Cooper K, Sharkey D, Siciliano Nascimento M, Etrusco Zaroni
Santos AC, Webb LV, Bates DG; manuscript submitted).

The study asks whether two bedside indices, ROX and ROX-HR, or eight machine
learning models, can identify early which children aged 24 months or younger will
fail high-flow nasal cannula (HFNC) therapy, and whether that holds in a hospital
the tool was not developed in.

**No patient data are in this repository.** See [`data/README.md`](data/README.md).

## The design, in brief

- **Three datasets.** A US paediatric ICU (Dataset 1) and two Brazilian cohorts
  (Datasets 2 and 3); 614 children, 130 failures.
- **Predictors.** Four routinely recorded vital signs — SpO₂, FiO₂, respiratory rate
  and heart rate — at HFNC initiation and at 2 hours, and the change between them:
  twelve predictors. ROX = (SpO₂/FiO₂)/respiratory rate; ROX-HR = ROX/heart rate × 100.
- **Outcome.** Escalation to non-invasive or invasive ventilation at any time after 2 hours.
- **External validation, twice.** Develop on Datasets 1+2 and validate on Dataset 3
  (split A); develop on Datasets 1+3 and validate on Dataset 2 (split B).
- **One threshold rule for every predictor.** The cut-off reaching at least 0.60
  sensitivity on development data with the highest specificity, frozen and applied
  unchanged to the validation data.
- **Models.** L1 and L2 logistic regression, random forest, extremely randomised
  trees, RBF support vector machine, histogram gradient boosting, XGBoost and
  TabPFN, each tuned by exhaustive grid search inside 200 × 5-fold nested
  cross-validation.
- **Comparison with ROX-HR.** Paired difference in balanced accuracy at the frozen
  operating points (primary), and in AUROC (DeLong), on the same children, with
  stratified bootstrap intervals.
- **Also.** Calibration, decision curves, SHAP importance, and the same analysis
  repeated at 6 and 12 hours.

Nothing is chosen after seeing the validation data: every cut-off,
hyperparameter and probability mapping is fitted on development data only.

## Layout

```
src/hfnc_rox/
  config.py           every setting that changes a number
  data.py             the three datasets -> one table per landmark
  models.py           the eight models and their grids
  metrics.py          AUROC, bootstrap intervals, DeLong, thresholds, calibration, net benefit
  analysis.py         one analysis: fit_models() then evaluate()
  shap_importance.py  SHAP importance
  tables.py           the paper's tables
scripts/
  1_build_data.py     data/raw/ -> data/processed/
  2_run_analysis.py   one landmark and split -> results/L02_A/ ...
  3_make_tables.py    results/tables/
  4_make_figures.py   results/figures/
cluster/run_all.sbatch  the six analyses as a SLURM array job
tests/                  statistics on known cases; the whole analysis on synthetic data
```

## Running it

```bash
pip install -r requirements.txt
python scripts/1_build_data.py                               # needs the raw files in data/raw/
python scripts/2_run_analysis.py --landmark 2 --split A --quick   # a smoke test, minutes
sbatch cluster/run_all.sbatch                                # the full analyses
python scripts/3_make_tables.py
python scripts/4_make_figures.py
```

The tests need no data: `pip install pytest && python -m pytest tests/`.

A full analysis fits about 600,000 models; the published analyses each ran as
one cluster job on 28 CPUs and one GPU (TabPFN uses the GPU; XGBoost runs faster
on CPUs for data this small). `--quick` runs the same code with 3 repeats and 200
bootstrap resamples; its numbers are not meaningful.

## Where each result comes from

| in the paper | analysis | produced by |
|---|---|---|
| Table 1, eTable 9 | 2 h, all datasets | `tables.cohort_characteristics` |
| eTable 1 | 2 h, all datasets | `tables.missingness` |
| eFigure 1 (flow) | CONSORT counts | `data/processed/consort.csv` |
| Table 2 | `L02_A`, `L02_B` | `tables.performance` |
| eTable 2 (ΔAUROC, DeLong) | `L02_A`, `L02_B` | `tables.auroc_differences` |
| eTable 3 (operating point transfer) | `L02_A`, `L02_B` | `tables.operating_point_transfer` |
| eTables 4 and 5 | `L06_*`, `L12_*` | `tables.performance` |
| eTable 6, eFigure 2 (decision curves) | `L02_A`, `L02_B` | `tables.decision_curve_summary`, `4_make_figures.py` |
| eTable 7, eFigure 3 (calibration) | `L02_A`, `L02_B` | `tables.calibration`, `4_make_figures.py` |
| eTable 8, Figure 2 (SHAP) | `L02_A`, `L02_B` | `tables.shap_shares`, `4_make_figures.py` |

Figure 1 (study design) is a drawing, not generated from results.

## Details that change numbers

- **SHAP importance is computed on the a-priori configuration of each model**
  (`models.untuned_models`), not on the tuned configuration. Table 2 reports the
  tuned models.
- **Baseline FiO₂** is missing for 62% of Dataset 2 and 15% of Dataset 1 and is
  filled with that dataset's median before modelling, as in the source studies.
  Table 1 includes those filled values.
- **The modelling tables are read back from CSV** before analysis. Re-reading changes
  the last binary digit of some derived values, and histogram gradient boosting
  bins on exact values, so skipping the round trip gives slightly different fits.
- **KernelSHAP** (logistic regression, SVM, TabPFN) samples at random; this code
  seeds it, so its values reproduce exactly from run to run.
- **Library versions** matter in the third decimal place for XGBoost and TabPFN;
  `requirements.txt` pins the versions used.

## How this code was checked against the published analysis

The published results were produced by a longer research pipeline. This
repository is that pipeline reduced to what the paper reports, and was checked
against it before release:

- `1_build_data.py` reproduces the original modelling tables for every landmark
  and split exactly, row order included, and the CONSORT counts.
- Run side by side with the original on the real data, with reduced repeats,
  both produce identical numbers for every model, index, interval, comparison,
  decision curve and tree-model SHAP value (312 comparisons, both splits, 2 and 6 hours).
- `evaluate()` and `tables.py`, fed the original full-scale model scores,
  reproduce every number in Table 2 and eTables 4–5 to the precision at which
  those numbers were stored (1,042 comparisons).

## Data availability

The datasets belong to the three source studies and are available from their
investigators on reasonable request, subject to each institution's approval.

## Licence

MIT — see [`LICENSE`](LICENSE).
