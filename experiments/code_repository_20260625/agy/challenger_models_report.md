# Challenger Models & External Methods Report

**Date:** 2026-06-25
**Scope:** Evaluation of external methods listed in `STUDY_FILES.md` (Variable selection, ARDL, ARIMAX, ML challengers, PCA/STL seasonality) for integration into the SIRENA-KBR system.

## 1. Seasonality & STL Smoothing
**Analyzed Files:**
- `experiments/code_repository_20260625/nested_extracted/aa972a9a24e63/Сглаживание ИПЦ/main.R`

**Methodology:**
An R script utilizing the `seasonal` package to apply X-13ARIMA-SEATS on disaggregated CPI subcomponents. Contains hardcoded fallback to `x11` for certain groups (e.g., "Масло и жиры").

**Immediate Usefulness: NONE**
SIRENA-KBR already possesses a mature, Python-native seasonal adjustment pipeline. According to `import/SEASONAL_ADJUSTMENT_GUIDE.md`, the repository uses `import/x13.py` (with SHA256 caching) and `import/jdemetra.py` (JDemetra+ TRAMO-SEATS wrapper). Introducing an R-based X-13 pipeline is completely redundant.

**What to Port / Ignore:**
- **Ignore entirely.** The current Python architecture is superior for production.

**Leakage & Backtest Risks:**
- **HIGH.** The R script runs `seas(time_series)` on the full length of the available data (`start` to `end`). In a backtest framework, performing seasonal adjustment on the full dataset causes severe lookahead bias, as future observations influence historical seasonal factors and outlier detection. SIRENA requires recursive, vintage-based SA.

---

## 2. ML Challengers
**Analyzed Files:**
- `experiments/code_repository_20260625/nested_extracted/a9c66f72e388c/Комбинированный прогноз ИПЦ моделей машинного обучения/forecasts/XGB.Rmd` (and other `.Rmd` files for Ensembles, Neural Networks)
- `methodology.Rmd`, `Описание.txt`

**Methodology:**
Evaluates ML algorithms (XGBoost, Random Forest, SVM, kNN, Neural Networks) over 3, 9, 12, and 24-month horizons using R's `tidymodels`. Performs hyperparameter tuning using a 100-point Latin hypercube grid on a rolling origin cross-validation setup (initial = 96 months, assess = 24 months). Features include `ipc`, `agro_price`, `deposit`, `demand`, `usd_cb`.

**Immediate Usefulness: MEDIUM**
SIRENA-KBR's production ensemble already contains tree-based ML methods (`ngboost_model.py`, `ebm.py`). While the R pipeline itself is not directly usable, standard XGBoost and Random Forest models are solid additions that can be natively implemented in Python.

**What to Port / Ignore:**
- **Port:** The XGBoost model architecture should be recreated natively in Python (e.g., `sirena/models/xgboost_model.py`) matching the `base.py` contract.
- **Port:** The feature variables explored (`agro_price`, `deposit`, `demand`) should be verified against the current `sirena/macro_features.py`. If absent, they should be added to the Python data loaders.
- **Ignore:** The R `tidymodels` code and the exhaustive grid-search per step. 

**Leakage & Backtest Risks:**
- **MEDIUM.** The script uses `rolling_origin` which is theoretically safe. However, in a SIRENA Python port, performing heavy hyperparameter grid searches dynamically during every step of the backtest is computationally expensive and risky. If ported, parameters should be pre-tuned or tuned strictly inside the `train` fold of each temporal slice to avoid data leakage.

---

## 3. Variable Selection, ARDL, ARIMAX
**Analyzed Files:**
- `experiments/code_repository_20260625/nested_extracted/ae00677697d94/ARIMAX.R`
- `experiments/code_repository_20260625/nested_extracted/a0ac088b0b1be/Методика отбора переменных, обеспечивающих надежные прогнозы/variable_models_code.prg`

**Methodology:**
The EViews script (`variable_models_code.prg`) runs an out-of-sample iterative loop testing lags for variable selection based on lowest RMSE and Outperform Ratio against an AR benchmark. The R script (`ARIMAX.R`) uses `auto.arima` with specified `xreg` external regressors to generate forecasts.

**Immediate Usefulness: LOW**
SIRENA-KBR rules (`docs/VAR_MODEL_RESEARCH.md`) specifically prohibit "naive grid-search alone" for VAR/ARIMA family models. The system relies on robust, theory-driven, parsimonious variable subsets rather than brute-force automated ARIMA selections. 

**What to Port / Ignore:**
- **Ignore:** Both the EViews selection script and the R `auto.arima` script. 
- **Port:** The conceptual metric of "Outperform Ratio" against an AR(1) baseline could be integrated into SIRENA's backtesting reports (`backtest_h1_metrics.csv`), if not already present.

**Leakage & Backtest Risks:**
- **HIGH.** The R script (`ARIMAX.R`) directly loads `Прогноз регрессоров.xlsx` (exogenous forecasts) and feeds them into `forecast(..., xreg=x)`. Unless these external variables are strictly deterministic (e.g., known tariff schedules), using externally provided future paths in backtesting creates massive lookahead bias. SIRENA rules explicitly state: "never use future actual USD/RUONIA/Ki values" for exogenous paths.
