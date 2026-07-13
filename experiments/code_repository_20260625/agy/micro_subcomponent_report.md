# Micro and Subcomponent Candidates: Analysis Report

**Date**: 2026-06-25
**Scope**: Feasibility study of 5 subcomponent forecasting candidates for SIRENA-KBR.

---

## 1. Mordovia Python
*   **Target Files**: `experiments/code_repository_20260625/nested_extracted/a0268154c4591/ВВГУ_Мордовия_КСП инфляции Python/infl.py`
*   **Methods Used**: The script loops over 47 distinct CPI components using `pmdarima.auto_arima`. It splits the variables into two lists: 7 non-seasonal components (modeled with `seasonal=False`) and 40 seasonal components (modeled with `seasonal=True, m=12`). 
*   **Feasibility for SIRENA-KBR**: Highly feasible as a Python baseline. It demonstrates that running parallel SARIMA models on a 40+ component level is computationally manageable.
*   **Blockers**: 
    *   Hardcoded absolute paths (e.g., `/Users/yuliya_egortseva/...`).
    *   Missing aggregation: The script outputs an excel file of raw component forecasts but completely lacks the weighting logic required to reconstruct the headline CPI.

## 2. Omsk 45 Components
*   **Target Files**: `experiments/code_repository_20260625/nested_extracted/afa0cc7c1c88b/ARIMA-45 (Омск, СГУ)/Скрипт/arima_omsk.prg`
*   **Methods Used**: Explicitly breaks down inflation into 45 specific items/categories (15 food, 20 non-food, 10 services). It applies classical `stl` decomposition, models the seasonally adjusted (SA) series with `autoarma`, adds back the seasonal factors, calculates rolling YoY MoM chains, and recombines the 45 forecasts using actual weight vectors (`_vklad`).
*   **Feasibility for SIRENA-KBR**: Granularity is ideal. 45 items capture structural shifts much better than 3 high-level subcomponents without being as noisy as a full 500-item Rosstat basket. The weighted reconstruction logic is sound.
*   **Blockers**: Written in EViews (`.prg`). Needs full translation into Python (using `statsmodels.tsa.seasonal.STL` and `pmdarima`).

## 3. Khabarovsk Components
*   **Target Files**: `experiments/code_repository_20260625/nested_extracted/a2363877985ef/Прогноз_ИПЦ_по компонентам_Хабаровск/khab_mod.prg`
*   **Methods Used**: Conceptually identical to the Omsk approach (same 45 components and weighting arithmetic). However, the code architecture is significantly better: it relies on a standardized `sa_arima` subroutine and utilizes X-13 ARIMA-SEATS (`tramo/seats`) for seasonal adjustment instead of simple STL. 
*   **Feasibility for SIRENA-KBR**: This is the best methodological template among the EViews scripts. SIRENA-KBR already supports JDemetra+ and X-13 (via `import/x13.py`), making a direct Python port highly synergistic.
*   **Blockers**: EViews `.prg` format. Python pipeline needs to wrap the X-13 SA logic together with a parallelized ARIMA.

## 4. Magadan Components
*   **Target Files**: `experiments/code_repository_20260625/nested_extracted/a8dceef0607a0/ИПЦ_Магадан/Magadan_sripts_cpi.prg`
*   **Methods Used**: Models only the 3 top-level subcomponents (food, non-food, services). After auto-ARMA and X-13, the script relies heavily on hardcoded expert multipliers (e.g., `* 0.8` or `* 0.9`) to force the forecasted indices to align with Central Bank inflation targets (e.g., forcing a 4% trajectory).
*   **Feasibility for SIRENA-KBR**: None. It is not a micro/subcomponent forecast. It demonstrates "policy-forcing" rather than objective statistical forecasting.
*   **Blockers**: Hardcoded manual adjustments; logically incompatible with SIRENA-KBR's objective ensemble design.

## 5. DGU Component ARMAX (ЮГУ)
*   **Target Files**: `experiments/code_repository_20260625/nested_extracted/ae00677697d94/ARIMAX.R`
*   **Methods Used**: A textbook R script utilizing the `forecast::auto.arima` package. It models the top-level index (`IPC_mom`) using 6 completely unmapped external regressors (`xreg = y[,2:7]`). 
*   **Feasibility for SIRENA-KBR**: Irrelevant for this task. It evaluates a single headline series and contains zero microcomponent logic.
*   **Blockers**: R script format; no component breakdown; unidentified external variables.

---

### Conclusion & Recommendation
For implementing a new microcomponent model in SIRENA-KBR, the **Khabarovsk / Omsk 45-component** taxonomy is recommended. The development path should involve translating Khabarovsk's `sa_arima` subroutine into Python, using `pmdarima` for auto-ARMA and `statsmodels` (or `JDemetra+` via project tools) for seasonal adjustment, followed by an explicit weighting reconstruction step.
