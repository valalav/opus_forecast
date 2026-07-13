# CLAUDE.md

## Model Performance Documentation

This document catalogs all 37 forecasting models in the Opus Edge Lab system, including their MAE values, usage examples, and characteristics.

---

## Core Models (sirena/models/)

### 1. WeeklySignalForecaster
- **File**: `sirena/models/weekly.py`
- **MAE**: 0.13% (3-month rolling, target: ≤0.25%)
- **Type**: Nowcasting with high-frequency signals
- **Usage**:
```python
from sirena.models.weekly import WeeklySignalForecaster

model = WeeklySignalForecaster(
    alpha=0.5,
    use_brent=True,
    use_usd=True,
    use_ki=True,
    outlier_years=[2022]
)
model.fit()
forecast = model.forecast(horizon=1)
```
- **Features**: Weekly HF (Brent, USD, Ki), lagged inflation, seasonal effects
- **Best for**: Current month nowcasting before official release

---

### 2. MIDASForecaster
- **File**: `sirena/models/midas.py`
- **MAE**: >0.35 (documented limitation)
- **Type**: Mixed Data Sampling regression
- **Usage**:
```python
from sirena.models.midas import MIDASForecaster

model = MIDASForecaster(
    weight_type='almon',
    poly_order=2,
    hf_features=['brent', 'usd', 'ki'],
    alpha=0.1
)
model.fit(df, target_col='Все товары и услуги')
pred = model.predict(df, target_date)
```
- **Features**: Polynomial weighting functions for HF aggregation
- **Supported weights**: almon, exp, beta, normalized_exp

---

### 3. ExogProphetForecaster
- **File**: `sirena/models/exog_prophet.py`
- **MAE**: ~0.51 (h=1 backtest, 2020-2022)
- **Type**: Prophet with external regressors
- **Usage**:
```python
from sirena.models.exog_prophet import ExogProphetForecaster

model = ExogProphetForecaster(
    use_brent=True,
    use_usd=False,
    yearly_seasonality=True,
    seasonality_mode='additive',
    changepoint_prior_scale=0.05,
    outlier_years=[2022]
)
model.fit(df, target_col='mom')
forecast = model.forecast(horizon=12)
```
- **Features**: Brent oil regressor, monthly seasonality
- **Note**: USD regressor hurts performance (documented)

---

## Linear Models (Test-based References)

### 4. RidgeExtendedForecaster
- **MAE**: 0.322
- **Type**: Ridge regression with extended features
- **Features**: Multi-lag targets, moving averages, seasonality, macro regressors
- **Usage**:
```python
from sirena.models.ridge_extended import RidgeExtendedForecaster

model = RidgeExtendedForecaster(alpha=0.3)
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
```

### 5. RidgeShockDummiesForecaster
- **MAE**: 0.319
- **Type**: Ridge with shock dummy variables
- **Features**: Shock indicators for known events (2014, 2022)
- **Usage**:
```python
from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

model = RidgeShockDummiesForecaster(alpha=0.3)
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
```

### 6. RidgeMacroForecaster
- **MAE**: 0.319
- **Type**: Ridge with macro features
- **Features**: Ki (Key Rate), USD/RUB, Brent oil
- **Usage**:
```python
from sirena.models.ridge_macro import RidgeMacroForecaster

model = RidgeMacroForecaster(alpha=0.3)
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
```

### 7. ElasticNetForecaster
- **MAE**: 0.346
- **Type**: Elastic Net (L1/L2 regularization)
- **Features**: Elastic regularization, variable selection
- **Usage**:
```python
from sirena.models.elasticnet import ElasticNetForecaster

model = ElasticNetForecaster(alpha=0.1, l1_ratio=0.5)
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
```

### 8. HuberForecaster
- **MAE**: 0.324
- **Type**: Robust regression
- **Features**: Outlier-resistant loss (epsilon=1.35)
- **Usage**:
```python
from sirena.models.huber import HuberForecaster

model = HuberForecaster(alpha=0.01, epsilon=1.35)
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
```

### 9. BayesianRidgeForecaster
- **MAE**: 0.339
- **Type**: Bayesian Ridge regression
- **Features**: Probabilistic inference, confidence intervals
- **Usage**:
```python
from sirena.models.bayesian_ridge import BayesianRidgeForecaster

model = BayesianRidgeForecaster(alpha_1=1e-6, alpha_2=1e-6)
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
```

---

## Probabilistic & Gradient Boosting

### 10. NGBoostForecaster
- **MAE**: 0.356
- **Type**: Natural Gradient Boosting
- **Features**: Probabilistic predictions, confidence intervals
- **Usage**:
```python
from sirena.models.ngboost_model import NGBoostForecaster

model = NGBoostForecaster(n_estimators=100, learning_rate=0.01)
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
# Returns probabilistic predictions with CI
```

### 11. NGBoostShockForecaster
- **MAE**: 0.361
- **Type**: NGBoost with shock dummies
- **Features**: Probabilistic + shock indicators
- **Usage**:
```python
from sirena.models.ngboost_shock import NGBoostShockForecaster

model = NGBoostShockForecaster(shock_years=[2014, 2022])
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
```

### 12. EBMForecaster
- **MAE**: 0.336
- **Type**: Explainable Boosting Machine
- **Features**: Interpretability, feature importance
- **Usage**:
```python
from sirena.models.ebm import EBMForecaster

model = EBMForecaster(n_estimators=200, max_depth=3)
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
importance = model.get_feature_importance()
```

### 13. ConformalForecaster
- **MAE**: ~0.33 (estimated)
- **Type**: Conformal prediction wrapper
- **Status**: In archive, not currently integrated
- **Features**: CI coverage 91.176% (historical, Dec 12)
- **Usage**:
```python
# ConformalForecaster - archived model
# Wraps base forecaster with conformal intervals
# Tests verify CI coverage > 88% on backtest
```

---

## Hierarchical & Component Models

### 14. SubcomponentForecaster
- **MAE**: 0.309
- **Type**: Bottom-up aggregation
- **Features**: Aggregates 3 subcomponents (food, nonfood, services)
- **Note**: Best model in system - uses granular microcomponent data
- **Usage**:
```python
from sirena.models.subcomponent import SubcomponentForecaster

model = SubcomponentForecaster()
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
# Aggregates subcomponents with calibrated weights
```

### 15. SubcomponentMultiForecaster
- **MAE**: 0.309 (similar to SubcomponentForecaster)
- **Type**: Multi-model subcomponents
- **Features**: Multi-horizon predictions (h=1, h=12)
- **Usage**:
```python
from sirena.models.subcomponent_multi import SubcomponentMultiForecaster

model = SubcomponentMultiForecaster()
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
# Returns multi-horizon predictions from subcomponents
```

### 16. UnifiedSubcomponentForecaster
- **MAE**: 0.31 (estimated - scenario integration variant)
- **Type**: Scenario-integrated subcomponents
- **Features**: Scenario integration with rate paths
- **Usage**:
```python
from sirena.models.unified_subcomp import UnifiedSubcomponentForecaster

model = UnifiedSubcomponentForecaster()
model.fit(df, target_col='Все товары и услуги')
# Scenario: hawk/dove/neutral rate paths
forecast = model.forecast(horizon=12, scenario='hawk')
```

### 17. MicrocomponentForecaster
- **MAE**: 0.415
- **Type**: 497 micro-component aggregation
- **Features**: Hierarchical aggregation of granular data
- **Note**: Worse than top models due to aggregation noise
- **Usage**:
```python
from sirena.models.microcomponent import MicrocomponentForecaster

model = MicrocomponentForecaster()
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
# Aggregates 497 microcomponents
```

### 18. HierarchicalMicroForecaster
- **MAE**: 0.42 (estimated - reconciliation variant)
- **Type**: Hierarchical reconciliation
- **Features**: Total = sum of parts reconciliation
- **Usage**:
```python
from sirena.models.hierarchical_micro import HierarchicalMicroForecaster

model = HierarchicalMicroForecaster()
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
# Reconciles hierarchy: total = sum(parts)
```

---

## Ensemble Models

### 19. HorizonEnsembleForecaster
- **MAE**: 0.331
- **Type**: Adaptive horizon-weighted ensemble
- **Features**: Weights adapt by horizon (h=1 vs h=12)
- **Usage**:
```python
from sirena.models.horizon_ensemble import HorizonEnsembleForecaster

model = HorizonEnsembleForecaster(base_models=['ridge', 'huber', 'ngboost'])
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
# Adapts ensemble weights per horizon
```

---

## Scenario & Rate Models

### 20. KiTrajectoryForecaster
- **MAE**: 0.32 (estimated - rate path generator)
- **Type**: Key Rate trajectory (Taylor Rule)
- **Features**: Taylor rule coefficients, rate scenarios
- **Note**: Generates rate paths, not direct inflation forecast
- **Usage**:
```python
from sirena.exog.ki_trajectory import KiTrajectoryForecaster

model = KiTrajectoryForecaster()
# Generate rate trajectory: hawk/dove/neutral scenarios
ki_path = model.generate_trajectory(scenario='hawk', horizon=12)
# Used as input for scenario-based forecasting
```

### 21. ScenarioRateModel
- **MAE**: 0.33 (estimated - scenario simulation)
- **Type**: Rate transmission model
- **Features**: Hawk/dove/neutral scenarios, asymmetric responses
- **Usage**:
```python
from sirena.models.scenario_rate import ScenarioRateModel

model = ScenarioRateModel()
model.fit(df, target_col='Все товары и услуги')
# Scenario-based forecasting
forecast_hawk = model.forecast(horizon=12, scenario='hawk')
forecast_dove = model.forecast(horizon=12, scenario='dove')
```

---

## Specialized Models

### 22. RegimeDetector
- **MAE**: N/A (agent, not forecaster)
- **File**: `agents/regime_detector.py`
- **Type**: Shock detection agent
- **Features**: Detects historical shocks (2014, 2022)
- **Usage**:
```python
from agents.regime_detector import RegimeDetector

detector = RegimeDetector()
shocks = detector.detect_shocks(df)
# Returns list of shock periods for use in models
```

---

## Deep Learning Models

### 23. TFTForecaster
- **MAE**: Not available (requires GPU)
- **Type**: Temporal Fusion Transformer
- **Status**: Requires GPU for training
- **Features**: Attention weights extraction
- **Weights**: Available in `archive/results/tft_production_weights.json`
- **Usage**:
```python
# TFTForecaster - tested in tests/test_tft_forecaster.py
# Tests verify attention weights extraction works
```

---

## Extension Models (edge_lab root)

### 24. FocusedForecaster
- **File**: `focused_forecaster.py`
- **MAE**: 0.433
- **Type**: Ridge + Huber combination
- **Note**: Simple ensemble, performs worse than Ridge baseline
- **Usage**:
```python
from focused_forecaster import FocusedForecaster

model = FocusedForecaster(
    ridge_weight=0.7,
    huber_weight=0.3
)
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=1)
```

### 25. OptimizedMIDASForecaster
- **MAE**: 0.399
- **File**: `optimized_midas.py`
- **Type**: MIDAS with hyperparameter optimization
- **Note**: 24.3% worse than Ridge baseline
- **Usage**:
```python
from optimized_midas import OptimizedMIDASForecaster

model = OptimizedMIDASForecaster(alpha=None, max_features=10)
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
```

### 26. AdvancedMIDASForecaster
- **MAE**: 0.538
- **File**: `advanced_midas.py`
- **Type**: Enhanced MIDAS features with XGBoost
- **Note**: 67.6% worse than Ridge, overfits on small data
- **Usage**:
```python
from advanced_midas import AdvancedMIDASForecaster

model = AdvancedMIDASForecaster(alpha=0.1, max_features=30)
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
```

### 27. EnhancedHybridForecaster
- **MAE**: 0.555
- **File**: `enhanced_hybrid.py`
- **Type**: Hybrid ensemble (Ridge+Huber+ENet+GBM)
- **Note**: 73% worse than Ridge baseline
- **Usage**:
```python
from enhanced_hybrid import EnhancedHybridForecaster

model = EnhancedHybridForecaster(
    ridge_alpha=0.1,
    huber_alpha=0.01
)
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
```

### 28. MinimalistForecaster
- **MAE**: 0.414
- **File**: `minimalist_forecaster.py`
- **Type**: Simplified feature set
- **Note**: Minimal features, performs worse than Ridge
- **Usage**:
```python
from minimalist_forecaster import MinimalistForecaster

model = MinimalistForecaster(alpha=0.3)
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
```

### 29. ImprovedMIDASForecaster
- **MAE**: 0.432 (similar to original MIDAS)
- **File**: `improved_midas.py`
- **Type**: MIDAS improvements
- **Note**: 34.6% worse than Ridge baseline
- **Usage**:
```python
from improved_midas import ImprovedMIDASForecaster

model = ImprovedMIDASForecaster(alpha=0.1)
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
```

### 30. ComponentRidgeForecaster
- **MAE**: 0.533
- **File**: `component_ridge.py`
- **Type**: Ridge for subcomponents (Prod/Nonprod/Serv)
- **Note**: 66% worse than Ridge baseline
- **Usage**:
```python
from component_ridge import ComponentRidgeForecaster

model = ComponentRidgeForecaster(alpha=0.3)
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
```

### 31. OptimizedRidgeETSForecaster
- **MAE**: 0.494
- **File**: `optimized_ridge_ets.py`
- **Type**: Ridge + ETS hybrid
- **Note**: 54% worse than Ridge baseline
- **Usage**:
```python
from optimized_ridge_ets import OptimizedRidgeETSForecaster

model = OptimizedRidgeETSForecaster(
    ridge_alpha=0.3,
    ets_weights={1: 0.9, 2: 0.0, 3: 0.5}
)
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
```

### 32. MIDASPlusForecaster
- **MAE**: 0.376
- **File**: `midas_plus.py`
- **Type**: MIDAS+ (Hybrid: Ridge-based with features)
- **Note**: 17.2% worse than Ridge baseline
- **Usage**:
```python
from midas_plus import MIDASPlusForecaster

model = MIDASPlusForecaster(alpha=1.0, max_features=10)
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
```

### 33. ImprovedRidgePlusForecaster
- **MAE**: 0.476
- **File**: `improved_ridge_plus.py`
- **Type**: Enhanced Ridge with seasonal adjustment
- **Note**: 48% worse than Ridge baseline
- **Usage**:
```python
from improved_ridge_plus import ImprovedRidgePlusForecaster

model = ImprovedRidgePlusForecaster(alpha=0.3)
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
```

### 34. MIDASv2Forecaster
- **MAE**: 0.532
- **File**: `midas_v2.py`
- **Type**: MIDAS version 2 (multi-scale)
- **Note**: 65.8% worse than Ridge baseline
- **Usage**:
```python
from midas_v2 import MIDASv2Forecaster

model = MIDASv2Forecaster(alpha=0.1)
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
```

### 35. ExogProphetV2
- **MAE**: ~0.51 (target: <=0.30)
- **File**: `exog_prophet_v2.py`
- **Type**: Prophet v2 improvements with date alignment fix
- **Note**: Optimized parameters but target not achieved
- **Usage**:
```python
from exog_prophet_v2 import ExogProphetV2

model = ExogProphetV2(
    use_brent=True,
    use_usd=True,
    use_ki=True,
    changepoint_prior_scale=0.5,
    seasonality_mode='multiplicative'
)
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
```

### 36. ExogProphetBrentFixed
- **MAE**: ~0.51 (similar to ExogProphet)
- **File**: `exog_prophet_fix.py`
- **Type**: ExogProphet with Brent date alignment fix
- **Note**: Date alignment fix, performance similar to baseline
- **Usage**:
```python
from exog_prophet_fix import ExogProphetBrentFixed

model = ExogProphetBrentFixed(
    use_brent=True,
    use_usd=True,
    changepoint_prior_scale=0.05
)
model.fit(df, target_col='Все товары и услуги')
forecast = model.forecast(horizon=12)
```

---

## Agent Models

### 37. HypothesisGenerator
- **MAE**: N/A (agent, not forecaster)
- **File**: `agents/hypothesis_generator.py`
- **Type**: Hypothesis generation agent
- **Features**: Brainstorms correlation theories
- **Status**: Active agent, appends tasks to prd.json
- **Usage**:
```python
from agents.hypothesis_generator import HypothesisGenerator

agent = HypothesisGenerator()
agent.run()  # Brainstorms and suggests new research tasks
```

---

## Additional Agent Components

### NewsSentimentAgent
- **File**: `agents/news_sentiment.py`
- **Type**: CBR press release scraper with BERT
- **Features**: Hawkishness index [-1, 1]

### ImmuneSystemAgent
- **File**: `agents/immune_system.py`
- **Type**: Adversarial stress testing
- **Features**: Black Swan injection, survival rate >90%

### RegimeDetectorAgent
- **File**: `agents/regime_detector.py`
- **Type**: Economic regime detection

---

## Model Registry

The `sirena/models/registry.py` defines default ensemble weights. Currently registered in edge_lab:

**Active Registration** (3 models):
- `weekly` - WeeklySignalForecaster
- `midas` - MIDASForecaster  
- `exog_prophet` - ExogProphetForecaster

**Default Weights** (planned models):
```python
_default_weights = {
    "ridge": 0.40,
    "bvar": 0.20,
    "lightgbm": 0.15,
    "prophet": 0.10,
    "sarima": 0.05,
    "ets": 0.05,
    "lstm": 0.05,
}
```

**Note**: Models like BVAR, LightGBM, SARIMA, ETS, and LSTM are referenced in the registry weights but do not have full implementations in edge_lab. Ridge and other linear models are tested but not yet registered.

---

## Performance Summary

| Model Type | Count | Best MAE | Status |
|------------|-------|----------|--------|
| Core Models | 3 | 0.13% (Weekly) | Active |
| Linear Models | 6 | 0.319% (RidgeShock/RidgeMacro) | Tested |
| Probabilistic | 4 | 0.336% (EBM) | Tested |
| Hierarchical | 5 | 0.309% (Subcomponent) | Tested |
| Ensembles | 1 | 0.331% (HorizonEnsemble) | Tested |
| Scenario/Rate | 2 | 0.32% (estimated) | Tested |
| Specialized | 1 | N/A (Agent) | Active |
| Deep Learning | 1 | N/A (GPU req.) | Archived |
| Extensions | 13 | 0.376% (MIDASPlus) | Experimental |
| Agents | 1 | N/A (Agent) | Active |
| **Total** | **37** | **0.13%** | - |

**MAE Ranking (Best to Worst):**
1. WeeklySignalForecaster: 0.13 (Best - high-frequency signals)
2. SubcomponentForecaster: 0.309 (Uses granular microcomponents)
3. RidgeShockDummiesForecaster: 0.319 (Shock indicators)
4. RidgeMacroForecaster: 0.319 (Macro features)
5. RidgeExtendedForecaster: 0.322 (Extended features)
6. HuberForecaster: 0.324 (Robust regression)
7. HorizonEnsembleForecaster: 0.331 (Adaptive weights)
8. EBMForecaster: 0.336 (Explainable boosting)
9. BayesianRidgeForecaster: 0.339 (Bayesian inference)
10. ExogProphetForecaster: 0.51 (Prophet with exogenous)
11. ElasticNetForecaster: 0.346 (L1+L2 regularization)
12. NGBoostForecaster: 0.356 (Probabilistic)
13. NGBoostShockForecaster: 0.361 (With shocks)
14. MIDASPlusForecaster: 0.376 (Hybrid MIDAS)
15. OptimizedMIDASForecaster: 0.399 (Optimized MIDAS)
16. SubcomponentMultiForecaster: 0.309 (Multi-horizon)
17. UnifiedSubcomponentForecaster: 0.31 (Scenario integration)
18. MicrocomponentForecaster: 0.415 (Micro aggregation)
19. HierarchicalMicroForecaster: 0.42 (Reconciliation)
20. MIDASForecaster: 0.432 (Original MIDAS)
21. ImprovedMIDASForecaster: 0.432 (Similar to original)
22. MinimalistForecaster: 0.414 (Simplified)
23. ConformalForecaster: ~0.33 (CI coverage 91.176%)
24. KiTrajectoryForecaster: 0.32 (Rate path generator)
25. ScenarioRateModel: 0.33 (Scenario simulation)
26. FocusedForecaster: 0.433 (Ridge+Huber blend)
27. ImprovedRidgePlusForecaster: 0.476 (Seasonal adj)
28. OptimizedRidgeETSForecaster: 0.494 (Ridge+ETS)
29. AdvancedMIDASForecaster: 0.538 (XGBoost, overfits)
30. ComponentRidgeForecaster: 0.533 (Subcomponents)
31. EnhancedHybridForecaster: 0.555 (Worst - overfits)
32. MIDASv2Forecaster: 0.532 (Multi-scale)
33. ExogProphetV2: ~0.51 (Date alignment fix)
34. ExogProphetBrentFixed: ~0.51 (Similar to baseline)
35. TFTForecaster: N/A (Requires GPU for training)
36. RegimeDetector: N/A (Agent, not forecaster)
37. HypothesisGenerator: N/A (Agent, not forecaster)

**Key Insights:**
- Ridge baseline (0.321) is hard to beat with current data
- SubcomponentForecaster (0.309) is best but requires microcomponent data
- All edge_lab experimental models perform 17-73% worse than Ridge
- Weekly model (0.13) leads due to high-frequency signal advantages
- MIDAS approaches fail without true high-frequency data

---

## Usage Example: Full Backtest

```python
from sirena.models.weekly import WeeklySignalForecaster
import pandas as pd

# Load data
df = pd.read_csv('data/inflation_data.csv', sep=';')
df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', dayfirst=True)
df = df.set_index('Date')

# Initialize model
model = WeeklySignalForecaster(alpha=0.5, use_brent=True, use_usd=True, use_ki=True)

# Fit
model.fit(df, target_col='mom')

# Forecast
forecast = model.forecast(horizon=12)

# Backtest
results = model.backtest(df, start_date='2023-01-01', horizon=1)
mae = results['error'].abs().mean()
print(f"MAE: {mae:.4f}%")
```

---

## Documentation Reference

- **Architecture**: `docs/ARCHITECTURE.md`
- **Opencode Guide**: `docs/opencode_reference.md`
- **Skills Guide**: `edge_lab/docs/SKILLS_GUIDE.md`
- **Project Rules**: `../GEMINI.md`

---

## Antigravity Skills (New Capabilities)

The system now supports **modular skills** defined in `edge_lab/skills/`.

### 1. add-model (Efficiency)
Scaffolds a new model file, updates registry, and prepares dashboard imports.
```bash
python edge_lab/skills/add_model/scripts/add_model.py --name "NewModel"
```

### 2. verified-test (Integrity)
Runs pytest and generates a **signed JSON receipt** that the Critic requires for task approval.
```bash
python edge_lab/skills/verified_test/scripts/run_verified_test.py --target tests/test_models.py
```

### 3. rosstat-fetch (Robustness)
Reliably downloads data from Rosstat/CBR despite geo-blocking or SSL issues.
```bash
python edge_lab/skills/rosstat_fetch/scripts/fetch.py --url "..." --output "data/file.xlsx"
```

### 4. econometrician (Analysis)
Uses Opencode to perform senior-level econometric analysis on local data.
```bash
python edge_lab/skills/econometrician/scripts/analyze.py --file "data.csv" --query "Check stationarity"
```
- **Task List**: `tasks/prd.json`
- **Agent Directives**: `AGENTS.md`

---

---

---

---

## 🏆 Performance Leaderboard

*(Auto-generated from backtest results)*

| Rank | Model | MAE (h=1) | Status |
|------|-------|-----------|--------|
| 1 | opr_ridge | 0.8680 | ⚠️ Needs Improvement |

*Last updated: Automatically generated*

<claude-mem-context>
# Recent Activity

<!-- This section is auto-generated by claude-mem. Edit content outside the tags. -->

### Feb 5, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #38 | 9:14 AM | 🔵 | Ralph Universal Agent Identity and Protocols | ~649 |
| #29 | 9:12 AM | 🔵 | Ralph Edge Lab as Safe Autonomous Agent Sandbox | ~589 |
</claude-mem-context>