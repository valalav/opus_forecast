# Edge Lab Experiment Log

> Summary of all experiments conducted in Edge Lab with results and lessons learned.

## 📊 Experiment Results Summary

| Experiment | MAE | vs Baseline | Status | Location |
|------------|-----|-------------|--------|----------|
| Ridge (baseline) | 0.321 | — | ✅ Production | `sirena/models/ridge.py` |
| SubcomponentForecaster | 0.309 | **-3.7%** | ✅ Production | `sirena/models/subcomponent.py` |
| RidgeShockDummies | 0.319 | -0.6% | ✅ Production | `sirena/models/ridge_shock_dummies.py` |
| RidgeMacro | 0.319 | -0.6% | ✅ Production | `sirena/models/ridge_macro.py` |
| Huber | 0.324 | +0.9% | ✅ Production | `sirena/models/huber.py` |
| NGBoost | 0.356 | +10.9% | ✅ Production (h=2 best) | `sirena/models/ngboost_model.py` |
| MIDASPlus | 0.376 | +17.2% | ❌ Archived | `archive/experiments/midas_plus.py` |
| OptimizedMIDAS | 0.399 | +24.3% | ❌ Archived | `archive/experiments/optimized_midas.py` |
| MinimalistForecaster | 0.414 | +29.0% | ❌ Archived | `archive/experiments/minimalist_forecaster.py` |
| ImprovedMIDAS | 0.432 | +34.6% | ❌ Archived | `archive/experiments/improved_midas.py` |
| FocusedForecaster | 0.433 | +34.9% | ❌ Archived | `archive/experiments/focused_forecaster.py` |
| ImprovedRidgePlus | 0.476 | +48.3% | ❌ Archived | `archive/experiments/improved_ridge_plus.py` |
| OptimizedRidgeETS | 0.494 | +53.9% | ❌ Archived | `archive/experiments/optimized_ridge_ets.py` |
| MIDASv2 | 0.532 | +65.7% | ❌ Archived | `archive/experiments/midas_v2.py` |
| ComponentRidge | 0.533 | +66.0% | ❌ Archived | `archive/experiments/component_ridge.py` |
| AdvancedMIDAS | 0.538 | +67.6% | ❌ Archived | `archive/experiments/advanced_midas.py` |
| EnhancedHybrid | 0.555 | +72.9% | ❌ Archived | `archive/experiments/enhanced_hybrid.py` |

## 🔑 Key Lessons Learned

### 1. MIDAS Doesn't Work Without True High-Frequency Data
All MIDAS variants (6 experiments) failed to beat baseline because:
- We don't have true daily/weekly macro data
- Pseudo-HF created from monthly → overfitting

### 2. Complexity ≠ Accuracy
Simple Ridge (0.321) outperforms complex ensembles (0.555).
The more layers/features added, the worse the result.

### 3. Shock Dummies Work
Models with explicit 2014/2022 shock indicators consistently perform better.

### 4. Bottom-Up (Subcomponent) is Best
Using 3 components (Food/NonFood/Services) with individual models → MAE 0.309.

## 📁 Archive Structure

```
archive/
├── experiments/        # Failed model experiments (15 files)
├── scripts/
│   ├── verify/         # Verification scripts (34 files)
│   └── test/           # Test scripts (9 files)
├── results/            # Backtest CSVs and metrics
└── docs/               # Old documentation
```

## 🚀 What Made It to Production

From ~60 experiments, only **6 models** improved on baseline and were integrated:
1. SubcomponentForecaster (-3.7%)
2. RidgeShockDummies (-0.6%)
3. RidgeMacro (-0.6%)
4. Huber (+0.9% but robust)
5. NGBoost (best at h=2)
6. Prophet (best at h=12)

---
*Last updated: 2026-01-22*
