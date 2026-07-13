# Model Maintenance Schedule

> **Version**: 1.0 | **Updated**: 2026-01-24

This document defines the maintenance procedures for SIRENA-KBR forecasting models, including retraining schedules and model retirement criteria.

---

## Table of Contents

1. [Overview](#overview)
2. [Retraining Schedule](#retraining-schedule)
3. [Model Retirement Criteria](#model-retirement-criteria)
4. [Performance Monitoring](#performance-monitoring)
5. [Automated Infrastructure](#automated-infrastructure)
6. [Emergency Procedures](#emergency-procedures)

---

## Overview

The SIRENA-KBR forecasting system uses an ensemble of models that must be periodically retrained with new data. This document establishes:

- **Standard retraining frequency** for all production models
- **Performance thresholds** for model retirement
- **Monitoring procedures** to detect degradation early
- **Automated infrastructure** for hands-off maintenance

### Key Principles

1. **Data Freshness**: Models must be retrained when new inflation data becomes available
2. **Performance Stability**: Models showing consistent degradation must be retired or investigated
3. **Graceful Degradation**: Ensemble automatically adjusts weights based on recent performance
4. **Audit Trail**: All maintenance activities are logged for traceability

---

## Retraining Schedule

### Standard Retraining Frequency

**Frequency**: Monthly

**Trigger**: First day of each month at 02:00 AM (server time)

**Rationale**:
- Rosstat releases monthly inflation data on the 1st-3rd of each month
- Retraining on the 1st ensures models are trained on the latest available data
- Monthly frequency balances computational cost with data freshness

### Retraining Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Monthly Retraining Pipeline                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                               │
│  1. [02:00] Load latest data from data/infl_kbr.csv          │
│     - Verify data integrity (no NaN, correct date range)          │
│                                                               │
│  2. [02:05] Load previous retrain log                         │
│     - Compare new vs previous MAE                                 │
│                                                               │
│  3. [02:10] Retrain all production models                      │
│     - Ridge, Huber, NGBoost, EBM, Subcomp, etc.                │
│     - Use MIN_TRAIN_SIZE = 24 observations                        │
│                                                               │
│  4. [02:45] Calculate MAE on last 12 months                    │
│     - Rolling window validation                                    │
│                                                               │
│  5. [02:50] Save model weights                                 │
│     - archive/weights/{model}_{timestamp}.pkl                      │
│     - Create symlink: {model}_latest.pkl                          │
│                                                               │
│  6. [02:55] Update ensemble weights                             │
│     - Recalculate based on recent MAE performance                   │
│                                                               │
│  7. [03:00] Log results and generate report                    │
│     - archive/weights/retrain_log.json                            │
│     - Send telegram alert if MAE degraded by >20%                │
│                                                               │
└─────────────────────────────────────────────────────────────────────┘
```

### Manual Retraining

Run the retraining pipeline manually if:

- New data arrives before scheduled run
- Model parameters need to be updated
- Testing a new model in production

```bash
# Dry run (validate setup)
python3 scripts/auto_retrain.py --dry-run

# Retrain specific models
python3 scripts/auto_retrain.py --models ridge,huber

# Full retrain with verbose output
python3 scripts/auto_retrain.py --verbose
```

### Model-Specific Retraining

Certain models require more frequent retraining:

| Model Type | Frequency | Rationale |
|------------|-----------|------------|
| **Weekly Nowcasters** | Weekly | High-frequency data (weekly prices) |
| **Regime-Adaptive Models** | Monthly | Regime detection runs monthly |
| **Standard Ensemble Models** | Monthly | Monthly inflation data cycle |
| **Scenario Models** | Quarterly | Scenario analysis is less time-sensitive |

---

## Model Retirement Criteria

### Performance-Based Retirement

A model should be **retired** if:

#### 1. Persistent Performance Degradation

- **Trigger**: MAE consistently > 1.5x historical baseline for **3 consecutive months**
- **Historical Baseline**: Average MAE over previous 12 months
- **Example**:
  - Model historical MAE (h=1): 0.320
  - Retirement threshold: 0.480
  - Current MAE: 0.520 (3 months) → **RETIRE**

#### 2. Outperformed by Baseline

- **Trigger**: MAE > **NaiveSeasonalForecaster** for **6 consecutive months**
- **Rationale**: If a model can't beat a naive baseline, it's not adding value
- **Naive Baseline**: Forecast = Last Year's Value (h=12)

#### 3. High KPI Violation Rate

- **Trigger**: KPI violation rate > 50% over last 6 months
- **KPI Violation**: |error| > 0.5 percentage points
- **Example**:
  - 6 months of backtest data
  - 4 months with |error| > 0.5 → **66% violation rate** → **RETIRE**

#### 4. Non-Convergence Issues

- **Trigger**: Model fails to fit or produces NaN predictions in **2 out of 3 retrain attempts**
- **Action**: Investigate algorithm or remove from production ensemble

### Maintenance-Based Retirement

A model should be **retired** if:

#### 1. Outdated Features

- **Trigger**: Key features become unavailable (e.g., discontinued data source)
- **Example**: Regional indicator 21 (ИБВЭД) has data only from 2019-2023

#### 2. Redundancy

- **Trigger**: New model shows **≥10% MAE improvement** with similar complexity
- **Action**: Retire old model and replace with new model in ensemble

### Retirement Levels

| Level | Criteria | Action |
|-------|-----------|--------|
| **⚠️ Warning** | MAE > 1.2x baseline | Monitor closely, prepare retirement plan |
| **🔴 Critical** | MAE > 1.5x baseline | Start retirement investigation |
| **💀 Retire** | Any retirement criteria met | Remove from ensemble, archive weights |

---

## Performance Monitoring

### Drift Detection

**Tool**: `scripts/drift_detector.py`

**Frequency**: Weekly (Mondays at 10:00 AM)

**Detection Logic**:
```
DRIFT_THRESHOLD = 1.5
historical_baseline = avg(MAE over last 6 months)

if current_MAE > (historical_baseline * DRIFT_THRESHOLD):
    ALERT: "Model drift detected!"
    Notify: Telegram, logs
    Action: Trigger manual retrain investigation
```

**Output**: `data/drift_report.json`

**Alert Levels**:
| Degradation | Action |
|-------------|---------|
| 0-20% | Normal (monthly retrain will handle) |
| 20-50% | Warning (investigate cause) |
| >50% | Critical (immediate retrain) |

### Metrics to Track

| Metric | Frequency | Threshold | Action |
|--------|-----------|------------|--------|
| **MAE (h=1)** | Monthly | >0.400 | Investigate |
| **MAE (h=12)** | Monthly | >0.500 | Investigate |
| **KPI Violation Rate** | Monthly | >50% | Retire model |
| **Drift Ratio** | Weekly | >1.5 | Manual review |
| **Ensemble Weight** | Monthly | <5% | Consider retirement |

### Baseline MAE Values (2025 Backtest)

These values serve as historical baselines for drift detection:

| Model | MAE h=1 | MAE h=2 | MAE h=12 | Status |
|-------|----------|----------|-----------|--------|
| Subcomp | 0.309 | 0.330 | 0.324 | ✅ Best h=1 |
| NGBoost | 0.326 | 0.290 | 0.382 | ✅ Best h=2 |
| Subcomp_Multi | 0.326 | 0.336 | 0.297 | ✅ Best h=12 |
| Huber | 0.324 | 0.297 | 0.331 | ✅ Stable |
| Ridge | 0.321 | 0.289 | 0.326 | Baseline |

---

## Automated Infrastructure

### Retraining Pipeline

**Script**: `scripts/auto_retrain.py` (Edge Lab)

**Cron Entry**:
```cron
# Monthly retrain on 1st at 2am
0 2 1 * * cd /home/valalav/_projects/sirena-kbr/edge_lab && \
  python3 scripts/auto_retrain.py >> logs/retrain.log 2>&1
```

**Features**:
- Automatically detects new data (infl_kbr.csv or enhanced_inflation_data.csv)
- Calculates MAE delta vs previous run
- Saves model weights with timestamp
- Maintains symlink to latest weights: `{model}_latest.pkl`
- Logs performance to `archive/weights/retrain_log.json`

### Drift Detection

**Script**: `scripts/drift_detector.py` (Edge Lab)

**Cron Entry**:
```cron
# Weekly drift detection on Mondays at 10am
0 10 * * 1 cd /home/valalav/_projects/sirena-kbr/edge_lab && \
  python3 scripts/drift_detector.py >> logs/drift.log 2>&1
```

**Features**:
- Compares current MAE vs 6-month historical baseline
- Customizable threshold (default: 1.5x)
- Generates JSON report with affected models
- Exits with code 1 if drift detected (for alerting)

### Backup System

**Script**: `scripts/backup_system.py` (Main Project)

**Cron Entry**:
```cron
# Daily backup at 3am (after retrain)
0 3 * * * cd /home/valalav/_projects/sirena-kbr && \
  python3 scripts/backup_system.py >> logs/backup.log 2>&1
```

**Features**:
- Backs up `data/` and `sirena/models/`
- Creates compressed tar.gz archives
- Retention policy: keep last 5 backups
- Automatic cleanup of old backups

### Telegram Alerting

**Script**: `scripts/telegram_alert.py`

**Alert Conditions**:
- ✅ Retrain completed successfully
- ⚠️ Model drift detected (>50% degradation)
- 🔴 Retraining failed
- 💀 Model retirement required

---

## Emergency Procedures

### Emergency Retraining

**Scenario**: Critical performance degradation or production incident

**Steps**:
1. Run drift detector manually: `python3 scripts/drift_detector.py`
2. Identify affected models from `data/drift_report.json`
3. Trigger manual retrain: `python3 scripts/auto_retrain.py --models <affected_models>`
4. Monitor retrain logs: `tail -f logs/retrain.log`
5. Verify new model performance in Dashboard
6. Send summary report to stakeholders

### Rollback Procedure

**Scenario**: New model performs worse than previous version

**Steps**:
1. Stop production ensemble
2. Restore previous weights from backup:
   ```bash
   # Remove symlink to latest
   rm archive/weights/{model}_latest.pkl
   # Recreate symlink to previous version
   ln -s archive/weights/{model}_{previous_timestamp}.pkl \
     archive/weights/{model}_latest.pkl
   ```
3. Restart ensemble
4. Document rollback in maintenance log

### Emergency Model Retirement

**Scenario**: Model causes system instability or crashes

**Steps**:
1. Remove model from ensemble in `sirena/forecast.py`
2. Set model weight to 0 in dashboard configuration
3. Run drift detector to assess remaining models
4. Archive problematic model weights: `mv archive/weights/{model}*.pkl archive/retired/`
5. Document retirement reason in `docs/RETIRED_MODELS.md`

---

## Appendix

### Maintenance Checklist

**Monthly**:
- [ ] Verify automatic retrain completed successfully
- [ ] Review MAE delta in `retrain_log.json`
- [ ] Check for drift alerts
- [ ] Update ensemble weights if needed
- [ ] Review backup retention (5 most recent)

**Quarterly**:
- [ ] Full backtest on all horizons (h=1, h=2, h=12)
- [ ] Compare vs baseline MAE values
- [ ] Review KPI violation rates
- [ ] Assess model retirement candidates

**Annually**:
- [ ] Comprehensive model performance review
- [ ] Evaluate new model candidates
- [ ] Update documentation (this file)
- [ ] Audit retention policies (backups, logs)

### Contact Points

| Issue | Contact |
|--------|---------|
| Retraining failures | System Administrator |
| Model retirement decisions | Data Science Team |
| Production incidents | On-call Engineer |
| Documentation updates | Technical Writer |

---

**Document Owner**: SIRENA-KBR Development Team
**Last Reviewed**: 2026-01-24
**Next Review**: 2026-04-24
