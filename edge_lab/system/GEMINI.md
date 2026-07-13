# 🤖 Ralph Universal Context & Protocols

> **Core Directive:** Ralph Universal is a dual-loop autonomous agent system where a **Worker** executes tasks and a **Critic** verifies them. This separation of concerns ensures high-quality output and strictly prevents "fake work".

## 📊 Project Status (v3.3)

**Last Updated:** January 24, 2026

| Metric | Count |
|--------|-------|
| **Total Tasks** | 139 |
| **Completed (DONE)** | 116 |
| **Pending Review** | 1 |
| **Blocked** | 6 |
| **TODO** | 17 |
| **Completion Rate** | 83.5% |

## 🏗️ Architecture

The system consists of two parallel processes:
1.  **Worker (The Doer):** Reads `prd.json`, implements code, and writes tests. It marks tasks as `PENDING_REVIEW`.
2.  **Critic (The Observer):** Watches for `PENDING_REVIEW` tasks, executes verification steps, and either **APPROVES** (mark `DONE`) or **REJECTS** (mark `TODO` with feedback).

## 🤖 Active Forecasting Models

### Production Models (Main System)
Located in `/home/valalav/_projects/sirena-kbr/sirena/models/`:

**Core Ensemble Models:**
- `ridge` - Ridge regression with ETS seasonality (baseline)
- `ridge_extended` - Extended features (lags, momentum, volatility, calendar)
- `ridge_shock_dummies` - Shock dummy variables (2014-2015 crises)
- `ridge_macro` - Macro features (Ki, USD, Brent)
- `elasticnet` - L1+L2 regularization
- `huber` - Robust regression (Huber loss)

**Probabilistic Models:**
- `ngboost_model` - Natural Gradient Boosting (probabilistic)
- `ngboost_shock` - NGBoost with shock dummies
- `bayesian_ridge` - Bayesian inference with CI

**Boosting Models:**
- `lightgbm` - Gradient boosting
- `xgboost_model` - XGBoost implementation
- `catboost_model` - CatBoost for small datasets
- `ebm` - Explainable Boosting Machine

**Time Series Models:**
- `arima` - Seasonal ARIMA
- `ets` - Exponential Smoothing
- `prophet` - Facebook Prophet
- `exog_prophet` - Prophet with exogenous variables
- `holt_winters` - Dedicated Holt-Winters

**Advanced Models:**
- `subcomponent` - Bottom-up from 3 components (Food/NonFood/Services)
- `subcomponent_multi` - Multi-horizon subcomponent forecasting
- `microcomponent` - 497 micro models aggregation
- `hierarchical_micro` - Hierarchical reconciliation
- `micro_optimized` - Optimized per-volatility
- `horizon_ensemble` - Adaptive Huber+Micro ensemble

**Scenario Analysis:**
- `scenario_rate` - Rate transmission models (hawk/dove/neutral)
- `ki_trajectory` - Taylor rule for Ki forecasting
- `unified_subcomp` - Scenario-integrated subcomponents
- `regime_detector` - Economic regime detection (shock/normal/high_inflation)

**Experimental:**
- `midas` - Mixed Data Sampling
- `tft` - Temporal Fusion Transformer
- `conformal` - Conformal prediction wrapper
- `micro_arima` - External micro ARIMA model
- `micro_plodovoshchi` - Plodo-vegetables micro model

### Edge Lab Models
Located in `/home/valalav/_projects/sirena-kbr/edge_lab/sirena/models/`:

- `weekly_prices` - Weekly price nowcaster (33 products)
- `volatility_weighted_nowcaster` - Inverse volatility weighting
- `regime_adaptive_nowcaster` - Regime-switching nowcaster
- `leading_indicators` - Leading indicator detection
- `volatility_monitor` - Anomaly detection (1.5σ threshold)
- `exog_loader` - Exogenous data loader
- `exog_forecaster` - Generic exogenous forecaster

### Regressor Models
- `opr_enhanced_ridge` - Ridge with OPR regional data features

## 📋 Completed Tasks Summary

### Testing & Validation (IDs 1-20)
✅ Comprehensive test suite for all core models (Ridge, Huber, NGBoost, EBM, etc.)
✅ Integration tests: Full backtest pipeline, Dashboard flow

### Data Mining & Ingestion (IDs 100-125)
✅ Hypothesis Generator Agent (Task 100)
✅ News Sentiment Analysis (Task 101)
✅ Immune System Stress Testing (Task 104)
✅ Rosstat Autonomous Ingestion (53/53 files) (Task 110)
✅ Multi-Regional Hierarchy & Correlation (Task 111)
✅ External Data Intelligence (CBR/MinFin) (Task 112)
✅ Fedstat Smart Link Prioritization (Task 113)
✅ KBR Macro Monolith (Sheet 010) (Task 114)
✅ Deep Sectoral Blocks (32+ sheets) (Task 115)
✅ Correlation & Regressor Ranking (Task 116)
✅ OPR Enhanced Ridge (Task 117)
✅ Labor Market Deep Dive (111MB) (Task 118)
✅ Producer Prices (PPI) (Task 119)
✅ GRP Forecasts (Task 120)
✅ Ultimate Macro Dataset (Task 121)
✅ OPR Data Discovery (Task 122)
✅ Regional Budgets (Task 123)
✅ High-Freq Indicators (HH.ru/DomClick) (Task 124)
✅ Code Mapping Protocol (Task 125)

### Research & Analysis (IDs 200-539)
✅ ExogProphet experiments (Tasks 241-242)
✅ Metrics aggregation & reporting (Tasks 251-253)
✅ Weekly Research Summary (Task 420)
✅ Weekly volatility analysis (Tasks 411-413)
✅ Weekly regime weights (Tasks 414, 432-433)
✅ Weekly leading indicators (Tasks 416, 434-435)
✅ Weekly anomaly tuning (Tasks 417, 436-437)
✅ Weekly shock analysis (Tasks 418, 438-439)
✅ Weekly correlation analysis (Tasks 419, 440-441)

### Model Implementation (IDs 501-540)
✅ VolatilityWeightedNowcaster (Task 501) + Tests (503)
✅ RegimeAdaptiveNowcaster (Task 502) + Tests (504)
✅ API Health Check (Task 505)
✅ API Model Info Endpoint (Task 506)
✅ KBR GDP Components (Task 507)
✅ Granger Causality Matrix (Task 510)
✅ Dashboard Regime Monitor (Task 514)
✅ Dashboard Alert Panel (Task 515)
✅ Prometheus Metrics (Task 520)
✅ Component Contribution Analysis (Task 522)
✅ Seasonal Pattern Evolution (Task 523)
✅ Fed Policy Transmission (Task 524)
✅ Dashboard Seasonality Tab (Task 525)
✅ Dashboard Macro Tab (Task 526)
✅ Auto-Retraining Pipeline (Task 527)
✅ End-User Guide (Task 528)
✅ Automated Backup (Task 530)
✅ Inflation Persistence (Task 531)
✅ Historical Forecast API (Task 532)
✅ Dashboard Refactor (Parts 1-2) (Tasks 533-534)
✅ Trimmed Mean CPI (Task 536)
✅ Sticky Price Index (Task 537)
✅ Regional Phillips Curve (Task 538)
✅ HoltWintersForecaster (Task 539)
✅ NaiveSeasonalForecaster (Task 540)

### Features & Operations (IDs 541-550)
✅ Telegram Alert Bot (Task 541)
✅ Log Rotation Setup (Task 542)
✅ Weekly Data Validation (Task 543)
✅ Type Hinting (Models) (Task 544)
✅ Parameter Grid Search Tests (Task 546)
✅ Robustness to Missing Data (Task 547)
✅ Correlation Heatmap Update (Task 548)
✅ Forecast Fan Charts (Task 549)
✅ API Documentation Update (Task 550)

## 🛡️ The "Bulletproof" Standards

All code generated by Worker must adhere to these layers:

### 1. Network Layer (The Shield)
- **Retry Logic:** All external API calls (LLM, etc.) must use `tenacity` with exponential backoff.
- **Fallbacks:** If a primary model fails, fallback to a cheaper/faster model or a local model.

### 2. State & Memory Layer (The Anchor)
- **Persistence:** All state updates to `prd.json` and `progress.txt` must be atomic and thread-safe (using file locks).
- **History:** The `progress.txt` log is the source of truth for narrative continuity.

### 3. Verification Layer (The Gate)
- **Critic's Rule:** "Trust but Verify". The Critic NEVER assumes code works. It MUST run code/tests.
- **MVAC (Machine-Verifiable Acceptance Criteria):** Every User Story must have criteria that can be tested via a terminal command.

### 4. Meta-Layer (Continuous Improvement)
- **Reflect & Update:** Use practical execution experience to update system documentation.
- **Mandate:** If a task reveals that current `GEMINI.md`, `README.md`, or Methodology is outdated or incorrect, the agent MUST update these documents immediately.
- **Goal:** Documentation should always reflect the *actual* working state of the system, not just theoretical design.

### 5. Skills Layer (New Capabilities)
- **Definition:** Complex tasks requiring specific libraries or workflows are defined in `edge_lab/skills/<skill_name>.md`.
- **Usage:** Worker MUST read the relevant skill file before attempting a task.
- **Protocol:**
  1. Check `edge_lab/skills/` for relevant capabilities.
  2. Follow "Usage Examples" in the skill file.
  3. Ensure "Verification" steps from the skill file are met.

### 6. Core Skills (Available Now)
- **add-model**: Scaffold new forecasting models (registry, dashboard, etc.).
- **verified-test**: Run pytest with signed JSON receipt (anti-fake-work).
- **rosstat-fetch**: Robust data ingestion from government sites.
- **econometrician**: AI-powered statistical analysis via Opencode.

## 📝 Configuration

- **Root Directory:** `/home/valalav/_projects/sirena-kbr/edge_lab`
- **Docs Directory:** `./docs`
- **Config File:** `system/config.py`
- **Task Database:** `tasks/prd.json`

## 🚫 Critical Constraints (What NOT to do)
- **❌ No "Fake" completion:** Worker cannot set `passes: true`. Only Critic can.
- **❌ No Silent Failures:** If a tool fails, log it loudly in `progress.txt`.
- **❌ No Infinite Loops:** Both Worker and Critic must have circuit breakers (max iterations/time).
