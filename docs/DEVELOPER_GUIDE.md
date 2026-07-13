# Developer Architecture Guide

> **Target Audience**: Contributors and developers working on SIRENA-КБR v5.0 forecasting system
> **Last Updated**: 2026-01-24

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Worker-Critic Architecture](#worker-critic-architecture)
3. [ModelRegistry Pattern](#modelregistry-pattern)
4. [Testing Standards](#testing-standards)
5. [Adding New Models](#adding-new-models)
6. [Code Conventions](#code-conventions)
7. [Common Workflows](#common-workflows)

---

## System Overview

SIRENA-КБR is an inflation forecasting system for the Kabardino-Balkarian Republic (КБR). The system consists of:

- **Forecasting Engine** (`sirena/`) - ML models, ensemble methods
- **Dashboard** (`dashboard.py`) - Streamlit visualization
- **API** (`api/`) - REST endpoints for forecasts
- **Edge Lab** (`edge_lab/`) - Autonomous agent system (Worker/Critic)

### Directory Structure

```
opus_forecast/
├── sirena/                 # Forecasting engine
│   ├── models/             # Individual forecaster implementations
│   │   ├── base.py        # BaseForecaster abstract class
│   │   ├── registry.py    # ModelRegistry factory
│   │   ├── ridge.py       # Ridge regression
│   │   ├── ngboost.py     # NGBoost probabilistic model
│   │   └── ...          # Other models
│   ├── forecast.py         # EnsembleForecaster
│   └── data/              # Data loaders (SA, weekly prices)
├── api/                   # FastAPI REST API
│   └── routes/
│       ├── forecast.py
│       └── health.py
├── tests/                  # Unit and integration tests
├── scripts/                # Utility scripts
├── docs/                   # Documentation
├── data/                   # Data files
├── edge_lab/              # Autonomous agent system
│   ├── system/
│   │   ├── orchestrator.py  # Main entry point
│   │   ├── worker.py       # Task executor
│   │   ├── critic.py       # Task verifier
│   │   └── config.py      # Configuration
│   └── tasks/
│       └── prd.json       # Product Requirements Document
└── dashboard.py            # Streamlit dashboard
```

---

## Worker-Critic Architecture

The **Edge Lab** implements a dual-process autonomous agent system inspired by the "Bulletproof" protocol. This architecture separates task execution from verification to prevent "fake work" and ensure high-quality output.

### Core Components

#### 1. Orchestrator (`edge_lab/system/orchestrator.py`)

The orchestrator spawns three parallel processes:
- **Worker Process**: Executes tasks from the PRD
- **Critic Process**: Reviews and verifies completed tasks
- **Refiner Process** (optional): Decomposes blocked tasks into subtasks

```python
def main():
    """Main entry point for the autonomous agent system."""
    # Start three parallel processes
    worker_proc = multiprocessing.Process(target=run_worker)
    critic_proc = multiprocessing.Process(target=run_critic)
    refiner_proc = multiprocessing.Process(target=run_refiner)

    worker_proc.start()
    critic_proc.start()
    refiner_proc.start()
```

**Process Communication**: All processes share state through `prd.json` and `progress.txt` files (via `StateManager` with file locks).

#### 2. Worker (`edge_lab/system/worker.py`)

The Worker is the **executor**. Its responsibilities:

1. **Task Selection**: Fetches next `TODO` task from `prd.json`
2. **Prompt Formulation**: Builds context from `GEMINI.md`, recent progress, and task details
3. **Code Generation**: Uses LLM to implement the solution
4. **Self-Verification**: Runs verification commands before marking complete
5. **Status Update**: Marks task as `PENDING_REVIEW` for Critic

**Worker Loop**:
```python
def main():
    for i in range(MAX_ITERATIONS):
        # 1. Fetch next TODO task
        tasks = state.read_prd().get("user_stories", [])
        next_task = next((t for t in tasks if t.get("status") == "TODO"), None)

        if not next_task:
            continue  # Wait or exit

        # 2. Build prompt with context
        prompt = build_prompt(next_task, methodology, progress)

        # 3. Call LLM via AgentWrapper
        response = agent.generate(prompt)

        # 4. Parse and execute
        execute_commands(response)

        # 5. Mark as PENDING_REVIEW
        state.update_task(next_task["id"], "PENDING_REVIEW")
```

**Safety Features**:
- **Timeout**: Tasks limited to `max_task_duration_seconds` (default 30 minutes)
- **Retry Logic**: Failed tasks tracked in `TASK_RETRY_COUNTS`
- **Exit Conditions**: Stops when all tasks are `DONE`

#### 3. Critic (`edge_lab/system/critic.py`)

The Critic is the **verifier**. Its responsibilities:

1. **Task Selection**: Monitors for `PENDING_REVIEW` tasks
2. **Criteria Verification**: Executes each acceptance criterion
3. **Decision Making**: Returns `APPROVE` or `REJECT`
4. **Feedback Generation**: Provides specific reasons for rejection

**Critic Loop**:
```python
def main():
    for i in range(MAX_ITERATIONS):
        # 1. Fetch PENDING_REVIEW tasks
        tasks = state.read_prd().get("user_stories", [])
        pending = [t for t in tasks if t.get("status") == "PENDING_REVIEW"]

        for task in pending:
            # 2. Verify each criterion
            criteria = task.get("acceptance_criteria", [])
            results = []
            for criterion in criteria:
                result = verify_criterion(criterion)
                results.append(result)

            # 3. Make decision
            all_passed = all(r["passed"] for r in results)
            decision = "APPROVE" if all_passed else "REJECT"

            # 4. Update status
            new_status = "DONE" if decision == "APPROVE" else "TODO"
            state.update_task(task["id"], new_status, feedback)
```

**Acceptance Criterion Formats**:

| Format | Example | Verification |
|---------|----------|--------------|
| `@file:` | `@file: docs/GUIDE.md exists (>200 lines)` | Check file existence and line count |
| `@functional:` | `@functional: pytest tests/test.py -v exits 0` | Run command, check exit code |
| `@metric:` | `@metric: Test count >= 5` | Parse output, validate metric |
| `@docs:` | `@docs: Explains Agent System` | Human review (manual check) |
| `@integration:` | `@integration: Macro features used in tests` | Code analysis |

**Critic Output Schema**:
```json
{
  "decision": "APPROVE|REJECT",
  "reason": "All criteria verified",
  "criteria_results": [
    {
      "criterion": "File exists",
      "passed": true,
      "evidence": "ls output"
    }
  ],
  "confidence": 0.95
}
```

#### 4. StateManager (`edge_lab/system/core/state.py`)

Thread-safe state management for `prd.json` and `progress.txt`:

```python
class StateManager:
    def __init__(self):
        self.prd_path = PRD_FILE
        self.progress_path = PROGRESS_FILE
        self.lock = FileLock(self.prd_path.parent / ".prd.lock")

    def update_task(self, task_id: int, status: str, feedback: str = ""):
        """Update task status atomically."""
        with self.lock:
            prd = self.read_prd()
            for task in prd["user_stories"]:
                if task["id"] == task_id:
                    task["status"] = status
                    task["feedback"] = feedback
                    break
            self._write_prd(prd)

    def log_progress(self, message: str):
        """Append to progress.txt with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.progress_path.open("a") as f:
            f.write(f"[{timestamp}] {message}\n")
```

### Task Lifecycle

```
TODO ──▶ [WORKER] ──▶ PENDING_REVIEW ──▶ [CRITIC] ──▶ DONE
                     │                          │
                     ▼                          ▼
                  Execute                Verify Criteria
                  Code                   Run Tests
                     │                          │
                     ▼                          ▼
                  Self-Verify             APPROVE or REJECT
                     │                          │
                     └─────┬──────────────────────┘
                           │
                           ▼
                        BLOCKED (after 3 retries)
                           │
                           ▼
                   [REFINER] Decompose
                           │
                           ▼
                      Subtasks (TODO)
```

### Key Principles

**1. Trust But Verify**
- Worker cannot mark tasks as `DONE`
- Only Critic can approve
- Critic MUST run verification commands

**2. No Fake Work**
- Tasks marked as `PENDING_REVIEW` without implementation are rejected
- Evidence must be provided (file outputs, test results)
- "I created the code" ≠ "code works"

**3. Machine-Verifiable Criteria**
- Acceptance criteria should be executable commands
- Examples: `pytest tests/`, `ls -la file.py`, `wc -l file.py`
- Manual checks (`@docs:`) kept minimal

---

## ModelRegistry Pattern

The `ModelRegistry` implements the **Factory pattern** for managing forecasting models. It provides a centralized registry with decorators for easy model registration.

### Core Components

#### 1. ModelRegistry Class (`sirena/models/registry.py`)

```python
class ModelRegistry:
    """Central registry for all forecasting models."""

    _models: Dict[str, Type[BaseForecaster]] = {}
    _default_weights: Dict[str, float] = {
        'ridge': 0.40,
        'bvar': 0.20,
        'lightgbm': 0.15,
        'prophet': 0.10,
        'sarima': 0.05,
        'ets': 0.05,
        'lstm': 0.05
    }

    @classmethod
    def register(cls, name: str):
        """Decorator for registering models."""
        def decorator(model_class: Type[BaseForecaster]):
            cls._models[name] = model_class
            model_class.name = name
            return model_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseForecaster]:
        """Retrieve model class by name."""
        if name not in cls._models:
            raise KeyError(f"Model '{name}' not registered")
        return cls._models[name]

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model names."""
        return list(cls._models.keys())

    @classmethod
    def get_default_weights(cls) -> Dict[str, float]:
        """Get ensemble default weights."""
        return cls._default_weights.copy()
```

### Registering a New Model

**Step 1**: Create model class inheriting from `BaseForecaster`

```python
from sirena.models.base import BaseForecaster
from sirena.models.registry import ModelRegistry
import pandas as pd
import numpy as np

@ModelRegistry.register("my_model")
class MyForecaster(BaseForecaster):
    """
    My custom forecasting model.

    Features:
    - Uses Lasso regression with automatic feature selection
    - Incorporates external macro variables
    - Provides confidence intervals
    """

    name = "my_model"
    MIN_TRAIN_SIZE = 36  # Requires 3 years of data

    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.model = None

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'MyForecaster':
        """Train the model on historical data."""
        # Feature engineering
        X, y = self._prepare_features(df, target_col)

        # Train model
        from sklearn.linear_model import Lasso
        self.model = Lasso(alpha=self.alpha)
        self.model.fit(X, y)

        self._is_fitted = True
        self._last_train_date = df.index[-1]
        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Generate predictions for specified horizon."""
        self._check_fitted()

        # Generate future features
        last_date = self._last_train_date
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                 periods=horizon, freq='MS')

        X_future = self._prepare_future_features(future_dates)

        # Predict
        predictions = self.model.predict(X_future)
        return predictions

    def backtest(self, df: pd.DataFrame, start_date: str = '2019-01-01',
                target_col: str = 'Все товары и услуги') -> pd.DataFrame:
        """Run historical backtest."""
        results = []

        test_dates = df.loc[start_date:].index
        for i, date in enumerate(test_dates):
            # Train on data before this date
            train = df.loc[:date - pd.DateOffset(months=1)]

            # Fit and predict
            self.fit(train, target_col)
            pred = self.forecast(horizon=1)[0]

            # Record results
            actual = df.loc[date, target_col]
            results.append({
                'date': date,
                'actual': actual,
                'prediction': pred,
                'error': actual - pred
            })

        return pd.DataFrame(results)
```

**Step 2**: Import in `sirena/models/__init__.py`

```python
# sirena/models/__init__.py
from .registry import ModelRegistry
from .base import BaseForecaster

# Import all models to trigger registration
from .ridge import RidgeForecaster
from .bvar import BVARForecaster
from .my_model import MyForecaster  # <-- Add your model here

__all__ = [
    'BaseForecaster',
    'ModelRegistry',
    'RidgeForecaster',
    'BVARForecaster',
    'MyForecaster',  # <-- Export your model
]
```

### Using Registered Models

```python
from sirena.models import ModelRegistry, BaseForecaster
import pandas as pd

# Get model class
model_class = ModelRegistry.get("my_model")

# Instantiate and train
model = model_class(alpha=0.5)
model.fit(df, 'Все товары и услуги')

# Forecast
forecast = model.forecast(horizon=12)

# List all available models
all_models = ModelRegistry.list_models()
print(f"Available models: {all_models}")
# Output: ['ridge', 'bvar', 'lightgbm', 'my_model', ...]
```

### Ensemble Integration

```python
from sirena.forecast import EnsembleForecaster

# Create ensemble with custom weights
ensemble = EnsembleForecaster(weights={
    'my_model': 0.20,  # 20% weight to new model
    'ridge': 0.40,
    'bvar': 0.20,
    'prophet': 0.20
})

# Fit and forecast
ensemble.fit(df)
forecast = ensemble.forecast(horizon=12)
```

---

## Testing Standards

All code must have comprehensive test coverage. We follow a **Red-Green-Refactor** approach with `pytest`.

### Test Structure

**Directory Layout**:
```
tests/
├── test_ridge.py
├── test_ngboost.py
├── test_volatility_weighted_nowcaster.py
├── test_regime_adaptive_nowcaster.py
└── integration/
    ├── test_backtest_framework.py
    └── test_api.py
```

### Writing Unit Tests

**Example**: Test for `VolatilityWeightedNowcaster`

```python
"""
Unit tests for VolatilityWeightedNowcaster model.

Tests cover:
- Model initialization
- fit() method
- predict() method
- forecast() method
- Inverse volatility calculation
- backtest() method
- Edge cases
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from sirena.models.volatility_weighted_nowcaster import VolatilityWeightedNowcaster


@pytest.fixture
def sample_monthly_data():
    """Generate sample monthly CPI data for testing."""
    dates = pd.date_range("2016-01-01", periods=96, freq="MS")
    np.random.seed(42)

    data = pd.DataFrame({
        "Все товары и услуги": 100.5 + np.random.randn(96) * 0.3,
        "Продовольственные товары": 100.6 + np.random.randn(96) * 0.4,
        "Непродовольственные товары": 100.3 + np.random.randn(96) * 0.2,
        "Услуги": 100.4 + np.random.randn(96) * 0.3,
    }, index=dates)

    return data


@pytest.fixture
def model():
    """Create a default model instance."""
    return VolatilityWeightedNowcaster()


class TestVolatilityWeightedNowcaster:
    """Test suite for VolatilityWeightedNowcaster."""

    def test_initialization(self, model):
        """Test model initialization with default parameters."""
        assert model.name == "volatility_weighted_nowcaster"
        assert model._is_fitted is False
        assert hasattr(model, 'volatility_weights')

    def test_fit(self, model, sample_monthly_data):
        """Test model fitting."""
        fitted_model = model.fit(sample_monthly_data)

        assert fitted_model._is_fitted is True
        assert fitted_model._last_train_date == sample_monthly_data.index[-1]

    def test_forecast(self, model, sample_monthly_data):
        """Test forecast generation."""
        model.fit(sample_monthly_data)

        forecast = model.forecast(horizon=12)

        assert isinstance(forecast, np.ndarray)
        assert len(forecast) == 12
        assert np.all(np.isfinite(forecast))

    def test_predict_method(self, model, sample_monthly_data):
        """Test predict() method for single point prediction."""
        model.fit(sample_monthly_data)

        target_date = pd.Timestamp("2025-01-01")
        result = model.predict(sample_monthly_data, target_date)

        assert 'prediction' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert isinstance(result['prediction'], (int, float))

    def test_inverse_volatility_weights(self, model, sample_monthly_data):
        """Test that weights are inversely proportional to volatility."""
        model.fit(sample_monthly_data)

        weights = model.volatility_weights
        volatilities = model.volatilities

        # Higher volatility should have lower weight
        for i, (w1, w2) in enumerate(zip(weights[:-1], weights[1:])):
            v1, v2 = volatilities[i], volatilities[i+1]
            if v1 > v2:
                assert w1 < w2, f"Weight {w1} should be less than {w2} for higher volatility {v1} > {v2}"

    def test_backtest_output_structure(self, model, sample_monthly_data):
        """Test backtest returns correct DataFrame structure."""
        result = model.backtest(sample_monthly_data, start_date='2024-01-01')

        assert isinstance(result, pd.DataFrame)
        assert 'date' in result.columns
        assert 'actual' in result.columns
        assert 'prediction' in result.columns
        assert 'error' in result.columns
        assert len(result) > 0

    def test_edge_case_empty_data(self, model):
        """Test behavior with empty dataframe."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError):
            model.fit(empty_df)

    def test_edge_case_min_train_size(self, model, sample_monthly_data):
        """Test that minimum train size is enforced."""
        short_df = sample_monthly_data.iloc[:20]  # Less than MIN_TRAIN_SIZE

        with pytest.raises(ValueError, match="minimum.*observations"):
            model.fit(short_df)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_volatility_weighted_nowcaster.py -v

# Run specific test
pytest tests/test_volatility_weighted_nowcaster.py::TestVolatilityWeightedNowcaster::test_fit -v

# Run with coverage
pytest tests/ --cov=sirena --cov-report=html

# Run with markers
pytest tests/ -m unit  # Only unit tests
pytest tests/ -m integration  # Only integration tests
```

### Test Requirements

For a model to be production-ready:

1. **Minimum 5 test cases** per model
2. **Coverage** of all public methods (`fit`, `forecast`, `predict`, `backtest`)
3. **Edge cases**: Empty data, minimum train size, missing values
4. **Output validation**: Correct types, finite values, expected columns
5. **Deterministic**: Use `np.random.seed()` in fixtures for reproducibility

### Integration Tests

For testing the full pipeline:

```python
"""
Integration tests for backtest framework.
"""

import pytest
import pandas as pd
from scripts.backtest_framework import BacktestRunner


def test_full_backtest_pipeline():
    """Test complete backtest from data loading to metrics."""
    runner = BacktestRunner()

    # Run backtest
    results = runner.run(
        start_date='2024-01-01',
        end_date='2024-12-31',
        models=['ridge', 'ngboost', 'huber']
    )

    # Verify output
    assert 'predictions' in results
    assert 'metrics' in results

    # Check that all models produced forecasts
    pred_df = results['predictions']
    for model in ['ridge', 'ngboost', 'huber']:
        assert model in pred_df.columns

    # Check metrics calculated
    metrics_df = results['metrics']
    assert 'MAE' in metrics_df.columns
    assert all(metrics_df['MAE'] > 0)
```

---

## Adding New Models

Complete checklist for adding a new forecasting model:

### 1. Create Model File

```bash
# Create new model file
touch sirena/models/my_new_model.py
```

### 2. Implement Model Class

```python
from sirena.models.base import BaseForecaster
from sirena.models.registry import ModelRegistry
import pandas as pd
import numpy as np

@ModelRegistry.register("my_model")
class MyNewModel(BaseForecaster):
    name = "my_model"
    MIN_TRAIN_SIZE = 24

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'MyNewModel':
        # Implementation
        self._is_fitted = True
        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        self._check_fitted()
        # Implementation
        return np.zeros(horizon)

    def backtest(self, df: pd.DataFrame, start_date: str = '2019-01-01',
                target_col: str = 'Все товары и услуги') -> pd.DataFrame:
        # Implementation
        return pd.DataFrame()
```

### 3. Register in `__init__.py`

```python
# sirena/models/__init__.py
from .my_new_model import MyNewModel

__all__ = [..., 'MyNewModel']
```

### 4. Create Tests

```bash
touch tests/test_my_new_model.py
```

Minimum 5 test cases covering all public methods.

### 5. Run Backtests

```bash
# Add model to backtest framework
# Edit scripts/backtest_framework.py

# Run backtests
python3 scripts/run_backtest_h1.py
python3 scripts/run_backtest_h2.py
python3 scripts/run_backtest_h12.py
```

### 6. Update Dashboard

**Tab 1 (Forecast)**: Add model forecast generation and trace
**Tab 3 (Backtest)**: Add model to comparison plots

### 7. Update Documentation

- Add model to `docs/MODELS.md`
- Update `CLAUDE.md` with performance metrics
- Document hyperparameters in `docs/DEVELOPER_GUIDE.md`

### 8. Verification

```bash
# Run verification script
python3 scripts/add_model_checklist.py MyNewModel

# Expected: ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ: 11/11
```

---

## Code Conventions

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `RidgeForecaster`, `EnsembleForecaster`)
- **Functions**: `snake_case` (e.g., `fit_model`, `calculate_mae`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MIN_TRAIN_SIZE`, `DEFAULT_HORIZON`)
- **Private methods**: `_leading_underscore` (e.g., `_prepare_features`)

### Docstring Format

```python
def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'RidgeForecaster':
    """
    Train the model on historical data.

    Args:
        df: DataFrame with datetime index and target variable
        target_col: Name of target column (default: 'Все товары и услуги')

    Returns:
        self for method chaining

    Raises:
        ValueError: If df has less than MIN_TRAIN_SIZE observations

    Example:
        >>> model = RidgeForecaster()
        >>> model.fit(df, 'Все товары и услуги')
    """
```

### Type Hints

All functions must include type hints:

```python
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Union[float, np.ndarray]]:
    """Generate prediction for target date."""
    pass
```

### Error Handling

```python
def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'MyModel':
    """Train the model with proper error handling."""

    # Validate input
    if df is None or len(df) == 0:
        raise ValueError("Input DataFrame cannot be empty")

    if len(df) < self.MIN_TRAIN_SIZE:
        raise ValueError(f"Minimum {self.MIN_TRAIN_SIZE} observations required, got {len(df)}")

    try:
        # Training logic
        self._train(df, target_col)
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}") from e
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

def fit(self, df: pd.DataFrame):
    logger.info(f"Training model on {len(df)} observations")
    # ... training logic
    logger.debug(f"Model parameters: {self.params}")
```

---

## Common Workflows

### Running a Backtest

```bash
# h=1 (1 month ahead - MAIN KPI)
python3 scripts/run_backtest_h1.py

# Check results
cat archive/results/backtest_h1_metrics.csv
cat archive/results/backtest_h1_predictions.csv
```

### Testing a New Model

```bash
# 1. Create test file
cat > tests/test_my_model.py << 'EOF'
import pytest
from sirena.models.my_model import MyForecaster

def test_initialization():
    model = MyForecaster()
    assert model.name == "my_model"
EOF

# 2. Run test
pytest tests/test_my_model.py -v

# 3. Verify
python3 -c "from sirena.models.my_model import MyForecaster; print('OK')"
```

### Updating Dashboard

```bash
# 1. Edit dashboard.py
vim dashboard.py

# 2. Restart dashboard
streamlit run dashboard.py --server.port 8503

# 3. Clear cache (press 'C' in dashboard or Settings → Clear cache)
```

### Generating Forecasts

```python
from sirena.models import ModelRegistry
from sirena.forecast import EnsembleForecaster
import pandas as pd

# Load data
df = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',', parse_dates=['Day'])
df = df.set_index('Day').pivot(columns='Товар', values='MoM')

# Create ensemble
ensemble = EnsembleForecaster()
ensemble.fit(df)

# Forecast
forecast = ensemble.forecast(horizon=12)
print(f"12-month forecast: {forecast}")
```

### Using the API

```bash
# Start API
uvicorn api.main:app --reload --port 8000

# Get forecast
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"horizon": 6, "target": "Все товары и услуги"}'

# Get model list
curl http://localhost:8000/models

# Health check
curl http://localhost:8000/health
```

---

## Resources

### Documentation
- [CLAUDE.md](../CLAUDE.md) - Project rules and model performance
- [docs/MODELS.md](MODELS.md) - Model descriptions and weights
- [docs/API.md](API.md) - REST API documentation
- [docs/BACKTEST_METHODOLOGY.md](BACKTEST_METHODOLOGY.md) - Backtest validation rules

### Code References
- [sirena/models/base.py](../sirena/models/base.py) - BaseForecaster interface
- [sirena/models/registry.py](../sirena/models/registry.py) - ModelRegistry implementation
- [edge_lab/system/orchestrator.py](../edge_lab/system/orchestrator.py) - Agent orchestrator
- [edge_lab/system/worker.py](../edge_lab/system/worker.py) - Worker agent
- [edge_lab/system/critic.py](../edge_lab/system/critic.py) - Critic agent

### External Tools
- [pytest](https://docs.pytest.org/) - Testing framework
- [Streamlit](https://streamlit.io/) - Dashboard framework
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
- [scikit-learn](https://scikit-learn.org/) - ML library

---

**Last Updated**: 2026-01-24
**Maintainers**: Edge Lab Development Team
