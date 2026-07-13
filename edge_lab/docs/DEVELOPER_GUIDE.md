# Developer Architecture Guide

> **Version**: 1.0 | **Updated**: 2026-01-24

This guide is for contributors who want to understand the architecture, add new models, or improve the system.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Worker-Critic System](#worker-critic-system)
3. [ModelRegistry Pattern](#modelregistry-pattern)
4. [Testing Standards](#testing-standards)
5. [Code Standards](#code-standards)
6. [Workflow Examples](#workflow-examples)
7. [Common Pitfalls](#common-pitfalls)
8. [Best Practices](#best-practices)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Opus Forecast System                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐          ┌──────────────┐               │
│  │  sirena/     │          │  edge_lab/   │               │
│  │              │          │              │               │
│  │  models/     │          │  system/      │               │
│  │  ├── base.py │◄──────►│  worker.py   │               │
│  │  ├── registry│◄──────►│  critic.py   │               │
│  │  ├── ridge.py│          │  orchestrator│               │
│  │  └── ...     │          │              │               │
│  └──────────────┘          └──────────────┘               │
│         │                         │                          │
│         │                         │                          │
│  ┌──────▼──────────┐   ┌──────▼──────────────┐        │
│  │  tests/        │   │  tasks/             │        │
│  │  test_*.py    │   │  prd.json          │        │
│  └───────────────┘   └─────────────────────┘        │
│                                                               │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **BaseForecaster** | `sirena/models/base.py` | Abstract base class for all models |
| **ModelRegistry** | `sirena/models/registry.py` | Factory pattern for model management |
| **Worker** | `edge_lab/system/worker.py` | Autonomous task execution |
| **Critic** | `edge_lab/system/critic.py` | Task verification |
| **Orchestrator** | `edge_lab/system/orchestrator.py` | Process manager |
| **StateManager** | `edge_lab/system/core/state.py` | Thread-safe task queue |

---

## Worker-Critic System

The Worker-Critic system is a **dual-loop autonomous agent system** that ensures high-quality output through separation of concerns.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   prd.json (Task Queue)             │
│  ┌──────────┐    ┌───────────────┐   ┌─────────┐│
│  │   TODO   │───▶│ PENDING_REVIEW│──▶│  DONE   ││
│  └──────────┘    └───────────────┘   └─────────┘│
│       ▲                │                   │         │
│       │                │                   │         │
│       └────────────────┘                   │         │
│         (on REJECT)                        │         │
└─────────────────────────────────────────────────────────┘
         ▲                    │
         │                    ▼
    ┌─────────┐         ┌──────────┐
    │ Worker  │◀────────│  Critic  │
    │(opencode)│         │(opencode)│
    └─────────┘         └──────────┘
```

### Worker Lifecycle

The Worker follows a **Red-Green-Refactor** methodology:

#### Phase 1: Research (Mandatory)

Before writing code, Worker must:

1. **List relevant files**:
   ```bash
   ls -la sirena/models/
   ```

2. **Check existing code**:
   ```bash
   head -100 sirena/models/base.py
   ```

3. **State approach**:
   ```
   "I will create a new model X by inheriting from BaseForecaster
    and implementing fit(), forecast(), and backtest() methods."
   ```

#### Phase 2: Implementation

1. **Work ONLY in PROJECT_ROOT**: `/home/valalav/_projects/sirena-kbr`
2. **Create REAL files**: Not just print statements
3. **Follow patterns**: Use existing models as templates

#### Phase 3: Self-Verification (CRITICAL)

Worker MUST run verification commands before claiming completion:

**For code tasks:**
```bash
python3 -m py_compile your_file.py
pytest tests/your_test.py -v
```

**For data tasks:**
```bash
head -5 data/output.csv  # Show data!
wc -l data/output.csv     # Show row count!
```

### Critic Lifecycle

The Critic acts as an **adversarial verifier**:

#### Verification Steps

1. **CHECK FILES EXIST**:
   ```bash
   ls -la claimed_file_path
   wc -l file (verify line count claims)
   ```

2. **CHECK CODE COMPILES**:
   ```bash
   python3 -m py_compile file.py
   ```

3. **RUN VERIFICATION COMMANDS**:
   ```bash
   pytest tests/test_x.py -v  # Actually run, don't trust output
   ```

4. **CHECK FOR REGRESSIONS**:
   - Does existing code still work?
   - Are imports broken?

#### Decision Protocol

```python
# Critic output schema
{
  "decision": "APPROVE|REJECT",
  "reason": "Detailed explanation",
  "criteria_results": [
    {"criterion": "File exists", "passed": true, "evidence": "ls output"}
  ],
  "confidence": 0.95
}
```

### Status Transitions

Valid transitions:
- `TODO → PENDING_REVIEW` (Worker completes)
- `PENDING_REVIEW → DONE` (Critic approves)
- `PENDING_REVIEW → TODO` (Critic rejects)
- `TODO → DONE` (Manual intervention only)

### Critical Safety Features

1. **Race Condition Protection** (`state.py`):
   ```python
   # Don't allow PENDING_REVIEW if task has rejection feedback
   if status == "PENDING_REVIEW" and "Reject" in old_feedback:
       return  # Block - let Worker handle rejection first
   ```

2. **Process Restart** (`orchestrator.py`):
   ```python
   # Automatically restart crashed processes
   if not p_worker.is_alive():
       p_worker = multiprocessing.Process(target=run_worker)
       p_worker.start()
   ```

3. **Safety Limits** (`config.py`):
   ```python
   SAFETY_LIMITS = {
       'max_task_duration_seconds': 1800,  # 30 min per task
       'max_retries_per_task': 3,          # → BLOCKED after 3 fails
       'max_file_size_mb': 100,
   }
   ```

---

## ModelRegistry Pattern

The ModelRegistry implements a **Factory pattern** for managing forecasting models.

### Core Concepts

```python
from sirena.models import BaseForecaster, ModelRegistry

# Decorator registration
@ModelRegistry.register("my_model")
class MyModel(BaseForecaster):
    name = "my_model"

    def fit(self, df):
        self._is_fitted = True
        return self

    def forecast(self, horizon=12):
        return np.array([...])

    def backtest(self, df, start_date):
        return pd.DataFrame({...})
```

### Registry API

#### 1. Register a Model (Decorator)

```python
from sirena.models.base import BaseForecaster
from sirena.models.registry import ModelRegistry

@ModelRegistry.register("ridge")
class RidgeForecaster(BaseForecaster):
    """
    Ridge regression forecaster with ETS seasonality.

    Parameters:
        alpha: L2 regularization strength
    """
    def fit(self, df, target_col='Все товары и услуги'):
        # Training logic
        self._is_fitted = True
        return self
```

#### 2. Register Programmatically

```python
ModelRegistry.register_model(
    name="custom_model",
    model_class=CustomModel
)
```

#### 3. Get a Model Instance

```python
# Get model with default parameters
model = ModelRegistry.get("ridge")

# Get model with custom parameters
model = ModelRegistry.get("ridge", alpha=0.5)
```

#### 4. List All Models

```python
models = ModelRegistry.list_models()
# Returns: ['ridge', 'bvar', 'prophet', ...]
```

#### 5. Get Default Weights

```python
weights = ModelRegistry.get_default_weights()
# Returns: {'ridge': 0.40, 'bvar': 0.20, ...}
```

### Implementation Details

#### Registry Storage

```python
class ModelRegistry:
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
```

#### Type Enforcement

```python
@classmethod
def register(cls, name: str):
    def decorator(model_class: Type[BaseForecaster]):
        # Ensure inheritance
        if not issubclass(model_class, BaseForecaster):
            raise TypeError(f"{model_class.__name__} must inherit from BaseForecaster")

        # Prevent duplicates
        if name in cls._models:
            raise ValueError(f"Model '{name}' already registered")

        model_class.name = name
        cls._models[name] = model_class
        return model_class
    return decorator
```

### Creating a New Model

#### Step 1: Create Model File

```python
# sirena/models/my_model.py

from typing import Optional
import pandas as pd
import numpy as np
from sirena.models.base import BaseForecaster
from sirena.models.registry import ModelRegistry

@ModelRegistry.register("my_model")
class MyForecaster(BaseForecaster):
    """
    Description of what this model does.

    Parameters:
        param1: Description
        param2: Description
    """

    name = "my_model"
    MIN_TRAIN_SIZE = 24

    def __init__(self, param1=1.0, param2=0.5, **kwargs):
        super().__init__(**kwargs)
        self.param1 = param1
        self.param2 = param2

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'BaseForecaster':
        """
        Train the model.

        Args:
            df: DataFrame with datetime index and target column
            target_col: Name of target column

        Returns:
            self for method chaining
        """
        # Feature engineering
        X = self._prepare_features(df)
        y = df[target_col].values

        # Training logic here
        self.model = SomeAlgorithm(self.param1, self.param2)
        self.model.fit(X, y)

        self._is_fitted = True
        self._last_train_date = df.index[-1]

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """
        Generate forecasts.

        Args:
            horizon: Number of periods to forecast

        Returns:
            numpy array with MoM % values
        """
        self._check_fitted()

        # Forecast logic here
        predictions = self.model.predict(horizon)

        return predictions

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = '2019-01-01',
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """
        Run historical backtest.

        Args:
            df: Historical data
            start_date: Backtest start date
            target_col: Target column name

        Returns:
            DataFrame with columns: date, actual, prediction, error
        """
        results = []

        for date in df.index[df.index >= start_date]:
            train = df[df.index < date]
            test = df[df.index >= date:date]

            if len(train) < self.MIN_TRAIN_SIZE:
                continue

            self.fit(train, target_col)
            pred = self.forecast(horizon=1)[0]
            actual = test[target_col].iloc[0]

            results.append({
                'date': date,
                'actual': actual,
                'prediction': pred,
                'error': pred - actual
            })

        return pd.DataFrame(results)

    def _prepare_features(self, df: pd.DataFrame):
        """Helper method for feature engineering."""
        # Add lag features
        for lag in [1, 2, 3, 6, 12]:
            df[f'lag_{lag}'] = df[target_col].shift(lag)

        # Add seasonal features
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter

        return df

    def _check_fitted(self):
        """Raise error if model not fitted."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before calling predict/forecast")
```

#### Step 2: Update `__init__.py`

```python
# sirena/models/__init__.py

from .my_model import MyForecaster
from .ridge import RidgeForecaster
# ... other imports

__all__ = [
    'MyForecaster',
    'RidgeForecaster',
    # ... other models
]
```

#### Step 3: Add to Registry (Optional)

If not using decorator, add programmatically:
```python
from sirena.models import MyForecaster
ModelRegistry.register_model("my_model", MyForecaster)
```

---

## Testing Standards

### Philosophy

**Trust but Verify**: Tests are the only way to prove code works.

### Test Structure

```python
"""
Unit tests for MyModel.

Tests cover:
- Model initialization
- fit() method
- forecast() method
- backtest() method
- Edge cases (insufficient data, etc.)
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.models.my_model import MyForecaster


@pytest.fixture
def sample_data():
    """Generate sample monthly CPI data."""
    dates = pd.date_range("2016-01-01", periods=120, freq="MS")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "Все товары и услуги": 100.5 + np.random.randn(120) * 0.3,
        },
        index=dates,
    )

    return data


@pytest.fixture
def model():
    """Create a default model instance."""
    return MyForecaster()


class TestMyForecaster:
    """Test suite for MyForecaster."""

    def test_initialization(self, model):
        """Test model initializes with correct defaults."""
        assert model.param1 == 1.0
        assert model.param2 == 0.5
        assert not model._is_fitted

    def test_fit_returns_self(self, model, sample_data):
        """Test fit() returns self for method chaining."""
        result = model.fit(sample_data)
        assert result is model
        assert model._is_fitted

    def test_fit_requires_minimum_data(self, model):
        """Test fit() raises error with insufficient data."""
        insufficient_data = pd.DataFrame(
            {"Все товары и услуги": [100, 101]},
            index=pd.date_range("2024-01-01", periods=2, freq="MS")
        )

        with pytest.raises(ValueError):
            model.fit(insufficient_data)

    def test_forecast_requires_fit(self, model, sample_data):
        """Test forecast() raises error if model not fitted."""
        with pytest.raises(ValueError, match="must be fitted"):
            model.forecast(horizon=12)

    def test_forecast_output_shape(self, model, sample_data):
        """Test forecast() returns correct shape."""
        model.fit(sample_data)
        forecast = model.forecast(horizon=12)

        assert isinstance(forecast, np.ndarray)
        assert forecast.shape == (12,)

    def test_forecast_values_in_range(self, model, sample_data):
        """Test forecast values are reasonable (MoM %)."""
        model.fit(sample_data)
        forecast = model.forecast(horizon=12)

        # MoM should typically be between -5% and +5%
        assert np.all(forecast > -10)
        assert np.all(forecast < 10)

    def test_backtest_returns_dataframe(self, model, sample_data):
        """Test backtest() returns DataFrame with correct columns."""
        result = model.backtest(sample_data, start_date="2020-01-01")

        assert isinstance(result, pd.DataFrame)
        assert 'date' in result.columns
        assert 'actual' in result.columns
        assert 'prediction' in result.columns
        assert 'error' in result.columns

    def test_backtest_calculates_mae(self, model, sample_data):
        """Test backtest() errors are reasonable."""
        result = model.backtest(sample_data, start_date="2020-01-01")

        mae = result['error'].abs().mean()
        # Should have reasonable MAE (< 2.0 for random data)
        assert mae < 2.0

    def test_backtest_skips_insufficient_train_data(self, model, sample_data):
        """Test backtest() skips dates with insufficient training data."""
        result = model.backtest(sample_data, start_date="2017-01-01")

        # Should start from where MIN_TRAIN_SIZE allows
        assert len(result) > 0
        assert result['date'].min() >= pd.Timestamp("2017-12-01")


class TestMyForecasterEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframe_raises_error(self, model):
        """Test empty DataFrame raises appropriate error."""
        empty_df = pd.DataFrame(columns=["Все товары и услуги"])

        with pytest.raises(ValueError):
            model.fit(empty_df)

    def test_missing_target_column(self, model, sample_data):
        """Test missing target column raises error."""
        data_no_target = sample_data.drop(columns=["Все товары и услуги"])

        with pytest.raises(KeyError):
            model.fit(data_no_target)

    def test_custom_parameters(self):
        """Test model accepts custom parameters."""
        custom_model = MyForecaster(param1=5.0, param2=2.5)
        assert custom_model.param1 == 5.0
        assert custom_model.param2 == 2.5
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_my_model.py -v

# Run specific test
pytest tests/test_my_model.py::TestMyForecaster::test_fit_returns_self -v

# Run with coverage
pytest tests/ --cov=sirena/models --cov-report=html

# Run tests with output capture
pytest tests/ -s  # Show print statements
```

### Integration Tests

For testing end-to-end workflows:

```python
# tests/test_integration.py

import pytest
import pandas as pd
from sirena.models import ModelRegistry


def test_model_registry_get():
    """Test ModelRegistry can retrieve models."""
    ridge = ModelRegistry.get("ridge")
    assert ridge is not None
    assert ridge.name == "ridge"


def test_model_registry_list():
    """Test ModelRegistry can list all models."""
    models = ModelRegistry.list_models()
    assert "ridge" in models
    assert "prophet" in models
    assert len(models) > 5


def test_full_forecast_pipeline():
    """Test complete forecast pipeline."""
    # Load data
    df = pd.read_csv("data/infl_kbr.csv", sep=';', decimal=',')

    # Train model
    model = ModelRegistry.get("ridge")
    model.fit(df, target_col='Все товары и услуги')

    # Generate forecast
    forecast = model.forecast(horizon=12)

    # Verify output
    assert len(forecast) == 12
    assert all(np.isfinite(forecast))
```

### Test Naming Conventions

```
tests/
├── test_models.py              # General model tests
├── test_ridge.py             # Specific model tests
├── test_integration.py         # Integration tests
├── test_api.py               # API tests
├── test_forecast.py           # Forecast pipeline tests
└── api/
    ├── test_endpoints.py       # API endpoint tests
    └── test_models.py         # API model tests
```

### Acceptance Criteria in Tests

Every test should verify acceptance criteria:

```python
def test_mae_below_threshold(self, model, sample_data):
    """@metric: MAE < 0.35 OR documented limitation"""
    result = model.backtest(sample_data, start_date="2020-01-01")
    mae = result['error'].abs().mean()

    try:
        assert mae < 0.35
    except AssertionError:
        # Document if impossible
        pytest.fail(f"MAE {mae:.3f} exceeds 0.35. "
                   "Document architectural limitation if expected.")
```

---

## Code Standards

### Python Standards

1. **PEP 8 Compliance**:
   - Use `snake_case` for functions and variables
   - Use `PascalCase` for classes
   - Maximum line length: 88 characters (black default)

2. **Type Hints**:
   ```python
   def fit(
       self,
       df: pd.DataFrame,
       target_col: str = 'Все товары и услуги'
   ) -> 'BaseForecaster':
       """Docstring here."""
   ```

3. **Docstrings**:
   ```python
   def forecast(self, horizon: int = 12) -> np.ndarray:
       """
       Generate forecasts for specified horizon.

       Args:
           horizon: Number of periods to forecast

       Returns:
           numpy array with MoM % values

       Raises:
           ValueError: If model not fitted
       """
   ```

### File Organization

```
sirena/
├── models/
│   ├── __init__.py           # Model imports
│   ├── base.py              # BaseForecaster abstract class
│   ├── registry.py          # ModelRegistry factory
│   ├── ridge.py            # RidgeForecaster
│   ├── prophet.py          # ProphetForecaster
│   └── ...                # Other models
├── data/
│   ├── loader.py            # Data loading utilities
│   └── weekly_loader.py     # Weekly data loader
└── forecast.py             # EnsembleForecaster
```

### Imports

```python
# Standard library first
import os
import sys
from typing import Dict, List, Optional

# Third-party libraries
import numpy as np
import pandas as pd
import pytest

# Project imports
from sirena.models.base import BaseForecaster
from sirena.models.registry import ModelRegistry
```

### Error Handling

```python
def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги'):
    """Train model with proper error handling."""
    # Validate inputs
    if df is None or len(df) == 0:
        raise ValueError("DataFrame cannot be empty")

    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame")

    if len(df) < self.MIN_TRAIN_SIZE:
        raise ValueError(
            f"Insufficient data: {len(df)} < {self.MIN_TRAIN_SIZE}"
        )

    try:
        # Training logic
        self.model.fit(X, y)
    except Exception as e:
        raise RuntimeError(f"Model training failed: {e}") from e
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

def fit(self, df, target_col='Все товары и услуги'):
    """Train model with logging."""
    logger.info(f"Training model with {len(df)} observations")

    try:
        # Training logic
        logger.info(f"Model trained successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
```

---

## Workflow Examples

### Example 1: Adding a Simple Model

```python
# sirena/models/simple_average.py

from sirena.models.base import BaseForecaster
from sirena.models.registry import ModelRegistry
import pandas as pd
import numpy as np

@ModelRegistry.register("simple_average")
class SimpleAverageForecaster(BaseForecaster):
    """
    Simple moving average forecaster for baseline comparison.
    """

    name = "simple_average"

    def __init__(self, window: int = 12, **kwargs):
        super().__init__(**kwargs)
        self.window = window

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги'):
        """Store data for averaging."""
        self.data = df[target_col].copy()
        self._is_fitted = True
        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Predict moving average."""
        self._check_fitted()
        last_avg = self.data.tail(self.window).mean()
        return np.full(horizon, last_avg)
```

### Example 2: Running Backtest

```python
from sirena.models import ModelRegistry
import pandas as pd

# Load data
df = pd.read_csv("data/infl_kbr.csv", sep=';', decimal=',')

# Get model
model = ModelRegistry.get("ridge")
model.fit(df, target_col='Все товары и услуги')

# Run backtest
results = model.backtest(df, start_date='2019-01-01')

# Calculate MAE
mae = results['error'].abs().mean()
print(f"MAE: {mae:.3f}")
```

### Example 3: Creating Test

```python
# tests/test_simple_average.py

import pytest
import pandas as pd
import numpy as np
from sirena.models import ModelRegistry

def test_simple_average_basic():
    """Test basic functionality."""
    dates = pd.date_range("2016-01-01", periods=24, freq="MS")
    df = pd.DataFrame({"Все товары и услуги": range(24)}, index=dates)

    model = ModelRegistry.get("simple_average")
    model.fit(df)
    forecast = model.forecast(horizon=3)

    assert forecast.shape == (3,)
    assert np.all(forecast == forecast[0])  # All same value
```

---

## Common Pitfalls

### 1. Forgetting to Call `super().__init__()`

```python
# ❌ Bad
class MyModel(BaseForecaster):
    def __init__(self):
        self._is_fitted = False  # Missing super()

# ✅ Good
class MyModel(BaseForecaster):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Initialize parent
        # Custom initialization
```

### 2. Not Setting `_is_fitted` Flag

```python
# ❌ Bad
def fit(self, df):
    self.model = SomeModel()
    # Forgot: self._is_fitted = True

# ✅ Good
def fit(self, df):
    self.model = SomeModel()
    self._is_fitted = True  # Must set this
```

### 3. Not Checking `_check_fitted()` Before Predict

```python
# ❌ Bad
def forecast(self, horizon):
    # Predict without checking if fitted
    return self.model.predict(horizon)

# ✅ Good
def forecast(self, horizon):
    self._check_fitted()  # Always check first
    return self.model.predict(horizon)
```

### 4. Not Handling Missing Values

```python
# ❌ Bad
def fit(self, df):
    X = df[['lag1', 'lag2']]
    y = df['target']
    # NaN values will cause errors

# ✅ Good
def fit(self, df):
    X = df[['lag1', 'lag2']].fillna(method='ffill')
    y = df['target'].fillna(method='ffill')
```

### 5. Tests Don't Actually Run Code

```python
# ❌ Bad
def test_fit_works():
    model.fit(df)
    assert True  # Never actually checks anything

# ✅ Good
def test_fit_sets_is_fitted():
    model = MyModel()
    assert not model._is_fitted  # Before

    model.fit(df)
    assert model._is_fitted  # After
```

### 6. Hardcoding File Paths

```python
# ❌ Bad
path = "/home/user/opus_forecast/data/file.csv"

# ✅ Good
from pathlib import Path
path = Path(__file__).parent.parent / "data" / "file.csv"
```

---

## Best Practices

### 1. Use Fixtures in Tests

```python
@pytest.fixture
def sample_cpi_data():
    """Reusable sample data."""
    dates = pd.date_range("2016-01-01", periods=120, freq="MS")
    np.random.seed(42)
    return pd.DataFrame({
        "Все товары и услуги": 100 + np.random.randn(120) * 0.5
    }, index=dates)

def test_model_fits(sample_cpi_data):
    """Use fixture."""
    model = ModelRegistry.get("ridge")
    model.fit(sample_cpi_data)
    assert model._is_fitted
```

### 2. Test Both Success and Failure Cases

```python
def test_fit_succeeds_with_valid_data(model, sample_data):
    """Test success case."""
    assert model.fit(sample_data) is model

def test_fit_fails_with_insufficient_data(model):
    """Test failure case."""
    insufficient = pd.DataFrame({"Все товары и услуги": [1, 2]})
    with pytest.raises(ValueError):
        model.fit(insufficient)
```

### 3. Document Model Parameters

```python
@ModelRegistry.register("my_model")
class MyForecaster(BaseForecaster):
    """
    My forecasting model.

    Parameters:
        alpha (float): Regularization strength. Default=1.0
            Higher values = more regularization.
        max_iter (int): Maximum iterations. Default=1000
            If not converged, warning is raised.

    Methods:
        fit(df, target_col): Train model
        forecast(horizon): Generate predictions
        backtest(df, start_date): Historical validation

    Example:
        >>> model = ModelRegistry.get("my_model")
        >>> model.fit(df)
        >>> pred = model.forecast(horizon=12)
    """
```

### 4. Use Context Managers for Resources

```python
# ❌ Bad
f = open("file.txt")
data = f.read()
f.close()  # Can be missed on error

# ✅ Good
with open("file.txt") as f:
    data = f.read()  # Automatically closed
```

### 5. Validate Input Early

```python
def fit(self, df, target_col='Все товары и услуги'):
    """Validate inputs first."""
    # Check DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df)}")

    if len(df) == 0:
        raise ValueError("DataFrame cannot be empty")

    # Check columns
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found")

    # Check minimum size
    if len(df) < self.MIN_TRAIN_SIZE:
        raise ValueError(
            f"Need at least {self.MIN_TRAIN_SIZE} observations"
        )

    # Now safe to proceed
    ...
```

### 6. Write Descriptive Test Names

```python
# ❌ Bad
def test_1(model):
    """Unclear what's being tested."""

# ✅ Good
def test_fit_returns_self_for_method_chaining(model):
    """Clear description of test intent."""
```

### 7. Use Parameterized Tests

```python
import pytest

@pytest.mark.parametrize("horizon,expected_shape", [
    (1, (1,)),
    (6, (6,)),
    (12, (12,)),
])
def test_forecast_shape(model, sample_data, horizon, expected_shape):
    """Test forecast output shape for different horizons."""
    model.fit(sample_data)
    forecast = model.forecast(horizon=horizon)
    assert forecast.shape == expected_shape
```

### 8. Keep Models Independent

```python
# ❌ Bad - tight coupling to dashboard
class MyModel(BaseForecaster):
    def forecast(self, horizon):
        # Direct coupling
        from dashboard import add_trace
        add_trace(self.name, predictions)

# ✅ Good - pure model
class MyModel(BaseForecaster):
    def forecast(self, horizon):
        # Just return predictions
        return predictions
```

---

## Summary

This guide covers:

1. ✅ **Worker-Critic Architecture**: Dual-loop system for quality assurance
2. ✅ **ModelRegistry Pattern**: Factory pattern for model management
3. ✅ **Testing Standards**: Comprehensive test coverage requirements
4. ✅ **Code Standards**: Python conventions and project structure
5. ✅ **Workflow Examples**: Practical code samples
6. ✅ **Common Pitfalls**: Mistakes to avoid
7. ✅ **Best Practices**: Guidelines for writing better code

For more information:
- **Architecture Details**: `edge_lab/docs/ARCHITECTURE.md`
- **Quality Standards**: `edge_lab/docs/QUALITY_MANIFESTO.md`
- **Opencode Reference**: `edge_lab/docs/opencode_reference.md`
- **Main Documentation**: `/home/valalav/_projects/sirena-kbr/CLAUDE.md`

---

## Appendix: Quick Reference

### ModelRegistry API

```python
# Register
@ModelRegistry.register("name")

# Get instance
ModelRegistry.get("name", **params)

# List all
ModelRegistry.list_models()

# Get weights
ModelRegistry.get_default_weights()
```

### BaseForecaster Methods

```python
# Required to implement
def fit(df, target_col) -> BaseForecaster
def forecast(horizon) -> np.ndarray
def backtest(df, start_date) -> pd.DataFrame

# Provided by base class
_check_fitted()  # Raises if not fitted
```

### Test Commands

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=sirena --cov-report=html

# Run specific test
pytest tests/test_model.py::test_name -v
```

### Verification Commands (Worker Self-Check)

```bash
# Code compiles?
python3 -m py_compile file.py

# Tests pass?
pytest tests/test_file.py -v

# File exists?
ls -la path/to/file

# Line count?
wc -l file.py

# Data looks correct?
head -5 data/output.csv
```

**Remember**: The Critic will run these commands too. Don't claim success without verification!
