# Edge Lab Developer Architecture Guide

> **Target Audience**: Contributors and developers working on the autonomous agent system
> **Last Updated**: 2025-01-25
> **Version**: 1.2

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Worker-Critic Architecture](#worker-critic-architecture)
3. [ModelRegistry Pattern](#modelregistry-pattern)
4. [Testing Standards](#testing-standards)
5. [Code Standards](#code-standards)
6. [Task Development Workflow](#task-development-workflow)

---

## System Overview

Edge Lab implements a **dual-loop autonomous agent system** inspired by the "Bulletproof" protocol. This architecture separates task execution from verification to ensure high-quality output and prevent "fake work".

### Core Principles

1. **Trust But Verify**: Never assume a task is done. Verify with code execution or rigorous review
2. **No Fake Work**: Do not mark tasks as DONE unless acceptance criteria are met and verified
3. **Evolution**: If you find a better way, update documentation to reflect reality

### Directory Structure

```
edge_lab/
├── system/                    # Core orchestration
│   ├── orchestrator.py        # Main entry point, process manager
│   ├── worker.py             # Task executor (opencode)
│   ├── critic.py             # Task verifier (opencode)
│   ├── config.py             # Configuration constants
│   └── core/
│       ├── state.py          # Thread-safe state management
│       └── agent_wrapper.py  # LLM interface wrapper
├── tasks/
│   ├── prd.json             # Product Requirements Document (task queue)
│   └── progress.txt        # Execution log
├── agents/                  # Generated autonomous agents
├── sirena/                  # Forecasting models (subset of main system)
│   └── models/
│       ├── base.py          # BaseForecaster abstract class
│       └── registry.py     # ModelRegistry factory
├── scripts/                 # Utility and analysis scripts
├── tests/                   # Unit and integration tests
└── docs/                    # Documentation
    ├── ARCHITECTURE.md      # Architecture and lessons learned
    └── DEVELOPER_GUIDE.md   # This file
```

---

## Worker-Critic Architecture

The autonomous system consists of three parallel processes managed by the Orchestrator:

### 1. Orchestrator (`system/orchestrator.py`)

The orchestrator is the process manager that spawns and monitors Worker, Critic, and optional Refiner processes.

```python
def main():
    """Main entry point for autonomous agent system."""
    # Start three parallel processes
    worker_proc = multiprocessing.Process(target=run_worker)
    critic_proc = multiprocessing.Process(target=run_critic)
    refiner_proc = multiprocessing.Process(target=run_refiner)

    worker_proc.start()
    critic_proc.start()
    refiner_proc.start()

    # Monitor and restart on crash
    while True:
        if not worker_proc.is_alive():
            worker_proc = multiprocessing.Process(target=run_worker)
            worker_proc.start()
        # ... similar for critic and refiner
```

**Key Responsibilities**:
- Spawn parallel processes
- Monitor process health
- Restart crashed processes
- Manage lifecycle (graceful shutdown)

### 2. Worker (`system/worker.py`)

The Worker is the **executor**. Its responsibilities:

1. **Task Selection**: Fetches next `TODO` task from `prd.json`
2. **Context Building**: Loads relevant documentation and recent progress
3. **Code Generation**: Uses LLM to implement solution
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
            time.sleep(5)
            continue

        # 2. Build prompt with context
        prompt = build_prompt(next_task, methodology, progress)

        # 3. Call LLM via AgentWrapper
        response = agent.generate(prompt)

        # 4. Parse and execute commands
        execute_commands(response)

        # 5. Mark as PENDING_REVIEW
        state.update_task(next_task["id"], "PENDING_REVIEW")
```

**Safety Features**:
- **Timeout**: Tasks limited to `max_task_duration_seconds` (default 30 minutes)
- **Retry Logic**: Failed tasks tracked in `TASK_RETRY_COUNTS`
- **Exit Conditions**: Stops when all tasks are `DONE`

**Worker Output Format**:

Worker must end its response with the completion token:

```
COMPLETED_TASK

=== Evidence of Work ===
Files created/modified:
- <path> (<line count> lines)

Verification commands run (I have actually run these):
$ <command>
<real_terminal_output_of_command>

=== END ===
```

### 3. Critic (`system/critic.py`)

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
|--------|----------|--------------|
| `@file:` | `@file: docs/GUIDE.md exists (>200 lines)` | Check file existence and line count |
| `@functional:` | `@functional: pytest tests/test.py -v exits 0` | Run command, check exit code |
| `@metric:` | `@metric: Test count >= 5` | Parse output, validate metric |
| `@docs:` | `@docs: Explains Agent System` | Human review (manual check) |
| `@integration:` | `@integration: Macro features used in tests` | Code analysis |

**Critic Output Schema**:
```json
{
  "decision": "APPROVE|REJECT",
  "reason": "All criteria verified" or specific rejection reason,
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

### 4. StateManager (`system/core/state.py`)

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

**Race Condition Protection** (v1.1 fix):
```python
# Don't allow PENDING_REVIEW if task has rejection feedback
if status == "PENDING_REVIEW" and old_feedback and "Reject" in old_feedback:
    return  # Block - let Worker handle rejection first
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

#### ModelRegistry Class (`sirena/models/registry.py`)

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
            if not issubclass(model_class, BaseForecaster):
                raise TypeError(f"{model_class.__name__} must inherit from BaseForecaster")
            if name in cls._models:
                raise ValueError(f"Model '{name}' already registered")
            model_class.name = name
            cls._models[name] = model_class
            return model_class
        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> BaseForecaster:
        """Retrieve model instance by name."""
        if name not in cls._models:
            available = ", ".join(cls._models.keys()) or "no models"
            raise KeyError(f"Model '{name}' not found. Available: {available}")
        return cls._models[name](**kwargs)

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
        """Train model on historical data."""
        X, y = self._prepare_features(df, target_col)

        from sklearn.linear_model import Lasso
        self.model = Lasso(alpha=self.alpha)
        self.model.fit(X, y)

        self._is_fitted = True
        self._last_train_date = df.index[-1]
        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Generate predictions for specified horizon."""
        self._check_fitted()

        last_date = self._last_train_date
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                  periods=horizon, freq='MS')
        X_future = self._prepare_future_features(future_dates)
        predictions = self.model.predict(X_future)
        return predictions

    def backtest(self, df: pd.DataFrame, start_date: str = '2019-01-01',
                target_col: str = 'Все товары и услуги') -> pd.DataFrame:
        """Run historical backtest."""
        results = []

        test_dates = df.loc[start_date:].index
        for date in test_dates:
            train = df.loc[:date - pd.DateOffset(months=1)]
            self.fit(train, target_col)
            pred = self.forecast(horizon=1)[0]

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
model_class = ModelRegistry.get_class("my_model")

# Instantiate and train
model = model_class(alpha=0.5)
model.fit(df, 'Все товары и услуги')

# Forecast
forecast = model.forecast(horizon=12)

# List all available models
all_models = ModelRegistry.list_models()
print(f"Available models: {all_models}")
# Output: ['ridge', 'bvar', 'lightgbm', 'my_model', ...]

# Get model info
info = ModelRegistry.info()
print(f"Model count: {info['model_count']}")
print(f"Default weights: {info['default_weights']}")
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
- forecast() method
- predict() method
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

For testing full pipeline:

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

## Code Standards

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `RidgeForecaster`, `EnsembleForecaster`)
- **Functions**: `snake_case` (e.g., `fit_model`, `calculate_mae`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MIN_TRAIN_SIZE`, `DEFAULT_HORIZON`)
- **Private methods**: `_leading_underscore` (e.g., `_prepare_features`)

### Docstring Format

```python
def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'RidgeForecaster':
    """
    Train model on historical data.

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
    """Train model with proper error handling."""

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

### Data Mining Protocol

For "Honest" Extraction (Tasks 113, 114, etc.):

1. **Safety First**: Never ping thousands of URLs blindly. Always implement rate limiting or local parsing first.
2. **Verify, Don't Assume**:
   - ❌ Bad Criterion: "Parse file X"
   - ✅ Good Criterion: "Output `data/result.csv` exists AND size > 100KB AND has > 1000 rows"
3. **Sample the Goods**: The Critic agent MUST read the first 5 lines of any generated CSV (`head -n 5`) to confirm the data is not garbage.
4. **Handle Huge Files**: For >100MB files, never use `pd.read_csv()` without `chunksize`. Prove memory safety.

---

## Task Development Workflow

### Creating a New Task

1. **Edit `tasks/prd.json`**:
   ```json
   {
     "id": 600,
     "title": "My New Feature",
     "priority": "medium",
     "status": "TODO",
     "description": "Brief description of the task",
     "acceptance_criteria": [
       "@file: my_feature.py exists (>50 lines)",
       "@functional: python3 my_feature.py exits 0",
       "@metric: Output contains expected data"
     ]
   }
   ```

2. **Acceptance Criteria Best Practices**:
   - Use machine-verifiable criteria (`@file:`, `@functional:`, `@metric:`)
   - Minimum 3 criteria per task
   - Include escape hatch for impossible tasks: "OR documented limitation"

### Worker Task Execution

When Worker executes a task:

1. **Research Phase**:
   ```bash
   # List relevant files
   ls -la target_dir

   # Read existing code
   head -100 file.py

   # State approach BEFORE implementing
   ```

2. **Implementation Phase**:
   - Work ONLY in: `/home/valalav/_projects/sirena-kbr/edge_lab`
   - DO NOT modify files outside this directory
   - Use 'Red-Green-Refactor' method
   - Create REAL output files (not just print statements)

3. **Self-Verification Phase**:
   ```bash
   # Verify coding tasks
   python3 -m py_compile my_file.py
   pytest test_file.py -v

   # Verify data tasks
   head -5 output.csv
   wc -l output.csv
   ```

4. **Completion Format**:
   ```
   COMPLETED_TASK

   === Evidence of Work ===
   Files created/modified:
   - <path> (<line count> lines)

   Verification commands run (I have actually run these):
   $ <command>
   <real_terminal_output_of_command>

   === END ===
   ```

### Critic Task Review

When Critic reviews a task:

1. **Fetch PENDING_REVIEW tasks** from `prd.json`
2. **Verify each criterion**:
   - `@file:`: Check existence and line count with `ls` and `wc -l`
   - `@functional:`: Run command, check exit code
   - `@metric:`: Parse output, validate metric
   - `@docs:`: Manual review (minimal usage)
   - `@integration:`: Code analysis with `grep` or `find`

3. **Make decision**:
   - If all passed: Mark as `DONE`
   - If any failed: Mark as `TODO` with specific feedback

4. **Generate feedback**:
   - Explain which criteria failed
   - Provide specific command output showing failure
   - Be precise and actionable

### Handling Blocked Tasks

After 3 failed attempts, task is marked `BLOCKED`:

1. **Refiner decomposes** into subtasks
2. **Each subtask** becomes a new `TODO` task
3. **Original task** reference kept in `parent_id`

---

## Configuration

### Constants (`system/config.py`)

```python
# Project paths
PROJECT_ROOT = Path("/home/valalav/_projects/sirena-kbr/edge_lab")
PRD_FILE = PROJECT_ROOT / "tasks" / "prd.json"
PROGRESS_FILE = PROJECT_ROOT / "tasks" / "progress.txt"

# Process configuration
MAX_ITERATIONS = 1000  # Was 50, now allows ~33 min runs (v1.2 fix)
MAX_TASK_DURATION_SECONDS = 1800  # 30 minutes

# Retry configuration
MAX_RETRIES = 3
TASK_RETRY_COUNTS: Dict[int, int] = {}

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

---

## Resources

### Documentation
- [AGENTS.md](../AGENTS.md) - Agent directives and protocols
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Architecture and lessons learned
- [SKILLS_GUIDE.md](./SKILLS_GUIDE.md) - Available skills for complex tasks
- [opencode_reference.md](./opencode_reference.md) - Tool reference

### Code References
- [sirena/models/base.py](../sirena/models/base.py) - BaseForecaster interface
- [sirena/models/registry.py](../sirena/models/registry.py) - ModelRegistry implementation
- [system/orchestrator.py](../system/orchestrator.py) - Agent orchestrator
- [system/worker.py](../system/worker.py) - Worker agent
- [system/critic.py](../system/critic.py) - Critic agent
- [system/core/state.py](../system/core/state.py) - Thread-safe state management

### External Tools
- [pytest](https://docs.pytest.org/) - Testing framework
- [streamlit](https://streamlit.io/) - Dashboard framework
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
- [scikit-learn](https://scikit-learn.org/) - ML library

---

**Last Updated**: 2025-01-25
**Version**: 1.2
**Maintainers**: Edge Lab Development Team
