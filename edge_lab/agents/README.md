# Agents Package

Autonomous agents for the Opus Autopoiesis system.

## Available Agents

### HypothesisGenerator (🧬 The Spark)

The Hypothesis Generator Agent brainstorms new correlation theories and appends tasks to the PRD.

**Location:** `agents/hypothesis_generator.py`

**Features:**
- Generates hypotheses about economic correlations (lag effects, volatility, momentum, interactions)
- Converts hypotheses to PRD task format
- Checks for duplicate tasks before appending
- Atomic PRD updates with temporary file pattern

**Usage:**

```python
from agents.hypothesis_generator import HypothesisGenerator
from pathlib import Path

# Initialize generator
generator = HypothesisGenerator(prd_path=Path("tasks/prd.json"))

# Generate hypotheses
hypotheses = generator.generate_hypotheses(count=5)

# Brainstorm and append to PRD
added_tasks = generator.brainstorm(count=3)
```

**Hypothesis Types:**
1. **Lag effects** - Test if variable with lag N improves MAE
2. **Volatility** - Test rolling std as feature
3. **Momentum** - Test first difference as feature
4. **Interactions** - Test variable combinations
5. **Thresholds** - Test nonlinear threshold effects
6. **Nonlinear** - Test log/square/sqrt transformations

**Variables Analyzed:**
- `usd_nom_i` - USD/RUB exchange rate
- `Ki_i` - Key rate (ЦБ РФ)
- `Ruonia` - Ruonia rate
- `brent` - Brent oil price
- `mom` - Inflation (MoM)
- `Prod` - Food prices
- `Nonprod` - Non-food prices
- `Serv` - Services prices

## Testing

Run tests with pytest:

```bash
python3 -m pytest agents/test_hypothesis_generator.py -v
```

All 11 tests should pass.

## Acceptance Criteria

✅ Agent appends new tasks to PRD
✅ Tests verify all functionality
✅ Atomic file operations prevent corruption
✅ Duplicate detection prevents redundant tasks

---

### ImmuneSystemTester (🛡️ The Shield)

The Immune System Agent generates synthetic "Black Swan" events to test model survival and resilience.

**Location:** `agents/immune_system.py`

**Features:**
- Generates 6 types of black swan events (extreme values, regime changes, missing data, etc.)
- Tests model resilience with configurable survival criteria
- Measures survival rate and identifies vulnerabilities
- Produces comprehensive markdown reports

**Black Swan Event Types:**
1. **EXTREME_VALUE** - Massive spike/drop in target variable
2. **REGIME_CHANGE** - Sudden shift in data distribution
3. **MISSING_DATA** - Simulate data gaps
4. **FEATURE_OUTLIER** - Extreme values in exogenous features
5. **VOLATILITY_EXPLOSION** - Sudden increase in noise/uncertainty
6. **CONSECUTIVE_SHOCKS** - Multiple consecutive extreme shocks

**Survival Criteria:**
- Model doesn't crash or raise exceptions
- Predictions within reasonable bounds (e.g., -5% to +10% MoM)
- MAE degradation below threshold (default 2x baseline)
- No NaN or infinite predictions

**Usage:**

```python
from agents.immune_system import ImmuneSystemTester, create_sample_model

# Create tester
tester = ImmuneSystemTester(
    target_col='target',
    survival_threshold_mae=2.0,
    prediction_bounds=(-5.0, 10.0)
)

# Stress test a single model
model = create_sample_model("MyModel")
report = tester.stress_test_model(model, train_data)

print(f"Survival Rate: {report.survival_rate:.1f}%")
print(f"Passed: {report.passed_tests}/{report.total_tests}")

# Test multiple models
models = [create_sample_model(f"Model{i}") for i in range(5)]
reports = tester.test_models(models, train_data)

# Generate comprehensive report
report_text = tester.generate_report(reports, output_path="stress_test_report.md")
```

**Testing:**

Run tests with pytest:

```bash
python3 -m pytest tests/test_immune_system.py -v
```

Run verification script:

```bash
python3 verify_immune_system.py
```

## Acceptance Criteria

✅ Immune System tests model resilience
✅ Survival Rate > 90% achieved
✅ All 23 unit tests pass
✅ Generates synthetic black swan events
✅ Identifies model vulnerabilities
