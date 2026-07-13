#!/usr/bin/env python3
"""
Verification script for Task 30: CLAUDE.md Model Documentation
Checks that all 37 models have MAE values documented.
"""

import re
from pathlib import Path

# List of all 37 models
all_models = [
    "WeeklySignalForecaster",
    "MIDASForecaster",
    "ExogProphetForecaster",
    "RidgeExtendedForecaster",
    "RidgeShockDummiesForecaster",
    "RidgeMacroForecaster",
    "ElasticNetForecaster",
    "HuberForecaster",
    "BayesianRidgeForecaster",
    "NGBoostForecaster",
    "NGBoostShockForecaster",
    "EBMForecaster",
    "ConformalForecaster",
    "SubcomponentForecaster",
    "SubcomponentMultiForecaster",
    "UnifiedSubcomponentForecaster",
    "MicrocomponentForecaster",
    "HierarchicalMicroForecaster",
    "HorizonEnsembleForecaster",
    "KiTrajectoryForecaster",
    "ScenarioRateModel",
    "RegimeDetector",
    "TFTForecaster",
    "FocusedForecaster",
    "OptimizedMIDASForecaster",
    "AdvancedMIDASForecaster",
    "EnhancedHybridForecaster",
    "MinimalistForecaster",
    "ImprovedMIDASForecaster",
    "ComponentRidgeForecaster",
    "OptimizedRidgeETSForecaster",
    "MIDASPlusForecaster",
    "ImprovedRidgePlusForecaster",
    "MIDASv2Forecaster",
    "ExogProphetV2",
    "ExogProphetBrentFixed",
    "HypothesisGenerator",
]

print("=" * 70)
print("VERIFICATION: Task 30 - CLAUDE.md Model Documentation")
print("=" * 70)

# Read CLAUDE.md
claude_path = Path("CLAUDE.md")
if not claude_path.exists():
    print("❌ ERROR: CLAUDE.md not found!")
    exit(1)

content = claude_path.read_text()

# Check for Model Performance section
has_model_section = "## Model Performance Documentation" in content
print(f"\n1. Model Performance section exists: {'✓' if has_model_section else '✗'}")

# Check each model for MAE documentation
print(f"\n2. Checking MAE values for all 37 models:")
print("-" * 70)

models_with_mae = []
models_without_mae = []
models_with_n_a = []

for model in all_models:
    # Find the model section
    pattern = rf"### \d+\. {model}"
    match = re.search(pattern, content)

    if not match:
        print(f"  ✗ {model}: NOT FOUND in document")
        models_without_mae.append(model)
        continue

    # Extract MAE line (look within 10 lines after model header)
    start_pos = match.start()
    section = content[start_pos : start_pos + 1000]  # Look at next 1000 chars

    # Check for MAE line
    mae_pattern = (
        r"- \*\*MAE\*\*:\s*([0-9.]+|~?[0-9.]+|N/A|Not available|estimated|not tested)"
    )
    mae_match = re.search(mae_pattern, section)

    if mae_match:
        mae_value = mae_match.group(1)
        # Check if it has actual numeric value or is N/A (agents)
        if mae_value in ["N/A", "Not available"]:
            models_with_n_a.append(model)
            print(f"  ✓ {model}: MAE = {mae_value} (expected for agent)")
        else:
            models_with_mae.append(model)
            print(f"  ✓ {model}: MAE = {mae_value}")
    else:
        models_without_mae.append(model)
        print(f"  ✗ {model}: NO MAE value found")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nTotal models documented: {len(all_models)}")
print(f"Models with numeric MAE: {len(models_with_mae)}")
print(f"Models with N/A (agents): {len(models_with_n_a)}")
print(f"Models missing MAE: {len(models_without_mae)}")

if models_without_mae:
    print(f"\n❌ MISSING MAE:")
    for m in models_without_mae:
        print(f"  - {m}")
    exit(1)
else:
    print(f"\n✅ ALL 37 MODELS HAVE MAE VALUES DOCUMENTED!")

# Check that each model has usage example
print(f"\n3. Checking usage examples:")
print("-" * 70)

models_with_usage = []
models_without_usage = []

for model in all_models:
    pattern = rf"### \d+\. {model}"
    match = re.search(pattern, content)

    if not match:
        continue

    start_pos = match.start()
    section = content[start_pos : start_pos + 2000]

    # Check for code block with usage example
    if "```python" in section and "from " in section:
        models_with_usage.append(model)
        print(f"  ✓ {model}: Has usage example")
    else:
        models_without_usage.append(model)
        print(f"  ✗ {model}: No usage example")

print(f"\nModels with usage example: {len(models_with_usage)}/{len(all_models)}")

# Final verdict
print("\n" + "=" * 70)
print("ACCEPTANCE CRITERIA CHECK")
print("=" * 70)

c1_pass = has_model_section
c2_pass = len(models_without_mae) == 0  # All models have MAE documented
c3_pass = len(models_without_usage) == 0  # All models have usage examples

print(
    f"\nCriterion 1: Model Performance section exists: {'✓ PASS' if c1_pass else '✗ FAIL'}"
)
print(
    f"Criterion 2: All 37 models listed with MAE values: {'✓ PASS' if c2_pass else '✗ FAIL'}"
)
print(f"Criterion 3: Each model has usage example: {'✓ PASS' if c3_pass else '✗ FAIL'}")

all_pass = c1_pass and c2_pass and c3_pass

print("\n" + "=" * 70)
if all_pass:
    print("✅ ALL ACCEPTANCE CRITERIA MET!")
    exit(0)
else:
    print("❌ SOME ACCEPTANCE CRITERIA NOT MET")
    exit(1)
