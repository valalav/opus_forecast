---
name: add-model
description: Use this skill when the user wants to add a new forecasting model to the system. It scaffolds the model file, registers it, and updates the dashboard.
---

# Add Model Skill

**Goal**: Scaffold all necessary files and configurations to add a new model to SIRENA-KBR.

## Instructions
1.  **Analyze Request**: Identify the `ModelName` (PascalCase) and the base algorithm (e.g., "Ridge", "Lasso", "RandomForest").
2.  **Run Scaffolding**: Execute `scripts/add_model.py` with the model name.
3.  **Verify**:
    - Check if `sirena/models/{model_name}.py` exists.
    - Check if `sirena/models/__init__.py` exports it.
    - Check if `dashboard.py` imports it and adds it to `ALL_MODELS`.

## Usage
```bash
python edge_lab/skills/add_model/scripts/add_model.py --name "MyNewModel" --base "Ridge"
```
