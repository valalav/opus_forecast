
import os
import argparse
import re
from pathlib import Path

# Paths
BASE_DIR = Path("/home/valalav/_projects/sirena-kbr")
MODELS_DIR = BASE_DIR / "sirena/models"
DASHBOARD_FILE = BASE_DIR / "dashboard.py"
BACKTEST_FRAMEWORK = BASE_DIR / "scripts/backtest_framework.py"

TEMPLATE_RIDGE = """
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sirena.models.base import BaseForecaster
from sirena.models.registry import ModelRegistry

@ModelRegistry.register("{model_name}")
class {class_name}(BaseForecaster):
    def __init__(self):
        super().__init__()
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=1.0))
        ])
        
    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги'):
        # Prepare data
        pass

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> dict:
        return {'prediction': 100.0, 'std': 0.0}
"""

def create_model_file(name: str):
    filename = f"{name.lower()}.py" # simple lower case conversion
    class_name = f"{name}Forecaster"
    filepath = MODELS_DIR / filename
    
    if filepath.exists():
        print(f"⚠️ Model file {filename} already exists. Skipping creation.")
        return

    content = TEMPLATE_RIDGE.format(model_name=name, class_name=class_name)
    filepath.write_text(content)
    print(f"✅ Created {filepath}")
    return filename

def update_init(module_name: str, class_name: str):
    init_file = MODELS_DIR / "__init__.py"
    content = init_file.read_text()
    
    import_line = f"from .{module_name.replace('.py', '')} import {class_name}"
    if import_line in content:
        print("⚠️ Import already in __init__.py")
        return

    # Add import
    lines = content.splitlines()
    last_import_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("from ."):
            last_import_idx = i
            
    lines.insert(last_import_idx + 1, import_line)
    
    # Update __all__
    # This is tricky with regex, simple append might be safer if structure allows
    # For now, let's just append the import if simpler
    
    init_file.write_text("\n".join(lines) + "\n")
    print(f"✅ Updated {init_file}")

def update_dashboard(model_name: str):
    # This requires precise parsing of dashboard.py
    # For MVP, we will print instructions as this is risky to automate without robust parsing
    print(f"⚠️ AUTOMATION LIMIT: Please manually add '{model_name}' to ALL_MODELS in dashboard.py")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Name of the model (PascalCase), e.g. SuperRidge")
    args = parser.parse_args()
    
    filename = create_model_file(args.name)
    if filename:
        update_init(filename, f"{args.name}Forecaster")
        update_dashboard(args.name)

if __name__ == "__main__":
    main()
