import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.getcwd())

import sirena.models
from sirena.models.registry import ModelRegistry

print("Registered Models:")
models = ModelRegistry.list_models()
for m in models:
    print(f"- {m}")

print("\nDefault Weights:")
print(ModelRegistry.get_all_weights())
