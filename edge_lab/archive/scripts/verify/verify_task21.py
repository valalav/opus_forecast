#!/usr/bin/env python3
"""
Verification script for Task 21: MIDAS Model - MAE Improvement

This script runs the analysis and provides exit code for the worker.
"""

import sys
from pathlib import Path

# Run the analysis script
analysis_script = Path(__file__).parent / "verify_midas_task21_analysis.py"
exec(open(analysis_script).read())
