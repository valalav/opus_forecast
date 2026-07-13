"""
Dashboard Utilities for SIRENA-KBR.

Extracted utility functions for dashboard.py to improve code organization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

ALL_MODELS = [
    "Ridge",
    "Ridge_Ext",
    "Bayes_Ridge",
    "ElasticNet",
    "Huber",
    "Ridge_Shock",
    "Ridge_Macro",
    "NGBoost",
    "NGBoost_Shock",
    "BVAR",
    "SARIMA",
    "LightGBM",
    "Prophet",
    "ETS",
    "EBM",
    "CatBoost",
    "Subcomp",
    "Subcomp_Multi",
    "Micro",
    "Micro_SM",
    "Ensemble",
]

MODEL_COLORS = {
    "Ridge": "#1f77b4",
    "Ridge_Ext": "#aec7e8",
    "Bayes_Ridge": "#ff7f0e",
    "ElasticNet": "#ffbb78",
    "Huber": "#2ca02c",
    "Ridge_Shock": "#98df8a",
    "Ridge_Macro": "#2ecc71",
    "NGBoost": "#d62728",
    "NGBoost_Shock": "#ff9896",
    "BVAR": "#9467bd",
    "SARIMA": "#c5b0d5",
    "LightGBM": "#8c564b",
    "Prophet": "#c49c94",
    "ETS": "#e377c2",
    "EBM": "#f7b6d2",
    "CatBoost": "#7f7f7f",
    "Subcomp": "#c7c7c7",
    "Subcomp_Multi": "#bcbd22",
    "Micro": "#17becf",
    "Micro_SM": "#1f9e89",
    "Ensemble": "#000000",
    "Actual": "#000000",
    "Факт": "#000000",
}

MONTH_NAMES_RU = [
    "Январь", "Февраль", "Март", "Апрель",
    "Май", "Июнь", "Июль", "Август",
    "Сентябрь", "Октябрь", "Ноябрь", "Декабрь",
]


# =============================================================================
# DATA UTILITIES
# =============================================================================

def load_inflation_data(filepath: str = "data/inflation_data.csv") -> pd.DataFrame:
    """Load inflation data from CSV file."""
    df = pd.read_csv(filepath)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    return df


def calculate_mom_change(series: pd.Series) -> pd.Series:
    """Calculate month-over-month change."""
    return series.diff()


def calculate_yoy_change(series: pd.Series) -> pd.Series:
    """Calculate year-over-year change."""
    return series.diff(12)


def get_last_n_months(df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Get last N months of data."""
    return df.tail(n)


# =============================================================================
# METRICS CALCULATIONS
# =============================================================================

def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(actual[mask] - predicted[mask]))


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2))


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    mask = ~(np.isnan(actual) | np.isnan(predicted)) & (actual != 0)
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def calculate_kpi_violations(errors: np.ndarray, threshold: float = 0.5) -> int:
    """Count KPI violations (errors exceeding threshold)."""
    mask = ~np.isnan(errors)
    return int(np.sum(np.abs(errors[mask]) > threshold))


def calculate_coverage(errors: np.ndarray, threshold: float = 0.5) -> float:
    """Calculate coverage rate (% of errors within threshold)."""
    mask = ~np.isnan(errors)
    if mask.sum() == 0:
        return np.nan
    return (np.abs(errors[mask]) <= threshold).mean() * 100


# =============================================================================
# FORMATTING UTILITIES
# =============================================================================

def format_date_ru(date: datetime) -> str:
    """Format date in Russian locale."""
    if pd.isna(date):
        return ""
    month_idx = date.month - 1
    return f"{MONTH_NAMES_RU[month_idx]} {date.year}"


def format_percent(value: float, decimals: int = 2) -> str:
    """Format value as percentage string."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}%"


def format_number(value: float, decimals: int = 3) -> str:
    """Format number with specified decimals."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


# =============================================================================
# MODEL RANKING
# =============================================================================

def rank_models_by_mae(
    predictions: pd.DataFrame,
    actual_col: str = "Actual",
    ascending: bool = True
) -> pd.DataFrame:
    """Rank models by MAE."""
    model_cols = [c for c in predictions.columns if c not in [actual_col, 'Date']]

    results = []
    for model in model_cols:
        mae = calculate_mae(
            predictions[actual_col].values,
            predictions[model].values
        )
        results.append({"model": model, "MAE": mae})

    df = pd.DataFrame(results)
    df = df.sort_values("MAE", ascending=ascending).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def get_model_color(model_name: str) -> str:
    """Get color for model, with fallback."""
    return MODEL_COLORS.get(model_name, "#808080")


# =============================================================================
# CHART HELPERS
# =============================================================================

def create_error_bands(
    center: np.ndarray,
    width: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Create upper and lower error bands."""
    return center + width, center - width


def interpolate_missing(series: pd.Series, method: str = "linear") -> pd.Series:
    """Interpolate missing values in series."""
    return series.interpolate(method=method)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_dataframe(
    df: pd.DataFrame,
    required_cols: List[str]
) -> Tuple[bool, List[str]]:
    """Validate DataFrame has required columns."""
    missing = [c for c in required_cols if c not in df.columns]
    return len(missing) == 0, missing


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Check data quality metrics."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_pct": (df.isna().sum().sum() / df.size) * 100,
        "duplicates": df.duplicated().sum(),
        "date_range": (df.index.min(), df.index.max()) if isinstance(df.index, pd.DatetimeIndex) else None,
    }


if __name__ == "__main__":
    # Quick test
    print("Dashboard Utils loaded successfully")
    print(f"Models: {len(ALL_MODELS)}")
    print(f"Colors: {len(MODEL_COLORS)}")
