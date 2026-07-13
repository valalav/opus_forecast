"""
ExogProphet Fix - Date Alignment Patch
=====================================
Wrapper class to fix Brent date alignment issue in ExogProphetForecaster
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.models.exog_prophet import ExogProphetForecaster


class ExogProphetBrentFixed(ExogProphetForecaster):
    """
    ExogProphetForecaster with fixed Brent date alignment.

    The original model has a date alignment issue:
    - brent_prices.csv: 2010-01-01, 2010-02-01, ... (1st of month)
    - inflation_data.csv: 2010-01-31, 2010-02-28, ... (end of month)

    This wrapper reindexes Brent data to match macro data dates.
    """

    def _load_brent_data_fixed(self) -> pd.DataFrame:
        """Load and reindex Brent data to match macro dates."""
        brent_path = Path(__file__).parent.parent.parent / "data" / "brent_prices.csv"

        if not brent_path.exists():
            return None

        # Load original brent data
        df = pd.read_csv(brent_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

        # Load macro data to get correct dates
        macro_df = self._load_macro_data()

        # Reindex brent to match macro dates
        # For each macro date (e.g., 2010-01-31), use brent from same month (e.g., 2010-01-01)
        df_reindexed = df.reindex(macro_df.index, method="ffill")

        return df_reindexed

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features with lags, using fixed Brent data.
        """
        result = df.copy()

        # Use fixed Brent loader
        fixed_brent_df = self._load_brent_data_fixed()

        # USD lag-2
        if self.use_usd and "usd_nom_i" in result.columns:
            result["usd_lag2"] = result["usd_nom_i"].shift(self.USD_LAG)
            result["usd_lag2"] = (result["usd_lag2"] - 100) / 10

        # Ki lag-6
        if self.use_ki and "Ki" in result.columns:
            result["ki_lag6"] = result["Ki"].shift(self.KI_LAG)
            result["ki_lag6"] = result["ki_lag6"] / 10

        # Brent lag-5 (using fixed aligned data)
        if self.use_brent and fixed_brent_df is not None:
            # Merge with fixed brent data
            result = result.join(fixed_brent_df[["brent"]], how="left")
            result["brent_lag5"] = result["brent"].shift(self.BRENT_LAG)
            result["brent_lag5"] = result["brent_lag5"] / 100
            result = result.drop("brent", axis=1, errors="ignore")

        return result

    def _load_brent_data(self) -> pd.DataFrame:
        """Override to use fixed version."""
        return self._load_brent_data_fixed()
