"""
ExogProphet V2 - Improved with date alignment fix and optimized parameters
========================================================================

Improvements:
1. Fixed Brent date alignment (1st of month → end of month)
2. Optimized hyperparameters for better MAE
3. Better handling of NaN values
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.models.exog_prophet import ExogProphetForecaster

# Prophet import
try:
    from prophet import Prophet
except ImportError:
    Prophet = None


class ExogProphetV2(ExogProphetForecaster):
    """
    ExogProphetForecaster with date alignment fix and optimized parameters.

    Key fixes:
    1. Brent data is reindexed to match macro data dates
    2. Optimized hyperparameters for MAE <= 0.30 target
    3. Better NaN handling for regressors
    """

    def __init__(self, **kwargs):
        """Initialize with optimized parameters."""
        # Set optimized defaults for MAE <= 0.30
        defaults = {
            "use_usd": True,
            "use_brent": True,
            "use_ki": True,
            "yearly_seasonality": True,
            "seasonality_mode": "multiplicative",  # Changed for better shock handling
            "changepoint_prior_scale": 0.5,  # Increased for more flexibility with shocks
            "seasonality_prior_scale": 20.0,  # Increased for better seasonality
            "outlier_years": [],  # Don't exclude any years - let model learn from shocks
        }
        # Override with any user-provided kwargs
        defaults.update(kwargs)

        super().__init__(**defaults)

    def _load_brent_data(self) -> pd.DataFrame:
        """
        Load Brent data with fixed date alignment.

        Overrides parent method to fix date alignment:
        - Brent data: 2010-01-01, 2010-02-01, ... (1st of month)
        - Macro data: 2010-01-31, 2010-02-28, ... (end of month)
        """
        brent_path = Path(__file__).parent.parent / "data" / "brent_prices.csv"

        if not brent_path.exists():
            return None

        # Load original brent data
        df = pd.read_csv(brent_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

        # Load macro data to get correct dates
        macro_df = self._load_macro_data()

        # Reindex brent to match macro dates using forward fill
        # For each macro date (e.g., 2010-01-31), use brent from same month (e.g., 2010-01-01)
        df_reindexed = df.reindex(macro_df.index, method="ffill")

        return df_reindexed

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features with lags.
        Uses fixed Brent data from _load_brent_data().
        """
        result = df.copy()

        # USD lag-2
        if self.use_usd and "usd_nom_i" in result.columns:
            result["usd_lag2"] = result["usd_nom_i"].shift(self.USD_LAG)
            result["usd_lag2"] = (result["usd_lag2"] - 100) / 10
            # Fill NaN with last known value
            result["usd_lag2"] = result["usd_lag2"].fillna(method="ffill").fillna(0)

        # Ki lag-6
        if self.use_ki and "Ki" in result.columns:
            result["ki_lag6"] = result["Ki"].shift(self.KI_LAG)
            result["ki_lag6"] = result["ki_lag6"] / 10
            # Fill NaN with last known value
            result["ki_lag6"] = result["ki_lag6"].fillna(method="ffill").fillna(1.6)

        # Brent lag-5 (brent_df is now fixed with correct dates)
        if self.use_brent and self.brent_df is not None:
            # Merge brent data - now dates should match
            result = result.join(self.brent_df[["brent"]], how="left")
            result["brent_lag5"] = result["brent"].shift(self.BRENT_LAG)
            result["brent_lag5"] = result["brent_lag5"] / 100
            result = result.drop("brent", axis=1, errors="ignore")
            # Fill NaN with last known value
            result["brent_lag5"] = (
                result["brent_lag5"].fillna(method="ffill").fillna(0.7)
            )

        return result

    def backtest(
        self,
        df,
        start_date: str = "2019-01-01",
        target_col: str = "Все товары и услуги",
        horizon: int = 1,
    ) -> pd.DataFrame:
        """
        Backtest with fixed Brent date alignment.

        Overridden to ensure Brent data is properly aligned during backtest.
        """
        results = []

        # Load data
        macro_df = self._load_macro_data()

        # Load Brent data (already has fixed date alignment via _load_brent_data)
        brent_df = self._load_brent_data()

        # Get target series
        if "mom" in macro_df.columns:
            series = macro_df["mom"] - 100
        else:
            series = (
                macro_df[target_col] - 100
                if target_col in macro_df.columns
                else macro_df.iloc[:, 0] - 100
            )

        test_dates = series[series.index >= start_date].index

        for target_date in test_dates:
            cutoff = target_date - pd.DateOffset(months=horizon)
            train_end = cutoff

            train_data = macro_df[macro_df.index <= train_end]
            if len(train_data) < self.MIN_TRAIN_SIZE:
                continue

            try:
                # Create and fit model on training data only
                model = ExogProphetV2(
                    use_usd=self.use_usd,
                    use_brent=self.use_brent,
                    use_ki=self.use_ki,
                    yearly_seasonality=self.yearly_seasonality,
                    seasonality_mode=self.seasonality_mode,
                    changepoint_prior_scale=self.changepoint_prior_scale,
                    seasonality_prior_scale=self.seasonality_prior_scale,
                    outlier_years=self.outlier_years,
                )

                # Override macro_df with truncated data
                model.macro_df = train_data
                # Reindex brent to match train_data dates (important for alignment)
                if brent_df is not None:
                    model.brent_df = brent_df.reindex(train_data.index, method="ffill")
                else:
                    model.brent_df = None

                # Prepare and fit
                prepared_df = model._prepare_features(model.macro_df)
                prophet_df = model._prepare_prophet_df(prepared_df, "mom")

                prophet_df["year"] = prophet_df["ds"].dt.year
                prophet_df = prophet_df[~prophet_df["year"].isin(model.outlier_years)]
                prophet_df = prophet_df.drop("year", axis=1)

                # Drop rows with NaN in target or regressors
                prophet_df = prophet_df.dropna(
                    subset=["y"] + model.regressors if model.regressors else ["y"]
                )

                if len(prophet_df) < model.MIN_TRAIN_SIZE:
                    continue

                model.last_date = prophet_df["ds"].max()

                model.model = Prophet(
                    yearly_seasonality=model.yearly_seasonality,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode=model.seasonality_mode,
                    changepoint_prior_scale=model.changepoint_prior_scale,
                    seasonality_prior_scale=model.seasonality_prior_scale,
                )

                model.model.add_seasonality(
                    name="monthly", period=30.5, fourier_order=5
                )

                for reg in model.regressors:
                    model.model.add_regressor(reg)

                model.model.fit(prophet_df)
                model._is_fitted = True

                # Forecast
                fc = model.forecast(horizon)
                pred = fc[-1]
                actual = series.loc[target_date]

                results.append(
                    {
                        "date": target_date,
                        "actual": actual,
                        "prediction": pred,
                        "error": actual - pred,
                    }
                )

            except Exception as e:
                print(f"Error at {target_date}: {e}")
                continue

        return pd.DataFrame(results)
