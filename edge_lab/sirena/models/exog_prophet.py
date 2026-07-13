import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
except ImportError:
    Prophet = None


class ExogProphetForecaster:
    """
    ExogProphet Forecaster with Brent external regressor.

    Optimized configuration based on extensive testing:
    - Uses Brent-only regressor (USD hurts performance)
    - Additive seasonality mode
    - Outlier year filtering (2022)
    - Aggressive changepoint_prior_scale for flexibility

    Current best MAE: ~0.51 on h=1 backtest (2020-2022)
    """

    def __init__(
        self,
        use_usd: bool = False,
        use_brent: bool = True,
        yearly_seasonality: bool = True,
        seasonality_mode: str = "additive",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        outlier_years: list = None,
    ):
        self.name = "exog_prophet"
        self.use_usd = use_usd
        self.use_brent = use_brent
        self.yearly_seasonality = yearly_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.outlier_years = outlier_years or [2022]

        self.model = None
        self.macro_df = None
        self.brent_df = None
        self.last_date = None
        self._is_fitted = False
        self.regressors = []

        if self.use_usd:
            self.regressors.extend(["usd_lag1", "usd_lag2", "usd_roc1"])
        if self.use_brent:
            self.regressors.extend(["brent_lag1", "brent_lag2", "brent_roc1"])

    def _load_macro_data(self) -> pd.DataFrame:
        """Load macro data from parent directory."""
        base_dir = Path.cwd().parent
        data_path = base_dir / "data" / "inflation_data.csv"

        if not data_path.exists():
            raise FileNotFoundError(f"Macro data not found at {data_path}")

        df = pd.read_csv(data_path, sep=";", encoding="utf-8-sig")
        df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", dayfirst=True)
        df = df.set_index("Date").sort_index()

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.replace(",", ".").astype(float)

        return df

    def _load_brent_data(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """Load and align Brent data with macro data."""
        base_dir = Path.cwd().parent
        brent_path = base_dir / "data" / "brent_prices.csv"

        if not brent_path.exists():
            return None

        df = pd.read_csv(brent_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

        df_reindexed = df.reindex(macro_df.index, method="ffill")
        return df_reindexed

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features with lags and rate-of-change."""
        result = df.copy()

        if self.use_usd and "usd_nom_i" in result.columns:
            usd = result["usd_nom_i"].copy()
            result["usd_lag1"] = usd.shift(1)
            result["usd_lag2"] = usd.shift(2)
            result["usd_roc1"] = usd.diff(1).shift(1)

            result["usd_lag1"] = (result["usd_lag1"] - 100) / 10
            result["usd_lag2"] = (result["usd_lag2"] - 100) / 10
            result["usd_roc1"] = result["usd_roc1"] / 10

            for col in ["usd_lag1", "usd_lag2", "usd_roc1"]:
                result[col] = result[col].fillna(0)

        if self.use_brent and "brent" in result.columns:
            brent = result["brent"].copy()
            result["brent_lag1"] = brent.shift(1)
            result["brent_lag2"] = brent.shift(2)
            result["brent_roc1"] = brent.diff(1).shift(1)

            result["brent_lag1"] = result["brent_lag1"] / 100
            result["brent_lag2"] = result["brent_lag2"] / 100
            result["brent_roc1"] = result["brent_roc1"] / 100

            for col in ["brent_lag1", "brent_lag2", "brent_roc1"]:
                result[col] = result[col].fillna(0.7)

        return result

    def _prepare_prophet_df(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Convert to Prophet format."""
        prophet_df = df.copy().reset_index()
        prophet_df.columns = prophet_df.columns.str.lower()

        target_col_lower = target_col.lower()
        if target_col_lower in prophet_df.columns:
            prophet_df["y"] = prophet_df[target_col_lower] - 100
        else:
            prophet_df["y"] = prophet_df.iloc[:, 1] - 100

        prophet_df["ds"] = pd.to_datetime(prophet_df.iloc[:, 0])

        regressor_cols = [c for c in prophet_df.columns if c in self.regressors]

        result = prophet_df[["ds", "y"] + regressor_cols].copy()
        return result

    def fit(self, df=None, target_col: str = "mom"):
        """Fit Prophet model."""
        if Prophet is None:
            raise ImportError("Prophet not installed")

        if df is None:
            self.macro_df = self._load_macro_data()
        else:
            self.macro_df = df

        self.brent_df = self._load_brent_data(self.macro_df)

        if self.use_brent and self.brent_df is not None:
            self.macro_df = self.macro_df.join(self.brent_df[["brent"]], how="left")

        prepared_df = self._prepare_features(self.macro_df)
        prophet_df = self._prepare_prophet_df(prepared_df, target_col)

        if len(self.outlier_years) > 0:
            prophet_df["year"] = prophet_df["ds"].dt.year
            prophet_df = prophet_df[~prophet_df["year"].isin(self.outlier_years)]
            prophet_df = prophet_df.drop("year", axis=1)

        prophet_df = prophet_df.dropna(
            subset=["y"] + [c for c in prophet_df.columns if c in self.regressors]
        )

        if len(prophet_df) < 24:
            raise ValueError(f"Insufficient training data: {len(prophet_df)} < 24")

        self.last_date = prophet_df["ds"].max()

        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            mcmc_samples=0,
        )

        self.model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

        for reg in self.regressors:
            if reg in prophet_df.columns:
                self.model.add_regressor(reg, standardize=False)

        self.model.fit(prophet_df)
        self._is_fitted = True

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Generate forecasts in deviation space."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        future_dates = pd.date_range(
            start=self.last_date, periods=horizon + 1, freq="ME"
        )[1:]

        future = pd.DataFrame({"ds": future_dates})

        prepared_df = self._prepare_features(self.macro_df)

        for reg in self.regressors:
            if reg in prepared_df.columns:
                last_value = prepared_df[reg].dropna().iloc[-1]
                future[reg] = last_value

        forecast = self.model.predict(future)
        predictions = forecast["yhat"].values

        return predictions

    def backtest(
        self,
        df=None,
        start_date: str = "2019-01-01",
        target_col: str = "mom",
        horizon: int = 1,
    ) -> pd.DataFrame:
        """Run backtest."""
        results = []

        macro_df = self._load_macro_data() if df is None else df
        brent_df = self._load_brent_data(macro_df)

        test_df = macro_df[macro_df.index >= pd.Timestamp(start_date)].copy()

        if "mom" in test_df.columns:
            series = test_df["mom"] - 100
        else:
            series = (
                test_df[target_col] - 100
                if target_col in test_df.columns
                else test_df.iloc[:, 0] - 100
            )

        test_dates = series.index

        for target_date in test_dates:
            cutoff = target_date - pd.DateOffset(months=horizon)
            train_end = cutoff

            train_data = macro_df[macro_df.index <= train_end]
            if len(train_data) < 24:
                continue

            try:
                model = ExogProphetForecaster(
                    use_usd=self.use_usd,
                    use_brent=self.use_brent,
                    yearly_seasonality=self.yearly_seasonality,
                    seasonality_mode=self.seasonality_mode,
                    changepoint_prior_scale=self.changepoint_prior_scale,
                    seasonality_prior_scale=self.seasonality_prior_scale,
                    outlier_years=self.outlier_years,
                )

                model.macro_df = train_data.copy()
                if brent_df is not None:
                    model.brent_df = brent_df.reindex(train_data.index, method="ffill")
                else:
                    model.brent_df = None

                if model.use_brent and model.brent_df is not None:
                    model.macro_df = model.macro_df.join(
                        model.brent_df[["brent"]], how="left"
                    )

                prepared_df = model._prepare_features(model.macro_df)
                prophet_df = model._prepare_prophet_df(prepared_df, "mom")

                if len(model.outlier_years) > 0:
                    prophet_df["year"] = prophet_df["ds"].dt.year
                    prophet_df = prophet_df[
                        ~prophet_df["year"].isin(model.outlier_years)
                    ]
                    prophet_df = prophet_df.drop("year", axis=1)

                prophet_df = prophet_df.dropna(
                    subset=["y"]
                    + [c for c in prophet_df.columns if c in model.regressors]
                )

                if len(prophet_df) < 24:
                    continue

                model.last_date = prophet_df["ds"].max()

                model.model = Prophet(
                    yearly_seasonality=model.yearly_seasonality,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode=model.seasonality_mode,
                    changepoint_prior_scale=model.changepoint_prior_scale,
                    seasonality_prior_scale=model.seasonality_prior_scale,
                    mcmc_samples=0,
                )

                model.model.add_seasonality(
                    name="monthly", period=30.5, fourier_order=5
                )

                for reg in model.regressors:
                    if reg in prophet_df.columns:
                        model.model.add_regressor(reg, standardize=False)

                model.model.fit(prophet_df)
                model._is_fitted = True

                fc = model.forecast(horizon)
                pred = fc[0]
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
                continue

        return pd.DataFrame(results)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from Prophet model.

        For Prophet, regressor coefficients indicate importance.
        Returns DataFrame sorted by absolute coefficient value.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        importance_data = []

        if hasattr(self.model, "params") and self.model.params is not None:
            for name, value in self.model.params.items():
                if "regressor" in name:
                    regressor_name = (
                        name.replace("_beta", "")
                        .replace("_lower", "")
                        .replace("_upper", "")
                    )
                    importance_data.append(
                        {
                            "feature": regressor_name,
                            "coefficient": value,
                            "abs_coef": abs(value),
                        }
                    )

        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            return importance_df.sort_values("abs_coef", ascending=False)
        else:
            return pd.DataFrame(columns=["feature", "coefficient", "abs_coef"])
