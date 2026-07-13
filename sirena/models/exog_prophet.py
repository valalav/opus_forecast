"""
ExogProphet — Prophet с экзогенными переменными для h=12
=========================================================

Optimized for MAE <= 0.30 on h=1 backtest:
- usd_lag1, usd_roc1: USD lag 1 + rate of change
- brent_lag2, brent_roc2: Brent lag 2 + rate of change
- ki_lag3: Ki lag 3

Improvements:
- Shorter lags for more responsive forecasts
- Rate-of-change features capture momentum
- Additive seasonality for stability
- Smaller changepoint prior scale
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
import warnings
import logging

warnings.filterwarnings("ignore")

from .base import BaseForecaster
from .registry import ModelRegistry

# Prophet check
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
except ImportError:
    PROPHET_AVAILABLE = False


@ModelRegistry.register("exog_prophet")
class ExogProphetForecaster(BaseForecaster):
    """
    Prophet с экзогенными переменными.

    Оптимизирован для h=12 (годовая траектория).
    Использует лаги экзогенных из тестов Грейнджера.
    """

    name = "exog_prophet"
    MIN_TRAIN_SIZE = 36  # Минимум 3 года для h=12

    # Optimized lags for MAE <= 0.30
    USD_LAG = 1
    BRENT_LAG = 2
    KI_LAG = 3

    def __init__(
        self,
        use_usd: bool = True,
        use_brent: bool = True,
        use_ki: bool = True,
        yearly_seasonality: bool = True,
        seasonality_mode: str = "additive",
        changepoint_prior_scale: float = 0.01,
        seasonality_prior_scale: float = 1.0,
        outlier_years: List[int] = None,
        **kwargs,
    ):
        """
        Args:
            use_usd: Использовать USD lag-1
            use_brent: Использовать Brent lag-2
            use_ki: Использовать Ki lag-3
            yearly_seasonality: Годовая сезонность
            seasonality_mode: 'additive' или 'multiplicative'
            changepoint_prior_scale: Prior для изменений тренда (меньше = стабильнее)
            seasonality_prior_scale: Prior для сезонности
            outlier_years: Годы-выбросы для исключения
        """
        super().__init__(**kwargs)

        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not installed. Run: pip install prophet")

        self.use_usd = use_usd
        self.use_brent = use_brent
        self.use_ki = use_ki
        self.yearly_seasonality = yearly_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.outlier_years = outlier_years or [2022]

        self.model = None
        self.last_date = None
        self.macro_df = None
        self.brent_df = None
        self.regressors = []

    def _load_macro_data(self) -> pd.DataFrame:
        """Загрузка макроданных из inflation_data.csv."""
        data_path = Path(__file__).parent.parent.parent / "data" / "inflation_data.csv"

        df = pd.read_csv(data_path, sep=";", decimal=",")
        df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
        df = df.set_index("Date").sort_index()

        return df

    def _load_brent_data(self) -> pd.DataFrame:
        """Загрузка данных Brent."""
        brent_path = Path(__file__).parent.parent.parent / "data" / "brent_prices.csv"

        if not brent_path.exists():
            return None

        df = pd.read_csv(brent_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

        return df

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовка признаков с лагами и темпами роста.

        Создает:
        - usd_lag1, usd_roc1: USD lag 1 + rate of change
        - brent_lag2, brent_roc2: Brent lag 2 + rate of change
        - ki_lag3: Ki lag 3
        """
        result = df.copy()

        # USD lag-1 + rate of change
        if self.use_usd and "usd_nom_i" in result.columns:
            result["usd_lag1"] = result["usd_nom_i"].shift(self.USD_LAG)
            result["usd_lag1"] = (result["usd_lag1"] - 100) / 10
            result["usd_roc1"] = result["usd_nom_i"].diff(1).shift(self.USD_LAG)
            result["usd_roc1"] = result["usd_roc1"].ffill().fillna(0)
            # Ensure no NaN in USD columns
            result["usd_lag1"] = result["usd_lag1"].ffill().fillna(0)

        # Ki lag-3
        if self.use_ki and "Ki" in result.columns:
            result["ki_lag3"] = result["Ki"].shift(self.KI_LAG)
            result["ki_lag3"] = result["ki_lag3"] / 10
            result["ki_lag3"] = result["ki_lag3"].ffill().fillna(0)

        # Brent lag-2 + rate of change
        if self.use_brent and self.brent_df is not None:
            result = result.join(self.brent_df[["brent"]], how="left")
            result["brent_lag2"] = result["brent"].shift(self.BRENT_LAG)
            result["brent_lag2"] = result["brent_lag2"] / 100
            result["brent_roc2"] = result["brent"].diff(1).shift(self.BRENT_LAG)
            result["brent_roc2"] = result["brent_roc2"].ffill().fillna(0)
            result = result.drop("brent", axis=1, errors="ignore")

        return result

    def _prepare_prophet_df(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Преобразование в формат Prophet с регрессорами."""
        # Target
        if target_col in df.columns:
            series = df[target_col].dropna()
        else:
            series = df.dropna()

        # Convert from index to percentage points
        if series.mean() > 50:
            series = series - 100

        prophet_df = pd.DataFrame({"ds": series.index, "y": series.values})

        # Add regressors
        self.regressors = []

        # USD features
        if self.use_usd and "usd_lag1" in df.columns:
            prophet_df["usd_lag1"] = df.loc[series.index, "usd_lag1"].values
            self.regressors.append("usd_lag1")
        if self.use_usd and "usd_roc1" in df.columns:
            prophet_df["usd_roc1"] = df.loc[series.index, "usd_roc1"].values
            self.regressors.append("usd_roc1")

        # Ki features
        if self.use_ki and "ki_lag3" in df.columns:
            prophet_df["ki_lag3"] = df.loc[series.index, "ki_lag3"].values
            self.regressors.append("ki_lag3")

        # Brent features
        if self.use_brent and "brent_lag2" in df.columns:
            prophet_df["brent_lag2"] = df.loc[series.index, "brent_lag2"].values
            self.regressors.append("brent_lag2")
        if self.use_brent and "brent_roc2" in df.columns:
            prophet_df["brent_roc2"] = df.loc[series.index, "brent_roc2"].values
            self.regressors.append("brent_roc2")

        # Only add regressors that are present
        valid_regressors = [r for r in self.regressors if r in prophet_df.columns]
        prophet_df = prophet_df[["ds", "y"] + valid_regressors]
        self.regressors = valid_regressors

        return prophet_df

    def fit(
        self, df: pd.DataFrame, target_col: str = "Все товары и услуги"
    ) -> "ExogProphetForecaster":
        """
        Обучение ExogProphet.

        Args:
            df: DataFrame с таргетом (можно передать любой, макро загрузится отдельно)
            target_col: Колонка с целевой переменной
        """
        # Load macro data
        self.macro_df = self._load_macro_data()
        self.brent_df = self._load_brent_data()

        # Prepare features with lags
        prepared_df = self._prepare_features(self.macro_df)

        # Convert target to prophet format
        prophet_df = self._prepare_prophet_df(prepared_df, "mom")

        # Exclude outlier years
        prophet_df["year"] = prophet_df["ds"].dt.year
        prophet_df = prophet_df[~prophet_df["year"].isin(self.outlier_years)]
        prophet_df = prophet_df.drop("year", axis=1)

        # Drop rows with NaN regressors (due to lags)
        prophet_df = prophet_df.dropna()

        self.last_date = prophet_df["ds"].max()

        # Create Prophet model
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            mcmc_samples=0,
        )

        # Add monthly seasonality
        self.model.add_seasonality(
            name="monthly",
            period=30.5,
            fourier_order=3,
        )

        # Add regressors
        for reg in self.regressors:
            self.model.add_regressor(reg, standardize=False)

        # Fit
        self.model.fit(prophet_df)

        self._is_fitted = True
        self._last_train_date = self.last_date

        return self

    def _forecast_exog(self, horizon: int) -> pd.DataFrame:
        """
        Прогноз экзогенных переменных на horizon месяцев.

        Использует naive (last value) для простоты.
        """
        from .exog_forecaster import ExogForecaster

        try:
            ef = ExogForecaster(
                ki_method="adaptive", usd_method="adaptive", brent_method="ar1"
            )
            ef.fit(self.macro_df, self.brent_df)
            return ef.forecast(horizon)
        except Exception:
            # Fallback: naive forecast
            return None

    def _prepare_future_regressors(
        self, future: pd.DataFrame, horizon: int
    ) -> pd.DataFrame:
        """
        Подготовка регрессоров для будущих дат.

        Логика лагов:
        - Для t+1: используем actual данные с нужным лагом
        - Для t+12: часть данных actual, часть forecasted
        """
        result = future.copy()

        # Get historical data
        hist_usd = (
            self.macro_df["usd_nom_i"].values
            if "usd_nom_i" in self.macro_df.columns
            else None
        )
        hist_ki = self.macro_df["Ki"].values if "Ki" in self.macro_df.columns else None
        hist_brent = (
            self.brent_df["brent"].values if self.brent_df is not None else None
        )

        # Get exog forecast
        exog_fc = self._forecast_exog(
            horizon + max(self.USD_LAG, self.BRENT_LAG, self.KI_LAG)
        )

        # Fill regressors for future dates
        future_mask = result["ds"] > self.last_date
        n_future = future_mask.sum()

        if "usd_lag2" in self.regressors:
            usd_values = []
            for i, row in result[future_mask].iterrows():
                # How many months ahead is this?
                months_ahead = (row["ds"].year - self.last_date.year) * 12 + (
                    row["ds"].month - self.last_date.month
                )

                # Lag-2 means we need data from months_ahead - 2
                data_offset = months_ahead - self.USD_LAG

                if data_offset <= 0:
                    # Use historical data
                    idx = len(hist_usd) + data_offset - 1
                    if idx >= 0:
                        val = (hist_usd[idx] - 100) / 10
                    else:
                        val = 0
                else:
                    # Use forecasted data
                    if exog_fc is not None and data_offset <= len(exog_fc):
                        val = (exog_fc.iloc[data_offset - 1]["usd_nom_i"] - 100) / 10
                    else:
                        val = 0
                usd_values.append(val)

            result.loc[future_mask, "usd_lag2"] = usd_values

        if "ki_lag6" in self.regressors:
            ki_values = []
            for i, row in result[future_mask].iterrows():
                months_ahead = (row["ds"].year - self.last_date.year) * 12 + (
                    row["ds"].month - self.last_date.month
                )
                data_offset = months_ahead - self.KI_LAG

                if data_offset <= 0:
                    idx = len(hist_ki) + data_offset - 1
                    if idx >= 0 and not np.isnan(hist_ki[idx]):
                        val = hist_ki[idx] / 10
                    else:
                        # Use last known value
                        val = (
                            hist_ki[~np.isnan(hist_ki)][-1] / 10
                            if len(hist_ki[~np.isnan(hist_ki)]) > 0
                            else 1.6
                        )
                else:
                    if exog_fc is not None and data_offset <= len(exog_fc):
                        val = exog_fc.iloc[data_offset - 1]["Ki"] / 10
                    else:
                        val = (
                            hist_ki[~np.isnan(hist_ki)][-1] / 10
                            if hist_ki is not None
                            and len(hist_ki[~np.isnan(hist_ki)]) > 0
                            else 1.6
                        )
                ki_values.append(val)

            result.loc[future_mask, "ki_lag6"] = ki_values

        if "brent_lag5" in self.regressors and hist_brent is not None:
            brent_values = []
            for i, row in result[future_mask].iterrows():
                months_ahead = (row["ds"].year - self.last_date.year) * 12 + (
                    row["ds"].month - self.last_date.month
                )
                data_offset = months_ahead - self.BRENT_LAG

                if data_offset <= 0:
                    idx = len(hist_brent) + data_offset - 1
                    if idx >= 0:
                        val = hist_brent[idx] / 100
                    else:
                        val = 0.7
                else:
                    if exog_fc is not None and data_offset <= len(exog_fc):
                        val = exog_fc.iloc[data_offset - 1]["Brent"] / 100
                    else:
                        val = hist_brent[-1] / 100
                brent_values.append(val)

            result.loc[future_mask, "brent_lag5"] = brent_values

        return result

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Прогноз ExogProphet."""
        self._check_fitted()

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=horizon, freq="MS")

        # Add regressor columns
        for reg in self.regressors:
            future[reg] = np.nan

        # Fill historical regressor values
        prepared_df = self._prepare_features(self.macro_df)
        hist_mask = future["ds"] <= self.last_date

        for reg in self.regressors:
            if reg in prepared_df.columns:
                # Match dates
                for idx, row in future[hist_mask].iterrows():
                    if row["ds"] in prepared_df.index:
                        future.loc[idx, reg] = prepared_df.loc[row["ds"], reg]

        # Fill future regressor values
        future = self._prepare_future_regressors(future, horizon)

        # Handle any remaining NaNs
        for reg in self.regressors:
            future[reg] = future[reg].fillna(method="ffill").fillna(0)

        # Predict
        forecast = self.model.predict(future)

        forecast_future = forecast[forecast["ds"] > self.last_date]
        return forecast_future["yhat"].values

    def forecast_with_intervals(self, horizon: int = 12) -> Dict[str, Any]:
        """Прогноз с интервалами."""
        self._check_fitted()

        future = self.model.make_future_dataframe(periods=horizon, freq="MS")

        # Add and fill regressors
        for reg in self.regressors:
            future[reg] = np.nan

        prepared_df = self._prepare_features(self.macro_df)
        hist_mask = future["ds"] <= self.last_date

        for reg in self.regressors:
            if reg in prepared_df.columns:
                for idx, row in future[hist_mask].iterrows():
                    if row["ds"] in prepared_df.index:
                        future.loc[idx, reg] = prepared_df.loc[row["ds"], reg]

        future = self._prepare_future_regressors(future, horizon)

        for reg in self.regressors:
            future[reg] = future[reg].fillna(method="ffill").fillna(0)

        forecast = self.model.predict(future)
        forecast_future = forecast[forecast["ds"] > self.last_date]

        return {
            "mean": forecast_future["yhat"].values,
            "lower": forecast_future["yhat_lower"].values,
            "upper": forecast_future["yhat_upper"].values,
            "dates": forecast_future["ds"].values,
            "trend": forecast_future["trend"].values,
        }

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """
        Прогноз на конкретную дату.

        Args:
            df: DataFrame (не используется, данные из внутреннего состояния)
            target_date: Целевая дата прогноза

        Returns:
            Dict с 'prediction' и метаданными
        """
        self._check_fitted()

        # Calculate horizon
        horizon = (target_date.year - self.last_date.year) * 12 + (
            target_date.month - self.last_date.month
        )

        if horizon <= 0:
            raise ValueError(
                f"Target date {target_date} must be after last training date {self.last_date}"
            )

        fc = self.forecast(horizon)

        return {
            "prediction": fc[-1] + 100,  # Convert back to index format
            "horizon": horizon,
            "regressors_used": self.regressors,
            "last_train_date": self.last_date,
        }

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = "2019-01-01",
        target_col: str = "Все товары и услуги",
        horizon: int = 1,
    ) -> pd.DataFrame:
        """Бэктестирование ExogProphet."""
        results = []

        # Load data
        macro_df = self._load_macro_data()
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

            # Need enough data
            train_data = macro_df[macro_df.index <= train_end]
            if len(train_data) < self.MIN_TRAIN_SIZE:
                continue

            try:
                # Create and fit model on training data only
                model = ExogProphetForecaster(
                    use_usd=self.use_usd,
                    use_brent=self.use_brent,
                    use_ki=self.use_ki,
                    yearly_seasonality=self.yearly_seasonality,
                    seasonality_mode=self.seasonality_mode,
                    outlier_years=self.outlier_years,
                )

                # Override macro_df with truncated data
                model.macro_df = train_data
                model.brent_df = (
                    brent_df[brent_df.index <= train_end]
                    if brent_df is not None
                    else None
                )

                # Prepare and fit
                prepared_df = model._prepare_features(model.macro_df)
                prophet_df = model._prepare_prophet_df(prepared_df, "mom")

                prophet_df["year"] = prophet_df["ds"].dt.year
                prophet_df = prophet_df[~prophet_df["year"].isin(model.outlier_years)]
                prophet_df = prophet_df.drop("year", axis=1)
                prophet_df = prophet_df.dropna()

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
                continue

        return pd.DataFrame(results)

    def get_regressor_importance(self) -> Dict[str, float]:
        """Получить важность регрессоров."""
        if self.model is None:
            return {}

        # Extract coefficients from model params
        importance = {}
        for reg in self.regressors:
            if reg in self.model.params:
                coef = (
                    self.model.params[reg][0][0]
                    if isinstance(self.model.params[reg], list)
                    else self.model.params[reg]
                )
                importance[reg] = abs(float(coef))

        return importance
