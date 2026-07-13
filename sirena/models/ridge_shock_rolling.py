"""
Ridge Shock Rolling
===================

Производная модель поверх RidgeShockDummies:
- сохраняет shock-dummy логику исходной модели;
- заменяет глобальную сезонную норму на rolling seasonality;
- не изменяет исходный `ridge_shock_dummies.py`.
"""

from typing import Any, Dict, Optional, cast

import pandas as pd

from .registry import ModelRegistry
from .ridge_shock_dummies import RidgeShockDummiesForecaster


@ModelRegistry.register("ridge_shock_rolling_24m")
class RidgeShockRollingForecaster(RidgeShockDummiesForecaster):
    """RidgeShockDummies с rolling seasonality на последних N месяцах."""

    name = "ridge_shock_rolling_24m"
    OUTLIER_YEARS = [2010]

    def __init__(
        self,
        alpha: Optional[float] = None,
        use_macro: bool = True,
        use_2022_dummy: bool = True,
        seasonality_window: int = 24,
        **kwargs,
    ):
        super().__init__(
            alpha=alpha,
            use_macro=use_macro,
            use_2022_dummy=use_2022_dummy,
            **kwargs,
        )
        self.alpha: float = self.alpha
        self.use_macro: bool = self.use_macro
        self.use_2022_dummy: bool = self.use_2022_dummy
        self.seasonality_window = seasonality_window

    def _compute_seasonal_norm(self, df: pd.DataFrame) -> pd.Series:
        """Вычислить сезонную норму на последних `seasonality_window` месяцах."""
        max_index = cast(pd.Timestamp, df.index.max())
        cutoff_date = max_index - pd.DateOffset(months=self.seasonality_window)
        recent_df = cast(pd.DataFrame, df[df.index >= cutoff_date].copy())
        year_mask = ~pd.Series(recent_df["year"], index=recent_df.index).isin(
            self.OUTLIER_YEARS
        )
        clean_df = cast(pd.DataFrame, recent_df.loc[year_mask].copy())

        if len(clean_df) < 12:
            clean_df = recent_df

        seasonal_norm = clean_df.groupby("month")["Все товары и услуги"].mean()
        return cast(pd.Series, seasonal_norm)

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = "2019-01-01",
        target_col: str = "Все товары и услуги",
    ) -> pd.DataFrame:
        """Бэктестирование производной модели с сохранением её параметров."""
        start = pd.Timestamp(start_date)

        valid_dates = df.dropna(subset=[target_col]).index
        test_dates = valid_dates[valid_dates >= start]

        results = []

        for target_date in test_dates:
            train_df = cast(pd.DataFrame, df[df.index < target_date].copy())

            if len(train_df[[target_col]].dropna()) < self.MIN_TRAIN_SIZE:
                continue

            try:
                model = RidgeShockRollingForecaster(
                    alpha=self.alpha,
                    use_macro=self.use_macro,
                    use_2022_dummy=self.use_2022_dummy,
                    seasonality_window=self.seasonality_window,
                )
                model.fit(train_df, target_col)

                test_df = cast(pd.DataFrame, df[df.index <= target_date].copy())
                pred_result = model.predict(test_df, target_date)
                actual = df.loc[target_date, target_col]

                results.append(
                    {
                        "date": target_date,
                        "actual": actual,
                        "prediction": pred_result["prediction"],
                        "error": actual - pred_result["prediction"],
                        "pred_ridge": pred_result["pred_ridge"],
                        "has_macro": pred_result.get("has_macro", False),
                        "use_2022_dummy": pred_result.get("use_2022_dummy", False),
                        "seasonality_window": self.seasonality_window,
                    }
                )
            except Exception:
                continue

        return pd.DataFrame(results)

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """Точечный прогноз с добавлением служебного признака окна сезонности."""
        result = super().predict(df, target_date)
        result["seasonality_window"] = self.seasonality_window
        return result
