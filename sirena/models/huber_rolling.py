from typing import Any, Dict, List, Optional, cast

import pandas as pd

from .huber import HuberForecaster
from .registry import ModelRegistry


@ModelRegistry.register("huber_rolling_24m")
class HuberRollingForecaster(HuberForecaster):
    """Huber с rolling seasonality по последним N месяцам."""

    name = "huber_rolling_24m"
    OUTLIER_YEARS = [2010]
    BASE_FEATURES = [
        "y_lag1",
        "y_lag2",
        "y_lag12",
        "y_lag3",
        "y_lag6",
        "y_ma3",
        "y_ma6",
        "d_y_lag1",
        "d_y_lag3",
        "y_vol3",
        "y_vol6",
        "month_sin",
        "month_cos",
        "quarter_sin",
        "quarter_cos",
        "is_jan",
        "is_dec",
        "is_tariff_month",
        "is_q1",
        "is_summer",
        "food_lag1",
        "nonfood_lag1",
        "services_lag1",
        "seasonal_norm",
        "deviation_lag1",
    ]
    MACRO_FEATURES = [
        "ruonia_diff_lag1",
        "spread_lag4",
        "ki_diff_lag6",
        "ki_vol",
    ]
    ETS_WEIGHTS = {
        1: 0.9,
        2: 0.0,
        3: 0.5,
        4: 0.3,
        5: 0.9,
        6: 0.5,
        7: 0.0,
        8: 0.5,
        9: 0.9,
        10: 0.9,
        11: 0.0,
        12: 0.0,
    }

    def __init__(
        self,
        epsilon: float = 1.35,
        alpha: float = 0.3,
        max_iter: int = 500,
        use_macro: bool = True,
        seasonality_window: int = 24,
        **kwargs: Any,
    ):
        super().__init__(
            epsilon=epsilon,
            alpha=alpha,
            max_iter=max_iter,
            use_macro=use_macro,
            **kwargs,
        )
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iter = max_iter
        self.use_macro = use_macro
        self.seasonality_window = seasonality_window

    def _compute_seasonal_norm(self, df: pd.DataFrame) -> pd.Series:
        cutoff_date = cast(pd.Timestamp, df.index.max()) - pd.DateOffset(
            months=self.seasonality_window
        )
        recent_df = cast(pd.DataFrame, df[df.index >= cutoff_date])
        clean_df = cast(
            pd.DataFrame, recent_df[~recent_df["year"].isin(self.OUTLIER_YEARS)]
        )
        if len(clean_df) < 12:
            clean_df = recent_df
        seasonal_norm = clean_df.groupby("month")["Все товары и услуги"].mean()
        return cast(pd.Series, seasonal_norm)

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        result = cast(Dict[str, Any], super().predict(df, target_date))
        result["seasonality_window"] = self.seasonality_window
        return result

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = "2019-01-01",
        target_col: str = "Все товары и услуги",
    ) -> pd.DataFrame:
        start = pd.Timestamp(start_date)
        valid_dates = df.dropna(subset=[target_col]).index
        test_dates = valid_dates[valid_dates >= start]

        results = []
        for target_date in test_dates:
            train_df = cast(pd.DataFrame, df[df.index < target_date].copy())
            target_subset = cast(pd.DataFrame, train_df[[target_col]].dropna())
            if len(target_subset) < self.MIN_TRAIN_SIZE:
                continue

            try:
                model = HuberRollingForecaster(
                    epsilon=self.epsilon,
                    alpha=self.alpha,
                    max_iter=self.max_iter,
                    use_macro=self.use_macro,
                    seasonality_window=self.seasonality_window,
                )
                model.fit(train_df, target_col)
                test_df = cast(pd.DataFrame, df[df.index <= target_date].copy())
                pred_result = model.predict(test_df, cast(pd.Timestamp, target_date))
                actual = df.loc[target_date, target_col]
                results.append(
                    {
                        "date": target_date,
                        "actual": actual,
                        "prediction": pred_result["prediction"],
                        "error": actual - pred_result["prediction"],
                        "pred_huber": pred_result["pred_huber"],
                        "scale": pred_result.get("scale"),
                        "has_macro": pred_result.get("has_macro", False),
                        "seasonality_window": pred_result.get("seasonality_window"),
                    }
                )
            except Exception:
                continue

        return pd.DataFrame(results)
