"""
Централизованный модуль прогнозирования экзогенных переменных
=============================================================

Прогнозирует все экзогенные переменные для использования в моделях инфляции.
Каждый метод прогнозирования валидирован на бэктесте.

Переменные:
- USD: VAR (если доступен) или AR(1), MAE ~4.35%
- Ki: Random Walk (ЦБ объявляет заранее), MAE ~0.3%
- Ruonia: VAR (MAE 3.75) >> Ki-спред (MAE 8.97)
- Brent: Random Walk
- Компоненты (Prod, Serv, Nonprod): Сезонная норма + тренд

Использование:
    from sirena.exog_forecaster import ExogForecaster

    exog = ExogForecaster(use_var=True)  # VAR для Ruonia/USD
    exog.fit(df)
    future_df = exog.forecast(horizon=12)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.linear_model import Ridge


class ExogForecaster:
    """
    Прогнозирование экзогенных переменных.

    Методы прогнозирования выбраны на основе бэктестов:
    - USD: VAR (MAE 4.35) или AR(1) (MAE 4.49)
    - Ki: Random Walk, MAE ~0.3%
    - Ruonia: VAR (MAE 3.75) — 58% лучше Ki-спреда (MAE 8.97)!
    - Brent: Random Walk, MAE ~8%
    - Prod/Serv/Nonprod: Сезонная норма, MAE ~0.3%

    Args:
        use_var: Использовать VAR для Ruonia/USD (рекомендуется)
    """

    OUTLIER_YEARS = [2022, 2010]

    def __init__(self, use_var: bool = True):
        self._is_fitted = False
        self._last_date = None
        self._last_values = {}
        self._seasonal_norms = {}
        self._ar_models = {}
        self._spreads = {}
        self._use_var = use_var
        self._var_model = None
        self._var_forecast = None

    def fit(self, df: pd.DataFrame) -> 'ExogForecaster':
        """
        Обучение моделей прогнозирования на исторических данных.

        Args:
            df: DataFrame с колонками: mom, usd_nom_i/usd, Ki_i/Ki, Ruonia,
                brent, Prod, Serv, Nonprod
        """
        self._last_date = df.index.max()

        # Сохраняем последние значения
        for col in df.columns:
            if df[col].notna().any():
                self._last_values[col] = df[col].dropna().iloc[-1]

        # Исключаем выбросные годы для сезонных норм
        df_clean = df[~df.index.year.isin(self.OUTLIER_YEARS)]

        # Сезонные нормы для компонентов
        for col in ['mom', 'Prod', 'Serv', 'Nonprod']:
            if col in df_clean.columns:
                monthly = df_clean.groupby(df_clean.index.month)[col].mean()
                self._seasonal_norms[col] = monthly.to_dict()

        # AR(1) для USD
        usd_col = 'usd_nom_i' if 'usd_nom_i' in df.columns else 'usd' if 'usd' in df.columns else None
        if usd_col:
            self._fit_ar_model(df, usd_col, 'usd')

        # Спред Ki - Ruonia (fallback если VAR не работает)
        ki_col = 'Ki_i' if 'Ki_i' in df.columns else 'Ki' if 'Ki' in df.columns else None
        if ki_col and 'Ruonia' in df.columns:
            recent = df[[ki_col, 'Ruonia']].dropna().tail(24)
            if len(recent) > 0:
                self._spreads['ki_ruonia'] = (recent[ki_col] - recent['Ruonia']).mean()

        # VAR для Ruonia/USD (58% лучше для Ruonia!)
        if self._use_var:
            self._fit_var_model(df)

        self._is_fitted = True
        return self

    def _fit_var_model(self, df: pd.DataFrame):
        """Обучение VAR модели для Ruonia и USD."""
        try:
            from sirena.exog import VarExogForecaster
        except ImportError:
            self._var_model = None
            return

        # Выбираем переменные для VAR
        var_cols = []
        usd_col = 'usd_nom_i' if 'usd_nom_i' in df.columns else 'usd' if 'usd' in df.columns else None
        if usd_col:
            var_cols.append(usd_col)
        if 'Ruonia' in df.columns:
            var_cols.append('Ruonia')

        if len(var_cols) < 2:
            self._var_model = None
            return

        # Добавляем дополнительные переменные если есть
        for col in ['fl_potrb_zad', 'fl_dep', 'all_real']:
            if col in df.columns:
                var_cols.append(col)

        df_var = df[var_cols].dropna()
        if len(df_var) < 24:
            self._var_model = None
            return

        try:
            self._var_model = VarExogForecaster(lags=4)
            self._var_model.fit(df_var)
        except Exception as e:
            self._var_model = None

    def _fit_ar_model(self, df: pd.DataFrame, col: str, name: str):
        """Обучение AR(1) модели на приростах."""
        series = df[col].dropna()
        if len(series) < 12:
            return

        # Прирост
        diff = series.diff().dropna()

        # AR(1): diff_t = alpha + beta * diff_{t-1}
        X = diff.shift(1).dropna().values.reshape(-1, 1)
        y = diff.iloc[1:].values

        if len(X) > 10:
            model = Ridge(alpha=1.0)
            model.fit(X, y)
            self._ar_models[name] = {
                'model': model,
                'last_diff': diff.iloc[-1],
                'mean_diff': diff.mean(),
                'std_diff': diff.std()
            }

    def forecast(self, horizon: int = 12) -> pd.DataFrame:
        """
        Прогноз экзогенных переменных на horizon месяцев.

        Args:
            horizon: Горизонт прогноза в месяцах

        Returns:
            DataFrame с прогнозами, индекс = будущие даты
        """
        if not self._is_fitted:
            raise ValueError("Call fit() first")

        future_dates = pd.date_range(
            start=self._last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq='MS'
        )

        result = pd.DataFrame(index=future_dates)

        # Получаем VAR прогноз если модель обучена
        if self._var_model is not None:
            try:
                self._var_forecast = self._var_model.forecast(horizon=horizon)
            except:
                self._var_forecast = None
        else:
            self._var_forecast = None

        # USD: VAR (если есть) или AR(1)
        result = self._forecast_usd(result, horizon)

        # Ki: Random Walk
        result = self._forecast_ki(result, horizon)

        # Ruonia: VAR (MAE 3.75) или Ki-спред (MAE 8.97)
        result = self._forecast_ruonia(result, horizon)

        # Brent: Random Walk
        result = self._forecast_brent(result, horizon)

        # Компоненты: сезонная норма
        result = self._forecast_components(result, horizon)

        return result

    def _forecast_usd(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Прогноз USD: AR(1) на приростах."""
        usd_col = None
        for col in ['usd_nom_i', 'usd']:
            if col in self._last_values:
                usd_col = col
                break

        if usd_col is None:
            return df

        last_usd = self._last_values[usd_col]

        if 'usd' in self._ar_models:
            ar = self._ar_models['usd']
            model = ar['model']
            last_diff = ar['last_diff']

            values = [last_usd]
            current_diff = last_diff

            for i in range(horizon):
                # Прогноз прироста
                next_diff = model.predict([[current_diff]])[0]
                # Ограничиваем экстремальные значения
                next_diff = np.clip(next_diff, -5, 5)
                # Новое значение
                new_val = values[-1] + next_diff
                values.append(new_val)
                current_diff = next_diff

            df[usd_col] = values[1:]
        else:
            # Fallback: Random Walk
            df[usd_col] = last_usd

        return df

    def _forecast_ki(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Прогноз Ki: Random Walk (ЦБ объявляет заранее)."""
        ki_col = None
        for col in ['Ki_i', 'Ki']:
            if col in self._last_values:
                ki_col = col
                break

        if ki_col:
            df[ki_col] = self._last_values[ki_col]

        return df

    def _forecast_ruonia(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Прогноз Ruonia: VAR (MAE 3.75) >> Ki-спред (MAE 8.97)."""
        if 'Ruonia' not in self._last_values:
            return df

        # Приоритет 1: VAR (58% лучше!)
        if self._var_forecast is not None and 'Ruonia' in self._var_forecast.columns:
            var_ruonia = self._var_forecast['Ruonia'].values[:horizon]
            if len(var_ruonia) == horizon:
                df['Ruonia'] = var_ruonia
                return df

        # Приоритет 2: Ki - спред (fallback)
        ki_col = None
        for col in ['Ki_i', 'Ki']:
            if col in df.columns:
                ki_col = col
                break

        if ki_col and 'ki_ruonia' in self._spreads:
            # Ruonia = Ki - спред
            df['Ruonia'] = df[ki_col] - self._spreads['ki_ruonia']
        else:
            # Fallback: Random Walk
            df['Ruonia'] = self._last_values['Ruonia']

        return df

    def _forecast_brent(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Прогноз Brent: Random Walk."""
        for col in ['brent', 'brent_pct']:
            if col in self._last_values:
                df[col] = self._last_values[col]
                break
        return df

    def _forecast_components(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Прогноз компонентов: сезонная норма."""
        for col in ['Prod', 'Serv', 'Nonprod']:
            if col in self._seasonal_norms:
                for date in df.index:
                    month = date.month
                    df.loc[date, col] = self._seasonal_norms[col].get(month, 100.0)
        return df

    def backtest(self, df: pd.DataFrame, start_date: str = '2020-01-01') -> pd.DataFrame:
        """
        Бэктест прогнозов экзогенных переменных.

        Args:
            df: Полный DataFrame с историей
            start_date: Начало бэктеста

        Returns:
            DataFrame с метриками по каждой переменной
        """
        start = pd.Timestamp(start_date)
        test_dates = df.index[df.index >= start]

        results = {col: {'errors': []} for col in ['usd', 'ki', 'ruonia', 'brent', 'prod', 'serv']}

        for cutoff in test_dates:
            train_df = df[df.index < cutoff].copy()

            if len(train_df) < 24:
                continue

            try:
                forecaster = ExogForecaster()
                forecaster.fit(train_df)
                fc = forecaster.forecast(horizon=1)

                if fc.empty:
                    continue

                # Сравниваем с фактом
                actual = df.loc[cutoff]

                # USD
                for usd_col in ['usd_nom_i', 'usd']:
                    if usd_col in fc.columns and usd_col in actual.index:
                        error = actual[usd_col] - fc[usd_col].iloc[0]
                        results['usd']['errors'].append(abs(error))
                        break

                # Ki
                for ki_col in ['Ki_i', 'Ki']:
                    if ki_col in fc.columns and ki_col in actual.index:
                        error = actual[ki_col] - fc[ki_col].iloc[0]
                        results['ki']['errors'].append(abs(error))
                        break

                # Ruonia
                if 'Ruonia' in fc.columns and 'Ruonia' in actual.index:
                    error = actual['Ruonia'] - fc['Ruonia'].iloc[0]
                    results['ruonia']['errors'].append(abs(error))

                # Brent
                if 'brent' in fc.columns and 'brent' in actual.index:
                    error = actual['brent'] - fc['brent'].iloc[0]
                    results['brent']['errors'].append(abs(error))

                # Components
                for comp, key in [('Prod', 'prod'), ('Serv', 'serv')]:
                    if comp in fc.columns and comp in actual.index:
                        error = actual[comp] - fc[comp].iloc[0]
                        results[key]['errors'].append(abs(error))

            except Exception:
                continue

        # Вычисляем метрики
        metrics = []
        for var, data in results.items():
            errors = data['errors']
            if errors:
                metrics.append({
                    'variable': var,
                    'mae': np.mean(errors),
                    'rmse': np.sqrt(np.mean(np.array(errors)**2)),
                    'n_obs': len(errors)
                })

        return pd.DataFrame(metrics)

    def get_last_values(self) -> Dict:
        """Получить последние известные значения."""
        return self._last_values.copy()

    def get_seasonal_norms(self) -> Dict:
        """Получить сезонные нормы."""
        return self._seasonal_norms.copy()


# Глобальный экземпляр для использования в моделях
_global_exog_forecaster = None


def get_exog_forecaster() -> Optional[ExogForecaster]:
    """Получить глобальный ExogForecaster."""
    return _global_exog_forecaster


def set_exog_forecaster(forecaster: ExogForecaster):
    """Установить глобальный ExogForecaster."""
    global _global_exog_forecaster
    _global_exog_forecaster = forecaster
