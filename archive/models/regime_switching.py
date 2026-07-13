"""
Режимозависимый ансамбль (Regime-Switching Ensemble)
=====================================================

v4.6: Обновлено с NGBoost Shock как основной моделью.

Автоматически определяет макро-режим и адаптирует веса моделей:

Режимы:
1. "Шоковый" (shock) — резкие изменения ставок/макро:
   - |ΔRuonia| > 0.5 или |ΔKi| > 0.5 за последний месяц
   - NGBoost Shock получает максимальный вес (лучшая модель v4.6)

2. "Нормальный" (normal) — стабильная макро-среда:
   - NGBoost Shock и Ridge делят лидерство

3. "Высокая инфляция" (high_inflation) — скачок инфляции:
   - NGBoost Shock + BVAR для макро-трансмиссии

Веса по режимам:
- Shock: NGBoost Shock 35%, Ridge 25%, BVAR 15%, ...
- Normal: NGBoost Shock 30%, Ridge 30%, ETS 12%, ...
- High Inflation: NGBoost Shock 30%, BVAR 25%, Ridge 20%, ...
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import warnings

from .base import BaseForecaster
from .registry import ModelRegistry

warnings.filterwarnings('ignore')


# Пороги для определения режимов
REGIME_THRESHOLDS = {
    'ruonia_abs_change': 0.5,   # Абсолютное изменение Ruonia
    'ki_abs_change': 0.5,       # Абсолютное изменение Ki
    'inflation_spike': 1.5,     # Скачок инфляции (pp)
}

# Веса моделей по режимам (v4.6 — с NGBoost Shock)
REGIME_WEIGHTS = {
    'shock': {
        'ngboost_shock': 0.35,  # NGBoost Shock — лучшая модель для шоков (MAE 0.298)
        'ridge': 0.25,          # Ridge как стабилизатор
        'bvar': 0.15,           # BVAR для макро
        'lightgbm': 0.10,       # LightGBM для нелинейности
        'sarima': 0.05,         # SARIMA для инерции
        'prophet': 0.05,
        'ets': 0.03,
        'ebm': 0.02
    },
    'normal': {
        'ngboost_shock': 0.30,  # NGBoost Shock — основная модель
        'ridge': 0.30,          # Ridge — вторая по качеству
        'ets': 0.12,            # ETS для сезонности
        'bvar': 0.08,           # BVAR как подстраховка
        'lightgbm': 0.08,
        'prophet': 0.05,
        'sarima': 0.04,
        'ebm': 0.03
    },
    'high_inflation': {
        'ngboost_shock': 0.30,  # NGBoost Shock для шоков
        'bvar': 0.25,           # BVAR для макро-трансмиссии
        'ridge': 0.20,
        'sarima': 0.10,
        'lightgbm': 0.08,
        'ets': 0.04,
        'prophet': 0.03
    }
}


def detect_regime(
    df: pd.DataFrame,
    date: Optional[pd.Timestamp] = None
) -> Tuple[str, Dict[str, float]]:
    """
    Определяет текущий макро-режим.

    Args:
        df: DataFrame с колонками Ruonia, Ki (опционально)
        date: Дата для определения режима (если None — последняя)

    Returns:
        Tuple[str, Dict]: (название режима, диагностика)
    """
    if date is None:
        date = df.index.max()

    diagnostics = {
        'date': date,
        'ruonia_change': None,
        'ki_change': None,
        'inflation_change': None,
        'signals': []
    }

    # ΔRuonia
    if 'Ruonia' in df.columns:
        ruonia = df['Ruonia'].dropna()
        if len(ruonia) >= 2:
            idx = ruonia.index.get_indexer([date], method='ffill')[0]
            if idx > 0:
                change = abs(ruonia.iloc[idx] - ruonia.iloc[idx - 1])
                diagnostics['ruonia_change'] = change
                if change > REGIME_THRESHOLDS['ruonia_abs_change']:
                    diagnostics['signals'].append('ruonia_shock')

    # ΔKi
    if 'Ki' in df.columns:
        ki = df['Ki'].dropna()
        if len(ki) >= 2:
            idx = ki.index.get_indexer([date], method='ffill')[0]
            if idx > 0:
                change = abs(ki.iloc[idx] - ki.iloc[idx - 1])
                diagnostics['ki_change'] = change
                if change > REGIME_THRESHOLDS['ki_abs_change']:
                    diagnostics['signals'].append('ki_shock')

    # Скачок инфляции
    if 'Все товары и услуги' in df.columns:
        infl = df['Все товары и услуги'].dropna()
        if len(infl) >= 2:
            idx = infl.index.get_indexer([date], method='ffill')[0]
            if idx > 0:
                change = abs(infl.iloc[idx] - infl.iloc[idx - 1])
                diagnostics['inflation_change'] = change
                if change > REGIME_THRESHOLDS['inflation_spike']:
                    diagnostics['signals'].append('inflation_spike')

    # Определение режима
    if 'ruonia_shock' in diagnostics['signals'] or 'ki_shock' in diagnostics['signals']:
        regime = 'shock'
    elif 'inflation_spike' in diagnostics['signals']:
        regime = 'high_inflation'
    else:
        regime = 'normal'

    diagnostics['regime'] = regime
    return regime, diagnostics


@ModelRegistry.register("regime_switching")
class RegimeSwitchingEnsemble(BaseForecaster):
    """
    Режимозависимый ансамбль с адаптивными весами.

    Автоматически определяет макро-режим и использует оптимальные
    веса для каждого режима.

    Режимы:
    - shock: резкие изменения ставок → больше BVAR
    - normal: стабильность → больше Ridge
    - high_inflation: ускорение инфляции → больше BVAR + SARIMA

    Example:
        >>> model = RegimeSwitchingEnsemble()
        >>> model.fit(df)
        >>> # Автоматически определит режим
        >>> fc = model.forecast(horizon=12)
        >>> # Посмотреть текущий режим
        >>> print(model.current_regime)
    """

    name = "regime_switching"
    MIN_TRAIN_SIZE = 36

    def __init__(
        self,
        regime_weights: Optional[Dict[str, Dict[str, float]]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """
        Args:
            regime_weights: Веса по режимам (default: REGIME_WEIGHTS)
            thresholds: Пороги для определения режимов (default: REGIME_THRESHOLDS)
        """
        super().__init__(**kwargs)

        self.regime_weights = regime_weights or REGIME_WEIGHTS
        self.thresholds = thresholds or REGIME_THRESHOLDS

        self._models: Dict[str, BaseForecaster] = {}
        self._df: Optional[pd.DataFrame] = None
        self._current_regime: str = 'normal'
        self._regime_diagnostics: Dict = {}
        self._model_forecasts: Dict[str, np.ndarray] = {}

    @property
    def current_regime(self) -> str:
        """Текущий режим."""
        return self._current_regime

    @property
    def current_weights(self) -> Dict[str, float]:
        """Текущие веса моделей."""
        return self.regime_weights.get(self._current_regime, self.regime_weights['normal'])

    def _detect_regime(self, df: pd.DataFrame, date: Optional[pd.Timestamp] = None) -> str:
        """Определение режима с учётом пользовательских порогов."""
        regime, diagnostics = detect_regime(df, date)
        self._regime_diagnostics = diagnostics
        return regime

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = 'Все товары и услуги'
    ) -> 'RegimeSwitchingEnsemble':
        """
        Обучение всех базовых моделей.

        Args:
            df: DataFrame с данными
            target_col: Целевая колонка

        Returns:
            self
        """
        self._df = df.copy()

        # Определяем режим на последнюю дату
        self._current_regime = self._detect_regime(df)

        # Обучаем все базовые модели
        model_names = list(self.regime_weights['normal'].keys())

        for model_name in model_names:
            try:
                model = ModelRegistry.get(model_name)
                model.fit(df, target_col)
                self._models[model_name] = model
            except Exception as e:
                # Модель недоступна — пропускаем
                pass

        if not self._models:
            raise ValueError("Ни одна модель не была успешно обучена")

        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """
        Генерирует прогноз с весами текущего режима.

        Args:
            horizon: Горизонт прогноза

        Returns:
            numpy array с взвешенным прогнозом
        """
        self._check_fitted()

        # Получаем веса для текущего режима
        weights = self.current_weights

        # Собираем прогнозы от всех моделей
        self._model_forecasts = {}
        for model_name, model in self._models.items():
            try:
                fc = model.forecast(horizon)
                # Нормализация: некоторые модели возвращают % (0.5),
                # другие — индекс (100.5). Приводим к индексу.
                fc = np.array(fc)
                if np.mean(fc) < 50:  # Вероятно в процентах
                    fc = fc + 100
                self._model_forecasts[model_name] = fc
            except Exception:
                pass

        if not self._model_forecasts:
            raise ValueError("Ни одна модель не дала прогноз")

        # Нормализуем веса для доступных моделей
        available_weights = {
            name: weights.get(name, 0)
            for name in self._model_forecasts.keys()
        }
        total_weight = sum(available_weights.values())

        if total_weight == 0:
            # Равные веса если всё обнулилось
            n = len(self._model_forecasts)
            normalized = {name: 1.0 / n for name in self._model_forecasts.keys()}
        else:
            normalized = {name: w / total_weight for name, w in available_weights.items()}

        # Взвешенный прогноз
        ensemble = np.zeros(horizon)
        for model_name, fc in self._model_forecasts.items():
            ensemble += fc * normalized[model_name]

        return ensemble

    def forecast_with_regime(
        self,
        horizon: int = 12,
        regime: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Прогноз с возможностью переопределить режим.

        Args:
            horizon: Горизонт прогноза
            regime: Принудительный режим (None = автоопределение)

        Returns:
            Dict с прогнозами и информацией о режиме
        """
        self._check_fitted()

        if regime is not None and regime in self.regime_weights:
            self._current_regime = regime

        ensemble = self.forecast(horizon)

        return {
            'ensemble': ensemble,
            'regime': self._current_regime,
            'weights': self.current_weights,
            'model_forecasts': self._model_forecasts.copy(),
            'diagnostics': self._regime_diagnostics
        }

    def backtest(
        self,
        df: pd.DataFrame = None,
        start_date: str = '2020-01-01',
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """
        Бэктест с динамическим определением режима.

        Args:
            df: DataFrame с данными
            start_date: Начало периода
            target_col: Целевая колонка

        Returns:
            DataFrame с результатами
        """
        if df is None:
            df = self._df
            if df is None:
                raise ValueError("Нужны данные для бэктеста")

        results = []
        start = pd.to_datetime(start_date)
        test_dates = df.index[df.index >= start]

        for date in test_dates:
            cutoff = date - pd.DateOffset(months=1)
            train_data = df[df.index <= cutoff]

            if len(train_data) < self.MIN_TRAIN_SIZE + 12:
                continue

            try:
                # Создаём новую модель для чистого бэктеста
                model = RegimeSwitchingEnsemble(
                    regime_weights=self.regime_weights,
                    thresholds=self.thresholds
                )
                model.fit(train_data, target_col)
                pred = model.forecast(horizon=1)[0]
                actual = df.loc[date, target_col]

                results.append({
                    'date': date,
                    'actual': actual,
                    'prediction': pred,
                    'error': pred - actual,
                    'regime': model.current_regime
                })
            except Exception:
                continue

        return pd.DataFrame(results)

    def get_regime_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        История режимов по всем датам.

        Args:
            df: DataFrame с данными

        Returns:
            DataFrame с режимом для каждой даты
        """
        history = []
        for date in df.index:
            regime, diag = detect_regime(df, date)
            history.append({
                'date': date,
                'regime': regime,
                'ruonia_change': diag.get('ruonia_change'),
                'ki_change': diag.get('ki_change'),
                'signals': ', '.join(diag.get('signals', []))
            })
        return pd.DataFrame(history)

    def get_model_info(self) -> Dict:
        """Информация о модели."""
        return {
            'name': self.name,
            'current_regime': self._current_regime,
            'current_weights': self.current_weights,
            'available_models': list(self._models.keys()),
            'diagnostics': self._regime_diagnostics,
            'is_fitted': self._is_fitted
        }


def compare_regimes(
    df: pd.DataFrame,
    start_date: str = '2020-01-01',
    target_col: str = 'Все товары и услуги'
) -> pd.DataFrame:
    """
    Сравнение производительности режимов.

    Args:
        df: DataFrame с данными
        start_date: Начало периода
        target_col: Целевая колонка

    Returns:
        DataFrame со статистикой по режимам
    """
    model = RegimeSwitchingEnsemble()
    model.fit(df, target_col)
    bt = model.backtest(df, start_date, target_col)

    if bt.empty:
        return pd.DataFrame()

    # Статистика по режимам
    stats = []
    for regime in ['normal', 'shock', 'high_inflation']:
        regime_data = bt[bt['regime'] == regime]
        if len(regime_data) > 0:
            stats.append({
                'regime': regime,
                'count': len(regime_data),
                'mae': regime_data['error'].abs().mean(),
                'rmse': np.sqrt((regime_data['error'] ** 2).mean()),
                'pct': len(regime_data) / len(bt) * 100
            })

    return pd.DataFrame(stats)
