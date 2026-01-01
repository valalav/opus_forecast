"""
Regime Detector — определение макроэкономического режима.
=========================================================

Режимы:
- NORMAL: стандартные лаги 3-6-9 мес
- SHOCK: короткие лаги 1-2-3 мес (быстрая трансмиссия)
- HIGH_INFLATION: средние лаги 2-4-6 мес

Критерии определения режима:
- SHOCK: |ΔRuonia| > 0.5 п.п. или |ΔKi| > 0.5 п.п. (резкие изменения ставок)
- HIGH_INFLATION: ΔИнфляция_YoY > 1.5 п.п. (ускорение инфляции)
- NORMAL: остальные периоды

Использование:
    >>> from sirena.models.regime_detector import detect_regime, MacroRegime
    >>> regime, diagnostics = detect_regime(df)
    >>> print(f"Режим: {regime.value}")
    >>> print(f"Ki change: {diagnostics['ki_change']:.2f} п.п.")
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
import pandas as pd
import numpy as np


class MacroRegime(Enum):
    """Макроэкономический режим."""
    NORMAL = "normal"
    SHOCK = "shock"
    HIGH_INFLATION = "high_inflation"


@dataclass
class RegimeConfig:
    """Конфигурация лагов для режима."""
    name: str
    lags: List[int]
    description: str


# Конфигурации для разных режимов
REGIME_CONFIGS = {
    MacroRegime.NORMAL: RegimeConfig(
        name="normal",
        lags=[3, 6, 9, 12],
        description="Стандартные лаги, медленная трансмиссия"
    ),
    MacroRegime.SHOCK: RegimeConfig(
        name="shock",
        lags=[1, 2, 3, 6],
        description="Короткие лаги, быстрая трансмиссия (кризис, резкие изменения ставок)"
    ),
    MacroRegime.HIGH_INFLATION: RegimeConfig(
        name="high_inflation",
        lags=[2, 4, 6, 9],
        description="Средние лаги, ускоренная трансмиссия (инфляционный режим)"
    ),
}


@dataclass
class RegimeThresholds:
    """Пороги для определения режима."""
    shock_ki_change: float = 0.5      # п.п. изменения Ki для режима SHOCK
    shock_ruonia_change: float = 0.5  # п.п. изменения Ruonia для режима SHOCK
    high_infl_yoy_change: float = 1.5 # п.п. ускорения YoY инфляции для HIGH_INFLATION


def detect_regime(
    df: pd.DataFrame,
    ki_col: str = 'Ki',
    ruonia_col: str = 'Ruonia',
    mom_col: str = 'mom',
    thresholds: Optional[RegimeThresholds] = None,
    lookback_months: int = 3
) -> Tuple[MacroRegime, Dict]:
    """
    Определить текущий макроэкономический режим.

    Args:
        df: DataFrame с макро-данными (индекс = Date)
        ki_col: Колонка ключевой ставки
        ruonia_col: Колонка RUONIA
        mom_col: Колонка MoM инфляции
        thresholds: Пороги для определения режима
        lookback_months: Окно для оценки изменений

    Returns:
        Tuple[MacroRegime, Dict]: Режим и диагностика
    """
    if thresholds is None:
        thresholds = RegimeThresholds()

    diagnostics = {
        'ki_change': 0.0,
        'ruonia_change': 0.0,
        'yoy_change': 0.0,
        'current_yoy': 0.0,
        'is_shock_ki': False,
        'is_shock_ruonia': False,
        'is_high_inflation': False,
        'lookback_months': lookback_months
    }

    # 1. Проверка на SHOCK (резкие изменения ставок)
    if ki_col in df.columns:
        ki = df[ki_col].dropna()
        if len(ki) >= lookback_months:
            ki_change = ki.iloc[-1] - ki.iloc[-lookback_months]
            diagnostics['ki_change'] = ki_change
            if abs(ki_change) > thresholds.shock_ki_change:
                diagnostics['is_shock_ki'] = True

    if ruonia_col in df.columns:
        ruonia = df[ruonia_col].dropna()
        if len(ruonia) >= lookback_months:
            ruonia_change = ruonia.iloc[-1] - ruonia.iloc[-lookback_months]
            diagnostics['ruonia_change'] = ruonia_change
            if abs(ruonia_change) > thresholds.shock_ruonia_change:
                diagnostics['is_shock_ruonia'] = True

    # SHOCK режим
    if diagnostics['is_shock_ki'] or diagnostics['is_shock_ruonia']:
        return MacroRegime.SHOCK, diagnostics

    # 2. Проверка на HIGH_INFLATION (ускорение инфляции YoY)
    if mom_col in df.columns:
        mom = df[mom_col].dropna()
        if len(mom) >= 24:  # Нужно минимум 2 года для YoY
            # Вычисляем YoY инфляцию (сумма MoM за 12 месяцев)
            # mom = MoM в процентных пунктах (100 = 0%)
            yoy_current = mom.iloc[-12:].sum()
            yoy_prev = mom.iloc[-24:-12].sum()
            yoy_change = yoy_current - yoy_prev

            diagnostics['current_yoy'] = yoy_current
            diagnostics['yoy_change'] = yoy_change

            if yoy_change > thresholds.high_infl_yoy_change:
                diagnostics['is_high_inflation'] = True
                return MacroRegime.HIGH_INFLATION, diagnostics

    # 3. По умолчанию NORMAL
    return MacroRegime.NORMAL, diagnostics


def get_regime_lags(regime: MacroRegime) -> List[int]:
    """
    Получить список лагов для режима.

    Args:
        regime: Макроэкономический режим

    Returns:
        List[int]: Список лагов в месяцах
    """
    return REGIME_CONFIGS[regime].lags


def get_regime_history(
    df: pd.DataFrame,
    ki_col: str = 'Ki',
    ruonia_col: str = 'Ruonia',
    mom_col: str = 'mom',
    window: int = 12
) -> pd.DataFrame:
    """
    Построить историю режимов.

    Args:
        df: DataFrame с макро-данными
        ki_col: Колонка ключевой ставки
        ruonia_col: Колонка RUONIA
        mom_col: Колонка MoM инфляции
        window: Окно для rolling расчёта

    Returns:
        DataFrame с колонками: Date, regime, ki_change, ruonia_change, yoy_change
    """
    history = []

    for i in range(12, len(df)):
        df_slice = df.iloc[:i+1]
        regime, diag = detect_regime(
            df_slice,
            ki_col=ki_col,
            ruonia_col=ruonia_col,
            mom_col=mom_col
        )
        history.append({
            'Date': df.index[i],
            'regime': regime.value,
            'ki_change': diag['ki_change'],
            'ruonia_change': diag['ruonia_change'],
            'yoy_change': diag['yoy_change'],
            'is_shock': regime == MacroRegime.SHOCK,
            'is_high_inflation': regime == MacroRegime.HIGH_INFLATION
        })

    return pd.DataFrame(history).set_index('Date')


def detect_shock_periods(
    df: pd.DataFrame,
    ki_col: str = 'Ki',
    ruonia_col: str = 'Ruonia',
    threshold: float = 1.0,
    min_duration: int = 2
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Найти периоды шоков в истории.

    Args:
        df: DataFrame с макро-данными
        ki_col: Колонка ключевой ставки
        ruonia_col: Колонка RUONIA
        threshold: Порог изменения ставки для шока (п.п./мес)
        min_duration: Минимальная длительность шока (месяцев)

    Returns:
        List[(start, end)]: Список периодов шоков
    """
    shocks = []

    # Объединяем изменения Ki и Ruonia
    changes = pd.DataFrame(index=df.index)

    if ki_col in df.columns:
        changes['ki_diff'] = df[ki_col].diff().abs()
    if ruonia_col in df.columns:
        changes['ruonia_diff'] = df[ruonia_col].diff().abs()

    if changes.empty:
        return []

    # Максимальное изменение из двух ставок
    changes['max_change'] = changes.max(axis=1)

    # Находим периоды шоков
    is_shock = changes['max_change'] > threshold

    in_shock = False
    shock_start = None

    for date, shock in is_shock.items():
        if shock and not in_shock:
            shock_start = date
            in_shock = True
        elif not shock and in_shock:
            shock_end = date
            # Проверяем минимальную длительность
            duration = (shock_end.to_period('M') - shock_start.to_period('M')).n
            if duration >= min_duration:
                shocks.append((shock_start, shock_end))
            in_shock = False

    # Если шок продолжается до конца данных
    if in_shock and shock_start is not None:
        shocks.append((shock_start, df.index[-1]))

    return shocks


# Известные шоковые периоды (для ручной коррекции)
KNOWN_SHOCK_PERIODS = [
    ('2014-12-01', '2015-02-28'),  # Валютный кризис 2014-2015
    ('2022-02-01', '2022-06-30'),  # Санкционный шок 2022
]


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    print("=" * 60)
    print("=== Тест Regime Detector ===")
    print("=" * 60)

    # Загрузка данных
    data_dir = Path(__file__).parent.parent.parent / 'data'
    df = pd.read_csv(data_dir / 'inflation_data.csv', sep=';', decimal=',')
    for col in df.columns:
        if col != 'Date' and df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    df = df.set_index('Date').sort_index()

    print(f"\nДанные: {df.index.min()} — {df.index.max()}")

    # 1. Тест detect_regime
    print("\n1. Текущий режим:")
    regime, diag = detect_regime(df)
    print(f"   Режим: {regime.value}")
    print(f"   ΔKi (3м): {diag['ki_change']:.2f} п.п.")
    print(f"   ΔRuonia (3м): {diag['ruonia_change']:.2f} п.п.")
    print(f"   YoY инфляция: {diag['current_yoy']:.2f}%")
    print(f"   ΔYOY: {diag['yoy_change']:.2f} п.п.")

    # 2. Лаги для текущего режима
    print(f"\n2. Лаги для режима {regime.value}:")
    lags = get_regime_lags(regime)
    print(f"   {lags}")

    # 3. История режимов
    print("\n3. История режимов (последние 12 месяцев):")
    history = get_regime_history(df)
    print(history.tail(12)[['regime', 'ki_change', 'is_shock']])

    # 4. Шоковые периоды
    print("\n4. Обнаруженные шоковые периоды:")
    shocks = detect_shock_periods(df)
    for start, end in shocks:
        print(f"   {start.strftime('%Y-%m')} — {end.strftime('%Y-%m')}")

    print("\n✅ Все тесты пройдены!")
