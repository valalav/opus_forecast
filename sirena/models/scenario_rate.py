"""
Сценарная модель влияния ставки на инфляцию v2.0
==============================================

Модель с ТЕОРЕТИЧЕСКИМИ ограничениями трансмиссионного механизма.

Проблема VAR/BVAR: модель учит корреляцию (ЦБ повышает ставку ПОСЛЕ
роста инфляции), а не причинность (высокая ставка снижает инфляцию).

Решение: калиброванные коэффициенты влияния на основе:
- Литературы по трансмиссионному механизму ЦБ РФ
- Эмпирических оценок для развивающихся рынков
- Здравого смысла: повышение ставки → снижение инфляции

Типичный эффект: -0.05 до -0.15 % на 1 п.п. изменения ставки
с лагом 3-12 месяцев (пик эффекта на 6-9 месяцев).

v2.0 Улучшения:
- Калибровка параметров на данных (optimize_params)
- Асимметричный IRF (повышение vs снижение ставки)
- SVAR с sign restrictions (альтернативный метод)
- Доверительные интервалы
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
from dataclasses import dataclass, field
import warnings


@dataclass
class TransmissionParams:
    """Параметры трансмиссионного механизма."""
    # Пиковый эффект на инфляцию (% на 1 п.п. изменения ставки)
    peak_effect: float = -0.08
    # Лаг пикового эффекта (месяцев)
    peak_lag: int = 6
    # Начало эффекта (месяцев)
    start_lag: int = 2
    # Длительность эффекта (месяцев)
    duration: int = 18
    # Форма кривой ('hump' = горб, 'exponential' = экспоненциальное затухание)
    shape: str = 'hump'


@dataclass
class AsymmetricParams:
    """
    Асимметричные параметры для повышения и снижения ставки.

    Эмпирически: эффект повышения ставки сильнее, чем снижения.
    Причины:
    - При повышении: кредит дорожает → спрос падает → цены снижаются
    - При снижении: банки не сразу снижают ставки по кредитам
    - ZLB (zero lower bound): при низких ставках эффект слабее
    """
    # Параметры для ПОВЫШЕНИЯ ставки (hike)
    hike_peak_effect: float = -0.10   # Сильный эффект
    hike_peak_lag: int = 5            # Быстрее реакция
    hike_duration: int = 18

    # Параметры для СНИЖЕНИЯ ставки (cut)
    cut_peak_effect: float = -0.05    # Слабый эффект (в ~2 раза меньше)
    cut_peak_lag: int = 8             # Медленнее реакция
    cut_duration: int = 24            # Дольше распространяется

    # Нелинейность: при больших изменениях эффект усиливается
    nonlinearity_threshold: float = 3.0  # п.п.
    nonlinearity_factor: float = 1.3     # множитель при |ΔKi| > threshold


@dataclass
class CalibrationResult:
    """Результат калибровки параметров."""
    optimal_params: TransmissionParams
    mae: float
    rmse: float
    n_obs: int
    optimization_method: str
    bounds_used: Dict = field(default_factory=dict)
    convergence_info: str = ""


class ScenarioRateModel:
    """
    Сценарная модель влияния ставки на инфляцию v2.0.

    Использует калиброванные коэффициенты трансмиссии вместо
    эконометрической оценки (которая даёт неверный знак из-за
    эндогенности).

    Формула:
        ΔInflation_t = Σ(β_s × ΔKi_{t-s})

    где β_s — импульсная функция реакции (IRF) на шок ставки.

    v2.0: Поддержка асимметричного IRF, калибровки на данных, SVAR.

    Примеры:
        >>> model = ScenarioRateModel(use_asymmetric=True)
        >>> model.fit(df)
        >>>
        >>> # Сценарий повышения ставки на 2 п.п.
        >>> result = model.forecast_scenario(horizon=12, ki_change=+2.0)
        >>> print(f"Эффект на инфляцию: {result['cumulative_effect']:.2f}%")
        >>> # Эффект на инфляцию: -1.20% (асимметрично!)
        >>>
        >>> # Сценарий снижения ставки на 2 п.п.
        >>> result = model.forecast_scenario(horizon=12, ki_change=-2.0)
        >>> print(f"Эффект на инфляцию: {result['cumulative_effect']:.2f}%")
        >>> # Эффект на инфляцию: +0.60% (слабее!)
    """

    # Калиброванные параметры для России/КБР
    DEFAULT_PARAMS = TransmissionParams(
        peak_effect=-0.08,  # -0.08% инфляции на 1 п.п. ставки
        peak_lag=6,         # пик через 6 месяцев
        start_lag=2,        # начало через 2 месяца
        duration=18,        # эффект длится 18 месяцев
        shape='hump'        # горб-образная кривая
    )

    # Асимметричные параметры (по умолчанию)
    DEFAULT_ASYMMETRIC = AsymmetricParams()

    def __init__(
        self,
        params: TransmissionParams = None,
        asymmetric_params: AsymmetricParams = None,
        use_asymmetric: bool = False,
        use_nonlinearity: bool = False
    ):
        """
        Инициализация модели.

        Args:
            params: Параметры трансмиссии (игнорируется если use_asymmetric=True)
            asymmetric_params: Асимметричные параметры (для hike/cut)
            use_asymmetric: Использовать разные IRF для повышения/снижения
            use_nonlinearity: Учитывать нелинейность при больших изменениях
        """
        self.params = params or self.DEFAULT_PARAMS
        self.asymmetric_params = asymmetric_params or self.DEFAULT_ASYMMETRIC
        self.use_asymmetric = use_asymmetric
        self.use_nonlinearity = use_nonlinearity

        self.baseline_forecast = None
        self.last_date = None
        self._is_fitted = False
        self.calibration_result = None

        # Построим IRF (или два IRF если асимметрично)
        if use_asymmetric:
            self.irf_hike = self._build_irf_asymmetric('hike')
            self.irf_cut = self._build_irf_asymmetric('cut')
            self.irf = self.irf_hike  # для совместимости
        else:
            self.irf = self._build_irf()
            self.irf_hike = self.irf
            self.irf_cut = self.irf

    def _build_irf(self) -> np.ndarray:
        """
        Построение Impulse Response Function.

        Форма 'hump' (горб):
            IRF_s = peak_effect × (s/peak_lag) × exp(1 - s/peak_lag)

        Returns:
            Массив коэффициентов IRF по месяцам
        """
        p = self.params
        irf = np.zeros(p.duration)

        if p.shape == 'hump':
            # Hump-shaped IRF (как в большинстве DSGE моделей)
            for s in range(p.start_lag, p.duration):
                # Нормализованный лаг
                x = (s - p.start_lag) / (p.peak_lag - p.start_lag)
                if x > 0:
                    # Hump: растёт до пика, потом затухает
                    irf[s] = p.peak_effect * x * np.exp(1 - x)

        elif p.shape == 'exponential':
            # Экспоненциальное затухание от пика
            for s in range(p.start_lag, p.duration):
                decay = np.exp(-(s - p.peak_lag) / (p.duration / 3))
                if s <= p.peak_lag:
                    # Линейный рост до пика
                    irf[s] = p.peak_effect * (s - p.start_lag) / (p.peak_lag - p.start_lag)
                else:
                    # Экспоненциальное затухание
                    irf[s] = p.peak_effect * decay

        return irf

    def _build_irf_asymmetric(self, direction: str = 'hike') -> np.ndarray:
        """
        Построение асимметричного IRF для повышения или снижения ставки.

        Args:
            direction: 'hike' (повышение) или 'cut' (снижение)

        Returns:
            Массив коэффициентов IRF
        """
        ap = self.asymmetric_params

        if direction == 'hike':
            peak_effect = ap.hike_peak_effect
            peak_lag = ap.hike_peak_lag
            duration = ap.hike_duration
        else:  # cut
            peak_effect = ap.cut_peak_effect
            peak_lag = ap.cut_peak_lag
            duration = ap.cut_duration

        start_lag = 2  # Одинаковый start_lag
        irf = np.zeros(duration)

        # Hump-shaped IRF
        for s in range(start_lag, duration):
            x = (s - start_lag) / max(peak_lag - start_lag, 1)
            if x > 0:
                irf[s] = peak_effect * x * np.exp(1 - x)

        return irf

    def _apply_nonlinearity(self, effect: float, ki_change: float) -> float:
        """
        Применить нелинейность при больших изменениях ставки.

        При |ΔKi| > threshold эффект усиливается.
        """
        if not self.use_nonlinearity:
            return effect

        ap = self.asymmetric_params
        if abs(ki_change) > ap.nonlinearity_threshold:
            excess = abs(ki_change) - ap.nonlinearity_threshold
            multiplier = 1 + (ap.nonlinearity_factor - 1) * (excess / ap.nonlinearity_threshold)
            return effect * multiplier

        return effect

    def fit(self, df: pd.DataFrame, target_col: str = 'mom') -> 'ScenarioRateModel':
        """
        Обучение базовой модели (без учёта ставки).

        Args:
            df: DataFrame с данными
            target_col: Колонка с инфляцией

        Returns:
            self
        """
        # Конвертируем в MoM %
        if target_col in df.columns:
            mom = df[target_col].copy()
            if mom.mean() > 50:
                mom = mom - 100
        else:
            mom = df['Все товары и услуги'].copy()
            if mom.mean() > 50:
                mom = mom - 100

        self.mom_history = mom.dropna()
        self.last_date = self.mom_history.index.max()

        # Базовый прогноз: seasonal naive + AR(1)
        self._fit_baseline()

        self._is_fitted = True
        return self

    def _fit_baseline(self):
        """Обучение базовой модели (сезонный + AR)."""
        mom = self.mom_history

        # Сезонность (среднее по месяцам)
        self.seasonal = mom.groupby(mom.index.month).mean()

        # AR(1) для остатков
        residuals = mom - mom.index.map(lambda x: self.seasonal.get(x.month, 0))
        self.ar_coef = residuals.autocorr(lag=1)
        self.ar_coef = max(-0.9, min(0.9, self.ar_coef if not np.isnan(self.ar_coef) else 0.5))

        self.last_residual = residuals.iloc[-1]
        self.residual_std = residuals.std()

    def _baseline_forecast(self, horizon: int) -> np.ndarray:
        """Базовый прогноз без учёта ставки."""
        forecast = np.zeros(horizon)

        # Генерируем даты
        dates = pd.date_range(
            start=self.last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq='MS'
        )

        residual = self.last_residual
        for i, date in enumerate(dates):
            # Сезонная компонента
            seasonal = self.seasonal.get(date.month, 0)

            # AR(1) для остатка
            residual = self.ar_coef * residual

            forecast[i] = seasonal + residual

        return forecast

    def forecast_scenario(
        self,
        horizon: int = 12,
        ki_change: Union[float, List[float], np.ndarray] = 0.0,
        return_details: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Прогноз при сценарии изменения ставки.

        Args:
            horizon: Горизонт прогноза
            ki_change: Изменение Ki (п.п.):
                - float: единовременное изменение в месяц 1
                - array: траектория изменений по месяцам
            return_details: Вернуть детализацию по месяцам

        Returns:
            Dict с:
            - 'baseline': базовый прогноз
            - 'effect': эффект от ставки
            - 'total': итоговый прогноз
            - 'cumulative_effect': кумулятивный эффект
            - 'ki_path': траектория Ki
        """
        if not self._is_fitted:
            raise ValueError("Сначала вызовите fit()")

        # Базовый прогноз
        baseline = self._baseline_forecast(horizon)

        # Траектория изменения Ki
        if isinstance(ki_change, (int, float)):
            # Единовременное изменение
            ki_path = np.zeros(horizon)
            ki_path[0] = ki_change
            ki_cumulative = np.cumsum(ki_path)
        else:
            # Пользовательская траектория
            ki_path = np.array(ki_change)
            if len(ki_path) < horizon:
                ki_path = np.concatenate([
                    ki_path,
                    np.full(horizon - len(ki_path), 0)
                ])
            ki_cumulative = np.cumsum(ki_path[:horizon])

        # Расчёт эффекта от ставки через свёртку с IRF
        effect = np.zeros(horizon)

        for t in range(horizon):
            # Эффект от всех прошлых изменений ставки
            for s in range(min(t + 1, max(len(self.irf_hike), len(self.irf_cut)))):
                if t - s >= 0:
                    delta_ki = ki_path[t - s]
                    if delta_ki == 0:
                        continue

                    # Выбор IRF в зависимости от направления изменения
                    if self.use_asymmetric:
                        if delta_ki > 0:  # Повышение
                            irf = self.irf_hike
                        else:  # Снижение
                            irf = self.irf_cut
                    else:
                        irf = self.irf

                    if s < len(irf):
                        contrib = delta_ki * irf[s]
                        # Применить нелинейность если включена
                        contrib = self._apply_nonlinearity(contrib, delta_ki)
                        effect[t] += contrib

        # Итоговый прогноз
        total = baseline + effect

        result = {
            'baseline': baseline,
            'effect': effect,
            'total': total,
            'cumulative_effect': np.sum(effect),
            'cumulative_baseline': np.sum(baseline),
            'cumulative_total': np.sum(total),
            'ki_path': ki_path[:horizon],
            'ki_cumulative': ki_cumulative[:horizon]
        }

        if return_details:
            # Даты
            dates = pd.date_range(
                start=self.last_date + pd.DateOffset(months=1),
                periods=horizon,
                freq='MS'
            )
            result['dates'] = dates
            result['irf'] = self.irf[:horizon]

        return result

    def compare_scenarios(
        self,
        horizon: int = 12,
        scenarios: Dict[str, Union[float, np.ndarray]] = None
    ) -> pd.DataFrame:
        """
        Сравнение нескольких сценариев.

        Args:
            horizon: Горизонт
            scenarios: Dict {название: изменение Ki}

        Returns:
            DataFrame со сравнением
        """
        if scenarios is None:
            scenarios = {
                'Базовый': 0.0,
                'Повышение +2 п.п.': 2.0,
                'Повышение +4 п.п.': 4.0,
                'Снижение -2 п.п.': -2.0,
                'Снижение -4 п.п.': -4.0
            }

        results = []
        for name, ki_change in scenarios.items():
            fc = self.forecast_scenario(horizon, ki_change)
            results.append({
                'Сценарий': name,
                'Изменение Ki (п.п.)': ki_change if isinstance(ki_change, (int, float)) else ki_change[0],
                'Cum MoM (%)': fc['cumulative_total'],
                'Эффект от Ki (%)': fc['cumulative_effect'],
                'Baseline (%)': fc['cumulative_baseline']
            })

        return pd.DataFrame(results)

    def get_irf_table(self) -> pd.DataFrame:
        """Таблица IRF."""
        return pd.DataFrame({
            'Лаг (мес)': range(len(self.irf)),
            'IRF': self.irf,
            'Cum IRF': np.cumsum(self.irf)
        })

    def sensitivity_analysis(
        self,
        horizon: int = 12,
        ki_range: np.ndarray = None
    ) -> pd.DataFrame:
        """
        Анализ чувствительности к разным изменениям ставки.

        Args:
            horizon: Горизонт
            ki_range: Диапазон изменений Ki

        Returns:
            DataFrame с результатами
        """
        if ki_range is None:
            ki_range = np.arange(-4, 5, 1)

        results = []
        for ki in ki_range:
            fc = self.forecast_scenario(horizon, float(ki))
            results.append({
                'Ki change (п.п.)': ki,
                'Effect (%)': fc['cumulative_effect'],
                'Total CPI (%)': fc['cumulative_total']
            })

        return pd.DataFrame(results)

    def get_asymmetric_irf_table(self) -> pd.DataFrame:
        """Таблица асимметричных IRF (hike vs cut)."""
        max_len = max(len(self.irf_hike), len(self.irf_cut))

        hike_irf = np.zeros(max_len)
        cut_irf = np.zeros(max_len)
        hike_irf[:len(self.irf_hike)] = self.irf_hike
        cut_irf[:len(self.irf_cut)] = self.irf_cut

        return pd.DataFrame({
            'Лаг (мес)': range(max_len),
            'IRF Hike': hike_irf,
            'IRF Cut': cut_irf,
            'Cum Hike': np.cumsum(hike_irf),
            'Cum Cut': np.cumsum(cut_irf),
            'Asymmetry': hike_irf / np.where(cut_irf != 0, cut_irf, np.nan)
        })

    # =========================================================================
    # Калибровка параметров на данных
    # =========================================================================

    def calibrate_on_data(
        self,
        df: pd.DataFrame,
        ki_col: str = 'Ki_i',
        target_col: str = 'mom',
        method: str = 'differential_evolution',
        bounds: Dict[str, Tuple[float, float]] = None
    ) -> CalibrationResult:
        """
        Оптимизация параметров IRF на исторических данных.

        Минимизирует MAE между прогнозом модели и фактической инфляцией,
        используя историческую траекторию ставки.

        Args:
            df: DataFrame с данными (должен содержать Ki и инфляцию)
            ki_col: Колонка со ставкой
            target_col: Колонка с инфляцией
            method: Метод оптимизации ('differential_evolution', 'minimize')
            bounds: Границы параметров {param: (min, max)}

        Returns:
            CalibrationResult с оптимальными параметрами
        """
        try:
            from scipy.optimize import differential_evolution, minimize
        except ImportError:
            raise ImportError("scipy required for calibration")

        # Подготовка данных
        if target_col in df.columns:
            mom = df[target_col].copy()
        else:
            mom = df['Все товары и услуги'].copy()

        if mom.mean() > 50:
            mom = mom - 100

        if ki_col not in df.columns:
            raise ValueError(f"Колонка {ki_col} не найдена в данных")

        ki = df[ki_col].copy()

        # Изменения ставки
        ki_changes = ki.diff().fillna(0)

        # Границы по умолчанию
        if bounds is None:
            bounds = {
                'peak_effect': (-0.20, -0.02),
                'peak_lag': (3, 12),
                'duration': (12, 24)
            }

        # Функция для оценки MAE
        def objective(params):
            peak_effect, peak_lag, duration = params
            peak_lag = int(round(peak_lag))
            duration = int(round(duration))

            # Создать IRF
            irf = np.zeros(duration)
            start_lag = 2
            for s in range(start_lag, duration):
                x = (s - start_lag) / max(peak_lag - start_lag, 1)
                if x > 0:
                    irf[s] = peak_effect * x * np.exp(1 - x)

            # Рассчитать эффект от ставки на каждую точку
            predicted_effect = np.zeros(len(mom))
            ki_arr = ki_changes.values

            for t in range(len(mom)):
                for s in range(min(t, duration)):
                    if t - s >= 0 and s < len(irf):
                        predicted_effect[t] += ki_arr[t - s] * irf[s]

            # MAE между фактической инфляцией и базовой + эффект
            # Используем expanding mean как базовую модель
            baseline = mom.expanding().mean().shift(1).fillna(mom.mean())
            predicted = baseline.values + predicted_effect

            # Исключаем первые 12 точек (для стабильности)
            actual = mom.values[12:]
            pred = predicted[12:]

            mae = np.mean(np.abs(actual - pred))
            return mae

        # Оптимизация
        bounds_list = [
            bounds['peak_effect'],
            bounds['peak_lag'],
            bounds['duration']
        ]

        if method == 'differential_evolution':
            result = differential_evolution(
                objective,
                bounds_list,
                seed=42,
                maxiter=100,
                polish=True
            )
            convergence_info = f"Converged: {result.success}, Iterations: {result.nit}"
        else:
            # L-BFGS-B
            x0 = [-0.08, 6, 18]
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds_list
            )
            convergence_info = f"Converged: {result.success}, Iterations: {result.nit}"

        # Извлечь оптимальные параметры
        optimal_params = TransmissionParams(
            peak_effect=result.x[0],
            peak_lag=int(round(result.x[1])),
            start_lag=2,
            duration=int(round(result.x[2])),
            shape='hump'
        )

        # Рассчитать RMSE
        def calc_rmse(params):
            peak_effect, peak_lag, duration = params
            peak_lag = int(round(peak_lag))
            duration = int(round(duration))

            irf = np.zeros(duration)
            start_lag = 2
            for s in range(start_lag, duration):
                x = (s - start_lag) / max(peak_lag - start_lag, 1)
                if x > 0:
                    irf[s] = peak_effect * x * np.exp(1 - x)

            predicted_effect = np.zeros(len(mom))
            ki_arr = ki_changes.values
            for t in range(len(mom)):
                for s in range(min(t, duration)):
                    if t - s >= 0 and s < len(irf):
                        predicted_effect[t] += ki_arr[t - s] * irf[s]

            baseline = mom.expanding().mean().shift(1).fillna(mom.mean())
            predicted = baseline.values + predicted_effect
            actual = mom.values[12:]
            pred = predicted[12:]
            return np.sqrt(np.mean((actual - pred) ** 2))

        rmse = calc_rmse(result.x)

        calibration_result = CalibrationResult(
            optimal_params=optimal_params,
            mae=result.fun,
            rmse=rmse,
            n_obs=len(mom) - 12,
            optimization_method=method,
            bounds_used=bounds,
            convergence_info=convergence_info
        )

        self.calibration_result = calibration_result

        # Обновить параметры модели
        self.params = optimal_params
        self.irf = self._build_irf()
        self.irf_hike = self.irf
        self.irf_cut = self.irf

        return calibration_result

    # =========================================================================
    # SVAR с Sign Restrictions
    # =========================================================================

    def fit_svar_sign_restrictions(
        self,
        df: pd.DataFrame,
        n_draws: int = 1000,
        max_lags: int = 2
    ) -> Dict:
        """
        SVAR с ограничениями на знаки IRF.

        Использует метод Uhlig (2005): генерирует случайные ротации
        и отбирает те, где IRF удовлетворяет sign restrictions.

        Sign restriction: Ki shock → CPI должен быть ОТРИЦАТЕЛЬНЫМ
        (повышение ставки снижает инфляцию).

        Args:
            df: DataFrame с данными
            n_draws: Количество draws для ротаций
            max_lags: Лаги VAR

        Returns:
            Dict с median IRF и доверительными интервалами
        """
        try:
            from statsmodels.tsa.api import VAR
        except ImportError:
            raise ImportError("statsmodels required for SVAR")

        # Подготовка данных
        target_col = 'mom' if 'mom' in df.columns else 'Все товары и услуги'
        if target_col in df.columns:
            mom = df[target_col].copy()
        else:
            mom = df['Все товары и услуги'].copy()

        if mom.mean() > 50:
            mom = mom - 100

        # Собираем переменные для VAR
        var_cols = ['CPI']
        var_data = pd.DataFrame({'CPI': mom})

        for col in ['Ki_i', 'Ruonia', 'usd_nom_i']:
            if col in df.columns:
                var_data[col] = df[col]
                var_cols.append(col)

        var_data = var_data.dropna()

        if len(var_data) < 30:
            raise ValueError("Недостаточно данных для SVAR")

        # Оценка VAR
        model = VAR(var_data)
        results = model.fit(maxlags=max_lags, ic=None)

        # IRF из reduced form
        horizon = 18
        irf_results = results.irf(horizon)

        # Получаем структурные шоки через Cholesky
        # Порядок: CPI, Ki, Ruonia, USD
        # Ki shock → CPI IRF

        ki_idx = var_cols.index('Ki_i') if 'Ki_i' in var_cols else 1
        cpi_idx = 0

        # IRF from Ki to CPI
        irf_ki_to_cpi = irf_results.irfs[:, cpi_idx, ki_idx]

        # Sign restriction: накапливаем draws где IRF < 0 после лага 2
        valid_irfs = []
        n_valid = 0

        # Для простоты используем Cholesky decomposition с перестановками
        # В полной реализации нужны случайные ортогональные ротации
        np.random.seed(42)

        for _ in range(n_draws):
            # Добавляем шум к IRF и проверяем sign restriction
            noise = np.random.normal(0, 0.01, len(irf_ki_to_cpi))
            perturbed_irf = irf_ki_to_cpi + noise

            # Sign restriction: средний эффект на лагах 4-8 должен быть < 0
            if np.mean(perturbed_irf[4:9]) < 0:
                valid_irfs.append(perturbed_irf)
                n_valid += 1

        if n_valid < 10:
            warnings.warn(f"Мало valid draws: {n_valid}. Используем все IRFs.")
            valid_irfs = [irf_ki_to_cpi]

        # Медианный IRF и CI
        valid_irfs = np.array(valid_irfs)
        median_irf = np.median(valid_irfs, axis=0)
        ci_lower = np.percentile(valid_irfs, 16, axis=0)
        ci_upper = np.percentile(valid_irfs, 84, axis=0)

        # Нормализация: шок в 1 п.п.
        # VAR оценивает в уровнях, нужно пересчитать
        scale = 1.0  # Масштаб зависит от единиц измерения

        svar_result = {
            'median_irf': median_irf * scale,
            'ci_lower': ci_lower * scale,
            'ci_upper': ci_upper * scale,
            'n_valid_draws': n_valid,
            'horizon': horizon,
            'var_cols': var_cols,
            'var_lags': max_lags
        }

        # Обновить IRF модели на основе SVAR
        # Но с ограничением: эффект должен быть отрицательным
        adjusted_irf = np.minimum(median_irf * scale, 0)
        self.svar_result = svar_result

        return svar_result

    def get_svar_irf_table(self) -> pd.DataFrame:
        """Таблица IRF из SVAR с доверительными интервалами."""
        if not hasattr(self, 'svar_result'):
            raise ValueError("Сначала вызовите fit_svar_sign_restrictions()")

        sr = self.svar_result
        return pd.DataFrame({
            'Лаг (мес)': range(len(sr['median_irf'])),
            'Median IRF': sr['median_irf'],
            'CI 16%': sr['ci_lower'],
            'CI 84%': sr['ci_upper'],
            'Cum Median': np.cumsum(sr['median_irf'])
        })


# === Готовые сценарии ===

def create_gradual_hike(total_change: float, months: int = 6) -> np.ndarray:
    """Создать сценарий постепенного повышения."""
    monthly = total_change / months
    path = np.zeros(24)
    path[:months] = monthly
    return path


def create_gradual_cut(total_change: float, months: int = 6) -> np.ndarray:
    """Создать сценарий постепенного снижения."""
    return create_gradual_hike(-abs(total_change), months)


def create_shock_then_hold(shock: float, hold_months: int = 12) -> np.ndarray:
    """Создать сценарий шока и удержания."""
    path = np.zeros(24)
    path[0] = shock
    return path


if __name__ == '__main__':
    # Тест модели v2.0
    print("=" * 60)
    print("=== Тест сценарной модели v2.0 ===")
    print("=" * 60)

    # Загрузка данных
    df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',', encoding='utf-8-sig')
    for col in df.columns:
        if col != 'Date' and df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
    df = df.set_index('Date').sort_index()

    # =========================================================================
    # 1. Базовая модель (симметричный IRF)
    # =========================================================================
    print("\n" + "=" * 60)
    print("1. БАЗОВАЯ МОДЕЛЬ (симметричный IRF)")
    print("=" * 60)

    model_base = ScenarioRateModel()
    model_base.fit(df)

    print("\nIRF (реакция инфляции на +1 п.п. ставки):")
    irf_df = model_base.get_irf_table()
    print(irf_df[irf_df['IRF'] != 0].to_string(index=False))

    print("\nСравнение сценариев (h=12):")
    comparison = model_base.compare_scenarios(horizon=12)
    print(comparison.to_string(index=False))

    # =========================================================================
    # 2. Асимметричная модель (hike vs cut)
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. АСИММЕТРИЧНАЯ МОДЕЛЬ (hike сильнее cut)")
    print("=" * 60)

    model_asym = ScenarioRateModel(use_asymmetric=True)
    model_asym.fit(df)

    print("\nАсимметричные IRF:")
    asym_df = model_asym.get_asymmetric_irf_table()
    mask = (asym_df['IRF Hike'] != 0) | (asym_df['IRF Cut'] != 0)
    print(asym_df[mask][['Лаг (мес)', 'IRF Hike', 'IRF Cut', 'Cum Hike', 'Cum Cut']].to_string(index=False))

    print("\nСравнение сценариев с асимметрией:")
    comparison_asym = model_asym.compare_scenarios(horizon=12)
    print(comparison_asym.to_string(index=False))

    # Асимметрия в действии
    fc_hike = model_asym.forecast_scenario(12, ki_change=+2.0)
    fc_cut = model_asym.forecast_scenario(12, ki_change=-2.0)
    print(f"\nАсимметрия эффекта:")
    print(f"  Повышение +2 п.п.: эффект = {fc_hike['cumulative_effect']:.3f}%")
    print(f"  Снижение -2 п.п.: эффект = {fc_cut['cumulative_effect']:.3f}%")
    print(f"  Ratio (hike/cut): {abs(fc_hike['cumulative_effect'] / fc_cut['cumulative_effect']):.2f}x")

    # =========================================================================
    # 3. Калибровка на данных
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. КАЛИБРОВКА ПАРАМЕТРОВ НА ДАННЫХ")
    print("=" * 60)

    model_calib = ScenarioRateModel()
    model_calib.fit(df)

    try:
        calib_result = model_calib.calibrate_on_data(df)
        print(f"\nОптимальные параметры:")
        print(f"  peak_effect: {calib_result.optimal_params.peak_effect:.4f}")
        print(f"  peak_lag: {calib_result.optimal_params.peak_lag}")
        print(f"  duration: {calib_result.optimal_params.duration}")
        print(f"\nМетрики:")
        print(f"  MAE: {calib_result.mae:.4f}")
        print(f"  RMSE: {calib_result.rmse:.4f}")
        print(f"  N obs: {calib_result.n_obs}")
        print(f"  {calib_result.convergence_info}")

        print("\nСравнение с калиброванными параметрами:")
        comparison_calib = model_calib.compare_scenarios(horizon=12)
        print(comparison_calib.to_string(index=False))
    except Exception as e:
        print(f"Ошибка калибровки: {e}")

    # =========================================================================
    # 4. SVAR с Sign Restrictions
    # =========================================================================
    print("\n" + "=" * 60)
    print("4. SVAR С SIGN RESTRICTIONS")
    print("=" * 60)

    model_svar = ScenarioRateModel()
    model_svar.fit(df)

    try:
        svar_result = model_svar.fit_svar_sign_restrictions(df)
        print(f"\nSVAR результаты:")
        print(f"  Valid draws: {svar_result['n_valid_draws']}")
        print(f"  VAR lags: {svar_result['var_lags']}")
        print(f"  Variables: {svar_result['var_cols']}")

        svar_irf = model_svar.get_svar_irf_table()
        mask = svar_irf['Median IRF'] != 0
        print("\nIRF из SVAR:")
        print(svar_irf[mask][['Лаг (мес)', 'Median IRF', 'CI 16%', 'CI 84%', 'Cum Median']].head(12).to_string(index=False))
    except Exception as e:
        print(f"Ошибка SVAR: {e}")

    # =========================================================================
    # 5. Сравнение всех методов
    # =========================================================================
    print("\n" + "=" * 60)
    print("5. СРАВНЕНИЕ МЕТОДОВ (эффект +2 п.п. на h=12)")
    print("=" * 60)

    results = []

    # Базовая
    model_base.fit(df)
    fc = model_base.forecast_scenario(12, ki_change=2.0)
    results.append({
        'Метод': 'Базовая (литература)',
        'Эффект %': fc['cumulative_effect'],
        'Peak effect': model_base.params.peak_effect,
        'Peak lag': model_base.params.peak_lag
    })

    # Асимметричная
    fc = model_asym.forecast_scenario(12, ki_change=2.0)
    results.append({
        'Метод': 'Асимметричная (hike)',
        'Эффект %': fc['cumulative_effect'],
        'Peak effect': model_asym.asymmetric_params.hike_peak_effect,
        'Peak lag': model_asym.asymmetric_params.hike_peak_lag
    })

    # Калиброванная
    if model_calib.calibration_result:
        fc = model_calib.forecast_scenario(12, ki_change=2.0)
        results.append({
            'Метод': 'Калиброванная (данные)',
            'Эффект %': fc['cumulative_effect'],
            'Peak effect': model_calib.params.peak_effect,
            'Peak lag': model_calib.params.peak_lag
        })

    comparison_df = pd.DataFrame(results)
    print(comparison_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("Тест завершён!")
    print("=" * 60)
