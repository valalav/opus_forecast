"""
Модуль для анализа сезонной структуры временных рядов.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from scipy.signal import periodogram
from typing import Dict, Optional
import warnings

from .base_analyzer import BaseAnalyzer

class SeasonalityAnalyzer(BaseAnalyzer):
    """
    Класс для анализа сезонной структуры временных рядов.
    
    Позволяет проводить декомпозицию временного ряда на тренд, сезонность
    и остатки, оценивать силу сезонности, выполнять спектральный анализ и
    формировать рекомендации по моделированию сезонных данных.
    """
    
    def __init__(self, 
                output_dir: Optional[str] = None,
                series_name: Optional[str] = None,
                seasonal_period: int = 12):
        """
        Инициализация анализатора сезонности.
        
        Args:
            output_dir: Директория для сохранения результатов
            series_name: Имя анализируемого ряда
            seasonal_period: Период сезонности (по умолчанию 12 для месячных данных)
        """
        super().__init__(output_dir, series_name)
        self.seasonal_period = seasonal_period
    
    def analyze(self, series: pd.Series, period: Optional[int] = None, 
               model: str = 'additive', save_plot: bool = True) -> Dict:
        """
        Анализ сезонности временного ряда.
        
        Args:
            series: Временной ряд для анализа
            period: Период сезонности (если None, используется self.seasonal_period)
            model: Модель декомпозиции ('additive' или 'multiplicative')
            save_plot: Сохранить график с результатами анализа
            
        Returns:
            Dict: результаты анализа сезонности
        """
        self.logger.info(f"Анализ сезонности для {self.series_name}")
        
        if period is None:
            period = self.seasonal_period
        
        # Предобработка ряда
        series_filled = self.preprocess_series(series)
        
        results = {
            'series_name': self.series_name,
            'seasonal_strength': None,
            'seasonality_detected': False,
            'decomposition': {
                'trend': None,
                'seasonal': None,
                'resid': None
            },
            'seasonal_pattern': None,
            'stl_decomposition': {
                'trend': None,
                'seasonal': None,
                'resid': None
            },
            'stl_seasonal_strength': None,
            'periodogram': None,
            'recommended_model': None
        }
        
        # Проверяем, достаточно ли данных для сезонной декомпозиции
        if len(series_filled) < 2 * period:
            self.logger.warning(f"Недостаточно данных для анализа сезонности с периодом {period}")
            results['error'] = f"Недостаточно данных для анализа сезонности с периодом {period}"
            return results
        
        try:
            # Классическая сезонная декомпозиция
            decomposition = seasonal_decompose(
                series_filled, 
                model=model, 
                period=period
            )
            
            # Сохраняем компоненты декомпозиции
            trend = pd.Series(decomposition.trend, index=series_filled.index)
            seasonal = pd.Series(decomposition.seasonal, index=series_filled.index)
            resid = pd.Series(decomposition.resid, index=series_filled.index)
            
            # Удаляем NaN из компонент
            trend = trend.dropna()
            seasonal = seasonal.dropna()
            resid = resid.dropna()
            
            # Определение силы сезонности
            if model == 'additive':
                seasonal_strength = max(0, 1 - np.var(resid) / np.var(seasonal + resid))
            else:  # multiplicative
                seasonal_strength = max(0, 1 - np.var(resid) / np.var(seasonal * resid))
                
            results['seasonal_strength'] = seasonal_strength
            results['seasonality_detected'] = seasonal_strength > 0.3  # Порог для определения сезонности
            
            self.logger.info(f"Сила сезонности: {seasonal_strength:.4f}")
            self.logger.info(f"Сезонность{' ' if results['seasonality_detected'] else ' не '}обнаружена")
            
            # Сохраняем компоненты
            results['decomposition']['trend'] = trend.tolist()
            results['decomposition']['seasonal'] = seasonal.tolist()
            results['decomposition']['resid'] = resid.tolist()
            
            # STL декомпозиция (Seasonal-Trend decomposition using LOESS)
            try:
                stl = STL(series_filled, period=period, seasonal=13, robust=True)
                stl_result = stl.fit()
                
                # Сохраняем компоненты STL
                results['stl_decomposition']['trend'] = stl_result.trend.dropna().tolist()
                results['stl_decomposition']['seasonal'] = stl_result.seasonal.dropna().tolist()
                results['stl_decomposition']['resid'] = stl_result.resid.dropna().tolist()
                
                # Оценка силы сезонности из STL
                stl_seasonal = pd.Series(stl_result.seasonal, index=series_filled.index).dropna()
                stl_resid = pd.Series(stl_result.resid, index=series_filled.index).dropna()
                
                stl_seasonal_strength = max(0, 1 - np.var(stl_resid) / np.var(stl_seasonal + stl_resid))
                results['stl_seasonal_strength'] = stl_seasonal_strength
                
                self.logger.info(f"Сила сезонности по STL: {stl_seasonal_strength:.4f}")
            except Exception as e:
                self.logger.error(f"Ошибка при выполнении STL декомпозиции: {str(e)}")
            
            # Периодограмма для выявления скрытых периодов
            try:
                # Вычисляем периодограмму
                f, Pxx = periodogram(series_filled.dropna().values)
                
                # Получаем периоды
                periods = 1 / f[1:]  # Исключаем первый элемент (частота 0)
                power = Pxx[1:]
                
                # Найдем наиболее значимые периоды
                significant_periods_idx = np.argsort(power)[-5:]  # Топ-5 по мощности
                significant_periods = periods[significant_periods_idx]
                significant_power = power[significant_periods_idx]
                
                results['periodogram'] = {
                    'periods': significant_periods.tolist(),
                    'power': significant_power.tolist()
                }
                
                # Найдем наиболее близкий к предполагаемому периоду
                closest_idx = np.argmin(np.abs(periods - period))
                results['periodogram']['closest_to_expected'] = {
                    'period': float(periods[closest_idx]),
                    'power': float(power[closest_idx]),
                    'rank': int(np.where(np.argsort(power)[::-1] == closest_idx)[0])  # Ранг по мощности
                }
                
                self.logger.info(f"Наиболее значимые периоды из спектрального анализа: {significant_periods.tolist()}")
            except Exception as e:
                self.logger.error(f"Ошибка при расчете периодограммы: {str(e)}")
            
            # Анализ сезонного паттерна
            if results['seasonality_detected']:
                # Извлекаем сезонный паттерн (значения для одного периода)
                seasonal_pattern = {}
                
                # Для месячных данных используем номера месяцев
                if period == 12 and isinstance(series.index, pd.DatetimeIndex):
                    # Группируем по месяцам и вычисляем среднее сезонное значение
                    monthly_seasonal = seasonal.groupby(seasonal.index.month).mean()
                    for month, value in monthly_seasonal.items():
                        seasonal_pattern[month] = value
                else:
                    # Для других периодов просто используем индекс от 0 до period-1
                    avg_seasonal = []
                    for i in range(period):
                        values = seasonal.iloc[i::period].values
                        avg_seasonal.append(np.mean(values))
                    
                    for i in range(period):
                        seasonal_pattern[i] = avg_seasonal[i]
                
                results['seasonal_pattern'] = seasonal_pattern
                
                # Рекомендация модели на основе силы сезонности
                if seasonal_strength > 0.7:
                    results['recommended_model'] = 'SARIMA или Prophet'
                elif seasonal_strength > 0.3:
                    results['recommended_model'] = 'SARIMA, ETS или Prophet'
                else:
                    results['recommended_model'] = 'ARIMA с сезонными индикаторами'
            else:
                results['recommended_model'] = 'ARIMA или ETS без сезонной компоненты'
            
            self.logger.info(f"Рекомендуемая модель: {results['recommended_model']}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при анализе сезонности: {str(e)}")
            results['error'] = str(e)
        
        # Визуализация
        if save_plot and 'error' not in results:
            self._plot_analysis(series_filled, results, period)
        
        return results
    
    def _plot_analysis(self, series: pd.Series, results: Dict, period: int):
        """
        Визуализация результатов анализа сезонности
        
        Args:
            series: временной ряд
            results: результаты анализа сезонности
            period: период сезонности
        """
        # График классической декомпозиции
        fig, axs = plt.subplots(4, 1, figsize=(12, 16))
        
        # Исходный ряд
        axs[0].plot(series.index, series.values)
        axs[0].set_title(f"Исходный ряд: {self.series_name}")
        axs[0].set_xlabel('Дата')
        axs[0].set_ylabel('Значение')
        axs[0].grid(True)
        
        # Тренд
        trend_values = results['decomposition']['trend']
        trend = pd.Series(trend_values, index=series.index[:len(trend_values)]).dropna()
        axs[1].plot(trend.index, trend.values)
        axs[1].set_title(f"Трендовая компонента")
        axs[1].set_xlabel('Дата')
        axs[1].set_ylabel('Значение')
        axs[1].grid(True)
        
        # Сезонная компонента
        seasonal_values = results['decomposition']['seasonal']
        seasonal = pd.Series(seasonal_values, index=series.index[:len(seasonal_values)]).dropna()
        axs[2].plot(seasonal.index, seasonal.values)
        axs[2].set_title(f"Сезонная компонента")
        axs[2].set_xlabel('Дата')
        axs[2].set_ylabel('Значение')
        axs[2].grid(True)
        
        # Остатки
        resid_values = results['decomposition']['resid']
        resid = pd.Series(resid_values, index=series.index[:len(resid_values)]).dropna()
        axs[3].plot(resid.index, resid.values)
        axs[3].set_title(f"Остаточная компонента")
        axs[3].set_xlabel('Дата')
        axs[3].set_ylabel('Значение')
        axs[3].grid(True)
        
        plt.tight_layout()
        
        # Добавляем аннотацию с результатами анализа
        plt.figtext(0.5, 0.01, 
                   f"Сила сезонности: {results['seasonal_strength']:.4f}\n"
                   f"Сезонность{' ' if results['seasonality_detected'] else ' не '}обнаружена\n"
                   f"Период сезонности: {period}\n"
                   f"Рекомендуемая модель: {results['recommended_model']}",
                   ha='center', bbox=dict(facecolor='white', alpha=0.8))
        
        # Сохраняем график декомпозиции
        filename = f"{self.series_name}_seasonality_analysis.png"
        self.save_plot(filename, fig)
        self.close_plot()
        
        # График STL декомпозиции, если есть результаты
        if 'stl_decomposition' in results and all(v is not None for v in results['stl_decomposition'].values()):
            fig, axs = plt.subplots(4, 1, figsize=(12, 16))
            
            # Исходный ряд
            axs[0].plot(series.index, series.values)
            axs[0].set_title(f"Исходный ряд: {self.series_name}")
            axs[0].grid(True)
            
            # STL тренд
            stl_trend = pd.Series(results['stl_decomposition']['trend'], index=series.index[:len(results['stl_decomposition']['trend'])])
            axs[1].plot(stl_trend.index, stl_trend.values)
            axs[1].set_title(f"STL трендовая компонента")
            axs[1].grid(True)
            
            # STL сезонная компонента
            stl_seasonal = pd.Series(results['stl_decomposition']['seasonal'], index=series.index[:len(results['stl_decomposition']['seasonal'])])
            axs[2].plot(stl_seasonal.index, stl_seasonal.values)
            axs[2].set_title(f"STL сезонная компонента")
            axs[2].grid(True)
            
            # STL остатки
            stl_resid = pd.Series(results['stl_decomposition']['resid'], index=series.index[:len(results['stl_decomposition']['resid'])])
            axs[3].plot(stl_resid.index, stl_resid.values)
            axs[3].set_title(f"STL остаточная компонента")
            axs[3].grid(True)
            
            plt.tight_layout()
            
            # Добавляем аннотацию с результатами STL
            if 'stl_seasonal_strength' in results:
                plt.figtext(0.5, 0.01, 
                           f"Сила сезонности по STL: {results['stl_seasonal_strength']:.4f}\n"
                           f"STL сезонность {'' if results['stl_seasonal_strength'] > 0.3 else 'не '}обнаружена",
                           ha='center', bbox=dict(facecolor='white', alpha=0.8))
            
            # Сохраняем график STL
            stl_filename = f"{self.series_name}_stl_decomposition.png"
            self.save_plot(stl_filename, fig)
            self.close_plot()
            
            self.logger.info(f"График STL декомпозиции сохранен: {stl_filename}")
        
        # Периодограмма, если есть результаты
        if 'periodogram' in results and results['periodogram'] is not None:
            plt.figure(figsize=(12, 6))
            
            # Получаем данные периодограммы
            periods = np.array(results['periodogram']['periods'])
            power = np.array(results['periodogram']['power'])
            
            # Сортируем по убыванию мощности для наглядности
            sorted_indices = np.argsort(power)[::-1]
            periods_sorted = periods[sorted_indices]
            power_sorted = power[sorted_indices]
            
            # Строим гистограмму для топ-10 периодов
            n_top = min(10, len(periods_sorted))
            plt.bar(range(n_top), power_sorted[:n_top], align='center')
            plt.xticks(range(n_top), [f"{p:.1f}" for p in periods_sorted[:n_top]])
            
            plt.title(f'Топ-{n_top} периодов по мощности для {self.series_name}')
            plt.xlabel('Период')
            plt.ylabel('Спектральная мощность')
            plt.grid(True, axis='y')
            
            # Сохраняем график периодограммы
            period_filename = f"{self.series_name}_periodogram.png"
            self.save_plot(period_filename)
            self.close_plot()
            
            self.logger.info(f"График периодограммы сохранен: {period_filename}")
        
        # График сезонного паттерна, если есть
        if 'seasonal_pattern' in results and results['seasonal_pattern']:
            plt.figure(figsize=(10, 6))
            
            pattern = results['seasonal_pattern']
            x = list(pattern.keys())
            y = list(pattern.values())
            
            plt.bar(x, y)
            plt.title(f"Сезонный паттерн для {self.series_name}")
            
            if period == 12:
                plt.xlabel('Месяц')
                # Добавляем названия месяцев
                month_names = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 
                             'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
                plt.xticks(range(1, 13), month_names)
            else:
                plt.xlabel('Индекс периода')
                
            plt.ylabel('Сезонный эффект')
            plt.grid(True, axis='y')
            
            pattern_filename = f"{self.series_name}_seasonal_pattern.png"
            self.save_plot(pattern_filename)
            self.close_plot()
            
            self.logger.info(f"График сезонного паттерна сохранен: {pattern_filename}")
