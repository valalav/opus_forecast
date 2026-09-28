(function (root) {
  'use strict';

  function isMonth(value) {
    return typeof value === 'string' && /^\d{4}-(0[1-9]|1[0-2])$/.test(value);
  }

  function finiteNumber(value) {
    return typeof value === 'number' && Number.isFinite(value);
  }

  function stepMonth(value) {
    return typeof value === 'string' && /^\d{4}-(0[1-9]|1[0-2])-01$/.test(value)
      ? value.slice(0, 7)
      : null;
  }

  function apply(steps, events) {
    if (!Array.isArray(steps) || !Array.isArray(events)) {
      throw new TypeError('Шаги прогноза и тарифные события должны быть массивами.');
    }

    const forecastMonths = new Set();
    for (const step of steps) {
      const month = step && typeof step === 'object' ? stepMonth(step.date) : null;
      if (!month) {
        throw new TypeError('Дата шага прогноза должна иметь формат YYYY-MM-01.');
      }
      if (!finiteNumber(step.all)) throw new TypeError('Значение прогноза должно быть числом.');
      if (forecastMonths.has(month)) throw new RangeError(`В прогнозе повторяется месяц ${month}.`);
      forecastMonths.add(month);
    }

    const totals = new Map();
    for (const event of events) {
      if (!event || typeof event !== 'object' || !isMonth(event.date)) {
        throw new TypeError('Дата тарифного события должна иметь формат YYYY-MM.');
      }
      const { date, rate, baseline, weight } = event;
      if (!finiteNumber(rate) || !finiteNumber(baseline) || !finiteNumber(weight)) {
        throw new TypeError('Ставка, базовый уровень и вес события должны быть числами.');
      }
      if (weight <= 0 || weight > 100) throw new RangeError('Вес события должен быть больше 0 и не превышать 100%.');
      if (rate <= -100 || baseline <= -100) throw new RangeError('Ставка и базовый уровень должны быть больше −100%.');
      if (!forecastMonths.has(date)) throw new RangeError(`Месяц события ${date} находится за пределами горизонта прогноза.`);

      const current = totals.get(date) || { weight: 0, delta: 0 };
      current.weight += weight;
      if (current.weight > 100) throw new RangeError(`Суммарный вес событий за ${date} превышает 100%.`);
      current.delta += weight * (rate - baseline) / 100;
      if (!finiteNumber(current.delta)) throw new RangeError(`Расчётная поправка за ${date} не является конечным числом.`);
      totals.set(date, current);
    }

    return steps.map((step) => {
      const month = stepMonth(step.date);
      const delta = totals.get(month)?.delta || 0;
      const adjusted = step.all + delta;
      if (!finiteNumber(adjusted) || adjusted <= -100) {
        throw new RangeError(`Скорректированный прогноз за ${month} должен быть конечным числом и больше −100%.`);
      }
      return { ...step, model_all: step.all, tariff_delta: delta, all: adjusted };
    });
  }

  root.TariffScenario = { apply };
})(globalThis);
