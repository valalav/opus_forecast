(function (root) {
  'use strict';

  const KINDS = new Set(['actual', 'plan', 'assumption']);
  const monthPattern = /^\d{4}-(0[1-9]|1[0-2])-01$/;
  const datePattern = /^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$/;
  const fields = ['date', 'rate', 'known_at', 'source', 'kind', 'weight', 'baseline'];

  function fail(message) { throw new TypeError(message); }
  function isRecord(value) { return value !== null && typeof value === 'object' && !Array.isArray(value); }
  function validMonth(value) { return typeof value === 'string' && monthPattern.test(value); }
  function validDate(value) {
    if (typeof value !== 'string' || !datePattern.test(value)) return false;
    const parsed = new Date(`${value}T00:00:00Z`);
    return parsed.toISOString().slice(0, 10) === value;
  }
  function finite(value) { return typeof value === 'number' && Number.isFinite(value); }

  function validate(calendar, expectedRegion) {
    if (!isRecord(calendar) || calendar.schema_version !== 1) fail('Календарь должен иметь schema_version: 1.');
    if (typeof calendar.region_code !== 'string' || !calendar.region_code.trim()) fail('В календаре не указан region_code.');
    if (expectedRegion !== undefined && String(expectedRegion) !== calendar.region_code) {
      fail(`Календарь региона ${calendar.region_code} нельзя применить к региону ${expectedRegion}.`);
    }
    if (typeof calendar.series_id !== 'string' || !calendar.series_id.trim()) fail('В календаре не указан series_id.');
    if (!Array.isArray(calendar.records)) fail('Поле records календаря должно быть массивом.');
    const keys = new Set();
    const records = calendar.records.map((record, index) => {
      if (!isRecord(record)) fail(`Запись ${index + 1} календаря должна быть объектом.`);
      if (!validMonth(record.date)) fail(`Запись ${index + 1}: дата должна иметь формат YYYY-MM-01.`);
      if (!finite(record.rate) || record.rate <= -100) fail(`Запись ${index + 1}: ставка должна быть числом больше −100%; ноль задаётся явно.`);
      if (!validDate(record.known_at)) fail(`Запись ${index + 1}: дата публикации должна иметь формат YYYY-MM-DD.`);
      if (typeof record.source !== 'string' || !record.source.trim()) fail(`Запись ${index + 1}: укажите источник.`);
      if (!KINDS.has(record.kind)) fail(`Запись ${index + 1}: вид должен быть actual, plan или assumption.`);
      if (record.kind === 'actual' && record.known_at < record.date) fail(`Запись ${index + 1}: факт не может быть опубликован до месяца наблюдения.`);
      if (record.weight !== undefined && (!finite(record.weight) || record.weight < 0 || record.weight > 100)) {
        fail(`Запись ${index + 1}: вес должен быть числом от 0 до 100%.`);
      }
      if (record.baseline !== undefined && (!finite(record.baseline) || record.baseline <= -100)) {
        fail(`Запись ${index + 1}: базовый уровень должен быть числом больше −100%.`);
      }
      const key = `${record.date}\u0000${record.known_at}`;
      if (keys.has(key)) fail(`Для месяца ${record.date.slice(0, 7)} уже есть версия с датой публикации ${record.known_at}.`);
      keys.add(key);
      const clean = { date: record.date, rate: record.rate, known_at: record.known_at, source: record.source, kind: record.kind };
      if (record.weight !== undefined) clean.weight = record.weight;
      if (record.baseline !== undefined) clean.baseline = record.baseline;
      return clean;
    });
    return { schema_version: 1, region_code: calendar.region_code, series_id: calendar.series_id, records };
  }

  function parseCSV(input, regionCode, seriesId = 'tariff-calendar') {
    if (typeof input !== 'string') fail('CSV-календарь должен быть текстом.');
    const text = input.replace(/^\uFEFF/, '');
    const firstLine = text.split(/\r?\n/, 1)[0] || '';
    const commaCount = (firstLine.match(/,/g) || []).length;
    const semicolonCount = (firstLine.match(/;/g) || []).length;
    const delimiter = semicolonCount > commaCount ? ';' : ',';
    const rows = [];
    let row = [], cell = '', quoted = false;
    for (let i = 0; i < text.length; i += 1) {
      const char = text[i];
      if (quoted) {
        if (char === '"' && text[i + 1] === '"') { cell += '"'; i += 1; }
        else if (char === '"') quoted = false;
        else cell += char;
      } else if (char === '"') quoted = true;
      else if (char === delimiter) { row.push(cell); cell = ''; }
      else if (char === '\n') { row.push(cell.replace(/\r$/, '')); rows.push(row); row = []; cell = ''; }
      else cell += char;
    }
    if (quoted) fail('В CSV не закрыта кавычка.');
    if (cell.length || row.length) { row.push(cell.replace(/\r$/, '')); rows.push(row); }
    const nonempty = rows.filter(cells => cells.some(value => value.trim() !== ''));
    if (!nonempty.length) fail('CSV-файл не содержит строк.');
    const header = nonempty.shift().map(value => value.trim().toLowerCase());
    for (const required of ['date', 'rate', 'known_at', 'source', 'kind']) {
      if (!header.includes(required)) fail(`В CSV отсутствует обязательный столбец ${required}.`);
    }
    const records = nonempty.map((cells, index) => {
      const values = Object.fromEntries(header.map((name, i) => [name, (cells[i] ?? '').trim()]));
      const record = { date: values.date, rate: values.rate === '' ? NaN : Number(values.rate), known_at: values.known_at, source: values.source, kind: values.kind };
      for (const optional of ['weight', 'baseline']) if (values[optional] !== undefined && values[optional] !== '') record[optional] = Number(values[optional]);
      if (!Number.isFinite(record.rate)) fail(`CSV-строка ${index + 2}: ставка должна быть числом.`);
      for (const optional of ['weight', 'baseline']) if (record[optional] !== undefined && !Number.isFinite(record[optional])) fail(`CSV-строка ${index + 2}: ${optional} должна быть числом.`);
      return record;
    });
    return validate({ schema_version: 1, region_code: String(regionCode ?? ''), series_id: String(seriesId ?? ''), records });
  }

  function exportCSV(calendar) {
    const valid = validate(calendar);
    const escape = value => `"${String(value ?? '').replaceAll('"', '""')}"`;
    return '\uFEFF' + [fields.join(';'), ...valid.records.map(record => fields.map(field => escape(record[field] ?? '')).join(';'))].join('\r\n');
  }

  function latestVintage(records, targetDate, forecastOrigin) {
    if (!validMonth(targetDate)) fail('Целевой месяц должен иметь формат YYYY-MM-01.');
    const originMonth = typeof forecastOrigin === 'string' && /^\d{4}-(0[1-9]|1[0-2])$/.test(forecastOrigin)
      ? forecastOrigin : typeof forecastOrigin === 'string' && validMonth(forecastOrigin) ? forecastOrigin.slice(0, 7) : null;
    if (!originMonth) fail('Дата отсечения должна иметь формат YYYY-MM или YYYY-MM-01.');
    const [year, month] = originMonth.split('-').map(Number);
    const knownBy = new Date(Date.UTC(year, month, 0)).toISOString().slice(0, 10);
    return records.filter(record => record.date === targetDate && record.known_at <= knownBy)
      .sort((a, b) => b.known_at.localeCompare(a.known_at))[0] || null;
  }

  root.TariffCalendar = { validate, parseCSV, exportCSV, latestVintage };
})(globalThis);
