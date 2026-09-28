/* Explicit local-folder loading for the portable regional inflation app. */
(function (root) {
  'use strict';
  const XLSX_RE = /ИПЦ с исключением сезонности.*регионы.*\.xlsx$/i;
  const JSON_NAME = 'regional_indices.json';
  const RATES_NAME = 'macro_rates.json';
  const CSV_NAME = 'inflation_data.csv';
  const MAX_XLSX = 200 * 1024 * 1024;

  function report(progress, message, fraction) { if (progress) progress({message, fraction}); }
  async function sha256(text) {
    if (!root.crypto?.subtle) return null;
    const digest = await root.crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
    return [...new Uint8Array(digest)].map(x => x.toString(16).padStart(2, '0')).join('');
  }
  function normalizePath(value) {
    const path = String(value || '').trim().replace(/\\/g, '/').replace(/\/+$/, '');
    if (!path) throw new Error('Укажите путь к папке с локальными файлами.');
    if (path.split('/').includes('..')) throw new Error('Путь не должен содержать переходы к родительским папкам.');
    return path;
  }
  async function api(path, file) {
    if (!root.LOCAL_SOURCES_CONFIG?.baseUrl) throw new Error('Локальный источник доступен после запуска Start.ps1.');
    const endpoint = new URL(file ? '/api/local-file' : '/api/local-sources', root.LOCAL_SOURCES_CONFIG.baseUrl);
    endpoint.searchParams.set('path', path);
    const response = await fetch(endpoint, {headers: {'X-Local-Token': root.LOCAL_SOURCES_CONFIG.token}});
    if (!response.ok) throw new Error('Локальная папка: HTTP ' + response.status + (response.status === 403 ? ' (проверка источника не пройдена)' : ''));
    return file ? response.arrayBuffer() : response.json();
  }
  function parseCsv(text) {
    const lines = text.replace(/^\uFEFF/, '').split(/\r?\n/).filter(Boolean);
    if (lines.length < 2) throw new Error('inflation_data.csv пуст или не содержит строк данных.');
    const split = line => {
      const out = []; let value = '', quoted = false;
      for (let i = 0; i < line.length; i++) {
        const c = line[i];
        if (c === '"' && quoted && line[i + 1] === '"') { value += '"'; i++; }
        else if (c === '"') quoted = !quoted;
        else if (c === ';' && !quoted) { out.push(value); value = ''; }
        else value += c;
      }
      out.push(value); return out;
    };
    const header = split(lines[0]).map(x => x.trim());
    const dateIndex = header.indexOf('Date'), kiIndex = header.indexOf('Ki'), ruoniaIndex = header.indexOf('Ruonia');
    if ([dateIndex, kiIndex, ruoniaIndex].some(x => x < 0)) throw new Error('inflation_data.csv должен содержать столбцы Date, Ki и Ruonia.');
    const numeric = value => {
      const n = Number(String(value || '').trim().replace(/\s/g, '').replace(',', '.'));
      return Number.isFinite(n) ? n : null;
    };
    const rows = new Map();
    for (const line of lines.slice(1)) {
      const values = split(line), rawDate = (values[dateIndex] || '').trim();
      const date = /^\d{4}-\d\d/.test(rawDate) ? rawDate.slice(0, 7) + '-01' : (/^\d\d\.\d\d\.\d{4}$/.test(rawDate) ? rawDate.slice(6) + '-' + rawDate.slice(3, 5) + '-01' : '');
      if (!date) continue;
      const ki = numeric(values[kiIndex]), ruonia = numeric(values[ruoniaIndex]);
      if (ki === null || ruonia === null) throw new Error('В inflation_data.csv отсутствует Ki или Ruonia за ' + date + '.');
      if (rows.has(date)) throw new Error('В inflation_data.csv повторяется месяц ' + date + '.');
      rows.set(date, {date, ki, ruonia});
    }
    return [...rows.values()].sort((a, b) => a.date.localeCompare(b.date));
  }
  async function decode(file, previousDataset, progress) {
    const name = file.name;
    if (name.toLowerCase().endsWith('.xlsx')) {
      if (file.size > MAX_XLSX) throw new Error('Файл XLSX превышает лимит 200 МБ.');
      if (typeof root.importWorkbook !== 'function') throw new Error('Не загружен модуль импорта Excel.');
      return root.importWorkbook(await file.arrayBuffer(), {previousDataset, source: {name, source_type: 'local-file'}}, progress);
    }
    if (name === JSON_NAME) {
      const dataset = JSON.parse(await file.text());
      if (!Array.isArray(dataset.regions) || !dataset.regions.length) throw new Error('regional_indices.json не содержит региональные ряды.');
      return dataset;
    }
    throw new Error('Поддерживаются только региональный XLSX и regional_indices.json.');
  }
  async function mergeMacros(dataset, macroFile, progress) {
    if (!macroFile) return dataset;
    report(progress, 'Проверка макрорядов Ki и Ruonia', 0.9);
    const rawText = await macroFile.text();
    let rows, packageData;
    if (macroFile.name === RATES_NAME) {
      packageData = JSON.parse(rawText);
      if (root.CbrRates?.merge) return root.CbrRates.merge(dataset, packageData);
      if (packageData.kind !== 'regional_macro_rates' || !Array.isArray(packageData.macro_rows)) throw new Error('macro_rates.json должен содержать kind=regional_macro_rates и месячные macro_rows.');
      rows = packageData.macro_rows;
    } else rows = parseCsv(rawText);
    const byDate = new Map(rows.map(r => [r.date, r]));
    const dates = new Set((dataset.regions || []).flatMap(r => [...(r.rows || []), ...(r.sa_rows || [])].map(x => x.date)));
    const missing = [...dates].filter(date => !byDate.has(date));
    if (missing.length) throw new Error(macroFile.name + ' не покрывает месяцы региональных рядов: ' + missing.slice(0, 4).join(', ') + (missing.length > 4 ? '…' : ''));
    dataset.macro_rows = rows;
    for (const region of dataset.regions) for (const key of ['rows', 'sa_rows']) for (const row of (region[key] || [])) {
      const macro = byDate.get(row.date);
      if (!Number.isFinite(macro?.ki) || !Number.isFinite(macro?.ruonia)) throw new Error('В источнике отсутствует Ki или Ruonia за ' + row.date + '.');
      row.ki = macro.ki; row.ruonia = macro.ruonia;
    }
    dataset.macro_source = packageData?.macro_source || {publisher:'Импорт CSV пользователя',name:macroFile.name,format:'csv;Date,Ki,Ruonia',sha256:await sha256(rawText),coverage:{from:rows[0]?.date,to:rows.at(-1)?.date},aggregation:{ki:'provided_monthly',ruonia:'provided_monthly'}};
    dataset.macro_daily={ki:[],ruonia:[]};
    dataset.source = {...(dataset.source || {}), macro_source: dataset.macro_source};
    return dataset;
  }
  async function filesFromDirectory(handle) {
    const files = [];
    for await (const [name, entry] of handle.entries()) if (entry.kind === 'file' && (XLSX_RE.test(name) || name === JSON_NAME || name === RATES_NAME || name === CSV_NAME)) files.push(await entry.getFile());
    return files;
  }
  function latest(files, predicate) { return files.filter(f => predicate(f.name)).sort((a, b) => b.name.localeCompare(a.name))[0] || null; }
  async function loadSelection(input, previousDataset, progress) {
    const files = Array.from(input || []);
    const unsupportedXlsx = files.find(f => f.name.toLowerCase().endsWith('.xlsx') && !XLSX_RE.test(f.name));
    if (unsupportedXlsx && !files.some(f => XLSX_RE.test(f.name))) throw new Error('Файл «' + unsupportedXlsx.name + '» не поддерживается. Нужен региональный выпуск «ИПЦ с исключением сезонности (регионы)» с листом db.');
    const workbook = latest(files, name => XLSX_RE.test(name)) || files.find(f => f.name === JSON_NAME);
    if (!workbook) throw new Error('Выберите региональный XLSX или regional_indices.json.');
    let dataset = await decode(workbook, previousDataset, progress);
    const macroFile = files.find(f => f.name === RATES_NAME) || files.find(f => f.name === CSV_NAME);
    dataset = await mergeMacros(dataset, macroFile, progress);
    return {dataset, provenance: {primary: workbook.name, macro: macroFile?.name || null, mode: 'file-picker'}, warnings: macroFile ? [] : ['Отдельный файл ставок не найден: ставки из пакета не обновлялись.']};
  }
  async function chooseDirectory() {
    if (!root.showDirectoryPicker) throw new Error('Выбор папки браузером недоступен. Запустите Start.ps1, чтобы включить путь к папке.');
    const handle = await root.showDirectoryPicker({mode: 'read'});
    return {handle, files: await filesFromDirectory(handle)};
  }
  async function probe() {
    if (!root.LOCAL_SOURCES_CONFIG?.baseUrl) return {available: false, mode: 'file-picker'};
    return api('.', false);
  }
  async function list(path) { const result = await api(normalizePath(path), false); return result.files; }
  async function load(path, previousDataset, progress) {
    const listing = await api(normalizePath(path), false), files = listing.files || [];
    const unsupportedXlsx = files.find(f => f.name.toLowerCase().endsWith('.xlsx') && !XLSX_RE.test(f.name));
    const workbook = latest(files, name => XLSX_RE.test(name)) || files.find(f => f.name === JSON_NAME);
    const macro = files.find(f => f.name === RATES_NAME) || files.find(f => f.name === CSV_NAME);
    if (!workbook && unsupportedXlsx) throw new Error('Файл «' + unsupportedXlsx.name + '» не поддерживается. Нужен региональный выпуск «ИПЦ с исключением сезонности (регионы)» с листом db.');
    if (!workbook) throw new Error('В указанной папке нет регионального XLSX или regional_indices.json.');
    let dataset;
    if (workbook.name === JSON_NAME) {
      dataset = JSON.parse(new TextDecoder().decode(await api(normalizePath(path + '/' + workbook.name), true)));
    } else {
      dataset = await root.importWorkbook(await api(normalizePath(path + '/' + workbook.name), true), {previousDataset, source: {name: workbook.name, source_type: 'local-server'}}, progress);
    }
    const macroFile = macro && {name: macro.name, text: async () => new TextDecoder().decode(await api(normalizePath(path + '/' + macro.name), true))};
    dataset = await mergeMacros(dataset, macroFile, progress);
    return {dataset, provenance: {primary: workbook.name, macro: macro?.name || null, mode: 'loopback-folder', directory: path}, warnings: macro ? [] : ['Отдельный файл ставок не найден: использованы ставки из пакета; обновите их с ЦБ или импортируйте файл.']};
  }
  async function fetchCbr(series, from, to) {
    if (!['ki','ruonia'].includes(series) || !/^\d{4}-\d\d-\d\d$/.test(from) || !/^\d{4}-\d\d-\d\d$/.test(to) || from > to) throw new Error('Укажите ряд Ki/Ruonia и корректный интервал дат.');
    if (!root.LOCAL_SOURCES_CONFIG?.baseUrl) throw new Error('Загрузка ЦБ РФ доступна после запуска Start.ps1.');
    const endpoint = new URL('/api/cbr-rates', root.LOCAL_SOURCES_CONFIG.baseUrl);
    endpoint.search = new URLSearchParams({series, from, to});
    const response = await fetch(endpoint, {headers: {'X-Local-Token': root.LOCAL_SOURCES_CONFIG.token}});
    if (!response.ok) throw new Error('ЦБ РФ: HTTP ' + response.status);
    return response.text();
  }
  root.LocalSources = {probe, list, load, loadSelection, chooseDirectory, fetchCbr, parseCsv};
})(globalThis);
