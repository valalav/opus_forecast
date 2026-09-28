/* Download, validate, aggregate and exchange official Bank of Russia rate series. */
(function (root) {
  'use strict';

  const BASE_URL = 'https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx';
  const OPS = {
    ki: {method: 'KeyRateXML', row: ['KR', 'KeyRate'], date: 'DT', value: 'Rate', url: BASE_URL + '?op=KeyRateXML'},
    ruonia: {method: 'RuoniaXML', row: ['ro', 'Ruonia'], date: 'D0', value: 'ruo', url: BASE_URL + '?op=RuoniaXML'}
  };
  const today = () => new Date().toISOString();
  const progress = (fn, message, fraction) => { if (fn) fn({message, fraction}); };

  function validDate(value, label) {
    const s = String(value || '');
    if (!/^\d{4}-\d{2}-\d{2}$/.test(s)) throw new Error(label + ' must use YYYY-MM-DD.');
    const d = new Date(s + 'T00:00:00Z');
    if (!Number.isFinite(d.getTime()) || d.toISOString().slice(0, 10) !== s) throw new Error('Invalid ' + label + ': ' + s);
    return s;
  }
  function monthDate(value) {
    const date = validDate(value, 'date');
    if (!date.endsWith('-01')) throw new Error('Monthly macro dates must be YYYY-MM-01: ' + date);
    return date;
  }
  function number(value, label) {
    const text=String(value ?? '').trim();
    if(!text)throw new Error('Пустое значение ставки: '+label);
    const n = Number(text.replace(',', '.'));
    if (!Number.isFinite(n)) throw new Error('Invalid finite number for ' + label + '.');
    return n;
  }
  function xmlDocument(text) {
    if (typeof root.DOMParser !== 'function') throw new Error('This browser does not provide an XML parser. Import a JSON package instead.');
    const doc = new root.DOMParser().parseFromString(String(text), 'application/xml');
    if (doc.querySelector('parsererror')) throw new Error('CBR response contains malformed XML.');
    const fault = [...doc.getElementsByTagName('*')].find(n => (n.localName || n.nodeName.split(':').pop()).toLowerCase() === 'faultstring');
    if (fault) throw new Error('CBR SOAP fault: ' + fault.textContent.trim());
    return doc;
  }
  function localName(node) { return (node.localName || node.nodeName.split(':').pop()).toLowerCase(); }
  function parseXml(text, series, from, to) {
    const spec = OPS[series];
    if (!spec) throw new Error('Unknown CBR series: ' + series);
    const doc = xmlDocument(text), rows = [];
    for (const node of [...doc.getElementsByTagName('*')]) {
      if (!spec.row.map(x => x.toLowerCase()).includes(localName(node))) continue;
      const descendants = [...node.getElementsByTagName('*')];
      if (descendants.some(child => spec.row.map(x => x.toLowerCase()).includes(localName(child)))) continue;
      const fields = {};
      for (const child of descendants) {
        const name = localName(child);
        if (name === spec.date.toLowerCase() || name === spec.value.toLowerCase()) fields[name] = child.textContent.trim();
      }
      const rawDate = fields[spec.date.toLowerCase()] || '';
      const date = rawDate.slice(0, 10);
      if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) throw new Error('Malformed date in CBR ' + series + ' XML: ' + rawDate);
      validDate(date, 'CBR observation date');
      const value = number(fields[spec.value.toLowerCase()], series + ' at ' + date);
      if (date < from || date > to) continue;
      rows.push({date, value});
    }
    rows.sort((a, b) => a.date.localeCompare(b.date));
    const seen = new Set();
    for (const row of rows) {
      if (seen.has(row.date)) throw new Error('Duplicate CBR ' + series + ' observation: ' + row.date);
      seen.add(row.date);
    }
    if (!rows.length) throw new Error('CBR returned no ' + series + ' observations for ' + from + ' through ' + to + '.');
    return rows;
  }
  function sha256(text) {
    if (!root.crypto?.subtle) return Promise.resolve(null);
    return root.crypto.subtle.digest('SHA-256', new TextEncoder().encode(text)).then(bytes =>
      [...new Uint8Array(bytes)].map(x => x.toString(16).padStart(2, '0')).join(''));
  }
  function soapRequest(series, from, to) {
    const spec = OPS[series];
    const method = spec.method;
    return `<?xml version="1.0" encoding="utf-8"?><soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"><soap:Body><${method} xmlns="http://web.cbr.ru/"><fromDate>${from}T00:00:00</fromDate><ToDate>${to}T00:00:00</ToDate></${method}></soap:Body></soap:Envelope>`;
  }
  async function fetchXml(series, from, to) {
    if (root.LocalSources?.fetchCbr) return root.LocalSources.fetchCbr(series, from, to);
    if (typeof root.fetch !== 'function') throw new Error('No browser fetch is available. Use local server or import CBR XML files.');
    const spec = OPS[series];
    let response;
    try {
      response = await root.fetch(BASE_URL, {method: 'POST', headers: {'Content-Type': 'text/xml; charset=utf-8', SOAPAction: 'http://web.cbr.ru/' + spec.method}, body: soapRequest(series, from, to)});
    } catch (error) {
      throw new Error('Direct CBR download failed (the browser may block cross-origin SOAP). Start the local server or import official XML files. ' + error.message);
    }
    if (!response.ok) throw new Error('CBR SOAP HTTP ' + response.status + '. Start the local server or import official XML files.');
    return response.text();
  }
  function inRange(row, from, to) { return row.date >= from && row.date <= to; }
  function validateDaily(rows, series, from, to) {
    if (!Array.isArray(rows)) throw new Error('Package daily.' + series + ' must be an array.');
    const out = rows.map(r => {
      const date = validDate(r?.date, series + ' observation date');
      if (date < from || date > to) throw new Error(series + ' observation outside package coverage: ' + date);
      return {date, value: number(r.value, series + ' at ' + date)};
    }).sort((a, b) => a.date.localeCompare(b.date));
    for (let i = 1; i < out.length; i++) if (out[i - 1].date === out[i].date) throw new Error('Duplicate ' + series + ' date: ' + out[i].date);
    return out;
  }
  function aggregate(daily, method, from, to) {
    if (!['last', 'mean'].includes(method)) throw new Error('Aggregation must be last or mean.');
    const groups = new Map();
    for (const row of daily) {
      if (!inRange(row, from, to)) continue;
      const month = row.date.slice(0, 7) + '-01';
      if (!groups.has(month)) groups.set(month, []);
      groups.get(month).push(row);
    }
    const now = new Date();
    const lastCompletedMonthEnd = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 0)).toISOString().slice(0, 10);
    const completedTo = to < lastCompletedMonthEnd ? to : lastCompletedMonthEnd;
    const out = new Map();
    for (const [month, rows] of groups) {
      const lastDay = new Date(Date.UTC(Number(month.slice(0, 4)), Number(month.slice(5, 7)), 0)).toISOString().slice(0, 10);
      if (from > month || completedTo < lastDay) continue;
      out.set(month, method === 'last' ? rows[rows.length - 1].value : rows.reduce((s, r) => s + r.value, 0) / rows.length);
    }
    return out;
  }
  function packageFromDaily(daily, params = {}, source = {}) {
    const from = validDate(params.from || source.coverage?.from || daily.ki[0]?.date, 'from');
    const to = validDate(params.to || source.coverage?.to || daily.ki.at(-1)?.date, 'to');
    if (from > to) throw new Error('from must be on or before to.');
    const methods = {ki: params.ki_method || 'last', ruonia: params.ruonia_method || 'mean'};
    if (!['last', 'mean'].includes(methods.ki) || !['last', 'mean'].includes(methods.ruonia)) throw new Error('Aggregation methods must be last or mean.');
    const ki = aggregate(daily.ki, methods.ki, from, to), ruonia = aggregate(daily.ruonia, methods.ruonia, from, to);
    const months = [...new Set([...ki.keys(), ...ruonia.keys()])].sort();
    const macro_rows = months.map(date => ({date, ki: ki.has(date) ? ki.get(date) : null, ruonia: ruonia.has(date) ? ruonia.get(date) : null}));
    const allDates = [...daily.ki, ...daily.ruonia].map(r => r.date).sort();
    return {
      schema_version: 1, kind: 'regional_macro_rates', macro_rows,
      macro_source: {...source, publisher: source.publisher || 'Банк России', coverage: {from, to, first_observation: allDates[0] || null, last_observation: allDates.at(-1) || null}, downloaded_at: source.downloaded_at || today(), aggregation: {...methods}},
      daily, aggregation: methods
    };
  }
  function validatePackage(pkg) {
    const monthly = validateMonthlyPackage(pkg);
    if (!pkg.daily || !Array.isArray(pkg.daily.ki) || !Array.isArray(pkg.daily.ruonia)) return monthly;
    if ((!pkg.daily.ki.length || !pkg.daily.ruonia.length) && pkg.macro_rows.length) return monthly;
    const from = validDate(pkg.macro_source?.coverage?.from, 'package coverage.from');
    const to = validDate(pkg.macro_source?.coverage?.to, 'package coverage.to');
    if (from > to) throw new Error('Package coverage.from must not exceed coverage.to.');
    const daily = {ki: validateDaily(pkg.daily?.ki, 'ki', from, to), ruonia: validateDaily(pkg.daily?.ruonia, 'ruonia', from, to)};
    const params = {from, to, ki_method: pkg.aggregation?.ki, ruonia_method: pkg.aggregation?.ruonia};
    const normalized = packageFromDaily(daily, params, pkg.macro_source || {});
    if (!Array.isArray(pkg.macro_rows)) throw new Error('Package macro_rows must be an array.');
    const rows = pkg.macro_rows.map(r => {
      const date = monthDate(r?.date);
      for (const key of ['ki', 'ruonia']) if (r[key] !== null && r[key] !== undefined) number(r[key], key + ' at ' + date);
      return {date, ki: r.ki == null ? null : number(r.ki, 'ki at ' + date), ruonia: r.ruonia == null ? null : number(r.ruonia, 'ruonia at ' + date)};
    }).sort((a, b) => a.date.localeCompare(b.date));
    for (let i = 1; i < rows.length; i++) if (rows[i - 1].date === rows[i].date) throw new Error('Duplicate monthly macro date: ' + rows[i].date);
    normalized.macro_rows = rows;
    normalized.macro_source = {...normalized.macro_source, ...pkg.macro_source};
    normalized.daily = daily;
    normalized.aggregation = {...params.ki_method && {ki: params.ki_method}, ...params.ruonia_method && {ruonia: params.ruonia_method}};
    return normalized;
  }
  async function download(options = {}, onProgress) {
    const from = validDate(options.from || '2016-01-01', 'from');
    const to = validDate(options.to || '2026-08-31', 'to');
    if (from > to) throw new Error('from must be on or before to.');
    const methods = {ki_method: options.ki_method || 'last', ruonia_method: options.ruonia_method || 'mean'};
    progress(onProgress, 'Загрузка ключевой ставки', 0.08);
    const kiXml = await fetchXml('ki', from, to);
    progress(onProgress, 'Загрузка RUONIA', 0.43);
    const ruoniaXml = await fetchXml('ruonia', from, to);
    progress(onProgress, 'Проверка XML и расчёт месячных значений', 0.7);
    const daily = {ki: parseXml(kiXml, 'ki', from, to), ruonia: parseXml(ruoniaXml, 'ruonia', from, to)};
    const hashes = await Promise.all([sha256(kiXml), sha256(ruoniaXml)]);
    const pkg = packageFromDaily(daily, {from, to, ...methods}, {publisher: 'Банк России', urls: {ki: OPS.ki.url, ruonia: OPS.ruonia.url}, sha256: {ki: hashes[0], ruonia: hashes[1]}, raw_xml: {ki: kiXml, ruonia: ruoniaXml}, downloaded_at: today(), aggregation: {ki: methods.ki_method, ruonia: methods.ruonia_method}});
    progress(onProgress, 'Данные готовы', 1);
    return validatePackage(pkg);
  }
  function merge(dataset, inputPackage) {
    const pkg = validatePackage(inputPackage);
    if (!dataset || !Array.isArray(dataset.regions)) throw new Error('Regional dataset must contain regions[].');
    const byDate = new Map(pkg.macro_rows.map(r => [r.date, r]));
    const clone = typeof structuredClone === 'function' ? structuredClone(dataset) : JSON.parse(JSON.stringify(dataset));
    clone.macro_rows = pkg.macro_rows.map(r => ({...r}));
    clone.macro_source = {...pkg.macro_source};
    clone.macro_daily = pkg.daily || {ki:[],ruonia:[]};
    for (const region of clone.regions) {
      for (const key of ['rows', 'raw_rows', 'sa_rows']) {
        if (!Array.isArray(region[key])) continue;
        for (const row of region[key]) {
          const macro = byDate.get(String(row.date).slice(0, 7) + '-01');
          row.ki = macro?.ki ?? null;
          row.ruonia = macro?.ruonia ?? null;
        }
      }
      for (const rep of Object.values(region.representations || {})) {
        for (const row of rep?.rows || []) {
          const macro = byDate.get(String(row.date).slice(0, 7) + '-01');
          row.ki = macro?.ki ?? null;
          row.ruonia = macro?.ruonia ?? null;
        }
      }
    }
    return clone;
  }
  async function importFiles(files, options = {}) {
    const list = Array.from(files || []);
    if (!list.length) throw new Error('Select a CBR package JSON or official XML files.');
    const jsonFile = list.find(f => /\.json$/i.test(f.name));
    if (jsonFile) return validatePackage(JSON.parse(await jsonFile.text()));
    const xmlFiles = [];
    for (const file of list.filter(f => /\.xml$/i.test(f.name))) xmlFiles.push({file, text: await file.text()});
    const daily = {ki: null, ruonia: null};
    for (const {file, text} of xmlFiles) {
      const name = file.name.toLowerCase();
      const series = /ruonia|руония/i.test(name) ? 'ruonia' : /keyrate|key_rate|ставк|ключев/i.test(name) ? 'ki' : null;
      const guessed = series || options.seriesByName?.[file.name];
      if (!guessed || !OPS[guessed]) throw new Error('Cannot identify rate series for ' + file.name + '; name it KeyRateXML.xml or RuoniaXML.xml.');
      if (daily[guessed]) throw new Error('Duplicate XML file for ' + guessed + '.');
      daily[guessed] = text;
    }
    if (daily.ki && daily.ruonia) {
      const from = validDate(options.from || '1900-01-01', 'from'), to = validDate(options.to || '2999-12-31', 'to');
      const observations = {ki: parseXml(daily.ki, 'ki', from, to), ruonia: parseXml(daily.ruonia, 'ruonia', from, to)};
      const actualFrom = options.from || [observations.ki[0].date, observations.ruonia[0].date].sort().at(0);
      const actualTo = options.to || [observations.ki.at(-1).date, observations.ruonia.at(-1).date].sort().at(-1);
      const hashes = await Promise.all([sha256(daily.ki), sha256(daily.ruonia)]);
      return packageFromDaily(observations, {from: actualFrom, to: actualTo, ki_method: options.ki_method || 'last', ruonia_method: options.ruonia_method || 'mean'}, {publisher: 'Банк России', urls: {ki: OPS.ki.url, ruonia: OPS.ruonia.url}, sha256: {ki: hashes[0], ruonia: hashes[1]}, raw_xml: daily, downloaded_at: today(), imported_from: xmlFiles.map(x => x.file.name)});
    }
    const csv = list.find(f => /\.csv$/i.test(f.name));
    if (csv && root.LocalSources?.parseCsv) {
      const macro_rows = root.LocalSources.parseCsv(await csv.text()).map(r => ({date: monthDate(r.date), ki: number(r.ki, 'ki'), ruonia: number(r.ruonia, 'ruonia')}));
      const from = options.from || macro_rows[0]?.date, to = options.to || macro_rows.at(-1)?.date;
      const monthlyPkg = {schema_version: 1, kind: 'regional_macro_rates', macro_rows, macro_source: {publisher: 'Imported CSV', imported_from: csv.name, downloaded_at: today(), coverage: {from, to}}, daily: {ki: [], ruonia: []}, aggregation: {ki: 'last', ruonia: 'mean'}};
      return validateMonthlyPackage(monthlyPkg);
    }
    throw new Error('Import requires one CBR package JSON or both KeyRateXML and RuoniaXML files.');
  }
  function validateMonthlyPackage(pkg) {
    if (!pkg || pkg.schema_version !== 1 || pkg.kind !== 'regional_macro_rates' || !Array.isArray(pkg.macro_rows)) throw new Error('Invalid regional macro rates package.');
    const from = validDate(pkg.macro_source?.coverage?.from, 'coverage.from'), to = validDate(pkg.macro_source?.coverage?.to, 'coverage.to');
    if(from>to)throw new Error('Package coverage.from must not exceed coverage.to.');
    const rows = pkg.macro_rows.map(r => ({date: monthDate(r.date), ki: r.ki == null ? null : number(r.ki, 'ki'), ruonia: r.ruonia == null ? null : number(r.ruonia, 'ruonia')})).sort((a,b)=>a.date.localeCompare(b.date));
    for (let i=1;i<rows.length;i++) if(rows[i-1].date===rows[i].date) throw new Error('Duplicate monthly macro date: '+rows[i].date);
    return {...pkg, macro_rows: rows, macro_source: {...pkg.macro_source, coverage: {from,to}}};
  }
  function fromPackage(pkg) { return validateMonthlyPackage(pkg); }
  function fromDataset(dataset, options = {}) {
    const rows = dataset?.macro_rows || [];
    const normalized = rows.map(r => ({date: monthDate(r.date), ki: r.ki == null ? null : number(r.ki, 'ki'), ruonia: r.ruonia == null ? null : number(r.ruonia, 'ruonia')})).sort((a,b)=>a.date.localeCompare(b.date));
    const from = options.from || dataset.macro_source?.coverage?.from || normalized[0]?.date, to = options.to || dataset.macro_source?.coverage?.to || normalized.at(-1)?.date;
    if (!from || !to) throw new Error('Dataset has no monthly macro rows to export.');
    return validateMonthlyPackage({schema_version: 1, kind: 'regional_macro_rates', macro_rows: normalized, macro_source: {...dataset.macro_source, publisher: dataset.macro_source?.publisher || 'Imported dataset', coverage: {from, to}, downloaded_at: dataset.macro_source?.downloaded_at || today()}, daily: dataset.macro_daily || {ki: [], ruonia: []}, aggregation: dataset.macro_source?.aggregation || {ki: 'last', ruonia: 'mean'}});
  }
  root.CbrRates = {download, merge, importFiles, fromDataset, fromPackage, parseXml, validatePackage, aggregate};
})(globalThis);
