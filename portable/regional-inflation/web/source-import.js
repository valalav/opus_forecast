/* Offline XLSX import and public Yandex Disk refresh. No credentials or server. */
(function (root) {
  'use strict';
  const NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main';
  const delay = () => new Promise(resolve => setTimeout(resolve, 0));
  function xml(text) {
    const doc = new DOMParser().parseFromString(text, 'application/xml');
    if (doc.querySelector('parsererror')) throw new Error('Некорректный XML внутри XLSX');
    return doc;
  }
  function archive(bytes) {
    const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    let eocd = -1;
    for (let p = data.length - 22; p >= Math.max(0, data.length - 65557); p--) {
      if (view.getUint32(p, true) === 0x06054b50) { eocd = p; break; }
    }
    if (eocd < 0) throw new Error('Файл не является поддерживаемым XLSX/ZIP');
    const count = view.getUint16(eocd + 10, true);
    let p = view.getUint32(eocd + 16, true);
    const entries = new Map(), decoder = new TextDecoder();
    for (let i = 0; i < count; i++) {
      if (view.getUint32(p, true) !== 0x02014b50) throw new Error('Повреждён каталог XLSX');
      const method = view.getUint16(p + 10, true), size = view.getUint32(p + 20, true);
      const fullSize = view.getUint32(p + 24, true), n = view.getUint16(p + 28, true);
      const extra = view.getUint16(p + 30, true), comment = view.getUint16(p + 32, true);
      const offset = view.getUint32(p + 42, true);
      const name = decoder.decode(data.subarray(p + 46, p + 46 + n));
      if (view.getUint32(offset, true) !== 0x04034b50) throw new Error('Повреждён раздел XLSX');
      const start = offset + 30 + view.getUint16(offset + 26, true) + view.getUint16(offset + 28, true);
      if (start + size > data.length) throw new Error('Неполная загрузка XLSX');
      entries.set(name, {name, method, size, fullSize, bytes: data.subarray(start, start + size)});
      p += 46 + n + extra + comment;
    }
    return entries;
  }
  async function inflate(entry, consume, progress) {
    if (!entry) throw new Error('В Excel отсутствует необходимый лист');
    if (entry.fullSize > 600 * 1024 * 1024) throw new Error('Лист слишком велик для этой версии приложения');
    if (entry.method === 0) { consume(entry.bytes, true); return; }
    if (entry.method !== 8 || !root.fflate) throw new Error('Недоступен распаковщик XLSX');
    let produced = 0;
    const unzip = new root.fflate.Inflate((part, final) => {
      produced += part.length;
      if (produced > 600 * 1024 * 1024) throw new Error('Превышен размер распакованного листа');
      consume(part, final);
    });
    for (let p = 0; p < entry.size; p += 65536) {
      unzip.push(entry.bytes.subarray(p, p + 65536), p + 65536 >= entry.size);
      if (progress) progress(Math.min(1, (p + 65536) / entry.size));
      await delay();
    }
  }
  async function textEntry(entry) {
    let text = ''; const decoder = new TextDecoder();
    await inflate(entry, (part, final) => { text += decoder.decode(part, {stream: !final}); });
    return text;
  }
  function decodeText(value) {
    return value.replace(/&#(x[\da-f]+|\d+);/gi, (_, n) => String.fromCodePoint(n[0].toLowerCase() === 'x' ? parseInt(n.slice(1), 16) : Number(n)))
      .replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&amp;/g, '&');
  }
  function cells(row, strings) {
    const out = {};
    for (const match of row.matchAll(/<c\b([^>]*)>([\s\S]*?)<\/c>/g)) {
      const ref = /\br="([A-Z]+)\d+"/.exec(match[1]); if (!ref) continue;
      const value = /<v(?:\s[^>]*)?>([^<]*)<\/v>/.exec(match[2]);
      if (/\bt="s"/.test(match[1])) out[ref[1]] = value ? strings[Number(value[1])] : null;
      else if (/\bt="inlineStr"/.test(match[1])) out[ref[1]] = [...match[2].matchAll(/<t(?:\s[^>]*)?>([\s\S]*?)<\/t>/g)].map(x => decodeText(x[1])).join('');
      else out[ref[1]] = value ? Number(value[1]) : null;
    }
    return out;
  }
  async function rows(entry, strings, consume, progress) {
    let buffer = ''; const decoder = new TextDecoder();
    await inflate(entry, (part, final) => {
      buffer += decoder.decode(part, {stream: !final});
      let end;
      while ((end = buffer.indexOf('</row>')) >= 0) {
        const segment = buffer.slice(0, end + 6); buffer = buffer.slice(end + 6);
        consume(cells(segment, strings));
      }
      if (buffer.length > 4 * 1024 * 1024) throw new Error('Неподдерживаемая структура строк Excel');
    }, progress);
  }
  function report(progress, message, fraction) { if (progress) progress({message, fraction}); }
  async function importWorkbook(bytes, options = {}, progress) {
    if (typeof options === 'string') options = {kind: options};
    const entries = archive(bytes);
    const workbook = xml(await textEntry(entries.get('xl/workbook.xml')));
    if (workbook.querySelector('workbookPr')?.getAttribute('date1904') === '1') throw new Error('Календарь Excel 1904 не поддерживается');
    const relDoc = xml(await textEntry(entries.get('xl/_rels/workbook.xml.rels')));
    const relations = new Map([...relDoc.getElementsByTagName('Relationship')].map(x => [x.getAttribute('Id'), x.getAttribute('Target')]));
    const sheets = new Map([...workbook.getElementsByTagNameNS(NS, 'sheet')].map(s => {
      const rid = s.getAttribute('r:id'); let target = relations.get(rid);
      if (target?.startsWith('/')) target = target.slice(1); else target = 'xl/' + target;
      return [s.getAttribute('name'), entries.get(target)];
    }));
    if (!sheets.has('db')) throw new Error('Нужен файл «ИПЦ с исключением сезонности (регионы)» с листом db. Он содержит и RAW mom, и SA mom_sa. Внутренняя модель Excel файла «ИПЦ полный» этой версией не читается.');
    report(progress, 'Чтение справочников Excel', 0);
    let strings = [];
    if (entries.has('xl/sharedStrings.xml')) {
      const shared = xml(await textEntry(entries.get('xl/sharedStrings.xml')));
      strings = [...shared.getElementsByTagNameNS(NS, 'si')].map(si => [...si.getElementsByTagNameNS(NS, 't')].map(t => t.textContent).join(''));
    }
    const previous = options.previousDataset || {};
    const known = new Map((previous.regions || []).map(r => [String(r.code), r]));
    const names = new Map();
    if (sheets.has('momsa_regions')) await rows(sheets.get('momsa_regions'), strings, row => {
      if (Number.isInteger(row.A) && typeof row.B === 'string') names.set(String(row.A), row.B);
    });
    const macroRows = previous.macro_rows || previous.regions?.find(r => String(r.code) === '7')?.rows || [];
    const macros = new Map(macroRows.map(r => [r.date, {ki: r.ki ?? null, ruonia: r.ruonia ?? null}]));
    const table = new Map(), fields = {1: 'y', 2: 'nonfood', 3: 'food', 4: 'services'};
    let headerChecked = false, scanned = 0;
    await rows(sheets.get('db'), strings, row => {
      if (!headerChecked) {
        if ([row.A, row.B, row.C, row.D, row.E].join('|') !== 'Day|item|region|mom|mom_sa') throw new Error('Формат листа db изменился: ожидаются Day,item,region,mom,mom_sa');
        headerChecked = true; return;
      }
      scanned++;
      if (!fields[row.B]) return;
      if (!Number.isInteger(row.C) || !Number.isFinite(row.A)) throw new Error('Некорректная дата/регион в Excel');
      const date = new Date(Date.UTC(1899, 11, 30) + row.A * 86400000).toISOString().slice(0, 7) + '-01';
      const code = String(row.C), key = code + '/' + date;
      if (!table.has(key)) table.set(key, {code, date, raw: {}, sa: {}});
      const record = table.get(key), field = fields[row.B];
      if (Object.hasOwn(record.raw, field)) throw new Error('Дубликат региона/месяца/показателя: ' + key);
      if (!Number.isFinite(row.D) || !Number.isFinite(row.E) || row.D <= 0 || row.E <= 0) throw new Error('Отсутствует положительный индекс: ' + key);
      record.raw[field] = row.D; record.sa[field] = row.E;
    }, f => report(progress, 'Разбор региональных рядов: ' + scanned.toLocaleString('ru-RU') + ' строк', f));
    const regions = new Map();
    for (const record of table.values()) {
      if (Object.keys(record.raw).length !== 4 || Object.keys(record.sa).length !== 4) throw new Error('Неполные четыре ряда: ' + record.code + '/' + record.date);
      if (!regions.has(record.code)) {
        const old = known.get(record.code);
        regions.set(record.code, {code: Number(record.code), name: names.get(record.code) || old?.name || 'Территория ' + record.code,
          kind: old?.kind || 'unclassified', status: 'available', rows: [], sa_rows: []});
      }
      const region = regions.get(record.code), macro = macros.get(record.date) || {ki: null, ruonia: null};
      region.rows.push({date: record.date, ...record.raw, ...macro});
      region.sa_rows.push({date: record.date, ...record.sa, ...macro});
    }
    for (const region of regions.values()) {
      region.rows.sort((a, b) => a.date.localeCompare(b.date)); region.sa_rows.sort((a, b) => a.date.localeCompare(b.date));
      for (let i = 1; i < region.rows.length; i++) {
        const a = new Date(region.rows[i - 1].date); a.setUTCMonth(a.getUTCMonth() + 1);
        if (a.toISOString().slice(0, 10) !== region.rows[i].date) region.status = 'gaps';
      }
    }
    if (!regions.size) throw new Error('В файле не найдены региональные агрегаты');
    const array = [...regions.values()].sort((a, b) => a.name.localeCompare(b.name, 'ru'));
    const digest = root.crypto?.subtle ? [...new Uint8Array(await root.crypto.subtle.digest('SHA-256', bytes))].map(x => x.toString(16).padStart(2, '0')).join('') : null;
    const dates = [...table.values()].map(x => x.date).sort();
    report(progress, 'Загружено территорий: ' + array.length, 1);
    return {schema_version: 1, source: {...(options.source || {}), format: 'xlsx/db', raw_field: 'mom', sa_field: 'mom_sa',
      sha256: digest, imported_at: new Date().toISOString(), evidence_level: 'C',
      note: 'RAW и SA из одного выпуска; импорт заменяет историю целиком, без склейки vintages'},
      coverage: {start: dates[0], end: dates.at(-1), regions: array.length}, macro_rows: macroRows, regions: array};
  }
  async function loadYandexDataset(publicUrl, options = {}, progress) {
    const endpoint = 'https://cloud-api.yandex.net/v1/disk/public/resources';
    const query = new URLSearchParams({public_key: publicUrl, limit: '1000'});
    report(progress, 'Проверка опубликованных файлов на Яндекс.Диске', 0);
    const response = await fetch(endpoint + '?' + query);
    if (!response.ok) throw new Error('Яндекс.Диск: HTTP ' + response.status);
    const listing = await response.json();
    const candidates = (listing._embedded?.items || [listing]).filter(f => f.type === 'file' && /ИПЦ с исключением сезонности.*регионы.*\.xlsx$/i.test(f.name));
    candidates.sort((a, b) => b.name.localeCompare(a.name));
    if (!candidates.length) throw new Error('В папке не найден региональный Excel с RAW/SA. Выберите файл вручную.');
    const selected = candidates[0];
    const downloadQuery = new URLSearchParams({public_key: publicUrl, path: selected.path});
    const downloadResponse = await fetch(endpoint + '/download?' + downloadQuery);
    if (!downloadResponse.ok) throw new Error('Не получена ссылка на скачивание: HTTP ' + downloadResponse.status);
    const download = await downloadResponse.json();
    if (!download.href?.startsWith('https://')) throw new Error('Яндекс.Диск не вернул HTTPS-ссылку');
    const dataResponse = await fetch(download.href);
    if (!dataResponse.ok) throw new Error('Загрузка Excel: HTTP ' + dataResponse.status);
    const total = Number(dataResponse.headers.get('content-length')) || selected.size;
    if (total > 200 * 1024 * 1024) throw new Error('Размер файла превышает лимит 200 МБ');
    const reader = dataResponse.body.getReader(), chunks = []; let received = 0;
    while (true) {
      const {done, value} = await reader.read(); if (done) break;
      chunks.push(value); received += value.length;
      if (received > 200 * 1024 * 1024) { await reader.cancel(); throw new Error('Превышен лимит загрузки'); }
      report(progress, 'Загрузка ' + selected.name + ': ' + (received / 1048576).toFixed(1) + ' МБ', received / total);
    }
    const bytes = new Uint8Array(received); let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
    return importWorkbook(bytes, {...options, source: {public_url: publicUrl, name: selected.name, modified: selected.modified, size: selected.size}}, progress);
  }
  root.importWorkbook = importWorkbook;
  root.loadYandexDataset = loadYandexDataset;
})(globalThis);
