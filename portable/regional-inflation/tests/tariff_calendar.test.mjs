import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import test from 'node:test';

const source = await readFile(new URL('../web/tariff-calendar.js', import.meta.url), 'utf8');
const context = vm.createContext({});
vm.runInContext(source, context);
const { validate, parseCSV, exportCSV, latestVintage } = context.TariffCalendar;

const record = (extra = {}) => ({
  date: '2024-01-01', rate: 0, known_at: '2024-01-15', source: 'Официальная публикация', kind: 'actual', ...extra,
});
const calendar = records => ({ schema_version: 1, region_code: '07', series_id: 'regulated-tariffs', records });

test('validates explicit zero values and preserves absent optional fields', () => {
  const clean = validate(calendar([record()]));
  assert.equal(clean.records[0].rate, 0);
  assert.equal(Object.hasOwn(clean.records[0], 'weight'), false);
  assert.equal(Object.hasOwn(clean.records[0], 'baseline'), false);
});

test('accepts multiple vintages for a month with different publication dates', () => {
  const clean = validate(calendar([
    record({ known_at: '2024-01-15', rate: 4 }),
    record({ known_at: '2024-02-02', rate: 5 }),
  ]));
  assert.equal(clean.records.length, 2);
});

test('rejects duplicate vintages with the same month and publication date', () => {
  assert.throws(() => validate(calendar([record(), record({ rate: 1 })])), /уже есть версия/);
});

test('selects the latest version published by the forecast origin month end', () => {
  const records = [
    record({ rate: 2, known_at: '2024-01-20' }),
    record({ rate: 3, known_at: '2024-02-20' }),
    record({ rate: 4, known_at: '2024-03-01' }),
  ];
  assert.equal(latestVintage(records, '2024-01-01', '2024-02-01').rate, 3);
  assert.equal(latestVintage(records, '2024-01-01', '2024-01').rate, 2);
  assert.equal(latestVintage(records, '2024-02-01', '2024-02-01'), null);
});

test('checks calendar schema, region, dates, source, kind, and actual publication timing', () => {
  assert.throws(() => validate(calendar([]), '01'), /нельзя применить/);
  assert.throws(() => validate({ ...calendar([]), schema_version: 2 }), /schema_version/);
  assert.throws(() => validate(calendar([record({ date: '2024-02-02' })])), /YYYY-MM-01/);
  assert.throws(() => validate(calendar([record({ known_at: '2024-02-30' })])), /YYYY-MM-DD/);
  assert.throws(() => validate(calendar([record({ source: '   ' })])), /источник/);
  assert.throws(() => validate(calendar([record({ kind: 'forecast' })])), /actual, plan или assumption/);
  assert.throws(() => validate(calendar([record({ known_at: '2023-12-31' })])), /не может быть опубликован/);
});

test('supports BOM, semicolon separators, quoted delimiters, and CSV export round-trip', () => {
  const csv = '\uFEFFdate;rate;known_at;source;kind;weight;baseline\r\n2024-01-01;0;2024-01-15;"Источник; с запятой, и кавычкой ""А""";actual;;';
  const parsed = parseCSV(csv, '07', 'regulated-tariffs');
  assert.equal(parsed.records[0].rate, 0);
  assert.equal(parsed.records[0].source, 'Источник; с запятой, и кавычкой "А"');
  assert.equal(Object.hasOwn(parsed.records[0], 'weight'), false);
  const roundtrip = parseCSV(exportCSV(parsed), '07', 'regulated-tariffs');
  assert.deepEqual({ ...roundtrip.records[0] }, { ...parsed.records[0] });
});

test('parses comma-delimited CSV and rejects blank or nonnumeric required rates', () => {
  const parsed = parseCSV('date,rate,known_at,source,kind\n2024-01-01,1.25,2024-01-15,"Source, primary",plan', '07');
  assert.equal(parsed.records[0].source, 'Source, primary');
  assert.equal(parsed.records[0].rate, 1.25);
  assert.throws(() => parseCSV('date;rate;known_at;source;kind\n2024-01-01;;2024-01-15;x;actual', '07'), /ставка должна быть числом/);
});

test('does not mutate calendar objects while validating', () => {
  const input = calendar([record({ weight: 5, baseline: 2 })]);
  const before = structuredClone(input);
  const result = validate(input);
  assert.deepEqual(input, before);
  assert.notEqual(result, input);
  assert.notEqual(result.records, input.records);
});
