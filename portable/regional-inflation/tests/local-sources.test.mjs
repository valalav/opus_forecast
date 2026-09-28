import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';

const context = vm.createContext({console, TextDecoder, TextEncoder, URL, URLSearchParams, fetch: async () => { throw new Error('unexpected fetch'); }});
vm.runInContext(await readFile(new URL('../web/local-sources.js', import.meta.url), 'utf8'), context);

test('parses regional macro CSV and rejects absent required columns', () => {
  const rows = context.LocalSources.parseCsv('Date;Ki;Ruonia\n2026-01-01;15,5;16,2\n');
  assert.deepEqual(JSON.parse(JSON.stringify(rows)), [{date:'2026-01-01',ki:15.5,ruonia:16.2}]);
  assert.throws(() => context.LocalSources.parseCsv('Date;Ki\n2026-01-01;1'), /Date, Ki и Ruonia/);
});

test('loads an explicitly selected regional dataset and merges macro rows', async () => {
  context.importWorkbook = async () => ({regions:[{code:7,rows:[{date:'2026-01-01',y:1}],sa_rows:[{date:'2026-01-01',y:1}]}]});
  const fake = (name, text) => ({name,size:text.length,text:async()=>text,arrayBuffer:async()=>new TextEncoder().encode(text).buffer});
  const result = await context.LocalSources.loadSelection([
    fake('2026-08 ИПЦ с исключением сезонности (регионы).xlsx','workbook'),
    fake('inflation_data.csv','Date;Ki;Ruonia\n2026-01-01;15;16\n')
  ], {}, null);
  assert.equal(result.dataset.regions[0].sa_rows[0].ki, 15);
  assert.equal(result.dataset.regions[0].rows[0].ruonia, 16);
  assert.equal(result.provenance.macro, 'inflation_data.csv');
});

test('fails when selected macro file does not cover regional history', async () => {
  context.importWorkbook = async () => ({regions:[{rows:[{date:'2026-01-01'}],sa_rows:[]}]});
  const fake = (name, text) => ({name,size:text.length,text:async()=>text,arrayBuffer:async()=>new TextEncoder().encode(text).buffer});
  await assert.rejects(context.LocalSources.loadSelection([
    fake('2026-08 ИПЦ с исключением сезонности (регионы).xlsx','workbook'), fake('inflation_data.csv','Date;Ki;Ruonia\n2025-12-01;15;16\n')
  ], {}, null), /не покрывает месяцы/);
});

test('uses macro_rates.json in preference to CSV through CbrRates.merge', async () => {
  context.importWorkbook = async () => ({regions:[]});
  let mergedPackage;
  context.CbrRates = {merge: (dataset, pkg) => { mergedPackage = pkg; return {...dataset, merged:true}; }};
  const fake = (name, text) => ({name,size:text.length,text:async()=>text,arrayBuffer:async()=>new TextEncoder().encode(text).buffer});
  const pkg = {schema_version:1,kind:'regional_macro_rates',macro_rows:[]};
  const result = await context.LocalSources.loadSelection([
    fake('2026-08 ИПЦ с исключением сезонности (регионы).xlsx','workbook'), fake('macro_rates.json',JSON.stringify(pkg)), fake('inflation_data.csv','Date;Ki;Ruonia\n2026-01-01;15;16\n')
  ], {}, null);
  assert.deepEqual(JSON.parse(JSON.stringify(mergedPackage)), pkg);
  assert.equal(result.dataset.merged, true);
});
