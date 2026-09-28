import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import {webcrypto} from 'node:crypto';

class Node {
  constructor(name) { this.nodeName = name; this.localName = name.split(':').pop(); this.children = []; this.textContent = ''; }
  getElementsByTagName() { return this.children.flatMap(child => [child, ...child.getElementsByTagName('*')]); }
}
class Parser {
  parseFromString(source) {
    const root = new Node('document'), stack = [root];
    for (const token of source.matchAll(/<([^!?\s/>]+)(?:\s[^>]*)?>|<\/([^>]+)>|([^<]+)/g)) {
      if (token[1]) {
        const node = new Node(token[1]); stack.at(-1).children.push(node);
        if (!token[0].endsWith('/>')) stack.push(node);
      } else if (token[2]) stack.pop();
      else if (token[3]) stack.at(-1).textContent += token[3].trim();
    }
    root.querySelector = () => null;
    root.getElementsByTagName = Node.prototype.getElementsByTagName;
    return root;
  }
}

const context = vm.createContext({TextEncoder, URL, URLSearchParams, Date, structuredClone, crypto: webcrypto, DOMParser: Parser, console});
vm.runInContext(await readFile(new URL('../web/cbr-rates.js', import.meta.url), 'utf8'), context);
const api = context.CbrRates;
const keyXml = '<KeyRateXMLResponse><KeyRateXMLResult><KeyRate><KR><DT>2023-08-01T00:00:00+03:00</DT><Rate>8.5</Rate></KR><KR><DT>2023-08-15T00:00:00+03:00</DT><Rate>12</Rate></KR></KeyRate></KeyRateXMLResult></KeyRateXMLResponse>';
const ruoniaXml = '<RuoniaXMLResponse><RuoniaXMLResult><Ruonia><ro><D0>2023-08-01T00:00:00+03:00</D0><ruo>8.1</ruo><DateUpdate>2023-08-02</DateUpdate></ro><ro><D0>2023-08-02T00:00:00+03:00</D0><ruo>8.3</ruo><DateUpdate>2023-08-03</DateUpdate></ro></Ruonia></RuoniaXMLResult></RuoniaXMLResponse>';

test('downloads through local adapter, parses SOAP namespace-safe, and aggregates full months', async () => {
  context.LocalSources = {fetchCbr: async series => series === 'ki' ? keyXml : ruoniaXml};
  const pkg = await api.download({from: '2023-08-01', to: '2023-08-31'});
  assert.equal(pkg.kind, 'regional_macro_rates');
  assert.deepEqual(JSON.parse(JSON.stringify(pkg.macro_rows)), [{date: '2023-08-01', ki: 12, ruonia: 8.2}]);
  assert.equal(pkg.daily.ki.length, 2);
  assert.match(pkg.macro_source.sha256.ki, /^[a-f0-9]{64}$/);
});

test('omits incomplete calendar months while retaining their daily observations', () => {
  const ki = api.parseXml(keyXml, 'ki', '2023-08-01', '2023-08-15');
  assert.equal(ki.length, 2);
  context.LocalSources = {fetchCbr: async series => series === 'ki' ? keyXml : ruoniaXml};
  return api.download({from: '2023-08-01', to: '2023-08-15'}).then(pkg => {
    assert.deepEqual(JSON.parse(JSON.stringify(pkg.macro_rows)), []);
    assert.equal(pkg.daily.ki.length, 2);
  });
});

test('merge replaces every regional rate series and nulls uncovered months on a clone', () => {
  const pkg = api.fromPackage({schema_version: 1, kind: 'regional_macro_rates', macro_rows: [{date: '2023-08-01', ki: 12, ruonia: 8.2}], macro_source: {coverage: {from: '2023-08-01', to: '2023-08-31'}}, daily: {ki: [], ruonia: []}, aggregation: {ki: 'last', ruonia: 'mean'}});
  const original = {regions: [{rows: [{date: '2023-08-01', ki: 1}, {date: '2023-09-01', ki: 2}], sa_rows: [{date: '2023-08-01'}], representations: {sa: {rows: [{date: '2023-09-01'}]}}}]};
  const result = api.merge(original, pkg);
  assert.notEqual(result, original);
  assert.deepEqual(JSON.parse(JSON.stringify(result.regions[0].rows)), [{date: '2023-08-01', ki: 12, ruonia: 8.2}, {date: '2023-09-01', ki: null, ruonia: null}]);
  assert.equal(result.regions[0].sa_rows[0].ki, 12);
  assert.equal(result.regions[0].representations.sa.rows[0].ruonia, null);
  assert.equal(original.regions[0].rows[0].ki, 1);
});

test('imports package JSON or both official XML files', async () => {
  const files = [
    {name: 'KeyRateXML.xml', text: async () => keyXml},
    {name: 'RuoniaXML.xml', text: async () => ruoniaXml}
  ];
  const pkg = await api.importFiles(files, {from: '2023-08-01', to: '2023-08-31'});
  assert.equal(pkg.macro_rows[0].ki, 12);
  const jsonFile = {name: 'macro_rates.json', text: async () => JSON.stringify(pkg)};
  assert.equal((await api.importFiles([jsonFile])).macro_rows.length, 1);
});

test('rejects duplicate XML observations and invalid package bounds', () => {
  assert.throws(() => api.parseXml(keyXml.replace('</KeyRate>', '<KR><DT>2023-08-01</DT><Rate>99</Rate></KR></KeyRate>'), 'ki', '2023-08-01', '2023-08-31'), /Duplicate/);
  assert.throws(() => api.fromPackage({schema_version: 1, kind: 'regional_macro_rates', macro_rows: [], macro_source: {coverage: {from: '2023-09-01', to: '2023-08-31'}}, daily: {ki: [], ruonia: []}, aggregation: {ki: 'last', ruonia: 'mean'}}), /must not exceed/);
});

test('blank rates are not silently converted to zero', () => {
  assert.throws(() => api.parseXml(keyXml.replace(/<Rate>[^<]*<\/Rate>/,'<Rate></Rate>'), 'ki', '2023-08-01', '2023-08-31'));
  assert.throws(() => api.fromPackage({schema_version:1,kind:'regional_macro_rates',macro_rows:[{date:'2023-08-01',ki:'',ruonia:8}],macro_source:{coverage:{from:'2023-08-01',to:'2023-08-31'}}}));
});
