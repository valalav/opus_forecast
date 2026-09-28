import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const web = path.join(root, 'web');
const arg = (name, fallback) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? path.resolve(process.argv[i + 1]) : fallback;
};

const templatePath = arg('template', path.join(web, 'index.html'));
const cssPath = arg('css', path.join(web, 'style.css'));
const appPath = arg('app', path.join(web, 'app.js'));
const importPath = arg('importer', path.join(web, 'source-import.js'));
const localSourcesPath = arg('local-sources', path.join(web, 'local-sources.js'));
const comparisonPath = arg('comparison', path.join(web, 'comparison.js'));
const cbrRatesPath = arg('cbr-rates', path.join(web, 'cbr-rates.js'));
const dataPath = arg('data', path.join(root, 'data', 'regional_indices.json'));
const wasmJsPath = arg('wasm-js', path.join(root, 'core', 'pkg', 'regional_inflation_core.js'));
const wasmPath = arg('wasm', path.join(root, 'core', 'pkg', 'regional_inflation_core_bg.wasm'));
const outputPath = arg('output', path.join(root, 'regional-inflation.html'));
const fflatePath = arg('fflate', path.join(root, 'node_modules', 'fflate', 'umd', 'index.js'));

const required = [templatePath, cssPath, appPath, importPath, localSourcesPath, comparisonPath, cbrRatesPath, dataPath, wasmJsPath, wasmPath, fflatePath];
for (const file of required) {
  try { await readFile(file); }
  catch { throw new Error(`Required build input is missing: ${file}`); }
}
const [template, css, app, importer, localSources, comparison, cbrRates, data, wasmJs, wasm, fflate] = await Promise.all(required.map(file => readFile(file)));
const encode = b => b.toString('base64');
const safeJson = b => JSON.stringify(JSON.parse(b.toString('utf8'))).replaceAll('<', '\\u003c').replaceAll('>', '\\u003e').replaceAll('&', '\\u0026');
let html = template.toString();
const replacements = [
  ['/*__INLINE_CSS__*/', css.toString()],
  ['/*__INLINE_WASM_BINDGEN__*/', `window.__WASM_BINDGEN_SOURCE__=${JSON.stringify(wasmJs.toString())};window.__REGIONAL_WASM_BASE64__=${JSON.stringify(encode(wasm))};window.__REGIONAL_DATA__=${safeJson(data)};`],
  ['/*__INLINE_APP__*/', `${fflate.toString()}\n${importer.toString()}\n${localSources.toString()}\n${comparison.toString()}\n${cbrRates.toString()}\n${app.toString()}`],
];
for (const [needle, value] of replacements) {
  if (!html.includes(needle)) throw new Error(`Template placeholder not found: ${needle}`);
  // Use a callback so `$&` in bundled/vendor JavaScript is not treated as a replacement token.
  html = html.replace(needle, () => value);
}
await mkdir(path.dirname(outputPath), { recursive: true });
await writeFile(outputPath, html, 'utf8');
console.log(`Built ${outputPath} (${Buffer.byteLength(html).toLocaleString()} bytes)`);
