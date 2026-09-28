import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import test from 'node:test';

const source = await readFile(new URL('../web/tariff-scenario.js', import.meta.url), 'utf8');
const context = vm.createContext({});
vm.runInContext(source, context);
const apply = context.TariffScenario.apply;

test('weighted tariff effect is added in percentage points', () => {
  const result = apply([{ date: '2026-10-01', all: 1.2 }], [
    { date: '2026-10', rate: 10, baseline: 0, weight: 5 },
  ]);
  assert.deepEqual({ ...result[0] }, {
    date: '2026-10-01', all: 1.7, model_all: 1.2, tariff_delta: 0.5,
  });
});

test('equal rate and baseline produce zero delta', () => {
  const result = apply([{ date: '2026-10-01', all: 1.2 }], [
    { date: '2026-10', rate: 8, baseline: 8, weight: 20 },
  ]);
  assert.equal(result[0].tariff_delta, 0);
  assert.equal(result[0].all, 1.2);
});

test('events add within a month and policy timing can cancel July and transfer to October', () => {
  const steps = [
    { date: '2026-07-01', all: 0.4 },
    { date: '2026-10-01', all: 0.5 },
  ];
  const result = apply(steps, [
    { date: '2026-07', rate: 0, baseline: 10, weight: 5 },
    { date: '2026-10', rate: 10, baseline: 0, weight: 5 },
    { date: '2026-10', rate: 2, baseline: 0, weight: 10 },
  ]);
  assert.equal(result[0].tariff_delta, -0.5);
  assert.equal(result[1].tariff_delta, 0.7);
});

test('rejects blank or invalid weights', () => {
  for (const weight of ['', null, 0, -1, 100.01]) {
    assert.throws(() => apply([{ date: '2026-10-01', all: 1 }], [
      { date: '2026-10', rate: 10, baseline: 0, weight },
    ]));
  }
});

test('rejects overlapping monthly weights above 100 percent', () => {
  assert.throws(() => apply([{ date: '2026-10-01', all: 1 }], [
    { date: '2026-10', rate: 10, baseline: 0, weight: 60 },
    { date: '2026-10', rate: 8, baseline: 0, weight: 41 },
  ]), /превышает 100/);
});

test('does not mutate the input steps', () => {
  const steps = [{ date: '2026-10-01', all: 1, extra: 'kept' }];
  const original = structuredClone(steps);
  const result = apply(steps, [{ date: '2026-10', rate: 10, baseline: 0, weight: 5 }]);
  assert.deepEqual(steps, original);
  assert.notEqual(result[0], steps[0]);
  assert.equal(result[0].extra, 'kept');
});

test('rejects events outside the forecast horizon', () => {
  assert.throws(() => apply([{ date: '2026-10-01', all: 1 }], [
    { date: '2026-11', rate: 10, baseline: 0, weight: 5 },
  ]), /за пределами горизонта/);
});

test('rejects malformed dates, absent event fields, and rates at or below minus 100', () => {
  assert.throws(() => apply([{ date: '2026-13-01', all: 1 }], []), /YYYY-MM-01/);
  assert.throws(() => apply([{ date: '2026-10-01', all: 1 }], [{ date: '2026-10', rate: 10, weight: 5 }]));
  assert.throws(() => apply([{ date: '2026-10-01', all: 1 }], [
    { date: '2026-10', rate: -100, baseline: 0, weight: 5 },
  ]), /больше −100/);
});

test('rejects duplicate forecast months and adjusted values at or below minus 100', () => {
  assert.throws(() => apply([
    { date: '2026-10-01', all: 1 },
    { date: '2026-10-01', all: 2 },
  ], []), /повторяется месяц/);
  assert.throws(() => apply([{ date: '2026-10-01', all: -99.8 }], [
    { date: '2026-10', rate: 0, baseline: 10, weight: 5 },
  ]), /больше −100/);
});
