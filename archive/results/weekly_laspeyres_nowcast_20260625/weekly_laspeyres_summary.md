# Weekly Laspeyres Nowcast Prototype

Target month: `2026-06`

## Coverage

- Weekly unique items: `116`
- Matched items: `108`
- Matched item weight: `0.3499`
- Exact matches: `106`
- Fuzzy matches: `2`

## June Signal

- Last weekly date: `2026-06-22`
- Cumulative observed matched-basket signal: `+1.673` pp
- Cumulative headline partial contribution: `+0.578` pp
- Cumulative component-scaled signal: `+2.124` pp

## Month-End Price Bridge

- Window: `2026-05-25` -> `2026-06-22`
- Observed matched-basket price index: `101.758`
- Headline partial index: `100.610`
- Component-scaled index: `102.218`

## Interpretation

`headline_partial_index` treats all unmatched CPI basket weights as no change.
`component_scaled_index` treats matched weekly items as representative within each broad component.
Both are diagnostics, not official CPI facts.
