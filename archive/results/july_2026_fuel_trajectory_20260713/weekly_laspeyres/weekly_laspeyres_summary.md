# Weekly Laspeyres Nowcast Prototype

Target month: `2026-07`

## Coverage

- Weekly unique items: `107`
- Matched items: `105`
- Matched item weight: `0.3469`
- Exact matches: `103`
- Fuzzy matches: `2`

## July 2026 Signal

- Last weekly date: `2026-07-06`
- Cumulative observed matched-basket signal: `+0.898` pp
- Cumulative headline partial contribution: `+0.312` pp
- Cumulative component-scaled signal: `+1.030` pp

## Month-End Price Bridge

- Window: `2026-06-29` -> `2026-07-06`
- Observed matched-basket price index: `100.888`
- Headline partial index: `100.308`
- Component-scaled index: `101.020`

## Interpretation

`headline_partial_index` treats all unmatched CPI basket weights as no change.
`component_scaled_index` treats matched weekly items as representative within each broad component.
Both are diagnostics, not official CPI facts.
