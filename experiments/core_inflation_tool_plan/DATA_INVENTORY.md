# Core Inflation Tool Data Inventory

Date: 2026-05-27

## Selected MVP Sources

| Source | Use | Reason |
|---|---|---|
| `data/kbr_indices.csv` | component MoM and YoY indices | Long table with `Date`, `Region_code`, `Item_code`, `MoM`, `YoY`; KBR rows are `Region_code = 7`. |
| `data/mom_sa_kbr.csv` | seasonally adjusted component MoM indices | Wide table with KBR SA component indices from 2016-01 to 2026-04; used for `*_sa` stable inflation indicators. |
| `data/access_weights.csv` | annual item weights | Has `Region_code`, `Item_code`, `Day`, `Weight_gross`; KBR rows exist for 2016-2026. |
| `data/items_names.csv` | item labels | Maps `Item_code` to names for readable contribution reports. |
| `data/inflation_data.csv` | headline check | Official monthly headline series in semicolon format with space-padded decimals. |

## Rejected Or Deferred Sources

| Source | Decision | Reason |
|---|---|---|
| `data/micro_sprav.csv` | deferred | Useful hierarchy/weights reference, but semicolon parsing and item coverage differ from `access_weights`. |
| `data/raw/sub_mom.csv` | deferred | Subcomponent-only wide format; not enough for full basket core calculation. |
| `data/raw/subcomp_sprav.csv` | deferred | Subcomponent dictionary; useful for hierarchy validation in a later pass. |
| `data/trimmed_mean_cpi.csv` | comparison only | Existing output artifact, not a primary input. |
| `data/sticky_price_index.csv` | comparison only | Existing output artifact with missing early values. |
| `data/inflation_persistence.csv` | comparison only | Persistence diagnostics by product, not monthly CPI facts. |

## Known Risks

- Annual weights are joined to monthly component observations by calendar year.
- Aggregate item codes are excluded by config to avoid mixing headline and component levels.
- The Excel audit showed MoM/YoY contamination risk; the CLI always emits a MoM/YoY distinctness diagnostic.
- `data/mom_sa_kbr.csv` has MoM SA indices but no YoY matrix, so MoM/YoY contamination is marked `expected_skip` for the SA branch.
- The exclusion list is intentionally conservative and must be business-reviewed before production use.
