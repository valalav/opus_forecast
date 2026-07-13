# Task 133: Weekly Data Standard & Governance

## Context
We are shifting from ad-hoc analysis to a production-grade "High-Frequency Monitor". This requires a rigid data schema to prevent drift and ensure model reliability.

## Objective
Establish the "Law of the Data" for weekly prices.

## Steps
1.  **Define Protocol**: Create `docs/WEEKLY_DATA_STANDARD.md` specifying:
    -   **File Naming**: `kbr_weekly_prices_YYYY_YYYY.csv`.
    -   **Columns**: `date`, `product_code`, `product_name`, `price`, `weight`, `flags`.
    -   **Types**: Strict typing (Date vs String).
    -   **Missingness Policy**: When to drop vs interpolate vs flag.
2.  **Create Validator**: Implement `scripts/validate_weekly_data.py` that:
    -   Checks column existence and types.
    -   Verifies date continuity (Monotonic freq=W-MON).
    -   Checks for duplicate keys (Product + Date).
    -   Validates price positiveness.
3.  **Refactor Ingester**: Update `agents/weekly_prices_ingester.py` to conform to this standard (if needed).

## Acceptance Criteria
- [ ] `docs/WEEKLY_DATA_STANDARD.md` exists and is comprehensive.
- [ ] `scripts/validate_weekly_data.py` passes on current dataset.
- [ ] Current data follows the standard (or is fixed to follow it).
