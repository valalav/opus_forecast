# Task 128: Precision Seasonal Calendar

## Context
"Vegetables rise in winter" is too vague. We need: "Cucumbers peak in Week 7 (mid-Feb) and trough in Week 32 (early Aug)."

## Objective
Create a machine-readable "Almanac" of seasonal cycles for every product.

## Methodology
1.  **Decomposition**: Apply STL (Seasonal-Trend-Loess) to `price` (log-transformed) or `wow_growth`.
    -   Period = 52 (Weekly).
2.  **Peak Detection**:
    -   Extract the seasonal component $S_t$.
    -   Find the week indices ($w \in [1, 52]$) where $S_t$ is maximal and minimal.
    -   Calculate "Seasonal Amplitude" (Max - Min).
3.  **Consistency Check**:
    -   Does the peak happen in the same week every year? Calculate `StdDev(Peak_Week)`.
    -   Keep only products with stable seasonality (StdDev < 4 weeks).

## Output Artifacts
-   `data/seasonal_calendar.json`:
    ```json
    {
      "1001": {
        "name": "Beef",
        "peak_week": 51,
        "trough_week": 20,
        "amplitude_pct": 2.5,
        "consistency_score": 0.9
      }
    }
    ```

## Acceptance Criteria
- [ ] JSON file generated with >100 products.
- [ ] Validation: Does the "Peak Week" align with common sense for 5 known seasonal items (Eggs, Cucumbers, Sugar)?
