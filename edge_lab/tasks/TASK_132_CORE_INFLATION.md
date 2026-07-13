# Task 132: True Core Inflation

## Context
Headline CPI is noisy. "Exclude Food & Energy" is the standard cleaning method, but is it the best for KBR? Maybe "Exclude Cucumbers & Plane Tickets" is better.

## Objective
Develop alternative **Core Inflation** measures that predict future headline inflation better than the official CPI.

## Algorithms to Implement
1.  **Trimmed Mean**:
    -   Sort components by MoM change.
    -   Cut the top $\alpha$% and bottom $\beta$% of *weights*.
    -   Calculate weighted average of the middle.
    -   *Experiment*: Optimize $\alpha, \beta$ (e.g., Trim 10%, Trim 20%).
2.  **Weighted Median**:
    -   The inflation rate of the component at the 50th cumulative weight percentile.
    -   Robust to extreme outliers.
3.  **Volatility Exclusion**:
    -   Permanently exclude the Top N most volatile items found in Task 127.

## Validation metric
**Cogley-Sargent Test**: A good core measure ($ \pi^C $) should predict future headline inflation ($ \pi^H $).
-   Regress: $ \pi^H_{t+h} - \pi^H_t = \alpha + \beta (\pi^C_t - \pi^H_t) + \epsilon $
-   If $\beta \approx 1$, Core attracts Headline (Good).

## Output
-   `data/kbr_core_inflation_measures.csv`.
-   `reports/core_measure_evaluation.md`.

## Acceptance Criteria
- [ ] At least 3 Core measures computed (Trimmed, Median, Exclusion).
- [ ] "Winning" Core measure identified based on predictive power.
