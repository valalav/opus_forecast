# Task 127: The Volatility Radar

## Context
Not all price changes matter. A 50% jump in "Matches" is noise; a 5% jump in "Beef" is a crisis. We need to filter signal from noise using volatility and weights.

## Objective
Identify the **Top 20 Risk Factors** in the KBR consumer basket.

## Analytical Approach
1.  **Load Data**: Weekly prices + Access Weights.
2.  **Feature Engineering**:
    -   `WoW_Growth`: Percentage change.
    -   `Volatility_12w`: Rolling 12-week standard deviation.
    -   `Impact_Score`: `Volatility_12w * Weight`.
3.  **Regime Detection**:
    -   Use `ruptures` or simple rolling window statistics to find *structural breaks* in volatility.
    -   Flag items that have transitioned from "Stable" to "Volatile" in 2024-2025.
4.  **Reporting**:
    -   Generate `reports/volatility_radar.md`.
    -   Visualize: Scalar Plot (X=Weight, Y=Volatility). Top-right quadrant = DANGER ZONE.

## Acceptance Criteria
- [ ] Report identifies top 20 "Impact" items.
- [ ] Report identifies at least 5 "Sleeping Giants" (High weight, historically low volatility).
- [ ] Analysis covers entire period 2008-2026.
