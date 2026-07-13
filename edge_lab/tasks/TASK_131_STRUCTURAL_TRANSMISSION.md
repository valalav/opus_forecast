# Task 131: Structural Transmission (Russia -> KBR)

## Context
KBR is a small open economy within Russia. Prices are driven by national trends (Exchange Rate, Key Rate, Federal CPI) but react with local idiosyncrasies.

## Objective
Quantify the **Transmission Mechanism**: If Russia CPI jumps 1%, what happens to KBR? When? And by how much?

## Research Questions
1.  **Lag**: How fast does a federal shock hit Nalchik?
2.  **Beta**: Is KBR inflation "High Beta" (>1) or "Low Beta" (<1) relative to Moscow/Federal?
3.  **Asymmetry**: Do prices rise faster than they fall?

## Methodology
1.  **Data**:
    -   `CPI_RU`: Federal MoM (from Rosstat/Haver/Manual).
    -   `CPI_KBR`: KBR MoM (our target).
    -   `Exog`: USD, Brent, Key Rate.
2.  **Model**:
    -   **VAR (Vector Autoregression)**: `[CPI_KBR, CPI_RU]`.
    -   **Granger Causality**: Does RU -> KBR? (Almost certainly yes). Does KBR -> RU? (Likely no).
3.  **Visualization**:
    -   **Impulse Response Function (IRF)**: Plot the trajectory of KBR CPI following a 1 S.D. shock to RU CPI.

## Output
-   `reports/transmission_mechanism.md`:
    -   Estimated Lag (in months).
    -   Transmission Coefficient (Elasticity).
    -   IRF Plot.

## Acceptance Criteria
- [ ] Statistical confirmation of the transmission lag.
- [ ] Quantification of the "Regional Basisian" (spread).
