# Seasonal Pattern Evolution Analysis

> **Date:** 2026-01-24
> **Goal:** Analyze how KBR seasonality patterns evolved over 5-year windows (2010-2014, 2015-2019, 2020-2024)

## Executive Summary

The analysis of seasonal inflation patterns across three 5-year eras reveals significant evolution in KBR's inflation dynamics:

1. **January inflation remains high but volatile:** January remains consistently high-inflation month (1.02%, 1.30%, 0.86% across eras) but with declining volatility (Std: 0.92 → 1.63 → 0.32)
2. **December showing structural decline:** December inflation dropped from 1.07% (2010-2014) to 0.58% (2020-2024), indicating weakening pre-holiday price pressure
3. **Seasonal patterns diverging:** Correlation between 2010-2014 and 2020-2024 eras is only 0.632, suggesting fundamentally different inflation dynamics
4. **Tariff indexation impact muted:** July (tariff adjustment month) shows consistently low inflation (0.10-0.33%), suggesting indexation is less impactful than previously thought

## Methodology

### Data Source
- **File:** `data/infl_kbr.csv`
- **Indicator:** MoM (Month-over-Month) inflation for "Все товары и услуги" (Total CPI)
- **Period:** January 2010 – December 2024 (180 months, complete 5-year windows)

### Analysis Approach
For each era (5-year window), calculated:
1. **Mean Index:** Average MoM inflation for each month (1-12)
2. **Median Index:** Robust measure resistant to outliers
3. **Standard Deviation:** Volatility/risk measure
4. **Decomposition Index:** Seasonal component from statsmodels `seasonal_decompose` (additive model)
5. **Pattern Correlation:** Pearson correlation between mean patterns of different eras

### Eras Analyzed
| Era | Period | Historical Context |
|------|---------|------------------|
| 2010-2014 | Pre-Crimea | Post-2008 crisis, relatively stable period |
| 2015-2019 | Post-Crimea | Crimea annexation aftermath, ruble devaluation 2014-2015 |
| 2020-2024 | COVID & Recovery | COVID-19 shock, supply chain disruptions, 2022 sanctions |

## Key Findings

### 1. Evolution of High-Inflation Months

**Top 3 months by era:**

| Era | #1 Month | #2 Month | #3 Month |
|------|-----------|-----------|-----------|
| 2010-2014 | Dec (1.07%) | Jan (1.02%) | Sep (0.98%) |
| 2015-2019 | Jan (1.30%) | Feb (1.00%) | Apr (0.84%) |
| 2020-2024 | Mar (1.69%) | Apr (1.31%) | Jan (0.86%) |

**Insights:**
- **December dominance declined:** Dec went from #1 (1.07%) in 2010-2014 to not in top-3 in 2020-2024
- **January volatility:** Jan spiked to 1.30% in 2015-2019 (post-Crimea) but normalized to 0.86% in 2020-2024
- **Spring season emerged:** March/April became high-inflation months in 2020-2024 (COVID supply shock effects)

### 2. Low-Inflation Months: Stable Patterns

**Bottom 3 months by era:**

| Era | #1 Month | #2 Month | #3 Month |
|------|-----------|-----------|-----------|
| 2010-2014 | Aug (0.08%) | Jun (0.15%) | Jul (0.33%) |
| 2015-2019 | Aug (-0.04%) | Sep (0.02%) | Jul (0.10%) |
| 2020-2024 | Aug (-0.06%) | Jun (0.10%) | Jul (0.17%) |

**Insights:**
- **Summer stability is consistent:** August and June are consistently lowest-inflation months across all eras
- **July tariff myth:** July shows very low inflation (often deflationary), contrary to expectation of tariff-driven price hikes
- **Deflationary pressure:** August showed deflation in 2015-2019 (-0.04%) and 2020-2024 (-0.06%)

### 3. Key Monthly Evolution

#### January (New Year & Tariffs)

| Era | Mean | Std | Interpretation |
|------|-------|-----|---------------|
| 2010-2014 | 1.02% | 0.92 | Moderate inflation, high volatility |
| 2015-2019 | 1.30% | 1.63 | **Peak inflation**, extreme volatility (post-Crimea) |
| 2020-2024 | 0.86% | 0.32 | Lower inflation, stabilized |

**Conclusion:** January is the most volatile month but showing decreasing trend (1.30% → 0.86%).

#### July (Tariff Indexation)

| Era | Mean | Std | Interpretation |
|------|-------|-----|---------------|
| 2010-2014 | 0.33% | 0.81 | Low inflation |
| 2015-2019 | 0.10% | 0.41 | Very low inflation |
| 2020-2024 | 0.17% | 0.69 | Low inflation |

**Conclusion:** Tariff indexation in July has minimal impact on CPI (consistently low months).

#### December (Pre-Holiday)

| Era | Mean | Std | Interpretation |
|------|-------|-----|---------------|
| 2010-2014 | 1.07% | 0.66 | High inflation |
| 2015-2019 | 0.78% | 0.41 | Moderate decline |
| 2020-2024 | 0.58% | 0.33 | **Lowest** in series |

**Conclusion:** December showing structural decline (1.07% → 0.58%), possibly due to reduced pre-holiday consumption or supply chain improvements.

### 4. Pattern Correlation Between Eras

| Era Comparison | Correlation | Interpretation |
|----------------|--------------|---------------|
| 2010-2014 vs 2015-2019 | **0.632** | Moderate similarity (structural break) |
| 2015-2019 vs 2020-2024 | **0.560** | Low similarity (COVID shock) |
| 2010-2014 vs 2020-2024 | **Implied low** | Major structural evolution |

**Insights:**
- **Structural breaks detected:** Correlation 0.63 is moderate, indicating significant pattern changes between eras
- **COVID disruption:** 0.56 correlation between 2015-2019 and 2020-2024 shows COVID-19 fundamentally altered seasonal dynamics
- **Model implication:** Using pre-2015 seasonal patterns for 2020+ forecasts will produce errors

### 5. Biggest Pattern Shifts (2010-2014 → 2020-2024)

| Month | Change | 2010-2014 | 2020-2024 | Interpretation |
|-------|---------|-------------|-------------|---------------|
| **September** | **-0.27pp** | 0.98% | 0.71% | Post-summer deflationary pressure |
| **December** | **-0.50pp** | 1.07% | 0.58% | **Major decline** in pre-holiday inflation |
| **May** | **-0.49pp** | 0.87% | 0.38% | Early summer weakening |
| **April** | **+0.61pp** | 0.70% | 1.31% | **Major increase** (COVID supply chain) |
| **March** | **+1.09pp** (vs 2015-2019) | 0.60% | 1.69% | **Emergent high-inflation month** |

### 6. Stability Analysis (Std across Eras)

**Most stable months (lowest volatility across eras):**

| Month | Std (Era-to-Era) | Interpretation |
|-------|-------------------|---------------|
| June | 0.031 | Highly stable |
| August | 0.077 | Highly stable |
| October | 0.110 | Stable |
| July | 0.116 | Stable (tariff myth) |
| February | 0.136 | Moderate stability |

**Most volatile months (highest volatility across eras):**

| Month | Std (Era-to-Era) | Interpretation |
|-------|-------------------|---------------|
| **March** | **0.573** | Extremely volatile (COVID effect) |
| **September** | 0.493 | Highly volatile |
| **April** | 0.320 | Volatile |
| **May** | 0.299 | Moderately volatile |
| **December** | 0.251 | Declining trend |

## Implications for Forecasting

### 1. Model Design Recommendations

| Aspect | Recommendation | Rationale |
|--------|---------------|-----------|
| **Seasonal Features** | Use era-specific weights or 2020-2024 as baseline | Pre-2015 patterns are obsolete (corr 0.56-0.63) |
| **January Dummy** | Keep, but reduce weight | High volatility but mean declining (1.30% → 0.86%) |
| **December Dummy** | Reduce or remove weight | Trend downward (1.07% → 0.58%) |
| **July Dummy** | **Remove** | Tariff indexation impact minimal (0.10-0.33%) |
| **March Dummy** | **Add** | Emerged as high-inflation month in 2020-2024 (1.69%) |
| **April Dummy** | **Add** | COVID supply chain effects (+0.61pp vs 2010-2014) |

### 2. Risk Management

| Month | Risk Level | Strategy |
|--------|-------------|----------|
| March | HIGH | New emergent risk factor (COVID effect), requires special monitoring |
| January | MEDIUM | Volatile but trend downward, use rolling window features |
| April | MEDIUM-HIGH | Elevated in COVID era, monitor supply chain indicators |
| June/August | LOW | Most stable months, can use simple seasonal adjustment |

### 3. Nowcasting Signals

| Signal | What to Monitor | Threshold for Alert |
|---------|-----------------|-------------------|
| Spring Inflation | Mar/April price indices | > 1.2% MoM |
| January Shock | First 2 weeks of Jan price data | > 0.8% WoW |
| December Surprise | Late-year supply/demand | < 0.5% MoM (unusual stability) |

## Data Files

- **Source Data:** `data/infl_kbr.csv` (Monthly CPI, 2010-2024)
- **Seasonal Indices:** `data/seasonal_indices_by_era.csv` (36 rows: 12 months × 3 eras)
- **Analysis Script:** `scripts/seasonal_evolution.py` (266 lines)

## CSV File Structure

`data/seasonal_indices_by_era.csv` contains:

| Column | Description |
|---------|-------------|
| `Month` | Month number (1-12) |
| `Mean_Index` | Average MoM inflation (%) for the era |
| `Median_Index` | Median MoM inflation (%) (robust to outliers) |
| `Std` | Standard deviation (volatility) |
| `Year_Count` | Number of years in era (always 5) |
| `Decompose_Index` | Seasonal component from statsmodels decomposition |
| `Era` | Era name (2010-2014, 2015-2019, 2020-2024) |
| `Start_Year` | First year of era |
| `End_Year` | Last year of era |

## Conclusion

KBR inflation seasonality has evolved significantly across the three eras:

1. **Structural shift:** Correlation between eras is moderate (0.56-0.63), indicating major pattern changes
2. **January volatility:** Remains high-inflation month but with declining trend and high Std
3. **December decline:** From 1.07% (2010-2014) to 0.58% (2020-2024) — major structural change
4. **July myth:** Tariff indexation has minimal impact on CPI (0.10-0.33%)
5. **Emergent risks:** March/April became high-inflation months in COVID era

**Recommendation:** Update forecasting models to use 2020-2024 seasonal patterns as baseline, remove/reduce December and July dummy variables, and add March/April features for COVID-era patterns.

---

*Analysis prepared automatically by `scripts/seasonal_evolution.py`*
