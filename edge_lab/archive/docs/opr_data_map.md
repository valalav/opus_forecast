# OPR Statistics Data Map

## Overview
Analysis of the OPR (ОПР) statistics dataset from `assets/charts/ОПР_статистика/`.

**Primary Files:**
- `Основная статистика ЮГУ.xlsx` (144MB) - Main YUGU (South Russia) data
- `New_Итоговый протокол+идеальные коды.xlsx` (281KB) - Indicator metadata

**Analysis Date: 2026-01-22**

---

## Data Structure

### Sheets in YUGU File
The main file contains 141 data sheets:

1. **Для главной** - Date reference sheet
2. **Главная** - Main table of contents
3. **Numbered sheets (100-996)** - Individual indicators

### Data Granularity
- **Frequency:** Mix of monthly, quarterly, and annual data
- **Geographic Coverage:**
  - Russian Federation (RF)
  - Southern Federal District (YUGU/SKFO)
  - Individual subjects including KBR
- **Time Periods:** Varies by indicator (typically from 2006-2024)

### Data Lag
Based on the indicator metadata:
- Monthly indicators: Typically 1-2 month lag
- Quarterly indicators: 2-3 month lag
- Annual indicators: 4-6 month lag

---

## Regions Available

The YUGU dataset includes data for the following regions:
- **KBR** - Kabardino-Balkarian Republic (target region)
- **YUGU** - Southern Federal District (macro proxy)
- **RF** - Russian Federation (national proxy)
- **Krasnodar** - Krasnodar Krai (regional proxy)
- **Stavropol** - Stavropol Krai (regional proxy)
- **Rostov** - Rostov Oblast (regional proxy)

Sheets with KBR data: 94

---

## Top-20 Proxy Series for KBR Inflation

The following series are ranked by their potential as macro-regressors for KBR inflation forecasting:

### 1. Sheet 100: ИПЦ

- **Data Mart:** 
- **Frequency:** месячная
- **Time Depth:** 
- **Relevance Score:** 125
- **Why Relevant:** DIRECT CPI - Primary inflation target, Monthly frequency

### 2. Sheet 101: ИПЦ

- **Data Mart:** 
- **Frequency:** месячная
- **Time Depth:** 
- **Relevance Score:** 125
- **Why Relevant:** DIRECT CPI - Primary inflation target, Monthly frequency

### 3. Sheet 392: Индекс производства (оперативные данные) (ОКВЭД 2)

- **Data Mart:** Росстат: Статистика промышленности и производства
- **Frequency:** месячная
- **Time Depth:** с 2015
- **Relevance Score:** 65
- **Why Relevant:** Production indicator, Monthly frequency

### 4. Sheet 400: Дефицит денежного дохода в стоимостном выражении (до 1998 г. - в трлн.руб.)

- **Data Mart:** Росстат: Доходы и уровень жизни населения
- **Frequency:** годовая
- **Time Depth:** c 1992 г.
- **Relevance Score:** 65
- **Why Relevant:** Income indicator, Deep time series

### 5. Sheet 403: Структура денежных доходов по источникам формирования

- **Data Mart:** Росстат: Доходы и уровень жизни населения
- **Frequency:** квартальная
- **Time Depth:** c 1990 г.
- **Relevance Score:** 65
- **Why Relevant:** Income indicator, Deep time series

### 6. Sheet 994: Задолженность по платежам в государственные внебюджетные фонды из общей суммы кредиторской задолженности крупных и средних предприятий и организаций c 2017 г.

- **Data Mart:** Росстат: Финансы, финансовая деятельность и информационное статистическое обеспечение оценки эффективности бюджетных расходов
- **Frequency:** месячная
- **Time Depth:** c 2017 г.
- **Relevance Score:** 55
- **Why Relevant:** Budget indicator, Monthly frequency

### 7. Sheet 340: Реальные денежные доходы

- **Data Mart:** 
- **Frequency:** квартальная
- **Time Depth:** 
- **Relevance Score:** 50
- **Why Relevant:** Income indicator

### 8. Sheet 341: Среднедушевые денежные доходы

- **Data Mart:** 
- **Frequency:** квартальная
- **Time Depth:** 
- **Relevance Score:** 50
- **Why Relevant:** Income indicator

### 9. Sheet 342: Совокупные денежные доходы

- **Data Mart:** 
- **Frequency:** квартальная
- **Time Depth:** 
- **Relevance Score:** 50
- **Why Relevant:** Income indicator

### 10. Sheet 501: Земельный налог по 2016 г.

- **Data Mart:** Росстат: Общеэкономические показатели деятельности организаций и мониторинги важнейших проблем социально-экономической сферы
- **Frequency:** квартальная
- **Time Depth:** c 2006 г.
- **Relevance Score:** 45
- **Why Relevant:** Budget indicator, Deep time series

### 11. Sheet 388: Объем выданных кредитов крупным ЮЛ

- **Data Mart:** Росстат: Статистика промышленности и производства
- **Frequency:** месячная
- **Time Depth:** c 2008 г.
- **Relevance Score:** 40
- **Why Relevant:** Monthly frequency, Deep time series

### 12. Sheet 389: Индекс промышленного производства (оперативные данные)

- **Data Mart:** Росстат: Статистика промышленности и производства
- **Frequency:** месячная
- **Time Depth:** с 2000 г.
- **Relevance Score:** 40
- **Why Relevant:** Monthly frequency, Deep time series

### 13. Sheet 421: Депозиты юридических лиц

- **Data Mart:** Росстат: Доходы и уровень жизни населения
- **Frequency:** месячная
- **Time Depth:** c 1992 г.
- **Relevance Score:** 40
- **Why Relevant:** Monthly frequency, Deep time series

### 14. Sheet 119: 

- **Data Mart:** 
- **Frequency:** месячная
- **Time Depth:** 
- **Relevance Score:** 25
- **Why Relevant:** Monthly frequency

### 15. Sheet 142: ИБВЭД

- **Data Mart:** 
- **Frequency:** месячная
- **Time Depth:** 
- **Relevance Score:** 25
- **Why Relevant:** Monthly frequency

### 16. Sheet 180: ИПП

- **Data Mart:** 
- **Frequency:** месячная
- **Time Depth:** 
- **Relevance Score:** 25
- **Why Relevant:** Monthly frequency

### 17. Sheet 181: ИПП

- **Data Mart:** 
- **Frequency:** месячная
- **Time Depth:** 
- **Relevance Score:** 25
- **Why Relevant:** Monthly frequency

### 18. Sheet 182: ИПП

- **Data Mart:** 
- **Frequency:** месячная
- **Time Depth:** 
- **Relevance Score:** 25
- **Why Relevant:** Monthly frequency

### 19. Sheet 183: ИПП

- **Data Mart:** 
- **Frequency:** месячная
- **Time Depth:** 
- **Relevance Score:** 25
- **Why Relevant:** Monthly frequency

### 20. Sheet 184: ИПП

- **Data Mart:** 
- **Frequency:** месячная
- **Time Depth:** 
- **Relevance Score:** 25
- **Why Relevant:** Monthly frequency

---

## Key Insights

1. **Direct Price Indicators (Top Priority):**
   - CPI (ИПЦ) data is available at both YUGU district and KBR regional levels
   - These are the primary targets for inflation forecasting
   - Monthly frequency allows for fine-grained analysis

2. **Production Indicators (High Priority):**
   - Industrial production indices by sector
   - Monthly data with good historical coverage
   - Strong correlation with inflation through supply-side factors

3. **Labor Market Indicators (Medium-High Priority):**
   - Wages and unemployment rates
   - Monthly frequency
   - Correlated with inflation through demand-side channels

4. **Housing Indicators (Medium Priority):**
   - Housing prices from DomClick and other sources
   - Both announced and actual transaction prices
   - Important component of the CPI basket

5. **Budget Indicators (Low-Medium Priority):**
   - Consolidated budgets of RF subjects
   - Monthly/quarterly data
   - Correlates with fiscal policy and inflation expectations

---

## Data Quality Notes

From the protocol file:
- Some indicators have missing data for certain time periods
- Quarterly cumulative data may have format inconsistencies
- A few indicators have data quality issues noted in the comments

---

## Recommendations for Forecasting Pipeline

1. **Primary Features:** Use CPI data from YUGU as a leading indicator for KBR
2. **Secondary Features:** Include production indices, wages, and unemployment
3. **Regional Hierarchies:** Leverage RF → YUGU → KBR hierarchical relationships
4. **Data Frequency:** Prioritize monthly indicators for model training
5. **Missing Data:** Implement interpolation for gaps in time series

---

## Next Steps

1. Extract KBR-specific time series from identified sheets
2. Calculate correlations between YUGU and KBR inflation
3. Build feature engineering pipeline for the top-20 proxies
4. Integrate with existing KBR inflation forecasting model
