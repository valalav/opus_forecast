---
name: econometrician
description: Use this skill when you need deep statistical analysis, anomaly detection, or hypothesis testing on data. It uses Opencode (LLM) to act as a Senior Econometrician.
---

# Econometrician Skill

**Goal**: Provide expert-level econometric analysis of time series data using `opencode` as the reasoning engine.

## Capabilities
1.  **Stationarity Checks**: Analyze trend/seasonality/noise.
2.  **Anomaly Explanation**: Look at data and propose economic reasons for spikes.
3.  **Model Critique**: Review model residuals for autocorrelation or heteroscedasticity.

## Instructions
1.  **Prepare Data**: Ensure data is in a CSV file (e.g., `data/cpi.csv`).
2.  **Execute**: Run `scripts/analyze.py --file <csv_path> --query <question>`.
3.  **Output**: The script will feed the data + question to `opencode` and return a Markdown report.

## Usage
```bash
python edge_lab/skills/econometrician/scripts/analyze.py --file "data/infl_kbr.csv" --query "Is the post-2022 inflation regime structurally different from 2018-2021?"
```

## Verification
- Check that the output contains "Econometric Analysis Report".
- Verify that specific tests (ADF, KPSS) were recommended or interpreted if relevant.
