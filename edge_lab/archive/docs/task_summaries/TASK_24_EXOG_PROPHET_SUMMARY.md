# Task 24: Improve ExogProphet - Summary

## Task Status: NOT COMPLETE ❌

### Acceptance Criteria
- **Required**: MAE <= 0.30
- **Actual**: MAE = 0.7738
- **Status**: FAILED ✗

---

## Findings

### 1. Brent Regressor Implementation
The Brent regressor IS already implemented in `/home/valalav/_projects/sirena-kbr/sirena/models/exog_prophet.py`:
- `use_brent` parameter (default: True)
- `BRENT_LAG = 5` constant
- `brent_lag5` feature in `_prepare_features()`
- Brent data loaded from `/home/valalav/_projects/sirena-kbr/data/brent_prices.csv`

### 2. Date Alignment Bug Discovered
A date alignment issue was found:
- **Brent data**: `2010-01-01, 2010-02-01, ...` (1st of month)
- **Macro data**: `2010-01-31, 2010-02-28, ...` (end of month)

This causes the `join()` operation to fail, resulting in all NaN `brent_lag5` values.

### 3. Fix Attempted
Created wrapper `ExogProphetBrentFixed` in `edge_lab/exog_prophet_fix.py` that reindexes Brent data to match macro dates.

### 4. Backtest Results
```
Test period: 2019-01-31 to 2025-12-31
Number of forecasts: 84
MAE: 0.7738
ME (Mean Error): 0.1621
RMSE: 1.0524
```

### 5. Comparison: With vs Without Brent
- MAE (without Brent): 0.7738
- MAE (with Brent): 0.7738
- Improvement: +0.00%

The Brent regressor shows **NO IMPROVEMENT**, suggesting the feature is either:
1. Not properly integrated despite the fix
2. Not predictive for the target period
3. Requires parameter tuning

---

## Test Files Created

1. `edge_lab/tests/test_exog_prophet_forecaster.py` - Unit tests (10/10 passed)
2. `edge_lab/debug_exog_prophet.py` - Debug script to identify issues
3. `edge_lab/verify_exog_prophet_brent.py` - Original verification
4. `edge_lab/verify_exog_prophet_brent_fixed.py` - Fixed version verification
5. `edge_lab/exog_prophet_fix.py` - Wrapper to fix date alignment

---

## Recommendations

1. **Core Issue**: The ExogProphetForecaster in sirena/models/exog_prophet.py needs the date alignment fix incorporated
2. **Performance**: Even with the fix, MAE (0.77) is far from target (0.30)
3. **Tuning**: Model hyperparameters may need optimization:
   - `changepoint_prior_scale`
   - `seasonality_prior_scale`
   - Lag periods (currently 2, 5, 6)
4. **Alternative**: Consider a different approach to integrating Brent oil prices

---

## Files Modified/Created (in edge_lab only)

| File | Purpose |
|------|---------|
| tests/test_exog_prophet_forecaster.py | Unit tests for ExogProphet |
| debug_exog_prophet.py | Debug script |
| verify_exog_prophet_brent.py | Original verification |
| verify_exog_prophet_brent_fixed.py | Fixed verification |
| exog_prophet_fix.py | Date alignment patch |

---

## Conclusion

The Brent regressor feature exists in the codebase but:
1. Has a date alignment bug preventing it from working correctly
2. Even after fixing the bug, performance (MAE=0.77) does NOT meet the acceptance criterion (MAE<=0.30)
3. The Brent regressor shows no measurable improvement when enabled

**Task 24 CANNOT be marked as complete** because the acceptance criterion is not met.

---

## Note

Since this WORKER agent is restricted from modifying files outside `/home/valalav/_projects/sirena-kbr/edge_lab`, the actual fix to `sirena/models/exog_prophet.py` cannot be applied. The patch was created in `edge_lab/exog_prophet_fix.py` as a workaround.
