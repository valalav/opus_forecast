# Task 129: Weekly Bridge Nowcaster

## Context
We wait 10-15 days after month-end for official CPI. Weekly data arrives on Wednesday. We can know the future.

## Objective
Build a model to predict the official **Monthly CPI** using accumulated **Weekly Data**.

## Model Design
**Equation:**
$$ \text{CPI}_{Month} = \alpha + \beta_1 \cdot \text{Accumulated\_Weekly\_Index} + \beta_2 \cdot \text{Lagged\_CPI} + \epsilon $$

### Sub-problems
1.  **Aggregation**: How to turn 4 (or 5) weekly price points into a monthly proxy?
    -   *Method A*: Average of weekly levels / Average of prev month levels.
    -   *Method B*: End-of-month / End-of-prev-month.
    -   *Method C*: Product of (1 + Weekly_Inflation).
    -   *Task*: Test which aggregation correlates best with Official MoM.
2.  **Partial Knowledge**:
    -   Model 1: `W1_only` (Day 7).
    -   Model 2: `W1+W2` (Day 14).
    -   Model 3: `W1+W2+W3` (Day 21).
    -   Model 4: `Full_Month` (Day 28+).

## Acceptance Criteria
- [ ] `models/nowcaster_w3.pkl` saved (Day 21 model).
- [ ] RMSE on test set (2024-2025) < 0.20 p.p.
- [ ] Report: "Accuracy gain vs Naive Baseline" (Does knowing 3 weeks of prices actually help?).
