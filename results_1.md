# AStats Evaluation: Ground Truth vs System Output

## Legend
- ✅ = Correctly identified/enforced
- ❌ = Missed or incorrect
- 🟡 = Partially correct

| # | Dataset | Data Type | Ground Truth: Required Tests | System Predicted: Allowed | Match? | Ground Truth: Forbidden Tests | System Predicted: Forbidden | Match? |
|---|---------|-----------|------------------------------|---------------------------|--------|-------------------------------|----------------------------|--------|
| 1 | Paired Pre/Post (60 rows, PatientID repeated) | Repeated Measures | Paired t-test, Wilcoxon signed-rank, Repeated Measures ANOVA, LMM/GEE | Repeated Measures ANOVA, LMM, GEE, Paired t-test, Wilcoxon signed-rank | ✅ | Independent t-test, OLS, One-way ANOVA, Mann-Whitney U | OLS, Independent t-test, One-way ANOVA (Independent), Mann-Whitney U | ✅ |
| 2 | A/B Test (100 rows, cross-sectional) | Independent Groups | Independent t-test, Mann-Whitney U, One-way ANOVA | Independent t-test, One-way ANOVA, Mann-Whitney U | ✅ | Paired t-test, Repeated Measures ANOVA, Wilcoxon signed-rank | Repeated Measures ANOVA, LMM, Paired t-test, Wilcoxon signed-rank | ✅ |
| 3 | Binary Outcome (201 rows, Subscribed 0/1) | Binary Classification | Logistic Regression, Chi-Square | Logistic Regression, Chi-Square Test | ✅ | OLS Linear Regression | OLS Linear Regression | ✅ |
| 4 | Survival Data (151 rows, Failed + DaysToFailure) | Time-to-Event / Censored | Cox Proportional Hazards, Kaplan-Meier, Log-Rank Test | Cox PH, Kaplan-Meier, Log-Rank Test | ✅ | OLS Linear Regression, Logistic Regression | OLS Linear Regression | 🟡 |
| 5 | Hierarchical (200 rows, Students nested in Schools) | Nested / Clustered | Mixed-Effects Models (LMM/HLM), Random Intercepts/Slopes | LMM / HLM, Random Intercepts/Slopes | ✅ | Standard OLS (ignoring nesting) | Standard OLS (ignoring nested structure) | ✅ |

## Detection Accuracy Summary

| Component | Accuracy |
|-----------|----------|
| Pandas Heuristic Detection (structure flags) | **5/5 (100%)** |
| Deterministic Constraint Engine (allowed methods) | **5/5 (100%)** |
| Deterministic Constraint Engine (forbidden methods) | **4.5/5 (90%)** |
| Overall Constraint Accuracy | **~97%** |

### Notes on Dataset 4 (Survival)
- The system correctly forbids OLS but does **not** explicitly forbid Logistic Regression for survival data (ground truth says it should be forbidden since the outcome is time-to-event, not binary classification).
- The `Failed` column triggers `has_binary_outcome: true` in addition to `has_survival_data: true` because it has exactly 2 unique values (0/1). This is a **false positive** on the binary flag — `Failed` is an event indicator, not a classification target. This is an area for refinement.
