"""
statistical_constraint_agent.py - Deterministic Rule Engine for Statistical Constraints.
Reads data_structure.json to produce constraints.json restricting analytical methods.
Zero LLM calls. 100% Rule-Based.
"""
import os
import json
from utils.file_utils import ensure_output_dir


def run(profile_text: str, data_structure_text: str, output_dir: str) -> str:
    """
    Run the Statistical Constraint rule engine.
    (profile_text is passed for compatibility with the pipeline, but rules focus on structure).

    Returns
    -------
    str : absolute path to the generated constraints.json
    """
    ensure_output_dir(output_dir)

    print("\n[StatisticalConstraintAgent] Evaluating statistical constraints via Deterministic Rules Engine...")
    
    # 1. Parse the structure inference
    try:
        structure = json.loads(data_structure_text)
    except Exception as e:
        print(f"[StatisticalConstraintAgent] Warning: Could not parse data structure JSON. Applying safe defaults.")
        structure = {}

    has_repeated = structure.get("has_repeated_measures", False)
    has_indep = structure.get("has_independent_groups", False)
    has_multiclass = structure.get("has_multiclass_nominal_outcome", False)

    allowed = []
    forbidden = []
    warnings = []

    # -------------------------------------------------------------
    # 2. Strict Rules Engine
    # -------------------------------------------------------------
    if has_repeated:
        allowed.extend([
            "Repeated Measures ANOVA",
            "Linear Mixed-Effects Models (LMM)",
            "Generalized Estimating Equations (GEE)",
            "Paired / Dependent t-test",
            "Wilcoxon signed-rank test"
        ])
        forbidden.extend([
            "Ordinary Least Squares (OLS) Linear Regression",
            "Independent / Two-Sample t-test",
            "One-way ANOVA (Independent)",
            "Mann-Whitney U test"
        ])
        warnings.append("[CRITICAL] Data contains repeated measures or dependent samples. Do not treat rows as independent observations. OLS is strictly forbidden.")
    else:
        allowed.extend([
            "Ordinary Least Squares (OLS) Linear Regression",
            "Independent / Two-Sample t-test",
            "One-way ANOVA",
            "Mann-Whitney U test"
        ])
        forbidden.extend([
            "Repeated Measures ANOVA",
            "Linear Mixed-Effects Models (LMM)",
            "Paired / Dependent t-test",
            "Wilcoxon signed-rank test"
        ])
        warnings.append("Observations are assumed independent (Cross-sectional data).")

    if has_indep:
        allowed.extend([
            "Independent group comparisons (e.g., test differences between groups)"
        ])
    else:
        warnings.append("No independent grouping detected. Focus on continuous regression or within-subject time trends instead of group-wise comparisons.")

    # RULE 3: Binary Outcome
    has_binary = structure.get("has_binary_outcome", False)
    if has_binary:
        # Remove OLS if it was added by a prior rule
        allowed = [m for m in allowed if "OLS" not in m and "Ordinary Least Squares" not in m]
        forbidden.append("Ordinary Least Squares (OLS) Linear Regression")
        allowed.append("Logistic Regression")
        allowed.append("Chi-Square Test (for categorical predictors)")
        warnings.append("CRITICAL: The target outcome is binary (0/1). OLS Linear Regression is FORBIDDEN. Use Logistic Regression.")

    # RULE 3B: Multiclass Nominal Outcome
    if has_multiclass:
        allowed = [m for m in allowed if "OLS" not in m and "Ordinary Least Squares" not in m]
        forbidden.append("Ordinary Least Squares (OLS) Linear Regression")
        allowed.append("Multinomial Logistic Regression")
        allowed.append("Softmax / Multiclass Classification Models")
        warnings.append("CRITICAL: The target outcome has 3+ unordered classes. Do not use OLS. Use multinomial logistic or another multiclass classification model.")

    # RULE 4: Survival / Time-to-Event Data
    has_survival = structure.get("has_survival_data", False)
    if has_survival:
        allowed = [m for m in allowed if "OLS" not in m and "Ordinary Least Squares" not in m]
        forbidden.append("Ordinary Least Squares (OLS) Linear Regression")
        allowed.append("Cox Proportional Hazards Model")
        allowed.append("Kaplan-Meier Estimator")
        allowed.append("Log-Rank Test")
        warnings.append("CRITICAL: This is survival/time-to-event data with censoring. Use Cox PH or Kaplan-Meier. Standard regression is FORBIDDEN.")

    # RULE 5: Hierarchical / Nested Data
    is_hierarchical = structure.get("is_hierarchical", False)
    if is_hierarchical:
        allowed = [m for m in allowed if "OLS" not in m and "Ordinary Least Squares" not in m]
        forbidden.append("Standard OLS Linear Regression (ignoring nested structure)")
        allowed.append("Linear Mixed-Effects Models (LMM) / Hierarchical Linear Modeling (HLM)")
        allowed.append("Random Intercepts/Slopes Models")
        warnings.append("CRITICAL: Data has hierarchical/nested structure. Observations are NOT independent. Use Mixed-Effects models.")

    # =============================================================
    # Phase 2: Assumption-Based Rules
    # =============================================================
    assumptions = structure.get("assumption_tests", {})

    # RULE 6: Non-Normal Data
    normality_violations = assumptions.get("normality_violations", [])
    if len(normality_violations) > 0:
        violated_cols = [v["column"] for v in normality_violations if isinstance(v, dict)]
        warnings.append(f"WARNING: Shapiro-Wilk test detected non-normal distributions in: {violated_cols}. Consider non-parametric alternatives or data transformations (log, Box-Cox) before using parametric tests.")
        allowed.append("Non-parametric alternatives (Mann-Whitney, Kruskal-Wallis, Spearman)")
        allowed.append("Data Transformation (log, sqrt, Box-Cox) before parametric tests")

    # RULE 7: Heteroscedasticity
    if assumptions.get("homoscedasticity_violated"):
        warnings.append("WARNING: Levene's test detected unequal variances across groups. Standard t-tests and ANOVA assume equal variances. Use Welch's t-test or robust standard errors.")
        allowed.append("Welch's t-test (does not assume equal variances)")
        allowed.append("Robust Standard Errors (HC3)")

    # RULE 8: Multicollinearity
    if assumptions.get("multicollinearity_detected"):
        high_vif = assumptions.get("high_vif_columns", [])
        allowed = [m for m in allowed if "OLS" not in m and "Ordinary Least Squares" not in m]
        forbidden.append("Plain OLS Linear Regression (multicollinearity inflates standard errors)")
        allowed.append("Ridge Regression (L2 regularization)")
        allowed.append("Lasso Regression (L1 regularization)")
        allowed.append("Elastic Net Regression")
        allowed.append("Principal Component Regression (PCR)")
        warnings.append(f"CRITICAL: High multicollinearity detected among predictors: {high_vif}. Plain OLS will produce unreliable coefficient estimates. Use regularized regression (Ridge/Lasso/Elastic Net) or remove collinear predictors.")

    # RULE 9: Small Sample Size (n < 30)
    if assumptions.get("is_small_sample"):
        n = assumptions.get("sample_size", 0)
        forbidden.append("Chi-Square Test (unreliable with small samples)")
        forbidden.append("Large-sample z-tests")
        allowed.append("Fisher's Exact Test")
        allowed.append("Permutation Tests")
        allowed.append("Exact / Non-parametric alternatives")
        warnings.append(f"CRITICAL: Sample size is very small (n={n}). The Central Limit Theorem may not apply. Use exact tests (Fisher's, permutation) instead of asymptotic tests (chi-square, z-test). Interpret all p-values with caution.")

    # RULE 9B: Sparse contingency tables
    if assumptions.get("has_sparse_contingency"):
        sparse_tables = assumptions.get("sparse_contingency_tables", [])
        forbidden.append("Chi-Square Test (sparse contingency tables)")
        allowed.append("Fisher's Exact Test")
        warnings.append(f"CRITICAL: Sparse contingency tables detected in {sparse_tables}. Chi-square assumptions are weak here; prefer Fisher's Exact Test or exact resampling methods.")

    # RULE 10: Ordinal Outcome
    if assumptions.get("has_ordinal_outcome"):
        ordinal_cols = assumptions.get("ordinal_columns", [])
        allowed = [m for m in allowed if "OLS" not in m and "Ordinary Least Squares" not in m]
        forbidden.append("OLS Linear Regression (treats ordinal as continuous)")
        forbidden.append("Standard Logistic Regression (treats ordinal as binary)")
        allowed.append("Ordinal Logistic Regression (Proportional Odds Model)")
        allowed.append("Kruskal-Wallis Test (non-parametric comparison)")
        allowed.append("Spearman Rank Correlation")
        warnings.append(f"CRITICAL: Columns {ordinal_cols} appear to be ordinal (e.g., Likert scale). OLS treats them as continuous (wrong) and standard Logistic treats them as binary (wrong). Use Ordinal Logistic Regression or non-parametric rank tests.")

    # RULE 11: Zero-Inflated Count Data
    if assumptions.get("has_zero_inflation"):
        zi_cols = assumptions.get("zero_inflated_columns", [])
        allowed = [m for m in allowed if "OLS" not in m and "Ordinary Least Squares" not in m]
        forbidden.append("OLS Linear Regression (ignores count/zero-inflation structure)")
        forbidden.append("Standard Poisson Regression (underestimates variance with zero-inflation)")
        allowed.append("Zero-Inflated Poisson (ZIP) Regression")
        allowed.append("Zero-Inflated Negative Binomial (ZINB) Regression")
        allowed.append("Hurdle Model")
        warnings.append(f"CRITICAL: Zero-inflated count data detected in {zi_cols}. Standard Poisson underestimates variance. Use Zero-Inflated Poisson/Negative Binomial or Hurdle models.")

    # RULE 12: Overdispersed counts without strong zero inflation
    if assumptions.get("has_overdispersed_counts"):
        od_cols = assumptions.get("overdispersed_columns", [])
        forbidden.append("Standard Poisson Regression (overdispersion violates equidispersion)")
        allowed.append("Negative Binomial Regression")
        warnings.append(f"CRITICAL: Overdispersed count outcomes detected in {od_cols}. Standard Poisson is too restrictive; prefer Negative Binomial models.")

    # RULE 13: Proportion outcome
    if assumptions.get("has_proportion_outcome"):
        prop_cols = assumptions.get("proportion_columns", [])
        allowed = [m for m in allowed if "OLS" not in m and "Ordinary Least Squares" not in m]
        forbidden.append("OLS Linear Regression (ignores bounded proportion outcome)")
        allowed.append("Beta Regression")
        allowed.append("Fractional Logistic Regression")
        warnings.append(f"CRITICAL: Proportion outcome detected in {prop_cols}. Outcomes bounded in [0, 1] should use beta or fractional-response models instead of OLS.")

    # RULE 14: Time series / serial dependence
    if assumptions.get("has_time_series"):
        ts_cols = assumptions.get("time_series_columns", [])
        allowed = [m for m in allowed if "OLS" not in m and "Ordinary Least Squares" not in m]
        forbidden.append("OLS Linear Regression (ignores serial dependence)")
        allowed.append("ARIMA")
        allowed.append("Autoregressive Regression")
        warnings.append(f"CRITICAL: Serial dependence detected in {ts_cols}. Use time-series models such as ARIMA or autoregressive regression rather than plain OLS.")

    # RULE 15: Perfect separation in binary classification
    if assumptions.get("has_perfect_separation"):
        sep_cols = assumptions.get("perfect_separation_columns", [])
        allowed.append("Firth Penalized Logistic Regression")
        allowed.append("Penalized Logistic Regression")
        warnings.append(f"CRITICAL: Perfect or near-perfect separation detected using predictors {sep_cols}. Standard logistic regression coefficients may diverge; use Firth or penalized logistic models.")

    # RULE 16: Structured missingness
    if assumptions.get("structured_missingness"):
        missing_cols = assumptions.get("structured_missingness_columns", [])
        allowed.append("Multiple Imputation")
        allowed.append("Missingness Diagnostics / Sensitivity Analysis")
        warnings.append(f"WARNING: Missingness appears structured rather than random: {missing_cols}. Consider multiple imputation and sensitivity analysis instead of naive complete-case analysis.")

    # Remove duplicates but keep order stable
    allowed = list(dict.fromkeys(allowed))
    forbidden = list(dict.fromkeys(forbidden))

    constraints = {
        "engine_type": "Deterministic Rule-Based Constraints",
        "data_structure_read": {
            "has_repeated_measures": has_repeated,
            "has_independent_groups": has_indep,
            "has_binary_outcome": has_binary,
            "has_multiclass_nominal_outcome": has_multiclass,
            "has_survival_data": has_survival,
            "is_hierarchical": is_hierarchical
        },
        "assumption_tests_read": {
            "normality_violated": len(normality_violations) > 0,
            "homoscedasticity_violated": assumptions.get("homoscedasticity_violated", False),
            "multicollinearity_detected": assumptions.get("multicollinearity_detected", False),
            "small_sample": assumptions.get("is_small_sample", False),
            "ordinal_outcome": assumptions.get("has_ordinal_outcome", False),
            "zero_inflated": assumptions.get("has_zero_inflation", False),
            "sparse_contingency": assumptions.get("has_sparse_contingency", False),
            "overdispersed_counts": assumptions.get("has_overdispersed_counts", False),
            "proportion_outcome": assumptions.get("has_proportion_outcome", False),
            "time_series": assumptions.get("has_time_series", False),
            "perfect_separation": assumptions.get("has_perfect_separation", False),
            "structured_missingness": assumptions.get("structured_missingness", False)
        },
        "allowed_methods": allowed,
        "forbidden_methods": forbidden,
        "methodological_warnings": warnings
    }

    out_path = os.path.join(output_dir, "constraints.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(constraints, f, indent=4)
        
    print(f"[StatisticalConstraintAgent] constraints.json saved -> {out_path}")
    return out_path
