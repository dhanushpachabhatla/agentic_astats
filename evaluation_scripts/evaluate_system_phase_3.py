"""
evaluate_system_phase_3.py - Phase 3 Roadmap Stress Suite for AStats

Generates 10 edge-case datasets that probe the current limits of the system.
Unlike Phases 1 and 2, this suite is designed as a roadmap:
it reports whether each scenario is currently SUPPORTED, PARTIAL, or a GAP.

Run with: python evaluate_system_phase_3.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Add project root to path so 'agents' imports resolve
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from agents import data_structure_agent, statistical_constraint_agent

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "eval_phase3")
DATA_DIR = os.path.join(PROJECT_ROOT, "sample_data", "eval_datasets_p3")


def setup_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# Dataset Generators (Phase 3)
# --------------------------------------------------------------------------

def generate_multiclass_nominal_outcome_data():
    """3-class unordered target: should eventually prefer multinomial models."""
    np.random.seed(201)
    n = 240
    age = np.random.randint(18, 70, n)
    income = np.random.normal(65000, 18000, n)
    browsing = np.random.gamma(shape=2.5, scale=8.0, size=n)
    logits = np.vstack([
        0.02 * age - 0.00001 * income + 0.03 * browsing,
        -0.01 * age + 0.000015 * income + 0.02 * browsing,
        np.full(n, 0.0)
    ]).T
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    outcome = [np.random.choice(["Store", "Online", "Partner"], p=p) for p in probs]

    df = pd.DataFrame({
        "CustomerNumber": np.arange(1, n + 1),
        "Age": age,
        "AnnualIncome": np.round(income, 2),
        "BrowsingMinutes": np.round(browsing, 2),
        "PurchaseChannel": outcome
    })
    path = os.path.join(DATA_DIR, "p3_1_multiclass_nominal.csv")
    df.to_csv(path, index=False)
    return path


def generate_binary_predictor_continuous_target_data():
    """Binary predictor only: should not be mistaken for a binary outcome."""
    np.random.seed(202)
    n = 180
    treatment = np.random.choice([0, 1], n)
    baseline = np.random.normal(55, 9, n)
    age = np.random.randint(25, 75, n)
    recovery = 42 + 6 * treatment + 0.6 * baseline - 0.08 * age + np.random.normal(0, 4, n)

    df = pd.DataFrame({
        "SubjectNumber": np.arange(1, n + 1),
        "TreatmentArm": treatment,
        "BaselineScore": np.round(baseline, 2),
        "Age": age,
        "RecoveryScore": np.round(recovery, 2)
    })
    path = os.path.join(DATA_DIR, "p3_2_binary_predictor_continuous_target.csv")
    df.to_csv(path, index=False)
    return path


def generate_string_ordinal_outcome_data():
    """String-coded Likert labels: should eventually map to ordinal methods."""
    np.random.seed(203)
    n = 220
    wait = np.random.gamma(shape=2.0, scale=7.5, size=n)
    age = np.random.randint(18, 80, n)
    labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    probs = [0.08, 0.17, 0.33, 0.27, 0.15]
    satisfaction = np.random.choice(labels, n, p=probs)

    df = pd.DataFrame({
        "RespondentNumber": np.arange(1, n + 1),
        "Age": age,
        "WaitMinutes": np.round(wait, 2),
        "SatisfactionLabel": satisfaction
    })
    path = os.path.join(DATA_DIR, "p3_3_string_ordinal.csv")
    df.to_csv(path, index=False)
    return path


def generate_sparse_subgroup_counts_data():
    """Large n overall, but rare cells where Fisher is safer than chi-square."""
    np.random.seed(204)
    n = 240
    exposure = np.array(["Rare"] * 8 + ["Common"] * (n - 8))
    np.random.shuffle(exposure)
    event = np.where(
        exposure == "Rare",
        np.random.choice([0, 1], n, p=[0.75, 0.25]),
        np.random.choice([0, 1], n, p=[0.96, 0.04])
    )

    df = pd.DataFrame({
        "PatientNumber": np.arange(1, n + 1),
        "ExposureGroup": exposure,
        "Complication": event
    })
    path = os.path.join(DATA_DIR, "p3_4_sparse_subgroup_counts.csv")
    df.to_csv(path, index=False)
    return path


def generate_overdispersed_count_data():
    """Count outcome with overdispersion but not strong zero inflation."""
    np.random.seed(205)
    n = 350
    risk = np.random.normal(0, 1, n)
    age = np.random.randint(18, 85, n)
    mean_count = np.exp(0.6 + 0.35 * risk + 0.01 * (age - 50))
    dispersion = 1.8
    probs = dispersion / (dispersion + mean_count)
    counts = np.random.negative_binomial(dispersion, probs)

    df = pd.DataFrame({
        "MemberNumber": np.arange(1, n + 1),
        "RiskScore": np.round(risk, 3),
        "Age": age,
        "NumVisits": counts
    })
    path = os.path.join(DATA_DIR, "p3_5_overdispersed_counts.csv")
    df.to_csv(path, index=False)
    return path


def generate_proportion_outcome_data():
    """Outcome bounded in [0, 1]: should eventually prefer beta/binomial models."""
    np.random.seed(206)
    n = 260
    spend = np.random.uniform(500, 5000, n)
    quality = np.random.uniform(0.2, 0.95, n)
    linear = -1.0 + 0.00035 * spend + 1.4 * quality
    mean_prop = 1 / (1 + np.exp(-linear))
    alpha = np.clip(mean_prop * 18, 1.1, None)
    beta = np.clip((1 - mean_prop) * 18, 1.1, None)
    conversion = np.random.beta(alpha, beta)

    df = pd.DataFrame({
        "CampaignNumber": np.arange(1, n + 1),
        "CampaignSpend": np.round(spend, 2),
        "LandingPageQuality": np.round(quality, 3),
        "ConversionRate": np.round(conversion, 4)
    })
    path = os.path.join(DATA_DIR, "p3_6_proportion_outcome.csv")
    df.to_csv(path, index=False)
    return path


def generate_hidden_hierarchy_data():
    """Nested data with no obvious *_id keyword to trigger hierarchy detection."""
    np.random.seed(207)
    n_sites = 6
    rows_per_site = 40
    sites = np.repeat(["North", "South", "East", "West", "Central", "Coastal"], rows_per_site)
    classrooms = np.tile(np.repeat(["A", "B", "C", "D"], 10), n_sites)
    teaching = np.random.choice(["Standard", "Inquiry"], n_sites * rows_per_site)
    site_effect = {
        "North": 4.5, "South": -2.0, "East": 3.0,
        "West": -1.0, "Central": 0.5, "Coastal": 2.0
    }
    score = np.array([
        72 + site_effect[site] + (4 if method == "Inquiry" else 0) + np.random.normal(0, 5)
        for site, method in zip(sites, teaching)
    ])

    df = pd.DataFrame({
        "LearnerNumber": np.arange(1, len(score) + 1),
        "Campus": sites,
        "Classroom": classrooms,
        "TeachingMethod": teaching,
        "TestScore": np.round(score, 2)
    })
    path = os.path.join(DATA_DIR, "p3_7_hidden_hierarchy.csv")
    df.to_csv(path, index=False)
    return path


def generate_time_series_data():
    """Autocorrelated time series: should eventually route to time-series methods."""
    np.random.seed(208)
    n = 120
    promo = np.random.normal(100, 12, n)
    sales = np.zeros(n)
    sales[0] = 200 + 0.8 * promo[0] + np.random.normal(0, 4)
    for t in range(1, n):
        sales[t] = 0.75 * sales[t - 1] + 0.45 * promo[t] + np.random.normal(0, 4)

    df = pd.DataFrame({
        "MonthIndex": np.arange(1, n + 1),
        "PromoSpend": np.round(promo, 2),
        "MonthlySales": np.round(sales, 2)
    })
    path = os.path.join(DATA_DIR, "p3_8_time_series.csv")
    df.to_csv(path, index=False)
    return path


def generate_perfect_separation_data():
    """Binary outcome with near-perfect predictor: should warn about separation."""
    np.random.seed(209)
    n = 180
    biomarker = np.random.normal(0, 1, n)
    red_flag = (biomarker > 0.4).astype(int)
    outcome = red_flag.copy()

    df = pd.DataFrame({
        "CaseNumber": np.arange(1, n + 1),
        "BiomarkerZ": np.round(biomarker, 3),
        "RedFlag": red_flag,
        "SevereEvent": outcome
    })
    path = os.path.join(DATA_DIR, "p3_9_perfect_separation.csv")
    df.to_csv(path, index=False)
    return path


def generate_missingness_pattern_data():
    """Missingness concentrated in one group: should eventually trigger missing-data strategy."""
    np.random.seed(210)
    n = 240
    group = np.random.choice(["Control", "HighRisk"], n, p=[0.7, 0.3])
    income = np.random.normal(55000, 14000, n)
    outcome = 30 + 0.0005 * income + np.where(group == "HighRisk", -5, 0) + np.random.normal(0, 3, n)
    missing_mask = (group == "HighRisk") & (np.random.rand(n) < 0.55)
    income[missing_mask] = np.nan

    df = pd.DataFrame({
        "ParticipantNumber": np.arange(1, n + 1),
        "RiskBand": group,
        "Income": np.round(income, 2),
        "OutcomeScore": np.round(outcome, 2)
    })
    path = os.path.join(DATA_DIR, "p3_10_missingness_pattern.csv")
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# Evaluation Definitions
# --------------------------------------------------------------------------

PHASE3_EVALUATIONS = [
    {
        "name": "P3-1: Multiclass Nominal Outcome",
        "generator": generate_multiclass_nominal_outcome_data,
        "future_need": "Support multinomial classification and forbid treating unordered multiclass targets as OLS regression problems.",
        "must_allow": ["Multinomial Logistic"],
        "must_forbid": ["OLS"],
        "must_detect_true": [],
        "must_detect_false": ["ordinal_outcome", "has_binary_outcome"],
    },
    {
        "name": "P3-2: Binary Predictor with Continuous Outcome",
        "generator": generate_binary_predictor_continuous_target_data,
        "future_need": "Do not confuse a binary predictor with a binary target.",
        "must_allow": ["OLS"],
        "must_forbid": [],
        "must_detect_true": [],
        "must_detect_false": ["has_binary_outcome"],
    },
    {
        "name": "P3-3: String-Coded Ordinal Outcome",
        "generator": generate_string_ordinal_outcome_data,
        "future_need": "Map ordered string labels to ordinal modeling choices.",
        "must_allow": ["Ordinal Logistic"],
        "must_forbid": ["OLS"],
        "must_detect_true": ["ordinal_outcome"],
        "must_detect_false": [],
    },
    {
        "name": "P3-4: Sparse Subgroup Counts",
        "generator": generate_sparse_subgroup_counts_data,
        "future_need": "Look beyond global n and catch sparse contingency tables where Fisher's Exact is safer than chi-square.",
        "must_allow": ["Fisher"],
        "must_forbid": ["Chi-Square"],
        "must_detect_true": [],
        "must_detect_false": [],
    },
    {
        "name": "P3-5: Overdispersed Counts without Zero Inflation",
        "generator": generate_overdispersed_count_data,
        "future_need": "Differentiate overdispersion from zero inflation and surface Negative Binomial models.",
        "must_allow": ["Negative Binomial"],
        "must_forbid": ["Poisson"],
        "must_detect_true": [],
        "must_detect_false": ["zero_inflated"],
    },
    {
        "name": "P3-6: Proportion Outcome",
        "generator": generate_proportion_outcome_data,
        "future_need": "Route bounded [0,1] outcomes to beta/binomial-style methods instead of plain linear regression.",
        "must_allow": ["Beta Regression"],
        "must_forbid": ["OLS"],
        "must_detect_true": [],
        "must_detect_false": [],
    },
    {
        "name": "P3-7: Hidden Hierarchy without ID Keywords",
        "generator": generate_hidden_hierarchy_data,
        "future_need": "Detect clustered or nested data even when columns are named Campus/Classroom instead of *_id.",
        "must_allow": ["Mixed-Effects"],
        "must_forbid": ["OLS"],
        "must_detect_true": ["is_hierarchical"],
        "must_detect_false": [],
    },
    {
        "name": "P3-8: Autocorrelated Time Series",
        "generator": generate_time_series_data,
        "future_need": "Detect serial dependence and route to time-series models such as ARIMA or autoregressive regression.",
        "must_allow": ["ARIMA"],
        "must_forbid": ["OLS"],
        "must_detect_true": [],
        "must_detect_false": [],
    },
    {
        "name": "P3-9: Perfect Separation in Binary Outcome",
        "generator": generate_perfect_separation_data,
        "future_need": "Warn when logistic regression is unstable because a predictor almost perfectly separates the classes.",
        "must_allow": ["Firth", "Penalized Logistic"],
        "must_forbid": [],
        "must_detect_true": ["has_binary_outcome"],
        "must_detect_false": [],
    },
    {
        "name": "P3-10: Group-Dependent Missingness",
        "generator": generate_missingness_pattern_data,
        "future_need": "Recommend missing-data diagnostics or imputation when missingness is structured rather than random.",
        "must_allow": ["Multiple Imputation"],
        "must_forbid": [],
        "must_detect_true": [],
        "must_detect_false": [],
    },
]


def run_constraint_pipeline(csv_path):
    """Runs the structure + constraint agents and returns parsed JSON."""
    struct_path = data_structure_agent.run(csv_path, OUTPUT_DIR)
    with open(struct_path, "r", encoding="utf-8") as f:
        struct_text = f.read()

    const_path = statistical_constraint_agent.run("", struct_text, OUTPUT_DIR)
    with open(const_path, "r", encoding="utf-8") as f:
        const_text = f.read()

    return json.loads(struct_text), json.loads(const_text)


def build_signal_map(constraints):
    signal_map = {}
    signal_map.update(constraints.get("data_structure_read", {}))
    signal_map.update(constraints.get("assumption_tests_read", {}))
    return signal_map


def evaluate_case(case, constraints):
    signal_map = build_signal_map(constraints)
    allowed_text = " ".join(constraints.get("allowed_methods", [])).lower()
    forbidden_text = " ".join(constraints.get("forbidden_methods", [])).lower()

    checks = []

    for flag in case.get("must_detect_true", []):
        checks.append({
            "label": f"flag `{flag}` should be true",
            "passed": bool(signal_map.get(flag, False)),
        })

    for flag in case.get("must_detect_false", []):
        checks.append({
            "label": f"flag `{flag}` should stay false",
            "passed": not bool(signal_map.get(flag, False)),
        })

    for method in case.get("must_allow", []):
        checks.append({
            "label": f"allowed methods should include `{method}`",
            "passed": method.lower() in allowed_text,
        })

    for method in case.get("must_forbid", []):
        checks.append({
            "label": f"forbidden methods should include `{method}`",
            "passed": method.lower() in forbidden_text,
        })

    passed = sum(1 for c in checks if c["passed"])
    total = len(checks)

    if passed == total:
        status = "SUPPORTED"
    elif passed == 0:
        status = "GAP"
    else:
        status = "PARTIAL"

    return status, checks, signal_map


def write_summary_json(summary_rows):
    out_path = os.path.join(OUTPUT_DIR, "phase3_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=4)
    return out_path


def execute_phase3():
    setup_directories()
    print("=" * 60)
    print("AStats Phase 3 Roadmap Stress Suite")
    print("Testing: unsupported or brittle statistical edge cases")
    print("=" * 60)

    summary_rows = []

    for case in PHASE3_EVALUATIONS:
        print(f"\n{'-' * 60}")
        print(f"Evaluating: {case['name']}")
        print(f"  Future Need: {case['future_need']}")
        csv_path = case["generator"]()

        try:
            structure, constraints = run_constraint_pipeline(csv_path)
            status, checks, signal_map = evaluate_case(case, constraints)

            print(f"  Status: {status}")
            for check in checks:
                marker = "PASS" if check["passed"] else "FAIL"
                print(f"    {marker}: {check['label']}")

            summary_rows.append({
                "name": case["name"],
                "status": status,
                "future_need": case["future_need"],
                "checks": checks,
                "signals": signal_map,
                "allowed_methods": constraints.get("allowed_methods", []),
                "forbidden_methods": constraints.get("forbidden_methods", []),
                "dataset_path": csv_path,
            })

        except Exception as e:
            print(f"  [ERROR] Pipeline crashed: {e}")
            summary_rows.append({
                "name": case["name"],
                "status": "ERROR",
                "future_need": case["future_need"],
                "error": str(e),
                "dataset_path": csv_path,
            })

    summary_path = write_summary_json(summary_rows)

    print("\n" + "=" * 60)
    print("Phase 3 Summary")
    print("=" * 60)
    for status in ["SUPPORTED", "PARTIAL", "GAP", "ERROR"]:
        count = sum(1 for row in summary_rows if row["status"] == status)
        print(f"  {status}: {count}")
    print(f"\nSummary JSON: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    execute_phase3()
