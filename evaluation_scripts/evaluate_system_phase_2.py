"""
evaluate_system_phase_2.py - Phase 2 Evaluation Suite for AStats
Tests 5 advanced statistical edge cases that basic LLMs commonly miss:
  1. Multicollinearity (correlated predictors)
  2. Non-Normal + Heteroscedastic data
  3. Small Sample Size (n < 30)
  4. Ordinal Outcome (Likert scale)
  5. Zero-Inflated Count Data

Evaluates ONLY the constraint engine output (constraints.json + data_structure.json).
No LLM planning calls needed — this tests the deterministic engine only.

Run with: python evaluate_system_phase_2.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Add project root to path so 'agents' and 'utils' imports resolve
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from agents import data_structure_agent, statistical_constraint_agent
from utils.file_utils import read_markdown

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "eval_phase2")
DATA_DIR = os.path.join(PROJECT_ROOT, "sample_data", "eval_datasets_p2")


def setup_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# Dataset Generators (Phase 2)
# --------------------------------------------------------------------------

def generate_multicollinear_data():
    """Dataset P2-1: Multicollinearity
    Height, Weight, and BMI are all highly correlated.
    System MUST detect multicollinearity and forbid plain OLS."""
    np.random.seed(101)
    n = 200
    height = np.random.normal(170, 10, n)  # cm
    weight = 0.8 * height + np.random.normal(0, 5, n)  # strongly correlated with height
    bmi = weight / ((height / 100) ** 2)  # deterministically derived
    blood_pressure = 50 + 0.3 * weight + np.random.normal(0, 8, n)

    df = pd.DataFrame({
        'PatientID': np.arange(1, n + 1),
        'Height_cm': np.round(height, 1),
        'Weight_kg': np.round(weight, 1),
        'BMI': np.round(bmi, 1),
        'BloodPressure': np.round(blood_pressure, 1)
    })
    path = os.path.join(DATA_DIR, "p2_1_multicollinear.csv")
    df.to_csv(path, index=False)
    return path


def generate_nonnormal_heteroscedastic_data():
    """Dataset P2-2: Non-Normal + Heteroscedastic
    Income is heavily right-skewed. Variance of spending differs by group.
    System MUST detect non-normality and heteroscedasticity."""
    np.random.seed(102)
    n = 150
    group = np.random.choice(['Low', 'Mid', 'High'], n)
    # Right-skewed income (exponential)
    income = np.where(group == 'Low', np.random.exponential(20000, n),
             np.where(group == 'Mid', np.random.exponential(50000, n),
                      np.random.exponential(120000, n)))
    # Heteroscedastic spending: variance depends on group
    spending = np.where(group == 'Low', np.random.normal(500, 100, n),
              np.where(group == 'Mid', np.random.normal(1500, 500, n),
                       np.random.normal(5000, 2000, n)))

    df = pd.DataFrame({
        'CustomerID': np.arange(1, n + 1),
        'IncomeGroup': group,
        'AnnualIncome': np.round(income, 2),
        'MonthlySpending': np.round(spending, 2)
    })
    path = os.path.join(DATA_DIR, "p2_2_nonnormal.csv")
    df.to_csv(path, index=False)
    return path


def generate_small_sample_data():
    """Dataset P2-3: Small Sample Size (n=20)
    System MUST detect small sample and recommend exact tests."""
    np.random.seed(103)
    n = 20
    df = pd.DataFrame({
        'SubjectID': np.arange(1, n + 1),
        'Treatment': np.random.choice(['Drug', 'Placebo'], n),
        'Outcome': np.random.normal(10, 3, n)
    })
    path = os.path.join(DATA_DIR, "p2_3_small_sample.csv")
    df.to_csv(path, index=False)
    return path


def generate_ordinal_outcome_data():
    """Dataset P2-4: Ordinal Outcome (Likert Scale 1-5)
    System MUST NOT use OLS (treats ordinal as continuous) or standard logistic.
    Should recommend Ordinal Logistic Regression."""
    np.random.seed(104)
    n = 200
    df = pd.DataFrame({
        'RespondentID': np.arange(1, n + 1),
        'Age': np.random.randint(18, 65, n),
        'Gender': np.random.choice(['M', 'F'], n),
        'Satisfaction': np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.15, 0.30, 0.35, 0.15])
    })
    path = os.path.join(DATA_DIR, "p2_4_ordinal.csv")
    df.to_csv(path, index=False)
    return path


def generate_zero_inflated_data():
    """Dataset P2-5: Zero-Inflated Count Data
    Number of insurance claims — majority are 0, rest are counts.
    System MUST NOT use OLS or standard Poisson."""
    np.random.seed(105)
    n = 300
    # 60% have zero claims, rest follow Poisson
    is_zero = np.random.choice([0, 1], n, p=[0.4, 0.6])
    claims = np.where(is_zero == 1, 0, np.random.poisson(lam=2, size=n))

    df = pd.DataFrame({
        'PolicyID': np.arange(1, n + 1),
        'Age': np.random.randint(18, 70, n),
        'YearsAsCustomer': np.random.randint(1, 20, n),
        'NumClaims': claims
    })
    path = os.path.join(DATA_DIR, "p2_5_zero_inflated.csv")
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# Evaluation Definitions
# --------------------------------------------------------------------------

PHASE2_EVALUATIONS = [
    {
        "name": "P2-1: Multicollinearity",
        "generator": generate_multicollinear_data,
        "check_field": "multicollinearity_detected",
        "expected_value": True,
        "must_allow": ["Ridge", "Lasso", "Elastic Net"],
        "must_forbid": ["Plain OLS"],
        "description": "Height/Weight/BMI are highly correlated"
    },
    {
        "name": "P2-2: Non-Normal + Heteroscedastic",
        "generator": generate_nonnormal_heteroscedastic_data,
        "check_field": "normality_violated",
        "expected_value": True,
        "must_allow": ["Non-parametric", "Welch"],
        "must_forbid": [],
        "description": "Right-skewed income + unequal variance across groups"
    },
    {
        "name": "P2-3: Small Sample (n=20)",
        "generator": generate_small_sample_data,
        "check_field": "small_sample",
        "expected_value": True,
        "must_allow": ["Fisher", "Permutation", "Exact"],
        "must_forbid": ["Chi-Square"],
        "description": "Only 20 observations — CLT doesn't apply"
    },
    {
        "name": "P2-4: Ordinal Outcome (Likert 1-5)",
        "generator": generate_ordinal_outcome_data,
        "check_field": "ordinal_outcome",
        "expected_value": True,
        "must_allow": ["Ordinal Logistic", "Kruskal-Wallis", "Spearman"],
        "must_forbid": ["OLS Linear Regression"],
        "description": "Satisfaction on 1-5 scale — not continuous, not binary"
    },
    {
        "name": "P2-5: Zero-Inflated Counts",
        "generator": generate_zero_inflated_data,
        "check_field": "zero_inflated",
        "expected_value": True,
        "must_allow": ["Zero-Inflated", "ZIP", "ZINB", "Hurdle"],
        "must_forbid": ["Standard Poisson"],
        "description": "60% zero claims — standard Poisson underestimates variance"
    }
]


def run_constraint_pipeline(csv_path):
    """Runs ONLY the structure + constraint agents (no LLM planning needed)."""
    # 1. Structure (uses local LLM + Pandas heuristics)
    struct_path = data_structure_agent.run(csv_path, OUTPUT_DIR)
    with open(struct_path, "r", encoding="utf-8") as f:
        struct_text = f.read()

    # 2. Constraints (100% deterministic)
    const_path = statistical_constraint_agent.run("", struct_text, OUTPUT_DIR)
    with open(const_path, "r", encoding="utf-8") as f:
        const_text = f.read()

    return json.loads(struct_text), json.loads(const_text)


def execute_phase2():
    setup_directories()
    print("=" * 60)
    print("AStats Phase 2 Evaluation Suite")
    print("Testing: Assumption Violations & Advanced Edge Cases")
    print("=" * 60)

    results = []

    for ev in PHASE2_EVALUATIONS:
        print(f"\n{'─'*60}")
        print(f"Evaluating: {ev['name']}")
        print(f"  Description: {ev['description']}")
        csv_path = ev['generator']()

        try:
            structure, constraints = run_constraint_pipeline(csv_path)

            # Check 1: Did the assumption test detect the issue?
            assumption_data = constraints.get("assumption_tests_read", {})
            detected = assumption_data.get(ev["check_field"], False)
            detection_pass = detected == ev["expected_value"]

            # Check 2: Are the required methods in the allowed list?
            allowed_text = " ".join(constraints.get("allowed_methods", [])).lower()
            forbidden_text = " ".join(constraints.get("forbidden_methods", [])).lower()

            has_required = all(
                req.lower() in allowed_text for req in ev["must_allow"]
            ) if ev["must_allow"] else True

            has_forbidden = all(
                fbd.lower() in forbidden_text for fbd in ev["must_forbid"]
            ) if ev["must_forbid"] else True

            overall = detection_pass and has_required and has_forbidden
            status = "✅ PASS" if overall else "❌ FAIL"

            report = f"  [{status}] {ev['name']}\n"
            report += f"    Detection ({ev['check_field']}): {'✅' if detection_pass else '❌'} (expected={ev['expected_value']}, got={detected})\n"
            report += f"    Required Methods in Allowed: {'✅' if has_required else '❌'} (looked for: {ev['must_allow']})\n"
            report += f"    Forbidden Methods in Forbidden: {'✅' if has_forbidden else '❌'} (looked for: {ev['must_forbid']})\n"

            print(report)
            results.append((ev['name'], status, report))

        except Exception as e:
            import traceback
            print(f"  ❌ [ERROR] Pipeline crashed: {e}")
            traceback.print_exc()
            results.append((ev['name'], "ERROR", str(e)))

    # Final Summary
    print("\n" + "=" * 60)
    print("Phase 2 Final Evaluation Summary")
    print("=" * 60)
    passed = sum(1 for _, s, _ in results if "PASS" in s)
    total = len(results)
    for name, status, _ in results:
        print(f"  {status} — {name}")
    print(f"\nScore: {passed}/{total}")
    print("=" * 60)


if __name__ == "__main__":
    execute_phase2()
