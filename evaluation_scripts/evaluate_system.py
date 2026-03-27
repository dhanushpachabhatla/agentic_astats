"""
evaluate_system.py - Automated Evaluation Suite for AStats
Generates 5 complex statistical edge-case datasets, runs them through the 
AStats pipeline, and checks if the system's generated constraints and plans 
match the Ground Truth expert expectations.

Run with: python evaluate_system.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Add project root to path so 'agents' and 'utils' imports resolve
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from agents import (
    data_understanding_agent,
    data_structure_agent,
    statistical_constraint_agent,
    planning_agent
)
from utils.file_utils import read_markdown

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "eval_runs")
DATA_DIR = os.path.join(PROJECT_ROOT, "sample_data", "eval_datasets")

def setup_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset Generation
# --------------------------------------------------------------------------

def generate_paired_data():
    """Dataset 1: Pre/Post test (Repeated Measures, 2 timepoints)"""
    np.random.seed(1)
    df = pd.DataFrame({
        'PatientID': np.repeat(np.arange(1, 31), 2),
        'Timepoint': np.tile(['Pre', 'Post'], 30),
        'Score': np.random.normal(50, 10, 60)
    })
    path = os.path.join(DATA_DIR, "eval1_paired.csv")
    df.to_csv(path, index=False)
    return path

def generate_ab_test_data():
    """Dataset 2: Classical A/B Test (Independent Groups, Cross-sectional)"""
    np.random.seed(2)
    df = pd.DataFrame({
        'UserID': np.arange(1, 101),
        'WebsiteVersion': np.random.choice(['A', 'B'], 100),
        'TimeOnPage': np.random.exponential(scale=30, size=100)
    })
    path = os.path.join(DATA_DIR, "eval2_ab_test.csv")
    df.to_csv(path, index=False)
    return path

def generate_binary_outcome_data():
    """Dataset 3: Binary Classification / Logistic Target
    Edge Case: System MUST NOT recommend OLS linear regression for a 0/1 outcome."""
    np.random.seed(3)
    df = pd.DataFrame({
        'CustomerID': np.arange(1, 202),  # 201 elements to match other columns
        'Age': np.random.randint(18, 70, 201),
        'Income': np.random.normal(50000, 15000, 201),
        'Subscribed': np.random.choice([0, 1], 201, p=[0.8, 0.2])
    })
    path = os.path.join(DATA_DIR, "eval3_binary.csv")
    df.to_csv(path, index=False)
    return path

def generate_survival_data():
    """Dataset 4: Time-to-Event / Survival Data
    Edge Case: Right-censored data require Cox PH or Kaplan-Meier, NOT standard regression."""
    np.random.seed(4)
    df = pd.DataFrame({
        'MachineID': np.arange(1, 152),  # 151 elements to match other columns
        'OperatingTemp': np.random.normal(70, 5, 151),
        'DaysToFailure': np.random.randint(10, 365, 151),
        'Failed': np.random.choice([0, 1], 151, p=[0.3, 0.7]) # 0 = censored
    })
    path = os.path.join(DATA_DIR, "eval4_survival.csv")
    df.to_csv(path, index=False)
    return path

def generate_hierarchical_data():
    """Dataset 5: Nested / Hierarchical Data
    Edge Case: Students within schools means observations are NOT independent. 
    Requires Mixed-Effects (HLM)."""
    np.random.seed(5)
    school_ids = np.repeat(np.arange(1, 11), 20) # 10 schools, 20 students each
    df = pd.DataFrame({
        'StudentID': np.arange(1, 201),
        'SchoolID': school_ids,
        'TeachingMethod': np.where(school_ids % 2 == 0, 'New', 'Standard'),
        'TestScore': np.random.normal(75, 10, 200)
    })
    path = os.path.join(DATA_DIR, "eval5_hierarchical.csv")
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# Evaluation Suite
# --------------------------------------------------------------------------

EVALUATIONS = [
    {
        "name": "Dataset 1: Paired Pre/Post",
        "generator": generate_paired_data,
        "goal": "Did the intervention improve scores from Pre to Post?",
        "must_have": ["Paired", "t-test", "Wilcoxon"],
        "must_NOT_have": ["Independent t-test", "Ordinary Least Squares"]
    },
    {
        "name": "Dataset 2: Independent A/B Test",
        "generator": generate_ab_test_data,
        "goal": "Which website version keeps users on the page longer?",
        "must_have": ["Independent", "Mann-Whitney", "t-test"],
        "must_NOT_have": ["Paired", "Repeated Measures"]
    },
    {
        "name": "Dataset 3: Binary Outcome",
        "generator": generate_binary_outcome_data,
        "goal": "Identify factors predicting if a customer subscribed.",
        "must_have": ["Logistic Regression", "Classification"],
        "must_NOT_have": ["Ordinary Least Squares", "Linear Regression"]
    },
    {
        "name": "Dataset 4: Survival / Censored Data",
        "generator": generate_survival_data,
        "goal": "What factors predict machine failure over time?",
        "must_have": ["Cox", "Kaplan-Meier", "Survival"],
        "must_NOT_have": ["Ordinary Least Squares", "Logistic Regression"]
    },
    {
        "name": "Dataset 5: Hierarchical Data",
        "generator": generate_hierarchical_data,
        "goal": "Does teaching method affect test scores across schools?",
        "must_have": ["Mixed-Effects", "Hierarchical", "HLM"],
        "must_NOT_have": ["Ordinary Least Squares", "Independent t-test"]
    }
]

def run_pipeline_headless(csv_path, goal):
    """Runs the 4 agents silently and returns the text of constraints.json and analysis_plan.md"""
    # 1. Understanding
    profile_path = data_understanding_agent.run(csv_path, OUTPUT_DIR)
    profile_text = read_markdown(profile_path)
    
    # 2. Structure
    struct_path = data_structure_agent.run(csv_path, OUTPUT_DIR)
    with open(struct_path, "r", encoding="utf-8") as f:
        struct_text = f.read()
        
    # 3. Constraints
    const_path = statistical_constraint_agent.run(profile_text, struct_text, OUTPUT_DIR)
    with open(const_path, "r", encoding="utf-8") as f:
        const_text = f.read()
        
    # 4. Planning
    plan_path = planning_agent.generate_plan(profile_text, struct_text, const_text, goal, OUTPUT_DIR)
    plan_text = read_markdown(plan_path)
    
    return const_text, plan_text

def execute_evaluations():
    setup_directories()
    print("="*60)
    print("AStats Evaluation Suite")
    print("="*60)
    
    results = []
    
    for eval_case in EVALUATIONS:
        print(f"\nEvaluating: {eval_case['name']}")
        csv_path = eval_case['generator']()
        
        try:
            const_text, plan_text = run_pipeline_headless(csv_path, eval_case['goal'])
            
            combined_text = (const_text + plan_text).lower()
            
            # Check requirements
            has_required = any(req.lower() in combined_text for req in eval_case['must_have'])
            has_forbidden = any(forbid.lower() in combined_text for forbid in eval_case['must_NOT_have'])
            
            if has_required and not has_forbidden:
                status = "PASS"
            else:
                status = "FAIL"
                
            report = f"[{status}] {eval_case['name']}\n"
            report += f"  - Goal: {eval_case['goal']}\n"
            report += f"  - Missing Required? {not has_required} (Looked for: {eval_case['must_have']})\n"
            report += f"  - Included Forbidden? {has_forbidden} (Looked for: {eval_case['must_NOT_have']})\n"
            
            print(report)
            results.append(report)
            
        except Exception as e:
            print(f"[ERROR] Pipeline failed on {eval_case['name']}: {e}")
            
    print("\n" + "="*60)
    print("Final Evaluation Summary")
    print("="*60)
    for res in results:
        print(res.split('\n')[0]) # Just print the PASS/FAIL line
    print("="*60)

if __name__ == "__main__":
    execute_evaluations()
