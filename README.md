# AStats — Agentic Statistical Analysis System

A modular **human-in-the-loop** agentic AI system for statistical analysis, built with a **Hybrid Local + Cloud LLM Architecture** and a **Deterministic Statistical Constraint Engine**.

## Architecture Overview

AStats uses a **3-layer hybrid approach** to balance cost, speed, and statistical rigor:

| Layer | Engine | Role |
|-------|--------|------|
| **Data Profiling** | Mistral 7B (Local via LM Studio) | Fast, private data understanding and structure inference |
| **Statistical Constraints** | Python Rules Engine (Zero LLM) | 100% deterministic — enforces valid test selection |
| **Analysis Planning** | Gemini API (Cloud) | High-level reasoning and plan generation |

## Project Structure

```
Astats preliminary work/
├── main.py                                 # Orchestrator — run this
├── generate_complex_data.py                # Synthetic data generator
├── requirements.txt
├── .env                                    # API keys (Gemini + LM Studio)
├── .env.example
│
├── agents/
│   ├── data_understanding_agent.py         # Local LLM → dataset_profile.md
│   ├── data_structure_agent.py             # Pandas heuristics + Local LLM → data_structure.json
│   ├── statistical_constraint_agent.py     # Deterministic rules → constraints.json
│   └── planning_agent.py                   # Cloud LLM → analysis_plan.md
│
├── evaluation_scripts/
│   ├── evaluate_system.py                  # Phase 1: 5 structural edge cases
│   └── evaluate_system_phase_2.py          # Phase 2: 5 assumption-based edge cases
│
├── testing/
│   ├── test_api.py                         # Gemini API key tester
│   ├── test_local_llm.py                   # LM Studio connection tester
│   └── stress_test_local_llm.py            # Context window stress test
│
├── utils/
│   ├── file_utils.py                       # CSV / markdown I/O helpers
│   └── notebook_utils.py                   # nbformat notebook builder
│
├── sample_data/                            # Input datasets
└── outputs/                                # Generated artifacts (runtime)
    ├── dataset_profile.md
    ├── data_structure.json
    ├── constraints.json
    ├── analysis_plan.md
    └── analysis_notebook.ipynb
```

## Methodology

### Phase 1 — Hybrid Inference Pipeline

The core pipeline runs 4 agents sequentially:

```
CSV + Goal
    │
    ▼
┌──────────────────────────────┐
│  1. Data Understanding Agent │  Local LLM (Mistral 7B)
│     Profiles CSV columns,    │  ──► dataset_profile.md
│     stats, missing values    │  ──► analysis_notebook.ipynb
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  2. Data Structure Agent     │  Pandas Heuristics + Local LLM
│     Detects: repeated        │  ──► data_structure.json
│     measures, binary,        │
│     survival, hierarchical   │
│     + assumption tests       │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  3. Constraint Engine        │  100% Deterministic Python
│     11 rules enforce valid   │  ──► constraints.json
│     statistical methods      │
│     Zero LLM. Zero halluc.   │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  4. Planning Agent           │  Cloud LLM (Gemini API)
│     Synthesizes profile +    │  ──► analysis_plan.md
│     structure + constraints  │
│     into actionable plan     │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  Human Review (terminal)     │
│  Feedback → Refine → Approve │
└──────────────────────────────┘
```

### Key Design Decision: Heuristic Override Pattern

Small local models (7B params) hallucinate structural flags. AStats solves this with:

```
LLM generates JSON → Pandas heuristics FORCIBLY CORRECT wrong flags → Constraint engine reads corrected JSON
```

Pandas math is ground truth. The LLM's output is treated as a **suggestion**, not a source of truth.

### Phase 2 — Statistical Assumption Testing

Added deterministic tests that run before the constraint engine:

| Test | Method | Trigger |
|------|--------|---------|
| Normality | Shapiro-Wilk (p < 0.05) | Per numeric column |
| Homoscedasticity | Levene's test (p < 0.05) | Groups × target |
| Multicollinearity | Pearson |r| > 0.85 | All numeric pairs |
| Small Sample | n < 30 | Row count |
| Ordinal Outcome | Consecutive ints, 3-7 categories | Per int column |
| Zero-Inflation | >30% zeros in non-negative int | Per count column |

### Constraint Engine Rules (11 Total)

| # | Rule | Trigger | Effect |
|---|------|---------|--------|
| 1 | Repeated Measures | `has_repeated_measures` | Allow Paired t-test, RM-ANOVA, LMM. Forbid OLS, independent tests. |
| 2 | Independent Groups | `has_independent_groups` | Allow independent t-test, ANOVA, Mann-Whitney. |
| 3 | Binary Outcome | `has_binary_outcome` | **Forbid OLS.** Allow Logistic Regression, Chi-Square. |
| 4 | Survival Data | `has_survival_data` | **Forbid OLS.** Allow Cox PH, Kaplan-Meier, Log-Rank. |
| 5 | Hierarchical | `is_hierarchical` | **Forbid OLS.** Allow Mixed-Effects / HLM. |
| 6 | Non-Normal | Shapiro-Wilk fail | Warn. Suggest non-parametric + transforms. |
| 7 | Heteroscedastic | Levene's fail | Suggest Welch's t-test, robust SEs. |
| 8 | Multicollinearity | High correlation | **Forbid plain OLS.** Require Ridge/Lasso/Elastic Net. |
| 9 | Small Sample | n < 30 | **Forbid Chi-Square.** Require Fisher's Exact, Permutation. |
| 10 | Ordinal Outcome | Likert-type | **Forbid OLS & Logistic.** Require Ordinal Logistic. |
| 11 | Zero-Inflated | >30% zeros | **Forbid OLS & Poisson.** Require ZIP/ZINB/Hurdle. |

## Evaluation Results

### Phase 1 — Structural Edge Cases (5/5 ✅)

| Dataset | Detection | Constraints Correct? |
|---------|-----------|---------------------|
| Paired Pre/Post | `has_repeated_measures: true` | ✅ |
| A/B Test | `has_independent_groups: true` | ✅ |
| Binary Outcome | `has_binary_outcome: true` | ✅ |
| Survival Data | `has_survival_data: true` | ✅ |
| Hierarchical | `is_hierarchical: true` | ✅ |

### Phase 2 — Assumption Edge Cases (5/5 ✅)

| Dataset | Detection Target | Constraints Correct? |
|---------|------------------|----------------------|
| Multicollinearity (Height/Weight/BMI) | `multicollinearity_detected: true` | ✅ |
| Non-Normal + Heteroscedastic | `normality_violated: true`, `homoscedasticity_violated: true` | ✅ |
| Small Sample (n=20) | `small_sample: true` | ✅ |
| Ordinal Outcome (Likert 1-5) | `ordinal_outcome: true` | ✅ |
| Zero-Inflated Counts (60% zeros) | `zero_inflated: true` | ✅ |

## Quick Start

### Prerequisites

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) with **Mistral 7B Instruct v0.1** loaded (port 1234)
- Gemini API key(s) in `.env`

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
python main.py --csv sample_data/iris.csv --goal "perform EDA and visualize the dataset"
```

### Run Evaluation Suites

```bash
# Phase 1: Structural edge cases
python evaluation_scripts/evaluate_system.py

# Phase 2: Assumption-based edge cases
python evaluation_scripts/evaluate_system_phase_2.py
```

## Environment

| Component | Details |
|-----------|---------|
| Local LLM | Mistral 7B Instruct v0.1 (Q4_K_M GGUF) via LM Studio |
| Cloud LLM | Google Gemini API (Flash/Pro) |
| Context Window | ~7300 tokens (tuned for local model) |
| Core Libraries | pandas, numpy, scipy, nbformat, openai |
