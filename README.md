# AStats — Agentic Statistical Analysis System

A minimal, modular **human-in-the-loop** agentic AI system for statistical analysis, built with Python and the **Gemini API** (no LangChain / LangGraph).

## Project Structure

```
Astats preliminary work/
├── main.py                          # Orchestrator — run this
├── requirements.txt
├── .env                             # Your Gemini API keys (already set up)
├── .env.example                     # Template
├── sample_data/
│   └── iris.csv                     # Bundled test dataset
├── agents/
│   ├── data_understanding_agent.py  # Profiles CSV → dataset_profile.md + notebook
│   └── planning_agent.py            # Generates & refines analysis_plan.md
├── utils/
│   ├── file_utils.py                # CSV / markdown I/O helpers
│   └── notebook_utils.py            # nbformat notebook builder
└── outputs/                         # Created at runtime
    ├── dataset_profile.md
    ├── analysis_plan.md
    └── analysis_notebook.ipynb
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the system

```bash
python main.py
```

Or with CLI arguments (skips the prompts):

```bash
python main.py --csv sample_data/iris.csv --goal "perform EDA and visualize the dataset"
```

### 3. Follow the workflow

```
[Agent 1] Data Understanding Agent
  → Reads your CSV
  → Calls Gemini to write outputs/dataset_profile.md
  → Builds outputs/analysis_notebook.ipynb

[Agent 2] Planning Agent
  → Reads dataset_profile.md + your goal
  → Calls Gemini to write outputs/analysis_plan.md

[Human-in-the-Loop]
  → Review the plan printed to terminal
  → Type feedback to refine it (e.g. "Add a PCA step after visualization")
  → Repeat until satisfied, then type "approve"
```

## Outputs

| File | Description |
|------|-------------|
| `outputs/dataset_profile.md` | Column types, stats, missing values, EDA insights (Gemini-generated) |
| `outputs/analysis_plan.md` | Numbered analysis plan tailored to your goal |
| `outputs/analysis_notebook.ipynb` | Jupyter notebook with EDA starter cells |

## API Keys

Three Gemini API keys (`GEMINI_KEY1`, `GEMINI_KEY2`, `GEMINI_KEY3`) are loaded from `.env`.  
The system automatically rotates keys if one fails.

## Workflow Diagram

```
CSV + Goal
    │
    ▼
┌─────────────────────────┐
│  Data Understanding     │  ──► dataset_profile.md
│  Agent (Gemini)         │  ──► analysis_notebook.ipynb
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Planning Agent         │  ──► analysis_plan.md
│  (Gemini)               │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Human Review           │
│  (terminal feedback)    │
└─────────────────────────┘
    │ feedback
    ▼
┌─────────────────────────┐
│  Plan Refinement        │  ──► analysis_plan.md (updated)
│  (Gemini)               │
└─────────────────────────┘
    │ approved
    ▼
  Done ✅
```
