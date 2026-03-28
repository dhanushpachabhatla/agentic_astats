"""
planning_agent.py - LLM-based agent that generates and refines analysis_plan.md.
Uses Gemini API with key rotation across GEMINI_API_KEY20 / GEMINI_KEY1/2/3.
"""
import os
from google import genai
from openai import OpenAI

from utils.file_utils import write_markdown


# Gemini key rotation - tries each key in order until one succeeds
_GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY20"),
    os.getenv("GEMINI_KEY1"),
    os.getenv("GEMINI_KEY2"),
    os.getenv("GEMINI_KEY3"),
]

_MODEL = "gemini-2.5-flash"
_PLANNING_BACKEND = os.getenv("PLANNING_BACKEND", "gemini").strip().lower()

# Local LM Studio client - mirrors the other local agents
_LOCAL_CLIENT = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
_LOCAL_MODEL = "local-model"


def _call_local_llm(prompt: str) -> str:
    """Send prompt to local LM Studio server."""
    try:
        response = _LOCAL_CLIENT.chat.completions.create(
            model=_LOCAL_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert statistical analysis planner. Follow the provided statistical constraints strictly and produce clear markdown."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2500
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Failed to call local LM Studio API for planning: {e}. Is a model loaded on port 1234?")


def _call_gemini(prompt: str) -> str:
    """Try each Gemini API key in round-robin until a call succeeds."""
    last_error = None
    for key in _GEMINI_KEYS:
        if not key:
            continue
        try:
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model=_MODEL,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            last_error = e
            print(f"[planning_agent] Key failed, trying next... ({e})")
    raise RuntimeError(f"All Gemini API keys failed. Last error: {last_error}")


def _call_planner(prompt: str) -> str:
    """
    Route planning calls through the configured backend.

    Supported values:
      - gemini: use Gemini only
      - local: use LM Studio only
      - auto: try LM Studio first, then fall back to Gemini
    """
    if _PLANNING_BACKEND == "local":
        return _call_local_llm(prompt)
    if _PLANNING_BACKEND == "auto":
        try:
            return _call_local_llm(prompt)
        except Exception as e:
            print(f"[planning_agent] Local planning failed, falling back to Gemini... ({e})")
            return _call_gemini(prompt)
    return _call_gemini(prompt)


def _generate_plan_prompt(profile_text: str, data_structure_text: str, constraint_text: str, user_goal: str) -> str:
    return f"""You are an expert data scientist and analysis planner.

User Goal: {user_goal}

--- DATASET PROFILE ---
{profile_text}
--- END PROFILE ---

--- DATA STRUCTURE EXPECTATIONS ---
{data_structure_text}
--- END DATA STRUCTURE ---

--- STATISTICAL CONSTRAINTS ---
{constraint_text}
--- END CONSTRAINTS ---

Based on the user's goal, the dataset profile, the inferred data structure, and the STRICT statistical constraints above, create a structured analysis plan.

Output this as a markdown file called analysis_plan.md. Structure it as follows:

# Analysis Plan

## User Goal
State the user's goal in one sentence.

## Dataset Summary
A 2-3 sentence recap of what the dataset contains.

## Analysis Steps

For each step provide:
- **Step N: [Step Name]**
  - **Objective:** What this step aims to achieve
  - **Method:** Specific technique or approach (e.g., seaborn pairplot, OLS regression)
  - **Expected Output:** What artifact or insight this produces (e.g., correlation heatmap, regression summary table)
  - **Priority:** High / Medium / Low

Include steps for (as relevant to the user goal and dataset):
1. Data Cleaning / Preprocessing
2. Exploratory Data Analysis (EDA)
3. Visualization
4. Feature Engineering (if applicable)
5. Statistical Testing or Modeling (if applicable)
6. Interpretation & Summary

## Notes
Any caveats, data quality issues to watch for, or assumptions made.

Be specific, actionable, and concise. Tailor the plan directly to the user goal and dataset.
"""


def _refine_plan_prompt(current_plan: str, feedback: str) -> str:
    return f"""You are an expert data scientist reviewing an analysis plan.

--- CURRENT ANALYSIS PLAN ---
{current_plan}
--- END PLAN ---

--- USER FEEDBACK ---
{feedback}
--- END FEEDBACK ---

Update the analysis plan based on the user's feedback.

Rules:
- Keep all steps the user did NOT mention changing
- Add, remove, or modify steps exactly as the user requested
- Maintain the same markdown structure as the original plan
- At the top of the plan, add a brief "## Revision Notes" section explaining what changed and why
- Be concise and specific

Output the complete updated analysis_plan.md.
"""


def generate_plan(profile_text: str, data_structure_text: str, constraints_text: str, user_goal: str, output_dir: str) -> str:
    """
    Generate an initial analysis plan from the dataset profile, structure, constraints, and user goal.

    Returns
    -------
    str : path to the written analysis_plan.md
    """
    print(f"\n[PlanningAgent] Generating analysis plan with backend: {_PLANNING_BACKEND}...")
    prompt = _generate_plan_prompt(profile_text, data_structure_text, constraints_text, user_goal)
    plan_md = _call_planner(prompt)

    plan_path = os.path.join(output_dir, "analysis_plan.md")
    write_markdown(plan_path, plan_md)
    print(f"[PlanningAgent] analysis_plan.md saved -> {plan_path}")
    return plan_path


def refine_plan(current_plan_text: str, feedback: str, output_dir: str) -> str:
    """
    Refine the existing analysis plan based on human feedback.

    Returns
    -------
    str : path to the updated analysis_plan.md
    """
    print(f"\n[PlanningAgent] Refining plan with backend: {_PLANNING_BACKEND}...")
    prompt = _refine_plan_prompt(current_plan_text, feedback)
    updated_plan_md = _call_planner(prompt)

    plan_path = os.path.join(output_dir, "analysis_plan.md")
    write_markdown(plan_path, updated_plan_md)
    print(f"[PlanningAgent] analysis_plan.md updated -> {plan_path}")
    return plan_path
