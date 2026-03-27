"""
data_structure_agent.py - Analyzes dataset structure using Pandas heuristics
and semantic LLM inference (hybrid approach) to infer independent/dependent groups,
repeated measures, etc. Produces data_structure.json.
"""
import os
import json
import pandas as pd
from openai import OpenAI

from utils.file_utils import read_csv, ensure_output_dir


# Connect to LM Studio Local Server
_LOCAL_CLIENT = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
_MODEL = "local-model"

def _call_local_llm(prompt: str) -> str:
    """Send prompt to local LM Studio server."""
    try:
        response = _LOCAL_CLIENT.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict data structure analyzer. ONLY output valid JSON. Do not include markdown formatting or conversational text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # Low temp for structured JSON extraction
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Failed to call local LM Studio API: {e}. Is the server running on port 1234?")


def _compute_heuristics(df: pd.DataFrame) -> dict:
    """Compute strict Pandas heuristics to guide (and ground) the LLM."""
    n_rows = len(df)
    heuristics = {
        "candidate_id_columns": [],
        "shows_repeated_measures": False,
        "candidate_grouping_columns": [],
        "numeric_targets": [],
        "binary_targets": [],
        "potential_survival_data": False,
        "potential_hierarchical_data": False
    }
    
    for col in df.columns:
        n_unique = df[col].nunique()
        max_freq = df[col].value_counts().max() if n_unique > 0 else 0
        dtype = str(df[col].dtype)
        
        col_lower = str(col).lower()

        # Heuristic: Potential ID column if it has many unique values but isn't a float
        if n_unique > (n_rows * 0.1) and "float" not in dtype:
            heuristics["candidate_id_columns"].append(col)
            # If an ID column has rows appearing multiple times, strongly implies repeated measures
            if max_freq > 1 and max_freq < n_rows:
                heuristics["shows_repeated_measures"] = True
                
        # Heuristic: Grouping columns have low cardinality
        if n_unique > 1 and n_unique <= 10 and "float" not in dtype:
            heuristics["candidate_grouping_columns"].append(col)
            
        # Heuristic: Binary target detection
        if n_unique == 2:
            heuristics["binary_targets"].append(col)
            
        # Heuristic: Numeric targets
        if "float" in dtype or "int" in dtype:
            if n_unique > 10:  # Not a binary or categorical int
                heuristics["numeric_targets"].append(col)
        
        # Heuristic: Hierarchical / Nested Data detection
        # Check ANY column (not just high-cardinality) for group-level ID patterns
        if "id" in col_lower and any(w in col_lower for w in ["school", "hospital", "clinic", "group", "class", "region", "country", "site", "cluster", "center"]):
            heuristics["potential_hierarchical_data"] = True
        
        # Heuristic: Survival Data detection (column name patterns)
        if any(w in col_lower for w in ["surv", "censor", "status", "event", "time_to", "failed", "failure", "death", "duration"]):
            heuristics["potential_survival_data"] = True
                
    return heuristics


def _compute_assumption_tests(df: pd.DataFrame, heuristics: dict) -> dict:
    """
    Phase 2: Run deterministic statistical assumption tests on the data.
    These tests are computed via scipy/pandas math — zero LLM involvement.
    Results are appended to the heuristics dict for the constraint engine.
    """
    from scipy import stats
    
    assumptions = {
        "normality_violations": [],     # Columns that fail Shapiro-Wilk
        "homoscedasticity_violated": False,
        "multicollinearity_detected": False,
        "high_vif_columns": [],
        "is_small_sample": False,       # n < 30
        "has_ordinal_outcome": False,
        "ordinal_columns": [],
        "has_zero_inflation": False,
        "zero_inflated_columns": [],
        "sample_size": len(df)
    }
    
    n_rows = len(df)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    # ----- 1. Small Sample Detection -----
    if n_rows < 30:
        assumptions["is_small_sample"] = True
    
    # ----- 2. Normality Test (Shapiro-Wilk) on numeric columns -----
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) >= 8 and len(col_data) <= 5000:  # Shapiro-Wilk bounds
            try:
                stat, p_value = stats.shapiro(col_data)
                if p_value < 0.05:
                    assumptions["normality_violations"].append({
                        "column": col,
                        "shapiro_p": round(float(p_value), 6),
                        "verdict": "NON-NORMAL"
                    })
            except Exception:
                pass
    
    # ----- 3. Homoscedasticity (Levene's Test) -----
    # Only meaningful if there's a grouping column and a numeric target
    grouping_cols = heuristics.get("candidate_grouping_columns", [])
    target_cols = heuristics.get("numeric_targets", [])
    
    if grouping_cols and target_cols:
        group_col = grouping_cols[0]
        target_col = target_cols[0]
        groups = [group_data[target_col].dropna().values 
                  for _, group_data in df.groupby(group_col) 
                  if len(group_data[target_col].dropna()) >= 2]
        if len(groups) >= 2:
            try:
                stat, p_value = stats.levene(*groups)
                if p_value < 0.05:
                    assumptions["homoscedasticity_violated"] = True
            except Exception:
                pass
    
    # ----- 4. Multicollinearity (VIF) -----
    if len(numeric_cols) >= 2:
        try:
            from numpy.linalg import LinAlgError
            numeric_df = df[numeric_cols].dropna()
            if len(numeric_df) > len(numeric_cols):
                # Add constant for VIF calculation
                X = numeric_df.values
                from numpy.linalg import inv
                corr_matrix = np.corrcoef(X, rowvar=False)
                # Check if any pair has |r| > 0.85
                import numpy as np
                for i in range(len(numeric_cols)):
                    for j in range(i + 1, len(numeric_cols)):
                        if abs(corr_matrix[i, j]) > 0.85:
                            assumptions["multicollinearity_detected"] = True
                            assumptions["high_vif_columns"].append(
                                f"{numeric_cols[i]} <-> {numeric_cols[j]} (r={corr_matrix[i,j]:.2f})"
                            )
        except Exception:
            pass
    
    # ----- 5. Ordinal Outcome Detection -----
    for col in df.columns:
        col_data = df[col].dropna()
        n_unique = col_data.nunique()
        dtype = str(df[col].dtype)
        # Ordinal: integer column with small range (3-7 categories, e.g., Likert 1-5)
        if "int" in dtype and 3 <= n_unique <= 7:
            # Check if values are consecutive integers
            unique_vals = sorted(col_data.unique())
            if unique_vals == list(range(min(unique_vals), max(unique_vals) + 1)):
                assumptions["has_ordinal_outcome"] = True
                assumptions["ordinal_columns"].append(col)
    
    # ----- 6. Zero-Inflated Count Data Detection -----
    for col in numeric_cols:
        col_data = df[col].dropna()
        if "int" in str(df[col].dtype) and (col_data >= 0).all():
            zero_pct = (col_data == 0).sum() / len(col_data)
            if zero_pct > 0.3:  # More than 30% zeros = potential zero-inflation
                assumptions["has_zero_inflation"] = True
                assumptions["zero_inflated_columns"].append({
                    "column": col,
                    "zero_percentage": round(float(zero_pct * 100), 1)
                })
    
    return assumptions


def _build_structure_prompt(dataset_name: str, head_str: str, dtypes_str: str, heuristics: dict) -> str:
    return f"""You are an expert data statistician.
Analyze this dataset to infer its underlying data-generating process and statistical structure.
We have pre-computed some strict Pandas heuristics to guide you. Combine your semantic understanding of the column names with these hard mathematical facts.

Dataset: {dataset_name}

--- PANDAS HEURISTICS (GROUND TRUTH) ---
{json.dumps(heuristics, indent=2)}
----------------------------------------

COLUMN TYPES:
{dtypes_str}

SAMPLE DATA (first 5 rows):
{head_str}

    Infer whether the data contains:
    - Repeated measures (e.g., 'time', 'visit', 'pre/post' columns)
    - Paired samples
    - Independent vs dependent groups
    - Hierarchical or grouped data patterns (e.g., 'subject_id' nested in 'school_id')
    - Binary Outcomes requiring Logistic Regression instead of OLS
    - Survival Data with right-censoring
    - What each variable's role likely is.
    
    Output your findings STRICTLY as a valid JSON object matching this exact structure (start with {{):
    
    {{
      "dataset_summary": "Brief summary",
      "has_repeated_measures": true/false,
      "has_independent_groups": true/false,
      "has_binary_outcome": true/false (Set to true if there is a primary boolean/binary 1/0 target variable),
      "has_survival_data": true/false,
      "is_hierarchical": true/false,
      "identified_roles": {{
        "column_name_1": "ID",
        "column_name_2": "Grouping Variable",
        "column_name_3": "Repeated Measure Timepoint",
        "column_name_4": "Continuous Target"
      }},
      "inferred_structure_notes": "Explain why it has repeated measures, independent groups, or other advanced structures."
    }}
    """


def run(csv_path: str, output_dir: str) -> str:
    ensure_output_dir(output_dir)
    print("\n[DataStructureAgent] Inferring data structure using Pandas + Local LM Studio...")
    df = read_csv(csv_path)

    heuristics = _compute_heuristics(df)
    head_str = df.head().to_string()
    dtypes_str = str(df.dtypes)
    dataset_name = os.path.basename(csv_path)

    prompt = _build_structure_prompt(dataset_name, head_str, dtypes_str, heuristics)
    response_text = _call_local_llm(prompt)

    # Clean up standard markdown wrapping if the LLM adds it
    cleaned_json = response_text.strip()
    if cleaned_json.startswith("```json"):
        cleaned_json = cleaned_json[7:]
    if cleaned_json.startswith("```"):
        cleaned_json = cleaned_json[3:]
    if cleaned_json.endswith("```"):
        cleaned_json = cleaned_json[:-3]
    cleaned_json = cleaned_json.strip()

    try:
        parsed_data = json.loads(cleaned_json)
    except json.JSONDecodeError:
        print("[DataStructureAgent] Warning: Output was not valid JSON. Using heuristic-only fallback.")
        parsed_data = {}

    # CRITICAL: Override LLM output with hard Pandas heuristics
    # The LLM (especially small local models) may hallucinate these flags.
    # Pandas math is ground truth and always wins.
    if heuristics.get("shows_repeated_measures"):
        parsed_data["has_repeated_measures"] = True
    if len(heuristics.get("binary_targets", [])) > 0:
        parsed_data["has_binary_outcome"] = True
    if heuristics.get("potential_survival_data"):
        parsed_data["has_survival_data"] = True
    if heuristics.get("potential_hierarchical_data"):
        parsed_data["is_hierarchical"] = True

    # Phase 2: Add assumption test results
    assumptions = _compute_assumption_tests(df, heuristics)
    parsed_data["assumption_tests"] = assumptions

    formatted_json = json.dumps(parsed_data, indent=4)

    out_path = os.path.join(output_dir, "data_structure.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(formatted_json)
        
    print(f"[DataStructureAgent] data_structure.json saved -> {out_path}")
    return out_path
