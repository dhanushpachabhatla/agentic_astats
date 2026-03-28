"""
data_structure_agent.py - Analyzes dataset structure using Pandas heuristics
and semantic LLM inference (hybrid approach) to infer independent/dependent groups,
repeated measures, etc. Produces data_structure.json.
"""
import os
import json
import re
import pandas as pd
import numpy as np
from openai import OpenAI

from utils.file_utils import read_csv, ensure_output_dir


# Connect to LM Studio Local Server
_LOCAL_CLIENT = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
_MODEL = "local-model"

OUTCOME_KEYWORDS = {
    "outcome", "target", "response", "label", "class", "event", "failure",
    "score", "satisfaction", "severity", "conversion", "rate", "probability",
    "claim", "claims", "count", "visits", "sales", "revenue", "channel",
    "purchase", "subscribed", "subscription", "churn", "complication"
}
GROUP_KEYWORDS = {
    "group", "arm", "treatment", "cohort", "version", "gender", "sex",
    "condition", "campus", "classroom", "school", "hospital", "site",
    "cluster", "region", "center", "centre", "ward", "branch", "method"
}
HIERARCHY_KEYWORDS = {
    "school", "hospital", "clinic", "campus", "classroom", "site", "cluster",
    "region", "center", "centre", "ward", "branch"
}
TIME_KEYWORDS = {
    "time", "date", "day", "days", "month", "months", "week", "weeks",
    "year", "years", "duration", "followup", "visit", "wave", "period", "index"
}
EVENT_KEYWORDS = {"event", "status", "failed", "failure", "death", "censor", "relapse"}
STRING_ORDINAL_ORDERS = [
    ["very low", "low", "medium", "high", "very high"],
    ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"],
    ["poor", "fair", "good", "very good", "excellent"],
]

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


def _column_tokens(col_name: str) -> set[str]:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(col_name))
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
    return {tok for tok in normalized.split("_") if tok}


def _has_any_token(tokens: set[str], keywords: set[str]) -> bool:
    return not keywords.isdisjoint(tokens)


def _is_text_like_dtype(dtype) -> bool:
    return (
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_categorical_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
    )


def _is_id_like_column(col_name: str, series: pd.Series) -> bool:
    tokens = _column_tokens(col_name)
    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    name_hint = (
        str(col_name).lower().endswith("id")
        or str(col_name).lower().endswith("number")
        or str(col_name).lower().endswith("key")
        or _has_any_token(tokens, {"id", "identifier", "subject", "patient", "customer", "policy", "respondent", "participant", "case", "member", "learner"})
    )
    unique_ratio = non_null.nunique() / len(non_null)
    return bool(name_hint and unique_ratio >= 0.9)


def _is_time_like_column(col_name: str, series: pd.Series) -> bool:
    tokens = _column_tokens(col_name)
    if _has_any_token(tokens, TIME_KEYWORDS):
        return True
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    return False


def _score_outcome_candidate(col_name: str, series: pd.Series, is_last_non_id: bool) -> int:
    tokens = _column_tokens(col_name)
    score = 0

    if is_last_non_id:
        score += 4
    if _has_any_token(tokens, OUTCOME_KEYWORDS):
        score += 5
    if _has_any_token(tokens, GROUP_KEYWORDS):
        score -= 3
    if _is_time_like_column(col_name, series):
        score -= 4

    non_null = series.dropna()
    if len(non_null) == 0:
        return score - 10

    n_unique = non_null.nunique()
    if pd.api.types.is_numeric_dtype(series):
        if n_unique > 8:
            score += 2
        elif n_unique == 2 and not _has_any_token(tokens, OUTCOME_KEYWORDS):
            score -= 1
    else:
        if 2 <= n_unique <= 12:
            score += 2

    return score


def _is_string_ordinal_series(series: pd.Series) -> bool:
    non_null = series.dropna()
    if len(non_null) == 0 or not _is_text_like_dtype(series.dtype):
        return False

    unique_vals = sorted({str(v).strip().lower() for v in non_null})
    if len(unique_vals) < 3 or len(unique_vals) > 7:
        return False

    for ordered_labels in STRING_ORDINAL_ORDERS:
        if set(unique_vals).issubset(set(ordered_labels)):
            ordered_present = [label for label in ordered_labels if label in unique_vals]
            if len(ordered_present) == len(unique_vals):
                return True
    return False


def _compute_heuristics(df: pd.DataFrame) -> dict:
    """Compute strict Pandas heuristics to guide (and ground) the LLM."""
    n_rows = len(df)
    heuristics = {
        "candidate_id_columns": [],
        "shows_repeated_measures": False,
        "candidate_grouping_columns": [],
        "numeric_targets": [],
        "binary_targets": [],
        "primary_outcome_column": None,
        "primary_outcome_type": "unknown",
        "has_multiclass_nominal_outcome": False,
        "potential_survival_data": False,
        "potential_hierarchical_data": False,
        "potential_time_series": False,
        "has_independent_groups": False
    }

    non_id_columns = []
    time_like_columns = []
    event_like_columns = []
    grouping_columns = []

    for col in df.columns:
        series = df[col]
        n_unique = df[col].nunique()
        max_freq = df[col].value_counts().max() if n_unique > 0 else 0
        dtype = str(df[col].dtype)

        if _is_id_like_column(col, series):
            heuristics["candidate_id_columns"].append(col)
            # If an ID column has rows appearing multiple times, strongly implies repeated measures
            if max_freq > 1 and max_freq < n_rows:
                heuristics["shows_repeated_measures"] = True
            continue

        non_id_columns.append(col)
        tokens = _column_tokens(col)

        if _is_time_like_column(col, series):
            time_like_columns.append(col)

        if _has_any_token(tokens, EVENT_KEYWORDS) and series.nunique(dropna=True) <= 3:
            event_like_columns.append(col)

        if n_unique > 1 and n_unique <= 10 and "float" not in dtype and not _is_time_like_column(col, series):
            heuristics["candidate_grouping_columns"].append(col)
            grouping_columns.append(col)

        if ("float" in dtype or "int" in dtype) and not _is_time_like_column(col, series):
            if n_unique > 10:
                heuristics["numeric_targets"].append(col)

    if non_id_columns:
        scored = []
        for idx, col in enumerate(non_id_columns):
            score = _score_outcome_candidate(col, df[col], idx == len(non_id_columns) - 1)
            scored.append((score, idx, col))
        heuristics["primary_outcome_column"] = max(scored, key=lambda item: (item[0], item[1]))[2]

    outcome_col = heuristics["primary_outcome_column"]
    if outcome_col:
        outcome_series = df[outcome_col]
        outcome_unique = outcome_series.dropna().nunique()
        if outcome_unique == 2 and not (_has_any_token(_column_tokens(outcome_col), EVENT_KEYWORDS) and len(time_like_columns) > 0):
            heuristics["binary_targets"] = [outcome_col]
            heuristics["primary_outcome_type"] = "binary"
        elif _is_string_ordinal_series(outcome_series):
            heuristics["primary_outcome_type"] = "ordinal"
        elif pd.api.types.is_integer_dtype(outcome_series) and 3 <= outcome_unique <= 7:
            unique_vals = sorted(outcome_series.dropna().unique())
            if unique_vals == list(range(min(unique_vals), max(unique_vals) + 1)):
                heuristics["primary_outcome_type"] = "ordinal"
            elif outcome_unique >= 3:
                heuristics["primary_outcome_type"] = "multiclass_nominal"
                heuristics["has_multiclass_nominal_outcome"] = True
        elif _is_text_like_dtype(outcome_series.dtype):
            if outcome_unique >= 3:
                heuristics["primary_outcome_type"] = "multiclass_nominal"
                heuristics["has_multiclass_nominal_outcome"] = True
        elif pd.api.types.is_numeric_dtype(outcome_series):
            heuristics["primary_outcome_type"] = "continuous"

    grouping_without_outcome = [col for col in grouping_columns if col != outcome_col]
    heuristics["has_independent_groups"] = bool(grouping_without_outcome) and not heuristics["shows_repeated_measures"]

    if len(time_like_columns) > 0 and len(event_like_columns) > 0:
        heuristics["potential_survival_data"] = True
        if heuristics["primary_outcome_type"] == "binary":
            heuristics["binary_targets"] = []
            heuristics["primary_outcome_type"] = "survival_event"

    if len(grouping_without_outcome) >= 2:
        heuristics["potential_hierarchical_data"] = True
    else:
        for col in grouping_without_outcome:
            if _has_any_token(_column_tokens(col), HIERARCHY_KEYWORDS):
                heuristics["potential_hierarchical_data"] = True
                break

    heuristics["potential_time_series"] = bool(time_like_columns)

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
        "has_multiclass_nominal_outcome": heuristics.get("has_multiclass_nominal_outcome", False),
        "multiclass_outcome_columns": [],
        "has_zero_inflation": False,
        "zero_inflated_columns": [],
        "has_overdispersed_counts": False,
        "overdispersed_columns": [],
        "has_sparse_contingency": False,
        "sparse_contingency_tables": [],
        "has_proportion_outcome": False,
        "proportion_columns": [],
        "has_time_series": False,
        "time_series_columns": [],
        "has_perfect_separation": False,
        "perfect_separation_columns": [],
        "structured_missingness": False,
        "structured_missingness_columns": [],
        "sample_size": len(df)
    }
    
    n_rows = len(df)
    candidate_id_cols = set(heuristics.get("candidate_id_columns", []))
    outcome_col = heuristics.get("primary_outcome_column")
    outcome_type = heuristics.get("primary_outcome_type")
    time_like_cols = [col for col in df.columns if _is_time_like_column(col, df[col])]
    numeric_cols = [
        col for col in df.select_dtypes(include=["number"]).columns.tolist()
        if col not in candidate_id_cols and col not in time_like_cols
    ]
    
    # ----- 1. Small Sample Detection -----
    if n_rows < 30:
        assumptions["is_small_sample"] = True
    
    # ----- 2. Normality Test (Shapiro-Wilk) on numeric columns -----
    normality_cols = numeric_cols if outcome_col is None else list(dict.fromkeys([outcome_col] + [col for col in numeric_cols if col != outcome_col]))
    for col in normality_cols:
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
    grouping_cols = [
        col for col in heuristics.get("candidate_grouping_columns", [])
        if col not in candidate_id_cols and col != outcome_col
    ]
    target_cols = [
        col for col in heuristics.get("numeric_targets", [])
        if col not in candidate_id_cols and col not in time_like_cols
    ]
    if outcome_col and outcome_col in target_cols:
        target_cols = [outcome_col] + [col for col in target_cols if col != outcome_col]

    if grouping_cols and target_cols:
        for group_col in grouping_cols:
            for target_col in target_cols:
                if group_col == target_col:
                    continue

                groups = [
                    group_data[target_col].dropna().values
                    for _, group_data in df.groupby(group_col)
                    if len(group_data[target_col].dropna()) >= 2
                ]
                if len(groups) < 2:
                    continue

                try:
                    stat, p_value = stats.levene(*groups)
                    if p_value < 0.05:
                        assumptions["homoscedasticity_violated"] = True
                        break
                except Exception:
                    continue

            if assumptions["homoscedasticity_violated"]:
                break
    
    # ----- 4. Multicollinearity (VIF) -----
    predictor_numeric_cols = [col for col in numeric_cols if col != outcome_col]
    if len(predictor_numeric_cols) >= 2:
        try:
            numeric_df = df[predictor_numeric_cols].dropna()
            if len(numeric_df) > len(predictor_numeric_cols):
                # Add constant for VIF calculation
                X = numeric_df.values
                corr_matrix = np.corrcoef(X, rowvar=False)
                # Check if any pair has |r| > 0.85
                for i in range(len(predictor_numeric_cols)):
                    for j in range(i + 1, len(predictor_numeric_cols)):
                        corr_abs = abs(corr_matrix[i, j])
                        # Round to the same precision used in reporting so borderline
                        # cases near the threshold are classified consistently.
                        if round(float(corr_abs), 2) >= 0.85:
                            assumptions["multicollinearity_detected"] = True
                            assumptions["high_vif_columns"].append(
                                f"{predictor_numeric_cols[i]} <-> {predictor_numeric_cols[j]} (r={corr_matrix[i,j]:.2f})"
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
        elif _is_string_ordinal_series(df[col]):
            assumptions["has_ordinal_outcome"] = True
            assumptions["ordinal_columns"].append(col)

    if outcome_type == "multiclass_nominal" and outcome_col:
        assumptions["has_multiclass_nominal_outcome"] = True
        assumptions["multiclass_outcome_columns"].append(outcome_col)
    
    # ----- 6. Zero-Inflated / Overdispersed Count Data Detection -----
    count_cols = []
    for col in numeric_cols:
        col_data = df[col].dropna()
        tokens = _column_tokens(col)
        is_count_named = _has_any_token(tokens, {"count", "counts", "claim", "claims", "visit", "visits", "num"})
        is_outcome_count = col == outcome_col
        if (
            "int" in str(df[col].dtype)
            and (col_data >= 0).all()
            and col_data.nunique() > 2
            and (is_count_named or is_outcome_count)
        ):
            count_cols.append(col)
            zero_pct = (col_data == 0).sum() / len(col_data)
            if zero_pct > 0.3:  # More than 30% zeros = potential zero-inflation
                assumptions["has_zero_inflation"] = True
                assumptions["zero_inflated_columns"].append({
                    "column": col,
                    "zero_percentage": round(float(zero_pct * 100), 1)
                })
            mean_val = float(col_data.mean())
            var_val = float(col_data.var(ddof=1)) if len(col_data) > 1 else 0.0
            if mean_val > 0 and var_val > mean_val * 1.5 and zero_pct <= 0.3:
                assumptions["has_overdispersed_counts"] = True
                assumptions["overdispersed_columns"].append({
                    "column": col,
                    "mean": round(mean_val, 3),
                    "variance": round(var_val, 3)
                })

    # ----- 7. Sparse contingency tables -----
    categorical_cols = [
        col for col in grouping_cols
        if df[col].dropna().nunique() >= 2
    ]
    if outcome_col and outcome_col not in candidate_id_cols and df[outcome_col].dropna().nunique() <= 10:
        categorical_targets = [outcome_col]
    else:
        categorical_targets = []

    for group_col in categorical_cols:
        for target_col in categorical_targets:
            if group_col == target_col:
                continue
            table = pd.crosstab(df[group_col], df[target_col])
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue
            try:
                _, _, _, expected = stats.chi2_contingency(table)
                if (expected < 5).any():
                    assumptions["has_sparse_contingency"] = True
                    assumptions["sparse_contingency_tables"].append(f"{group_col} x {target_col}")
            except Exception:
                continue

    # ----- 8. Proportion outcome -----
    if outcome_col and pd.api.types.is_numeric_dtype(df[outcome_col]):
        outcome_data = df[outcome_col].dropna()
        if len(outcome_data) > 0 and outcome_data.nunique() > 10 and ((outcome_data >= 0) & (outcome_data <= 1)).all():
            assumptions["has_proportion_outcome"] = True
            assumptions["proportion_columns"].append(outcome_col)

    # ----- 9. Time-series signal -----
    if outcome_col and outcome_col in df.columns and len(time_like_cols) > 0 and pd.api.types.is_numeric_dtype(df[outcome_col]):
        for time_col in time_like_cols:
            ordered = df[[time_col, outcome_col]].dropna().sort_values(time_col)
            if len(ordered) >= 20:
                autocorr = ordered[outcome_col].autocorr(lag=1)
                if pd.notna(autocorr) and abs(float(autocorr)) >= 0.6:
                    assumptions["has_time_series"] = True
                    assumptions["time_series_columns"].append({
                        "time_column": time_col,
                        "outcome_column": outcome_col,
                        "lag1_autocorr": round(float(autocorr), 3)
                    })

    # ----- 10. Perfect separation in binary outcomes -----
    if outcome_col and outcome_type == "binary":
        outcome_data = df[outcome_col].dropna()
        for col in df.columns:
            if col in candidate_id_cols or col == outcome_col:
                continue
            predictor = df[[col, outcome_col]].dropna()
            if predictor.empty:
                continue
            if predictor[col].nunique() == 2:
                table = pd.crosstab(predictor[col], predictor[outcome_col])
                if table.shape == (2, 2) and ((table == 0).any(axis=1).all() or (table == 0).any(axis=0).all()):
                    assumptions["has_perfect_separation"] = True
                    assumptions["perfect_separation_columns"].append(col)

    # ----- 11. Structured missingness -----
    cols_with_missing = [col for col in df.columns if df[col].isna().sum() > 0]
    for missing_col in cols_with_missing:
        missing_indicator = df[missing_col].isna().astype(int)
        for group_col in grouping_cols:
            group_missing = pd.DataFrame({"group": df[group_col], "missing": missing_indicator}).dropna()
            if group_missing["group"].nunique() < 2:
                continue
            missing_rates = group_missing.groupby("group")["missing"].mean()
            if (missing_rates.max() - missing_rates.min()) >= 0.2:
                assumptions["structured_missingness"] = True
                assumptions["structured_missingness_columns"].append({
                    "column": missing_col,
                    "grouping_column": group_col,
                    "missing_rate_range": round(float(missing_rates.max() - missing_rates.min()), 3)
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
    try:
        response_text = _call_local_llm(prompt)
    except RuntimeError as e:
        print(f"[DataStructureAgent] Warning: {e}")
        print("[DataStructureAgent] Falling back to heuristic-only structure inference.")
        response_text = "{}"

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
    parsed_data["has_repeated_measures"] = heuristics.get("shows_repeated_measures", False)
    parsed_data["has_independent_groups"] = heuristics.get("has_independent_groups", False)
    parsed_data["has_binary_outcome"] = len(heuristics.get("binary_targets", [])) > 0
    parsed_data["has_multiclass_nominal_outcome"] = heuristics.get("has_multiclass_nominal_outcome", False)
    parsed_data["has_survival_data"] = heuristics.get("potential_survival_data", False)
    parsed_data["is_hierarchical"] = heuristics.get("potential_hierarchical_data", False)
    parsed_data["primary_outcome_column"] = heuristics.get("primary_outcome_column")
    parsed_data["primary_outcome_type"] = heuristics.get("primary_outcome_type", "unknown")

    # Phase 2: Add assumption test results
    assumptions = _compute_assumption_tests(df, heuristics)
    parsed_data["assumption_tests"] = assumptions

    formatted_json = json.dumps(parsed_data, indent=4)

    out_path = os.path.join(output_dir, "data_structure.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(formatted_json)
        
    print(f"[DataStructureAgent] data_structure.json saved -> {out_path}")
    return out_path
