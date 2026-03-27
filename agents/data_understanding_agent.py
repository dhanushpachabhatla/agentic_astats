"""
data_understanding_agent.py - Reads a CSV, computes stats locally,
then calls Gemini to generate dataset_profile.md. Also builds
an initial analysis_notebook.ipynb using nbformat.
"""
import os
import io
import pandas as pd
from openai import OpenAI

from utils.file_utils import read_csv, write_markdown, ensure_output_dir
from utils.notebook_utils import (
    create_notebook, add_markdown_cell, add_code_cell, save_notebook
)


# Connect to LM Studio Local Server
_LOCAL_CLIENT = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
_MODEL = "local-model"  # LM Studio ignores this and uses the loaded model


def _call_local_llm(prompt: str) -> str:
    """Send prompt to local LM Studio server."""
    try:
        response = _LOCAL_CLIENT.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert data scientist passing information to a planning agent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Failed to call local LM Studio API: {e}. Is the server running on port 1234?")


def _compute_local_stats(df) -> str:
    """Return a plain-text summary of the DataFrame for the Gemini prompt."""
    buf = io.StringIO()

    buf.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n")

    buf.write("Column Types:\n")
    for col, dtype in df.dtypes.items():
        buf.write(f"  {col}: {dtype}\n")

    buf.write("\nMissing Values per Column:\n")
    missing = df.isnull().sum()
    for col, count in missing.items():
        pct = round(100 * count / len(df), 2)
        buf.write(f"  {col}: {count} ({pct}%)\n")

    buf.write("\nBasic Statistics (numeric columns):\n")
    buf.write(df.describe().to_string())
    buf.write("\n")

    # Value counts / unique for categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        buf.write("\nCategorical Columns - Unique Value Counts:\n")
        for col in cat_cols:
            buf.write(f"  {col}: {df[col].nunique()} unique values\n")
            top5 = df[col].value_counts().head(5)
            buf.write(f"    Top 5: {dict(top5)}\n")

    return buf.getvalue()


def _build_profile_prompt(dataset_name: str, stats_text: str) -> str:
    return f"""You are a data scientist performing Exploratory Data Analysis (EDA).

Dataset: {dataset_name}

--- COMPUTED STATISTICS ---
{stats_text}
--- END STATISTICS ---

Using the above statistics, write a comprehensive dataset_profile.md report in Markdown. Include:

1. **Dataset Overview** - what the dataset appears to contain, number of rows/columns
2. **Column Descriptions** - for each column: data type, missing %, key observations
3. **Basic Statistics Summary** - highlight min, max, mean, std for key numeric columns
4. **Missing Values Analysis** - which columns have missing data and recommended handling
5. **Categorical Variables** - top categories and their distribution
6. **Key EDA Insights** - at least 5 interesting observations or patterns from the stats
7. **Potential Data Quality Issues** - skewness, outliers, high cardinality, etc.
8. **Recommended Next Steps** - suggested analyses based on what you see

Write clearly and concisely. Use markdown headers, bullet points, and tables where helpful.
"""


def _build_notebook(df, csv_path: str, output_dir: str) -> str:
    """Build an initial Jupyter notebook with exploration cells."""
    nb = create_notebook()
    csv_filename = os.path.basename(csv_path)

    add_markdown_cell(nb, f"# AStats — Initial Data Exploration\n**Dataset:** `{csv_filename}`")
    add_code_cell(nb, "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n%matplotlib inline\n")
    add_markdown_cell(nb, "## 1. Load Dataset")
    add_code_cell(nb, f"df = pd.read_csv(r'{os.path.abspath(csv_path)}')\ndf.head()")
    add_markdown_cell(nb, "## 2. Dataset Info")
    add_code_cell(nb, "df.info()\ndf.describe(include='all')")

    path = os.path.join(output_dir, "analysis_notebook.ipynb")
    save_notebook(nb, path)
    return path


def run(csv_path: str, output_dir: str) -> str:
    """
    Run the Data Understanding Agent.

    Parameters
    ----------
    csv_path   : path to the input CSV file
    output_dir : directory where outputs will be written

    Returns
    -------
    str : absolute path to the generated dataset_profile.md
    """
    ensure_output_dir(output_dir)

    print("\n[DataUnderstandingAgent] Loading dataset...")
    df = read_csv(csv_path)
    print(f"  -> {df.shape[0]} rows x {df.shape[1]} columns")

    print("[DataUnderstandingAgent] Computing local statistics...")
    stats_text = _compute_local_stats(df)

    print("[DataUnderstandingAgent] Calling Local LM Studio to generate dataset profile...")
    dataset_name = os.path.basename(csv_path)
    prompt = _build_profile_prompt(dataset_name, stats_text)
    profile_md = _call_local_llm(prompt)

    profile_path = os.path.join(output_dir, "dataset_profile.md")
    write_markdown(profile_path, profile_md)
    print(f"[DataUnderstandingAgent] dataset_profile.md saved -> {profile_path}")

    print("[DataUnderstandingAgent] Building analysis notebook...")
    notebook_path = _build_notebook(df, csv_path, output_dir)
    print(f"[DataUnderstandingAgent] analysis_notebook.ipynb saved -> {notebook_path}")

    return profile_path
