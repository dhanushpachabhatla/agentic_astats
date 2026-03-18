"""
file_utils.py — Helpers for reading/writing CSV, markdown, and text files.
"""
import os
import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def read_markdown(path: str) -> str:
    """Read a markdown file and return its contents as a string."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Markdown file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_markdown(path: str, content: str) -> None:
    """Write a string to a markdown file, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[file_utils] Written: {path}")


def ensure_output_dir(output_dir: str) -> None:
    """Create the output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
