"""
notebook_utils.py — Helpers for creating Jupyter notebooks using nbformat.
"""
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def create_notebook() -> nbformat.NotebookNode:
    """Create a new empty v4 Jupyter notebook."""
    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    }
    return nb


def add_markdown_cell(nb: nbformat.NotebookNode, text: str) -> None:
    """Append a markdown cell to the notebook."""
    nb.cells.append(new_markdown_cell(text))


def add_code_cell(nb: nbformat.NotebookNode, code: str) -> None:
    """Append a code cell to the notebook."""
    nb.cells.append(new_code_cell(code))


def save_notebook(nb: nbformat.NotebookNode, path: str) -> None:
    """Save the notebook to disk, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[notebook_utils] Notebook saved: {path}")
