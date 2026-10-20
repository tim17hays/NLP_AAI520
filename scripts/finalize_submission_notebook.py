"""Apply the maintained report narrative without executing model code."""
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "notebooks/final_multi_agent_financial_analysis.ipynb"
    notebook = nbformat.read(path, as_version=4)
    narrative = json.loads((ROOT / "docs/team/final_report_narrative.json").read_text())
    for cell in notebook.cells:
        if cell.cell_type == "markdown" and cell.id in narrative:
            cell.source = narrative[cell.id]
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print("Report narrative updated; experiment outputs preserved.")


if __name__ == "__main__":
    main()
