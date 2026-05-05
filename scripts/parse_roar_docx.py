#!/usr/bin/env python3
"""
Parse a ROAR .docx and output extracted sections as JSON to stdout.
Matches the logic in background/labeled_data_parser_docx.ipynb.
Usage: python parse_roar_docx.py <path-to.docx>
Exit code 0 with JSON on success; non-zero on error.
"""

import json
import re
import sys
from pathlib import Path

try:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph
except ImportError:
    print(json.dumps({"error": "python-docx not installed"}), file=sys.stderr)
    sys.exit(2)


def iter_block_items(doc):
    """Iterate through paragraphs and tables in document order."""
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, doc)
        elif child.tag.endswith("}tbl"):
            yield Table(child, doc)


def parse_roar_document(file_path: str) -> dict:
    """
    Extract department, plo, methods, results_conclusions, improvement_plan
    from a ROAR .docx. Returns a dict suitable for ExtractedSections.
    """
    doc = Document(file_path)
    row_data = {
        "department": None,
        "plo": "",
        "methods": "",
        "results_conclusions": "",
        "improvement_plan": "",
    }
    current_section = None

    def handle_text(text: str) -> None:
        nonlocal current_section
        text = (text or "").strip()
        if not text:
            return
        if "Department/Academic Program:" in text:
            row_data["department"] = text.split(":", 1)[-1].strip()
            current_section = None
            return
        t = text.lower()
        if "specific program learning outcome" in t:
            current_section = "plo"
            return
        if t.startswith("methods"):
            current_section = "methods"
            return
        if "results and conclusions" in t:
            current_section = "results_conclusions"
            return
        if "improvement plan" in t:
            current_section = "improvement_plan"
            return
        if current_section:
            row_data[current_section] += text + " "

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            handle_text(block.text)
        elif isinstance(block, Table):
            for row in block.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        handle_text(p.text)

    for k in ("plo", "methods", "results_conclusions", "improvement_plan"):
        row_data[k] = row_data[k].strip()

    # Strip PLO/SLO number prefix
    row_data["plo"] = re.sub(
        r"^\s*\(?\b(?:PLO|SLO)\s*\d+[^\w]*", "", row_data["plo"]
    ).strip()

    # Remove prompt text if present (legacy template text)
    row_data["methods"] = row_data["methods"].replace(
        "What kind of direct assessment did you use? (i.e. describe how you selected the sample of work that was assessed, the rubric used, etc. Best practices and SACSCOC requires the use of direct assessment.) If you also used indirect assessment, please describe that as well.",
        "",
    ).strip()
    row_data["results_conclusions"] = row_data["results_conclusions"].replace(
        "What were the results of your evaluation?",
        "",
    ).strip()
    row_data["improvement_plan"] = row_data["improvement_plan"].replace(
        "Based on your results, what changes (if any) will you make? What is your timeline to make these changes? What is your timeline for assessing the impact of these changes?",
        "",
    ).strip()

    # Return only non-empty or present keys for ExtractedSections
    out = {}
    if row_data["department"]:
        out["department"] = row_data["department"]
    if row_data["plo"]:
        out["plo"] = row_data["plo"]
    if row_data["methods"]:
        out["methods"] = row_data["methods"]
    if row_data["results_conclusions"]:
        out["results_conclusions"] = row_data["results_conclusions"]
    if row_data["improvement_plan"]:
        out["improvement_plan"] = row_data["improvement_plan"]
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: parse_roar_docx.py <path.docx>"}), file=sys.stderr)
        return 1
    path = Path(sys.argv[1])
    if not path.exists():
        print(json.dumps({"error": f"File not found: {path}"}), file=sys.stderr)
        return 1
    if path.suffix.lower() != ".docx":
        print(json.dumps({"error": "Expected .docx file"}), file=sys.stderr)
        return 1
    try:
        extracted = parse_roar_document(str(path))
        print(json.dumps(extracted))
        return 0
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
