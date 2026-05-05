"""
batch_run.py — Run the ROAR evaluation pipeline on every .docx file in the
pipeline directory (including good_roars/ and bad_roars/ subfolders) and
print a summary table comparing pipeline predictions to human labels.

Usage:
    python batch_run.py                      # scan current dir + subdirs
    python batch_run.py --dir path/to/roars  # specific root directory
    python batch_run.py --json               # output results as JSON
    python batch_run.py --threshold 0.5      # custom pass/fail threshold
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# Pipeline predicts "good" when weighted_score >= this threshold
DEFAULT_THRESHOLD = 0.5


def find_roar_files(root: Path) -> list[tuple[Path, str]]:
    """
    Return (path, human_label) pairs for every .docx under *root*.

    Label is derived from the parent directory name:
        good_roars/ → "good"
        bad_roars/  → "bad"
        anything else → "unknown"
    """
    pairs: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*.docx")):
        parent = path.parent.name.lower()
        if "good" in parent:
            label = "good"
        elif "bad" in parent:
            label = "bad"
        else:
            label = "unknown"
        pairs.append((path, label))
    return pairs


def run_one(path: Path) -> dict:
    from pipeline.extractor import SectionExtractor
    from pipeline.roar_pipeline import ROARPipeline

    pipeline = ROARPipeline()
    start = time.time()
    try:
        sections = SectionExtractor.extract_from_docx(path)
        result   = pipeline.run(pre_sections=sections)
        return {
            "file":           path.name,
            "folder":         path.parent.name,
            "plo":            result.final_scores.plo,
            "methods":        result.final_scores.methods,
            "results":        result.final_scores.results,
            "plan":           result.final_scores.plan,
            "weighted_score": result.weighted_score,
            "consistent":     result.consistent,
            "iterations":     result.iterations,
            "elapsed_sec":    round(time.time() - start, 1),
            "error":          None,
            "reasoning":      result.reasoning.model_dump() if result.reasoning else {},
        }
    except Exception as exc:
        logging.exception("Pipeline failed on %s", path.name)
        return {
            "file": path.name, "folder": path.parent.name,
            "plo": None, "methods": None, "results": None, "plan": None,
            "weighted_score": None, "consistent": None, "iterations": None,
            "elapsed_sec": round(time.time() - start, 1),
            "error": str(exc)[:200], "reasoning": {},
        }


def _badge(score) -> str:
    if score is None: return " ? "
    return " 1 " if score == 1 else " 0 "


def print_table(rows: list[dict], threshold: float) -> None:
    sep = "─" * 108

    print()
    print(sep)
    print(f"  {'FILE':<44} {'LABEL':>7} {'PLO':>3} {'MTH':>3} {'RES':>3} {'PLN':>3}"
          f"  {'SCORE':>6}  {'PRED':>4}  {'MATCH':>5}  {'SEC':>5}")
    print(sep)

    correct = 0
    scored  = 0
    for r in rows:
        label = r.get("human_label", "unknown")
        if r["error"]:
            print(f"  {r['file']:<44}  {label:>7}  ERROR: {r['error'][:50]}")
            continue

        pred       = "good" if r["weighted_score"] >= threshold else "bad"
        match      = "  OK " if (label != "unknown" and pred == label) else ("MISS " if label != "unknown" else "  -- ")
        score_str  = f"{r['weighted_score']*100:5.1f}%"

        if label != "unknown":
            scored += 1
            if pred == label:
                correct += 1

        print(
            f"  {r['file']:<44}"
            f"  {label:>7}"
            f"  {_badge(r['plo'])}"
            f"  {_badge(r['methods'])}"
            f"  {_badge(r['results'])}"
            f"  {_badge(r['plan'])}"
            f"  {score_str}"
            f"  {pred:>4}"
            f"  {match}"
            f"  {r['elapsed_sec']:>5.1f}s"
        )
    print(sep)

    # Summary
    successful = [r for r in rows if r["error"] is None]
    if successful:
        avg = sum(r["weighted_score"] for r in successful) / len(successful)
        print(f"\n  Files: {len(rows)}  |  Errors: {len(rows)-len(successful)}"
              f"  |  Avg score: {avg*100:.1f}%"
              f"  |  Threshold: {threshold*100:.0f}%"
              + (f"  |  Accuracy: {correct}/{scored} = {correct/scored*100:.0f}%"
                 if scored > 0 else ""))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ROAR evaluation")
    parser.add_argument("--dir",       default=".", metavar="PATH")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Score threshold for good/bad prediction (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--json",      action="store_true", dest="json_out")
    args = parser.parse_args()

    root  = Path(args.dir).resolve()
    pairs = find_roar_files(root)

    if not pairs:
        print(f"No .docx files found under {root}")
        sys.exit(1)

    print(f"\nFound {len(pairs)} ROAR file(s) under {root}")
    print(f"Evaluator : LLaMA 4 Maverick (Azure AI Foundry MaaS)")
    print(f"Verifier  : GPT-5.4-mini     (Azure OpenAI)")
    print(f"Threshold : {args.threshold*100:.0f}% → good / below → bad\n")

    rows = []
    for i, (path, label) in enumerate(pairs, 1):
        print(f"[{i:02}/{len(pairs):02}] [{label:>7}]  {path.name} ...", end="", flush=True)
        r = run_one(path)
        r["human_label"] = label
        if r["error"]:
            print(f"  ERROR")
        else:
            pred = "good" if r["weighted_score"] >= args.threshold else "bad"
            match = "OK" if pred == label else ("MISS" if label != "unknown" else "--")
            print(f"  score={r['weighted_score']*100:.0f}%  pred={pred}  [{match}]  ({r['elapsed_sec']}s)")
        rows.append(r)

    if args.json_out:
        print(json.dumps(rows, indent=2))
    else:
        print_table(rows, args.threshold)


if __name__ == "__main__":
    main()
