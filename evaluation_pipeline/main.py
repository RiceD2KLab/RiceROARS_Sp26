"""
main.py – CLI entry point for the ROAR quality assessment pipeline.

Usage examples
--------------
# Evaluate a PDF ROAR document (default: LLaMA via Ollama + Qwen3.5 via vLLM)
python main.py --pdf path/to/roar.pdf

# Evaluate a plain-text file
python main.py --text path/to/roar.txt

# Pass raw text inline (quick test)
python main.py --inline "PLO: ... Methods: ... Results: ... Plan: ..."

# Override individual models
python main.py --pdf roar.pdf --evaluator llama3:8b --verifier Qwen/Qwen3.5-0.8B

# Override backends (e.g. run everything through Ollama)
python main.py --pdf roar.pdf --evaluator-backend ollama --verifier-backend ollama

# Output structured JSON for downstream processing
python main.py --pdf roar.pdf --json

# Verbose / debug logging
python main.py --pdf roar.pdf --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import config
from pipeline.roar_pipeline import ROARPipeline


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("roar_main")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_SEP = "─" * 60

_SECTION_LABELS = {
    "plo":     "PLO          (weight 25%)",
    "methods": "Methods      (weight 30%)",
    "results": "Results      (weight 30%)",
    "plan":    "Plan         (weight 15%)",
}


def _print_result(result) -> None:
    print()
    print(_SEP)
    print("  ROAR Quality Assessment – Final Report")
    print(_SEP)

    print("\nSECTION SCORES")
    for field, label in _SECTION_LABELS.items():
        score = getattr(result.final_scores, field)
        badge = "✓ PASS" if score == 1 else "✗ FAIL"
        print(f"  {label:<30}  {badge}  (score={score})")

    pct = result.weighted_score * 100
    print(f"\nWEIGHTED QUALITY SCORE : {result.weighted_score:.4f}  ({pct:.1f} / 100)")
    print(f"CONSISTENT             : {'Yes' if result.consistent else 'No'}")
    print(f"FEEDBACK ITERATIONS    : {result.iterations}")

    if result.reasoning:
        print("\nREASONING")
        for field in ("plo", "methods", "results", "plan"):
            text = getattr(result.reasoning, field)
            if text:
                print(f"\n  [{field.upper()}]\n  {text}")

    print()
    print(_SEP)


def _print_json(result) -> None:
    output = {
        "final_scores":   result.final_scores.model_dump(),
        "weighted_score": result.weighted_score,
        "consistent":     result.consistent,
        "iterations":     result.iterations,
        "reasoning":      result.reasoning.model_dump() if result.reasoning else None,
    }
    print(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="roar_eval",
        description="Prompt-based multi-model pipeline for ROAR quality assessment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input source (mutually exclusive)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf",    metavar="PATH", help="Path to a ROAR PDF file")
    src.add_argument("--docx",   metavar="PATH", help="Path to a Rice ROAR Word (.docx) file")
    src.add_argument("--text",   metavar="PATH", help="Path to a plain-text ROAR file")
    src.add_argument("--inline", metavar="TEXT", help="Inline ROAR text (quick test)")

    # Model names
    p.add_argument(
        "--evaluator",
        default=config.EVALUATOR_MODEL,
        metavar="MODEL",
        help="Model identifier for the primary evaluator (LLaMA)",
    )
    p.add_argument(
        "--verifier",
        default=config.VERIFIER_MODEL,
        metavar="MODEL",
        help="Model identifier for the verifier (Qwen3.5-0.8B)",
    )

    # Backend selection
    p.add_argument(
        "--evaluator-backend",
        default=config.EVALUATOR_BACKEND,
        choices=["ollama", "openai"],
        dest="evaluator_backend",
        help="Backend for the evaluator",
    )
    p.add_argument(
        "--verifier-backend",
        default=config.VERIFIER_BACKEND,
        choices=["ollama", "openai"],
        dest="verifier_backend",
        help="Backend for the verifier (default: openai → vLLM / SGLang)",
    )

    # Pipeline tuning
    p.add_argument(
        "--max-iter",
        type=int,
        default=config.MAX_ITERATIONS,
        dest="max_iter",
        help="Maximum feedback-loop iterations",
    )

    # Output format
    p.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print result as JSON instead of formatted report",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # -------------------------------------------------------------------
    # Load document  (text / pre-extracted sections for .docx)
    # -------------------------------------------------------------------
    from pipeline.extractor import SectionExtractor

    # pre_sections is set when we can extract sections directly (e.g. .docx
    # table mapping) — the pipeline will skip its own extraction step.
    pre_sections = None
    document     = ""

    if args.pdf:
        path = Path(args.pdf)
        if not path.exists():
            logger.error("PDF file not found: %s", path)
            sys.exit(1)
        logger.info("Loading PDF: %s", path)
        document = SectionExtractor.load_pdf(path)

    elif args.docx:
        path = Path(args.docx)
        if not path.exists():
            logger.error(".docx file not found: %s", path)
            sys.exit(1)
        logger.info("Loading Word document: %s", path)
        # Directly map the ROAR table — no LLM extraction needed
        pre_sections = SectionExtractor.extract_from_docx(path)
        logger.info(
            "Sections extracted from .docx table:\n"
            "  PLO     : %.80s…\n"
            "  Methods : %.80s…\n"
            "  Results : %.80s…\n"
            "  Plan    : %.80s…",
            pre_sections.plo, pre_sections.methods,
            pre_sections.results, pre_sections.plan,
        )

    elif args.text:
        path = Path(args.text)
        if not path.exists():
            logger.error("Text file not found: %s", path)
            sys.exit(1)
        logger.info("Loading text file: %s", path)
        document = path.read_text(encoding="utf-8")

    else:
        document = args.inline

    # -------------------------------------------------------------------
    # Run pipeline
    # -------------------------------------------------------------------
    logger.info(
        "Starting pipeline  |  evaluator=%s (%s)  verifier=%s (%s)  max_iter=%d",
        args.evaluator, args.evaluator_backend,
        args.verifier,  args.verifier_backend,
        args.max_iter,
    )

    pipeline = ROARPipeline(
        evaluator_model=args.evaluator,
        verifier_model=args.verifier,
        evaluator_backend=args.evaluator_backend,
        verifier_backend=args.verifier_backend,
        max_iterations=args.max_iter,
    )

    result = pipeline.run(document, pre_sections=pre_sections)

    # Ensure stdout can handle non-ASCII characters on Windows
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    # -------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------
    if args.json_output:
        _print_json(result)
    else:
        _print_result(result)


if __name__ == "__main__":
    main()
