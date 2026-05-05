#!/usr/bin/env python3
"""
Run the ROAR evaluation pipeline for the Next.js API.

Reads a PDF or .docx path, prints a single JSON object to stdout (no extra logs).

Production defaults (override via .env): model profile MS3 (DeepSeek-V3.2 + o4-mini
verifier), prompt set B (Chain-of-Thought Evidence Anchoring), matching the team's
final pipeline (MS3 strict + prompt set B).

Usage:
  python3 scripts/run_roar_evaluation.py --pdf /path/to/file.pdf
  python3 scripts/run_roar_evaluation.py --docx /path/to/file.docx

Exit code 0 on success, 1 on failure (error details are still JSON on stdout).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = REPO_ROOT / "evaluation_pipeline"
sys.path.insert(0, str(EVAL_ROOT))

# Load env before importing config-dependent pipeline code
from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")
load_dotenv(EVAL_ROOT / ".env")

logging.basicConfig(level=logging.WARNING)


def _build_pipeline(
    evaluator_model_override: str | None = None,
    verifier_model_override: str | None = None,
):
    """ROARPipeline from ROAR_MODEL_PROFILE + ROAR_PROMPT_SET + deployment env overrides."""
    from model_sets import MODEL_SETS
    from pipeline.roar_pipeline import ROARPipeline
    from prompt_sets import PROMPT_SETS

    pk = os.getenv("ROAR_PROMPT_SET", "B_ChainOfThought")
    mk = os.getenv("ROAR_MODEL_PROFILE", "MS3_DeepSeek_o4mini")

    if pk not in PROMPT_SETS:
        raise ValueError(
            f"Unknown ROAR_PROMPT_SET={pk!r}. Choose from: {list(PROMPT_SETS.keys())}"
        )
    if mk not in MODEL_SETS:
        raise ValueError(
            f"Unknown ROAR_MODEL_PROFILE={mk!r}. Choose from: {list(MODEL_SETS.keys())}"
        )

    ps = PROMPT_SETS[pk]
    ms = MODEL_SETS[mk]
    ev = ms["evaluator"]
    ve = ms["verifier"]

    ev_model = evaluator_model_override or os.getenv("EVALUATOR_MODEL", ev["model"])
    ve_model = verifier_model_override or os.getenv("VERIFIER_MODEL", ve["model"])
    max_iter = int(os.getenv("MAX_ITERATIONS", "2"))

    return ROARPipeline(
        evaluator_model=ev_model,
        verifier_model=ve_model,
        evaluator_backend=ev["backend"],
        verifier_backend=ve["backend"],
        max_iterations=max_iter,
        prompt_set=ps,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", type=Path, metavar="PATH", help="Path to a ROAR PDF")
    group.add_argument("--docx", type=Path, metavar="PATH", help="Path to a ROAR .docx")
    parser.add_argument(
        "--evaluator-model",
        dest="evaluator_model_override",
        help="Optional evaluator deployment/model override for this run.",
    )
    parser.add_argument(
        "--verifier-model",
        dest="verifier_model_override",
        help="Optional verifier deployment/model override for this run.",
    )
    args = parser.parse_args()

    try:
        from pipeline.extractor import SectionExtractor

        pre_sections = None
        document = ""

        if args.pdf is not None:
            path = args.pdf
            if not path.is_file():
                raise FileNotFoundError(f"PDF not found: {path}")
            document = SectionExtractor.load_pdf(path)
        else:
            path = args.docx
            assert path is not None
            if not path.is_file():
                raise FileNotFoundError(f".docx not found: {path}")
            pre_sections = SectionExtractor.extract_from_docx(path)

        pipeline = _build_pipeline(
            evaluator_model_override=args.evaluator_model_override,
            verifier_model_override=args.verifier_model_override,
        )
        result = pipeline.run(document, pre_sections=pre_sections)

        sections = result.sections
        if sections is None:
            raise RuntimeError("Pipeline returned no sections")

        extracted = {
            "plo": sections.plo,
            "methods": sections.methods,
            "results_conclusions": sections.results,
            "improvement_plan": sections.plan,
        }

        fs = result.final_scores.model_dump()
        section_scores = {
            "plo": fs["plo"],
            "methods": fs["methods"],
            "results": fs["results"],
            "plan": fs["plan"],
        }
        strict_all_pass = all(section_scores[k] == 1 for k in section_scores)

        pk = os.getenv("ROAR_PROMPT_SET", "B_ChainOfThought")
        mk = os.getenv("ROAR_MODEL_PROFILE", "MS3_DeepSeek_o4mini")
        strategy = os.getenv("ROAR_CLASSIFICATION_STRATEGY", "strict").strip().lower()

        payload = {
            "ok": True,
            "extracted": extracted,
            "sectionScores": section_scores,
            "weightedScore": result.weighted_score,
            "consistent": result.consistent,
            "iterations": result.iterations,
            "strictAllPass": strict_all_pass,
            "classificationStrategy": strategy,
            "roarPromptSet": pk,
            "roarModelProfile": mk,
            "evaluatorModel": args.evaluator_model_override
            or os.getenv("EVALUATOR_MODEL", ""),
            "verifierModel": args.verifier_model_override
            or os.getenv("VERIFIER_MODEL", ""),
        }
        sys.stdout.write(json.dumps(payload, ensure_ascii=True))
        sys.stdout.flush()
    except Exception as exc:
        debug_context = {
            "roarPromptSet": os.getenv("ROAR_PROMPT_SET", "B_ChainOfThought"),
            "roarModelProfile": os.getenv("ROAR_MODEL_PROFILE", "MS3_DeepSeek_o4mini"),
            "evaluatorApiBase": os.getenv("EVALUATOR_API_BASE", ""),
            "evaluatorApiVersion": os.getenv("EVALUATOR_API_VERSION", ""),
            "azureEndpoint": os.getenv("AZURE_ENDPOINT", ""),
            "azureApiVersion": os.getenv("AZURE_API_VERSION", ""),
            "verifierAzureEndpoint": os.getenv("VERIFIER_AZURE_ENDPOINT", ""),
            "verifierAzureApiVersion": os.getenv("VERIFIER_AZURE_API_VERSION", ""),
        }
        err = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "detail": traceback.format_exc(),
            "debugContext": debug_context,
        }
        sys.stdout.write(json.dumps(err, ensure_ascii=True))
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
