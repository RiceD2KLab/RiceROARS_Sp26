"""
ROAR pipeline orchestrator.

Wires together all four stages defined in the paper:
  1. Section Extraction
  2. Initial Scoring   (LLaMA primary evaluator)
  3. Verification      (Qwen3.5-0.8B verifier via OpenAI-compatible API)
  4. Feedback Loop     (LLaMA, up to max_iter iterations)

and computes the final weighted quality score.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import config
from models.schemas import AuditStep, PipelineResult, ROARSections, SectionScores
from pipeline.extractor import SectionExtractor
from pipeline.evaluator import ROAREvaluator
from pipeline.verifier import ROARVerifier
from pipeline.feedback import FeedbackModule
from utils.llm_factory import Backend

logger = logging.getLogger(__name__)


class ROARPipeline:
    """
    End-to-end ROAR quality assessment pipeline.

    Parameters
    ----------
    evaluator_model:
        Model identifier for LLaMA (primary evaluator + feedback).
        Defaults to ``config.EVALUATOR_MODEL``.
    verifier_model:
        Model identifier for Qwen3.5 (verifier).
        Defaults to ``config.VERIFIER_MODEL``.
    evaluator_backend:
        Backend for the evaluator: ``"ollama"`` or ``"openai"``.
        Defaults to ``config.EVALUATOR_BACKEND``.
    verifier_backend:
        Backend for the verifier: ``"ollama"`` or ``"openai"``.
        Defaults to ``config.VERIFIER_BACKEND``.
    max_iterations:
        Maximum number of feedback-loop iterations before the pipeline
        terminates regardless of consistency.  Defaults to ``config.MAX_ITERATIONS``.
    """

    def __init__(
        self,
        evaluator_model: Optional[str] = None,
        verifier_model: Optional[str] = None,
        evaluator_backend: Optional[Backend] = None,
        verifier_backend: Optional[Backend] = None,
        max_iterations: Optional[int] = None,
        prompt_set: Optional[dict] = None,
    ) -> None:
        self._max_iter = max_iterations if max_iterations is not None else config.MAX_ITERATIONS

        ev_model   = evaluator_model or config.EVALUATOR_MODEL
        ve_model   = verifier_model  or config.VERIFIER_MODEL
        ev_backend: Backend = evaluator_backend or config.EVALUATOR_BACKEND  # type: ignore[assignment]
        ve_backend: Backend = verifier_backend  or config.VERIFIER_BACKEND   # type: ignore[assignment]

        ps = prompt_set or {}
        self._extractor = SectionExtractor(model=ev_model, backend=ev_backend)
        self._evaluator = ROAREvaluator(
            model=ev_model, backend=ev_backend,
            scoring_prompt=ps.get("scoring"),
        )
        self._verifier = ROARVerifier(
            model=ve_model, backend=ve_backend,
            verifier_prompt=ps.get("verifier"),
        )
        self._feedback = FeedbackModule(
            model=ev_model, backend=ev_backend,
            feedback_prompt=ps.get("feedback"),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        document: str = "",
        *,
        pre_sections: Optional[ROARSections] = None,
    ) -> PipelineResult:
        """
        Run the full pipeline on a ROAR document.

        Parameters
        ----------
        document:
            Raw text of the ROAR.  Ignored when *pre_sections* is supplied.
        pre_sections:
            Already-extracted :class:`ROARSections` (e.g. from
            ``SectionExtractor.extract_from_docx``).  When provided, Step 1
            (section extraction) is skipped entirely.

        Returns
        -------
        PipelineResult
            Final scores, weighted quality score, iteration count, and
            consistency flag.
        """
        # ---------------------------------------------------------------
        # Step 1 – Section Extraction  (skip if sections already provided)
        # ---------------------------------------------------------------
        if pre_sections is not None:
            logger.info("[Step 1] Using pre-extracted sections (skipping LLM extraction).")
            sections: ROARSections = pre_sections
        else:
            logger.info("[Step 1] Extracting sections from document…")
            sections = self._extractor.extract(document)
        logger.debug(
            "Extracted sections:\n  PLO: %.80s…\n  Methods: %.80s…\n"
            "  Results: %.80s…\n  Plan: %.80s…",
            sections.plo, sections.methods, sections.results, sections.plan,
        )

        # Audit trail — records every intermediate step for reporting
        audit: list[AuditStep] = []

        # ---------------------------------------------------------------
        # Step 2 – Initial Scoring
        # ---------------------------------------------------------------
        logger.info("[Step 2] Running initial scoring (LLaMA evaluator)…")
        evaluator_output = self._evaluator.evaluate(sections)
        logger.info("Initial scores: %s", evaluator_output.scores.model_dump())
        audit.append(AuditStep(
            step_name="Step 2 – Initial Scoring (LLaMA Evaluator)",
            scores=evaluator_output.scores,
            reasoning=evaluator_output.reasoning,
        ))

        # ---------------------------------------------------------------
        # Steps 3 + 4 + 5 – Verify → consistency check → feedback loop
        # ---------------------------------------------------------------
        iteration  = 0
        consistent = False

        while iteration < self._max_iter:
            # Step 3 – Verification
            logger.info(
                "[Step 3] Verifying scores (GPT Verifier) – iteration %d…",
                iteration + 1,
            )
            verifier_output = self._verifier.verify(sections, evaluator_output)
            logger.info(
                "Verification: consistent=%s, differences=%d",
                verifier_output.consistent,
                len(verifier_output.differences),
            )
            audit.append(AuditStep(
                step_name=f"Step 3 – Verification (GPT Verifier) iteration {iteration + 1}",
                scores=verifier_output.verified_scores,
                consistent=verifier_output.consistent,
                differences=verifier_output.differences,
            ))

            # Step 4 – Consistency check
            if verifier_output.consistent:
                consistent = True
                logger.info("[Step 4] Scores are consistent.  Pipeline converged.")
                break

            # Step 5 – Feedback loop
            iteration += 1
            logger.info(
                "[Step 5] Inconsistency detected – applying feedback "
                "(iteration %d / %d)…",
                iteration, self._max_iter,
            )
            feedback_output  = self._feedback.apply_feedback(evaluator_output, verifier_output)
            evaluator_output = self._feedback.feedback_to_evaluator_output(
                feedback_output, evaluator_output
            )
            logger.info("Revised scores: %s", evaluator_output.scores.model_dump())
            audit.append(AuditStep(
                step_name=f"Step 5 – Feedback / Revision (LLaMA) iteration {iteration}",
                scores=evaluator_output.scores,
                reasoning=evaluator_output.reasoning,
                changes=feedback_output.changes_explained,
            ))

        if not consistent:
            logger.warning(
                "Pipeline reached max_iter=%d without convergence.  "
                "Returning last revised scores.",
                self._max_iter,
            )

        # ---------------------------------------------------------------
        # Compute weighted score
        # ---------------------------------------------------------------
        weighted = self._weighted_score(evaluator_output.scores)
        logger.info("Final weighted score: %.4f", weighted)

        return PipelineResult(
            final_scores=evaluator_output.scores,
            weighted_score=weighted,
            reasoning=evaluator_output.reasoning,
            iterations=iteration,
            consistent=consistent,
            sections=sections,
            audit_trail=audit,
        )

    def run_from_pdf(self, path: str | Path) -> PipelineResult:
        """Convenience wrapper: load a PDF and run the pipeline."""
        text = SectionExtractor.load_pdf(path)
        return self.run(text)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_score(scores: SectionScores) -> float:
        total = 0.0
        for field, weight in config.WEIGHTS.items():
            total += getattr(scores, field) * weight
        return round(total, 4)
