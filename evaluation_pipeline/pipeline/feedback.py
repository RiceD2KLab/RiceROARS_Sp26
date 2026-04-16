"""
Feedback module – Step 5 of the ROAR pipeline.

Uses LLaMA and the feedback prompt (Appendix A.3) to help the primary
evaluator reconsider its predictions based on disagreements identified by
the Qwen verifier.
"""

from __future__ import annotations

import json
from typing import List, Optional

import config
from models.schemas import (
    Change,
    EvaluatorOutput,
    FeedbackOutput,
    SectionReasoning,
    SectionScores,
    VerifierOutput,
)
from prompts.templates import FEEDBACK_PROMPT
from utils.json_parser import RobustJsonOutputParser
from utils.llm_factory import Backend, build_llm


class FeedbackModule:
    """
    Feedback loop module backed by LLaMA.

    Parameters
    ----------
    model:
        Model identifier.  Defaults to ``config.EVALUATOR_MODEL``.
    backend:
        ``"ollama"`` or ``"openai"``.  Defaults to ``config.EVALUATOR_BACKEND``.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        backend: Optional[Backend] = None,
        feedback_prompt=None,
    ) -> None:
        self._model = model or config.EVALUATOR_MODEL
        self._backend: Backend = backend or config.EVALUATOR_BACKEND  # type: ignore[assignment]
        self._feedback_prompt = feedback_prompt or FEEDBACK_PROMPT
        self._chain = self._build_chain()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_feedback(
        self,
        evaluator_output: EvaluatorOutput,
        verifier_output: VerifierOutput,
    ) -> FeedbackOutput:
        """
        Ask LLaMA to revise its scores using the verifier's feedback.

        Parameters
        ----------
        evaluator_output:
            The most recent primary evaluator output.
        verifier_output:
            The verifier output that contains the list of differences.

        Returns
        -------
        FeedbackOutput
            Revised scores and explanations of each change.
        """
        previous_scores_str = json.dumps(
            evaluator_output.scores.model_dump(), indent=2
        )
        differences_str = json.dumps(
            [d.model_dump() for d in verifier_output.differences], indent=2
        )

        raw: dict = self._chain.invoke(
            {
                "previous_scores": previous_scores_str,
                "differences":     differences_str,
            }
        )
        return self._parse(raw, evaluator_output.scores)

    def feedback_to_evaluator_output(
        self,
        feedback: FeedbackOutput,
        previous: EvaluatorOutput,
    ) -> EvaluatorOutput:
        """
        Convert a FeedbackOutput back into an EvaluatorOutput so the
        verifier can re-check it in the next iteration.
        """
        prev_reasoning = previous.reasoning.model_dump()
        for change in feedback.changes_explained:
            field = change.field.lower()
            if field in prev_reasoning:
                prev_reasoning[field] = change.reason

        return EvaluatorOutput(
            scores=feedback.revised_scores,
            reasoning=SectionReasoning(**prev_reasoning),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_chain(self):
        if self._backend == "azure_openai":
            llm = build_llm(
                self._model,
                self._backend,
                azure_temperature=config.EVALUATOR_AZURE_TEMPERATURE,
                azure_max_tokens=config.EVALUATOR_AZURE_MAX_TOKENS,
            )
        else:
            llm = build_llm(
                self._model,
                self._backend,
                openai_base_url=config.EVALUATOR_API_BASE,
                openai_api_key=config.EVALUATOR_API_KEY,
                openai_temperature=config.EVALUATOR_TEMPERATURE,
                openai_top_p=config.EVALUATOR_TOP_P,
                openai_presence_penalty=config.EVALUATOR_PRESENCE_PENALTY,
                openai_max_tokens=config.EVALUATOR_MAX_TOKENS,
                openai_default_query={"api-version": config.EVALUATOR_API_VERSION},
            )
        return self._feedback_prompt | llm | RobustJsonOutputParser()

    @staticmethod
    def _parse(raw: dict, previous_scores: SectionScores) -> FeedbackOutput:
        rs_raw = raw.get("revised_scores", {})
        revised_scores = SectionScores(
            plo=int(rs_raw.get("plo",     previous_scores.plo)),
            methods=int(rs_raw.get("methods", previous_scores.methods)),
            results=int(rs_raw.get("results", previous_scores.results)),
            plan=int(rs_raw.get("plan",    previous_scores.plan)),
        )

        changes_raw: list = raw.get("changes_explained", [])
        changes: List[Change] = [
            Change(
                field=str(c.get("field", "")),
                old_score=int(c.get("old_score", 0)),
                new_score=int(c.get("new_score", 0)),
                reason=str(c.get("reason", "")),
            )
            for c in changes_raw
        ]

        return FeedbackOutput(revised_scores=revised_scores, changes_explained=changes)
