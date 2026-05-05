"""
Primary evaluator – Step 2 of the ROAR pipeline.

Uses LLaMA (configured via EVALUATOR_MODEL) and the weighted audit prompt
(Appendix A.1) to produce an initial structured JSON score for each section.
"""

from __future__ import annotations

import json
from typing import Optional

import config
from models.schemas import EvaluatorOutput, ROARSections, SectionReasoning, SectionScores
from prompts.templates import SCORING_PROMPT
from utils.json_parser import RobustJsonOutputParser
from utils.llm_factory import Backend, build_llm


class ROAREvaluator:
    """
    Primary evaluator backed by LLaMA.

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
        scoring_prompt=None,
    ) -> None:
        self._model = model or config.EVALUATOR_MODEL
        self._backend: Backend = backend or config.EVALUATOR_BACKEND  # type: ignore[assignment]
        self._scoring_prompt = scoring_prompt or SCORING_PROMPT
        self._chain = self._build_chain()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, sections: ROARSections) -> EvaluatorOutput:
        """
        Score all four ROAR sections.

        Parameters
        ----------
        sections:
            Extracted ROAR sections.

        Returns
        -------
        EvaluatorOutput
            Structured scores and per-section reasoning.
        """
        raw: dict = self._chain.invoke(
            {
                "plo":     sections.plo,
                "methods": sections.methods,
                "results": sections.results,
                "plan":    sections.plan,
            }
        )
        return self._parse(raw)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_chain(self):
        if self._backend == "azure_openai":
            # o4-mini reasoning model: temperature=1.0, larger token budget.
            # Presence_penalty and top_p are not passed — unsupported by o-series.
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
        return self._scoring_prompt | llm | RobustJsonOutputParser()

    @staticmethod
    def _parse(raw: dict) -> EvaluatorOutput:
        scores_raw    = raw.get("scores", {})
        reasoning_raw = raw.get("reasoning", {})

        scores = SectionScores(
            plo=int(scores_raw.get("plo", 0)),
            methods=int(scores_raw.get("methods", 0)),
            results=int(scores_raw.get("results", 0)),
            plan=int(scores_raw.get("plan", 0)),
        )
        reasoning = SectionReasoning(
            plo=reasoning_raw.get("plo", ""),
            methods=reasoning_raw.get("methods", ""),
            results=reasoning_raw.get("results", ""),
            plan=reasoning_raw.get("plan", ""),
        )
        return EvaluatorOutput(scores=scores, reasoning=reasoning)

    def scores_to_json_str(self, output: EvaluatorOutput) -> str:
        """Serialise an EvaluatorOutput to a compact JSON string for prompt injection."""
        return json.dumps(
            {
                "scores":    output.scores.model_dump(),
                "reasoning": output.reasoning.model_dump(),
            },
            indent=2,
        )
