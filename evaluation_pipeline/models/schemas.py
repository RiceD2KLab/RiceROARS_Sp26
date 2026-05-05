from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input / extracted sections
# ---------------------------------------------------------------------------

class ROARSections(BaseModel):
    """The four structured sections extracted from a ROAR document."""

    plo: str = Field(..., description="Program Learning Objective text")
    methods: str = Field(..., description="Assessment methods description")
    results: str = Field(..., description="Reported results (numeric data + analysis)")
    plan: str = Field(..., description="Improvement plan text")


# ---------------------------------------------------------------------------
# Scoring / evaluator output  (Appendix A.1)
# ---------------------------------------------------------------------------

class SectionScores(BaseModel):
    """Binary scores (0 or 1) for each of the four ROAR sections."""

    plo: int = Field(..., ge=0, le=1)
    methods: int = Field(..., ge=0, le=1)
    results: int = Field(..., ge=0, le=1)
    plan: int = Field(..., ge=0, le=1)


class SectionReasoning(BaseModel):
    """Free-text reasoning for each section score."""

    plo: str
    methods: str
    results: str
    plan: str


class EvaluatorOutput(BaseModel):
    """Structured JSON produced by the LLaMA primary evaluator."""

    scores: SectionScores
    reasoning: SectionReasoning


# ---------------------------------------------------------------------------
# Verifier output  (Appendix A.2)
# ---------------------------------------------------------------------------

class Difference(BaseModel):
    """A single scoring disagreement identified by the verifier."""

    field: str
    llama_score: int = Field(..., ge=0, le=1)
    correct_score: int = Field(..., ge=0, le=1)
    reason: str


class VerifierOutput(BaseModel):
    """Structured JSON produced by the Qwen verifier."""

    verified_scores: SectionScores
    consistent: bool
    differences: List[Difference]


# ---------------------------------------------------------------------------
# Feedback / revised output  (Appendix A.3)
# ---------------------------------------------------------------------------

class Change(BaseModel):
    """A single score revision made after verifier feedback."""

    field: str
    old_score: int = Field(..., ge=0, le=1)
    new_score: int = Field(..., ge=0, le=1)
    reason: str


class FeedbackOutput(BaseModel):
    """Structured JSON produced after the LLaMA feedback loop iteration."""

    revised_scores: SectionScores
    changes_explained: List[Change]


# ---------------------------------------------------------------------------
# Audit trail (step-by-step internal evidence)
# ---------------------------------------------------------------------------

class AuditStep(BaseModel):
    """One recorded step inside the pipeline (scoring, verification, feedback)."""

    step_name: str
    scores: SectionScores
    reasoning: Optional[SectionReasoning] = None
    consistent: Optional[bool] = None
    differences: Optional[List[Difference]] = None
    changes: Optional[List[Change]] = None


# ---------------------------------------------------------------------------
# Final pipeline result
# ---------------------------------------------------------------------------

class PipelineResult(BaseModel):
    """Aggregated result returned by the full ROAR evaluation pipeline."""

    final_scores: SectionScores
    weighted_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: Optional[SectionReasoning] = None
    iterations: int = Field(..., ge=0)
    consistent: bool
    # Populated when the pipeline is run — contains every intermediate step
    sections: Optional[ROARSections] = None
    audit_trail: List[AuditStep] = Field(default_factory=list)
