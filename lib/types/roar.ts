/**
 * Shared types for ROAR analysis API and UI.
 * Single contract for mock and real pipeline responses.
 */

export type RoarFlag = {
  id: string;
  category: "structural" | "semantic";
  code: string;
  message: string;
  section?: "plo" | "methods" | "results" | "improvement_plan";
};

export type ExtractedSections = {
  department?: string;
  plo?: string;
  methods?: string;
  results_conclusions?: string;
  improvement_plan?: string;
};

export type SectionScores = {
  plo: 0 | 1;
  methods: 0 | 1;
  results: 0 | 1;
  plan: 0 | 1;
};

export type RoarAnalysisResult = {
  filename: string;
  extracted?: ExtractedSections;
  flags: RoarFlag[];
  sectionScores?: SectionScores;
  /** Weighted quality score from the evaluation pipeline (0–1). */
  weightedScore?: number;
  /** Whether evaluator and verifier agreed within the feedback loop. */
  pipelineConsistent?: boolean;
  /** Feedback-loop iterations completed. */
  pipelineIterations?: number;
  /** Set when the LLM pipeline was skipped or failed (e.g. missing API keys). */
  analysisNote?: string;
  /** True when all four section scores are 1 (team “strict” good ROAR). */
  strictAllPass?: boolean;
  /** strict = all sections must pass; weighted = legacy weighted threshold. */
  classificationStrategy?: "strict" | "weighted";
  /** Prompt set key from evaluation_pipeline (e.g. B_ChainOfThought). */
  roarPromptSet?: string;
  /** Model profile key (e.g. MS3_DeepSeek_o1mini). */
  roarModelProfile?: string;
  /** Evaluator deployment/model used for this request. */
  evaluatorModel?: string;
  /** Verifier deployment/model used for this request. */
  verifierModel?: string;
};
