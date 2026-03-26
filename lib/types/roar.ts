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
};
