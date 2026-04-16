"""
LangChain PromptTemplates for every stage of the ROAR pipeline.

Prompt engineering notes
------------------------
1.  JSON templates use descriptive placeholders (<SCORE>, <0_OR_1>, etc.)
    instead of literal 0 values.  Using `0` as a placeholder caused GPT
    models to copy it literally — treating "0" as the "correct" answer.

2.  The verifier prompt explicitly defines the scoring scale (1=PASS, 0=FAIL)
    and instructs the model to AGREE by default.  A previous version caused
    GPT-5.4-mini to invert all scores because it misread the placeholder.

3.  The scoring prompt clarifies edge cases that caused systematic errors:
    - PLO: broad learner outcomes are acceptable (score 1 unless absent).
    - Plan: "no changes needed + we will continue X" counts as actionable.

4.  All prompts use "Output STRICT JSON ONLY" with the explicit note
    "do not add any text before or after the JSON object" to reduce
    preamble pollution that causes JSON parse failures.
"""

from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# A.0  Section extraction prompt  (pre-processing step)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["document"],
    template="""You are an expert at analysing ROAR (Rice Outcome Assessment Report) documents.
Extract the four key sections from the document below.

DOCUMENT:
{document}

Rules:
- Extract text verbatim where possible.
- If a section is absent, use the string "Not provided".

Output STRICT JSON ONLY — do not add any text before or after the JSON object:
{{
  "plo": "<extracted PLO text>",
  "methods": "<extracted assessment methods text>",
  "results": "<extracted results text>",
  "plan": "<extracted improvement plan text>"
}}""",
)

# ---------------------------------------------------------------------------
# A.1  Scoring prompt  (primary evaluator – LLaMA)
# ---------------------------------------------------------------------------

SCORING_PROMPT = PromptTemplate(
    input_variables=["plo", "methods", "results", "plan"],
    template="""You are evaluating a Rice University ROAR (Outcome Assessment Report).
For each section, check the DISQUALIFIERS first. If any disqualifier is met, score = 0.
Only award score = 1 if you can quote explicit evidence satisfying ALL requirements.

SCORING SCALE:  1 = PASS  |  0 = FAIL

═══════════════════════════════════════════════════════════════════════
PLO — Program Learning Outcome (weight 25%)

  DISQUALIFIERS → score = 0 if:
    • The section is absent or blank.
    • The text describes ONLY program activities, not student learning.

  PASS → score = 1 if:
    • The text describes what students will ACHIEVE, KNOW, or BE ABLE TO DO.
    • Broad outcomes are acceptable ("students will develop X", "students will
      demonstrate Y"). The outcome does not need to be narrowly worded.

═══════════════════════════════════════════════════════════════════════
METHODS — Assessment Methods (weight 30%)

  DISQUALIFIERS → score = 0 if ANY of these are true:
    • ❌ The ONLY assessment tool mentioned is course grades, GPA, or
         degree completion rates (these are INDIRECT measures — not direct).
    • ❌ There is NO mention of a rubric, scoring criteria, or evaluation
         scale anywhere in the section. "Direct observation" without criteria
         does NOT count.
    • ❌ The section is absent or blank.

  PASS → score = 1 only if BOTH of these are explicitly stated:
    ✓ A DIRECT measure: dissertation review, oral exam, presentation,
      project, rubric-graded assignment, committee evaluation, etc.
    ✓ A RUBRIC or CRITERIA: a scoring scale, criteria checklist, or
      explicit evaluation form with defined standards.

  EVIDENCE REQUIRED: Quote the exact text showing the rubric/criteria.
  If you cannot find a rubric in the text, score = 0.

═══════════════════════════════════════════════════════════════════════
RESULTS — Assessment Results (weight 30%)

  DISQUALIFIERS → score = 0 if ANY of these are true:
    • ❌ The results are ONLY qualitative ("students did well", "most met
         the goal", "all dissertations were substantive", "students exceeded
         expectations") with NO numbers whatsoever.
    • ❌ The results list only individual letter grades (A, B, C) without
         any aggregation (averages, percentages, pass rates).
    • ❌ There is NO comparison to a threshold, benchmark, or standard.
    • ❌ The section is absent or blank.

  PASS → score = 1 only if BOTH of these are present:
    ✓ NUMBERS: actual percentages, averages, scores, or rates.
    ✓ ANALYSIS: comparison to a stated benchmark or threshold.

  EVIDENCE REQUIRED: Quote the exact numbers from the text.
  If you cannot find numbers, score = 0.

═══════════════════════════════════════════════════════════════════════
IMPROVEMENT PLAN — Next cycle plan (weight 15%)

  DISQUALIFIERS → score = 0 if ANY of these are true:
    • ❌ The section is absent, blank, "N/A", or "None".
    • ❌ States only "no changes" or "no changes are needed" with NO
         description of what will happen or continue next cycle.

  PASS → score = 1 if:
    ✓ A specific next step is described, OR
    ✓ "No changes are needed" AND the text also describes what
      practice will continue (e.g., "We will continue X in AY 24-25").

═══════════════════════════════════════════════════════════════════════

INPUT:
PLO: {plo}
Methods: {methods}
Results: {results}
Plan: {plan}

Output STRICT JSON ONLY — do not add any text before or after the JSON object:
{{
  "scores": {{
    "plo": <SCORE>,
    "methods": <SCORE>,
    "results": <SCORE>,
    "plan": <SCORE>
  }},
  "reasoning": {{
    "plo": "<quote the student outcome text, or state why it fails>",
    "methods": "<quote the rubric/criteria text — or name the missing element>",
    "results": "<quote the specific numbers and threshold — or name the missing element>",
    "plan": "<quote the next-step text — or state why it fails>"
  }}
}}

Replace every <SCORE> with 0 or 1 (integer, not string).""",
)

# ---------------------------------------------------------------------------
# A.2  Verifier prompt  (GPT-5.4-mini)
# ---------------------------------------------------------------------------

VERIFIER_PROMPT = PromptTemplate(
    input_variables=["plo", "methods", "results", "plan", "llama_output"],
    template="""You are a QUALITY CONTROL verifier for ROAR assessment scores.
Your job is to check whether the primary evaluator followed the scoring rules correctly.

SCORING SCALE:  1 = PASS (meets requirements)  |  0 = FAIL (does not meet requirements)

SCORING RULES YOU MUST ENFORCE:

METHODS (score = 0) if:
  • Course grades, GPA, or degree completion are the ONLY assessment tool.
    (These are INDIRECT measures. They do NOT qualify as direct assessment.)
  • NO rubric, scoring criteria, or evaluation scale is explicitly mentioned.
    ("Assessed by committee" without criteria = 0.)

RESULTS (score = 0) if:
  • Results are ONLY qualitative ("all students passed", "most met the goal",
    "students did well") with NO actual numbers, percentages, or scores.
  • Only individual letter grades (A, B, C) are listed with NO aggregation,
    NO average, and NO comparison to a threshold or benchmark.

PLAN (score = 0) if:
  • Section is absent, "N/A", or "None".
  • States ONLY "no changes" or "no changes will be made" with NO description
    of what practice will continue next cycle.
  PLAN passes (score = 1) if: "no changes" + describes what will continue.

IMPORTANT DEFAULTS:
- Agree with the evaluator UNLESS a specific rule above is violated.
- Do NOT override a 0 just because a section seems "good enough."
- Do NOT override a 1 without citing a specific rule violation.
- If consistent is true, "differences" MUST be an empty array [].

ORIGINAL ROAR SECTIONS:
PLO: {plo}
Methods: {methods}
Results: {results}
Plan: {plan}

PRIMARY EVALUATOR SCORES AND REASONING:
{llama_output}

Output STRICT JSON ONLY — do not add any text before or after the JSON object:
{{
  "verified_scores": {{
    "plo": <SCORE>,
    "methods": <SCORE>,
    "results": <SCORE>,
    "plan": <SCORE>
  }},
  "consistent": <BOOL>,
  "differences": [
    {{
      "field": "<plo|methods|results|plan>",
      "llama_score": <SCORE>,
      "correct_score": <SCORE>,
      "reason": "<cite the specific rule that was violated>"
    }}
  ]
}}

Replace every <SCORE> with 0 or 1 (integer, not string).
Replace <BOOL> with true or false (no quotes).""",
)

# ---------------------------------------------------------------------------
# A.3  Feedback prompt  (primary evaluator – LLaMA, reconsideration step)
# ---------------------------------------------------------------------------

FEEDBACK_PROMPT = PromptTemplate(
    input_variables=["previous_scores", "differences"],
    template="""You previously evaluated a ROAR report.
A quality-control verifier found specific errors in your scoring.
Review each difference carefully and decide whether to update your score.

SCORING SCALE:  1 = PASS  |  0 = FAIL

YOUR PREVIOUS SCORES:
{previous_scores}

VERIFIER FEEDBACK (fields where your score may be wrong):
{differences}

INSTRUCTIONS:
- For each item in the verifier feedback, decide whether the correction is justified.
- If the verifier gives a convincing reason, update the score.
- If you believe your original score was correct, keep it.
- Explain every change (or non-change) concisely.

Output STRICT JSON ONLY — do not add any text before or after the JSON object:
{{
  "revised_scores": {{
    "plo": <SCORE>,
    "methods": <SCORE>,
    "results": <SCORE>,
    "plan": <SCORE>
  }},
  "changes_explained": [
    {{
      "field": "<plo|methods|results|plan>",
      "old_score": <SCORE>,
      "new_score": <SCORE>,
      "reason": "<why you changed or kept the score>"
    }}
  ]
}}

Replace every <SCORE> with 0 or 1 (integer, not string).""",
)
