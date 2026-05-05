"""
prompt_sets.py — Three distinct prompt engineering strategies for the ROAR pipeline.

Each set contains a scorer prompt (LLaMA evaluator) and a verifier prompt
(GPT-5.4-mini).  The feedback prompt is shared across all sets since its
purpose (reconcile a specific disagreement) is the same regardless of strategy.

Strategy overview
─────────────────
SET A  Disqualifier-First (current production prompt)
       Leads with explicit FAIL conditions marked with ❌.
       The model checks disqualifiers before looking for passing evidence.
       Strength  : precise failure detection via named anti-patterns
       Weakness  : long prompt; model may over-apply rules

SET B  Chain-of-Thought Evidence Anchoring
       Asks the model to QUOTE verbatim text before scoring.
       Forces grounding — if evidence cannot be found, score = 0.
       Uses yes/no questions per criterion rather than free-form criteria.
       Strength  : transparent, traceable reasoning
       Weakness  : longer output; model must quote accurately

SET C  Few-Shot Exemplar Comparison
       Provides concrete PASS and FAIL examples for every section type.
       No abstract rules — the model learns by analogy.
       Strength  : intuitive for models trained on examples
       Weakness  : examples may not cover all edge cases
"""

from langchain_core.prompts import PromptTemplate

# ═══════════════════════════════════════════════════════════════════════════
# SET A — Disqualifier-First  (current production prompts)
# ═══════════════════════════════════════════════════════════════════════════

_SCORING_A = PromptTemplate(
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
    • ❌ States only "no changes" or "no changes will be made" with NO
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

_VERIFIER_A = PromptTemplate(
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


# ═══════════════════════════════════════════════════════════════════════════
# SET B — Chain-of-Thought Evidence Anchoring
#
# Strategy: force the model to QUOTE text before scoring.
# If it cannot find a quote, the section fails.  Yes/no questions replace
# abstract rules to reduce ambiguity.
# ═══════════════════════════════════════════════════════════════════════════

_SCORING_B = PromptTemplate(
    input_variables=["plo", "methods", "results", "plan"],
    template="""You are evaluating a ROAR (Rice Outcome Assessment Report).
Follow these EXACT steps for each section. Do not skip steps.

STEP 1 — QUOTE THE EVIDENCE
For each section, copy the single most relevant sentence verbatim from the text.
If the section is blank or absent, write "NOT FOUND".

STEP 2 — ANSWER BINARY QUESTIONS
Based ONLY on your quoted sentence, answer yes or no:

PLO (weight 25%):
  Q: Does the quoted text describe what students will achieve, learn, or be able to do?
  (yes → score 1, no → score 0)

METHODS (weight 30%) — must answer YES to BOTH:
  Q1: Does the quoted text name a direct assessment of student work?
      (Dissertation review, oral exam, presentation, committee evaluation qualify.
       Course letter grades alone do NOT qualify.)
  Q2: Does the quoted text explicitly mention a rubric, criteria, or scoring scale?
      (Must be stated — "committee determines pass/fail" without criteria = no.)
  (yes to both → score 1, no to either → score 0)

RESULTS (weight 30%) — must answer YES to BOTH:
  Q1: Does the quoted text contain actual numbers — percentages, averages, or counts?
      (Individual letter grades like "A, B, C" without aggregation = no.)
  Q2: Does the quoted text compare results to a threshold, standard, or benchmark?
      ("All students passed" without a defined standard = no.)
  (yes to both → score 1, no to either → score 0)

PLAN (weight 15%):
  Q: Does the quoted text describe a specific action for the next cycle?
     Acceptable: "We will continue X", "We will add Y", "No changes; we will continue Z."
     NOT acceptable: "N/A", "None", "No changes" with nothing else.
  (yes → score 1, no → score 0)

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
    "plo": "QUOTE: '<verbatim quote>' | ANSWER: yes/no | SCORE: <0 or 1>",
    "methods": "QUOTE: '<verbatim quote>' | Q1: yes/no | Q2: yes/no | SCORE: <0 or 1>",
    "results": "QUOTE: '<verbatim quote>' | Q1: yes/no | Q2: yes/no | SCORE: <0 or 1>",
    "plan": "QUOTE: '<verbatim quote>' | ANSWER: yes/no | SCORE: <0 or 1>"
  }}
}}

Replace every <SCORE> with 0 or 1 (integer, not string).""",
)

_VERIFIER_B = PromptTemplate(
    input_variables=["plo", "methods", "results", "plan", "llama_output"],
    template="""You are verifying ROAR assessment scores. The primary evaluator quoted text
and answered yes/no questions to produce each score. Your job is to check the logic.

SCORING SCALE:  1 = PASS  |  0 = FAIL

CHECK THESE SPECIFIC LOGIC ERRORS:

METHODS: If the evaluator's quoted text does NOT contain a rubric/criteria word
  (rubric, criteria, scale, graded, evaluation form, criterion), the score
  MUST be 0, regardless of other reasoning.

RESULTS: If the evaluator's quoted text does NOT contain a number (%, score,
  average, rate, count), the score MUST be 0.

PLAN: If the quoted text is "N/A", "None", "No changes" with nothing else,
  the score MUST be 0.

AGREE with the evaluator unless you can identify a specific logic error above.
If consistent is true, "differences" MUST be an empty array [].

ORIGINAL ROAR SECTIONS:
PLO: {plo}
Methods: {methods}
Results: {results}
Plan: {plan}

PRIMARY EVALUATOR OUTPUT (includes quotes and answers):
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
      "reason": "<state the logic error found>"
    }}
  ]
}}

Replace every <SCORE> with 0 or 1 (integer, not string).
Replace <BOOL> with true or false (no quotes).""",
)


# ═══════════════════════════════════════════════════════════════════════════
# SET C — Few-Shot Exemplar Comparison
#
# Strategy: no abstract rules — show the model concrete PASS and FAIL
# examples for each section type and ask it to compare and classify.
# ═══════════════════════════════════════════════════════════════════════════

_SCORING_C = PromptTemplate(
    input_variables=["plo", "methods", "results", "plan"],
    template="""You are scoring a Rice University ROAR (Outcome Assessment Report).
Use the examples below to calibrate your scores. Compare the INPUT to the examples.

SCORING SCALE:  1 = PASS  |  0 = FAIL

────────────────────────────────────────────────────────────────────────
PLO — Program Learning Outcome (weight 25%)

PASS examples (score = 1):
  • "Students will develop skills to pursue professional endeavors within and
    outside the academy successfully."
  • "Students will demonstrate a comprehensive understanding of the history and
    theory of the discipline."
  • "Students will communicate research effectively by writing clearly and cogently."
  • "Students will execute and present original research in their discipline."

FAIL examples (score = 0):
  • "The program will provide courses in research methods." [describes program, not students]
  • "PLO 3: Dissertation completion." [no learner outcome described]
  • [blank or absent]

────────────────────────────────────────────────────────────────────────
METHODS — Assessment Methods (weight 30%)

PASS examples (score = 1) — must show BOTH a direct measure AND a rubric:
  • "Students are evaluated by their dissertation committee using a 5-criterion
    rubric graded 1–5 on clarity, originality, methodology, analysis, and presentation."
  • "A mock job talk is scored by faculty using an oral rubric covering research
    articulation, Q&A performance, and visual quality."
  • "Each syllabus is evaluated by the DGS and dissertation chair on whether it
    identifies an important topic, uses rigorous readings, and develops engaging exercises."

FAIL examples (score = 0) — missing direct measure OR rubric:
  • "Students are assessed through course grades in CHEM 211 and CHEM 365."
    [course grades only — indirect; no rubric]
  • "Faculty observe student dissertations and determine whether standards are met."
    [no rubric or criteria defined]
  • "Direct observation of student work in the form of dissertations."
    [no rubric; no criteria]

────────────────────────────────────────────────────────────────────────
RESULTS — Assessment Results (weight 30%)

PASS examples (score = 1) — must show BOTH numbers AND comparison to benchmark:
  • "The cumulative score across four presentations was 4.2 (84%). All outcomes
    exceed the threshold of 80% of students who achieve a score of 3.5 or better."
  • "Average score across all syllabi: 4.6/5 (92%), above the 80% benchmark."
  • "78% of students scored above the 3.5 threshold (target: 80%). One student
    did not meet expectations."

FAIL examples (score = 0) — no numbers OR no benchmark:
  • "All dissertations addressed the field in substantive ways."
    [qualitative — no numbers]
  • "This learning objective is being accomplished by PhD students."
    [qualitative — no numbers, no benchmark]
  • "1 student C+; 2 students above B; 3 students A."
    [individual grades — no aggregation, no benchmark comparison]
  • "All fourth year students have passed this requirement."
    [no numbers, no benchmark defined]

────────────────────────────────────────────────────────────────────────
IMPROVEMENT PLAN — Next cycle plan (weight 15%)

PASS examples (score = 1):
  • "In AY 24-25 we will continue to hold mock job talks evaluated by committees."
  • "Students at risk will receive additional mentoring. Impact assessed next year."
  • "No changes are needed. We will continue this evaluation process next cycle."

FAIL examples (score = 0):
  • "N/A"
  • "None"
  • "No changes will be made." [no description of what continues]
  • "We are satisfied with these results." [no next step]
  • "Based on these results, no changes will be made." [no continuation plan]

────────────────────────────────────────────────────────────────────────

Compare the following ROAR sections to the examples above and assign scores:

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
    "plo": "<which example this most resembles and why>",
    "methods": "<which example this most resembles and why — name missing element if 0>",
    "results": "<which example this most resembles and why — name missing element if 0>",
    "plan": "<which example this most resembles and why>"
  }}
}}

Replace every <SCORE> with 0 or 1 (integer, not string).""",
)

_VERIFIER_C = PromptTemplate(
    input_variables=["plo", "methods", "results", "plan", "llama_output"],
    template="""You are verifying ROAR assessment scores. Use the examples below
to check whether the primary evaluator's scores match the correct standard.

SCORING SCALE:  1 = PASS  |  0 = FAIL

KEY PASS/FAIL SIGNALS:

METHODS = 0 (FAIL) when:  → Only course grades mentioned (no rubric/criteria)
  → "Direct observation" or "committee determines" without stated criteria
METHODS = 1 (PASS) when:  → Both a direct measure AND rubric/criteria are named

RESULTS = 0 (FAIL) when:  → Only qualitative ("all did well", "most passed")
  → Only individual letter grades without aggregation or benchmark comparison
RESULTS = 1 (PASS) when:  → Actual % or scores AND compared to a threshold

PLAN = 0 (FAIL) when:  → "N/A", "None", or bare "no changes" with nothing else
PLAN = 1 (PASS) when:  → Describes a specific action OR "no changes + continue X"

Default: AGREE with the evaluator. Only correct when a key signal above is violated.
If consistent is true, "differences" MUST be an empty array [].

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
      "reason": "<cite the key signal that was violated>"
    }}
  ]
}}

Replace every <SCORE> with 0 or 1 (integer, not string).
Replace <BOOL> with true or false (no quotes).""",
)


# ═══════════════════════════════════════════════════════════════════════════
# Shared feedback prompt (same for all sets — its task is always identical)
# ═══════════════════════════════════════════════════════════════════════════

from prompts.templates import FEEDBACK_PROMPT as _FEEDBACK_SHARED

# ── Public API ───────────────────────────────────────────────────────────────

PROMPT_SETS: dict[str, dict] = {
    "A_DisqualifierFirst": {
        "label":       "Set A — Disqualifier-First",
        "description": "Leads with explicit FAIL conditions (❌). Check disqualifiers before evidence.",
        "scoring":     _SCORING_A,
        "verifier":    _VERIFIER_A,
        "feedback":    _FEEDBACK_SHARED,
    },
    "B_ChainOfThought": {
        "label":       "Set B — Chain-of-Thought Evidence Anchoring",
        "description": "Quote verbatim text first; answer yes/no questions; score from answers.",
        "scoring":     _SCORING_B,
        "verifier":    _VERIFIER_B,
        "feedback":    _FEEDBACK_SHARED,
    },
    "C_FewShot": {
        "label":       "Set C — Few-Shot Exemplar Comparison",
        "description": "Calibrate against concrete PASS/FAIL examples; no abstract rules.",
        "scoring":     _SCORING_C,
        "verifier":    _VERIFIER_C,
        "feedback":    _FEEDBACK_SHARED,
    },
}
