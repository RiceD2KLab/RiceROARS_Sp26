# ROARS — Rice Outcomes Assessment Reporting Screening

A pre-screening system that ingests **Nuventive** (or legacy) ROAR submissions—PDF or Word—and outputs **interpretable flags** for potentially problematic reports, so Institutional Effectiveness (IE) reviewers can focus attention where it is most needed.

---

## Table of Contents

- [What Are ROARs?](#what-are-roars)
- [Project Background and Significance](#project-background-and-significance)
- [Project Objectives](#project-objectives)
- [System Overview](#system-overview)
- [Flag Types: What Makes a "Bad" ROAR?](#flag-types-what-makes-a-bad-roar)
- [Data and Reference Materials](#data-and-reference-materials)
- [Technical Pipeline (Planned / In Progress)](#technical-pipeline-planned--in-progress)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Running with Docker](#running-with-docker)

---

## What Are ROARs?

**ROAR** = **R**ice **O**utcomes **A**ssessment **R**eporting. Each academic degree program at Rice University submits annual reports describing:

- **Program Learning Outcomes (PLOs)** — what students should know or be able to do upon completion
- **Assessment methods** — how the program directly assessed student work (rubrics, exams, portfolios, etc.)
- **Results and conclusions** — aggregated results, comparison to targets, and interpretation
- **Improvement plan** — actions to be taken when targets are not met (or to sustain improvement)

Reporting is aligned with **SACSCOC** (Southern Association of Colleges and Schools Commission on Colleges) expectations for continuous improvement and institutional effectiveness. As of **Spring 2026**, programs submit ROARs and Follow-Up reports via the online platform **Nuventive Improve**; legacy submissions exist as Word documents or exported PDFs.

---

## Project Background and Significance

- **Sponsors:** Rice University's Office of Information Technology (OIT) and Office of Institutional Effectiveness (OIE), within the IDEAS (Institutional Data, Evaluation, Analytics and Strategy) Office.
- **Problem:** Review of ROAR submissions is currently manual. Staff must verify alignment between PLOs and methods, consistency of results with methods, and quality of analysis and improvement plans. Volume and complexity lead to long review timelines and delayed feedback.
- **Goal:** Build an automated **"first set of eyes"** that flags major structural and semantic issues so reviewers can prioritize which submissions need closer attention. The system is **decision-support**, not a replacement for human judgment.
- **Long-term:** The tool may be extended so programs can self-screen before submitting to OIE, or to support similar assessment workflows in non-academic units.

---

## Project Objectives

1. **Structured data extraction** — Parse ROAR documents (PDF/Word) and extract key fields: PLOs, assessment methods, results/analysis, and improvement plans.
2. **Automated completeness and compliance checks** — Detect missing fields, formatting issues, duplicated content, and other structural problems.
3. **PLO–program correlation analysis** — Identify which programs or PLO characteristics tend to be associated with lower-quality submissions.
4. **Semantic evaluation** — Assess alignment between PLOs and methods, and between methods and reported results.
5. **Flagging system** — Produce interpretable flags (e.g., vague analysis, misaligned assessment, missing performance targets) to prioritize manual review.
6. **Validation framework** — Test the system on labeled and mock ROAR data to ensure reliable identification of major issues.
7. **Reviewer-friendly output** — Design output (e.g., summary report or categorized flags) suitable for OIE workflows and, eventually, a reviewer dashboard.

---

## System Overview

```
┌─────────────────────┐     ┌──────────────────────────────┐     ┌─────────────────────────┐
│  Nuventive PDFs /    │     │  Parse → Extract → Evaluate   │     │  Flags + optional       │
│  Legacy ROAR .docx   │ ──► │  (rules + optional ML/LLM)    │ ──► │  section-level scores   │
└─────────────────────┘     └──────────────────────────────┘     └─────────────────────────┘
```

- **Input:** One or more ROAR documents (Nuventive-style PDFs or legacy Word files).
- **Processing:** Parse and segment into PLO, Methods, Results/Conclusions, and Improvement Plan; run rule-based and/or model-based checks.
- **Output:** A set of **flags** (and optionally section-level pass/fail or scores) that reviewers can use to triage submissions.

---

## Flag Types: What Makes a "Bad" ROAR?

Flags fall into two broad categories, consistent with the project's Initial Report and the Rice Outcomes Assessment Reporting Handbook.

### Structural / Rule-based

- **Missing performance targets** — No clear statement of the desired level of achievement or comparison of results to target.
- **Missing or thin sections** — Little or no content in Methods, Results, or Improvement Plan.
- **Duplicated or copied content** — E.g., text reused from a previous year without updating.
- **Use of course grades as sole measure** — Handbook specifies PLO assessment should use direct measures of student work (rubrics, exams, portfolios), not overall course grades alone.
- **Missing or vague improvement plan** — Especially when results do not meet target; required for "closing the loop."

### Semantic / Quality

- **Vague or passive wording** — PLO or methods that are not specific or measurable.
- **Multiple goals in one PLO** — One outcome statement mixing several achievements (harder to align with a single measure).
- **Misalignment** — Methods do not clearly assess the stated PLO; or results do not map to the described method.
- **Weak analysis** — Results reported without clear interpretation or comparison to target.

Evaluation criteria used in experiments (e.g., in the "specific prompt" runs) align with the handbook: e.g., action verbs, learner perspective, direct assessment, clear targets, and current-year data in results.

---

## Data and Reference Materials

All reference data and documentation live in the **`background/`** folder. Do not commit sensitive or personally identifiable data; the materials below are for project context and development only.

### Handbook and Project Plan

| Item | Description |
|------|-------------|
| `RICE OUTCOMES ASSESSMENT REPORTING HANDBOOK_ January 2026_final.pdf` | Official Rice ROAR handbook: process, section requirements, direct assessment, targets, Bloom's taxonomy, Nuventive Improve. |
| `ROARS Initial Report (2).pdf` | Project plan: objectives, data description, pipeline design, validation, and expected next steps (e.g., flagging system, reviewer interface). |

### Labeled ROAR Samples

| Item | Description |
|------|-------------|
| `GOOD ROARS/` | **26** expert-labeled acceptable ROARs (`.docx`), one file per PLO-level report. |
| `BAD ROARS/` | **13** expert-labeled unacceptable ROARs (`.docx`). |
| `labeled_data_parser_docx.ipynb` | Notebook that parses these `.docx` files into structured rows: `filename`, `department`, `plo`, `methods`, `results_conclusions`, `improvement_plan`, `label` (1 = good, 0 = bad). Output: `labeled_dataset.csv`. |

These labeled samples are the primary source for validation and for training or prompting evaluation models.

### Model Evaluation Results

| Item | Description |
|------|-------------|
| `qwen/qwen results - initial .csv` | Parsed ROAR rows plus section-level scores (`plo_score`, `methods_score`, `results_score`, `plan_score`) from an LLM (Qwen) with a generic pass/fail prompt. |
| `qwen/qwen results - specific prompt.csv` | Same structure, with a **handbook-aligned prompt** (criteria for PLO, Methods, Results, Improvement Plan). |
| `LLaMa/LLaMa_simple_prompt_evaluations.xlsx` | LLaMa-based section evaluations (simple prompt). |
| `LLaMa/LLaMa_specific_prompt_evaluations.xlsx` | LLaMa-based section evaluations (specific/criteria-based prompt). |
| `Gemini Initial Audit Results on _Good PLOs_.xlsx` | Audit of "good" PLO wording; useful as a benchmark for PLO quality. |

### PLO Reference Data (Rice General Announcements)

| Item | Description |
|------|-------------|
| `RiceROARS_Sp26-main/webscraper.py` | Scrapes Rice GA (ga.rice.edu) for departments and credentials and extracts PLOs from credential pages. |
| `RiceROARS_Sp26-main/normalized_plos.py` | Converts scraped credential–PLO data into a long-format table (one row per outcome). |
| `RiceROARS_Sp26-main/rice_ga_credential_plos.csv` | Per-credential PLOs (department, credential, level, URLs, `plos_list`, etc.). |
| `RiceROARS_Sp26-main/PLOs_long.csv` | Long form: `department_name`, `credential_title`, `level`, `outcome_text`, `credential_url`. |

These CSVs support matching extracted PLOs from ROARs to official program PLOs and analyzing alignment.

---

## Technical Pipeline (Planned / In Progress)

1. **Parsing** — Extract text from Nuventive PDFs or Word ROARs and segment into: Department/Program, PLO, Methods, Results and Conclusions, Improvement Plan. The `labeled_data_parser_docx.ipynb` logic is a reference for section detection in Word; PDF parsing may use similar header/section rules or layout-based extraction.
2. **Rule-based checks** — Missing sections, length thresholds, presence of "course grade" language, presence of numeric targets, etc.
3. **Model-based evaluation** — Optional use of LLMs (e.g., Qwen, LLaMa, Gemini) or fine-tuned models with handbook-aligned criteria to score or flag PLO, Methods, Results, and Improvement Plan. Section-level scores (e.g., 0/1) can be aggregated into document-level flags.
4. **Validation** — Evaluate on the labeled GOOD/BAD ROAR set and any sponsor-provided mock data; report recall/precision/F1 for "bad" ROAR detection and flag categories.
5. **Output** — Categorized flags and, if desired, section-level pass/fail or scores, in a form suitable for a **reviewer dashboard** (e.g., list of flagged submissions with reasons).

---

## Repository Structure

```
roars/
├── README.md                 # This file
├── app/                      # Next.js frontend (upload, results, dashboard)
│   ├── layout.tsx
│   ├── page.tsx
│   └── globals.css
├── background/               # Reference materials (do not deploy)
│   ├── GOOD ROARS/           # Labeled acceptable ROAR .docx
│   ├── BAD ROARS/            # Labeled unacceptable ROAR .docx
│   ├── qwen/                 # Qwen LLM evaluation CSVs
│   ├── LLaMa/                # LLaMa evaluation spreadsheets
│   ├── RiceROARS_Sp26-main/  # PLO scrapers and CSVs
│   ├── labeled_data_parser_docx.ipynb
│   ├── RICE OUTCOMES ASSESSMENT REPORTING HANDBOOK_ January 2026_final.pdf
│   ├── ROARS Initial Report (2).pdf
│   └── Gemini Initial Audit Results on _Good PLOs_.xlsx
├── package.json
├── next.config.ts
└── ...
```

The **frontend** in `app/` is intended to support:

- Upload of Nuventive PDFs (and optionally Word files).
- Display of processing status and results.
- A **dashboard** of flagged ROARs with clear flag categories and, where applicable, section-level feedback.

---

## Getting Started

### Prerequisites

- Node.js 18+ (for the Next.js app).
- Python 3 (if running parsers or notebooks in `background/`).

### Run the frontend

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the app.

### .docx parsing (real extracted sections)

Uploaded **.docx** files are parsed by a Python script that reuses the logic from `background/labeled_data_parser_docx.ipynb`. The API route calls this script to extract **department**, **PLO**, **methods**, **results/conclusions**, and **improvement plan**; the UI then shows these in the Details panel.

To enable .docx parsing:

1. Install Python 3 and ensure `python3` (or `python` on Windows) is on your PATH.
2. Install the parser dependency:

   ```bash
   pip install -r scripts/requirements.txt
   ```
   or `pip install python-docx`.

3. From the project root, the API route runs `python3 scripts/parse_roar_docx.py <temp-path>` for each uploaded .docx. If the script is missing or fails (e.g. python-docx not installed), the app falls back to mock extracted sections for that file.

**PDF** uploads still receive mock extracted data; PDF parsing is not yet implemented.

### Use background materials

- Read the **Handbook** and **Initial Report** in `background/` for full context on ROAR structure and project objectives.
- Use **GOOD ROARS** and **BAD ROARS** plus `labeled_data_parser_docx.ipynb` to reproduce or extend the labeled dataset.
- Use the **qwen** and **LLaMa** result files to compare prompt designs and section-level scoring.
- Use **RiceROARS_Sp26-main** scripts and CSVs to link ROARs to official PLOs by program.

---

## Running with Docker

You can run ROARS without installing Node or Python locally by using the provided `Dockerfile`. The image bundles:

- Node 20 for the Next.js app.
- Python 3 + `python-docx` for the `.docx` parser.

### Build the image

From the project root:

```bash
docker build -t roars .
```

### Run the container

```bash
docker run --rm -p 3000:3000 roars
```

Then open [http://localhost:3000](http://localhost:3000) in your browser.

Inside the container, the API route uses the bundled Python (`python3`) and `scripts/parse_roar_docx.py` to parse uploaded `.docx` files. End users do **not** need Python installed on their machines.

---

## Summary

**ROARS** is a pre-screening system that:

- **Takes in:** Nuventive PDFs (or legacy ROAR Word documents).
- **Does:** Parses sections, runs rule-based and optional model-based checks aligned with the Rice ROAR Handbook.
- **Outputs:** Interpretable **flags** for bad or at-risk ROARs (structural and semantic), so OIE reviewers can focus effort and programs can improve submissions.

The `background/` folder contains the project definition, handbook, labeled examples, evaluation results, and PLO reference data needed to design and validate the pipeline and to build the frontend (upload + flags dashboard).
