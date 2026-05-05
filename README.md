# ROARS — Rice Outcomes Assessment Reporting Screening

A pre-screening system that ingests ROAR submissions (PDF or Word) and produces **section-level scores and flags** so Institutional Effectiveness (IE) reviewers can focus attention where it is most needed.

---

## What Are ROARs?

**ROAR** = **R**ice **O**utcomes **A**ssessment **R**eporting. Each academic degree program at Rice University submits annual reports describing:

- **Program Learning Outcomes (PLOs)** — what students should know or be able to do
- **Assessment Methods** — how the program directly assessed student work (rubrics, exams, portfolios, etc.)
- **Results and Conclusions** — aggregated results, comparison to targets, and interpretation
- **Improvement Plan** — actions to be taken when targets are not met

---

## System Overview

```
┌─────────────────────┐     ┌──────────────────────────────┐     ┌─────────────────────────┐
│  ROAR .docx / .pdf   │     │  Extract → Evaluate → Verify  │     │  Section scores + flags  │
│  (upload via UI)     │ ──► │  (multi-model LLM pipeline)   │ ──► │  (pass/fail per section) │
└─────────────────────┘     └──────────────────────────────┘     └─────────────────────────┘
```

The pipeline uses a **multi-model evaluator–verifier architecture** with an iterative feedback loop:

1. **Input & Decomposition** — Parse the ROAR into four sections (PLO, Methods, Results, Plan)
2. **Initial Evaluation** — DeepSeek-V3.2 scores each section (pass/fail) with Chain-of-Thought reasoning
3. **Verification** — o4-mini independently checks the evaluator's scores against scoring rules
4. **Feedback Loop** — If disagreements exist, the evaluator revises; loop repeats until consensus (max 2 iterations)
5. **Final Output** — Section scores, weighted quality score, strict classification, and flags

**Default configuration:** MS3 (DeepSeek-V3.2 evaluator + o4-mini verifier) with Chain-of-Thought Evidence Anchoring prompts and strict classification (all sections must pass).

---

## Repository Structure

```
roars/
├── app/                      # Next.js frontend
│   ├── layout.tsx            # Root layout
│   ├── page.tsx              # Main UI (upload, model selection, results table, detail panel)
│   ├── globals.css           # Tailwind styles
│   └── api/analyze/
│       └── route.ts          # POST endpoint — runs Python pipeline via subprocess
├── evaluation_pipeline/      # Python backend (the core evaluation engine)
│   ├── config.py             # Azure endpoints, API keys, sampling parameters
│   ├── model_sets.py         # Model configurations (MS1/MS2/MS3)
│   ├── prompt_sets.py        # Prompt strategies (A: Disqualifier-First, B: Chain-of-Thought, C: Few-Shot)
│   ├── main.py               # Standalone CLI
│   ├── pipeline/
│   │   ├── extractor.py      # Step 1: Section extraction from .docx/.pdf
│   │   ├── evaluator.py      # Step 2: Primary LLM scoring
│   │   ├── verifier.py       # Step 3: Independent verification
│   │   ├── feedback.py       # Step 5: Feedback reconciliation
│   │   └── roar_pipeline.py  # Orchestrator (wires all steps together)
│   ├── models/schemas.py     # Pydantic data models
│   ├── prompts/templates.py  # Shared prompt templates
│   ├── utils/
│   │   ├── llm_factory.py    # LLM client builder (Azure OpenAI, Ollama, etc.)
│   │   └── json_parser.py    # Robust JSON extraction from LLM output
│   └── .env.example          # Template for API keys and endpoints
├── scripts/
│   ├── run_roar_evaluation.py  # Bridge: Next.js API → evaluation_pipeline
│   ├── parse_roar_docx.py      # Legacy .docx parser (fallback)
│   └── requirements.txt        # Python dependencies
├── lib/types/roar.ts         # Shared TypeScript types
├── Dockerfile                # Multi-stage build (Node + Python)
├── docker-compose.yml        # One-command local deployment
├── package.json
└── tsconfig.json
```

---

## Getting Started

### Prerequisites

- **Node.js 18+** (for the Next.js app)
- **Python 3.10+** on PATH as `python` or `python3` (set `PYTHON` env var to override)

### Install and run

```bash
# Install Node dependencies
npm install

# Install Python dependencies
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r scripts/requirements.txt

# Configure LLM endpoints
cp evaluation_pipeline/.env.example evaluation_pipeline/.env
# Edit .env with your Azure API keys and endpoints

# Start the dev server
npx next dev
```

Open [http://localhost:3000](http://localhost:3000) to use the app.

### LLM Configuration

Copy `evaluation_pipeline/.env.example` to `evaluation_pipeline/.env` and fill in:

| Variable | Description |
|----------|-------------|
| `EVALUATOR_API_BASE` | Azure AI Foundry endpoint for DeepSeek-V3.2 |
| `EVALUATOR_API_KEY` | API key for the evaluator endpoint |
| `AZURE_ENDPOINT` | Azure OpenAI endpoint (for o4-mini verifier) |
| `AZURE_API_KEY` | API key for Azure OpenAI |

Without valid credentials, uploads fall back to **legacy .docx parsing** (sections extracted but not scored) or **placeholder content** for PDFs.

### Environment Variables (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `ROAR_MODEL_PROFILE` | `MS3_DeepSeek_o4mini` | Model set to use |
| `ROAR_PROMPT_SET` | `B_ChainOfThought` | Prompt strategy |
| `ROAR_CLASSIFICATION_STRATEGY` | `strict` | `strict` (all sections must pass) or `weighted` |
| `ROAR_PIPELINE_TIMEOUT_MS` | `300000` | Max time per file (ms) |
| `MAX_ITERATIONS` | `2` | Max feedback loop iterations |

---

## Running with Docker

```bash
# Build
docker build -t roars .

# Run (with env file for API keys)
docker run --rm -p 3000:3000 --env-file evaluation_pipeline/.env roars
```

Or use Docker Compose:

```bash
docker compose up --build     # Start
docker compose down            # Stop
```

Open [http://localhost:3000](http://localhost:3000).

---

## Model Configurations

| Set | Evaluator | Verifier | Use Case |
|-----|-----------|----------|----------|
| MS1 | LLaMA-4-Maverick | GPT-5.4-mini | Baseline |
| MS2 | o4-mini (reasoning) | gpt-4.1 | Reasoning evaluator |
| **MS3** | **DeepSeek-V3.2** | **o4-mini (reasoning)** | **Production default** |

The UI provides dropdown selectors to override the evaluator and verifier deployment for individual requests.

---

## Scoring Rules

Each section is scored pass (1) or fail (0):

- **PLO** (25%): Must describe what students will achieve, know, or be able to do
- **Methods** (30%): Must name a direct assessment measure AND a rubric/scoring criteria
- **Results** (30%): Must contain actual numbers AND comparison to a benchmark
- **Plan** (15%): Must describe a specific next step or continuation plan

**Strict classification:** A ROAR passes only if all four sections score 1. Any failure flags the document for human review.
