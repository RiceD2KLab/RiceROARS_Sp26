# ROARS - Rice Outcomes Assessment Reporting Screening

ROARS is a pre-screening web app for Rice Outcomes Assessment Reporting (ROAR) submissions. It accepts `.pdf` or `.docx` reports and runs a multi-model evaluation pipeline that extracts report sections, scores them against section-level quality rules, and flags reports that need human review.

## What The App Does

Each ROAR report is evaluated across four sections:

- **Program Learning Outcomes (PLOs)**: what students should know or be able to do
- **Assessment Methods**: how student work was directly assessed
- **Results and Conclusions**: numeric results and comparison to targets
- **Improvement Plan**: follow-up actions or continuation plans

The final project configuration uses **DeepSeek-V3.2** as the evaluator and **o4-mini** as the verifier. The evaluator scores the report, the verifier independently checks the scoring, and the pipeline can loop through feedback until the two models agree or the maximum iteration count is reached.

## Repository Structure

```text
app/                         Next.js frontend and API route
app/api/analyze/route.ts     Upload endpoint that runs the Python pipeline
evaluation_pipeline/         Core Python ROAR evaluation pipeline
evaluation_pipeline/.env.example
                             Environment variable template, safe to commit
evaluation_pipeline/config.py
                             LLM endpoints, model names, and pipeline settings
evaluation_pipeline/model_sets.py
                             Model profile definitions
evaluation_pipeline/prompt_sets.py
                             Prompt strategy definitions
scripts/run_roar_evaluation.py
                             Bridge script used by the Next.js API
scripts/requirements.txt     Python dependencies
lib/types/roar.ts            Shared TypeScript result types
Dockerfile                   Container build
docker-compose.yml           Local Docker Compose setup
```

## Prerequisites

- Node.js 18+
- Python 3.10+
- Azure AI Foundry / Azure OpenAI credentials for the evaluator and verifier deployments

## Local Setup

Install dependencies:

```bash
npm install
python -m venv .venv
.venv\Scripts\activate
pip install -r scripts/requirements.txt
```

On macOS/Linux, activate the virtual environment with:

```bash
source .venv/bin/activate
```

Create your private environment file:

```bash
copy evaluation_pipeline\.env.example evaluation_pipeline\.env
```

On macOS/Linux:

```bash
cp evaluation_pipeline/.env.example evaluation_pipeline/.env
```

Then edit `evaluation_pipeline/.env` and replace the placeholder values with real credentials. Do not commit `.env`.

Start the app:

```bash
npx next dev
```

Open http://localhost:3000.

## Environment Variables

The app loads environment variables from `evaluation_pipeline/.env`. The bridge script also supports a repo-root `.env`, but `evaluation_pipeline/.env` is the recommended location for this project.

Required for the final project configuration:

| Variable | Example | Purpose |
| --- | --- | --- |
| `EVALUATOR_BACKEND` | `openai` | Uses the OpenAI-compatible client for Azure AI Foundry MaaS |
| `EVALUATOR_MODEL` | `DeepSeek-V3.2-evaluator` | Evaluator deployment/model name |
| `EVALUATOR_API_BASE` | `https://<resource>.services.ai.azure.com/models/` | Azure AI Foundry models endpoint |
| `EVALUATOR_API_KEY` | `<secret>` | Key for the evaluator endpoint |
| `EVALUATOR_API_VERSION` | `2024-05-01-preview` | API version for the evaluator endpoint |
| `VERIFIER_BACKEND` | `azure_openai` | Uses Azure OpenAI for the verifier |
| `VERIFIER_MODEL` | `o4-mini-verifier` | Verifier model/deployment label |
| `VERIFIER_AZURE_DEPLOYMENT` | `o4-mini-verifier` | Azure OpenAI verifier deployment name |
| `AZURE_ENDPOINT` | `https://<resource>.cognitiveservices.azure.com/` | Shared Azure OpenAI endpoint |
| `AZURE_API_KEY` | `<secret>` | Azure OpenAI key |
| `AZURE_API_VERSION` | `2025-01-01-preview` | Azure OpenAI API version |

Useful defaults:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ROAR_MODEL_PROFILE` | `MS3_DeepSeek_o4mini` | Final project model set |
| `ROAR_PROMPT_SET` | `B_ChainOfThought` | Final project prompt strategy |
| `ROAR_CLASSIFICATION_STRATEGY` | `strict` | Requires all four sections to pass |
| `MAX_ITERATIONS` | `2` | Maximum evaluator-verifier feedback loops |
| `ROAR_PIPELINE_TIMEOUT_MS` | `300000` | Next.js API timeout per file, in milliseconds |
| `PYTHON` | auto-detected | Optional Python executable override |

Reasoning-model settings:

```env
VERIFIER_AZURE_TEMPERATURE=1.0
VERIFIER_AZURE_MAX_TOKENS=16384
```

These are included in `.env.example` because o-series reasoning deployments require temperature `1.0` and need a larger token budget.

## Running With Docker

Build and run directly:

```bash
docker build -t roars .
docker run --rm -p 3000:3000 --env-file evaluation_pipeline/.env roars
```

Or use Docker Compose:

```bash
docker compose up --build
docker compose down
```

## Scoring Rules

Each section receives a binary score:

- **PLO**: must describe what students will achieve, know, or be able to do
- **Methods**: must name a direct assessment measure and a rubric/scoring criterion
- **Results**: must include actual numbers and comparison to a benchmark
- **Plan**: must describe a specific next step or continuation plan

The final classification strategy is strict: a ROAR passes only when all four sections receive `1`. Any section score of `0` flags the document for human review.

## Sharing Credentials With An Evaluator

Do not upload a real `.env` file or API keys to GitHub. Commit only `evaluation_pipeline/.env.example`.

For grading, send the instructor the private values separately and ask them to create `evaluation_pipeline/.env` from the template. A short email can be:

```text
Hi Professor,

I uploaded the project code to GitHub without API credentials. To run the full ROARS evaluation pipeline, please copy evaluation_pipeline/.env.example to evaluation_pipeline/.env and fill in the Azure AI Foundry / Azure OpenAI values I am sending separately.

The required variables are EVALUATOR_API_KEY, EVALUATOR_API_BASE, AZURE_API_KEY, and AZURE_ENDPOINT. The model/deployment names and API versions are already shown in the template.

Best,
<your name>
```

Use a private channel for the real key values. If a key was ever pasted into a public repo, chat, screenshot, or shared document, rotate it before grading.
