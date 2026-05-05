import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Backend selection (per model role)
# ---------------------------------------------------------------------------
# "ollama"        → ChatOllama       (local Ollama server)
# "openai"        → ChatOpenAI       (any OpenAI-compatible endpoint: Azure AI Foundry MaaS, vLLM…)
# "azure_openai"  → AzureChatOpenAI  (Azure OpenAI service via cognitiveservices.azure.com)
#                   Requires VERIFIER_AZURE_ENDPOINT, VERIFIER_AZURE_API_KEY,
#                   and VERIFIER_AZURE_API_VERSION to be set.
# "local_qwen"    → ChatOpenAI pre-configured for a locally-served Qwen3.5-0.8B instance
#                   (HuggingFace Transformers server on http://localhost:8000/v1).
#                   Kept in code for reference; not active by default.
EVALUATOR_BACKEND: str = os.getenv("EVALUATOR_BACKEND", "openai")
VERIFIER_BACKEND:  str = os.getenv("VERIFIER_BACKEND",  "azure_openai")

# ---------------------------------------------------------------------------
# Evaluator model  (LLaMA – Azure AI Foundry)
# ---------------------------------------------------------------------------
EVALUATOR_MODEL: str = os.getenv("EVALUATOR_MODEL", "llama-evaluator")

# Azure AI Foundry models endpoint.
# The `models/` base URL + api-version query param is the correct format for
# Azure AI Foundry serverless (MaaS) deployments like Llama 4 Maverick.
# The OpenAI SDK appends "chat/completions" to the base URL automatically.
EVALUATOR_API_BASE: str = os.getenv(
    "EVALUATOR_API_BASE",
    "https://rice-edbi-azure-d2k-roar-foundry.services.ai.azure.com/models/",
)
EVALUATOR_API_KEY:     str = os.getenv("EVALUATOR_API_KEY", "")
EVALUATOR_API_VERSION: str = os.getenv("EVALUATOR_API_VERSION", "2024-05-01-preview")

# LLaMA scoring: low temperature for deterministic JSON output.
# No vLLM-specific extras (top_k / min_p / repetition_penalty) – Azure ignores
# or rejects unknown extra_body fields.
EVALUATOR_TEMPERATURE:       float = float(os.getenv("EVALUATOR_TEMPERATURE", "0.0"))
EVALUATOR_TOP_P:             float = 1.0
EVALUATOR_PRESENCE_PENALTY:  float = 0.0   # no penalty needed for structured scoring
EVALUATOR_MAX_TOKENS:        int   = 4096  # scoring responses are short

# ---------------------------------------------------------------------------
# Verifier model  (Qwen3.5-0.8B)
# ---------------------------------------------------------------------------
VERIFIER_MODEL: str = os.getenv("VERIFIER_MODEL", "Qwen/Qwen3.5-0.8B")

# ── "local_qwen" backend ──────────────────────────────────────────────────
# Fixed endpoint for a locally-served HuggingFace Transformers server.
# Override via env vars if you run the server on a different host/port.
LOCAL_QWEN_API_BASE: str = os.getenv("LOCAL_QWEN_API_BASE", "http://localhost:8000/v1")
LOCAL_QWEN_API_KEY:  str = os.getenv("LOCAL_QWEN_API_KEY",  "EMPTY")

# ── Shared Azure OpenAI service settings (cognitiveservices.azure.com) ────
# Both evaluator and verifier can use this same endpoint when their backend
# is set to "azure_openai".  Individual EVALUATOR_AZURE_* / VERIFIER_AZURE_*
# overrides take precedence if set.
AZURE_ENDPOINT:    str = os.getenv(
    "AZURE_ENDPOINT",
    "https://rice-edbi-azure-d2k-roar-foundry.cognitiveservices.azure.com/",
)
AZURE_API_VERSION: str = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
AZURE_API_KEY:     str = os.getenv("AZURE_API_KEY", "")

# ── Per-role overrides (optional – fall back to shared AZURE_* above) ──────
VERIFIER_AZURE_ENDPOINT:    str = os.getenv("VERIFIER_AZURE_ENDPOINT",    AZURE_ENDPOINT)
VERIFIER_AZURE_DEPLOYMENT:  str = os.getenv("VERIFIER_AZURE_DEPLOYMENT",  "gpt_verifier")
VERIFIER_AZURE_API_VERSION: str = os.getenv("VERIFIER_AZURE_API_VERSION", AZURE_API_VERSION)
VERIFIER_AZURE_API_KEY:     str = os.getenv("VERIFIER_AZURE_API_KEY",     AZURE_API_KEY)

EVALUATOR_AZURE_ENDPOINT:    str = os.getenv("EVALUATOR_AZURE_ENDPOINT",    AZURE_ENDPOINT)
EVALUATOR_AZURE_DEPLOYMENT:  str = os.getenv("EVALUATOR_AZURE_DEPLOYMENT",  EVALUATOR_MODEL)
EVALUATOR_AZURE_API_VERSION: str = os.getenv("EVALUATOR_AZURE_API_VERSION", AZURE_API_VERSION)
EVALUATOR_AZURE_API_KEY:     str = os.getenv("EVALUATOR_AZURE_API_KEY",     AZURE_API_KEY)

# Evaluator sampling — o4-mini is a REASONING model:
#   • temperature MUST be 1.0 (o-series models do not support temperature < 1)
#   • max_completion_tokens should be generous to allow reasoning trace + JSON output
#   • presence_penalty / top_p are NOT passed (unsupported by o-series)
EVALUATOR_AZURE_TEMPERATURE:  float = float(os.getenv("EVALUATOR_AZURE_TEMPERATURE", "1.0"))
EVALUATOR_AZURE_MAX_TOKENS:   int   = int(os.getenv("EVALUATOR_AZURE_MAX_TOKENS",   "16384"))

# Verifier sampling — configurable so o4-mini (reasoning) and GPT (standard) both work.
# For GPT-class verifiers (gpt-5.4-mini, gpt-4.1): temperature=0.0
# For o4-mini verifier (reasoning model):            temperature=1.0, max_tokens=16384
VERIFIER_TEMPERATURE:         float = float(os.getenv("VERIFIER_TEMPERATURE",          "0.0"))
VERIFIER_MAX_TOKENS:          int   = 4096
VERIFIER_AZURE_TEMPERATURE:   float = float(os.getenv("VERIFIER_AZURE_TEMPERATURE",    "0.0"))
VERIFIER_AZURE_MAX_TOKENS:    int   = int(os.getenv("VERIFIER_AZURE_MAX_TOKENS",       "4096"))

# ── "openai" backend fallback ─────────────────────────────────────────────
# Used when VERIFIER_BACKEND=openai (e.g. vLLM / SGLang with explicit creds).
VERIFIER_API_BASE: str = os.getenv("VERIFIER_API_BASE", "http://localhost:8000/v1")
VERIFIER_API_KEY:  str = os.getenv("VERIFIER_API_KEY",  "EMPTY")

# Qwen3.5 recommended sampling parameters – non-thinking mode, text tasks
# (from the official Qwen3.5 best-practices guide)
QWEN_TEMPERATURE:        float = 1.0
QWEN_TOP_P:              float = 1.0
QWEN_TOP_K:              int   = 20
QWEN_MIN_P:              float = 0.0
QWEN_PRESENCE_PENALTY:   float = 2.0
QWEN_REPETITION_PENALTY: float = 1.0
QWEN_MAX_TOKENS:         int   = 32_768

# ---------------------------------------------------------------------------
# Ollama fallback  (used when *_BACKEND = "ollama")
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL:   str   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TEMPERATURE: float = 0.0
OLLAMA_FORMAT:      str   = "json"

# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "2"))

# ---------------------------------------------------------------------------
# Scoring weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHTS: dict[str, float] = {
    "methods": 0.30,
    "results": 0.30,
    "plo":     0.25,
    "plan":    0.15,
}
