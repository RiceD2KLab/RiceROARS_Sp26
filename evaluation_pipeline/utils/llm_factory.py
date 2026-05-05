"""
LLM factory – build the right LangChain chat model for each pipeline role.

╔══════════════════════════════════════════════════════════════════════════╗
║  HOW TO RUN THE LOCAL QWEN3.5-0.8B SERVER  (backend = "local_qwen")     ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  1. Install HuggingFace Transformers with serving extras:                ║
║                                                                          ║
║     pip install "transformers[serving] @                                 ║
║         git+https://github.com/huggingface/transformers.git@main"        ║
║                                                                          ║
║  2. Start the OpenAI-compatible server:                                  ║
║                                                                          ║
║     transformers serve                                                   ║
║         --force-model Qwen/Qwen3.5-0.8B                                 ║
║         --port 8000                                                      ║
║         --continuous-batching                                            ║
║                                                                          ║
║  3. The API will be available at:  http://localhost:8000/v1              ║
║                                                                          ║
║  4. No API key is needed – use api_key="EMPTY"                           ║
║                                                                          ║
║  5. Verify the server is up:                                             ║
║     curl http://localhost:8000/v1/models                                 ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

Supported backends
------------------
"ollama"
    Uses ``langchain_ollama.ChatOllama``.  Ollama must be running locally
    (default: http://localhost:11434).  ``format="json"`` enforces JSON output.

"openai"
    Uses ``langchain_openai.ChatOpenAI`` pointed at any OpenAI-compatible
    endpoint.  The caller must supply ``openai_base_url`` and
    ``openai_api_key`` explicitly (e.g. Azure AI Foundry MaaS endpoint for
    the evaluator).  Standard OpenAI params are accepted; non-standard vLLM
    extras (top_k, min_p, …) can be forwarded via ``openai_extra_body``.
    Leave ``openai_extra_body`` empty for Azure – Azure rejects unknown fields.

"azure_openai"  ← VERIFIER DEFAULT
    Uses ``langchain_openai.AzureChatOpenAI`` for Azure OpenAI deployments
    served through ``cognitiveservices.azure.com``.  Unlike the ``"openai"``
    backend (which points at a plain OpenAI-compatible URL), this backend
    requires an ``api_version`` and an ``azure_deployment`` name.
    Config keys: VERIFIER_AZURE_ENDPOINT, VERIFIER_AZURE_DEPLOYMENT,
                 VERIFIER_AZURE_API_VERSION, VERIFIER_AZURE_API_KEY.
    Callers need zero extra arguments – everything comes from config / .env.

"local_qwen"  ← kept for reference, currently inactive
    A named preset for a locally-served Qwen3.5-0.8B instance (HuggingFace
    Transformers server, see the startup instructions in the module docstring).
    Uses ``ChatOpenAI`` internally with Qwen3.5 sampling params pre-baked.
    Switch back by setting VERIFIER_BACKEND=local_qwen in .env.
"""

from __future__ import annotations

from typing import Any, Literal

Backend = Literal["ollama", "openai", "azure_openai", "local_qwen"]


def build_llm(
    model: str,
    backend: Backend,
    *,
    # ── Ollama knobs ──────────────────────────────────────────────────────
    ollama_base_url: str | None = None,
    ollama_temperature: float | None = None,
    ollama_format: str | None = None,
    # ── OpenAI-compat knobs (required for "openai" backend) ───────────────
    # Leave None when using "local_qwen" – the factory supplies its own.
    openai_base_url: str | None = None,
    openai_api_key: str | None = None,
    openai_temperature: float = 0.0,
    openai_top_p: float = 1.0,
    openai_presence_penalty: float = 0.0,
    openai_max_tokens: int = 4096,
    # Extra body: only forwarded when non-empty.
    # Use for server-specific params (e.g. vLLM top_k / min_p).
    # Do NOT populate for Azure AI Foundry – Azure rejects unknown fields.
    openai_extra_body: dict[str, Any] | None = None,
    # Default query params appended to every request URL.
    # Required for Azure AI Foundry MaaS endpoints: {"api-version": "2024-05-01-preview"}.
    openai_default_query: dict[str, Any] | None = None,
    # ── Azure OpenAI knobs (override config defaults per call) ────────────
    # Use these to pass role-specific values without changing global config.
    # Critical for o4-mini: temperature MUST be 1.0 (reasoning model).
    azure_temperature: float | None = None,
    azure_max_tokens:  int   | None = None,
):
    """
    Return a LangChain chat model configured for *backend*.

    Parameters
    ----------
    model:
        Model identifier – Ollama name, HuggingFace model ID, or Azure
        deployment name.
    backend:
        One of ``"ollama"``, ``"openai"``, or ``"local_qwen"``.
    openai_base_url / openai_api_key:
        **Required** for the ``"openai"`` backend; ignored for
        ``"local_qwen"`` (the factory provides hardcoded local defaults).
    openai_extra_body:
        Server-specific params forwarded verbatim in the HTTP body.
        Pass Qwen extras for vLLM/SGLang; leave ``None`` for Azure.
    """
    import config  # local import keeps config mockable in tests

    if backend == "ollama":
        return _build_ollama(
            model=model,
            base_url=ollama_base_url or config.OLLAMA_BASE_URL,
            temperature=(
                ollama_temperature
                if ollama_temperature is not None
                else config.OLLAMA_TEMPERATURE
            ),
            fmt=ollama_format or config.OLLAMA_FORMAT,
        )

    if backend == "openai":
        if not openai_base_url:
            raise ValueError(
                "openai_base_url is required for the 'openai' backend. "
                "Pass config.EVALUATOR_API_BASE or config.VERIFIER_API_BASE."
            )
        if not openai_api_key:
            raise ValueError(
                "openai_api_key is required for the 'openai' backend. "
                "Pass config.EVALUATOR_API_KEY or config.VERIFIER_API_KEY."
            )
        return _build_openai(
            model=model,
            base_url=openai_base_url,
            api_key=openai_api_key,
            temperature=openai_temperature,
            top_p=openai_top_p,
            presence_penalty=openai_presence_penalty,
            max_tokens=openai_max_tokens,
            extra_body=openai_extra_body or {},
            default_query=openai_default_query or {},
        )

    if backend == "azure_openai":
        # ── Azure OpenAI service (cognitiveservices.azure.com) ────────────
        # Works for both evaluator and verifier roles.
        # Callers can pass azure_temperature / azure_max_tokens to override
        # the config defaults (important for o4-mini reasoning model which
        # requires temperature=1 and larger max_completion_tokens).
        _temp      = azure_temperature if azure_temperature is not None else config.VERIFIER_TEMPERATURE
        _max_tok   = azure_max_tokens  if azure_max_tokens  is not None else config.VERIFIER_MAX_TOKENS
        return _build_azure_openai(
            deployment=model,
            endpoint=config.AZURE_ENDPOINT,
            api_version=config.AZURE_API_VERSION,
            api_key=config.AZURE_API_KEY,
            temperature=_temp,
            max_tokens=_max_tok,
        )

    if backend == "local_qwen":
        # ── Pre-baked Qwen3.5 preset (HuggingFace Transformers server) ────
        # The HF Transformers server (`transformers serve`) only accepts
        # standard OpenAI parameters.  vLLM-specific extras (top_k, min_p,
        # repetition_penalty) are intentionally omitted here — the server
        # returns HTTP 422 "Unexpected fields" if they are included.
        #
        # Qwen3.5 non-thinking mode, text tasks
        # (from official Qwen3.5 best-practices guide)
        return _build_openai(
            model=model,
            base_url=config.LOCAL_QWEN_API_BASE,
            api_key=config.LOCAL_QWEN_API_KEY,
            temperature=config.QWEN_TEMPERATURE,            # 1.0
            top_p=config.QWEN_TOP_P,                        # 1.0
            presence_penalty=config.QWEN_PRESENCE_PENALTY,  # 2.0
            max_tokens=config.QWEN_MAX_TOKENS,              # 32 768
            extra_body={},   # empty → no extra_body injected
        )

    raise ValueError(
        f"Unknown backend {backend!r}. "
        "Valid choices are 'ollama', 'openai', 'azure_openai', and 'local_qwen'."
    )


# ---------------------------------------------------------------------------
# Private builders
# ---------------------------------------------------------------------------

def _build_azure_openai(
    *,
    deployment: str,
    endpoint: str,
    api_version: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
):
    """
    Build an AzureChatOpenAI client for deployments served through
    cognitiveservices.azure.com (Azure OpenAI service).

    This is distinct from the ``"openai"`` backend (which uses ChatOpenAI
    with a plain base_url) because Azure OpenAI requires:
      - ``azure_deployment``  – the deployment name (e.g. "gpt_verifier")
      - ``azure_endpoint``    – the cognitiveservices URL
      - ``api_version``       – e.g. "2024-12-01-preview"
    """
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=endpoint,
        api_version=api_version,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _build_ollama(*, model: str, base_url: str, temperature: float, fmt: str):
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        format=fmt,                 # forces Ollama into strict JSON-only mode
    )


def _build_openai(
    *,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    max_tokens: int,
    extra_body: dict[str, Any],
    default_query: dict[str, Any],
):
    from langchain_openai import ChatOpenAI

    # top_p and presence_penalty are standard OpenAI params → pass directly.
    # model_kwargs is reserved for server-specific extras only (extra_body).
    model_kwargs: dict[str, Any] = {}
    if extra_body:
        # Only attach extra_body when non-empty.
        # Azure AI Foundry rejects unknown extra_body keys (HTTP 422).
        # The HF Transformers server also rejects vLLM-specific fields.
        model_kwargs["extra_body"] = extra_body

    return ChatOpenAI(
        model=model,
        openai_api_base=base_url,
        openai_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        presence_penalty=presence_penalty,
        model_kwargs=model_kwargs if model_kwargs else {},
        # default_query appends ?key=val to every request URL.
        # Required for Azure AI Foundry MaaS: {"api-version": "2024-05-01-preview"}
        **({"default_query": default_query} if default_query else {}),
    )
