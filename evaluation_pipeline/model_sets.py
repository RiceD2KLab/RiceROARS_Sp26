"""
model_sets.py — Three model set configurations for the ROAR ablation study.

Each set specifies the evaluator and verifier model + backend.
Fill in the Azure deployment names after deploying the models.

See model_set_rationale.txt for the full selection rationale.

Usage in evaluation_test.py / model_evaluation_test.py:
    from model_sets import MODEL_SETS
"""

from __future__ import annotations

# ── Deployment names ──────────────────────────────────────────────────────────
# Update these after deploying the models in Azure.

# Set 2 — deploy on Azure OpenAI (cognitiveservices.azure.com)  ← ACTIVE
O4_MINI_DEPLOYMENT   = "o4-mini-evaluator"   # deployed ✓
GPT41_DEPLOYMENT     = "gpt-4.1-verifier"    # deployed ✓  (note: hyphen, not dot)
O4_MINI_VER_DEPLOY   = "o4-mini-verifier"    # deploy when running Set 2 / legacy MS3 tests

# Set 3 — deploy on Azure AI Foundry MaaS (services.ai.azure.com) + Azure OpenAI verifier
DEEPSEEK_DEPLOYMENT  = "DeepSeek-V3.2-evaluator"  # Azure AI Foundry MaaS deployment name
# Final production verifier (MS3 strict + prompt B): o4-mini reasoning model on Azure OpenAI
O4_MINI_VER_DEPLOY_MS3 = "o4-mini-verifier"


MODEL_SETS: dict[str, dict] = {

    # ── Baseline (current production setup) ───────────────────────────────────
    "MS1_LLaMA_GPTmini": {
        "label":       "Set 1 — LLaMA-4-Maverick + GPT-5.4-mini (baseline)",
        "description": "Current production setup.  Large MoE evaluator with a fast GPT verifier.",
        "evaluator": {
            "model":   "llama-evaluator",    # Azure AI Foundry MaaS deployment name
            "backend": "openai",
            # endpoint + api_key come from config.py (EVALUATOR_API_BASE / EVALUATOR_API_KEY)
        },
        "verifier": {
            "model":   "gpt_verifier",       # Azure OpenAI deployment name
            "backend": "azure_openai",
            # endpoint + api_key come from config.py (VERIFIER_AZURE_* keys)
        },
    },

    # ── Model Set 2 — OpenAI Reasoning ───────────────────────────────────────
    # Hypothesis: a reasoning evaluator (o4-mini) applies multi-criteria rubrics
    # more systematically; a more capable verifier (gpt-4.1) catches more errors.
    "MS2_o4mini_GPT41": {
        "label":       "Set 2 — o4-mini (reasoning) + gpt-4.1",
        "description": (
            "Reasoning evaluator: o4-mini thinks step-by-step before scoring, "
            "reducing false positives on vague sections.  "
            "gpt-4.1 verifier is more capable than gpt-5.4-mini at rule enforcement."
        ),
        "evaluator": {
            "model":   O4_MINI_DEPLOYMENT,
            "backend": "azure_openai",
            # Uses same cognitiveservices endpoint as the verifier.
            # Set EVALUATOR_AZURE_DEPLOYMENT=o4-mini-evaluator in .env,
            # or override below.
        },
        "verifier": {
            "model":   GPT41_DEPLOYMENT,   # "gpt-4.1-verifier"
            "backend": "azure_openai",
        },
    },

    # ── Model Set 3 — DeepSeek + Reasoning Verifier ───────────────────────────
    # Hypothesis: DeepSeek-V3.2 as evaluator tests a different model family
    # (different training biases vs LLaMA).  o4-mini as verifier catches criterion
    # violations more reliably than a standard chat model.
    "MS3_DeepSeek_o4mini": {
        "label":       "Set 3 — DeepSeek-V3.2 + o4-mini (reasoning verifier)",
        "description": (
            "DeepSeek-V3.2 evaluator: top-tier open-weights model with strong "
            "structured output and different training biases than LLaMA.  "
            "o4-mini reasoning verifier catches criterion violations "
            "that standard chat verifiers miss."
        ),
        "evaluator": {
            "model":   DEEPSEEK_DEPLOYMENT,
            "backend": "openai",
            # Uses Azure AI Foundry MaaS endpoint — same pattern as llama-evaluator.
            # Endpoint: config.EVALUATOR_API_BASE  (models/ path + api-version)
        },
        "verifier": {
            "model":   O4_MINI_VER_DEPLOY_MS3,
            "backend": "azure_openai",
        },
    },
}

# Alias for backward compatibility with env vars that reference the old key.
MODEL_SETS["MS3_DeepSeek_o1mini"] = MODEL_SETS["MS3_DeepSeek_o4mini"]
