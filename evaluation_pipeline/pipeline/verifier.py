"""
Verifier – Step 3 of the ROAR pipeline.

Uses Qwen3.5-0.8B and the verification prompt (Appendix A.2) to independently
check the correctness and consistency of the primary evaluator's scores.

╔══════════════════════════════════════════════════════════════════════════╗
║  RUNNING THE LOCAL QWEN3.5-0.8B SERVER  (default backend: local_qwen)   ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Step 1 – Install HuggingFace Transformers with serving extras:          ║
║                                                                          ║
║    pip install "transformers[serving] @                                  ║
║        git+https://github.com/huggingface/transformers.git@main"         ║
║                                                                          ║
║  Step 2 – Start the server (run in a separate terminal):                 ║
║                                                                          ║
║    transformers serve                                                    ║
║        --force-model Qwen/Qwen3.5-0.8B                                  ║
║        --port 8000                                                       ║
║        --continuous-batching                                             ║
║                                                                          ║
║  Step 3 – Verify it's running:                                           ║
║                                                                          ║
║    curl http://localhost:8000/v1/models                                  ║
║                                                                          ║
║  Step 4 – Run the pipeline (the verifier will connect automatically):    ║
║                                                                          ║
║    python main.py --pdf your_roar.pdf                                    ║
║                                                                          ║
║  Notes:                                                                  ║
║  • The server downloads Qwen/Qwen3.5-0.8B on first run (~1.6 GB).       ║
║  • No API key is required – the server accepts api_key="EMPTY".          ║
║  • To switch backends: set VERIFIER_BACKEND=openai in your .env          ║
║    and supply VERIFIER_API_BASE / VERIFIER_API_KEY for a remote server.  ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

Backend routing
---------------
"azure_openai"  (ACTIVE DEFAULT)
    Uses AzureChatOpenAI for the GPT-5.4-mini deployment (gpt_verifier) on
    Azure OpenAI service (cognitiveservices.azure.com).  All settings come
    from config.VERIFIER_AZURE_* — no extra arguments needed in this module.
    Much faster (~2-5 sec) and more consistent than the local Qwen server.

"local_qwen"  (kept for reference — server not required to be running)
    The factory pre-configures ChatOpenAI for the local HF Transformers server
    at http://localhost:8000/v1 with all Qwen3.5 sampling parameters baked in.
    Switch back by setting VERIFIER_BACKEND=local_qwen in .env and restarting
    the server: transformers serve Qwen/Qwen3.5-0.8B --port 8000 --continuous-batching

"openai"
    A remote OpenAI-compatible server (e.g. vLLM / SGLang).
    Reads VERIFIER_API_BASE and VERIFIER_API_KEY from config / .env.

"ollama"
    Local Ollama server.  Reads OLLAMA_BASE_URL from config / .env.
"""

from __future__ import annotations

import json
from typing import List, Optional

import config
from models.schemas import (
    Difference,
    EvaluatorOutput,
    ROARSections,
    SectionScores,
    VerifierOutput,
)
from prompts.templates import VERIFIER_PROMPT
from utils.json_parser import RobustJsonOutputParser
from utils.llm_factory import Backend, build_llm


class ROARVerifier:
    """
    Verifier backed by Qwen3.5-0.8B.

    Parameters
    ----------
    model:
        Model identifier / Azure deployment name.  Defaults to
        ``config.VERIFIER_MODEL``.  For ``"azure_openai"`` backend this is
        the Azure deployment name (e.g. ``"gpt_verifier"``).
    backend:
        ``"azure_openai"`` (default), ``"local_qwen"``, ``"openai"``, or
        ``"ollama"``.  See the module docstring for backend details.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        backend: Optional[Backend] = None,
        verifier_prompt=None,
    ) -> None:
        self._model = model or config.VERIFIER_MODEL
        self._backend: Backend = backend or config.VERIFIER_BACKEND  # type: ignore[assignment]
        self._verifier_prompt = verifier_prompt or VERIFIER_PROMPT
        self._chain = self._build_chain()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        sections: ROARSections,
        evaluator_output: EvaluatorOutput,
    ) -> VerifierOutput:
        """
        Verify the evaluator's scores against the original sections.

        Returns
        -------
        VerifierOutput
            Verified scores, consistency flag, and list of differences.
        """
        llama_output_str = json.dumps(
            {
                "scores":    evaluator_output.scores.model_dump(),
                "reasoning": evaluator_output.reasoning.model_dump(),
            },
            indent=2,
        )

        raw: dict = self._chain.invoke(
            {
                "plo":          sections.plo,
                "methods":      sections.methods,
                "results":      sections.results,
                "plan":         sections.plan,
                "llama_output": llama_output_str,
            }
        )
        return self._parse(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_chain(self):
        """
        Build the LangChain LCEL chain appropriate for the active backend.

        "local_qwen"
            The factory handles all Qwen3.5 settings (endpoint, sampling
            params, vLLM extra_body) – this branch is a one-liner.

        "openai"
            Passes VERIFIER_API_BASE / VERIFIER_API_KEY from config.
            Also forwards Qwen3.5 sampling params and vLLM-compatible
            extra_body fields (top_k, min_p, repetition_penalty).

        "ollama"
            Uses Ollama's local server; no extra params needed.
        """
        if self._backend == "azure_openai":
            # ── Azure OpenAI verifier (gpt-4.1 or o4-mini) ────────────────
            # VERIFIER_AZURE_TEMPERATURE in .env controls the temperature:
            #   GPT-class verifiers (gpt-4.1, gpt-5.4-mini): 0.0 (deterministic)
            #   o4-mini-verifier (reasoning model):           1.0 (required)
            llm = build_llm(
                self._model, self._backend,
                azure_temperature=config.VERIFIER_AZURE_TEMPERATURE,
                azure_max_tokens=config.VERIFIER_AZURE_MAX_TOKENS,
            )

        elif self._backend == "local_qwen":
            # ── Local HF Transformers server (Qwen3.5-0.8B) ───────────────
            # Kept for reference. To use: set VERIFIER_BACKEND=local_qwen in
            # .env and start: transformers serve Qwen/Qwen3.5-0.8B --port 8000
            llm = build_llm(self._model, self._backend)

        elif self._backend == "openai":
            # ── Remote OpenAI-compatible server (vLLM / SGLang) ───────────
            llm = build_llm(
                self._model,
                self._backend,
                openai_base_url=config.VERIFIER_API_BASE,
                openai_api_key=config.VERIFIER_API_KEY,
                openai_temperature=config.QWEN_TEMPERATURE,
                openai_top_p=config.QWEN_TOP_P,
                openai_presence_penalty=config.QWEN_PRESENCE_PENALTY,
                openai_max_tokens=config.QWEN_MAX_TOKENS,
                openai_extra_body={
                    "top_k": config.QWEN_TOP_K,
                    "min_p": config.QWEN_MIN_P,
                    "repetition_penalty": config.QWEN_REPETITION_PENALTY,
                },
            )

        else:
            # ── Ollama fallback ────────────────────────────────────────────
            llm = build_llm(self._model, self._backend)

        return self._verifier_prompt | llm | RobustJsonOutputParser()

    @staticmethod
    def _parse(raw: dict) -> VerifierOutput:
        vs_raw = raw.get("verified_scores", {})
        verified_scores = SectionScores(
            plo=int(vs_raw.get("plo", 0)),
            methods=int(vs_raw.get("methods", 0)),
            results=int(vs_raw.get("results", 0)),
            plan=int(vs_raw.get("plan", 0)),
        )

        diffs_raw: list = raw.get("differences", [])
        differences: List[Difference] = [
            Difference(
                field=str(d.get("field", "")),
                llama_score=int(d.get("llama_score", 0)),
                correct_score=int(d.get("correct_score", 0)),
                reason=str(d.get("reason", "")),
            )
            for d in diffs_raw
        ]

        consistent: bool = bool(raw.get("consistent", True))

        # Guard: if the model claims consistent=true but there are actual
        # score differences, override so the feedback loop fires correctly.
        if differences and any(d.llama_score != d.correct_score for d in differences):
            consistent = False

        return VerifierOutput(
            verified_scores=verified_scores,
            consistent=consistent,
            differences=differences,
        )
