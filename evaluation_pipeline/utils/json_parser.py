"""
Robust JSON output parser for LLM responses.

LLMs sometimes:
  - Wrap JSON in markdown code fences
  - Add preamble text before the JSON ("Here is the output: {...")
  - Mix a sentence fragment into the opening brace ("{\" happy to help...")
  - Use trailing commas or single quotes

This parser tries multiple strategies to extract valid JSON.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.output_parsers import BaseOutputParser


class RobustJsonOutputParser(BaseOutputParser[dict[str, Any]]):
    """Parse LLM output to a Python dict, tolerating common formatting noise."""

    def parse(self, text: str) -> dict[str, Any]:
        cleaned = self._strip_noise(text)

        # Strategy 1: parse the whole cleaned string
        result = self._try_parse(cleaned)
        if result is not None:
            return result

        # Strategy 2: find EVERY { position and try each as a JSON start
        # (handles preamble text mixed before the JSON object)
        for start in self._find_all_brace_starts(cleaned):
            end = cleaned.rfind("}") + 1
            if end > start:
                result = self._try_parse(cleaned[start:end])
                if result is not None:
                    return result

        # Strategy 3: attempt light repairs (trailing commas, single quotes)
        repaired = self._repair(cleaned)
        result = self._try_parse(repaired)
        if result is not None:
            return result

        raise ValueError(
            f"Could not parse JSON from LLM response.\n"
            f"Cleaned text:\n{cleaned[:500]}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_noise(text: str) -> str:
        """Remove markdown fences, leading/trailing whitespace, and preambles."""
        # Remove ```json ... ``` or ``` ... ``` blocks (keep interior content)
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        # Remove common preamble phrases up to and including their line
        text = re.sub(
            r"(?i)^(here is|here'?s|output|result|below is)[^\n]*\n",
            "",
            text.strip(),
        )
        return text.strip()

    @staticmethod
    def _find_all_brace_starts(text: str) -> list[int]:
        """Return all positions of '{' in text, ordered from last to first."""
        positions = [i for i, ch in enumerate(text) if ch == "{"]
        # Try later starts first (more likely to skip noisy preamble)
        return list(reversed(positions))

    @staticmethod
    def _try_parse(text: str) -> dict[str, Any] | None:
        """Return parsed dict or None if parsing fails."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _repair(text: str) -> str:
        """
        Apply light repairs to near-valid JSON:
          - Remove trailing commas before ] or }
          - Replace single-quoted string delimiters with double quotes
          - Remove JavaScript-style // comments
        """
        # Remove JS single-line comments
        text = re.sub(r"//[^\n]*", "", text)
        # Remove trailing commas before closing brackets
        text = re.sub(r",\s*([}\]])", r"\1", text)
        # Replace single-quote string delimiters (heuristic — may break real apostrophes)
        # Only replace when clearly acting as a key/value delimiter
        text = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', text)  # keys
        text = re.sub(r"(:\s*)'([^']*)'", r'\1"\2"', text)  # values
        return text

    @property
    def _type(self) -> str:  # noqa: D102
        return "robust_json"
