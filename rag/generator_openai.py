from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI


class OpenAIGenerator:
  

    def __init__(self, model: str = "gpt-5-mini", api_key: Optional[str] = None):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def _system(self, kind: str) -> str:
        base = (
            "You are an information extraction system for annual reports.\n"
            "Use ONLY the provided CONTEXT.\n"
            "Do not guess.\n"
            "Do not add explanations.\n"
        )
        if kind == "boolean":
            return base + "Output must be exactly: True or False."
        if kind == "number":
            return base + "Output must be exactly: one number (digits with optional decimal) OR N/A."
        if kind in ("name", "names"):
            return base + "Output must be exactly: the requested name(s) OR N/A. No extra text."
        return base + "Output must be only the answer."

    def _instructions(self, kind: str) -> str:
        if kind == "boolean":
            return (
                "Return ONLY 'True' or 'False'. "
                "If there is no mention in the context, return 'False'."
            )
        if kind == "number":
            return (
                "Return ONLY ONE number (e.g., 123 or 123.45) OR 'N/A'. "
                "No currency symbols, no units, no words."
            )
        if kind == "name":
            return "Return ONLY the exact name OR 'N/A'. No extra words."
        if kind == "names":
            return (
                "Return ONLY the exact titles/names separated by commas OR 'N/A'. "
                "No bullets, no numbering, no extra words."
            )
        return "Return ONLY the answer."

    def answer(
        self,
        question_text: str,
        kind: str,
        context: str,
        model_override: str | None = None,
    ) -> str:
        prompt = (
            f"QUESTION:\n{question_text}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"INSTRUCTIONS:\n{self._instructions(kind)}\n"
        )

        resp = self.client.responses.create(
            model=model_override or self.model,
            input=[
                {"role": "system", "content": self._system(kind)},
                {"role": "user", "content": prompt},
            ],
      
        )
        return (resp.output_text or "").strip()

