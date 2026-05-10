"""DeepEval LLM-as-judge wrapper.

DeepEval's built-in metrics (Faithfulness, AnswerRelevancy, Hallucination,
GEval, etc.) call an LLM under the hood to score responses. By default they
use OpenAI. To use anything else — DeepSeek, our local LM Studio Gemma, or
another OpenAI-compatible endpoint — we have to wrap the client in DeepEval's
`DeepEvalBaseLLM` interface.

We expose a single factory `make_judge()` that picks the most capable judge
available, in this priority:

  1. DeepSeek API  (env: DATASET_LLM_URL + DATASET_LLM_KEY)
  2. Local LM Studio (env: LOCAL_LLM_URL)

Using DeepSeek as the judge is strongly preferred — judging our own
fine-tuned Gemma with the same Gemma is methodologically circular ("the
model agrees with itself"). DeepSeek-V3 is a much stronger judge than a 4B
local model and produces more reliable scores.
"""
from __future__ import annotations

import json
import os
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI


class OpenAICompatibleJudge(DeepEvalBaseLLM):
    """Adapter from any OpenAI-compatible chat endpoint to DeepEvalBaseLLM."""

    def __init__(self, base_url: str, api_key: str, model: str, label: str):
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
        self._model = model
        self._label = label

    def load_model(self):  # required by DeepEvalBaseLLM
        return self._client

    def get_model_name(self) -> str:
        return self._label

    def _chat(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
        )
        return (resp.choices[0].message.content or "").strip()

    def generate(self, prompt: str, schema: Any = None) -> Any:
        text = self._chat(prompt)
        if schema is None:
            return text
        # DeepEval's newer metrics pass a pydantic schema and expect a parsed
        # instance back. Attempt JSON-load + schema validate; fall back to raw
        # text if the model didn't comply.
        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```", 2)[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()
            return schema(**json.loads(cleaned))
        except Exception:
            return text

    async def a_generate(self, prompt: str, schema: Any = None) -> Any:
        # DeepEval calls a_generate for async metrics; we just call sync since
        # the OpenAI client is synchronous here. (Could swap for AsyncOpenAI
        # if eval throughput becomes a bottleneck.)
        return self.generate(prompt, schema)


def make_judge() -> OpenAICompatibleJudge:
    """Pick the best available judge based on env vars."""
    deepseek_url = os.environ.get("DATASET_LLM_URL")
    deepseek_key = os.environ.get("DATASET_LLM_KEY")
    deepseek_model = os.environ.get("DATASET_LLM_MODEL", "deepseek-chat")
    if deepseek_url and deepseek_key:
        return OpenAICompatibleJudge(
            base_url=deepseek_url,
            api_key=deepseek_key,
            model=deepseek_model,
            label=f"judge-{deepseek_model}",
        )

    local_url = os.environ.get("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1")
    local_model = os.environ.get("LOCAL_LLM_MODEL", "gemma-3-4b")
    return OpenAICompatibleJudge(
        base_url=local_url,
        api_key="not-needed",
        model=local_model,
        label=f"judge-{local_model}",
    )
