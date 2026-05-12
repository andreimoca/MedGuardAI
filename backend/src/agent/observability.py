"""Logging callback for the clinical agent.

Attaches to the LangGraph invocation so every tool call shows up in the
backend logs with its inputs and a preview of its output. Enabled by default;
toggle off by setting MEDGUARD_TOOL_LOGS=0 in the environment.
"""
import logging
import os
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("medguard.tools")
if not logger.handlers:
    # Match the basicConfig used by api/main.py so the formatting is consistent.
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _truncate(text: str, limit: int = 240) -> str:
    text = " ".join(str(text).split())  # collapse whitespace
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


class ToolLoggingHandler(BaseCallbackHandler):
    """Log every tool start/end at INFO level."""

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or "unknown_tool"
        args = inputs if inputs else input_str
        logger.info(f"→ TOOL CALL  {name}({_truncate(str(args), 180)})")

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        text = output.content if hasattr(output, "content") else str(output)
        logger.info(f"← TOOL DONE  {_truncate(text)}")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        logger.error(f"✗ TOOL ERROR {type(error).__name__}: {error}")


def get_callbacks() -> list[BaseCallbackHandler]:
    if os.environ.get("MEDGUARD_TOOL_LOGS", "1") == "0":
        return []
    return [ToolLoggingHandler()]
