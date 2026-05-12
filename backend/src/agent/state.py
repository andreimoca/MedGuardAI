from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    patient_context: dict[str, Any]
    # Retrieved RAG context, populated by the upfront retrieve_node so the
    # agent always reasons over grounded FDA-label snippets even if it
    # never explicitly calls the retrieve_drug_info tool.
    retrieved_context: str
    # Phase 5 — safety pipeline state. Set by input_guard / output_guard so
    # the conditional edges can route correctly and so observers can inspect
    # what the guards decided after the run.
    input_blocked: bool
    safety_signals: dict[str, Any]
