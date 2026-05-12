"""Public agent facade.

Historically this was a single linear RAG chain. It is now a thin wrapper
over the LangGraph tool-calling agent in `agent/graph.py`. Public methods are
preserved so the FastAPI layer at `api/main.py` and the existing test suite
keep working unchanged.
"""
import os
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from agent.graph import build_agent_app
from agent.observability import get_callbacks
from agent.tools.emergency import is_emergency

VECTOR_DB_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed", "chroma_db"
)

# Each agent<->tools round trip costs 2 graph steps, plus input_guard,
# retrieve, agent, and output_guard. A query that legitimately needs 4 tools
# (allergies + interactions + dosage + a re-retrieval) already costs ~12, so
# the old limit of 12 tripped on exactly that case. 30 leaves comfortable
# headroom while still bounding runaway tool loops.
RECURSION_LIMIT = 30


class ClinicalAgent:
    def __init__(self):
        # Vector store kept here so /health can still introspect retriever status.
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vectorstore = Chroma(
                persist_directory=VECTOR_DB_DIR, embedding_function=embeddings
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        except Exception as e:
            print(f"Failed to load Vector DB: {e}. Run build_vector_db.py.")
            self.vectorstore = None
            self.retriever = None

        try:
            self.app = build_agent_app()
            print("LangGraph agent compiled with 6 clinical tools.")
        except Exception as e:
            print(f"WARNING: Failed to compile LangGraph agent: {e}")
            self.app = None

    def format_patient_context(self, context: Dict[str, Any]) -> str:
        """Format the patient's parameters into a strict context string."""
        return f"""
Patient Profile:
- Age: {context.get('age', 'Unknown')}
- Weight: {context.get('weight', 'Unknown')} kg
- Known Allergies: {', '.join(context.get('allergies', ['None']))}
- Pre-existing Conditions: {', '.join(context.get('conditions', ['None']))}
"""

    def check_for_emergency(self, query: str) -> bool:
        """Lightweight pre-LLM guardrail; the agent itself calls a richer
        emergency_classifier tool, but this fast path lets the API short-circuit
        before any LLM round-trip."""
        return is_emergency(query)

    def process_query(self, user_query: str, patient_context: Dict[str, Any]) -> str:
        """Run the LangGraph agent and return the final assistant message."""
        if not self.app:
            return "System Error: Agent not initialized correctly."

        if self.check_for_emergency(user_query):
            return (
                "[EMERGENCY GUARDRAIL TRIPPED] Please seek emergency medical "
                "assistance immediately. Do not wait."
            )

        try:
            result = self.app.invoke(
                {
                    "messages": [HumanMessage(content=user_query)],
                    "patient_context": patient_context or {},
                },
                config={"recursion_limit": RECURSION_LIMIT, "callbacks": get_callbacks()},
            )
        except Exception as e:
            print(f"Agent execution failed: {e}")
            return (
                "I do not have sufficient information in the official documentation "
                "to answer this safely. Please consult a healthcare professional."
            )

        for message in reversed(result.get("messages", [])):
            if isinstance(message, AIMessage) and not message.tool_calls:
                return message.content or ""
        return "No response generated."

    def stream_query(self, user_query: str, patient_context: Dict[str, Any]):
        """Run the agent and yield progress events as plain dicts.

        Yields:
          {"type": "tool", "name": <tool name>, "args": {...}} for every tool
          the agent invokes, in order, and finally exactly one terminal event:
          {"type": "answer", "answer": <str>, "status": "success"|"emergency"|"error"}.

        This powers the streaming UI so the user can see "checking interaction
        between X and Y…" while the agent works, then the message resolves to
        the final answer.
        """
        if not self.app:
            yield {"type": "answer", "answer": "System Error: Agent not initialized correctly.", "status": "error"}
            return

        if self.check_for_emergency(user_query):
            yield {
                "type": "answer",
                "answer": (
                    "[EMERGENCY GUARDRAIL TRIPPED] Please seek emergency medical "
                    "assistance immediately. Do not wait."
                ),
                "status": "emergency",
            }
            return

        collected: list = []
        try:
            for chunk in self.app.stream(
                {
                    "messages": [HumanMessage(content=user_query)],
                    "patient_context": patient_context or {},
                },
                config={"recursion_limit": RECURSION_LIMIT, "callbacks": get_callbacks()},
                stream_mode="updates",
            ):
                for node_update in chunk.values():
                    if not isinstance(node_update, dict):
                        continue
                    for message in node_update.get("messages", []) or []:
                        collected.append(message)
                        if isinstance(message, AIMessage) and message.tool_calls:
                            for call in message.tool_calls:
                                yield {
                                    "type": "tool",
                                    "name": call.get("name", "tool"),
                                    "args": call.get("args", {}) or {},
                                }
        except Exception as e:
            print(f"Agent execution failed: {e}")
            yield {
                "type": "answer",
                "answer": (
                    "I do not have sufficient information in the official documentation "
                    "to answer this safely. Please consult a healthcare professional."
                ),
                "status": "success",
            }
            return

        for message in reversed(collected):
            if isinstance(message, AIMessage) and not message.tool_calls:
                yield {"type": "answer", "answer": message.content or "", "status": "success"}
                return
        yield {"type": "answer", "answer": "No response generated.", "status": "success"}


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    agent = ClinicalAgent()
    context = {
        "age": 45,
        "weight": 80,
        "allergies": ["Penicillin"],
        "conditions": ["Hypertension"],
    }
    query = "Is it safe for me to take ibuprofen for a headache?"
    print(f"\nUser Query: {query}")
    print(agent.process_query(query, context))
