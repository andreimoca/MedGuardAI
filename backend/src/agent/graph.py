"""LangGraph state graph for the MedGuardAI clinical agent.

Architecture: hybrid retrieval-augmented agent.

    START -> retrieve -> agent -> [tools -> agent]* -> END

`retrieve` runs once upfront and pulls top-2 RAG chunks for the user's query
into state.retrieved_context. `agent` is the LLM bound to all six clinical
tools, with the retrieved context injected into the system prompt so it
always reasons over grounded FDA-label snippets. `tools` executes any tool
calls the LLM emits; the loop continues until the LLM emits a final answer
with no further tool calls.
"""
import logging
import os

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.tools import ALL_TOOLS
from agent.tools.retrieval import format_docs, get_retriever

logger = logging.getLogger("medguard.tools")

SYSTEM_PROMPT = """You are MedGuardAI, a safety-first clinical assistant.

You always have RAG context retrieved from the local FDA-label vector store
for the user's most recent question, shown below as RETRIEVED CONTEXT. Use it
as the PRIMARY source of truth.

You also have these tools and SHOULD use them when the question warrants:
- retrieve_drug_info(query): additional semantic search if you need context on
  a different drug than the one(s) already retrieved.
- check_drug_interactions(drug_a, drug_b): documented interaction between two drugs.
- dosage_calculator(drug, age, weight_kg): FDA dosage section + patient flags.
- check_allergies(drug, patient_allergies): allergy & cross-reactivity check.
- emergency_classifier(query): tier the user's symptoms (HIGH / MEDIUM / NONE).
- fetch_openfda_live(drug_name): live FDA lookup for drugs not in the local store.

Decision policy:
1. If the user describes any acute symptom, call emergency_classifier first.
   On HIGH or MEDIUM, stop and reply with exactly:
   "[EMERGENCY] Please seek emergency medical assistance or call your local
   emergency number immediately."
2. For dosage questions, call dosage_calculator with the patient's age/weight.
3. For "can I mix X and Y" or "I am on Y, can I take X" questions, call
   check_drug_interactions.
4. For ANY medication recommendation when the patient profile lists allergies,
   call check_allergies BEFORE giving the recommendation.
5. NEVER hallucinate dosages, contraindications, or interactions. If neither
   the retrieved context nor the tools answer the question, say:
   "I do not have sufficient information in the official documentation to
   answer this safely. Please consult a healthcare professional."
6. Cite the drug name(s) you used as your source in the final answer.

Patient profile:
{patient_profile}

RETRIEVED CONTEXT (from FDA labels, top-2 semantic matches for the user's query):
{retrieved_context}
"""


def _format_patient_profile(ctx: dict) -> str:
    return (
        f"- Age: {ctx.get('age', 'Unknown')} years\n"
        f"- Weight: {ctx.get('weight', 'Unknown')} kg\n"
        f"- Known allergies: {', '.join(ctx.get('allergies') or ['None'])}\n"
        f"- Pre-existing conditions: {', '.join(ctx.get('conditions') or ['None'])}"
    )


def _latest_human_query(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _build_llm() -> ChatOpenAI:
    base_url = os.environ.get("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1")
    model = os.environ.get("LOCAL_LLM_MODEL", "gemma-3-4b")
    return ChatOpenAI(base_url=base_url, api_key="not-needed", model=model, temperature=0.0)


def build_agent_app(llm: ChatOpenAI | None = None):
    """Compile the LangGraph state graph. `llm` is injectable for testing."""
    llm = llm or _build_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def retrieve_node(state: AgentState) -> dict:
        """Always-on RAG: pull top-2 chunks for the latest user query."""
        # Skip re-retrieval if state already has context (e.g. follow-up turn).
        if state.get("retrieved_context"):
            return {}
        query = _latest_human_query(state["messages"])
        if not query:
            return {"retrieved_context": ""}
        try:
            retriever = get_retriever()
            docs = retriever.invoke(query)
            context = format_docs(docs) or "(no relevant documents found)"
        except Exception as exc:
            logger.warning(f"upfront retrieval failed: {exc}")
            context = "(retrieval unavailable)"
        logger.info(
            f"⟳ RAG PREFETCH for query={query[:80]!r} → "
            f"{len(context)} chars of context"
        )
        return {"retrieved_context": context}

    def agent_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        messages = state["messages"]
        profile = _format_patient_profile(state.get("patient_context") or {})
        context = state.get("retrieved_context") or "(none)"
        system_msg = SystemMessage(
            content=SYSTEM_PROMPT.format(
                patient_profile=profile, retrieved_context=context
            )
        )
        response = llm_with_tools.invoke([system_msg, *messages])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()
