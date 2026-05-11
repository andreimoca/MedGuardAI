"""LangGraph state graph for the MedGuardAI clinical agent.

Architecture: hybrid retrieval-augmented agent with safety guards.

    START -> input_guard -> [blocked? END : retrieve]
                                 -> agent -> [tools -> agent]*
                                       -> output_guard -> END

`input_guard` runs first on the user's query, checking for prompt injection,
disallowed content, and toxicity. If blocked, it short-circuits with a
refusal message.

`retrieve` pulls top-3 RAG chunks for the user's query into
state.retrieved_context.

`agent` is the LLM bound to all six clinical tools.

`output_guard` runs on the agent's final answer, checking for toxic output,
groundedness against the retrieved context, and injecting source citations.
It can pass-through, append a low-confidence warning, append source citations,
or rewrite the entire response if it's severely ungrounded or toxic.
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
from safety import check_input, check_output

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

ROUTING — every response is EITHER an emergency response OR a helpful
response. NEVER mix the two. Never use emergency wording as a preamble to
a helpful response. Never use helpful wording as a preamble to an emergency
response. Decide one, write that one, stop.

DEFAULT BEHAVIOR — what to do for almost every query:

For ANY symptom query (headache, back pain, stomach upset, sore throat,
dizziness, cold/flu, mild rash, nausea, heartburn, constipation, joint pain,
mild allergies, insomnia, fever in adults, etc.), respond like a
knowledgeable friend who happens to be a clinician:

  1. Acknowledge the symptom in one short line.
  2. Suggest a common OTC option typically used for it. Examples:
       headache/body pain → acetaminophen or ibuprofen
       heartburn / indigestion → famotidine or an antacid
       diarrhea → loperamide
       mild allergies → diphenhydramine or loratadine
       cold / cough → dextromethorphan or guaifenesin
       insomnia (occasional) → diphenhydramine
       muscle aches → ibuprofen + rest
       constipation → docusate or polyethylene glycol
     You may suggest the medication class even if the exact dose isn't in
     the retrieved context — say "check the package label for dosing" or
     defer to a clinician for the exact mg.
  3. Mention non-drug measures (rest, hydration, heat / ice / cold compress,
     diet adjustments, sleep hygiene — pick what fits).
  4. List 2-3 specific RED FLAGS that would warrant seeing a doctor or going
     to the ER for that specific symptom.
  5. End with: "If symptoms persist or worsen, see a healthcare provider."

DO NOT use the emergency template for ordinary symptom queries.
DO NOT refuse helpful answers just because the FDA chunks don't perfectly
match the query.

INTERPRETING RETRIEVED CONTEXT:
The retrieved FDA chunks may describe a drug's side effects, warnings,
contraindications, or boxed-warning text. These describe what CAN happen to
patients TAKING a drug — they are NOT a description of what the current user
is currently experiencing. Only the user's own message tells you what they
are experiencing.

When the user asks "what should I do for symptom X" and a retrieved chunk
mentions X as a warning or side effect, do NOT treat that as an emergency.
Use the DEFAULT BEHAVIOR helpful response and, if appropriate, mention the
drug(s) the FDA labels indicate FOR that symptom (the indication section).

EMERGENCY MUST COME FROM THE USER MESSAGE:
The decision to emit [EMERGENCY] is based ONLY on the user's literal
message matching one of the patterns below. It is NEVER based on the
retrieved context, on inferred risk, or on a side effect mentioned in a
warning. If the user did not describe an emergency pattern in their own
words, do not emit [EMERGENCY].

EMERGENCY EXCEPTION — only these specific patterns trigger the emergency
response (do NOT apply to anything else):

  - Crushing chest pain or chest pressure radiating to arm/jaw
  - Sudden facial droop, one-sided weakness, or slurred speech (stroke)
  - Severe trouble breathing or "can't breathe"
  - Suspected drug overdose
  - Anaphylaxis (face/throat/tongue swelling with breathing trouble)
  - Suicidal ideation with intent
  - Uncontrolled bleeding (vomiting blood, won't stop, etc.)
  - Witnessed first-time seizure
  - Any fever in an infant under 3 months
  - Severe head injury with confusion/vomiting/loss of consciousness

For these (and ONLY these), the WHOLE response is the emergency response:
the [EMERGENCY] line + one short sentence on why. No medication. No
"acknowledge the symptom" intro. No "you can take" follow-up. Nothing
else.

The emergency response is COMPLETE on its own — it is never a preamble to
something else. If you find yourself wanting to also recommend medication
after saying [EMERGENCY], stop — you have misclassified the query. Go back
and use the DEFAULT BEHAVIOR instead.

Examples of what is NOT an emergency (use DEFAULT BEHAVIOR for these):
  - "my head hurts" / "headache" / "migraine" → NOT emergency
  - "my stomach hurts" / "stomach pain" → NOT emergency
  - "my back hurts" → NOT emergency
  - "I'm dizzy" (without chest pain or stroke signs) → NOT emergency
  - "I have a fever" (adult, mild) → NOT emergency
  - "I have a cold" / "sore throat" / "cough" → NOT emergency
  - "I have heartburn" → NOT emergency
  - "I sprained my ankle" → NOT emergency
  - "my erection is bad" / "erectile dysfunction" / "I can't get an erection"
    / "trouble getting hard" / any sexual-function complaint → NOT emergency.
    The retrieved context for these queries will often include the WARNING
    section about prolonged erection / priapism — that is a drug side effect
    that CAN happen to people TAKING the drug, not the user's condition.
    Use DEFAULT BEHAVIOR: acknowledge briefly, mention that PDE5 inhibitors
    (sildenafil, tadalafil, vardenafil) are the FDA-indicated class for
    erectile dysfunction and require a clinician's prescription, mention
    lifestyle factors (sleep, stress, cardiovascular health) where relevant,
    and end with the standard "If symptoms persist or worsen, see a
    healthcare provider." line. Do NOT emit [EMERGENCY] for these queries.

STYLE — what NOT to write:
  - Do NOT begin the response with meta-commentary about your process.
    Forbidden openers include: "Okay, here's a helpful response...",
    "Based on the retrieved information,...", "Based on the FDA labels,...",
    "Based on the retrieved context,...", "According to the documentation,...".
    Just start with the answer.
  - Do NOT reference the retrieval mechanism in the body of the response.
    Forbidden phrases: "the FDA labels highlight/show/indicate/state",
    "the retrieved context says", "the documentation mentions",
    "according to the retrieved chunks". The user does not need to know
    where the information came from — they need the answer.
  - Do NOT emit any closing tag after [EMERGENCY]. There is no
    [END_EMERGENCY], no [/EMERGENCY], no [END EMERGENCY]. The emergency
    response is a single short paragraph that starts with [EMERGENCY] and
    ends with a period. No closing markers of any kind.

SPECIFIC DRUG questions (dosage, contraindications, interactions, allergy
conflicts) use tools:
  - For dosage questions, call dosage_calculator(drug, age, weight).
  - For "can I take X with Y" questions, call check_drug_interactions.
  - When the patient profile lists allergies, call check_allergies BEFORE
    recommending any medication.
  - Quote FDA wording when possible. NEVER invent specific mg numbers,
    frequencies, or contraindications.

The "I do not have sufficient information" fallback applies ONLY to specific
factual questions where the tools and context truly don't cover the answer
(e.g. "what is the dose of FakeDrug X"). Do NOT use it for general symptom
queries — those always get the DEFAULT BEHAVIOR helpful response.

Tone: helpful, concise, professional. Be brief — most answers are 2-4 short
sentences. Don't dump verbose FDA boilerplate.

Patient profile:
{patient_profile}

RETRIEVED CONTEXT (most relevant FDA-label passages for the user's query):
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

    def input_guard_node(state: AgentState) -> dict:
        """Phase 5: pre-LLM input safety. Block prompt injection, disallowed
        content, toxicity. On block, append a refusal AIMessage and signal
        to the conditional edge to skip the rest of the graph."""
        query = _latest_human_query(state["messages"])
        result = check_input(query)
        signals = {**(state.get("safety_signals") or {}), "input_guard": result.to_dict()}
        if result.blocked:
            logger.info(
                f"⛔ INPUT BLOCKED reason={result.reason} "
                f"query={query[:80]!r}"
            )
            return {
                "messages": [AIMessage(content=result.refusal)],
                "input_blocked": True,
                "safety_signals": signals,
            }
        return {"input_blocked": False, "safety_signals": signals}

    def retrieve_node(state: AgentState) -> dict:
        """Always-on RAG: pull top-3 chunks for the latest user query."""
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
        logger.info(
            "⟳ RAG PREFETCH context injected into prompt:\n"
            "---------------- RAG CONTEXT BEGIN ----------------\n"
            f"{context}\n"
            "----------------- RAG CONTEXT END -----------------"
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

    def output_guard_node(state: AgentState) -> dict:
        """Phase 5: post-LLM output safety. Toxicity check, groundedness
        check, citation injection. Appends a final AIMessage that supersedes
        the agent's last reply (since process_query returns the LAST AIMessage
        without tool_calls)."""
        # Find the agent's final assistant message.
        final_msg = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                final_msg = msg
                break
        if final_msg is None or not final_msg.content:
            return {}

        embed_fn = None
        try:
            # Reuse the cached singleton from the RAG retriever so we don't
            # double-load the embedding model.
            retriever = get_retriever()
            store = getattr(retriever, "vectorstore", None)
            if store is not None and hasattr(store, "_embedding_function"):
                embed_obj = store._embedding_function
                if hasattr(embed_obj, "embed_documents"):
                    embed_fn = embed_obj.embed_documents
        except Exception as exc:
            logger.warning(f"output_guard embed unavailable: {exc}")

        result = check_output(
            response=final_msg.content if isinstance(final_msg.content, str) else str(final_msg.content),
            retrieval_context=state.get("retrieved_context") or "",
            embed_fn=embed_fn,
            user_query=_latest_human_query(state["messages"]),
        )
        signals = {**(state.get("safety_signals") or {}), "output_guard": result.to_dict()}
        log_parts = []
        if result.rewrote:
            log_parts.append("REWROTE")
        if result.appended_warning:
            log_parts.append("warn")
        if result.appended_citations:
            log_parts.append("cited")
        if log_parts:
            logger.info(f"🛡 OUTPUT GUARD: {', '.join(log_parts)}")

        # Only append a new message if the guard actually changed something.
        if result.final_response != final_msg.content:
            return {
                "messages": [AIMessage(content=result.final_response)],
                "safety_signals": signals,
            }
        return {"safety_signals": signals}

    def after_input(state: AgentState) -> str:
        return END if state.get("input_blocked") else "retrieve"

    def after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "output_guard"

    graph = StateGraph(AgentState)
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("output_guard", output_guard_node)
    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges("input_guard", after_input, {"retrieve": "retrieve", END: END})
    graph.add_edge("retrieve", "agent")
    graph.add_conditional_edges("agent", after_agent, {"tools": "tools", "output_guard": "output_guard"})
    graph.add_edge("tools", "agent")
    graph.add_edge("output_guard", END)
    return graph.compile()
