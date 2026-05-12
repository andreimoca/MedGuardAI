"""Tests for the LangGraph clinical agent.

Two layers:
1. Unit tests for each tool, exercising real data files (no LLM).
2. Graph-level tests that stub the LLM with a scripted FakeChat model and
   assert the agent routes correctly to the right tool for the right query.
"""
import os
import sys
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Match the import strategy used by the existing test suite.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from agent.graph import build_agent_app
from agent.tools.allergies import check_allergies
from agent.tools.dosage import dosage_calculator
from agent.tools.emergency import emergency_classifier, is_emergency
from agent.tools.interactions import check_drug_interactions


# ----------------------------- TOOL UNIT TESTS ----------------------------- #

class TestEmergencyTool:
    def test_high_severity_overdose(self):
        result = emergency_classifier.invoke({"query": "I think I had an overdose"})
        assert result.startswith("HIGH")

    def test_high_severity_chest_pain(self):
        result = emergency_classifier.invoke({"query": "crushing chest pain right now"})
        assert result.startswith("HIGH")

    def test_normal_query_is_none(self):
        result = emergency_classifier.invoke({"query": "what is the dose of vitamin C?"})
        assert result.startswith("NONE")

    def test_helper_matches_tool(self):
        # The fast-path helper used by the FastAPI layer must agree with the tool.
        assert is_emergency("anaphylaxis after amoxicillin") is True
        assert is_emergency("headache treatment options") is False


class TestAllergyTool:
    def test_direct_allergy_match(self):
        result = check_allergies.invoke(
            {"drug": "naproxen", "patient_allergies": ["Naproxen"]}
        )
        assert "ALLERGY MATCH" in result

    def test_cross_reactivity_aspirin_to_naproxen(self):
        result = check_allergies.invoke(
            {"drug": "naproxen", "patient_allergies": ["Aspirin"]}
        )
        # Aspirin-NSAID cross reactivity is documented in naproxen's label.
        assert "CROSS-REACTIVITY" in result or "ALLERGY MATCH" in result

    def test_no_allergies(self):
        result = check_allergies.invoke({"drug": "naproxen", "patient_allergies": []})
        assert "no allergies" in result.lower()


class TestDosageTool:
    def test_dosage_returns_section(self):
        result = dosage_calculator.invoke(
            {"drug": "naproxen", "age": 45, "weight_kg": 80}
        )
        assert "DOSAGE" in result.upper() or "naproxen" in result.lower()

    def test_pediatric_flag(self):
        result = dosage_calculator.invoke(
            {"drug": "naproxen", "age": 10, "weight_kg": 35}
        )
        assert "PEDIATRIC" in result

    def test_geriatric_flag(self):
        result = dosage_calculator.invoke(
            {"drug": "naproxen", "age": 75, "weight_kg": 70}
        )
        assert "GERIATRIC" in result

    def test_unknown_drug(self):
        result = dosage_calculator.invoke(
            {"drug": "fictionium", "age": 30, "weight_kg": 70}
        )
        assert "No FDA label" in result


class TestInteractionsTool:
    def test_naproxen_warfarin_interaction(self):
        # Naproxen's label explicitly discusses warfarin / anticoagulants.
        result = check_drug_interactions.invoke(
            {"drug_a": "naproxen", "drug_b": "warfarin"}
        )
        assert "warfarin" in result.lower()

    def test_unknown_first_drug(self):
        result = check_drug_interactions.invoke(
            {"drug_a": "fictionium", "drug_b": "warfarin"}
        )
        assert "No FDA label" in result


# ----------------------------- GRAPH-LEVEL TESTS ----------------------------- #

class _ScriptedChatModel(BaseChatModel):
    """A minimal BaseChatModel that returns a pre-scripted list of AIMessages.

    The agent loop calls the LLM repeatedly; each call returns the next message
    in the script. Tool messages from `ToolNode` get appended to the state and
    inspected by the test.
    """

    script: list[AIMessage]
    call_index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.call_index >= len(self.script):
            # Default: a final answer with no tool calls so the loop terminates.
            msg = AIMessage(content="done")
        else:
            msg = self.script[self.call_index]
            self.call_index += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        # The graph calls `.bind_tools()`. Ignore tool schemas for the stub.
        return self


def _make_tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


class TestGraphRouting:
    def test_emergency_query_routes_to_emergency_classifier(self):
        script = [
            _make_tool_call("emergency_classifier", {"query": "crushing chest pain"}),
            AIMessage(
                content="[EMERGENCY] Please seek emergency medical assistance "
                "or call your local emergency number immediately."
            ),
        ]
        llm = _ScriptedChatModel(script=script)
        app = build_agent_app(llm=llm)

        result = app.invoke(
            {
                "messages": [HumanMessage(content="I have crushing chest pain")],
                "patient_context": {"age": 60, "weight": 80, "allergies": [], "conditions": []},
            }
        )

        tool_calls = [
            m for m in result["messages"] if isinstance(m, AIMessage) and m.tool_calls
        ]
        assert tool_calls, "expected at least one tool call"
        assert tool_calls[0].tool_calls[0]["name"] == "emergency_classifier"
        assert "[EMERGENCY]" in result["messages"][-1].content

    def test_dosage_query_routes_to_dosage_calculator(self):
        script = [
            _make_tool_call(
                "dosage_calculator", {"drug": "naproxen", "age": 45, "weight_kg": 80}
            ),
            AIMessage(content="The recommended dose is 250 mg twice daily."),
        ]
        llm = _ScriptedChatModel(script=script)
        app = build_agent_app(llm=llm)

        result = app.invoke(
            {
                "messages": [
                    HumanMessage(content="What dose of naproxen should I take?")
                ],
                "patient_context": {"age": 45, "weight": 80, "allergies": [], "conditions": []},
            }
        )

        names = [
            m.tool_calls[0]["name"]
            for m in result["messages"]
            if isinstance(m, AIMessage) and m.tool_calls
        ]
        assert "dosage_calculator" in names

    def test_interaction_query_routes_to_interactions_tool(self):
        script = [
            _make_tool_call(
                "check_drug_interactions",
                {"drug_a": "naproxen", "drug_b": "warfarin"},
            ),
            AIMessage(content="There is a documented interaction with warfarin."),
        ]
        llm = _ScriptedChatModel(script=script)
        app = build_agent_app(llm=llm)

        result = app.invoke(
            {
                "messages": [
                    HumanMessage(content="Can I take naproxen with warfarin?")
                ],
                "patient_context": {"age": 50, "weight": 75, "allergies": [], "conditions": []},
            }
        )

        names = [
            m.tool_calls[0]["name"]
            for m in result["messages"]
            if isinstance(m, AIMessage) and m.tool_calls
        ]
        assert "check_drug_interactions" in names

    def test_multi_step_routing(self):
        """Agent calls allergies then retrieval before answering."""
        script = [
            _make_tool_call(
                "check_allergies",
                {"drug": "naproxen", "patient_allergies": ["Aspirin"]},
                call_id="c1",
            ),
            _make_tool_call(
                "retrieve_drug_info", {"query": "naproxen for headache"}, call_id="c2"
            ),
            AIMessage(content="Given aspirin allergy, naproxen carries cross-reactivity risk."),
        ]
        llm = _ScriptedChatModel(script=script)
        app = build_agent_app(llm=llm)

        result = app.invoke(
            {
                "messages": [
                    HumanMessage(content="I'm allergic to aspirin. Can I take naproxen?")
                ],
                "patient_context": {
                    "age": 40,
                    "weight": 70,
                    "allergies": ["Aspirin"],
                    "conditions": [],
                },
            }
        )

        names = [
            m.tool_calls[0]["name"]
            for m in result["messages"]
            if isinstance(m, AIMessage) and m.tool_calls
        ]
        assert names == ["check_allergies", "retrieve_drug_info"]
        assert "cross-reactivity" in result["messages"][-1].content.lower()
