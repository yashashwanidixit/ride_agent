"""Unit tests for the Stage 6 sequential agent loop in Stage5Orchestrator.

These tests prove that Stage5Orchestrator:
- depends only on the LLMProvider abstraction (never a concrete provider)
- only reaches the underlying ride tools through MCPClient - never directly
- runs a generic loop whose sequence is entirely driven by what the
  injected LLMProvider returns, with no hardcoded tool ordering

All LLM interaction is mocked via ScriptedLLMProvider/FakeLLMProvider,
which implement the LLMProvider abstraction directly - none of these
tests instantiate OllamaProvider or a real OpenAICompatibleProvider.
MCPClient is mocked too (MagicMock), so none of these tests spawn the
real MCP server or hit real services. That real path is exercised
manually via the CLI (see the accompanying instructions).
"""

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from ride_agent.agent.llm.base import LLMProvider, ToolCall, ToolDefinition
from ride_agent.agent.orchestrator import Stage5Orchestrator


class FakeTool:
    """Stand-in for an MCP-discovered tool object."""

    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.input_schema = input_schema


GEOCODE_TOOL = FakeTool(
    name="geocode_location",
    description="Converts a place name into coordinates.",
    input_schema={
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
)
RIDE_TIME_TOOL = FakeTool(
    name="get_ride_time",
    description="Ride time between two coordinate pairs.",
    input_schema={"type": "object", "properties": {}},
)
RIDE_PRICE_TOOL = FakeTool(
    name="get_ride_price",
    description="Estimated fare from distance and duration.",
    input_schema={"type": "object", "properties": {}},
)

ALL_TOOLS = [GEOCODE_TOOL, RIDE_TIME_TOOL, RIDE_PRICE_TOOL]


def make_fake_mcp_client(tools, call_tool_side_effect=None, call_tool_return=None):
    mcp_client = MagicMock()
    mcp_client.discover_tools.return_value = tools
    if call_tool_side_effect is not None:
        mcp_client.call_tool.side_effect = call_tool_side_effect
    else:
        mcp_client.call_tool.return_value = call_tool_return or {"success": True}
    return mcp_client


class ScriptedLLMProvider(LLMProvider):
    """LLMProvider whose responses are scripted in advance.

    Each scripted entry is either {"tool_call": ToolCall(...)} or
    {"final_answer": "..."}. This lets tests drive the orchestrator
    through a specific sequence of tool calls / final answers without
    depending on Ollama's or OpenAI's actual wire format - proving the
    orchestrator works against the LLMProvider abstraction alone.
    """

    def __init__(self, script: List[Dict[str, Any]], repeat_last: bool = False):
        self._script = list(script)
        self._repeat_last = repeat_last
        self.call_count = 0
        self.received_messages_snapshots: List[List[Dict[str, Any]]] = []
        self.closed = False

    def is_available(self) -> bool:
        return True

    def call_with_tools(self, messages, tools):
        self.call_count += 1
        self.received_messages_snapshots.append([dict(m) for m in messages])

        if not self._script:
            if self._repeat_last:
                return self._last_response
            raise AssertionError("ScriptedLLMProvider ran out of scripted responses")

        response = self._script.pop(0)
        self._last_response = response
        return response

    def extract_tool_call(self, response) -> Optional[ToolCall]:
        return response.get("tool_call")

    def extract_assistant_message(self, response) -> Dict[str, Any]:
        tool_call = response.get("tool_call")
        if tool_call:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "function": {"name": tool_call.name, "arguments": tool_call.arguments},
                    }
                ],
            }
        return {"role": "assistant", "content": response.get("final_answer", "")}

    def format_tool_result_message(self, tool_call, result) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        }

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------
# TEST 1 - single tool call then final answer
# ---------------------------------------------------------------------

def test_single_tool_call_then_final_answer():
    script = [
        {"tool_call": ToolCall(name="geocode_location", arguments={"location": "IIT Patna"}, id="1")},
        {"final_answer": "IIT Patna is at roughly 25.59N, 85.14E."},
    ]
    provider = ScriptedLLMProvider(script)
    mcp_client = make_fake_mcp_client(
        tools=[GEOCODE_TOOL],
        call_tool_return={"success": True, "latitude": 25.59, "longitude": 85.14},
    )

    orchestrator = Stage5Orchestrator(llm_provider=provider, mcp_client=mcp_client)
    result = orchestrator.process_user_request("Find the coordinates of IIT Patna.")

    assert provider.call_count == 2  # LLM called twice
    mcp_client.call_tool.assert_called_once_with(
        "geocode_location", {"location": "IIT Patna"}
    )
    assert result == "IIT Patna is at roughly 25.59N, 85.14E."


# ---------------------------------------------------------------------
# TEST 2 - multiple sequential tool calls, order driven entirely by the LLM
# ---------------------------------------------------------------------

def test_multiple_sequential_tool_calls_in_llm_determined_order():
    script = [
        {"tool_call": ToolCall(name="geocode_location", arguments={"location": "IIT Patna"}, id="1")},
        {"tool_call": ToolCall(name="geocode_location", arguments={"location": "Patna Airport"}, id="2")},
        {"tool_call": ToolCall(
            name="get_ride_time",
            arguments={
                "pickup_latitude": 25.59, "pickup_longitude": 85.14,
                "destination_latitude": 25.59, "destination_longitude": 85.09,
            },
            id="3",
        )},
        {"tool_call": ToolCall(
            name="get_ride_price",
            arguments={"distance_km": 10, "duration_minutes": 25},
            id="4",
        )},
        {"final_answer": "It's about 10km, 25 minutes, and roughly Rs. 425."},
    ]
    provider = ScriptedLLMProvider(script)

    call_tool_results = [
        {"success": True, "latitude": 25.59, "longitude": 85.14},
        {"success": True, "latitude": 25.59, "longitude": 85.09},
        {"success": True, "distance_km": 10, "duration_minutes": 25},
        {"success": True, "fare": 425},
    ]
    mcp_client = make_fake_mcp_client(tools=ALL_TOOLS, call_tool_side_effect=call_tool_results)

    orchestrator = Stage5Orchestrator(llm_provider=provider, mcp_client=mcp_client)
    result = orchestrator.process_user_request(
        "I want to go from IIT Patna to Patna airport. Distance, time, and fare please."
    )

    assert provider.call_count == 5  # initial + one more after each of the 4 tool results

    # The exact order was dictated entirely by the scripted LLM responses,
    # never hardcoded in the orchestrator.
    actual_calls = [call.args[0] for call in mcp_client.call_tool.call_args_list]
    assert actual_calls == [
        "geocode_location",
        "geocode_location",
        "get_ride_time",
        "get_ride_price",
    ]
    assert result == "It's about 10km, 25 minutes, and roughly Rs. 425."


# ---------------------------------------------------------------------
# TEST 3 - no tool call at all
# ---------------------------------------------------------------------

def test_no_tool_call_returns_final_answer_immediately():
    script = [{"final_answer": "I can help with rides, but that's not something I can look up."}]
    provider = ScriptedLLMProvider(script)
    mcp_client = make_fake_mcp_client(tools=ALL_TOOLS)

    orchestrator = Stage5Orchestrator(llm_provider=provider, mcp_client=mcp_client)
    result = orchestrator.process_user_request("What's the weather like?")

    assert provider.call_count == 1
    mcp_client.call_tool.assert_not_called()
    assert result == "I can help with rides, but that's not something I can look up."


# ---------------------------------------------------------------------
# TEST 4 - maximum tool-call safety limit
# ---------------------------------------------------------------------

def test_max_tool_call_limit_terminates_the_loop():
    infinite_tool_call = {
        "tool_call": ToolCall(name="geocode_location", arguments={"location": "X"}, id="loop")
    }
    provider = ScriptedLLMProvider([infinite_tool_call], repeat_last=True)
    mcp_client = make_fake_mcp_client(tools=[GEOCODE_TOOL], call_tool_return={"success": True})

    orchestrator = Stage5Orchestrator(
        llm_provider=provider, mcp_client=mcp_client, max_tool_calls=3
    )
    result = orchestrator.process_user_request("Keep going forever.")

    assert provider.call_count == 3
    assert mcp_client.call_tool.call_count == 3
    assert "allowed number of tool calls" in result


# ---------------------------------------------------------------------
# TEST 5 - MCP tool execution failure
# ---------------------------------------------------------------------

def test_tool_execution_failure_is_surfaced_not_hidden():
    script = [
        {"tool_call": ToolCall(name="geocode_location", arguments={"location": "Nowhere"}, id="1")},
        {"final_answer": "I couldn't find that location, could you clarify it?"},
    ]
    provider = ScriptedLLMProvider(script)
    mcp_client = make_fake_mcp_client(tools=[GEOCODE_TOOL])
    mcp_client.call_tool.side_effect = ValueError("Nominatim returned no results")

    orchestrator = Stage5Orchestrator(llm_provider=provider, mcp_client=mcp_client)
    result = orchestrator.process_user_request("Find the coordinates of Nowhere.")

    assert provider.call_count == 2
    # The failure must have been fed back to the LLM as an error, not
    # silently dropped or disguised as success.
    second_call_messages = provider.received_messages_snapshots[1]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "Nominatim returned no results" in tool_messages[0]["content"]
    assert '"success": false' in tool_messages[0]["content"]
    assert result == "I couldn't find that location, could you clarify it?"


# ---------------------------------------------------------------------
# Additional failure-mode coverage from the "Failure Handling" spec
# ---------------------------------------------------------------------

def test_unknown_tool_is_rejected_but_loop_continues():
    script = [
        {"tool_call": ToolCall(name="delete_everything", arguments={}, id="1")},
        {"final_answer": "I don't have a tool for that."},
    ]
    provider = ScriptedLLMProvider(script)
    mcp_client = make_fake_mcp_client(tools=[GEOCODE_TOOL])

    orchestrator = Stage5Orchestrator(llm_provider=provider, mcp_client=mcp_client)
    result = orchestrator.process_user_request("Do something malicious.")

    mcp_client.call_tool.assert_not_called()
    assert provider.call_count == 2
    assert result == "I don't have a tool for that."


def test_malformed_arguments_are_rejected_but_loop_continues():
    script = [
        {"tool_call": ToolCall(name="geocode_location", arguments="not-a-dict", id="1")},
        {"final_answer": "I hit an internal error with that request."},
    ]
    provider = ScriptedLLMProvider(script)
    mcp_client = make_fake_mcp_client(tools=[GEOCODE_TOOL])

    orchestrator = Stage5Orchestrator(llm_provider=provider, mcp_client=mcp_client)
    result = orchestrator.process_user_request("Find the coordinates of IIT Patna.")

    mcp_client.call_tool.assert_not_called()
    assert provider.call_count == 2
    assert result == "I hit an internal error with that request."


# ---------------------------------------------------------------------
# Provider-independence checks (carried over from Stage 5)
# ---------------------------------------------------------------------

def test_orchestrator_does_not_import_concrete_providers():
    import ride_agent.agent.orchestrator as orchestrator_module

    with open(orchestrator_module.__file__, encoding="utf-8") as f:
        source = f.read()

    assert "OllamaProvider" not in source
    assert "OpenAICompatibleProvider" not in source


def test_orchestrator_works_identically_with_a_mocked_openai_style_provider():
    """Swapping ScriptedLLMProvider for a MagicMock spec'd on
    OpenAICompatibleProvider must not require any change to
    orchestrator.py - only the injected provider changes."""

    from ride_agent.agent.llm.openai import OpenAICompatibleProvider

    mocked_openai_provider = MagicMock(spec=OpenAICompatibleProvider)
    mocked_openai_provider.is_available.return_value = True

    tool_call = ToolCall(name="geocode_location", arguments={"location": "IIT Patna"}, id="1")
    mocked_openai_provider.call_with_tools.side_effect = [
        {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": []}}]},
        {"choices": [{"message": {"role": "assistant", "content": "Here you go."}}]},
    ]
    mocked_openai_provider.extract_tool_call.side_effect = [tool_call, None]
    mocked_openai_provider.extract_assistant_message.side_effect = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "assistant", "content": "Here you go."},
    ]
    mocked_openai_provider.format_tool_result_message.return_value = {
        "role": "tool", "tool_call_id": "1", "content": "{}"
    }

    mcp_client = make_fake_mcp_client(tools=[GEOCODE_TOOL], call_tool_return={"success": True})

    orchestrator = Stage5Orchestrator(
        llm_provider=mocked_openai_provider, mcp_client=mcp_client
    )
    result = orchestrator.process_user_request("Find the coordinates of IIT Patna.")

    mcp_client.call_tool.assert_called_once_with(
        "geocode_location", {"location": "IIT Patna"}
    )
    assert result == "Here you go."