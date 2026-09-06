"""Unit tests for the Stage 5 orchestrator.

These tests prove that Stage5Orchestrator depends only on the
LLMProvider abstraction (never a concrete provider) and only reaches
the underlying ride tools through MCPClient - never directly.
"""

from typing import Any, List, Optional
from unittest.mock import MagicMock

from ride_agent.agent.llm.base import LLMProvider, ToolCall, ToolDefinition
from ride_agent.agent.orchestrator import Stage5Orchestrator


class FakeTool:
    """Stand-in for an MCP-discovered tool object."""

    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.input_schema = input_schema


class FakeLLMProvider(LLMProvider):
    """Minimal LLMProvider used to prove orchestrator provider-independence."""

    def __init__(self, tool_call: Optional[ToolCall]):
        self._tool_call = tool_call
        self.received_tools: List[ToolDefinition] = []
        self.closed = False

    def is_available(self) -> bool:
        return True

    def call_with_tools(self, user_message: str, tools: List[ToolDefinition]) -> Any:
        self.received_tools = tools
        return {"fake": True}

    def extract_tool_call(self, response: Any) -> Optional[ToolCall]:
        return self._tool_call

    def close(self) -> None:
        self.closed = True


def make_fake_mcp_client(tools, call_tool_return=None):
    mcp_client = MagicMock()
    mcp_client.discover_tools.return_value = tools
    mcp_client.call_tool.return_value = call_tool_return or {"success": True}
    return mcp_client


GEOCODE_TOOL = FakeTool(
    name="geocode_location",
    description="Converts a place name into coordinates.",
    input_schema={
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
)


def test_orchestrator_discovers_tools_and_passes_definitions_to_provider():
    tool_call = ToolCall(name="geocode_location", arguments={"location": "IIT Patna"})
    provider = FakeLLMProvider(tool_call=tool_call)
    mcp_client = make_fake_mcp_client(tools=[GEOCODE_TOOL])

    orchestrator = Stage5Orchestrator(llm_provider=provider, mcp_client=mcp_client)
    orchestrator.process_user_request("Find the coordinates of IIT Patna.")

    assert len(provider.received_tools) == 1
    assert provider.received_tools[0].name == "geocode_location"
    assert provider.received_tools[0].input_schema == GEOCODE_TOOL.input_schema


def test_orchestrator_sends_valid_tool_call_through_mcp_client():
    tool_call = ToolCall(name="geocode_location", arguments={"location": "IIT Patna"})
    provider = FakeLLMProvider(tool_call=tool_call)
    mcp_client = make_fake_mcp_client(tools=[GEOCODE_TOOL])

    orchestrator = Stage5Orchestrator(llm_provider=provider, mcp_client=mcp_client)
    result = orchestrator.process_user_request("Find the coordinates of IIT Patna.")

    mcp_client.call_tool.assert_called_once_with(
        "geocode_location", {"location": "IIT Patna"}
    )
    assert result == {"success": True}


def test_orchestrator_rejects_unknown_tool():
    tool_call = ToolCall(name="delete_everything", arguments={})
    provider = FakeLLMProvider(tool_call=tool_call)
    mcp_client = make_fake_mcp_client(tools=[GEOCODE_TOOL])

    orchestrator = Stage5Orchestrator(llm_provider=provider, mcp_client=mcp_client)
    result = orchestrator.process_user_request("Do something malicious.")

    assert result is None
    mcp_client.call_tool.assert_not_called()


def test_orchestrator_rejects_malformed_arguments():
    tool_call = ToolCall(name="geocode_location", arguments="not-a-dict")
    provider = FakeLLMProvider(tool_call=tool_call)
    mcp_client = make_fake_mcp_client(tools=[GEOCODE_TOOL])

    orchestrator = Stage5Orchestrator(llm_provider=provider, mcp_client=mcp_client)
    result = orchestrator.process_user_request("Find the coordinates of IIT Patna.")

    assert result is None
    mcp_client.call_tool.assert_not_called()


def test_orchestrator_returns_none_when_llm_does_not_request_a_tool():
    provider = FakeLLMProvider(tool_call=None)
    mcp_client = make_fake_mcp_client(tools=[GEOCODE_TOOL])

    orchestrator = Stage5Orchestrator(llm_provider=provider, mcp_client=mcp_client)
    result = orchestrator.process_user_request("Just chatting, no tool needed.")

    assert result is None
    mcp_client.call_tool.assert_not_called()


def test_orchestrator_does_not_import_concrete_providers():
    import ride_agent.agent.orchestrator as orchestrator_module

    with open(orchestrator_module.__file__, encoding="utf-8") as f:
        source = f.read()

    assert "OllamaProvider" not in source
    assert "OpenAICompatibleProvider" not in source


def test_orchestrator_works_identically_with_mocked_openai_provider():
    """Swapping FakeLLMProvider for a mocked OpenAICompatibleProvider must
    not require any change to orchestrator.py - only the injected
    provider changes."""

    from ride_agent.agent.llm.openai import OpenAICompatibleProvider

    mocked_openai_provider = MagicMock(spec=OpenAICompatibleProvider)
    mocked_openai_provider.is_available.return_value = True
    mocked_openai_provider.call_with_tools.return_value = {"choices": []}
    mocked_openai_provider.extract_tool_call.return_value = ToolCall(
        name="geocode_location", arguments={"location": "IIT Patna"}
    )

    mcp_client = make_fake_mcp_client(tools=[GEOCODE_TOOL])

    orchestrator = Stage5Orchestrator(
        llm_provider=mocked_openai_provider, mcp_client=mcp_client
    )
    result = orchestrator.process_user_request("Find the coordinates of IIT Patna.")

    mcp_client.call_tool.assert_called_once_with(
        "geocode_location", {"location": "IIT Patna"}
    )
    assert result == {"success": True}