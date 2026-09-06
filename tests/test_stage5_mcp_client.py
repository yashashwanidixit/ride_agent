"""Unit tests for the real MCP-protocol MCPClient (Stage 5).

Unit tests mock mcp.Client entirely - no real subprocess is spawned.
The one integration test that spawns the real server.py is marked and
skipped by default.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ride_agent.agent.mcp_client import MCPClient, MCPToolDefinition


class FakeMCPTool:
    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.input_schema = input_schema


class FakeListToolsResult:
    def __init__(self, tools):
        self.tools = tools


class FakeCallToolResult:
    def __init__(self, structured_content=None, content=None, is_error=False):
        self.structured_content = structured_content
        self.content = content or []
        self.is_error = is_error


def make_fake_client(list_tools_result=None, call_tool_result=None):
    """Build a mock that behaves like `async with Client(...) as client: ...`."""
    fake_client = MagicMock()
    fake_client.list_tools = AsyncMock(return_value=list_tools_result)
    fake_client.call_tool = AsyncMock(return_value=call_tool_result)

    fake_client_cm = MagicMock()
    fake_client_cm.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client_cm.__aexit__ = AsyncMock(return_value=False)

    return fake_client_cm


def test_discover_tools_uses_real_list_tools_not_hardcoded():
    fake_tools = [
        FakeMCPTool(
            name="geocode_location",
            description="Converts a place into coordinates.",
            input_schema={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        ),
        FakeMCPTool(name="get_ride_time", description="Ride time.", input_schema={}),
        FakeMCPTool(name="get_ride_price", description="Ride price.", input_schema={}),
    ]
    fake_client_cm = make_fake_client(list_tools_result=FakeListToolsResult(fake_tools))

    with patch("ride_agent.agent.mcp_client.Client", return_value=fake_client_cm):
        mcp_client = MCPClient(server_script_path="fake/server.py")
        discovered = mcp_client.discover_tools()

    names = [t.name for t in discovered]
    assert names == ["geocode_location", "get_ride_time", "get_ride_price"]
    assert isinstance(discovered[0], MCPToolDefinition)
    assert discovered[0].input_schema["properties"]["location"]["type"] == "string"


def test_call_tool_sends_request_through_mcp_client_not_direct_import():
    fake_tools = [FakeMCPTool(name="geocode_location", description="", input_schema={})]
    discover_cm = make_fake_client(list_tools_result=FakeListToolsResult(fake_tools))

    call_result = FakeCallToolResult(
        structured_content={
            "success": True,
            "latitude": 25.5941,
            "longitude": 85.1376,
            "display_name": "IIT Patna",
        }
    )
    call_cm = make_fake_client(call_tool_result=call_result)

    with patch("ride_agent.agent.mcp_client.Client", side_effect=[discover_cm, call_cm]):
        mcp_client = MCPClient(server_script_path="fake/server.py")
        mcp_client.discover_tools()
        result = mcp_client.call_tool("geocode_location", {"location": "IIT Patna"})

    assert result["success"] is True
    assert result["display_name"] == "IIT Patna"


def test_call_tool_rejects_undiscovered_tool_name():
    mcp_client = MCPClient(server_script_path="fake/server.py")
    # discover_tools() was never called, so discovered_tools is empty.
    with pytest.raises(ValueError):
        mcp_client.call_tool("geocode_location", {"location": "IIT Patna"})


def test_call_tool_raises_on_tool_error_result():
    fake_tools = [FakeMCPTool(name="geocode_location", description="", input_schema={})]
    discover_cm = make_fake_client(list_tools_result=FakeListToolsResult(fake_tools))

    error_content_block = MagicMock()
    error_content_block.text = "location not found"
    call_result = FakeCallToolResult(is_error=True, content=[error_content_block])
    call_cm = make_fake_client(call_tool_result=call_result)

    with patch("ride_agent.agent.mcp_client.Client", side_effect=[discover_cm, call_cm]):
        mcp_client = MCPClient(server_script_path="fake/server.py")
        mcp_client.discover_tools()

        with pytest.raises(ValueError, match="location not found"):
            mcp_client.call_tool("geocode_location", {"location": "Nowhere"})


def test_mcp_tool_definition_has_no_ollama_specific_method():
    tool = MCPToolDefinition(name="x", description="y", input_schema={})
    assert not hasattr(tool, "to_ollama_tool_format")


@pytest.mark.integration
def test_real_discover_tools_against_actual_server():
    """Real integration test: spawns the actual server.py over stdio.

    Skipped by default. Run explicitly with:
        python -m pytest tests/test_stage5_mcp_client.py -m integration -q
    Requires the real server.py and its dependencies (Nominatim/OSRM
    reachability is not needed just for discovery).
    """
    mcp_client = MCPClient(server_script_path="src/ride_agent/server.py")
    tools = mcp_client.discover_tools()

    names = {t.name for t in tools}
    assert names == {"geocode_location", "get_ride_time", "get_ride_price"}