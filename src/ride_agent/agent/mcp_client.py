"""MCP Client wrapper for discovering and invoking MCP tools.

This uses the real MCP Python SDK v2 client (`mcp.Client` +
`mcp.StdioServerParameters`) to communicate with the actual MCP server
process over stdio JSON-RPC. It does NOT hardcode tool definitions and
does NOT call tool implementations directly in-process - both
discovery and execution go through the real MCP protocol against
server.py.

The public methods (discover_tools, call_tool, close) are kept
synchronous on purpose, even though the underlying SDK client is
async-only, so that Stage5Orchestrator (which is synchronous) does not
need to change.
"""

import asyncio
import logging
import sys
from typing import Any, Dict, List

from mcp import Client, MCPError, StdioServerParameters

logger = logging.getLogger(__name__)


class MCPToolDefinition:
    """Represents a discovered MCP tool."""

    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.input_schema = input_schema


class MCPClient:
    """Client for connecting to and invoking MCP server tools over real MCP (stdio)."""

    def __init__(self, server_script_path: str = "src/ride_agent/server.py"):
        """Initialize MCP client.

        Args:
            server_script_path: Path to the MCP server script. Launched
                as `sys.executable server_script_path` over stdio.
        """
        self.server_script_path = server_script_path
        self.discovered_tools: Dict[str, MCPToolDefinition] = {}

    def _server_params(self) -> StdioServerParameters:
        return StdioServerParameters(
            command=sys.executable,
            args=[self.server_script_path],
        )

    async def _discover_tools_async(self) -> List[MCPToolDefinition]:
        async with Client(self._server_params()) as client:
            result = await client.list_tools()
            return [
                MCPToolDefinition(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.input_schema or {},
                )
                for tool in result.tools
            ]

    def discover_tools(self) -> List[MCPToolDefinition]:
        """Discover available tools from the real MCP server via list_tools().

        Returns:
            List of discovered MCPToolDefinition objects, sourced from the
            actual running MCP server - never a hardcoded list.
        """
        try:
            tools = asyncio.run(self._discover_tools_async())
        except MCPError as e:
            logger.error(f"MCP error during tool discovery: {e}")
            raise

        self.discovered_tools = {tool.name: tool for tool in tools}
        logger.info(f"Discovered {len(tools)} MCP tools: {[t.name for t in tools]}")
        return tools

    async def _call_tool_async(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        async with Client(self._server_params()) as client:
            result = await client.call_tool(tool_name, arguments)

            if getattr(result, "is_error", False):
                error_text = ""
                if result.content:
                    first_block = result.content[0]
                    error_text = getattr(first_block, "text", str(first_block))
                raise ValueError(f"MCP tool '{tool_name}' returned an error: {error_text}")

            if result.structured_content is not None:
                return result.structured_content

            # Fall back to parsing text content if the tool did not
            # produce structured_content for some reason.
            if result.content:
                first_block = result.content[0]
                text = getattr(first_block, "text", None)
                if text is not None:
                    import json

                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return {"result": text}

            return {}

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool through the real MCP server (stdio JSON-RPC).

        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            The tool's structured result, as actually executed by the
            MCP server process (which in turn calls the real
            geocoding/routing/pricing services).

        Raises:
            ValueError: If the tool is not in the last-discovered tool
                set, or if the MCP server reports a tool execution error.
            MCPError: If the MCP protocol call itself fails (raised by
                the SDK, e.g. the server rejects the request).
        """
        if tool_name not in self.discovered_tools:
            raise ValueError(f"Unknown MCP tool: {tool_name}")

        try:
            result = asyncio.run(self._call_tool_async(tool_name, arguments))
            logger.info(f"Called {tool_name} with args {arguments}")
            return result
        except MCPError as e:
            logger.error(f"MCP error calling {tool_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error calling {tool_name}: {e}")
            raise

    def close(self) -> None:
        """No-op: each call opens and cleanly closes its own MCP session/subprocess."""
        return None