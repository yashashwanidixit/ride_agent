"""Orchestrator for LLM + MCP integration (Stage 5).

The orchestrator depends ONLY on the LLMProvider abstraction and on
MCPClient. It has no knowledge of Ollama, OpenAI, Qwen, GPT, or any
other specific provider/model, and it never imports a concrete
LLMProvider implementation.
"""

import logging
from typing import Any, Dict, List, Optional

from ride_agent.agent.llm.base import LLMProvider, ToolCall, ToolDefinition
from ride_agent.agent.mcp_client import MCPClient
from ride_agent.agent.prompts import RIDE_CONCIERGE_SYSTEM_PROMPT
logger = logging.getLogger(__name__)


def _tool_to_definition(tool: Any) -> ToolDefinition:
    """Convert an MCP-discovered tool object into a provider-neutral ToolDefinition.

    Assumes the discovered tool object exposes `name`, `description`,
    and an input schema under either `input_schema` (snake_case) or
    `inputSchema` (the standard MCP protocol field name used by the
    official `mcp` Python SDK's list_tools() result).

    If your current MCPClient's Tool wrapper uses different attribute
    names than these, this function needs to be adjusted - see the
    accompanying explanation for what to check.
    """

    name = getattr(tool, "name", None)
    if not name:
        raise ValueError("Discovered MCP tool is missing a name.")

    description = getattr(tool, "description", "") or ""

    input_schema = getattr(tool, "input_schema", None)
    if input_schema is None:
        input_schema = getattr(tool, "inputSchema", None)
    if input_schema is None:
        input_schema = {}

    return ToolDefinition(
        name=name,
        description=description,
        input_schema=input_schema,
    )


class Stage5Orchestrator:
    """Coordinates LLM tool calling with MCP tool execution.

    Provider-independent: constructed with an LLMProvider instance via
    dependency injection. Never instantiates OllamaProvider or
    OpenAICompatibleProvider itself.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        mcp_server_path: str = "src/ride_agent/server.py",
        mcp_client: Optional[MCPClient] = None,
    ):
        self.llm_provider = llm_provider
        self.mcp_client = mcp_client or MCPClient(server_script_path=mcp_server_path)

    def initialize(self) -> bool:
        try:
            if not self.llm_provider.is_available():
                logger.error("LLM provider not available")
                return False

            tools = self.mcp_client.discover_tools()

            if not tools:
                logger.error("No MCP tools discovered")
                return False

            logger.info(f"Successfully discovered {len(tools)} MCP tools")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def process_user_request(
        self,
        user_message: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            discovered_tools = self.mcp_client.discover_tools()

            tool_definitions: List[ToolDefinition] = [
                _tool_to_definition(tool) for tool in discovered_tools
            ]

            logger.info(
                f"Available MCP tools: {[t.name for t in tool_definitions]}"
            )

            messages = [
                {
                    "role": "system",
                    "content": RIDE_CONCIERGE_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ]

            llm_response = self.llm_provider.call_with_tools(
                messages,
                tool_definitions,
)

            tool_call: Optional[ToolCall] = self.llm_provider.extract_tool_call(
                llm_response
            )

            if not tool_call:
                logger.info("LLM did not request a tool")
                return None

            if not isinstance(tool_call.arguments, dict):
                logger.error(
                    f"Malformed tool call arguments for '{tool_call.name}': "
                    f"expected an object, got {type(tool_call.arguments)!r}"
                )
                return None

            valid_tool_names = {tool.name for tool in tool_definitions}

            if tool_call.name not in valid_tool_names:
                logger.error(f"LLM requested unknown tool: {tool_call.name}")
                return None

            result = self.mcp_client.call_tool(
                tool_call.name,
                tool_call.arguments,
            )

            return result

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise

    def cleanup(self) -> None:
        try:
            self.llm_provider.close()
            self.mcp_client.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")