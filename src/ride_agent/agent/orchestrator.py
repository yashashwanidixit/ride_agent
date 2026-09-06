"""Orchestrator for LLM + MCP integration (Stage 6: sequential agent loop).

The orchestrator depends ONLY on the LLMProvider abstraction and on
MCPClient. It has no knowledge of Ollama, OpenAI, Qwen, GPT, or any
other specific provider/model, and it never imports a concrete
LLMProvider implementation.

Stage 6 change from Stage 5: process_user_request() no longer makes a
single LLM call and returns a raw tool result. It now runs a loop -
call the LLM, execute a tool if requested, feed the result back,
repeat - until the LLM produces a final answer with no further tool
call, or a safety limit is reached. process_user_request() now returns
the final natural-language answer (str) rather than a raw tool dict.
"""

import logging
from typing import Any, Dict, List, Optional

from ride_agent.agent.llm.base import LLMProvider, ToolCall, ToolDefinition
from ride_agent.agent.mcp_client import MCPClient
from ride_agent.agent.prompts import RIDE_CONCIERGE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOOL_CALLS = 10

_FALLBACK_LIMIT_REACHED_MESSAGE = (
    "I wasn't able to finish this request within the allowed number of "
    "tool calls. Could you try rephrasing it or breaking it into smaller "
    "steps?"
)


def _tool_to_definition(tool: Any) -> ToolDefinition:
    """Convert an MCP-discovered tool object into a provider-neutral ToolDefinition.

    Assumes the discovered tool object exposes `name`, `description`,
    and an input schema under either `input_schema` (snake_case) or
    `inputSchema` (the standard MCP protocol field name used by the
    official `mcp` Python SDK's list_tools() result).
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

    Retains the name Stage5Orchestrator (rather than renaming to
    Stage6Orchestrator) since it is the same orchestration component
    evolving across stages, and both this codebase and the CLI already
    reference it under this name.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        mcp_server_path: str = "src/ride_agent/server.py",
        mcp_client: Optional[MCPClient] = None,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    ):
        self.llm_provider = llm_provider
        self.mcp_client = mcp_client or MCPClient(server_script_path=mcp_server_path)
        self.max_tool_calls = max_tool_calls

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

    def process_user_request(self, user_message: str) -> str:
        """Run the Stage 6 agent loop for a single user request.

        Returns the final natural-language answer produced by the LLM.
        The exact sequence and number of tool calls is decided entirely
        by the LLM's responses; nothing here hardcodes which tool comes
        next or how many are needed.
        """
        discovered_tools = self.mcp_client.discover_tools()
        tool_definitions: List[ToolDefinition] = [
            _tool_to_definition(tool) for tool in discovered_tools
        ]
        valid_tool_names = {tool.name for tool in tool_definitions}

        logger.info(f"Available MCP tools: {[t.name for t in tool_definitions]}")

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": RIDE_CONCIERGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        for iteration in range(1, self.max_tool_calls + 1):
            llm_response = self.llm_provider.call_with_tools(messages, tool_definitions)

            assistant_message = self.llm_provider.extract_assistant_message(llm_response)
            tool_call: Optional[ToolCall] = self.llm_provider.extract_tool_call(llm_response)

            if not tool_call:
                final_answer = assistant_message.get("content") or ""
                logger.info("LLM produced a final answer; ending the loop.")
                return final_answer

            messages.append(assistant_message)

            # Every branch below must append SOME tool-result message
            # before the loop continues, even on rejection/failure -
            # OpenAI-compatible APIs require a tool response for every
            # tool_call the assistant made, or the next call fails.

            if not isinstance(tool_call.arguments, dict):
                logger.error(
                    f"Malformed tool call arguments for '{tool_call.name}': "
                    f"expected an object, got {type(tool_call.arguments)!r}"
                )
                error_result = {
                    "success": False,
                    "error": f"Malformed arguments for tool '{tool_call.name}'.",
                }
                messages.append(
                    self.llm_provider.format_tool_result_message(tool_call, error_result)
                )
                continue

            if tool_call.name not in valid_tool_names:
                logger.error(f"LLM requested unknown tool: {tool_call.name}")
                error_result = {
                    "success": False,
                    "error": f"Unknown tool '{tool_call.name}'. Not available via MCP.",
                }
                messages.append(
                    self.llm_provider.format_tool_result_message(tool_call, error_result)
                )
                continue

            logger.info(
                f"LLM requested tool: {tool_call.name} args={tool_call.arguments}"
            )

            try:
                tool_result = self.mcp_client.call_tool(tool_call.name, tool_call.arguments)
                logger.info(f"MCP tool executed: {tool_call.name}")
                logger.info(f"Tool result: {tool_result}")
            except Exception as e:
                logger.error(f"Tool execution failed for {tool_call.name}: {e}")
                tool_result = {"success": False, "error": str(e)}

            messages.append(
                self.llm_provider.format_tool_result_message(tool_call, tool_result)
            )

        logger.error(
            f"Maximum tool-call limit ({self.max_tool_calls}) reached without a final answer."
        )
        return _FALLBACK_LIMIT_REACHED_MESSAGE

    def cleanup(self) -> None:
        try:
            self.llm_provider.close()
            self.mcp_client.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")