"""Provider-neutral interfaces for the Stage 5/6 LLM integration layer.

This module defines the contract every LLM provider (Ollama,
OpenAI-compatible, or any future provider) must implement, along with
the provider-neutral data structures used to pass tool definitions,
tool calls, and conversation messages between the MCP layer, the LLM
provider, and the orchestrator.

Nothing in this module is aware of Ollama, OpenAI, Qwen, GPT, or any
specific model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolDefinition:
    """Provider-neutral representation of a single tool.

    Built from MCP tool-discovery data (name, description, input
    schema). This is the only tool representation the orchestrator and
    LLM providers should depend on.
    """

    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """Provider-neutral representation of a tool call produced by an LLM.

    Every LLMProvider implementation is responsible for translating its
    own raw API response into this shape. The orchestrator never
    inspects a raw provider response directly.

    `id` is optional because not every provider's wire format assigns
    an id to a tool call (Ollama's /api/chat typically does not).
    OpenAI-compatible APIs require it to correlate a tool-result
    message back to the tool call it answers, via
    format_tool_result_message().
    """

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None


class LLMProvider(ABC):
    """Abstract interface every LLM provider must implement.

    The orchestrator depends ONLY on this interface. It must never
    import or reference a concrete provider class (OllamaProvider,
    OpenAICompatibleProvider, or any future provider).
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the provider/model is currently reachable/usable."""
        raise NotImplementedError

    @abstractmethod
    def call_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[ToolDefinition],
    ) -> Any:
        """Send the conversation history and available tools to the model.

        Returns the raw, provider-specific response object/dict. This
        raw response must ONLY ever be consumed by this same provider's
        extract_tool_call() / extract_assistant_message() - never by
        the orchestrator.
        """
        raise NotImplementedError

    @abstractmethod
    def extract_tool_call(self, response: Any) -> Optional[ToolCall]:
        """Normalize a raw provider response into a ToolCall, or None."""
        raise NotImplementedError

    @abstractmethod
    def extract_assistant_message(self, response: Any) -> Dict[str, Any]:
        """Return the assistant's message from a raw response, in this
        provider's own wire format, suitable for appending directly to
        the conversation history passed into the next call_with_tools().

        Must always include at least "role" and "content" keys. When the
        response includes a tool call, this message must also carry
        whatever provider-specific tool-call representation that
        provider's own wire format expects (e.g. an OpenAI-style
        "tool_calls" list), since it is fed straight back to the same
        provider on the next turn.
        """
        raise NotImplementedError

    @abstractmethod
    def format_tool_result_message(
        self,
        tool_call: ToolCall,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the tool-result message, in this provider's own wire
        format, to append to conversation history after executing
        tool_call and obtaining result (which may represent a failure -
        callers should not assume result implies success)."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any provider resources (connections, sessions, etc.).

        Default no-op; providers that hold resources should override
        this.
        """
        return None