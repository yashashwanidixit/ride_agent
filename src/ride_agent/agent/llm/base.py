"""Provider-neutral interfaces for the Stage 5 LLM integration layer.

This module defines the contract every LLM provider (Ollama,
OpenAI-compatible, or any future provider) must implement, along with
the provider-neutral data structures used to pass tool definitions and
tool calls between the MCP layer, the LLM provider, and the
orchestrator.

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
    """

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


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
        """Send the user message and available tools to the model.

        Returns the raw, provider-specific response object/dict. This
        raw response must ONLY ever be consumed by this same provider's
        extract_tool_call() implementation - never by the orchestrator.
        """
        raise NotImplementedError

    @abstractmethod
    def extract_tool_call(self, response: Any) -> Optional[ToolCall]:
        """Normalize a raw provider response into a ToolCall, or None."""
        raise NotImplementedError
    
    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        """Release any provider resources (connections, sessions, etc.).

        Default no-op; providers that hold resources should override
        this.
        """
        return None