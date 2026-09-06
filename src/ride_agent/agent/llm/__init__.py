"""LLM provider package: provider-neutral interfaces plus concrete providers.

Nothing outside this package should import ollama.py or openai.py
directly except application-level wiring code (e.g. a factory or a
main entrypoint) that decides which provider to construct.
"""

from ride_agent.agent.llm.base import LLMProvider, ToolCall, ToolDefinition
from ride_agent.agent.llm.ollama import OllamaProvider
from ride_agent.agent.llm.openai import OpenAICompatibleProvider

__all__ = [
    "LLMProvider",
    "ToolCall",
    "ToolDefinition",
    "OllamaProvider",
    "OpenAICompatibleProvider",
]