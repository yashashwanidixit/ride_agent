"""Ollama-specific implementation of the LLMProvider interface.

All Ollama API details (HTTP endpoints, request/response shapes, the
"/api/chat" tool-calling format) live in this module only. Nothing
outside this file needs to know how Ollama represents tools or tool
calls.
"""

import json
import logging
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:
    httpx = None

from ride_agent.agent.llm.base import LLMProvider, ToolCall, ToolDefinition

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """LLMProvider implementation backed by a local Ollama server.

    The model name is NOT hardcoded here. It must be supplied by the
    caller (application config / factory), based on whatever model is
    actually installed locally (verify with `ollama list`).
    """

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        if not model_name:
            raise ValueError(
                "OllamaProvider requires an explicit model_name. "
                "Run 'ollama list' to see installed models and pass one in."
            )

        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/chat"
        self.client: Optional["httpx.Client"] = None

    def is_available(self) -> bool:
        if httpx is None:
            logger.error("httpx not available. Cannot check Ollama availability.")
            return False

        try:
            with httpx.Client() as client:
                response = client.get(f"{self.base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    model_names = [m.get("name") for m in models]
                    available = any(self.model_name in name for name in model_names)

                    if available:
                        logger.info(f"Ollama model {self.model_name} is available")
                    else:
                        logger.warning(
                            f"Ollama running but {self.model_name} not found. "
                            f"Available: {model_names}"
                        )

                    return available
        except Exception as e:
            logger.error(f"Ollama not available: {e}")
            return False

        return False

    def get_client(self) -> "httpx.Client":
        if self.client is None:
            if httpx is None:
                raise RuntimeError("httpx not installed")

            self.client = httpx.Client(timeout=30)

        return self.client

    @staticmethod
    def _to_ollama_tool_format(tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert provider-neutral ToolDefinitions into Ollama's tool format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
        ]

    def call_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[ToolDefinition],
    ) -> Dict[str, Any]:
        if httpx is None:
            raise RuntimeError("httpx not installed. Cannot call Ollama.")

        client = self.get_client()

        payload = {
            "model": self.model_name,
            "messages": messages,
            "tools": self._to_ollama_tool_format(tools),
            "stream": False,
        }

        response = client.post(self.api_url, json=payload)
        response.raise_for_status()

        result = response.json()

        logger.info(f"LLM response: {json.dumps(result, indent=2)[:500]}")

        return result

    def extract_tool_call(self, response: Dict[str, Any]) -> Optional[ToolCall]:
        try:
            message = response.get("message", {})
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                return None

            tool_call = tool_calls[0]
            function = tool_call.get("function", {})

            name = function.get("name")
            arguments = function.get("arguments", {})

            if not name:
                return None

            return ToolCall(name=name, arguments=arguments)

        except Exception as e:
            logger.error(f"Error extracting tool call: {e}")
            return None

    def close(self) -> None:
        if self.client:
            self.client.close()
            self.client = None