"""OpenAI-compatible implementation of the LLMProvider interface.

Targets the widely-used OpenAI "chat completions" tool-calling contract
(model, base_url, api_key, tools=[{"type": "function", ...}]), so it
works with the official OpenAI API as well as any OpenAI-compatible
endpoint implementing the same contract.

All OpenAI-specific request/response handling lives in this file only.
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


class OpenAICompatibleProvider(LLMProvider):
    """LLMProvider implementation for OpenAI-compatible chat completion APIs.

    model_name, api_key, and base_url are all supplied by the caller.
    Nothing here is hardcoded to a specific model or a specific vendor.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
    ):
        if not model_name:
            raise ValueError("OpenAICompatibleProvider requires an explicit model_name.")
        if not api_key:
            raise ValueError("OpenAICompatibleProvider requires an api_key.")

        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/chat/completions"
        self.client: Optional["httpx.Client"] = None

    def is_available(self) -> bool:
        if httpx is None:
            logger.error("httpx not available. Cannot check provider availability.")
            return False

        try:
            with httpx.Client() as client:
                response = client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5,
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"OpenAI-compatible endpoint not available: {e}")
            return False

    def get_client(self) -> "httpx.Client":
        if self.client is None:
            if httpx is None:
                raise RuntimeError("httpx not installed")
            self.client = httpx.Client(timeout=30)
        return self.client

    @staticmethod
    def _to_openai_tool_format(tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
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
            raise RuntimeError("httpx not installed. Cannot call OpenAI-compatible API.")

        client = self.get_client()

        payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "tools": self._to_ollama_tool_format(tools),
                    "stream": False,
                }

        response = client.post(
            self.api_url,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()

        result = response.json()

        logger.info(f"LLM response: {json.dumps(result, indent=2)[:500]}")

        return result

    def extract_tool_call(self, response: Dict[str, Any]) -> Optional[ToolCall]:
        try:
            choices = response.get("choices", [])
            if not choices:
                return None

            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                return None

            tool_call = tool_calls[0]
            function = tool_call.get("function", {})

            name = function.get("name")
            raw_arguments = function.get("arguments", "{}")

            if not name:
                return None

            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    logger.error(f"Could not parse tool arguments JSON: {raw_arguments}")
                    return None
            elif isinstance(raw_arguments, dict):
                arguments = raw_arguments
            else:
                arguments = {}

            return ToolCall(name=name, arguments=arguments)

        except Exception as e:
            logger.error(f"Error extracting tool call: {e}")
            return None

    def close(self) -> None:
        if self.client:
            self.client.close()
            self.client = None