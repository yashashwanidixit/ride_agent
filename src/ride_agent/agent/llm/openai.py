"""OpenAI-compatible implementation of the LLMProvider interface.

Targets the widely-used OpenAI "chat completions" tool-calling contract
(model, base_url, api_key, tools=[{"type": "function", ...}]), so it
works with the official OpenAI API as well as any OpenAI-compatible
endpoint implementing the same contract.

All OpenAI-specific request/response handling lives in this file only.
The Stage 6 agent loop itself lives in the orchestrator, not here.
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
    "tools": self._to_openai_tool_format(tools),
    "reasoning_effort": "none",
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

        if response.status_code >= 400:
            logger.error(
                "OpenAI API error %s: %s",
                response.status_code,
                response.text,
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
            call_id = tool_call.get("id")

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

            return ToolCall(name=name, arguments=arguments, id=call_id)

        except Exception as e:
            logger.error(f"Error extracting tool call: {e}")
            return None

    def extract_assistant_message(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """The choices[0].message object is already exactly the shape
        the chat completions API expects back in the messages list on
        the next turn, so we return it as-is (with a safe default)."""
        choices = response.get("choices", [])
        if not choices:
            return {"role": "assistant", "content": ""}
        return choices[0].get("message") or {"role": "assistant", "content": ""}

    def format_tool_result_message(
        self,
        tool_call: ToolCall,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """OpenAI-compatible APIs require tool_call_id to correlate this
        response with a specific entry in the assistant's tool_calls."""
        if not tool_call.id:
            logger.warning(
                f"Tool call for '{tool_call.name}' has no id; the "
                "OpenAI-compatible API may reject a tool_call_id-less "
                "tool message on the next turn."
            )
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        }

    def close(self) -> None:
        if self.client:
            self.client.close()
            self.client = None