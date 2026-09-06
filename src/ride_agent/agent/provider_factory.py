"""Small factory for selecting an LLMProvider from configuration.

This is the ONLY place provider-name branching should exist in this
codebase. The orchestrator must never branch on provider name/type -
it only ever sees an LLMProvider instance.
"""

import os
from typing import Optional

from ride_agent.agent.llm.base import LLMProvider
from ride_agent.agent.llm.ollama import OllamaProvider
from ride_agent.agent.llm.openai import OpenAICompatibleProvider


def create_llm_provider(
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMProvider:
    """Create an LLMProvider based on explicit args or environment variables.

    Falls back to these environment variables when an argument is None:
        RIDE_AGENT_LLM_PROVIDER   ("ollama" or "openai", default "ollama")
        RIDE_AGENT_LLM_MODEL
        RIDE_AGENT_LLM_BASE_URL
        RIDE_AGENT_LLM_API_KEY
    """

    provider_name = (
        provider_name or os.environ.get("RIDE_AGENT_LLM_PROVIDER", "ollama")
    ).lower()
    model_name = model_name or os.environ.get("RIDE_AGENT_LLM_MODEL")

    if provider_name == "ollama":
        if not model_name:
            raise ValueError(
                "No model_name provided for OllamaProvider. "
                "Run 'ollama list' and supply an installed model name "
                "(via argument or the RIDE_AGENT_LLM_MODEL env var)."
            )

        kwargs = {}
        resolved_base_url = base_url or os.environ.get("RIDE_AGENT_LLM_BASE_URL")
        if resolved_base_url:
            kwargs["base_url"] = resolved_base_url

        return OllamaProvider(model_name=model_name, **kwargs)

    if provider_name == "openai":
        resolved_api_key = api_key or os.environ.get("RIDE_AGENT_LLM_API_KEY")
        resolved_base_url = base_url or os.environ.get(
            "RIDE_AGENT_LLM_BASE_URL", "https://api.openai.com/v1"
        )

        if not model_name:
            raise ValueError("No model_name provided for OpenAICompatibleProvider.")
        if not resolved_api_key:
            raise ValueError("No api_key provided for OpenAICompatibleProvider.")

        return OpenAICompatibleProvider(
            model_name=model_name,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
        )

    raise ValueError(f"Unknown provider_name: {provider_name!r}")