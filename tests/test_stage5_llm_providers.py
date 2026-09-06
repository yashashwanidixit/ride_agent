"""Unit tests for Stage 5 LLM provider abstraction and normalization.

These tests do NOT require a real Ollama instance or real OpenAI
credentials - all HTTP calls are mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

from ride_agent.agent.llm.base import LLMProvider, ToolCall, ToolDefinition
from ride_agent.agent.llm.ollama import OllamaProvider
from ride_agent.agent.llm.openai import OpenAICompatibleProvider


SAMPLE_TOOLS = [
    ToolDefinition(
        name="geocode_location",
        description="Converts a place name into coordinates.",
        input_schema={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    )
]


def test_ollama_provider_implements_llm_provider():
    provider = OllamaProvider(model_name="fake-model")
    assert isinstance(provider, LLMProvider)


def test_openai_provider_implements_llm_provider():
    provider = OpenAICompatibleProvider(model_name="fake-model", api_key="fake-key")
    assert isinstance(provider, LLMProvider)


def test_ollama_provider_requires_model_name():
    with pytest.raises(ValueError):
        OllamaProvider(model_name="")


def test_openai_provider_requires_model_name_and_key():
    with pytest.raises(ValueError):
        OpenAICompatibleProvider(model_name="", api_key="key")

    with pytest.raises(ValueError):
        OpenAICompatibleProvider(model_name="model", api_key="")


def test_ollama_tool_call_normalization():
    provider = OllamaProvider(model_name="fake-model")

    raw_response = {
        "message": {
            "tool_calls": [
                {
                    "function": {
                        "name": "geocode_location",
                        "arguments": {"location": "IIT Patna"},
                    }
                }
            ]
        }
    }

    tool_call = provider.extract_tool_call(raw_response)

    assert tool_call == ToolCall(
        name="geocode_location",
        arguments={"location": "IIT Patna"},
    )


def test_ollama_no_tool_call_returns_none():
    provider = OllamaProvider(model_name="fake-model")
    assert provider.extract_tool_call({"message": {}}) is None


def test_openai_tool_call_normalization_with_json_string_arguments():
    provider = OpenAICompatibleProvider(model_name="fake-model", api_key="fake-key")

    raw_response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "geocode_location",
                                "arguments": '{"location": "IIT Patna"}',
                            }
                        }
                    ]
                }
            }
        ]
    }

    tool_call = provider.extract_tool_call(raw_response)

    assert tool_call == ToolCall(
        name="geocode_location",
        arguments={"location": "IIT Patna"},
    )


def test_openai_no_tool_call_returns_none():
    provider = OpenAICompatibleProvider(model_name="fake-model", api_key="fake-key")
    assert provider.extract_tool_call({"choices": [{"message": {}}]}) is None


def test_ollama_and_openai_normalize_to_the_same_tool_call_shape():
    ollama_provider = OllamaProvider(model_name="fake-model")
    openai_provider = OpenAICompatibleProvider(model_name="fake-model", api_key="fake-key")

    ollama_raw = {
        "message": {
            "tool_calls": [
                {"function": {"name": "geocode_location", "arguments": {"location": "IIT Patna"}}}
            ]
        }
    }

    openai_raw = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "geocode_location",
                                "arguments": '{"location": "IIT Patna"}',
                            }
                        }
                    ]
                }
            }
        ]
    }

    assert ollama_provider.extract_tool_call(ollama_raw) == openai_provider.extract_tool_call(
        openai_raw
    )


@patch("ride_agent.agent.llm.ollama.httpx")
def test_ollama_call_with_tools_sends_ollama_tool_format(mock_httpx):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"tool_calls": []}}
    mock_response.raise_for_status.return_value = None
    mock_client.post.return_value = mock_response
    mock_httpx.Client.return_value = mock_client

    provider = OllamaProvider(model_name="fake-model")
    provider.call_with_tools("Find IIT Patna", SAMPLE_TOOLS)

    sent_payload = mock_client.post.call_args.kwargs["json"]
    assert sent_payload["model"] == "fake-model"
    assert sent_payload["tools"][0]["function"]["name"] == "geocode_location"
    assert sent_payload["tools"][0]["function"]["parameters"] == SAMPLE_TOOLS[0].input_schema


@patch("ride_agent.agent.llm.openai.httpx")
def test_openai_call_with_tools_sends_openai_tool_format_and_auth_header(mock_httpx):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {}}]}
    mock_response.raise_for_status.return_value = None
    mock_client.post.return_value = mock_response
    mock_httpx.Client.return_value = mock_client

    provider = OpenAICompatibleProvider(model_name="fake-model", api_key="fake-key")
    provider.call_with_tools("Find IIT Patna", SAMPLE_TOOLS)

    call_kwargs = mock_client.post.call_args.kwargs
    assert call_kwargs["json"]["tools"][0]["function"]["name"] == "geocode_location"
    assert call_kwargs["headers"]["Authorization"] == "Bearer fake-key"