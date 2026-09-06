"""Unit tests for the LLM provider factory (the only place provider-name
branching should exist)."""

import pytest

from ride_agent.agent.llm.ollama import OllamaProvider
from ride_agent.agent.llm.openai import OpenAICompatibleProvider
from ride_agent.agent.provider_factory import create_llm_provider


def test_factory_creates_ollama_provider():
    provider = create_llm_provider(provider_name="ollama", model_name="fake-model")
    assert isinstance(provider, OllamaProvider)
    assert provider.model_name == "fake-model"


def test_factory_creates_openai_provider():
    provider = create_llm_provider(
        provider_name="openai", model_name="fake-model", api_key="fake-key"
    )
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model_name == "fake-model"


def test_factory_requires_model_name_for_ollama():
    with pytest.raises(ValueError):
        create_llm_provider(provider_name="ollama", model_name=None)


def test_factory_requires_api_key_for_openai():
    with pytest.raises(ValueError):
        create_llm_provider(provider_name="openai", model_name="fake-model", api_key=None)


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError):
        create_llm_provider(provider_name="not-a-real-provider", model_name="fake-model")