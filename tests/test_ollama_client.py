"""Tests for OllamaClient (Phase 1)."""

import pytest
from local_agent.config import Config
from local_agent.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
)


def test_ollama_live_simple_message():
    """Verify that Ollama responds to a basic user message."""
    client = OllamaClient()
    messages = [{"role": "user", "content": "Respond with the word PONG only."}]

    response = client.chat(messages=messages)

    assert "role" in response
    assert response["role"] == "assistant"
    assert "content" in response
    assert len(response["content"].strip()) > 0
    assert "PONG" in response["content"].upper()


def test_ollama_connection_error_useful_message():
    """Verify that an unreachable URL produces a friendly, actionable error."""
    # Use an invalid port where no service runs
    bad_config = Config(ollama_base_url="http://127.0.0.1:59999", request_timeout=2.0)
    client = OllamaClient(config=bad_config)

    with pytest.raises(OllamaConnectionError) as exc_info:
        client.chat(messages=[{"role": "user", "content": "hi"}])

    error_message = str(exc_info.value)
    assert "Could not connect to Ollama" in error_message
    assert "ollama serve" in error_message


def test_ollama_missing_model_useful_message():
    """Verify that requesting a nonexistent model suggests running ollama pull."""
    client = OllamaClient()

    with pytest.raises(OllamaModelNotFoundError) as exc_info:
        client.chat(
            messages=[{"role": "user", "content": "hi"}],
            model="nonexistent-model-xyz",
        )

    error_message = str(exc_info.value)
    assert "was not found in Ollama" in error_message
    assert "ollama pull nonexistent-model-xyz" in error_message
