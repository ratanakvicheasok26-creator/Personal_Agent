"""Tests for terminal chatbot loop (Phase 2)."""

from typing import Any
import pytest
from local_agent.config import Config
from local_agent.main import run_chat_loop
from local_agent.ollama_client import OllamaClient, OllamaConnectionError


class FakeOllamaClient(OllamaClient):
    """Fake client to test multi-turn and error handling without running Ollama."""

    def __init__(self, responses: list[dict[str, Any]] | None = None, should_fail: bool = False):
        super().__init__(config=Config())
        self.responses = list(responses or [])
        self.calls: list[list[dict[str, Any]]] = []
        self.should_fail = should_fail

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(list(messages))
        if self.should_fail:
            raise OllamaConnectionError("Could not connect to Ollama.")
        if self.responses:
            return self.responses.pop(0)
        return {"role": "assistant", "content": "Default test response"}


def test_clean_exit_commands():
    """Verify that 'exit', 'quit', and 'q' immediately terminate the loop."""
    for cmd in ["exit", "quit", "q", "EXIT"]:
        inputs = iter([cmd])
        outputs: list[str] = []

        fake_client = FakeOllamaClient()
        run_chat_loop(
            client=fake_client,
            input_fn=lambda _: next(inputs),
            print_fn=lambda msg="", **kwargs: outputs.append(str(msg)),
        )

        assert any("Goodbye!" in out for out in outputs)
        assert len(fake_client.calls) == 0


def test_clean_exit_on_keyboard_interrupt():
    """Verify that Ctrl+C (KeyboardInterrupt) exits cleanly without throwing an exception."""
    def raise_interrupt(_: str) -> str:
        raise KeyboardInterrupt()

    outputs: list[str] = []
    fake_client = FakeOllamaClient()

    run_chat_loop(
        client=fake_client,
        input_fn=raise_interrupt,
        print_fn=lambda msg="", **kwargs: outputs.append(str(msg)),
    )

    assert any("Goodbye!" in out for out in outputs)


def test_multi_turn_history_retained():
    """Verify that messages across turns accumulate in conversation history."""
    inputs = iter(["Hello", "I am Alice", "exit"])
    outputs: list[str] = []

    fake_client = FakeOllamaClient(
        responses=[
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
        ]
    )

    run_chat_loop(
        client=fake_client,
        input_fn=lambda _: next(inputs),
        print_fn=lambda msg="", **kwargs: outputs.append(str(msg)),
    )

    # There were 2 conversational turns
    assert len(fake_client.calls) == 2

    # Turn 1 had 1 message
    assert len(fake_client.calls[0]) == 1
    assert fake_client.calls[0][0] == {"role": "user", "content": "Hello"}

    # Turn 2 had 3 messages (user, assistant, user)
    assert len(fake_client.calls[1]) == 3
    assert fake_client.calls[1][0] == {"role": "user", "content": "Hello"}
    assert fake_client.calls[1][1] == {"role": "assistant", "content": "Hello! How can I help?"}
    assert fake_client.calls[1][2] == {"role": "user", "content": "I am Alice"}

    assert any("Nice to meet you, Alice!" in out for out in outputs)


def test_error_handling_does_not_crash_loop():
    """Verify that an Ollama error is reported without breaking the loop or corrupting history."""
    inputs = iter(["Hello", "exit"])
    outputs: list[str] = []

    fake_client = FakeOllamaClient(should_fail=True)

    run_chat_loop(
        client=fake_client,
        input_fn=lambda _: next(inputs),
        print_fn=lambda msg="", **kwargs: outputs.append(str(msg)),
    )

    assert any("[Error]" in out for out in outputs)
    assert any("Goodbye!" in out for out in outputs)
