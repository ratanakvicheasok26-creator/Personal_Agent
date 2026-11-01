"""Tests for Agent tool loop implementation (Phase 5)."""

import os
from typing import Any
import pytest

from local_agent.config import Config
from local_agent.ollama_client import OllamaClient
from local_agent.mcp_client import MCPClient
from local_agent.agent import Agent, mcp_to_ollama_tool


class MockOllamaClient(OllamaClient):
    """Mock client to script Ollama responses for deterministic testing of the tool loop."""

    def __init__(self, responses: list[dict[str, Any]]):
        super().__init__(config=Config())
        self.responses = list(responses)
        self.history_snapshots: list[list[dict[str, Any]]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.history_snapshots.append(list(messages))
        if self.responses:
            return self.responses.pop(0)
        return {"role": "assistant", "content": "Default mock response"}


class MockMCPClient(MCPClient):
    """Mock MCP client that simulates tool execution and security policy."""

    def __init__(self, tool_results: dict[str, str] | None = None, config: Config | None = None):
        cfg = config or Config()
        super().__init__(config=cfg)
        self.tool_results = tool_results or {}
        self.executed_tools: list[tuple[str, dict[str, Any]]] = []
        self._is_running = True

    def list_tools(self, filter_allowed: bool = True) -> list[dict[str, Any]]:
        raw_tools = [
            {
                "name": "read_file",
                "description": "Read file contents",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "list_directory",
                "description": "List directory contents",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
            {
                "name": "write_file",
                "description": "Write to file",
                "input_schema": {"type": "object"},
            },
        ]
        if filter_allowed:
            return [t for t in raw_tools if t["name"] in self.config.allowed_tools]
        return raw_tools

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> str:
        args = arguments or {}
        self.executed_tools.append((name, args))
        if name in self.tool_results:
            return self.tool_results[name]
        return f"Mock output for {name}"


def test_schema_adapter_converts_mcp_to_ollama_format():
    """Verify that mcp_to_ollama_tool converts input_schema to function parameters."""
    mcp_tool = {
        "name": "read_file",
        "description": "Read file contents as text",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "path to file"}},
            "required": ["path"],
        },
    }

    ollama_tool = mcp_to_ollama_tool(mcp_tool)

    assert ollama_tool["type"] == "function"
    assert ollama_tool["function"]["name"] == "read_file"
    assert ollama_tool["function"]["description"] == "Read file contents as text"
    assert ollama_tool["function"]["parameters"] == mcp_tool["input_schema"]


def test_agent_tool_loop_end_to_end_grounded():
    """Verify tool loop: user ask -> model calls tool -> tool returns -> model answers grounded."""
    tool_results = {"read_file": "Secret code: ALPHA-99"}
    mock_mcp = MockMCPClient(tool_results=tool_results)

    # Step 1: Model issues a tool call
    # Step 2: Model receives tool result and produces grounded answer
    mock_ollama = MockOllamaClient(
        responses=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "secret.txt"},
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "The secret code in secret.txt is ALPHA-99.",
            },
        ]
    )

    agent = Agent(ollama_client=mock_ollama, mcp_client=mock_mcp)
    messages = [{"role": "user", "content": "What is the code in secret.txt?"}]

    final_answer = agent.run_turn(messages)

    assert "ALPHA-99" in final_answer
    assert len(mock_mcp.executed_tools) == 1
    assert mock_mcp.executed_tools[0] == ("read_file", {"path": "secret.txt"})

    # Check conversation history contains user, assistant tool_call, tool response, and final assistant
    assert len(messages) == 4
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert "tool_calls" in messages[1]
    assert messages[2]["role"] == "tool"
    assert "ALPHA-99" in messages[2]["content"]
    assert messages[3]["role"] == "assistant"
    assert "ALPHA-99" in messages[3]["content"]


def test_agent_blocks_disallowed_tools():
    """Verify security rule: model attempting to call a forbidden tool is blocked by host."""
    mock_mcp = MockMCPClient()

    # Model attempts to call write_file (not in allowed_tools)
    mock_ollama = MockOllamaClient(
        responses=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write_file",
                            "arguments": {"path": "hacked.txt", "content": "payload"},
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "I cannot write to files because write_file is disallowed.",
            },
        ]
    )

    agent = Agent(ollama_client=mock_ollama, mcp_client=mock_mcp)
    messages = [{"role": "user", "content": "Write hacked.txt"}]

    final_answer = agent.run_turn(messages)

    # Tool was NOT executed on the MCP client
    assert len(mock_mcp.executed_tools) == 0

    # The tool result message returned a security error to the model
    tool_msg = next(m for m in messages if m.get("role") == "tool")
    assert "Security Error" in tool_msg["content"]
    assert "write_file" in tool_msg["content"]
    assert "disallowed" in final_answer.lower() or "cannot" in final_answer.lower()


def test_agent_stops_at_max_iterations():
    """Verify runaway prevention: agent terminates safely if model loops tool calls endlessly."""
    mock_mcp = MockMCPClient()

    # Endless loop of tool calls
    infinite_tool_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "loop.txt"},
                }
            }
        ],
    }
    mock_ollama = MockOllamaClient(responses=[infinite_tool_call] * 10)

    cfg = Config(max_iterations=3)
    agent = Agent(ollama_client=mock_ollama, mcp_client=mock_mcp, config=cfg)
    messages = [{"role": "user", "content": "Keep reading loop.txt forever"}]

    answer = agent.run_turn(messages)

    assert "reached the maximum limit of 3" in answer
    # Exactly max_iterations tool calls were executed
    assert len(mock_mcp.executed_tools) == 3


def test_agent_handles_multiple_tool_calls_in_one_turn():
    """Verify handling multiple tool calls issued in a single response."""
    mock_mcp = MockMCPClient(
        tool_results={
            "read_file": "file content",
            "list_directory": "dir content",
        }
    )

    mock_ollama = MockOllamaClient(
        responses=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "list_directory", "arguments": {"path": "."}}},
                    {"function": {"name": "read_file", "arguments": {"path": "a.txt"}}},
                ],
            },
            {
                "role": "assistant",
                "content": "Found directory and read file.",
            },
        ]
    )

    agent = Agent(ollama_client=mock_ollama, mcp_client=mock_mcp)
    messages = [{"role": "user", "content": "Check folder and read a.txt"}]

    final_answer = agent.run_turn(messages)

    assert "Found directory and read file" in final_answer
    assert len(mock_mcp.executed_tools) == 2
    assert mock_mcp.executed_tools[0][0] == "list_directory"
    assert mock_mcp.executed_tools[1][0] == "read_file"

    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_messages) == 2


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Live integration test with Ollama and Filesystem MCP server",
)
def test_agent_live_end_to_end_file_find():
    """Live verification: Ollama reads test.txt via real Filesystem MCP server."""
    cfg = Config()
    workspace = cfg.workspace_dir
    os.makedirs(workspace, exist_ok=True)
    test_path = os.path.join(workspace, "phase5_target.txt")
    with open(test_path, "w") as f:
        f.write("Secret keyword is BUMBLEBEE.\n")

    with MCPClient(config=cfg) as mcp_client:
        agent = Agent(mcp_client=mcp_client, config=cfg)
        messages = [
            {
                "role": "user",
                "content": "Please read phase5_target.txt and tell me the secret keyword.",
            }
        ]
        response = agent.run_turn(messages)

        assert "BUMBLEBEE" in response.upper()
