"""Tests for MCPClient implementation (Phase 4)."""

import os
import pytest
from local_agent.config import Config
from local_agent.mcp_client import (
    MCPClient,
    MCPConnectionError,
    MCPToolNotAllowedError,
    MCPToolExecutionError,
)


@pytest.fixture(scope="module")
def workspace_with_file():
    """Ensure the test workspace has a dedicated test file."""
    config = Config()
    workspace = config.workspace_dir
    os.makedirs(workspace, exist_ok=True)
    sample_file = os.path.join(workspace, "phase4_sample.txt")
    with open(sample_file, "w") as f:
        f.write("Hello from Phase 4 workspace!\n")
    # Also restore test.txt for Phase 3 tests if needed
    base_file = os.path.join(workspace, "test.txt")
    with open(base_file, "w") as f:
        f.write("Hello from workspace!\n")
    return workspace, sample_file


def test_mcp_client_not_started_raises_error():
    """Verify that calling methods before start() raises MCPConnectionError."""
    client = MCPClient()
    with pytest.raises(MCPConnectionError) as exc_info:
        client.list_tools()
    assert "Call start() first" in str(exc_info.value)

    with pytest.raises(MCPConnectionError) as exc_info:
        client.call_tool("read_file", {"path": "test.txt"})
    assert "Call start() first" in str(exc_info.value)


def test_mcp_client_start_and_stop():
    """Verify clean start and stop lifecycle."""
    client = MCPClient()
    client.start()
    assert client._is_running is True

    client.stop()
    assert client._is_running is False


def test_mcp_client_context_manager():
    """Verify MCPClient works as a context manager with automatic cleanup."""
    with MCPClient() as client:
        assert client._is_running is True
        tools = client.list_tools()
        assert len(tools) > 0
    assert client._is_running is False


def test_mcp_client_list_tools(workspace_with_file):
    """Verify discovering tools and applying the host-side allowlist."""
    with MCPClient() as client:
        # Default filtered tools
        allowed_tools = client.list_tools(filter_allowed=True)
        tool_names = [t["name"] for t in allowed_tools]

        assert "read_file" in tool_names
        assert "list_directory" in tool_names
        # Host-side security rule: write/edit/delete tools must not be in allowed list
        assert "write_file" not in tool_names
        assert "delete_file" not in tool_names

        # Verify schema structure
        for tool in allowed_tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert isinstance(tool["input_schema"], dict)

        # Unfiltered tools discovery
        all_tools = client.list_tools(filter_allowed=False)
        assert len(all_tools) >= len(allowed_tools)


def test_mcp_client_call_tool_read_file(workspace_with_file):
    """Verify calling read_file returns the file content."""
    workspace, sample_file = workspace_with_file

    with MCPClient() as client:
        output = client.call_tool("read_file", {"path": sample_file})
        assert "Hello from Phase 4 workspace!" in output


def test_mcp_client_blocks_disallowed_tool():
    """Verify security rule: calling tools outside allowlist is rejected by host."""
    with MCPClient() as client:
        with pytest.raises(MCPToolNotAllowedError) as exc_info:
            client.call_tool("write_file", {"path": "danger.txt", "content": "bad"})

        assert "not allowed by security policy" in str(exc_info.value)


def test_mcp_client_denies_outside_workspace():
    """Verify security rule: access outside the workspace is denied with MCPToolExecutionError."""
    with MCPClient() as client:
        with pytest.raises(MCPToolExecutionError) as exc_info:
            client.call_tool("read_file", {"path": "/etc/passwd"})

        assert "Access denied" in str(exc_info.value) or "error" in str(exc_info.value).lower()


def test_mcp_client_caps_result_size(workspace_with_file):
    """Verify security rule: output exceeding max_tool_result_chars is truncated."""
    workspace, sample_file = workspace_with_file
    # Create client with a small limit for testing
    small_limit_config = Config(max_tool_result_chars=10)

    with MCPClient(config=small_limit_config) as client:
        output = client.call_tool("read_file", {"path": sample_file})
        assert len(output) > 10  # Because of the truncation notice
        assert "Output truncated: exceeded limit of 10 characters" in output
        assert output.startswith("Hello from")
