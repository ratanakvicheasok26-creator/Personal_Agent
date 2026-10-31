"""Tests for Filesystem MCP Server verification (Phase 3)."""

import json
import os
import subprocess
import pytest
from local_agent.config import Config


@pytest.fixture(scope="module")
def mcp_server():
    """Start the official filesystem MCP server pointing to the configured workspace."""
    config = Config()
    workspace = config.workspace_dir
    os.makedirs(workspace, exist_ok=True)

    # Ensure a sample file exists for testing
    sample_file = os.path.join(workspace, "test.txt")
    if not os.path.exists(sample_file):
        with open(sample_file, "w") as f:
            f.write("Hello from workspace!\n")

    proc = subprocess.Popen(
        ["npx", "-y", "@modelcontextprotocol/server-filesystem", workspace],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def send_rpc(msg: dict):
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()

    def read_rpc() -> dict:
        while True:
            line = proc.stdout.readline()
            if not line:
                stderr = proc.stderr.read()
                raise RuntimeError(f"Server closed unexpectedly. Stderr: {stderr}")
            line = line.strip()
            if line:
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

    # Initialize handshake
    send_rpc({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
    })
    init_res = read_rpc()
    assert "result" in init_res
    send_rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})

    yield {"proc": proc, "send": send_rpc, "read": read_rpc, "workspace": workspace}

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_mcp_server_discovers_tools(mcp_server):
    """Verify that the Filesystem MCP Server starts and tools are discoverable."""
    send = mcp_server["send"]
    read = mcp_server["read"]

    send({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    })
    res = read()
    tools = res.get("result", {}).get("tools", [])
    tool_names = [t["name"] for t in tools]

    assert "read_file" in tool_names
    assert "list_directory" in tool_names
    assert "search_files" in tool_names


def test_mcp_server_reads_workspace_file(mcp_server):
    """Verify that files within the workspace can be listed and read."""
    send = mcp_server["send"]
    read = mcp_server["read"]
    test_file = os.path.join(mcp_server["workspace"], "test.txt")

    send({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "read_file",
            "arguments": {"path": test_file},
        },
    })
    res = read()
    content = res.get("result", {}).get("content", [{}])[0].get("text", "")
    assert "Hello from workspace!" in content


def test_mcp_server_denies_outside_workspace(mcp_server):
    """Verify that access outside the workspace is denied by the server."""
    send = mcp_server["send"]
    read = mcp_server["read"]

    send({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "read_file",
            "arguments": {"path": "/etc/passwd"},
        },
    })
    res = read()
    result_data = res.get("result", {})
    is_error = result_data.get("isError", False) or "error" in res
    error_text = ""
    if "content" in result_data and result_data["content"]:
        error_text = result_data["content"][0].get("text", "")

    assert is_error or "Access denied" in error_text
