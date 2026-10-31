"""Client for interacting with the Filesystem MCP Server using the official Python SDK."""

import asyncio
import os
import threading
from typing import Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from local_agent.config import Config


class MCPClientError(Exception):
    """Base exception for all MCP client errors."""


class MCPConnectionError(MCPClientError):
    """Raised when connecting to or communicating with the MCP server fails."""


class MCPToolNotAllowedError(MCPClientError):
    """Raised when a requested tool is not permitted by host security policy."""


class MCPToolExecutionError(MCPClientError):
    """Raised when tool execution fails or the server returns an error."""


class MCPClient:
    """Synchronous Python client for the MCP Filesystem Server.

    Manages the server subprocess lifecycle, tool discovery, and secure tool execution.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session: ClientSession | None = None
        self._stop_event: asyncio.Event | None = None
        self._is_running = False

    def start(self, timeout: float = 15.0) -> None:
        """Start the filesystem MCP server process in the background.

        Args:
            timeout: Maximum seconds to wait for server initialization.

        Raises:
            MCPConnectionError: If the server fails to start or initialize.
        """
        if self._is_running:
            return

        # Ensure the configured workspace directory exists
        workspace = os.path.abspath(self.config.workspace_dir)
        os.makedirs(workspace, exist_ok=True)

        ready_event = threading.Event()
        startup_error: list[Exception] = []

        def _worker_loop() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def _run_server_lifecycle() -> None:
                self._stop_event = asyncio.Event()
                params = StdioServerParameters(
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-filesystem", workspace],
                )
                try:
                    async with stdio_client(params) as (read_stream, write_stream):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            self._session = session
                            self._is_running = True
                            ready_event.set()
                            # Keep running until stop() signals self._stop_event
                            await self._stop_event.wait()
                except Exception as exc:
                    startup_error.append(exc)
                    ready_event.set()
                finally:
                    self._is_running = False
                    self._session = None

            try:
                self._loop.run_until_complete(_run_server_lifecycle())
            finally:
                self._loop.close()

        self._thread = threading.Thread(target=_worker_loop, daemon=True)
        self._thread.start()

        if not ready_event.wait(timeout=timeout):
            self.stop()
            raise MCPConnectionError(
                f"Timed out after {timeout} seconds waiting for MCP filesystem server to start."
            )

        if startup_error:
            self.stop()
            raise MCPConnectionError(
                f"Failed to start MCP filesystem server: {startup_error[0]}"
            )

    def stop(self, timeout: float = 5.0) -> None:
        """Shut down the MCP server process cleanly."""
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=timeout)

        self._is_running = False
        self._session = None
        self._loop = None
        self._thread = None
        self._stop_event = None

    def __enter__(self) -> "MCPClient":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def list_tools(self, filter_allowed: bool = True) -> list[dict[str, Any]]:
        """Ask the server what tools it has available.

        Args:
            filter_allowed: If True, only tools in config.allowed_tools are returned.

        Returns:
            List of tool dictionaries with 'name', 'description', and 'input_schema'.

        Raises:
            MCPConnectionError: If the MCP server is not currently running.
        """
        if not self._is_running or not self._session or not self._loop:
            raise MCPConnectionError("MCP server is not connected. Call start() first.")

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._session.list_tools(),
                self._loop,
            )
            response = future.result(timeout=10.0)
        except Exception as exc:
            raise MCPConnectionError(f"Failed to list tools from MCP server: {exc}") from exc

        tools: list[dict[str, Any]] = []
        for tool in response.tools:
            if filter_allowed and tool.name not in self.config.allowed_tools:
                continue

            tools.append({
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.input_schema or {},
            })

        return tools

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> str:
        """Execute a tool on the MCP server and return the resulting text.

        Args:
            name: Name of the tool to execute (e.g. 'read_file').
            arguments: Arguments for the tool (e.g. {'path': 'test.txt'}).
            timeout: Timeout in seconds for tool execution.

        Returns:
            The text output returned by the tool, capped by config.max_tool_result_chars.

        Raises:
            MCPConnectionError: If the server is not running.
            MCPToolNotAllowedError: If the tool is not in config.allowed_tools.
            MCPToolExecutionError: If the tool execution fails or returns an error.
        """
        if not self._is_running or not self._session or not self._loop:
            raise MCPConnectionError("MCP server is not connected. Call start() first.")

        # Security check: Read-only allowlist
        if name not in self.config.allowed_tools:
            raise MCPToolNotAllowedError(
                f"Tool '{name}' is not allowed by security policy. Allowed tools: {list(self.config.allowed_tools)}"
            )

        args = arguments or {}

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._session.call_tool(name, args),
                self._loop,
            )
            result = future.result(timeout=timeout)
        except Exception as exc:
            raise MCPConnectionError(f"Failed to communicate with MCP server: {exc}") from exc

        # Extract text content from result
        content_parts: list[str] = []
        for item in result.content:
            if hasattr(item, "text") and item.text:
                content_parts.append(item.text)

        result_text = "\n".join(content_parts).strip()

        # Handle tool errors reported by server (e.g. access denied)
        if result.is_error:
            raise MCPToolExecutionError(
                f"Tool '{name}' returned an error: {result_text or 'Unknown error'}"
            )

        # Security check: Cap tool result size
        max_chars = self.config.max_tool_result_chars
        if len(result_text) > max_chars:
            result_text = (
                result_text[:max_chars]
                + f"\n... [Output truncated: exceeded limit of {max_chars} characters]"
            )

        return result_text


if __name__ == "__main__":
    import os

    cfg = Config()
    # Ensure workspace and sample file exist
    sample = os.path.join(cfg.workspace_dir, "test.txt")
    if not os.path.exists(sample):
        os.makedirs(cfg.workspace_dir, exist_ok=True)
        with open(sample, "w") as f:
            f.write("Hello from workspace!\n")

    print("=== Phase 4: MCP Client Demonstration ===")
    print("1. Connecting to Filesystem MCP Server...")
    with MCPClient(config=cfg) as client:
        print("   Connected successfully.")

        print("\n2. Discovering allowed tools and inspecting schemas:")
        tools = client.list_tools(filter_allowed=True)
        for t in tools:
            print(f"   - {t['name']}: {t['description']}")

        print("\n3. Executing direct tool call (read_file on test.txt):")
        result = client.call_tool("read_file", {"path": sample})
        print(f"   Output: {result.strip()}")

        print("\n4. Testing error handling (disallowed tool):")
        try:
            client.call_tool("write_file", {"path": "bad.txt", "content": "test"})
        except MCPToolNotAllowedError as e:
            print(f"   Caught expected security error: {e}")

        print("\n5. Testing error handling (outside workspace boundary):")
        try:
            client.call_tool("read_file", {"path": "/etc/passwd"})
        except MCPToolExecutionError as e:
            print(f"   Caught expected boundary error: {e}")

    print("\n6. Disconnected cleanly.")
    print("=========================================")
