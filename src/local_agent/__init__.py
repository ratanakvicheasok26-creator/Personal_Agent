"""Local AI Agent package."""

from local_agent.config import Config
from local_agent.ollama_client import OllamaClient
from local_agent.mcp_client import (
    MCPClient,
    MCPClientError,
    MCPConnectionError,
    MCPToolNotAllowedError,
    MCPToolExecutionError,
)
from local_agent.agent import Agent, mcp_to_ollama_tool

__version__ = "0.1.0"

__all__ = [
    "Config",
    "OllamaClient",
    "MCPClient",
    "MCPClientError",
    "MCPConnectionError",
    "MCPToolNotAllowedError",
    "MCPToolExecutionError",
    "Agent",
    "mcp_to_ollama_tool",
]
