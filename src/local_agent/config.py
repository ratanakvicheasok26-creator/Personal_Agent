"""Configuration settings for the local AI agent."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Config:
    """Application settings with environment variable fallbacks."""

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    default_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    request_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "60.0"))

    # MCP Filesystem Server settings
    workspace_dir: str = os.getenv(
        "AGENT_WORKSPACE",
        os.path.expanduser("~/Documents/AI-Agent-Workspace"),
    )
    # Host-side read-only allowlist (security rule)
    allowed_tools: tuple[str, ...] = (
        "read_file",
        "read_text_file",
        "list_directory",
        "list_directory_with_sizes",
        "directory_tree",
        "search_files",
        "get_file_info",
        "list_allowed_directories",
    )
    # Result size cap (security rule)
    max_tool_result_chars: int = int(os.getenv("MAX_TOOL_RESULT_CHARS", "10000"))
