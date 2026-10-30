"""Configuration settings for the local AI agent."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Config:
    """Application settings with environment variable fallbacks."""

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    default_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    request_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "60.0"))
