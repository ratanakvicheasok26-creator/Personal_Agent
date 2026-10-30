"""Client for communicating with the local Ollama REST API."""

from typing import Any
import httpx

from local_agent.config import Config


class OllamaError(Exception):
    """Base exception for Ollama errors."""


class OllamaConnectionError(OllamaError):
    """Raised when Ollama cannot be reached (e.g. server is down)."""


class OllamaModelNotFoundError(OllamaError):
    """Raised when the requested model is not installed in Ollama."""


class OllamaTimeoutError(OllamaError):
    """Raised when a request to Ollama times out."""


class OllamaClient:
    """Simple HTTP client for the Ollama chat API."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Send a list of messages to Ollama and return the assistant response message.

        Args:
            messages: Conversation history, e.g. [{"role": "user", "content": "Hello"}]
            model: Model name. If None, uses config.default_model.
            tools: Optional tool definitions formatted for Ollama.

        Returns:
            The assistant's message dictionary, e.g. {"role": "assistant", "content": "..."}

        Raises:
            OllamaConnectionError: If Ollama is not running.
            OllamaModelNotFoundError: If the specified model is not downloaded.
            OllamaTimeoutError: If the request times out.
            OllamaError: For any other API or HTTP error.
        """
        target_model = model or self.config.default_model
        payload: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        url = f"{self.config.ollama_base_url.rstrip('/')}/api/chat"

        try:
            with httpx.Client(timeout=self.config.request_timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("message", {})

        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at '{self.config.ollama_base_url}'.\n"
                "Is the Ollama service running?\n"
                "Try running 'ollama serve' in your terminal or check 'systemctl status ollama'."
            ) from exc

        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Request to Ollama timed out after {self.config.request_timeout} seconds.\n"
                "The model might still be loading into RAM. Consider increasing OLLAMA_TIMEOUT."
            ) from exc

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise OllamaModelNotFoundError(
                    f"Model '{target_model}' was not found in Ollama.\n"
                    f"Please run 'ollama pull {target_model}' to download it first."
                ) from exc

            error_text = exc.response.text.strip()
            raise OllamaError(
                f"Ollama API returned HTTP {exc.response.status_code}: {error_text}"
            ) from exc

        except Exception as exc:
            raise OllamaError(f"Unexpected error communicating with Ollama: {exc}") from exc
