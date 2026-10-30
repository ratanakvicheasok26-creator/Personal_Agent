"""Terminal chat interface for the local AI agent."""

import sys
from typing import Callable
from local_agent.config import Config
from local_agent.ollama_client import OllamaClient, OllamaError

EXIT_COMMANDS = {"exit", "quit", "q", ":q"}


def run_chat_loop(
    client: OllamaClient | None = None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
) -> None:
    """Run an interactive multi-turn chat session in the terminal.

    Args:
        client: OllamaClient instance. If None, a default one is created.
        input_fn: Function to read user input (defaults to builtin `input`).
        print_fn: Function to display output (defaults to builtin `print`).
    """
    if client is None:
        client = OllamaClient()

    model_name = client.config.default_model
    print_fn("=" * 50)
    print_fn(f"Local AI Agent - Terminal Chat (Model: {model_name})")
    print_fn("Type your message and press Enter.")
    print_fn("Type 'exit', 'quit', or press Ctrl+C to end the chat.")
    print_fn("=" * 50 + "\n")

    messages: list[dict[str, str]] = []

    while True:
        try:
            user_input = input_fn("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print_fn("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print_fn("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            print_fn("\nThinking...", end="\r", flush=True)
            response = client.chat(messages=messages)
            assistant_content = response.get("content", "").strip()
            print_fn(" " * 20, end="\r")  # Clear "Thinking..." line
            print_fn(f"Assistant: {assistant_content}\n")
            messages.append({"role": "assistant", "content": assistant_content})
        except OllamaError as err:
            print_fn(" " * 20, end="\r")  # Clear "Thinking..." line
            print_fn(f"\n[Error] {err}\n")
            # Remove the last user message so the failed turn doesn't corrupt history
            messages.pop()


def main() -> None:
    """Entry point for the command-line script."""
    try:
        config = Config()
        client = OllamaClient(config=config)
        run_chat_loop(client=client)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
