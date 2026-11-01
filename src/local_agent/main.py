"""Terminal chat interface for the local AI agent."""

import sys
from typing import Any, Callable
from local_agent.config import Config
from local_agent.ollama_client import OllamaClient, OllamaError
from local_agent.mcp_client import MCPClient
from local_agent.agent import Agent

EXIT_COMMANDS = {"exit", "quit", "q", ":q"}


def run_chat_loop(
    client: OllamaClient | None = None,
    agent: Agent | None = None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
) -> None:
    """Run an interactive multi-turn chat session in the terminal.

    Args:
        client: Optional OllamaClient instance (used if agent is None).
        agent: Optional Agent instance. If None, created with client.
        input_fn: Function to read user input (defaults to builtin `input`).
        print_fn: Function to display output (defaults to builtin `print`).
    """
    if agent is None:
        if client is None:
            client = OllamaClient()
        agent = Agent(ollama_client=client, config=client.config)

    model_name = agent.config.default_model
    print_fn("=" * 50)
    print_fn(f"Local AI Agent - Terminal Chat (Model: {model_name})")
    print_fn("Type your message and press Enter.")
    print_fn("Type 'exit', 'quit', or press Ctrl+C to end the chat.")
    print_fn("=" * 50 + "\n")

    messages: list[dict[str, Any]] = []

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
            assistant_content = agent.run_turn(messages=messages)
            print_fn(" " * 20, end="\r")  # Clear "Thinking..." line
            print_fn(f"Assistant: {assistant_content.strip()}\n")
        except OllamaError as err:
            print_fn(" " * 20, end="\r")  # Clear "Thinking..." line
            print_fn(f"\n[Error] {err}\n")
            # Remove the last user message so the failed turn doesn't corrupt history
            messages.pop()
        except Exception as err:
            print_fn(" " * 20, end="\r")  # Clear "Thinking..." line
            print_fn(f"\n[Error] {err}\n")
            messages.pop()


def main() -> None:
    """Entry point for the command-line script."""
    try:
        config = Config()
        ollama_client = OllamaClient(config=config)
        with MCPClient(config=config) as mcp_client:
            agent = Agent(
                ollama_client=ollama_client,
                mcp_client=mcp_client,
                config=config,
                on_tool_call=lambda name, args: print(f"[Tool] Invoking {name}({args})..."),
            )
            run_chat_loop(agent=agent)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
