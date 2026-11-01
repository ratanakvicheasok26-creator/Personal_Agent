# Local AI Agent (Ollama + MCP)

A lightweight, secure terminal AI assistant running completely offline on your local Linux machine. It uses a local Ollama model paired with the official Model Context Protocol (MCP) Filesystem Server to inspect, search, and read files within a designated workspace.

Built with Python 3.11+ and the official MCP Python SDK, adhering strictly to a minimalistic architecture with zero bloat (no LangChain, no vector databases, no web frameworks).

---

## Architecture

The Python Host acts as the central coordinator (orchestrator), owning the conversation history, validation, security filtering, and the iterative tool execution loop:

```
                      +-------------------+
                      |   User Terminal   |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |    Python Host    |
                      |  (Agent Loop)     |
                      +---+-----------+---+
                          |           |
             Ollama API   |           |  MCP SDK (JSON-RPC)
       (schema adapted)   |           |
                          v           v
                 +------------+   +------------------------+
                 |   Ollama   |   | Filesystem MCP Server  |
                 | (Local LLM)|   | (Node.js via stdio)    |
                 +------------+   +-----------+------------+
                                              |
                                              v
                                  +------------------------+
                                  |    Target Workspace    |
                                  | ~/Documents/...        |
                                  +------------------------+
```

### Components
1. **Host (`Agent` in `src/local_agent/agent.py`)**:
   * Translates MCP tool definitions into Ollama tool/function schemas (`mcp_to_ollama_tool`).
   * Enforces a host-side allowlist so forbidden tools (e.g., `write_file`, `delete_file`) are rejected before execution.
   * Manages conversational turns and loop termination (`max_iterations`).
2. **Ollama Client (`src/local_agent/ollama_client.py`)**:
   * Communicates with Ollama's HTTP API (`/api/chat`).
   * Handles timeouts, model status, and error translation.
3. **MCP Client (`src/local_agent/mcp_client.py`)**:
   * Spawns the official `@modelcontextprotocol/server-filesystem` child process via `npx` over `stdio`.
   * Enforces result truncation caps to prevent memory bloat or model context overflow.
4. **Terminal Chat (`src/local_agent/main.py`)**:
   * Multi-turn conversational interface with clean exit handling (`exit`, `quit`, `:q`, Ctrl+C).

---

## Prerequisites

Ensure your system has the following installed:
* **Linux OS** (Ubuntu 22.04+ or similar)
* **Python 3.11+**
* **`uv` package manager**: Modern, fast Python package and environment manager
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
* **Node.js (v18+) and `npx`**: Required to run the official MCP filesystem server
  ```bash
  node --version && npx --version
  ```
* **Ollama**: Local model server
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```
* **Ollama Tool-Capable Model**:
  Recommended smallest tool-capable model:
  ```bash
  ollama pull qwen2.5:3b
  ```

---

## Installation & Setup

1. **Clone or navigate to the repository:**
   ```bash
   cd Personal_Agent
   ```

2. **Sync Python virtual environment with `uv`:**
   ```bash
   uv sync
   ```

3. **Prepare the secure workspace folder:**
   ```bash
   mkdir -p ~/Documents/AI-Agent-Workspace
   ```
   Add sample files to test:
   ```bash
   echo "Meeting notes: Project Alpha deadline is Friday." > ~/Documents/AI-Agent-Workspace/notes.txt
   ```

4. **Verify Ollama is active:**
   ```bash
   ollama list
   ```

---

## Configuration

All configuration is managed through environment variables with sensible defaults in `src/local_agent/config.py`:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL of the local Ollama daemon |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Default Ollama model to use |
| `OLLAMA_TIMEOUT` | `120.0` | Timeout in seconds for Ollama requests |
| `AGENT_WORKSPACE` | `~/Documents/AI-Agent-Workspace` | Absolute path to the sandboxed folder |
| `MAX_TOOL_RESULT_CHARS` | `10000` | Maximum characters returned by any single tool |
| `MAX_TOOL_ITERATIONS` | `5` | Maximum tool execution loops per user turn |

---

## Running the Agent

Start the terminal agent:
```bash
uv run local-agent
```
Or directly via Python module:
```bash
uv run python -m local_agent.main
```

### Example Session
```text
==================================================
Local AI Agent - Terminal Chat (Model: qwen2.5:3b)
Type your message and press Enter.
Type 'exit', 'quit', or press Ctrl+C to end the chat.
==================================================

You: What files are in the workspace?
[Tool] Invoking list_directory({'path': '.'})...
Assistant: In your workspace, there is currently one file: `notes.txt`.

You: What is inside notes.txt?
[Tool] Invoking read_file({'path': 'notes.txt'})...
Assistant: The file `notes.txt` contains: "Meeting notes: Project Alpha deadline is Friday."

You: exit
Goodbye!
```

---

## Security Limits & Sandboxing

The agent enforces multi-layered defense-in-depth principles:

1. **Workspace Sandboxing:** The MCP filesystem server process is spawned with only the allowed directory argument (`~/Documents/AI-Agent-Workspace`). Attempts to traverse outside via `../` or absolute paths are blocked by the server.
2. **Host-Side Read-Only Allowlist:** The Python host will never invoke mutation tools (`write_file`, `create_directory`, `move_file`). Only safe inspection tools are permitted:
   * `read_file`, `read_text_file`
   * `list_directory`, `list_directory_with_sizes`, `directory_tree`
   * `search_files`, `get_file_info`, `list_allowed_directories`
3. **No Code or Shell Execution:** The agent has zero access to bash, terminal execution, or arbitrary Python evaluation.
4. **Untrusted Data Isolation:** The system prompt explicitly instructs the LLM that all retrieved file contents and tool outputs are untrusted data, not prompt instructions.
5. **Output Capping:** Tool results are strictly capped at 10,000 characters to prevent denial-of-service and context blowing.
6. **Runaway Loop Protection:** The agent loop is hard-capped at 5 iterations per turn.
7. **Offline Operation:** Once the model is pulled and npm dependencies are cached, the agent runs completely offline without sending telemetry or data externally.

---

## Troubleshooting

### 1. `Could not connect to Ollama`
* Check if Ollama is running:
  ```bash
  systemctl status ollama
  # Or start it manually:
  ollama serve
  ```

### 2. Request timed out after 60s / 120s
* Models running on laptop CPUs may take longer during initial prompt evaluation with tool definitions. Increase the timeout:
  ```bash
  export OLLAMA_TIMEOUT=180.0
  uv run local-agent
  ```

### 3. `Failed to start MCP filesystem server`
* Ensure `npx` is functional and can download or run `@modelcontextprotocol/server-filesystem`:
  ```bash
  npx -y @modelcontextprotocol/server-filesystem --help
  ```

### 4. `Security Error: Tool 'write_file' is not allowed`
* The agent is intentionally read-only. Any attempt by the model to modify or delete files is blocked by design.

---

## Testing

Run the automated test suite using `pytest`:

```bash
# Run all unit tests (offline and deterministic with mocks)
uv run pytest -k "not live"

# Run all tests including live integration with local Ollama and MCP
uv run pytest
```
