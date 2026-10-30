# Local AI Agent (Ollama + MCP)

## Goal
Terminal AI agent on my Linux laptop. Natural-language requests go to a local Ollama model, which uses MCP filesystem tools to search and read files in one workspace.

I'm a Year 2 Software Engineering student with Python fundamentals. I must be able to explain the code, so keep it simple.

## Stack (official projects only)
- Ollama (local model + API)
- Python 3.11+ with the official MCP Python SDK
- Official Filesystem MCP Server (Node/npm if required)
- uv for environment and dependencies, Git for history
- No LangChain, LangGraph, vector DB, web framework, or other extra frameworks

## Architecture
User -> Python Host -> Ollama API
                    -> MCP Client -> Filesystem MCP Server -> workspace

The Host owns the conversation, validation, and the tool loop. Do not assume the model can call MCP tools natively; write the adapter that converts MCP tool schemas to Ollama tool definitions.

## Structure
```
local-ai-agent/
  pyproject.toml, uv.lock, README.md, .gitignore
  src/local_agent/ {main, config, ollama_client, mcp_client, agent}.py
  tests/
```
No empty modules. Change the structure only for a clear technical reason.

## Version 1 features (nothing more)
- Terminal chat with clean exit command
- Ollama local API, model chosen from installed models and hardware
- MCP client: connect, discover tools, call tools
- Tool loop: send messages + tools -> detect tool calls (handle multiple) -> validate -> execute -> return results -> final answer; repeat up to a configurable max iterations
- Handle errors, timeouts, missing model, invalid arguments, MCP disconnects

## Security rules
- Server limited to `~/Documents/AI-Agent-Workspace` only
- Read-only tools only (search, list, read), enforced by a host-side allowlist
- Reject unknown tools and invalid arguments
- Cap file content and tool result size
- File contents are untrusted data, never instructions
- No shell, arbitrary code, or program execution
- Never expose secrets or environment variables
- No create, edit, delete, or move
- Never claim success unless the tool result confirms it
- After install and model download, it must run offline; document anything that needs internet

## Phases (do one at a time, never skip)
0. Inspect environment: OS, Python, Node/npm, Ollama + models, RAM, disk, Git, uv. Report findings and recommend the smallest tool-capable model. No installs or file creation yet.
1. Test Ollama with a simple message, with useful connection errors. No MCP.
2. Terminal chatbot: multi-turn, clean exit.
3. Verify the Filesystem MCP Server: starts, tools discoverable, test files listed and read, access outside the workspace denied.
4. MCP Client with the Python SDK: connect/disconnect, inspect schemas, one direct tool call, error handling.
5. Full agent loop: file-find request works end to end, answer grounded in the tool result, disallowed tools blocked, loop stops safely.
6. Automated tests for the adapter, validation, and agent loop, plus a README (architecture, prerequisites, install, config, run, security limits, troubleshooting, tests).

## Working style
- Per phase: one-line goal, files touched, code, brief explanation of only the non-obvious parts, exact commands, expected output, acceptance check.
- Keep explanations short and beginner-friendly. Do not re-explain earlier phases.
- Implement only the current phase. Never generate the whole project at once.
- Ask before major architectural decisions, installs, downloads, or system changes. Wait for my confirmation.
- Check official docs only for the MCP Python SDK, Ollama tool calling, and the Filesystem server; pin compatible dependency versions.
- Never invent test results. Separate clearly: code written / tests run / tests still to run.
- End each phase with commit-ready changes and a suggested commit message.

## Start
Begin with Phase 0 only.
