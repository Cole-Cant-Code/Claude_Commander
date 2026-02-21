# Claude Commander

**One prompt, thirteen models.** An [MCP](https://modelcontextprotocol.io/) server that orchestrates multiple LLMs through Ollama — so your AI coding agent can debate, vote, review, benchmark, and stress-test ideas across models in a single tool call.

> *"Instead of asking one model and hoping it's right, ask thirteen and find out where they agree."*

**Works with [Claude Code](https://docs.anthropic.com/en/docs/claude-code) and [Codex](https://openai.com/index/codex/).** Set up once, switch between clients freely — same server, same tools, same results. Also supports Gemini CLI and Kimi.

---

## Why?

Every LLM has blind spots. A single model can hallucinate confidently, miss edge cases, or anchor on a mediocre approach. Claude Commander fixes this by giving your agent access to **30+ orchestration tools** that compose multiple models into collaborative patterns:

- **Get a second opinion** — or a thirteenth — on any question
- **Catch bugs that one reviewer misses** — parallel code review from three independent models, merged by severity
- **Stress-test your code** — iterative red-team attacks find flaws a single pass never would
- **Eliminate AI slop** — detect filler, hallucinated citations, and vague hedging before it ships
- **Find the best answer** — anonymous blind taste tests and peer-ranked leaderboards, no bias

All of this happens through standard MCP tool calls. No new CLI to learn, no new UI. Your existing agent — whether it's Claude Code or Codex — just gains superpowers.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Tools at a Glance](#tools-at-a-glance)
  - [Primitives](#primitives)
  - [Orchestration](#orchestration)
  - [Verification & Quality](#verification--quality)
  - [Profiles & Pipelines](#profiles--pipelines)
  - [Task Execution](#task-execution)
- [Models](#models)
- [Usage Examples](#usage-examples)
- [Client Setup](#client-setup)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Known Limitations](#known-limitations)
- [Testing](#testing)
- [Dependencies](#dependencies)

---

## Quick Start

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/), and a running [Ollama](https://ollama.com/) instance.

```bash
# 1. Clone & install
git clone https://github.com/Cole-Cant-Code/Claude_Commander.git
cd Claude_Commander
uv sync

# 2. Point to your Ollama instance
export OLLAMA_BASE_URL="http://your-ollama-host:11434"

# 3. Add to your MCP client (example: Claude Code)
claude mcp add claude-commander -- \
  uv run --project /path/to/Claude_Commander \
  fastmcp run /path/to/Claude_Commander/src/claude_commander/server.py:mcp
```

That's it. Your agent now has access to all 30+ tools. See [Client Setup](#client-setup) for Codex, Gemini CLI, and Kimi configs.

---

## Tools at a Glance

### Primitives

| Tool | What it does |
|---|---|
| [`call_model`](src/claude_commander/server.py) | Call a single model or [profile](#profiles--pipelines) |
| [`auto_call`](src/claude_commander/server.py) | Auto-route to the best-fit model with fallback retries and optional time budgets |
| [`swarm`](src/claude_commander/server.py) | Fan-out to multiple models in parallel (default 6, up to 13) |
| [`list_models`](src/claude_commander/registry.py) | Show all registered models with availability status |
| [`health_check`](src/claude_commander/server.py) | Verify Ollama connectivity |

### Orchestration

| Tool | Pattern | What it does |
|---|---|---|
| [`debate`](src/claude_commander/server.py) | Multi-round sequential | Two models argue back and forth, each seeing the full transcript |
| [`vote`](src/claude_commander/server.py) | Parallel + extraction | Models vote from fixed options; multi-strategy parser extracts results |
| [`consensus`](src/claude_commander/server.py) | Parallel + judge | Swarm a question, then a judge synthesizes agreement and disagreement |
| [`code_review`](src/claude_commander/server.py) | Parallel + merge | Independent reviewers find bugs, security issues, and performance problems — merged by severity |
| [`multi_solve`](src/claude_commander/server.py) | Parallel | Multiple models independently solve the same coding problem |
| [`benchmark`](src/claude_commander/server.py) | Prompt x model matrix | Run N prompts across M models with per-model latency stats |
| [`rank`](src/claude_commander/server.py) | Parallel + peer eval | Models answer, then peer-judge each other 1–10 for a leaderboard |
| [`chain`](src/claude_commander/server.py) | Sequential pipeline | Output of step N feeds into step N+1 for iterative refinement |
| [`map_reduce`](src/claude_commander/server.py) | Fan-out + reduce | Parallel responses synthesized through a custom reducer |
| [`blind_taste_test`](src/claude_commander/server.py) | Anonymous comparison | Responses labeled A/B/C with a reveal mapping — no anchoring bias |
| [`contrarian`](src/claude_commander/server.py) | Thesis + antithesis | One model answers, another finds logical gaps and argues alternatives |

### Verification & Quality

These tools help you **catch problems before they ship**:

| Tool | What it does |
|---|---|
| [`verify`](src/claude_commander/server.py) | Cross-model fact verification — extracts individual claims and checks each one |
| [`red_team`](src/claude_commander/server.py) | Iterative adversarial stress testing — attacker finds flaws, defender patches, attacker escalates |
| [`quality_gate`](src/claude_commander/server.py) | Pass/fail evaluation against criteria — a checkpoint for AI pipelines |
| [`detect_slop`](src/claude_commander/server.py) | Multi-model detection of filler phrases, hallucinated citations, and vague hedging |

### Profiles & Pipelines

Save and reuse model configurations. Profiles store model + parameters; pipelines chain profiles into multi-step workflows.

| Tool | What it does |
|---|---|
| [`create_profile`](src/claude_commander/profile_store.py) | Save a named model + temperature + system prompt combo |
| [`get_profile`](src/claude_commander/profile_store.py) / [`list_profiles`](src/claude_commander/profile_store.py) | Inspect or list saved profiles |
| [`clone_profile`](src/claude_commander/profile_store.py) | Duplicate a profile with modifications |
| [`delete_profile`](src/claude_commander/profile_store.py) | Remove a custom profile |
| [`create_pipeline`](src/claude_commander/pipeline_store.py) / [`run_pipeline`](src/claude_commander/pipeline_store.py) | Save and execute multi-step profile chains |
| [`list_pipelines`](src/claude_commander/pipeline_store.py) / [`delete_pipeline`](src/claude_commander/pipeline_store.py) | Manage saved pipelines |

**12 builtin profiles** ship out of the box (see [`defaults.py`](src/claude_commander/defaults.py)):

`fast-general` · `deep-reasoner` · `code-specialist` · `thinking-judge` · `creative-writer` · `factual-analyst` · `vision-analyzer` · `quick-draft` · `strict-verifier` · `adversarial-attacker` · `quality-judge` · `slop-detector`

**6 builtin pipelines:** `draft-then-refine` · `code-review-pipeline` · `creative-to-critical` · `verify-then-refine` · `red-team-then-harden` · `full-quality-check`

### Task Execution

| Tool | What it does |
|---|---|
| [`exec_task`](src/claude_commander/cli.py) | Delegate coding tasks to local CLI agents — Claude Code, Codex, Gemini CLI, or Kimi |

---

## Models

13 Ollama cloud models across four categories, plus 4 local CLI agents:

| Category | Models |
|---|---|
| **General** | GLM-5, GLM-4.7, MiniMax M2.5, MiniMax M2.1, GPT-OSS 20B, GPT-OSS 120B, Qwen3 Next 80B, Kimi K2.5 |
| **Code** | Qwen3 Coder Next |
| **Reasoning** | DeepSeek v3.2, Kimi K2 Thinking |
| **Vision** | Qwen3 VL 235B, Qwen3 VL 235B Instruct |
| **CLI Agents** | Claude Code, Codex, Gemini CLI, Kimi CLI |

All tools accept an optional `models` parameter to override defaults. The full catalog is defined in [`registry.py`](src/claude_commander/registry.py).

---

## Usage Examples

These are MCP tool calls — your client invokes them directly.

**Get a code review from three independent models:**
```python
code_review(code="def connect(host): ...", language="python")
```

**Debate a design decision:**
```python
debate(prompt="Is Rust better than Go for CLI tools?", rounds=3)
```

**Auto-route with fallback and time budget:**
```python
auto_call(prompt="Review this function for bugs", task="code", strategy="quality", max_time_ms=15000)
```

**Vote on an architecture choice:**
```python
vote(prompt="Best approach?", options=["monolith", "microservices", "serverless"])
```

**Red-team your API design:**
```python
red_team(content="POST /users accepts {name, email, role} and creates a user...", rounds=3)
```

**Run a sequential refinement pipeline:**
```python
chain(
  prompt="Explain quantum computing to a 10-year-old",
  pipeline=["glm-5:cloud", "deepseek-v3.2:cloud", "kimi-k2-thinking:cloud"]
)
```

**Anonymous blind taste test:**
```python
blind_taste_test(prompt="Explain monads simply", count=4)
```

**Peer-ranked leaderboard:**
```python
rank(prompt="Write a haiku about recursion")
```

---

## Client Setup

Both primary clients are shown below. The same server works identically with either — pick whichever you prefer, or use both.

### Claude Code

```bash
claude mcp add claude-commander -- \
  uv run --project /path/to/Claude_Commander \
  fastmcp run /path/to/Claude_Commander/src/claude_commander/server.py:mcp
```

Or add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "claude-commander": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/Claude_Commander", "fastmcp", "run", "/path/to/Claude_Commander/src/claude_commander/server.py:mcp"],
      "env": { "OLLAMA_BASE_URL": "http://your-ollama-host:11434" }
    }
  }
}
```

Agent instructions: [`CLAUDE.md`](CLAUDE.md)

### Codex

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.codex-commander]
command = "uv"
args = ["run", "--project", "/path/to/Claude_Commander", "fastmcp", "run", "/path/to/Claude_Commander/src/claude_commander/server.py:mcp"]

[mcp_servers.codex-commander.env]
MCP_SERVER_NAME = "Codex Commander"
OLLAMA_BASE_URL = "http://your-ollama-host:11434"
```

Agent instructions: [`AGENTS.md`](AGENTS.md)

<details>
<summary><strong>Gemini CLI</strong></summary>

```bash
gemini mcp add -e OLLAMA_BASE_URL=http://your-ollama-host:11434 gemini-commander -- \
  uv run --project /path/to/Claude_Commander \
  fastmcp run /path/to/Claude_Commander/src/claude_commander/server.py:mcp
```

Or add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "gemini-commander": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/Claude_Commander", "fastmcp", "run", "/path/to/Claude_Commander/src/claude_commander/server.py:mcp"],
      "env": { "OLLAMA_BASE_URL": "http://your-ollama-host:11434" }
    }
  }
}
```
</details>

<details>
<summary><strong>Kimi CLI</strong></summary>

Add to `~/.kimi/mcp.json`:

```json
{
  "mcpServers": {
    "kimi-commander": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/Claude_Commander", "fastmcp", "run", "/path/to/Claude_Commander/src/claude_commander/server.py:mcp"],
      "env": { "OLLAMA_BASE_URL": "http://your-ollama-host:11434" }
    }
  }
}
```
</details>

---

## Architecture

```
CLAUDE.md              ← Agent instructions for Claude Code
AGENTS.md              ← Agent instructions for Codex (and other MCP clients)

src/claude_commander/
  server.py            ← FastMCP server: 30+ tools & all orchestration logic
  registry.py          ← 17-model catalog (13 cloud + 4 CLI) with categories & strengths
  ollama.py            ← Async HTTP client for the Ollama API (aiohttp)
  cli.py               ← Async subprocess runner for CLI agents (claude, codex, gemini, kimi)
  models.py            ← ~40 Pydantic result types for structured tool output
  profile_store.py     ← JSON-persisted reusable model profiles
  pipeline_store.py    ← JSON-persisted multi-step pipeline definitions
  defaults.py          ← 12 builtin profiles + 6 builtin pipelines
  resolver.py          ← Profile name → model ID + parameter resolution

tests/                 ← 209 tests, all mocked — no Ollama instance needed
  test_server.py       ← Core tool tests
  test_advanced.py     ← Orchestration tools + helper tests
  test_verification.py ← Verification & anti-slop tool tests
  test_cli.py          ← CLI subprocess caller tests
  test_ollama.py       ← HTTP client tests
  test_registry.py     ← Model catalog tests
  test_resolver.py     ← Profile resolver tests
  test_defaults.py     ← Builtin seed tests
  test_pipeline_store.py ← Pipeline persistence tests
```

Key design decisions:
- **Semaphore-bounded concurrency** — max 13 parallel Ollama calls to prevent overwhelming the backend
- **Thinking model auto-retry** — if a reasoning model exhausts its token budget on chain-of-thought and returns empty content, the server automatically retries with a bumped `num_predict`
- **Structured output everywhere** — every tool returns typed Pydantic models, not raw strings
- **Profile-first API** — any parameter that takes a model ID also accepts a profile name, so saved configs compose seamlessly with all tools

---

## Configuration

<details>
<summary><strong>Auto-routing config</strong></summary>

`auto_call` supports config-driven routing via `CLAUDE_COMMANDER_ROUTING_CONFIG` (defaults to `~/.claude-commander/auto_routing.json`):

```json
{
  "default_profile": "default",
  "profiles": {
    "default": {
      "code": {
        "balanced": ["qwen3-coder-next:cloud", "deepseek-v3.2:cloud"]
      }
    },
    "fast-local": {
      "general": {
        "fast": ["glm-4.7:cloud", "gpt-oss:20b-cloud"]
      }
    }
  }
}
```

Select a routing profile at runtime: `auto_call(prompt="...", routing_profile="fast-local")`

</details>

<details>
<summary><strong>Environment variables</strong></summary>

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `CLAUDE_COMMANDER_ROUTING_CONFIG` | `~/.claude-commander/auto_routing.json` | Auto-routing profile config path |

</details>

---

## Known Limitations

- **Thinking model token floor.** Reasoning models (`deepseek-v3.2`, `kimi-k2-thinking`, `glm-5`) need at least ~200 tokens or they consume their entire budget on chain-of-thought. The server auto-retries, but avoid setting `max_tokens` below 200–300.

- **Consensus scope.** `consensus` defaults to all 17 registered models. Pass an explicit `models` list to control scope and cost.

- **CLI agent prerequisites.** `exec_task` agents (`claude:cli`, `codex:cli`, etc.) require their binaries on `$PATH`. Missing binaries fail gracefully per-model without blocking other models in a swarm.

- **No token-usage metadata.** Responses include `elapsed_seconds` but not token counts. For cost-conscious usage, prefer compact tools (`vote`, `quality_gate`) or set `max_tokens` to 300–500.

---

## Testing

```bash
uv sync --extra dev
uv run pytest tests/ -v
```

209 tests, all mocked at the [`call_ollama`](src/claude_commander/ollama.py) boundary — no running Ollama instance needed.

---

## Dependencies

| Package | Version | Role |
|---|---|---|
| [FastMCP](https://github.com/jlowin/fastmcp) | >= 2.14 | MCP server framework |
| [Pydantic](https://docs.pydantic.dev/) | >= 2.8 | Structured result models |
| [aiohttp](https://docs.aiohttp.org/) | >= 3.9 | Async HTTP client for Ollama |
