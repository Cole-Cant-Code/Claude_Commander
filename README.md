# Claude Commander

An MCP server that lets you call multiple Ollama models and compose them into
useful patterns — debates, voting, code review, benchmarks, and more.

## What it does

Claude Commander sits between your MCP client (Claude Code, Codex, etc.) and an
Ollama instance running cloud-proxied models. It exposes 16 tools:

**Primitives**

| Tool | What it does |
|---|---|
| `call_model` | Call a single model |
| `auto_call` | Auto-route to a best-fit model, with fallback retries |
| `swarm` | Call multiple models in parallel |
| `list_models` | Show registered models + availability |
| `health_check` | Check Ollama connectivity |

**Orchestration**

| Tool | Pattern | What it does |
|---|---|---|
| `debate` | Sequential multi-round | Two models argue back and forth, each seeing the full transcript |
| `vote` | Parallel + extraction | Models vote from fixed options; multi-strategy parser extracts results |
| `consensus` | Parallel + judge | Swarm a question, then a judge synthesizes agreement/disagreement |
| `code_review` | Parallel + merge | Multiple reviewers independently review code, findings merged by severity |
| `multi_solve` | Parallel | Multiple models independently solve the same coding problem |
| `benchmark` | Parallel matrix | Run N prompts x M models, get latency stats |
| `rank` | Parallel + peer eval | Models answer, then peer-judge each other 1-10 for a leaderboard |
| `chain` | Sequential pipeline | Output of model N feeds into model N+1 |
| `map_reduce` | Parallel + reduce | Fan-out to many models, fan-in through a reducer with custom instructions |
| `blind_taste_test` | Parallel + anonymize | Anonymous "Response A/B/C" comparison with a reveal mapping |
| `contrarian` | Two-phase | One model answers, another finds logical gaps and argues alternatives |

## Models

13 Ollama cloud models across four categories:

| Category | Models |
|---|---|
| **General** | GLM-5, GLM-4.7, MiniMax M2.5, MiniMax M2.1, GPT-OSS 20B, GPT-OSS 120B, Qwen3 Next 80B, Kimi K2.5 |
| **Code** | Qwen3 Coder Next |
| **Reasoning** | DeepSeek v3.2, Kimi K2 Thinking |
| **Vision** | Qwen3 VL 235B, Qwen3 VL 235B Instruct |

All tools accept optional `models` parameters to override defaults.

## Setup

Requires Python 3.11+ and a running Ollama instance.

```bash
# clone and install
git clone https://github.com/Cole-Cant-Code/Claude_Commander.git
cd Claude_Commander
uv sync --extra dev

# set Ollama endpoint (defaults to localhost:11434)
export OLLAMA_BASE_URL="http://your-ollama-host:11434"
```

Agent instructions (for any client): [`AGENTS.md`](AGENTS.md)

### Claude Code

Server name: `claude-commander`

```bash
claude mcp add claude-commander -- uv run --project /path/to/Claude_Commander fastmcp run /path/to/Claude_Commander/src/claude_commander/server.py:mcp
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

### Codex

Server name: `codex-commander`

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.codex-commander]
command = "uv"
args = ["run", "--project", "/path/to/Claude_Commander", "fastmcp", "run", "/path/to/Claude_Commander/src/claude_commander/server.py:mcp"]

[mcp_servers.codex-commander.env]
MCP_SERVER_NAME = "Codex Commander"
OLLAMA_BASE_URL = "http://your-ollama-host:11434"
```

### Gemini CLI

Server name: `gemini-commander`

```bash
gemini mcp add -e OLLAMA_BASE_URL=http://your-ollama-host:11434 gemini-commander -- uv run --project /path/to/Claude_Commander fastmcp run /path/to/Claude_Commander/src/claude_commander/server.py:mcp
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

### Kimi CLI

Server name: `kimi-commander`

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

## Usage examples

These are MCP tool calls. Your client (Claude Code, Codex, Gemini, Kimi, etc.) invokes them directly.

**Debate** — two models argue about a topic:
```
debate(prompt="Is Rust better than Go?", rounds=3)
```

**Auto call** — route + fallback in one call:
```
auto_call(prompt="Review this Python function for bugs", task="code", strategy="quality")
```

**Vote** — ask all models a yes/no question:
```
vote(prompt="Is the sky blue?")
vote(prompt="Best approach?", options=["monolith", "microservices", "serverless"])
```

**Code review** — three reviewers, merged by severity:
```
code_review(code="def connect(host): ...", language="python")
```

**Chain** — sequential refinement pipeline:
```
chain(
  prompt="Explain quantum computing",
  pipeline=["glm-5:cloud", "deepseek-v3.2:cloud", "kimi-k2-thinking:cloud"]
)
```

**Rank** — peer-evaluated leaderboard:
```
rank(prompt="Write a haiku about recursion")
```

**Blind taste test** — anonymous comparison:
```
blind_taste_test(prompt="Explain monads", count=4)
```

## Tool details

### auto_call

```
auto_call(
  prompt,
  task="auto",
  strategy="balanced",
  routing_profile?,
  models?,
  max_attempts=3,
  max_time_ms?,
  ...
)
```

Automatically routes the prompt to a best-fit model (or profile) and retries
with ordered fallbacks if the first attempt errors or returns empty content.
Tasks: `auto`, `general`, `code`, `reasoning`, `creative`, `verification`, `vision`.
Strategies: `fast`, `balanced`, `quality`.

`max_time_ms` applies a total wall-clock budget for all attempts. If budget runs
out, `auto_call` stops and returns an error with `budget_exhausted=true`.

Routing can be config-driven via `CLAUDE_COMMANDER_ROUTING_CONFIG`
(defaults to `~/.claude-commander/auto_routing.json`). Optional format:

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

Select a profile at runtime with `routing_profile="fast-local"`.

### debate

```
debate(prompt, model_a?, model_b?, rounds=3)
```

Multi-round back-and-forth. Each round's model sees the full transcript so far.
Defaults: `deepseek-v3.2:cloud` vs `glm-5:cloud`.

### vote

```
vote(prompt, options=["yes","no"], models?)
```

All models answer with a system prompt forcing them to lead with their choice.
Votes are extracted via a cascade: first-word match, phrase patterns
(`"I vote X"`, `"my answer is X"`), occurrence counting, then abstain fallback.
Returns tally, majority, and agreement percentage.

### consensus

```
consensus(prompt, models?, judge_model?)
```

Phase 1: swarm to all models. Phase 2: `kimi-k2-thinking:cloud` (default judge)
identifies agreement, disagreement, and produces a unified answer.

### code_review

```
code_review(code, language?, review_models?, merge_model?)
```

Default reviewers: `qwen3-coder-next:cloud`, `deepseek-v3.2:cloud`, `gpt-oss:120b-cloud`.
System prompt targets bugs, security, performance, readability with line-number references.
Merge phase deduplicates and sorts findings critical-to-minor.

### multi_solve

```
multi_solve(problem, language?, models?)
```

Sends the problem to code + reasoning models by default. Each produces a complete
solution with comments. Useful for comparing algorithmic approaches.

### benchmark

```
benchmark(prompts, models?)
```

Runs every prompt against every model in parallel (semaphore-bounded). Returns
a structured matrix with per-cell latency and per-model average latency stats.

### rank

```
rank(prompt, models?, judge_count=3)
```

All models answer, then randomly-selected peers score each answer 1-10.
Score extraction handles `"8/10"`, `"Score: 8"`, `"Rating: 8"`, `"8 out of 10"`.
Returns a sorted leaderboard with per-judge breakdowns.

### chain

```
chain(prompt, pipeline, pass_context=True)
```

Sequential pipeline. With `pass_context=True` (default), each step sees all prior
outputs. With `False`, each step only sees the immediately previous output.

### map_reduce

```
map_reduce(prompt, mapper_models?, reducer_model?, reduce_prompt?)
```

Like `consensus` but combines information rather than finding agreement.
Custom `reduce_prompt` lets you control the synthesis instruction.

### blind_taste_test

```
blind_taste_test(prompt, count=3)
```

Randomly selects `count` models (seeded by prompt hash for reproducibility),
shuffles responses into "Response A", "Response B", etc. The `reveal` dict
maps labels to model IDs.

### contrarian

```
contrarian(prompt, thesis_model?, antithesis_model?)
```

Phase 1: `qwen3-next:80b-cloud` answers normally. Phase 2: `deepseek-v3.2:cloud`
gets a system prompt to find logical gaps, challenge assumptions, and argue
alternatives — substantively, not for its own sake.

## Project structure

```
AGENTS.md                        # Agent instructions (any MCP client)

src/claude_commander/
  __init__.py                    # version
  registry.py                    # 13-model catalog with strengths + categories
  ollama.py                      # async HTTP client (aiohttp)
  models.py                      # Pydantic result types
  profile_store.py               # reusable profile persistence
  server.py                      # FastMCP tools + orchestration logic

tests/
  test_registry.py               # model catalog tests
  test_ollama.py                 # HTTP client tests (mocked)
  test_server.py                 # original 4 tools (mocked)
  test_advanced.py               # 11 orchestration tools + helpers (mocked)
```

## Known Limitations

**Thinking model token exhaustion.** Setting `max_tokens` below ~200 causes
thinking/reasoning models (`deepseek-v3.2`, `kimi-k2-thinking`, `glm-5`) to consume
their entire budget on internal chain-of-thought and return empty content. The server
detects this and retries with a bumped `num_predict` (~306), but the safe practice is
to never go below 200–300 tokens for calls involving these models.

**Consensus scope.** `consensus` defaults to all registered models — 13 cloud + 4 CLI
agents (17 total). Pass an explicit `models` list to control scope and context cost.

**CLI agent prerequisites.** CLI agents (`claude:cli`, `codex:cli`, `gemini:cli`,
`kimi:cli`) require their binaries on `$PATH`. Missing binaries fail with
`No such file or directory`; timeouts/OOM exit with code `-9`. These failures are
per-model and don't block other models in a swarm.

**No token-usage metadata.** Responses include `elapsed_seconds` per model but not
token counts. Context-conscious callers should prefer compact tools (`vote`,
`quality_gate`) or set `max_tokens` to 300–500 instead of the 4096 default.

## Tests

```bash
uv run pytest tests/ -v
```

209 tests, all mocked at the `call_ollama` boundary — no Ollama instance needed.

## Dependencies

- [FastMCP](https://github.com/jlowin/fastmcp) >= 2.14 — MCP server framework
- [Pydantic](https://docs.pydantic.dev/) >= 2.8 — result models
- [aiohttp](https://docs.aiohttp.org/) >= 3.9 — async HTTP for Ollama API
