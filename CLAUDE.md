# Claude Code — Commander GM Instructions

You have access to **Claude Commander**, an MCP server that orchestrates 13 Ollama
cloud models. Use it as your GM (game master) to query, compare, and compose
model responses.

## MCP Setup

Already configured as a project-level server. If missing, add to `~/.claude.json`
under `projects` or globally:

```json
{
  "claude-commander": {
    "command": "uv",
    "args": ["run", "--project", "/Users/cole/Claude_Commander", "fastmcp", "run", "claude_commander.server:mcp"],
    "env": { "OLLAMA_BASE_URL": "http://100.64.0.7:11434" }
  }
}
```

## Available Tools

**Primitives** — `call_model`, `swarm`, `list_models`, `health_check`

**Orchestration** — `debate`, `vote`, `consensus`, `code_review`, `multi_solve`,
`benchmark`, `rank`, `chain`, `map_reduce`, `blind_taste_test`, `contrarian`

## When to Use Commander

- **Need a second opinion**: `call_model` or `swarm` a question to get external perspectives
- **Evaluating options**: `vote` with custom options, or `rank` to get a scored leaderboard
- **Code quality**: `code_review` fans out to 3 code-specialized models and merges findings
- **Deep analysis**: `chain` a reasoning pipeline (e.g., generalist -> specialist -> critic)
- **Fact-checking yourself**: `contrarian` generates a devil's-advocate counterargument
- **Comparing approaches**: `blind_taste_test` for unbiased A/B/C comparison

## Model Categories

| Category | Best for |
|----------|----------|
| **Code** | `qwen3-coder-next:cloud` — generation, architecture, debugging |
| **Reasoning** | `deepseek-v3.2:cloud`, `kimi-k2-thinking:cloud` — math, logic, chain-of-thought |
| **Vision** | `qwen3-vl:235b-*:cloud` — image understanding |
| **General** | GLM-5, MiniMax M2.5, GPT-OSS, Qwen3 Next, Kimi K2.5 |

## Tips

- `swarm` with no `models` parameter hits all 13 in parallel
- `call_model` accepts `role_label` and `tags` for tracking in multi-step workflows
- Thinking models (`deepseek-v3.2`, `kimi-k2-thinking`) return a `thinking` field with chain-of-thought
- All responses are truncated to 200 chars in return values; use `call_model` directly for full output
- The Ollama endpoint is on Tailscale at `100.64.0.7:11434`
