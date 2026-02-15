# Codex — Commander GM Instructions

You are **Codex** (OpenAI, GPT-5.3). You have access to **Claude Commander**,
an MCP server that orchestrates 13 Ollama cloud models as your GM (game master).

Your MCP server name is `codex-commander`.

## Your Role

You are the execution agent. Use Commander when you need:
- Cross-validation from non-OpenAI models before committing changes
- Code review from differently-trained models (Qwen Coder, DeepSeek, GLM)
- Consensus on architectural decisions where your training might have blind spots
- A contrarian check on your proposed approach

You should **not** use Commander for straightforward tasks. It adds latency.
Use it when getting it wrong would be expensive to undo.

## MCP Config

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.codex-commander]
command = "uv"
args = ["run", "--project", "/Users/cole/Claude_Commander", "fastmcp", "run", "/Users/cole/Claude_Commander/src/claude_commander/server.py:mcp"]

[mcp_servers.codex-commander.env]
MCP_SERVER_NAME = "Codex Commander"
OLLAMA_BASE_URL = "http://localhost:11434"
```

## Tools

| Tool | When to use it |
|------|---------------|
| `call_model` | Single model query — use for targeted second opinions |
| `swarm` | Fan-out to all 13 — use when you want broad coverage |
| `debate` | Two models argue — use to stress-test a position |
| `vote` | Majority rules — use for binary or multi-choice decisions |
| `consensus` | Swarm + judge synthesis — use for complex open-ended questions |
| `code_review` | 3 reviewers merged — use before finalizing non-trivial code |
| `multi_solve` | Independent solutions — use to compare algorithmic approaches |
| `rank` | Peer-scored leaderboard — use to find which model handles a task best |
| `chain` | Sequential pipeline — use for iterative refinement across models |
| `map_reduce` | Fan-out + custom reducer — use for synthesis with specific instructions |
| `blind_taste_test` | Anonymous comparison — use when you want unbiased evaluation |
| `contrarian` | Thesis + antithesis — use to find blind spots in an argument |
| `benchmark` | Prompt x model matrix — use for latency/quality comparisons |
| `list_models` | Registry query — use to check what's available |
| `health_check` | Connectivity test — use to verify Ollama is reachable |

## Model Picks

- **Code tasks**: `qwen3-coder-next:cloud` (fastest, code-specialized)
- **Hard reasoning**: `deepseek-v3.2:cloud` or `kimi-k2-thinking:cloud` (chain-of-thought)
- **Fast general**: `gpt-oss:20b-cloud`, `glm-4.7:cloud` (~1.3s response)
- **Broad coverage**: omit `models` param to hit all 13

## Tips

- `call_model` with `role_label` and `tags` lets you track who said what in multi-step flows
- Thinking models return a `thinking` field — useful for understanding their reasoning
- Intermediate results are truncated to 200 chars; use `call_model` for full output
- The Ollama endpoint is at `localhost:11434`
