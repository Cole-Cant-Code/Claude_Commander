# Commander — Claude Code Instructions

You have access to **Claude Commander**, an MCP server that orchestrates
13 Ollama cloud models plus 4 CLI agents. Use it as a multi-model GM (game master).

> **Coexistence rule.** This MCP server is also used by Codex and other clients.
> You are the primary agent in this session — Commander is your *tool*, not your
> replacement. **Never use `exec_task` with `codex:cli` to hand off your task.**
> Only use `exec_task` for isolated subtasks where a specific CLI agent's strengths
> genuinely help (e.g., a narrow code generation request). You own the task; Commander
> provides second opinions.

> **Cognitive independence rule.** The value of multi-model orchestration is that
> each model thinks differently. **Do not poison the pool with your own conclusions.**
>
> - **Ask neutral questions.** Write `consensus(prompt="What are the trade-offs
>   of microservices vs monolith?")` — not `"I think microservices are better,
>   do you agree?"`. Leading prompts anchor every model on your framing and
>   destroy the diversity you're paying for.
> - **Never inject your preferred answer** into the `prompt` or `system_prompt`
>   of Commander tools. If you already have an opinion, state it to the user
>   separately — don't bake it into the question you send to 13 models.
> - **Don't cherry-pick models** to manufacture agreement. If you only include
>   models you expect to agree with you, you're running a confirmation machine,
>   not a consensus tool. Omit the `models` parameter or use a representative set.
> - **Report results faithfully.** When presenting Commander output to the user,
>   include dissenting views and minority positions — don't summarize away
>   disagreement to make the result look cleaner.

## When to Use Commander

Use Commander when you need:
- External validation or a second opinion from differently-trained models
- Multiple independent perspectives on ambiguous questions
- Code review from models with different strengths (Qwen Coder, DeepSeek, GLM)
- A devil's advocate / contrarian check on your proposed approach
- Consensus on architectural decisions
- Fact verification or quality gating before shipping output
- Detection of AI slop (filler, hallucinated citations, vague hedging)

**Don't** use Commander for straightforward tasks. It adds latency and token cost.
Use it when diversity of thought matters or when getting it wrong would be expensive.

## Tools

### Core

| Tool | When to use it |
|------|---------------|
| `call_model` | Single model query — use for targeted second opinions |
| `auto_call` | Auto-routed query with fallback retries — use when you don't care which model answers |
| `swarm` | Fan-out to 6+ models — use when you want broad coverage |
| `list_models` | Registry query — use to check what's available |
| `health_check` | Connectivity test — use to verify Ollama is reachable |

### Orchestration

| Tool | When to use it |
|------|---------------|
| `debate` | Two models argue — use to stress-test a position |
| `vote` | Majority rules — use for binary or multi-choice decisions |
| `consensus` | Swarm + judge synthesis — use for complex open-ended questions |
| `code_review` | 3 reviewers merged by severity — use before finalizing non-trivial code |
| `multi_solve` | Independent solutions — use to compare algorithmic approaches |
| `rank` | Peer-scored leaderboard — use to find which model handles a task best |
| `chain` | Sequential pipeline — use for iterative refinement across models |
| `map_reduce` | Fan-out + custom reducer — use for synthesis with specific instructions |
| `blind_taste_test` | Anonymous comparison — use when you want unbiased evaluation |
| `contrarian` | Thesis + antithesis — use to find blind spots in an argument |
| `benchmark` | Prompt x model matrix — use for latency/quality comparisons |

### Verification & Quality

| Tool | When to use it |
|------|---------------|
| `verify` | Cross-model fact check — use to validate claims before presenting them |
| `red_team` | Iterative adversarial attack — use to stress-test code, APIs, or arguments |
| `quality_gate` | Pass/fail against criteria — use as a checkpoint before shipping |
| `detect_slop` | AI garbage detection — use to catch filler, hallucinations, and hedging |

### Profiles & Pipelines

| Tool | When to use it |
|------|---------------|
| `create_profile` / `clone_profile` | Save a reusable model + params combo |
| `list_profiles` / `get_profile` | Browse or inspect saved profiles |
| `create_pipeline` / `run_pipeline` | Save and run multi-step profile chains |
| `list_pipelines` | Browse saved pipelines |

### Task Execution

| Tool | When to use it |
|------|---------------|
| `exec_task` | Delegate an **isolated subtask** to a CLI agent (see coexistence rule above) |

## Model Picks

- **Code tasks**: `qwen3-coder-next:cloud` (fastest, code-specialized)
- **Hard reasoning**: `deepseek-v3.2:cloud` or `kimi-k2-thinking:cloud` (chain-of-thought)
- **Fast general**: `gpt-oss:20b-cloud`, `glm-4.7:cloud` (~1.3s response)
- **Broad coverage**: omit `models` param to hit all 13

## Tips

- `call_model` with `role_label` and `tags` lets you track who said what in multi-step flows
- Thinking models return a `thinking` field — useful for understanding their reasoning
- Intermediate results are truncated to 200 chars; use `call_model` for full output
- Profiles can be used anywhere a model ID is accepted — `call_model(model="deep-reasoner")`
- See [README.md](README.md) for setup examples per client

## Watch Out

- **Token floor for thinking models.** Don't set `max_tokens` below 200. Thinking models
  (`deepseek-v3.2`, `kimi-k2-thinking`, `glm-5`) burn tokens on internal reasoning before
  producing output. At ~50 tokens they return empty strings. Safe minimum: 200–300.
- **`consensus` calls everything.** It defaults to all 13 cloud models + 4 CLI agents (17
  total). Always pass an explicit `models` list if context budget matters.
- **CLI agents can fail silently.** `kimi:cli` requires `kimi` on PATH, `codex:cli` requires
  `codex`, etc. Missing binaries fail with `No such file or directory`; timeouts exit with
  code -9. These don't block other models in a swarm, but check `list_models` first.
- **No token counting in responses.** You get `elapsed_seconds` but not token usage. To
  control context intake: prefer `vote` (tally only), `quality_gate` (score only), or pass
  `max_tokens: 300–500` instead of the 4096 default.
