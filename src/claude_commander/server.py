"""Claude Commander MCP server — model orchestration tools for Ollama."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from claude_commander import __version__
from claude_commander.models import (
    AutoCallResult,
    BenchmarkCell,
    BenchmarkResult,
    BlindResponse,
    BlindTasteResult,
    CallResult,
    ChainResult,
    ChainStep,
    ClaimVerification,
    CodeReviewResult,
    ConsensusResult,
    ContrarianResult,
    CriterionScore,
    DebateResult,
    DebateRound,
    DetectSlopResult,
    ExecTaskResult,
    HealthStatus,
    MapReduceResult,
    ModelAvailability,
    ModelScore,
    MultiSolveResult,
    PipelineData,
    PipelineResult,
    PipelineStepResult,
    ProfileData,
    ProfileListResult,
    QualityGateResult,
    RankResult,
    RedTeamExchange,
    RedTeamResult,
    SlopSignal,
    SurvivingIssue,
    SwarmResult,
    VerifyResult,
    Vote,
    VoteResult,
)
from claude_commander.cli import call_cli
from claude_commander.ollama import OLLAMA_BASE_URL, call_ollama, check_ollama
from claude_commander.pipeline_store import get_pipeline_store
from claude_commander.profile_store import get_profile_store
from claude_commander.registry import MODELS, get_model
from claude_commander.resolver import _UNSET, merge_overrides, resolve

# Keep the historical name by default, but allow config-level aliases (e.g. "GM Commander")
# without changing existing Claude configs.
_INSTRUCTIONS = """\
Claude Commander orchestrates 13 cloud models via Ollama. Every tool that takes \
a `model` parameter also accepts a profile name (run `list_profiles` to see them).

WHEN TO USE WHICH TOOL:
- Need one second opinion? → call_model
- Need automatic routing + fallback? → auto_call
- Need broad coverage? → swarm (6 by default, count=13 for all), consensus (swarm + judge synthesis)
- Need to stress-test content? → red_team (iterative attacker/defender)
- Need a quality checkpoint? → quality_gate (pass/fail against criteria)
- Need to verify facts? → verify (cross-model claim checking)
- Need to detect AI slop? → detect_slop (filler/hallucination detection)
- Need diverse solutions? → multi_solve, blind_taste_test
- Need iterative refinement? → chain, run_pipeline
- Need adversarial perspectives? → debate, contrarian
- Need code written/executed locally? → exec_task (delegates to codex, claude, kimi, or gemini CLI)

Read the full guide: use resources/read on "commander://guide"\
"""

mcp = FastMCP(
    os.getenv("MCP_SERVER_NAME", "Claude Commander"),
    instructions=_INSTRUCTIONS,
)

MAX_CONCURRENCY = 13


# ---------------------------------------------------------------------------
# Embedded guide — readable by any LLM client via resources/read
# ---------------------------------------------------------------------------

_GUIDE = """\
# Claude Commander — Usage Guide

You are connected to Claude Commander, an MCP server that orchestrates 13 cloud \
models through Ollama. This guide tells you everything you need to use it well.

## Core Concept

Every tool that accepts a `model` parameter also accepts a **profile name**. \
Profiles bundle a model ID with a system prompt, temperature, and other settings. \
Use `list_profiles` to see all available profiles. Use `list_models` to see the \
raw model registry.

---

## Tool Reference

### Generation & Orchestration

| Tool | What it does | When to reach for it |
|------|-------------|---------------------|
| `call_model` | Single model/profile call | Targeted second opinion on a specific question |
| `auto_call` | Auto-route to best-fit model with fallback retries | Quick "pick-for-me" calls with resilience |
| `swarm` | Fan-out to models in parallel (6 default, count=13 for all) | Broad coverage, seeing how many models agree |
| `consensus` | Swarm + judge synthesizes a unified answer | Complex open-ended questions needing a single answer |
| `chain` | Sequential pipeline — each model builds on prior output | Iterative refinement across different model strengths |
| `run_pipeline` | Execute a saved pipeline by name | Reusable multi-step workflows |
| `map_reduce` | Fan-out + custom reducer prompt | Synthesis with specific aggregation instructions |

### Comparison & Evaluation

| Tool | What it does | When to reach for it |
|------|-------------|---------------------|
| `debate` | Two models argue for multiple rounds | Stress-testing a position or exploring both sides |
| `contrarian` | Thesis then devil's-advocate antithesis | Finding blind spots in an argument |
| `vote` | Multiple models vote on options | Binary or multi-choice decisions |
| `rank` | Models answer, then peer-judge each other 1-10 | Finding which model handles a task best |
| `blind_taste_test` | Anonymous A/B/C comparison | Unbiased evaluation without model-name bias |
| `benchmark` | Prompt x model latency/quality matrix | Performance comparisons |

### Verification & Quality (Anti-Slop)

| Tool | What it does | When to reach for it |
|------|-------------|---------------------|
| `verify` | Cross-model fact verification with per-claim verdicts | Checking factual accuracy of existing content |
| `red_team` | Iterative adversarial stress testing (attacker vs defender) | Finding weaknesses in arguments, code, or plans |
| `quality_gate` | Pass/fail scoring against criteria with threshold | Quality checkpoint before shipping AI-generated content |
| `detect_slop` | Multi-model AI garbage detection | Catching filler phrases, vague generalities, hallucinated citations |

### Code-Specific

| Tool | What it does | When to reach for it |
|------|-------------|---------------------|
| `code_review` | 3 parallel expert reviews merged by severity | Before finalizing non-trivial code |
| `multi_solve` | Multiple independent solutions to the same problem | Comparing algorithmic approaches |

### Management

| Tool | What it does |
|------|-------------|
| `list_models` | Show all 13 registered models with availability |
| `health_check` | Verify Ollama connectivity |
| `create_profile` / `clone_profile` / `get_profile` / `list_profiles` / `delete_profile` | Manage reusable model profiles |
| `create_pipeline` / `list_pipelines` / `run_pipeline` / `delete_pipeline` | Manage reusable multi-step pipelines |

---

## Model Roster (17 models)

### Reasoning (chain-of-thought, thinking field populated)
- `deepseek-v3.2:cloud` — Math, formal logic, competitive programming. Best attacker/verifier.
- `kimi-k2-thinking:cloud` — Extended chain-of-thought, self-correction. Default judge.

### Code
- `qwen3-coder-next:cloud` — Code generation, architecture, debugging. Fastest for code tasks.

### Vision
- `qwen3-vl:235b-cloud` — Visual reasoning, image analysis.
- `qwen3-vl:235b-instruct-cloud` — Multimodal instruction following.

### General
- `glm-5:cloud` — Academic reasoning, structured analysis.
- `glm-4.7:cloud` — Fast general purpose, structured output.
- `gpt-oss:120b-cloud` — Broad coverage, general reasoning.
- `gpt-oss:20b-cloud` — Fastest inference, good for drafts.
- `qwen3-next:80b-cloud` — Balanced generalist, multilingual.
- `kimi-k2.5:cloud` — Fast, good at agentic execution.
- `minimax-m2.5:cloud` — Agentic execution, function calling.
- `minimax-m2.1:cloud` — Lightweight general purpose.

### CLI (local AI coding agents — slower but full tool-use capable)
- `claude:cli` — Claude Code CLI. Reasoning, code generation, agentic tool use.
- `gemini:cli` — Gemini CLI. Reasoning, multimodal, search-grounded.
- `codex:cli` — Codex CLI. Code generation, sandboxed execution.
- `kimi:cli` — Kimi CLI. Code generation, agentic tool use.

---

## Builtin Profiles (12)

| Profile | Model | Temp | Use for |
|---------|-------|------|---------|
| `fast-general` | glm-4.7 | 0.7 | Quick general tasks |
| `deep-reasoner` | deepseek-v3.2 | 0.3 | Hard reasoning problems |
| `code-specialist` | qwen3-coder-next | 0.2 | Code generation and debugging |
| `thinking-judge` | kimi-k2-thinking | 0.4 | Evaluation and judging |
| `creative-writer` | glm-5 | 0.9 | Creative writing, brainstorming |
| `factual-analyst` | gpt-oss:120b | 0.3 | Precise factual analysis |
| `vision-analyzer` | qwen3-vl:235b | 0.5 | Image and visual reasoning |
| `quick-draft` | gpt-oss:20b | 0.8 | Fast low-latency drafting |
| `strict-verifier` | deepseek-v3.2 | 0.2 | Rigorous fact-checking |
| `adversarial-attacker` | deepseek-v3.2 | 0.3 | Finding flaws and edge cases |
| `quality-judge` | kimi-k2-thinking | 0.2 | Structured quality scoring |
| `slop-detector` | gpt-oss:120b | 0.2 | Detecting AI filler content |

---

## Builtin Pipelines (6)

| Pipeline | Steps | Use for |
|----------|-------|---------|
| `draft-then-refine` | quick-draft → deep-reasoner | Fast draft + deep refinement |
| `code-review-pipeline` | code-specialist → deep-reasoner → factual-analyst | Write + review + validate |
| `creative-to-critical` | creative-writer → thinking-judge | Brainstorm + evaluate |
| `verify-then-refine` | quick-draft → strict-verifier → deep-reasoner | Generate + verify + fix |
| `red-team-then-harden` | deep-reasoner → adversarial-attacker → thinking-judge | Generate + stress-test + harden |
| `full-quality-check` | strict-verifier → slop-detector → quality-judge | Triple-check existing content |

---

## Usage Patterns

### Pattern 1: Generate and verify
```
1. call_model(model="quick-draft", prompt="Write about X")
2. verify(content=<step 1 output>)
3. If verify returns fail/mixed → call_model(model="deep-reasoner", prompt="Fix these issues: ...")
```
Or just: `run_pipeline(name="verify-then-refine", prompt="Write about X")`

### Pattern 2: Quality gate before shipping
```
1. <generate content with any tool>
2. quality_gate(content=<output>, threshold=7.0)
3. If failed → fix blocking issues and re-check
```

### Pattern 3: Adversarial hardening
```
1. call_model(model="deep-reasoner", prompt="Propose architecture for X")
2. red_team(content=<step 1 output>, rounds=3)
3. Address surviving_issues from the result
```

### Pattern 4: Slop detection on existing content
```
1. detect_slop(content=<any text>)
2. If verdict is "significant" or "severe" → rewrite flagged sections
```

### Pattern 5: Multi-model consensus for high-stakes questions
```
consensus(prompt="Should we use microservices or monolith for this project?")
→ Gets answers from all 13 models, judge synthesizes into unified recommendation
```

---

## Tips

- Thinking models (`deepseek-v3.2`, `kimi-k2-thinking`) return a `thinking` field with their \
chain-of-thought reasoning — useful for understanding *why* they reached a conclusion.
- Results from swarm/consensus/code_review truncate individual responses to 200 chars. \
Use `call_model` directly if you need the full untruncated output from a specific model.
- `red_team` stops early if the attacker says "NO NEW ISSUES FOUND" — well-written content \
finishes in 1 round, weak content gets the full adversarial treatment.
- `quality_gate` fails if *any* criterion scores below `threshold - 2`, even if the overall \
average is above threshold. This prevents one catastrophic weakness from being hidden by \
high scores elsewhere.
- `detect_slop` merges signals across detectors and escalates severity — if one detector says \
"minor" but another says "major" for the same signal type, the result uses "major".
- All tools accept profile names wherever model IDs are expected. Create custom profiles \
with `create_profile` to save your preferred configurations.
- `auto_call` supports `max_time_ms` and config-driven routing profiles from \
`CLAUDE_COMMANDER_ROUTING_CONFIG` (default: `~/.claude-commander/auto_routing.json`).

---

## Known Limitations & Gotchas

### Token budget and thinking models

Setting `max_tokens` below ~200 will break thinking/reasoning models (`deepseek-v3.2`, \
`kimi-k2-thinking`, `glm-5`). These models spend tokens on internal chain-of-thought \
*before* producing visible output. At low budgets (e.g. 50), all tokens are consumed by \
reasoning and the response comes back empty. The server detects this ("reasoning-token \
exhaustion") and retries once with `num_predict` bumped to ~306, but this is a fallback, \
not a fix. **Safe minimum: 200–300 tokens for any call involving thinking models.**

### Consensus and swarm scope

`consensus` defaults to **all registered models** — 13 cloud models plus CLI agents. This \
means a single `consensus` call can produce 17 individual responses plus a judge synthesis. \
If context budget matters, always pass an explicit `models` list to scope it down. The same \
applies to `swarm` when called with `count=13`.

### CLI agent availability

CLI agents (`claude:cli`, `codex:cli`, `gemini:cli`, `kimi:cli`) require their respective \
binaries on `$PATH`. If a binary is missing, the call fails with `No such file or directory`. \
If the agent times out or runs out of memory, it exits with code `-9`. These failures are \
reported per-model in results but do not block other models in a swarm. **Check availability \
with `list_models` before relying on CLI agents in pipelines.**

### Context intake

There is currently no token-counting metadata in responses. You get `elapsed_seconds` per \
model but not token usage. When orchestrating multi-model calls, be aware that all raw \
outputs (including `thinking` fields from reasoning models) land in the caller's context. \
Strategies to control intake:
- Use `consensus` or `map_reduce` and read only the synthesis, not individual responses.
- Use `vote` for decisions — returns a tally, not prose.
- Use `quality_gate` for pass/fail checks — returns a score, not full analysis.
- Set `max_tokens` to 300–500 instead of the default 4096 when full responses aren't needed.
- Pass an explicit `models` list (3–5 models) instead of hitting all 13+.
"""


@mcp.resource("commander://guide", name="Usage Guide", description="Full reference guide for all Claude Commander tools, models, profiles, and pipelines")
def get_guide() -> str:
    """Return the full Claude Commander usage guide."""
    return _GUIDE

# ---------------------------------------------------------------------------
# Role constants
# ---------------------------------------------------------------------------

DEFAULT_JUDGE = "kimi-k2-thinking:cloud"
DEFAULT_DEBATE = ("deepseek-v3.2:cloud", "glm-5:cloud")
CODE_MODELS = ["qwen3-coder-next:cloud", "deepseek-v3.2:cloud", "gpt-oss:120b-cloud"]
RANK_DEFAULT_MODELS = [
    "deepseek-v3.2:cloud",
    "qwen3-coder-next:cloud",
    "glm-5:cloud",
    "kimi-k2.5:cloud",
    "minimax-m2.5:cloud",
]
# Default swarm: 6 models covering reasoning, code, general, and vision.
# Pass count=13 or models=list(MODELS.keys()) to hit all.
DEFAULT_SWARM_COUNT = 6
DEFAULT_SWARM_MODELS = [
    "deepseek-v3.2:cloud",       # reasoning
    "kimi-k2-thinking:cloud",    # reasoning / thinking
    "qwen3-coder-next:cloud",    # code
    "gpt-oss:120b-cloud",        # general (large)
    "glm-5:cloud",               # general (academic)
    "qwen3-vl:235b-cloud",       # vision
]

AUTO_TASKS = {"general", "code", "reasoning", "creative", "verification", "vision"}
AUTO_STRATEGIES = {"fast", "balanced", "quality"}
AUTO_ROUTING_CONFIG_ENV = "CLAUDE_COMMANDER_ROUTING_CONFIG"
AUTO_ROUTING_DEFAULT_PATH = Path.home() / ".claude-commander" / "auto_routing.json"
AUTO_ROUTING_DEFAULT_PROFILE = "default"

AUTO_ROUTING_DEFAULTS: dict[str, dict[str, list[str]]] = {
    "general": {
        "fast": ["gpt-oss:20b-cloud", "glm-4.7:cloud", "kimi-k2.5:cloud"],
        "balanced": ["glm-5:cloud", "qwen3-next:80b-cloud", "gpt-oss:120b-cloud"],
        "quality": ["gpt-oss:120b-cloud", "glm-5:cloud", "deepseek-v3.2:cloud"],
    },
    "code": {
        "fast": ["qwen3-coder-next:cloud", "glm-4.7:cloud", "gpt-oss:20b-cloud"],
        "balanced": ["qwen3-coder-next:cloud", "deepseek-v3.2:cloud", "gpt-oss:120b-cloud"],
        "quality": ["qwen3-coder-next:cloud", "deepseek-v3.2:cloud", "kimi-k2-thinking:cloud"],
    },
    "reasoning": {
        "fast": ["glm-5:cloud", "qwen3-next:80b-cloud", "kimi-k2.5:cloud"],
        "balanced": ["deepseek-v3.2:cloud", "kimi-k2-thinking:cloud", "glm-5:cloud"],
        "quality": ["deepseek-v3.2:cloud", "kimi-k2-thinking:cloud", "gpt-oss:120b-cloud"],
    },
    "creative": {
        "fast": ["gpt-oss:20b-cloud", "kimi-k2.5:cloud", "glm-4.7:cloud"],
        "balanced": ["glm-5:cloud", "qwen3-next:80b-cloud", "gpt-oss:120b-cloud"],
        "quality": ["glm-5:cloud", "qwen3-next:80b-cloud", "kimi-k2-thinking:cloud"],
    },
    "verification": {
        "fast": ["glm-4.7:cloud", "gpt-oss:20b-cloud", "kimi-k2.5:cloud"],
        "balanced": ["deepseek-v3.2:cloud", "gpt-oss:120b-cloud", "kimi-k2-thinking:cloud"],
        "quality": ["deepseek-v3.2:cloud", "kimi-k2-thinking:cloud", "gpt-oss:120b-cloud"],
    },
    "vision": {
        "fast": ["qwen3-vl:235b-instruct-cloud", "qwen3-vl:235b-cloud", "glm-4.7:cloud"],
        "balanced": ["qwen3-vl:235b-cloud", "qwen3-vl:235b-instruct-cloud", "glm-5:cloud"],
        "quality": ["qwen3-vl:235b-cloud", "qwen3-vl:235b-instruct-cloud", "deepseek-v3.2:cloud"],
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

TRUNCATE_LEN = 200


def _truncate(text: str, limit: int = TRUNCATE_LEN) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _truncate_result(r: CallResult, limit: int = TRUNCATE_LEN) -> CallResult:
    return CallResult(
        model=r.model,
        content=_truncate(r.content, limit),
        thinking=r.thinking[:limit] + "..." if r.thinking and len(r.thinking) > limit else r.thinking,
        elapsed_seconds=r.elapsed_seconds,
        status=r.status,
        error=r.error,
        role_label=r.role_label,
        tags=r.tags,
        warnings=r.warnings,
    )


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _copy_routing_table(table: dict[str, dict[str, list[str]]]) -> dict[str, dict[str, list[str]]]:
    return {
        task: {strategy: list(candidates) for strategy, candidates in strategies.items()}
        for task, strategies in table.items()
    }


def _normalize_candidates(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    candidates = [item.strip() for item in raw if isinstance(item, str) and item.strip()]
    return _dedupe_keep_order(candidates)


def _merge_routing_table(
    base: dict[str, dict[str, list[str]]],
    override: Any,
) -> dict[str, dict[str, list[str]]]:
    merged = _copy_routing_table(base)
    if not isinstance(override, dict):
        return merged

    for task, strategies in override.items():
        if task not in AUTO_TASKS or not isinstance(strategies, dict):
            continue
        for strategy, raw_candidates in strategies.items():
            if strategy not in AUTO_STRATEGIES:
                continue
            candidates = _normalize_candidates(raw_candidates)
            if candidates:
                merged[task][strategy] = candidates
    return merged


def _load_routing_profiles() -> tuple[dict[str, dict[str, dict[str, list[str]]]], str, str, list[str]]:
    warnings: list[str] = []
    config_path = Path(os.getenv(AUTO_ROUTING_CONFIG_ENV, str(AUTO_ROUTING_DEFAULT_PATH))).expanduser()
    builtin_profiles = {AUTO_ROUTING_DEFAULT_PROFILE: _copy_routing_table(AUTO_ROUTING_DEFAULTS)}

    if not config_path.exists():
        return builtin_profiles, AUTO_ROUTING_DEFAULT_PROFILE, "builtin", warnings

    try:
        raw = json.loads(config_path.read_text())
    except Exception as exc:
        warnings.append(f"Failed to read routing config at {config_path}: {exc}")
        return builtin_profiles, AUTO_ROUTING_DEFAULT_PROFILE, f"file:{config_path}", warnings

    if not isinstance(raw, dict):
        warnings.append(f"Routing config at {config_path} must be a JSON object.")
        return builtin_profiles, AUTO_ROUTING_DEFAULT_PROFILE, f"file:{config_path}", warnings

    if "profiles" in raw:
        raw_profiles = raw.get("profiles")
        default_profile = raw.get("default_profile", AUTO_ROUTING_DEFAULT_PROFILE)
    else:
        raw_profiles = {AUTO_ROUTING_DEFAULT_PROFILE: raw}
        default_profile = AUTO_ROUTING_DEFAULT_PROFILE

    if not isinstance(raw_profiles, dict) or not raw_profiles:
        warnings.append(f"Routing config at {config_path} has no valid 'profiles' mapping.")
        return builtin_profiles, AUTO_ROUTING_DEFAULT_PROFILE, f"file:{config_path}", warnings

    profiles: dict[str, dict[str, dict[str, list[str]]]] = {}
    for name, profile_raw in raw_profiles.items():
        if not isinstance(name, str) or not name.strip():
            continue
        profiles[name.strip()] = _merge_routing_table(AUTO_ROUTING_DEFAULTS, profile_raw)

    if not profiles:
        warnings.append(f"Routing config at {config_path} had no usable profiles; using builtin defaults.")
        return builtin_profiles, AUTO_ROUTING_DEFAULT_PROFILE, f"file:{config_path}", warnings

    if not isinstance(default_profile, str) or default_profile not in profiles:
        fallback = sorted(profiles)[0]
        warnings.append(
            f"default_profile '{default_profile}' not found; using '{fallback}' instead."
        )
        default_profile = fallback

    return profiles, default_profile, f"file:{config_path}", warnings


def _resolve_routing_profile(
    routing_profile: str | None,
) -> tuple[dict[str, dict[str, list[str]]], str, str, list[str]]:
    profiles, default_profile, source, warnings = _load_routing_profiles()
    if routing_profile is None:
        selected = default_profile
    else:
        selected = routing_profile.strip()
        if not selected:
            raise ValueError("routing_profile cannot be empty.")
        if selected not in profiles:
            known = ", ".join(sorted(profiles))
            raise ValueError(f"Unknown routing_profile '{routing_profile}'. Available: {known}")

    return profiles[selected], selected, source, warnings


def _infer_auto_task(prompt: str, system_prompt: str | None = None) -> str:
    text = f"{system_prompt or ''}\n{prompt}".lower()

    if any(k in text for k in ("image", "photo", "screenshot", "diagram", "ocr", "vision")):
        return "vision"
    if any(
        k in text for k in (
            "code",
            "function",
            "bug",
            "debug",
            "refactor",
            "unit test",
            "stack trace",
            "python",
            "typescript",
            "javascript",
            "rust",
            "java",
            "sql",
        )
    ):
        return "code"
    if any(
        k in text for k in (
            "fact-check",
            "fact check",
            "verify",
            "citation",
            "source",
            "accuracy",
            "hallucination",
        )
    ):
        return "verification"
    if any(k in text for k in ("story", "poem", "creative", "brainstorm", "tagline")):
        return "creative"
    if any(k in text for k in ("analyze", "prove", "reason", "tradeoff", "derive", "theorem")):
        return "reasoning"
    return "general"


def _extract_vote(response: str, options: list[str]) -> tuple[str, str]:
    lower = response.lower().strip()
    options_lower = [o.lower() for o in options]

    first_word = re.sub(r"[^a-z0-9]", "", lower.split()[0]) if lower else ""
    for i, opt in enumerate(options_lower):
        if first_word == opt:
            return options[i], "high"

    for i, opt in enumerate(options_lower):
        patterns = [
            rf"\bi vote {re.escape(opt)}\b",
            rf"\bmy answer is {re.escape(opt)}\b",
            rf"\bi choose {re.escape(opt)}\b",
            rf"\bi pick {re.escape(opt)}\b",
            rf"\bthe answer is {re.escape(opt)}\b",
        ]
        for pat in patterns:
            if re.search(pat, lower):
                return options[i], "medium"

    counts = []
    for i, opt in enumerate(options_lower):
        counts.append((lower.count(opt), i))
    counts.sort(key=lambda x: x[0], reverse=True)
    if counts and counts[0][0] > 0:
        if len(counts) == 1 or counts[0][0] > counts[1][0]:
            return options[counts[0][1]], "low"

    return "abstain", "none"


def _extract_score(response: str) -> float:
    for pattern, group in [
        (r"(\d+(?:\.\d+)?)\s*/\s*10", 1),
        (r"[Ss]core:\s*(\d+(?:\.\d+)?)", 1),
        (r"[Rr]ating:\s*(\d+(?:\.\d+)?)", 1),
        (r"(\d+(?:\.\d+)?)\s+out\s+of\s+10", 1),
    ]:
        m = re.search(pattern, response)
        if m:
            return min(10.0, max(1.0, float(m.group(group))))

    m = re.search(r"\b(\d+(?:\.\d+)?)\b", response)
    if m:
        val = float(m.group(1))
        if 1.0 <= val <= 10.0:
            return val
    return 5.0


def _pick_models_by_category(categories: list[str]) -> list[str]:
    return [mid for mid, info in MODELS.items() if info.category in categories]


# ---------------------------------------------------------------------------
# Verification constants
# ---------------------------------------------------------------------------

VERIFY_DEFAULT_MODELS = ["deepseek-v3.2:cloud", "kimi-k2-thinking:cloud", "gpt-oss:120b-cloud"]
SLOP_DEFAULT_MODELS = ["deepseek-v3.2:cloud", "gpt-oss:120b-cloud", "kimi-k2.5:cloud"]
DEFAULT_QUALITY_CRITERIA = ["accuracy", "completeness", "clarity", "specificity", "logical_coherence"]


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def _extract_verdict(response: str, options: list[str]) -> str:
    """Extract a verdict from prose. Same cascade as _extract_vote: explicit markers → first word → occurrence counting."""
    lower = response.lower().strip()
    options_lower = [o.lower() for o in options]

    # Check for explicit markers first
    for i, opt in enumerate(options_lower):
        patterns = [
            rf"\bverdict:\s*{re.escape(opt)}\b",
            rf"\bresult:\s*{re.escape(opt)}\b",
            rf"\boverall:\s*{re.escape(opt)}\b",
            rf"\bconclusion:\s*{re.escape(opt)}\b",
        ]
        for pat in patterns:
            if re.search(pat, lower):
                return options[i]

    # First word match
    first_word = re.sub(r"[^a-z0-9]", "", lower.split()[0]) if lower else ""
    for i, opt in enumerate(options_lower):
        if first_word == opt:
            return options[i]

    # Occurrence counting
    counts = []
    for i, opt in enumerate(options_lower):
        counts.append((lower.count(opt), i))
    counts.sort(key=lambda x: x[0], reverse=True)
    if counts and counts[0][0] > 0:
        if len(counts) == 1 or counts[0][0] > counts[1][0]:
            return options[counts[0][1]]

    return options[0]  # default to first option


def _extract_issues(response: str) -> list[str]:
    """Pull bullet/numbered issues from text."""
    issues: list[str] = []
    for line in response.split("\n"):
        stripped = line.strip()
        # Match bullet points: - text, * text, or numbered: 1. text
        m = re.match(r"^(?:[-*]|\d+\.)\s+(.+)", stripped)
        if m:
            issues.append(m.group(1).strip())
            continue
        # Match "Issue:" markers
        m = re.match(r"^[Ii]ssue:\s*(.+)", stripped)
        if m:
            issues.append(m.group(1).strip())
    return issues


def _extract_severity(text: str) -> str:
    """Map text to critical/major/minor via keyword matching."""
    lower = text.lower()
    if any(kw in lower for kw in ("critical", "severe", "fatal", "breaking")):
        return "critical"
    if any(kw in lower for kw in ("major", "significant", "important", "serious")):
        return "major"
    return "minor"


def _severity_rank(severity: str) -> int:
    """Numeric rank for severity comparison."""
    return {"critical": 3, "major": 2, "minor": 1}.get(severity, 0)


def _parse_claims(response: str) -> list[ClaimVerification]:
    """Parse CLAIM: ... | VERDICT: ... lines with optional CONFIDENCE/SOURCES fields."""
    claims: list[ClaimVerification] = []
    pattern = re.compile(
        r"CLAIM:\s*(.+?)\s*\|\s*VERDICT:\s*(\w+)"
        r"(?:\s*\|\s*CONFIDENCE:\s*([\d.]+))?"
        r"(?:\s*\|\s*SOURCES:\s*(.+))?"
    )
    for line in response.split("\n"):
        m = pattern.search(line)
        if m:
            claim_text = m.group(1).strip()
            verdict = m.group(2).strip().lower()
            confidence = float(m.group(3)) if m.group(3) else 0.5
            sources = [s.strip() for s in m.group(4).split(",")] if m.group(4) else []
            claims.append(
                ClaimVerification(
                    claim=claim_text,
                    verdict=verdict,
                    confidence=min(1.0, max(0.0, confidence)),
                    sources_cited=sources,
                )
            )
    return claims


def _parse_criteria_scores(response: str, criteria: list[str]) -> list[CriterionScore]:
    """Parse CRITERION: ... | SCORE: ... lines with fallback to _extract_score()."""
    scores: list[CriterionScore] = []
    found: set[str] = set()
    pattern = re.compile(
        r"(?:CRITERION|CRITERIA):\s*(.+?)\s*\|\s*SCORE:\s*(\d+(?:\.\d+)?)"
        r"(?:\s*\|\s*ISSUES:\s*(.+))?"
    )
    for line in response.split("\n"):
        m = pattern.search(line)
        if m:
            criterion = m.group(1).strip().lower()
            score = min(10.0, max(1.0, float(m.group(2))))
            issues = [i.strip() for i in m.group(3).split(";")] if m.group(3) else []
            scores.append(CriterionScore(criterion=criterion, score=score, issues=issues))
            found.add(criterion)

    # Fallback: try to find scores for any missing criteria
    for c in criteria:
        if c.lower() not in found:
            # Look for the criterion name near a score
            pattern_fallback = re.compile(
                rf"{re.escape(c)}.*?(\d+(?:\.\d+)?)\s*/\s*10", re.IGNORECASE
            )
            m = pattern_fallback.search(response)
            if m:
                score = min(10.0, max(1.0, float(m.group(1))))
                scores.append(CriterionScore(criterion=c.lower(), score=score))
            else:
                scores.append(CriterionScore(criterion=c.lower(), score=5.0))

    return scores


def _extract_surviving_issues(exchanges: list[RedTeamExchange]) -> list[SurvivingIssue]:
    """Pull unresolved issues from the final attack round."""
    if not exchanges:
        return []

    final_attack = exchanges[-1].attack
    issues: list[SurvivingIssue] = []

    # Try structured [SEVERITY] markers first
    severity_pattern = re.compile(r"\[(CRITICAL|MAJOR|MINOR)]\s*(.+)", re.IGNORECASE)
    for line in final_attack.split("\n"):
        m = severity_pattern.search(line.strip())
        if m:
            severity = m.group(1).lower()
            issue_text = m.group(2).strip()
            issues.append(
                SurvivingIssue(
                    issue=issue_text,
                    severity=severity,
                    first_raised_round=exchanges[-1].round_number,
                )
            )

    # Fallback to _extract_issues if no structured markers
    if not issues:
        raw_issues = _extract_issues(final_attack)
        for issue_text in raw_issues:
            issues.append(
                SurvivingIssue(
                    issue=issue_text,
                    severity=_extract_severity(issue_text),
                    first_raised_round=exchanges[-1].round_number,
                )
            )

    return issues


def _resolve_model(name_or_model: str) -> str:
    """Resolve a profile name or raw model ID and validate it.

    Returns the underlying model ID. Raises ValueError for unknown names.
    """
    model_id, _ = resolve(name_or_model)
    get_model(model_id)  # validates the underlying model exists in registry
    return model_id


async def _call_resolved(
    name_or_model: str,
    prompt: str,
    *,
    system_prompt: object = _UNSET,
    temperature: object = _UNSET,
    top_p: object = _UNSET,
    max_tokens: object = _UNSET,
    timeout_seconds: object = _UNSET,
    response_format: object = _UNSET,
    role_label: object = _UNSET,
    tags: object = _UNSET,
) -> CallResult:
    """Resolve a profile/model name and call Ollama or CLI with merged params."""
    model_id, profile_params = resolve(name_or_model)
    info = get_model(model_id)

    merged = merge_overrides(
        profile_params,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        response_format=response_format,
        role_label=role_label,
        tags=tags,
    )

    if info.is_cli:
        return await call_cli(model_id, prompt, **merged)
    return await call_ollama(model_id, prompt, **merged)


# ---------------------------------------------------------------------------
# Core tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def call_model(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 4096,
    response_format: str | dict[str, Any] | None = None,
    role_label: str = "",
    tags: list[str] | None = None,
) -> CallResult:
    """Call one model or profile. Accepts model IDs or profile names."""
    return await _call_resolved(
        model,
        prompt,
        system_prompt=system_prompt if system_prompt is not None else _UNSET,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        response_format=response_format if response_format is not None else _UNSET,
        role_label=role_label,
        tags=tags if tags is not None else _UNSET,
    )


@mcp.tool()
async def auto_call(
    prompt: str,
    task: str = "auto",
    strategy: str = "balanced",
    routing_profile: str | None = None,
    models: list[str] | None = None,
    max_attempts: int = 3,
    max_time_ms: int | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 4096,
    response_format: str | dict[str, Any] | None = None,
    role_label: str = "",
    tags: list[str] | None = None,
) -> AutoCallResult:
    """Auto-route prompt to best-fit model/profile, retrying with fallbacks on failure."""
    requested_task = task.lower().strip()
    strategy_name = strategy.lower().strip()
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1.")
    if max_time_ms is not None and max_time_ms < 1:
        raise ValueError("max_time_ms must be >= 1 when provided.")
    if models is not None and routing_profile is not None:
        raise ValueError("routing_profile cannot be used together with explicit models override.")
    if strategy_name not in AUTO_STRATEGIES:
        allowed = ", ".join(sorted(AUTO_STRATEGIES))
        raise ValueError(f"Unknown strategy '{strategy}'. Allowed: {allowed}")

    resolved_task = _infer_auto_task(prompt, system_prompt) if requested_task == "auto" else requested_task
    if resolved_task not in AUTO_TASKS:
        allowed = ", ".join(["auto", *sorted(AUTO_TASKS)])
        raise ValueError(f"Unknown task '{task}'. Allowed: {allowed}")

    routing_source = "manual_override"
    effective_routing_profile = "manual_override"
    routing_warnings: list[str] = []
    if models is not None:
        if not models:
            raise ValueError("If provided, models must contain at least one model/profile name.")
        candidate_names = _dedupe_keep_order(models)
    else:
        routing_table, effective_routing_profile, routing_source, routing_warnings = (
            _resolve_routing_profile(routing_profile)
        )
        candidate_names = _dedupe_keep_order(routing_table[resolved_task][strategy_name])

    for name in candidate_names:
        _resolve_model(name)

    attempts = candidate_names[:max_attempts]
    start = time.monotonic()
    attempted_models: list[str] = []
    failure_notes: list[str] = []
    last_result: CallResult | None = None
    budget_exhausted = False

    for name in attempts:
        timeout_override: object = _UNSET
        if max_time_ms is not None:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            if elapsed_ms >= max_time_ms:
                budget_exhausted = True
                failure_notes.append(
                    f"Budget exhausted before attempting '{name}' ({int(elapsed_ms)}ms elapsed)."
                )
                break
            remaining_ms = max_time_ms - elapsed_ms
            timeout_override = max(1, int(remaining_ms / 1000))

        result = await _call_resolved(
            name,
            prompt,
            system_prompt=system_prompt if system_prompt is not None else _UNSET,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout_seconds=timeout_override,
            response_format=response_format if response_format is not None else _UNSET,
            role_label=role_label,
            tags=tags if tags is not None else _UNSET,
        )
        last_result = result
        attempted_models.append(result.model)
        if result.status == "ok" and result.content.strip():
            total_elapsed = round(time.monotonic() - start, 2)
            return AutoCallResult(
                prompt_snippet=prompt[:500],
                requested_task=requested_task,
                resolved_task=resolved_task,
                strategy=strategy_name,
                routing_profile=effective_routing_profile,
                routing_source=routing_source,
                routing_warnings=routing_warnings,
                candidate_models=attempts,
                attempted_models=attempted_models,
                selected_model=result.model,
                fallback_used=len(attempted_models) > 1,
                budget_ms=max_time_ms,
                budget_exhausted=False,
                result=result,
                total_elapsed_seconds=total_elapsed,
            )

        reason = result.error or "empty content"
        failure_notes.append(f"{result.model}: {reason}")

    total_elapsed = round(time.monotonic() - start, 2)
    failed_model = attempted_models[-1] if attempted_models else ""
    error_msg = (
        "auto_call stopped due to max_time_ms budget before a successful response."
        if budget_exhausted
        else "All auto_call attempts failed."
    )
    failure_result = CallResult(
        model=failed_model,
        status="error",
        error=error_msg,
        warnings=failure_notes,
    )
    if last_result is not None:
        failure_result.content = last_result.content
        failure_result.thinking = last_result.thinking
        failure_result.elapsed_seconds = last_result.elapsed_seconds
        failure_result.role_label = last_result.role_label
        failure_result.tags = last_result.tags
        failure_result.warnings = _dedupe_keep_order(last_result.warnings + routing_warnings + failure_notes)
    else:
        failure_result.warnings = _dedupe_keep_order(routing_warnings + failure_notes)

    return AutoCallResult(
        prompt_snippet=prompt[:500],
        requested_task=requested_task,
        resolved_task=resolved_task,
        strategy=strategy_name,
        routing_profile=effective_routing_profile,
        routing_source=routing_source,
        routing_warnings=routing_warnings,
        candidate_models=attempts,
        attempted_models=attempted_models,
        selected_model="",
        fallback_used=len(attempted_models) > 1,
        budget_ms=max_time_ms,
        budget_exhausted=budget_exhausted,
        result=failure_result,
        total_elapsed_seconds=total_elapsed,
    )


@mcp.tool()
async def swarm(
    prompt: str,
    models: list[str] | None = None,
    count: int | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 4096,
    response_format: str | dict[str, Any] | None = None,
) -> SwarmResult:
    """Call models/profiles in parallel. Defaults to 6 diverse models. Pass count=13 for all."""
    if models:
        target_ids = models
    elif count is not None:
        # Caller asked for a specific count — slice from all models
        target_ids = list(MODELS.keys())[:min(count, len(MODELS))]
    else:
        target_ids = DEFAULT_SWARM_MODELS[:DEFAULT_SWARM_COUNT]
    resolved_ids = [_resolve_model(mid) for mid in target_ids]

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _bounded(name: str) -> CallResult:
        async with sem:
            return await _call_resolved(
                name, prompt,
                system_prompt=system_prompt if system_prompt is not None else _UNSET,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                response_format=response_format if response_format is not None else _UNSET,
            )

    start = time.monotonic()
    results = list(await asyncio.gather(*[_bounded(name) for name in target_ids]))
    total_elapsed = round(time.monotonic() - start, 2)

    succeeded = sum(1 for r in results if r.status == "ok")
    return SwarmResult(
        results=[_truncate_result(r) for r in results],
        total_elapsed_seconds=total_elapsed,
        models_called=len(resolved_ids),
        models_succeeded=succeeded,
        models_failed=len(resolved_ids) - succeeded,
    )


@mcp.tool()
async def list_models() -> list[ModelAvailability]:
    """List registered models with availability status."""
    connected = await check_ollama()
    return [
        ModelAvailability(
            model_id=m.model_id,
            display_name=m.display_name,
            category=m.category,
            strengths=m.strengths,
            available=connected,
        )
        for m in MODELS.values()
    ]


@mcp.tool()
async def health_check() -> HealthStatus:
    """Check server health and Ollama connectivity."""
    connected = await check_ollama()
    return HealthStatus(
        version=__version__,
        ollama_url=OLLAMA_BASE_URL,
        ollama_connected=connected,
        registered_models=len(MODELS),
    )


# ---------------------------------------------------------------------------
# Debate
# ---------------------------------------------------------------------------


@mcp.tool()
async def debate(
    prompt: str,
    model_a: str | None = None,
    model_b: str | None = None,
    rounds: int = 3,
) -> DebateResult:
    """Multi-round debate. Models/profiles alternate, each seeing full transcript."""
    a = model_a or DEFAULT_DEBATE[0]
    b = model_b or DEFAULT_DEBATE[1]
    _resolve_model(a)
    _resolve_model(b)

    debate_rounds: list[DebateRound] = []
    transcript = f"Debate topic: {prompt}\n\n"

    start = time.monotonic()
    for i in range(1, rounds + 1):
        current_model = a if i % 2 == 1 else b
        other_model = b if i % 2 == 1 else a

        system = (
            f"You are in a structured debate (round {i}/{rounds}). "
            f"You are arguing your perspective on the topic. "
            f"Your opponent is {other_model}. Be substantive and direct."
        )

        result = await _call_resolved(current_model, transcript, system_prompt=system)
        debate_rounds.append(
            DebateRound(round_number=i, model=current_model, content=result.content)
        )
        transcript += f"--- Round {i} ({current_model}) ---\n{result.content}\n\n"

    total_elapsed = round(time.monotonic() - start, 2)
    return DebateResult(
        model_a=a,
        model_b=b,
        prompt=prompt,
        rounds=debate_rounds,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Vote
# ---------------------------------------------------------------------------


@mcp.tool()
async def vote(
    prompt: str,
    options: list[str] | None = None,
    models: list[str] | None = None,
) -> VoteResult:
    """Models/profiles vote on a question. Returns tally, majority, agreement %."""
    opts = options or ["yes", "no"]
    target_ids = models if models else list(MODELS.keys())
    for mid in target_ids:
        _resolve_model(mid)

    options_str = ", ".join(opts)
    system = (
        f"You must vote on the following question. The valid options are: {options_str}. "
        f"Start your response with your chosen option word, then explain briefly."
    )

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _bounded(name: str) -> CallResult:
        async with sem:
            return await _call_resolved(name, prompt, system_prompt=system)

    results = list(await asyncio.gather(*[_bounded(mid) for mid in target_ids]))

    votes: list[Vote] = []
    tally: dict[str, int] = {o: 0 for o in opts}
    tally["abstain"] = 0

    for r in results:
        extracted, confidence = _extract_vote(r.content, opts)
        votes.append(
            Vote(
                model=r.model,
                raw_response=_truncate(r.content),
                extracted_vote=extracted,
                confidence=confidence,
            )
        )
        if extracted in tally:
            tally[extracted] += 1
        else:
            tally["abstain"] += 1

    if tally["abstain"] == 0:
        del tally["abstain"]

    majority = max(tally, key=lambda k: tally[k]) if tally else ""
    majority_count = tally.get(majority, 0)
    agreement_pct = round(majority_count / len(target_ids) * 100, 1) if target_ids else 0.0

    return VoteResult(
        prompt=prompt,
        votes=votes,
        tally=tally,
        majority=majority,
        total_models=len(target_ids),
        agreement_pct=agreement_pct,
    )


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------


@mcp.tool()
async def consensus(
    prompt: str,
    models: list[str] | None = None,
    judge_model: str | None = None,
) -> ConsensusResult:
    """Swarm then judge-synthesize into a unified answer. Accepts profiles."""
    judge = judge_model or DEFAULT_JUDGE
    target_ids = models if models else list(MODELS.keys())
    for mid in target_ids:
        _resolve_model(mid)
    _resolve_model(judge)

    start = time.monotonic()

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _bounded(name: str) -> CallResult:
        async with sem:
            return await _call_resolved(name, prompt)

    responses = list(await asyncio.gather(*[_bounded(mid) for mid in target_ids]))

    response_block = "\n\n".join(
        f"[{r.model}]: {r.content}" for r in responses if r.status == "ok"
    )
    synthesis_prompt = (
        f"Original question: {prompt}\n\n"
        f"The following models provided responses:\n\n{response_block}\n\n"
        f"Identify areas of agreement and disagreement among these responses. "
        f"Then produce a single, unified answer that captures the consensus."
    )
    synthesis_result = await _call_resolved(judge, synthesis_prompt)

    total_elapsed = round(time.monotonic() - start, 2)
    return ConsensusResult(
        prompt=prompt,
        individual_responses=[_truncate_result(r) for r in responses],
        synthesis=synthesis_result.content,
        judge_model=judge,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Code Review
# ---------------------------------------------------------------------------


@mcp.tool()
async def code_review(
    code: str,
    language: str | None = None,
    review_models: list[str] | None = None,
    merge_model: str | None = None,
) -> CodeReviewResult:
    """Parallel code reviews merged and sorted by severity. Accepts profiles."""
    reviewers = review_models or CODE_MODELS
    merger = merge_model or DEFAULT_JUDGE
    for mid in reviewers:
        _resolve_model(mid)
    _resolve_model(merger)

    lang_hint = f" ({language})" if language else ""
    system = (
        f"You are an expert code reviewer. Review the following{lang_hint} code for: "
        f"bugs, security issues, performance problems, and readability. "
        f"Reference line numbers where applicable. Suggest specific fixes."
    )

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _bounded(name: str) -> CallResult:
        async with sem:
            return await _call_resolved(name, code, system_prompt=system)

    start = time.monotonic()
    reviews = list(await asyncio.gather(*[_bounded(mid) for mid in reviewers]))

    review_block = "\n\n".join(
        f"[Review by {r.model}]:\n{r.content}" for r in reviews if r.status == "ok"
    )
    merge_prompt = (
        f"The following independent code reviews were produced for this code:\n\n"
        f"{review_block}\n\n"
        f"Deduplicate the findings and produce a single merged review. "
        f"Sort issues by severity (critical -> major -> minor). "
        f"Preserve specific fix suggestions."
    )
    merge_result = await _call_resolved(merger, merge_prompt)

    total_elapsed = round(time.monotonic() - start, 2)
    return CodeReviewResult(
        code_snippet=code[:500],
        reviews=[_truncate_result(r) for r in reviews],
        merged_review=merge_result.content,
        reviewer_models=reviewers,
        merge_model=merger,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Multi-Solve
# ---------------------------------------------------------------------------


@mcp.tool()
async def multi_solve(
    problem: str,
    language: str | None = None,
    models: list[str] | None = None,
) -> MultiSolveResult:
    """Multiple models/profiles solve the same problem independently."""
    target_ids = models or (
        _pick_models_by_category(["code", "reasoning"])
        + ["qwen3-next:80b-cloud", "gpt-oss:120b-cloud"]
    )
    seen: set[str] = set()
    deduped: list[str] = []
    for mid in target_ids:
        if mid not in seen:
            seen.add(mid)
            deduped.append(mid)
    target_ids = deduped

    for mid in target_ids:
        _resolve_model(mid)

    lang_instruction = f" Write the solution in {language}." if language else ""
    system = (
        f"Write a complete, working solution to the given problem.{lang_instruction} "
        f"Include brief comments explaining your approach."
    )

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _bounded(name: str) -> CallResult:
        async with sem:
            return await _call_resolved(name, problem, system_prompt=system)

    start = time.monotonic()
    solutions = list(await asyncio.gather(*[_bounded(mid) for mid in target_ids]))
    total_elapsed = round(time.monotonic() - start, 2)

    return MultiSolveResult(
        problem=problem,
        language=language,
        solutions=solutions,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@mcp.tool()
async def benchmark(
    prompts: list[str],
    models: list[str] | None = None,
) -> BenchmarkResult:
    """Prompt x model matrix. Returns per-model latency stats. Accepts profiles."""
    target_ids = models if models else list(MODELS.keys())
    for mid in target_ids:
        _resolve_model(mid)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _run_cell(name: str, prompt_idx: int) -> BenchmarkCell:
        async with sem:
            result = await _call_resolved(name, prompts[prompt_idx])
            return BenchmarkCell(
                model=result.model,
                prompt_index=prompt_idx,
                content=_truncate(result.content),
                elapsed_seconds=result.elapsed_seconds,
                status=result.status,
            )

    start = time.monotonic()
    tasks = [
        _run_cell(mid, pi)
        for pi in range(len(prompts))
        for mid in target_ids
    ]
    cells = list(await asyncio.gather(*tasks))
    total_elapsed = round(time.monotonic() - start, 2)

    model_times: dict[str, list[float]] = {mid: [] for mid in target_ids}
    for cell in cells:
        if cell.status == "ok":
            model_times[cell.model].append(cell.elapsed_seconds)
    model_stats = {
        mid: round(sum(times) / len(times), 2) if times else 0.0
        for mid, times in model_times.items()
    }

    return BenchmarkResult(
        prompts=prompts,
        models=target_ids,
        results=cells,
        model_stats=model_stats,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Rank
# ---------------------------------------------------------------------------


@mcp.tool()
async def rank(
    prompt: str,
    models: list[str] | None = None,
    judge_count: int = 3,
) -> RankResult:
    """Models/profiles answer, then peer-judge each other 1-10. Returns leaderboard."""
    target_ids = models or RANK_DEFAULT_MODELS
    for mid in target_ids:
        _resolve_model(mid)

    start = time.monotonic()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _answer(name: str) -> CallResult:
        async with sem:
            return await _call_resolved(name, prompt)

    answers = list(await asyncio.gather(*[_answer(mid) for mid in target_ids]))

    judge_system = (
        "Rate the following response on a scale of 1 to 10. "
        "Start your reply with 'Score: X/10' then explain briefly."
    )

    async def _judge(judge_id: str, answer: CallResult) -> tuple[str, str, float]:
        async with sem:
            judge_prompt = (
                f"Question: {prompt}\n\n"
                f"Response by {answer.model}:\n{answer.content}"
            )
            result = await _call_resolved(judge_id, judge_prompt, system_prompt=judge_system)
            score = _extract_score(result.content)
            return answer.model, judge_id, score

    judge_tasks = []
    rng = random.Random(hash(prompt))
    for answer in answers:
        candidates = [mid for mid in target_ids if mid != answer.model]
        actual_count = min(judge_count, len(candidates))
        judges = rng.sample(candidates, actual_count) if candidates else []
        for jid in judges:
            judge_tasks.append(_judge(jid, answer))

    judge_results = list(await asyncio.gather(*judge_tasks))

    scores_map: dict[str, dict[str, float]] = {mid: {} for mid in target_ids}
    for target_model, judge_model, score in judge_results:
        scores_map[target_model][judge_model] = score

    leaderboard = []
    for mid in target_ids:
        received = scores_map[mid]
        avg = round(sum(received.values()) / len(received), 2) if received else 0.0
        answer_content = next((a.content for a in answers if a.model == mid), "")
        leaderboard.append(
            ModelScore(
                model=mid,
                avg_score=avg,
                scores_received=received,
                response=_truncate(answer_content),
            )
        )
    leaderboard.sort(key=lambda s: s.avg_score, reverse=True)

    total_elapsed = round(time.monotonic() - start, 2)
    return RankResult(
        prompt=prompt,
        leaderboard=leaderboard,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------


@mcp.tool()
async def chain(
    prompt: str,
    pipeline: list[str],
    pass_context: bool = True,
) -> ChainResult:
    """Sequential pipeline — each model/profile builds on prior outputs."""
    for mid in pipeline:
        _resolve_model(mid)

    steps: list[ChainStep] = []
    all_outputs: list[str] = []

    start = time.monotonic()
    for i, model_id in enumerate(pipeline):
        if i == 0:
            step_prompt = prompt
        else:
            if pass_context:
                context = "\n\n".join(
                    f"--- Step {j + 1} ({pipeline[j]}) ---\n{all_outputs[j]}"
                    for j in range(i)
                )
            else:
                context = f"Previous analysis by {pipeline[i - 1]}:\n{all_outputs[-1]}"

            step_prompt = (
                f"{context}\n\n"
                f"Original question: {prompt}\n\n"
                f"Build on this — extend, refine, or correct."
            )

        step_start = time.monotonic()
        result = await _call_resolved(model_id, step_prompt)
        step_elapsed = round(time.monotonic() - step_start, 2)

        steps.append(
            ChainStep(
                step=i + 1,
                model=model_id,
                content=result.content,
                thinking=result.thinking,
                elapsed_seconds=step_elapsed,
            )
        )
        all_outputs.append(result.content)

    total_elapsed = round(time.monotonic() - start, 2)
    return ChainResult(
        prompt=prompt,
        steps=steps,
        final_output=all_outputs[-1] if all_outputs else "",
        pipeline=pipeline,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Map-Reduce
# ---------------------------------------------------------------------------


@mcp.tool()
async def map_reduce(
    prompt: str,
    mapper_models: list[str] | None = None,
    reducer_model: str | None = None,
    reduce_prompt: str | None = None,
) -> MapReduceResult:
    """Fan-out to models/profiles, fan-in with a reducer. Custom reduce_prompt supported."""
    mappers = mapper_models if mapper_models else list(MODELS.keys())
    reducer = reducer_model or DEFAULT_JUDGE
    for mid in mappers:
        _resolve_model(mid)
    _resolve_model(reducer)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _map(name: str) -> CallResult:
        async with sem:
            return await _call_resolved(name, prompt)

    start = time.monotonic()
    mapped = list(await asyncio.gather(*[_map(mid) for mid in mappers]))

    response_block = "\n\n".join(
        f"[{r.model}]: {r.content}" for r in mapped if r.status == "ok"
    )
    n = sum(1 for r in mapped if r.status == "ok")
    default_reduce = (
        f"Synthesize these {n} responses into a single comprehensive answer. "
        f"Preserve the strongest points from each."
    )
    reduce_instruction = reduce_prompt or default_reduce
    full_reduce_prompt = (
        f"Original question: {prompt}\n\n"
        f"Responses from {n} models:\n\n{response_block}\n\n"
        f"Instructions: {reduce_instruction}"
    )
    reduce_result = await _call_resolved(reducer, full_reduce_prompt)

    total_elapsed = round(time.monotonic() - start, 2)
    return MapReduceResult(
        prompt=prompt,
        mapped_responses=[_truncate_result(r) for r in mapped],
        reduced_output=reduce_result.content,
        reducer_model=reducer,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Blind Taste Test
# ---------------------------------------------------------------------------


@mcp.tool()
async def blind_taste_test(
    prompt: str,
    count: int = 3,
) -> BlindTasteResult:
    """Anonymous A/B/C comparison. Reveal mapping shows which model was which."""
    all_ids = list(MODELS.keys())
    count = min(count, len(all_ids))

    seed = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    selected = rng.sample(all_ids, count)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _call(name: str) -> CallResult:
        async with sem:
            return await _call_resolved(name, prompt)

    start = time.monotonic()
    results = list(await asyncio.gather(*[_call(mid) for mid in selected]))

    indexed = list(enumerate(results))
    rng.shuffle(indexed)

    labels = [chr(65 + i) for i in range(count)]
    responses: list[BlindResponse] = []
    reveal: dict[str, str] = {}
    for label, (_, r) in zip(labels, indexed):
        responses.append(BlindResponse(label=f"Response {label}", content=r.content))
        reveal[f"Response {label}"] = r.model

    total_elapsed = round(time.monotonic() - start, 2)
    return BlindTasteResult(
        prompt=prompt,
        responses=responses,
        reveal=reveal,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Contrarian
# ---------------------------------------------------------------------------


@mcp.tool()
async def contrarian(
    prompt: str,
    thesis_model: str | None = None,
    antithesis_model: str | None = None,
) -> ContrarianResult:
    """Thesis then devil's-advocate antithesis. Accepts profiles."""
    t_model = thesis_model or "qwen3-next:80b-cloud"
    a_model = antithesis_model or "deepseek-v3.2:cloud"
    _resolve_model(t_model)
    _resolve_model(a_model)

    start = time.monotonic()

    thesis_result = await _call_resolved(t_model, prompt)

    anti_system = (
        "You are a critical analyst. Find logical gaps, challenge assumptions, "
        "identify missing perspectives, and argue alternatives. "
        "Be substantive, not contrarian for its own sake."
    )
    anti_prompt = (
        f"Original question: {prompt}\n\n"
        f"Thesis (by {t_model}):\n{thesis_result.content}\n\n"
        f"Provide a substantive counterargument."
    )
    anti_result = await _call_resolved(a_model, anti_prompt, system_prompt=anti_system)

    total_elapsed = round(time.monotonic() - start, 2)
    return ContrarianResult(
        prompt=prompt,
        thesis_model=t_model,
        thesis=thesis_result.content,
        antithesis_model=a_model,
        antithesis=anti_result.content,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


@mcp.tool()
async def verify(
    content: str,
    verifier_models: list[str] | None = None,
    judge_model: str | None = None,
) -> VerifyResult:
    """Cross-model fact verification. Evaluates claims in existing content (unlike consensus which generates answers)."""
    verifiers = verifier_models or VERIFY_DEFAULT_MODELS
    judge = judge_model or DEFAULT_JUDGE
    for mid in verifiers:
        _resolve_model(mid)
    _resolve_model(judge)

    system = (
        "You are a rigorous fact-checker. Analyze the following content and evaluate each factual claim.\n"
        "For each claim, output a line in this exact format:\n"
        "CLAIM: <the claim> | VERDICT: <supported/unsupported/contradicted/unverifiable> | CONFIDENCE: <0.0-1.0> | SOURCES: <source1, source2>\n\n"
        "Also note any internal contradictions in the content."
    )

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _bounded(name: str) -> CallResult:
        async with sem:
            return await _call_resolved(name, content, system_prompt=system, temperature=0.2)

    start = time.monotonic()
    reports = list(await asyncio.gather(*[_bounded(mid) for mid in verifiers]))

    # Judge synthesizes verifier reports
    report_block = "\n\n".join(
        f"[Verifier: {r.model}]:\n{r.content}" for r in reports if r.status == "ok"
    )
    synthesis_prompt = (
        f"Content under review:\n{content}\n\n"
        f"Verification reports from {len(verifiers)} models:\n\n{report_block}\n\n"
        "Synthesize these reports into per-claim verdicts. Use this format for each claim:\n"
        "CLAIM: <claim> | VERDICT: <supported/unsupported/contradicted/unverifiable> | CONFIDENCE: <0.0-1.0> | SOURCES: <sources>\n\n"
        "Then state OVERALL VERDICT: pass/fail/mixed\n"
        "List any INTERNAL CONTRADICTIONS found."
    )
    judge_result = await _call_resolved(judge, synthesis_prompt, temperature=0.2)

    # Parse structured output
    claims = _parse_claims(judge_result.content)
    overall = _extract_verdict(judge_result.content, ["pass", "fail", "mixed"])

    # Extract contradictions
    contradictions: list[str] = []
    in_contradictions = False
    for line in judge_result.content.split("\n"):
        if "contradiction" in line.lower():
            in_contradictions = True
            # Check if this line itself has content after the marker
            m = re.search(r"contradictions?:?\s*(.+)", line, re.IGNORECASE)
            if m and m.group(1).strip() and not m.group(1).strip().lower().startswith("none"):
                contradictions.append(m.group(1).strip())
            continue
        if in_contradictions:
            stripped = line.strip()
            if stripped and stripped[0] in "-*•":
                contradictions.append(stripped.lstrip("-*• ").strip())
            elif stripped and re.match(r"\d+\.", stripped):
                contradictions.append(re.sub(r"^\d+\.\s*", "", stripped).strip())

    # Compute overall confidence
    if claims:
        overall_confidence = round(sum(c.confidence for c in claims) / len(claims), 2)
    else:
        overall_confidence = 0.0

    total_elapsed = round(time.monotonic() - start, 2)
    return VerifyResult(
        content_snippet=content[:500],
        claims=claims,
        overall_verdict=overall,
        overall_confidence=overall_confidence,
        internal_contradictions=contradictions,
        verifier_models=verifiers,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Red Team
# ---------------------------------------------------------------------------


@mcp.tool()
async def red_team(
    content: str,
    rounds: int = 3,
    attacker_model: str | None = None,
    defender_model: str | None = None,
) -> RedTeamResult:
    """Iterative adversarial stress testing. Attacker finds flaws, defender addresses them, attacker escalates."""
    attacker = attacker_model or "deepseek-v3.2:cloud"
    defender = defender_model or "kimi-k2-thinking:cloud"
    _resolve_model(attacker)
    _resolve_model(defender)

    exchanges: list[RedTeamExchange] = []
    transcript = f"Content under review:\n{content}\n\n"

    attack_system = (
        "You are an adversarial red-teamer. Your job is to find weaknesses, flaws, "
        "logical errors, missing edge cases, and potential failures in the content. "
        "Be specific and escalate with each round. "
        "If you cannot find any new issues, respond with exactly: NO NEW ISSUES FOUND"
    )
    defense_system = (
        "You are defending content against adversarial critique. "
        "Address each issue raised point by point. Acknowledge valid concerns, "
        "provide fixes or mitigations, and push back where criticism is unfounded."
    )

    start = time.monotonic()
    converged = False

    for i in range(1, rounds + 1):
        # Attacker turn
        attack_prompt = (
            f"{transcript}"
            f"Round {i}: Find new flaws, weaknesses, and edge cases not yet addressed."
        )
        attack_result = await _call_resolved(
            attacker, attack_prompt, system_prompt=attack_system, temperature=0.3
        )
        attack_text = attack_result.content

        # Check for convergence
        if "NO NEW ISSUES FOUND" in attack_text.upper():
            converged = True
            exchanges.append(
                RedTeamExchange(
                    round_number=i,
                    attack=attack_text,
                    attack_model=attacker,
                    defense="(converged — no defense needed)",
                    defense_model=defender,
                )
            )
            break

        # Defender turn
        defense_prompt = (
            f"{transcript}"
            f"Round {i} attack by {attacker}:\n{attack_text}\n\n"
            f"Address each issue raised."
        )
        defense_result = await _call_resolved(
            defender, defense_prompt, system_prompt=defense_system, temperature=0.3
        )

        exchange = RedTeamExchange(
            round_number=i,
            attack=attack_text,
            attack_model=attacker,
            defense=defense_result.content,
            defense_model=defender,
        )
        exchanges.append(exchange)
        transcript += (
            f"--- Round {i} Attack ({attacker}) ---\n{attack_text}\n\n"
            f"--- Round {i} Defense ({defender}) ---\n{defense_result.content}\n\n"
        )

    # Extract surviving issues and compute robustness score
    surviving = _extract_surviving_issues(exchanges)
    if converged and not surviving:
        robustness_score = 10.0
    else:
        # Decrease score per surviving issue weighted by severity
        penalty = sum(_severity_rank(s.severity) for s in surviving)
        robustness_score = max(1.0, min(10.0, 10.0 - penalty))

    total_elapsed = round(time.monotonic() - start, 2)
    return RedTeamResult(
        content_snippet=content[:500],
        exchanges=exchanges,
        surviving_issues=surviving,
        robustness_score=robustness_score,
        rounds_completed=len(exchanges),
        converged=converged,
        attacker_model=attacker,
        defender_model=defender,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Quality Gate
# ---------------------------------------------------------------------------


@mcp.tool()
async def quality_gate(
    content: str,
    criteria: list[str] | None = None,
    threshold: float = 7.0,
    judge_model: str | None = None,
) -> QualityGateResult:
    """Pass/fail evaluation against quality criteria. Checkpoint for AI pipelines."""
    judge = judge_model or DEFAULT_JUDGE
    _resolve_model(judge)
    criteria_list = criteria or DEFAULT_QUALITY_CRITERIA

    criteria_block = "\n".join(f"- {c}" for c in criteria_list)
    system = (
        "You are a quality evaluator. Score the following content on each criterion from 1-10.\n"
        "Use this exact format for each criterion:\n"
        "CRITERION: <name> | SCORE: <1-10> | ISSUES: <issue1; issue2>\n\n"
        "Then provide:\n"
        "OVERALL VERDICT: pass or fail\n"
        "BLOCKING ISSUES: <list any critical problems>"
    )
    prompt = (
        f"Content to evaluate:\n{content}\n\n"
        f"Evaluate on these criteria:\n{criteria_block}\n\n"
        f"Threshold for passing: {threshold}/10"
    )

    start = time.monotonic()
    result = await _call_resolved(judge, prompt, system_prompt=system, temperature=0.2)

    # Parse criteria scores
    criteria_scores = _parse_criteria_scores(result.content, criteria_list)

    # Compute overall score
    if criteria_scores:
        overall_score = round(sum(c.score for c in criteria_scores) / len(criteria_scores), 2)
    else:
        overall_score = 5.0

    # Determine pass/fail: overall >= threshold AND no criterion catastrophically low
    catastrophic_threshold = threshold - 2.0
    blocking: list[str] = []
    for cs in criteria_scores:
        if cs.score < catastrophic_threshold:
            blocking.append(f"{cs.criterion}: score {cs.score} below catastrophic threshold {catastrophic_threshold}")
    passed = overall_score >= threshold and not blocking

    # Also extract any explicit blocking issues from the response
    blocking_from_text = _extract_issues(result.content)
    for issue in blocking_from_text:
        if "block" in issue.lower() or "critical" in issue.lower():
            blocking.append(issue)

    total_elapsed = round(time.monotonic() - start, 2)
    return QualityGateResult(
        content_snippet=content[:500],
        passed=passed,
        overall_score=overall_score,
        threshold=threshold,
        criteria_scores=criteria_scores,
        blocking_issues=blocking,
        judge_model=judge,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Detect Slop
# ---------------------------------------------------------------------------


@mcp.tool()
async def detect_slop(
    content: str,
    detector_models: list[str] | None = None,
) -> DetectSlopResult:
    """Detect low-quality AI-generated filler content using multiple models in parallel."""
    detectors = detector_models or SLOP_DEFAULT_MODELS
    for mid in detectors:
        _resolve_model(mid)

    signal_types = (
        "filler_phrases, vague_generalities, hallucinated_citations, "
        "confident_unsupported_claims, generic_structure, hedging_without_substance, "
        "repetitive_padding"
    )
    system = (
        f"You are an AI content quality detector. Analyze the following content for signs of "
        f"low-quality AI-generated 'slop'. Look for these signal types: {signal_types}.\n\n"
        "For each signal found, output a line in this format:\n"
        "SIGNAL: <type> | SEVERITY: <critical/major/minor> | DESCRIPTION: <what you found> | "
        "EXAMPLES: <example1; example2>\n\n"
        "End with: SLOP SCORE: <0-10> (0 = clean, 10 = pure slop)"
    )

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _bounded(name: str) -> CallResult:
        async with sem:
            return await _call_resolved(name, content, system_prompt=system, temperature=0.2)

    start = time.monotonic()
    reports = list(await asyncio.gather(*[_bounded(mid) for mid in detectors]))

    # Parse signals from each detector
    signal_pattern = re.compile(
        r"SIGNAL:\s*(.+?)\s*\|\s*SEVERITY:\s*(\w+)\s*\|\s*DESCRIPTION:\s*([^|]+?)(?:\s*\|\s*EXAMPLES:\s*(.+))?\s*$"
    )
    score_pattern = re.compile(r"SLOP\s*SCORE:\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

    all_signals: dict[str, SlopSignal] = {}
    scores: list[float] = []

    for report in reports:
        if report.status != "ok":
            continue
        for line in report.content.split("\n"):
            m = signal_pattern.search(line)
            if m:
                signal_type = m.group(1).strip().lower().replace(" ", "_")
                severity = m.group(2).strip().lower()
                if severity not in ("critical", "major", "minor"):
                    severity = _extract_severity(severity)
                description = m.group(3).strip()
                examples = [e.strip() for e in m.group(4).split(";")] if m.group(4) else []

                if signal_type in all_signals:
                    existing = all_signals[signal_type]
                    # Escalate severity
                    if _severity_rank(severity) > _severity_rank(existing.severity):
                        all_signals[signal_type] = SlopSignal(
                            signal_type=signal_type,
                            description=description,
                            examples=list(set(existing.examples + examples)),
                            severity=severity,
                        )
                    else:
                        # Deduplicate examples
                        all_signals[signal_type] = SlopSignal(
                            signal_type=existing.signal_type,
                            description=existing.description,
                            examples=list(set(existing.examples + examples)),
                            severity=existing.severity,
                        )
                else:
                    all_signals[signal_type] = SlopSignal(
                        signal_type=signal_type,
                        description=description,
                        examples=examples,
                        severity=severity,
                    )

        # Extract score
        m = score_pattern.search(report.content)
        if m:
            scores.append(min(10.0, max(0.0, float(m.group(1)))))

    # Compute aggregated score
    if scores:
        slop_score = round(sum(scores) / len(scores), 1)
        agreement_pct = round(
            (1.0 - (max(scores) - min(scores)) / 10.0) * 100, 1
        ) if len(scores) > 1 else 100.0
    else:
        slop_score = 0.0
        agreement_pct = 0.0

    # Determine verdict
    if slop_score <= 2:
        verdict = "clean"
    elif slop_score <= 4:
        verdict = "mild"
    elif slop_score <= 7:
        verdict = "significant"
    else:
        verdict = "severe"

    total_elapsed = round(time.monotonic() - start, 2)
    return DetectSlopResult(
        content_snippet=content[:500],
        slop_score=slop_score,
        verdict=verdict,
        signals=list(all_signals.values()),
        detector_models=detectors,
        agreement_pct=agreement_pct,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Profile tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def create_profile(
    name: str,
    model: str,
    description: str = "",
    system_prompt: str = "",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 4096,
    timeout_seconds: int = 120,
    role_label: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Create a named profile (model + params). Use anywhere model IDs are accepted."""
    get_model(model)  # validate underlying model
    profile = ProfileData(
        model=model,
        description=description,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        role_label=role_label,
        tags=tags or [],
    )
    store = get_profile_store()
    store.save(name, profile)
    return {"name": name, **profile.model_dump()}


@mcp.tool()
async def get_profile(name: str) -> dict[str, Any]:
    """View a profile's full configuration."""
    store = get_profile_store()
    profile = store.load(name)
    if profile is None:
        raise ValueError(f"Profile '{name}' not found.")
    return {"name": name, **profile.model_dump()}


@mcp.tool()
async def list_profiles() -> ProfileListResult:
    """List all saved profiles."""
    store = get_profile_store()
    profiles = store.list_all()
    return ProfileListResult(profiles=profiles, count=len(profiles))


@mcp.tool()
async def clone_profile(
    source: str,
    target: str,
    description: str | None = None,
    system_prompt: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    role_label: str | None = None,
) -> dict[str, Any]:
    """Clone an existing profile with optional modifications."""
    store = get_profile_store()
    overrides: dict[str, Any] = {}
    if description is not None:
        overrides["description"] = description
    if system_prompt is not None:
        overrides["system_prompt"] = system_prompt
    if temperature is not None:
        overrides["temperature"] = temperature
    if top_p is not None:
        overrides["top_p"] = top_p
    if max_tokens is not None:
        overrides["max_tokens"] = max_tokens
    if role_label is not None:
        overrides["role_label"] = role_label
    cloned = store.clone(source, target, **overrides)
    return {"name": target, **cloned.model_dump()}


@mcp.tool()
async def delete_profile(name: str) -> dict[str, str]:
    """Delete a non-builtin profile."""
    store = get_profile_store()
    store.delete(name)
    return {"deleted": name}


# ---------------------------------------------------------------------------
# Pipeline tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def create_pipeline(
    name: str,
    steps: list[str],
    description: str = "",
) -> dict[str, Any]:
    """Save a named sequence of profile/model steps as a pipeline."""
    # Validate all steps resolve
    for step in steps:
        _resolve_model(step)
    pipeline = PipelineData(name=name, description=description, steps=steps)
    store = get_pipeline_store()
    store.save(name, pipeline)
    return {"name": name, **pipeline.model_dump()}


@mcp.tool()
async def list_pipelines() -> dict[str, Any]:
    """List all saved pipelines."""
    store = get_pipeline_store()
    pipelines = store.list_all()
    return {"pipelines": pipelines, "count": len(pipelines)}


@mcp.tool()
async def run_pipeline(
    name: str,
    prompt: str,
    pass_context: bool = True,
) -> PipelineResult:
    """Execute a saved pipeline. Each step passes context to the next."""
    store = get_pipeline_store()
    pipeline = store.load(name)
    if pipeline is None:
        raise ValueError(f"Pipeline '{name}' not found.")

    for step_name in pipeline.steps:
        _resolve_model(step_name)

    steps: list[PipelineStepResult] = []
    all_outputs: list[str] = []

    start = time.monotonic()
    for i, step_name in enumerate(pipeline.steps):
        model_id, profile_params = resolve(step_name)
        role_label = profile_params.get("role_label", "")

        if i == 0:
            step_prompt = prompt
        else:
            if pass_context:
                context = "\n\n".join(
                    f"--- Step {j + 1} ({pipeline.steps[j]}) ---\n{all_outputs[j]}"
                    for j in range(i)
                )
            else:
                context = (
                    f"Previous analysis by {pipeline.steps[i - 1]}:\n{all_outputs[-1]}"
                )
            step_prompt = (
                f"{context}\n\n"
                f"Original question: {prompt}\n\n"
                f"Build on this — extend, refine, or correct."
            )

        step_start = time.monotonic()
        result = await _call_resolved(step_name, step_prompt)
        step_elapsed = round(time.monotonic() - step_start, 2)

        steps.append(
            PipelineStepResult(
                step=i + 1,
                model=result.model,
                role_label=role_label,
                content=result.content,
                thinking=result.thinking,
                elapsed_seconds=step_elapsed,
            )
        )
        all_outputs.append(result.content)

    total_elapsed = round(time.monotonic() - start, 2)
    return PipelineResult(
        prompt=prompt,
        steps=steps,
        final_output=all_outputs[-1] if all_outputs else "",
        pipeline_name=name,
        total_elapsed_seconds=total_elapsed,
    )


@mcp.tool()
async def delete_pipeline(name: str) -> dict[str, str]:
    """Delete a non-builtin pipeline."""
    store = get_pipeline_store()
    store.delete(name)
    return {"deleted": name}


# ---------------------------------------------------------------------------
# Local code execution
# ---------------------------------------------------------------------------

CLI_AGENTS = ["codex:cli", "claude:cli", "kimi:cli", "gemini:cli"]


@mcp.tool()
async def exec_task(
    task: str,
    agent: str = "codex:cli",
    working_dir: str = ".",
    timeout_seconds: int = 300,
) -> ExecTaskResult:
    """Delegate a coding task to a local CLI agent for execution.

    The agent runs locally on this machine and can read/write files, execute
    code, install packages, and perform any coding task autonomously.

    Args:
        task: What to do — be specific (e.g. "create a Python script that
              parses CSV files in ./data and outputs summary stats").
        agent: Which CLI agent to use.  Options: codex:cli (default),
               claude:cli, kimi:cli, gemini:cli.
        working_dir: Directory the agent operates in (default: current dir).
        timeout_seconds: Max execution time (default: 300s / 5 min).
    """
    if agent not in CLI_AGENTS:
        return ExecTaskResult(
            task=task,
            agent=agent,
            status="error",
            error=f"Unknown agent '{agent}'. Choose from: {', '.join(CLI_AGENTS)}",
        )

    info = MODELS.get(agent)
    if info is None:
        return ExecTaskResult(
            task=task, agent=agent, status="error", error=f"Agent '{agent}' not in registry."
        )

    import os

    abs_dir = os.path.abspath(os.path.expanduser(working_dir))
    if not os.path.isdir(abs_dir):
        return ExecTaskResult(
            task=task,
            agent=agent,
            working_dir=abs_dir,
            status="error",
            error=f"Directory does not exist: {abs_dir}",
        )

    result = await call_cli(
        agent,
        task,
        timeout_seconds=timeout_seconds,
        cwd=abs_dir,
    )

    return ExecTaskResult(
        task=task,
        agent=agent,
        working_dir=abs_dir,
        output=result.content,
        status=result.status,
        error=result.error or "",
        elapsed_seconds=result.elapsed_seconds,
    )
