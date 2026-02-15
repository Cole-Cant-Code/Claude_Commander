"""Claude Commander MCP server — 4 tools for Ollama model orchestration."""

from __future__ import annotations

import asyncio
import time

from fastmcp import FastMCP

from claude_commander import __version__
from claude_commander.models import (
    CallResult,
    HealthStatus,
    ModelAvailability,
    SwarmResult,
)
from claude_commander.ollama import OLLAMA_BASE_URL, call_ollama, check_ollama
from claude_commander.registry import MODELS, get_model

mcp = FastMCP("Claude Commander")

MAX_CONCURRENCY = 13


@mcp.tool()
async def call_model(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> CallResult:
    """Call a single Ollama model and return its response.

    Args:
        model: Model ID from the registry (e.g. "deepseek-v3.2:cloud").
        prompt: The user prompt to send.
        system_prompt: Optional system prompt.
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens to generate.
    """
    get_model(model)  # validate model exists
    return await call_ollama(
        model,
        prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@mcp.tool()
async def swarm(
    prompt: str,
    models: list[str] | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> SwarmResult:
    """Call multiple models in parallel and collect results.

    Args:
        prompt: The user prompt to send to all models.
        models: List of model IDs. Defaults to all 13 registered models.
        system_prompt: Optional system prompt for all models.
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens per model.
    """
    target_ids = models if models else list(MODELS.keys())
    for mid in target_ids:
        get_model(mid)  # validate all models exist up front

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _bounded(model_id: str) -> CallResult:
        async with sem:
            return await call_ollama(
                model_id,
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    start = time.monotonic()
    results = list(await asyncio.gather(*[_bounded(mid) for mid in target_ids]))
    total_elapsed = round(time.monotonic() - start, 2)

    succeeded = sum(1 for r in results if r.status == "ok")
    return SwarmResult(
        results=results,
        total_elapsed_seconds=total_elapsed,
        models_called=len(target_ids),
        models_succeeded=succeeded,
        models_failed=len(target_ids) - succeeded,
    )


@mcp.tool()
async def list_models() -> list[ModelAvailability]:
    """List all registered models with live Ollama availability check."""
    connected = await check_ollama()
    return [
        ModelAvailability(
            model_id=m.model_id,
            display_name=m.display_name,
            category=m.category,
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
