"""Async Ollama HTTP client using aiohttp."""

from __future__ import annotations

import os
import time

import aiohttp

from claude_commander.models import CallResult

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


async def call_ollama(
    model: str,
    prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    timeout_seconds: int = 120,
) -> CallResult:
    """Send a single chat completion request to Ollama."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "stream": False,
        "messages": messages,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    start = time.monotonic()
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_seconds),
            ) as resp,
        ):
            raw = await resp.json()

        content = raw.get("message", {}).get("content", "")
        elapsed = round(time.monotonic() - start, 2)
        return CallResult(model=model, content=content, elapsed_seconds=elapsed)

    except Exception as exc:
        elapsed = round(time.monotonic() - start, 2)
        return CallResult(
            model=model, status="error", error=str(exc), elapsed_seconds=elapsed
        )


async def check_ollama() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(
                OLLAMA_BASE_URL,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp,
        ):
            return resp.status == 200
    except Exception:
        return False
