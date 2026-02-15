"""Async Ollama HTTP client using aiohttp."""

from __future__ import annotations

import os
import time
from typing import Any

import aiohttp

from claude_commander.models import CallResult
from claude_commander.registry import MODELS

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


async def call_ollama(
    model: str,
    prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 4096,
    timeout_seconds: int = 120,
    response_format: str | dict[str, Any] | None = None,
    role_label: str = "",
    tags: list[str] | None = None,
) -> CallResult:
    """Send a single chat completion request to Ollama."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    else:
        info = MODELS.get(model)
        name = info.display_name if info else model
        messages.append({
            "role": "system",
            "content": f"You are {name}. Answer the user's question directly.",
        })
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "messages": messages,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
        },
    }
    if response_format is not None:
        payload["format"] = response_format

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

        msg = raw.get("message", {})
        content = msg.get("content", "")
        thinking = msg.get("thinking", None)
        elapsed = round(time.monotonic() - start, 2)

        warnings: list[str] = []
        if not content and resp.status == 200:
            warnings.append(
                f"Model {model} returned empty content "
                f"(possible reasoning-token exhaustion). "
                f"Elapsed: {elapsed}s, max_tokens: {max_tokens}"
            )

        return CallResult(
            model=model,
            content=content,
            thinking=thinking,
            elapsed_seconds=elapsed,
            role_label=role_label,
            tags=tags or [],
            warnings=warnings,
        )

    except Exception as exc:
        elapsed = round(time.monotonic() - start, 2)
        return CallResult(
            model=model,
            status="error",
            error=str(exc),
            elapsed_seconds=elapsed,
            role_label=role_label,
            tags=tags or [],
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
