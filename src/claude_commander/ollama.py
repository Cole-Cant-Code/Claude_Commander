"""Async Ollama HTTP client using aiohttp."""

from __future__ import annotations

import os
import time
from typing import Any

import aiohttp

from claude_commander.models import CallResult
from claude_commander.registry import MODELS

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Some Ollama-compatible backends emit "thinking" tokens that can consume the
# entire `num_predict` budget, leaving an empty `message.content`. To keep MCP
# semantics reliable, we do a single automatic retry with a larger `num_predict`
# when the backend reports thinking but returns empty content on HTTP 200.
_EMPTY_CONTENT_RETRY_ATTEMPTS = 2
_EMPTY_CONTENT_RETRY_BUMP = 256
_EMPTY_CONTENT_RETRY_MIN_PREDICT = 128

# Proactive floor for thinking models — applied *before* the first attempt
# to avoid the empty-content-then-retry dance entirely.
_THINKING_MODEL_MIN_PREDICT = 256


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

    start = time.monotonic()
    try:
        warnings: list[str] = []
        content = ""
        thinking = None
        last_status = 0
        last_num_predict = max_tokens

        # Enforce a minimum token budget for thinking models so their
        # chain-of-thought doesn't consume the entire num_predict.
        info = MODELS.get(model)
        if info and info.is_thinking and max_tokens < _THINKING_MODEL_MIN_PREDICT:
            warnings.append(
                f"max_tokens={max_tokens} is below the {_THINKING_MODEL_MIN_PREDICT} "
                f"floor for thinking model {model}; raised automatically."
            )
            max_tokens = _THINKING_MODEL_MIN_PREDICT

        async with aiohttp.ClientSession() as session:
            for attempt in range(1, _EMPTY_CONTENT_RETRY_ATTEMPTS + 1):
                num_predict = max_tokens
                if attempt > 1:
                    # Give the backend enough room to emit internal reasoning plus
                    # at least a small visible completion.
                    num_predict = max(
                        _EMPTY_CONTENT_RETRY_MIN_PREDICT,
                        max_tokens + _EMPTY_CONTENT_RETRY_BUMP,
                    )

                payload_messages = messages
                if attempt > 1 and messages and messages[0].get("role") == "system":
                    # Nudge the backend away from emitting "thinking" and towards
                    # producing a visible completion.
                    payload_messages = [
                        {
                            "role": "system",
                            "content": (
                                messages[0].get("content", "")
                                + "\n\nIMPORTANT: Reply with the final answer only. "
                                "Do not include internal reasoning. Follow the user's format strictly."
                            ),
                        },
                        *messages[1:],
                    ]

                payload: dict[str, Any] = {
                    "model": model,
                    "stream": False,
                    "messages": payload_messages,
                    "options": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": num_predict,
                    },
                }
                if response_format is not None:
                    payload["format"] = response_format

                async with session.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds),
                ) as resp:
                    raw = await resp.json()
                    last_status = resp.status

                last_num_predict = num_predict
                msg = raw.get("message", {})
                content = msg.get("content", "") or ""
                thinking = msg.get("thinking", None)

                if content:
                    break

                if last_status == 200:
                    warnings.append(
                        f"Model {model} returned empty content "
                        f"(possible reasoning-token exhaustion). "
                        f"Attempt {attempt}/{_EMPTY_CONTENT_RETRY_ATTEMPTS}, "
                        f"num_predict: {num_predict}"
                    )

                # Only retry if the backend indicates it produced thinking/reasoning.
                if attempt >= _EMPTY_CONTENT_RETRY_ATTEMPTS or not thinking:
                    break

        elapsed = round(time.monotonic() - start, 2)

        if not content and last_status == 200 and not warnings:
            warnings.append(
                f"Model {model} returned empty content "
                f"(possible reasoning-token exhaustion). "
                f"Elapsed: {elapsed}s, max_tokens: {max_tokens}, "
                f"num_predict: {last_num_predict}"
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
