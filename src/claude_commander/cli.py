"""Async CLI subprocess caller — runs AI CLI tools non-interactively."""

from __future__ import annotations

import asyncio
import json
import time

from claude_commander.models import CallResult
from claude_commander.registry import MODELS


async def call_cli(
    model: str,
    prompt: str,
    *,
    system_prompt: str | None = None,
    timeout_seconds: int = 120,
    role_label: str = "",
    tags: list[str] | None = None,
    cwd: str | None = None,
    **_kwargs: object,
) -> CallResult:
    """Run a CLI model as a subprocess and capture its output.

    The CLI command template is stored in the model registry. The ``{prompt}``
    placeholder is replaced with the actual prompt text.  Extra kwargs
    (temperature, top_p, etc.) are accepted but silently ignored — CLI tools
    don't expose those knobs.
    """
    info = MODELS.get(model)
    if info is None or not info.is_cli:
        return CallResult(
            model=model,
            status="error",
            error=f"Model '{model}' is not a registered CLI model.",
            role_label=role_label,
            tags=tags or [],
        )

    # Build the full prompt: prepend system prompt if provided
    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"

    # Substitute {prompt} in the command template
    cmd = [part.replace("{prompt}", full_prompt) for part in info.cli_command]

    start = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_seconds
        )
        elapsed = round(time.monotonic() - start, 2)

        output = stdout.decode("utf-8", errors="replace").strip()

        # Codex --json returns JSONL events; extract the final message
        if model == "codex:cli" and output:
            output = _extract_codex_output(output)

        if proc.returncode != 0:
            err_text = stderr.decode("utf-8", errors="replace").strip()
            return CallResult(
                model=model,
                content=output,
                status="error",
                error=f"CLI exited with code {proc.returncode}: {err_text}",
                elapsed_seconds=elapsed,
                role_label=role_label,
                tags=tags or [],
            )

        warnings: list[str] = []
        if not output:
            warnings.append(f"CLI model {model} returned empty output after {elapsed}s")

        return CallResult(
            model=model,
            content=output,
            elapsed_seconds=elapsed,
            role_label=role_label,
            tags=tags or [],
            warnings=warnings,
        )

    except asyncio.TimeoutError:
        elapsed = round(time.monotonic() - start, 2)
        # Try to kill the timed-out process
        try:
            proc.kill()  # type: ignore[possibly-undefined]
        except Exception:
            pass
        return CallResult(
            model=model,
            status="error",
            error=f"CLI timed out after {timeout_seconds}s",
            elapsed_seconds=elapsed,
            role_label=role_label,
            tags=tags or [],
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


def _extract_codex_output(raw: str) -> str:
    """Extract the final assistant message from Codex JSONL output.

    Codex ``exec --json`` emits newline-delimited JSON events. We look for
    the last ``message`` event with ``role: assistant`` and return its text
    content, falling back to the raw output if parsing fails.
    """
    last_message = ""
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Codex events have varying shapes; look for assistant content
        if isinstance(event, dict):
            role = event.get("role", "")
            if role == "assistant":
                content = event.get("content", "")
                if content:
                    last_message = content
            # Also check nested message structure
            msg = event.get("message", {})
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content:
                    last_message = content
    return last_message or raw
