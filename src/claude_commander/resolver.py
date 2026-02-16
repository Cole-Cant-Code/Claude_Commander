"""Profile resolver — translates profile names to model IDs + params."""

from __future__ import annotations

from typing import Any

from claude_commander.profile_store import get_profile_store

# Sentinel: "caller did not provide this kwarg"
_UNSET = object()


def resolve(name_or_model: str) -> tuple[str, dict[str, Any]]:
    """Resolve a profile name or raw model ID into (model_id, params).

    If *name_or_model* matches a saved profile, returns the profile's model
    and all its parameters as a dict.  Otherwise treats it as a raw model ID
    and returns it with an empty param dict (the caller's own defaults apply).
    """
    store = get_profile_store()
    profile = store.load(name_or_model)
    if profile is not None:
        params = profile.model_dump()
        model_id = params.pop("model")
        # Strip metadata fields — not relevant for call_ollama
        for key in ("description", "parent", "builtin", "tags"):
            params.pop(key, None)
        return model_id, params

    # Not a profile — treat as raw model ID (validation happens downstream)
    return name_or_model, {}


def merge_overrides(profile_params: dict[str, Any], **caller_kwargs: Any) -> dict[str, Any]:
    """Merge caller-provided kwargs over profile defaults.

    Only keys where the caller actually provided a value (not ``_UNSET``)
    override the profile default.  This lets tools distinguish between
    "caller passed temperature=0.5" and "caller didn't mention temperature".
    """
    merged = dict(profile_params)
    for key, val in caller_kwargs.items():
        if val is not _UNSET:
            merged[key] = val
    return merged
