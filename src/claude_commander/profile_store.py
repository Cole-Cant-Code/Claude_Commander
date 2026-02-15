"""Reusable profile storage with JSON persistence."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from claude_commander.models import ProfileData

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_DEFAULT_DIR = Path.home() / ".claude-commander" / "profiles"


class ProfileStore:
    """In-memory profile store with disk persistence."""

    def __init__(self, persist_dir: Path | None = None) -> None:
        self._profiles: dict[str, ProfileData] = {}
        self._persist_dir = persist_dir or _DEFAULT_DIR
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._load_all()

    def save(self, name: str, profile: ProfileData) -> None:
        if not _NAME_RE.match(name):
            raise ValueError(
                f"Invalid profile name '{name}'. "
                "Use only letters, digits, hyphens, and underscores."
            )
        self._profiles[name] = profile
        self._persist(name, profile)

    def load(self, name: str) -> ProfileData | None:
        return self._profiles.get(name)

    def delete(self, name: str) -> bool:
        if name not in self._profiles:
            return False
        del self._profiles[name]
        path = self._persist_dir / f"{name}.json"
        path.unlink(missing_ok=True)
        return True

    def list_all(self) -> dict[str, dict[str, Any]]:
        return {name: p.model_dump() for name, p in sorted(self._profiles.items())}

    def _persist(self, name: str, profile: ProfileData) -> None:
        try:
            path = self._persist_dir / f"{name}.json"
            path.write_text(json.dumps(profile.model_dump(), indent=2))
        except Exception:
            logger.exception("Failed to persist profile %s", name)

    def _load_all(self) -> None:
        if not self._persist_dir.exists():
            return
        for path in self._persist_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                self._profiles[path.stem] = ProfileData(**data)
            except Exception:
                logger.warning("Skipping corrupt profile file: %s", path)


# Module-level singleton
_store: ProfileStore | None = None


def get_profile_store() -> ProfileStore:
    """Return the singleton ProfileStore instance."""
    global _store
    if _store is None:
        _store = ProfileStore()
    return _store
