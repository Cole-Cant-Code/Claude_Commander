"""Pipeline storage with JSON persistence — mirrors ProfileStore pattern."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from claude_commander.models import PipelineData

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_DEFAULT_DIR = Path.home() / ".claude-commander" / "pipelines"


class PipelineStore:
    """In-memory pipeline store with disk persistence."""

    def __init__(self, persist_dir: Path | None = None) -> None:
        self._pipelines: dict[str, PipelineData] = {}
        self._persist_dir = persist_dir or _DEFAULT_DIR
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._load_all()

    def save(self, name: str, pipeline: PipelineData) -> None:
        if not _NAME_RE.match(name):
            raise ValueError(
                f"Invalid pipeline name '{name}'. "
                "Use only letters, digits, hyphens, and underscores."
            )
        self._pipelines[name] = pipeline
        self._persist(name, pipeline)

    def load(self, name: str) -> PipelineData | None:
        return self._pipelines.get(name)

    def delete(self, name: str) -> bool:
        pipeline = self._pipelines.get(name)
        if pipeline is None:
            return False
        if pipeline.builtin:
            raise ValueError(
                f"Cannot delete builtin pipeline '{name}'. Clone it instead."
            )
        del self._pipelines[name]
        path = self._persist_dir / f"{name}.json"
        path.unlink(missing_ok=True)
        return True

    def list_all(self) -> dict[str, dict[str, Any]]:
        return {name: p.model_dump() for name, p in sorted(self._pipelines.items())}

    def has_pipeline(self, name: str) -> bool:
        return name in self._pipelines

    def clone(self, source: str, target: str, **overrides: Any) -> PipelineData:
        """Clone an existing pipeline with optional field overrides."""
        src = self._pipelines.get(source)
        if src is None:
            raise ValueError(f"Source pipeline '{source}' not found.")
        if not _NAME_RE.match(target):
            raise ValueError(
                f"Invalid pipeline name '{target}'. "
                "Use only letters, digits, hyphens, and underscores."
            )
        data = src.model_dump()
        data["name"] = target
        data["builtin"] = False
        for key, val in overrides.items():
            if key in data:
                data[key] = val
        cloned = PipelineData(**data)
        self.save(target, cloned)
        return cloned

    def _persist(self, name: str, pipeline: PipelineData) -> None:
        try:
            path = self._persist_dir / f"{name}.json"
            path.write_text(json.dumps(pipeline.model_dump(), indent=2))
        except Exception:
            logger.exception("Failed to persist pipeline %s", name)

    def _load_all(self) -> None:
        if not self._persist_dir.exists():
            return
        for path in self._persist_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                self._pipelines[path.stem] = PipelineData(**data)
            except Exception:
                logger.warning("Skipping corrupt pipeline file: %s", path)


# Module-level singleton
_store: PipelineStore | None = None


def get_pipeline_store() -> PipelineStore:
    """Return the singleton PipelineStore instance."""
    global _store
    if _store is None:
        _store = PipelineStore()
    return _store
