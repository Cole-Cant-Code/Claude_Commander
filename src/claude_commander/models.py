"""Pydantic result models returned by MCP tools."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CallResult(BaseModel):
    """Result of a single model call."""

    model: str
    content: str = ""
    elapsed_seconds: float = 0.0
    status: str = "ok"
    error: str | None = None


class SwarmResult(BaseModel):
    """Aggregated results from a parallel swarm call."""

    results: list[CallResult] = Field(default_factory=list)
    total_elapsed_seconds: float = 0.0
    models_called: int = 0
    models_succeeded: int = 0
    models_failed: int = 0


class ModelAvailability(BaseModel):
    """Single model's registry info + live availability."""

    model_id: str
    display_name: str
    category: str
    available: bool


class HealthStatus(BaseModel):
    """Server health and connectivity status."""

    server: str = "claude-commander"
    version: str
    ollama_url: str
    ollama_connected: bool
    registered_models: int
