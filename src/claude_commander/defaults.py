"""Seed profiles and pipelines — builtin defaults for Claude Commander."""

from __future__ import annotations

from typing import TYPE_CHECKING

from claude_commander.models import PipelineData, ProfileData

if TYPE_CHECKING:
    from claude_commander.pipeline_store import PipelineStore
    from claude_commander.profile_store import ProfileStore


# ---------------------------------------------------------------------------
# Builtin profiles
# ---------------------------------------------------------------------------

_BUILTIN_PROFILES: dict[str, ProfileData] = {
    "fast-general": ProfileData(
        model="glm-4.7:cloud",
        temperature=0.7,
        description="Fast general-purpose model for quick tasks",
        builtin=True,
    ),
    "deep-reasoner": ProfileData(
        model="deepseek-v3.2:cloud",
        temperature=0.3,
        description="Extended chain-of-thought for hard reasoning problems",
        builtin=True,
    ),
    "code-specialist": ProfileData(
        model="qwen3-coder-next:cloud",
        temperature=0.2,
        description="Code generation, architecture, and debugging",
        builtin=True,
    ),
    "thinking-judge": ProfileData(
        model="kimi-k2-thinking:cloud",
        temperature=0.4,
        description="Thinking model for evaluation and judging tasks",
        builtin=True,
    ),
    "creative-writer": ProfileData(
        model="glm-5:cloud",
        temperature=0.9,
        description="High-temperature creative writing and brainstorming",
        builtin=True,
    ),
    "factual-analyst": ProfileData(
        model="gpt-oss:120b-cloud",
        temperature=0.3,
        description="Precise factual analysis with broad coverage",
        builtin=True,
    ),
    "vision-analyzer": ProfileData(
        model="qwen3-vl:235b-cloud",
        temperature=0.5,
        description="Multimodal visual reasoning and image analysis",
        builtin=True,
    ),
    "quick-draft": ProfileData(
        model="gpt-oss:20b-cloud",
        temperature=0.8,
        description="Lightweight fast drafting — low latency, good enough",
        builtin=True,
    ),
    "strict-verifier": ProfileData(
        model="deepseek-v3.2:cloud",
        temperature=0.2,
        system_prompt=(
            "You are a rigorous fact-checker. Verify every claim against known facts. "
            "Flag unsupported assertions, find errors, and cite sources when possible."
        ),
        description="Rigorous fact-checking, finding errors",
        builtin=True,
    ),
    "adversarial-attacker": ProfileData(
        model="deepseek-v3.2:cloud",
        temperature=0.3,
        system_prompt=(
            "You are an adversarial red-teamer. Find weaknesses, flaws, edge cases, "
            "and potential failures. Be specific and escalate with each round."
        ),
        description="Finding weaknesses, flaws, edge cases",
        builtin=True,
    ),
    "quality-judge": ProfileData(
        model="kimi-k2-thinking:cloud",
        temperature=0.2,
        system_prompt=(
            "You are a quality evaluator. Provide structured scoring against criteria. "
            "Be precise with scores and specific about issues found."
        ),
        description="Structured quality evaluation, scoring",
        builtin=True,
    ),
    "slop-detector": ProfileData(
        model="gpt-oss:120b-cloud",
        temperature=0.2,
        system_prompt=(
            "You are an AI content quality detector. Identify generic filler phrases, "
            "vague generalities, hallucinated citations, and confident unsupported claims."
        ),
        description="Detecting generic filler, hallucinated citations",
        builtin=True,
    ),
}


# ---------------------------------------------------------------------------
# Builtin pipelines
# ---------------------------------------------------------------------------

_BUILTIN_PIPELINES: dict[str, PipelineData] = {
    "draft-then-refine": PipelineData(
        name="draft-then-refine",
        description="Quick draft followed by deep reasoning refinement",
        steps=["quick-draft", "deep-reasoner"],
        builtin=True,
    ),
    "code-review-pipeline": PipelineData(
        name="code-review-pipeline",
        description="Code specialist writes, reasoner reviews, analyst validates",
        steps=["code-specialist", "deep-reasoner", "factual-analyst"],
        builtin=True,
    ),
    "creative-to-critical": PipelineData(
        name="creative-to-critical",
        description="Creative brainstorm followed by critical thinking evaluation",
        steps=["creative-writer", "thinking-judge"],
        builtin=True,
    ),
    "verify-then-refine": PipelineData(
        name="verify-then-refine",
        description="Generate, verify, fix issues",
        steps=["quick-draft", "strict-verifier", "deep-reasoner"],
        builtin=True,
    ),
    "red-team-then-harden": PipelineData(
        name="red-team-then-harden",
        description="Generate, stress-test, harden",
        steps=["deep-reasoner", "adversarial-attacker", "thinking-judge"],
        builtin=True,
    ),
    "full-quality-check": PipelineData(
        name="full-quality-check",
        description="Triple-check existing content",
        steps=["strict-verifier", "slop-detector", "quality-judge"],
        builtin=True,
    ),
}


# ---------------------------------------------------------------------------
# Seeding functions
# ---------------------------------------------------------------------------


def seed_profiles(store: ProfileStore) -> None:
    """Populate the store with builtin profiles (idempotent)."""
    for name, profile in _BUILTIN_PROFILES.items():
        if not store.has_profile(name):
            store.save(name, profile)


def seed_pipelines(store: PipelineStore) -> None:
    """Populate the store with builtin pipelines (idempotent)."""
    for name, pipeline in _BUILTIN_PIPELINES.items():
        if not store.has_pipeline(name):
            store.save(name, pipeline)
