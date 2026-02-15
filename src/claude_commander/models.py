"""Pydantic result models returned by MCP tools."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CallResult(BaseModel):
    """Result of a single model call."""

    model: str
    content: str = ""
    thinking: str | None = None
    elapsed_seconds: float = 0.0
    status: str = "ok"
    error: str | None = None
    role_label: str = ""
    tags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


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
    strengths: list[str] = Field(default_factory=list)
    available: bool


class HealthStatus(BaseModel):
    """Server health and connectivity status."""

    server: str = "claude-commander"
    version: str
    ollama_url: str
    ollama_connected: bool
    registered_models: int


# ---------------------------------------------------------------------------
# Debate
# ---------------------------------------------------------------------------

class DebateRound(BaseModel):
    """A single round in a multi-round debate."""

    round_number: int
    model: str
    content: str


class DebateResult(BaseModel):
    """Full transcript of a multi-round debate between two models."""

    model_a: str
    model_b: str
    prompt: str
    rounds: list[DebateRound] = Field(default_factory=list)
    total_elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Vote
# ---------------------------------------------------------------------------

class Vote(BaseModel):
    """A single model's vote with extraction metadata."""

    model: str
    raw_response: str
    extracted_vote: str
    confidence: str


class VoteResult(BaseModel):
    """Aggregated voting results from multiple models."""

    prompt: str
    votes: list[Vote] = Field(default_factory=list)
    tally: dict[str, int] = Field(default_factory=dict)
    majority: str = ""
    total_models: int = 0
    agreement_pct: float = 0.0


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------

class ConsensusResult(BaseModel):
    """Parallel responses plus a judge-synthesized consensus."""

    prompt: str
    individual_responses: list[CallResult] = Field(default_factory=list)
    synthesis: str = ""
    judge_model: str = ""
    total_elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Code Review
# ---------------------------------------------------------------------------

class CodeReviewResult(BaseModel):
    """Parallel expert reviews merged into a single report."""

    code_snippet: str
    reviews: list[CallResult] = Field(default_factory=list)
    merged_review: str = ""
    reviewer_models: list[str] = Field(default_factory=list)
    merge_model: str = ""
    total_elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Multi-Solve
# ---------------------------------------------------------------------------

class MultiSolveResult(BaseModel):
    """Multiple independent solutions to a coding problem."""

    problem: str
    language: str | None = None
    solutions: list[CallResult] = Field(default_factory=list)
    total_elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class BenchmarkCell(BaseModel):
    """One cell in the prompt x model benchmark matrix."""

    model: str
    prompt_index: int
    content: str = ""
    elapsed_seconds: float = 0.0
    status: str = "ok"


class BenchmarkResult(BaseModel):
    """Full benchmark matrix with per-model statistics."""

    prompts: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    results: list[BenchmarkCell] = Field(default_factory=list)
    model_stats: dict[str, float] = Field(default_factory=dict)
    total_elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Rank
# ---------------------------------------------------------------------------

class ModelScore(BaseModel):
    """A single model's ranking entry with judge breakdown."""

    model: str
    avg_score: float = 0.0
    scores_received: dict[str, float] = Field(default_factory=dict)
    response: str = ""


class RankResult(BaseModel):
    """Peer-evaluated leaderboard of model responses."""

    prompt: str
    leaderboard: list[ModelScore] = Field(default_factory=list)
    total_elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

class ChainStep(BaseModel):
    """One step in a sequential model pipeline."""

    step: int
    model: str
    content: str = ""
    thinking: str | None = None
    elapsed_seconds: float = 0.0


class ChainResult(BaseModel):
    """Result of a sequential model pipeline."""

    prompt: str
    steps: list[ChainStep] = Field(default_factory=list)
    final_output: str = ""
    pipeline: list[str] = Field(default_factory=list)
    total_elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Map-Reduce
# ---------------------------------------------------------------------------

class MapReduceResult(BaseModel):
    """Fan-out / fan-in result with mapped responses and reduced output."""

    prompt: str
    mapped_responses: list[CallResult] = Field(default_factory=list)
    reduced_output: str = ""
    reducer_model: str = ""
    total_elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Blind Taste Test
# ---------------------------------------------------------------------------

class BlindResponse(BaseModel):
    """An anonymised model response (label hides identity)."""

    label: str
    content: str = ""


class BlindTasteResult(BaseModel):
    """Anonymised side-by-side comparison with reveal mapping."""

    prompt: str
    responses: list[BlindResponse] = Field(default_factory=list)
    reveal: dict[str, str] = Field(default_factory=dict)
    total_elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Contrarian
# ---------------------------------------------------------------------------

class ContrarianResult(BaseModel):
    """Thesis and substantive devil's-advocate antithesis."""

    prompt: str
    thesis_model: str = ""
    thesis: str = ""
    antithesis_model: str = ""
    antithesis: str = ""
    total_elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

class ProfileData(BaseModel):
    """Reusable model profile with all configuration options."""

    model: str
    role_label: str = ""
    system_prompt: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4096
    timeout_seconds: int = 120
    response_format: str | dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)
    description: str = ""
    parent: str = ""
    builtin: bool = False


class ProfileListResult(BaseModel):
    """List of all saved profiles."""

    profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    count: int = 0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PipelineData(BaseModel):
    """Saved pipeline definition — an ordered sequence of profile/model steps."""

    name: str
    description: str = ""
    steps: list[str] = Field(default_factory=list)
    builtin: bool = False


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


class ClaimVerification(BaseModel):
    """Per-claim verdict from cross-model fact verification."""

    claim: str
    verdict: str = "unverifiable"  # supported/unsupported/contradicted/unverifiable
    confidence: float = 0.0  # 0-1
    reasoning: str = ""
    sources_cited: list[str] = Field(default_factory=list)


class VerifyResult(BaseModel):
    """Cross-model fact verification result."""

    content_snippet: str
    claims: list[ClaimVerification] = Field(default_factory=list)
    overall_verdict: str = "mixed"  # pass/fail/mixed
    overall_confidence: float = 0.0
    internal_contradictions: list[str] = Field(default_factory=list)
    verifier_models: list[str] = Field(default_factory=list)
    total_elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Red Team
# ---------------------------------------------------------------------------


class RedTeamExchange(BaseModel):
    """One attack/defense round in adversarial stress testing."""

    round_number: int
    attack: str
    attack_model: str
    defense: str
    defense_model: str


class SurvivingIssue(BaseModel):
    """An unresolved issue from red-team testing."""

    issue: str
    severity: str = "minor"  # critical/major/minor
    first_raised_round: int = 1


class RedTeamResult(BaseModel):
    """Result of iterative adversarial stress testing."""

    content_snippet: str
    exchanges: list[RedTeamExchange] = Field(default_factory=list)
    surviving_issues: list[SurvivingIssue] = Field(default_factory=list)
    robustness_score: float = 5.0  # 1-10
    rounds_completed: int = 0
    converged: bool = False
    attacker_model: str = ""
    defender_model: str = ""
    total_elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Quality Gate
# ---------------------------------------------------------------------------


class CriterionScore(BaseModel):
    """Score for a single quality criterion."""

    criterion: str
    score: float = 5.0  # 1-10
    max_score: float = 10.0
    issues: list[str] = Field(default_factory=list)


class QualityGateResult(BaseModel):
    """Pass/fail evaluation against quality criteria."""

    content_snippet: str
    passed: bool = False
    overall_score: float = 0.0
    threshold: float = 7.0
    criteria_scores: list[CriterionScore] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    judge_model: str = ""
    total_elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Detect Slop
# ---------------------------------------------------------------------------


class SlopSignal(BaseModel):
    """A detected signal of low-quality AI-generated content."""

    signal_type: str  # filler_phrases, vague_generalities, etc.
    description: str = ""
    examples: list[str] = Field(default_factory=list)
    severity: str = "minor"  # critical/major/minor


class DetectSlopResult(BaseModel):
    """AI garbage detection result."""

    content_snippet: str
    slop_score: float = 0.0  # 0-10
    verdict: str = "clean"  # clean/mild/significant/severe
    signals: list[SlopSignal] = Field(default_factory=list)
    detector_models: list[str] = Field(default_factory=list)
    agreement_pct: float = 0.0
    total_elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Exec Task
# ---------------------------------------------------------------------------


class ExecTaskResult(BaseModel):
    """Result from delegating a coding task to a local CLI agent."""

    task: str
    agent: str
    working_dir: str = ""
    output: str = ""
    status: str = "ok"
    error: str = ""
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PipelineStepResult(BaseModel):
    """One step in a sequential pipeline."""

    step: int
    model: str
    role_label: str = ""
    content: str = ""
    thinking: str | None = None
    elapsed_seconds: float = 0.0


class PipelineResult(BaseModel):
    """Result of running a profile-based pipeline."""

    prompt: str
    steps: list[PipelineStepResult] = Field(default_factory=list)
    final_output: str = ""
    pipeline_name: str = ""
    total_elapsed_seconds: float = 0.0
