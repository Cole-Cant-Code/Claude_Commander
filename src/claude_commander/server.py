"""Claude Commander MCP server — 15 tools for Ollama model orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
import time

from fastmcp import FastMCP

from claude_commander import __version__
from claude_commander.models import (
    BenchmarkCell,
    BenchmarkResult,
    BlindResponse,
    BlindTasteResult,
    CallResult,
    ChainResult,
    ChainStep,
    CodeReviewResult,
    ConsensusResult,
    ContrarianResult,
    DebateResult,
    DebateRound,
    HealthStatus,
    MapReduceResult,
    ModelAvailability,
    ModelScore,
    MultiSolveResult,
    RankResult,
    SwarmResult,
    Vote,
    VoteResult,
)
from claude_commander.ollama import OLLAMA_BASE_URL, call_ollama, check_ollama
from claude_commander.registry import MODELS, get_model

mcp = FastMCP("Claude Commander")

MAX_CONCURRENCY = 13

# ---------------------------------------------------------------------------
# Role constants
# ---------------------------------------------------------------------------

DEFAULT_JUDGE = "kimi-k2-thinking:cloud"
DEFAULT_DEBATE = ("deepseek-v3.2:cloud", "glm-5:cloud")
CODE_MODELS = ["qwen3-coder-next:cloud", "deepseek-v3.2:cloud", "gpt-oss:120b-cloud"]
RANK_DEFAULT_MODELS = [
    "deepseek-v3.2:cloud",
    "qwen3-coder-next:cloud",
    "glm-5:cloud",
    "kimi-k2.5:cloud",
    "minimax-m2.5:cloud",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _extract_vote(response: str, options: list[str]) -> tuple[str, str]:
    """Parse a model response to extract a vote from the given options.

    Strategy cascade:
    1. First-word exact match (case-insensitive)
    2. Phrase patterns like "I vote X", "my answer is X", "I choose X"
    3. Occurrence counting — pick the option mentioned most
    4. Fallback: "abstain" with low confidence
    """
    lower = response.lower().strip()
    options_lower = [o.lower() for o in options]

    # Strategy 1: first word match (strip trailing punctuation)
    first_word = re.sub(r"[^a-z0-9]", "", lower.split()[0]) if lower else ""
    for i, opt in enumerate(options_lower):
        if first_word == opt:
            return options[i], "high"

    # Strategy 2: phrase patterns
    for i, opt in enumerate(options_lower):
        patterns = [
            rf"\bi vote {re.escape(opt)}\b",
            rf"\bmy answer is {re.escape(opt)}\b",
            rf"\bi choose {re.escape(opt)}\b",
            rf"\bi pick {re.escape(opt)}\b",
            rf"\bthe answer is {re.escape(opt)}\b",
        ]
        for pat in patterns:
            if re.search(pat, lower):
                return options[i], "medium"

    # Strategy 3: occurrence counting
    counts = []
    for i, opt in enumerate(options_lower):
        counts.append((lower.count(opt), i))
    counts.sort(key=lambda x: x[0], reverse=True)
    if counts and counts[0][0] > 0:
        # Only use if there's a clear winner
        if len(counts) == 1 or counts[0][0] > counts[1][0]:
            return options[counts[0][1]], "low"

    # Strategy 4: abstain
    return "abstain", "none"


def _extract_score(response: str) -> float:
    """Parse a judge response to extract a numeric score (1-10).

    Regex cascade:
    1. "X/10" pattern
    2. "Score: X" pattern
    3. "Rating: X" pattern
    4. "X out of 10" pattern
    5. Any standalone number 1-10
    6. Default: 5.0
    """
    # Pattern 1: X/10
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", response)
    if m:
        return min(10.0, max(1.0, float(m.group(1))))

    # Pattern 2: Score: X
    m = re.search(r"[Ss]core:\s*(\d+(?:\.\d+)?)", response)
    if m:
        return min(10.0, max(1.0, float(m.group(1))))

    # Pattern 3: Rating: X
    m = re.search(r"[Rr]ating:\s*(\d+(?:\.\d+)?)", response)
    if m:
        return min(10.0, max(1.0, float(m.group(1))))

    # Pattern 4: X out of 10
    m = re.search(r"(\d+(?:\.\d+)?)\s+out\s+of\s+10", response)
    if m:
        return min(10.0, max(1.0, float(m.group(1))))

    # Pattern 5: any standalone 1-10
    m = re.search(r"\b(\d+(?:\.\d+)?)\b", response)
    if m:
        val = float(m.group(1))
        if 1.0 <= val <= 10.0:
            return val

    return 5.0


def _pick_models_by_category(categories: list[str]) -> list[str]:
    """Return model IDs whose category is in the given list."""
    return [mid for mid, info in MODELS.items() if info.category in categories]


# ---------------------------------------------------------------------------
# Original 4 tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def call_model(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> CallResult:
    """Call one model. Use list_models to see valid IDs."""
    get_model(model)  # validate model exists
    return await call_ollama(
        model,
        prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@mcp.tool()
async def swarm(
    prompt: str,
    models: list[str] | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> SwarmResult:
    """Call multiple models in parallel. Defaults to all 13."""
    target_ids = models if models else list(MODELS.keys())
    for mid in target_ids:
        get_model(mid)  # validate all models exist up front

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _bounded(model_id: str) -> CallResult:
        async with sem:
            return await call_ollama(
                model_id,
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    start = time.monotonic()
    results = list(await asyncio.gather(*[_bounded(mid) for mid in target_ids]))
    total_elapsed = round(time.monotonic() - start, 2)

    succeeded = sum(1 for r in results if r.status == "ok")
    return SwarmResult(
        results=results,
        total_elapsed_seconds=total_elapsed,
        models_called=len(target_ids),
        models_succeeded=succeeded,
        models_failed=len(target_ids) - succeeded,
    )


@mcp.tool()
async def list_models() -> list[ModelAvailability]:
    """List registered models with availability status."""
    connected = await check_ollama()
    return [
        ModelAvailability(
            model_id=m.model_id,
            display_name=m.display_name,
            category=m.category,
            available=connected,
        )
        for m in MODELS.values()
    ]


@mcp.tool()
async def health_check() -> HealthStatus:
    """Check server health and Ollama connectivity."""
    connected = await check_ollama()
    return HealthStatus(
        version=__version__,
        ollama_url=OLLAMA_BASE_URL,
        ollama_connected=connected,
        registered_models=len(MODELS),
    )


# ---------------------------------------------------------------------------
# Tool 1: debate
# ---------------------------------------------------------------------------


@mcp.tool()
async def debate(
    prompt: str,
    model_a: str | None = None,
    model_b: str | None = None,
    rounds: int = 3,
) -> DebateResult:
    """Multi-round debate. Models alternate, each seeing full transcript."""
    a = model_a or DEFAULT_DEBATE[0]
    b = model_b or DEFAULT_DEBATE[1]
    get_model(a)
    get_model(b)

    debate_rounds: list[DebateRound] = []
    transcript = f"Debate topic: {prompt}\n\n"

    start = time.monotonic()
    for i in range(1, rounds + 1):
        current_model = a if i % 2 == 1 else b
        other_model = b if i % 2 == 1 else a

        system = (
            f"You are in a structured debate (round {i}/{rounds}). "
            f"You are arguing your perspective on the topic. "
            f"Your opponent is {other_model}. Be substantive and direct."
        )

        result = await call_ollama(current_model, transcript, system_prompt=system)
        debate_rounds.append(
            DebateRound(round_number=i, model=current_model, content=result.content)
        )
        transcript += f"--- Round {i} ({current_model}) ---\n{result.content}\n\n"

    total_elapsed = round(time.monotonic() - start, 2)
    return DebateResult(
        model_a=a,
        model_b=b,
        prompt=prompt,
        rounds=debate_rounds,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Tool 2: vote
# ---------------------------------------------------------------------------


@mcp.tool()
async def vote(
    prompt: str,
    options: list[str] | None = None,
    models: list[str] | None = None,
) -> VoteResult:
    """Models vote on a question. Returns tally, majority, agreement %."""
    opts = options or ["yes", "no"]
    target_ids = models if models else list(MODELS.keys())
    for mid in target_ids:
        get_model(mid)

    options_str = ", ".join(opts)
    system = (
        f"You must vote on the following question. The valid options are: {options_str}. "
        f"Start your response with your chosen option word, then explain briefly."
    )

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _bounded(model_id: str) -> CallResult:
        async with sem:
            return await call_ollama(model_id, prompt, system_prompt=system)

    start = time.monotonic()
    results = list(await asyncio.gather(*[_bounded(mid) for mid in target_ids]))

    votes: list[Vote] = []
    tally: dict[str, int] = {o: 0 for o in opts}
    tally["abstain"] = 0

    for r in results:
        extracted, confidence = _extract_vote(r.content, opts)
        votes.append(
            Vote(
                model=r.model,
                raw_response=r.content,
                extracted_vote=extracted,
                confidence=confidence,
            )
        )
        if extracted in tally:
            tally[extracted] += 1
        else:
            tally["abstain"] += 1

    # Remove abstain key if no abstentions
    if tally["abstain"] == 0:
        del tally["abstain"]

    majority = max(tally, key=lambda k: tally[k]) if tally else ""
    majority_count = tally.get(majority, 0)
    agreement_pct = round(majority_count / len(target_ids) * 100, 1) if target_ids else 0.0

    total_elapsed = round(time.monotonic() - start, 2)
    return VoteResult(
        prompt=prompt,
        votes=votes,
        tally=tally,
        majority=majority,
        total_models=len(target_ids),
        agreement_pct=agreement_pct,
    )


# ---------------------------------------------------------------------------
# Tool 3: consensus
# ---------------------------------------------------------------------------


@mcp.tool()
async def consensus(
    prompt: str,
    models: list[str] | None = None,
    judge_model: str | None = None,
) -> ConsensusResult:
    """Swarm then judge-synthesize into a unified answer."""
    judge = judge_model or DEFAULT_JUDGE
    target_ids = models if models else list(MODELS.keys())
    for mid in target_ids:
        get_model(mid)
    get_model(judge)

    start = time.monotonic()

    # Phase 1: parallel swarm
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _bounded(model_id: str) -> CallResult:
        async with sem:
            return await call_ollama(model_id, prompt)

    responses = list(await asyncio.gather(*[_bounded(mid) for mid in target_ids]))

    # Phase 2: synthesis
    response_block = "\n\n".join(
        f"[{r.model}]: {r.content}" for r in responses if r.status == "ok"
    )
    synthesis_prompt = (
        f"Original question: {prompt}\n\n"
        f"The following models provided responses:\n\n{response_block}\n\n"
        f"Identify areas of agreement and disagreement among these responses. "
        f"Then produce a single, unified answer that captures the consensus."
    )
    synthesis_result = await call_ollama(judge, synthesis_prompt)

    total_elapsed = round(time.monotonic() - start, 2)
    return ConsensusResult(
        prompt=prompt,
        individual_responses=responses,
        synthesis=synthesis_result.content,
        judge_model=judge,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Tool 4: code_review
# ---------------------------------------------------------------------------


@mcp.tool()
async def code_review(
    code: str,
    language: str | None = None,
    review_models: list[str] | None = None,
    merge_model: str | None = None,
) -> CodeReviewResult:
    """Parallel code reviews merged and sorted by severity."""
    reviewers = review_models or CODE_MODELS
    merger = merge_model or DEFAULT_JUDGE
    for mid in reviewers:
        get_model(mid)
    get_model(merger)

    lang_hint = f" ({language})" if language else ""
    system = (
        f"You are an expert code reviewer. Review the following{lang_hint} code for: "
        f"bugs, security issues, performance problems, and readability. "
        f"Reference line numbers where applicable. Suggest specific fixes."
    )

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _bounded(model_id: str) -> CallResult:
        async with sem:
            return await call_ollama(model_id, code, system_prompt=system)

    start = time.monotonic()
    reviews = list(await asyncio.gather(*[_bounded(mid) for mid in reviewers]))

    # Merge phase
    review_block = "\n\n".join(
        f"[Review by {r.model}]:\n{r.content}" for r in reviews if r.status == "ok"
    )
    merge_prompt = (
        f"The following independent code reviews were produced for this code:\n\n"
        f"{review_block}\n\n"
        f"Deduplicate the findings and produce a single merged review. "
        f"Sort issues by severity (critical -> major -> minor). "
        f"Preserve specific fix suggestions."
    )
    merge_result = await call_ollama(merger, merge_prompt)

    total_elapsed = round(time.monotonic() - start, 2)
    return CodeReviewResult(
        code_snippet=code[:500],  # truncate for result readability
        reviews=reviews,
        merged_review=merge_result.content,
        reviewer_models=reviewers,
        merge_model=merger,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Tool 5: multi_solve
# ---------------------------------------------------------------------------


@mcp.tool()
async def multi_solve(
    problem: str,
    language: str | None = None,
    models: list[str] | None = None,
) -> MultiSolveResult:
    """Multiple models solve the same problem independently."""
    target_ids = models or (
        _pick_models_by_category(["code", "reasoning"])
        + ["qwen3-next:80b-cloud", "gpt-oss:120b-cloud"]
    )
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for mid in target_ids:
        if mid not in seen:
            seen.add(mid)
            deduped.append(mid)
    target_ids = deduped

    for mid in target_ids:
        get_model(mid)

    lang_instruction = f" Write the solution in {language}." if language else ""
    system = (
        f"Write a complete, working solution to the given problem.{lang_instruction} "
        f"Include brief comments explaining your approach."
    )

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _bounded(model_id: str) -> CallResult:
        async with sem:
            return await call_ollama(model_id, problem, system_prompt=system)

    start = time.monotonic()
    solutions = list(await asyncio.gather(*[_bounded(mid) for mid in target_ids]))
    total_elapsed = round(time.monotonic() - start, 2)

    return MultiSolveResult(
        problem=problem,
        language=language,
        solutions=solutions,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Tool 6: benchmark
# ---------------------------------------------------------------------------


@mcp.tool()
async def benchmark(
    prompts: list[str],
    models: list[str] | None = None,
) -> BenchmarkResult:
    """Prompt x model matrix. Returns per-model latency stats."""
    target_ids = models if models else list(MODELS.keys())
    for mid in target_ids:
        get_model(mid)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _run_cell(model_id: str, prompt_idx: int) -> BenchmarkCell:
        async with sem:
            result = await call_ollama(model_id, prompts[prompt_idx])
            return BenchmarkCell(
                model=model_id,
                prompt_index=prompt_idx,
                content=result.content,
                elapsed_seconds=result.elapsed_seconds,
                status=result.status,
            )

    start = time.monotonic()
    tasks = [
        _run_cell(mid, pi)
        for pi in range(len(prompts))
        for mid in target_ids
    ]
    cells = list(await asyncio.gather(*tasks))
    total_elapsed = round(time.monotonic() - start, 2)

    # Compute per-model average latency
    model_times: dict[str, list[float]] = {mid: [] for mid in target_ids}
    for cell in cells:
        if cell.status == "ok":
            model_times[cell.model].append(cell.elapsed_seconds)
    model_stats = {
        mid: round(sum(times) / len(times), 2) if times else 0.0
        for mid, times in model_times.items()
    }

    return BenchmarkResult(
        prompts=prompts,
        models=target_ids,
        results=cells,
        model_stats=model_stats,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Tool 7: rank
# ---------------------------------------------------------------------------


@mcp.tool()
async def rank(
    prompt: str,
    models: list[str] | None = None,
    judge_count: int = 3,
) -> RankResult:
    """Models answer, then peer-judge each other 1-10. Returns leaderboard."""
    target_ids = models or RANK_DEFAULT_MODELS
    for mid in target_ids:
        get_model(mid)

    start = time.monotonic()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    # Phase 1: all models answer
    async def _answer(model_id: str) -> CallResult:
        async with sem:
            return await call_ollama(model_id, prompt)

    answers = list(await asyncio.gather(*[_answer(mid) for mid in target_ids]))

    # Phase 2: peer evaluation
    judge_system = (
        "Rate the following response on a scale of 1 to 10. "
        "Start your reply with 'Score: X/10' then explain briefly."
    )

    async def _judge(judge_id: str, answer: CallResult) -> tuple[str, str, float]:
        """Returns (target_model, judge_model, score)."""
        async with sem:
            judge_prompt = (
                f"Question: {prompt}\n\n"
                f"Response by {answer.model}:\n{answer.content}"
            )
            result = await call_ollama(judge_id, judge_prompt, system_prompt=judge_system)
            score = _extract_score(result.content)
            return answer.model, judge_id, score

    judge_tasks = []
    rng = random.Random(hash(prompt))
    for answer in answers:
        # Pick judges that are NOT the answer's own model
        candidates = [mid for mid in target_ids if mid != answer.model]
        actual_count = min(judge_count, len(candidates))
        judges = rng.sample(candidates, actual_count) if candidates else []
        for jid in judges:
            judge_tasks.append(_judge(jid, answer))

    judge_results = list(await asyncio.gather(*judge_tasks))

    # Aggregate scores
    scores_map: dict[str, dict[str, float]] = {mid: {} for mid in target_ids}
    for target_model, judge_model, score in judge_results:
        scores_map[target_model][judge_model] = score

    leaderboard = []
    for mid in target_ids:
        received = scores_map[mid]
        avg = round(sum(received.values()) / len(received), 2) if received else 0.0
        answer_content = next((a.content for a in answers if a.model == mid), "")
        leaderboard.append(
            ModelScore(
                model=mid,
                avg_score=avg,
                scores_received=received,
                response=answer_content,
            )
        )
    leaderboard.sort(key=lambda s: s.avg_score, reverse=True)

    total_elapsed = round(time.monotonic() - start, 2)
    return RankResult(
        prompt=prompt,
        leaderboard=leaderboard,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Tool 8: chain
# ---------------------------------------------------------------------------


@mcp.tool()
async def chain(
    prompt: str,
    pipeline: list[str],
    pass_context: bool = True,
) -> ChainResult:
    """Sequential pipeline — each model builds on prior outputs."""
    for mid in pipeline:
        get_model(mid)

    steps: list[ChainStep] = []
    all_outputs: list[str] = []

    start = time.monotonic()
    for i, model_id in enumerate(pipeline):
        if i == 0:
            step_prompt = prompt
        else:
            if pass_context:
                context = "\n\n".join(
                    f"--- Step {j + 1} ({pipeline[j]}) ---\n{all_outputs[j]}"
                    for j in range(i)
                )
            else:
                context = f"Previous analysis by {pipeline[i - 1]}:\n{all_outputs[-1]}"

            step_prompt = (
                f"{context}\n\n"
                f"Original question: {prompt}\n\n"
                f"Build on this — extend, refine, or correct."
            )

        step_start = time.monotonic()
        result = await call_ollama(model_id, step_prompt)
        step_elapsed = round(time.monotonic() - step_start, 2)

        steps.append(
            ChainStep(
                step=i + 1,
                model=model_id,
                content=result.content,
                elapsed_seconds=step_elapsed,
            )
        )
        all_outputs.append(result.content)

    total_elapsed = round(time.monotonic() - start, 2)
    return ChainResult(
        prompt=prompt,
        steps=steps,
        final_output=all_outputs[-1] if all_outputs else "",
        pipeline=pipeline,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Tool 9: map_reduce
# ---------------------------------------------------------------------------


@mcp.tool()
async def map_reduce(
    prompt: str,
    mapper_models: list[str] | None = None,
    reducer_model: str | None = None,
    reduce_prompt: str | None = None,
) -> MapReduceResult:
    """Fan-out to models, fan-in with a reducer. Custom reduce_prompt supported."""
    mappers = mapper_models if mapper_models else list(MODELS.keys())
    reducer = reducer_model or DEFAULT_JUDGE
    for mid in mappers:
        get_model(mid)
    get_model(reducer)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _map(model_id: str) -> CallResult:
        async with sem:
            return await call_ollama(model_id, prompt)

    start = time.monotonic()
    mapped = list(await asyncio.gather(*[_map(mid) for mid in mappers]))

    # Reduce phase
    response_block = "\n\n".join(
        f"[{r.model}]: {r.content}" for r in mapped if r.status == "ok"
    )
    n = sum(1 for r in mapped if r.status == "ok")
    default_reduce = (
        f"Synthesize these {n} responses into a single comprehensive answer. "
        f"Preserve the strongest points from each."
    )
    reduce_instruction = reduce_prompt or default_reduce
    full_reduce_prompt = (
        f"Original question: {prompt}\n\n"
        f"Responses from {n} models:\n\n{response_block}\n\n"
        f"Instructions: {reduce_instruction}"
    )
    reduce_result = await call_ollama(reducer, full_reduce_prompt)

    total_elapsed = round(time.monotonic() - start, 2)
    return MapReduceResult(
        prompt=prompt,
        mapped_responses=mapped,
        reduced_output=reduce_result.content,
        reducer_model=reducer,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Tool 10: blind_taste_test
# ---------------------------------------------------------------------------


@mcp.tool()
async def blind_taste_test(
    prompt: str,
    count: int = 3,
) -> BlindTasteResult:
    """Anonymous A/B/C comparison. Reveal mapping shows which model was which."""
    all_ids = list(MODELS.keys())
    count = min(count, len(all_ids))

    # Deterministic selection seeded by prompt hash for reproducibility
    seed = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    selected = rng.sample(all_ids, count)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _call(model_id: str) -> CallResult:
        async with sem:
            return await call_ollama(model_id, prompt)

    start = time.monotonic()
    results = list(await asyncio.gather(*[_call(mid) for mid in selected]))

    # Shuffle presentation order
    indexed = list(enumerate(results))
    rng.shuffle(indexed)

    labels = [chr(65 + i) for i in range(count)]  # A, B, C, ...
    responses: list[BlindResponse] = []
    reveal: dict[str, str] = {}
    for label, (_, r) in zip(labels, indexed):
        responses.append(BlindResponse(label=f"Response {label}", content=r.content))
        reveal[f"Response {label}"] = r.model

    total_elapsed = round(time.monotonic() - start, 2)
    return BlindTasteResult(
        prompt=prompt,
        responses=responses,
        reveal=reveal,
        total_elapsed_seconds=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Tool 11: contrarian
# ---------------------------------------------------------------------------


@mcp.tool()
async def contrarian(
    prompt: str,
    thesis_model: str | None = None,
    antithesis_model: str | None = None,
) -> ContrarianResult:
    """Thesis then devil's-advocate antithesis. Challenges assumptions."""
    t_model = thesis_model or "qwen3-next:80b-cloud"
    a_model = antithesis_model or "deepseek-v3.2:cloud"
    get_model(t_model)
    get_model(a_model)

    start = time.monotonic()

    # Phase 1: thesis
    thesis_result = await call_ollama(t_model, prompt)

    # Phase 2: antithesis
    anti_system = (
        "You are a critical analyst. Find logical gaps, challenge assumptions, "
        "identify missing perspectives, and argue alternatives. "
        "Be substantive, not contrarian for its own sake."
    )
    anti_prompt = (
        f"Original question: {prompt}\n\n"
        f"Thesis (by {t_model}):\n{thesis_result.content}\n\n"
        f"Provide a substantive counterargument."
    )
    anti_result = await call_ollama(a_model, anti_prompt, system_prompt=anti_system)

    total_elapsed = round(time.monotonic() - start, 2)
    return ContrarianResult(
        prompt=prompt,
        thesis_model=t_model,
        thesis=thesis_result.content,
        antithesis_model=a_model,
        antithesis=anti_result.content,
        total_elapsed_seconds=total_elapsed,
    )
