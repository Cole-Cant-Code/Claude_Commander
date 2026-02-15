"""Tests for the 11 advanced MCP tools + helper functions (mocked Ollama)."""

from unittest.mock import AsyncMock, patch

import pytest

from claude_commander.models import CallResult
from claude_commander.server import (
    _extract_score,
    _extract_vote,
    _pick_models_by_category,
    benchmark,
    blind_taste_test,
    chain,
    code_review,
    consensus,
    contrarian,
    debate,
    map_reduce,
    multi_solve,
    rank,
    vote,
)

# FastMCP's @mcp.tool() wraps functions in FunctionTool objects.
# Access the original async functions via .fn for direct testing.
_debate = debate.fn
_vote = vote.fn
_consensus = consensus.fn
_code_review = code_review.fn
_multi_solve = multi_solve.fn
_benchmark = benchmark.fn
_rank = rank.fn
_chain = chain.fn
_map_reduce = map_reduce.fn
_blind_taste_test = blind_taste_test.fn
_contrarian = contrarian.fn


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------


class TestExtractVote:
    def test_first_word_match(self):
        v, c = _extract_vote("yes I think so", ["yes", "no"])
        assert v == "yes"
        assert c == "high"

    def test_first_word_match_case_insensitive(self):
        v, c = _extract_vote("No, definitely not", ["yes", "no"])
        assert v == "no"
        assert c == "high"

    def test_phrase_i_vote(self):
        v, c = _extract_vote("After careful consideration, I vote yes on this", ["yes", "no"])
        assert v == "yes"
        assert c == "medium"

    def test_phrase_my_answer_is(self):
        v, c = _extract_vote("my answer is no because reasons", ["yes", "no"])
        assert v == "no"
        assert c == "medium"

    def test_phrase_i_choose(self):
        v, c = _extract_vote("I think I choose yes", ["yes", "no"])
        assert v == "yes"
        assert c == "medium"

    def test_occurrence_counting(self):
        v, c = _extract_vote(
            "Well, banana is great, banana is tasty, banana wins", ["apple", "banana"]
        )
        assert v == "banana"
        assert c == "low"

    def test_abstain_on_no_match(self):
        v, c = _extract_vote("I have no opinion on this matter", ["agree", "disagree"])
        assert v == "abstain"
        assert c == "none"

    def test_empty_response(self):
        v, c = _extract_vote("", ["yes", "no"])
        assert v == "abstain"
        assert c == "none"

    def test_multi_option(self):
        v, c = _extract_vote("red is the best color", ["red", "blue", "green"])
        assert v == "red"
        assert c == "high"

    def test_tied_occurrences_abstain(self):
        # Equal counts of both options — no clear winner
        v, c = _extract_vote("maybe yes or maybe no", ["yes", "no"])
        assert v == "abstain"
        assert c == "none"


class TestExtractScore:
    def test_x_out_of_10(self):
        assert _extract_score("Score: 8/10. Good response.") == 8.0

    def test_score_colon(self):
        assert _extract_score("Score: 7 - decent answer") == 7.0

    def test_rating_colon(self):
        assert _extract_score("Rating: 9.5 — excellent!") == 9.5

    def test_x_out_of_10_words(self):
        assert _extract_score("I'd give this 6 out of 10") == 6.0

    def test_clamp_high(self):
        assert _extract_score("Score: 15/10") == 10.0

    def test_clamp_low(self):
        assert _extract_score("Score: 0/10") == 1.0

    def test_standalone_number(self):
        assert _extract_score("Hmm, about a 7 I'd say") == 7.0

    def test_no_number_default(self):
        assert _extract_score("This is a good response overall") == 5.0

    def test_decimal_score(self):
        assert _extract_score("7.5/10") == 7.5

    def test_number_out_of_range_default(self):
        # 42 is outside 1-10, falls through to default
        assert _extract_score("about 42 points") == 5.0


class TestPickModelsByCategory:
    def test_code_category(self):
        models = _pick_models_by_category(["code"])
        assert "qwen3-coder-next:cloud" in models
        assert "glm-5:cloud" not in models

    def test_reasoning_category(self):
        models = _pick_models_by_category(["reasoning"])
        assert "deepseek-v3.2:cloud" in models
        assert "kimi-k2-thinking:cloud" in models

    def test_multiple_categories(self):
        models = _pick_models_by_category(["code", "reasoning"])
        assert "qwen3-coder-next:cloud" in models
        assert "deepseek-v3.2:cloud" in models

    def test_empty_category(self):
        models = _pick_models_by_category(["nonexistent"])
        assert models == []


# ---------------------------------------------------------------------------
# Mock helper
# ---------------------------------------------------------------------------


def _make_mock(content: str = "mock response"):
    """Create a mock call_ollama that returns predictable content."""

    async def mock_call(model, prompt, **kw):
        return CallResult(model=model, content=content, elapsed_seconds=0.1)

    return mock_call


def _make_mock_varied():
    """Create a mock that returns model-specific content."""

    async def mock_call(model, prompt, **kw):
        return CallResult(model=model, content=f"response from {model}", elapsed_seconds=0.1)

    return mock_call


# ---------------------------------------------------------------------------
# Tool tests: debate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_debate_default_models():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()):
        result = await _debate("Is Rust better than Go?")
    assert result.model_a == "deepseek-v3.2:cloud"
    assert result.model_b == "glm-5:cloud"
    assert len(result.rounds) == 3
    assert result.rounds[0].model == "deepseek-v3.2:cloud"  # odd round
    assert result.rounds[1].model == "glm-5:cloud"  # even round
    assert result.rounds[2].model == "deepseek-v3.2:cloud"  # odd round


@pytest.mark.asyncio
async def test_debate_custom_models_and_rounds():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()):
        result = await _debate(
            "tabs vs spaces",
            model_a="glm-5:cloud",
            model_b="kimi-k2.5:cloud",
            rounds=2,
        )
    assert result.model_a == "glm-5:cloud"
    assert result.model_b == "kimi-k2.5:cloud"
    assert len(result.rounds) == 2


@pytest.mark.asyncio
async def test_debate_invalid_model():
    with pytest.raises(ValueError, match="Unknown model"):
        await _debate("test", model_a="fake:model")


@pytest.mark.asyncio
async def test_debate_transcript_builds():
    """Each round should see the growing transcript (call_ollama receives it)."""
    prompts_received = []

    async def tracking_mock(model, prompt, **kw):
        prompts_received.append(prompt)
        return CallResult(model=model, content=f"round by {model}", elapsed_seconds=0.1)

    with patch("claude_commander.server.call_ollama", side_effect=tracking_mock):
        await _debate("topic", rounds=2)

    # Round 2 prompt should contain round 1 content
    assert "round by" in prompts_received[1]


# ---------------------------------------------------------------------------
# Tool tests: vote
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vote_unanimous_yes():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock("yes absolutely")):
        result = await _vote("Is the sky blue?", models=["glm-5:cloud", "kimi-k2.5:cloud"])
    assert result.majority == "yes"
    assert result.agreement_pct == 100.0
    assert result.tally["yes"] == 2
    assert "abstain" not in result.tally


@pytest.mark.asyncio
async def test_vote_custom_options():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock("red is the best")):
        result = await _vote(
            "Best color?", options=["red", "blue", "green"], models=["glm-5:cloud"]
        )
    assert result.majority == "red"


@pytest.mark.asyncio
async def test_vote_with_abstain():
    with patch(
        "claude_commander.server.call_ollama",
        side_effect=_make_mock("I cannot decide on this"),
    ):
        result = await _vote(
            "Pick one", options=["agree", "disagree"], models=["glm-5:cloud"]
        )
    assert result.votes[0].extracted_vote == "abstain"
    assert "abstain" in result.tally


@pytest.mark.asyncio
async def test_vote_mixed():
    call_count = 0

    async def alternating(model, prompt, **kw):
        nonlocal call_count
        call_count += 1
        answer = "yes for sure" if call_count % 2 == 1 else "no way"
        return CallResult(model=model, content=answer, elapsed_seconds=0.1)

    with patch("claude_commander.server.call_ollama", side_effect=alternating):
        result = await _vote("test?", models=["glm-5:cloud", "kimi-k2.5:cloud"])
    assert result.tally["yes"] == 1
    assert result.tally["no"] == 1


# ---------------------------------------------------------------------------
# Tool tests: consensus
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consensus_basic():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()):
        result = await _consensus(
            "What is 2+2?", models=["glm-5:cloud", "kimi-k2.5:cloud"]
        )
    assert len(result.individual_responses) == 2
    assert result.judge_model == "kimi-k2-thinking:cloud"
    assert result.synthesis  # non-empty


@pytest.mark.asyncio
async def test_consensus_custom_judge():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()):
        result = await _consensus(
            "test", models=["glm-5:cloud"], judge_model="deepseek-v3.2:cloud"
        )
    assert result.judge_model == "deepseek-v3.2:cloud"


# ---------------------------------------------------------------------------
# Tool tests: code_review
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_code_review_basic():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()):
        result = await _code_review("def foo(): pass")
    assert len(result.reviews) == 3  # CODE_MODELS has 3 entries
    assert result.merge_model == "kimi-k2-thinking:cloud"
    assert result.merged_review  # non-empty
    assert result.code_snippet == "def foo(): pass"


@pytest.mark.asyncio
async def test_code_review_with_language():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()):
        result = await _code_review("fn main() {}", language="rust")
    assert result.code_snippet == "fn main() {}"


@pytest.mark.asyncio
async def test_code_review_custom_reviewers():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()):
        result = await _code_review(
            "x = 1",
            review_models=["glm-5:cloud"],
            merge_model="deepseek-v3.2:cloud",
        )
    assert len(result.reviews) == 1
    assert result.merge_model == "deepseek-v3.2:cloud"


# ---------------------------------------------------------------------------
# Tool tests: multi_solve
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_solve_default_models():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()):
        result = await _multi_solve("FizzBuzz")
    # Should include code + reasoning + 2 generals (deduped)
    assert len(result.solutions) >= 3
    assert result.problem == "FizzBuzz"


@pytest.mark.asyncio
async def test_multi_solve_with_language():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()):
        result = await _multi_solve("sort a list", language="Python")
    assert result.language == "Python"


@pytest.mark.asyncio
async def test_multi_solve_custom_models():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()):
        result = await _multi_solve(
            "FizzBuzz", models=["glm-5:cloud", "deepseek-v3.2:cloud"]
        )
    assert len(result.solutions) == 2


# ---------------------------------------------------------------------------
# Tool tests: benchmark
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_benchmark_basic():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()):
        result = await _benchmark(
            ["hello", "world"], models=["glm-5:cloud", "kimi-k2.5:cloud"]
        )
    assert len(result.prompts) == 2
    assert len(result.models) == 2
    assert len(result.results) == 4  # 2 prompts * 2 models
    assert "glm-5:cloud" in result.model_stats
    assert result.model_stats["glm-5:cloud"] == 0.1


@pytest.mark.asyncio
async def test_benchmark_single_prompt():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()):
        result = await _benchmark(["single"], models=["glm-5:cloud"])
    assert len(result.results) == 1
    assert result.results[0].prompt_index == 0


# ---------------------------------------------------------------------------
# Tool tests: rank
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rank_basic():
    async def score_mock(model, prompt, **kw):
        # Return score-like content for judge calls
        if "Rate the following" in kw.get("system_prompt", ""):
            return CallResult(model=model, content="Score: 8/10 Good!", elapsed_seconds=0.1)
        return CallResult(model=model, content=f"answer from {model}", elapsed_seconds=0.1)

    with patch("claude_commander.server.call_ollama", side_effect=score_mock):
        result = await _rank(
            "Explain recursion",
            models=["glm-5:cloud", "kimi-k2.5:cloud", "deepseek-v3.2:cloud"],
            judge_count=2,
        )
    assert len(result.leaderboard) == 3
    # All should get score 8.0 since mock always returns 8/10
    for entry in result.leaderboard:
        assert entry.avg_score == 8.0


@pytest.mark.asyncio
async def test_rank_default_models():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock("Score: 7/10")):
        result = await _rank("test prompt")
    assert len(result.leaderboard) == 5  # RANK_DEFAULT_MODELS


@pytest.mark.asyncio
async def test_rank_sorted_descending():
    call_counter = {"n": 0}

    async def varying_scores(model, prompt, **kw):
        call_counter["n"] += 1
        if "Rate the following" in kw.get("system_prompt", ""):
            # Give different scores based on which model is being judged
            if "glm-5:cloud" in prompt:
                return CallResult(model=model, content="Score: 9/10", elapsed_seconds=0.1)
            elif "kimi-k2.5:cloud" in prompt:
                return CallResult(model=model, content="Score: 5/10", elapsed_seconds=0.1)
            return CallResult(model=model, content="Score: 7/10", elapsed_seconds=0.1)
        return CallResult(model=model, content=f"answer from {model}", elapsed_seconds=0.1)

    with patch("claude_commander.server.call_ollama", side_effect=varying_scores):
        result = await _rank(
            "test",
            models=["glm-5:cloud", "kimi-k2.5:cloud", "deepseek-v3.2:cloud"],
            judge_count=2,
        )
    # Leaderboard should be sorted descending by avg_score
    scores = [e.avg_score for e in result.leaderboard]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Tool tests: chain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chain_basic():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()):
        result = await _chain(
            "What is AI?",
            pipeline=["glm-5:cloud", "deepseek-v3.2:cloud", "kimi-k2.5:cloud"],
        )
    assert len(result.steps) == 3
    assert result.pipeline == ["glm-5:cloud", "deepseek-v3.2:cloud", "kimi-k2.5:cloud"]
    assert result.final_output  # non-empty
    assert result.steps[0].step == 1
    assert result.steps[2].step == 3


@pytest.mark.asyncio
async def test_chain_context_passing():
    """With pass_context=True, later steps should see all prior outputs."""
    prompts_received = []

    async def tracking_mock(model, prompt, **kw):
        prompts_received.append(prompt)
        return CallResult(model=model, content=f"output-{model}", elapsed_seconds=0.1)

    with patch("claude_commander.server.call_ollama", side_effect=tracking_mock):
        await _chain(
            "question",
            pipeline=["glm-5:cloud", "deepseek-v3.2:cloud", "kimi-k2.5:cloud"],
            pass_context=True,
        )

    # Step 3 (index 2) should contain output from step 1
    assert "output-glm-5:cloud" in prompts_received[2]


@pytest.mark.asyncio
async def test_chain_no_context():
    """With pass_context=False, step 3 should only see step 2 output."""
    prompts_received = []

    async def tracking_mock(model, prompt, **kw):
        prompts_received.append(prompt)
        return CallResult(model=model, content=f"output-{model}", elapsed_seconds=0.1)

    with patch("claude_commander.server.call_ollama", side_effect=tracking_mock):
        await _chain(
            "question",
            pipeline=["glm-5:cloud", "deepseek-v3.2:cloud", "kimi-k2.5:cloud"],
            pass_context=False,
        )

    # Step 3 should contain step 2 output but NOT formatted as "Step 1"
    assert "output-deepseek-v3.2:cloud" in prompts_received[2]
    assert "Step 1" not in prompts_received[2]


@pytest.mark.asyncio
async def test_chain_invalid_model():
    with pytest.raises(ValueError, match="Unknown model"):
        await _chain("test", pipeline=["fake:model"])


# ---------------------------------------------------------------------------
# Tool tests: map_reduce
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_map_reduce_basic():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()):
        result = await _map_reduce(
            "What is AI?", mapper_models=["glm-5:cloud", "kimi-k2.5:cloud"]
        )
    assert len(result.mapped_responses) == 2
    assert result.reducer_model == "kimi-k2-thinking:cloud"
    assert result.reduced_output  # non-empty


@pytest.mark.asyncio
async def test_map_reduce_custom_reduce_prompt():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()):
        result = await _map_reduce(
            "test",
            mapper_models=["glm-5:cloud"],
            reduce_prompt="Just pick the best one.",
        )
    assert result.reduced_output  # non-empty


@pytest.mark.asyncio
async def test_map_reduce_custom_reducer():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()):
        result = await _map_reduce(
            "test",
            mapper_models=["glm-5:cloud"],
            reducer_model="deepseek-v3.2:cloud",
        )
    assert result.reducer_model == "deepseek-v3.2:cloud"


# ---------------------------------------------------------------------------
# Tool tests: blind_taste_test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blind_taste_test_basic():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()), \
         patch("claude_commander.server.call_cli", side_effect=_make_mock_varied()):
        result = await _blind_taste_test("Tell me a joke")
    assert len(result.responses) == 3  # default count
    assert len(result.reveal) == 3
    # Labels should be Response A, B, C
    labels = [r.label for r in result.responses]
    assert "Response A" in labels
    assert "Response B" in labels
    assert "Response C" in labels


@pytest.mark.asyncio
async def test_blind_taste_test_deterministic():
    """Same prompt should select same models."""
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()), \
         patch("claude_commander.server.call_cli", side_effect=_make_mock_varied()):
        r1 = await _blind_taste_test("same prompt")
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()), \
         patch("claude_commander.server.call_cli", side_effect=_make_mock_varied()):
        r2 = await _blind_taste_test("same prompt")
    assert r1.reveal == r2.reveal


@pytest.mark.asyncio
async def test_blind_taste_test_custom_count():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()), \
         patch("claude_commander.server.call_cli", side_effect=_make_mock_varied()):
        result = await _blind_taste_test("test", count=5)
    assert len(result.responses) == 5


@pytest.mark.asyncio
async def test_blind_taste_test_count_capped():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()), \
         patch("claude_commander.server.call_cli", side_effect=_make_mock_varied()):
        result = await _blind_taste_test("test", count=100)
    assert len(result.responses) == 17  # max = total models


# ---------------------------------------------------------------------------
# Tool tests: contrarian
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contrarian_basic():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()):
        result = await _contrarian("AI will replace all jobs")
    assert result.thesis_model == "qwen3-next:80b-cloud"
    assert result.antithesis_model == "deepseek-v3.2:cloud"
    assert result.thesis  # non-empty
    assert result.antithesis  # non-empty


@pytest.mark.asyncio
async def test_contrarian_custom_models():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()):
        result = await _contrarian(
            "test",
            thesis_model="glm-5:cloud",
            antithesis_model="kimi-k2.5:cloud",
        )
    assert result.thesis_model == "glm-5:cloud"
    assert result.antithesis_model == "kimi-k2.5:cloud"


@pytest.mark.asyncio
async def test_contrarian_invalid_model():
    with pytest.raises(ValueError, match="Unknown model"):
        await _contrarian("test", thesis_model="fake:model")


@pytest.mark.asyncio
async def test_contrarian_antithesis_sees_thesis():
    """Antithesis model should receive the thesis in its prompt."""
    prompts_received = []

    async def tracking_mock(model, prompt, **kw):
        prompts_received.append(prompt)
        return CallResult(model=model, content=f"content from {model}", elapsed_seconds=0.1)

    with patch("claude_commander.server.call_ollama", side_effect=tracking_mock):
        await _contrarian("claim X")

    # Second call (antithesis) should include thesis content
    assert "content from qwen3-next:80b-cloud" in prompts_received[1]
