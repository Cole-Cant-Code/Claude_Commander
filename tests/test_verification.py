"""Tests for verification & anti-slop tools + helper functions (mocked Ollama)."""

from unittest.mock import AsyncMock, patch

import pytest

from claude_commander.models import CallResult, RedTeamExchange
from claude_commander.server import (
    _extract_issues,
    _extract_severity,
    _extract_surviving_issues,
    _extract_verdict,
    _parse_claims,
    _parse_criteria_scores,
    _severity_rank,
    detect_slop,
    quality_gate,
    red_team,
    verify,
)

# FastMCP's @mcp.tool() wraps functions in FunctionTool objects.
_verify = verify.fn
_red_team = red_team.fn
_quality_gate = quality_gate.fn
_detect_slop = detect_slop.fn


# ---------------------------------------------------------------------------
# Helper: _extract_verdict
# ---------------------------------------------------------------------------


class TestExtractVerdict:
    def test_explicit_verdict_marker(self):
        assert _extract_verdict("Verdict: pass", ["pass", "fail", "mixed"]) == "pass"

    def test_explicit_result_marker(self):
        assert _extract_verdict("Result: fail due to errors", ["pass", "fail", "mixed"]) == "fail"

    def test_explicit_overall_marker(self):
        assert _extract_verdict("Overall: mixed results here", ["pass", "fail", "mixed"]) == "mixed"

    def test_conclusion_marker(self):
        assert _extract_verdict("Conclusion: pass with notes", ["pass", "fail", "mixed"]) == "pass"

    def test_first_word_match(self):
        assert _extract_verdict("fail — too many errors", ["pass", "fail", "mixed"]) == "fail"

    def test_occurrence_counting(self):
        assert _extract_verdict(
            "this could pass and that could pass but unlikely fail", ["pass", "fail"]
        ) == "pass"

    def test_defaults_to_first_option(self):
        assert _extract_verdict("nothing relevant here", ["pass", "fail"]) == "pass"

    def test_case_insensitive(self):
        assert _extract_verdict("VERDICT: PASS", ["pass", "fail"]) == "pass"


# ---------------------------------------------------------------------------
# Helper: _extract_issues
# ---------------------------------------------------------------------------


class TestExtractIssues:
    def test_bullet_dash(self):
        text = "Issues found:\n- Missing error handling\n- No input validation"
        issues = _extract_issues(text)
        assert len(issues) == 2
        assert "Missing error handling" in issues
        assert "No input validation" in issues

    def test_bullet_asterisk(self):
        text = "* First issue\n* Second issue"
        issues = _extract_issues(text)
        assert len(issues) == 2

    def test_numbered(self):
        text = "1. First\n2. Second\n3. Third"
        issues = _extract_issues(text)
        assert len(issues) == 3
        assert "First" in issues

    def test_issue_marker(self):
        text = "Issue: Something is wrong\nIssue: Another problem"
        issues = _extract_issues(text)
        assert len(issues) == 2

    def test_no_issues(self):
        text = "Everything looks good. No problems found."
        issues = _extract_issues(text)
        assert issues == []

    def test_mixed_formats(self):
        text = "- Bullet issue\n1. Numbered issue\nIssue: Marker issue"
        issues = _extract_issues(text)
        assert len(issues) == 3


# ---------------------------------------------------------------------------
# Helper: _extract_severity
# ---------------------------------------------------------------------------


class TestExtractSeverity:
    def test_critical_keywords(self):
        assert _extract_severity("This is a critical security flaw") == "critical"
        assert _extract_severity("severe data loss risk") == "critical"
        assert _extract_severity("fatal error in processing") == "critical"
        assert _extract_severity("breaking change detected") == "critical"

    def test_major_keywords(self):
        assert _extract_severity("major performance regression") == "major"
        assert _extract_severity("significant memory leak") == "major"
        assert _extract_severity("important edge case missed") == "major"

    def test_minor_default(self):
        assert _extract_severity("small typo in output") == "minor"
        assert _extract_severity("could be slightly improved") == "minor"


# ---------------------------------------------------------------------------
# Helper: _severity_rank
# ---------------------------------------------------------------------------


class TestSeverityRank:
    def test_critical(self):
        assert _severity_rank("critical") == 3

    def test_major(self):
        assert _severity_rank("major") == 2

    def test_minor(self):
        assert _severity_rank("minor") == 1

    def test_unknown(self):
        assert _severity_rank("unknown") == 0


# ---------------------------------------------------------------------------
# Helper: _parse_claims
# ---------------------------------------------------------------------------


class TestParseClaims:
    def test_full_format(self):
        text = "CLAIM: The sky is blue | VERDICT: supported | CONFIDENCE: 0.95 | SOURCES: Wikipedia, NASA"
        claims = _parse_claims(text)
        assert len(claims) == 1
        assert claims[0].claim == "The sky is blue"
        assert claims[0].verdict == "supported"
        assert claims[0].confidence == 0.95
        assert "Wikipedia" in claims[0].sources_cited
        assert "NASA" in claims[0].sources_cited

    def test_minimal_format(self):
        text = "CLAIM: Water is wet | VERDICT: supported"
        claims = _parse_claims(text)
        assert len(claims) == 1
        assert claims[0].claim == "Water is wet"
        assert claims[0].verdict == "supported"
        assert claims[0].confidence == 0.5  # default

    def test_multiple_claims(self):
        text = (
            "CLAIM: Earth is round | VERDICT: supported | CONFIDENCE: 0.99\n"
            "CLAIM: Moon is cheese | VERDICT: contradicted | CONFIDENCE: 0.01\n"
            "CLAIM: Dark matter exists | VERDICT: unverifiable | CONFIDENCE: 0.6"
        )
        claims = _parse_claims(text)
        assert len(claims) == 3
        assert claims[1].verdict == "contradicted"
        assert claims[2].verdict == "unverifiable"

    def test_no_claims(self):
        text = "This content looks fine overall. No specific claims to verify."
        claims = _parse_claims(text)
        assert claims == []

    def test_confidence_clamped(self):
        text = "CLAIM: Test | VERDICT: supported | CONFIDENCE: 1.5"
        claims = _parse_claims(text)
        assert claims[0].confidence == 1.0


# ---------------------------------------------------------------------------
# Helper: _parse_criteria_scores
# ---------------------------------------------------------------------------


class TestParseCriteriaScores:
    def test_structured_format(self):
        text = "CRITERION: accuracy | SCORE: 8 | ISSUES: minor factual error; typo in date"
        scores = _parse_criteria_scores(text, ["accuracy"])
        assert len(scores) == 1
        assert scores[0].criterion == "accuracy"
        assert scores[0].score == 8.0
        assert len(scores[0].issues) == 2

    def test_fallback_to_slash_format(self):
        text = "Clarity: 7/10. The writing is mostly clear."
        scores = _parse_criteria_scores(text, ["clarity"])
        assert len(scores) == 1
        assert scores[0].score == 7.0

    def test_default_score_for_missing(self):
        text = "No structured output at all here."
        scores = _parse_criteria_scores(text, ["accuracy", "clarity"])
        assert len(scores) == 2
        assert all(s.score == 5.0 for s in scores)

    def test_mixed_found_and_missing(self):
        text = "CRITERION: accuracy | SCORE: 9\nSome other text."
        scores = _parse_criteria_scores(text, ["accuracy", "clarity"])
        assert len(scores) == 2
        accuracy = next(s for s in scores if s.criterion == "accuracy")
        clarity = next(s for s in scores if s.criterion == "clarity")
        assert accuracy.score == 9.0
        assert clarity.score == 5.0  # fallback default


# ---------------------------------------------------------------------------
# Helper: _extract_surviving_issues
# ---------------------------------------------------------------------------


class TestExtractSurvivingIssues:
    def test_structured_severity_markers(self):
        exchange = RedTeamExchange(
            round_number=3,
            attack="[CRITICAL] SQL injection possible\n[MINOR] Typo in docs",
            attack_model="attacker",
            defense="addressed",
            defense_model="defender",
        )
        issues = _extract_surviving_issues([exchange])
        assert len(issues) == 2
        assert issues[0].severity == "critical"
        assert issues[0].issue == "SQL injection possible"
        assert issues[1].severity == "minor"

    def test_fallback_to_bullet_extraction(self):
        exchange = RedTeamExchange(
            round_number=2,
            attack="Still remaining:\n- Critical memory leak in handler\n- Minor formatting issue",
            attack_model="attacker",
            defense="noted",
            defense_model="defender",
        )
        issues = _extract_surviving_issues([exchange])
        assert len(issues) == 2
        assert issues[0].severity == "critical"  # "critical" keyword in text

    def test_empty_exchanges(self):
        assert _extract_surviving_issues([]) == []

    def test_first_raised_round(self):
        exchange = RedTeamExchange(
            round_number=3,
            attack="[MAJOR] Still not fixed",
            attack_model="attacker",
            defense="ok",
            defense_model="defender",
        )
        issues = _extract_surviving_issues([exchange])
        assert issues[0].first_raised_round == 3


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_mock(content: str = "mock response"):
    async def mock_call(model, prompt, **kw):
        return CallResult(model=model, content=content, elapsed_seconds=0.1)
    return mock_call


def _make_mock_varied():
    async def mock_call(model, prompt, **kw):
        return CallResult(model=model, content=f"response from {model}", elapsed_seconds=0.1)
    return mock_call


# ---------------------------------------------------------------------------
# Tool tests: verify
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_basic():
    """Verify tool should call verifiers in parallel then judge."""
    call_count = {"n": 0}

    async def mock_call(model, prompt, **kw):
        call_count["n"] += 1
        if call_count["n"] <= 3:
            # Verifier responses
            return CallResult(
                model=model,
                content="CLAIM: The sky is blue | VERDICT: supported | CONFIDENCE: 0.9",
                elapsed_seconds=0.1,
            )
        # Judge synthesis
        return CallResult(
            model=model,
            content=(
                "CLAIM: The sky is blue | VERDICT: supported | CONFIDENCE: 0.95\n"
                "OVERALL VERDICT: pass\n"
                "INTERNAL CONTRADICTIONS: None found"
            ),
            elapsed_seconds=0.1,
        )

    with patch("claude_commander.server.call_ollama", side_effect=mock_call):
        result = await _verify("The sky is blue and water is wet.")

    assert result.overall_verdict == "pass"
    assert len(result.verifier_models) == 3
    assert result.content_snippet.startswith("The sky is blue")


@pytest.mark.asyncio
async def test_verify_custom_models():
    async def mock_call(model, prompt, **kw):
        return CallResult(
            model=model,
            content="CLAIM: Test | VERDICT: supported | CONFIDENCE: 0.8\nOverall: pass",
            elapsed_seconds=0.1,
        )

    with patch("claude_commander.server.call_ollama", side_effect=mock_call):
        result = await _verify(
            "Test content",
            verifier_models=["glm-5:cloud"],
            judge_model="deepseek-v3.2:cloud",
        )

    assert result.verifier_models == ["glm-5:cloud"]


@pytest.mark.asyncio
async def test_verify_invalid_model():
    with pytest.raises(ValueError, match="Unknown model"):
        await _verify("test", verifier_models=["fake:model"])


@pytest.mark.asyncio
async def test_verify_fail_verdict():
    async def mock_call(model, prompt, **kw):
        return CallResult(
            model=model,
            content=(
                "CLAIM: X causes Y | VERDICT: contradicted | CONFIDENCE: 0.9\n"
                "OVERALL VERDICT: fail"
            ),
            elapsed_seconds=0.1,
        )

    with patch("claude_commander.server.call_ollama", side_effect=mock_call):
        result = await _verify("X causes Y for certain.")

    assert result.overall_verdict == "fail"


# ---------------------------------------------------------------------------
# Tool tests: red_team
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_red_team_basic():
    call_count = {"n": 0}

    async def mock_call(model, prompt, **kw):
        call_count["n"] += 1
        if "attacker" in kw.get("system_prompt", "").lower() or "red-team" in kw.get("system_prompt", "").lower():
            return CallResult(
                model=model,
                content="[MAJOR] Missing input validation\n- Edge case not handled",
                elapsed_seconds=0.1,
            )
        return CallResult(
            model=model,
            content="Addressed: added input validation and edge case handling.",
            elapsed_seconds=0.1,
        )

    with patch("claude_commander.server.call_ollama", side_effect=mock_call):
        result = await _red_team("def process(x): return x + 1", rounds=2)

    assert result.rounds_completed == 2
    assert result.attacker_model == "deepseek-v3.2:cloud"
    assert result.defender_model == "kimi-k2-thinking:cloud"
    assert len(result.exchanges) == 2


@pytest.mark.asyncio
async def test_red_team_convergence():
    call_count = {"n": 0}

    async def mock_call(model, prompt, **kw):
        call_count["n"] += 1
        if "adversarial" in kw.get("system_prompt", "").lower():
            return CallResult(
                model=model,
                content="NO NEW ISSUES FOUND — the content is solid.",
                elapsed_seconds=0.1,
            )
        return CallResult(model=model, content="Defense response", elapsed_seconds=0.1)

    with patch("claude_commander.server.call_ollama", side_effect=mock_call):
        result = await _red_team("Well-written content here", rounds=5)

    assert result.converged is True
    assert result.rounds_completed == 1  # stopped on first round
    assert result.robustness_score == 10.0


@pytest.mark.asyncio
async def test_red_team_custom_models():
    with patch("claude_commander.server.call_ollama", side_effect=_make_mock_varied()):
        result = await _red_team(
            "test",
            attacker_model="glm-5:cloud",
            defender_model="kimi-k2.5:cloud",
            rounds=1,
        )
    assert result.attacker_model == "glm-5:cloud"
    assert result.defender_model == "kimi-k2.5:cloud"


@pytest.mark.asyncio
async def test_red_team_invalid_model():
    with pytest.raises(ValueError, match="Unknown model"):
        await _red_team("test", attacker_model="fake:model")


@pytest.mark.asyncio
async def test_red_team_transcript_builds():
    """Each round should see the growing transcript."""
    prompts_received = []

    async def tracking_mock(model, prompt, **kw):
        prompts_received.append(prompt)
        return CallResult(model=model, content=f"content by {model}", elapsed_seconds=0.1)

    with patch("claude_commander.server.call_ollama", side_effect=tracking_mock):
        await _red_team("original content", rounds=2)

    # Round 2 attacker (index 2) should see round 1 attack content
    assert "content by" in prompts_received[2]


@pytest.mark.asyncio
async def test_red_team_robustness_scoring():
    """Surviving issues should decrease the robustness score."""
    async def mock_call(model, prompt, **kw):
        if "adversarial" in kw.get("system_prompt", "").lower():
            return CallResult(
                model=model,
                content=(
                    "[CRITICAL] Security vulnerability\n"
                    "[MAJOR] Performance issue\n"
                    "[MINOR] Style concern"
                ),
                elapsed_seconds=0.1,
            )
        return CallResult(model=model, content="Addressed some issues.", elapsed_seconds=0.1)

    with patch("claude_commander.server.call_ollama", side_effect=mock_call):
        result = await _red_team("test content", rounds=1)

    # penalty = 3 (critical) + 2 (major) + 1 (minor) = 6
    assert result.robustness_score == 4.0


# ---------------------------------------------------------------------------
# Tool tests: quality_gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quality_gate_pass():
    async def mock_call(model, prompt, **kw):
        return CallResult(
            model=model,
            content=(
                "CRITERION: accuracy | SCORE: 9\n"
                "CRITERION: completeness | SCORE: 8\n"
                "CRITERION: clarity | SCORE: 9\n"
                "CRITERION: specificity | SCORE: 8\n"
                "CRITERION: logical_coherence | SCORE: 9\n"
                "OVERALL VERDICT: pass"
            ),
            elapsed_seconds=0.1,
        )

    with patch("claude_commander.server.call_ollama", side_effect=mock_call):
        result = await _quality_gate("High quality content here.")

    assert result.passed is True
    assert result.overall_score >= 7.0
    assert result.judge_model == "kimi-k2-thinking:cloud"
    assert len(result.criteria_scores) == 5


@pytest.mark.asyncio
async def test_quality_gate_fail_low_score():
    async def mock_call(model, prompt, **kw):
        return CallResult(
            model=model,
            content=(
                "CRITERION: accuracy | SCORE: 3\n"
                "CRITERION: completeness | SCORE: 4\n"
                "CRITERION: clarity | SCORE: 3\n"
                "CRITERION: specificity | SCORE: 4\n"
                "CRITERION: logical_coherence | SCORE: 3\n"
                "OVERALL VERDICT: fail"
            ),
            elapsed_seconds=0.1,
        )

    with patch("claude_commander.server.call_ollama", side_effect=mock_call):
        result = await _quality_gate("Poor content.")

    assert result.passed is False
    assert result.overall_score < 7.0


@pytest.mark.asyncio
async def test_quality_gate_fail_catastrophic_criterion():
    """Even if overall is above threshold, a catastrophically low criterion blocks."""
    async def mock_call(model, prompt, **kw):
        return CallResult(
            model=model,
            content=(
                "CRITERION: accuracy | SCORE: 4\n"  # catastrophic: 4 < 7.0 - 2.0
                "CRITERION: completeness | SCORE: 9\n"
                "CRITERION: clarity | SCORE: 9\n"
                "CRITERION: specificity | SCORE: 9\n"
                "CRITERION: logical_coherence | SCORE: 9\n"
            ),
            elapsed_seconds=0.1,
        )

    with patch("claude_commander.server.call_ollama", side_effect=mock_call):
        result = await _quality_gate("Content with one bad criterion.")

    # Overall: (4+9+9+9+9)/5 = 8.0 >= 7.0, but accuracy 4.0 < 5.0 (threshold-2)
    assert result.passed is False
    assert len(result.blocking_issues) > 0


@pytest.mark.asyncio
async def test_quality_gate_custom_criteria():
    async def mock_call(model, prompt, **kw):
        return CallResult(
            model=model,
            content="CRITERION: originality | SCORE: 8\nCRITERION: depth | SCORE: 7",
            elapsed_seconds=0.1,
        )

    with patch("claude_commander.server.call_ollama", side_effect=mock_call):
        result = await _quality_gate(
            "Test content",
            criteria=["originality", "depth"],
            threshold=7.0,
        )

    assert result.passed is True
    criteria_names = [c.criterion for c in result.criteria_scores]
    assert "originality" in criteria_names
    assert "depth" in criteria_names


@pytest.mark.asyncio
async def test_quality_gate_custom_threshold():
    async def mock_call(model, prompt, **kw):
        return CallResult(
            model=model,
            content="CRITERION: accuracy | SCORE: 8\nCRITERION: clarity | SCORE: 8",
            elapsed_seconds=0.1,
        )

    with patch("claude_commander.server.call_ollama", side_effect=mock_call):
        result = await _quality_gate(
            "Test", criteria=["accuracy", "clarity"], threshold=9.0
        )

    assert result.passed is False
    assert result.threshold == 9.0


@pytest.mark.asyncio
async def test_quality_gate_invalid_model():
    with pytest.raises(ValueError, match="Unknown model"):
        await _quality_gate("test", judge_model="fake:model")


# ---------------------------------------------------------------------------
# Tool tests: detect_slop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_slop_clean():
    async def mock_call(model, prompt, **kw):
        return CallResult(
            model=model,
            content="No slop signals detected.\nSLOP SCORE: 1",
            elapsed_seconds=0.1,
        )

    with patch("claude_commander.server.call_ollama", side_effect=mock_call):
        result = await _detect_slop("Well-written, specific technical content.")

    assert result.verdict == "clean"
    assert result.slop_score <= 2.0
    assert len(result.detector_models) == 3


@pytest.mark.asyncio
async def test_detect_slop_severe():
    async def mock_call(model, prompt, **kw):
        return CallResult(
            model=model,
            content=(
                "SIGNAL: filler_phrases | SEVERITY: major | DESCRIPTION: Heavy use of filler | EXAMPLES: in conclusion; it's worth noting\n"
                "SIGNAL: vague_generalities | SEVERITY: critical | DESCRIPTION: No specifics | EXAMPLES: many experts agree\n"
                "SLOP SCORE: 9"
            ),
            elapsed_seconds=0.1,
        )

    with patch("claude_commander.server.call_ollama", side_effect=mock_call):
        result = await _detect_slop("In conclusion, it's worth noting that many experts agree...")

    assert result.verdict == "severe"
    assert result.slop_score > 7.0
    assert len(result.signals) >= 2


@pytest.mark.asyncio
async def test_detect_slop_signal_aggregation():
    """Same signal from multiple detectors should merge with severity escalation."""
    call_count = {"n": 0}

    async def mock_call(model, prompt, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return CallResult(
                model=model,
                content="SIGNAL: filler_phrases | SEVERITY: minor | DESCRIPTION: Some filler | EXAMPLES: in conclusion\nSLOP SCORE: 4",
                elapsed_seconds=0.1,
            )
        elif call_count["n"] == 2:
            return CallResult(
                model=model,
                content="SIGNAL: filler_phrases | SEVERITY: major | DESCRIPTION: Lots of filler | EXAMPLES: it's worth noting\nSLOP SCORE: 5",
                elapsed_seconds=0.1,
            )
        else:
            return CallResult(
                model=model,
                content="SIGNAL: filler_phrases | SEVERITY: minor | DESCRIPTION: Filler found | EXAMPLES: as we can see\nSLOP SCORE: 4",
                elapsed_seconds=0.1,
            )

    with patch("claude_commander.server.call_ollama", side_effect=mock_call):
        result = await _detect_slop("Content with some filler.")

    # Should merge into one signal with escalated severity
    filler_signals = [s for s in result.signals if s.signal_type == "filler_phrases"]
    assert len(filler_signals) == 1
    assert filler_signals[0].severity == "major"  # escalated from minor
    # Examples should be deduplicated and merged
    assert len(filler_signals[0].examples) >= 2


@pytest.mark.asyncio
async def test_detect_slop_custom_models():
    async def mock_call(model, prompt, **kw):
        return CallResult(
            model=model, content="No issues.\nSLOP SCORE: 1", elapsed_seconds=0.1
        )

    with patch("claude_commander.server.call_ollama", side_effect=mock_call):
        result = await _detect_slop("Test", detector_models=["glm-5:cloud"])

    assert result.detector_models == ["glm-5:cloud"]


@pytest.mark.asyncio
async def test_detect_slop_invalid_model():
    with pytest.raises(ValueError, match="Unknown model"):
        await _detect_slop("test", detector_models=["fake:model"])


@pytest.mark.asyncio
async def test_detect_slop_verdict_thresholds():
    """Test all four verdict thresholds."""
    for score, expected_verdict in [(1, "clean"), (3, "mild"), (6, "significant"), (9, "severe")]:
        def _make_score_mock(s):
            async def mock_call(model, prompt, **kw):
                return CallResult(
                    model=model, content=f"SLOP SCORE: {s}", elapsed_seconds=0.1
                )
            return mock_call

        with patch("claude_commander.server.call_ollama", side_effect=_make_score_mock(score)):
            result = await _detect_slop("Test content")

        assert result.verdict == expected_verdict, f"Score {score} should give {expected_verdict}, got {result.verdict}"


@pytest.mark.asyncio
async def test_detect_slop_agreement_pct():
    """Agreement should be high when detectors give similar scores."""
    call_count = {"n": 0}

    async def mock_call(model, prompt, **kw):
        call_count["n"] += 1
        score = 5 if call_count["n"] <= 2 else 5  # all same
        return CallResult(
            model=model, content=f"SLOP SCORE: {score}", elapsed_seconds=0.1
        )

    with patch("claude_commander.server.call_ollama", side_effect=mock_call):
        result = await _detect_slop("Test")

    assert result.agreement_pct == 100.0
