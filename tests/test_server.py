"""Tests for the MCP server tools (mocked Ollama layer)."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from claude_commander.models import CallResult
from claude_commander.server import auto_call, call_model, health_check, list_models, swarm

# FastMCP's @mcp.tool() wraps functions in FunctionTool objects.
# Access the original async functions via .fn for direct testing.
_auto_call = auto_call.fn
_call_model = call_model.fn
_swarm = swarm.fn
_list_models = list_models.fn
_health_check = health_check.fn


@pytest.mark.asyncio
async def test_call_model_valid():
    fake = CallResult(model="deepseek-v3.2:cloud", content="hi", elapsed_seconds=0.5)
    with patch("claude_commander.server.call_ollama", new_callable=AsyncMock, return_value=fake):
        result = await _call_model("deepseek-v3.2:cloud", "say hi")
    assert result.content == "hi"
    assert result.model == "deepseek-v3.2:cloud"


@pytest.mark.asyncio
async def test_call_model_invalid():
    with pytest.raises(ValueError, match="Unknown model"):
        await _call_model("fake:model", "hello")


@pytest.mark.asyncio
async def test_auto_call_code_routing():
    async def mock_call(model, prompt, **kw):
        return CallResult(model=model, content=f"from {model}", elapsed_seconds=0.1)

    with patch("claude_commander.server.call_ollama", side_effect=mock_call):
        result = await _auto_call("Review this Python function", task="code", max_attempts=1)

    assert result.resolved_task == "code"
    assert result.selected_model == "qwen3-coder-next:cloud"
    assert result.result.status == "ok"
    assert result.fallback_used is False


@pytest.mark.asyncio
async def test_auto_call_fallback_after_error():
    async def flaky_call(model, prompt, **kw):
        if model == "deepseek-v3.2:cloud":
            return CallResult(model=model, status="error", error="timeout", elapsed_seconds=0.1)
        return CallResult(model=model, content=f"ok from {model}", elapsed_seconds=0.1)

    with patch("claude_commander.server.call_ollama", side_effect=flaky_call):
        result = await _auto_call("Analyze tradeoffs", task="reasoning", max_attempts=2)

    assert result.resolved_task == "reasoning"
    assert result.fallback_used is True
    assert len(result.attempted_models) == 2
    assert result.result.status == "ok"
    assert result.selected_model != "deepseek-v3.2:cloud"


@pytest.mark.asyncio
async def test_auto_call_model_override():
    async def mock_call(model, prompt, **kw):
        return CallResult(model=model, content="ok", elapsed_seconds=0.1)

    with patch("claude_commander.server.call_ollama", side_effect=mock_call):
        result = await _auto_call("hello", models=["glm-4.7:cloud"], max_attempts=1)

    assert result.candidate_models == ["glm-4.7:cloud"]
    assert result.selected_model == "glm-4.7:cloud"


@pytest.mark.asyncio
async def test_auto_call_uses_routing_profile_config(tmp_path):
    config_path = tmp_path / "routing.json"
    config_path.write_text(
        json.dumps(
            {
                "default_profile": "speedy",
                "profiles": {
                    "speedy": {"code": {"balanced": ["glm-4.7:cloud"]}},
                    "quality-first": {
                        "code": {"balanced": ["qwen3-coder-next:cloud"]}
                    },
                },
            }
        )
    )

    async def mock_call(model, prompt, **kw):
        return CallResult(model=model, content=f"ok from {model}", elapsed_seconds=0.1)

    with patch.dict(
        "os.environ",
        {"CLAUDE_COMMANDER_ROUTING_CONFIG": str(config_path)},
        clear=False,
    ):
        with patch("claude_commander.server.call_ollama", side_effect=mock_call):
            result = await _auto_call(
                "Review this Python function",
                task="code",
                strategy="balanced",
                routing_profile="speedy",
                max_attempts=1,
            )

    assert result.selected_model == "glm-4.7:cloud"
    assert result.routing_profile == "speedy"
    assert result.routing_source.startswith("file:")


@pytest.mark.asyncio
async def test_auto_call_budget_exhausted_before_fallback():
    async def slow_error(model, prompt, **kw):
        await asyncio.sleep(0.01)
        return CallResult(model=model, status="error", error="timeout", elapsed_seconds=0.01)

    with patch("claude_commander.server.call_ollama", side_effect=slow_error):
        result = await _auto_call(
            "Analyze tradeoffs",
            task="reasoning",
            max_attempts=3,
            max_time_ms=1,
        )

    assert result.budget_exhausted is True
    assert len(result.attempted_models) == 1
    assert "budget" in (result.result.error or "").lower()


@pytest.mark.asyncio
async def test_auto_call_invalid_routing_profile(tmp_path):
    config_path = tmp_path / "routing.json"
    config_path.write_text(json.dumps({"profiles": {"default": {}}}))

    with patch.dict(
        "os.environ",
        {"CLAUDE_COMMANDER_ROUTING_CONFIG": str(config_path)},
        clear=False,
    ):
        with pytest.raises(ValueError, match="Unknown routing_profile"):
            await _auto_call("hello", routing_profile="missing")


@pytest.mark.asyncio
async def test_auto_call_rejects_routing_profile_with_models():
    with pytest.raises(ValueError, match="routing_profile cannot be used together"):
        await _auto_call("hello", models=["glm-4.7:cloud"], routing_profile="default")


@pytest.mark.asyncio
async def test_auto_call_invalid_task():
    with pytest.raises(ValueError, match="Unknown task"):
        await _auto_call("hello", task="not-a-task")


@pytest.mark.asyncio
async def test_swarm_subset():
    async def mock_call(model, prompt, **kw):
        return CallResult(model=model, content=f"from {model}", elapsed_seconds=0.1)

    with patch("claude_commander.server.call_ollama", side_effect=mock_call):
        result = await _swarm("test", models=["glm-5:cloud", "deepseek-v3.2:cloud"])

    assert result.models_called == 2
    assert result.models_succeeded == 2
    assert result.models_failed == 0
    assert len(result.results) == 2


@pytest.mark.asyncio
async def test_swarm_defaults_to_six():
    async def mock_call(model, prompt, **kw):
        return CallResult(model=model, content="ok", elapsed_seconds=0.01)

    with patch("claude_commander.server.call_ollama", side_effect=mock_call):
        result = await _swarm("test")

    assert result.models_called == 6


@pytest.mark.asyncio
async def test_swarm_count_overrides_default():
    async def mock_call(model, prompt, **kw):
        return CallResult(model=model, content="ok", elapsed_seconds=0.01)

    with patch("claude_commander.server.call_ollama", side_effect=mock_call):
        result = await _swarm("test", count=13)

    assert result.models_called == 13


@pytest.mark.asyncio
async def test_swarm_invalid_model():
    with pytest.raises(ValueError, match="Unknown model"):
        await _swarm("test", models=["fake:model"])


@pytest.mark.asyncio
async def test_list_models():
    with patch("claude_commander.server.check_ollama", new_callable=AsyncMock, return_value=True):
        result = await _list_models()
    assert len(result) == 17
    assert all(m.available for m in result)


@pytest.mark.asyncio
async def test_health_check_connected():
    with patch("claude_commander.server.check_ollama", new_callable=AsyncMock, return_value=True):
        result = await _health_check()
    assert result.ollama_connected is True
    assert result.registered_models == 17
    assert result.version == "0.1.0"


@pytest.mark.asyncio
async def test_health_check_disconnected():
    with patch(
        "claude_commander.server.check_ollama", new_callable=AsyncMock, return_value=False
    ):
        result = await _health_check()
    assert result.ollama_connected is False
