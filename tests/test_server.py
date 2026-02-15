"""Tests for the MCP server tools (mocked Ollama layer)."""

from unittest.mock import AsyncMock, patch

import pytest

from claude_commander.models import CallResult
from claude_commander.server import call_model, health_check, list_models, swarm

# FastMCP's @mcp.tool() wraps functions in FunctionTool objects.
# Access the original async functions via .fn for direct testing.
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
