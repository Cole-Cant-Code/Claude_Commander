"""Tests for the Ollama client (mocked, no real server needed)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claude_commander.ollama import call_ollama, check_ollama


@pytest.mark.asyncio
async def test_call_ollama_success():
    mock_resp = AsyncMock()
    mock_resp.json = AsyncMock(
        return_value={"message": {"content": "Hello!"}}
    )
    mock_resp.status = 200
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("claude_commander.ollama.aiohttp.ClientSession", return_value=mock_session):
        result = await call_ollama("glm-5:cloud", "say hello")

    assert result.status == "ok"
    assert result.content == "Hello!"
    assert result.model == "glm-5:cloud"
    assert result.elapsed_seconds >= 0


@pytest.mark.asyncio
async def test_call_ollama_error():
    mock_session = MagicMock()
    mock_session.post = MagicMock(side_effect=Exception("connection refused"))
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("claude_commander.ollama.aiohttp.ClientSession", return_value=mock_session):
        result = await call_ollama("glm-5:cloud", "say hello")

    assert result.status == "error"
    assert "connection refused" in result.error


@pytest.mark.asyncio
async def test_check_ollama_connected():
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("claude_commander.ollama.aiohttp.ClientSession", return_value=mock_session):
        assert await check_ollama() is True


@pytest.mark.asyncio
async def test_check_ollama_unreachable():
    mock_session = MagicMock()
    mock_session.get = MagicMock(side_effect=Exception("unreachable"))
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("claude_commander.ollama.aiohttp.ClientSession", return_value=mock_session):
        assert await check_ollama() is False
