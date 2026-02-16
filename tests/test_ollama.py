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
async def test_call_ollama_retries_on_empty_content_with_thinking():
    # First response: backend used all token budget on "thinking", leaving empty content.
    mock_resp1 = AsyncMock()
    mock_resp1.json = AsyncMock(return_value={"message": {"content": "", "thinking": "..."}})
    mock_resp1.status = 200
    mock_resp1.__aenter__ = AsyncMock(return_value=mock_resp1)
    mock_resp1.__aexit__ = AsyncMock(return_value=False)

    # Second response: returns the actual content.
    mock_resp2 = AsyncMock()
    mock_resp2.json = AsyncMock(return_value={"message": {"content": "ping"}})
    mock_resp2.status = 200
    mock_resp2.__aenter__ = AsyncMock(return_value=mock_resp2)
    mock_resp2.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(side_effect=[mock_resp1, mock_resp2])
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("claude_commander.ollama.aiohttp.ClientSession", return_value=mock_session):
        result = await call_ollama("gpt-oss:20b-cloud", "say ping", max_tokens=20)

    assert result.status == "ok"
    assert result.content == "ping"
    assert mock_session.post.call_count == 2
    assert result.warnings  # should record the empty-content attempt


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
