"""Tests for the CLI subprocess caller (mocked subprocess layer)."""

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from claude_commander.cli import call_cli, _extract_codex_output
from claude_commander.models import CallResult
from claude_commander.registry import MODELS


# ---------------------------------------------------------------------------
# _extract_codex_output
# ---------------------------------------------------------------------------


class TestExtractCodexOutput:
    def test_extracts_assistant_message(self):
        raw = '{"role": "assistant", "content": "Hello world"}\n'
        assert _extract_codex_output(raw) == "Hello world"

    def test_extracts_last_assistant_message(self):
        raw = (
            '{"role": "assistant", "content": "First"}\n'
            '{"role": "user", "content": "Follow up"}\n'
            '{"role": "assistant", "content": "Second"}\n'
        )
        assert _extract_codex_output(raw) == "Second"

    def test_nested_message_structure(self):
        raw = '{"message": {"role": "assistant", "content": "Nested response"}}\n'
        assert _extract_codex_output(raw) == "Nested response"

    def test_falls_back_to_raw(self):
        raw = "Just plain text output, not JSON"
        assert _extract_codex_output(raw) == raw

    def test_empty_string(self):
        assert _extract_codex_output("") == ""

    def test_ignores_non_assistant_roles(self):
        raw = '{"role": "system", "content": "You are helpful"}\n'
        assert _extract_codex_output(raw) == raw  # falls back to raw

    def test_skips_invalid_json_lines(self):
        raw = (
            "not json\n"
            '{"role": "assistant", "content": "Valid"}\n'
            "also not json\n"
        )
        assert _extract_codex_output(raw) == "Valid"


# ---------------------------------------------------------------------------
# call_cli
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_cli_success():
    """CLI call should capture stdout and return it as content."""
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"Hello from CLI", b""))
    mock_proc.returncode = 0

    with patch("claude_commander.cli.asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await call_cli("claude:cli", "say hello")

    assert result.status == "ok"
    assert result.content == "Hello from CLI"
    assert result.model == "claude:cli"
    assert result.elapsed_seconds >= 0


@pytest.mark.asyncio
async def test_call_cli_with_system_prompt():
    """System prompt should be prepended to the user prompt."""
    captured_cmd = []
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"response", b""))
    mock_proc.returncode = 0

    async def capture_exec(*args, **kwargs):
        captured_cmd.extend(args)
        return mock_proc

    with patch("claude_commander.cli.asyncio.create_subprocess_exec", side_effect=capture_exec):
        await call_cli("claude:cli", "user prompt", system_prompt="Be concise.")

    # The full prompt should contain both system and user parts
    cmd_str = " ".join(captured_cmd)
    assert "Be concise." in cmd_str
    assert "user prompt" in cmd_str


@pytest.mark.asyncio
async def test_call_cli_nonzero_exit():
    """Non-zero exit code should return error status."""
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"partial output", b"something went wrong"))
    mock_proc.returncode = 1

    with patch("claude_commander.cli.asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await call_cli("claude:cli", "test")

    assert result.status == "error"
    assert "exited with code 1" in result.error
    assert result.content == "partial output"


@pytest.mark.asyncio
async def test_call_cli_timeout():
    """Timeout should return error status."""
    import asyncio as aio

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(side_effect=aio.TimeoutError)
    mock_proc.kill = MagicMock()

    with patch("claude_commander.cli.asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await call_cli("claude:cli", "test", timeout_seconds=1)

    assert result.status == "error"
    assert "timed out" in result.error


@pytest.mark.asyncio
async def test_call_cli_unknown_model():
    """Non-CLI model should return error."""
    result = await call_cli("deepseek-v3.2:cloud", "test")
    assert result.status == "error"
    assert "not a registered CLI model" in result.error


@pytest.mark.asyncio
async def test_call_cli_empty_output_warning():
    """Empty output should produce a warning."""
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
    mock_proc.returncode = 0

    with patch("claude_commander.cli.asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await call_cli("claude:cli", "test")

    assert result.status == "ok"
    assert len(result.warnings) == 1
    assert "empty output" in result.warnings[0]


@pytest.mark.asyncio
async def test_call_cli_codex_json_extraction():
    """Codex CLI output should be parsed from JSONL."""
    jsonl = '{"role": "assistant", "content": "The answer is 42"}\n'
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(jsonl.encode(), b""))
    mock_proc.returncode = 0

    with patch("claude_commander.cli.asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await call_cli("codex:cli", "what is the answer?")

    assert result.content == "The answer is 42"


@pytest.mark.asyncio
async def test_call_cli_role_label_and_tags():
    """role_label and tags should be passed through."""
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"output", b""))
    mock_proc.returncode = 0

    with patch("claude_commander.cli.asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await call_cli(
            "claude:cli", "test", role_label="verifier", tags=["qa"]
        )

    assert result.role_label == "verifier"
    assert result.tags == ["qa"]


# ---------------------------------------------------------------------------
# Registry: CLI models
# ---------------------------------------------------------------------------


class TestCliRegistry:
    def test_cli_models_exist(self):
        for mid in ["claude:cli", "gemini:cli", "codex:cli", "kimi:cli"]:
            assert mid in MODELS, f"Missing CLI model: {mid}"

    def test_cli_models_are_flagged(self):
        for mid in ["claude:cli", "gemini:cli", "codex:cli", "kimi:cli"]:
            assert MODELS[mid].is_cli is True

    def test_cli_models_have_commands(self):
        for mid in ["claude:cli", "gemini:cli", "codex:cli", "kimi:cli"]:
            assert len(MODELS[mid].cli_command) > 0

    def test_cli_commands_have_prompt_placeholder(self):
        for mid in ["claude:cli", "gemini:cli", "codex:cli", "kimi:cli"]:
            cmd = " ".join(MODELS[mid].cli_command)
            assert "{prompt}" in cmd, f"{mid} command missing {{prompt}} placeholder"

    def test_non_cli_models_not_flagged(self):
        for mid, info in MODELS.items():
            if mid.endswith(":cloud"):
                assert info.is_cli is False, f"{mid} should not be CLI"


# ---------------------------------------------------------------------------
# Integration: _call_resolved routes CLI models
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_model_routes_to_cli():
    """call_model with a CLI model ID should route through call_cli, not call_ollama."""
    from claude_commander.server import call_model

    _call_model = call_model.fn

    fake_cli = AsyncMock(
        return_value=CallResult(model="claude:cli", content="from cli", elapsed_seconds=0.5)
    )

    with patch("claude_commander.server.call_cli", fake_cli), \
         patch("claude_commander.server.call_ollama", side_effect=AssertionError("should not be called")):
        result = await _call_model("claude:cli", "test prompt")

    assert result.content == "from cli"
    fake_cli.assert_called_once()


@pytest.mark.asyncio
async def test_call_model_routes_ollama_for_cloud():
    """call_model with an Ollama model should route through call_ollama, not call_cli."""
    from claude_commander.server import call_model

    _call_model = call_model.fn

    fake_ollama = AsyncMock(
        return_value=CallResult(model="glm-5:cloud", content="from ollama", elapsed_seconds=0.1)
    )

    with patch("claude_commander.server.call_ollama", fake_ollama), \
         patch("claude_commander.server.call_cli", side_effect=AssertionError("should not be called")):
        result = await _call_model("glm-5:cloud", "test prompt")

    assert result.content == "from ollama"
    fake_ollama.assert_called_once()


# ---------------------------------------------------------------------------
# exec_task tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exec_task_success():
    """exec_task should delegate to call_cli and return ExecTaskResult."""
    from claude_commander.server import exec_task

    _exec_task = exec_task.fn

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"Done: created file.py", b""))
    mock_proc.returncode = 0

    with patch("claude_commander.cli.asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await _exec_task("create file.py", agent="codex:cli", working_dir="/tmp")

    assert result.status == "ok"
    assert result.output == "Done: created file.py"
    assert result.agent == "codex:cli"
    assert result.working_dir == "/tmp"


@pytest.mark.asyncio
async def test_exec_task_invalid_agent():
    """exec_task with unknown agent should return error."""
    from claude_commander.server import exec_task

    _exec_task = exec_task.fn

    result = await _exec_task("do stuff", agent="fake:cli")
    assert result.status == "error"
    assert "Unknown agent" in result.error


@pytest.mark.asyncio
async def test_exec_task_bad_directory():
    """exec_task with nonexistent directory should return error."""
    from claude_commander.server import exec_task

    _exec_task = exec_task.fn

    result = await _exec_task("do stuff", agent="codex:cli", working_dir="/nonexistent/path/xyz")
    assert result.status == "error"
    assert "does not exist" in result.error


@pytest.mark.asyncio
async def test_exec_task_passes_cwd():
    """exec_task should pass cwd to the subprocess."""
    captured_kwargs = {}
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))
    mock_proc.returncode = 0

    async def capture_exec(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_proc

    from claude_commander.server import exec_task

    _exec_task = exec_task.fn

    with patch("claude_commander.cli.asyncio.create_subprocess_exec", side_effect=capture_exec):
        await _exec_task("test task", agent="claude:cli", working_dir="/tmp")

    assert captured_kwargs.get("cwd") == "/tmp"


@pytest.mark.asyncio
async def test_exec_task_default_agent():
    """exec_task defaults to codex:cli."""
    from claude_commander.server import exec_task

    _exec_task = exec_task.fn

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"done", b""))
    mock_proc.returncode = 0

    with patch("claude_commander.cli.asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await _exec_task("build it", working_dir="/tmp")

    assert result.agent == "codex:cli"
    assert result.status == "ok"
