"""Tests for the model registry."""

import pytest

from claude_commander.registry import MODELS, ModelInfo, get_model


def test_registry_has_17_models():
    assert len(MODELS) == 17


def test_all_entries_are_model_info():
    for model in MODELS.values():
        assert isinstance(model, ModelInfo)


def test_model_ids_are_keys():
    for model_id, info in MODELS.items():
        assert model_id == info.model_id


def test_categories_are_valid():
    valid = {"general", "code", "vision", "reasoning", "cli"}
    for model in MODELS.values():
        assert model.category in valid, f"{model.model_id} has invalid category"


def test_get_model_valid():
    m = get_model("deepseek-v3.2:cloud")
    assert m.display_name == "DeepSeek v3.2"
    assert m.category == "reasoning"


def test_get_model_invalid():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("nonexistent:model")


EXPECTED_IDS = [
    "glm-5:cloud",
    "minimax-m2.5:cloud",
    "qwen3-coder-next:cloud",
    "gpt-oss:20b-cloud",
    "gpt-oss:120b-cloud",
    "qwen3-vl:235b-instruct-cloud",
    "qwen3-vl:235b-cloud",
    "kimi-k2-thinking:cloud",
    "qwen3-next:80b-cloud",
    "deepseek-v3.2:cloud",
    "minimax-m2.1:cloud",
    "glm-4.7:cloud",
    "kimi-k2.5:cloud",
    "claude:cli",
    "gemini:cli",
    "codex:cli",
    "kimi:cli",
]


def test_all_expected_models_present():
    for model_id in EXPECTED_IDS:
        assert model_id in MODELS, f"Missing model: {model_id}"
