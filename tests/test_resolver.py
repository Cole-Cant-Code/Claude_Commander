"""Tests for the profile resolver module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from claude_commander.models import ProfileData
from claude_commander.profile_store import ProfileStore
from claude_commander.resolver import _UNSET, merge_overrides, resolve


@pytest.fixture()
def tmp_store(tmp_path: Path) -> ProfileStore:
    """Create a ProfileStore with a temp directory and no auto-seeding."""
    store = ProfileStore(persist_dir=tmp_path / "profiles", seed=False)
    store.save(
        "test-profile",
        ProfileData(
            model="deepseek-v3.2:cloud",
            temperature=0.3,
            system_prompt="You are a test assistant.",
            role_label="tester",
            description="A test profile",
        ),
    )
    return store


class TestResolve:
    def test_resolve_known_profile(self, tmp_store: ProfileStore) -> None:
        with patch("claude_commander.resolver.get_profile_store", return_value=tmp_store):
            model_id, params = resolve("test-profile")
        assert model_id == "deepseek-v3.2:cloud"
        assert params["temperature"] == 0.3
        assert params["system_prompt"] == "You are a test assistant."
        assert params["role_label"] == "tester"
        # Metadata fields stripped
        assert "description" not in params
        assert "parent" not in params
        assert "builtin" not in params
        assert "tags" not in params

    def test_resolve_raw_model_id(self, tmp_store: ProfileStore) -> None:
        with patch("claude_commander.resolver.get_profile_store", return_value=tmp_store):
            model_id, params = resolve("glm-5:cloud")
        assert model_id == "glm-5:cloud"
        assert params == {}

    def test_resolve_unknown_name_treated_as_raw(self, tmp_store: ProfileStore) -> None:
        with patch("claude_commander.resolver.get_profile_store", return_value=tmp_store):
            model_id, params = resolve("nonexistent-anything")
        assert model_id == "nonexistent-anything"
        assert params == {}


class TestMergeOverrides:
    def test_unset_preserves_profile_defaults(self) -> None:
        profile = {"temperature": 0.3, "system_prompt": "hello"}
        merged = merge_overrides(profile, temperature=_UNSET, system_prompt=_UNSET)
        assert merged["temperature"] == 0.3
        assert merged["system_prompt"] == "hello"

    def test_caller_values_override_profile(self) -> None:
        profile = {"temperature": 0.3, "system_prompt": "hello"}
        merged = merge_overrides(profile, temperature=0.9, system_prompt="world")
        assert merged["temperature"] == 0.9
        assert merged["system_prompt"] == "world"

    def test_mixed_unset_and_provided(self) -> None:
        profile = {"temperature": 0.3, "top_p": 0.9, "max_tokens": 4096}
        merged = merge_overrides(
            profile,
            temperature=0.5,
            top_p=_UNSET,
            max_tokens=_UNSET,
        )
        assert merged["temperature"] == 0.5
        assert merged["top_p"] == 0.9
        assert merged["max_tokens"] == 4096

    def test_new_keys_from_caller(self) -> None:
        profile = {"temperature": 0.3}
        merged = merge_overrides(profile, extra_key="new_value")
        assert merged["extra_key"] == "new_value"
        assert merged["temperature"] == 0.3

    def test_none_is_a_valid_override(self) -> None:
        """None is not _UNSET — it's a real value that should override."""
        profile = {"system_prompt": "hello"}
        merged = merge_overrides(profile, system_prompt=None)
        assert merged["system_prompt"] is None


class TestUnsetSentinel:
    def test_unset_identity(self) -> None:
        assert _UNSET is _UNSET

    def test_unset_is_not_none(self) -> None:
        assert _UNSET is not None

    def test_unset_is_falsy_object(self) -> None:
        # object() instances are truthy
        assert _UNSET
