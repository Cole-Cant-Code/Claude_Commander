"""Tests for the PipelineStore persistence layer."""

from pathlib import Path

import pytest

from claude_commander.models import PipelineData
from claude_commander.pipeline_store import PipelineStore


@pytest.fixture()
def store(tmp_path: Path) -> PipelineStore:
    """Create a PipelineStore with a temp directory."""
    return PipelineStore(persist_dir=tmp_path / "pipelines")


@pytest.fixture()
def store_with_builtin(tmp_path: Path) -> PipelineStore:
    """Create a store with one builtin and one user pipeline."""
    s = PipelineStore(persist_dir=tmp_path / "pipelines")
    s.save(
        "builtin-pipe",
        PipelineData(
            name="builtin-pipe",
            description="A builtin",
            steps=["glm-5:cloud", "deepseek-v3.2:cloud"],
            builtin=True,
        ),
    )
    s.save(
        "user-pipe",
        PipelineData(
            name="user-pipe",
            description="A user pipeline",
            steps=["glm-5:cloud"],
        ),
    )
    return s


class TestCRUD:
    def test_save_and_load(self, store: PipelineStore) -> None:
        pipeline = PipelineData(
            name="test", description="desc", steps=["glm-5:cloud"]
        )
        store.save("test", pipeline)
        loaded = store.load("test")
        assert loaded is not None
        assert loaded.name == "test"
        assert loaded.steps == ["glm-5:cloud"]

    def test_load_missing_returns_none(self, store: PipelineStore) -> None:
        assert store.load("nonexistent") is None

    def test_has_pipeline(self, store: PipelineStore) -> None:
        assert not store.has_pipeline("x")
        store.save("x", PipelineData(name="x", steps=["glm-5:cloud"]))
        assert store.has_pipeline("x")

    def test_delete_user_pipeline(self, store_with_builtin: PipelineStore) -> None:
        assert store_with_builtin.delete("user-pipe") is True
        assert store_with_builtin.load("user-pipe") is None

    def test_delete_missing_returns_false(self, store: PipelineStore) -> None:
        assert store.delete("no-such") is False

    def test_list_all_sorted(self, store: PipelineStore) -> None:
        store.save("beta", PipelineData(name="beta", steps=["glm-5:cloud"]))
        store.save("alpha", PipelineData(name="alpha", steps=["glm-5:cloud"]))
        names = list(store.list_all().keys())
        assert names == ["alpha", "beta"]


class TestBuiltinProtection:
    def test_cannot_delete_builtin(self, store_with_builtin: PipelineStore) -> None:
        with pytest.raises(ValueError, match="builtin"):
            store_with_builtin.delete("builtin-pipe")
        # Verify it's still there
        assert store_with_builtin.load("builtin-pipe") is not None


class TestClone:
    def test_clone_basic(self, store_with_builtin: PipelineStore) -> None:
        cloned = store_with_builtin.clone("builtin-pipe", "my-clone")
        assert cloned.name == "my-clone"
        assert cloned.steps == ["glm-5:cloud", "deepseek-v3.2:cloud"]
        assert cloned.builtin is False

    def test_clone_with_overrides(self, store_with_builtin: PipelineStore) -> None:
        cloned = store_with_builtin.clone(
            "builtin-pipe",
            "custom",
            description="My custom version",
            steps=["glm-5:cloud"],
        )
        assert cloned.description == "My custom version"
        assert cloned.steps == ["glm-5:cloud"]

    def test_clone_missing_source_raises(self, store: PipelineStore) -> None:
        with pytest.raises(ValueError, match="not found"):
            store.clone("nonexistent", "target")

    def test_clone_is_deletable(self, store_with_builtin: PipelineStore) -> None:
        store_with_builtin.clone("builtin-pipe", "deletable")
        assert store_with_builtin.delete("deletable") is True


class TestNameValidation:
    def test_valid_names(self, store: PipelineStore) -> None:
        for name in ["alpha", "a-b", "a_b", "A1", "test-123"]:
            store.save(name, PipelineData(name=name, steps=["glm-5:cloud"]))
            assert store.has_pipeline(name)

    def test_invalid_names(self, store: PipelineStore) -> None:
        for name in ["has space", "has.dot", "has/slash", ""]:
            with pytest.raises(ValueError, match="Invalid"):
                store.save(name, PipelineData(name=name, steps=["glm-5:cloud"]))


class TestPersistence:
    def test_reload_from_disk(self, tmp_path: Path) -> None:
        persist_dir = tmp_path / "pipelines"
        s1 = PipelineStore(persist_dir=persist_dir)
        s1.save("saved", PipelineData(name="saved", steps=["glm-5:cloud"]))

        # New store instance loads from same directory
        s2 = PipelineStore(persist_dir=persist_dir)
        loaded = s2.load("saved")
        assert loaded is not None
        assert loaded.steps == ["glm-5:cloud"]
