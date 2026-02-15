"""Tests for seed profiles and pipelines from defaults.py."""

from pathlib import Path

import pytest

from claude_commander.defaults import (
    _BUILTIN_PIPELINES,
    _BUILTIN_PROFILES,
    seed_pipelines,
    seed_profiles,
)
from claude_commander.pipeline_store import PipelineStore
from claude_commander.profile_store import ProfileStore


@pytest.fixture()
def profile_store(tmp_path: Path) -> ProfileStore:
    return ProfileStore(persist_dir=tmp_path / "profiles", seed=False)


@pytest.fixture()
def pipeline_store(tmp_path: Path) -> PipelineStore:
    return PipelineStore(persist_dir=tmp_path / "pipelines")


class TestSeedProfiles:
    def test_seeds_all_builtin_profiles(self, profile_store: ProfileStore) -> None:
        seed_profiles(profile_store)
        for name in _BUILTIN_PROFILES:
            assert profile_store.has_profile(name), f"Missing profile: {name}"

    def test_expected_profile_count(self, profile_store: ProfileStore) -> None:
        seed_profiles(profile_store)
        assert len(profile_store.list_all()) == 12

    def test_profiles_are_builtin(self, profile_store: ProfileStore) -> None:
        seed_profiles(profile_store)
        for name in _BUILTIN_PROFILES:
            profile = profile_store.load(name)
            assert profile is not None
            assert profile.builtin is True

    def test_idempotent_reseeding(self, profile_store: ProfileStore) -> None:
        seed_profiles(profile_store)
        first_run = profile_store.list_all()
        seed_profiles(profile_store)
        second_run = profile_store.list_all()
        assert first_run == second_run

    def test_does_not_overwrite_existing(self, profile_store: ProfileStore) -> None:
        """If a user has modified a profile with the same name, seeding skips it."""
        from claude_commander.models import ProfileData

        profile_store.save(
            "fast-general",
            ProfileData(model="glm-5:cloud", temperature=0.99),
        )
        seed_profiles(profile_store)
        loaded = profile_store.load("fast-general")
        assert loaded is not None
        assert loaded.temperature == 0.99  # not overwritten

    def test_specific_profile_models(self, profile_store: ProfileStore) -> None:
        seed_profiles(profile_store)
        expected = {
            "fast-general": "glm-4.7:cloud",
            "deep-reasoner": "deepseek-v3.2:cloud",
            "code-specialist": "qwen3-coder-next:cloud",
            "thinking-judge": "kimi-k2-thinking:cloud",
            "creative-writer": "glm-5:cloud",
            "factual-analyst": "gpt-oss:120b-cloud",
            "vision-analyzer": "qwen3-vl:235b-cloud",
            "quick-draft": "gpt-oss:20b-cloud",
        }
        for name, expected_model in expected.items():
            profile = profile_store.load(name)
            assert profile is not None
            assert profile.model == expected_model, f"{name}: expected {expected_model}"


class TestSeedPipelines:
    def test_seeds_all_builtin_pipelines(self, pipeline_store: PipelineStore) -> None:
        seed_pipelines(pipeline_store)
        for name in _BUILTIN_PIPELINES:
            assert pipeline_store.has_pipeline(name), f"Missing pipeline: {name}"

    def test_expected_pipeline_count(self, pipeline_store: PipelineStore) -> None:
        seed_pipelines(pipeline_store)
        assert len(pipeline_store.list_all()) == 6

    def test_pipelines_are_builtin(self, pipeline_store: PipelineStore) -> None:
        seed_pipelines(pipeline_store)
        for name in _BUILTIN_PIPELINES:
            pipeline = pipeline_store.load(name)
            assert pipeline is not None
            assert pipeline.builtin is True

    def test_idempotent_reseeding(self, pipeline_store: PipelineStore) -> None:
        seed_pipelines(pipeline_store)
        first_run = pipeline_store.list_all()
        seed_pipelines(pipeline_store)
        second_run = pipeline_store.list_all()
        assert first_run == second_run

    def test_specific_pipeline_steps(self, pipeline_store: PipelineStore) -> None:
        seed_pipelines(pipeline_store)
        expected = {
            "draft-then-refine": ["quick-draft", "deep-reasoner"],
            "code-review-pipeline": ["code-specialist", "deep-reasoner", "factual-analyst"],
            "creative-to-critical": ["creative-writer", "thinking-judge"],
        }
        for name, expected_steps in expected.items():
            pipeline = pipeline_store.load(name)
            assert pipeline is not None
            assert pipeline.steps == expected_steps
