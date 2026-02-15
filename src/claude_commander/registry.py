"""Hardcoded registry of 13 Ollama cloud models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Metadata for a registered Ollama model."""

    model_id: str
    display_name: str
    category: str  # general, code, vision, reasoning
    strengths: list[str] = Field(default_factory=list)
    default_temperature: float = 0.7
    is_thinking: bool = False


MODELS: dict[str, ModelInfo] = {
    info.model_id: info
    for info in [
        ModelInfo(
            model_id="glm-5:cloud",
            display_name="GLM-5",
            category="general",
            strengths=["academic reasoning", "structured analysis"],
        ),
        ModelInfo(
            model_id="minimax-m2.5:cloud",
            display_name="MiniMax M2.5",
            category="general",
            strengths=["agentic execution", "function calling"],
        ),
        ModelInfo(
            model_id="qwen3-coder-next:cloud",
            display_name="Qwen3 Coder Next",
            category="code",
            strengths=["code generation", "software architecture", "debugging"],
            default_temperature=0.5,
        ),
        ModelInfo(
            model_id="gpt-oss:20b-cloud",
            display_name="GPT-OSS 20B",
            category="general",
            strengths=["fast inference", "instruction following"],
        ),
        ModelInfo(
            model_id="gpt-oss:120b-cloud",
            display_name="GPT-OSS 120B",
            category="general",
            strengths=["general reasoning", "broad coverage"],
        ),
        ModelInfo(
            model_id="qwen3-vl:235b-instruct-cloud",
            display_name="Qwen3 VL 235B Instruct",
            category="vision",
            strengths=["multimodal", "image understanding"],
        ),
        ModelInfo(
            model_id="qwen3-vl:235b-cloud",
            display_name="Qwen3 VL 235B",
            category="vision",
            strengths=["multimodal", "visual reasoning"],
        ),
        ModelInfo(
            model_id="kimi-k2-thinking:cloud",
            display_name="Kimi K2 Thinking",
            category="reasoning",
            strengths=["extended chain-of-thought", "self-correction"],
            default_temperature=0.5,
            is_thinking=True,
        ),
        ModelInfo(
            model_id="qwen3-next:80b-cloud",
            display_name="Qwen3 Next 80B",
            category="general",
            strengths=["balanced generalist", "multilingual"],
        ),
        ModelInfo(
            model_id="deepseek-v3.2:cloud",
            display_name="DeepSeek v3.2",
            category="reasoning",
            strengths=["math", "formal logic", "competitive programming"],
            default_temperature=0.3,
            is_thinking=True,
        ),
        ModelInfo(
            model_id="minimax-m2.1:cloud",
            display_name="MiniMax M2.1",
            category="general",
            strengths=["general purpose", "lightweight"],
        ),
        ModelInfo(
            model_id="glm-4.7:cloud",
            display_name="GLM-4.7",
            category="general",
            strengths=["general purpose", "structured output"],
        ),
        ModelInfo(
            model_id="kimi-k2.5:cloud",
            display_name="Kimi K2.5",
            category="general",
            strengths=["speed", "agentic execution"],
        ),
    ]
}


def get_model(model_id: str) -> ModelInfo:
    """Look up a model by ID, raising ValueError if unknown."""
    if model_id not in MODELS:
        known = ", ".join(sorted(MODELS))
        raise ValueError(f"Unknown model '{model_id}'. Available: {known}")
    return MODELS[model_id]
