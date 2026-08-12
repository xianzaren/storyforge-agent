from __future__ import annotations

import os
from pathlib import Path

from .llm import CompatibleLLMClient


class OpenAICompatibleVisionTagger:
    """Turn one representative frame into reviewable semantic suggestions."""

    SYSTEM = (
        "You annotate a single representative video frame for an editor. "
        "Return JSON only. Never infer identities, dialogue, events outside the frame, or uncertain facts."
    )
    USER = (
        "Describe only visible evidence. Return: "
        '{"summary":"short factual description",'
        '"tags":["scene","objects","action","shot type"],'
        '"confidence":0.0}. Use the same language as clearly visible text, otherwise Chinese.'
    )

    def __init__(self, client: CompatibleLLMClient | None = None) -> None:
        self.client = client or CompatibleLLMClient()

    @property
    def enabled(self) -> bool:
        return self.client.vision_enabled

    def __call__(self, thumbnail: Path) -> tuple[str, list[str], float, str]:
        payload = self.client.generate_vision_json(thumbnail, self.SYSTEM, self.USER)
        summary = str(payload.get("summary") or "").strip()
        tags = payload.get("tags") or []
        if not summary or not isinstance(tags, list):
            raise ValueError("Vision response must contain summary and tags")
        confidence = float(payload.get("confidence", 0.5))
        provider = f"openai-compatible:{os.getenv('STORYFORGE_VISION_MODEL', 'unknown')}"
        return summary, [str(item) for item in tags[:12]], confidence, provider
