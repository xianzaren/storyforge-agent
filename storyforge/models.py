from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Scene:
    scene_id: int
    title: str
    narration: str
    visual_query: str
    on_screen_text: str
    duration_seconds: float
    asset_path: str | None = None
    audio_path: str | None = None
    clip_path: str | None = None
    asset_type: str | None = None
    asset_source_duration: float | None = None
    asset_segment_start_seconds: float = 0.0
    asset_segment_duration: float | None = None
    asset_shot_index: int | None = None
    asset_width: int | None = None
    asset_height: int | None = None
    asset_has_audio: bool = False
    asset_match_score: int = 0
    asset_generated: bool = False
    asset_start_seconds: float = 0.0
    asset_looped: bool = False
    asset_selection_reason: str | None = None
    asset_reuse_index: int = 0


@dataclass
class ProjectState:
    task_id: str
    topic: str
    language: str
    target_duration: int
    status: str = "created"
    current_step: str = "created"
    scenes: list[Scene] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    retries: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectState":
        data = dict(data)
        data["scenes"] = [Scene(**item) for item in data.get("scenes", [])]
        return cls(**data)

    def save(self, path: Path) -> None:
        import json

        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
