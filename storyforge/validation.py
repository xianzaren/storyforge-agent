from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .utils import run_command


@dataclass
class AuditIssue:
    code: str
    message: str
    severity: str = "error"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunAudit:
    run_dir: str
    passed: bool
    issues: list[AuditIssue]
    facts: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "passed": self.passed,
            "issues": [asdict(issue) for issue in self.issues],
            "facts": self.facts,
        }


REQUIRED_STEPS = ["media_analysis", "script", "storyboard", "assets", "audio", "subtitles", "render", "quality"]
REQUIRED_FILES = [
    "state.json",
    "workflow_settings.json",
    "media_analysis.json",
    "events.jsonl",
    "script.json",
    "storyboard.json",
    "assets_manifest.json",
    "tts_manifest.json",
    "subtitles.srt",
    "final_raw.mp4",
    "final.mp4",
    "quality_report.json",
]


def audit_run(
    run_dir: Path,
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
    require_video_assets: bool = False,
    require_distinct_shots: int = 0,
) -> RunAudit:
    run_dir = run_dir.resolve()
    issues: list[AuditIssue] = []
    facts: dict[str, Any] = {}

    for name in REQUIRED_FILES:
        path = run_dir / name
        if not path.exists() or path.stat().st_size == 0:
            issues.append(AuditIssue("missing_artifact", f"Missing or empty artifact: {name}", details={"path": str(path)}))

    state = _read_json(run_dir / "state.json", issues)
    quality = _read_json(run_dir / "quality_report.json", issues)
    manifest = _read_json(run_dir / "assets_manifest.json", issues)
    settings = _read_json(run_dir / "workflow_settings.json", issues)

    facts["status"] = state.get("status")
    facts["quality_passed"] = quality.get("passed")
    facts["warnings"] = state.get("warnings", [])
    facts["tts_providers"] = state.get("artifacts", {}).get("tts_providers")
    if state.get("status") not in {"completed", "completed_with_warnings"}:
        issues.append(AuditIssue(
            "workflow_status",
            "Workflow did not finish successfully",
            details={"status": state.get("status")},
        ))
    if quality.get("passed") is not True:
        issues.append(AuditIssue("quality_failed", "quality_report.json did not pass", details={"checks": quality.get("checks", {})}))
    if state.get("artifacts", {}).get("subtitles_burned") != "true":
        issues.append(AuditIssue("subtitle_flag", "Final video is not marked as subtitle-burned"))

    _audit_events(run_dir / "events.jsonl", issues, facts)
    _audit_manifest(manifest, issues, facts, require_video_assets, require_distinct_shots)

    video_path = run_dir / "final.mp4"
    if video_path.exists():
        probe = _probe_video(video_path, issues)
        facts["probe"] = probe
        streams = probe.get("streams", [])
        video_stream = next((item for item in streams if item.get("codec_type") == "video"), None)
        audio_stream = next((item for item in streams if item.get("codec_type") == "audio"), None)
        if not video_stream:
            issues.append(AuditIssue("video_stream", "Final MP4 has no video stream"))
        if not audio_stream:
            issues.append(AuditIssue("audio_stream", "Final MP4 has no audio stream"))
        if video_stream and expected_width and video_stream.get("width") != expected_width:
            issues.append(AuditIssue("video_width", "Final video width is incorrect", details={"expected": expected_width, "actual": video_stream.get("width")}))
        if video_stream and expected_height and video_stream.get("height") != expected_height:
            issues.append(AuditIssue("video_height", "Final video height is incorrect", details={"expected": expected_height, "actual": video_stream.get("height")}))
        mean_volume = _mean_volume(video_path, issues)
        facts["mean_volume_db"] = mean_volume
        if mean_volume is not None and mean_volume <= -60:
            issues.append(AuditIssue("silent_audio", "Final video audio appears silent", details={"mean_volume_db": mean_volume}))

    if settings:
        facts["settings"] = settings
        if expected_width and settings.get("width") != expected_width:
            issues.append(AuditIssue("settings_width", "Workflow settings width does not match the requested test width"))
        if expected_height and settings.get("height") != expected_height:
            issues.append(AuditIssue("settings_height", "Workflow settings height does not match the requested test height"))

    return RunAudit(str(run_dir), not any(issue.severity == "error" for issue in issues), issues, facts)


def _read_json(path: Path, issues: list[AuditIssue]) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("top-level JSON value must be an object")
        return value
    except Exception as exc:
        issues.append(AuditIssue("invalid_json", f"Cannot parse {path.name}", details={"error": f"{type(exc).__name__}: {exc}"}))
        return {}


def _audit_events(path: Path, issues: list[AuditIssue], facts: dict[str, Any]) -> None:
    if not path.exists():
        return
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            issues.append(AuditIssue("invalid_event", "Invalid JSONL event", details={"line": line_number, "error": str(exc)}))
    facts["event_count"] = len(events)
    facts["retry_count"] = sum(item.get("event") == "step_retry" for item in events)
    failed_events = [item for item in events if item.get("event") in {"step_failed", "workflow_failed"}]
    if failed_events:
        issues.append(AuditIssue("failure_events", "Run contains failure events", details={"events": failed_events}))
    for step in REQUIRED_STEPS:
        started = next((index for index, item in enumerate(events) if item.get("event") == "step_started" and item.get("step") == step), None)
        completed = next((index for index, item in enumerate(events) if item.get("event") == "step_completed" and item.get("step") == step), None)
        if started is None or completed is None or completed <= started:
            issues.append(AuditIssue("event_sequence", f"Step event sequence is incomplete: {step}"))


def _audit_manifest(
    manifest: dict[str, Any],
    issues: list[AuditIssue],
    facts: dict[str, Any],
    require_video_assets: bool,
    require_distinct_shots: int,
) -> None:
    assignments = manifest.get("assignments", [])
    if not isinstance(assignments, list) or not assignments:
        issues.append(AuditIssue("asset_assignments", "Asset manifest has no scene assignments"))
        return
    video_assignments = [item for item in assignments if item.get("asset_type") == "video"]
    shot_ids = [item.get("shot_index") for item in video_assignments if item.get("shot_index") is not None]
    facts["scene_count"] = len(assignments)
    facts["video_scene_count"] = len(video_assignments)
    facts["selected_shot_ids"] = shot_ids
    facts["looped_scene_count"] = sum(bool(item.get("looped")) for item in assignments)
    facts["source_audio_scene_count"] = sum(bool(item.get("source_audio_used")) for item in assignments)
    if require_video_assets and not video_assignments:
        issues.append(AuditIssue("video_assets", "No video asset was selected during the stage-two test"))
    if require_distinct_shots and len(set(shot_ids)) < require_distinct_shots:
        issues.append(AuditIssue(
            "shot_diversity",
            "Not enough distinct video shots were selected",
            details={"required": require_distinct_shots, "selected": shot_ids},
        ))
    for item in video_assignments:
        if (
            item.get("source_has_audio")
            and not item.get("source_audio_ignored")
            and not item.get("source_audio_used")
        ):
            issues.append(AuditIssue(
                "source_audio",
                "A source video audio track is neither marked as used nor ignored",
                details={"scene_id": item.get("scene_id")},
            ))


def _probe_video(path: Path, issues: list[AuditIssue]) -> dict[str, Any]:
    try:
        result = run_command([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,size,bit_rate:stream=codec_type,codec_name,width,height,sample_rate,channels",
            "-of", "json", str(path),
        ])
        return json.loads(result.stdout)
    except Exception as exc:
        issues.append(AuditIssue("ffprobe", "ffprobe could not inspect final.mp4", details={"error": f"{type(exc).__name__}: {exc}"}))
        return {}


def _mean_volume(path: Path, issues: list[AuditIssue]) -> float | None:
    try:
        result = run_command(["ffmpeg", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"])
        match = re.search(r"mean_volume:\s*(-?(?:inf|\d+(?:\.\d+)?))\s*dB", result.stderr)
        if not match:
            issues.append(AuditIssue("volume_probe", "FFmpeg did not report mean audio volume", severity="warning"))
            return None
        return float(match.group(1))
    except Exception as exc:
        issues.append(AuditIssue("volume_probe", "Cannot measure final audio volume", severity="warning", details={"error": f"{type(exc).__name__}: {exc}"}))
        return None
