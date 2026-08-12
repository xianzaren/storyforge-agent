from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .media import MediaAnalysis, MediaAnalysisTool
from .models import ProjectState, Scene
from .tools import AssetTool, QualityTool, RenderTool, ScriptTool, StoryboardTool, SubtitleTool, TTSTool
from .utils import ffprobe_duration, slugify, write_json


@dataclass
class WorkflowConfig:
    runs_dir: Path = Path("artifacts/runs")
    assets_dir: Path | None = None
    width: int = 1280
    height: int = 720
    max_retries: int = 1
    enable_scene_detection: bool = True
    scene_detection_threshold: float = 0.35
    max_video_segments: int = 24
    fit_mode: str = "pad"
    transition_seconds: float = 0.35
    use_unmatched_assets: bool = True
    subtitle_source: str = "generated_narration"
    audio_mode: str = "narration_replace"
    source_audio_volume: float = 1.0
    narration_volume: float = 1.0


class WorkflowAgent:
    def __init__(self, config: WorkflowConfig | None = None, on_event: Callable[[str, dict], None] | None = None) -> None:
        self.config = config or WorkflowConfig()
        self.on_event = on_event
        if self.config.subtitle_source not in {"generated_narration", "source_audio"}:
            raise ValueError("subtitle_source must be generated_narration or source_audio")
        if self.config.audio_mode not in {"narration_replace", "source_original", "source_narration_mix"}:
            raise ValueError("Unsupported audio_mode")
        self.media_tool = MediaAnalysisTool()
        self.media_analyses: list[MediaAnalysis] = []
        self.script_tool = ScriptTool()
        self.storyboard_tool = StoryboardTool()
        self.asset_tool = AssetTool(
            self.config.width,
            self.config.height,
            enable_scene_detection=self.config.enable_scene_detection,
            scene_detection_threshold=self.config.scene_detection_threshold,
            max_video_segments=self.config.max_video_segments,
            use_unmatched_assets=self.config.use_unmatched_assets,
        )
        self.tts_tool = TTSTool()
        self.subtitle_tool = SubtitleTool()
        self.render_tool = RenderTool()
        self.quality_tool = QualityTool()

    def run(self, topic: str, target_duration: int = 30, language: str = "en") -> ProjectState:
        task_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{slugify(topic)}-{uuid.uuid4().hex[:5]}"
        run_dir = self.config.runs_dir / task_id
        for name in ["assets", "audio", "clips"]:
            (run_dir / name).mkdir(parents=True, exist_ok=True)
        state = ProjectState(task_id=task_id, topic=topic, language=language, target_duration=target_duration)
        settings_path = run_dir / "workflow_settings.json"
        write_json(settings_path, {
            "width": self.config.width,
            "height": self.config.height,
            "fit_mode": self.config.fit_mode,
            "transition_seconds": self.config.transition_seconds,
            "enable_scene_detection": self.config.enable_scene_detection,
            "scene_detection_threshold": self.config.scene_detection_threshold,
            "max_video_segments": self.config.max_video_segments,
            "max_retries": self.config.max_retries,
            "use_unmatched_assets": self.config.use_unmatched_assets,
            "subtitle_source": self.config.subtitle_source,
            "audio_mode": self.config.audio_mode,
            "source_audio_volume": self.config.source_audio_volume,
            "narration_volume": self.config.narration_volume,
        })
        state.artifacts["workflow_settings"] = str(settings_path)

        try:
            self._step(state, run_dir, "media_analysis", lambda: self._analyze_media(state, run_dir))
            self._step(state, run_dir, "script", lambda: self._create_script(state, run_dir))
            self._step(state, run_dir, "storyboard", lambda: self._create_storyboard(state, run_dir))
            self._step(state, run_dir, "assets", lambda: self._resolve_assets(state, run_dir))
            self._step(state, run_dir, "audio", lambda: self._create_audio(state, run_dir))
            self._step(state, run_dir, "subtitles", lambda: self._create_subtitles(state, run_dir))
            self._step(state, run_dir, "render", lambda: self._render(state, run_dir))
            self._step(state, run_dir, "quality", lambda: self._quality(state, run_dir))
            state.status = (
                "completed"
                if state.artifacts.get("quality_passed") == "true" and not state.warnings
                else "completed_with_warnings"
            )
            state.current_step = "done"
            self._event(run_dir, "workflow_completed", {"status": state.status, "warnings": state.warnings})
        except Exception as exc:
            state.status = "failed"
            state.error = f"{type(exc).__name__}: {exc}"
            self._event(run_dir, "workflow_failed", {"error": state.error})
        finally:
            state.save(run_dir / "state.json")
        return state

    def _analyze_media(self, state: ProjectState, run_dir: Path) -> None:
        transcription_requested = self.config.subtitle_source == "source_audio"
        self.media_analyses = self.media_tool.run(
            self.config.assets_dir,
            language=state.language,
            transcribe=transcription_requested,
        )
        manifest_path = self.media_tool.write_manifest(run_dir / "media_analysis.json", self.media_analyses)
        state.artifacts["media_analysis"] = str(manifest_path)
        transcript_segments = sum(len(item.transcript_segments) for item in self.media_analyses)
        state.artifacts["transcription_provider"] = ",".join(sorted({
            item.transcription_provider for item in self.media_analyses
        })) or "none"
        self._event(run_dir, "media_analyzed", {
            "video_count": len(self.media_analyses),
            "audio_video_count": sum(item.has_audio for item in self.media_analyses),
            "transcript_segments": transcript_segments,
            "transcription_requested": transcription_requested,
        })
        if transcription_requested and not transcript_segments:
            errors = [item.transcription_error for item in self.media_analyses if item.transcription_error]
            reason = errors[0] if errors else "No uploaded video with recognisable speech was found."
            state.warnings.append(
                "Source-audio subtitles were requested but transcription was unavailable; "
                f"the workflow will use the topic script instead. {reason}"
            )
            self._event(run_dir, "transcription_fallback", {"reason": reason})

    def _step(self, state: ProjectState, run_dir: Path, name: str, action: Callable[[], None]) -> None:
        for attempt in range(self.config.max_retries + 1):
            state.current_step = name
            state.status = "running"
            state.error = None
            state.save(run_dir / "state.json")
            self._event(run_dir, "step_started", {"step": name, "attempt": attempt + 1})
            try:
                action()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                state.error = error
                state.save(run_dir / "state.json")
                self._event(run_dir, "step_failed", {"step": name, "attempt": attempt + 1, "error": error})
                if attempt >= self.config.max_retries:
                    raise
                retry_number = attempt + 1
                state.retries[name] = retry_number
                self._event(run_dir, "step_retry", {"step": name, "retry": retry_number})
                continue

            state.error = None
            if attempt:
                message = f"Step '{name}' recovered after {attempt} retry attempt(s)."
                if message not in state.warnings:
                    state.warnings.append(message)
            state.save(run_dir / "state.json")
            self._event(run_dir, "step_completed", {"step": name, "attempt": attempt + 1})
            return

    def _create_script(self, state: ProjectState, run_dir: Path) -> None:
        transcript_scenes = []
        if self.config.subtitle_source == "source_audio":
            transcript_scenes = self.media_tool.transcript_scenes(
                self.media_analyses,
                target_duration=state.target_duration,
                language=state.language,
            )
        if transcript_scenes:
            state.scenes = transcript_scenes
            provider = "source-transcript"
        else:
            state.scenes = self.script_tool.run(state.topic, state.target_duration, state.language)
            provider = self.script_tool.last_provider
        path = run_dir / "script.json"
        write_json(path, {
            "topic": state.topic,
            "provider": provider,
            "scenes": [scene.__dict__ for scene in state.scenes],
        })
        state.artifacts["script"] = str(path)
        state.artifacts["script_provider"] = provider
        if not transcript_scenes and self.script_tool.last_error:
            warning = f"LLM generation failed; template fallback was used. {self.script_tool.last_error}"
            state.warnings.append(warning)
            self._event(run_dir, "model_fallback", {"reason": self.script_tool.last_error})

    def _create_storyboard(self, state: ProjectState, run_dir: Path) -> None:
        state.scenes = self.storyboard_tool.run(state.scenes)
        storyboard = run_dir / "storyboard.json"
        write_json(storyboard, {"scenes": [scene.__dict__ for scene in state.scenes]})
        state.artifacts["storyboard"] = str(storyboard)

    def _resolve_assets(self, state: ProjectState, run_dir: Path) -> None:
        records = self.asset_tool.index(self.config.assets_dir)
        self._event(run_dir, "assets_indexed", {
            "records": len(records),
            "image_records": sum(record.media_type == "image" for record in records),
            "video_segments": sum(record.media_type == "video" for record in records),
            "scene_detection": self.config.enable_scene_detection,
        })
        for warning in self.asset_tool.last_warnings:
            state.warnings.append(warning)
            self._event(run_dir, "asset_index_warning", {"warning": warning})
        video_duration_by_path: dict[str, float] = {}
        for record in records:
            if record.media_type == "video" and record.source_duration_seconds:
                key = str(record.path.resolve())
                video_duration_by_path[key] = max(
                    video_duration_by_path.get(key, 0.0),
                    record.source_duration_seconds,
                )
        available_video_duration = sum(video_duration_by_path.values())
        if video_duration_by_path and available_video_duration + 0.05 < state.target_duration:
            warning = (
                f"Uploaded video duration is about {available_video_duration:.1f}s, shorter than the "
                f"{state.target_duration}s target; some source footage will be reused or looped."
            )
            state.warnings.append(warning)
            self._event(run_dir, "insufficient_video_coverage", {
                "available_video_seconds": round(available_video_duration, 3),
                "target_seconds": state.target_duration,
            })
        used_assets: set[str] = set()
        asset_usage_counts: dict[str, int] = {}
        for scene in state.scenes:
            if scene.source_media_path and scene.transcript_start_seconds is not None:
                source_path = Path(scene.source_media_path)
                source_record = next((record for record in records if record.path.resolve() == source_path.resolve()), None)
                if source_record:
                    scene.asset_path = str(source_path)
                    scene.asset_type = "video"
                    scene.asset_source_duration = source_record.source_duration_seconds or source_record.duration_seconds
                    scene.asset_segment_start_seconds = scene.transcript_start_seconds
                    scene.asset_segment_duration = max(
                        0.35,
                        (scene.transcript_end_seconds or scene.transcript_start_seconds + scene.duration_seconds)
                        - scene.transcript_start_seconds,
                    )
                    scene.asset_shot_index = None
                    scene.asset_width = source_record.width
                    scene.asset_height = source_record.height
                    scene.asset_has_audio = source_record.has_audio
                    scene.asset_match_score = 1
                    scene.asset_generated = False
                    scene.asset_selection_reason = "source_transcript_timeline"
                    scene.asset_reuse_index = asset_usage_counts.get(str(source_path.resolve()), 0)
                    asset_usage_counts[str(source_path.resolve())] = scene.asset_reuse_index + 1
                    continue
            path, generated = self.asset_tool.match_or_create(
                scene,
                records,
                run_dir / "assets",
                used_assets=used_assets,
                usage_counts=asset_usage_counts,
            )
            scene.asset_path = str(path)
            if generated:
                state.retries["asset_recovery"] = state.retries.get("asset_recovery", 0) + 1
                self._event(run_dir, "local_recovery", {"scene_id": scene.scene_id, "reason": "no_matching_asset", "action": "generated_title_card"})
            elif scene.asset_selection_reason == "unmatched_user_fallback":
                self._event(run_dir, "unmatched_asset_fallback", {
                    "scene_id": scene.scene_id,
                    "asset_path": scene.asset_path,
                    "action": "used_uploaded_asset_without_keyword_match",
                })
        if any(scene.asset_selection_reason == "unmatched_user_fallback" for scene in state.scenes):
            state.warnings.append(
                "Uploaded assets were used without a filename/tag match. "
                "Add metadata.json tags for more relevant automatic selection."
            )
        storyboard = Path(state.artifacts["storyboard"])
        write_json(storyboard, {"scenes": [scene.__dict__ for scene in state.scenes]})
        manifest_path = run_dir / "assets_manifest.json"
        write_json(manifest_path, {
            "indexed_assets": [record.to_dict() for record in records],
            "assignments": [self._asset_assignment(scene) for scene in state.scenes],
        })
        state.artifacts["assets_manifest"] = str(manifest_path)

    def _create_audio(self, state: ProjectState, run_dir: Path) -> None:
        providers = []
        manifest = []
        for scene in state.scenes:
            scene.audio_mode = self.config.audio_mode
            can_use_source = scene.asset_type == "video" and scene.asset_has_audio and scene.asset_path
            if self.config.audio_mode == "source_original" and can_use_source:
                actual_path = Path(scene.asset_path or "")
                provider = "source-original"
                scene.audio_path = str(actual_path)
                scene.source_audio_used = True
            else:
                audio_path = run_dir / "audio" / f"scene_{scene.scene_id:02}.wav"
                actual_path, provider = self.tts_tool.run(
                    scene.narration, audio_path, state.language, scene.duration_seconds
                )
                scene.audio_path = str(actual_path)
                scene.duration_seconds = max(scene.duration_seconds, ffprobe_duration(actual_path) + 0.15)
                scene.source_audio_used = bool(
                    self.config.audio_mode == "source_narration_mix" and can_use_source
                )
                if self.config.audio_mode == "source_original" and not can_use_source:
                    warning = f"Scene {scene.scene_id} has no usable source audio; generated narration was used."
                    state.warnings.append(warning)
                    self._event(run_dir, "source_audio_fallback", {"scene_id": scene.scene_id})
            providers.append(provider)
            manifest.append({
                "scene_id": scene.scene_id,
                "provider": provider,
                "audio_path": str(actual_path),
                "duration_seconds": scene.duration_seconds,
                "audio_mode": scene.audio_mode,
                "source_audio_used": scene.source_audio_used,
            })
        manifest_path = run_dir / "tts_manifest.json"
        write_json(manifest_path, {"scenes": manifest})
        state.artifacts["tts_manifest"] = str(manifest_path)
        state.artifacts["tts_providers"] = ",".join(sorted(set(providers)))
        if "silent-fallback" in providers:
            state.warnings.append("TTS provider unavailable for at least one scene; a silent placeholder track was used.")

    def _create_subtitles(self, state: ProjectState, run_dir: Path) -> None:
        path = self.subtitle_tool.run(state.scenes, run_dir / "subtitles.srt")
        state.artifacts["subtitles"] = str(path)

    def _render(self, state: ProjectState, run_dir: Path) -> None:
        clips = []
        for scene in state.scenes:
            path = run_dir / "clips" / f"scene_{scene.scene_id:02}.mp4"
            self.render_tool.render_scene(
                scene,
                path,
                self.config.width,
                self.config.height,
                fit_mode=self.config.fit_mode,
                transition_seconds=self.config.transition_seconds,
                audio_mode=scene.audio_mode,
                source_audio_volume=self.config.source_audio_volume,
                narration_volume=self.config.narration_volume,
            )
            scene.clip_path = str(path)
            clips.append(path)
        raw_video = self.render_tool.concatenate(clips, run_dir / "final_raw.mp4")
        final_video = self.render_tool.burn_subtitles(
            raw_video,
            Path(state.artifacts["subtitles"]),
            run_dir / "final.mp4",
        )
        state.artifacts["raw_video"] = str(raw_video)
        state.artifacts["video"] = str(final_video)
        state.artifacts["subtitles_burned"] = "true"
        manifest_path = Path(state.artifacts["assets_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["assignments"] = [self._asset_assignment(scene) for scene in state.scenes]
        write_json(manifest_path, manifest)

    @staticmethod
    def _asset_assignment(scene: Scene) -> dict:
        return {
            "scene_id": scene.scene_id,
            "asset_path": scene.asset_path,
            "asset_type": scene.asset_type,
            "generated": scene.asset_generated,
            "match_score": scene.asset_match_score,
            "source_duration_seconds": scene.asset_source_duration,
            "segment_start_seconds": round(scene.asset_segment_start_seconds, 3),
            "segment_duration_seconds": scene.asset_segment_duration,
            "shot_index": scene.asset_shot_index,
            "source_width": scene.asset_width,
            "source_height": scene.asset_height,
            "source_has_audio": scene.asset_has_audio,
            "source_audio_ignored": scene.asset_type == "video" and scene.asset_has_audio and not scene.source_audio_used,
            "source_audio_used": scene.source_audio_used,
            "audio_mode": scene.audio_mode,
            "subtitle_source": scene.subtitle_source,
            "transcript_start_seconds": scene.transcript_start_seconds,
            "transcript_end_seconds": scene.transcript_end_seconds,
            "trim_start_seconds": round(scene.asset_start_seconds, 3),
            "output_duration_seconds": round(scene.duration_seconds, 3),
            "looped": scene.asset_looped,
            "selection_reason": scene.asset_selection_reason,
            "reuse_index": scene.asset_reuse_index,
            "clip_path": scene.clip_path,
        }

    def _quality(self, state: ProjectState, run_dir: Path) -> None:
        subtitles_path = Path(state.artifacts["subtitles"])
        report = self.quality_tool.run(
            Path(state.artifacts["video"]),
            state.scenes,
            subtitles_path,
            expected_width=self.config.width,
            expected_height=self.config.height,
        )
        if not report["passed"] and state.retries.get("quality", 0) < self.config.max_retries:
            state.retries["quality"] = state.retries.get("quality", 0) + 1
            self._event(run_dir, "quality_retry", {"attempt": state.retries["quality"], "report": report})
            self._render(state, run_dir)
            report = self.quality_tool.run(
                Path(state.artifacts["video"]),
                state.scenes,
                subtitles_path,
                expected_width=self.config.width,
                expected_height=self.config.height,
            )
        write_json(run_dir / "quality_report.json", report)
        state.artifacts["quality_report"] = str(run_dir / "quality_report.json")
        state.artifacts["quality_passed"] = str(report["passed"]).lower()
        if not report["passed"]:
            state.warnings.append("Quality checks did not all pass; inspect quality_report.json.")

    def _event(self, run_dir: Path, event: str, payload: dict) -> None:
        item = {"timestamp": time.time(), "event": event, **payload}
        with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        if self.on_event:
            try:
                self.on_event(event, item)
            except Exception:
                # UI callbacks are observers and must not break the media workflow.
                pass
