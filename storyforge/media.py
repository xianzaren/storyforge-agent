from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .models import Scene
from .utils import require_binary, run_command


@dataclass
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str


@dataclass
class MediaAnalysis:
    path: str
    duration_seconds: float
    width: int | None
    height: int | None
    has_audio: bool
    transcription_provider: str = "not-requested"
    detected_language: str | None = None
    transcript_segments: list[TranscriptSegment] = field(default_factory=list)
    transcription_error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class TranscriptNLPTool:
    """Small deterministic transcript cleaner that works without a model API."""

    FILLER_PREFIX = re.compile(
        r"^(?:(?:um+|uh+|erm+|you know|like|嗯+|呃+|额+|那个)[,，。.!！?？\s]*)+",
        flags=re.IGNORECASE,
    )

    @classmethod
    def clean(cls, text: str) -> str:
        value = re.sub(r"\s+", " ", text).strip()
        value = cls.FILLER_PREFIX.sub("", value).strip()
        # Collapse immediate repeated words produced by hesitant speech.
        value = re.sub(r"\b([A-Za-z0-9']+)\s+\1\b", r"\1", value, flags=re.IGNORECASE)
        value = re.sub(r"([\u4e00-\u9fff]{1,3})[，,\s]+\1", r"\1", value)
        return value.strip(" ,，")

    @classmethod
    def clean_segments(cls, segments: Iterable[TranscriptSegment]) -> list[TranscriptSegment]:
        cleaned: list[TranscriptSegment] = []
        for item in segments:
            text = cls.clean(item.text)
            if text and item.end_seconds > item.start_seconds:
                cleaned.append(TranscriptSegment(
                    max(0.0, float(item.start_seconds)),
                    max(0.0, float(item.end_seconds)),
                    text,
                ))
        return cleaned

    @classmethod
    def caption_segments(
        cls,
        segments: Iterable[TranscriptSegment],
        *,
        max_words: int = 10,
        max_cjk_chars: int = 18,
    ) -> list[TranscriptSegment]:
        """Split long speech segments into readable captions while retaining timing."""
        result: list[TranscriptSegment] = []
        for item in segments:
            text = item.text.strip()
            if not text:
                continue
            sentence_parts = [part.strip() for part in re.split(
                r"(?<=[.!?。！？；;])\s*", text
            ) if part.strip()]
            chunks: list[str] = []
            for part in sentence_parts:
                if re.search(r"[\u4e00-\u9fff]", part):
                    chunks.extend(
                        part[index:index + max_cjk_chars]
                        for index in range(0, len(part), max_cjk_chars)
                    )
                else:
                    words = part.split()
                    chunks.extend(
                        " ".join(words[index:index + max_words])
                        for index in range(0, len(words), max_words)
                    )
            chunks = [chunk for chunk in chunks if chunk]
            if len(chunks) <= 1:
                result.append(item)
                continue
            weights = [max(1, len(re.sub(r"\s+", "", chunk))) for chunk in chunks]
            total_weight = sum(weights)
            total_duration = item.end_seconds - item.start_seconds
            cursor = item.start_seconds
            for index, (chunk, weight) in enumerate(zip(chunks, weights)):
                end = (
                    item.end_seconds
                    if index == len(chunks) - 1
                    else cursor + total_duration * weight / total_weight
                )
                result.append(TranscriptSegment(cursor, end, chunk))
                cursor = end
        return result


Transcriber = Callable[[Path, str], tuple[list[TranscriptSegment], str | None, str]]


class MediaAnalysisTool:
    SUPPORTED_VIDEOS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

    def __init__(self, transcriber: Transcriber | None = None) -> None:
        self.transcriber = transcriber
        self._whisper_model = None

    def run(
        self,
        assets_dir: Path | None,
        *,
        language: str,
        transcribe: bool,
    ) -> list[MediaAnalysis]:
        if not assets_dir or not assets_dir.exists():
            return []
        analyses: list[MediaAnalysis] = []
        for path in sorted(assets_dir.rglob("*")):
            if path.suffix.lower() not in self.SUPPORTED_VIDEOS:
                continue
            item = self._probe(path)
            if transcribe and item.has_audio:
                try:
                    segments, detected_language, provider = (
                        self.transcriber(path, language)
                        if self.transcriber
                        else self._faster_whisper(path, language)
                    )
                    cleaned = TranscriptNLPTool.clean_segments(segments)
                    item.transcript_segments = TranscriptNLPTool.caption_segments(cleaned)
                    item.detected_language = detected_language
                    item.transcription_provider = provider
                    if not item.transcript_segments:
                        item.transcription_error = "No speech segments were detected in the source audio."
                except Exception as exc:
                    item.transcription_provider = "unavailable"
                    item.transcription_error = f"{type(exc).__name__}: {exc}"
            elif transcribe:
                item.transcription_provider = "no-audio"
                item.transcription_error = "The uploaded video has no audio stream."
            analyses.append(item)
        return analyses

    @staticmethod
    def transcript_scenes(
        analyses: list[MediaAnalysis],
        *,
        target_duration: int,
        language: str,
    ) -> list[Scene]:
        candidates = [item for item in analyses if item.transcript_segments]
        if not candidates:
            return []
        source = max(candidates, key=lambda item: sum(
            segment.end_seconds - segment.start_seconds for segment in item.transcript_segments
        ))
        scenes: list[Scene] = []
        output_duration = 0.0
        for segment in source.transcript_segments:
            if output_duration >= target_duration:
                break
            source_duration = max(0.6, segment.end_seconds - segment.start_seconds)
            duration = min(source_duration, target_duration - output_duration)
            if duration < 0.35:
                break
            title = f"原声 {len(scenes) + 1}" if language.lower().startswith("zh") else f"Source {len(scenes) + 1}"
            scenes.append(Scene(
                scene_id=len(scenes) + 1,
                title=title,
                narration=segment.text,
                visual_query=Path(source.path).stem,
                on_screen_text=segment.text[:30],
                duration_seconds=duration,
                source_media_path=source.path,
                transcript_start_seconds=segment.start_seconds,
                transcript_end_seconds=min(segment.end_seconds, segment.start_seconds + duration),
                subtitle_source="source_transcript",
            ))
            output_duration += duration
        return scenes

    @staticmethod
    def write_manifest(path: Path, analyses: list[MediaAnalysis]) -> Path:
        path.write_text(
            json.dumps({"media": [item.to_dict() for item in analyses]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _probe(path: Path) -> MediaAnalysis:
        require_binary("ffprobe")
        result = run_command([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration:stream=codec_type,width,height", "-of", "json", str(path),
        ])
        payload = json.loads(result.stdout)
        streams = payload.get("streams", [])
        video = next((item for item in streams if item.get("codec_type") == "video"), {})
        duration = float(payload.get("format", {}).get("duration") or 0.0)
        return MediaAnalysis(
            path=str(path),
            duration_seconds=duration,
            width=int(video["width"]) if video.get("width") else None,
            height=int(video["height"]) if video.get("height") else None,
            has_audio=any(item.get("codec_type") == "audio" for item in streams),
        )

    def _faster_whisper(
        self,
        path: Path,
        language: str,
    ) -> tuple[list[TranscriptSegment], str | None, str]:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Speech transcription requires optional dependency 'faster-whisper'. "
                "Install it with: pip install faster-whisper"
            ) from exc
        if self._whisper_model is None:
            self._whisper_model = WhisperModel(
                os.getenv("STORYFORGE_WHISPER_MODEL", "tiny"),
                device=os.getenv("STORYFORGE_WHISPER_DEVICE", "cpu"),
                compute_type=os.getenv("STORYFORGE_WHISPER_COMPUTE_TYPE", "int8"),
            )
        requested_language = None if language == "auto" else language.split("-")[0].lower()
        raw_segments, info = self._whisper_model.transcribe(
            str(path),
            language=requested_language,
            vad_filter=True,
            beam_size=5,
        )
        segments = [TranscriptSegment(float(item.start), float(item.end), item.text) for item in raw_segments]
        return segments, getattr(info, "language", requested_language), "faster-whisper"
