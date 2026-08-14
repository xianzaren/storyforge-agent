from __future__ import annotations

import csv
import json
import math
import re
import time
import uuid
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .audio_features import AudioFeatureTool, BoundaryCandidate
from .media import MediaAnalysisTool, TranscriptNLPTool, TranscriptSegment
from .packager import MaterialPackager, PackageConfig
from .utils import require_binary, run_command, seconds_to_srt, slugify, write_json


@dataclass
class AnalysisConfig:
    output_dir: Path = Path("artifacts/analysis")
    scene_threshold: float = 0.30
    similarity_threshold: float = 0.90
    min_segment_seconds: float = 0.6
    max_segments: int = 120
    transcribe: bool = True
    export_clips: bool = False
    enable_audio_boundaries: bool = True
    audio_window_seconds: float = 2.0
    audio_hop_seconds: float = 1.0
    audio_change_threshold: float = 0.45
    emotion_persistence_windows: int = 2
    boundary_merge_seconds: float = 1.0
    build_material_packages: bool = False
    package_merge_threshold: float = 0.45
    max_package_seconds: float = 30.0
    export_package_media: bool = True


@dataclass
class SegmentAnnotation:
    segment_id: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    thumbnail_path: str
    cluster_id: int
    cluster_similarity: float
    boundary_reasons: list[str] = field(default_factory=list)
    boundary_score: float = 1.0
    transcript: str = ""
    keywords: list[str] = field(default_factory=list)
    content_summary: str = ""
    visual_tags: list[str] = field(default_factory=list)
    content_confidence: float = 0.0
    content_provider: str = "local-features"
    suggested_emotion: str = "中性"
    emotion_confidence: float = 0.5
    emotion_evidence: list[str] = field(default_factory=list)
    rms_db: float | None = None
    peak_db: float | None = None
    motion_score: float = 0.0
    highlight_score: float = 0.0
    speech_present: bool = False
    user_label: str = ""
    review_emotion: str = ""
    review_notes: str = ""
    keep: bool = True
    clip_path: str | None = None


@dataclass
class VideoAnalysisResult:
    analysis_id: str
    source_path: str
    duration_seconds: float
    width: int | None
    height: int | None
    has_audio: bool
    detected_language: str | None
    transcription_provider: str
    status: str
    segments: list[SegmentAnnotation]
    artifacts: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    packages: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "VideoAnalysisResult":
        data = dict(value)
        data["segments"] = [SegmentAnnotation(**item) for item in data.get("segments", [])]
        return cls(**data)

    @classmethod
    def load(cls, path: Path) -> "VideoAnalysisResult":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


ProgressCallback = Callable[[str, dict], None]
VisualTagger = Callable[[Path], tuple[str, list[str], float, str]]


class VideoAnalysisAgent:
    """Analyse and annotate footage without making editorial decisions for the user."""

    POSITIVE_WORDS = {
        "win", "won", "great", "amazing", "excited", "success", "love", "happy", "yes",
        "赢", "胜利", "成功", "开心", "高兴", "太好了", "精彩", "喜欢", "厉害",
    }
    TENSION_WORDS = {
        "danger", "warning", "urgent", "fight", "attack", "run", "careful", "trouble",
        "危险", "警告", "紧张", "快跑", "攻击", "战斗", "小心", "麻烦",
    }
    SAD_WORDS = {
        "sad", "sorry", "lost", "lose", "failed", "pain", "cry", "miss",
        "难过", "抱歉", "失败", "输了", "痛苦", "哭", "遗憾", "失去",
    }
    FUNNY_WORDS = {
        "funny", "laugh", "joke", "haha", "hilarious", "lol",
        "搞笑", "好笑", "哈哈", "笑死", "玩笑", "有趣",
    }
    STOP_WORDS = {
        "the", "and", "that", "this", "with", "from", "your", "have", "for", "are",
        "was", "but", "you", "into", "about", "then", "they", "its", "is", "to", "of",
        "的", "了", "和", "是", "在", "有", "也", "就", "都", "与", "及", "这", "那",
    }

    def __init__(
        self,
        config: AnalysisConfig | None = None,
        *,
        media_tool: MediaAnalysisTool | None = None,
        visual_tagger: VisualTagger | None = None,
        on_event: ProgressCallback | None = None,
    ) -> None:
        self.config = config or AnalysisConfig()
        self.media_tool = media_tool or MediaAnalysisTool()
        self.visual_tagger = visual_tagger
        self.on_event = on_event

    def run(self, video_path: Path, language: str = "auto") -> VideoAnalysisResult:
        video_path = video_path.resolve()
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        analysis_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{slugify(video_path.stem)}-{uuid.uuid4().hex[:5]}"
        run_dir = self.config.output_dir / analysis_id
        thumbnails_dir = run_dir / "thumbnails"
        clips_dir = run_dir / "clips"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        if self.config.export_clips:
            clips_dir.mkdir(parents=True, exist_ok=True)
        events_path = run_dir / "events.jsonl"
        warnings: list[str] = []

        self._event(events_path, "analysis_started", {"source_path": str(video_path)})
        media = self.media_tool._probe(video_path)
        self._event(events_path, "media_probed", media.to_dict())

        transcript_segments: list[TranscriptSegment] = []
        provider = "not-requested"
        detected_language = None
        if self.config.transcribe and media.has_audio:
            try:
                raw, detected_language, provider = (
                    self.media_tool.transcriber(video_path, language)
                    if self.media_tool.transcriber
                    else self.media_tool._faster_whisper(video_path, language)
                )
                transcript_segments = TranscriptNLPTool.caption_segments(
                    TranscriptNLPTool.clean_segments(raw)
                )
                if not transcript_segments:
                    warnings.append("No speech was detected; visual and acoustic annotation continued.")
                self._event(events_path, "transcription_completed", {
                    "provider": provider,
                    "language": detected_language,
                    "segments": len(transcript_segments),
                })
            except Exception as exc:
                provider = "unavailable"
                warnings.append(f"Speech transcription unavailable: {type(exc).__name__}: {exc}")
                self._event(events_path, "transcription_fallback", {"reason": warnings[-1]})
        elif self.config.transcribe:
            provider = "no-audio"
            warnings.append("The source has no audio stream; speech and acoustic analysis were skipped.")

        audio_samples = self._extract_audio(video_path, run_dir / "analysis_audio.wav") if media.has_audio else None
        audio_windows = AudioFeatureTool.analyse(
            audio_samples,
            media.duration_seconds,
            transcript_segments,
            window_seconds=self.config.audio_window_seconds,
            hop_seconds=self.config.audio_hop_seconds,
        )
        audio_candidates = (
            AudioFeatureTool.change_boundaries(
                audio_windows,
                threshold=self.config.audio_change_threshold,
                persistence_windows=self.config.emotion_persistence_windows,
                merge_seconds=self.config.boundary_merge_seconds,
            )
            if self.config.enable_audio_boundaries
            else []
        )
        visual_boundaries = self._shot_boundaries(video_path, media.duration_seconds)
        boundary_candidates = self._fuse_boundaries(
            visual_boundaries,
            audio_candidates,
            media.duration_seconds,
        )
        boundaries = [item.time_seconds for item in boundary_candidates]
        self._event(events_path, "shots_detected", {
            "boundaries": len(boundaries),
            "segments": max(0, len(boundaries) - 1),
            "visual_candidates": max(0, len(visual_boundaries) - 2),
            "audio_emotion_candidates": len(audio_candidates),
        })
        self._event(events_path, "audio_timeline_analysed", {
            "windows": len(audio_windows),
            "emotion_boundaries": [item.to_dict() for item in audio_candidates],
        })
        segments: list[SegmentAnnotation] = []
        features: list[np.ndarray] = []
        for index, (start_boundary, end_boundary) in enumerate(
            zip(boundary_candidates, boundary_candidates[1:]), 1
        ):
            start, end = start_boundary.time_seconds, end_boundary.time_seconds
            duration = end - start
            if duration < self.config.min_segment_seconds:
                continue
            thumbnail = thumbnails_dir / f"segment_{index:03}.jpg"
            feature, motion = self._visual_features(video_path, start, end, thumbnail, run_dir)
            text = self._transcript_for_range(transcript_segments, start, end)
            rms_db, peak_db = self._audio_metrics(audio_samples, start, end)
            segment = SegmentAnnotation(
                segment_id=len(segments) + 1,
                start_seconds=round(start, 3),
                end_seconds=round(end, 3),
                duration_seconds=round(duration, 3),
                thumbnail_path=str(thumbnail),
                cluster_id=0,
                cluster_similarity=0.0,
                boundary_reasons=start_boundary.reasons,
                boundary_score=start_boundary.score,
                transcript=text,
                keywords=self._keywords(text),
                rms_db=rms_db,
                peak_db=peak_db,
                motion_score=round(motion, 3),
                speech_present=bool(text),
            )
            segments.append(segment)
            features.append(feature)
            self._describe_content(segment, thumbnail)
            self._event(events_path, "segment_analysed", {
                "segment_id": segment.segment_id,
                "start_seconds": segment.start_seconds,
                "end_seconds": segment.end_seconds,
            })
            if self.config.export_clips:
                segment.clip_path = str(self._export_clip(video_path, segment, clips_dir))

        self._cluster_segments(segments, features)
        self._annotate_emotions(segments)
        self._write_outputs(run_dir, video_path, transcript_segments, segments, audio_windows, audio_candidates)
        packages: list[dict] = []
        if self.config.build_material_packages and segments:
            try:
                built = MaterialPackager(PackageConfig(
                    merge_threshold=self.config.package_merge_threshold,
                    max_package_seconds=self.config.max_package_seconds,
                    export_media=self.config.export_package_media,
                )).build(video_path, segments, run_dir / "packages")
                packages = [item.to_dict() for item in built]
                self._event(events_path, "material_packages_created", {
                    "package_count": len(packages),
                    "similar_group_count": len({item["similar_group_id"] for item in packages}),
                })
            except Exception as exc:
                warnings.append(f"Material package export unavailable: {type(exc).__name__}: {exc}")
                self._event(events_path, "material_packages_fallback", {"reason": warnings[-1]})
        artifacts = {
            "analysis_json": str(run_dir / "analysis.json"),
            "segments_csv": str(run_dir / "segments.csv"),
            "transcript_srt": str(run_dir / "transcript.srt"),
            "audio_timeline": str(run_dir / "audio_timeline.json"),
            "events": str(events_path),
            "thumbnails": str(thumbnails_dir),
        }
        if packages:
            artifacts["packages"] = str(run_dir / "packages")
            artifacts["package_manifest"] = str(run_dir / "packages" / "package_manifest.json")
            if self.config.export_package_media:
                artifacts["packages_archive"] = str(run_dir / "packages.zip")
        if self.config.export_clips:
            artifacts["clips"] = str(clips_dir)
        result = VideoAnalysisResult(
            analysis_id=analysis_id,
            source_path=str(video_path),
            duration_seconds=round(media.duration_seconds, 3),
            width=media.width,
            height=media.height,
            has_audio=media.has_audio,
            detected_language=detected_language,
            transcription_provider=provider,
            status="completed_with_warnings" if warnings else "completed",
            segments=segments,
            artifacts=artifacts,
            warnings=warnings,
            packages=packages,
        )
        write_json(run_dir / "analysis.json", result.to_dict())
        (run_dir / "analysis_audio.wav").unlink(missing_ok=True)
        self._event(events_path, "analysis_completed", {
            "status": result.status,
            "segment_count": len(segments),
            "cluster_count": len({item.cluster_id for item in segments}),
        })
        return result

    def export_selection(
        self,
        result: VideoAnalysisResult,
        segment_ids: list[int],
        output_dir: Path | None = None,
    ) -> Path:
        selected = [item for item in result.segments if item.segment_id in set(segment_ids)]
        if not selected:
            raise ValueError("Select at least one segment")
        root = output_dir or Path(result.artifacts["analysis_json"]).parent / "selection"
        root.mkdir(parents=True, exist_ok=True)
        clips = [self._export_clip(Path(result.source_path), item, root) for item in selected]
        concat_file = root / "clips.txt"
        concat_file.write_text(
            "\n".join(f"file '{path.resolve().as_posix()}'" for path in clips), encoding="utf-8"
        )
        output = root / "selected_segments.mp4"
        run_command([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", "-movflags", "+faststart", str(output),
        ])
        write_json(root / "selection.json", {
            "source_path": result.source_path,
            "selected_segment_ids": segment_ids,
            "segments": [asdict(item) for item in selected],
            "output": str(output),
        })
        return output

    def _shot_boundaries(self, path: Path, duration: float) -> list[float]:
        require_binary("ffmpeg")
        result = run_command([
            "ffmpeg", "-hide_banner", "-i", str(path),
            "-vf", f"select='gt(scene,{max(0.05, min(0.95, self.config.scene_threshold)):.3f})',showinfo",
            "-an", "-f", "null", "-",
        ])
        detected = [float(item) for item in re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", result.stderr)]
        values = sorted({0.0, *[max(0.0, min(duration, item)) for item in detected], duration})
        if len(values) - 1 > self.config.max_segments:
            values = values[:self.config.max_segments] + [duration]
        return values

    def _fuse_boundaries(
        self,
        visual_boundaries: list[float],
        audio_candidates: list[BoundaryCandidate],
        duration: float,
    ) -> list[BoundaryCandidate]:
        internal = [
            BoundaryCandidate(value, 1.0, ["画面转场"], "visual")
            for value in visual_boundaries[1:-1]
        ] + [
            item for item in audio_candidates if 0 < item.time_seconds < duration
        ]
        grouped: list[BoundaryCandidate] = []
        for candidate in sorted(internal, key=lambda item: item.time_seconds):
            previous = grouped[-1] if grouped else None
            can_merge = (
                previous is not None
                and candidate.time_seconds - previous.time_seconds <= self.config.boundary_merge_seconds
                and not (previous.source == "visual" and candidate.source == "visual")
            )
            if not can_merge:
                grouped.append(candidate)
                continue
            assert previous is not None
            previous.reasons = list(dict.fromkeys([*previous.reasons, *candidate.reasons]))
            # A visual cut is frame-derived and remains the anchor when audio evidence is nearby.
            if candidate.score > previous.score and previous.source != "visual":
                candidate.reasons = previous.reasons
                grouped[-1] = candidate
        selected = [BoundaryCandidate(0.0, 1.0, ["视频开始"], "timeline")]
        minimum = max(0.0, self.config.min_segment_seconds)
        for candidate in grouped:
            if candidate.time_seconds - selected[-1].time_seconds < minimum:
                if len(selected) > 1 and candidate.score > selected[-1].score:
                    selected[-1] = candidate
                continue
            selected.append(candidate)
        if len(selected) > 1 and duration - selected[-1].time_seconds < minimum:
            selected.pop()
        if duration > selected[-1].time_seconds:
            selected.append(BoundaryCandidate(duration, 1.0, ["视频结束"], "timeline"))
        if len(selected) - 1 > self.config.max_segments:
            strongest = sorted(
                selected[1:-1], key=lambda item: item.score, reverse=True
            )[: max(0, self.config.max_segments - 1)]
            selected = [selected[0], *sorted(strongest, key=lambda item: item.time_seconds), selected[-1]]
        return selected

    @staticmethod
    def _merge_short_segments(boundaries: list[float], minimum: float) -> list[float]:
        if len(boundaries) <= 2 or minimum <= 0:
            return boundaries
        merged = [boundaries[0]]
        for boundary in boundaries[1:-1]:
            if boundary - merged[-1] >= minimum:
                merged.append(boundary)
        end = boundaries[-1]
        if len(merged) > 1 and end - merged[-1] < minimum:
            merged.pop()
        if end > merged[-1]:
            merged.append(end)
        return merged

    @staticmethod
    def _extract_audio(path: Path, output: Path) -> tuple[np.ndarray, int] | None:
        try:
            run_command([
                "ffmpeg", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(output),
            ])
            with wave.open(str(output), "rb") as handle:
                rate = handle.getframerate()
                samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16).astype(np.float32)
            return samples / 32768.0, rate
        except Exception:
            return None

    @staticmethod
    def _audio_metrics(
        audio: tuple[np.ndarray, int] | None,
        start: float,
        end: float,
    ) -> tuple[float | None, float | None]:
        if audio is None:
            return None, None
        samples, rate = audio
        chunk = samples[max(0, int(start * rate)):min(len(samples), int(end * rate))]
        if not len(chunk):
            return None, None
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        peak = float(np.max(np.abs(chunk)))
        return round(20 * math.log10(max(rms, 1e-8)), 2), round(20 * math.log10(max(peak, 1e-8)), 2)

    def _visual_features(
        self,
        path: Path,
        start: float,
        end: float,
        thumbnail: Path,
        run_dir: Path,
    ) -> tuple[np.ndarray, float]:
        duration = end - start
        first = run_dir / "_feature_a.jpg"
        last = run_dir / "_feature_b.jpg"
        self._frame(path, start + duration * 0.2, first)
        self._frame(path, start + duration * 0.5, thumbnail)
        self._frame(path, start + duration * 0.8, last)
        first_feature = self._image_feature(first)
        middle_feature = self._image_feature(thumbnail)
        last_feature = self._image_feature(last)
        motion = 1.0 - self._cosine(first_feature, last_feature)
        first.unlink(missing_ok=True)
        last.unlink(missing_ok=True)
        return middle_feature, max(0.0, min(1.0, motion))

    @staticmethod
    def _frame(path: Path, timestamp: float, output: Path) -> None:
        run_command([
            "ffmpeg", "-y", "-ss", f"{max(0.0, timestamp):.3f}", "-i", str(path),
            "-frames:v", "1", "-vf", "scale=320:-2", "-q:v", "3", str(output),
        ])

    @staticmethod
    def _image_feature(path: Path) -> np.ndarray:
        with Image.open(path).convert("RGB") as image:
            pixels = np.asarray(image.resize((64, 64)), dtype=np.float32)
        values = []
        for channel in range(3):
            hist, _ = np.histogram(pixels[:, :, channel], bins=16, range=(0, 256), density=False)
            values.extend(hist.astype(np.float32))
        feature = np.asarray(values, dtype=np.float32)
        norm = np.linalg.norm(feature)
        return feature / norm if norm else feature

    @staticmethod
    def _cosine(first: np.ndarray, second: np.ndarray) -> float:
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        return float(np.dot(first, second) / denominator) if denominator else 0.0

    def _cluster_segments(self, segments: list[SegmentAnnotation], features: list[np.ndarray]) -> None:
        centroids: list[np.ndarray] = []
        counts: list[int] = []
        for segment, feature in zip(segments, features):
            similarities = [self._cosine(feature, centroid) for centroid in centroids]
            best_index = int(np.argmax(similarities)) if similarities else -1
            best_similarity = similarities[best_index] if similarities else 0.0
            if best_index >= 0 and best_similarity >= self.config.similarity_threshold:
                segment.cluster_id = best_index + 1
                segment.cluster_similarity = round(best_similarity, 3)
                counts[best_index] += 1
                centroid = centroids[best_index] * (counts[best_index] - 1) + feature
                centroids[best_index] = centroid / counts[best_index]
            else:
                centroids.append(feature.copy())
                counts.append(1)
                segment.cluster_id = len(centroids)
                segment.cluster_similarity = 1.0

    def _annotate_emotions(self, segments: list[SegmentAnnotation]) -> None:
        rms_values = [item.rms_db for item in segments if item.rms_db is not None]
        quiet = float(np.percentile(rms_values, 35)) if rms_values else -35.0
        loud = float(np.percentile(rms_values, 70)) if rms_values else -18.0
        for item in segments:
            lower = item.transcript.lower()
            evidence: list[str] = []
            scores = {"兴奋": 0.0, "紧张": 0.0, "低落": 0.0, "轻松/幽默": 0.0, "平静": 0.0}
            word_sets = [
                ("兴奋", self.POSITIVE_WORDS),
                ("紧张", self.TENSION_WORDS),
                ("低落", self.SAD_WORDS),
                ("轻松/幽默", self.FUNNY_WORDS),
            ]
            for label, words in word_sets:
                matched = [word for word in words if word in lower]
                if matched:
                    scores[label] += min(0.65, 0.3 + len(matched) * 0.12)
                    evidence.append(f"文本关键词：{', '.join(matched[:3])}")
            if item.rms_db is not None:
                if item.rms_db >= loud:
                    scores["兴奋"] += 0.18
                    scores["紧张"] += 0.15
                    evidence.append("声音能量较高")
                elif item.rms_db <= quiet:
                    scores["平静"] += 0.25
                    evidence.append("声音能量较低")
            if item.motion_score >= 0.20:
                scores["兴奋"] += 0.12
                scores["紧张"] += 0.12
                evidence.append("画面变化较明显")
            elif item.motion_score <= 0.05:
                scores["平静"] += 0.18
                evidence.append("画面较稳定")
            if not item.transcript and item.rms_db is None:
                scores["平静"] += 0.12
            label, score = max(scores.items(), key=lambda pair: pair[1])
            if score < 0.18:
                label, score = "中性", 0.5
                evidence.append("缺少明确情绪信号")
            else:
                score = min(0.92, 0.52 + score * 0.5)
            item.suggested_emotion = label
            item.emotion_confidence = round(score, 2)
            item.emotion_evidence = evidence[:4]
            energy_score = 0.0 if item.rms_db is None else max(0.0, min(1.0, (item.rms_db + 45) / 35))
            speech_score = min(1.0, len(item.transcript) / 80) if item.transcript else 0.0
            item.highlight_score = round(min(1.0, 0.45 * energy_score + 0.3 * item.motion_score + 0.25 * speech_score), 2)

    def _describe_content(self, segment: SegmentAnnotation, thumbnail: Path) -> None:
        """Attach evidence-backed hints; semantic claims require speech or a visual tagger."""
        if self.visual_tagger:
            try:
                summary, tags, confidence, provider = self.visual_tagger(thumbnail)
                segment.content_summary = summary.strip()
                segment.visual_tags = [str(item).strip() for item in tags if str(item).strip()]
                segment.content_confidence = round(max(0.0, min(1.0, float(confidence))), 2)
                segment.content_provider = provider
                return
            except Exception as exc:
                segment.visual_tags.append(f"视觉模型失败:{type(exc).__name__}")

        with Image.open(thumbnail).convert("RGB") as image:
            pixels = np.asarray(image.resize((64, 64)), dtype=np.float32)
        brightness = float(pixels.mean())
        channel_means = pixels.mean(axis=(0, 1))
        dominant = ["偏红", "偏绿", "偏蓝"][int(np.argmax(channel_means))]
        light = "明亮" if brightness >= 165 else "昏暗" if brightness <= 85 else "中等亮度"
        movement = (
            "高画面变化" if segment.motion_score >= 0.2
            else "低画面变化" if segment.motion_score <= 0.05
            else "中等画面变化"
        )
        segment.visual_tags.extend([light, dominant, movement])
        if segment.transcript:
            segment.content_summary = segment.transcript[:160]
            segment.content_confidence = 0.7
            segment.content_provider = "speech-transcript+local-features"
        else:
            segment.content_summary = "未识别具体语义，建议人工标注或接入视觉模型"
            segment.content_confidence = 0.2

    @classmethod
    def _keywords(cls, text: str) -> list[str]:
        english = re.findall(r"[A-Za-z][A-Za-z']{2,}", text.lower())
        chinese = re.findall(r"[\u4e00-\u9fff]{2,6}", text)
        counts: dict[str, int] = {}
        for word in [*english, *chinese]:
            if word not in cls.STOP_WORDS:
                counts[word] = counts.get(word, 0) + 1
        return [item[0] for item in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:6]]

    @staticmethod
    def _transcript_for_range(items: list[TranscriptSegment], start: float, end: float) -> str:
        values = [item.text for item in items if min(end, item.end_seconds) > max(start, item.start_seconds)]
        return " ".join(dict.fromkeys(values)).strip()

    def _write_outputs(
        self,
        run_dir: Path,
        source_path: Path,
        transcript: list[TranscriptSegment],
        segments: list[SegmentAnnotation],
        audio_windows: list,
        audio_candidates: list[BoundaryCandidate],
    ) -> None:
        with (run_dir / "segments.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "segment_id", "start_seconds", "end_seconds", "duration_seconds", "cluster_id",
                "boundary_reasons", "boundary_score",
                "cluster_similarity", "transcript", "keywords", "suggested_emotion",
                "content_summary", "visual_tags", "content_confidence", "content_provider",
                "emotion_confidence", "emotion_evidence", "rms_db", "peak_db", "motion_score",
                "highlight_score", "speech_present", "keep", "user_label", "thumbnail_path", "clip_path",
                "review_emotion", "review_notes",
            ])
            writer.writeheader()
            for item in segments:
                row = asdict(item)
                row["boundary_reasons"] = "|".join(item.boundary_reasons)
                row["keywords"] = "|".join(item.keywords)
                row["visual_tags"] = "|".join(item.visual_tags)
                row["emotion_evidence"] = "|".join(item.emotion_evidence)
                writer.writerow(row)
        (run_dir / "transcript.srt").write_text("\n\n".join(
            f"{index}\n{seconds_to_srt(item.start_seconds)} --> {seconds_to_srt(item.end_seconds)}\n{item.text}"
            for index, item in enumerate(transcript, 1)
        ), encoding="utf-8")
        write_json(run_dir / "audio_timeline.json", {
            "window_seconds": self.config.audio_window_seconds,
            "hop_seconds": self.config.audio_hop_seconds,
            "change_threshold": self.config.audio_change_threshold,
            "persistence_windows": self.config.emotion_persistence_windows,
            "windows": [item.to_dict() for item in audio_windows],
            "emotion_boundaries": [item.to_dict() for item in audio_candidates],
        })
        write_json(run_dir / "analysis.json", {
            "source_path": str(source_path),
            "segments": [asdict(item) for item in segments],
        })

    @staticmethod
    def _export_clip(source: Path, segment: SegmentAnnotation, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"segment_{segment.segment_id:03}.mp4"
        run_command([
            "ffmpeg", "-y", "-ss", f"{segment.start_seconds:.3f}", "-i", str(source),
            "-t", f"{segment.duration_seconds:.3f}", "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(output),
        ])
        return output

    def _event(self, path: Path, event: str, payload: dict) -> None:
        item = {"timestamp": time.time(), "event": event, **payload}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        if self.on_event:
            self.on_event(event, item)
