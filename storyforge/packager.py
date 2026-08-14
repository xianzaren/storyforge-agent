from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw

from .utils import run_command, seconds_to_srt, write_json


class PackageSegment(Protocol):
    segment_id: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    thumbnail_path: str
    cluster_id: int
    transcript: str
    keywords: list[str]
    visual_tags: list[str]
    content_summary: str
    content_confidence: float
    suggested_emotion: str
    emotion_confidence: float
    highlight_score: float
    boundary_reasons: list[str]
    review_emotion: str
    user_label: str
    keep: bool


@dataclass
class MaterialPackage:
    package_id: str
    title: str
    segment_ids: list[int]
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    dominant_cluster_id: int
    similar_group_id: int = 0
    content_summary: str = ""
    content_tags: list[str] = field(default_factory=list)
    dominant_emotion: str = "中性"
    emotion_trend: list[str] = field(default_factory=list)
    highlight_score: float = 0.0
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    needs_review: bool = True
    package_dir: str = ""
    preview_path: str = ""
    contact_sheet_path: str = ""
    subtitle_path: str = ""
    clip_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PackageConfig:
    merge_threshold: float = 0.45
    max_package_seconds: float = 30.0
    export_media: bool = True


class MaterialPackager:
    """Build reviewable, contiguous event packages without making final edit choices."""

    def __init__(self, config: PackageConfig | None = None) -> None:
        self.config = config or PackageConfig()

    def build(
        self,
        source: Path,
        segments: list[PackageSegment],
        output_dir: Path,
    ) -> list[MaterialPackage]:
        output_dir.mkdir(parents=True, exist_ok=True)
        groups = self._contiguous_groups([item for item in segments if item.keep])
        packages = [self._annotate(index, group) for index, group in enumerate(groups, 1)]
        self._assign_similar_groups(packages)
        if self.config.export_media:
            for package, group in zip(packages, groups):
                self._export_package(source, package, group, output_dir)
        write_json(output_dir / "package_manifest.json", {
            "source_path": str(source),
            "package_count": len(packages),
            "similar_group_count": len({item.similar_group_id for item in packages}),
            "packages": [item.to_dict() for item in packages],
        })
        if self.config.export_media:
            shutil.make_archive(str(output_dir), "zip", root_dir=output_dir)
        return packages

    def _contiguous_groups(self, segments: list[PackageSegment]) -> list[list[PackageSegment]]:
        if not segments:
            return []
        groups: list[list[PackageSegment]] = [[segments[0]]]
        for current in segments[1:]:
            active = groups[-1]
            previous = active[-1]
            duration = current.end_seconds - active[0].start_seconds
            timeline_gap = current.start_seconds - previous.end_seconds > 0.25
            hard_change = any(
                "文本情绪变化" in reason or "声学状态变化" in reason
                for reason in current.boundary_reasons
            )
            visual_change = "画面转场" in current.boundary_reasons and current.cluster_id != previous.cluster_id
            score = self._adjacency_score(previous, current)
            if (
                hard_change
                or visual_change
                or timeline_gap
                or duration > self.config.max_package_seconds
                or score < self.config.merge_threshold
            ):
                groups.append([current])
            else:
                active.append(current)
        return groups

    @classmethod
    def _adjacency_score(cls, first: PackageSegment, second: PackageSegment) -> float:
        cluster = 0.55 if first.cluster_id == second.cluster_id else 0.0
        first_emotion = first.review_emotion or first.suggested_emotion
        second_emotion = second.review_emotion or second.suggested_emotion
        emotion = 0.20 if first_emotion == second_emotion else 0.0
        keywords = 0.20 * cls._jaccard(set(first.keywords), set(second.keywords))
        tags = 0.05 * cls._jaccard(set(first.visual_tags), set(second.visual_tags))
        return round(cluster + emotion + keywords + tags, 3)

    @staticmethod
    def _jaccard(first: set[str], second: set[str]) -> float:
        union = first | second
        return len(first & second) / len(union) if union else 0.0

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))

    def _annotate(self, index: int, group: list[PackageSegment]) -> MaterialPackage:
        cluster = Counter(item.cluster_id for item in group).most_common(1)[0][0]
        effective_emotions = [item.review_emotion or item.suggested_emotion for item in group]
        emotions = self._dedupe(effective_emotions)
        dominant_emotion = Counter(effective_emotions).most_common(1)[0][0]
        tags = self._dedupe([
            *[item.user_label for item in group],
            *[value for item in group for value in item.keywords],
            *[value for item in group for value in item.visual_tags],
        ])[:12]
        semantic = [item.content_summary for item in group if item.content_confidence >= 0.5]
        transcript = self._dedupe([item.transcript for item in group if item.transcript])
        summary = " ".join(semantic or transcript)[:240]
        if not summary:
            summary = "未识别具体内容，等待人工复核"
        confidence = sum(item.content_confidence for item in group) / len(group)
        emotion_confidence = sum(item.emotion_confidence for item in group) / len(group)
        total_confidence = round(0.65 * confidence + 0.35 * emotion_confidence, 2)
        evidence = self._dedupe([
            f"连续时间范围 {group[0].start_seconds:.1f}–{group[-1].end_seconds:.1f}s",
            f"主要相似镜头组 {cluster}",
            *[reason for item in group for reason in item.boundary_reasons],
        ])[:8]
        if summary.startswith("未识别具体内容"):
            title_hint = next(
                (tag for tag in tags if "变化" not in tag and "偏" not in tag),
                "待复核素材",
            )
        else:
            title_hint = summary.split("。", 1)[0].split(".", 1)[0].strip()[:36]
        return MaterialPackage(
            package_id=f"package_{index:03}",
            title=f"{title_hint} · {dominant_emotion}",
            segment_ids=[item.segment_id for item in group],
            start_seconds=group[0].start_seconds,
            end_seconds=group[-1].end_seconds,
            duration_seconds=round(group[-1].end_seconds - group[0].start_seconds, 3),
            dominant_cluster_id=cluster,
            content_summary=summary,
            content_tags=tags,
            dominant_emotion=dominant_emotion,
            emotion_trend=emotions,
            highlight_score=round(max(item.highlight_score for item in group), 2),
            confidence=total_confidence,
            evidence=evidence,
            needs_review=total_confidence < 0.65,
        )

    def _assign_similar_groups(self, packages: list[MaterialPackage]) -> None:
        representatives: list[MaterialPackage] = []
        for package in packages:
            match = next((
                item for item in representatives
                if item.dominant_cluster_id == package.dominant_cluster_id
                or self._jaccard(set(item.content_tags), set(package.content_tags)) >= 0.5
            ), None)
            if match:
                package.similar_group_id = match.similar_group_id
            else:
                package.similar_group_id = len(representatives) + 1
                representatives.append(package)

    def _export_package(
        self,
        source: Path,
        package: MaterialPackage,
        segments: list[PackageSegment],
        root: Path,
    ) -> None:
        package_dir = root / package.package_id
        clips_dir = package_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        clips: list[Path] = []
        for segment in segments:
            clip = clips_dir / f"segment_{segment.segment_id:03}.mp4"
            run_command([
                "ffmpeg", "-y", "-ss", f"{segment.start_seconds:.3f}", "-i", str(source),
                "-t", f"{segment.duration_seconds:.3f}", "-map", "0:v:0", "-map", "0:a?",
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(clip),
            ])
            clips.append(clip)
        concat = package_dir / "clips.txt"
        concat.write_text("\n".join(
            f"file '{clip.resolve().as_posix()}'" for clip in clips
        ), encoding="utf-8")
        preview = package_dir / "preview.mp4"
        run_command([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
            "-c", "copy", "-movflags", "+faststart", str(preview),
        ])
        subtitle = package_dir / "transcript.srt"
        cursor = 0.0
        cues: list[str] = []
        for segment in segments:
            if segment.transcript:
                cues.append(
                    f"{len(cues) + 1}\n{seconds_to_srt(cursor)} --> "
                    f"{seconds_to_srt(cursor + segment.duration_seconds)}\n{segment.transcript}"
                )
            cursor += segment.duration_seconds
        subtitle.write_text("\n\n".join(cues), encoding="utf-8")
        sheet = package_dir / "contact_sheet.jpg"
        self._contact_sheet(segments, sheet)
        package.package_dir = str(package_dir)
        package.preview_path = str(preview)
        package.contact_sheet_path = str(sheet)
        package.subtitle_path = str(subtitle)
        package.clip_paths = [str(item) for item in clips]
        write_json(package_dir / "package.json", package.to_dict())

    @staticmethod
    def _contact_sheet(segments: list[PackageSegment], output: Path) -> None:
        images: list[Image.Image] = []
        for segment in segments:
            with Image.open(segment.thumbnail_path).convert("RGB") as image:
                images.append(image.resize((320, 180)).copy())
        columns = min(3, max(1, len(images)))
        rows = (len(images) + columns - 1) // columns
        canvas = Image.new("RGB", (columns * 320, rows * 210), "#111827")
        draw = ImageDraw.Draw(canvas)
        for index, (segment, image) in enumerate(zip(segments, images)):
            x, y = (index % columns) * 320, (index // columns) * 210
            canvas.paste(image, (x, y))
            draw.text((x + 8, y + 186), f"#{segment.segment_id}  {segment.start_seconds:.1f}-{segment.end_seconds:.1f}s", fill="white")
        canvas.save(output, quality=88)
