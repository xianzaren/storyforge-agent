from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFont

from .llm import CompatibleLLMClient
from .models import Scene
from .utils import ffprobe_duration, require_binary, run_command, seconds_to_srt, write_json


EventCallback = Callable[[str, dict], None]


class ScriptTool:
    def __init__(self, llm: CompatibleLLMClient | None = None) -> None:
        self.llm = llm or CompatibleLLMClient()
        self.last_provider = "template-offline"
        self.last_error: str | None = None

    def run(self, topic: str, target_duration: int, language: str) -> list[Scene]:
        count = max(3, min(6, round(target_duration / 6)))
        self.last_provider = "template-offline"
        self.last_error = None
        if self.llm.enabled:
            try:
                result = self.llm.generate_json(
                    "You create concise short-video scripts. Return JSON only with a scenes array.",
                    f"Create {count} scenes about {topic!r} in language {language}. "
                    f"Target total duration {target_duration}s. Each scene needs title, narration, "
                    "visual_query, on_screen_text, duration_seconds.",
                )
                scenes = self._parse_scenes(result.get("scenes", []), target_duration)
                if scenes:
                    self.last_provider = "openai-compatible"
                    return scenes
                self.last_error = "The model response did not contain any valid scenes."
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_provider = "template-fallback"
        return self._fallback(topic, target_duration, language, count)

    @staticmethod
    def _parse_scenes(items: list[dict], target_duration: int) -> list[Scene]:
        if not isinstance(items, list):
            raise ValueError("scenes must be a list")
        scenes: list[Scene] = []
        for index, item in enumerate(items, 1):
            if not isinstance(item, dict):
                continue
            narration = str(item.get("narration", "")).strip()
            if not narration:
                continue
            try:
                duration = float(item.get("duration_seconds", max(3, target_duration / max(1, len(items)))))
            except (TypeError, ValueError):
                duration = max(3, target_duration / max(1, len(items)))
            if not math.isfinite(duration) or duration <= 0:
                duration = max(3, target_duration / max(1, len(items)))
            scenes.append(Scene(
                scene_id=index,
                title=str(item.get("title", f"Scene {index}")),
                narration=narration,
                visual_query=str(item.get("visual_query", item.get("title", ""))),
                on_screen_text=str(item.get("on_screen_text", item.get("title", ""))),
                duration_seconds=duration,
            ))
        return scenes

    @staticmethod
    def _fallback(topic: str, target_duration: int, language: str, count: int) -> list[Scene]:
        duration = target_duration / count
        if language.lower().startswith("zh"):
            templates = [
                ("开场", f"用几十秒快速了解{topic}。", "主题概览"),
                ("为什么重要", f"{topic}正在改变内容生产的速度与方式。", "真实使用场景"),
                ("核心方法", "关键是把复杂目标拆成清晰步骤，并为每一步选择合适工具。", "工作流与工具"),
                ("可靠性", "结构化结果、状态记录和局部重试让自动化流程更稳定。", "质量检查"),
                ("总结", f"从一个小型可运行流程开始，再持续优化{topic}。", "下一步行动"),
                ("结束", "把创意交给人，把重复工作交给自动化。", "创作与自动化"),
            ]
        else:
            templates = [
                ("Hook", f"Here is a quick way to understand {topic}.", "fast topic introduction"),
                ("Why it matters", f"{topic} is changing how creators turn ideas into finished content.", "creator workflow"),
                ("Core method", "Break the goal into clear steps, then give each step the right tool.", "workflow and tools"),
                ("Reliability", "Structured outputs, event logs, and local retries make the pipeline dependable.", "quality control"),
                ("Takeaway", f"Start with one working flow, measure it, and improve {topic} scene by scene.", "iteration"),
                ("Close", "Keep human judgment in the loop and automate the repetitive work.", "human and AI collaboration"),
            ]
        selected = templates[:count]
        return [Scene(i, title, narration, query, title, duration) for i, (title, narration, query) in enumerate(selected, 1)]


class StoryboardTool:
    """Normalise script scenes into render-ready storyboard entries."""

    def run(self, scenes: list[Scene]) -> list[Scene]:
        if not scenes:
            raise ValueError("Cannot create a storyboard without scenes")
        for index, scene in enumerate(scenes, 1):
            scene.scene_id = index
            scene.title = scene.title.strip() or f"Scene {index}"
            scene.narration = scene.narration.strip()
            if not scene.narration:
                raise ValueError(f"Scene {index} has no narration")
            scene.visual_query = scene.visual_query.strip() or scene.title
            scene.on_screen_text = scene.on_screen_text.strip() or scene.title
            if not math.isfinite(scene.duration_seconds) or scene.duration_seconds <= 0:
                raise ValueError(f"Scene {index} has an invalid duration")
        return scenes


@dataclass(frozen=True)
class AssetRecord:
    path: Path
    keywords: set[str]
    media_type: str
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    has_audio: bool = False
    segment_start_seconds: float = 0.0
    shot_index: int | None = None
    source_duration_seconds: float | None = None

    @property
    def key(self) -> str:
        return f"{self.path.resolve()}#{self.shot_index if self.shot_index is not None else 'full'}"

    def to_dict(self) -> dict:
        value = asdict(self)
        value["path"] = str(self.path)
        value["keywords"] = sorted(self.keywords)
        return value


class AssetTool:
    SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png"}
    SUPPORTED_VIDEOS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    SUPPORTED = SUPPORTED_IMAGES | SUPPORTED_VIDEOS

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        enable_scene_detection: bool = True,
        scene_detection_threshold: float = 0.35,
        max_video_segments: int = 24,
        use_unmatched_assets: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.enable_scene_detection = enable_scene_detection
        self.scene_detection_threshold = max(0.05, min(0.95, scene_detection_threshold))
        self.max_video_segments = max(1, max_video_segments)
        self.use_unmatched_assets = use_unmatched_assets
        self.last_warnings: list[str] = []

    def index(self, assets_dir: Path | None) -> list[AssetRecord]:
        self.last_warnings = []
        if not assets_dir or not assets_dir.exists():
            return []
        metadata_path = assets_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        records: list[AssetRecord] = []
        for path in assets_dir.rglob("*"):
            if path.suffix.lower() not in self.SUPPORTED:
                continue
            metadata_value = metadata.get(path.name, [])
            tags = metadata_value.get("tags", []) if isinstance(metadata_value, dict) else metadata_value
            if not isinstance(tags, list):
                tags = [tags]
            words = self._keywords(path.stem)
            for tag in tags:
                words.update(self._keywords(str(tag)))
            try:
                if path.suffix.lower() in self.SUPPORTED_IMAGES:
                    with Image.open(path) as image:
                        width, height = image.size
                    records.append(AssetRecord(path, words, "image", width=width, height=height))
                else:
                    records.extend(self._video_records(path, words))
            except Exception as exc:
                self.last_warnings.append(f"Skipped unreadable asset '{path.name}': {type(exc).__name__}: {exc}")
        return records

    def match_or_create(
        self,
        scene: Scene,
        records: list[AssetRecord],
        output_dir: Path,
        used_assets: set[str] | None = None,
        usage_counts: dict[str, int] | None = None,
    ) -> tuple[Path, bool]:
        query_words = self._keywords(f"{scene.visual_query} {scene.title}")
        used_assets = used_assets if used_assets is not None else set()
        usage_counts = usage_counts if usage_counts is not None else {}
        ranked = sorted(
            records,
            key=lambda item: (
                len(query_words & item.keywords),
                item.key not in used_assets,
                item.media_type == "video",
            ),
            reverse=True,
        )
        if ranked:
            record = ranked[0]
            score = len(query_words & record.keywords)
            if score > 0 or self.use_unmatched_assets:
                scene.asset_type = record.media_type
                scene.asset_source_duration = record.source_duration_seconds or record.duration_seconds
                scene.asset_segment_start_seconds = record.segment_start_seconds
                scene.asset_segment_duration = record.duration_seconds
                scene.asset_shot_index = record.shot_index
                scene.asset_width = record.width
                scene.asset_height = record.height
                scene.asset_has_audio = record.has_audio
                scene.asset_match_score = score
                scene.asset_generated = False
                scene.asset_selection_reason = "keyword_match" if score > 0 else "unmatched_user_fallback"
                scene.asset_reuse_index = usage_counts.get(record.key, 0)
                usage_counts[record.key] = scene.asset_reuse_index + 1
                used_assets.add(record.key)
                return record.path, False
        path = output_dir / f"scene_{scene.scene_id:02}_card.png"
        self._title_card(scene, path)
        scene.asset_type = "image"
        scene.asset_source_duration = None
        scene.asset_segment_start_seconds = 0.0
        scene.asset_segment_duration = None
        scene.asset_shot_index = None
        scene.asset_width = self.width
        scene.asset_height = self.height
        scene.asset_has_audio = False
        scene.asset_match_score = 0
        scene.asset_generated = True
        scene.asset_selection_reason = "generated_title_card"
        scene.asset_reuse_index = 0
        return path, True

    @staticmethod
    def _keywords(value: str) -> set[str]:
        normalised = re.sub(r"[_\-]+", " ", value.lower())
        words = set(re.findall(r"[a-z0-9]+", normalised))
        for chunk in re.findall(r"[\u4e00-\u9fff]+", normalised):
            words.add(chunk)
            words.update(chunk[index:index + 2] for index in range(max(0, len(chunk) - 1)))
        return words

    def _video_records(self, path: Path, words: set[str]) -> list[AssetRecord]:
        require_binary("ffprobe")
        result = run_command([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration:stream=codec_type,width,height", "-of", "json", str(path),
        ])
        probe = json.loads(result.stdout)
        streams = probe.get("streams", [])
        video_stream = next((item for item in streams if item.get("codec_type") == "video"), {})
        duration_value = probe.get("format", {}).get("duration")
        duration = float(duration_value) if duration_value not in (None, "N/A") else None
        width = int(video_stream["width"]) if video_stream.get("width") else None
        height = int(video_stream["height"]) if video_stream.get("height") else None
        has_audio = any(item.get("codec_type") == "audio" for item in streams)
        if not duration or not self.enable_scene_detection:
            return [AssetRecord(
                path, words, "video", duration, width, height, has_audio,
                source_duration_seconds=duration,
            )]

        boundaries = [0.0, *self._detect_scene_boundaries(path), duration]
        boundaries = sorted({max(0.0, min(duration, item)) for item in boundaries})
        segments: list[AssetRecord] = []
        for shot_index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
            segment_duration = end - start
            if segment_duration < 0.4:
                continue
            segments.append(AssetRecord(
                path=path,
                keywords=set(words),
                media_type="video",
                duration_seconds=segment_duration,
                width=width,
                height=height,
                has_audio=has_audio,
                segment_start_seconds=start,
                shot_index=shot_index,
                source_duration_seconds=duration,
            ))
            if len(segments) >= self.max_video_segments:
                break
        return segments or [AssetRecord(
            path, words, "video", duration, width, height, has_audio,
            source_duration_seconds=duration,
        )]

    def _detect_scene_boundaries(self, path: Path) -> list[float]:
        require_binary("ffmpeg")
        result = run_command([
            "ffmpeg", "-hide_banner", "-i", str(path),
            "-vf", f"select='gt(scene,{self.scene_detection_threshold:.3f})',showinfo",
            "-an", "-f", "null", "-",
        ])
        values = [float(item) for item in re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", result.stderr)]
        return values[: max(0, self.max_video_segments - 1)]

    def _title_card(self, scene: Scene, path: Path) -> None:
        colors = [(20, 35, 70), (34, 65, 110), (67, 56, 125), (16, 91, 104), (91, 55, 98), (28, 78, 70)]
        base = colors[(scene.scene_id - 1) % len(colors)]
        image = Image.new("RGB", (self.width, self.height), base)
        draw = ImageDraw.Draw(image)
        for y in range(self.height):
            factor = y / self.height
            color = tuple(max(0, min(255, int(c + 35 * factor))) for c in base)
            draw.line((0, y, self.width, y), fill=color)
        font_large = self._font(58)
        font_small = self._font(30)
        draw.rounded_rectangle((80, 80, self.width - 80, self.height - 80), radius=28, fill=(0, 0, 0, 90), outline=(255, 255, 255), width=2)
        draw.text((120, 145), f"{scene.scene_id:02}", font=font_small, fill=(130, 220, 255))
        title = scene.on_screen_text or scene.title
        lines = self._wrap(draw, title, font_large, self.width - 260)
        y = 255
        for line in lines[:3]:
            draw.text((120, y), line, font=font_large, fill="white")
            y += 82
        draw.text((120, self.height - 145), "STORYFORGE AGENT", font=font_small, fill=(210, 225, 240))
        image.save(path)

    @staticmethod
    def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "C:/Windows/Fonts/msyh.ttc",
        ]
        for item in candidates:
            if Path(item).exists():
                return ImageFont.truetype(item, size)
        return ImageFont.load_default()

    @staticmethod
    def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
        if not text:
            return [""]
        tokens = list(text) if re.search(r"[\u4e00-\u9fff]", text) else text.split()
        separator = "" if len(tokens) and len(tokens[0]) == 1 and re.search(r"[\u4e00-\u9fff]", text) else " "
        lines, current = [], ""
        for token in tokens:
            candidate = token if not current else current + separator + token
            if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = token
        if current:
            lines.append(current)
        return lines


class TTSTool:
    def run(self, text: str, output_path: Path, language: str, target_duration: float) -> tuple[Path, str]:
        require_binary("ffmpeg")
        if language.lower().startswith("zh"):
            try:
                import edge_tts  # type: ignore
                edge_output = output_path.with_suffix(".mp3")
                asyncio.run(edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural").save(str(edge_output)))
                return edge_output, "edge-tts"
            except Exception:
                self._silent(output_path, target_duration)
                return output_path, "silent-fallback"
        escaped = text.replace("'", "").replace(":", " ")[:900]
        try:
            run_command(["ffmpeg", "-y", "-f", "lavfi", "-i", f"flite=text='{escaped}':voice=slt", "-ar", "44100", str(output_path)])
            return output_path, "ffmpeg-flite"
        except Exception:
            self._silent(output_path, target_duration)
            return output_path, "silent-fallback"

    @staticmethod
    def _silent(output_path: Path, duration: float) -> None:
        run_command(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", f"{max(1, duration):.3f}", str(output_path)])


class SubtitleTool:
    def run(self, scenes: list[Scene], path: Path) -> Path:
        cursor = 0.0
        blocks = []
        for index, scene in enumerate(scenes, 1):
            end = cursor + scene.duration_seconds
            blocks.append(f"{index}\n{seconds_to_srt(cursor)} --> {seconds_to_srt(end)}\n{scene.narration}\n")
            cursor = end
        path.write_text("\n".join(blocks), encoding="utf-8")
        return path


class RenderTool:
    def render_scene(
        self,
        scene: Scene,
        output_path: Path,
        width: int,
        height: int,
        fit_mode: str = "pad",
        transition_seconds: float = 0.35,
    ) -> Path:
        if not scene.asset_path or not scene.audio_path:
            raise ValueError("Scene is missing asset or audio")
        require_binary("ffmpeg")
        audio_duration = max(1.0, ffprobe_duration(Path(scene.audio_path)))
        duration = max(scene.duration_seconds, audio_duration + 0.15)
        scene.duration_seconds = duration
        asset_path = Path(scene.asset_path)
        is_video = scene.asset_type == "video" or asset_path.suffix.lower() in AssetTool.SUPPORTED_VIDEOS
        video_input: list[str]
        video_options: list[str] = []
        input_filters: list[str] = []
        if is_video:
            source_duration = scene.asset_source_duration or ffprobe_duration(asset_path)
            scene.asset_source_duration = source_duration
            segment_start = scene.asset_segment_start_seconds
            available_duration = scene.asset_segment_duration or max(0.0, source_duration - segment_start)
            if available_duration + 0.05 < duration:
                scene.asset_looped = True
                scene.asset_start_seconds = segment_start
                frame_count = max(1, math.ceil(available_duration * 25))
                video_input = ["-i", str(asset_path)]
                input_filters = [
                    f"trim=start={segment_start:.3f}:duration={available_duration:.3f}",
                    "setpts=PTS-STARTPTS",
                    "fps=25",
                    f"loop=loop=-1:size={frame_count}:start=0",
                    "setpts=N/25/TB",
                ]
            else:
                scene.asset_looped = False
                slack = max(0.0, available_duration - duration)
                if scene.asset_reuse_index:
                    position = (scene.asset_reuse_index * 0.38196601125) % 1.0
                    scene.asset_start_seconds = segment_start + slack * position
                else:
                    scene.asset_start_seconds = segment_start + slack / 2
                video_input = ["-ss", f"{scene.asset_start_seconds:.3f}", "-i", str(asset_path)]
        else:
            scene.asset_looped = False
            scene.asset_start_seconds = 0.0
            video_input = ["-loop", "1", "-i", str(asset_path)]
            video_options = ["-tune", "stillimage"]

        run_command([
            "ffmpeg", "-y", *video_input, "-i", scene.audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-vf", self._visual_filter(
                width,
                height,
                fit_mode,
                duration,
                transition_seconds,
                input_filters=input_filters,
            ),
            *self._audio_transition_options(duration, transition_seconds),
            "-c:v", "libx264", "-preset", "veryfast", *video_options,
            "-c:a", "aac", "-b:a", "128k", "-t", f"{duration:.3f}", str(output_path),
        ])
        return output_path

    @staticmethod
    def _visual_filter(
        width: int,
        height: int,
        fit_mode: str,
        duration: float,
        transition_seconds: float,
        input_filters: list[str] | None = None,
    ) -> str:
        filters = list(input_filters or [])
        if fit_mode == "crop":
            filters.extend([
                f"scale={width}:{height}:force_original_aspect_ratio=increase",
                f"crop={width}:{height}",
            ])
        else:
            filters.extend([
                f"scale={width}:{height}:force_original_aspect_ratio=decrease",
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            ])
        filters.extend(["fps=25", "format=yuv420p"])
        fade_duration = min(max(0.0, transition_seconds), max(0.0, duration / 3))
        if fade_duration > 0:
            filters.extend([
                f"fade=t=in:st=0:d={fade_duration:.3f}",
                f"fade=t=out:st={max(0.0, duration - fade_duration):.3f}:d={fade_duration:.3f}",
            ])
        return ",".join(filters)

    @staticmethod
    def _audio_transition_options(duration: float, transition_seconds: float) -> list[str]:
        fade_duration = min(max(0.0, transition_seconds), max(0.0, duration / 3))
        if fade_duration <= 0:
            return []
        return [
            "-af",
            (
                f"afade=t=in:st=0:d={fade_duration:.3f},"
                f"afade=t=out:st={max(0.0, duration - fade_duration):.3f}:d={fade_duration:.3f}"
            ),
        ]

    def concatenate(self, clips: list[Path], output_path: Path) -> Path:
        if not clips:
            raise ValueError("No clips were provided for concatenation")
        concat_file = output_path.parent / "clips.txt"
        concat_file.write_text("\n".join(f"file '{p.resolve().as_posix()}'" for p in clips), encoding="utf-8")
        run_command(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(output_path)])
        return output_path

    def burn_subtitles(self, input_path: Path, subtitles_path: Path, output_path: Path) -> Path:
        if not subtitles_path.exists() or not subtitles_path.read_text(encoding="utf-8").strip():
            raise ValueError("Subtitle file is missing or empty")
        escaped_path = subtitles_path.resolve().as_posix().replace(":", r"\:").replace("'", r"\'")
        subtitle_filter = (
            f"subtitles=filename='{escaped_path}':"
            "force_style='FontName=Microsoft YaHei,FontSize=24,"
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            "BorderStyle=1,Outline=2,Shadow=0,MarginV=96,Alignment=2'"
        )
        run_command([
            "ffmpeg", "-y", "-i", str(input_path), "-vf", subtitle_filter,
            "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(output_path),
        ])
        return output_path


class QualityTool:
    def run(
        self,
        final_video: Path,
        scenes: list[Scene],
        subtitles_path: Path | None = None,
        expected_width: int | None = None,
        expected_height: int | None = None,
    ) -> dict:
        expected = sum(scene.duration_seconds for scene in scenes)
        actual = ffprobe_duration(final_video) if final_video.exists() else 0.0
        missing = [scene.scene_id for scene in scenes if not scene.asset_path or not Path(scene.asset_path).exists()]
        missing_audio = [scene.scene_id for scene in scenes if not scene.audio_path or not Path(scene.audio_path).exists()]
        missing_clips = [scene.scene_id for scene in scenes if not scene.clip_path or not Path(scene.clip_path).exists()]
        duration_delta = abs(actual - expected)
        streams: list[dict] = []
        if final_video.exists():
            probe = run_command([
                "ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height",
                "-of", "json", str(final_video),
            ])
            streams = json.loads(probe.stdout).get("streams", [])
        subtitle_ready = bool(
            subtitles_path
            and subtitles_path.exists()
            and subtitles_path.read_text(encoding="utf-8").strip()
        )
        video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
        resolution_matches = (
            not expected_width
            or not expected_height
            or (
                video_stream.get("width") == expected_width
                and video_stream.get("height") == expected_height
            )
        )
        checks = {
            "video_exists": final_video.exists() and final_video.stat().st_size > 1024,
            "video_stream_exists": any(stream.get("codec_type") == "video" for stream in streams),
            "audio_stream_exists": any(stream.get("codec_type") == "audio" for stream in streams),
            "resolution_matches_requested_format": resolution_matches,
            "all_scenes_have_assets": not missing,
            "all_scenes_have_audio": not missing_audio,
            "all_scenes_have_rendered_clips": not missing_clips,
            "subtitles_generated_and_burned": subtitle_ready,
            "duration_within_tolerance": duration_delta <= max(1.5, expected * 0.08),
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "expected_duration_seconds": round(expected, 3),
            "actual_duration_seconds": round(actual, 3),
            "duration_delta_seconds": round(duration_delta, 3),
            "missing_asset_scene_ids": missing,
            "missing_audio_scene_ids": missing_audio,
            "missing_clip_scene_ids": missing_clips,
            "video_asset_scene_ids": [scene.scene_id for scene in scenes if scene.asset_type == "video"],
            "streams": streams,
        }
