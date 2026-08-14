from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from storyforge import AnalysisConfig, VideoAnalysisAgent, VideoAnalysisResult
from storyforge.analyzer import SegmentAnnotation
from storyforge.audio_features import AudioFeatureTool
from storyforge.media import MediaAnalysisTool, TranscriptSegment
from storyforge.packager import MaterialPackager, PackageConfig
from storyforge.utils import ffprobe_duration, run_command
from storyforge.vision import OpenAICompatibleVisionTagger


class AnalysisUnitTests(unittest.TestCase):
    @staticmethod
    def _package_segment(
        segment_id: int,
        start: float,
        cluster: int,
        emotion: str = "平静",
        reasons: list[str] | None = None,
    ) -> SegmentAnnotation:
        return SegmentAnnotation(
            segment_id=segment_id,
            start_seconds=start,
            end_seconds=start + 2,
            duration_seconds=2,
            thumbnail_path="unused.jpg",
            cluster_id=cluster,
            cluster_similarity=1,
            boundary_reasons=reasons or (["视频开始"] if segment_id == 1 else []),
            content_summary="测试内容",
            content_confidence=0.7,
            suggested_emotion=emotion,
            emotion_confidence=0.7,
        )

    def test_emotion_annotation_includes_evidence_and_highlight_score(self) -> None:
        segment = SegmentAnnotation(
            segment_id=1,
            start_seconds=0,
            end_seconds=2,
            duration_seconds=2,
            thumbnail_path="unused.jpg",
            cluster_id=1,
            cluster_similarity=1,
            transcript="Amazing, we won this fight!",
            rms_db=-8,
            peak_db=-2,
            motion_score=0.4,
            speech_present=True,
        )
        VideoAnalysisAgent()._annotate_emotions([segment])
        self.assertIn(segment.suggested_emotion, {"兴奋", "紧张"})
        self.assertGreater(segment.emotion_confidence, 0.5)
        self.assertTrue(segment.emotion_evidence)
        self.assertGreater(segment.highlight_score, 0.5)

    def test_transcript_alignment_only_uses_overlapping_speech(self) -> None:
        transcript = [
            TranscriptSegment(0.0, 1.0, "first"),
            TranscriptSegment(1.2, 2.0, "second"),
            TranscriptSegment(2.5, 3.0, "third"),
        ]
        value = VideoAnalysisAgent._transcript_for_range(transcript, 0.8, 2.2)
        self.assertEqual(value, "first second")

    def test_short_shots_are_merged_without_leaving_timeline_gaps(self) -> None:
        merged = VideoAnalysisAgent._merge_short_segments([0.0, 0.2, 1.0, 1.2], 0.6)
        self.assertEqual(merged, [0.0, 1.2])

    def test_nearby_visual_cuts_are_not_merged_as_duplicate_audio_candidates(self) -> None:
        agent = VideoAnalysisAgent(AnalysisConfig(
            min_segment_seconds=0.4,
            boundary_merge_seconds=1.0,
        ))
        boundaries = agent._fuse_boundaries([0.0, 1.0, 2.0, 3.0], [], 3.0)
        self.assertEqual([item.time_seconds for item in boundaries], [0.0, 1.0, 2.0, 3.0])

    def test_vision_tagger_validates_structured_model_response(self) -> None:
        class FakeClient:
            vision_enabled = True

            @staticmethod
            def generate_vision_json(path, system, user):
                return {"summary": "A person beside a bicycle", "tags": ["person", "bicycle"], "confidence": 0.84}

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.jpg"
            image.write_bytes(b"not-read-by-fake-client")
            summary, tags, confidence, provider = OpenAICompatibleVisionTagger(FakeClient())(image)
        self.assertEqual(summary, "A person beside a bicycle")
        self.assertEqual(tags, ["person", "bicycle"])
        self.assertEqual(confidence, 0.84)
        self.assertTrue(provider.startswith("openai-compatible:"))

    def test_persistent_audio_and_text_emotion_change_creates_boundary(self) -> None:
        rate = 4000
        first_time = np.arange(rate * 4, dtype=np.float32) / rate
        second_time = np.arange(rate * 4, dtype=np.float32) / rate
        samples = np.concatenate([
            0.03 * np.sin(2 * np.pi * 180 * first_time),
            0.75 * np.sin(2 * np.pi * 900 * second_time),
        ]).astype(np.float32)
        transcript = [
            TranscriptSegment(0.0, 3.7, "I am sad and sorry we lost."),
            TranscriptSegment(4.3, 8.0, "Amazing, we won! Great success!"),
        ]
        windows = AudioFeatureTool.analyse(
            (samples, rate), 8.0, transcript, window_seconds=1.0, hop_seconds=0.5
        )
        boundaries = AudioFeatureTool.change_boundaries(
            windows, threshold=0.45, persistence_windows=2, merge_seconds=1.0
        )
        self.assertTrue(boundaries)
        closest = min(boundaries, key=lambda item: abs(item.time_seconds - 4.0))
        self.assertLess(abs(closest.time_seconds - 4.0), 1.1)
        self.assertTrue(any("情绪变化" in reason for reason in closest.reasons))
        self.assertTrue(any("能量" in reason or "声学状态" in reason for reason in closest.reasons))

    def test_one_window_spike_is_ignored_by_persistence_rule(self) -> None:
        rate = 2000
        samples = np.full(rate * 8, 0.01, dtype=np.float32)
        samples[rate * 3:rate * 4] = 0.9
        windows = AudioFeatureTool.analyse(
            (samples, rate), 8.0, [], window_seconds=1.0, hop_seconds=1.0
        )
        boundaries = AudioFeatureTool.change_boundaries(
            windows, threshold=0.45, persistence_windows=2, merge_seconds=0.8
        )
        self.assertEqual(boundaries, [])

    def test_material_packages_split_on_emotion_boundary_and_link_similar_groups(self) -> None:
        segments = [
            self._package_segment(1, 0, 1),
            self._package_segment(2, 2, 1),
            self._package_segment(3, 4, 2, "紧张", ["文本情绪变化:平静→紧张"]),
            self._package_segment(4, 6, 1, "平静", ["画面转场"]),
        ]
        packager = MaterialPackager(PackageConfig(export_media=False))
        with tempfile.TemporaryDirectory() as directory:
            packages = packager.build(Path("source.mp4"), segments, Path(directory))
        self.assertEqual([item.segment_ids for item in packages], [[1, 2], [3], [4]])
        self.assertEqual(packages[0].similar_group_id, packages[2].similar_group_id)
        self.assertNotEqual(packages[0].similar_group_id, packages[1].similar_group_id)

    def test_material_packages_respect_keep_and_manual_labels(self) -> None:
        first = self._package_segment(1, 0, 1)
        first.user_label = "开场介绍"
        first.review_emotion = "期待"
        omitted = self._package_segment(2, 2, 1)
        omitted.keep = False
        third = self._package_segment(3, 4, 1)
        packager = MaterialPackager(PackageConfig(export_media=False))
        with tempfile.TemporaryDirectory() as directory:
            packages = packager.build(Path("source.mp4"), [first, omitted, third], Path(directory))
        self.assertEqual([item.segment_ids for item in packages], [[1], [3]])
        self.assertIn("开场介绍", packages[0].content_tags)
        self.assertEqual(packages[0].dominant_emotion, "期待")


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is not available")
class AnalysisIntegrationTests(unittest.TestCase):
    @staticmethod
    def _transcriber(path: Path, language: str):
        return ([
            TranscriptSegment(0.0, 0.9, "Amazing, we won this round."),
            TranscriptSegment(1.0, 1.9, "Warning, danger ahead."),
            TranscriptSegment(2.0, 2.9, "The red view returns."),
        ], "en", "test-transcriber")

    @staticmethod
    def _make_three_shot_video(path: Path) -> None:
        run_command([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=320x180:r=25:d=1",
            "-f", "lavfi", "-i", "color=c=white:s=320x180:r=25:d=1",
            "-f", "lavfi", "-i", "color=c=black:s=320x180:r=25:d=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map", "[v]", "-map", "3:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path),
        ])

    def test_video_analysis_slices_clusters_transcribes_and_exports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "red_green_red.mp4"
            self._make_three_shot_video(source)
            agent = VideoAnalysisAgent(
                AnalysisConfig(
                    output_dir=root / "analysis",
                    scene_threshold=0.10,
                    similarity_threshold=0.95,
                    min_segment_seconds=0.4,
                    transcribe=True,
                    build_material_packages=True,
                ),
                media_tool=MediaAnalysisTool(self._transcriber),
            )
            result = agent.run(source, language="en")

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.transcription_provider, "test-transcriber")
            self.assertGreaterEqual(len(result.segments), 3)
            self.assertEqual(result.segments[0].cluster_id, result.segments[2].cluster_id)
            self.assertNotEqual(result.segments[0].cluster_id, result.segments[1].cluster_id)
            self.assertIn("won", result.segments[0].transcript)
            self.assertEqual(result.segments[0].suggested_emotion, "兴奋")
            self.assertGreater(result.segments[0].content_confidence, 0.5)

            for key in ["analysis_json", "segments_csv", "transcript_srt", "events", "thumbnails"]:
                self.assertTrue(Path(result.artifacts[key]).exists(), key)
            self.assertTrue(Path(result.artifacts["package_manifest"]).exists())
            self.assertTrue(Path(result.artifacts["packages_archive"]).exists())
            self.assertGreaterEqual(len(result.packages), 3)
            first_package = result.packages[0]
            for key in ["preview_path", "contact_sheet_path", "subtitle_path"]:
                self.assertTrue(Path(first_package[key]).exists(), key)
            self.assertTrue(all(Path(path).exists() for path in first_package["clip_paths"]))
            self.assertEqual(result.packages[0]["similar_group_id"], result.packages[2]["similar_group_id"])
            loaded = VideoAnalysisResult.load(Path(result.artifacts["analysis_json"]))
            self.assertEqual(len(loaded.segments), len(result.segments))

            output = agent.export_selection(result, [1, 3])
            self.assertTrue(output.exists() and output.stat().st_size > 1024)
            self.assertGreater(ffprobe_duration(output), 1.5)
            selection = json.loads((output.parent / "selection.json").read_text(encoding="utf-8"))
            self.assertEqual(selection["selected_segment_ids"], [1, 3])

    def test_no_transcription_does_not_invent_semantic_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "silent.mp4"
            run_command([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=320x180:d=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
            ])
            result = VideoAnalysisAgent(AnalysisConfig(
                output_dir=root / "analysis",
                transcribe=False,
                min_segment_seconds=0.2,
            )).run(source)
            self.assertEqual(len(result.segments), 1)
            self.assertFalse(result.segments[0].speech_present)
            self.assertIn("未识别具体语义", result.segments[0].content_summary)
            self.assertLess(result.segments[0].content_confidence, 0.5)

    def test_static_video_is_split_by_persistent_audio_emotion_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "static_with_emotion_change.mp4"
            run_command([
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=25:d=8",
                "-f", "lavfi", "-i", "sine=frequency=180:duration=4",
                "-f", "lavfi", "-i", "sine=frequency=900:duration=4",
                "-filter_complex",
                "[1:a]volume=0.03[a1];[2:a]volume=0.9[a2];[a1][a2]concat=n=2:v=0:a=1[a]",
                "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", str(source),
            ])

            def changing_transcriber(path: Path, language: str):
                return ([
                    TranscriptSegment(0.0, 3.7, "I am sad and sorry we lost."),
                    TranscriptSegment(4.3, 8.0, "Amazing, we won! Great success!"),
                ], "en", "emotion-test-transcriber")

            result = VideoAnalysisAgent(
                AnalysisConfig(
                    output_dir=root / "analysis",
                    scene_threshold=0.30,
                    min_segment_seconds=1.0,
                    audio_window_seconds=1.0,
                    audio_hop_seconds=0.5,
                    audio_change_threshold=0.45,
                    emotion_persistence_windows=2,
                    transcribe=True,
                ),
                media_tool=MediaAnalysisTool(changing_transcriber),
            ).run(source, language="en")
            self.assertGreaterEqual(len(result.segments), 2)
            audio_starts = [
                item for item in result.segments[1:]
                if any("情绪变化" in reason or "声学状态变化" in reason for reason in item.boundary_reasons)
            ]
            self.assertTrue(audio_starts)
            self.assertLess(abs(audio_starts[0].start_seconds - 4.0), 1.2)
            timeline = json.loads(Path(result.artifacts["audio_timeline"]).read_text(encoding="utf-8"))
            self.assertGreater(len(timeline["windows"]), 8)
            self.assertTrue(timeline["emotion_boundaries"])


if __name__ == "__main__":
    unittest.main()
