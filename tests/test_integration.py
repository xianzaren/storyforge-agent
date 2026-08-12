from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from storyforge import WorkflowAgent, WorkflowConfig
from storyforge.utils import run_command


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is not available")
class WorkflowIntegrationTests(unittest.TestCase):
    def test_offline_workflow_generates_subtitled_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = WorkflowAgent(WorkflowConfig(
                runs_dir=Path(directory),
                width=640,
                height=360,
                max_retries=1,
            )).run("Offline workflow integration test", target_duration=8, language="en")

            self.assertIn(state.status, {"completed", "completed_with_warnings"})
            self.assertIsNone(state.error)
            self.assertEqual(state.artifacts["quality_passed"], "true")
            self.assertEqual(state.artifacts["subtitles_burned"], "true")

            final_video = Path(state.artifacts["video"])
            raw_video = Path(state.artifacts["raw_video"])
            subtitles = Path(state.artifacts["subtitles"])
            report = json.loads(Path(state.artifacts["quality_report"]).read_text(encoding="utf-8"))

            self.assertTrue(final_video.exists() and final_video.stat().st_size > 1024)
            self.assertTrue(raw_video.exists() and raw_video.stat().st_size > 1024)
            self.assertTrue(subtitles.exists() and subtitles.stat().st_size > 0)
            self.assertTrue(report["checks"]["video_stream_exists"])
            self.assertTrue(report["checks"]["audio_stream_exists"])
            self.assertTrue(report["checks"]["subtitles_generated_and_burned"])

    def test_image_and_video_assets_are_mixed_and_rendered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets_dir = root / "assets"
            assets_dir.mkdir()
            Image.new("RGB", (320, 180), (30, 80, 150)).save(assets_dir / "fast_topic_introduction.png")
            video_path = assets_dir / "creator_workflow.mp4"
            run_command([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=320x180:d=2",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-shortest",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(video_path),
            ])
            long_video_path = assets_dir / "workflow_tools_long.mp4"
            run_command([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25:duration=8",
                "-f", "lavfi", "-i", "sine=frequency=660:duration=8", "-shortest",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(long_video_path),
            ])

            state = WorkflowAgent(WorkflowConfig(
                runs_dir=root / "runs",
                assets_dir=assets_dir,
                width=640,
                height=360,
                max_retries=1,
                enable_scene_detection=False,
            )).run("Mixed media demo", target_duration=8, language="en")

            self.assertIn(state.status, {"completed", "completed_with_warnings"})
            self.assertEqual(state.artifacts["quality_passed"], "true")
            asset_types = {scene.asset_type for scene in state.scenes}
            self.assertEqual(asset_types, {"image", "video"})

            manifest = json.loads(Path(state.artifacts["assets_manifest"]).read_text(encoding="utf-8"))
            video_assignments = [item for item in manifest["assignments"] if item["asset_type"] == "video"]
            self.assertTrue(video_assignments)
            self.assertTrue(any(item["looped"] for item in video_assignments))
            self.assertTrue(any(not item["looped"] and item["trim_start_seconds"] > 0 for item in video_assignments))
            self.assertTrue(all(item["source_audio_ignored"] for item in video_assignments))
            self.assertTrue(Path(state.artifacts["video"]).exists())

    def test_scene_detection_indexes_distinct_video_shots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "three_scenes.mp4"
            run_command([
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "color=c=red:s=320x180:d=1",
                "-f", "lavfi", "-i", "color=c=green:s=320x180:d=1",
                "-f", "lavfi", "-i", "color=c=blue:s=320x180:d=1",
                "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
                "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video_path),
            ])
            from storyforge.tools import AssetTool

            records = AssetTool(
                640,
                360,
                enable_scene_detection=True,
                scene_detection_threshold=0.15,
            ).index(root)
            video_records = [record for record in records if record.media_type == "video"]
            # Scene detection is threshold-based; adjacent synthetic colours may
            # merge, but the file must still be split into multiple candidates.
            self.assertGreaterEqual(len(video_records), 2)
            self.assertEqual(video_records[0].segment_start_seconds, 0.0)
            self.assertTrue(all(record.shot_index is not None for record in video_records))

    def test_unmatched_mcn_upload_is_used_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets_dir = root / "assets"
            assets_dir.mkdir()
            video_path = assets_dir / "MCN.mp4"
            run_command([
                "ffmpeg", "-y", "-f", "lavfi", "-i",
                "testsrc2=size=320x180:rate=25:duration=5",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=5", "-shortest",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(video_path),
            ])

            state = WorkflowAgent(WorkflowConfig(
                runs_dir=root / "runs",
                assets_dir=assets_dir,
                width=640,
                height=360,
                max_retries=1,
                enable_scene_detection=False,
                use_unmatched_assets=True,
            )).run("How AI helps creators", target_duration=8, language="en")

            self.assertIn(state.status, {"completed", "completed_with_warnings"})
            self.assertEqual(state.artifacts["quality_passed"], "true")
            self.assertTrue(all(scene.asset_type == "video" for scene in state.scenes))
            self.assertTrue(all(Path(scene.asset_path or "").name == "MCN.mp4" for scene in state.scenes))
            self.assertTrue(all(
                scene.asset_selection_reason == "unmatched_user_fallback" for scene in state.scenes
            ))
            self.assertEqual(
                [scene.asset_reuse_index for scene in state.scenes],
                list(range(len(state.scenes))),
            )
            self.assertTrue(any("shorter" in warning for warning in state.warnings))


if __name__ == "__main__":
    unittest.main()
