from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from storyforge.models import ProjectState, Scene
from storyforge.tools import AssetRecord, AssetTool, RenderTool, ScriptTool, StoryboardTool, SubtitleTool
from storyforge.workflow import WorkflowAgent, WorkflowConfig


class CoreTests(unittest.TestCase):
    def test_fallback_script_is_structured(self) -> None:
        scenes = ScriptTool().run("AI for creators", 24, "en")
        self.assertGreaterEqual(len(scenes), 3)
        self.assertTrue(all(scene.narration and scene.visual_query for scene in scenes))

    def test_llm_json_response_is_parsed(self) -> None:
        class FakeLLM:
            enabled = True

            @staticmethod
            def generate_json(system: str, user: str) -> dict:
                return {"scenes": [{
                    "title": "Model scene",
                    "narration": "Structured narration",
                    "visual_query": "structured visual",
                    "on_screen_text": "Model",
                    "duration_seconds": 5,
                }]}

        tool = ScriptTool(FakeLLM())  # type: ignore[arg-type]
        scenes = tool.run("topic", 5, "en")
        self.assertEqual(tool.last_provider, "openai-compatible")
        self.assertEqual(scenes[0].title, "Model scene")

    def test_llm_failure_uses_recorded_template_fallback(self) -> None:
        class FailingLLM:
            enabled = True

            @staticmethod
            def generate_json(system: str, user: str) -> dict:
                raise TimeoutError("model timeout")

        tool = ScriptTool(FailingLLM())  # type: ignore[arg-type]
        scenes = tool.run("topic", 12, "en")
        self.assertEqual(tool.last_provider, "template-fallback")
        self.assertIn("TimeoutError", tool.last_error or "")
        self.assertGreaterEqual(len(scenes), 3)

    def test_asset_recovery_creates_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scene = Scene(1, "Hook", "Narration", "no asset", "Hello", 4)
            path, generated = AssetTool(640, 360).match_or_create(scene, [], Path(directory))
            self.assertTrue(generated)
            self.assertTrue(path.exists())

    def test_video_asset_is_preferred_for_equal_keyword_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "creator.jpg"
            video_path = root / "creator.mp4"
            records = [
                AssetRecord(image_path, {"creator"}, "image", width=640, height=360),
                AssetRecord(video_path, {"creator"}, "video", duration_seconds=2, width=640, height=360),
            ]
            scene = Scene(1, "Creator", "Narration", "creator", "Creator", 4)
            path, generated = AssetTool(640, 360).match_or_create(scene, records, root)
            self.assertFalse(generated)
            self.assertEqual(path, video_path)
            self.assertEqual(scene.asset_type, "video")
            self.assertEqual(scene.asset_match_score, 1)

    def test_unmatched_uploaded_video_is_used_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "MCN.mp4"
            records = [AssetRecord(video_path, {"mcn"}, "video", duration_seconds=10)]
            scene = Scene(1, "开场", "旁白", "主题概览", "开场", 4)

            path, generated = AssetTool(640, 360).match_or_create(scene, records, root)

            self.assertFalse(generated)
            self.assertEqual(path, video_path)
            self.assertEqual(scene.asset_match_score, 0)
            self.assertEqual(scene.asset_selection_reason, "unmatched_user_fallback")

    def test_strict_matching_can_still_generate_a_title_card(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "MCN.mp4"
            records = [AssetRecord(video_path, {"mcn"}, "video", duration_seconds=10)]
            scene = Scene(1, "开场", "旁白", "主题概览", "开场", 4)

            path, generated = AssetTool(640, 360, use_unmatched_assets=False).match_or_create(
                scene, records, root
            )

            self.assertTrue(generated)
            self.assertNotEqual(path, video_path)
            self.assertEqual(scene.asset_selection_reason, "generated_title_card")

    def test_unused_video_shots_are_selected_before_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "workflow.mp4"
            records = [
                AssetRecord(video_path, {"workflow"}, "video", 3, shot_index=1),
                AssetRecord(video_path, {"workflow"}, "video", 3, segment_start_seconds=3, shot_index=2),
            ]
            used: set[str] = set()
            first = Scene(1, "Workflow", "First", "workflow", "First", 2)
            second = Scene(2, "Workflow", "Second", "workflow", "Second", 2)
            tool = AssetTool(640, 360)
            tool.match_or_create(first, records, root, used_assets=used)
            tool.match_or_create(second, records, root, used_assets=used)
            self.assertNotEqual(first.asset_shot_index, second.asset_shot_index)

    def test_crop_and_transition_filter_is_constructed(self) -> None:
        value = RenderTool._visual_filter(720, 1280, "crop", 5, 0.4)
        self.assertIn("force_original_aspect_ratio=increase", value)
        self.assertIn("crop=720:1280", value)
        self.assertIn("fade=t=in", value)
        self.assertIn("fade=t=out", value)

    def test_subtitles_have_valid_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scenes = [Scene(1, "A", "First", "a", "A", 3), Scene(2, "B", "Second", "b", "B", 4)]
            path = SubtitleTool().run(scenes, Path(directory) / "out.srt")
            text = path.read_text(encoding="utf-8")
            self.assertIn("00:00:00,000 --> 00:00:03,000", text)
            self.assertIn("00:00:03,000 --> 00:00:07,000", text)

    def test_storyboard_normalises_render_fields(self) -> None:
        scenes = [Scene(99, "  Hook  ", "  Narration  ", "", "", 3)]
        result = StoryboardTool().run(scenes)
        self.assertEqual(result[0].scene_id, 1)
        self.assertEqual(result[0].title, "Hook")
        self.assertEqual(result[0].visual_query, "Hook")
        self.assertEqual(result[0].on_screen_text, "Hook")

    def test_failed_step_is_retried_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            state = ProjectState("test", "topic", "en", 10)
            attempts = 0

            def recover_on_retry() -> None:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("temporary failure")

            agent = WorkflowAgent(WorkflowConfig(runs_dir=run_dir, max_retries=1))
            agent._step(state, run_dir, "recoverable", recover_on_retry)

            self.assertEqual(attempts, 2)
            self.assertEqual(state.retries["recoverable"], 1)
            self.assertTrue(any("recovered" in warning for warning in state.warnings))
            events = (run_dir / "events.jsonl").read_text(encoding="utf-8")
            self.assertIn('"event": "step_retry"', events)


if __name__ == "__main__":
    unittest.main()
