from __future__ import annotations

import argparse
import sys
from pathlib import Path

from storyforge import WorkflowAgent, WorkflowConfig


def positive_duration(value: str) -> int:
    duration = int(value)
    if duration <= 0:
        raise argparse.ArgumentTypeError("duration must be greater than zero")
    return duration


def unit_interval(value: str) -> float:
    number = float(value)
    if not 0.05 <= number <= 0.95:
        raise argparse.ArgumentTypeError("value must be between 0.05 and 0.95")
    return number


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Generate a short video with the StoryForge workflow agent.")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--duration", type=positive_duration, default=30)
    parser.add_argument("--language", default="en", choices=["en", "zh"])
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--max-retries", type=int, default=1, choices=range(0, 4), metavar="0-3")
    parser.add_argument("--fit-mode", choices=["pad", "crop"], default="pad")
    parser.add_argument("--aspect-ratio", choices=["16:9", "9:16", "1:1"], default="16:9")
    parser.add_argument("--transition-seconds", type=float, default=0.35)
    parser.add_argument("--scene-threshold", type=unit_interval, default=0.35)
    parser.add_argument("--disable-scene-detection", action="store_true")
    parser.add_argument(
        "--strict-asset-matching",
        action="store_true",
        help="Generate title cards instead of using uploaded assets whose filename/tags do not match a scene.",
    )
    args = parser.parse_args()
    dimensions = {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (1080, 1080)}
    width, height = dimensions[args.aspect_ratio]

    def show(event: str, item: dict) -> None:
        detail = item.get("step") or item.get("reason") or ""
        print(f"[{event}] {detail}")

    state = WorkflowAgent(WorkflowConfig(
        runs_dir=args.runs_dir,
        assets_dir=args.assets_dir,
        max_retries=args.max_retries,
        width=width,
        height=height,
        fit_mode=args.fit_mode,
        transition_seconds=max(0.0, args.transition_seconds),
        enable_scene_detection=not args.disable_scene_detection,
        scene_detection_threshold=args.scene_threshold,
        use_unmatched_assets=not args.strict_asset_matching,
    ), on_event=show).run(
        topic=args.topic, target_duration=args.duration, language=args.language
    )
    print(f"status={state.status}")
    print(f"task_id={state.task_id}")
    if state.artifacts.get("video"):
        print(f"video={state.artifacts['video']}")
    for warning in state.warnings:
        print(f"warning={warning}")
    if state.error:
        print(state.error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
