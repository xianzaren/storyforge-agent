from __future__ import annotations

import argparse
import json
from pathlib import Path

from storyforge import AnalysisConfig, OpenAICompatibleVisionTagger, VideoAnalysisAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyse a video into shots, similarity clusters, transcript and review labels."
    )
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("--language", default="auto", choices=["auto", "zh", "en"])
    parser.add_argument("--scene-threshold", type=float, default=0.30)
    parser.add_argument("--similarity-threshold", type=float, default=0.90)
    parser.add_argument("--min-segment-seconds", type=float, default=0.60)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/analysis"))
    parser.add_argument("--no-transcribe", action="store_true")
    parser.add_argument("--no-audio-boundaries", action="store_true")
    parser.add_argument("--audio-window", type=float, default=2.0)
    parser.add_argument("--audio-hop", type=float, default=1.0)
    parser.add_argument("--audio-change-threshold", type=float, default=0.45)
    parser.add_argument("--emotion-persistence-windows", type=int, default=2)
    parser.add_argument("--export-clips", action="store_true")
    parser.add_argument("--vision", action="store_true", help="Use configured OpenAI-compatible vision model")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = AnalysisConfig(
        output_dir=args.output_dir,
        scene_threshold=args.scene_threshold,
        similarity_threshold=args.similarity_threshold,
        min_segment_seconds=args.min_segment_seconds,
        transcribe=not args.no_transcribe,
        export_clips=args.export_clips,
        enable_audio_boundaries=not args.no_audio_boundaries,
        audio_window_seconds=args.audio_window,
        audio_hop_seconds=args.audio_hop,
        audio_change_threshold=args.audio_change_threshold,
        emotion_persistence_windows=args.emotion_persistence_windows,
    )
    agent = VideoAnalysisAgent(
        config,
        visual_tagger=OpenAICompatibleVisionTagger() if args.vision else None,
        on_event=lambda event, payload: print(f"[{event}]", flush=True),
    )
    result = agent.run(args.video, language=args.language)
    print(json.dumps({
        "status": result.status,
        "analysis_id": result.analysis_id,
        "segments": len(result.segments),
        "clusters": len({item.cluster_id for item in result.segments}),
        "artifacts": result.artifacts,
        "warnings": result.warnings,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
