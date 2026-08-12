from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import streamlit as st

from storyforge import WorkflowAgent, WorkflowConfig


st.set_page_config(page_title="StoryForge Agent", page_icon="🎬", layout="wide")
st.title("StoryForge Agent")
st.caption("主题 → 结构化脚本 → 分镜与素材 → 配音与字幕 → FFmpeg 成片 → 质量检查")

with st.sidebar:
    topic = st.text_input("视频主题", "How AI helps creators")
    duration = st.slider("目标时长（秒）", 12, 60, 24, 4)
    language = st.selectbox("语言", ["en", "zh"])
    aspect_ratio = st.selectbox("画面比例", ["16:9 横屏", "9:16 竖屏", "1:1 方形"])
    fit_mode = st.selectbox("素材适配", ["pad", "crop"], format_func=lambda item: "完整留边" if item == "pad" else "铺满裁剪")
    subtitle_source = st.selectbox(
        "字幕来源",
        ["generated_narration", "source_audio"],
        format_func=lambda item: "根据主题生成" if item == "generated_narration" else "识别上传视频原声",
        help="原声识别需要安装 faster-whisper；不可用或视频没有语音时会明确降级到主题脚本。",
    )
    audio_mode = st.selectbox(
        "成片音轨",
        ["narration_replace", "source_original", "source_narration_mix"],
        format_func=lambda item: {
            "narration_replace": "AI 配音替换原声",
            "source_original": "保留原音轨",
            "source_narration_mix": "原声 + AI 配音混合",
        }[item],
    )
    source_audio_volume = st.slider(
        "原声音量",
        0.0,
        2.0,
        1.0 if audio_mode == "source_original" else 0.3,
        0.05,
        disabled=audio_mode == "narration_replace",
    )
    narration_volume = st.slider(
        "AI 配音音量",
        0.0,
        2.0,
        1.0,
        0.05,
        disabled=audio_mode == "source_original",
    )
    transition_seconds = st.slider("淡入淡出（秒）", 0.0, 1.0, 0.35, 0.05)
    enable_scene_detection = st.checkbox("自动镜头检测", value=True)
    scene_threshold = st.slider("镜头变化阈值", 0.1, 0.8, 0.35, 0.05, disabled=not enable_scene_detection)
    use_unmatched_assets = st.checkbox(
        "未匹配关键词时仍使用上传素材",
        value=True,
        help="建议保持开启。关闭后，文件名或 metadata.json 标签未命中分镜关键词时会生成标题卡。",
    )
    uploads = st.file_uploader(
        "上传可选图片或视频素材",
        type=["png", "jpg", "jpeg", "mp4", "mov", "mkv", "avi", "webm"],
        accept_multiple_files=True,
        help="视频素材会自动裁剪或循环，原声默认静音并使用 AI 旁白。",
    )

events_box = st.empty()
event_lines: list[str] = []

if st.button("生成视频", type="primary", use_container_width=True):
    assets_dir = None
    temp_dir = None
    if uploads:
        temp_dir = tempfile.TemporaryDirectory()
        assets_dir = Path(temp_dir.name)
        for upload in uploads:
            (assets_dir / upload.name).write_bytes(upload.getbuffer())

    def on_event(event: str, item: dict) -> None:
        event_lines.append(f"{event}: {item.get('step') or item.get('reason') or ''}")
        events_box.code("\n".join(event_lines[-12:]), language="text")

    with st.spinner("Agent 正在执行工作流..."):
        dimensions = {
            "16:9 横屏": (1280, 720),
            "9:16 竖屏": (720, 1280),
            "1:1 方形": (1080, 1080),
        }
        width, height = dimensions[aspect_ratio]
        agent = WorkflowAgent(WorkflowConfig(
            runs_dir=Path("artifacts/runs"),
            assets_dir=assets_dir,
            width=width,
            height=height,
            fit_mode=fit_mode,
            transition_seconds=transition_seconds,
            enable_scene_detection=enable_scene_detection,
            scene_detection_threshold=scene_threshold,
            use_unmatched_assets=use_unmatched_assets,
            subtitle_source=subtitle_source,
            audio_mode=audio_mode,
            source_audio_volume=source_audio_volume,
            narration_volume=narration_volume,
        ), on_event=on_event)
        state = agent.run(topic=topic, target_duration=duration, language=language)

    if temp_dir:
        temp_dir.cleanup()

    if state.status == "failed":
        st.error(state.error)
    else:
        if state.status == "completed_with_warnings":
            st.warning("任务已完成，但存在降级或质量警告。")
            for warning in state.warnings:
                st.caption(f"⚠️ {warning}")
        else:
            st.success(f"任务完成：{state.status}")
        left, right = st.columns([2, 1])
        with left:
            video_path = Path(state.artifacts["video"])
            st.video(video_path.read_bytes())
            st.download_button("下载 MP4", video_path.read_bytes(), file_name=f"{state.task_id}.mp4", mime="video/mp4")
        with right:
            if uploads:
                st.subheader("媒体理解")
                st.caption(
                    f"脚本来源：{state.artifacts.get('script_provider', 'unknown')} · "
                    f"转写引擎：{state.artifacts.get('transcription_provider', 'none')}"
                )
                st.subheader("素材采用情况")
                st.dataframe(
                    [{
                        "分镜": scene.scene_id,
                        "素材": Path(scene.asset_path).name if scene.asset_path else "",
                        "类型": scene.asset_type,
                        "采用原因": scene.asset_selection_reason,
                        "匹配分": scene.asset_match_score,
                        "镜头": scene.asset_shot_index,
                        "取片起点": round(scene.asset_start_seconds, 2),
                        "循环": scene.asset_looped,
                        "音轨": scene.audio_mode,
                        "保留原声": scene.source_audio_used,
                        "字幕来源": scene.subtitle_source,
                    } for scene in state.scenes],
                    use_container_width=True,
                    hide_index=True,
                )
            st.subheader("任务状态")
            st.json(state.to_dict())
