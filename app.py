from __future__ import annotations

import hashlib
import os
from pathlib import Path

import streamlit as st

from storyforge import (
    AnalysisConfig,
    OpenAICompatibleVisionTagger,
    VideoAnalysisAgent,
    VideoAnalysisResult,
)
from storyforge.utils import write_json


ARTIFACTS = Path("artifacts")
INPUTS = ARTIFACTS / "analysis-inputs"


def save_upload(upload) -> Path:
    payload = upload.getvalue()
    digest = hashlib.sha256(payload).hexdigest()[:12]
    safe_name = Path(upload.name).name
    target = INPUTS / digest / safe_name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size != len(payload):
        target.write_bytes(payload)
    return target


def load_current_result() -> VideoAnalysisResult | None:
    value = st.session_state.get("analysis_result_path")
    if not value:
        return None
    path = Path(value)
    if not path.exists():
        return None
    return VideoAnalysisResult.load(path)


def rows_from_result(result: VideoAnalysisResult) -> list[dict]:
    return [
        {
            "保留": item.keep,
            "片段": item.segment_id,
            "开始": item.start_seconds,
            "结束": item.end_seconds,
            "时长": item.duration_seconds,
            "切点依据": "；".join(item.boundary_reasons),
            "切点分数": item.boundary_score,
            "镜头组": item.cluster_id,
            "内容摘要": item.content_summary,
            "语音文本": item.transcript,
            "AI情绪建议": item.suggested_emotion,
            "情绪置信度": item.emotion_confidence,
            "人工情绪": item.review_emotion,
            "人工内容标签": item.user_label,
            "复核备注": item.review_notes,
            "高光分": item.highlight_score,
            "证据": "；".join(item.emotion_evidence),
            "视觉特征": "；".join(item.visual_tags),
        }
        for item in result.segments
    ]


def apply_review(result: VideoAnalysisResult, rows: list[dict]) -> Path:
    indexed = {int(row["片段"]): row for row in rows}
    for item in result.segments:
        row = indexed.get(item.segment_id)
        if not row:
            continue
        item.keep = bool(row.get("保留", True))
        item.review_emotion = str(row.get("人工情绪") or "").strip()
        item.user_label = str(row.get("人工内容标签") or "").strip()
        item.review_notes = str(row.get("复核备注") or "").strip()
    output = Path(result.artifacts["analysis_json"]).parent / "reviewed_analysis.json"
    result.artifacts["reviewed_json"] = str(output)
    write_json(output, result.to_dict())
    st.session_state["analysis_result_path"] = str(output)
    return output


st.set_page_config(page_title="StoryForge 素材分析助手", page_icon="🎞️", layout="wide")
st.title("StoryForge · AI 视频素材分析与前期标注助手")
st.caption("理解长视频、自动切片、聚类相似镜头，并给出带证据的内容与情绪建议；最终剪辑决定由你完成。")

with st.expander("当前能力边界", expanded=False):
    st.markdown(
        "- 本地完成：转场切片、缩略图、颜色相似度聚类、画面变化、音频能量、片段导出。\n"
        "- 可选 Whisper：识别语音并按时间轴对齐，不会因为视频无台词而生成虚构字幕。\n"
        "- 当前规则只提供可验证的视觉特征；人物、物体、动作等具体语义需接入视觉模型后才会输出。"
    )

settings, workspace = st.columns([1, 2.4])
with settings:
    st.subheader("1. 导入并分析")
    upload = st.file_uploader(
        "上传一个长视频",
        type=["mp4", "mov", "mkv", "avi", "webm"],
        key="analysis_video",
    )
    language = st.selectbox("语音语言", ["auto", "zh", "en"], key="analysis_language")
    transcribe = st.checkbox(
        "识别原视频语音（需要 faster-whisper）",
        value=True,
        key="analysis_transcribe",
    )
    vision_configured = bool(os.getenv("STORYFORGE_API_KEY") and os.getenv("STORYFORGE_VISION_MODEL"))
    use_vision = st.checkbox(
        "识别人物/物体/场景/动作（可选视觉模型）",
        value=False,
        disabled=not vision_configured,
        help="配置 STORYFORGE_VISION_MODEL 和 OpenAI-compatible API 后启用。",
        key="analysis_vision",
    )
    if not vision_configured:
        st.caption("视觉模型未配置：本次只输出可验证的亮度、色彩和画面变化特征。")
    scene_threshold = st.slider(
        "切镜敏感度",
        min_value=0.10,
        max_value=0.70,
        value=0.30,
        step=0.05,
        help="数值越小，切出的片段通常越多。",
        key="analysis_scene_threshold",
    )
    similarity_threshold = st.slider(
        "相似镜头聚类阈值",
        min_value=0.70,
        max_value=0.99,
        value=0.90,
        step=0.01,
        help="数值越高，只有更相似的镜头才会进入同一组。",
        key="analysis_similarity_threshold",
    )
    enable_audio_boundaries = st.checkbox(
        "使用语音情绪与声音变化辅助切片",
        value=True,
        help="需要音轨；语音文本情绪需要启用转写。瞬时变化会由持续性规则过滤。",
        key="analysis_audio_boundaries",
    )
    with st.expander("音频切点高级参数", expanded=False):
        audio_window_seconds = st.slider(
            "分析窗口（秒）", 0.5, 4.0, 2.0, 0.5, key="analysis_audio_window"
        )
        audio_hop_seconds = st.slider(
            "窗口步长（秒）", 0.25, 2.0, 1.0, 0.25, key="analysis_audio_hop"
        )
        audio_change_threshold = st.slider(
            "持续变化阈值", 0.20, 0.90, 0.45, 0.05,
            help="越低越容易因声音或文本情绪变化产生切点。",
            key="analysis_audio_threshold",
        )
        emotion_persistence_windows = st.slider(
            "最少持续窗口数", 1, 4, 2, 1,
            help="设为 2 可过滤单个窗口的瞬时噪声。",
            key="analysis_audio_persistence",
        )
    analyse = st.button("开始分析", type="primary", use_container_width=True, key="run_analysis")
    progress_box = st.empty()

with workspace:
    if upload:
        st.video(upload.getvalue())
        st.caption(f"输入文件：{upload.name} · {len(upload.getvalue()) / 1024 / 1024:.1f} MB")
    else:
        st.info("上传视频后，系统会先生成分析和标注建议，不会直接替你剪掉任何内容。")

if analyse:
    if upload is None:
        st.error("请先上传视频。")
    else:
        source = save_upload(upload)
        event_names = {
            "analysis_started": "读取视频",
            "media_probed": "检查音视频流",
            "transcription_completed": "语音识别完成",
            "transcription_fallback": "语音识别已降级",
            "shots_detected": "镜头切分完成",
            "segment_analysed": "正在提取镜头特征",
            "analysis_completed": "分析完成",
        }

        def on_event(event: str, payload: dict) -> None:
            progress_box.info(event_names.get(event, event))

        agent = VideoAnalysisAgent(
            AnalysisConfig(
                output_dir=ARTIFACTS / "analysis",
                scene_threshold=scene_threshold,
                similarity_threshold=similarity_threshold,
                transcribe=transcribe,
                enable_audio_boundaries=enable_audio_boundaries,
                audio_window_seconds=audio_window_seconds,
                audio_hop_seconds=min(audio_hop_seconds, audio_window_seconds),
                audio_change_threshold=audio_change_threshold,
                emotion_persistence_windows=emotion_persistence_windows,
            ),
            visual_tagger=OpenAICompatibleVisionTagger() if use_vision else None,
            on_event=on_event,
        )
        try:
            with st.spinner("正在逐镜头分析，长视频需要一些时间……"):
                result = agent.run(source, language=language)
            st.session_state["analysis_result_path"] = result.artifacts["analysis_json"]
            progress_box.success(f"分析完成：{len(result.segments)} 个片段")
        except Exception as exc:
            st.exception(exc)

result = load_current_result()
if result:
    st.divider()
    st.subheader("2. 查看分析结果")
    clusters = len({item.cluster_id for item in result.segments})
    speech_segments = sum(item.speech_present for item in result.segments)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("视频时长", f"{result.duration_seconds:.1f} 秒")
    c2.metric("自动片段", len(result.segments))
    c3.metric("相似镜头组", clusters)
    c4.metric("含语音片段", speech_segments)
    st.caption(
        f"转写引擎：{result.transcription_provider} · 状态：{result.status} · "
        f"分析目录：{Path(result.artifacts['analysis_json']).parent}"
    )
    for warning in result.warnings:
        st.warning(warning)

    left_filter, right_filter = st.columns(2)
    cluster_options = ["全部", *[str(value) for value in sorted({item.cluster_id for item in result.segments})]]
    with left_filter:
        cluster_filter = st.selectbox("筛选镜头组", cluster_options, key="cluster_filter")
    with right_filter:
        emotion_options = ["全部", *sorted({item.suggested_emotion for item in result.segments})]
        emotion_filter = st.selectbox("筛选AI情绪建议", emotion_options, key="emotion_filter")

    visible = [
        item for item in result.segments
        if (cluster_filter == "全部" or item.cluster_id == int(cluster_filter))
        and (emotion_filter == "全部" or item.suggested_emotion == emotion_filter)
    ]
    if visible:
        with st.expander("镜头缩略图", expanded=True):
            columns = st.columns(4)
            for index, item in enumerate(visible[:24]):
                with columns[index % 4]:
                    thumbnail = Path(item.thumbnail_path)
                    if thumbnail.exists():
                        st.image(str(thumbnail), use_container_width=True)
                    st.caption(
                        f"#{item.segment_id} · {item.start_seconds:.1f}–{item.end_seconds:.1f}s · "
                        f"组 {item.cluster_id} · {item.suggested_emotion}"
                    )

    st.subheader("3. 人工复核与选择")
    st.caption("可编辑“保留、人工情绪、人工内容标签、复核备注”。AI建议及其证据保持只读。")
    editable_rows = rows_from_result(result)
    edited = st.data_editor(
        editable_rows,
        hide_index=True,
        use_container_width=True,
        disabled=[
            "片段", "开始", "结束", "时长", "切点依据", "切点分数", "镜头组", "内容摘要", "语音文本",
            "AI情绪建议", "情绪置信度", "高光分", "证据", "视觉特征",
        ],
        key=f"review_editor_{result.analysis_id}",
    )
    reviewed_rows = edited.to_dict("records") if hasattr(edited, "to_dict") else list(edited)

    save_col, export_col = st.columns(2)
    with save_col:
        if st.button("保存人工标注", use_container_width=True, key="save_review"):
            reviewed_path = apply_review(result, reviewed_rows)
            st.success(f"已保存：{reviewed_path.name}")
    with export_col:
        if st.button("导出勾选片段为 MP4", use_container_width=True, key="export_review"):
            apply_review(result, reviewed_rows)
            selected_ids = [int(row["片段"]) for row in reviewed_rows if bool(row.get("保留"))]
            try:
                with st.spinner("正在无损衔接所选片段……"):
                    output = VideoAnalysisAgent().export_selection(result, selected_ids)
                st.session_state["selection_video"] = str(output)
                st.success(f"已导出 {len(selected_ids)} 个片段")
            except Exception as exc:
                st.exception(exc)

    selection_video = st.session_state.get("selection_video")
    if selection_video and Path(selection_video).exists():
        st.video(Path(selection_video).read_bytes())
        st.download_button(
            "下载所选片段 MP4",
            Path(selection_video).read_bytes(),
            file_name="selected_segments.mp4",
            mime="video/mp4",
        )

    st.subheader("4. 下载分析资产")
    download_columns = st.columns(4)
    downloads = [
        ("分析 JSON", result.artifacts["analysis_json"], "application/json"),
        ("片段 CSV", result.artifacts["segments_csv"], "text/csv"),
        ("语音字幕 SRT", result.artifacts["transcript_srt"], "text/plain"),
        ("音频时间窗 JSON", result.artifacts["audio_timeline"], "application/json"),
    ]
    for column, (label, path_value, mime) in zip(download_columns, downloads):
        path = Path(path_value)
        with column:
            st.download_button(label, path.read_bytes(), file_name=path.name, mime=mime, use_container_width=True)

st.divider()
st.caption("原来的主题生成与自动成片功能已保留在左侧页面“自动成片”，但不再是本项目的主要定位。")
