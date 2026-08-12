# StoryForge Agent

一个面向短视频创作的可运行 AI Agent MVP。它把“主题”转成结构化脚本和分镜，匹配本地图片或视频素材，生成配音与字幕，通过 FFmpeg 合成带硬字幕的视频，并对失败步骤自动重试、对输出执行质量检查。

## 这个项目展示什么

- Agent / Tool / Workflow 分层：编排器只负责状态推进，具体能力由工具实现。
- 结构化输出：脚本和分镜保存为 JSON，便于检查、修改和复现。
- 智能剪辑基础：自动检测视频镜头，将长视频拆成镜头候选，并优先选择尚未使用的匹配镜头。
- 素材组织：扫描本地素材库，根据文件名和元数据进行场景匹配。
- 自动视频生产：配音、SRT 字幕、字幕烧录、场景片段和最终 MP4 全流程生成。
- 质量闭环：检查素材、音频、字幕、媒体流、音视频时长和产物完整性；步骤异常时按策略重试并记录恢复过程。
- 可观测性：每个任务均保存状态快照与 JSONL 事件日志。

> 默认离线模式无需模型 API：脚本使用确定性模板，画面使用自动生成的标题卡，英文配音使用 FFmpeg flite。配置 OpenAI-compatible API 后可生成更自然的结构化脚本；安装 `edge-tts` 后可生成中文配音。

## 环境要求

- Python 3.10+
- FFmpeg / ffprobe

安装：

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 一分钟运行

离线 CLI：

```bash
python cli.py --topic "How AI helps creators" --duration 24 --language en --max-retries 1

# 竖屏、铺满裁剪、镜头检测与淡入淡出
python cli.py --topic "AI creator workflow" --duration 16 --language en \
  --assets-dir examples/stage2_assets --aspect-ratio 9:16 --fit-mode crop \
  --scene-threshold 0.25 --transition-seconds 0.35
```

启动 Web 界面：

```bash
streamlit run app.py
```

输出位于 `runs/<task_id>/`，包括：

```text
state.json          # 当前任务状态
workflow_settings.json # 本次画幅、镜头检测、转场和重试设置
events.jsonl        # 每步事件日志
script.json         # 结构化脚本
storyboard.json     # 分镜与素材选择
assets_manifest.json # 素材元数据、匹配结果、裁剪和循环信息
subtitles.srt       # 独立 SRT 字幕
tts_manifest.json   # 每个场景的 TTS 提供者和音频路径
final_raw.mp4       # 字幕烧录前的中间视频
final.mp4           # 带配音和硬字幕的最终视频
quality_report.json # 质量检查结果
```

## 使用自己的素材

把图片或视频放入某个目录，并通过 `--assets-dir` 传入。支持：

```text
.jpg .jpeg .png .mp4 .mov .mkv .avi .webm
```

```bash
python cli.py --topic "Solar energy basics" --assets-dir examples/assets
```

素材匹配会使用文件名关键词。默认情况下，只要用户上传了素材，即使文件名/标签没有命中分镜关键词也会采用上传素材，不再静默替换为标题卡；Web 页面会显示每个分镜的素材采用原因。需要旧的严格匹配行为时，可在 Web 关闭“未匹配关键词时仍使用上传素材”，或在 CLI 增加 `--strict-asset-matching`。也可在素材目录放置 `metadata.json`：

```json
{
  "solar_panel.jpg": ["solar", "energy", "panel"],
  "city_evening.png": ["city", "future", "technology"],
  "creator_workflow.mp4": {"tags": ["creator", "workflow", "AI"]}
}
```

视频素材会由 ffprobe 读取时长、分辨率和音轨信息，并使用 FFmpeg 场景变化检测拆成镜头候选。匹配时优先使用尚未用过的镜头；同一长镜头再次使用时会选择不同时间位置，短镜头只在自身边界内循环。如果上传视频总时长短于目标成片时长，界面会明确提示素材必然重复。素材原声默认静音，最终只使用 Agent 生成的旁白。图片和视频可以在同一个任务中混合使用。所有镜头选择会写入 `assets_manifest.json`。

CLI 和 Web 均支持 16:9、9:16、1:1，支持完整留边（`pad`）或铺满裁剪（`crop`），并可调节画面与音频淡入淡出时间。镜头检测可关闭，阈值也可调整。

完全没有上传素材时，Agent 会生成场景标题卡并记录一次局部恢复事件。开启严格匹配后，没有匹配素材也会生成标题卡。

## 配置模型 API（可选）

项目使用 OpenAI-compatible Chat Completions 接口，不绑定特定供应商：

```bash
export STORYFORGE_API_KEY="..."
export STORYFORGE_BASE_URL="https://api.openai.com/v1"
export STORYFORGE_MODEL="your-model-name"
python cli.py --topic "A 30-second introduction to multimodal AI"
```

模型必须返回 JSON；解析失败时会自动退回离线脚本生成器，不中断整条流水线。

## 中文配音（可选）

```bash
pip install edge-tts
python cli.py --topic "三十秒了解多模态模型" --language zh
```

如果 `edge-tts` 不可用，系统仍会生成视频、字幕与占位音轨，并在质量报告中记录降级信息。

## 测试

推荐使用一键自动化入口：

```powershell
# 环境预检 + 快速核心测试
.\.venv\Scripts\python.exe test.py smoke

# 完整回归：全部测试 + 二阶段多镜头竖屏成片 + Streamlit 交互
.\.venv\Scripts\python.exe test.py build

# 可选：调用真实 OpenAI-compatible API
.\.venv\Scripts\python.exe test.py llm-test
```

每次执行都会在 `test_results/<时间>-<命令>/` 生成：

```text
report.json          # 机器可读结果和 Bug 列表
report.md            # 人工可读测试报告与复现命令
logs/                # 每个测试阶段的完整日志
stage2_audit.json    # 媒体流、音量、镜头选择和事件链审计
stage2_runs/         # 二阶段验收视频及全部中间产物
```

`build` 会检查 Python 包、FFmpeg/ffprobe、`subtitles`/`flite` 滤镜和 `libx264` 编码器，并验证状态、事件顺序、图片/视频素材、镜头检测与去重、镜头内循环、竖屏裁剪、转场、AI 旁白非静音、字幕、分辨率及质量报告。任一阶段失败都会生成带 `BUG-xxx` 编号、错误栈、日志路径和复现命令的 Bug 记录。

也可以直接运行底层测试：

```bash
python -m unittest discover -s tests -v
python cli.py --topic "How AI helps creators" --duration 16 --language en
```

当前 19 项测试覆盖模型 JSON 解析与降级、分镜规范化、素材恢复、未匹配上传素材回退、严格匹配模式、视频优先匹配、镜头检测、镜头去重、画幅与转场滤镜、SRT 时间轴、失败步骤重试、成功警告状态审计、产物审计、Bug 报告生成，以及图片/视频混合和 `MCN.mp4` 式无关键词素材的真实 FFmpeg 端到端成片。

## 架构

```text
User / Streamlit / CLI
          |
     WorkflowAgent
          |
  +-------+--------+---------+---------+
  | Script Tool    | Asset Tool        |
  | Storyboard Tool| TTS + Subtitle    |
  | Render Tool    | Quality Tool      |
  +-------+--------+---------+---------+
          |
   state.json + events.jsonl + final.mp4
```

## 适合作为下一轮优化的方向

- 接入真实素材搜索 API 与版权信息记录。
- 增加 CLIP/SigLIP 视觉模型，对镜头与分镜做语义一致性评分。
- 用状态图框架替换当前轻量编排器，支持人工审核节点。
- 增加 Whisper 语音转写、镜头级局部重写、局部重新配音和并行渲染。
- 增加对象存储、队列和多任务并发。
