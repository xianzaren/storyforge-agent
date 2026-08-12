from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from storyforge import WorkflowAgent, WorkflowConfig
from storyforge.tools import ScriptTool
from storyforge.utils import run_command
from storyforge.validation import RunAudit, audit_run


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = ROOT / "test_results"


class SkipStage(RuntimeError):
    pass


@dataclass
class StageResult:
    stage_id: str
    name: str
    status: str
    duration_seconds: float
    reproduce: str
    log_file: str
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None


class TestWorkflow:
    def __init__(self, command: str, results_root: Path) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.command = command
        self.result_dir = (results_root / f"{stamp}-{command}").resolve()
        self.logs_dir = self.result_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.stages: list[StageResult] = []

    def run_stage(
        self,
        stage_id: str,
        name: str,
        action: Callable[[], dict[str, Any] | None],
        reproduce: str,
    ) -> StageResult:
        print(f"\n[{stage_id}] {name}")
        started = time.perf_counter()
        log_path = self.logs_dir / f"{stage_id}.log"
        status = "passed"
        details: dict[str, Any] = {}
        error = None
        trace = None
        with log_path.open("w", encoding="utf-8") as handle:
            try:
                with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                    details = action() or {}
            except SkipStage as exc:
                status = "skipped"
                error = str(exc)
                handle.write(f"SKIPPED: {error}\n")
            except Exception as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                trace = traceback.format_exc()
                handle.write(trace)
        duration = round(time.perf_counter() - started, 3)
        result = StageResult(
            stage_id=stage_id,
            name=name,
            status=status,
            duration_seconds=duration,
            reproduce=reproduce,
            log_file=str(log_path),
            details=details,
            error=error,
            traceback=trace,
        )
        self.stages.append(result)
        marker = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}[status]
        print(f"  {marker} ({duration:.3f}s)")
        if error:
            print(f"  {error}")
        return result

    @property
    def passed(self) -> bool:
        return not any(stage.status == "failed" for stage in self.stages)

    def write_report(self) -> tuple[Path, Path]:
        bugs = []
        for index, stage in enumerate((item for item in self.stages if item.status == "failed"), 1):
            bugs.append({
                "bug_id": f"BUG-{index:03}",
                "stage": stage.stage_id,
                "title": f"{stage.name} failed",
                "error": stage.error,
                "reproduce": stage.reproduce,
                "log_file": stage.log_file,
                "traceback": stage.traceback,
            })
        report = {
            "schema_version": 1,
            "command": self.command,
            "status": "passed" if self.passed else "failed",
            "generated_at": datetime.now().astimezone().isoformat(),
            "project_root": str(ROOT),
            "environment": {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "llm_api_configured": bool(os.getenv("STORYFORGE_API_KEY") and os.getenv("STORYFORGE_MODEL")),
            },
            "summary": {
                "total": len(self.stages),
                "passed": sum(item.status == "passed" for item in self.stages),
                "failed": sum(item.status == "failed" for item in self.stages),
                "skipped": sum(item.status == "skipped" for item in self.stages),
            },
            "stages": [asdict(stage) for stage in self.stages],
            "bugs": bugs,
        }
        json_path = self.result_dir / "report.json"
        markdown_path = self.result_dir / "report.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        markdown_path.write_text(self._markdown_report(report), encoding="utf-8")
        return json_path, markdown_path

    @staticmethod
    def _markdown_report(report: dict[str, Any]) -> str:
        summary = report["summary"]
        lines = [
            "# StoryForge automated test report",
            "",
            f"- Status: **{report['status'].upper()}**",
            f"- Command: `{report['command']}`",
            f"- Generated: {report['generated_at']}",
            f"- Results: {summary['passed']} passed, {summary['failed']} failed, {summary['skipped']} skipped",
            "",
            "## Stages",
            "",
            "| Stage | Status | Duration | Reproduce |",
            "|---|---|---:|---|",
        ]
        for stage in report["stages"]:
            lines.append(
                f"| {stage['stage_id']} {stage['name']} | {stage['status']} | "
                f"{stage['duration_seconds']:.3f}s | `{stage['reproduce']}` |"
            )
        lines.extend(["", "## Bugs", ""])
        if not report["bugs"]:
            lines.append("No bugs were detected by this run.")
        for bug in report["bugs"]:
            lines.extend([
                f"### {bug['bug_id']}: {bug['title']}",
                "",
                f"- Error: `{bug['error']}`",
                f"- Reproduce: `{bug['reproduce']}`",
                f"- Log: `{bug['log_file']}`",
                "",
            ])
        return "\n".join(lines) + "\n"


def preflight() -> dict[str, Any]:
    if sys.version_info < (3, 10):
        raise RuntimeError(f"Python 3.10+ is required, got {sys.version.split()[0]}")
    packages = {}
    for distribution in ["Pillow", "streamlit"]:
        packages[distribution] = importlib.metadata.version(distribution)
    binaries = {}
    for name in ["ffmpeg", "ffprobe"]:
        path = shutil.which(name)
        if not path:
            raise RuntimeError(f"Required executable is missing from PATH: {name}")
        binaries[name] = path
    filters = run_command(["ffmpeg", "-hide_banner", "-filters"]).stdout
    encoders = run_command(["ffmpeg", "-hide_banner", "-encoders"]).stdout
    for required_filter in ["subtitles", "flite"]:
        if required_filter not in filters:
            raise RuntimeError(f"FFmpeg filter is unavailable: {required_filter}")
    if "libx264" not in encoders:
        raise RuntimeError("FFmpeg encoder is unavailable: libx264")
    print(json.dumps({"packages": packages, "binaries": binaries}, ensure_ascii=False, indent=2))
    return {"packages": packages, "binaries": binaries}


def unit_tests(pattern: str = "test*.py") -> dict[str, Any]:
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-v"]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    print(completed.stdout)
    print(completed.stderr, file=sys.stderr)
    if completed.returncode:
        raise RuntimeError(f"unittest exited with code {completed.returncode}")
    match = __import__("re").search(r"Ran\s+(\d+)\s+tests?", completed.stderr + completed.stdout)
    return {"test_count": int(match.group(1)) if match else None, "command": command}


def stage_two_acceptance(result_dir: Path) -> dict[str, Any]:
    assets_dir = result_dir / "stage2_assets"
    runs_dir = result_dir / "stage2_runs"
    assets_dir.mkdir(parents=True, exist_ok=True)
    source_video = assets_dir / "creator_workflow_multishot.mp4"
    run_command([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=320x180:d=2",
        "-f", "lavfi", "-i", "color=c=white:s=320x180:d=2",
        "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=523:duration=6",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
        "-map", "[v]", "-map", "3:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", str(source_video),
    ])
    (assets_dir / "metadata.json").write_text(json.dumps({
        source_video.name: {"tags": [
            "fast topic introduction", "creator workflow", "workflow and tools",
            "quality control", "iteration", "human AI collaboration",
        ]}
    }, indent=2), encoding="utf-8")
    state = WorkflowAgent(WorkflowConfig(
        runs_dir=runs_dir,
        assets_dir=assets_dir,
        width=360,
        height=640,
        fit_mode="crop",
        transition_seconds=0.2,
        enable_scene_detection=True,
        scene_detection_threshold=0.15,
        max_retries=1,
    )).run("AI creator workflow", target_duration=8, language="en")
    run_dir = runs_dir / state.task_id
    audit = audit_run(
        run_dir,
        expected_width=360,
        expected_height=640,
        require_video_assets=True,
        require_distinct_shots=3,
    )
    (result_dir / "stage2_audit.json").write_text(
        json.dumps(audit.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2))
    if not audit.passed:
        raise RuntimeError("Stage-two artifact audit failed: " + "; ".join(issue.message for issue in audit.issues))
    return {
        "run_dir": str(run_dir),
        "video": str(run_dir / "final.mp4"),
        "audit": audit.to_dict(),
    }


def streamlit_acceptance() -> dict[str, Any]:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app.py"))
    app.run(timeout=30)
    if app.exception:
        raise RuntimeError(f"Streamlit initial render failed: {[item.value for item in app.exception]}")
    app.text_input[0].input("Automated web workflow")
    app.slider[0].set_value(12)
    app.selectbox[0].select(app.selectbox[0].options[0])
    app.selectbox[1].select(app.selectbox[1].options[1])
    app.selectbox[2].select(app.selectbox[2].options[1])
    app.slider[1].set_value(0.2)
    app.checkbox[0].check()
    app.slider[2].set_value(0.25)
    app.button[0].click()
    app.run(timeout=120)
    errors = [item.value for item in app.error]
    exceptions = [str(item.value) for item in app.exception]
    success = [item.value for item in app.success]
    print(json.dumps({"success": success, "errors": errors, "exceptions": exceptions}, ensure_ascii=False, indent=2))
    if errors or exceptions or not any("completed" in str(item) for item in success):
        raise RuntimeError("Streamlit generation workflow did not complete successfully")
    candidates = sorted((ROOT / "runs").glob("*-automated-web-workflow-*"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError("Cannot find the Streamlit-generated run directory")
    audit = audit_run(candidates[0], expected_width=720, expected_height=1280)
    if not audit.passed:
        raise RuntimeError("Streamlit artifact audit failed: " + "; ".join(issue.message for issue in audit.issues))
    return {"run_dir": str(candidates[0]), "audit": audit.to_dict()}


def live_llm_test() -> dict[str, Any]:
    if not os.getenv("STORYFORGE_API_KEY") or not os.getenv("STORYFORGE_MODEL"):
        raise SkipStage("STORYFORGE_API_KEY and STORYFORGE_MODEL are not configured")
    tool = ScriptTool()
    scenes = tool.run("A concise introduction to automated video editing", 12, "en")
    if tool.last_provider != "openai-compatible":
        raise RuntimeError(f"Live model call fell back to {tool.last_provider}: {tool.last_error}")
    if not scenes:
        raise RuntimeError("Live model returned no valid scenes")
    return {"provider": tool.last_provider, "scene_count": len(scenes)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command StoryForge regression and bug-report workflow")
    parser.add_argument("command", nargs="?", choices=["smoke", "build", "llm-test"], default="build")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_ROOT)
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    workflow = TestWorkflow(args.command, args.results_dir)
    workflow.run_stage("preflight", "Environment and FFmpeg capabilities", preflight, "python test.py smoke")
    if args.command == "smoke":
        workflow.run_stage("unit", "Fast core tests", lambda: unit_tests("test_core.py"), "python test.py smoke")
    elif args.command == "llm-test":
        workflow.run_stage("llm", "Live OpenAI-compatible model", live_llm_test, "python test.py llm-test")
    else:
        workflow.run_stage("unit", "All unit and integration tests", unit_tests, "python test.py build")
        workflow.run_stage(
            "stage2",
            "Stage-two multi-shot vertical video acceptance",
            lambda: stage_two_acceptance(workflow.result_dir),
            "python test.py build",
        )
        workflow.run_stage("streamlit", "Streamlit interactive generation", streamlit_acceptance, "python test.py build")
    json_report, markdown_report = workflow.write_report()
    print("\n=== StoryForge test workflow ===")
    print(f"status={'PASSED' if workflow.passed else 'FAILED'}")
    print(f"json_report={json_report}")
    print(f"markdown_report={markdown_report}")
    return 0 if workflow.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
