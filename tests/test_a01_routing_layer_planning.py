from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from tmux_core.runtime.vendor_catalog import get_default_model_for_vendor, get_model_choices, get_normalized_effort_choices
from A01_Routing_LayerPlanning import (
    DEFAULT_MODEL_BY_VENDOR,
    TerminalProgressMonitor,
    build_parser,
    collect_cli_request,
    determine_exit_code,
    display_status_label,
    format_batch_summary,
    main,
    normalize_effort_choice,
    normalize_model_choice,
    normalize_run_init_choice,
    normalize_vendor_choice,
    prompt_effort,
    prompt_confirmation,
    prompt_project_dir,
    prompt_model,
    prompt_proxy_port,
    prompt_vendor,
    render_live_progress_frame,
    render_live_progress_line,
    render_routing_failure_summary,
    render_runtime_start_summary,
    render_requirements_stage_placeholder,
    run_routing_stage,
    summarize_live_result_counts,
    split_target_dirs_text,
)
from T09_terminal_ops import PromptBackRequested
from T03_agent_init_workflow import BatchInitResult, DirectoryInitResult, RoutingCleanupResult, RunManifest, RunStore, TargetSelection, WorkerManifestEntry


def _write_valid_routing_layer(project_dir: Path) -> None:
    (project_dir / "docs").mkdir(exist_ok=True)
    (project_dir / "AGENTS.md").write_text("ok", encoding="utf-8")
    (project_dir / "docs" / "repo_map.json").write_text(
        json.dumps({"modules": [{"id": "M01", "name": "root"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (project_dir / "docs" / "task_routes.json").write_text(
        json.dumps({"routes": [{"id": "R01", "first_read_modules": ["M01"], "pitfall_ids": ["P01"]}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (project_dir / "docs" / "pitfalls.json").write_text(
        json.dumps({"pitfalls": [{"id": "P01", "title": "risk"}]}, ensure_ascii=False),
        encoding="utf-8",
    )


class RoutingLayerCliTests(unittest.TestCase):
    def test_terminal_progress_monitor_redraws_in_place_for_tty(self):
        class _TTYBuffer(io.StringIO):
            def isatty(self) -> bool:
                return True

        stream = _TTYBuffer()
        monitor = TerminalProgressMonitor(
            run_id="run_demo",
            runtime_root="/tmp/runtime",
            selection=TargetSelection(
                project_dir="/tmp/project",
                selected_dirs=("/tmp/project",),
                skipped_dirs=(),
                forced_dirs=(),
                project_missing_files=(),
            ),
            stream=stream,
        )
        monitor._display_line("line1")
        monitor._display_line("line2-updated")
        output = stream.getvalue()
        self.assertIn("\r", output)
        self.assertNotIn("\x1b[2J", output)
        self.assertNotIn("\x1b[2F", output)
        self.assertEqual(monitor.interval_sec, 0.2)

    def test_normalize_vendor_choice_supports_aliases(self):
        self.assertEqual("claude", normalize_vendor_choice("claude code"))
        self.assertEqual("opencode", normalize_vendor_choice("4"))
        self.assertEqual("codex", normalize_vendor_choice("codex"))
        with self.assertRaises(ValueError):
            normalize_vendor_choice("qwen")
        with self.assertRaises(ValueError):
            normalize_vendor_choice("kimi")

    def test_normalize_model_and_effort_support_numeric_aliases(self):
        self.assertEqual("gpt-5.4", normalize_model_choice("codex", "1"))
        self.assertEqual("sonnet", normalize_model_choice("claude", "1"))
        default_opencode_model = get_default_model_for_vendor("opencode")
        opencode_reasoning_model = next(
            item.model_id
            for item in get_model_choices("opencode")
            if len(item.reasoning.normalized_reasoning_levels) > 1
        )
        self.assertEqual(default_opencode_model, normalize_model_choice("opencode", "1"))
        self.assertEqual("medium", normalize_effort_choice("codex", "gpt-5.4", "2"))
        self.assertEqual("medium", normalize_effort_choice("opencode", opencode_reasoning_model, "2"))
        self.assertEqual("max", normalize_effort_choice("codex", "gpt-5.4", "5"))

    def test_prompt_model_uses_scanned_opencode_models(self):
        opencode_models = get_model_choices("opencode")
        self.assertGreaterEqual(len(opencode_models), 2)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            with patch("builtins.input", side_effect=["2"]):
                model = prompt_model("opencode", DEFAULT_MODEL_BY_VENDOR["opencode"])
        self.assertEqual(model, opencode_models[1].model_id)
        self.assertNotIn("自定义输入 provider/model", stdout.getvalue())

    def test_collect_cli_request_normalizes_opencode_model_and_effort_in_parameter_mode(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _write_valid_routing_layer(project_dir)
            args = parser.parse_args(
                [
                    "--project-dir",
                    tmpdir,
                    "--run-init",
                    "yes",
                    "--vendor",
                    "opencode",
                    "--model",
                    "1",
                    "--effort",
                    "1",
                ]
            )
            request = collect_cli_request(args)
        self.assertEqual(request.vendor, "opencode")
        self.assertEqual(request.model, get_default_model_for_vendor("opencode"))
        self.assertEqual(request.reasoning_effort, get_normalized_effort_choices("opencode", request.model)[0])

    def test_build_parser_accepts_requirement_name_for_workflow_compat(self):
        parser = build_parser()
        args = parser.parse_args(["--project-dir", "/tmp/project", "--requirement-name", "需求A"])
        self.assertEqual(args.project_dir, "/tmp/project")
        self.assertEqual(args.requirement_name, "需求A")

    def test_normalize_run_init_choice_parses_yes_no_variants(self):
        self.assertTrue(normalize_run_init_choice("yes"))
        self.assertTrue(normalize_run_init_choice("1"))
        self.assertFalse(normalize_run_init_choice("no"))
        self.assertFalse(normalize_run_init_choice("0"))

    def test_split_target_dirs_text_keeps_non_empty_items(self):
        self.assertEqual(
            split_target_dirs_text("api, core/calculation , , ./docs"),
            ["api", "core/calculation", "./docs"],
        )

    def test_prompt_project_dir_requires_absolute_path_and_reuses_invalid_input(self):
        metadata_calls: list[dict[str, object]] = []

        @contextmanager
        def capture_prompt_metadata(**metadata):
            metadata_calls.append(metadata)
            yield

        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_mock = Mock(side_effect=["relative/path", tmpdir])
            with patch("A01_Routing_LayerPlanning.prompt_with_default", prompt_mock), patch(
                "A01_Routing_LayerPlanning.prompt_metadata",
                side_effect=capture_prompt_metadata,
            ), patch("A01_Routing_LayerPlanning.message") as message_mock:
                result = prompt_project_dir()

        self.assertEqual(result, str(Path(tmpdir).resolve()))
        self.assertEqual(prompt_mock.call_args_list[0].args, ("输入项目工作目录", ""))
        self.assertEqual(prompt_mock.call_args_list[1].args, ("输入项目工作目录", "relative/path"))
        self.assertEqual(metadata_calls[1]["error_message"], "目录无效: 请输入绝对路径")
        message_mock.assert_any_call("目录无效: 请输入绝对路径")

    def test_display_status_label_prioritizes_failed_and_forced(self):
        failed = DirectoryInitResult(work_dir="/tmp/a", forced=True, status="failed", rounds_used=1)
        forced = DirectoryInitResult(work_dir="/tmp/b", forced=True, status="passed", rounds_used=0)
        skipped = DirectoryInitResult(work_dir="/tmp/c", forced=False, status="skipped", rounds_used=0)
        self.assertEqual(display_status_label(failed), "failed")
        self.assertEqual(display_status_label(forced), "forced")
        self.assertEqual(display_status_label(skipped), "skipped")

    def test_format_batch_summary_contains_directory_lines(self):
        batch = BatchInitResult(
            run_id="run_demo",
            runtime_dir="/tmp/runtime/run_demo",
            selection=TargetSelection(
                project_dir="/tmp/project",
                selected_dirs=("/tmp/project",),
                skipped_dirs=(),
                forced_dirs=("/tmp/project",),
                project_missing_files=("AGENTS.md",),
            ),
            config={
                "vendor": "codex",
                "model": DEFAULT_MODEL_BY_VENDOR["codex"],
                "reasoning_effort": "high",
                "proxy_url": "",
                "reasoning_note": "reasoning_effort=high",
            },
            results=[
                DirectoryInitResult(
                    work_dir="/tmp/project",
                    forced=True,
                    status="passed",
                    rounds_used=1,
                    last_audit_summary="Ready",
                ),
                DirectoryInitResult(
                    work_dir="/tmp/project/core",
                    forced=False,
                    status="failed",
                    rounds_used=2,
                    failure_reason="max_refine_rounds_reached",
                    last_audit_summary="Need fixes",
                ),
            ],
        )
        summary = format_batch_summary(batch)
        self.assertIn("run_id: run_demo", summary)
        self.assertIn("- /tmp/project: forced | audit=Ready", summary)
        self.assertIn("failure=max_refine_rounds_reached", summary)

    def test_determine_exit_code_returns_non_zero_on_failure(self):
        batch = BatchInitResult(
            run_id="run_demo",
            runtime_dir="/tmp/runtime/run_demo",
            selection=TargetSelection(
                project_dir="/tmp/project",
                selected_dirs=("/tmp/project",),
                skipped_dirs=(),
                forced_dirs=(),
                project_missing_files=(),
            ),
            config={
                "vendor": "codex",
                "model": "gpt-5.4",
                "reasoning_effort": "high",
                "proxy_url": "",
                "reasoning_note": "reasoning_effort=high",
            },
            results=[
                DirectoryInitResult(work_dir="/tmp/project", forced=False, status="failed", rounds_used=1),
            ],
        )
        self.assertEqual(determine_exit_code(batch), 1)

    def test_render_requirements_stage_placeholder_mentions_cleanup_and_next_stage(self):
        text = render_requirements_stage_placeholder(
            ["aginit-a", "aginit-b"],
            RoutingCleanupResult(
                removed_intermediate_files=("a", "b"),
                removed_runtime_dirs=("runtime",),
            ),
        )
        self.assertIn("路由层配置完成", text)
        self.assertIn("已清理路由层 tmux 会话: 2", text)
        self.assertIn("已清理阶段中间文件: 2", text)
        self.assertIn("已清理阶段运行目录: 1", text)
        self.assertIn("进入需求录入阶段（占位）", text)
        self.assertIn("A02_RequirementIntake.py", text)

    def test_render_routing_failure_summary_lists_failed_targets_without_next_stage(self):
        batch = BatchInitResult(
            run_id="run_demo",
            runtime_dir="/tmp/runtime/run_demo",
            selection=TargetSelection(
                project_dir="/tmp/project",
                selected_dirs=("/tmp/project", "/tmp/project/core"),
                skipped_dirs=(),
                forced_dirs=(),
                project_missing_files=(),
            ),
            config={
                "vendor": "codex",
                "model": "gpt-5.4",
                "reasoning_effort": "high",
                "proxy_url": "",
                "reasoning_note": "reasoning_effort=high",
            },
            results=[
                DirectoryInitResult(work_dir="/tmp/project", forced=False, status="passed", rounds_used=1),
                DirectoryInitResult(
                    work_dir="/tmp/project/core",
                    forced=False,
                    status="failed",
                    rounds_used=1,
                    failure_reason="create_command_failed: prompt timeout",
                ),
            ],
        )

        text = render_routing_failure_summary(batch, ["aginit-core"])

        self.assertIn("路由层初始化未完全通过", text)
        self.assertIn("/tmp/project/core", text)
        self.assertIn("create_command_failed: prompt timeout", text)
        self.assertIn("已清理路由层 tmux 会话: 1", text)
        self.assertNotIn("进入需求录入阶段（占位）", text)
        self.assertNotIn("A02_RequirementIntake.py", text)

    def test_render_runtime_start_summary_includes_tmux_sessions_and_attach_commands(self):
        run_store = type(
            "RunStoreStub",
            (),
            {"manifest": type("ManifestStub", (), {"run_id": "run_demo", "runtime_dir": "/tmp/runtime/run_demo"})()},
        )()
        live_workers = [
            type("HandleStub", (), {"session_name": "aginit-demo-1", "work_dir": "/tmp/project", "forced": True})(),
            type("HandleStub", (), {"session_name": "aginit-demo-2", "work_dir": "/tmp/project/core", "forced": False})(),
        ]
        text = render_runtime_start_summary(
            run_store=run_store,
            live_workers=live_workers,
            immediate_results=[],
        )
        self.assertIn("路由层初始化已启动", text)
        self.assertIn("run_id: run_demo", text)
        self.assertIn("tmux sessions:", text)
        self.assertIn("aginit-demo-1 | /tmp/project | forced", text)
        self.assertIn("tmux attach -t aginit-demo-2", text)

    def test_summarize_live_result_counts_distinguishes_running_and_finished(self):
        run_store = type(
            "RunStoreStub",
            (),
            {
                "manifest": RunManifest(
                    manifest_version=1,
                    run_id="run_demo",
                    runtime_dir="/tmp/runtime/run_demo",
                    project_dir="/tmp/project",
                    selection={},
                    config={},
                    status="running",
                    created_at="2026-04-13T12:00:00",
                    updated_at="2026-04-13T12:00:00",
                    workers=[
                        WorkerManifestEntry(work_dir="/tmp/project", result_status="pending", workflow_stage="create_running"),
                        WorkerManifestEntry(work_dir="/tmp/project/a", result_status="passed", workflow_stage="completed"),
                        WorkerManifestEntry(work_dir="/tmp/project/b", result_status="failed", workflow_stage="completed"),
                    ],
                )
            },
        )()
        counts = summarize_live_result_counts(run_store)
        self.assertEqual(counts["running"], 1)
        self.assertEqual(counts["passed"], 1)
        self.assertEqual(counts["failed"], 1)
        self.assertEqual(counts["pending"], 0)

    def test_render_live_progress_frame_contains_stage_phase_health_and_counts(self):
        run_store = type(
            "RunStoreStub",
            (),
            {
                "manifest": RunManifest(
                    manifest_version=1,
                    run_id="run_demo",
                    runtime_dir="/tmp/runtime/run_demo",
                    project_dir="/tmp/project",
                    selection={},
                    config={},
                    status="running",
                    created_at="2026-04-13T12:00:00",
                    updated_at="2026-04-13T12:00:00",
                    workers=[
                        WorkerManifestEntry(
                            work_dir="/tmp/project",
                            session_name="aginit-demo-1",
                            workflow_stage="create_running",
                            result_status="pending",
                            agent_state="BUSY",
                            health_status="alive",
                            note="turn:create_routing_layer",
                        )
                    ],
                )
            },
        )()
        selection = TargetSelection(
            project_dir="/tmp/project",
            selected_dirs=("/tmp/project",),
            skipped_dirs=(),
            forced_dirs=(),
            project_missing_files=(),
        )
        text = render_live_progress_frame(run_store=run_store, selection=selection, tick=1)
        self.assertIn("路由层初始化运行中... ⠙", text)
        self.assertIn("counts: pending=0 running=1 passed=0 failed=0 skipped=0", text)
        self.assertIn("stage=create_running | state=BUSY | health=alive", text)
        self.assertIn("note=turn:create_routing_layer", text)

    def test_render_live_progress_line_contains_spinner_counts_and_focus(self):
        run_store = type(
            "RunStoreStub",
            (),
            {
                "manifest": RunManifest(
                    manifest_version=1,
                    run_id="run_demo",
                    runtime_dir="/tmp/runtime/run_demo",
                    project_dir="/tmp/project",
                    selection={},
                    config={},
                    status="running",
                    created_at="2026-04-13T12:00:00",
                    updated_at="2026-04-13T12:00:00",
                    workers=[
                        WorkerManifestEntry(
                            work_dir="/tmp/project",
                            session_name="aginit-demo-1",
                            workflow_stage="create_running",
                            result_status="pending",
                            agent_state="BUSY",
                            health_status="alive",
                            note="create_routing_layer",
                        )
                    ],
                )
            },
        )()
        selection = TargetSelection(
            project_dir="/tmp/project",
            selected_dirs=("/tmp/project",),
            skipped_dirs=(),
            forced_dirs=(),
            project_missing_files=(),
        )
        text = render_live_progress_line(run_store=run_store, selection=selection, tick=1)
        self.assertIn("⠙ 路由层初始化中", text)
        self.assertIn("pending=0 running=1 passed=0 failed=0 skipped=0", text)
        self.assertIn("project:create_running/BUSY", text)
        self.assertIn("create_routing_layer", text)

    def test_render_live_progress_displays_prelaunch_dead_state_as_starting(self):
        run_store = type(
            "RunStoreStub",
            (),
            {
                "manifest": RunManifest(
                    manifest_version=1,
                    run_id="run_demo",
                    runtime_dir="/tmp/runtime/run_demo",
                    project_dir="/tmp/project",
                    selection={},
                    config={},
                    status="running",
                    created_at="2026-04-13T12:00:00",
                    updated_at="2026-04-13T12:00:00",
                    workers=[
                        WorkerManifestEntry(
                            work_dir="/tmp/project",
                            session_name="aginit-demo-1",
                            workflow_stage="create_running",
                            result_status="running",
                            agent_state="DEAD",
                            agent_started=False,
                            pane_id="",
                            health_status="missing_session",
                            note="create_routing_layer",
                        )
                    ],
                )
            },
        )()
        selection = TargetSelection(
            project_dir="/tmp/project",
            selected_dirs=("/tmp/project",),
            skipped_dirs=(),
            forced_dirs=(),
            project_missing_files=(),
        )

        frame = render_live_progress_frame(run_store=run_store, selection=selection, tick=1)
        line = render_live_progress_line(run_store=run_store, selection=selection, tick=1)

        self.assertIn("state=STARTING", frame)
        self.assertIn("project:create_running/STARTING", line)

    def test_render_live_progress_line_refreshes_worker_state_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "worker.state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "agent_state": "BUSY",
                        "agent_started": True,
                        "agent_alive": True,
                        "health_status": "alive",
                        "note": "turn:create_routing_layer",
                    }
                ),
                encoding="utf-8",
            )
            entry = WorkerManifestEntry(
                work_dir="/tmp/project",
                session_name="aginit-demo-1",
                workflow_stage="create_running",
                result_status="running",
                agent_state="STARTING",
                health_status="unknown",
                note="create_routing_layer",
                state_path=str(state_path),
            )

            class _RunStoreStub:
                def __init__(self) -> None:
                    self.manifest = RunManifest(
                        manifest_version=1,
                        run_id="run_demo",
                        runtime_dir="/tmp/runtime/run_demo",
                        project_dir="/tmp/project",
                        selection={},
                        config={},
                        status="running",
                        created_at="2026-04-13T12:00:00",
                        updated_at="2026-04-13T12:00:00",
                        workers=[entry],
                    )

                def update_worker_state_from_file(self, work_dir, state_path, *, preserve_workflow_fields=False):  # noqa: ANN001
                    payload = json.loads(Path(state_path).read_text(encoding="utf-8"))
                    self.manifest.workers[0].agent_state = str(payload["agent_state"])
                    self.manifest.workers[0].health_status = str(payload["health_status"])
                    self.manifest.workers[0].note = str(payload["note"])
                    return self.manifest.workers[0]

            selection = TargetSelection(
                project_dir="/tmp/project",
                selected_dirs=("/tmp/project",),
                skipped_dirs=(),
                forced_dirs=(),
                project_missing_files=(),
            )
            text = render_live_progress_line(run_store=_RunStoreStub(), selection=selection, tick=0)

        self.assertIn("project:create_running/BUSY", text)
        self.assertIn("turn:create_routing_layer", text)

    def test_collect_cli_request_uses_project_arg_without_prompting_for_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = build_parser()
            args = parser.parse_args(
                [
                    "--project-dir",
                    tmpdir,
                    "--vendor",
                    "codex",
                    "--model",
                    "gpt-5.4",
                    "--effort",
                    "high",
                    "--run-init",
                    "yes",
                    "--proxy-port",
                    "7890",
                    "--yes",
                ]
            )
            with patch("builtins.input", side_effect=AssertionError("input should not be called")):
                request = collect_cli_request(args)
            self.assertEqual(request.project_dir, str(Path(tmpdir).resolve()))
            self.assertEqual(request.vendor, "codex")
            self.assertEqual(request.proxy_port, "7890")

    def test_collect_cli_request_with_project_dir_only_still_prompts_run_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _write_valid_routing_layer(project_dir)
            parser = build_parser()
            args = parser.parse_args(["--project-dir", tmpdir])
            with patch("builtins.input", side_effect=["no"]):
                request = collect_cli_request(args)
        self.assertFalse(request.run_init)
        self.assertEqual(request.target_dirs, ())
        self.assertEqual(request.vendor, "codex")
        self.assertEqual(request.model, "gpt-5.4")
        self.assertEqual(request.reasoning_effort, "high")
        self.assertEqual(request.proxy_port, "")

    def test_collect_cli_request_can_back_from_run_init_to_project_dir(self):
        parser = build_parser()
        args = parser.parse_args([])
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _write_valid_routing_layer(project_dir)
            with patch("A01_Routing_LayerPlanning.prompt_project_dir", side_effect=[tmpdir, tmpdir]) as project_prompt, patch(
                "A01_Routing_LayerPlanning.prompt_run_init",
                side_effect=[PromptBackRequested(), False],
            ):
                request = collect_cli_request(args)
        self.assertFalse(request.run_init)
        self.assertEqual(project_prompt.call_count, 2)

    def test_collect_cli_request_can_back_from_run_init_to_provided_project_dir_when_allowed(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _write_valid_routing_layer(project_dir)
            args = parser.parse_args(["--project-dir", tmpdir, "--allow-project-dir-back"])
            with patch("A01_Routing_LayerPlanning.prompt_project_dir", return_value=tmpdir) as project_prompt, patch(
                "A01_Routing_LayerPlanning.prompt_run_init",
                side_effect=[PromptBackRequested(), False],
            ):
                request = collect_cli_request(args)
        self.assertFalse(request.run_init)
        project_prompt.assert_called_once_with(tmpdir)

    def test_collect_cli_request_can_back_from_model_to_vendor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _write_valid_routing_layer(project_dir)
            parser = build_parser()
            args = parser.parse_args(["--project-dir", tmpdir])
            with patch("A01_Routing_LayerPlanning.prompt_run_init", return_value=True), patch(
                "A01_Routing_LayerPlanning.prompt_target_dirs",
                return_value=(),
            ), patch(
                "A01_Routing_LayerPlanning.prompt_vendor",
                side_effect=["gemini", "codex"],
            ) as vendor_prompt, patch(
                "A01_Routing_LayerPlanning.prompt_model",
                side_effect=[PromptBackRequested(), "gpt-5.4"],
            ), patch(
                "A01_Routing_LayerPlanning.prompt_effort",
                return_value="high",
            ), patch(
                "A01_Routing_LayerPlanning.prompt_proxy_port",
                return_value="",
            ):
                request = collect_cli_request(args)
        self.assertEqual(request.vendor, "codex")
        self.assertEqual(request.model, "gpt-5.4")
        self.assertEqual(vendor_prompt.call_count, 2)

    def test_collect_cli_request_with_project_dir_only_prompts_target_dirs_and_effort(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _write_valid_routing_layer(project_dir)
            parser = build_parser()
            args = parser.parse_args(["--project-dir", tmpdir])
            with patch(
                "builtins.input",
                side_effect=[
                    "yes",
                    "",
                    "1",
                    "1",
                    "3",
                    "1",
                ],
            ):
                request = collect_cli_request(args)
        self.assertTrue(request.run_init)
        self.assertEqual(request.target_dirs, ())
        self.assertEqual(request.vendor, "codex")
        self.assertEqual(request.model, "gpt-5.4")
        self.assertEqual(request.reasoning_effort, "high")
        self.assertEqual(request.proxy_port, "")

    def test_collect_cli_request_with_yes_only_still_uses_interactive_prompts(self):
        parser = build_parser()
        args = parser.parse_args(["--yes"])
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _write_valid_routing_layer(project_dir)
            with patch(
                "builtins.input",
                side_effect=[
                    tmpdir,   # 项目工作目录
                    "no",     # run_init
                ],
            ):
                request = collect_cli_request(args)
        self.assertFalse(request.run_init)
        self.assertEqual(request.vendor, "codex")
        self.assertEqual(request.model, "gpt-5.4")
        self.assertEqual(request.reasoning_effort, "high")
        self.assertEqual(request.proxy_port, "")
        self.assertTrue(request.auto_confirm)

    def test_collect_cli_request_forces_init_when_project_routing_is_missing(self):
        parser = build_parser()
        args = parser.parse_args(["--yes"])
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "app.py").write_text("print('ok')\n", encoding="utf-8")
            with patch(
                "A01_Routing_LayerPlanning.prompt_project_dir",
                return_value=tmpdir,
            ), patch(
                "A01_Routing_LayerPlanning.prompt_run_init",
                side_effect=AssertionError("缺少路由层文件时不应先询问 run_init"),
            ), patch(
                "A01_Routing_LayerPlanning.prompt_target_dirs",
                return_value=(),
            ), patch(
                "A01_Routing_LayerPlanning.prompt_vendor",
                return_value="codex",
            ), patch(
                "A01_Routing_LayerPlanning.prompt_model",
                return_value="gpt-5.4",
            ), patch(
                "A01_Routing_LayerPlanning.prompt_effort",
                return_value="high",
            ), patch(
                "A01_Routing_LayerPlanning.prompt_proxy_port",
                return_value="",
            ), patch("sys.stdout", new=stdout):
                with patch("builtins.input", side_effect=AssertionError("不应触发底层 input")):
                    request = collect_cli_request(args)
        self.assertTrue(request.run_init)
        self.assertEqual(request.target_dirs, ())
        self.assertIn("当前项目路由层文件缺失, 强制执行路由初始化", stdout.getvalue())

    def test_collect_cli_request_skips_empty_project_missing_routing_without_prompts(self):
        parser = build_parser()
        args = parser.parse_args(["--yes"])
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "A01_Routing_LayerPlanning.prompt_project_dir",
            return_value=tmpdir,
        ), patch(
            "A01_Routing_LayerPlanning.prompt_run_init",
            side_effect=AssertionError("空项目缺少路由层时不应询问 run_init"),
        ), patch(
            "A01_Routing_LayerPlanning.prompt_target_dirs",
            side_effect=AssertionError("空项目缺少路由层时不应询问 target dirs"),
        ), patch(
            "A01_Routing_LayerPlanning.prompt_vendor",
            side_effect=AssertionError("空项目缺少路由层时不应询问 vendor"),
        ), patch(
            "A01_Routing_LayerPlanning.prompt_model",
            side_effect=AssertionError("空项目缺少路由层时不应询问 model"),
        ), patch(
            "A01_Routing_LayerPlanning.prompt_effort",
            side_effect=AssertionError("空项目缺少路由层时不应询问 effort"),
        ), patch(
            "A01_Routing_LayerPlanning.prompt_proxy_port",
            side_effect=AssertionError("空项目缺少路由层时不应询问 proxy"),
        ):
            with patch("builtins.input", side_effect=AssertionError("不应触发底层 input")):
                request = collect_cli_request(args)
        self.assertFalse(request.run_init)
        self.assertEqual(request.target_dirs, ())
        self.assertEqual(request.vendor, "codex")
        self.assertEqual(request.model, "gpt-5.4")
        self.assertEqual(request.reasoning_effort, "high")
        self.assertEqual(request.proxy_port, "")

    def test_run_routing_stage_skips_empty_project_with_specific_message(self):
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "A01_Routing_LayerPlanning.run_batch_initialization",
            side_effect=AssertionError("空项目不应启动路由初始化"),
        ), patch("sys.stdout", new=stdout):
            result = run_routing_stage(["--project-dir", tmpdir, "--yes", "--legacy-cli"])
        self.assertTrue(result.skipped)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("当前项目未检测到业务文件，跳过路由层初始化。", stdout.getvalue())

    def test_prompt_functions_print_vendor_model_effort_and_proxy_lists(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            with patch("builtins.input", side_effect=["1", "1", "5", "2"]):
                vendor = prompt_vendor("codex")
                model = prompt_model(vendor, DEFAULT_MODEL_BY_VENDOR[vendor])
                effort = prompt_effort(vendor, model, "high")
                proxy_port = prompt_proxy_port("")
        output = stdout.getvalue()
        self.assertEqual(vendor, "codex")
        self.assertEqual(model, "gpt-5.4")
        self.assertEqual(effort, "max")
        self.assertEqual(proxy_port, "10900")
        self.assertIn("选择厂商", output)
        self.assertIn("选择 codex 模型", output)
        self.assertIn("选择 gpt-5.4 推理强度", output)
        self.assertIn("选择代理端口", output)
        self.assertIn("gpt-5.4-mini", output)
        self.assertIn("xhigh", output)

    def test_prompt_functions_include_role_label_when_provided(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            with patch("builtins.input", side_effect=["1", "1", "5", "2"]):
                vendor = prompt_vendor("codex", role_label="审核器-R1")
                model = prompt_model(vendor, DEFAULT_MODEL_BY_VENDOR[vendor], role_label="审核器-R1")
                effort = prompt_effort(vendor, model, "high", role_label="审核器-R1")
                proxy_port = prompt_proxy_port("", role_label="审核器-R1")
        output = stdout.getvalue()
        self.assertEqual(vendor, "codex")
        self.assertEqual(model, "gpt-5.4")
        self.assertEqual(effort, "max")
        self.assertEqual(proxy_port, "10900")
        self.assertIn("为 审核器-R1 选择厂商", output)
        self.assertIn("为 审核器-R1 选择 codex 模型", output)
        self.assertIn("为 审核器-R1 选择 gpt-5.4 推理强度", output)
        self.assertIn("为 审核器-R1 选择代理端口", output)

    def test_collect_cli_request_scopes_routing_init_selection_prompts_to_router(self):
        parser = build_parser()
        args = parser.parse_args([])
        observed: dict[str, object] = {}

        def fake_prompt_vendor(default="codex", *, role_label=""):  # noqa: ANN001
            observed["vendor_role_label"] = role_label
            return "codex"

        def fake_prompt_model(vendor, default=None, *, role_label=""):  # noqa: ANN001
            observed["model_role_label"] = role_label
            return "gpt-5.4"

        def fake_prompt_effort(vendor, model, default="high", *, role_label=""):  # noqa: ANN001
            observed["effort_role_label"] = role_label
            return "high"

        def fake_prompt_proxy_port(default="", *, role_label=""):  # noqa: ANN001
            observed["proxy_role_label"] = role_label
            return ""

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "app.py").write_text("print('ok')\n", encoding="utf-8")
            with patch(
                "A01_Routing_LayerPlanning.prompt_project_dir",
                return_value=tmpdir,
            ), patch(
                "A01_Routing_LayerPlanning.prompt_run_init",
                return_value=True,
            ), patch(
                "A01_Routing_LayerPlanning.prompt_target_dirs",
                return_value=(),
            ), patch(
                "A01_Routing_LayerPlanning._predict_routing_role_label",
                return_value="路由器-天捷星",
            ), patch(
                "A01_Routing_LayerPlanning.prompt_vendor",
                side_effect=fake_prompt_vendor,
            ), patch(
                "A01_Routing_LayerPlanning.prompt_model",
                side_effect=fake_prompt_model,
            ), patch(
                "A01_Routing_LayerPlanning.prompt_effort",
                side_effect=fake_prompt_effort,
            ), patch(
                "A01_Routing_LayerPlanning.prompt_proxy_port",
                side_effect=fake_prompt_proxy_port,
            ):
                request = collect_cli_request(args)

        self.assertTrue(request.run_init)
        self.assertEqual(observed["vendor_role_label"], "路由器-天捷星")
        self.assertEqual(observed["model_role_label"], "路由器-天捷星")
        self.assertEqual(observed["effort_role_label"], "路由器-天捷星")
        self.assertEqual(observed["proxy_role_label"], "路由器-天捷星")

    def test_prompt_proxy_port_allows_custom_input(self):
        with patch("builtins.input", side_effect=["4", "http://127.0.0.1:10900"]):
            proxy_port = prompt_proxy_port("")
        self.assertEqual(proxy_port, "http://127.0.0.1:10900")

    def test_prompt_confirmation_force_yes_rejects_no(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout), patch("builtins.input", side_effect=["no", "yes"]):
            result = prompt_confirmation("summary", force_yes=True)
        self.assertTrue(result)
        self.assertIn("必须执行初始化", stdout.getvalue())

    def test_main_starts_and_stops_terminal_progress_monitor(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            request_args = parser.parse_args(
                [
                    "--project-dir",
                    tmpdir,
                    "--vendor",
                    "codex",
                    "--model",
                    "gpt-5.4",
                    "--effort",
                    "high",
                    "--run-init",
                    "yes",
                    "--yes",
                ]
            )
            batch_result = BatchInitResult(
                run_id="run_demo",
                runtime_dir=str(project_dir / ".routing_init_runtime" / "run_demo"),
                selection=TargetSelection(
                    project_dir=str(project_dir),
                    selected_dirs=(str(project_dir),),
                    skipped_dirs=(),
                    forced_dirs=(),
                    project_missing_files=(),
                ),
                config={
                    "vendor": "codex",
                    "model": "gpt-5.4",
                    "reasoning_effort": "high",
                    "proxy_url": "",
                    "reasoning_note": "reasoning_effort=high",
                },
                results=[DirectoryInitResult(work_dir=str(project_dir), forced=False, status="passed", rounds_used=0)],
            )
            start_stop_events: list[str] = []

            class FakeMonitor:
                def __init__(self, *, run_id, runtime_root, selection, stream=None, interval_sec=1.0):  # noqa: ANN001
                    start_stop_events.append(f"init:{run_id}")

                def start(self):
                    start_stop_events.append("start")

                def stop(self):
                    start_stop_events.append("stop")

            def fake_run_batch_initialization(*, selection, config, max_refine_rounds, on_workers_prepared, **kwargs):  # noqa: ANN001
                run_store = type(
                    "RunStoreStub",
                    (),
                    {
                        "manifest": type(
                            "ManifestStub",
                            (),
                            {"run_id": "run_demo", "runtime_dir": str(project_dir / ".routing_init_runtime" / "run_demo")},
                        )(),
                    },
                )()
                live_workers = [type("HandleStub", (), {"session_name": "aginit-demo-1", "work_dir": str(project_dir), "forced": False})()]
                on_workers_prepared(run_store, live_workers, [])
                return batch_result

            with patch("A01_Routing_LayerPlanning.collect_cli_request", return_value=collect_cli_request(request_args)), patch(
                "A01_Routing_LayerPlanning.TerminalProgressMonitor", FakeMonitor
            ), patch(
                "A01_Routing_LayerPlanning.run_batch_initialization", side_effect=fake_run_batch_initialization
            ), patch(
                "A01_Routing_LayerPlanning.RunStore.load",
                return_value=type("LoadedRunStore", (), {"manifest": type("ManifestStub", (), {"run_id": "run_demo", "runtime_dir": str(project_dir / '.routing_init_runtime' / 'run_demo')})()})(),
            ), patch(
                "A01_Routing_LayerPlanning.kill_run_tmux_sessions",
                return_value=[],
            ), patch(
                "A01_Routing_LayerPlanning.cleanup_routing_stage_artifacts",
                return_value=RoutingCleanupResult(),
            ):
                stdout = io.StringIO()
                with patch("sys.stdout", stdout):
                    exit_code = main(["--project-dir", tmpdir, "--vendor", "codex", "--model", "gpt-5.4", "--effort", "high", "--run-init", "yes", "--yes"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(start_stop_events, ["init:run_demo", "start", "stop"])


if __name__ == "__main__":
    unittest.main()
    main,
