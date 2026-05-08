from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from B01_terminal_interaction import (
    AgentInitControlCenter,
    collect_b01_request,
    main,
    parse_control_command,
    prompt_yes_no,
    render_control_help,
    run_terminal_control_loop,
)
from T03_agent_init_workflow import determine_batch_worker_count
from A01_Routing_LayerPlanning import build_parser


class B01TerminalInteractionTests(unittest.TestCase):
    def test_parse_control_command_supports_aliases(self):
        self.assertEqual(parse_control_command("").action, "status")
        self.assertEqual(parse_control_command("list").action, "status")
        self.assertEqual(parse_control_command("open 2").action, "attach")
        self.assertEqual(parse_control_command("logs 3").action, "transcript")
        self.assertEqual(parse_control_command("quit").action, "exit")

    def test_prompt_yes_no_parses_yes_and_no(self):
        with patch("builtins.input", side_effect=["yes", "no"]):
            self.assertTrue(prompt_yes_no("测试", False))
            self.assertFalse(prompt_yes_no("测试", True))

    def test_collect_b01_request_interactive_order_skips_target_dirs_when_no_init(self):
        parser = build_parser()
        args = parser.parse_args([])
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "docs").mkdir()
            (project_dir / "AGENTS.md").write_text("ok", encoding="utf-8")
            (project_dir / "docs" / "repo_map.json").write_text("{}", encoding="utf-8")
            (project_dir / "docs" / "task_routes.json").write_text("{}", encoding="utf-8")
            (project_dir / "docs" / "pitfalls.json").write_text("{}", encoding="utf-8")
            with patch("builtins.input", side_effect=[tmpdir, "no"]):
                request = collect_b01_request(args)
        self.assertFalse(request.run_init)
        self.assertEqual(request.target_dirs, ())
        self.assertEqual(request.vendor, "codex")
        self.assertEqual(request.model, "gpt-5.4")
        self.assertEqual(request.reasoning_effort, "high")
        self.assertEqual(request.proxy_port, "")

    def test_collect_b01_request_forces_init_when_project_routing_is_missing(self):
        parser = build_parser()
        args = parser.parse_args([])
        stdout = io.StringIO()
        def fake_prompt_yes_no(prompt_text: str, default: bool = False) -> bool:
            if prompt_text == "是否需要生成项目路由层":
                raise AssertionError("缺少路由层文件时不应先询问 run_init")
            if prompt_text == "是否使用代理端口":
                return False
            raise AssertionError(prompt_text)
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "B01_terminal_interaction.prompt_project_dir",
            return_value=tmpdir,
        ), patch(
            "B01_terminal_interaction.prompt_yes_no",
            side_effect=fake_prompt_yes_no,
        ), patch(
            "B01_terminal_interaction.prompt_target_dirs",
            return_value=(),
        ), patch(
            "B01_terminal_interaction.prompt_vendor",
            return_value="codex",
        ), patch(
            "B01_terminal_interaction.prompt_model",
            return_value="gpt-5.4",
        ), patch(
            "B01_terminal_interaction.prompt_effort",
            return_value="high",
        ), patch("sys.stdout", new=stdout):
            Path(tmpdir, "app.py").write_text("print('ok')\n", encoding="utf-8")
            with patch("builtins.input", side_effect=["no"]):
                request = collect_b01_request(args)
        self.assertTrue(request.run_init)
        self.assertEqual(request.target_dirs, ())
        self.assertIn("当前项目路由层文件缺失, 强制执行路由初始化", stdout.getvalue())

    def test_collect_b01_request_skips_empty_project_missing_routing_without_prompts(self):
        parser = build_parser()
        args = parser.parse_args(["--yes"])
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "B01_terminal_interaction.prompt_project_dir",
            return_value=tmpdir,
        ), patch(
            "B01_terminal_interaction.prompt_yes_no",
            side_effect=AssertionError("空项目缺少路由层时不应询问 yes/no"),
        ), patch(
            "B01_terminal_interaction.prompt_target_dirs",
            side_effect=AssertionError("空项目缺少路由层时不应询问 target dirs"),
        ), patch(
            "B01_terminal_interaction.prompt_vendor",
            side_effect=AssertionError("空项目缺少路由层时不应询问 vendor"),
        ), patch(
            "B01_terminal_interaction.prompt_model",
            side_effect=AssertionError("空项目缺少路由层时不应询问 model"),
        ), patch(
            "B01_terminal_interaction.prompt_effort",
            side_effect=AssertionError("空项目缺少路由层时不应询问 effort"),
        ):
            with patch("builtins.input", side_effect=AssertionError("不应触发底层 input")):
                request = collect_b01_request(args)
        self.assertFalse(request.run_init)
        self.assertEqual(request.target_dirs, ())
        self.assertEqual(request.vendor, "codex")
        self.assertEqual(request.model, "gpt-5.4")
        self.assertEqual(request.reasoning_effort, "high")
        self.assertEqual(request.proxy_port, "")

    def test_b01_main_uses_force_yes_confirmation_when_project_routing_is_missing(self):
        parser = build_parser()
        args = parser.parse_args([])
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "app.py").write_text("print('ok')\n", encoding="utf-8")
            with patch(
                "builtins.input",
                side_effect=[tmpdir, "", "1", "1", "3", "no"],
            ), patch(
                "B01_terminal_interaction.prompt_confirmation",
                return_value=True,
            ) as mocked_confirm, patch(
                "B01_terminal_interaction.AgentInitControlCenter.create_new"
            ) as mocked_create, patch(
                "B01_terminal_interaction.run_terminal_control_loop",
                return_value=SimpleNamespace(results=[]),
            ), patch(
                "B01_terminal_interaction.determine_exit_code",
                return_value=0,
            ):
                request = collect_b01_request(args)
                config = SimpleNamespace(vendor="codex", model="gpt-5.4", reasoning_effort="high", proxy_url="")
                selection = SimpleNamespace(
                    should_run=True,
                    project_missing_files=("AGENTS.md",),
                    project_dir=tmpdir,
                    selected_dirs=(tmpdir,),
                    skipped_dirs=(),
                    forced_dirs=(tmpdir,),
                )
                preflight_summary = "summary"
                if not request.auto_confirm and not mocked_confirm(preflight_summary, force_yes=bool(selection.project_missing_files)):
                    self.fail("forced confirmation should continue")
                mocked_confirm.assert_called_once_with(preflight_summary, force_yes=True)
                mocked_create.assert_not_called()

    def test_b01_main_skips_empty_project_with_specific_message(self):
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "B01_terminal_interaction.maybe_launch_tui",
            return_value=(False, ["--project-dir", tmpdir, "--yes", "--legacy-cli"]),
        ), patch(
            "B01_terminal_interaction.AgentInitControlCenter.create_new",
            side_effect=AssertionError("空项目不应创建控制中心"),
        ), patch("sys.stdout", new=stdout):
            exit_code = main([])
        self.assertEqual(exit_code, 0)
        self.assertIn("当前项目未检测到业务文件，跳过路由层初始化。", stdout.getvalue())

    def test_collect_b01_request_with_yes_only_still_prompts_for_run_init_and_proxy(self):
        parser = build_parser()
        args = parser.parse_args(["--yes"])
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "docs").mkdir()
            (project_dir / "AGENTS.md").write_text("ok", encoding="utf-8")
            (project_dir / "docs" / "repo_map.json").write_text("{}", encoding="utf-8")
            (project_dir / "docs" / "task_routes.json").write_text("{}", encoding="utf-8")
            (project_dir / "docs" / "pitfalls.json").write_text("{}", encoding="utf-8")
            with patch(
                "builtins.input",
                side_effect=[
                    tmpdir,   # 项目工作目录
                    "no",     # 是否生成项目路由层
                ],
            ):
                request = collect_b01_request(args)
        self.assertFalse(request.run_init)
        self.assertEqual(request.target_dirs, ())
        self.assertEqual(request.vendor, "codex")
        self.assertEqual(request.model, "gpt-5.4")
        self.assertEqual(request.reasoning_effort, "high")
        self.assertEqual(request.proxy_port, "")
        self.assertTrue(request.auto_confirm)

    def test_collect_b01_request_with_project_dir_only_still_prompts_run_init(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "docs").mkdir()
            (project_dir / "AGENTS.md").write_text("ok", encoding="utf-8")
            (project_dir / "docs" / "repo_map.json").write_text("{}", encoding="utf-8")
            (project_dir / "docs" / "task_routes.json").write_text("{}", encoding="utf-8")
            (project_dir / "docs" / "pitfalls.json").write_text("{}", encoding="utf-8")
            args = parser.parse_args(["--project-dir", tmpdir])
            with patch("builtins.input", side_effect=["no"]):
                request = collect_b01_request(args)
        self.assertFalse(request.run_init)
        self.assertEqual(request.target_dirs, ())
        self.assertEqual(request.vendor, "codex")
        self.assertEqual(request.model, "gpt-5.4")
        self.assertEqual(request.reasoning_effort, "high")
        self.assertEqual(request.proxy_port, "")

    def test_collect_b01_request_with_project_dir_only_prompts_target_dirs_effort_and_proxy(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "docs").mkdir()
            (project_dir / "AGENTS.md").write_text("ok", encoding="utf-8")
            (project_dir / "docs" / "repo_map.json").write_text("{}", encoding="utf-8")
            (project_dir / "docs" / "task_routes.json").write_text("{}", encoding="utf-8")
            (project_dir / "docs" / "pitfalls.json").write_text("{}", encoding="utf-8")
            args = parser.parse_args(["--project-dir", tmpdir])
            with patch(
                "builtins.input",
                side_effect=[
                    "yes",
                    "",
                    "1",
                    "1",
                    "3",
                    "no",
                ],
            ):
                request = collect_b01_request(args)
        self.assertTrue(request.run_init)
        self.assertEqual(request.target_dirs, ())
        self.assertEqual(request.vendor, "codex")
        self.assertEqual(request.model, "gpt-5.4")
        self.assertEqual(request.reasoning_effort, "high")
        self.assertEqual(request.proxy_port, "")

    def test_determine_batch_worker_count_allows_nested_directories_to_run_in_parallel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = (Path(tmpdir) / "project").resolve()
            nested = (root / "core").resolve()
            other = (Path(tmpdir) / "other").resolve()
            nested.mkdir(parents=True)
            other.mkdir(parents=True)
            self.assertEqual(determine_batch_worker_count([root, nested], max_workers=4), 4)
            self.assertEqual(determine_batch_worker_count([root, other], max_workers=4), 4)

    def test_render_control_help_contains_attach_and_wait(self):
        help_text = render_control_help()
        self.assertIn("attach <编号|session_name>", help_text)
        self.assertIn("detach <编号|session_name>", help_text)
        self.assertIn("restart <编号|session_name>", help_text)
        self.assertIn("kill <编号|session_name>", help_text)
        self.assertIn("retry <编号|session_name>", help_text)
        self.assertIn("resume <run_id>", help_text)
        self.assertIn("Ctrl-b d", help_text)
        self.assertIn("wait", help_text)

    def test_transition_to_requirements_phase_keeps_failed_run_in_control_stage(self):
        control_center = AgentInitControlCenter.__new__(AgentInitControlCenter)
        batch_result = SimpleNamespace(results=[SimpleNamespace(status="failed")])
        transition_text = AgentInitControlCenter.transition_to_requirements_phase(control_center, batch_result)
        self.assertIn("当前不进入需求录入阶段", transition_text)
        self.assertIn("retry", transition_text)

    def test_get_handle_accepts_session_name_for_live_worker(self):
        control_center = AgentInitControlCenter.__new__(AgentInitControlCenter)
        control_center.selection = SimpleNamespace(selected_dirs=("/tmp/project",))
        handle = SimpleNamespace(session_name="aginit-demo", work_dir="/tmp/project")
        control_center.handle_by_dir = {"/tmp/project": handle}
        control_center.live_workers = [handle]
        control_center.run_store = SimpleNamespace(
            ensure_worker=lambda work_dir: SimpleNamespace(
                work_dir=work_dir,
                session_name="aginit-demo",
                transcript_path="/tmp/transcript.md",
                state_path="",
                result_status="pending",
                note="pending",
                health_note="",
                forced=False,
            ),
            manifest=SimpleNamespace(workers=[SimpleNamespace(work_dir="/tmp/project", session_name="aginit-demo")]),
        )
        control_center._sync_entry_from_state = lambda work_dir: None
        self.assertIs(control_center.get_handle("aginit-demo"), handle)
        self.assertIs(control_center.get_handle("1"), handle)

    def test_get_target_accepts_session_name_from_manifest_without_live_worker(self):
        control_center = AgentInitControlCenter.__new__(AgentInitControlCenter)
        control_center.selection = SimpleNamespace(selected_dirs=("/tmp/project",))
        control_center.handle_by_dir = {}
        control_center.live_workers = []
        control_center.run_store = SimpleNamespace(
            ensure_worker=lambda work_dir: SimpleNamespace(
                work_dir=work_dir,
                session_name="aginit-demo",
                transcript_path="/tmp/transcript.md",
                state_path="",
                result_status="failed",
                note="failed",
                health_note="",
                forced=False,
            ),
            manifest=SimpleNamespace(workers=[SimpleNamespace(work_dir="/tmp/project", session_name="aginit-demo")]),
        )
        target = control_center.get_target("aginit-demo")
        self.assertEqual(target.work_dir, "/tmp/project")
        self.assertEqual(target.session_name, "aginit-demo")
        self.assertIsNone(target.live_handle)

    def test_parser_accepts_resume_run_argument(self):
        parser = build_parser()
        args = parser.parse_args(["--resume-run", "run_demo"])
        self.assertEqual(args.resume_run, "run_demo")

    def test_restart_worker_rejects_manifest_only_target_without_live_handle(self):
        control_center = AgentInitControlCenter.__new__(AgentInitControlCenter)
        control_center.selection = SimpleNamespace(selected_dirs=("/tmp/project",))
        control_center.handle_by_dir = {}
        control_center.live_workers = []
        control_center.tmux_runtime = SimpleNamespace(kill_session=lambda *args, **kwargs: None)
        control_center.run_store = SimpleNamespace(
            ensure_worker=lambda work_dir: SimpleNamespace(
                work_dir=work_dir,
                session_name="aginit-demo",
                transcript_path="/tmp/transcript.md",
                state_path="",
                result_status="failed",
                note="failed",
                health_note="",
                forced=False,
            ),
            manifest=SimpleNamespace(workers=[SimpleNamespace(work_dir="/tmp/project", session_name="aginit-demo")]),
        )
        with self.assertRaisesRegex(RuntimeError, "没有 live worker"):
            control_center.restart_worker("1")

    def test_restart_worker_updates_run_store_for_live_handle(self):
        binding_calls: list[tuple[str, dict[str, object]]] = []
        event_calls: list[tuple[str, dict[str, object]]] = []

        class FakeWorker:
            session_name = "aginit-demo"
            agent_ready = True
            recoverable = False
            agent_state = "STARTING"
            restart_calls = 0

            def request_restart(self):
                self.restart_calls += 1
                self.agent_ready = False
                self.recoverable = True
                self.agent_state = "STARTING"
                return self.session_name

        handle = SimpleNamespace(session_name="aginit-demo", work_dir="/tmp/project", worker=FakeWorker())
        control_center = AgentInitControlCenter.__new__(AgentInitControlCenter)
        control_center.selection = SimpleNamespace(selected_dirs=("/tmp/project",))
        control_center.handle_by_dir = {"/tmp/project": handle}
        control_center.live_workers = [handle]
        control_center.run_store = SimpleNamespace(
            ensure_worker=lambda work_dir: SimpleNamespace(
                work_dir=work_dir,
                session_name="aginit-demo",
                transcript_path="/tmp/transcript.md",
                state_path="",
                result_status="running",
                note="running",
                health_note="",
                forced=False,
            ),
            update_worker_binding=lambda work_dir, **fields: binding_calls.append((work_dir, fields)),
            append_event=lambda event, **payload: event_calls.append((event, payload)),
            manifest=SimpleNamespace(workers=[SimpleNamespace(work_dir="/tmp/project", session_name="aginit-demo")]),
        )
        session_name = control_center.restart_worker("1")
        self.assertEqual(session_name, "aginit-demo")
        self.assertEqual(handle.worker.restart_calls, 1)
        self.assertFalse(handle.worker.agent_ready)
        self.assertTrue(handle.worker.recoverable)
        self.assertEqual(handle.worker.agent_state, "STARTING")
        self.assertEqual(binding_calls, [("/tmp/project", {"recoverable": True, "health_note": "manual_restart_requested"})])
        self.assertEqual(event_calls, [("manual_restart", {"work_dir": "/tmp/project", "session_name": "aginit-demo"})])

    def test_retry_worker_recreates_handle_and_resubmits(self):
        binding_calls: list[tuple[str, dict[str, object]]] = []
        event_calls: list[tuple[str, dict[str, object]]] = []
        submitted: list[object] = []
        kill_calls: list[tuple[str, bool]] = []

        class FakeWorker:
            def __init__(self, **kwargs):
                self.session_name = "aginit-new"
                self.runtime_dir = Path("/tmp/runtime")
                self.state_path = Path("/tmp/runtime/state.json")
                self.transcript_path = Path("/tmp/runtime/transcript.md")
                self.kwargs = kwargs

            def runtime_metadata(self):
                return {
                    "worker_id": "project",
                    "session_name": "aginit-new",
                    "pane_id": "%1",
                    "runtime_dir": "/tmp/runtime",
                    "work_dir": self.kwargs["work_dir"],
                    "log_path": "/tmp/runtime/worker.log",
                    "raw_log_path": "/tmp/runtime/worker.raw.log",
                    "state_path": "/tmp/runtime/state.json",
                    "transcript_path": "/tmp/runtime/transcript.md",
                }

        entry = SimpleNamespace(
            work_dir="/tmp/project",
            session_name="aginit-old",
            transcript_path="/tmp/transcript.md",
            state_path="",
            result_status="failed",
            note="failed",
            health_note="",
            forced=False,
            retry_count=2,
        )
        old_handle = SimpleNamespace(session_name="aginit-old", work_dir="/tmp/project", worker=SimpleNamespace())

        control_center = AgentInitControlCenter.__new__(AgentInitControlCenter)
        control_center.selection = SimpleNamespace(selected_dirs=("/tmp/project",))
        control_center.handle_by_dir = {"/tmp/project": old_handle}
        control_center.live_workers = [old_handle]
        control_center.results_by_dir = {"/tmp/project": SimpleNamespace(status="failed")}
        control_center.config = SimpleNamespace()
        control_center.run_root = Path("/tmp/run")
        control_center.worker_factory = FakeWorker
        control_center.tmux_runtime = SimpleNamespace(
            kill_session=lambda session_name, missing_ok=True: kill_calls.append((session_name, missing_ok)) or session_name
        )
        control_center._submit_handle = lambda handle: submitted.append(handle)
        control_center.run_store = SimpleNamespace(
            ensure_worker=lambda work_dir: entry,
            update_worker_binding=lambda work_dir, **fields: binding_calls.append((work_dir, fields)),
            append_event=lambda event, **payload: event_calls.append((event, payload)),
            manifest=SimpleNamespace(workers=[SimpleNamespace(work_dir="/tmp/project", session_name="aginit-old")]),
        )

        session_name = control_center.retry_worker("1")
        self.assertEqual(session_name, "aginit-new")
        self.assertEqual(control_center.handle_by_dir["/tmp/project"].worker.session_name, "aginit-new")
        self.assertNotIn("/tmp/project", control_center.results_by_dir)
        self.assertEqual(len(submitted), 1)
        self.assertEqual(submitted[0].work_dir, "/tmp/project")
        self.assertEqual(kill_calls, [("aginit-old", True)])
        self.assertEqual(binding_calls[0][0], "/tmp/project")
        self.assertEqual(binding_calls[0][1]["retry_count"], 3)
        self.assertEqual(event_calls, [("manual_retry", {"work_dir": "/tmp/project", "session_name": "aginit-new"})])

    def test_run_terminal_control_loop_rejects_resume_before_current_run_finishes(self):
        batch_result = SimpleNamespace()
        center = SimpleNamespace(
            run_id="run_current",
            max_refine_rounds=3,
            start=lambda: None,
            render_status=lambda: "status",
            all_done=lambda: False,
            wait_until_complete=lambda: batch_result,
            can_switch_runs=lambda: False,
            close=lambda: None,
        )
        stdout = io.StringIO()
        with (
            patch("builtins.input", side_effect=["resume run_other", "wait", "exit"]),
            patch("sys.stdout", new=stdout),
            patch("A01_Routing_LayerPlanning.format_batch_summary", return_value="summary"),
            patch("B01_terminal_interaction.AgentInitControlCenter.from_existing_run") as mocked_resume,
        ):
            result = run_terminal_control_loop(center)
        self.assertIs(result, batch_result)
        self.assertIn("当前 run 尚未完成", stdout.getvalue())
        mocked_resume.assert_not_called()

    def test_run_terminal_control_loop_switches_to_resumed_run_after_completion(self):
        old_result = SimpleNamespace(name="old")
        new_result = SimpleNamespace(name="new")
        old_center = SimpleNamespace(
            run_id="run_old",
            max_refine_rounds=3,
            selection=SimpleNamespace(project_dir="/tmp/project"),
            start=lambda: None,
            render_status=lambda: "old-status",
            all_done=lambda: True,
            wait_until_complete=lambda: old_result,
            can_switch_runs=lambda: True,
            close=lambda: setattr(old_center, "closed", True),
            transition_to_requirements_phase=lambda result: "路由层配置完成\n进入需求录入阶段（占位）\n下一步请运行: python3 A02_RequirementIntake.py",
        )
        old_center.closed = False
        new_center = SimpleNamespace(
            run_id="run_new",
            start=lambda: setattr(new_center, "started", True),
            render_status=lambda: "new-status",
            all_done=lambda: True,
            wait_until_complete=lambda: new_result,
            can_switch_runs=lambda: True,
            close=lambda: setattr(new_center, "closed", True),
            transition_to_requirements_phase=lambda result: "路由层配置完成\n进入需求录入阶段（占位）\n下一步请运行: python3 A02_RequirementIntake.py",
        )
        new_center.started = False
        new_center.closed = False
        stdout = io.StringIO()
        with (
            patch("builtins.input", side_effect=["resume run_new", "exit"]),
            patch("sys.stdout", new=stdout),
            patch("A01_Routing_LayerPlanning.format_batch_summary", return_value="summary"),
            patch("B01_terminal_interaction.AgentInitControlCenter.from_existing_run", return_value=new_center) as mocked_resume,
        ):
            result = run_terminal_control_loop(old_center)
        self.assertIs(result, new_result)
        self.assertTrue(old_center.closed)
        self.assertTrue(new_center.started)
        self.assertTrue(new_center.closed)
        mocked_resume.assert_called_once_with(run_id="run_new", project_dir="/tmp/project", max_refine_rounds=3)
        self.assertIn("已切换到 run: run_new", stdout.getvalue())

    def test_main_resume_run_uses_current_project_dir(self):
        resumed_center = SimpleNamespace()
        resumed_result = SimpleNamespace(results=[])
        resumed_center.start = lambda: None
        resumed_center.render_status = lambda: "status"
        resumed_center.all_done = lambda: True
        resumed_center.wait_until_complete = lambda: resumed_result
        resumed_center.can_switch_runs = lambda: True
        resumed_center.close = lambda: None
        resumed_center.transition_to_requirements_phase = lambda result: ""
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved_project_dir = str(Path(tmpdir).resolve())
            with (
                patch("B01_terminal_interaction.maybe_launch_tui", return_value=(False, ["--project-dir", tmpdir, "--resume-run", "run_demo"])),
                patch("B01_terminal_interaction.AgentInitControlCenter.from_existing_run", return_value=resumed_center) as mocked_resume,
                patch("B01_terminal_interaction.run_terminal_control_loop", return_value=resumed_result),
                patch("B01_terminal_interaction.determine_exit_code", return_value=0),
            ):
                exit_code = main([])

        self.assertEqual(exit_code, 0)
        mocked_resume.assert_called_once_with(run_id="run_demo", project_dir=resolved_project_dir, max_refine_rounds=3)

    def test_pending_work_count_counts_unfinished_dirs(self):
        control_center = AgentInitControlCenter.__new__(AgentInitControlCenter)
        control_center.selection = SimpleNamespace(selected_dirs=("/tmp/project",))
        control_center.results_by_dir = {}
        self.assertEqual(control_center.pending_work_count(), 1)

    def test_build_worker_snapshots_refreshes_latest_worker_state_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = str((Path(tmpdir) / "project").resolve())
            state_path = Path(tmpdir) / "runtime" / "worker.state.json"
            state_path.parent.mkdir(parents=True)
            state_path.write_text(
                json.dumps(
                    {
                        "session_name": "sess-routing",
                        "runtime_dir": str(state_path.parent),
                        "workflow_stage": "pending",
                        "result_status": "pending",
                        "agent_state": "BUSY",
                        "agent_started": True,
                        "agent_alive": True,
                        "health_status": "alive",
                        "retry_count": 1,
                        "last_heartbeat_at": "2026-05-03T13:00:00",
                        "state_path": str(state_path),
                        "transcript_path": str(state_path.parent / "transcript.md"),
                    }
                ),
                encoding="utf-8",
            )
            entry = SimpleNamespace(
                work_dir=project_dir,
                session_name="sess-routing",
                transcript_path="",
                state_path=str(state_path),
                result_status="running",
                workflow_stage="create_running",
                agent_state="STARTING",
                health_status="unknown",
                health_note="",
                retry_count=0,
                note="create_routing_layer",
                forced=True,
                current_turn_status_path="",
            )

            def update_worker_state_from_file(work_dir, path, preserve_workflow_fields=False):  # noqa: ANN001
                self.assertTrue(preserve_workflow_fields)
                self.assertEqual(work_dir, project_dir)
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
                entry.session_name = payload["session_name"]
                entry.agent_state = payload["agent_state"]
                entry.health_status = payload["health_status"]
                entry.retry_count = payload["retry_count"]
                entry.transcript_path = payload["transcript_path"]
                return entry

            control_center = AgentInitControlCenter.__new__(AgentInitControlCenter)
            control_center.selection = SimpleNamespace(selected_dirs=(project_dir,))
            control_center.results_by_dir = {}
            control_center.futures = {}
            control_center.handle_by_dir = {}
            control_center.tmux_runtime = SimpleNamespace(session_exists=lambda name: name == "sess-routing")
            control_center.run_store = SimpleNamespace(
                ensure_worker=lambda work_dir: entry,
                update_worker_state_from_file=update_worker_state_from_file,
            )

            snapshots = control_center.build_worker_snapshots()

        self.assertEqual(snapshots[0]["agent_state"], "BUSY")
        self.assertEqual(snapshots[0]["health_status"], "alive")
        self.assertEqual(snapshots[0]["retry_count"], 1)

    def test_refresh_worker_health_keeps_prelaunch_active_worker_starting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = str((Path(tmpdir) / "project").resolve())
            runtime_dir = Path(tmpdir) / "runtime"
            entry = SimpleNamespace(
                work_dir=project_dir,
                session_name="sess-routing",
                runtime_dir=str(runtime_dir),
                pane_id="",
                result_status="running",
                workflow_stage="create_running",
                agent_state="DEAD",
                agent_started=False,
                health_status="dead",
                health_note="missing_session",
                retry_count=0,
                note="create_routing_layer",
                recoverable=True,
                forced=True,
                state_path="",
                transcript_path="",
                current_turn_status_path="",
            )

            def ensure_worker(*, work_dir):  # noqa: ANN001
                self.assertEqual(work_dir, project_dir)
                return entry

            def sync_worker_snapshot(work_dir, **fields):  # noqa: ANN001
                self.assertEqual(work_dir, project_dir)
                for key, value in fields.items():
                    if hasattr(entry, key):
                        setattr(entry, key, value)
                return entry

            def refresh_health(**_kwargs):  # noqa: ANN001
                raise AssertionError("prelaunch worker should not be probed as dead")

            worker = SimpleNamespace(
                session_name="sess-routing",
                runtime_dir=runtime_dir,
                state_path=runtime_dir / "worker.state.json",
                refresh_health=refresh_health,
            )
            handle = SimpleNamespace(work_dir=project_dir, worker=worker, forced=True)
            control_center = AgentInitControlCenter.__new__(AgentInitControlCenter)
            control_center.live_workers = [handle]
            control_center.results_by_dir = {}
            control_center.run_store = SimpleNamespace(
                ensure_worker=ensure_worker,
                sync_worker_snapshot=sync_worker_snapshot,
            )

            control_center.refresh_worker_health()

        self.assertEqual(entry.agent_state, "STARTING")
        self.assertEqual(entry.health_status, "unknown")
        self.assertEqual(entry.health_note, "launch pending")

    def test_run_terminal_control_loop_announces_stage_transition_and_cleans_tmux(self):
        batch_result = SimpleNamespace(run_id="run_demo")
        transition_calls = []
        center = SimpleNamespace(
            run_id="run_demo",
            max_refine_rounds=3,
            start=lambda: None,
            render_status=lambda: "status",
            all_done=lambda: True,
            wait_until_complete=lambda: batch_result,
            can_switch_runs=lambda: True,
            close=lambda: None,
            transition_to_requirements_phase=lambda result: transition_calls.append(result) or "路由层配置完成\n进入需求录入阶段（占位）\n下一步请运行: python3 A02_RequirementIntake.py",
        )
        stdout = io.StringIO()
        with (
            patch("builtins.input", side_effect=["exit"]),
            patch("sys.stdout", new=stdout),
            patch("A01_Routing_LayerPlanning.format_batch_summary", return_value="summary"),
        ):
            result = run_terminal_control_loop(center)
        self.assertIs(result, batch_result)
        self.assertEqual(transition_calls, [batch_result])
        self.assertIn("路由层配置完成", stdout.getvalue())
        self.assertIn("进入需求录入阶段（占位）", stdout.getvalue())
        self.assertIn("A02_RequirementIntake.py", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
