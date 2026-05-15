from __future__ import annotations

from pathlib import Path
import subprocess
import unittest
from unittest.mock import Mock, patch

from T09_terminal_ops import (
    BridgePromptRequest,
    BridgeTerminalUI,
    PROMPT_BACK_VALUE,
    PromptBackRequested,
    StdioTerminalUI,
    ensure_tui_dependencies_installed,
    terminal_ui_is_interactive,
    message,
    maybe_launch_tui,
    prompt_metadata,
    prompt_select_option,
    prompt_with_default,
    set_terminal_ui,
    use_terminal_ui,
)


class _FakeProgress:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.lines: list[str] = []

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def _display_line(self, line: str) -> None:
        self.lines.append(line)


class _FakeUI:
    def __init__(self):
        self.messages: list[str] = []

    def message(self, *objects, sep=" ", end="\n", flush=False):  # noqa: ANN001
        self.messages.append(sep.join(str(item) for item in objects) + end)

    def prompt_text(self, prompt_text: str, default: str = "", allow_empty: bool = False) -> str:
        return default or "value"

    def prompt_select(
        self,
        *,
        title: str,
        options,
        default_value: str,
        prompt_text: str = "请选择",
        preview_path=None,
        preview_title: str = "",
        is_hitl: bool = False,
        extra_payload=None,
    ):  # noqa: ANN001
        _ = is_hitl
        _ = extra_payload
        return default_value

    def prompt_multiline(self, *, title: str, empty_retry_message: str = "输入不能为空，请重试。") -> str:
        return "multi"

    def clear_pending_tty_input(self) -> None:
        return None

    def create_progress_monitor(self, *, frame_builder, stream=None, interval_sec: float = 0.2):  # noqa: ANN001
        return _FakeProgress()

    def attach_external_process(self, command, *, cwd=None, env=None):  # noqa: ANN001
        return 0


class T09TerminalOpsTests(unittest.TestCase):
    def tearDown(self) -> None:
        set_terminal_ui(StdioTerminalUI())

    def test_message_routes_through_current_terminal_ui(self):
        fake = _FakeUI()
        with use_terminal_ui(fake):
            message("hello", "world")
        self.assertEqual(fake.messages, ["hello world\n"])

    def test_prompt_with_default_routes_through_current_terminal_ui(self):
        fake = _FakeUI()
        with use_terminal_ui(fake):
            value = prompt_with_default("输入项目工作目录", "/tmp/demo")
        self.assertEqual(value, "/tmp/demo")

    def test_bridge_terminal_ui_emits_log_and_prompt_events(self):
        events: list[tuple[str, dict[str, object]]] = []

        def emit_event(event_type: str, payload: dict[str, object]) -> None:
            events.append((event_type, payload))

        def request_prompt(request: BridgePromptRequest) -> dict[str, object]:
            self.assertEqual(request.prompt_type, "text")
            return {"value": "abc"}

        ui = BridgeTerminalUI(emit_event=emit_event, request_prompt=request_prompt)
        ui.message("line")
        value = ui.prompt_text("输入")
        self.assertEqual(value, "abc")
        self.assertEqual(events[0][0], "log.append")
        self.assertEqual(events[0][1]["text"], "line\n")

    def test_bridge_terminal_ui_prompt_metadata_supports_back_request(self):
        captured: list[BridgePromptRequest] = []

        def request_prompt(request: BridgePromptRequest) -> dict[str, object]:
            captured.append(request)
            return {"value": PROMPT_BACK_VALUE}

        ui = BridgeTerminalUI(emit_event=lambda *_args, **_kwargs: None, request_prompt=request_prompt)
        with self.assertRaises(PromptBackRequested):
            with prompt_metadata(allow_back=True, back_value=PROMPT_BACK_VALUE, stage_key="routing", stage_step_index=2):
                ui.prompt_text("输入")

        self.assertEqual(captured[0].payload["allow_back"], True)
        self.assertEqual(captured[0].payload["back_value"], PROMPT_BACK_VALUE)
        self.assertEqual(captured[0].payload["stage_key"], "routing")
        self.assertEqual(captured[0].payload["stage_step_index"], 2)

    def test_bridge_terminal_ui_back_value_is_plain_text_without_allow_back(self):
        ui = BridgeTerminalUI(
            emit_event=lambda *_args, **_kwargs: None,
            request_prompt=lambda _request: {"value": PROMPT_BACK_VALUE},
        )
        self.assertEqual(ui.prompt_text("输入"), PROMPT_BACK_VALUE)

    def test_bridge_terminal_ui_multiline_prompt_preserves_hitl_paths(self):
        captured: list[BridgePromptRequest] = []

        def request_prompt(request: BridgePromptRequest) -> dict[str, object]:
            captured.append(request)
            return {"value": "abc"}

        ui = BridgeTerminalUI(emit_event=lambda *_args, **_kwargs: None, request_prompt=request_prompt)
        value = ui.prompt_multiline(
            title="HITL 第 1 轮回复",
            question_path="/tmp/question.md",
            answer_path="/tmp/answer.md",
        )
        self.assertEqual(value, "abc")
        self.assertEqual(captured[0].payload["question_path"], str(Path("/tmp/question.md").resolve()))
        self.assertEqual(captured[0].payload["answer_path"], str(Path("/tmp/answer.md").resolve()))
        self.assertFalse(captured[0].payload["is_hitl"])

    def test_bridge_terminal_ui_multiline_prompt_marks_explicit_hitl_requests(self):
        captured: list[BridgePromptRequest] = []

        def request_prompt(request: BridgePromptRequest) -> dict[str, object]:
            captured.append(request)
            return {"value": "abc"}

        ui = BridgeTerminalUI(emit_event=lambda *_args, **_kwargs: None, request_prompt=request_prompt)
        ui.prompt_multiline(
            title="请回复",
            question_path="/tmp/question.md",
            answer_path="/tmp/answer.md",
            is_hitl=True,
        )
        self.assertTrue(captured[0].payload["is_hitl"])

    def test_bridge_terminal_ui_select_prompt_preserves_preview_metadata(self):
        captured: list[BridgePromptRequest] = []

        def request_prompt(request: BridgePromptRequest) -> dict[str, object]:
            captured.append(request)
            return {"value": "yes"}

        ui = BridgeTerminalUI(emit_event=lambda *_args, **_kwargs: None, request_prompt=request_prompt)
        value = ui.prompt_select(
            title="是否继续",
            options=(("yes", "yes"), ("no", "no")),
            default_value="no",
            preview_path="/tmp/clarification.md",
            preview_title="需求澄清文档",
        )
        self.assertEqual(value, "yes")
        self.assertEqual(captured[0].payload["preview_path"], str(Path("/tmp/clarification.md").resolve()))
        self.assertEqual(captured[0].payload["preview_title"], "需求澄清文档")

    def test_bridge_terminal_ui_select_prompt_passes_hitl_recovery_metadata(self):
        captured: list[BridgePromptRequest] = []

        def request_prompt(request: BridgePromptRequest) -> dict[str, object]:
            captured.append(request)
            return {"value": "retry_after_manual_model_change"}

        ui = BridgeTerminalUI(emit_event=lambda *_args, **_kwargs: None, request_prompt=request_prompt)
        value = ui.prompt_select(
            title="HITL: 智能体启动超时",
            options=(("retry_after_manual_model_change", "我已手动更换模型，继续尝试"),),
            default_value="retry_after_manual_model_change",
            is_hitl=True,
            extra_payload={
                "recovery_kind": "agent_ready_timeout",
                "session_name": "开发工程师-天魁星",
                "role_label": "开发工程师",
                "can_skip": False,
            },
        )

        self.assertEqual(value, "retry_after_manual_model_change")
        self.assertTrue(captured[0].payload["is_hitl"])
        self.assertEqual(captured[0].payload["recovery_kind"], "agent_ready_timeout")
        self.assertEqual(captured[0].payload["session_name"], "开发工程师-天魁星")
        self.assertFalse(captured[0].payload["can_skip"])

    def test_prompt_select_option_keeps_legacy_terminal_ui_compatible(self):
        class LegacySelectUI(_FakeUI):
            def prompt_select(
                self,
                *,
                title: str,
                options,
                default_value: str,
                prompt_text: str = "请选择",
                preview_path=None,
                preview_title: str = "",
            ):  # noqa: ANN001
                return default_value

        with use_terminal_ui(LegacySelectUI()):
            value = prompt_select_option(
                title="请选择",
                options=(("yes", "yes"),),
                default_value="yes",
            )
        self.assertEqual(value, "yes")

    def test_bridge_terminal_ui_progress_monitor_emits_stage_context(self):
        events: list[tuple[str, dict[str, object]]] = []

        def emit_event(event_type: str, payload: dict[str, object]) -> None:
            events.append((event_type, payload))

        ui = BridgeTerminalUI(
            emit_event=emit_event,
            request_prompt=lambda _request: {"value": "ok"},
            progress_context_provider=lambda: {"action": "stage.a05.start", "stage_seq": 12},
        )
        monitor = ui.create_progress_monitor(frame_builder=lambda _tick: "running")
        monitor.start()
        monitor.stop()
        progress_events = [payload for event_type, payload in events if event_type.startswith("progress.")]
        self.assertTrue(progress_events)
        self.assertTrue(all(item["action"] == "stage.a05.start" for item in progress_events))
        self.assertTrue(all(item["stage_seq"] == 12 for item in progress_events))

    def test_terminal_ui_is_interactive_when_bridge_ui_is_active(self):
        def emit_event(_event_type: str, _payload: dict[str, object]) -> None:
            return None

        def request_prompt(_request: BridgePromptRequest) -> dict[str, object]:
            return {"value": "ok"}

        ui = BridgeTerminalUI(emit_event=emit_event, request_prompt=request_prompt)
        with use_terminal_ui(ui):
            self.assertTrue(terminal_ui_is_interactive())

    def test_stdio_attach_external_process_waits_for_child_shutdown_after_keyboard_interrupt(self):
        ui = StdioTerminalUI()
        child = Mock()
        child.wait.side_effect = [KeyboardInterrupt(), 0]
        with patch("T09_terminal_ops.subprocess.Popen", return_value=child):
            self.assertEqual(ui.attach_external_process(["bun", "run"]), 130)
        child.send_signal.assert_called_once()
        child.terminate.assert_not_called()
        child.kill.assert_not_called()

    def test_stdio_attach_external_process_escalates_to_sigterm_and_sigkill_when_child_hangs(self):
        ui = StdioTerminalUI()
        child = Mock()
        child.wait.side_effect = [
            KeyboardInterrupt(),
            subprocess.TimeoutExpired(cmd=["bun", "run"], timeout=30.0),
            subprocess.TimeoutExpired(cmd=["bun", "run"], timeout=10.0),
            0,
        ]
        with patch("T09_terminal_ops.subprocess.Popen", return_value=child):
            self.assertEqual(ui.attach_external_process(["bun", "run"]), 130)
        child.send_signal.assert_called_once()
        child.terminate.assert_called_once_with()
        child.kill.assert_called_once_with()

    def test_maybe_launch_tui_falls_back_to_legacy_when_help_requested(self):
        redirected, payload = maybe_launch_tui(["--help"], route="home", action="workflow.a00.start")
        self.assertFalse(redirected)
        self.assertEqual(payload, ["--help"])

    def test_maybe_launch_tui_raises_when_launcher_fails(self):
        with (
            patch("T09_terminal_ops.sys.stdin.isatty", return_value=True),
            patch("T09_terminal_ops.sys.stdout.isatty", return_value=True),
            patch("T09_terminal_ops.ensure_tui_dependencies_installed"),
            patch("T09_terminal_ops.attach_external_process", side_effect=RuntimeError("bun missing")),
            patch("T09_terminal_ops.sys.argv", ["A00_main_tui.py"]),
        ):
            with self.assertRaisesRegex(RuntimeError, "bun missing"):
                maybe_launch_tui(None, route="home", action="workflow.a00.start")

    def test_maybe_launch_tui_installs_dependencies_before_launching_tui(self):
        with (
            patch("T09_terminal_ops.sys.stdin.isatty", return_value=True),
            patch("T09_terminal_ops.sys.stdout.isatty", return_value=True),
            patch("T09_terminal_ops.ensure_tui_dependencies_installed") as ensure_install,
            patch("T09_terminal_ops.attach_external_process", return_value=0) as attach_process,
            patch("T09_terminal_ops.sys.argv", ["A00_main_tui.py"]),
        ):
            redirected, payload = maybe_launch_tui(None, route="home", action="workflow.a00.start")
        self.assertTrue(redirected)
        self.assertEqual(payload, 0)
        ensure_install.assert_called_once_with()
        attach_process.assert_called_once()

    def test_maybe_launch_tui_forwards_explicit_cli_args_to_tui(self):
        with (
            patch("T09_terminal_ops.sys.stdin.isatty", return_value=True),
            patch("T09_terminal_ops.sys.stdout.isatty", return_value=True),
            patch("T09_terminal_ops.ensure_tui_dependencies_installed") as ensure_install,
            patch("T09_terminal_ops.attach_external_process", return_value=0) as attach_process,
            patch("T09_terminal_ops.sys.argv", ["A06_TaskSplit.py", "--project-dir", "/tmp/project"]),
        ):
            redirected, payload = maybe_launch_tui(None, route="task-split", action="stage.a06.start")
        self.assertTrue(redirected)
        self.assertEqual(payload, 0)
        ensure_install.assert_called_once_with()
        command = attach_process.call_args.args[0]
        self.assertIn("--argv-json", command)
        self.assertEqual(command[command.index("--argv-json") + 1], '["--project-dir", "/tmp/project"]')

    def test_maybe_launch_tui_preserves_legacy_cli_escape_hatch(self):
        redirected, payload = maybe_launch_tui(["--legacy-cli", "--help"], route="home", action="workflow.a00.start")
        self.assertFalse(redirected)
        self.assertEqual(payload, ["--help"])

    def test_maybe_launch_tui_honors_no_tui_escape_hatch(self):
        redirected, payload = maybe_launch_tui(["--no-tui"], route="home", action="workflow.a00.start")
        self.assertFalse(redirected)
        self.assertEqual(payload, [])

    def test_ensure_tui_dependencies_installed_runs_bun_install_when_required_packages_are_missing(self):
        result = Mock(returncode=0, stderr="", stdout="")
        with (
            patch("T09_terminal_ops.tui_package_dir", return_value=Path("/tmp/tui")),
            patch("pathlib.Path.exists", return_value=True),
            patch("T09_terminal_ops._missing_tui_dependency_names", side_effect=[["solid-js"], []]),
            patch("T09_terminal_ops.message"),
            patch("T09_terminal_ops.subprocess.run", return_value=result) as run_install,
        ):
            ensure_tui_dependencies_installed()
        run_install.assert_called_once_with(
            ["bun", "install", "--frozen-lockfile"],
            cwd="/tmp/tui",
            check=False,
            capture_output=True,
            text=True,
        )

    def test_ensure_tui_dependencies_installed_raises_when_install_finishes_with_missing_packages(self):
        result = Mock(returncode=0, stderr="", stdout="")
        with (
            patch("T09_terminal_ops.tui_package_dir", return_value=Path("/tmp/tui")),
            patch("pathlib.Path.exists", return_value=True),
            patch("T09_terminal_ops._missing_tui_dependency_names", side_effect=[["solid-js"], ["solid-js"]]),
            patch("T09_terminal_ops.message"),
            patch("T09_terminal_ops.subprocess.run", return_value=result),
        ):
            with self.assertRaisesRegex(RuntimeError, "OpenTUI 依赖安装后仍缺少: solid-js"):
                ensure_tui_dependencies_installed()


if __name__ == "__main__":
    unittest.main()
