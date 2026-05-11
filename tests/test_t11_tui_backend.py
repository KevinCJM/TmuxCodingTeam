from __future__ import annotations

import io
import json
import os
import queue
import signal
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from A02_RequirementIntake import NOTION_RUNTIME_ROOT_NAME
from A04_RequirementsReview import REQUIREMENTS_REVIEW_RUNTIME_ROOT_NAME, build_requirements_review_paths
from A05_DetailedDesign import DETAILED_DESIGN_RUNTIME_ROOT_NAME, build_detailed_design_paths
from A06_TaskSplit import TASK_SPLIT_RUNTIME_ROOT_NAME, build_task_split_paths
from A07_Development import DEVELOPMENT_RUNTIME_ROOT_NAME, build_development_paths, build_reviewer_artifact_paths
from A08_OverallReview import build_overall_review_paths
from T03_agent_init_workflow import ROUTING_RUNTIME_ROOT_NAME, build_routing_runtime_root, required_routing_layer_paths
from T08_pre_development import update_pre_development_task_status
from T12_requirements_common import build_requirements_clarification_paths
from T11_tui_backend import (
    ControlSessionState,
    HumanAttentionManager,
    PendingPromptState,
    PromptBroker,
    TuiBackendServer,
    main as backend_main,
)
from T10_tui_protocol import build_request
from T09_terminal_ops import BridgePromptRequest
from tmux_core.runtime.tmux_runtime import clear_runtime_shutdown_request, runtime_shutdown_requested


class _FakeTarget:
    def __init__(self, *, session_name: str = "sess-1", transcript_path: str = "/tmp/transcript.md", work_dir: str = "/tmp/demo"):
        self.session_name = session_name
        self.transcript_path = transcript_path
        self.work_dir = work_dir


class _FakeCenter:
    def __init__(self, *, run_id: str = "run_demo", done: bool = False):
        self.run_id = run_id
        self.run_root = "/tmp/runtime"
        self._done = done
        self.closed = False
        self.cleaned = False
        self.selection = SimpleNamespace(project_dir="/tmp/project")

    def refresh_worker_health(self) -> None:
        return None

    def build_status_rows(self):
        return [{"index": 1, "session_name": "sess-1", "status": "running"}]

    def build_worker_snapshots(self):
        return [
            {
                "index": 1,
                "session_name": "sess-1",
                "work_dir": "/tmp/project",
                "status": "running",
                "workflow_stage": "create_running",
                "agent_state": "READY",
                "health_status": "healthy",
                "retry_count": 0,
                "note": "running",
                "transcript_path": "/tmp/transcript.md",
                "turn_status_path": "/tmp/turn_status.json",
                "question_path": "",
                "answer_path": "",
                "artifact_paths": [],
            }
        ]

    def all_done(self) -> bool:
        return self._done

    def wait_until_complete(self):
        return type("Batch", (), {"run_id": self.run_id, "runtime_dir": "/tmp/runtime", "config": {"vendor": "codex", "model": "gpt-5.4", "reasoning_effort": "high", "proxy_url": ""}, "results": []})()

    def transition_to_requirements_phase(self, batch_result):  # noqa: ANN001
        return "进入需求录入阶段（占位）"

    def can_switch_runs(self) -> bool:
        return True

    def render_status(self) -> str:
        return "status text"

    def get_target(self, argument: str):  # noqa: ARG002
        return _FakeTarget()

    def detach(self, argument: str):  # noqa: ARG002
        return "sess-1"

    def kill_worker(self, argument: str):  # noqa: ARG002
        return "sess-1"

    def restart_worker(self, argument: str):  # noqa: ARG002
        return "sess-1"

    def retry_worker(self, argument: str):  # noqa: ARG002
        return "sess-1"

    def close(self) -> None:
        self.closed = True

    def start(self) -> None:
        return None

    def cleanup_routing_tmux_sessions(self):
        self.cleaned = True
        return ["sess-1"]


class T11TuiBackendTests(unittest.TestCase):
    def test_prompt_broker_roundtrip(self):
        events: list[tuple[str, dict[str, object]]] = []
        broker = PromptBroker(lambda event_type, payload: events.append((event_type, payload)))

        def resolve_later() -> None:
            broker.resolve(str(events[0][1]["id"]), {"value": "ok"})

        import threading

        threading.Timer(0.01, resolve_later).start()
        payload = broker.request(type("Req", (), {"prompt_type": "text", "payload": {"prompt_text": "输入"}})())
        self.assertEqual(payload["value"], "ok")
        self.assertEqual(events[0][0], "prompt.request")

    def test_prompt_broker_assigns_distinct_ids_to_sequential_prompts_on_same_thread(self):
        events: list[tuple[str, dict[str, object]]] = []
        broker = PromptBroker(lambda event_type, payload: events.append((event_type, payload)))

        def resolve_last(value: str) -> None:
            broker.resolve(str(events[-1][1]["id"]), {"value": value})

        import threading

        threading.Timer(0.01, resolve_last, args=("first",)).start()
        first = broker.request(type("Req", (), {"prompt_type": "select", "payload": {"prompt_text": "第一个"}})())
        threading.Timer(0.01, resolve_last, args=("second",)).start()
        second = broker.request(type("Req", (), {"prompt_type": "select", "payload": {"prompt_text": "第二个"}})())

        prompt_ids = [str(payload["id"]) for event_type, payload in events if event_type == "prompt.request"]
        self.assertEqual(first["value"], "first")
        self.assertEqual(second["value"], "second")
        self.assertEqual(len(prompt_ids), 2)
        self.assertNotEqual(prompt_ids[0], prompt_ids[1])

    def test_prompt_broker_shutdown_unblocks_pending_prompt(self):
        events: list[tuple[str, dict[str, object]]] = []
        broker = PromptBroker(lambda event_type, payload: events.append((event_type, payload)))
        started = threading.Event()
        errors: list[str] = []

        def wait_for_prompt() -> None:
            try:
                started.set()
                broker.request(type("Req", (), {"prompt_type": "text", "payload": {"prompt_text": "输入"}})())
            except RuntimeError as error:
                errors.append(str(error))

        worker = threading.Thread(target=wait_for_prompt)
        worker.start()
        self.assertTrue(started.wait(timeout=2.0))
        while not events:
            time.sleep(0.01)
        broker.shutdown("backend exiting")
        worker.join(timeout=2.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, ["backend exiting"])

    def test_pending_hitl_snapshot_can_be_derived_from_active_prompt(self):
        server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
        server._pending_prompt = PendingPromptState(  # noqa: SLF001
            prompt_id="prompt_1",
            prompt_type="multiline",
            payload={
                "title": "HITL 第 1 轮回复",
                "question_path": "/tmp/question.md",
                "answer_path": "/tmp/answer.md",
                "session_name": "需求分析师-地奇星",
            },
        )
        hitl = server._build_hitl_snapshot()  # noqa: SLF001
        app = server._build_app_snapshot()  # noqa: SLF001
        self.assertTrue(hitl["pending"])
        self.assertEqual(hitl["summary"], "HITL 第 1 轮回复")
        self.assertEqual(hitl["question_path"], "/tmp/question.md")
        self.assertEqual(hitl["answer_path"], "/tmp/answer.md")
        self.assertEqual(hitl["attach_command"], "tmux attach -t 需求分析师-地奇星")
        self.assertTrue(app["pending_hitl"])

    def test_pending_hitl_snapshot_prefers_explicit_hitl_flag(self):
        server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
        server._pending_prompt = PendingPromptState(  # noqa: SLF001
            prompt_id="prompt_1",
            prompt_type="multiline",
            payload={
                "title": "请回复",
                "question_path": "/tmp/question.md",
                "answer_path": "/tmp/answer.md",
                "is_hitl": True,
                "attach_command": "tmux attach -t 测试工程师-天暴星",
            },
        )
        hitl = server._build_hitl_snapshot()  # noqa: SLF001
        self.assertTrue(hitl["pending"])
        self.assertEqual(hitl["question_path"], "/tmp/question.md")
        self.assertEqual(hitl["attach_command"], "tmux attach -t 测试工程师-天暴星")

    def test_human_attention_manager_repeats_until_resolved(self):
        notifications: list[tuple[str, str, str]] = []
        manager = HumanAttentionManager(
            adapter_name_provider=lambda: "tui",
            platform_name="darwin",
            osascript_path="/usr/bin/osascript",
            interval_sec=0.05,
            notifier=lambda title, subtitle, body: notifications.append((title, subtitle, body)) or None,
        )

        manager.start_prompt(
            prompt_id="prompt_1",
            prompt_type="multiline",
            payload={"title": "请回复", "is_hitl": True},
            stage_label="任务开发",
        )
        time.sleep(0.13)
        snapshot = manager.snapshot()
        manager.resolve_prompt("prompt_1")
        notification_count = len(notifications)
        time.sleep(0.08)

        self.assertGreaterEqual(notification_count, 2)
        self.assertTrue(snapshot["pending"])
        self.assertEqual(snapshot["reason"], "hitl")
        self.assertEqual(snapshot["body"], "HITL 待处理")
        self.assertTrue(all(item[1] == "任务开发" for item in notifications[:notification_count]))
        self.assertFalse(manager.snapshot()["pending"])

    def test_human_attention_manager_keeps_other_unresolved_prompts_active(self):
        notifications: list[tuple[str, str, str]] = []
        manager = HumanAttentionManager(
            adapter_name_provider=lambda: "tui",
            platform_name="darwin",
            osascript_path="/usr/bin/osascript",
            interval_sec=0.05,
            notifier=lambda title, subtitle, body: notifications.append((title, subtitle, body)) or None,
        )

        manager.start_prompt(
            prompt_id="prompt_1",
            prompt_type="multiline",
            payload={"title": "请回复", "is_hitl": True},
            stage_label="任务开发",
        )
        manager.start_prompt(
            prompt_id="prompt_2",
            prompt_type="select",
            payload={"title": "请选择 reviewer 模型"},
            stage_label="任务拆分",
        )
        time.sleep(0.08)
        manager.resolve_prompt("prompt_2")
        time.sleep(0.08)
        snapshot = manager.snapshot()
        manager.shutdown()

        self.assertTrue(snapshot["pending"])
        self.assertEqual(snapshot["reason"], "hitl")
        self.assertIn(("TmuxCodingTeam 需要人工介入", "任务拆分", "请选择 reviewer 模型"), notifications)

    def test_human_attention_manager_initial_notify_is_async(self):
        manager = HumanAttentionManager(
            adapter_name_provider=lambda: "tui",
            platform_name="darwin",
            osascript_path="/usr/bin/osascript",
            interval_sec=60,
            notifier=lambda _title, _subtitle, _body: time.sleep(0.2) or None,
        )

        started = time.perf_counter()
        manager.start_prompt(
            prompt_id="prompt_1",
            prompt_type="text",
            payload={"prompt_text": "请输入项目目录"},
            stage_label="路由初始化",
        )
        elapsed = time.perf_counter() - started
        manager.shutdown()

        self.assertLess(elapsed, 0.1)

    def test_human_attention_manager_suppresses_initial_notify_while_tui_presence_is_recent(self):
        notifications: list[tuple[str, str, str]] = []
        presence_until = [time.monotonic() + 0.06]

        def presence_provider() -> dict[str, object]:
            delay = max(presence_until[0] - time.monotonic(), 0.0)
            return {
                "recent": delay > 0,
                "active_until": "soon",
                "delay_sec": delay,
            }

        manager = HumanAttentionManager(
            adapter_name_provider=lambda: "tui",
            platform_name="darwin",
            osascript_path="/usr/bin/osascript",
            interval_sec=0.05,
            presence_provider=presence_provider,
            notifier=lambda title, subtitle, body: notifications.append((title, subtitle, body)) or None,
        )

        try:
            manager.start_prompt(
                prompt_id="prompt_1",
                prompt_type="select",
                payload={"title": "请选择 reviewer 模型"},
                stage_label="任务拆分",
            )
            deadline = time.time() + 0.5
            while time.time() < deadline and not manager.snapshot().get("suppressed_due_to_presence"):
                time.sleep(0.005)
            suppressed_snapshot = manager.snapshot()
            self.assertFalse(suppressed_snapshot["pending"])
            self.assertTrue(suppressed_snapshot["suppressed_due_to_presence"])
            self.assertEqual(notifications, [])

            deadline = time.time() + 0.5
            while time.time() < deadline and not notifications:
                time.sleep(0.005)
            self.assertGreaterEqual(len(notifications), 1)
            self.assertTrue(manager.snapshot()["pending"])
        finally:
            manager.shutdown()

    def test_human_attention_manager_skips_repeat_notify_while_tui_presence_is_recent(self):
        notifications: list[tuple[str, str, str]] = []
        presence_until = [0.0]

        def presence_provider() -> dict[str, object]:
            delay = max(presence_until[0] - time.monotonic(), 0.0)
            return {
                "recent": delay > 0,
                "active_until": "soon",
                "delay_sec": delay,
            }

        manager = HumanAttentionManager(
            adapter_name_provider=lambda: "tui",
            platform_name="darwin",
            osascript_path="/usr/bin/osascript",
            interval_sec=0.03,
            presence_provider=presence_provider,
            notifier=lambda title, subtitle, body: notifications.append((title, subtitle, body)) or None,
        )

        try:
            manager.start_prompt(
                prompt_id="prompt_1",
                prompt_type="text",
                payload={"prompt_text": "请输入项目目录"},
                stage_label="路由初始化",
            )
            deadline = time.time() + 0.5
            while time.time() < deadline and not notifications:
                time.sleep(0.005)
            self.assertEqual(len(notifications), 1)

            presence_until[0] = time.monotonic() + 0.08
            time.sleep(0.05)
            self.assertEqual(len(notifications), 1)
            self.assertTrue(manager.snapshot()["suppressed_due_to_presence"])

            deadline = time.time() + 0.5
            while time.time() < deadline and len(notifications) < 2:
                time.sleep(0.005)
            self.assertGreaterEqual(len(notifications), 2)
        finally:
            manager.shutdown()

    def test_human_attention_manager_is_disabled_when_not_tui_or_not_macos(self):
        notifications: list[tuple[str, str, str]] = []
        manager = HumanAttentionManager(
            adapter_name_provider=lambda: "web",
            platform_name="linux",
            osascript_path="/usr/bin/osascript",
            notifier=lambda title, subtitle, body: notifications.append((title, subtitle, body)) or None,
        )

        manager.start_prompt(
            prompt_id="prompt_1",
            prompt_type="select",
            payload={"title": "请选择 reviewer 模型"},
            stage_label="任务拆分",
        )

        self.assertEqual(notifications, [])
        self.assertFalse(manager.snapshot()["pending"])

    def test_ui_presence_action_refreshes_tui_presence_window(self):
        server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
        server.attach_adapter("tui")

        result = server.dispatch_action(
            "ui.presence",
            {"reason": "keyboard", "shell_focus": "content"},
            respond=False,
        )

        self.assertTrue(result["accepted"])
        self.assertTrue(server.is_tui_presence_recent())
        self.assertTrue(server.presence_expires_at())

    def test_ui_presence_action_is_ignored_for_web_adapter(self):
        server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
        server._adapter_name = "web"  # noqa: SLF001

        result = server.dispatch_action(
            "ui.presence",
            {"reason": "keyboard", "shell_focus": "content"},
            respond=False,
        )

        self.assertFalse(result["accepted"])
        self.assertFalse(server.is_tui_presence_recent())

    def test_prompt_open_starts_attention_manager_for_current_stage(self):
        server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
        calls: list[dict[str, object]] = []
        server._attention_manager = SimpleNamespace(  # noqa: SLF001
            start_prompt=lambda **kwargs: calls.append(dict(kwargs)),
            resolve_prompt=lambda *_args, **_kwargs: None,
            snapshot=lambda: {"pending": False, "reason": "", "started_at": ""},
            shutdown=lambda: None,
        )
        server._set_context(project_dir="/tmp/project", requirement_name="需求A", action="stage.a06.start")  # noqa: SLF001

        server._handle_prompt_open(  # noqa: SLF001
            "prompt_1",
            BridgePromptRequest(
                prompt_type="select",
                payload={"title": "请选择 reviewer 模型"},
            ),
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["prompt_id"], "prompt_1")
        self.assertEqual(calls[0]["stage_label"], "任务拆分")

    def test_app_snapshot_exposes_pending_attention_fields(self):
        server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
        server._attention_manager = SimpleNamespace(  # noqa: SLF001
            start_prompt=lambda **kwargs: None,
            resolve_prompt=lambda *_args, **_kwargs: None,
            snapshot=lambda: {
                "pending": True,
                "reason": "select",
                "started_at": "2026-04-23T10:00:00+08:00",
            },
            shutdown=lambda: None,
        )

        snapshot = server._build_app_snapshot()  # noqa: SLF001

        self.assertTrue(snapshot["pending_attention"])
        self.assertEqual(snapshot["pending_attention_reason"], "select")
        self.assertEqual(snapshot["pending_attention_since"], "2026-04-23T10:00:00+08:00")

    def test_stage_status_treats_awaiting_reconfig_worker_as_running(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            runtime_dir = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME / "worker-awaiting-reconfig"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            (runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "development-developer",
                        "session_name": "开发工程师-天魁星",
                        "work_dir": str(project_dir),
                        "project_dir": str(project_dir),
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a07.start",
                        "result_status": "failed",
                        "status": "failed",
                        "agent_state": "DEAD",
                        "health_status": "awaiting_reconfig",
                        "health_note": "需要重新选择模型",
                        "note": "awaiting_reconfig",
                        "updated_at": "2026-04-23T10:00:00+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="stage.a07.start")  # noqa: SLF001

            status = server._infer_runtime_stage_status("stage.a07.start")  # noqa: SLF001
            failed = server._failed_stage_worker_summaries("stage.a07.start")  # noqa: SLF001

        self.assertEqual(status, "running")
        self.assertEqual(failed, [])

    def test_manual_reconfiguration_error_pending_detects_awaiting_reconfig_worker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            runtime_dir = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME / "worker-awaiting-reconfig"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            (runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "development-developer",
                        "session_name": "开发工程师-柳土獐",
                        "work_dir": str(project_dir),
                        "project_dir": str(project_dir),
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a07.start",
                        "result_status": "running",
                        "agent_state": "STARTING",
                        "health_status": "awaiting_reconfig",
                        "health_note": "需要重新选择模型",
                        "note": "awaiting_reconfig",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="stage.a07.start")  # noqa: SLF001

            pending = server._manual_reconfiguration_error_pending(  # noqa: SLF001
                action="stage.a07.start",
                error=RuntimeError("检测到 开发工程师-柳土獐 需要重新启动或重建。原因: tmux pane missing"),
            )

        self.assertTrue(pending)

    def test_empty_hitl_question_file_is_not_reported_as_pending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _, _, ask_human_path, _ = build_requirements_clarification_paths(project_dir, "贪吃蛇")
            ask_human_path.parent.mkdir(parents=True, exist_ok=True)
            ask_human_path.write_text("\n", encoding="utf-8")
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a04.start")  # noqa: SLF001
            hitl = server._build_hitl_snapshot()  # noqa: SLF001
        self.assertFalse(hitl["pending"])
        self.assertEqual(hitl["question_path"], "")

    def test_file_driven_hitl_is_ignored_during_routing_stage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _, _, ask_human_path, _ = build_requirements_clarification_paths(project_dir, "贪吃蛇")
            development_paths = build_development_paths(project_dir, "贪吃蛇")
            ask_human_path.parent.mkdir(parents=True, exist_ok=True)
            ask_human_path.write_text("旧的需求澄清问题\n", encoding="utf-8")
            development_paths["ask_human_path"].parent.mkdir(parents=True, exist_ok=True)
            development_paths["ask_human_path"].write_text("旧的开发澄清问题\n", encoding="utf-8")
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a01.start")  # noqa: SLF001
            hitl = server._build_hitl_snapshot()  # noqa: SLF001

        self.assertFalse(hitl["pending"])
        self.assertEqual(hitl["question_path"], "")

    def test_requirements_file_hitl_is_ignored_during_intake_stage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _, _, ask_human_path, _ = build_requirements_clarification_paths(project_dir, "贪吃蛇")
            ask_human_path.parent.mkdir(parents=True, exist_ok=True)
            ask_human_path.write_text("旧的需求澄清问题\n", encoding="utf-8")
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a02.start")  # noqa: SLF001
            hitl = server._build_hitl_snapshot()  # noqa: SLF001

        self.assertFalse(hitl["pending"])
        self.assertEqual(hitl["question_path"], "")

    def test_hitl_snapshot_can_be_derived_from_current_stage_worker_question(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            requirement_name = "需求A"
            paths = build_development_paths(project_dir, requirement_name)
            for path in (
                paths["task_md_path"],
                paths["task_json_path"],
                paths["developer_output_path"],
                paths["merged_review_path"],
                paths["detailed_design_path"],
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("ok\n", encoding="utf-8")
            paths["task_json_path"].write_text(json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False), encoding="utf-8")
            paths["ask_human_path"].write_text("请确认评审冲突\n", encoding="utf-8")
            runtime_dir = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME / "worker-review"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            (runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "development-developer",
                        "session_name": "开发工程师-天魁星",
                        "work_dir": str(project_dir),
                        "project_dir": str(project_dir),
                        "requirement_name": requirement_name,
                        "workflow_action": "stage.a07.start",
                        "result_status": "running",
                        "workflow_stage": "turn_running",
                        "agent_state": "READY",
                        "health_status": "alive",
                        "question_path": str(paths["ask_human_path"]),
                        "answer_path": str(paths["hitl_record_path"]),
                        "updated_at": "2026-04-23T10:00:00+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), requirement_name=requirement_name, action="stage.a07.start")  # noqa: SLF001
            hitl = server._build_hitl_snapshot()  # noqa: SLF001

        self.assertTrue(hitl["pending"])
        self.assertEqual(hitl["question_path"], str(paths["ask_human_path"]))

    def test_answered_hitl_question_is_not_re_reported_until_question_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _, _, ask_human_path, _ = build_requirements_clarification_paths(project_dir, "贪吃蛇")
            ask_human_path.parent.mkdir(parents=True, exist_ok=True)
            ask_human_path.write_text("请补充边界条件\n", encoding="utf-8")
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime.session_exists = lambda session_name: session_name == "需求分析师-天慧星"  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a05.start")  # noqa: SLF001
            server._pending_prompt = PendingPromptState(  # noqa: SLF001
                prompt_id="prompt_1",
                prompt_type="multiline",
                payload={
                    "title": "HITL 第 1 轮回复",
                    "question_path": str(ask_human_path),
                    "answer_path": str(project_dir / "贪吃蛇_人机交互澄清记录.md"),
                },
            )

            server._handle_prompt_resolved("prompt_1", {"value": "这里是答复"})  # noqa: SLF001
            answered = server._build_hitl_snapshot()  # noqa: SLF001

            ask_human_path.write_text("请补充异常分支\n", encoding="utf-8")
            next_question = server._build_hitl_snapshot()  # noqa: SLF001

        self.assertFalse(answered["pending"])
        self.assertTrue(next_question["pending"])
        self.assertEqual(next_question["question_path"], str(ask_human_path))

    def test_review_snapshot_lists_ba_and_all_reviewers_from_runtime_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_requirements_review_paths(project_dir, "贪吃蛇")
            for file_path in (
                paths["merged_review_path"],
                paths["ba_feedback_path"],
                paths["ask_human_path"],
                paths["hitl_record_path"],
            ):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("", encoding="utf-8")
            runtime_root = project_dir / REQUIREMENTS_REVIEW_RUNTIME_ROOT_NAME
            session_names = [
                "评审分析师-天佑星",
                "审核器-天平星",
                "审核器-地隐星",
                "审核器-天哭星",
            ]
            for index, session_name in enumerate(session_names, start=1):
                runtime_dir = runtime_root / f"worker-{index}"
                runtime_dir.mkdir(parents=True, exist_ok=True)
                (runtime_dir / "worker.state.json").write_text(
                    json.dumps(
                        {
                            "session_name": session_name,
                            "work_dir": str(project_dir),
                            "result_status": "running",
                            "workflow_stage": "turn_running",
                            "agent_state": "BUSY",
                            "health_status": "alive",
                            "retry_count": 0,
                            "note": "",
                            "transcript_path": "",
                            "updated_at": "2026-04-17T18:00:00+08:00",
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime.session_exists = lambda name: True  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a04.start")  # noqa: SLF001

            review = server._build_review_snapshot()  # noqa: SLF001

        self.assertEqual([worker["session_name"] for worker in review["workers"]], session_names)

    def test_review_snapshot_does_not_materialize_hitl_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            original_requirement_path, requirements_clear_path, _, hitl_record_path = build_requirements_clarification_paths(project_dir, "需求A")
            original_requirement_path.write_text("原始需求正文\n", encoding="utf-8")
            requirements_clear_path.write_text("需求澄清正文\n", encoding="utf-8")
            self.assertFalse(hitl_record_path.exists())

            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="stage.a04.start")  # noqa: SLF001

            review = server._build_review_snapshot()  # noqa: SLF001

        self.assertEqual(review["requirement_name"], "需求A")
        self.assertFalse(hitl_record_path.exists())

    def test_review_snapshot_includes_reused_analyst_from_clarification_runtime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            review_runtime_dir = project_dir / REQUIREMENTS_REVIEW_RUNTIME_ROOT_NAME / "reviewer"
            review_runtime_dir.mkdir(parents=True, exist_ok=True)
            (review_runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "requirements-review-r1",
                        "session_name": "审核器-天损星",
                        "work_dir": str(project_dir.resolve()),
                        "project_dir": str(project_dir.resolve()),
                        "requirement_name": "日志持久化",
                        "workflow_action": "stage.a04.start",
                        "result_status": "running",
                        "workflow_stage": "turn_running",
                        "agent_state": "BUSY",
                        "agent_started": True,
                        "health_status": "alive",
                        "current_task_runtime_status": "running",
                        "note": "turn:requirements_review_init_R1_round_1",
                        "updated_at": "2026-05-05T13:18:22",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            clarification_runtime_dir = project_dir / ".requirements_clarification_runtime" / "ba"
            clarification_runtime_dir.mkdir(parents=True, exist_ok=True)
            (clarification_runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "requirements-analyst",
                        "session_name": "分析师-天闲星",
                        "work_dir": str(project_dir.resolve()),
                        "result_status": "succeeded",
                        "workflow_stage": "pending",
                        "agent_state": "READY",
                        "agent_started": True,
                        "health_status": "alive",
                        "current_task_runtime_status": "done",
                        "note": "done:requirements_clarification_round_1",
                        "updated_at": "2026-05-05T13:16:41",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime.session_exists = lambda name: name in {"审核器-天损星", "分析师-天闲星"}  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="日志持久化", action="stage.a04.start")  # noqa: SLF001

            review = server._build_review_snapshot()  # noqa: SLF001

        self.assertEqual(
            [worker["session_name"] for worker in review["workers"]],
            ["审核器-天损星", "分析师-天闲星"],
        )

    def test_design_snapshot_lists_ba_and_all_reviewers_from_runtime_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_detailed_design_paths(project_dir, "贪吃蛇")
            for file_path in (
                paths["detailed_design_path"],
                paths["merged_review_path"],
                paths["ba_feedback_path"],
                paths["ask_human_path"],
                paths["hitl_record_path"],
            ):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("", encoding="utf-8")
            runtime_root = project_dir / DETAILED_DESIGN_RUNTIME_ROOT_NAME
            session_names = [
                "需求分析师-天佑星",
                "开发工程师-天魁星",
                "测试工程师-天英星",
                "审核员-天机星",
            ]
            for index, session_name in enumerate(session_names, start=1):
                runtime_dir = runtime_root / f"worker-{index}"
                runtime_dir.mkdir(parents=True, exist_ok=True)
                (runtime_dir / "worker.state.json").write_text(
                    json.dumps(
                        {
                            "session_name": session_name,
                            "work_dir": str(project_dir),
                            "result_status": "running",
                            "workflow_stage": "turn_running",
                            "agent_state": "BUSY",
                            "health_status": "alive",
                            "retry_count": 0,
                            "note": "",
                            "transcript_path": "",
                            "updated_at": "2026-04-18T10:00:00+08:00",
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime.session_exists = lambda name: True  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a05.start")  # noqa: SLF001

            design = server._build_design_snapshot()  # noqa: SLF001

        self.assertEqual([worker["session_name"] for worker in design["workers"]], session_names)

    def test_design_snapshot_includes_reused_ba_from_previous_stage_runtime_roots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_detailed_design_paths(project_dir, "贪吃蛇")
            for file_path in (
                paths["detailed_design_path"],
                paths["merged_review_path"],
                paths["ba_feedback_path"],
                paths["ask_human_path"],
                paths["hitl_record_path"],
            ):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("", encoding="utf-8")

            clarification_runtime_dir = project_dir / ".requirements_clarification_runtime" / "ba"
            clarification_runtime_dir.mkdir(parents=True, exist_ok=True)
            (clarification_runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "需求分析师-天佑星",
                        "work_dir": str(project_dir),
                        "result_status": "running",
                        "workflow_stage": "turn_running",
                        "agent_state": "BUSY",
                        "health_status": "alive",
                        "retry_count": 0,
                        "note": "",
                        "transcript_path": "",
                        "updated_at": "2026-04-19T10:00:00+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            design_runtime_dir = project_dir / DETAILED_DESIGN_RUNTIME_ROOT_NAME / "reviewer"
            design_runtime_dir.mkdir(parents=True, exist_ok=True)
            (design_runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "开发工程师-天魁星",
                        "work_dir": str(project_dir),
                        "result_status": "running",
                        "workflow_stage": "turn_running",
                        "agent_state": "BUSY",
                        "health_status": "alive",
                        "retry_count": 0,
                        "note": "",
                        "transcript_path": "",
                        "updated_at": "2026-04-19T10:01:00+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime.session_exists = lambda name: True  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a05.start")  # noqa: SLF001

            design = server._build_design_snapshot()  # noqa: SLF001

        self.assertEqual(
            [worker["session_name"] for worker in design["workers"]],
            ["开发工程师-天魁星", "需求分析师-天佑星"],
        )

    def test_design_snapshot_keeps_completed_ba_handoff_when_reviewers_are_scoped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_detailed_design_paths(project_dir, "日志持久化")
            for file_path in (
                paths["detailed_design_path"],
                paths["merged_review_path"],
                paths["ba_feedback_path"],
                paths["ask_human_path"],
                paths["hitl_record_path"],
            ):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("", encoding="utf-8")

            clarification_runtime_dir = project_dir / ".requirements_clarification_runtime" / "ba"
            clarification_runtime_dir.mkdir(parents=True, exist_ok=True)
            (clarification_runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "requirements-analyst",
                        "session_name": "分析师-天闲星",
                        "work_dir": str(project_dir.resolve()),
                        "result_status": "succeeded",
                        "workflow_stage": "pending",
                        "agent_state": "READY",
                        "agent_started": True,
                        "health_status": "alive",
                        "current_task_runtime_status": "done",
                        "note": "done:generate_detailed_design",
                        "updated_at": "2026-05-05T12:01:02",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            design_runtime_dir = project_dir / DETAILED_DESIGN_RUNTIME_ROOT_NAME / "日志持久化" / "reviewer"
            design_runtime_dir.mkdir(parents=True, exist_ok=True)
            (design_runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "detailed-design-review-架构师",
                        "session_name": "架构师-奎木狼",
                        "work_dir": str(project_dir.resolve()),
                        "project_dir": str(project_dir.resolve()),
                        "requirement_name": "日志持久化",
                        "workflow_action": "stage.a05.start",
                        "result_status": "succeeded",
                        "workflow_stage": "pending",
                        "agent_state": "READY",
                        "agent_started": True,
                        "health_status": "alive",
                        "current_task_runtime_status": "done",
                        "note": "done:detailed_design_review_init_架构师_round_1",
                        "updated_at": "2026-05-05T12:02:02",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime.session_exists = lambda name: name in {"分析师-天闲星", "架构师-奎木狼"}  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="日志持久化", action="stage.a05.start")  # noqa: SLF001

            design = server._build_design_snapshot()  # noqa: SLF001

        self.assertEqual(
            [worker["session_name"] for worker in design["workers"]],
            ["架构师-奎木狼", "分析师-天闲星"],
        )

    def test_stage_a05_status_is_not_polluted_by_failed_review_worker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            review_runtime_dir = project_dir / REQUIREMENTS_REVIEW_RUNTIME_ROOT_NAME / "worker-failed"
            review_runtime_dir.mkdir(parents=True, exist_ok=True)
            (review_runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "requirements-review-r1",
                        "session_name": "审核器-毕月乌",
                        "work_dir": str(project_dir),
                        "result_status": "failed",
                        "workflow_stage": "pending",
                        "agent_state": "DEAD",
                        "health_status": "alive",
                        "note": "error:requirements_review_round_2",
                        "updated_at": "2026-04-20T14:06:58+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            design_runtime_dir = project_dir / DETAILED_DESIGN_RUNTIME_ROOT_NAME / "worker-running"
            design_runtime_dir.mkdir(parents=True, exist_ok=True)
            (design_runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "detailed-design-analyst",
                        "session_name": "需求分析师-天慧星",
                        "work_dir": str(project_dir),
                        "result_status": "running",
                        "workflow_stage": "pending",
                        "agent_state": "BUSY",
                        "health_status": "alive",
                        "note": "turn:detailed_design_feedback_round_2",
                        "updated_at": "2026-04-20T14:05:58+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime.session_exists = lambda session_name: session_name == "需求分析师-天慧星"  # noqa: SLF001
            server._tmux_runtime.backend.session_exists = lambda session_name: session_name == "需求分析师-天慧星"  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a05.start")  # noqa: SLF001
            with patch("tmux_core.bridge.backend.load_worker_from_state_path", return_value=None):
                status = server._infer_runtime_stage_status("stage.a05.start")  # noqa: SLF001

        self.assertEqual(status, "running")

    def test_stage_a05_status_ignores_previous_stage_failed_workers_before_design_runtime_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            review_runtime_dir = project_dir / REQUIREMENTS_REVIEW_RUNTIME_ROOT_NAME / "worker-failed"
            review_runtime_dir.mkdir(parents=True, exist_ok=True)
            (review_runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "requirements-review-analyst",
                        "session_name": "需求分析师-天慧星",
                        "work_dir": str(project_dir),
                        "result_status": "failed",
                        "workflow_stage": "pending",
                        "agent_state": "DEAD",
                        "health_status": "alive",
                        "note": "error:requirements_review_feedback_round_2",
                        "updated_at": "2026-04-22T17:01:00+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            clarification_runtime_dir = project_dir / ".requirements_clarification_runtime" / "worker-failed"
            clarification_runtime_dir.mkdir(parents=True, exist_ok=True)
            (clarification_runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "requirements-analyst",
                        "session_name": "分析师-天富星",
                        "work_dir": str(project_dir),
                        "result_status": "failed",
                        "workflow_stage": "pending",
                        "agent_state": "DEAD",
                        "health_status": "alive",
                        "note": "error:requirements_clarification_round_1",
                        "updated_at": "2026-04-22T17:02:00+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a05.start")  # noqa: SLF001

            workers = server._current_stage_workers("stage.a05.start")  # noqa: SLF001
            status = server._infer_runtime_stage_status("stage.a05.start")  # noqa: SLF001

        self.assertEqual(workers, [])
        self.assertEqual(status, "")

    def test_stage_a05_status_ignores_running_nested_clarification_analyst_for_status_inference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            clarification_runtime_dir = project_dir / ".requirements_clarification_runtime" / "worker-running"
            clarification_runtime_dir.mkdir(parents=True, exist_ok=True)
            (clarification_runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "requirements-analyst",
                        "session_name": "分析师-天慧星",
                        "work_dir": str(project_dir),
                        "result_status": "running",
                        "workflow_stage": "turn_running",
                        "agent_state": "BUSY",
                        "health_status": "alive",
                        "note": "turn:requirements_clarification_round_1",
                        "updated_at": "2026-04-20T14:05:58+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime.session_exists = lambda session_name: session_name == "分析师-天慧星"  # noqa: SLF001
            server._tmux_runtime.backend.session_exists = lambda session_name: session_name == "分析师-天慧星"  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a05.start")  # noqa: SLF001
            with patch("tmux_core.bridge.backend.load_worker_from_state_path", return_value=None):
                status = server._infer_runtime_stage_status("stage.a05.start")  # noqa: SLF001

        self.assertEqual(status, "")

    def test_stage_a05_status_treats_dead_workers_as_failed_not_running(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            design_runtime_dir = project_dir / DETAILED_DESIGN_RUNTIME_ROOT_NAME / "worker-dead"
            design_runtime_dir.mkdir(parents=True, exist_ok=True)
            (design_runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "detailed-design-reviewer",
                        "session_name": "审核员-地异星",
                        "work_dir": str(project_dir),
                        "status": "running",
                        "result_status": "running",
                        "workflow_stage": "turn_running",
                        "agent_state": "DEAD",
                        "health_status": "dead",
                        "note": "tmux session missing",
                        "updated_at": "2026-04-22T10:05:58+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a05.start")  # noqa: SLF001

            status = server._infer_runtime_stage_status("stage.a05.start")  # noqa: SLF001

        self.assertEqual(status, "failed")

    def test_task_split_snapshot_lists_ba_and_all_reviewers_from_runtime_roots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_task_split_paths(project_dir, "贪吃蛇")
            for file_path in (
                paths["task_md_path"],
                paths["task_json_path"],
                paths["merged_review_path"],
                paths["ba_feedback_path"],
                paths["ask_human_path"],
                paths["detailed_design_path"],
            ):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("ok\n", encoding="utf-8")
            runtime_root = project_dir / TASK_SPLIT_RUNTIME_ROOT_NAME
            session_names = [
                "需求分析师-天佑星",
                "开发工程师-天魁星",
                "测试工程师-天英星",
            ]
            for index, session_name in enumerate(session_names, start=1):
                runtime_dir = runtime_root / f"worker-{index}"
                runtime_dir.mkdir(parents=True, exist_ok=True)
                (runtime_dir / "worker.state.json").write_text(
                    json.dumps(
                        {
                            "session_name": session_name,
                            "work_dir": str(project_dir),
                            "result_status": "running",
                            "workflow_stage": "turn_running",
                            "agent_state": "BUSY",
                            "health_status": "alive",
                            "retry_count": 0,
                            "note": "",
                            "transcript_path": "",
                            "updated_at": "2026-04-20T10:00:00+08:00",
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime.session_exists = lambda name: True  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a06.start")  # noqa: SLF001

            task_split = server._build_task_split_snapshot()  # noqa: SLF001

        self.assertEqual([worker["session_name"] for worker in task_split["workers"]], session_names)
        self.assertTrue(any(item["label"] == "任务单" for item in task_split["files"]))
        self.assertTrue(any(item["label"] == "任务单 JSON" for item in task_split["files"]))

    def test_development_snapshot_lists_workers_and_stage_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_development_paths(project_dir, "贪吃蛇")
            for file_path, content in (
                (paths["task_md_path"], "任务单\n"),
                (
                    paths["task_json_path"],
                    json.dumps({"M1": {"M1-T1": True}, "M2": {"M2-T1": False, "M2-T2": True}}, ensure_ascii=False, indent=2),
                ),
                (paths["ask_human_path"], ""),
                (paths["developer_output_path"], "开发内容\n"),
                (paths["merged_review_path"], "评审记录\n"),
                (paths["detailed_design_path"], "详细设计\n"),
            ):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")
            runtime_root = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME
            session_names = [
                "开发工程师-天魁星",
                "测试工程师-天英星",
            ]
            for index, session_name in enumerate(session_names, start=1):
                runtime_dir = runtime_root / f"worker-{index}"
                runtime_dir.mkdir(parents=True, exist_ok=True)
                (runtime_dir / "worker.state.json").write_text(
                    json.dumps(
                        {
                            "worker_id": "development-developer" if index == 1 else "development-review-测试工程师",
                            "session_name": session_name,
                            "work_dir": str(project_dir),
                            "result_status": "running",
                            "workflow_stage": "turn_running",
                            "agent_state": "BUSY",
                            "health_status": "alive",
                            "retry_count": 0,
                            "note": "",
                            "transcript_path": "",
                            "updated_at": "2026-04-21T10:00:00+08:00",
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime.session_exists = lambda name: True  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a07.start")  # noqa: SLF001

            development = server._build_development_snapshot()  # noqa: SLF001

        self.assertEqual([worker["session_name"] for worker in development["workers"]], session_names)
        self.assertTrue(any(item["label"] == "任务单" for item in development["files"]))
        self.assertTrue(any(item["label"] == "与人类交流" for item in development["files"]))
        self.assertEqual(development["current_milestone_key"], "M2")
        self.assertFalse(development["all_tasks_completed"])
        self.assertEqual([item["key"] for item in development["milestones"]], ["M1", "M2"])
        self.assertEqual(
            development["milestones"][1]["tasks"],
            [
                {"key": "M2-T1", "completed": False},
                {"key": "M2-T2", "completed": True},
            ],
        )

    def test_build_development_snapshot_omits_milestones_when_task_json_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_development_paths(project_dir, "贪吃蛇")
            for file_path, content in (
                (paths["task_md_path"], "任务单\n"),
                (paths["task_json_path"], json.dumps({"M1": {"M1-T1": "invalid"}}, ensure_ascii=False, indent=2)),
                (paths["detailed_design_path"], "详细设计\n"),
            ):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime.session_exists = lambda session_name: session_name == "开发工程师-天魁星"  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a07.start")  # noqa: SLF001

            development = server._build_development_snapshot()  # noqa: SLF001

        self.assertIn("task_json_invalid", development["blockers"])
        self.assertEqual(development["milestones"], [])
        self.assertEqual(development["current_milestone_key"], "")
        self.assertFalse(development["all_tasks_completed"])

    def test_overall_review_snapshot_lists_workers_files_and_blockers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_overall_review_paths(project_dir, "贪吃蛇")
            for file_path, content in (
                (paths["original_requirement_path"], "原始需求\n"),
                (paths["requirements_clear_path"], "需求澄清\n"),
                (paths["task_md_path"], "任务单\n"),
                (paths["task_json_path"], json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False, indent=2)),
                (paths["developer_output_path"], "工程师开发内容\n"),
                (paths["merged_review_path"], "合并复核记录\n"),
                (paths["detailed_design_path"], "详细设计\n"),
            ):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")
            runtime_root = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME
            session_names = [
                "开发工程师-天魁星",
                "测试工程师-天英星",
            ]
            for index, session_name in enumerate(session_names, start=1):
                runtime_dir = runtime_root / f"worker-a08-{index}"
                runtime_dir.mkdir(parents=True, exist_ok=True)
                (runtime_dir / "worker.state.json").write_text(
                    json.dumps(
                        {
                            "worker_id": "development-developer" if index == 1 else "development-review-测试工程师",
                            "session_name": session_name,
                            "work_dir": str(project_dir),
                            "result_status": "running",
                            "workflow_stage": "turn_running",
                            "workflow_action": "stage.a08.start",
                            "agent_state": "BUSY",
                            "health_status": "alive",
                            "retry_count": 0,
                            "note": "",
                            "transcript_path": "",
                            "updated_at": "2026-04-23T10:00:00+08:00",
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime.session_exists = lambda name: True  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a08.start")  # noqa: SLF001

            overall_review = server._build_overall_review_snapshot()  # noqa: SLF001

        self.assertEqual([worker["session_name"] for worker in overall_review["workers"]], session_names)
        self.assertTrue(any(item["label"] == "任务单 JSON" for item in overall_review["files"]))
        self.assertTrue(any(item["label"] == "复核完成状态" for item in overall_review["files"]))
        self.assertEqual(overall_review["blockers"], ["overall_review_not_passed"])

    def test_hitl_snapshot_prefers_development_question_when_a07_is_active(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_development_paths(project_dir, "贪吃蛇")
            paths["ask_human_path"].parent.mkdir(parents=True, exist_ok=True)
            paths["ask_human_path"].write_text("请确认数据库字段映射\n", encoding="utf-8")
            paths["hitl_record_path"].write_text("历史回复\n", encoding="utf-8")
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime.session_exists = lambda session_name: session_name == "开发工程师-天魁星"  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a07.start")  # noqa: SLF001

            hitl = server._build_hitl_snapshot()  # noqa: SLF001

        self.assertTrue(hitl["pending"])
        self.assertEqual(hitl["question_path"], str(paths["ask_human_path"]))

    def test_hitl_snapshot_prefers_development_question_even_without_current_action(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_development_paths(project_dir, "贪吃蛇")
            paths["ask_human_path"].parent.mkdir(parents=True, exist_ok=True)
            paths["ask_human_path"].write_text("请确认任务边界\n", encoding="utf-8")
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="idle")  # noqa: SLF001

            hitl = server._build_hitl_snapshot()  # noqa: SLF001

        self.assertTrue(hitl["pending"])
        self.assertEqual(hitl["question_path"], str(paths["ask_human_path"]))

    def test_stage_a07_status_prefers_running_worker_over_failed_worker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            runtime_root = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME
            for worker_name, status, updated_at in (
                ("worker-failed", "failed", "2026-04-21T09:00:00+08:00"),
                ("worker-running", "running", "2026-04-21T10:00:00+08:00"),
            ):
                runtime_dir = runtime_root / worker_name
                runtime_dir.mkdir(parents=True, exist_ok=True)
                (runtime_dir / "worker.state.json").write_text(
                    json.dumps(
                        {
                            "worker_id": "development-review-架构师" if "failed" in worker_name else "development-developer",
                            "session_name": "架构师-天机星" if "failed" in worker_name else "开发工程师-天魁星",
                            "work_dir": str(project_dir),
                            "result_status": status,
                            "workflow_stage": "turn_running",
                            "agent_state": "BUSY",
                            "health_status": "alive",
                            "note": "",
                            "updated_at": updated_at,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime.session_exists = lambda session_name: session_name == "开发工程师-天魁星"  # noqa: SLF001
            server._tmux_runtime.backend.session_exists = lambda session_name: session_name == "开发工程师-天魁星"  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a07.start")  # noqa: SLF001
            with patch("tmux_core.bridge.backend.load_worker_from_state_path", return_value=None):
                status = server._infer_runtime_stage_status("stage.a07.start")  # noqa: SLF001

        self.assertEqual(status, "running")

    def test_stage_a07_status_reports_failed_when_any_current_worker_failed_and_none_running(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            runtime_root = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME
            for worker_name, session_name, status, updated_at in (
                ("worker-failed", "需求分析师-虚日鼠", "failed", "2026-04-21T09:00:00+08:00"),
                ("worker-succeeded", "开发工程师-天魁星", "succeeded", "2026-04-21T10:00:00+08:00"),
            ):
                runtime_dir = runtime_root / worker_name
                runtime_dir.mkdir(parents=True, exist_ok=True)
                (runtime_dir / "worker.state.json").write_text(
                    json.dumps(
                        {
                            "worker_id": "development-review-需求分析师" if "failed" in worker_name else "development-developer",
                            "session_name": session_name,
                            "work_dir": str(project_dir),
                            "result_status": status,
                            "workflow_stage": "pending",
                            "agent_state": "DEAD" if status == "failed" else "READY",
                            "health_status": "alive",
                            "note": "error:development_reviewer_init_需求分析师" if status == "failed" else "done:development_developer_init",
                            "updated_at": updated_at,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a07.start")  # noqa: SLF001

            status = server._infer_runtime_stage_status("stage.a07.start")  # noqa: SLF001

        self.assertEqual(status, "failed")

    def test_stage_a07_runtime_state_change_recovers_running_from_failed_when_tasks_remain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_development_paths(project_dir, "需求A")
            paths["task_json_path"].parent.mkdir(parents=True, exist_ok=True)
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            runtime_root = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME
            for worker_name, worker_id, session_name in (
                ("worker-developer", "development-developer", "开发工程师-天魁星"),
                ("worker-reviewer", "development-review-审核员", "审核员-天伤星"),
            ):
                runtime_dir = runtime_root / worker_name
                runtime_dir.mkdir(parents=True, exist_ok=True)
                (runtime_dir / "worker.state.json").write_text(
                    json.dumps(
                        {
                            "worker_id": worker_id,
                            "session_name": session_name,
                            "work_dir": str(project_dir),
                            "project_dir": str(project_dir.resolve()),
                            "requirement_name": "需求A",
                            "workflow_action": "stage.a07.start",
                            "result_status": "succeeded",
                            "agent_state": "READY",
                            "agent_started": True,
                            "agent_alive": True,
                            "current_command": "codex",
                            "health_status": "alive",
                            "updated_at": "2026-04-24T12:00:00+08:00",
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)
            server._tmux_runtime.session_exists = lambda _session_name: True  # noqa: SLF001
            server._tmux_runtime.backend.session_exists = lambda _session_name: True  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="stage.a07.start")  # noqa: SLF001
            server._display_action = "stage.a07.start"  # noqa: SLF001
            server._display_status = "failed"  # noqa: SLF001
            server._display_stage_seq = 7  # noqa: SLF001
            server._stage_seq_counter = 7  # noqa: SLF001

            with patch("tmux_core.bridge.backend.load_worker_from_state_path", return_value=None):
                server._bridge_ui.notify_runtime_state_changed()  # noqa: SLF001
                server._flush_dirty_snapshots()  # noqa: SLF001
                messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]

            stage_events = [item for item in messages if item.get("kind") == "event" and item.get("type") == "stage.changed"]
            state_path = project_dir / ".tmux_workflow" / "需求A" / "stages" / "stage_a07_start.state.json"
            state_payload = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertTrue(stage_events)
        self.assertEqual(stage_events[-1]["payload"]["action"], "stage.a07.start")
        self.assertEqual(stage_events[-1]["payload"]["status"], "running")
        self.assertGreater(int(stage_events[-1]["payload"]["stage_seq"]), 7)
        self.assertEqual(state_payload["action"], "stage.a07.start")
        self.assertEqual(state_payload["status"], "running")
        self.assertEqual(state_payload["source"], "runtime_inference")
        self.assertEqual(int(state_payload["stage_seq"]), int(stage_events[-1]["payload"]["stage_seq"]))

    def test_stage_a07_failed_live_reviewer_with_current_review_output_does_not_fail_stage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_development_paths(project_dir, "需求A")
            paths["task_json_path"].parent.mkdir(parents=True, exist_ok=True)
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _review_md_path, review_json_path = build_reviewer_artifact_paths(project_dir, "需求A", "测试工程师-天寿星")
            review_json_path.write_text(
                json.dumps([{"task_name": "M1-T1", "review_pass": True}], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            runtime_dir = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME / "worker-reviewer"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            (runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "development-review-测试工程师",
                        "session_name": "测试工程师-天寿星",
                        "work_dir": str(project_dir),
                        "project_dir": str(project_dir.resolve()),
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a07.start",
                        "status": "failed",
                        "result_status": "failed",
                        "agent_state": "READY",
                        "agent_started": True,
                        "health_status": "alive",
                        "current_turn_status_path": str(review_json_path),
                        "note": "error:development_review_init_M1-T1_测试工程师_round_1_repair_1",
                        "updated_at": "2026-04-24T12:00:00+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime.session_exists = lambda _session_name: True  # noqa: SLF001
            server._tmux_runtime.backend.session_exists = lambda _session_name: True  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="stage.a07.start")  # noqa: SLF001

            with patch("tmux_core.bridge.backend.load_worker_from_state_path", return_value=None):
                status = server._infer_runtime_stage_status("stage.a07.start")  # noqa: SLF001
                failed_summaries = server._failed_stage_worker_summaries("stage.a07.start")  # noqa: SLF001

        self.assertEqual(status, "running")
        self.assertEqual(failed_summaries, [])

    def test_stage_a07_runtime_state_change_keeps_failed_when_tasks_remain_but_session_is_gone(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_development_paths(project_dir, "需求A")
            paths["task_json_path"].parent.mkdir(parents=True, exist_ok=True)
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            runtime_dir = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME / "worker-reviewer"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            (runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "development-review-审核员",
                        "session_name": "审核员-天伤星",
                        "work_dir": str(project_dir),
                        "project_dir": str(project_dir.resolve()),
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a07.start",
                        "result_status": "succeeded",
                        "agent_state": "READY",
                        "agent_started": True,
                        "agent_alive": True,
                        "current_command": "codex",
                        "health_status": "alive",
                        "updated_at": "2026-04-24T12:00:00+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)
            server._tmux_runtime.session_exists = lambda _session_name: False  # noqa: SLF001
            server._tmux_runtime.backend.session_exists = lambda _session_name: False  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="stage.a07.start")  # noqa: SLF001
            server._display_action = "stage.a07.start"  # noqa: SLF001
            server._display_status = "failed"  # noqa: SLF001
            server._display_stage_seq = 7  # noqa: SLF001
            server._stage_seq_counter = 7  # noqa: SLF001

            with patch("tmux_core.bridge.backend.load_worker_from_state_path", return_value=None):
                server._bridge_ui.notify_runtime_state_changed()  # noqa: SLF001
                messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
                action, status, _stage_seq = server._derive_display_stage_state(  # noqa: SLF001
                    preferred_status="failed",
                    preferred_action="stage.a07.start",
                )

        stage_events = [item for item in messages if item.get("kind") == "event" and item.get("type") == "stage.changed"]
        self.assertEqual(action, "stage.a07.start")
        self.assertEqual(status, "failed")
        self.assertFalse(
            any(
                item.get("payload", {}).get("action") == "stage.a07.start"
                and item.get("payload", {}).get("status") == "running"
                for item in stage_events
            )
        )

    def test_stage_a08_status_does_not_treat_succeeded_reviewers_as_running_when_developer_dead(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            runtime_root = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME
            for worker_name, session_name, worker_id, status, agent_state, health_status in (
                ("worker-developer", "开发工程师-地默星", "development-developer", "running", "DEAD", "dead"),
                ("worker-reviewer", "审核员-地阖星", "development-review-审核员", "succeeded", "READY", "alive"),
            ):
                runtime_dir = runtime_root / worker_name
                runtime_dir.mkdir(parents=True, exist_ok=True)
                (runtime_dir / "worker.state.json").write_text(
                    json.dumps(
                        {
                            "worker_id": worker_id,
                            "session_name": session_name,
                            "work_dir": str(project_dir),
                            "project_dir": str(project_dir.resolve()),
                            "requirement_name": "需求A",
                            "workflow_action": "stage.a08.start",
                            "result_status": status,
                            "agent_state": agent_state,
                            "health_status": health_status,
                            "updated_at": "2026-04-24T12:00:00+08:00",
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime = SimpleNamespace(session_exists=lambda name: False)  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="stage.a08.start")  # noqa: SLF001

            status = server._infer_runtime_stage_status("stage.a08.start")  # noqa: SLF001

        self.assertEqual(status, "failed")

    def test_stage_a08_status_recovers_running_from_failed_when_review_state_not_passed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            runtime_root = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME
            runtime_dir = runtime_root / "worker-reviewer"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            (runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "development-review-审核员",
                        "session_name": "审核员-天伤星",
                        "work_dir": str(project_dir),
                        "project_dir": str(project_dir.resolve()),
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a08.start",
                        "result_status": "succeeded",
                        "agent_state": "READY",
                        "agent_started": True,
                        "agent_alive": True,
                        "current_command": "codex",
                        "health_status": "alive",
                        "updated_at": "2026-04-24T12:00:00+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime.session_exists = lambda _session_name: True  # noqa: SLF001
            server._tmux_runtime.backend.session_exists = lambda _session_name: True  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="stage.a08.start")  # noqa: SLF001
            server._display_status = "failed"  # noqa: SLF001

            with patch("tmux_core.bridge.backend.load_worker_from_state_path", return_value=None):
                action, status, _stage_seq = server._derive_display_stage_state(  # noqa: SLF001
                    preferred_status="failed",
                    preferred_action="stage.a08.start",
                )

        self.assertEqual(action, "stage.a08.start")
        self.assertEqual(status, "running")

    def test_stage_a08_completed_state_is_not_overridden_by_residual_live_worker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_overall_review_paths(project_dir, "需求A")
            paths["task_json_path"].parent.mkdir(parents=True, exist_ok=True)
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            paths["state_path"].write_text(
                json.dumps({"passed": True}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            runtime_dir = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME / "worker-reviewer"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            (runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "development-review-审核员",
                        "session_name": "审核员-地阖星",
                        "work_dir": str(project_dir),
                        "project_dir": str(project_dir.resolve()),
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a08.start",
                        "result_status": "succeeded",
                        "agent_state": "READY",
                        "agent_started": True,
                        "agent_alive": True,
                        "current_command": "codex",
                        "health_status": "alive",
                        "updated_at": "2026-04-24T12:00:00+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime.session_exists = lambda _session_name: True  # noqa: SLF001
            server._tmux_runtime.backend.session_exists = lambda _session_name: True  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="stage.a08.start")  # noqa: SLF001
            server._display_status = "completed"  # noqa: SLF001

            with patch("tmux_core.bridge.backend.load_worker_from_state_path", return_value=None):
                runtime_status = server._infer_runtime_stage_status("stage.a08.start")  # noqa: SLF001
                action, status, _stage_seq = server._derive_display_stage_state(  # noqa: SLF001
                    preferred_status="completed",
                    preferred_action="stage.a08.start",
                )

        self.assertEqual(runtime_status, "")
        self.assertEqual(action, "stage.a08.start")
        self.assertEqual(status, "completed")

    def test_failed_stage_worker_summaries_filter_to_current_requirement_and_action(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            runtime_root = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME
            for worker_name, requirement_name, workflow_action, session_name, note in (
                ("worker-current", "需求A", "stage.a07.start", "需求分析师-虚日鼠", "error:reqA"),
                ("worker-other-requirement", "需求B", "stage.a07.start", "审核员-地阖星", "error:reqB"),
                ("worker-other-action", "需求A", "stage.a08.start", "架构师-鬼金羊", "error:a08"),
            ):
                runtime_dir = runtime_root / worker_name
                runtime_dir.mkdir(parents=True, exist_ok=True)
                (runtime_dir / "worker.state.json").write_text(
                    json.dumps(
                        {
                            "worker_id": "development-review-测试",
                            "session_name": session_name,
                            "work_dir": str(project_dir),
                            "project_dir": str(project_dir.resolve()),
                            "requirement_name": requirement_name,
                            "workflow_action": workflow_action,
                            "result_status": "failed",
                            "health_status": "alive",
                            "note": note,
                            "updated_at": "2026-04-24T12:00:00+08:00",
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="stage.a07.start")  # noqa: SLF001

            summaries = server._failed_stage_worker_summaries("stage.a07.start")  # noqa: SLF001

        self.assertEqual(summaries, ["需求分析师-虚日鼠: error:reqA"])

    def test_stage_a07_status_treats_session_created_worker_as_running(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_development_paths(project_dir, "需求A")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            runtime_dir = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME / "worker-starting"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            (runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "development-developer",
                        "session_name": "开发工程师-地默星",
                        "work_dir": str(project_dir),
                        "result_status": "ready",
                        "workflow_stage": "pending",
                        "current_command": "zsh",
                        "agent_state": "STARTING",
                        "agent_started": False,
                        "health_status": "alive",
                        "note": "session_created",
                        "updated_at": "2026-04-22T10:00:00+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime.session_exists = lambda session_name: session_name == "开发工程师-地默星"  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="stage.a07.start")  # noqa: SLF001

            with patch("tmux_core.bridge.backend.load_worker_from_state_path", return_value=None):
                snapshot = server._build_development_snapshot()  # noqa: SLF001
                status = server._infer_runtime_stage_status("stage.a07.start")  # noqa: SLF001

        self.assertEqual(snapshot["workers"][0]["agent_state"], "STARTING")
        self.assertEqual(status, "running")

    def test_stage_a07_session_created_worker_with_missing_session_stays_starting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_development_paths(project_dir, "需求A")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            runtime_dir = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME / "worker-starting"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            (runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "development-review-测试工程师",
                        "session_name": "测试工程师-天暴星",
                        "work_dir": str(project_dir),
                        "project_dir": str(project_dir),
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a07.start",
                        "status": "ready",
                        "result_status": "running",
                        "workflow_stage": "pending",
                        "current_task_runtime_status": "running",
                        "pane_id": "%7",
                        "agent_state": "STARTING",
                        "agent_started": False,
                        "health_status": "alive",
                        "health_note": "alive",
                        "note": "session_created",
                        "updated_at": "2026-04-22T10:00:00+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime.session_exists = lambda _session_name: False  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="stage.a07.start")  # noqa: SLF001

            with patch("tmux_core.bridge.backend.load_worker_from_state_path", return_value=None):
                snapshot = server._build_development_snapshot()  # noqa: SLF001
                status = server._infer_runtime_stage_status("stage.a07.start")  # noqa: SLF001

        self.assertEqual(snapshot["workers"][0]["agent_state"], "STARTING")
        self.assertEqual(snapshot["workers"][0]["health_status"], "unknown")
        self.assertEqual(snapshot["workers"][0]["health_note"], "launch pending")
        self.assertEqual(status, "running")

    def test_stage_a07_metadata_only_worker_without_pane_is_starting_not_dead(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_development_paths(project_dir, "需求A")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            runtime_dir = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME / "worker-starting-no-pane"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            (runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "development-developer",
                        "session_name": "开发工程师-柳土獐",
                        "work_dir": str(project_dir),
                        "project_dir": str(project_dir),
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a07.start",
                        "status": "running",
                        "result_status": "running",
                        "workflow_stage": "pending",
                        "pane_id": "",
                        "agent_state": "STARTING",
                        "agent_started": False,
                        "health_status": "unknown",
                        "health_note": "launch pending",
                        "note": "awaiting_reconfig",
                        "updated_at": "2026-04-22T10:00:00+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime.session_exists = lambda _session_name: False  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="stage.a07.start")  # noqa: SLF001

            with patch("tmux_core.bridge.backend.load_worker_from_state_path", return_value=None):
                snapshot = server._build_development_snapshot()  # noqa: SLF001
                status = server._infer_runtime_stage_status("stage.a07.start")  # noqa: SLF001

        self.assertEqual(snapshot["workers"][0]["agent_state"], "STARTING")
        self.assertEqual(snapshot["workers"][0]["health_status"], "unknown")
        self.assertEqual(status, "running")

    def test_stage_snapshots_treat_prelaunch_dead_workers_as_starting_across_review_design_split_and_overall(self):
        cases = (
            ("stage.a04.start", REQUIREMENTS_REVIEW_RUNTIME_ROOT_NAME, "requirements-review-analyst", "_build_review_snapshot"),
            ("stage.a05.start", DETAILED_DESIGN_RUNTIME_ROOT_NAME, "detailed-design-analyst", "_build_design_snapshot"),
            ("stage.a06.start", TASK_SPLIT_RUNTIME_ROOT_NAME, "task-split-analyst", "_build_task_split_snapshot"),
            ("stage.a08.start", DEVELOPMENT_RUNTIME_ROOT_NAME, "development-review-审核员", "_build_overall_review_snapshot"),
        )
        for action, runtime_root_name, worker_id, builder_name in cases:
            with self.subTest(action=action), tempfile.TemporaryDirectory() as tmpdir:
                project_dir = Path(tmpdir)
                runtime_dir = project_dir / runtime_root_name / "worker-prelaunch"
                runtime_dir.mkdir(parents=True, exist_ok=True)
                (runtime_dir / "worker.state.json").write_text(
                    json.dumps(
                        {
                            "worker_id": worker_id,
                            "session_name": "预启动-虚日鼠",
                            "work_dir": str(project_dir),
                            "project_dir": str(project_dir),
                            "requirement_name": "需求A",
                            "workflow_action": action,
                            "status": "running",
                            "result_status": "running",
                            "workflow_stage": "pending",
                            "pane_id": "",
                            "agent_state": "DEAD",
                            "agent_started": False,
                            "health_status": "missing_session",
                            "health_note": "missing_session",
                            "note": "turn:init",
                            "updated_at": "2026-04-22T10:00:00+08:00",
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
                server._tmux_runtime.session_exists = lambda _session_name: False  # noqa: SLF001
                server._set_context(project_dir=str(project_dir), requirement_name="需求A", action=action)  # noqa: SLF001

                with patch("tmux_core.bridge.backend.load_worker_from_state_path", return_value=None):
                    snapshot = getattr(server, builder_name)()
                    status = server._infer_runtime_stage_status(action)  # noqa: SLF001

            self.assertEqual(snapshot["workers"][0]["agent_state"], "STARTING")
            self.assertEqual(snapshot["workers"][0]["health_status"], "unknown")
            self.assertNotEqual(status, "failed")

    def test_stage_a06_status_prefers_running_task_split_worker_over_failed_reviewers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            runtime_root = project_dir / DETAILED_DESIGN_RUNTIME_ROOT_NAME

            failed_runtime_dir = runtime_root / "worker-failed"
            failed_runtime_dir.mkdir(parents=True, exist_ok=True)
            (failed_runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "detailed-design-review-测试工程师",
                        "session_name": "测试工程师-井木犴",
                        "work_dir": str(project_dir),
                        "result_status": "failed",
                        "workflow_stage": "pending",
                        "agent_state": "DEAD",
                        "health_status": "alive",
                        "note": "error:task_split_review_init_测试工程师_round_1",
                        "current_turn_phase": "任务拆分",
                        "updated_at": "2026-04-21T12:25:20+08:00",
                        "last_heartbeat_at": "2026-04-21T12:27:31+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            running_runtime_dir = runtime_root / "worker-running"
            running_runtime_dir.mkdir(parents=True, exist_ok=True)
            (running_runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "detailed-design-review-架构师",
                        "session_name": "架构师-昴日鸡",
                        "work_dir": str(project_dir),
                        "result_status": "running",
                        "workflow_stage": "turn_running",
                        "agent_state": "BUSY",
                        "health_status": "alive",
                        "note": "turn:task_split_review_init_架构师_round_1",
                        "current_turn_phase": "任务拆分",
                        "updated_at": "2026-04-21T13:25:49+08:00",
                        "last_heartbeat_at": "2026-04-21T13:25:49+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime.session_exists = lambda session_name: session_name == "架构师-昴日鸡"  # noqa: SLF001
            server._tmux_runtime.backend.session_exists = lambda session_name: session_name == "架构师-昴日鸡"  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a06.start")  # noqa: SLF001
            with patch("tmux_core.bridge.backend.load_worker_from_state_path", return_value=None):
                status = server._infer_runtime_stage_status("stage.a06.start")  # noqa: SLF001

        self.assertEqual(status, "running")

    def test_task_split_snapshot_filters_workers_from_previous_workflow_start(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            record_dir = project_dir / ".tmux_workflow" / "需求A" / "stages"
            record_dir.mkdir(parents=True)
            (record_dir / "workflow_a00_start.state.json").write_text(
                json.dumps(
                    {
                        "action": "workflow.a00.start",
                        "status": "running",
                        "project_dir": str(project_dir),
                        "requirement_name": "需求A",
                        "stage_seq": 1,
                        "source": "runner_start",
                        "updated_at": "2026-05-03T09:00:00+08:00",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            runtime_root = project_dir / DETAILED_DESIGN_RUNTIME_ROOT_NAME
            older_root = runtime_root / "worker-old"
            newer_root = runtime_root / "worker-new"
            older_root.mkdir(parents=True)
            newer_root.mkdir(parents=True)
            common = {
                "worker_id": "detailed-design-review-测试工程师",
                "work_dir": str(project_dir),
                "project_dir": str(project_dir),
                "requirement_name": "需求A",
                "workflow_action": "stage.a06.start",
                "result_status": "succeeded",
                "workflow_stage": "pending",
                "agent_state": "READY",
                "health_status": "alive",
            }
            (older_root / "worker.state.json").write_text(
                json.dumps(
                    {
                        **common,
                        "session_name": "测试工程师-旧",
                        "note": "done:task_split_review_again_round_2",
                        "updated_at": "2026-05-02T21:27:03+08:00",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (newer_root / "worker.state.json").write_text(
                json.dumps(
                    {
                        **common,
                        "session_name": "测试工程师-新",
                        "result_status": "running",
                        "workflow_stage": "turn_running",
                        "agent_state": "BUSY",
                        "note": "turn:task_split_review_again_round_2",
                        "updated_at": "2026-05-03T09:27:03+08:00",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime = SimpleNamespace(session_exists=lambda name: True)  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="stage.a06.start")  # noqa: SLF001

            snapshot = server._build_task_split_snapshot()  # noqa: SLF001
            status = server._infer_runtime_stage_status("stage.a06.start")  # noqa: SLF001

        self.assertEqual([worker["session_name"] for worker in snapshot["workers"]], ["测试工程师-新"])
        self.assertEqual(status, "running")

    def test_stage_a06_status_ignores_failed_design_reviewer_outside_task_split_phase(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            runtime_root = project_dir / DETAILED_DESIGN_RUNTIME_ROOT_NAME / "worker-failed"
            runtime_root.mkdir(parents=True, exist_ok=True)
            (runtime_root / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "detailed-design-review-架构师",
                        "session_name": "架构师-箕水豹",
                        "work_dir": str(project_dir),
                        "result_status": "failed",
                        "workflow_stage": "pending",
                        "agent_state": "DEAD",
                        "health_status": "alive",
                        "note": "timeout:detailed_design_review_again_架构师_round_4",
                        "current_turn_phase": "详细设计",
                        "updated_at": "2026-04-21T12:55:37+08:00",
                        "last_heartbeat_at": "2026-04-21T12:55:37+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a06.start")  # noqa: SLF001

            workers = server._current_stage_workers("stage.a06.start")  # noqa: SLF001
            status = server._infer_runtime_stage_status("stage.a06.start")  # noqa: SLF001

        self.assertEqual(workers, [])
        self.assertEqual(status, "")

    def test_stage_snapshots_exclude_dead_unscoped_workers_when_context_known(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            design_runtime_dir = project_dir / DETAILED_DESIGN_RUNTIME_ROOT_NAME / "worker-stale-design"
            design_runtime_dir.mkdir(parents=True, exist_ok=True)
            (design_runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "detailed-design-review-审核员",
                        "session_name": "审核员-天暴星",
                        "work_dir": str(project_dir),
                        "result_status": "running",
                        "workflow_stage": "turn_running",
                        "agent_state": "READY",
                        "health_status": "alive",
                        "note": "turn:detailed_design_review_init_审核员_round_1",
                        "current_turn_status_path": str(project_dir / "missing_design_review.json"),
                        "updated_at": "2026-04-22T22:58:03+08:00",
                        "last_heartbeat_at": "2026-04-22T22:58:03+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            task_split_runtime_dir = project_dir / TASK_SPLIT_RUNTIME_ROOT_NAME / "worker-stale-task-split"
            task_split_runtime_dir.mkdir(parents=True, exist_ok=True)
            (task_split_runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "task-split-review-审核员",
                        "session_name": "审核员-天暴星",
                        "work_dir": str(project_dir),
                        "result_status": "running",
                        "workflow_stage": "turn_running",
                        "agent_state": "READY",
                        "health_status": "alive",
                        "note": "task_split_review_init_审核员_round_1",
                        "current_turn_phase": "任务拆分",
                        "current_turn_status_path": str(project_dir / "missing_task_split_review.json"),
                        "updated_at": "2026-04-22T22:58:04+08:00",
                        "last_heartbeat_at": "2026-04-22T22:58:04+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime = SimpleNamespace(session_exists=lambda name: False)  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a06.start")  # noqa: SLF001

            design = server._build_design_snapshot()  # noqa: SLF001
            task_split = server._build_task_split_snapshot()  # noqa: SLF001

        self.assertEqual(design["workers"], [])
        self.assertEqual(task_split["workers"], [])

    def test_stage_snapshots_prefer_scoped_workers_when_requirement_is_known(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            design_runtime_dir = project_dir / DETAILED_DESIGN_RUNTIME_ROOT_NAME / "worker-live-design"
            design_runtime_dir.mkdir(parents=True, exist_ok=True)
            (design_runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "detailed-design-review-审核员",
                        "session_name": "审核员-天暴星",
                        "work_dir": str(project_dir),
                        "result_status": "running",
                        "workflow_stage": "turn_running",
                        "agent_state": "READY",
                        "health_status": "alive",
                        "note": "turn:detailed_design_review_init_审核员_round_1",
                        "current_turn_status_path": str(project_dir / "missing_design_review.json"),
                        "updated_at": "2026-04-22T22:58:03+08:00",
                        "last_heartbeat_at": "2026-04-22T22:58:03+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            scoped_runtime_dir = project_dir / DETAILED_DESIGN_RUNTIME_ROOT_NAME / "worker-scoped-design"
            scoped_runtime_dir.mkdir(parents=True, exist_ok=True)
            (scoped_runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "detailed-design-review-审核员",
                        "session_name": "审核员-天寿星",
                        "work_dir": str(project_dir),
                        "project_dir": str(project_dir.resolve()),
                        "requirement_name": "贪吃蛇",
                        "workflow_action": "stage.a05.start",
                        "result_status": "running",
                        "workflow_stage": "turn_running",
                        "agent_state": "READY",
                        "health_status": "alive",
                        "note": "turn:detailed_design_review_init_审核员_round_1",
                        "updated_at": "2026-04-22T22:58:04+08:00",
                        "last_heartbeat_at": "2026-04-22T22:58:04+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime = SimpleNamespace(session_exists=lambda name: name in {"审核员-天暴星", "审核员-天寿星"})  # noqa: SLF001
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a05.start")  # noqa: SLF001

            design = server._build_design_snapshot()  # noqa: SLF001

        self.assertEqual([worker["session_name"] for worker in design["workers"]], ["审核员-天寿星"])

    def test_worker_context_filter_separates_requirements_in_same_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = str(Path(tmpdir).resolve())
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=project_dir, requirement_name="需求A", action="stage.a07.start")  # noqa: SLF001

            workers = server._filter_workers_for_current_context(  # noqa: SLF001
                [
                    {
                        "session_name": "开发工程师-A",
                        "project_dir": project_dir,
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a07.start",
                    },
                    {
                        "session_name": "开发工程师-B",
                        "project_dir": project_dir,
                        "requirement_name": "需求B",
                        "workflow_action": "stage.a07.start",
                    },
                    {
                        "session_name": "开发工程师-未标记需求",
                        "project_dir": project_dir,
                        "workflow_action": "stage.a07.start",
                    },
                ],
                "stage.a07.start",
            )

        self.assertEqual([worker["session_name"] for worker in workers], ["开发工程师-A"])

    def test_worker_context_filter_does_not_fallback_to_unscoped_when_other_requirement_is_scoped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = str(Path(tmpdir).resolve())
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=project_dir, requirement_name="需求C", action="stage.a07.start")  # noqa: SLF001

            workers = server._filter_workers_for_current_context(  # noqa: SLF001
                [
                    {
                        "session_name": "开发工程师-B",
                        "project_dir": project_dir,
                        "requirement_name": "需求B",
                        "workflow_action": "stage.a07.start",
                    },
                    {
                        "session_name": "开发工程师-未标记需求",
                        "project_dir": project_dir,
                        "workflow_action": "stage.a07.start",
                    },
                ],
                "stage.a07.start",
            )

        self.assertEqual(workers, [])

    def test_worker_context_filter_does_not_leak_same_requirement_from_other_action(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = str(Path(tmpdir).resolve())
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=project_dir, requirement_name="需求A", action="stage.a07.start")  # noqa: SLF001

            workers = server._filter_workers_for_current_context(  # noqa: SLF001
                [
                    {
                        "session_name": "开发工程师-A-复核",
                        "project_dir": project_dir,
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a08.start",
                    }
                ],
                "stage.a07.start",
            )

        self.assertEqual(workers, [])

    def test_worker_context_filter_keeps_legacy_unscoped_worker_when_no_scoped_metadata_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = str(Path(tmpdir).resolve())
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=project_dir, requirement_name="需求A", action="stage.a07.start")  # noqa: SLF001

            workers = server._filter_workers_for_current_context(  # noqa: SLF001
                [
                    {
                        "session_name": "开发工程师-旧运行态",
                        "project_dir": project_dir,
                    }
                ],
                "stage.a07.start",
            )

        self.assertEqual([worker["session_name"] for worker in workers], ["开发工程师-旧运行态"])

    def test_hitl_prompt_open_emits_question_into_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            question_path = Path(tmpdir) / "question.md"
            question_path.write_text("- [阻断] 需要补充碰撞规则\n", encoding="utf-8")
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)

            server._handle_prompt_open(  # noqa: SLF001
                "prompt_hitl",
                BridgePromptRequest(
                    prompt_type="multiline",
                    payload={
                        "title": "HITL 第 1 轮回复",
                        "question_path": str(question_path),
                        "answer_path": str(Path(tmpdir) / "answer.md"),
                    },
                ),
            )

            messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
            log_events = [item for item in messages if item.get("kind") == "event" and item.get("type") == "log.append"]
        self.assertTrue(log_events)
        self.assertTrue(any("HITL 问题文档" in str(item["payload"].get("text", "")) for item in log_events))
        self.assertTrue(any("需要补充碰撞规则" in str(item["payload"].get("text", "")) for item in log_events))
        self.assertTrue(any(item["payload"].get("log_kind") == "hitl" for item in log_events))
        self.assertTrue(any(item["payload"].get("hitl_round") == 1 for item in log_events))
        self.assertTrue(any(item["payload"].get("log_title") == "HITL 第 1 轮" for item in log_events))

    def test_prompt_response_backfills_project_dir_into_app_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)
            server._pending_prompt = PendingPromptState(  # noqa: SLF001
                prompt_id="prompt_project",
                prompt_type="text",
                payload={"prompt_text": "输入项目工作目录"},
            )
            server._prompt_broker._pending["prompt_project"] = queue.Queue(maxsize=1)  # noqa: SLF001
            server.handle_request(
                build_request(
                    "prompt.response",
                    {"prompt_id": "prompt_project", "value": tmpdir},
                    message_id="req_project",
                )
            )
            app = server._build_app_snapshot()  # noqa: SLF001
        self.assertEqual(app["project_dir"], str(Path(tmpdir).resolve()))

    def test_project_dir_prompt_clears_stale_requirement_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir="/tmp/old-project", requirement_name="需求A", action="stage.a04.start")  # noqa: SLF001
            server._pending_prompt = PendingPromptState(  # noqa: SLF001
                prompt_id="prompt_project",
                prompt_type="text",
                payload={"prompt_text": "输入项目工作目录"},
            )
            server._prompt_broker._pending["prompt_project"] = queue.Queue(maxsize=1)  # noqa: SLF001
            server.handle_request(
                build_request(
                    "prompt.response",
                    {"prompt_id": "prompt_project", "value": tmpdir},
                    message_id="req_project",
                )
            )
            app = server._build_app_snapshot()  # noqa: SLF001
        self.assertEqual(app["project_dir"], str(Path(tmpdir).resolve()))
        self.assertEqual(app["requirement_name"], "")

    def test_prompt_response_backfills_requirement_name_from_text_and_select_prompts(self):
        server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
        server._pending_prompt = PendingPromptState(  # noqa: SLF001
            prompt_id="prompt_requirement",
            prompt_type="text",
            payload={"prompt_text": "输入需求名称"},
        )
        server._prompt_broker._pending["prompt_requirement"] = queue.Queue(maxsize=1)  # noqa: SLF001
        server.handle_request(
            build_request(
                "prompt.response",
                {"prompt_id": "prompt_requirement", "value": "贪吃蛇"},
                message_id="req_requirement",
            )
        )
        self.assertEqual(server._build_app_snapshot()["requirement_name"], "贪吃蛇")  # noqa: SLF001

        server._pending_prompt = PendingPromptState(  # noqa: SLF001
            prompt_id="prompt_existing_requirement",
            prompt_type="select",
            payload={
                "prompt_text": "选择已有需求或创建新需求",
                "options": [
                    {"value": "需求A", "label": "需求A"},
                    {"value": "__create_new__", "label": "创建新需求"},
                ],
            },
        )
        server._prompt_broker._pending["prompt_existing_requirement"] = queue.Queue(maxsize=1)  # noqa: SLF001
        server.handle_request(
            build_request(
                "prompt.response",
                {"prompt_id": "prompt_existing_requirement", "value": "需求A"},
                message_id="req_existing_requirement",
            )
        )
        self.assertEqual(server._build_app_snapshot()["requirement_name"], "需求A")  # noqa: SLF001

    def test_prompt_response_ignores_invalid_existing_requirement_placeholder_value(self):
        server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
        server._pending_prompt = PendingPromptState(  # noqa: SLF001
            prompt_id="prompt_existing_requirement",
            prompt_type="select",
            payload={
                "prompt_text": "选择已有需求或创建新需求",
                "options": [
                    {"value": "需求A", "label": "需求A"},
                    {"value": "__create_new__", "label": "创建新需求"},
                ],
            },
        )
        server._prompt_broker._pending["prompt_existing_requirement"] = queue.Queue(maxsize=1)  # noqa: SLF001
        server.handle_request(
            build_request(
                "prompt.response",
                {"prompt_id": "prompt_existing_requirement", "value": "现有需求"},
                message_id="req_existing_requirement_invalid",
            )
        )
        self.assertEqual(server._build_app_snapshot()["requirement_name"], "")  # noqa: SLF001

    def test_handle_bootstrap_writes_response(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        server.handle_request(build_request("app.bootstrap", {}, message_id="req_1"))
        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        self.assertEqual(messages[0]["kind"], "response")
        self.assertEqual(messages[0]["id"], "req_1")
        self.assertIn("routes", messages[0]["payload"])
        self.assertIn("design", messages[0]["payload"]["routes"])
        self.assertIn("task-split", messages[0]["payload"]["routes"])
        self.assertIn("development", messages[0]["payload"]["routes"])
        self.assertIn("overall-review", messages[0]["payload"]["routes"])
        self.assertIn("stage.a08.start", messages[0]["payload"]["commands"])
        self.assertIn("capabilities", messages[0]["payload"])
        self.assertIn("snapshots", messages[0]["payload"])
        self.assertEqual(len(messages), 1)

    def test_stage_a05_start_runs_in_background(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        with patch("T11_tui_backend.run_detailed_design_stage", return_value=SimpleNamespace(project_dir="/tmp/project", requirement_name="需求A", passed=True)):
            server.handle_request(
                build_request(
                    "stage.a05.start",
                    {"argv": ["--project-dir", "/tmp/project", "--requirement-name", "需求A"]},
                    message_id="req_a05",
                )
            )
            for worker in list(server._workers.values()):  # noqa: SLF001
                worker.join(timeout=2.0)
            server._flush_dirty_snapshots()  # noqa: SLF001
        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        self.assertTrue(any(item.get("kind") == "response" and item.get("id") == "req_a05" for item in messages))

    def test_stage_a05_start_logs_error_when_stage_fails(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        with patch("T11_tui_backend.run_detailed_design_stage", side_effect=RuntimeError("design failed")):
            server.handle_request(
                build_request(
                    "stage.a05.start",
                    {"argv": ["--project-dir", "/tmp/project", "--requirement-name", "需求A"]},
                    message_id="req_a05_failed",
                )
            )
            for worker in list(server._workers.values()):  # noqa: SLF001
                worker.join(timeout=2.0)
            server._flush_dirty_snapshots()  # noqa: SLF001
        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        log_events = [item for item in messages if item.get("kind") == "event" and item.get("type") == "log.append"]
        self.assertTrue(any("design failed" in str(item.get("payload", {}).get("text", "")) for item in log_events))
        responses = [item for item in messages if item.get("kind") == "response" and item.get("id") == "req_a05_failed"]
        self.assertTrue(responses)
        self.assertFalse(responses[-1]["ok"])
        stage_events = [item for item in messages if item.get("kind") == "event" and item.get("type") == "stage.changed"]
        self.assertTrue(stage_events)
        self.assertEqual(stage_events[-1]["payload"]["status"], "failed")
        self.assertGreater(int(stage_events[-1]["payload"]["stage_seq"]), 0)

    def test_bridge_progress_events_follow_current_stage_sequence(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)

        server._bridge_ui.notify_stage_action_changed("stage.a05.start")  # noqa: SLF001
        monitor = server._bridge_ui.create_progress_monitor(frame_builder=lambda _tick: "running")  # noqa: SLF001
        monitor.start()
        monitor.stop()

        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        stage_events = [item for item in messages if item.get("kind") == "event" and item.get("type") == "stage.changed"]
        progress_events = [item for item in messages if item.get("kind") == "event" and str(item.get("type", "")).startswith("progress.")]
        self.assertTrue(stage_events)
        self.assertTrue(progress_events)
        stage_seq = int(stage_events[-1]["payload"]["stage_seq"])
        self.assertGreater(stage_seq, 0)
        self.assertTrue(all(item["payload"]["action"] == "stage.a05.start" for item in progress_events))
        self.assertTrue(all(int(item["payload"]["stage_seq"]) == stage_seq for item in progress_events))

    def test_stage_a05_start_success_survives_snapshot_emit_failure(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        with patch("T11_tui_backend.run_detailed_design_stage", return_value=SimpleNamespace(project_dir="/tmp/project", requirement_name="需求A", passed=True)), patch.object(
            server,
            "_build_app_snapshot",
            side_effect=RuntimeError("snapshot broken"),
        ):
            server.handle_request(
                build_request(
                    "stage.a05.start",
                    {"argv": ["--project-dir", "/tmp/project", "--requirement-name", "需求A"]},
                    message_id="req_a05_snapshot",
                )
            )
            for worker in list(server._workers.values()):  # noqa: SLF001
                worker.join(timeout=2.0)
            server._flush_dirty_snapshots()  # noqa: SLF001
        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        responses = [item for item in messages if item.get("kind") == "response" and item.get("id") == "req_a05_snapshot"]
        self.assertTrue(responses)
        self.assertTrue(responses[-1]["ok"])
        stage_events = [item for item in messages if item.get("kind") == "event" and item.get("type") == "stage.changed"]
        self.assertEqual(stage_events[-1]["payload"]["status"], "completed")
        log_events = [item for item in messages if item.get("kind") == "event" and item.get("type") == "log.append"]
        self.assertTrue(any("snapshot broken" in str(item.get("payload", {}).get("text", "")) for item in log_events))

    def test_stage_a06_start_runs_in_background(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        with patch("T11_tui_backend.run_task_split_stage", return_value=SimpleNamespace(project_dir="/tmp/project", requirement_name="需求A", passed=True)):
            server.handle_request(
                build_request(
                    "stage.a06.start",
                    {"argv": ["--project-dir", "/tmp/project", "--requirement-name", "需求A"]},
                    message_id="req_a06",
                )
            )
            for worker in list(server._workers.values()):  # noqa: SLF001
                worker.join(timeout=2.0)
        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        self.assertTrue(any(item.get("kind") == "response" and item.get("id") == "req_a06" for item in messages))

    def test_stage_a07_start_runs_in_background(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_development_paths(project_dir, "需求A")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)

            def fake_run_development_stage(argv, preserve_workers=False):  # noqa: ANN001
                self.assertTrue(preserve_workers)
                return SimpleNamespace(project_dir=str(project_dir), requirement_name="需求A", completed=True)

            with patch("T11_tui_backend.run_development_stage", side_effect=fake_run_development_stage):
                server.handle_request(
                    build_request(
                        "stage.a07.start",
                        {"argv": ["--project-dir", str(project_dir), "--requirement-name", "需求A"]},
                        message_id="req_a07",
                    )
                )
                for worker in list(server._workers.values()):  # noqa: SLF001
                    worker.join(timeout=2.0)
        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        self.assertTrue(any(item.get("kind") == "response" and item.get("id") == "req_a07" and item.get("ok") for item in messages))

    def test_stage_a08_start_runs_in_background(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_overall_review_paths(project_dir, "需求A")
            for file_path, content in (
                (paths["original_requirement_path"], "原始需求\n"),
                (paths["requirements_clear_path"], "需求澄清\n"),
                (paths["task_md_path"], "任务单\n"),
                (paths["task_json_path"], json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False, indent=2)),
                (paths["detailed_design_path"], "详细设计\n"),
                (paths["state_path"], json.dumps({"passed": True}, ensure_ascii=False, indent=2)),
            ):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)
            with patch("T11_tui_backend.run_overall_review_stage", return_value=SimpleNamespace(project_dir=str(project_dir), requirement_name="需求A", completed=True)):
                server.handle_request(
                    build_request(
                        "stage.a08.start",
                        {"argv": ["--project-dir", str(project_dir), "--requirement-name", "需求A"]},
                        message_id="req_a08",
                    )
                )
                for worker in list(server._workers.values()):  # noqa: SLF001
                    worker.join(timeout=2.0)
        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        self.assertTrue(any(item.get("kind") == "response" and item.get("id") == "req_a08" and item.get("ok") for item in messages))

    def test_stage_a08_start_rejects_completed_result_when_review_state_not_passed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_overall_review_paths(project_dir, "需求A")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)
            with patch("T11_tui_backend.run_overall_review_stage", return_value=SimpleNamespace(project_dir=str(project_dir), requirement_name="需求A", completed=True)):
                server.handle_request(
                    build_request(
                        "stage.a08.start",
                        {"argv": ["--project-dir", str(project_dir), "--requirement-name", "需求A"]},
                        message_id="req_a08_invalid_completed",
                    )
                )
                for worker in list(server._workers.values()):  # noqa: SLF001
                    worker.join(timeout=2.0)
            messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
            failure_path = project_dir / ".tmux_workflow" / "需求A" / "stages" / "stage_a08_start.failure.json"
            failure_exists = failure_path.exists()
            failure_payload = json.loads(failure_path.read_text(encoding="utf-8")) if failure_exists else {}
            state_path = project_dir / ".tmux_workflow" / "需求A" / "stages" / "stage_a08_start.state.json"
            state_exists = state_path.exists()
            state_payload = json.loads(state_path.read_text(encoding="utf-8")) if state_exists else {}

        responses = [item for item in messages if item.get("kind") == "response" and item.get("id") == "req_a08_invalid_completed"]
        self.assertTrue(responses)
        self.assertFalse(responses[-1]["ok"])
        self.assertIn("复核阶段返回 completed，但复核完成状态未通过", responses[-1]["error"])
        self.assertTrue(failure_exists)
        self.assertEqual(failure_payload["action"], "stage.a08.start")
        self.assertIn("复核完成状态未通过", failure_payload["error"])
        self.assertTrue(state_exists)
        self.assertEqual(state_payload["action"], "stage.a08.start")
        self.assertEqual(state_payload["status"], "failed")
        self.assertEqual(state_payload["source"], "runner_failure")
        self.assertEqual(state_payload["failure_path"], str(failure_path.resolve()))
        self.assertIn("复核完成状态未通过", state_payload["message"])

    def test_stage_a07_runtime_failure_overrides_stale_completed_display_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            runtime_root = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME / "worker-failed"
            runtime_root.mkdir(parents=True, exist_ok=True)
            (runtime_root / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "development-review-需求分析师",
                        "session_name": "需求分析师-虚日鼠",
                        "work_dir": str(project_dir),
                        "result_status": "failed",
                        "workflow_stage": "pending",
                        "agent_state": "DEAD",
                        "health_status": "alive",
                        "note": "error:development_reviewer_init_需求分析师",
                        "updated_at": "2026-04-21T16:52:24+08:00",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="stage.a07.start")  # noqa: SLF001
            server._display_status = "completed"  # noqa: SLF001

            server._bridge_ui.notify_runtime_state_changed()  # noqa: SLF001
            server._flush_dirty_snapshots()  # noqa: SLF001
            messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]

        stage_events = [item for item in messages if item.get("kind") == "event" and item.get("type") == "stage.changed"]
        self.assertTrue(stage_events)
        self.assertEqual(stage_events[-1]["payload"]["action"], "stage.a07.start")
        self.assertEqual(stage_events[-1]["payload"]["status"], "failed")

    def test_stage_a07_recoverable_worker_failure_does_not_flash_failed_while_runner_alive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_development_paths(project_dir, "需求A")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            runtime_root = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME / "worker-failed"
            runtime_root.mkdir(parents=True, exist_ok=True)
            (runtime_root / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "development-review-需求分析师",
                        "session_name": "需求分析师-虚日鼠",
                        "work_dir": str(project_dir),
                        "project_dir": str(project_dir.resolve()),
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a07.start",
                        "result_status": "failed",
                        "status": "failed",
                        "workflow_stage": "pending",
                        "agent_state": "DEAD",
                        "health_status": "alive",
                        "note": "error:development_review_init_M1-T1_R1_round_1",
                        "updated_at": "2026-04-21T16:52:24+08:00",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="stage.a07.start")  # noqa: SLF001
            server._display_action = "stage.a07.start"  # noqa: SLF001
            server._display_status = "running"  # noqa: SLF001
            server._display_stage_seq = 7  # noqa: SLF001
            server._workers["req_a07"] = SimpleNamespace(  # noqa: SLF001
                name="tui-backend-stage.a07.start-req_a07",
                is_alive=lambda: True,
            )

            with patch("tmux_core.bridge.backend.load_worker_from_state_path", return_value=None):
                action, status, _stage_seq = server._derive_display_stage_state()  # noqa: SLF001
                server._bridge_ui.notify_runtime_state_changed()  # noqa: SLF001
                messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]

        stage_events = [item for item in messages if item.get("kind") == "event" and item.get("type") == "stage.changed"]
        self.assertEqual(action, "stage.a07.start")
        self.assertEqual(status, "running")
        self.assertFalse(
            any(
                item.get("payload", {}).get("action") == "stage.a07.start"
                and item.get("payload", {}).get("status") == "failed"
                for item in stage_events
            )
        )

    def test_stage_a07_runtime_state_change_does_not_emit_failed_for_prelaunch_worker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_development_paths(project_dir, "需求A")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            runtime_dir = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME / "worker-starting"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            (runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "development-developer",
                        "session_name": "开发工程师-地默星",
                        "work_dir": str(project_dir),
                        "result_status": "ready",
                        "workflow_stage": "pending",
                        "current_command": "zsh",
                        "agent_state": "STARTING",
                        "agent_started": False,
                        "health_status": "alive",
                        "note": "session_created",
                        "updated_at": "2026-04-22T16:52:24+08:00",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="stage.a07.start")  # noqa: SLF001
            server._display_status = "running"  # noqa: SLF001
            server._tmux_runtime.session_exists = lambda session_name: session_name == "开发工程师-地默星"  # noqa: SLF001

            with patch("tmux_core.bridge.backend.load_worker_from_state_path", return_value=None):
                server._bridge_ui.notify_runtime_state_changed()  # noqa: SLF001
                messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]

        stage_events = [item for item in messages if item.get("kind") == "event" and item.get("type") == "stage.changed"]
        self.assertFalse(
            any(
                item.get("payload", {}).get("action") == "stage.a07.start"
                and item.get("payload", {}).get("status") == "failed"
                for item in stage_events
            )
        )

    def test_refresh_running_worker_snapshot_rechecks_busy_session_even_after_succeeded_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            runtime_dir = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME / "worker-reviewer"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            state_path = runtime_dir / "worker.state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "worker_id": "development-review-审核员",
                        "session_name": "审核员-天伤星",
                        "work_dir": str(project_dir),
                        "result_status": "succeeded",
                        "status": "ready",
                        "agent_state": "BUSY",
                        "agent_started": True,
                        "agent_alive": True,
                        "current_command": "codex",
                        "health_status": "alive",
                        "updated_at": "2026-04-24T12:00:00+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            class _FakeWorker:
                def refresh_health(self, *, notify_on_change: bool = False) -> None:  # noqa: ARG002
                    state_path.write_text(
                        json.dumps(
                            {
                                "worker_id": "development-review-审核员",
                                "session_name": "审核员-天伤星",
                                "work_dir": str(project_dir),
                                "result_status": "succeeded",
                                "status": "ready",
                                "agent_state": "READY",
                                "agent_started": True,
                                "agent_alive": True,
                                "current_command": "codex",
                                "health_status": "alive",
                                "updated_at": "2026-04-24T12:01:00+08:00",
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime.session_exists = lambda session_name: session_name == "审核员-天伤星"  # noqa: SLF001
            server._tmux_runtime.backend.session_exists = lambda session_name: session_name == "审核员-天伤星"  # noqa: SLF001

            with patch("tmux_core.bridge.backend.load_worker_from_state_path", return_value=_FakeWorker()):
                snapshot = server._refresh_running_worker_snapshot_if_needed(state_path)  # noqa: SLF001

        self.assertEqual(snapshot["agent_state"], "READY")

    def test_stage_a07_start_rejects_completed_result_when_task_json_still_has_false_and_logs_failed_worker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_development_paths(project_dir, "需求A")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            runtime_dir = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME / "worker-failed"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            (runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "development-review-需求分析师",
                        "session_name": "需求分析师-虚日鼠",
                        "work_dir": str(project_dir),
                        "result_status": "failed",
                        "workflow_stage": "pending",
                        "agent_state": "DEAD",
                        "health_status": "alive",
                        "note": "error:development_reviewer_init_需求分析师",
                        "updated_at": "2026-04-21T16:52:24+08:00",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)
            with patch("tmux_core.bridge.backend.run_development_stage", return_value=SimpleNamespace(project_dir=str(project_dir), requirement_name="需求A", completed=True)):
                server.handle_request(
                    build_request(
                        "stage.a07.start",
                        {"argv": ["--project-dir", str(project_dir), "--requirement-name", "需求A"]},
                        message_id="req_a07_invalid_completed",
                    )
                )
                for worker in list(server._workers.values()):  # noqa: SLF001
                    worker.join(timeout=5.0)
            messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
            failure_path = project_dir / ".tmux_workflow" / "需求A" / "stages" / "stage_a07_start.failure.json"
            failure_exists = failure_path.exists()
            failure_payload = json.loads(failure_path.read_text(encoding="utf-8")) if failure_exists else {}
            state_path = project_dir / ".tmux_workflow" / "需求A" / "stages" / "stage_a07_start.state.json"
            state_exists = state_path.exists()
            state_payload = json.loads(state_path.read_text(encoding="utf-8")) if state_exists else {}

        responses = [item for item in messages if item.get("kind") == "response" and item.get("id") == "req_a07_invalid_completed"]
        self.assertTrue(responses)
        self.assertFalse(responses[-1]["ok"])
        self.assertIn("任务开发阶段返回 completed，但任务单 JSON 仍存在未完成任务: M1-T1", responses[-1]["error"])
        self.assertIn("需求分析师-虚日鼠: error:development_reviewer_init_需求分析师", responses[-1]["error"])
        self.assertTrue(failure_exists)
        self.assertEqual(failure_payload["action"], "stage.a07.start")
        self.assertIn("任务单 JSON 仍存在未完成任务", failure_payload["error"])
        self.assertTrue(state_exists)
        self.assertEqual(state_payload["action"], "stage.a07.start")
        self.assertEqual(state_payload["status"], "failed")
        self.assertEqual(state_payload["source"], "runner_failure")
        self.assertEqual(state_payload["failure_path"], str(failure_path.resolve()))
        self.assertIn("任务单 JSON 仍存在未完成任务", state_payload["message"])
        log_texts = [
            str(item.get("payload", {}).get("text", ""))
            for item in messages
            if item.get("kind") == "event" and item.get("type") == "log.append"
        ]
        self.assertTrue(any("需求分析师-虚日鼠: error:development_reviewer_init_需求分析师" in text for text in log_texts))
        stage_events = [item for item in messages if item.get("kind") == "event" and item.get("type") == "stage.changed"]
        self.assertTrue(stage_events)
        self.assertEqual(stage_events[-1]["payload"]["status"], "failed")

    def test_stage_a07_workers_scan_requirement_scoped_runtime_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            runtime_dir = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME / "需求A" / "development-developer-aaaa"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            (runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "development-developer",
                        "session_name": "开发工程师-天速星",
                        "work_dir": str(project_dir),
                        "status": "ready",
                        "agent_state": "READY",
                        "health_status": "alive",
                        "project_dir": str(project_dir.resolve()),
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a07.start",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="stage.a07.start")  # noqa: SLF001

            workers = server._scan_current_development_workers(str(project_dir))  # noqa: SLF001

        self.assertEqual(len(workers), 1)
        self.assertEqual(workers[0]["session_name"], "开发工程师-天速星")

    def test_prompt_broker_ignores_duplicate_resolution_when_queue_is_full(self):
        broker = PromptBroker(lambda *_args, **_kwargs: None)
        prompt_queue: queue.Queue[dict[str, object]] = queue.Queue(maxsize=1)
        prompt_queue.put({"value": "first"})
        broker._pending["prompt_1"] = prompt_queue  # noqa: SLF001
        broker.resolve("prompt_1", {"value": "second"})
        self.assertEqual(prompt_queue.get_nowait()["value"], "first")

    def test_prompt_broker_unblocks_waiter_before_resolved_callback_finishes(self):
        prompt_opened = threading.Event()
        callback_entered = threading.Event()
        release_callback = threading.Event()
        received: queue.Queue[dict[str, object]] = queue.Queue(maxsize=1)

        def on_prompt_resolved(_prompt_id, _payload):  # noqa: ANN001
            callback_entered.set()
            release_callback.wait(timeout=2.0)

        broker = PromptBroker(
            lambda *_args, **_kwargs: prompt_opened.set(),
            on_prompt_resolved=on_prompt_resolved,
        )

        def request_prompt() -> None:
            received.put(
                broker.request(
                    BridgePromptRequest(
                        prompt_type="select",
                        payload={"title": "HITL", "default_value": "recheck"},
                    )
                )
            )

        requester = threading.Thread(target=request_prompt)
        requester.start()
        self.assertTrue(prompt_opened.wait(timeout=1.0))
        prompt_id = next(iter(broker._pending))  # noqa: SLF001
        resolver = threading.Thread(target=lambda: broker.resolve(prompt_id, {"value": "recheck"}))
        resolver.start()
        try:
            self.assertTrue(callback_entered.wait(timeout=1.0))
            self.assertEqual(received.get(timeout=1.0)["value"], "recheck")
        finally:
            release_callback.set()
            requester.join(timeout=2.0)
            resolver.join(timeout=2.0)

    def test_prompt_resolved_schedules_lightweight_snapshot_after_unblocking(self):
        server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
        server._pending_prompts["prompt_1"] = PendingPromptState(  # noqa: SLF001
            prompt_id="prompt_1",
            prompt_type="select",
            payload={"title": "HITL: 架构师 需要人工介入", "is_hitl": True},
        )
        server._pending_prompt = server._pending_prompts["prompt_1"]  # noqa: SLF001

        with patch.object(server, "_emit_snapshot_update") as emit_snapshot, patch.object(
            server,
            "_schedule_snapshot_update",
        ) as schedule_snapshot:
            server._handle_prompt_resolved("prompt_1", {"value": "recheck_after_manual_intervention"})  # noqa: SLF001

        emit_snapshot.assert_not_called()
        schedule_snapshot.assert_called_once()
        self.assertEqual(schedule_snapshot.call_args.kwargs["sections"], {"app", "hitl"})
        self.assertFalse(schedule_snapshot.call_args.kwargs["refresh_worker_health"])

    def test_prompt_open_schedules_lightweight_snapshot_without_sync_emit(self):
        server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
        request = BridgePromptRequest(
            prompt_type="select",
            payload={"title": "HITL: 架构师 需要人工介入", "is_hitl": True},
        )

        with patch.object(server, "_emit_snapshot_update", side_effect=AssertionError("sync snapshot should not run")) as emit_snapshot, patch.object(
            server,
            "_schedule_snapshot_update",
        ) as schedule_snapshot:
            server._handle_prompt_open("prompt_1", request)  # noqa: SLF001

        emit_snapshot.assert_not_called()
        schedule_snapshot.assert_called_once()
        self.assertEqual(schedule_snapshot.call_args.kwargs["sections"], {"app", "hitl"})
        self.assertFalse(schedule_snapshot.call_args.kwargs["refresh_worker_health"])

    def test_workflow_a00_start_runs_in_background(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        release = threading.Event()

        def delayed_main(_argv):  # noqa: ANN001
            release.wait(timeout=2.0)
            return 0

        with patch("T11_tui_backend.a00_main", side_effect=delayed_main):
            server.handle_request(build_request("workflow.a00.start", {"argv": []}, message_id="req_2"))
            workers = list(server._workers.values())  # noqa: SLF001
            try:
                self.assertTrue(workers)
                self.assertFalse(workers[0].daemon)
            finally:
                release.set()
                for worker in workers:
                    worker.join(timeout=2.0)
        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        self.assertTrue(any(item.get("kind") == "event" and item.get("type") == "stage.changed" for item in messages))
        self.assertTrue(any(item.get("kind") == "response" and item.get("id") == "req_2" for item in messages))

    def test_workflow_a00_start_deduplicates_while_runner_active(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        started = threading.Event()
        release = threading.Event()
        calls: list[list[str]] = []

        def delayed_main(argv):  # noqa: ANN001
            calls.append(list(argv))
            started.set()
            release.wait(timeout=2.0)
            return 0

        with patch("T11_tui_backend.a00_main", side_effect=delayed_main):
            server.handle_request(build_request("workflow.a00.start", {"argv": []}, message_id="req_1"))
            self.assertTrue(started.wait(timeout=2.0))
            server.handle_request(build_request("workflow.a00.start", {"argv": []}, message_id="req_2"))
            release.set()
            for worker in list(server._workers.values()):  # noqa: SLF001
                worker.join(timeout=2.0)

        self.assertEqual(len(calls), 1)
        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        duplicate_responses = [item for item in messages if item.get("kind") == "response" and item.get("id") == "req_2"]
        self.assertTrue(duplicate_responses)
        self.assertFalse(duplicate_responses[-1]["ok"])
        self.assertIn("已有同一任务在运行", duplicate_responses[-1]["error"])
        self.assertTrue(duplicate_responses[-1]["payload"]["already_running"])
        self.assertFalse(any(item.get("kind") == "event" and item.get("type") == "error" for item in messages))

    def test_workflow_a00_start_nonzero_exit_marks_failed(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        with patch("T11_tui_backend.a00_main", return_value=1):
            server.handle_request(build_request("workflow.a00.start", {"argv": []}, message_id="req_nonzero"))
            for worker in list(server._workers.values()):  # noqa: SLF001
                worker.join(timeout=2.0)

        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        responses = [item for item in messages if item.get("kind") == "response" and item.get("id") == "req_nonzero"]
        self.assertTrue(responses)
        self.assertFalse(responses[-1]["ok"])
        self.assertIn("workflow.a00.start exited with non-zero code: 1", responses[-1]["error"])
        stage_events = [item for item in messages if item.get("kind") == "event" and item.get("type") == "stage.changed"]
        self.assertTrue(stage_events)
        self.assertEqual(stage_events[-1]["payload"]["action"], "workflow.a00.start")
        self.assertEqual(stage_events[-1]["payload"]["status"], "failed")

    def test_workflow_a00_ready_timeout_enters_hitl_without_failed_response(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)

        with patch("T11_tui_backend.a00_main", side_effect=RuntimeError("Timed out waiting for agent ready.\nmock screen")):
            server.handle_request(build_request("workflow.a00.start", {"argv": []}, message_id="req_ready_timeout"))
            prompt_id = ""
            prompt_messages: list[dict[str, object]] = []
            messages_before_resolve: list[dict[str, object]] = []
            deadline = time.time() + 2.0
            while time.time() < deadline:
                messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
                prompt_messages = [
                    item
                    for item in messages
                    if item.get("kind") == "event" and item.get("type") == "prompt.request"
                ]
                if prompt_messages:
                    prompt_id = str(prompt_messages[-1]["payload"]["id"])
                app_snapshots = [item for item in messages if item.get("kind") == "event" and item.get("type") == "snapshot.app"]
                if prompt_id and any(item["payload"].get("pending_hitl") for item in app_snapshots):
                    messages_before_resolve = messages
                    break
                time.sleep(0.01)

            self.assertTrue(prompt_id)
            self.assertTrue(messages_before_resolve)
            server.handle_request(
                build_request(
                    "prompt.response",
                    {"prompt_id": prompt_id, "value": "retry_after_manual_model_change"},
                    message_id="req_ready_timeout_prompt",
                )
            )
            for worker in list(server._workers.values()):  # noqa: SLF001
                worker.join(timeout=2.0)

        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        prompt_payload = prompt_messages[-1]["payload"]
        self.assertTrue(prompt_payload["is_hitl"])
        self.assertEqual(prompt_payload["recovery_kind"], "agent_ready_timeout")
        self.assertFalse(prompt_payload["can_skip"])
        stage_events = [item for item in messages_before_resolve if item.get("kind") == "event" and item.get("type") == "stage.changed"]
        self.assertTrue(any(item["payload"]["status"] == "awaiting-input" for item in stage_events))
        app_snapshots = [item for item in messages_before_resolve if item.get("kind") == "event" and item.get("type") == "snapshot.app"]
        self.assertTrue(any(item["payload"].get("pending_hitl") for item in app_snapshots))
        self.assertEqual(len(app_snapshots), 1)
        hitl_snapshots = [item for item in messages_before_resolve if item.get("kind") == "event" and item.get("type") == "snapshot.hitl"]
        self.assertEqual(len(hitl_snapshots), 1)
        responses = [item for item in messages if item.get("kind") == "response" and item.get("id") == "req_ready_timeout"]
        self.assertTrue(responses)
        self.assertTrue(responses[-1]["ok"])
        self.assertNotIn("请求失败", str(responses[-1]))
        self.assertFalse(any(item.get("kind") == "event" and item.get("type") == "error" for item in messages))
        self.assertFalse(
            any(
                item.get("kind") == "event"
                and item.get("type") == "stage.changed"
                and item.get("payload", {}).get("status") == "failed"
                for item in messages
            )
        )

    def test_prompt_shutdown_does_not_mark_stage_failed(self):
        clear_runtime_shutdown_request()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                project_dir = Path(tmpdir)
                writer = io.StringIO()
                server = TuiBackendServer(reader=io.StringIO(), writer=writer)

                def fake_runner(argv, preserve_workers=True):  # noqa: ANN001
                    _ = argv
                    _ = preserve_workers
                    from T09_terminal_ops import prompt_select_option

                    return prompt_select_option(
                        title="HITL: 开发工程师 需要人工介入",
                        options=(("recheck", "我已进入 tmux/修正文件，重新检查"),),
                        default_value="recheck",
                        prompt_text="请选择恢复方式",
                        is_hitl=True,
                    )

                with patch("T11_tui_backend.run_development_stage", side_effect=fake_runner):
                    server.handle_request(
                        build_request(
                            "stage.a07.start",
                            {"argv": ["--project-dir", str(project_dir), "--requirement-name", "需求A"]},
                            message_id="req_prompt_shutdown",
                        )
                    )
                    deadline = time.time() + 2.0
                    while time.time() < deadline:
                        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
                        if any(item.get("kind") == "event" and item.get("type") == "prompt.request" for item in messages):
                            break
                        time.sleep(0.01)
                    server.shutdown(cleanup_tmux=False)
                    for worker in list(server._workers.values()):  # noqa: SLF001
                        worker.join(timeout=2.0)

                messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
                failure_path = (
                    project_dir
                    / ".tmux_workflow"
                    / "需求A"
                    / "stages"
                    / "stage_a07_start.failure.json"
                )
                failure_exists = failure_path.exists()

            self.assertFalse(failure_exists)
            self.assertFalse(any(item.get("kind") == "event" and item.get("type") == "error" for item in messages))
            self.assertFalse(
                any(
                    item.get("kind") == "event"
                    and item.get("type") == "stage.changed"
                    and item.get("payload", {}).get("status") == "failed"
                    for item in messages
                )
            )
        finally:
            clear_runtime_shutdown_request()

    def test_workflow_a00_failure_uses_current_stage_action_and_seq(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)

        def fake_runner(argv):  # noqa: ANN001
            server._handle_runtime_stage_change("stage.a05.start")  # noqa: SLF001
            return 1

        with patch("T11_tui_backend.a00_main", side_effect=fake_runner):
            server.handle_request(build_request("workflow.a00.start", {"argv": []}, message_id="req_stage_fail"))
            for worker in list(server._workers.values()):  # noqa: SLF001
                worker.join(timeout=2.0)

        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        stage_events = [item for item in messages if item.get("kind") == "event" and item.get("type") == "stage.changed"]
        review_events = [item for item in stage_events if item.get("payload", {}).get("action") == "stage.a05.start"]
        self.assertGreaterEqual(len(review_events), 2)
        self.assertEqual(review_events[-1]["payload"]["status"], "failed")
        self.assertEqual(review_events[-1]["payload"]["stage_seq"], review_events[0]["payload"]["stage_seq"])

    def test_requirement_concurrency_conflict_is_reported_as_error(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)

        def fake_runner(_argv):  # noqa: ANN001
            server._handle_runtime_stage_change("stage.a02.start")  # noqa: SLF001
            raise RuntimeError(
                "并发冲突：同项目同需求已有运行中任务（lock_key=/tmp/project::需求A; "
                "holder=pid=1, thread_name=tui-backend-workflow.a00.start-req_1）"
            )

        with patch("T11_tui_backend.a00_main", side_effect=fake_runner):
            server.handle_request(build_request("workflow.a00.start", {"argv": []}, message_id="req_conflict"))
            for worker in list(server._workers.values()):  # noqa: SLF001
                worker.join(timeout=2.0)

        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        responses = [item for item in messages if item.get("kind") == "response" and item.get("id") == "req_conflict"]
        self.assertTrue(responses)
        self.assertFalse(responses[-1]["ok"])
        self.assertIn("并发冲突", responses[-1]["error"])
        self.assertTrue(responses[-1]["payload"]["already_running"])
        self.assertTrue(any(item.get("kind") == "event" and item.get("type") == "error" for item in messages))
        self.assertTrue(
            any(
                item.get("kind") == "event"
                and item.get("type") == "stage.changed"
                and item.get("payload", {}).get("status") == "failed"
                for item in messages
            )
        )

    def test_backend_main_cleans_tmux_on_normal_eof(self):
        shutdown_calls: list[bool] = []

        class _FakeServer:
            def protocol_log_sink(self):
                return io.StringIO()

            def serve_forever(self):
                return 0

            def shutdown(self, *, cleanup_tmux: bool):
                shutdown_calls.append(cleanup_tmux)
                return []

        original_stdout = sys.stdout
        try:
            with patch("T11_tui_backend.TuiBackendServer", return_value=_FakeServer()), patch(
                "T11_tui_backend.signal.getsignal",
                return_value=signal.SIG_DFL,
            ), patch("T11_tui_backend.signal.signal"):
                exit_code = backend_main([])
        finally:
            sys.stdout = original_stdout

        self.assertEqual(exit_code, 0)
        self.assertEqual(shutdown_calls, [True])

    def test_backend_main_cleans_tmux_on_sigterm(self):
        shutdown_calls: list[bool] = []
        handlers: dict[int, object] = {}

        class _FakeServer:
            def protocol_log_sink(self):
                return io.StringIO()

            def serve_forever(self):
                handler = handlers[signal.SIGTERM]
                handler(signal.SIGTERM, None)
                return 0

            def shutdown(self, *, cleanup_tmux: bool):
                shutdown_calls.append(cleanup_tmux)
                return []

        def fake_signal(signum, handler):  # noqa: ANN001
            handlers[int(signum)] = handler

        original_stdout = sys.stdout
        try:
            with patch("T11_tui_backend.TuiBackendServer", return_value=_FakeServer()), patch(
                "T11_tui_backend.signal.getsignal",
                return_value=signal.SIG_DFL,
            ), patch("T11_tui_backend.signal.signal", side_effect=fake_signal):
                with self.assertRaises(SystemExit) as context:
                    backend_main([])
        finally:
            sys.stdout = original_stdout

        self.assertEqual(context.exception.code, 128 + int(signal.SIGTERM))
        self.assertEqual(shutdown_calls, [True, True])

    def test_stage_a01_start_auto_chains_to_requirement_intake(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)
            intake_calls: list[list[str]] = []

            with patch(
                "T11_tui_backend.run_routing_stage",
                return_value=SimpleNamespace(project_dir=tmpdir, exit_code=0),
            ), patch(
                "T11_tui_backend.run_requirement_intake_stage",
                side_effect=lambda argv: intake_calls.append(list(argv)) or SimpleNamespace(
                    project_dir=tmpdir,
                    requirement_name="贪吃蛇",
                ),
            ):
                server.handle_request(
                    build_request(
                        "stage.a01.start",
                        {"argv": ["--project-dir", tmpdir, "--requirement-name", "贪吃蛇", "--reuse-existing-original-requirement", "--yes"]},
                        message_id="req_a01",
                    )
                )
                for _ in range(10):
                    workers = list(server._workers.values())  # noqa: SLF001
                    if not workers:
                        break
                    for worker in workers:
                        worker.join(timeout=2.0)

            self.assertEqual(len(intake_calls), 1)
            self.assertEqual(intake_calls[0][:2], ["--project-dir", tmpdir])
            self.assertIn("--requirement-name", intake_calls[0])
            self.assertIn("贪吃蛇", intake_calls[0])
            self.assertIn("--reuse-existing-original-requirement", intake_calls[0])
            self.assertIn("--yes", intake_calls[0])
            messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
            self.assertTrue(any(item.get("kind") == "event" and item.get("type") == "log.append" and "自动进入需求录入阶段" in item.get("payload", {}).get("text", "") for item in messages))

    def test_stage_a01_start_does_not_chain_when_routing_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)

            with patch(
                "T11_tui_backend.run_routing_stage",
                return_value=SimpleNamespace(project_dir=tmpdir, exit_code=1),
            ), patch(
                "T11_tui_backend.run_requirement_intake_stage",
                side_effect=AssertionError("routing failed 时不应自动进入需求录入"),
            ):
                server.handle_request(
                    build_request(
                        "stage.a01.start",
                        {"argv": ["--project-dir", tmpdir]},
                        message_id="req_a01_failed",
                    )
                )
                for worker in list(server._workers.values()):  # noqa: SLF001
                    worker.join(timeout=2.0)

    def test_workflow_start_clears_stale_stage_state_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            record_dir = project_dir / ".tmux_workflow" / "需求A" / "stages"
            record_dir.mkdir(parents=True)
            stale_path = record_dir / "stage_a07_start.state.json"
            stale_path.write_text(
                json.dumps(
                    {
                        "action": "stage.a07.start",
                        "status": "awaiting-input",
                        "project_dir": str(project_dir),
                        "requirement_name": "需求A",
                        "stage_seq": 8,
                        "source": "runtime_inference",
                        "updated_at": "2026-04-01T10:00:00+08:00",
                        "failure_path": "",
                        "message": "",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="workflow.a00.start")  # noqa: SLF001

            server._emit_display_stage_state(  # noqa: SLF001
                preferred_status="running",
                preferred_action="workflow.a00.start",
                preferred_stage_seq=1,
                source="runner_start",
                force=True,
            )

            workflow_path = record_dir / "workflow_a00_start.state.json"
            stale_exists = stale_path.exists()
            workflow_exists = workflow_path.exists()

        self.assertFalse(stale_exists)
        self.assertTrue(workflow_exists)

    def test_running_stage_state_clears_same_action_stale_failure_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            record_dir = project_dir / ".tmux_workflow" / "需求A" / "stages"
            record_dir.mkdir(parents=True)
            failure_payload = {
                "action": "stage.a07.start",
                "status": "failed",
                "project_dir": str(project_dir),
                "requirement_name": "需求A",
                "error": "old failure",
            }
            failure_path = record_dir / "stage_a07_start.failure.json"
            latest_path = record_dir / "latest_failure.json"
            failure_path.write_text(json.dumps(failure_payload, ensure_ascii=False), encoding="utf-8")
            latest_path.write_text(json.dumps(failure_payload, ensure_ascii=False), encoding="utf-8")
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="stage.a07.start")  # noqa: SLF001

            server._emit_display_stage_state(  # noqa: SLF001
                preferred_status="running",
                preferred_action="stage.a07.start",
                preferred_stage_seq=12,
                source="runtime_inference",
                force=True,
            )
            failure_exists = failure_path.exists()
            latest_exists = latest_path.exists()

        self.assertFalse(failure_exists)
        self.assertFalse(latest_exists)

    def test_runtime_stage_change_marks_previous_forward_stage_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            record_dir = project_dir / ".tmux_workflow" / "需求A" / "stages"
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="workflow.a00.start")  # noqa: SLF001

            server._handle_runtime_stage_change("stage.a03.start")  # noqa: SLF001
            server._handle_runtime_stage_change("stage.a04.start")  # noqa: SLF001

            previous_payload = json.loads((record_dir / "stage_a03_start.state.json").read_text(encoding="utf-8"))
            current_payload = json.loads((record_dir / "stage_a04_start.state.json").read_text(encoding="utf-8"))

        self.assertEqual(previous_payload["action"], "stage.a03.start")
        self.assertEqual(previous_payload["status"], "completed")
        self.assertEqual(previous_payload["source"], "runtime_inference")
        self.assertIn("stage.a04.start", previous_payload["message"])
        self.assertEqual(current_payload["action"], "stage.a04.start")
        self.assertEqual(current_payload["status"], "running")

    def test_runtime_stage_change_marks_previous_backward_stage_superseded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            record_dir = project_dir / ".tmux_workflow" / "需求A" / "stages"
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="workflow.a00.start")  # noqa: SLF001

            server._handle_runtime_stage_change("stage.a05.start")  # noqa: SLF001
            server._handle_runtime_stage_change("stage.a04.start")  # noqa: SLF001

            previous_payload = json.loads((record_dir / "stage_a05_start.state.json").read_text(encoding="utf-8"))
            current_payload = json.loads((record_dir / "stage_a04_start.state.json").read_text(encoding="utf-8"))

        self.assertEqual(previous_payload["action"], "stage.a05.start")
        self.assertEqual(previous_payload["status"], "superseded")
        self.assertEqual(previous_payload["source"], "runtime_inference")
        self.assertIn("stage.a04.start", previous_payload["message"])
        self.assertEqual(current_payload["action"], "stage.a04.start")
        self.assertEqual(current_payload["status"], "running")

    def test_nonzero_routing_error_includes_failed_target_details(self):
        server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
        result = SimpleNamespace(
            exit_code=1,
            batch_result=SimpleNamespace(
                results=[
                    SimpleNamespace(work_dir="/tmp/project", status="passed"),
                    SimpleNamespace(
                        work_dir="/tmp/project/core",
                        status="failed",
                        failure_reason="create_command_failed: prompt timeout",
                    ),
                ]
            ),
        )

        with self.assertRaises(RuntimeError) as context:
            server._raise_for_nonzero_exit_code(action="stage.a01.start", stage_seq=1, result=result)  # noqa: SLF001

        message = str(context.exception)
        self.assertIn("stage.a01.start exited with non-zero code: 1", message)
        self.assertIn("failed routing targets", message)
        self.assertIn("/tmp/project/core", message)
        self.assertIn("create_command_failed: prompt timeout", message)

    def test_workflow_status_downgrades_to_awaiting_input_when_runner_leaves_file_driven_hitl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            requirement_name = "贪吃蛇"
            _, _, ask_human_path, _ = build_requirements_clarification_paths(project_dir, requirement_name)
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)

            def fake_runner(argv):  # noqa: ANN001
                server._handle_runtime_stage_change("stage.a04.start")  # noqa: SLF001
                ask_human_path.parent.mkdir(parents=True, exist_ok=True)
                ask_human_path.write_text("请补充碰撞规则。", encoding="utf-8")
                return SimpleNamespace(project_dir=str(project_dir), requirement_name=requirement_name)

            with patch("T11_tui_backend.a00_main", side_effect=fake_runner):
                server.handle_request(
                    build_request(
                        "workflow.a00.start",
                        {"argv": ["--project-dir", str(project_dir), "--requirement-name", requirement_name]},
                        message_id="req_hitl",
                    )
                )
                for worker in list(server._workers.values()):  # noqa: SLF001
                    worker.join(timeout=2.0)
                server._flush_dirty_snapshots()  # noqa: SLF001

        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        stage_events = [item for item in messages if item.get("kind") == "event" and item.get("type") == "stage.changed"]
        self.assertTrue(stage_events)
        self.assertEqual(stage_events[-1]["payload"]["action"], "stage.a04.start")
        self.assertEqual(stage_events[-1]["payload"]["status"], "awaiting-input")
        app_events = [item for item in messages if item.get("kind") == "event" and item.get("type") == "snapshot.app"]
        self.assertTrue(app_events)
        self.assertTrue(app_events[-1]["payload"]["pending_hitl"])

    def test_failed_status_is_not_masked_by_file_detected_hitl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _, _, ask_human_path, _ = build_requirements_clarification_paths(project_dir, "贪吃蛇")
            ask_human_path.parent.mkdir(parents=True, exist_ok=True)
            ask_human_path.write_text("请补充碰撞规则。", encoding="utf-8")
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a04.start")  # noqa: SLF001

            action, status, _stage_seq = server._derive_display_stage_state(  # noqa: SLF001
                preferred_status="failed",
                preferred_action="stage.a04.start",
            )

        self.assertEqual(action, "stage.a04.start")
        self.assertEqual(status, "failed")

    def test_control_open_returns_snapshot_for_existing_session(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        server._controls["run_demo"] = ControlSessionState(control_id="run_demo", center=_FakeCenter())  # noqa: SLF001
        server.handle_request(build_request("control.b01.open", {"control_id": "run_demo"}, message_id="req_3"))
        server._flush_dirty_snapshots()  # noqa: SLF001
        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        payload = messages[0]["payload"]
        self.assertTrue(payload["supported"])
        self.assertEqual(payload["control_id"], "run_demo")
        self.assertEqual(payload["status_text"], "status text")
        self.assertEqual(payload["workers"][0]["session_name"], "sess-1")
        event_types = [item.get("type") for item in messages[1:] if item.get("kind") == "event"]
        self.assertEqual(event_types, ["snapshot.control"])

    def test_worker_attach_returns_tmux_attach_command(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        server._controls["run_demo"] = ControlSessionState(control_id="run_demo", center=_FakeCenter())  # noqa: SLF001
        server.handle_request(
            build_request(
                "worker.attach",
                {"control_id": "run_demo", "argument": "1"},
                message_id="req_4",
            )
        )
        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        payload = messages[0]["payload"]
        self.assertEqual(payload["attach_command"], ["tmux", "attach", "-t", "sess-1"])
        self.assertEqual(payload["work_dir"], "/tmp/demo")

    def test_run_resume_replaces_previous_control_session(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        old_center = _FakeCenter(run_id="run_old", done=True)
        server._controls["run_old"] = ControlSessionState(control_id="run_old", center=old_center)  # noqa: SLF001
        with patch("T11_tui_backend.AgentInitControlCenter.from_existing_run", return_value=_FakeCenter(run_id="run_new")):
            server.handle_request(
                build_request(
                    "run.resume",
                    {"control_id": "run_old", "run_id": "run_new"},
                    message_id="req_5",
                )
            )
        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        payload = messages[0]["payload"]
        self.assertTrue(old_center.closed)
        self.assertEqual(payload["control_id"], "run_new")

    def test_run_resume_cleans_failed_run_sessions_before_switch(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        failed_center = _FakeCenter(run_id="run_failed", done=True)
        failed_batch = type(
            "Batch",
            (),
            {
                "run_id": "run_failed",
                "runtime_dir": "/tmp/runtime",
                "config": {"vendor": "codex", "model": "gpt-5.4", "reasoning_effort": "high", "proxy_url": ""},
                "results": [type("Item", (), {"status": "failed", "work_dir": "/tmp/demo", "failure_reason": "", "last_audit_summary": ""})()],
            },
        )()
        server._controls["run_failed"] = ControlSessionState(
            control_id="run_failed",
            center=failed_center,
            final_result=failed_batch,
        )  # noqa: SLF001
        with patch("T11_tui_backend.AgentInitControlCenter.from_existing_run", return_value=_FakeCenter(run_id="run_new")):
            server.handle_request(
                build_request(
                    "run.resume",
                    {"control_id": "run_failed", "run_id": "run_new"},
                    message_id="req_7",
                )
            )
        self.assertTrue(failed_center.cleaned)

    def test_worker_retry_clears_completed_snapshot_state(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        session = ControlSessionState(
            control_id="run_demo",
            center=_FakeCenter(done=False),
            final_result=object(),
            transition_text="old transition",
        )
        server._controls["run_demo"] = session  # noqa: SLF001
        server.handle_request(
            build_request(
                "worker.retry",
                {"control_id": "run_demo", "argument": "1"},
                message_id="req_6",
            )
        )
        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        payload = messages[0]["payload"]
        self.assertFalse(payload["done"])
        self.assertEqual(payload["transition_text"], "")

    def test_protocol_log_sink_converts_stdout_noise_into_log_event(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        with redirect_stdout(server.protocol_log_sink()):
            print("警告：文件不存在 -> /tmp/demo")
        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        self.assertTrue(any(item.get("kind") == "event" and item.get("type") == "log.append" for item in messages))
        payloads = [item.get("payload", {}) for item in messages if item.get("type") == "log.append"]
        self.assertTrue(any("警告：文件不存在" in str(payload.get("text", "")) for payload in payloads))

    def test_shutdown_closes_controls_and_cleans_tmux_when_requested(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        center = _FakeCenter(run_id="run_demo", done=False)
        server._controls["run_demo"] = ControlSessionState(control_id="run_demo", center=center)  # noqa: SLF001
        with patch("T11_tui_backend.cleanup_registered_tmux_workers", return_value=["sess-1"]) as cleanup:
            cleaned = server.shutdown(cleanup_tmux=True)
        self.assertEqual(cleaned, ["sess-1"])
        self.assertTrue(center.closed)
        cleanup.assert_called_once_with(reason="tui_backend_shutdown")

    def test_shutdown_requests_runtime_shutdown_and_new_server_clears_it(self):
        clear_runtime_shutdown_request()
        try:
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)
            self.assertFalse(runtime_shutdown_requested())
            server.shutdown(cleanup_tmux=False)
            self.assertTrue(runtime_shutdown_requested())

            TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            self.assertFalse(runtime_shutdown_requested())
        finally:
            clear_runtime_shutdown_request()

    def test_shutdown_cleanup_can_run_after_initial_non_cleanup_shutdown(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        first_cleaned = server.shutdown(cleanup_tmux=False)
        with patch("T11_tui_backend.cleanup_registered_tmux_workers", return_value=["sess-late"]), patch.object(
            server,
            "_cleanup_visible_tmux_workers",
            return_value=[],
        ), patch.object(
            server,
            "_cleanup_project_runtime_tmux_workers",
            return_value=[],
        ):
            second_cleaned = server.shutdown(cleanup_tmux=True)
        self.assertEqual(first_cleaned, [])
        self.assertEqual(second_cleaned, ["sess-late"])

    def test_shutdown_waits_for_runner_threads_before_tmux_cleanup(self):
        clear_runtime_shutdown_request()
        try:
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)
            runner_stopped = threading.Event()
            cleanup_observed_runner_stopped: list[bool] = []

            def runner():
                while not runtime_shutdown_requested():
                    time.sleep(0.01)
                runner_stopped.set()

            thread = threading.Thread(target=runner, name="tui-backend-unit-runner")
            thread.start()
            server._workers["unit"] = thread  # noqa: SLF001

            def fake_cleanup(*, reason):  # noqa: ANN001
                cleanup_observed_runner_stopped.append(runner_stopped.is_set())
                return []

            with patch("T11_tui_backend.cleanup_registered_tmux_workers", side_effect=fake_cleanup), patch.object(
                server,
                "_cleanup_visible_tmux_workers",
                return_value=[],
            ), patch.object(
                server,
                "_cleanup_project_runtime_tmux_workers",
                return_value=[],
            ):
                server.shutdown(cleanup_tmux=True)

            thread.join(timeout=1.0)
            self.assertTrue(runner_stopped.is_set())
            self.assertEqual(cleanup_observed_runner_stopped, [True])
        finally:
            clear_runtime_shutdown_request()

    def test_shutdown_cleans_visible_unregistered_tmux_workers_when_requested(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        server._set_context(project_dir="/tmp/project", requirement_name="需求A", action="stage.a07.start")  # noqa: SLF001
        killed_sessions: list[tuple[str, bool]] = []
        server._tmux_runtime = SimpleNamespace(  # noqa: SLF001
            kill_session=lambda session_name, *, missing_ok=True: killed_sessions.append((session_name, missing_ok)) or session_name,
        )
        visible_worker = {
            "worker_id": "development-review-测试工程师",
            "session_name": "测试工程师-天寿星",
            "project_dir": "/tmp/project",
            "requirement_name": "需求A",
            "workflow_action": "stage.a07.start",
            "status": "failed",
            "agent_state": "READY",
            "health_status": "alive",
            "session_exists": True,
        }

        def fake_current_stage_workers(action):  # noqa: ANN001
            return [visible_worker] if action == "stage.a07.start" else []

        with patch("T11_tui_backend.cleanup_registered_tmux_workers", return_value=["registered-sess"]), patch.object(
            server,
            "_current_stage_workers",
            side_effect=fake_current_stage_workers,
        ):
            cleaned = server.shutdown(cleanup_tmux=True)

        self.assertEqual(killed_sessions, [("测试工程师-天寿星", True)])
        self.assertEqual(cleaned, ["registered-sess", "测试工程师-天寿星"])

    def test_shutdown_cleans_current_project_runtime_state_workers_regardless_status(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir) / "project"
            other_project_dir = Path(tmp_dir) / "other-project"
            project_dir.mkdir()
            other_project_dir.mkdir()
            current_worker_root = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME / "M1-T1" / "dev-worker"
            current_worker_root.mkdir(parents=True)
            (current_worker_root / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "开发工程师-地雄星",
                        "project_dir": str(project_dir),
                        "work_dir": str(project_dir / "src"),
                        "status": "succeeded",
                        "result_status": "succeeded",
                        "agent_state": "READY",
                        "health_status": "alive",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            routing_worker_root = project_dir / ROUTING_RUNTIME_ROOT_NAME / "run_1" / "router"
            routing_worker_root.mkdir(parents=True)
            (routing_worker_root / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "路由器-天异星",
                        "project_dir": str(project_dir),
                        "work_dir": str(project_dir),
                        "status": "completed",
                        "agent_state": "READY",
                        "health_status": "alive",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            foreign_worker_root = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME / "foreign"
            foreign_worker_root.mkdir(parents=True)
            (foreign_worker_root / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "其他项目-不应清理",
                        "project_dir": str(other_project_dir),
                        "work_dir": str(other_project_dir / "src"),
                        "status": "succeeded",
                        "agent_state": "READY",
                        "health_status": "alive",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)
            server._set_context(project_dir=str(project_dir), requirement_name="需求A", action="stage.a07.start")  # noqa: SLF001
            killed_sessions: list[tuple[str, bool]] = []
            live_sessions = {"开发工程师-地雄星", "路由器-天异星", "其他项目-不应清理"}
            server._tmux_runtime = SimpleNamespace(  # noqa: SLF001
                session_exists=lambda session_name: session_name in live_sessions,
                kill_session=lambda session_name, *, missing_ok=True: killed_sessions.append((session_name, missing_ok)) or session_name,
            )

            with patch("T11_tui_backend.cleanup_registered_tmux_workers", return_value=[]), patch.object(
                server,
                "_cleanup_visible_tmux_workers",
                return_value=[],
            ):
                cleaned = server.shutdown(cleanup_tmux=True)

        self.assertEqual(killed_sessions, [("路由器-天异星", True), ("开发工程师-地雄星", True)])
        self.assertEqual(cleaned, ["开发工程师-地雄星", "路由器-天异星"])

    def test_run_list_reads_existing_manifests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir(parents=True)
            runtime_root = build_routing_runtime_root(project_dir)
            run_root = runtime_root / "run_demo"
            run_root.mkdir(parents=True)
            (run_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "manifest_version": 1,
                        "run_id": "run_demo",
                        "runtime_dir": str(run_root),
                        "project_dir": str(project_dir),
                        "selection": {"project_dir": str(project_dir), "selected_dirs": [], "skipped_dirs": [], "forced_dirs": [], "project_missing_files": []},
                        "config": {"vendor": "codex", "model": "gpt-5.4", "reasoning_effort": "high", "proxy_url": ""},
                        "status": "completed",
                        "created_at": "2026-04-16T10:00:00",
                        "updated_at": "2026-04-16T10:00:00",
                        "workers": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)
            server._set_context(project_dir=str(project_dir), action="stage.a01.start")  # noqa: SLF001
            server.handle_request(build_request("run.list", {}, message_id="req_8"))
        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        payload = messages[0]["payload"]
        self.assertEqual(payload["runs"][0]["run_id"], "run_demo")

    def test_run_list_returns_empty_without_project_context(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        server.handle_request(build_request("run.list", {}, message_id="req_8_empty"))
        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        self.assertEqual(messages[0]["payload"]["runs"], [])

    def test_run_resume_requires_project_context_or_payload_project_dir(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        with self.assertRaisesRegex(ValueError, "当前项目内的 routing run"):
            server.handle_request(
                build_request(
                    "run.resume",
                    {"run_id": "run_demo"},
                    message_id="req_resume_missing_project",
                )
            )

    def test_workflow_a00_stage_label_follows_file_state_progression(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="workflow.a00.start")  # noqa: SLF001

            self.assertEqual(server._build_app_snapshot()["active_stage_label"], "路由初始化")  # noqa: SLF001

            for file_path in required_routing_layer_paths(project_dir):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("ok", encoding="utf-8")
            self.assertEqual(server._build_app_snapshot()["active_stage_label"], "需求录入")  # noqa: SLF001

            original_requirement_path, requirements_clear_path, _, _ = build_requirements_clarification_paths(project_dir, "贪吃蛇")
            original_requirement_path.write_text("原始需求", encoding="utf-8")
            self.assertEqual(server._build_app_snapshot()["active_stage_label"], "需求澄清")  # noqa: SLF001

            requirements_clear_path.write_text("需求澄清", encoding="utf-8")
            self.assertEqual(server._build_app_snapshot()["active_stage_label"], "需求评审")  # noqa: SLF001
            self.assertFalse((project_dir / "贪吃蛇_人机交互澄清记录.md").exists())

            review_paths = build_requirements_review_paths(project_dir, "贪吃蛇")
            review_paths["merged_review_path"].write_text("评审完成", encoding="utf-8")
            self.assertEqual(server._build_app_snapshot()["active_stage_label"], "详细设计")  # noqa: SLF001

            update_pre_development_task_status(project_dir, "贪吃蛇", task_key="详细设计", completed=True)
            self.assertEqual(server._build_app_snapshot()["active_stage_label"], "任务拆分")  # noqa: SLF001

            update_pre_development_task_status(project_dir, "贪吃蛇", task_key="任务拆分", completed=True)
            self.assertEqual(server._build_app_snapshot()["active_stage_label"], "任务开发")  # noqa: SLF001

            development_paths = build_development_paths(project_dir, "贪吃蛇")
            development_paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.assertEqual(server._build_app_snapshot()["active_stage_label"], "复核")  # noqa: SLF001

            overall_review_paths = build_overall_review_paths(project_dir, "贪吃蛇")
            overall_review_paths["state_path"].write_text(
                json.dumps({"passed": True}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.assertEqual(server._build_app_snapshot()["active_stage_label"], "测试")  # noqa: SLF001

    def test_app_snapshot_prefers_display_action_for_active_stage_label(self):
        server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
        server._display_action = "stage.a07.start"  # noqa: SLF001
        server._context.current_action = ""  # noqa: SLF001

        app = server._build_app_snapshot()  # noqa: SLF001

        self.assertEqual(app["active_stage"], "stage.a07.start")
        self.assertEqual(app["active_stage_label"], "任务开发")

    def test_runtime_scanned_worker_snapshots_include_session_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            worker_root = runtime_root / "worker-1"
            worker_root.mkdir(parents=True)
            state_path = worker_root / "worker.state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "session_name": "sess-runtime",
                        "work_dir": "/tmp/project",
                        "status": "running",
                        "workflow_stage": "create_running",
                        "agent_state": "READY",
                        "health_status": "alive",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime = SimpleNamespace(session_exists=lambda name: name == "sess-runtime")  # noqa: SLF001
            workers = server._scan_runtime_workers(runtime_root)  # noqa: SLF001
        self.assertEqual(workers[0]["session_name"], "sess-runtime")
        self.assertTrue(workers[0]["session_exists"])

    def test_runtime_worker_lightweight_scan_skips_health_refresh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            worker_root = runtime_root / "worker-1"
            worker_root.mkdir(parents=True)
            (worker_root / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "sess-runtime",
                        "work_dir": "/tmp/project",
                        "status": "running",
                        "agent_state": "BUSY",
                        "health_status": "alive",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime = SimpleNamespace(session_exists=lambda name: name == "sess-runtime", backend=object())  # noqa: SLF001
            with patch("T11_tui_backend.load_worker_from_state_path") as load_worker:
                workers = server._scan_runtime_workers(runtime_root, refresh_health=False)  # noqa: SLF001

        load_worker.assert_not_called()
        self.assertEqual(workers[0]["session_name"], "sess-runtime")

    def test_runtime_worker_scan_tolerates_concurrently_deleted_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            live_root = runtime_root / "worker-live"
            disappearing_root = runtime_root / "worker-disappearing"
            live_root.mkdir(parents=True)
            disappearing_root.mkdir(parents=True)
            (live_root / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "sess-live",
                        "work_dir": "/tmp/project",
                        "status": "running",
                        "agent_state": "BUSY",
                        "health_status": "alive",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            real_scandir = os.scandir

            def flaky_scandir(path):  # noqa: ANN001
                if Path(path) == disappearing_root:
                    raise FileNotFoundError(str(path))
                return real_scandir(path)

            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime = SimpleNamespace(session_exists=lambda name: name == "sess-live", backend=object())  # noqa: SLF001
            with patch("tmux_core.bridge.backend.os.scandir", side_effect=flaky_scandir), patch(
                "T11_tui_backend.load_worker_from_state_path"
            ) as load_worker:
                workers = server._scan_runtime_workers(runtime_root, refresh_health=False)  # noqa: SLF001

        load_worker.assert_not_called()
        self.assertEqual([worker["session_name"] for worker in workers], ["sess-live"])

    def test_handle_runtime_state_change_schedules_lightweight_snapshot_without_stage_inference(self):
        server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
        server._display_action = "stage.a07.start"  # noqa: SLF001
        with patch.object(server, "_infer_runtime_stage_status", side_effect=AssertionError("stage inference should be async")) as infer_status, patch.object(
            server,
            "_schedule_snapshot_update",
        ) as schedule_snapshot:
            server._handle_runtime_state_change()  # noqa: SLF001

        schedule_snapshot.assert_called_once()
        infer_status.assert_not_called()
        self.assertFalse(schedule_snapshot.call_args.kwargs["refresh_worker_health"])

    def test_stage_action_change_schedules_lightweight_snapshot_without_sync_emit(self):
        server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
        with patch.object(server, "_emit_snapshot_update", side_effect=AssertionError("sync snapshot should not run")) as emit_snapshot, patch.object(
            server,
            "_schedule_snapshot_update",
        ) as schedule_snapshot, patch.object(server, "_infer_runtime_stage_status", return_value=""), patch.object(
            server,
            "_build_hitl_snapshot",
            return_value={"pending": False},
        ):
            server._handle_runtime_stage_change("stage.a07.start")  # noqa: SLF001

        emit_snapshot.assert_not_called()
        schedule_snapshot.assert_called_once()
        self.assertEqual(schedule_snapshot.call_args.kwargs["sections"], {"app"})
        self.assertFalse(schedule_snapshot.call_args.kwargs["refresh_worker_health"])

    def test_runner_completion_schedules_snapshot_without_waiting_for_heavy_emit(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)

        with patch.object(server, "_emit_snapshot_update", side_effect=AssertionError("sync snapshot should not run")) as emit_snapshot, patch.object(
            server,
            "_schedule_snapshot_update",
        ) as schedule_snapshot, patch.object(server, "_maybe_chain_after_stage_success") as chain_after_success, patch.object(
            server,
            "_infer_runtime_stage_status",
            return_value="",
        ), patch.object(server, "_build_hitl_snapshot", return_value={"pending": False}):
            server._run_in_thread("req_flow", "workflow.a00.start", lambda: 0, argv=[], respond=True)  # noqa: SLF001
            for worker in list(server._workers.values()):  # noqa: SLF001
                worker.join(timeout=2.0)

        emit_snapshot.assert_not_called()
        self.assertTrue(schedule_snapshot.called)
        self.assertTrue(any(not call.kwargs.get("refresh_worker_health", True) for call in schedule_snapshot.call_args_list))
        chain_after_success.assert_called_once()

    def test_runner_failure_schedules_snapshot_without_sync_emit(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)

        with patch.object(server, "_emit_snapshot_update", side_effect=AssertionError("sync snapshot should not run")) as emit_snapshot, patch.object(
            server,
            "_schedule_snapshot_update",
        ) as schedule_snapshot, patch.object(server, "_infer_runtime_stage_status", return_value=""), patch.object(
            server,
            "_build_hitl_snapshot",
            return_value={"pending": False},
        ):
            server._run_in_thread("req_fail", "workflow.a00.start", lambda: (_ for _ in ()).throw(RuntimeError("boom")), respond=True)  # noqa: SLF001
            for worker in list(server._workers.values()):  # noqa: SLF001
                worker.join(timeout=2.0)

        emit_snapshot.assert_not_called()
        self.assertTrue(schedule_snapshot.called)
        self.assertTrue(any(not call.kwargs.get("refresh_worker_health", True) for call in schedule_snapshot.call_args_list))

    def test_runtime_triggered_lightweight_snapshot_includes_dispatch_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            requirement_name = "需求A"
            worker_root = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME / requirement_name / "worker-1"
            worker_root.mkdir(parents=True)
            (worker_root / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "development-review-测试工程师",
                        "session_name": "测试工程师-天慧星",
                        "work_dir": str(project_dir),
                        "project_dir": str(project_dir),
                        "requirement_name": requirement_name,
                        "workflow_action": "stage.a07.start",
                        "status": "running",
                        "agent_state": "READY",
                        "health_status": "alive",
                        "dispatch_state": "submitting",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime = SimpleNamespace(session_exists=lambda name: name == "测试工程师-天慧星", backend=object())  # noqa: SLF001
            server._context.project_dir = str(project_dir)  # noqa: SLF001
            server._context.requirement_name = requirement_name  # noqa: SLF001
            emitted: list[tuple[str, dict[str, object]]] = []
            with patch.object(server, "emit_event", side_effect=lambda event, payload: emitted.append((event, payload))), patch(
                "T11_tui_backend.load_worker_from_state_path"
            ) as load_worker:
                server._emit_snapshot_update(  # noqa: SLF001
                    stage_routes=("development",),
                    refresh_worker_health=False,
                )

        load_worker.assert_not_called()
        stage_events = [payload for event, payload in emitted if event == "snapshot.stage"]
        self.assertEqual(stage_events[0]["route"], "development")
        workers = stage_events[0]["snapshot"]["workers"]  # type: ignore[index]
        self.assertEqual(workers[0]["dispatch_state"], "submitting")

    def test_flow_snapshot_flush_does_not_refresh_control_worker_health(self):
        class RaisingRefreshCenter(_FakeCenter):
            def refresh_worker_health(self) -> None:
                raise AssertionError("flow snapshot should not refresh worker health")

        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        server._controls["run_demo"] = ControlSessionState(control_id="run_demo", center=RaisingRefreshCenter())  # noqa: SLF001

        server._schedule_flow_snapshot_update(sections={"app", "control"})  # noqa: SLF001
        server._flush_dirty_snapshots()  # noqa: SLF001

        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        event_types = [item.get("type") for item in messages if item.get("kind") == "event"]
        self.assertIn("snapshot.app", event_types)
        self.assertIn("snapshot.control", event_types)

    def test_development_snapshot_recovers_sparse_health_state_from_tmux_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            requirement_name = "需求A"
            runtime_dir = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME / requirement_name / "development-review-abcd1234"
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "agent_alive": True,
                        "agent_started": True,
                        "agent_ready": True,
                        "agent_state": "READY",
                        "health_status": "alive",
                        "health_note": "alive",
                        "updated_at": "2026-05-09T15:40:17",
                        "last_heartbeat_at": "2026-05-09T15:40:17",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            class FakeTmuxRuntime:
                backend = None

                def session_exists(self, name):
                    return name == "测试工程师-天慧星"

                def session_matches_worker_state(self, name, state, state_path):  # noqa: ARG002
                    return name == "测试工程师-天慧星"

                def worker_identity_for_runtime_dir(self, current_runtime_dir):
                    if Path(current_runtime_dir).resolve() != runtime_dir.resolve():
                        return {}
                    return {
                        "session_name": "测试工程师-天慧星",
                        "session_exists": True,
                        "worker_id": "development-review-测试工程师",
                        "work_dir": str(project_dir),
                        "project_dir": str(project_dir),
                        "requirement_name": requirement_name,
                        "workflow_action": "stage.a07.start",
                    }

            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime = FakeTmuxRuntime()  # noqa: SLF001
            server._context.project_dir = str(project_dir)  # noqa: SLF001
            server._context.requirement_name = requirement_name  # noqa: SLF001

            snapshot = server._build_development_snapshot()  # noqa: SLF001

        self.assertEqual(len(snapshot["workers"]), 1)
        self.assertEqual(snapshot["workers"][0]["session_name"], "测试工程师-天慧星")
        self.assertEqual(snapshot["workers"][0]["worker_id"], "development-review-测试工程师")
        self.assertEqual(snapshot["workers"][0]["requirement_name"], requirement_name)
        self.assertEqual(snapshot["workers"][0]["workflow_action"], "stage.a07.start")
        self.assertTrue(snapshot["workers"][0]["session_exists"])

    def test_runtime_scanned_running_worker_snapshots_mark_dead_when_session_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            worker_root = runtime_root / "worker-1"
            worker_root.mkdir(parents=True)
            state_path = worker_root / "worker.state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "session_name": "sess-missing",
                        "work_dir": "/tmp/project",
                        "status": "running",
                        "workflow_stage": "turn_running",
                        "agent_state": "READY",
                        "health_status": "alive",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime = SimpleNamespace(session_exists=lambda name: False)  # noqa: SLF001
            workers = server._scan_runtime_workers(runtime_root)  # noqa: SLF001
        self.assertFalse(workers[0]["session_exists"])
        self.assertEqual(workers[0]["agent_state"], "DEAD")

    def test_runtime_scanned_worker_snapshot_rejects_tmux_session_from_other_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            project_dir = runtime_root / "project-a"
            worker_root = runtime_root / "worker-1"
            worker_root.mkdir(parents=True)
            state_path = worker_root / "worker.state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "session_name": "shared-session",
                        "work_dir": str(project_dir),
                        "project_dir": str(project_dir),
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a08.start",
                        "status": "running",
                        "workflow_stage": "turn_running",
                        "agent_state": "READY",
                        "health_status": "alive",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            class FakeTmuxRuntime:
                backend = None

                def session_exists(self, name):
                    return name == "shared-session"

                def session_matches_worker_state(self, name, state, state_path):
                    return False

            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime = FakeTmuxRuntime()  # noqa: SLF001
            workers = server._scan_runtime_workers(runtime_root)  # noqa: SLF001

        self.assertFalse(workers[0]["session_exists"])
        self.assertEqual(workers[0]["agent_state"], "DEAD")

    def test_runtime_scanned_busy_worker_keeps_live_session_when_context_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            project_dir = runtime_root / "project-a"
            worker_root = runtime_root / "worker-1"
            worker_root.mkdir(parents=True)
            state_path = worker_root / "worker.state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "session_name": "shared-session",
                        "work_dir": str(project_dir),
                        "project_dir": str(project_dir),
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a08.start",
                        "status": "running",
                        "result_status": "running",
                        "workflow_stage": "turn_running",
                        "agent_state": "BUSY",
                        "current_task_runtime_status": "running",
                        "health_status": "alive",
                        "health_note": "alive",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            class FakeTmuxRuntime:
                backend = None

                def session_exists(self, name):
                    return name == "shared-session"

                def session_matches_worker_state(self, name, state, state_path):
                    return False

            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime = FakeTmuxRuntime()  # noqa: SLF001
            workers = server._scan_runtime_workers(runtime_root)  # noqa: SLF001

        self.assertTrue(workers[0]["session_exists"])
        self.assertEqual(workers[0]["agent_state"], "BUSY")
        self.assertEqual(workers[0]["health_status"], "alive")

    def test_runtime_scanned_busy_worker_still_marks_dead_when_session_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            project_dir = runtime_root / "project-a"
            worker_root = runtime_root / "worker-1"
            worker_root.mkdir(parents=True)
            state_path = worker_root / "worker.state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "session_name": "shared-session",
                        "work_dir": str(project_dir),
                        "project_dir": str(project_dir),
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a08.start",
                        "status": "running",
                        "result_status": "running",
                        "workflow_stage": "turn_running",
                        "agent_state": "BUSY",
                        "current_task_runtime_status": "running",
                        "health_status": "alive",
                        "health_note": "alive",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            class FakeTmuxRuntime:
                backend = None

                def session_exists(self, name):
                    return False

                def session_matches_worker_state(self, name, state, state_path):
                    return False

            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime = FakeTmuxRuntime()  # noqa: SLF001
            workers = server._scan_runtime_workers(runtime_root)  # noqa: SLF001

        self.assertFalse(workers[0]["session_exists"])
        self.assertEqual(workers[0]["agent_state"], "DEAD")
        self.assertEqual(workers[0]["health_status"], "dead")

    def test_runtime_scanned_running_worker_snapshots_refresh_health_via_tmux_runtime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            worker_root = runtime_root / "worker-1"
            worker_root.mkdir(parents=True)
            state_path = worker_root / "worker.state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "worker_id": "requirements-review-r3",
                        "session_name": "sess-runtime",
                        "pane_id": "%1",
                        "work_dir": str(runtime_root),
                        "status": "running",
                        "workflow_stage": "turn_running",
                        "agent_state": "READY",
                        "health_status": "alive",
                        "health_note": "alive",
                        "config": {
                            "vendor": "gemini",
                            "model": "pro",
                            "resolved_model": "pro",
                            "reasoning_effort": "high",
                            "proxy_url": "",
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            class _RefreshedWorker:
                def refresh_health(self, **kwargs) -> None:
                    self.kwargs = kwargs
                    payload = json.loads(state_path.read_text(encoding="utf-8"))
                    payload["agent_state"] = "BUSY"
                    payload["health_note"] = "alive"
                    payload["last_heartbeat_at"] = "2026-04-20T16:00:00+08:00"
                    state_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime = SimpleNamespace(  # noqa: SLF001
                session_exists=lambda name: name == "sess-runtime",
                backend=object(),
            )
            with patch("tmux_core.bridge.backend.load_worker_from_state_path", return_value=_RefreshedWorker()):
                workers = server._scan_runtime_workers(runtime_root)  # noqa: SLF001

        self.assertEqual(workers[0]["agent_state"], "BUSY")
        self.assertEqual(workers[0]["health_note"], "alive")
        self.assertEqual(workers[0]["vendor"], "gemini")
        self.assertEqual(workers[0]["model"], "pro")
        self.assertEqual(workers[0]["resolved_model"], "pro")
        self.assertEqual(workers[0]["reasoning_effort"], "high")

    def test_runtime_scanned_stale_dead_worker_refreshes_when_session_is_live(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            worker_root = runtime_root / "worker-1"
            worker_root.mkdir(parents=True)
            state_path = worker_root / "worker.state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "worker_id": "development-developer",
                        "session_name": "sess-runtime",
                        "pane_id": "%1",
                        "work_dir": str(runtime_root),
                        "status": "failed",
                        "result_status": "failed",
                        "workflow_stage": "turn_running",
                        "agent_state": "DEAD",
                        "health_status": "dead",
                        "health_note": "generic turn error",
                        "config": {
                            "vendor": "codex",
                            "model": "gpt-5.4",
                            "reasoning_effort": "high",
                            "proxy_url": "",
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            class _RefreshedWorker:
                def refresh_health(self, **kwargs) -> None:  # noqa: ANN003
                    payload = json.loads(state_path.read_text(encoding="utf-8"))
                    payload["agent_state"] = "READY"
                    payload["health_status"] = "alive"
                    payload["health_note"] = "alive"
                    payload["last_heartbeat_at"] = "2026-04-20T16:00:00+08:00"
                    state_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime = SimpleNamespace(  # noqa: SLF001
                session_exists=lambda name: name == "sess-runtime",
                backend=object(),
            )
            with patch("tmux_core.bridge.backend.load_worker_from_state_path", return_value=_RefreshedWorker()):
                workers = server._scan_runtime_workers(runtime_root)  # noqa: SLF001

        self.assertTrue(workers[0]["session_exists"])
        self.assertEqual(workers[0]["agent_state"], "READY")
        self.assertEqual(workers[0]["health_status"], "alive")

    def test_runtime_scan_does_not_infer_busy_from_current_command_when_agent_state_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            worker_root = runtime_root / "worker-1"
            worker_root.mkdir(parents=True)
            (worker_root / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "requirements-review-r3",
                        "session_name": "sess-runtime",
                        "pane_id": "%1",
                        "work_dir": str(runtime_root),
                        "status": "running",
                        "workflow_stage": "turn_running",
                        "agent_state": "not-a-runtime-state",
                        "agent_alive": True,
                        "agent_started": True,
                        "current_command": "codex",
                        "health_status": "alive",
                        "health_note": "alive",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime = SimpleNamespace(  # noqa: SLF001
                session_exists=lambda name: name == "sess-runtime",
                backend=None,
            )

            workers = server._scan_runtime_workers(runtime_root)  # noqa: SLF001

        self.assertEqual(workers[0]["agent_state"], "")

    def test_runtime_scan_refresh_disables_reentrant_runtime_notifications(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            worker_root = runtime_root / "worker-1"
            worker_root.mkdir(parents=True)
            state_path = worker_root / "worker.state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "worker_id": "requirements-review-r3",
                        "session_name": "sess-runtime",
                        "pane_id": "%1",
                        "work_dir": str(runtime_root),
                        "status": "running",
                        "workflow_stage": "turn_running",
                        "health_status": "alive",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            class _RefreshedWorker:
                def __init__(self) -> None:
                    self.kwargs = {}

                def refresh_health(self, **kwargs) -> None:
                    self.kwargs = kwargs

            worker = _RefreshedWorker()
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime = SimpleNamespace(  # noqa: SLF001
                session_exists=lambda name: name == "sess-runtime",
                backend=object(),
            )
            with patch("tmux_core.bridge.backend.load_worker_from_state_path", return_value=worker):
                server._scan_runtime_workers(runtime_root)  # noqa: SLF001

        self.assertEqual(worker.kwargs, {"notify_on_change": False})

    def test_runtime_scanned_worker_snapshots_keep_alive_health_note_for_ready_agent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            worker_root = runtime_root / "worker-1"
            worker_root.mkdir(parents=True)
            state_path = worker_root / "worker.state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "session_name": "sess-reviewer",
                        "work_dir": "/tmp/project",
                        "result_status": "succeeded",
                        "workflow_stage": "turn_done",
                        "agent_state": "READY",
                        "health_status": "alive",
                        "health_note": "alive",
                        "updated_at": "2026-04-17T10:00:00+08:00",
                        "last_heartbeat_at": "2026-04-17T10:00:02+08:00",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime = SimpleNamespace(session_exists=lambda name: name == "sess-reviewer")  # noqa: SLF001
            workers = server._scan_runtime_workers(runtime_root)  # noqa: SLF001
        self.assertEqual(workers[0]["agent_state"], "READY")
        self.assertEqual(workers[0]["health_note"], "alive")

    def test_runtime_scanned_worker_snapshots_keep_running_ready_agent_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            worker_root = runtime_root / "worker-1"
            worker_root.mkdir(parents=True)
            state_path = worker_root / "worker.state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "session_name": "sess-reviewer",
                        "work_dir": "/tmp/project",
                        "result_status": "running",
                        "workflow_stage": "turn_running",
                        "agent_state": "READY",
                        "agent_started": True,
                        "agent_alive": True,
                        "current_command": "gemini",
                        "current_task_runtime_status": "running",
                        "health_status": "alive",
                        "health_note": "alive",
                        "updated_at": "2026-04-17T10:00:00+08:00",
                        "last_heartbeat_at": "2026-04-17T10:00:02+08:00",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime = SimpleNamespace(session_exists=lambda name: name == "sess-reviewer", backend=None)  # noqa: SLF001
            workers = server._scan_runtime_workers(runtime_root)  # noqa: SLF001
        self.assertEqual(workers[0]["agent_state"], "READY")
        self.assertEqual(workers[0]["health_note"], "alive")

    def test_runtime_scanned_worker_snapshot_normalizes_stale_running_when_agent_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            worker_root = runtime_root / "worker-1"
            worker_root.mkdir(parents=True)
            state_path = worker_root / "worker.state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "session_name": "sess-reviewer",
                        "work_dir": "/tmp/project",
                        "status": "ready",
                        "result_status": "running",
                        "workflow_stage": "pending",
                        "agent_state": "READY",
                        "agent_started": True,
                        "agent_alive": True,
                        "current_command": "gemini",
                        "current_task_runtime_status": "running",
                        "health_status": "alive",
                        "health_note": "alive",
                        "note": "agent_ready",
                        "updated_at": "2026-04-17T10:00:00+08:00",
                        "last_heartbeat_at": "2026-04-17T10:00:02+08:00",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime = SimpleNamespace(session_exists=lambda name: name == "sess-reviewer", backend=None)  # noqa: SLF001
            workers = server._scan_runtime_workers(runtime_root)  # noqa: SLF001
        self.assertEqual(workers[0]["status"], "ready")
        self.assertEqual(workers[0]["agent_state"], "READY")
        self.assertEqual(workers[0]["current_task_runtime_status"], "")

    def test_runtime_scanned_worker_snapshots_prefer_fresher_alive_health_note_while_turn_is_running(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            worker_root = runtime_root / "worker-1"
            worker_root.mkdir(parents=True)
            state_path = worker_root / "worker.state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "session_name": "sess-active",
                        "work_dir": "/tmp/project",
                        "result_status": "running",
                        "workflow_stage": "turn_running",
                        "agent_state": "BUSY",
                        "health_status": "alive",
                        "health_note": "alive",
                        "updated_at": "2026-04-17T10:00:00+08:00",
                        "last_heartbeat_at": "2026-04-17T10:00:03+08:00",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime = SimpleNamespace(session_exists=lambda name: name == "sess-active")  # noqa: SLF001
            workers = server._scan_runtime_workers(runtime_root)  # noqa: SLF001
        self.assertEqual(workers[0]["agent_state"], "BUSY")
        self.assertEqual(workers[0]["health_note"], "alive")

    def test_runtime_scanned_worker_snapshots_keep_newer_alive_health_note_while_turn_is_running(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            worker_root = runtime_root / "worker-1"
            worker_root.mkdir(parents=True)
            state_path = worker_root / "worker.state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "session_name": "sess-active",
                        "work_dir": "/tmp/project",
                        "result_status": "running",
                        "workflow_stage": "turn_running",
                        "agent_state": "BUSY",
                        "health_status": "alive",
                        "health_note": "alive",
                        "updated_at": "2026-04-17T10:00:03+08:00",
                        "last_heartbeat_at": "2026-04-17T10:00:00+08:00",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._tmux_runtime = SimpleNamespace(session_exists=lambda name: name == "sess-active")  # noqa: SLF001
            workers = server._scan_runtime_workers(runtime_root)  # noqa: SLF001
        self.assertEqual(workers[0]["agent_state"], "BUSY")
        self.assertEqual(workers[0]["health_note"], "alive")

    def test_requirements_snapshot_prefers_latest_failed_worker_when_session_name_reused(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            runtime_root = project_dir / ".requirements_clarification_runtime"
            older_root = runtime_root / "worker-old"
            newer_root = runtime_root / "worker-new"
            older_root.mkdir(parents=True)
            newer_root.mkdir(parents=True)
            (older_root / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "分析师-参水猿",
                        "work_dir": str(project_dir),
                        "result_status": "succeeded",
                        "workflow_stage": "pending",
                        "agent_state": "READY",
                        "health_status": "alive",
                        "note": "done:requirements_clarification_round_2",
                        "updated_at": "2026-04-20T14:04:58+08:00",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (newer_root / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "分析师-参水猿",
                        "work_dir": str(project_dir),
                        "result_status": "failed",
                        "workflow_stage": "pending",
                        "agent_state": "DEAD",
                        "health_status": "alive",
                        "note": "error:requirements_clarification_round_3",
                        "updated_at": "2026-04-20T14:06:58+08:00",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), requirement_name="基金数据生成器", action="stage.a03.start")  # noqa: SLF001

            snapshot = server._build_requirements_snapshot()  # noqa: SLF001

        self.assertEqual(len(snapshot["workers"]), 1)
        self.assertEqual(snapshot["workers"][0]["session_name"], "分析师-参水猿")
        self.assertEqual(snapshot["workers"][0]["status"], "failed")
        self.assertEqual(snapshot["workers"][0]["note"], "error:requirements_clarification_round_3")

    def test_runtime_state_change_surfaces_failed_requirements_worker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            runtime_root = project_dir / ".requirements_clarification_runtime" / "worker-latest"
            runtime_root.mkdir(parents=True)
            (runtime_root / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "分析师-参水猿",
                        "work_dir": str(project_dir),
                        "result_status": "failed",
                        "workflow_stage": "pending",
                        "agent_state": "DEAD",
                        "health_status": "alive",
                        "note": "error:requirements_clarification_round_3",
                        "updated_at": "2026-04-20T14:06:58+08:00",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)
            server._set_context(project_dir=str(project_dir), requirement_name="基金数据生成器", action="stage.a03.start")  # noqa: SLF001
            server._display_status = "completed"  # noqa: SLF001

            server._bridge_ui.notify_runtime_state_changed()  # noqa: SLF001
            server._flush_dirty_snapshots()  # noqa: SLF001
            messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]

        stage_events = [item for item in messages if item.get("kind") == "event" and item.get("type") == "stage.changed"]
        self.assertTrue(stage_events)
        self.assertEqual(stage_events[-1]["payload"]["action"], "stage.a03.start")
        self.assertEqual(stage_events[-1]["payload"]["status"], "failed")

    def test_stage_a03_status_is_not_polluted_by_failed_requirement_intake_worker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            runtime_root = project_dir / NOTION_RUNTIME_ROOT_NAME / "worker-failed"
            runtime_root.mkdir(parents=True)
            (runtime_root / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "录入-危月燕",
                        "work_dir": str(project_dir),
                        "result_status": "failed",
                        "workflow_stage": "pending",
                        "agent_state": "DEAD",
                        "health_status": "alive",
                        "note": "error:notion_round_1",
                        "updated_at": "2026-04-20T14:06:58+08:00",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), requirement_name="基金数据生成器", action="stage.a03.start")  # noqa: SLF001

            a02_status = server._infer_runtime_stage_status("stage.a02.start")  # noqa: SLF001
            a03_status = server._infer_runtime_stage_status("stage.a03.start")  # noqa: SLF001

        self.assertEqual(a02_status, "failed")
        self.assertEqual(a03_status, "")

    def test_routing_snapshot_uses_latest_run_workers_without_active_control_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir(parents=True)
            runtime_root = build_routing_runtime_root(project_dir)
            for file_path in required_routing_layer_paths(project_dir):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("ok", encoding="utf-8")

            run_root = runtime_root / "run_demo"
            run_root.mkdir(parents=True)
            (run_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "manifest_version": 1,
                        "run_id": "run_demo",
                        "runtime_dir": str(run_root),
                        "project_dir": str(project_dir),
                        "selection": {"project_dir": str(project_dir), "selected_dirs": [], "skipped_dirs": [], "forced_dirs": [], "project_missing_files": []},
                        "config": {"vendor": "codex", "model": "gpt-5.4", "reasoning_effort": "high", "proxy_url": ""},
                        "status": "running",
                        "created_at": "2026-04-16T10:00:00",
                        "updated_at": "2026-04-16T10:00:00",
                        "workers": [
                            {
                                "work_dir": str(project_dir),
                                "session_name": "sess-routing",
                                "workflow_stage": "create_running",
                                "result_status": "running",
                                "agent_state": "READY",
                                "health_status": "alive",
                                "current_task_runtime_status": "running",
                                "current_turn_status_path": str(run_root / "turn_status.json"),
                                "state_path": "",
                                "transcript_path": "",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), action="workflow.a00.start")  # noqa: SLF001
            server._tmux_runtime = SimpleNamespace(session_exists=lambda name: name == "sess-routing")  # noqa: SLF001
            snapshot = server._build_routing_snapshot()  # noqa: SLF001
        self.assertEqual(snapshot["workers"][0]["session_name"], "sess-routing")
        self.assertTrue(snapshot["workers"][0]["session_exists"])
        self.assertEqual(snapshot["workers"][0]["agent_state"], "BUSY")
        self.assertEqual(snapshot["workers"][0]["current_task_runtime_status"], "running")

    def test_manifest_backed_prelaunch_routing_worker_with_missing_session_stays_starting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir(parents=True)
            runtime_root = build_routing_runtime_root(project_dir)
            for file_path in required_routing_layer_paths(project_dir):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("ok", encoding="utf-8")

            run_root = runtime_root / "run_demo"
            run_root.mkdir(parents=True)
            (run_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "manifest_version": 1,
                        "run_id": "run_demo",
                        "runtime_dir": str(run_root),
                        "project_dir": str(project_dir),
                        "selection": {"project_dir": str(project_dir), "selected_dirs": [], "skipped_dirs": [], "forced_dirs": [], "project_missing_files": []},
                        "config": {"vendor": "codex", "model": "gpt-5.4", "reasoning_effort": "high", "proxy_url": ""},
                        "status": "running",
                        "created_at": "2026-04-16T10:00:00",
                        "updated_at": "2026-04-16T10:00:00",
                        "workers": [
                            {
                                "work_dir": str(project_dir),
                                "session_name": "sess-routing-prelaunch",
                                "workflow_stage": "pending",
                                "result_status": "pending",
                                "agent_state": "STARTING",
                                "agent_alive": False,
                                "agent_started": False,
                                "health_status": "unknown",
                                "state_path": "",
                                "transcript_path": "",
                                "note": "worker_prepared",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), action="stage.a01.start")  # noqa: SLF001
            server._tmux_runtime = SimpleNamespace(session_exists=lambda name: False)  # noqa: SLF001

            snapshot = server._build_routing_snapshot()  # noqa: SLF001
            status = server._infer_runtime_stage_status("stage.a01.start")  # noqa: SLF001

        self.assertFalse(snapshot["workers"][0]["session_exists"])
        self.assertEqual(snapshot["workers"][0]["agent_state"], "STARTING")
        self.assertEqual(snapshot["workers"][0]["health_status"], "unknown")
        self.assertEqual(status, "running")

    def test_manifest_backed_active_prelaunch_routing_worker_with_missing_session_stays_starting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir(parents=True)
            runtime_root = build_routing_runtime_root(project_dir)
            for file_path in required_routing_layer_paths(project_dir):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("ok", encoding="utf-8")

            run_root = runtime_root / "run_demo"
            run_root.mkdir(parents=True)
            (run_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "manifest_version": 1,
                        "run_id": "run_demo",
                        "runtime_dir": str(run_root),
                        "project_dir": str(project_dir),
                        "selection": {"project_dir": str(project_dir), "selected_dirs": [], "skipped_dirs": [], "forced_dirs": [], "project_missing_files": []},
                        "config": {"vendor": "codex", "model": "gpt-5.4", "reasoning_effort": "high", "proxy_url": ""},
                        "status": "running",
                        "created_at": "2026-04-16T10:00:00",
                        "updated_at": "2026-04-16T10:00:00",
                        "workers": [
                            {
                                "work_dir": str(project_dir),
                                "session_name": "sess-routing-launching",
                                "workflow_stage": "create_running",
                                "result_status": "running",
                                "agent_state": "DEAD",
                                "agent_alive": False,
                                "agent_started": False,
                                "health_status": "dead",
                                "health_note": "missing_session",
                                "state_path": "",
                                "transcript_path": "",
                                "note": "create_routing_layer",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), action="stage.a01.start")  # noqa: SLF001
            server._tmux_runtime = SimpleNamespace(session_exists=lambda name: False)  # noqa: SLF001

            snapshot = server._build_routing_snapshot()  # noqa: SLF001
            status = server._infer_runtime_stage_status("stage.a01.start")  # noqa: SLF001

        self.assertFalse(snapshot["workers"][0]["session_exists"])
        self.assertEqual(snapshot["workers"][0]["workflow_stage"], "create_running")
        self.assertEqual(snapshot["workers"][0]["agent_state"], "STARTING")
        self.assertEqual(snapshot["workers"][0]["health_status"], "unknown")
        self.assertEqual(snapshot["workers"][0]["health_note"], "launch pending")
        self.assertEqual(status, "running")

    def test_manifest_backed_busy_routing_worker_marks_stage_running(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir(parents=True)
            runtime_root = build_routing_runtime_root(project_dir)
            for file_path in required_routing_layer_paths(project_dir):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("ok", encoding="utf-8")

            run_root = runtime_root / "run_demo"
            run_root.mkdir(parents=True)
            (run_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "manifest_version": 1,
                        "run_id": "run_demo",
                        "runtime_dir": str(run_root),
                        "project_dir": str(project_dir),
                        "selection": {"project_dir": str(project_dir), "selected_dirs": [], "skipped_dirs": [], "forced_dirs": [], "project_missing_files": []},
                        "config": {"vendor": "codex", "model": "gpt-5.4", "reasoning_effort": "high", "proxy_url": ""},
                        "status": "running",
                        "created_at": "2026-04-16T10:00:00",
                        "updated_at": "2026-04-16T10:00:00",
                        "workers": [
                            {
                                "work_dir": str(project_dir),
                                "session_name": "sess-routing-busy",
                                "workflow_stage": "create_running",
                                "result_status": "running",
                                "agent_state": "BUSY",
                                "agent_alive": True,
                                "agent_started": True,
                                "health_status": "alive",
                                "state_path": "",
                                "transcript_path": "",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), action="stage.a01.start")  # noqa: SLF001
            server._tmux_runtime = SimpleNamespace(session_exists=lambda name: name == "sess-routing-busy")  # noqa: SLF001

            status = server._infer_runtime_stage_status("stage.a01.start")  # noqa: SLF001

        self.assertEqual(status, "running")

    def test_completed_routing_contract_ignores_manifest_only_missing_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir(parents=True)
            runtime_root = build_routing_runtime_root(project_dir)
            for file_path in required_routing_layer_paths(project_dir):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("ok", encoding="utf-8")

            run_root = runtime_root / "run_demo"
            run_root.mkdir(parents=True)
            (run_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "manifest_version": 1,
                        "run_id": "run_demo",
                        "runtime_dir": str(run_root),
                        "project_dir": str(project_dir),
                        "selection": {"project_dir": str(project_dir), "selected_dirs": [], "skipped_dirs": [], "forced_dirs": [], "project_missing_files": []},
                        "config": {"vendor": "codex", "model": "gpt-5.4", "reasoning_effort": "high", "proxy_url": ""},
                        "status": "running",
                        "created_at": "2026-04-16T10:00:00",
                        "updated_at": "2026-04-16T10:00:00",
                        "workers": [
                            {
                                "work_dir": str(project_dir),
                                "session_name": "sess-routing-cleaned",
                                "workflow_stage": "create_running",
                                "result_status": "running",
                                "agent_state": "READY",
                                "health_status": "alive",
                                "state_path": "",
                                "transcript_path": "",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), action="stage.a01.start")  # noqa: SLF001
            server._tmux_runtime = SimpleNamespace(session_exists=lambda name: False)  # noqa: SLF001

            status = server._infer_runtime_stage_status("stage.a01.start")  # noqa: SLF001
            action, display_status, _stage_seq = server._derive_display_stage_state(  # noqa: SLF001
                preferred_status="completed",
                preferred_action="stage.a01.start",
            )

        self.assertEqual(status, "")
        self.assertEqual(action, "stage.a01.start")
        self.assertEqual(display_status, "completed")

    def test_manifest_backed_running_worker_snapshot_marks_dead_when_session_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir(parents=True)
            runtime_root = build_routing_runtime_root(project_dir)
            for file_path in required_routing_layer_paths(project_dir):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("ok", encoding="utf-8")

            run_root = runtime_root / "run_demo"
            run_root.mkdir(parents=True)
            (run_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "manifest_version": 1,
                        "run_id": "run_demo",
                        "runtime_dir": str(run_root),
                        "project_dir": str(project_dir),
                        "selection": {"project_dir": str(project_dir), "selected_dirs": [], "skipped_dirs": [], "forced_dirs": [], "project_missing_files": []},
                        "config": {"vendor": "codex", "model": "gpt-5.4", "reasoning_effort": "high", "proxy_url": ""},
                        "status": "running",
                        "created_at": "2026-04-16T10:00:00",
                        "updated_at": "2026-04-16T10:00:00",
                        "workers": [
                            {
                                "work_dir": str(project_dir),
                                "session_name": "sess-routing-dead",
                                "workflow_stage": "create_running",
                                "result_status": "running",
                                "agent_state": "READY",
                                "health_status": "alive",
                                "state_path": "",
                                "transcript_path": "",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), action="workflow.a00.start")  # noqa: SLF001
            server._tmux_runtime = SimpleNamespace(session_exists=lambda name: False)  # noqa: SLF001

            snapshot = server._build_routing_snapshot()  # noqa: SLF001

        self.assertEqual(snapshot["workers"][0]["session_name"], "sess-routing-dead")
        self.assertFalse(snapshot["workers"][0]["session_exists"])
        self.assertEqual(snapshot["workers"][0]["agent_state"], "DEAD")
        self.assertEqual(snapshot["workers"][0]["health_status"], "dead")
        self.assertEqual(snapshot["workers"][0]["health_note"], "tmux session missing")

    def test_routing_setup_prompt_suppresses_manifest_only_dead_workers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir(parents=True)
            runtime_root = build_routing_runtime_root(project_dir)
            for file_path in required_routing_layer_paths(project_dir):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("ok", encoding="utf-8")

            run_root = runtime_root / "run_demo"
            run_root.mkdir(parents=True)
            (run_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "manifest_version": 1,
                        "run_id": "run_demo",
                        "runtime_dir": str(run_root),
                        "project_dir": str(project_dir),
                        "selection": {"project_dir": str(project_dir), "selected_dirs": [], "skipped_dirs": [], "forced_dirs": [], "project_missing_files": []},
                        "config": {"vendor": "codex", "model": "gpt-5.4", "reasoning_effort": "high", "proxy_url": ""},
                        "status": "running",
                        "created_at": "2026-04-16T10:00:00",
                        "updated_at": "2026-04-16T10:00:00",
                        "workers": [
                            {
                                "work_dir": str(project_dir),
                                "session_name": "sess-routing-dead",
                                "workflow_stage": "create_running",
                                "result_status": "running",
                                "agent_state": "READY",
                                "health_status": "alive",
                                "state_path": "",
                                "transcript_path": "",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
            server._set_context(project_dir=str(project_dir), action="stage.a01.start")  # noqa: SLF001
            server._tmux_runtime = SimpleNamespace(session_exists=lambda name: False)  # noqa: SLF001
            server._pending_prompt = PendingPromptState(  # noqa: SLF001
                prompt_id="prompt-routing",
                prompt_type="select",
                payload={
                    "stage_key": "routing",
                    "stage_step_index": 1,
                    "prompt_text": "是否执行 AGENT初始化",
                    "options": [{"value": "yes", "label": "yes"}, {"value": "no", "label": "no"}],
                },
            )

            snapshot = server._build_routing_snapshot()  # noqa: SLF001

        self.assertEqual(snapshot["workers"], [])

    def test_routing_skip_prompt_resolution_clears_manifest_only_workers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir(parents=True)
            runtime_root = build_routing_runtime_root(project_dir)
            for file_path in required_routing_layer_paths(project_dir):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("ok", encoding="utf-8")

            run_root = runtime_root / "run_demo"
            run_root.mkdir(parents=True)
            (run_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "manifest_version": 1,
                        "run_id": "run_demo",
                        "runtime_dir": str(run_root),
                        "project_dir": str(project_dir),
                        "selection": {"project_dir": str(project_dir), "selected_dirs": [], "skipped_dirs": [], "forced_dirs": [], "project_missing_files": []},
                        "config": {"vendor": "codex", "model": "gpt-5.4", "reasoning_effort": "high", "proxy_url": ""},
                        "status": "running",
                        "created_at": "2026-04-16T10:00:00",
                        "updated_at": "2026-04-16T10:00:00",
                        "workers": [
                            {
                                "work_dir": str(project_dir),
                                "session_name": "sess-routing-dead",
                                "workflow_stage": "create_running",
                                "result_status": "running",
                                "agent_state": "READY",
                                "health_status": "alive",
                                "state_path": "",
                                "transcript_path": "",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)
            server._set_context(project_dir=str(project_dir), action="stage.a01.start")  # noqa: SLF001
            server._tmux_runtime = SimpleNamespace(session_exists=lambda name: False)  # noqa: SLF001
            server._pending_prompts["prompt-routing"] = PendingPromptState(  # noqa: SLF001
                prompt_id="prompt-routing",
                prompt_type="select",
                payload={
                    "stage_key": "routing",
                    "stage_step_index": 1,
                    "prompt_text": "是否执行 AGENT初始化",
                    "options": [{"value": "yes", "label": "yes"}, {"value": "no", "label": "no"}],
                },
            )
            server._pending_prompt = server._pending_prompts["prompt-routing"]  # noqa: SLF001

            server._handle_prompt_resolved("prompt-routing", {"value": "no"})  # noqa: SLF001
            server._flush_dirty_snapshots()  # noqa: SLF001
            snapshot = server._build_routing_snapshot()  # noqa: SLF001
            events = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]

        self.assertEqual(snapshot["workers"], [])
        self.assertIsNone(server._pending_prompt)  # noqa: SLF001
        self.assertEqual(server._pending_prompts, {})  # noqa: SLF001
        self.assertTrue(
            any(
                item.get("kind") == "event"
                and item.get("type") == "snapshot.stage"
                and item.get("payload", {}).get("route") == "routing"
                and item.get("payload", {}).get("snapshot", {}).get("workers") == []
                for item in events
            )
        )

    def test_bridge_ui_runtime_state_change_notifier_debounces_app_and_current_stage_snapshots(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        server._set_context(action="stage.a03.start")  # noqa: SLF001
        server._bridge_ui.notify_runtime_state_changed()  # noqa: SLF001
        server._flush_dirty_snapshots()  # noqa: SLF001
        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        event_types = [item.get("type") for item in messages if item.get("kind") == "event"]
        self.assertIn("snapshot.app", event_types)
        self.assertIn("snapshot.stage", event_types)
        self.assertIn("snapshot.control", event_types)

    def test_bridge_ui_runtime_state_change_refreshes_requirements_worker_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            runtime_root = project_dir / ".requirements_analysis_runtime" / "requirements-analyst-demo"
            runtime_root.mkdir(parents=True)
            (runtime_root / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "sess-requirements",
                        "work_dir": str(project_dir),
                        "status": "running",
                        "workflow_stage": "requirements_analysis",
                        "agent_state": "READY",
                        "health_status": "alive",
                        "retry_count": 0,
                        "note": "requirements_analysis_round_1",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)
            server._set_context(project_dir=str(project_dir), requirement_name="贪吃蛇", action="stage.a02.start")  # noqa: SLF001
            server._tmux_runtime = SimpleNamespace(session_exists=lambda session_name: session_name == "sess-requirements")  # noqa: SLF001
            server._bridge_ui.notify_runtime_state_changed()  # noqa: SLF001
            server._flush_dirty_snapshots()  # noqa: SLF001
            messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        requirement_snapshots = [
            item["payload"]["snapshot"]
            for item in messages
            if item.get("kind") == "event"
            and item.get("type") == "snapshot.stage"
            and item.get("payload", {}).get("route") == "requirements"
        ]
        self.assertTrue(requirement_snapshots)
        latest_snapshot = requirement_snapshots[-1]
        self.assertEqual(latest_snapshot["workers"][0]["session_name"], "sess-requirements")
        self.assertTrue(latest_snapshot["workers"][0]["session_exists"])

    def test_runtime_state_change_emits_hitl_snapshot_when_worker_question_is_pending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            requirement_name = "需求A"
            paths = build_development_paths(project_dir, requirement_name)
            for path in (
                paths["task_md_path"],
                paths["task_json_path"],
                paths["developer_output_path"],
                paths["merged_review_path"],
                paths["detailed_design_path"],
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("ok\n", encoding="utf-8")
            paths["task_json_path"].write_text(json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False), encoding="utf-8")
            paths["ask_human_path"].write_text("请确认评审冲突\n", encoding="utf-8")
            runtime_dir = project_dir / DEVELOPMENT_RUNTIME_ROOT_NAME / "worker-hitl"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            (runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "development-developer",
                        "session_name": "开发工程师-天魁星",
                        "work_dir": str(project_dir),
                        "project_dir": str(project_dir),
                        "requirement_name": requirement_name,
                        "workflow_action": "stage.a07.start",
                        "result_status": "running",
                        "workflow_stage": "turn_running",
                        "agent_state": "READY",
                        "health_status": "alive",
                        "question_path": str(paths["ask_human_path"]),
                        "answer_path": str(paths["hitl_record_path"]),
                        "updated_at": "2026-04-23T10:00:00+08:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)
            server._set_context(project_dir=str(project_dir), requirement_name=requirement_name, action="stage.a07.start")  # noqa: SLF001
            server._bridge_ui.notify_runtime_state_changed()  # noqa: SLF001
            server._flush_dirty_snapshots()  # noqa: SLF001
            messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]

        hitl_events = [
            item["payload"]
            for item in messages
            if item.get("kind") == "event" and item.get("type") == "snapshot.hitl"
        ]
        self.assertTrue(hitl_events)
        self.assertTrue(hitl_events[-1]["pending"])
        self.assertEqual(hitl_events[-1]["question_path"], str(paths["ask_human_path"]))

    def test_app_recent_artifacts_use_cache_and_current_stage_when_stage_update_is_partial(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            routing_file = root / "routing.md"
            development_file = root / "development.md"
            routing_file.write_text("routing\n", encoding="utf-8")
            development_file.write_text("development\n", encoding="utf-8")
            files_by_route = {
                "routing": [{"path": str(routing_file)}],
                "development": [{"path": str(development_file)}],
            }
            built_routes: list[str] = []

            def fake_stage_snapshot(route: str) -> dict[str, object]:
                built_routes.append(route)
                return {"files": files_by_route.get(route, []), "workers": []}

            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)
            with patch.object(server, "_build_stage_snapshot_by_route", side_effect=fake_stage_snapshot):
                server._emit_snapshot_update(include_app=True, include_all_stages=True)  # noqa: SLF001
                built_routes.clear()
                server._emit_snapshot_update(include_app=True, stage_routes=("development",))  # noqa: SLF001
            messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]

        app_events = [
            item["payload"]
            for item in messages
            if item.get("kind") == "event" and item.get("type") == "snapshot.app"
        ]
        artifact_paths = {item["path"] for item in app_events[-1]["recent_artifacts"]}
        self.assertIn(str(routing_file.resolve()), artifact_paths)
        self.assertIn(str(development_file.resolve()), artifact_paths)
        self.assertEqual(built_routes, ["development"])

    def test_artifact_item_builder_filters_empty_and_missing_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            existing = Path(tmpdir) / "artifact.md"
            existing.write_text("artifact\n", encoding="utf-8")
            missing = Path(tmpdir) / "missing.md"

            items = TuiBackendServer._artifact_items_from_candidates(["", str(missing), str(existing)])  # noqa: SLF001

        self.assertEqual([item["path"] for item in items], [str(existing.resolve())])

    def test_artifact_index_is_scoped_by_project_and_requirement(self):
        with tempfile.TemporaryDirectory() as project_a, tempfile.TemporaryDirectory() as project_b:
            old_file = Path(project_a) / "old.md"
            new_file = Path(project_b) / "new.md"
            old_file.write_text("old\n", encoding="utf-8")
            new_file.write_text("new\n", encoding="utf-8")
            server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())

            server._set_context(project_dir=project_a, requirement_name="req-a")  # noqa: SLF001
            seeded = server._build_artifacts_snapshot(  # noqa: SLF001
                stages={"development": {"files": [{"path": str(old_file)}]}},
                control={"workers": []},
            )
            self.assertEqual([item["path"] for item in seeded["items"]], [str(old_file.resolve())])

            server._set_context(project_dir=project_b, requirement_name="req-b")  # noqa: SLF001
            current = server._build_artifacts_snapshot(  # noqa: SLF001
                stages={"development": {"files": [{"path": str(new_file)}]}},
                control={"workers": []},
            )
            partial = server._build_artifacts_snapshot(stages={}, control={"workers": []})  # noqa: SLF001

        self.assertEqual([item["path"] for item in current["items"]], [str(new_file.resolve())])
        self.assertEqual([item["path"] for item in partial["items"]], [str(new_file.resolve())])

    def test_snapshot_stage_registry_rejects_unknown_and_dedupes_routes(self):
        server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
        with self.assertRaises(KeyError):
            server._build_stage_snapshot_by_route("missing")  # noqa: SLF001

        with patch.object(server, "_build_routing_snapshot", return_value={"files": [], "workers": []}) as builder:
            snapshots = server._build_stage_snapshots(["routing", "routing"])  # noqa: SLF001

        self.assertEqual(list(snapshots), ["routing"])
        builder.assert_called_once()

    def test_snapshot_update_logs_builder_failures_and_continues_with_fallbacks(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        with patch.object(server, "_build_stage_snapshots", side_effect=RuntimeError("stage boom")), patch.object(
            server,
            "_build_control_snapshot_for_session",
            side_effect=RuntimeError("control boom"),
        ), patch.object(server, "_build_hitl_snapshot", side_effect=RuntimeError("hitl boom")), patch.object(
            server._attention_manager,
            "snapshot",
            side_effect=RuntimeError("attention boom"),
        ), patch.object(server, "_build_artifacts_snapshot", side_effect=RuntimeError("artifacts boom")):
            server._emit_snapshot_update(  # noqa: SLF001
                include_app=True,
                include_control=True,
                include_hitl=True,
                include_artifacts=True,
                stage_routes=("routing",),
            )

        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        log_text = "\n".join(str(item.get("payload", {}).get("text", "")) for item in messages if item.get("type") == "log.append")
        self.assertIn("stage boom", log_text)
        self.assertIn("control boom", log_text)
        self.assertIn("hitl boom", log_text)
        self.assertIn("attention boom", log_text)
        self.assertIn("artifacts boom", log_text)
        self.assertTrue(any(item.get("type") == "snapshot.app" for item in messages))

    def test_snapshot_dirty_scheduler_noops_empty_updates_and_shutdown_cancels_timer(self):
        server = TuiBackendServer(reader=io.StringIO(), writer=io.StringIO())
        server._schedule_snapshot_update(sections=set(), stage_routes=())  # noqa: SLF001
        self.assertIsNone(server._snapshot_debounce_timer)  # noqa: SLF001

        server._schedule_snapshot_update(sections={"app"}, delay_sec=10.0)  # noqa: SLF001
        self.assertIsNotNone(server._snapshot_debounce_timer)  # noqa: SLF001
        server._schedule_snapshot_update(sections={"hitl"}, delay_sec=10.0)  # noqa: SLF001
        self.assertEqual(server._snapshot_dirty_sections, {"app", "hitl"})  # noqa: SLF001
        server.shutdown(cleanup_tmux=False)
        self.assertIsNone(server._snapshot_debounce_timer)  # noqa: SLF001

    def test_worker_control_actions_emit_control_and_app_snapshots_only(self):
        for action in ("worker.detach", "worker.kill", "worker.restart", "worker.retry"):
            writer = io.StringIO()
            server = TuiBackendServer(reader=io.StringIO(), writer=writer)
            server._controls["run_demo"] = ControlSessionState(control_id="run_demo", center=_FakeCenter())  # noqa: SLF001
            server.handle_request(build_request(action, {"control_id": "run_demo", "argument": "1"}, message_id=f"req_{action}"))
            server._flush_dirty_snapshots()  # noqa: SLF001
            messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
            event_types = [item.get("type") for item in messages if item.get("kind") == "event"]
            self.assertIn("snapshot.app", event_types)
            self.assertIn("snapshot.control", event_types)
            self.assertNotIn("snapshot.stage", event_types)

    def test_emit_all_snapshots_uses_single_full_bundle_path(self):
        writer = io.StringIO()
        server = TuiBackendServer(reader=io.StringIO(), writer=writer)
        server._emit_all_snapshots()  # noqa: SLF001
        messages = [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]
        event_types = [item.get("type") for item in messages if item.get("kind") == "event"]
        self.assertIn("snapshot.app", event_types)
        self.assertIn("snapshot.control", event_types)
        self.assertIn("snapshot.hitl", event_types)
        self.assertIn("snapshot.artifacts", event_types)


if __name__ == "__main__":
    unittest.main()
