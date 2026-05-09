from __future__ import annotations

import io
import json
import subprocess
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

from tmux_core.runtime.vendor_catalog import get_default_model_for_vendor, get_model_choices
from T02_tmux_agents import (
    AgentRuntimeState,
    AgentRunConfig,
    GeminiOutputDetector,
    HealthSupervisor,
    OpenCodeOutputDetector,
    CodexOutputDetector,
    ClaudeOutputDetector,
    LaunchCoordinator,
    WrapperState,
    WorkerObservation,
    WorkerHealthSnapshot,
    WorkerStatus,
    TIMEOUT_EXIT_CODE,
    TaskResultContract,
    TmuxBackend,
    TmuxBatchWorker,
    TurnFileContract,
    TurnFileResult,
    TmuxRuntimeController,
    Vendor,
    TASK_RESULT_CONTRACT_ERROR_PREFIX,
    TURN_ARTIFACT_CONTRACT_ERROR_PREFIX,
    RuntimeShutdownRequested,
    cleanup_registered_tmux_workers,
    clear_runtime_shutdown_request,
    build_prompt_header,
    build_proxy_env,
    build_session_name,
    extract_final_protocol_token,
    is_provider_runtime_error,
    list_occupied_tmux_session_names,
    load_worker_from_state_path,
    normalize_proxy_url,
    request_runtime_shutdown,
    read_text_tail,
    TERMINAL_ACTIVITY_IDLE_WINDOW_SEC,
    _session_name_lease_lock,
)
from tmux_core.runtime.contracts import finalize_task_result, write_task_status
from tmux_core.runtime.tmux_runtime import (
    is_worker_death_error,
    worker_state_has_launch_evidence,
    worker_state_is_prelaunch_active,
)
from tmux_core.stage_kernel.role_orchestration import ensure_reviewers_ready


class TmuxAgentsTests(unittest.TestCase):
    @staticmethod
    def _health_snapshot(*, agent_state: str, health_status: str = "alive") -> WorkerHealthSnapshot:
        return WorkerHealthSnapshot(
            session_exists=True,
            health_status=health_status,
            health_note=health_status,
            agent_state=agent_state,
            pane_title="TmuxCodingTeam",
            last_heartbeat_at="2026-04-24T00:00:00",
            last_log_offset=0,
            current_command="codex",
            current_path="/tmp",
            pane_id="%1",
            session_name="demo",
        )

    def test_session_name_lease_lock_is_reentrant_in_process(self):
        with _session_name_lease_lock():
            with _session_name_lease_lock():
                pass

    def test_health_supervisor_uses_adaptive_intervals_and_stops_terminal_health(self):
        supervisor = HealthSupervisor(
            refresh_callback=lambda: None,
            interval_sec=2.0,
            ready_interval_sec=15.0,
            idle_interval_sec=30.0,
            idle_after_sec=0.0,
        )

        self.assertFalse(supervisor._update_next_interval(None))  # noqa: SLF001
        self.assertEqual(supervisor._next_interval_sec, 2.0)  # noqa: SLF001
        self.assertFalse(supervisor._update_next_interval(self._health_snapshot(agent_state="READY")))  # noqa: SLF001
        self.assertEqual(supervisor._next_interval_sec, 30.0)  # noqa: SLF001
        self.assertFalse(supervisor._update_next_interval(self._health_snapshot(agent_state="BUSY")))  # noqa: SLF001
        self.assertEqual(supervisor._next_interval_sec, 2.0)  # noqa: SLF001
        self.assertFalse(supervisor._update_next_interval(self._health_snapshot(agent_state="DEAD", health_status="missing_session")))  # noqa: SLF001
        self.assertTrue(supervisor._update_next_interval(self._health_snapshot(agent_state="DEAD", health_status="missing_session")))  # noqa: SLF001

    def test_health_supervisor_run_loop_stops_after_terminal_snapshot(self):
        calls = []

        def refresh():
            calls.append(True)
            return self._health_snapshot(agent_state="DEAD", health_status="missing_session")

        supervisor = HealthSupervisor(refresh_callback=refresh, interval_sec=0.01)
        supervisor.start()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not supervisor.stopped():
            time.sleep(0.01)

        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(supervisor.stopped())
        supervisor.stop()

    def test_task_runtime_paths_do_not_collide_when_labels_share_truncated_prefix(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="runtime-path-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4"),
                runtime_root=Path(tmp_dir) / "runtime",
            )

            original_path = worker._build_task_result_path(  # noqa: SLF001
                label="detailed_design_review_limit_human_reply",
                attempt=1,
            )
            repair_path = worker._build_task_result_path(  # noqa: SLF001
                label="detailed_design_review_limit_human_reply_repair_1",
                attempt=1,
            )

            self.assertNotEqual(original_path, repair_path)
            self.assertEqual(original_path.name.count("_attempt_1_result.json"), 1)
            self.assertEqual(repair_path.name.count("_attempt_1_result.json"), 1)

    def test_health_supervisor_can_restart_after_self_stop(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="health-restart-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            first = HealthSupervisor(
                refresh_callback=lambda: self._health_snapshot(agent_state="DEAD", health_status="missing_session"),
                interval_sec=0.01,
            )
            first.start()
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and not first.stopped():
                time.sleep(0.01)
            worker.health_supervisor = first

            worker._ensure_health_supervisor_started()  # noqa: SLF001

            self.assertIsNot(worker.health_supervisor, first)
            self.assertTrue(worker.health_supervisor.is_alive())
            worker._stop_health_supervisor()  # noqa: SLF001

    def test_health_supervisor_recovers_interval_after_refresh_exception(self):
        supervisor: HealthSupervisor

        def refresh():
            supervisor.stop()
            raise RuntimeError("probe failed")

        supervisor = HealthSupervisor(refresh_callback=refresh, interval_sec=0.01)
        supervisor.start()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not supervisor.stopped():
            time.sleep(0.01)

        self.assertTrue(supervisor.stopped())
        self.assertEqual(supervisor._next_interval_sec, 0.01)  # noqa: SLF001

    def test_normalize_proxy_url_accepts_port_or_url(self):
        self.assertEqual("http://127.0.0.1:7890", normalize_proxy_url("7890"))
        self.assertEqual("http://127.0.0.1:10809", normalize_proxy_url(10809))
        self.assertEqual("http://127.0.0.1:8899", normalize_proxy_url("127.0.0.1:8899"))
        self.assertEqual("https://proxy.example.com:8443", normalize_proxy_url("https://proxy.example.com:8443"))

    def test_build_proxy_env_populates_all_expected_keys(self):
        env = build_proxy_env("http://127.0.0.1:7890")
        self.assertEqual("http://127.0.0.1:7890", env["HTTP_PROXY"])
        self.assertEqual("socks5h://127.0.0.1:7890", env["all_proxy"])

    def test_normalize_proxy_url_accepts_explicit_socks_urls(self):
        self.assertEqual("socks5h://127.0.0.1:10900", normalize_proxy_url("socks5h://127.0.0.1:10900"))

    def test_prompt_header_contains_reasoning_note(self):
        header = build_prompt_header(Vendor.CODEX, "gpt-5.4-mini", "xhigh")
        self.assertIn("vendor: codex", header)
        self.assertIn("reasoning_effort=xhigh", header)
        self.assertIn("tmux_interactive_conversation", header)

    def test_removed_qwen_and_kimi_vendors_are_rejected(self):
        for vendor in ("qwen", "kimi"):
            with self.subTest(vendor=vendor), self.assertRaises(ValueError):
                AgentRunConfig(vendor=vendor, model="default")

    def test_provider_runtime_error_detection_covers_timeout_and_quota(self):
        self.assertTrue(is_provider_runtime_error("SSE read timed out while streaming"))
        self.assertTrue(is_provider_runtime_error("backend unavailable: quota exceeded"))
        self.assertFalse(is_provider_runtime_error("regular validation failure"))

    def test_provider_runtime_error_marks_worker_reconfigurable(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="provider-timeout-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4"),
                runtime_root=Path(tmp_dir) / "runtime",
            )

            worker.mark_provider_runtime_error(reason_text="SSE read timed out")
            state = worker.read_state()

            self.assertEqual(state["health_status"], "provider_runtime_error")
            self.assertEqual(state["last_provider_error"], "SSE read timed out")
            self.assertEqual(state["agent_state"], AgentRuntimeState.STARTING.value)

    def test_gemini_ready_detection_requires_input_box_marker(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="gemini-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="auto"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            not_ready_visible = """
Gemini CLI v0.37.1
Waiting for authentication... (Press Esc or Ctrl+C to cancel)
"""
            ready_visible = """
? for shortcuts
YOLO Ctrl+Y
*   Type your message or @path/to/file
workspace (/directory)
"""
            busy_visible = """
Thinking... (esc to cancel, 43s)
*   Type your message or @path/to/file
workspace (/directory)
"""
            self.assertFalse(worker._visible_indicates_agent_ready(not_ready_visible))
            self.assertFalse(worker._visible_indicates_agent_ready(busy_visible))
            self.assertTrue(worker._visible_indicates_agent_ready(ready_visible))

    def test_codex_ready_detection_rejects_prompt_marker_during_mcp_boot(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="codex-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            not_ready_visible = """
• Starting MCP servers (0/2): ossinsight, playwright (0s • esc to interrupt)
› Find and fix a bug in @filename
"""
            ready_visible = """
› Find and fix a bug in @filename
  gpt-5.4-mini high · ~/Desktop/KevinGit/My_C_Tools
"""
            self.assertTrue(worker._visible_indicates_agent_starting(not_ready_visible))
            self.assertFalse(worker._visible_indicates_agent_ready(not_ready_visible))
            self.assertFalse(worker._visible_indicates_agent_ready(ready_visible))

    def test_codex_ready_detection_rejects_working_marker_near_input_box(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="codex-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.5"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            visible = """
• Working (6s • esc to interrupt)

› Explain this codebase

  gpt-5.5 xhigh · ~/Desktop/DRL_PM
"""
            observation = WorkerObservation(
                visible_text=visible,
                raw_log_delta="",
                raw_log_tail=visible,
                current_command="node",
                current_path=tmp_dir,
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-05-08T17:30:00",
                pane_title="⠼ DRL_PM",
            )

            worker.agent_started = True
            worker.pane_id = "%1"
            self.assertFalse(worker._visible_indicates_agent_ready(visible))
            self.assertEqual(worker.get_agent_state(observation), AgentRuntimeState.BUSY)

            queued_visible = """
Messages to be submitted after next tool call:
```text
a07.developer.refine_code
```

› Explain this codebase

  gpt-5.5 xhigh · ~/Desktop/DRL_PM
"""
            self.assertFalse(worker._visible_indicates_agent_ready(queued_visible))

    def test_codex_ready_detection_uses_title_with_update_banner(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="codex-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            visible = """
╭─────────────────────────────────────────────────╮
│ ✨ Update available! 0.120.0 -> 0.121.0         │
╰─────────────────────────────────────────────────╯

› Run /review on my current changes
  gpt-5.4 high · ~/Desktop/KevinGit/My_C_Tools
"""
            self.assertFalse(worker._visible_indicates_agent_starting(visible))
            self.assertFalse(worker._visible_indicates_agent_ready(visible))
            worker.agent_started = True
            worker.pane_id = "%1"
            observation = WorkerObservation(
                visible_text=visible,
                raw_log_delta="",
                raw_log_tail=visible,
                current_command="node",
                current_path=tmp_dir,
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-05-08T18:00:00",
                pane_title="My_C_Tools",
            )
            self.assertEqual(worker.get_agent_state(observation), AgentRuntimeState.READY)

    def test_codex_ready_detection_ignores_stale_boot_prompts_after_input_box(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="codex-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.3-codex"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            visible = """
Do you trust the contents of this directory?
› 1. Yes, continue
  Press enter to continue

╭─────────────────────────────────────────────────╮
│ ✨ Update available! 0.125.0 -> 0.128.0         │
╰─────────────────────────────────────────────────╯

⚠ MCP startup incomplete (failed: notion)

› Find and fix a bug in @filename

  gpt-5.3-codex high · ~/Desktop/tmux_tui_simple_20260503_172545
"""
            observation = WorkerObservation(
                visible_text=visible,
                raw_log_delta="",
                raw_log_tail=visible,
                current_command="node",
                current_path=tmp_dir,
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-05-03T17:36:00",
                pane_title="tmux_tui_simple_20260...",
            )

            self.assertFalse(worker._visible_indicates_agent_starting(visible))
            self.assertFalse(worker._visible_indicates_agent_ready(visible))
            self.assertFalse(worker._visible_ready_signature(observation))
            worker.agent_started = True
            worker.pane_id = "%1"
            self.assertEqual(worker.get_agent_state(observation), AgentRuntimeState.READY)
            self.assertFalse(worker._maybe_handle_codex_boot_prompt(visible))

    def test_gemini_ready_detection_rejects_trust_prompt(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="gemini-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            trust_visible = """
Do you trust the files in this folder?
1. Trust folder
2. Trust parent folder
3. Don't trust
"""
            self.assertFalse(worker._visible_indicates_agent_ready(trust_visible))

    def test_opencode_ready_detection_accepts_input_prompt_and_completed_footer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="opencode-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="default"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            booting_visible = """
Performing one time database migration...
Database migration complete.
"""
            waiting_visible = """
Ask anything... "Fix a TODO in the codebase"
tab agents  ctrl+p commands
"""
            completed_visible = """
Created hello.txt with the content hi in /private/tmp/opencode-permtest.NOSf72/.

10.4K  ctrl+p commands
"""
            wrapped_completed_visible = """
Created hello.txt with the content hi in /private/tmp/opencode-permtest.NOSf72/.

10.4K  ctrl+p
commands
"""
            extreme_wrapped_completed_visible = """
Created hello.txt with the content hi in /private/tmp/opencode-permtest.NOSf72/.

85.2K (4ctrl+p
command
s
"""
            busy_visible = """
Thinking: The file has been created successfully.
■■■⬝⬝⬝⬝⬝  esc interrupt                         tab agents  ctrl+p commands
"""
            extreme_wrapped_busy_visible = """
Thinking: The file has been created successfully.
esc
interrupt
85.2K (4ctrl+p
command
s
"""
            self.assertTrue(worker._visible_indicates_agent_starting(booting_visible))
            self.assertFalse(worker._visible_indicates_agent_ready(booting_visible))
            self.assertTrue(worker._visible_indicates_agent_ready(waiting_visible))
            self.assertTrue(worker._visible_indicates_agent_ready(completed_visible))
            self.assertTrue(worker._visible_indicates_agent_ready(wrapped_completed_visible))
            self.assertTrue(worker._visible_indicates_agent_ready(extreme_wrapped_completed_visible))
            self.assertFalse(worker._visible_indicates_agent_ready(busy_visible))
            self.assertFalse(worker._visible_indicates_agent_ready(extreme_wrapped_busy_visible))

    def test_supported_vendor_state_classification_is_deterministic(self):
        cases = (
            (
                "codex",
                "gpt-5.4-mini",
                "",
                {"visible_text": "› Continue", "raw_log_tail": "› Continue", "current_command": "codex", "pane_title": "TmuxCodingTeam"},
                {"visible_text": "› Continue", "raw_log_tail": "› Continue", "current_command": "codex", "pane_title": "⠋ TmuxCodingTeam"},
            ),
            (
                "claude",
                "sonnet",
                "10900",
                {"visible_text": "❯", "raw_log_tail": "❯", "current_command": "claude", "pane_title": "✳ Claude Code"},
                {"visible_text": "Moseying…", "raw_log_tail": "Moseying…", "current_command": "claude", "pane_title": "⠋ Claude Code"},
            ),
            (
                "gemini",
                "auto",
                "10900",
                {"visible_text": "Type your message or @path/to/file", "raw_log_tail": "Type your message or @path/to/file", "current_command": "gemini", "pane_title": "◇ Ready"},
                {"visible_text": "Working…", "raw_log_tail": "Working…", "current_command": "gemini", "pane_title": "✦ Working"},
            ),
            (
                "opencode",
                "default",
                "",
                {"visible_text": "Ask anything...\nctrl+p commands", "raw_log_tail": "Ask anything...\nctrl+p commands", "current_command": "node", "pane_title": "OpenCode"},
                {"visible_text": "esc interrupt", "raw_log_tail": "esc interrupt", "current_command": "node", "pane_title": "OpenCode"},
            ),
        )
        def fake_resolve_launch(vendor_id, requested_model, requested_effort):
            return SimpleNamespace(
                resolved_model=str(requested_model or "default"),
                resolved_variant="",
                reasoning_control_mode="test",
                catalog_source_kind="test",
                confidence="high",
                native_reasoning_level=str(requested_effort or "high"),
                supports_reasoning=True,
                notes=(),
            )

        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch(
            "tmux_core.runtime.tmux_runtime.resolve_launch",
            side_effect=fake_resolve_launch,
        ):
            for vendor, model, proxy_url, ready_fields, busy_fields in cases:
                with self.subTest(vendor=vendor):
                    worker = TmuxBatchWorker(
                        worker_id=f"{vendor}-state-worker",
                        work_dir=tmp_dir,
                        config=AgentRunConfig(vendor=vendor, model=model, proxy_url=proxy_url),
                        runtime_root=Path(tmp_dir) / f"{vendor}-runtime",
                    )
                    worker.pane_id = "%1"
                    ready_observation = WorkerObservation(
                        current_path=tmp_dir,
                        pane_dead=False,
                        session_exists=True,
                        log_mtime=0.0,
                        observed_at="2026-04-24T00:00:00",
                        raw_log_delta="",
                        **ready_fields,
                    )
                    busy_observation = WorkerObservation(
                        current_path=tmp_dir,
                        pane_dead=False,
                        session_exists=True,
                        log_mtime=0.0,
                        observed_at="2026-04-24T00:00:01",
                        raw_log_delta="",
                        **busy_fields,
                    )
                    shell_observation = WorkerObservation(
                        visible_text="",
                        raw_log_delta="",
                        raw_log_tail="",
                        current_command="zsh",
                        current_path=tmp_dir,
                        pane_dead=False,
                        session_exists=True,
                        log_mtime=0.0,
                        observed_at="2026-04-24T00:00:02",
                        pane_title=str(ready_fields["pane_title"]),
                    )
                    worker.agent_started = False
                    self.assertEqual(worker.get_agent_state(ready_observation), AgentRuntimeState.STARTING)
                    worker.agent_started = True
                    self.assertEqual(worker.get_agent_state(ready_observation), AgentRuntimeState.READY)
                    self.assertEqual(worker.get_agent_state(busy_observation), AgentRuntimeState.BUSY)
                    self.assertEqual(worker.get_agent_state(shell_observation), AgentRuntimeState.DEAD)
                    if proxy_url:
                        self.assertEqual(worker.config.proxy_url, "http://127.0.0.1:10900")
                        self.assertEqual(build_proxy_env(worker.config.proxy_url)["HTTP_PROXY"], "http://127.0.0.1:10900")

    def test_claude_busy_surface_overrides_prompt_and_stale_ready_markers(self):
        busy_visible = """
❯ 请执行这个命令并告诉我结果: sleep 8 && echo state-test-done

⏺ Bash(sleep 8 && echo state-test-done)
  ⎿  Running…

✶ Nesting… (8s · ↓ 134 tokens · thinking with high effort)

────────────────────────────────────────────────────────────────────────────────
❯
────────────────────────────────────────────────────────────────────────────────
  ⏵⏵ bypass permissions on (shift+tab to cycle) · esc to interrupt
""".strip()
        detector_state = ClaudeOutputDetector().classify_agent_state(
            WorkerObservation(
                visible_text=busy_visible,
                raw_log_delta="",
                raw_log_tail="old ready prompt\n❯\n[[ACX_TURN:old:DONE]]",
                current_command="claude.exe",
                current_path="/tmp/project",
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-24T00:00:00",
                pane_title="",
            )
        )
        self.assertEqual(detector_state, AgentRuntimeState.BUSY)

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="claude-busy-state-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="claude", model="sonnet"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            observation = WorkerObservation(
                visible_text=busy_visible,
                raw_log_delta="",
                raw_log_tail="old ready prompt\n❯",
                current_command="claude.exe",
                current_path=tmp_dir,
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-24T00:00:01",
                pane_title="⠐ Execute test command with sleep delay",
            )
            self.assertEqual(worker.get_agent_state(observation), AgentRuntimeState.BUSY)
            self.assertFalse(worker._visible_indicates_agent_ready(busy_visible))  # noqa: SLF001

            star_busy_observation = WorkerObservation(
                visible_text=busy_visible,
                raw_log_delta="",
                raw_log_tail="old ready prompt\n❯",
                current_command="claude.exe",
                current_path=tmp_dir,
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-24T00:00:02",
                pane_title="⠐ Nesting…",
            )
            self.assertEqual(worker.get_agent_state(star_busy_observation), AgentRuntimeState.BUSY)

            ready_observation = WorkerObservation(
                visible_text="❯",
                raw_log_delta="",
                raw_log_tail="❯",
                current_command="claude.exe",
                current_path=tmp_dir,
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-24T00:00:03",
                pane_title="✳ Claude Code",
            )
            self.assertEqual(worker.get_agent_state(ready_observation), AgentRuntimeState.READY)

            completed_ready_surface = """
⏺ Bash(sleep 2 && echo claude-ready-regression-done)
  ⎿  claude-ready-regression-done

⏺ 命令执行成功，输出为：claude-ready-regression-done

✻ Brewed for 20s

────────────────────────────────────────────────────────────────────────────────
❯
────────────────────────────────────────────────────────────────────────────────
  ⏵⏵ bypass permissions on (shift+tab to cycle)
""".strip()
            completed_ready_observation = WorkerObservation(
                visible_text=completed_ready_surface,
                raw_log_delta="",
                raw_log_tail="",
                current_command="claude.exe",
                current_path=tmp_dir,
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-24T00:00:04",
                pane_title="✳ Execute command and report results",
            )
            self.assertEqual(worker.get_agent_state(completed_ready_observation), AgentRuntimeState.READY)

            ready_with_draft_surface = """
⏺ 审核通过

✻ Churned for 1m 6s

────────────────────────────────────────────────────────────────────────────────
❯ 继续审计 M1-T2
────────────────────────────────────────────────────────────────────────────────
  ⏵⏵ bypass permissions on (shift+tab to cycle)
""".strip()
            ready_with_draft_observation = WorkerObservation(
                visible_text=ready_with_draft_surface,
                raw_log_delta="",
                raw_log_tail="",
                current_command="claude.exe",
                current_path=tmp_dir,
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-24T00:00:05",
                pane_title="✳ Reinforcement learning asset allocation test review",
            )
            self.assertEqual(worker.get_agent_state(ready_with_draft_observation), AgentRuntimeState.READY)

    def test_codex_title_detection_accepts_work_dir_basename(self):
        class CodexStateWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

        with tempfile.TemporaryDirectory(prefix="tmux-api-v3-") as tmp_dir:
            work_dir = Path(tmp_dir) / "tmux-api-v3"
            work_dir.mkdir()
            worker = CodexStateWorker(
                worker_id="codex-title-worker",
                work_dir=work_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            ready_observation = WorkerObservation(
                visible_text="› Continue with the current task",
                raw_log_delta="",
                raw_log_tail="› Continue with the current task",
                current_command="node",
                current_path=str(work_dir),
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-21T00:00:00",
                pane_title="tmux-api-v3",
            )
            busy_observation = WorkerObservation(
                visible_text="› Continue with the current task",
                raw_log_delta="",
                raw_log_tail="› Continue with the current task",
                current_command="node",
                current_path=str(work_dir),
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-21T00:00:01",
                pane_title="⠋ tmux-api-v3",
            )
            shell_observation = WorkerObservation(
                visible_text="",
                raw_log_delta="",
                raw_log_tail="",
                current_command="zsh",
                current_path=str(work_dir),
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-21T00:00:02",
                pane_title="tmux-api-v3",
            )

            self.assertTrue(worker._title_indicates_ready("tmux-api-v3"))
            self.assertTrue(worker._title_indicates_busy("⠋ tmux-api-v3"))
            self.assertTrue(worker._title_indicates_busy("⠧tmux-api-v3"))

            worker.agent_started = False
            self.assertEqual(worker.get_agent_state(ready_observation), AgentRuntimeState.STARTING)
            self.assertEqual(worker.get_agent_state(shell_observation), AgentRuntimeState.STARTING)

            worker.agent_started = True
            self.assertEqual(worker.get_agent_state(ready_observation), AgentRuntimeState.READY)
            self.assertEqual(worker.get_agent_state(busy_observation), AgentRuntimeState.BUSY)
            self.assertEqual(worker.get_agent_state(shell_observation), AgentRuntimeState.DEAD)

    def test_prelaunch_shell_window_keeps_opencode_worker_starting(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir)
            worker = TmuxBatchWorker(
                worker_id="opencode-prelaunch-worker",
                work_dir=work_dir,
                config=AgentRunConfig(vendor="opencode", model="kimi-code/kimi-for-coding"),
                runtime_root=work_dir / "runtime",
            )
            worker.pane_id = "%1"
            observation = WorkerObservation(
                visible_text="",
                raw_log_delta="",
                raw_log_tail="",
                current_command="zsh",
                current_path=str(work_dir),
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-22T10:00:00",
                pane_title="OpenCode",
            )

            with mock.patch.object(worker, "target_exists", return_value=True):
                self.assertEqual(worker.get_agent_state(observation), AgentRuntimeState.STARTING)

    def test_wrapper_state_maps_not_ready_and_ready(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="wrapper-state-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.agent_state = AgentRuntimeState.STARTING
            worker.agent_ready = False
            self.assertEqual(
                worker._infer_wrapper_state(
                    current_command="codex",
                    visible_text="Starting MCP servers (0/2): ossinsight, playwright",
                ),
                WrapperState.NOT_READY,
            )
            worker.agent_state = AgentRuntimeState.BUSY
            worker.agent_ready = True
            self.assertEqual(
                worker._infer_wrapper_state(
                    current_command="codex",
                    visible_text="Thinking...",
                ),
                WrapperState.NOT_READY,
            )
            worker.agent_state = AgentRuntimeState.READY
            worker.agent_ready = True
            worker.agent_started = True
            worker.last_pane_title = "TmuxCodingTeam"
            self.assertEqual(
                worker._infer_wrapper_state(
                    current_command="codex",
                    visible_text="› Find and fix a bug in @filename",
                ),
                WrapperState.READY,
            )

    def test_wrapper_state_keeps_pre_ready_processing_in_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="pre-ready-processing-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.agent_state = AgentRuntimeState.BUSY
            worker.agent_ready = False
            self.assertEqual(
                worker._infer_wrapper_state(
                    current_command="codex",
                    visible_text="",
                ),
                WrapperState.NOT_READY,
            )

    def test_wrapper_state_keeps_codex_and_gemini_startup_pages_in_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            codex_worker = TmuxBatchWorker(
                worker_id="codex-wrapper-state-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime-codex",
            )
            codex_worker.agent_state = AgentRuntimeState.READY
            codex_worker.agent_ready = False
            self.assertEqual(
                codex_worker._infer_wrapper_state(
                    current_command="codex",
                    visible_text="• Starting MCP servers (0/2): ossinsight, playwright\n› Find and fix a bug in @filename",
                ),
                WrapperState.NOT_READY,
            )

            gemini_worker = TmuxBatchWorker(
                worker_id="gemini-wrapper-state-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime-gemini",
            )
            gemini_worker.agent_state = AgentRuntimeState.STARTING
            gemini_worker.agent_ready = True
            self.assertEqual(
                gemini_worker._infer_wrapper_state(
                    current_command="gemini",
                    visible_text="Waiting for authentication... (Press Esc or Ctrl+C to cancel)",
                ),
                WrapperState.NOT_READY,
            )

    def test_wrapper_state_uses_recent_terminal_hash_changes_as_not_ready_signal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="activity-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="claude", model="haiku"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.agent_state = AgentRuntimeState.READY
            worker.agent_ready = True
            worker._update_terminal_activity("frame-1", observed_at="2026-04-15T00:00:00")
            self.assertEqual(
                worker._infer_wrapper_state(
                    current_command="claude",
                    visible_text="❯",
                ),
                WrapperState.NOT_READY,
            )

    def test_wrapper_state_returns_ready_after_terminal_hash_stabilizes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="stable-activity-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="claude", model="haiku"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.agent_state = AgentRuntimeState.READY
            worker.agent_ready = True
            worker.agent_started = True
            worker.last_pane_title = "✳ Claude Code"
            worker._update_terminal_activity("frame-1", observed_at="2026-04-15T00:00:00")
            worker._last_terminal_change_monotonic = time.monotonic() - TERMINAL_ACTIVITY_IDLE_WINDOW_SEC - 0.1
            worker.terminal_recently_changed = False
            self.assertEqual(
                worker._infer_wrapper_state(
                    current_command="claude",
                    visible_text="❯",
                ),
                WrapperState.READY,
            )

    def test_gemini_output_detector_ignores_footer_and_keeps_protocol_tokens(self):
        output = """
✦ Completed the routing layer generation.

[[ACX_TURN:demo1234:DONE]]
[[ROUTING_CREATE:DONE]]

? for shortcuts
YOLO Ctrl+Y
▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
*   Type your message or @path/to/file
▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
workspace (/directory)                                                     branch                          sandbox                                             /model
~/Desktop/KevinGit/PyFinance/ReturnClassification                          main                            no sandbox                          gemini-3-flash-preview
"""
        message = GeminiOutputDetector().extract_last_message(output)
        self.assertIn("[[ACX_TURN:demo1234:DONE]]", message)
        self.assertIn("[[ROUTING_CREATE:DONE]]", message)
        self.assertNotIn("Type your message or @path/to/file", message)

    def test_provider_detectors_classify_vendor_prompts(self):
        gemini_phase = GeminiOutputDetector().classify_agent_state(
            WorkerObservation(
                visible_text="Waiting for authentication... (Press Esc or Ctrl+C to cancel)",
                raw_log_delta="",
                raw_log_tail="Waiting for authentication... (Press Esc or Ctrl+C to cancel)",
                current_command="gemini",
                current_path="/tmp/project",
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
        )
        self.assertEqual(gemini_phase, AgentRuntimeState.STARTING)

        gemini_trust_phase = GeminiOutputDetector().classify_agent_state(
            WorkerObservation(
                visible_text="Do you trust the files in this folder?\n1. Trust folder\n2. Trust parent folder\n3. Don't trust",
                raw_log_delta="",
                raw_log_tail="Do you trust the files in this folder?\n1. Trust folder\n2. Trust parent folder\n3. Don't trust",
                current_command="gemini",
                current_path="/tmp/project",
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
        )
        self.assertEqual(gemini_trust_phase, AgentRuntimeState.STARTING)

        gemini_ready_over_stale_auth_phase = GeminiOutputDetector().classify_agent_state(
            WorkerObservation(
                visible_text="*   Type your message or @path/to/file\nworkspace (/directory)",
                raw_log_delta="",
                raw_log_tail="Waiting for authentication... (Press Esc or Ctrl+C to cancel)\n*   Type your message or @path/to/file\nworkspace (/directory)",
                current_command="gemini",
                current_path="/tmp/project",
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
        )
        self.assertEqual(gemini_ready_over_stale_auth_phase, AgentRuntimeState.READY)

        gemini_processing_over_stale_auth_phase = GeminiOutputDetector().classify_agent_state(
            WorkerObservation(
                visible_text="Working… (My_C_Tools)",
                raw_log_delta="",
                raw_log_tail="Waiting for authentication... (Press Esc or Ctrl+C to cancel)\nWorking… (My_C_Tools)",
                current_command="gemini",
                current_path="/tmp/project",
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
        )
        self.assertEqual(gemini_processing_over_stale_auth_phase, AgentRuntimeState.BUSY)

        gemini_thinking_with_input_box_phase = GeminiOutputDetector().classify_agent_state(
            WorkerObservation(
                visible_text="Thinking... (esc to cancel, 43s)\n*   Type your message or @path/to/file",
                raw_log_delta="",
                raw_log_tail="Thinking... (esc to cancel, 43s)\n*   Type your message or @path/to/file",
                current_command="gemini",
                current_path="/tmp/project",
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
        )
        self.assertEqual(gemini_thinking_with_input_box_phase, AgentRuntimeState.BUSY)

        codex_phase = CodexOutputDetector().classify_agent_state(
            WorkerObservation(
                visible_text="Update available!\nUpdate now\nSkip until next version\nPress enter to continue",
                raw_log_delta="",
                raw_log_tail="Update available!\nUpdate now\nSkip until next version\nPress enter to continue",
                current_command="codex",
                current_path="/tmp/project",
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
        )
        self.assertEqual(codex_phase, AgentRuntimeState.STARTING)

        codex_waiting_phase = CodexOutputDetector().classify_agent_state(
            WorkerObservation(
                visible_text="› Find and fix a bug in @filename\n  gpt-5.4-mini high · ~/project",
                raw_log_delta="",
                raw_log_tail="› Find and fix a bug in @filename\n  gpt-5.4-mini high · ~/project",
                current_command="codex",
                current_path="/tmp/project",
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
                pane_title="project",
            )
        )
        self.assertEqual(codex_waiting_phase, AgentRuntimeState.READY)

        codex_waiting_with_update_banner_phase = CodexOutputDetector().classify_agent_state(
            WorkerObservation(
                visible_text="Update available!\n› Run /review on my current changes\n  gpt-5.4 high · ~/project",
                raw_log_delta="",
                raw_log_tail="Update available!\n› Run /review on my current changes\n  gpt-5.4 high · ~/project",
                current_command="codex",
                current_path="/tmp/project",
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
                pane_title="project",
            )
        )
        self.assertEqual(codex_waiting_with_update_banner_phase, AgentRuntimeState.READY)

        codex_booting_phase = CodexOutputDetector().classify_agent_state(
            WorkerObservation(
                visible_text="• Starting MCP servers (0/2): ossinsight, playwright (0s • esc to interrupt)\n› Find and fix a bug in @filename",
                raw_log_delta="",
                raw_log_tail="• Starting MCP servers (0/2): ossinsight, playwright (0s • esc to interrupt)\n› Find and fix a bug in @filename",
                current_command="codex",
                current_path="/tmp/project",
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
        )
        self.assertEqual(codex_booting_phase, AgentRuntimeState.STARTING)

        codex_processing_phase = CodexOutputDetector().classify_agent_state(
            WorkerObservation(
                visible_text="• Identifying C++ files and tests (25s • esc to interrupt)\n› Run /review on my current changes",
                raw_log_delta="",
                raw_log_tail="• Identifying C++ files and tests (25s • esc to interrupt)\n› Run /review on my current changes",
                current_command="codex",
                current_path="/tmp/project",
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
                pane_title="⠋ project",
            )
        )
        self.assertEqual(codex_processing_phase, AgentRuntimeState.BUSY)

        claude_phase = ClaudeOutputDetector().classify_agent_state(
            WorkerObservation(
                visible_text="❯",
                raw_log_delta="",
                raw_log_tail="❯",
                current_command="claude",
                current_path="/tmp/project",
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
        )
        self.assertEqual(claude_phase, AgentRuntimeState.READY)

    def test_build_launch_command_variants_include_expected_flags(self):
        work_dir = Path("/tmp/project")

        codex_cmd = AgentRunConfig(vendor=Vendor.CODEX, model="gpt-5.4", reasoning_effort="xhigh").build_launch_command(work_dir)
        self.assertIn("codex --model", codex_cmd)
        self.assertIn("--cd /tmp/project", codex_cmd)

        claude_cmd = AgentRunConfig(vendor=Vendor.CLAUDE, model="sonnet", reasoning_effort="max").build_launch_command(work_dir)
        self.assertIn("claude --model", claude_cmd)
        self.assertIn("--effort max", claude_cmd)

        gemini_cmd = AgentRunConfig(vendor=Vendor.GEMINI, model="auto", reasoning_effort="medium").build_launch_command(work_dir)
        self.assertIn("gemini --model flash", gemini_cmd)
        self.assertIn("--model flash", gemini_cmd)

        opencode_default_cmd = AgentRunConfig(vendor=Vendor.OPENCODE, model="default").build_launch_command(work_dir)
        self.assertIn("opencode /tmp/project --pure", opencode_default_cmd)
        self.assertIn(f"--model {get_default_model_for_vendor('opencode')}", opencode_default_cmd)
        self.assertNotIn("--dangerously-skip-permissions", opencode_default_cmd)

        mapped_opencode_model = next(
            item.model_id
            for item in get_model_choices("opencode")
            if item.reasoning.reasoning_control_mode == "mapped"
        )
        opencode_model_cmd = AgentRunConfig(
            vendor=Vendor.OPENCODE,
            model=mapped_opencode_model,
            reasoning_effort="max",
        ).build_launch_command(work_dir)
        self.assertIn(f"opencode /tmp/project --pure --model {mapped_opencode_model}", opencode_model_cmd)
        self.assertNotIn("--variant", opencode_model_cmd)
        self.assertNotIn("--dangerously-skip-permissions", opencode_model_cmd)

    def test_build_session_name_is_stable_and_bounded(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir)
            left = build_session_name("owner-worker", work_dir, Vendor.CODEX)
            right = build_session_name("owner-worker", work_dir, Vendor.CODEX)
            self.assertEqual(left, right)
            self.assertTrue(left.startswith("执行者-"))
            self.assertNotRegex(left, r"[\s/\\\\]")
            self.assertLessEqual(len(left), 60)

    def test_build_session_name_advances_when_preferred_name_is_occupied(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir)
            preferred = build_session_name("requirements-analyst", work_dir, Vendor.CODEX)
            fallback = build_session_name(
                "requirements-analyst",
                work_dir,
                Vendor.CODEX,
                occupied_session_names=[preferred],
            )
            self.assertTrue(preferred.startswith("分析师-"))
            self.assertTrue(fallback.startswith("分析师-"))
            self.assertNotEqual(preferred, fallback)

    def test_build_session_name_maps_routing_and_reviewer_roles(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir) / "repo-map"
            work_dir.mkdir()
            routing_session = build_session_name(work_dir.name, work_dir, Vendor.CODEX)
            review_ba_session = build_session_name("requirements-review-analyst", work_dir, Vendor.CODEX)
            reviewer_session = build_session_name("requirements-review-r2", work_dir, Vendor.CODEX)
            design_ba_session = build_session_name("detailed-design-analyst", work_dir, Vendor.CODEX)
            design_reviewer_session = build_session_name("detailed-design-review-开发工程师", work_dir, Vendor.CODEX)
            task_split_ba_session = build_session_name("task-split-analyst", work_dir, Vendor.CODEX)
            task_split_reviewer_session = build_session_name("task-split-review-开发工程师", work_dir, Vendor.CODEX)
            development_worker_session = build_session_name("development-developer", work_dir, Vendor.CODEX)
            development_reviewer_session = build_session_name("development-review-测试工程师", work_dir, Vendor.CODEX)
            self.assertTrue(routing_session.startswith("路由器-"))
            self.assertTrue(review_ba_session.startswith("需求分析师-"))
            self.assertTrue(reviewer_session.startswith("审核器-"))
            self.assertTrue(design_ba_session.startswith("需求分析师-"))
            self.assertTrue(design_reviewer_session.startswith("开发工程师-"))
            self.assertTrue(task_split_ba_session.startswith("需求分析师-"))
            self.assertTrue(task_split_reviewer_session.startswith("开发工程师-"))
            self.assertTrue(development_worker_session.startswith("开发工程师-"))
            self.assertTrue(development_reviewer_session.startswith("测试工程师-"))
            self.assertNotIn("-R2-", reviewer_session)

    def test_detailed_design_reviewer_worker_preserves_raw_worker_id_for_session_mapping(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="detailed-design-review-开发工程师",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
            )

        self.assertEqual(worker.worker_id, "detailed-design-review-开发工程师")
        self.assertTrue(worker.runtime_dir.name.startswith("detailed-design-review-"))
        self.assertTrue(worker.session_name.startswith("开发工程师-"))

    def test_load_worker_from_state_path_restores_detailed_design_reviewer_role_mapping(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="detailed-design-review-开发工程师",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker._write_state(WorkerStatus.READY, note="saved")  # noqa: SLF001
            restored = load_worker_from_state_path(worker.state_path)

        self.assertIsNotNone(restored)
        self.assertEqual(restored.worker_id, "detailed-design-review-开发工程师")
        self.assertTrue(restored.session_name.startswith("开发工程师-"))

    def test_task_split_reviewer_worker_preserves_raw_worker_id_for_session_mapping(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="task-split-review-开发工程师",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
            )

        self.assertEqual(worker.worker_id, "task-split-review-开发工程师")
        self.assertTrue(worker.runtime_dir.name.startswith("task-split-review-"))
        self.assertTrue(worker.session_name.startswith("开发工程师-"))

    def test_load_worker_from_state_path_restores_task_split_reviewer_role_mapping(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="task-split-review-开发工程师",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker._write_state(WorkerStatus.READY, note="saved")  # noqa: SLF001
            restored = load_worker_from_state_path(worker.state_path)

        self.assertIsNotNone(restored)
        self.assertEqual(restored.worker_id, "task-split-review-开发工程师")
        self.assertTrue(restored.session_name.startswith("开发工程师-"))

    def test_development_worker_preserves_raw_worker_id_for_session_mapping(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="development-developer",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
            )

        self.assertEqual(worker.worker_id, "development-developer")
        self.assertTrue(worker.runtime_dir.name.startswith("development-developer"))
        self.assertTrue(worker.session_name.startswith("开发工程师-"))

    def test_development_reviewer_worker_preserves_raw_worker_id_for_session_mapping(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="development-review-测试工程师",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
            )

        self.assertEqual(worker.worker_id, "development-review-测试工程师")
        self.assertTrue(worker.runtime_dir.name.startswith("development-review-"))
        self.assertTrue(worker.session_name.startswith("测试工程师-"))

    def test_load_worker_from_state_path_restores_development_reviewer_role_mapping(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="development-review-测试工程师",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker._write_state(WorkerStatus.READY, note="saved")  # noqa: SLF001
            restored = load_worker_from_state_path(worker.state_path)

        self.assertIsNotNone(restored)
        self.assertEqual(restored.worker_id, "development-review-测试工程师")
        self.assertTrue(restored.session_name.startswith("测试工程师-"))

    def test_write_state_notifies_runtime_state_change(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="reviewer-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            with mock.patch("T02_tmux_agents._notify_runtime_state_changed_best_effort") as notifier:
                worker._write_state(WorkerStatus.RUNNING, note="test")  # noqa: SLF001
        notifier.assert_called_once_with()

    def test_write_ready_state_clears_stale_running_runtime_fields(self):
        class ReadyStateWorker(TmuxBatchWorker):
            def is_agent_alive(self, observation=None):  # noqa: ANN001, ARG002
                return True

            def get_agent_state(self, observation=None):  # noqa: ANN001, ARG002
                return AgentRuntimeState.READY

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = ReadyStateWorker(
                worker_id="reviewer-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.agent_started = True
            worker.current_task_runtime_status = "running"
            worker._write_state(  # noqa: SLF001
                WorkerStatus.RUNNING,
                note="turn:review",
                extra={
                    "result_status": "running",
                    "current_task_runtime_status": "running",
                },
            )

            worker.current_task_runtime_status = "running"
            worker._write_state(WorkerStatus.READY, note="agent_ready")  # noqa: SLF001
            state = worker.read_state()

        self.assertEqual(state["status"], WorkerStatus.READY.value)
        self.assertEqual(state["result_status"], WorkerStatus.READY.value)
        self.assertEqual(state["agent_state"], AgentRuntimeState.READY.value)
        self.assertEqual(state["current_task_runtime_status"], "")

    def test_read_state_recovers_from_trailing_json_garbage(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="reviewer-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.state_path.write_text(
                "{\n"
                '  "state_revision": 3,\n'
                '  "note": "ok"\n'
                '}\n "note": "broken"\n}',
                encoding="utf-8",
            )
            recovered = worker.read_state()
            persisted = json.loads(worker.state_path.read_text(encoding="utf-8"))

        self.assertEqual(recovered.get("state_revision"), 3)
        self.assertEqual(recovered.get("note"), "ok")
        self.assertEqual(persisted.get("state_revision"), 3)
        self.assertEqual(persisted.get("note"), "ok")

    def test_write_state_tolerates_preexisting_corrupted_state_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="reviewer-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.state_path.write_text(
                "{\n"
                '  "state_revision": 1,\n'
                '  "workflow_stage": "pending"\n'
                '}\n "workflow_stage": "broken"\n}',
                encoding="utf-8",
            )
            with mock.patch.object(worker, "is_agent_alive", return_value=True), mock.patch.object(
                worker,
                "get_agent_state",
                return_value=type("State", (), {"value": "READY"})(),
            ), mock.patch("T02_tmux_agents._notify_runtime_state_changed_best_effort"):
                worker._write_state(WorkerStatus.RUNNING, note="recovered")  # noqa: SLF001
            persisted = json.loads(worker.state_path.read_text(encoding="utf-8"))

        self.assertEqual(persisted.get("status"), "running")
        self.assertEqual(persisted.get("note"), "recovered")

    def test_build_session_name_maps_routing_role_for_non_ascii_work_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir) / "测试项目"
            work_dir.mkdir()
            routing_session = build_session_name("测试项目", work_dir, Vendor.CODEX)
            self.assertTrue(routing_session.startswith("路由器-"))

    def test_extract_final_protocol_token_requires_final_line(self):
        self.assertEqual(
            extract_final_protocol_token("line\n[[ROUTING_CREATE:DONE]]", ["[[ROUTING_CREATE:DONE]]"]),
            "[[ROUTING_CREATE:DONE]]",
        )
        self.assertEqual(
            extract_final_protocol_token("line\n⏺ [[ROUTING_CREATE:DONE]]", ["[[ROUTING_CREATE:DONE]]"]),
            "[[ROUTING_CREATE:DONE]]",
        )
        self.assertEqual(
            extract_final_protocol_token("line\n✦ [[ROUTING_CREATE:DONE]]", ["[[ROUTING_CREATE:DONE]]"]),
            "[[ROUTING_CREATE:DONE]]",
        )
        self.assertEqual(
            extract_final_protocol_token("line\n[[ROUTING_CREATE:DONE]]\nextra", ["[[ROUTING_CREATE:DONE]]"]),
            "",
        )
        self.assertEqual(
            extract_final_protocol_token(
                "line\n[[ROUTING_CREATE:DONE]]\n✢ Improvising… (38s · ↓ 1.4k tokens)\n❯",
                ["[[ROUTING_CREATE:DONE]]"],
            ),
            "[[ROUTING_CREATE:DONE]]",
        )
        self.assertEqual(
            extract_final_protocol_token(
                "line\n[[ROUTING_AUDIT:WRITTEN]]\n*   输入您的消息或 @ 文件路径",
                ["[[ROUTING_AUDIT:WRITTEN]]"],
            ),
            "[[ROUTING_AUDIT:WRITTEN]]",
        )

    def test_run_turn_timeout_records_failed_state(self):
        class TimeoutWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.state_notes = []

            def _write_state(self, status, *, note, extra=None):
                self.state_notes.append((status.value, note, extra or {}))

            def _append_transcript(self, title, body):
                return None

            def ensure_agent_ready(self, timeout_sec=60.0):
                self.pane_id = "%1"
                self.agent_ready = True
                return None

            def target_exists(self, target=None):
                return True

            def capture_visible(self, tail_lines=500):
                return "fake-visible"

            def _send_text(self, text, enter_count=None):
                return None

            def _wait_for_turn_reply(self, **kwargs):
                raise TimeoutError("timed out")

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TimeoutWorker(
                worker_id="calc-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            result = worker.run_turn(label="timeout_case", prompt="hello", required_tokens=["[[DONE]]"], timeout_sec=0.05)
            self.assertFalse(result.ok)
            self.assertEqual(worker.results[-1].exit_code, TIMEOUT_EXIT_CODE)
            self.assertEqual(worker.state_notes[-1][0], "failed")
            self.assertTrue(worker.state_notes[-1][1].startswith("timeout:"))

    def test_run_turn_does_not_send_prompt_after_runtime_shutdown(self):
        class ShutdownBeforeSubmitWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.sent = False

            def _append_transcript(self, title, body):
                return None

            def _ensure_agent_ready_for_turn_start(self, **kwargs):
                self.pane_id = "%1"
                self.agent_ready = True
                request_runtime_shutdown("unit_test")

            def _send_text(self, text, enter_count=None):
                self.sent = True
                raise AssertionError("prompt should not be sent after shutdown")

        clear_runtime_shutdown_request()
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                worker = ShutdownBeforeSubmitWorker(
                    worker_id="shutdown-worker",
                    work_dir=tmp_dir,
                    config=AgentRunConfig(vendor="codex", model="gpt-5"),
                    runtime_root=Path(tmp_dir) / "runtime",
                )
                with self.assertRaises(RuntimeShutdownRequested):
                    worker.run_turn(label="shutdown_case", prompt="hello", required_tokens=["[[DONE]]"], timeout_sec=0.05)
                self.assertFalse(worker.sent)
        finally:
            clear_runtime_shutdown_request()

    def test_run_turn_restores_running_state_after_agent_ready(self):
        class SubmitStateWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.state_notes = []

            def _write_state(self, status, *, note, extra=None):
                self.state_notes.append((status.value, note, extra or {}))

            def _append_transcript(self, title, body):
                return None

            def ensure_agent_ready(self, timeout_sec=60.0):
                self.pane_id = "%1"
                self.agent_ready = True
                self._write_state(type("Status", (), {"value": "ready"})(), note="agent_ready", extra={})

            def target_exists(self, target=None):
                return True

            def capture_visible(self, tail_lines=500):
                return "fake-visible"

            def _send_text(self, text, enter_count=None):
                assert self.state_notes[-1][0] == "running"
                assert self.state_notes[-1][1].startswith("turn:")
                assert self.current_task_status_path
                payload = json.loads(Path(self.current_task_status_path).read_text(encoding="utf-8"))
                assert payload == {"status": "running"}
                assert self.current_task_status_path not in text

            def _wait_for_turn_reply(self, **kwargs):
                return "ok\n[[DONE]]"

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = SubmitStateWorker(
                worker_id="calc-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            result = worker.run_turn(label="submit_case", prompt="hello", required_tokens=["[[DONE]]"], timeout_sec=0.05)
            self.assertTrue(result.ok)
            running_notes = [item for item in worker.state_notes if item[0] == "running"]
            self.assertGreaterEqual(len(running_notes), 2)

    def test_run_turn_extends_initial_ready_wait_when_agent_is_busy(self):
        class BusyThenReadyWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.ensure_timeouts = []
                self.sent_prompts = []

            def _write_state(self, status, *, note, extra=None):
                return None

            def _append_transcript(self, title, body):
                return None

            def ensure_agent_ready(self, timeout_sec=60.0):
                self.ensure_timeouts.append(timeout_sec)
                self.pane_id = "%1"
                self.agent_started = True
                self.current_command = "node"
                self.current_path = str(self.work_dir)
                if len(self.ensure_timeouts) == 1:
                    self.agent_ready = False
                    self.agent_state = AgentRuntimeState.BUSY
                    self.wrapper_state = WrapperState.NOT_READY
                    raise RuntimeError("Timed out waiting for agent ready.\nworking")
                self.agent_ready = True
                self.agent_state = AgentRuntimeState.READY
                self.wrapper_state = WrapperState.READY
                self.last_pane_title = "TmuxCodingTeam"

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                if not self.agent_ready:
                    return WorkerObservation(
                        visible_text="working",
                        raw_log_delta="",
                        raw_log_tail="esc to interrupt",
                        current_command="node",
                        current_path=str(self.work_dir),
                        pane_dead=False,
                        session_exists=True,
                        log_mtime=0.0,
                        observed_at="2026-04-28T17:23:00",
                        pane_title="⠋ TmuxCodingTeam",
                    )
                return WorkerObservation(
                    visible_text="›",
                    raw_log_delta="",
                    raw_log_tail="›",
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-28T17:24:00",
                    pane_title="TmuxCodingTeam",
                )

            def target_exists(self, target=None):
                return True

            def capture_visible(self, tail_lines=500):
                return "fake-visible"

            def _send_text(self, text, enter_count=None):
                self.sent_prompts.append(text)
                write_task_status(self.current_task_status_path, status="done")

            def _wait_for_turn_reply(self, **kwargs):
                return "ok\n[[DONE]]"

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = BusyThenReadyWorker(
                worker_id="busy-start-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            result = worker.run_turn(label="busy_ready_start", prompt="hello", required_tokens=["[[DONE]]"], timeout_sec=2.0)

        self.assertTrue(result.ok)
        self.assertEqual(len(worker.sent_prompts), 1)
        self.assertGreaterEqual(len(worker.ensure_timeouts), 2)
        self.assertEqual(worker.ensure_timeouts[0], 60.0)
        self.assertEqual(worker.ensure_timeouts[1], 2.0)

    def test_run_turn_keeps_extending_initial_ready_wait_while_agent_remains_busy(self):
        class BusyTwiceThenReadyWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.ensure_timeouts = []
                self.sent_prompts = []

            def _write_state(self, status, *, note, extra=None):
                return None

            def _append_transcript(self, title, body):
                return None

            def ensure_agent_ready(self, timeout_sec=60.0):
                self.ensure_timeouts.append(timeout_sec)
                self.pane_id = "%1"
                self.agent_started = True
                self.current_command = "node"
                self.current_path = str(self.work_dir)
                if len(self.ensure_timeouts) <= 2:
                    self.agent_ready = False
                    self.agent_state = AgentRuntimeState.BUSY
                    self.wrapper_state = WrapperState.NOT_READY
                    raise RuntimeError("Timed out waiting for agent ready.\nworking")
                self.agent_ready = True
                self.agent_state = AgentRuntimeState.READY
                self.wrapper_state = WrapperState.READY
                self.last_pane_title = "TmuxCodingTeam"

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="working" if not self.agent_ready else "›",
                    raw_log_delta="",
                    raw_log_tail="esc to interrupt" if not self.agent_ready else "›",
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-28T17:23:00",
                    pane_title="⠋ TmuxCodingTeam" if not self.agent_ready else "TmuxCodingTeam",
                )

            def target_exists(self, target=None):
                return True

            def capture_visible(self, tail_lines=500):
                return "fake-visible"

            def _send_text(self, text, enter_count=None):
                self.sent_prompts.append(text)
                write_task_status(self.current_task_status_path, status="done")

            def _wait_for_turn_reply(self, **kwargs):
                return "ok\n[[DONE]]"

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = BusyTwiceThenReadyWorker(
                worker_id="busy-twice-start-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            result = worker.run_turn(label="busy_twice_ready_start", prompt="hello", required_tokens=["[[DONE]]"], timeout_sec=2.0)

        self.assertTrue(result.ok)
        self.assertEqual(len(worker.sent_prompts), 1)
        self.assertEqual(worker.ensure_timeouts[:3], [60.0, 2.0, 2.0])

    def test_run_turn_relaunches_stale_busy_agent_after_completed_contract_turn(self):
        class StaleBusyAfterDoneWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.ensure_timeouts = []
                self.sent_prompts = []
                self.restart_count = 0
                self.state_notes = []

            def _write_state(self, status, *, note, extra=None):
                self.state_notes.append((status.value, note, extra or {}))

            def _append_transcript(self, title, body):
                return None

            def ensure_agent_ready(self, timeout_sec=60.0):
                self.ensure_timeouts.append(timeout_sec)
                self.pane_id = "%1"
                self.agent_started = True
                self.current_command = "node"
                self.current_path = str(self.work_dir)
                if self.restart_count == 0:
                    self.agent_ready = False
                    self.agent_state = AgentRuntimeState.BUSY
                    self.wrapper_state = WrapperState.NOT_READY
                    raise RuntimeError("Timed out waiting for agent ready.\nBuild · Big Pickle")
                self.agent_ready = True
                self.agent_state = AgentRuntimeState.READY
                self.wrapper_state = WrapperState.READY
                self.last_pane_title = "TmuxCodingTeam"

            def request_restart(self):
                self.restart_count += 1
                self.agent_ready = False
                self.agent_started = False
                self.agent_state = AgentRuntimeState.STARTING
                self.wrapper_state = WrapperState.NOT_READY
                return self.session_name

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="Build · Big Pickle" if self.restart_count == 0 else "›",
                    raw_log_delta="",
                    raw_log_tail="Build · Big Pickle\nesc interrupt" if self.restart_count == 0 else "›",
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-05-04T06:24:00",
                    pane_title="OC | 需求文档一致性审计" if self.restart_count == 0 else "TmuxCodingTeam",
                )

            def target_exists(self, target=None):
                return True

            def capture_visible(self, tail_lines=500):
                return "Build · Big Pickle"

            def _send_text(self, text, enter_count=None):
                self.sent_prompts.append(text)
                write_task_status(self.current_task_status_path, status="done")

            def _wait_for_turn_reply(self, **kwargs):
                return "ok\n[[DONE]]"

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = StaleBusyAfterDoneWorker(
                worker_id="stale-busy-after-done-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="opencode/big-pickle"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.agent_started = True
            worker.current_task_runtime_status = "done"
            result = worker.run_turn(label="stale_busy_after_done", prompt="hello", required_tokens=["[[DONE]]"], timeout_sec=2.0)

        self.assertFalse(result.ok)
        self.assertEqual(worker.restart_count, 0)
        self.assertEqual(len(worker.sent_prompts), 0)
        self.assertEqual(worker.ensure_timeouts, [60.0])
        self.assertIn("系统不会自动重启该智能体", result.clean_output)
        self.assertIn(f"tmux attach -t {worker.session_name}", result.clean_output)

    def test_run_turn_relaunches_stale_busy_agent_before_repair_after_failed_turn(self):
        class StaleBusyAfterFailedWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.ensure_timeouts = []
                self.sent_prompts = []
                self.restart_count = 0

            def read_state(self):
                return {"status": "failed"}

            def _write_state(self, status, *, note, extra=None):
                return None

            def _append_transcript(self, title, body):
                return None

            def ensure_agent_ready(self, timeout_sec=60.0):
                self.ensure_timeouts.append(timeout_sec)
                self.pane_id = "%1"
                self.agent_started = True
                self.current_command = "node"
                self.current_path = str(self.work_dir)
                if self.restart_count == 0:
                    self.agent_ready = False
                    self.agent_state = AgentRuntimeState.BUSY
                    self.wrapper_state = WrapperState.NOT_READY
                    raise RuntimeError("Timed out waiting for agent ready.\nBuild · Big Pickle")
                self.agent_ready = True
                self.agent_state = AgentRuntimeState.READY
                self.wrapper_state = WrapperState.READY

            def request_restart(self):
                self.restart_count += 1
                self.agent_ready = False
                self.agent_started = False
                self.agent_state = AgentRuntimeState.STARTING
                self.wrapper_state = WrapperState.NOT_READY
                return self.session_name

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="Build · Big Pickle" if self.restart_count == 0 else "›",
                    raw_log_delta="",
                    raw_log_tail="Build · Big Pickle\nesc interrupt" if self.restart_count == 0 else "›",
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-05-04T07:20:00",
                    pane_title="OC | Review Repair" if self.restart_count == 0 else "TmuxCodingTeam",
                )

            def target_exists(self, target=None):
                return True

            def capture_visible(self, tail_lines=500):
                return "Build · Big Pickle"

            def _send_text(self, text, enter_count=None):
                self.sent_prompts.append(text)
                write_task_status(self.current_task_status_path, status="done")

            def _wait_for_turn_reply(self, **kwargs):
                return "ok\n[[DONE]]"

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = StaleBusyAfterFailedWorker(
                worker_id="stale-busy-after-failed-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="opencode/big-pickle"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.agent_started = True
            worker.current_task_runtime_status = "running"
            result = worker.run_turn(label="repair_turn", prompt="repair", required_tokens=["[[DONE]]"], timeout_sec=2.0)

        self.assertFalse(result.ok)
        self.assertEqual(worker.restart_count, 0)
        self.assertEqual(len(worker.sent_prompts), 0)
        self.assertEqual(worker.ensure_timeouts, [60.0])
        self.assertIn("系统不会自动重启该智能体", result.clean_output)
        self.assertIn(f"tmux attach -t {worker.session_name}", result.clean_output)

    def test_run_turn_waits_for_active_busy_agent_after_failed_turn(self):
        class ActiveBusyAfterFailedWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.ensure_timeouts = []
                self.sent_prompts = []
                self.state_notes = []

            def read_state(self):
                return {"status": "failed"}

            def _write_state(self, status, *, note, extra=None):
                self.state_notes.append((status.value, note, extra or {}))

            def _append_transcript(self, title, body):
                return None

            def ensure_agent_ready(self, timeout_sec=60.0):
                self.ensure_timeouts.append(timeout_sec)
                self.pane_id = "%1"
                self.agent_started = True
                self.current_command = "node"
                self.current_path = str(self.work_dir)
                if len(self.ensure_timeouts) == 1:
                    self.agent_ready = False
                    self.agent_state = AgentRuntimeState.BUSY
                    self.wrapper_state = WrapperState.NOT_READY
                    raise RuntimeError("Timed out waiting for agent ready.\nBuild · Big Pickle")
                self.agent_ready = True
                self.agent_state = AgentRuntimeState.READY
                self.wrapper_state = WrapperState.READY
                self.last_pane_title = "TmuxCodingTeam"

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                self.last_terminal_signature = "active-progress"
                self.last_terminal_changed_at = "2026-05-06T11:28:40"
                self._last_terminal_change_monotonic = time.monotonic()
                self.terminal_recently_changed = True
                return WorkerObservation(
                    visible_text="Build · Big Pickle",
                    raw_log_delta="",
                    raw_log_tail="Build · Big Pickle\nesc interrupt",
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-05-06T11:28:40",
                    pane_title="OC | Review Repair",
                )

            def target_exists(self, target=None):
                return True

            def capture_visible(self, tail_lines=500):
                return "Build · Big Pickle"

            def _send_text(self, text, enter_count=None):
                self.sent_prompts.append(text)

            def _wait_for_turn_reply(self, **kwargs):
                return "ok\n[[DONE]]"

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = ActiveBusyAfterFailedWorker(
                worker_id="active-busy-after-failed-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="opencode/big-pickle"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.agent_started = True
            worker.current_task_runtime_status = "running"
            result = worker.run_turn(label="repair_turn", prompt="repair", required_tokens=["[[DONE]]"], timeout_sec=2.0)

        self.assertTrue(result.ok)
        self.assertEqual(worker.ensure_timeouts, [60.0, 2.0])
        self.assertEqual(len(worker.sent_prompts), 1)
        self.assertNotIn("系统不会自动重启该智能体", result.clean_output)

    def test_run_turn_failure_does_not_leave_runtime_status_running(self):
        class FailingTurnWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.state_updates = []

            def _write_state(self, status, *, note, extra=None):
                self.state_updates.append((status.value, note, extra or {}))

            def _append_transcript(self, title, body):
                return None

            def ensure_agent_ready(self, timeout_sec=60.0):
                self.pane_id = "%1"
                self.agent_started = True
                self.agent_ready = True
                self.agent_state = AgentRuntimeState.READY
                self.wrapper_state = WrapperState.READY
                self.current_command = "node"
                self.current_path = str(self.work_dir)

            def target_exists(self, target=None):
                return False

            def _send_text(self, text, enter_count=None):
                return None

            def _wait_for_turn_reply(self, **kwargs):
                raise RuntimeError("reply contract failed")

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = FailingTurnWorker(
                worker_id="failed-turn-runtime-status-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="opencode/big-pickle"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            result = worker.run_turn(label="bad_turn", prompt="hello", required_tokens=["[[DONE]]"], timeout_sec=2.0)

        self.assertFalse(result.ok)
        self.assertEqual(worker.state_updates[-1][0], "failed")
        self.assertEqual(worker.state_updates[-1][2]["current_task_runtime_status"], "")

    def test_ensure_reviewers_ready_blocks_failed_busy_worker_without_restart(self):
        class FailedBusyWorker:
            def __init__(self):
                self.restart_count = 0
                self.ensure_timeouts = []

            def read_state(self):
                return {
                    "status": "failed" if self.restart_count == 0 else "ready",
                    "result_status": "failed" if self.restart_count == 0 else "pending",
                    "current_task_runtime_status": "running",
                }

            def get_agent_state(self, observation=None):
                return AgentRuntimeState.BUSY if self.restart_count == 0 else AgentRuntimeState.READY

            def ensure_agent_ready(self, timeout_sec=60.0):
                self.ensure_timeouts.append(timeout_sec)

            def _restart_stale_busy_agent_for_turn_start(self, *, timeout_sec, reason):
                self.restart_count += 1
                self.restart_reason = reason
                self.restart_timeout = timeout_sec

        worker = FailedBusyWorker()

        with self.assertRaises(RuntimeError) as raised:
            ensure_reviewers_ready(None, [worker], timeout_sec=3.0, allow_completed_nonready=True)

        self.assertEqual(worker.restart_count, 0)
        self.assertIn("需要人工介入", str(raised.exception))
        self.assertIn("系统不会自动重启该智能体", str(raised.exception))
        self.assertEqual(worker.ensure_timeouts, [])

    def test_run_turn_keeps_waiting_for_same_completion_turn_when_timeout_agent_still_busy(self):
        class SlowBusyCompletionWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.sent_prompts = []
                self.wait_calls = 0
                self.state_notes = []

            def _write_state(self, status, *, note, extra=None):
                self.state_notes.append((status.value, note, extra or {}))

            def _append_transcript(self, title, body):
                return None

            def ensure_agent_ready(self, timeout_sec=60.0):
                self.pane_id = "%1"
                self.agent_ready = True
                self.agent_started = True
                self.agent_state = AgentRuntimeState.READY
                self.wrapper_state = WrapperState.READY
                self.current_command = "node"
                self.current_path = str(self.work_dir)
                self.last_pane_title = "TmuxCodingTeam"

            def target_exists(self, target=None):
                return True

            def capture_visible(self, tail_lines=500):
                return "working"

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="working",
                    raw_log_delta="",
                    raw_log_tail="esc to interrupt",
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-28T19:16:56",
                    pane_title="⠋ TmuxCodingTeam",
                )

            def _send_text(self, text, enter_count=None):
                self.sent_prompts.append(text)

            def _wait_for_prompt_submission(self, *, prompt, timeout_sec):
                return self.observe()

            def wait_for_turn_artifacts(self, *, contract, task_status_path=None, timeout_sec):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    raise TimeoutError("等待 turn 文件结果超时")
                artifact_path.write_text("", encoding="utf-8")
                contract.status_path.write_text(
                    json.dumps(
                        {
                            "schema_version": "1.0",
                            "turn_id": contract.turn_id,
                            "phase": contract.phase,
                            "status": "done",
                            "written_at": "2026-04-28T19:17:00+08:00",
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                if task_status_path is not None:
                    write_task_status(task_status_path, status="done")
                self.current_task_runtime_status = "done"
                return contract.validator(contract.status_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            contract_path = root / "review.json"
            artifact_path = root / "review.md"

            def validator(path: Path) -> TurnFileResult:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return TurnFileResult(
                    status_path=str(path),
                    payload=payload,
                    artifact_paths={"review_md": str(artifact_path)},
                    artifact_hashes={"review_md": "sha256:md"},
                    validated_at="2026-04-28T19:17:01",
                )

            worker = SlowBusyCompletionWorker(
                worker_id="slow-busy-completion-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5"),
                runtime_root=root / "runtime",
            )
            result = worker.run_turn(
                label="overall_review_again_测试工程师_round_5",
                prompt="write review files only",
                completion_contract=TurnFileContract(
                    turn_id="overall_review_全面复核_测试工程师",
                    phase="复核阶段",
                    status_path=contract_path,
                    validator=validator,
                    quiet_window_sec=0.0,
                ),
                timeout_sec=1.0,
            )

        self.assertTrue(result.ok)
        self.assertEqual(len(worker.sent_prompts), 1)
        self.assertEqual(worker.wait_calls, 2)
        self.assertTrue(any(note.startswith("still_running:") for _, note, _ in worker.state_notes))

    def test_run_turn_keeps_waiting_when_prompt_confirmation_times_out_but_codex_is_busy(self):
        class BusyAfterPromptTimeoutWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.sent_prompts = []
                self.wait_calls = 0

            def _append_transcript(self, title, body):
                return None

            def ensure_agent_ready(self, timeout_sec=60.0):
                self.pane_id = "%1"
                self.agent_ready = True
                self.agent_started = True
                self.agent_state = AgentRuntimeState.READY
                self.wrapper_state = WrapperState.READY
                self.current_command = "node"
                self.current_path = str(self.work_dir)
                self.last_pane_title = "TmuxCodingTeam"

            def target_exists(self, target=None):
                return True

            def capture_visible(self, tail_lines=500):
                return "• Planning scan and write actions (20s • esc to interrupt)"

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="• Planning scan and write actions (20s • esc to interrupt)\n\n› Explain this codebase",
                    raw_log_delta="",
                    raw_log_tail="• Planning scan and write actions (20s • esc to interrupt)",
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-30T10:32:39",
                    pane_title="⠋ TmuxCodingTeam",
                )

            def _send_text(self, text, enter_count=None):
                self.sent_prompts.append(text)

            def _wait_for_prompt_submission(self, *, prompt, timeout_sec):
                raise TimeoutError("等待智能体确认收到 prompt 超时")

            def wait_for_turn_artifacts(self, *, contract, task_status_path=None, timeout_sec):
                self.wait_calls += 1
                artifact_path.write_text("", encoding="utf-8")
                contract.status_path.write_text(
                    json.dumps(
                        {
                            "schema_version": "1.0",
                            "turn_id": contract.turn_id,
                            "phase": contract.phase,
                            "status": "done",
                            "written_at": "2026-04-30T10:33:00+08:00",
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                if task_status_path is not None:
                    write_task_status(task_status_path, status="done")
                self.current_task_runtime_status = "done"
                return contract.validator(contract.status_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            contract_path = root / "routing_status.json"
            artifact_path = root / "AGENTS.md"

            def validator(path: Path) -> TurnFileResult:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return TurnFileResult(
                    status_path=str(path),
                    payload=payload,
                    artifact_paths={"agents": str(artifact_path)},
                    artifact_hashes={"agents": "sha256:agents"},
                    validated_at="2026-04-30T10:33:01",
                )

            worker = BusyAfterPromptTimeoutWorker(
                worker_id="busy-after-prompt-timeout-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5"),
                runtime_root=root / "runtime",
            )
            result = worker.run_turn(
                label="create_routing_layer",
                prompt="write routing files only",
                completion_contract=TurnFileContract(
                    turn_id="create_routing_layer_1",
                    phase="routing_layer_create",
                    status_path=contract_path,
                    validator=validator,
                    quiet_window_sec=0.0,
                ),
                timeout_sec=1.0,
            )

        self.assertTrue(result.ok)
        self.assertEqual(len(worker.sent_prompts), 1)
        self.assertEqual(worker.wait_calls, 1)

    def test_run_turn_does_not_retry_when_agent_becomes_busy_after_prompt_timeout(self):
        class DelayedBusyAfterPromptTimeoutWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.sent_prompts = []
                self.wait_calls = 0
                self.post_timeout_observes = 0
                self.prompt_timeout_seen = False

            def _append_transcript(self, title, body):
                return None

            def ensure_agent_ready(self, timeout_sec=60.0):
                self.pane_id = "%1"
                self.agent_ready = True
                self.agent_started = True
                self.agent_state = AgentRuntimeState.READY
                self.wrapper_state = WrapperState.READY
                self.current_command = "node"
                self.current_path = str(self.work_dir)
                self.last_pane_title = "◇  Ready"

            def target_exists(self, target=None):
                return True

            def capture_visible(self, tail_lines=500):
                return "fake-visible"

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                if self.prompt_timeout_seen:
                    self.post_timeout_observes += 1
                busy = self.prompt_timeout_seen and self.post_timeout_observes >= 2
                visible_text = (
                    "⠋ Thinking... (esc to cancel)\n *   Type your message or @path/to/file"
                    if busy
                    else "*   Type your message or @path/to/file"
                )
                pane_title = "✦  Working..." if busy else "◇  Ready"
                return WorkerObservation(
                    visible_text=visible_text,
                    raw_log_delta=visible_text,
                    raw_log_tail=visible_text,
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-05-02T21:10:49",
                    pane_title=pane_title,
                )

            def _send_text(self, text, enter_count=None):
                self.sent_prompts.append(text)

            def _wait_for_prompt_submission(self, *, prompt, timeout_sec):
                self.prompt_timeout_seen = True
                raise TimeoutError("等待智能体确认收到 prompt 超时")

            def wait_for_turn_artifacts(self, *, contract, task_status_path=None, timeout_sec):
                self.wait_calls += 1
                artifact_path.write_text("", encoding="utf-8")
                contract.status_path.write_text(
                    json.dumps(
                        {
                            "schema_version": "1.0",
                            "turn_id": contract.turn_id,
                            "phase": contract.phase,
                            "status": "done",
                            "written_at": "2026-05-02T21:11:47+08:00",
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                if task_status_path is not None:
                    write_task_status(task_status_path, status="done")
                self.current_task_runtime_status = "done"
                return contract.validator(contract.status_path)

        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch(
            "tmux_core.runtime.tmux_runtime.time.sleep",
            return_value=None,
        ):
            root = Path(tmp_dir)
            contract_path = root / "requirements_status.json"
            artifact_path = root / "requirements.md"

            def validator(path: Path) -> TurnFileResult:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return TurnFileResult(
                    status_path=str(path),
                    payload=payload,
                    artifact_paths={"requirements": str(artifact_path)},
                    artifact_hashes={"requirements": "sha256:req"},
                    validated_at="2026-05-02T21:11:48",
                )

            worker = DelayedBusyAfterPromptTimeoutWorker(
                worker_id="delayed-busy-after-prompt-timeout-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=root / "runtime",
            )
            result = worker.run_turn(
                label="requirements_clarification_round_1",
                prompt="write requirements files only",
                completion_contract=TurnFileContract(
                    turn_id="requirements_clarification_1",
                    phase="requirements_clarification",
                    status_path=contract_path,
                    validator=validator,
                    quiet_window_sec=0.0,
                ),
                timeout_sec=1.0,
            )

        self.assertTrue(result.ok)
        self.assertEqual(len(worker.sent_prompts), 1)
        self.assertEqual(worker.wait_calls, 1)
        self.assertGreaterEqual(worker.post_timeout_observes, 2)

    def test_run_turn_marks_wrapper_ready_after_success(self):
        class SubmitStateWorker(TmuxBatchWorker):
            def _append_transcript(self, title, body):
                return None

            def ensure_agent_ready(self, timeout_sec=60.0):
                self.pane_id = "%1"
                self.agent_ready = True
                self.agent_started = True
                self.agent_state = AgentRuntimeState.READY
                self.last_pane_title = "TmuxCodingTeam"
                self.current_command = "codex"
                self.current_path = str(self.work_dir)

            def target_exists(self, target=None):
                return True

            def capture_visible(self, tail_lines=500):
                return "fake-visible"

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                self.agent_state = AgentRuntimeState.READY
                return WorkerObservation(
                    visible_text="❯",
                    raw_log_delta="",
                    raw_log_tail="❯",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-15T00:00:00",
                    pane_title="TmuxCodingTeam",
                )

            def _send_text(self, text, enter_count=None):
                write_task_status(self.current_task_status_path, status="done")

            def _wait_for_turn_reply(self, **kwargs):
                return "ok\n[[DONE]]"

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = SubmitStateWorker(
                worker_id="success-ready-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            result = worker.run_turn(label="submit_case", prompt="hello", required_tokens=["[[DONE]]"], timeout_sec=0.05)
            self.assertTrue(result.ok)
            self.assertEqual(worker.wrapper_state, WrapperState.READY)

    def test_wait_for_turn_artifacts_returns_after_stable_file_validation(self):
        class FileContractWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="processing",
                    raw_log_delta="delta",
                    raw_log_tail="tail",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-12T00:00:00",
                )

            def capture_visible(self, tail_lines=500):
                return "visible"

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            artifact_path = root / "artifact.txt"
            artifact_path.write_text("artifact-body", encoding="utf-8")
            status_path = root / "turn_status.json"
            status_path.write_text(json.dumps({"ok": True}), encoding="utf-8")

            def validator(path: Path) -> TurnFileResult:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not payload.get("ok"):
                    raise ValueError("not ready")
                return TurnFileResult(
                    status_path=str(path),
                    payload=payload,
                    artifact_paths={"artifact.txt": str(artifact_path)},
                    artifact_hashes={"artifact.txt": "sha256:test"},
                    validated_at="2026-04-12T00:00:00",
                )

            worker = FileContractWorker(
                worker_id="file-contract-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            result = worker.wait_for_turn_artifacts(
                contract=TurnFileContract(
                    turn_id="create_routing_layer_1",
                    phase="routing_layer_create",
                    status_path=status_path,
                    validator=validator,
                    quiet_window_sec=1.0,
                ),
                timeout_sec=2.0,
            )
            self.assertEqual(Path(result.status_path).resolve(), status_path.resolve())
            self.assertEqual(Path(result.artifact_paths["artifact.txt"]).resolve(), artifact_path.resolve())

    def test_wait_for_turn_artifacts_ignores_transient_target_exists_false_after_observe(self):
        class FileContractWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return False

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="processing",
                    raw_log_delta="delta",
                    raw_log_tail="tail",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-12T00:00:00",
                )

            def capture_visible(self, tail_lines=500):
                return "visible"

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            artifact_path = root / "artifact.txt"
            artifact_path.write_text("artifact-body", encoding="utf-8")
            status_path = root / "turn_status.json"
            status_path.write_text(json.dumps({"ok": True}), encoding="utf-8")

            def validator(path: Path) -> TurnFileResult:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return TurnFileResult(
                    status_path=str(path),
                    payload=payload,
                    artifact_paths={"artifact.txt": str(artifact_path)},
                    artifact_hashes={"artifact.txt": "sha256:test"},
                    validated_at="2026-04-12T00:00:00",
                )

            worker = FileContractWorker(
                worker_id="transient-target-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            result = worker.wait_for_turn_artifacts(
                contract=TurnFileContract(
                    turn_id="audit_routing_layer_2",
                    phase="routing_layer_audit",
                    status_path=status_path,
                    validator=validator,
                    quiet_window_sec=1.0,
                ),
                timeout_sec=2.0,
            )
            self.assertEqual(Path(result.status_path).resolve(), status_path.resolve())

    def test_wait_for_turn_artifacts_requires_done_status(self):
        class FileContractWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.observe_count = 0

            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                self.observe_count += 1
                if self.observe_count == 1:
                    task_status_path.write_text('{"status": "running"}', encoding="utf-8")
                else:
                    task_status_path.write_text('{"status": "done"}', encoding="utf-8")
                return WorkerObservation(
                    visible_text="processing",
                    raw_log_delta="delta",
                    raw_log_tail="tail",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-12T00:00:00",
                )

            def capture_visible(self, tail_lines=500):
                return "visible"

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            artifact_path = root / "artifact.txt"
            artifact_path.write_text("artifact-body", encoding="utf-8")
            status_path = root / "turn_status.json"
            status_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
            task_status_path = root / "task_runtime.json"

            def validator(path: Path) -> TurnFileResult:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return TurnFileResult(
                    status_path=str(path),
                    payload=payload,
                    artifact_paths={"artifact.txt": str(artifact_path)},
                    artifact_hashes={"artifact.txt": "sha256:test"},
                    validated_at="2026-04-12T00:00:00",
                )

            worker = FileContractWorker(
                worker_id="file-contract-signal-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            result = worker.wait_for_turn_artifacts(
                contract=TurnFileContract(
                    turn_id="create_routing_layer_1",
                    phase="routing_layer_create",
                    status_path=status_path,
                    validator=validator,
                    quiet_window_sec=1.0,
                ),
                task_status_path=task_status_path,
                timeout_sec=2.0,
            )
            self.assertEqual(Path(result.status_path).resolve(), status_path.resolve())
            self.assertGreaterEqual(worker.observe_count, 2)

    def test_wait_for_turn_artifacts_marks_task_done_when_files_stable_and_agent_ready(self):
        class FileContractReadyWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.observe_count = 0

            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                self.observe_count += 1
                self.agent_state = AgentRuntimeState.READY
                return WorkerObservation(
                    visible_text="› Continue working in @filename",
                    raw_log_delta="" if self.observe_count == 1 else "delta",
                    raw_log_tail="› Continue working in @filename",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-17T00:00:00",
                    pane_title="TmuxCodingTeam",
                )

            def capture_visible(self, tail_lines=500):
                return "› Continue working in @filename"

            def _maybe_send_task_completion_nudge(self, **kwargs):
                return False

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            artifact_path = root / "artifact.txt"
            artifact_path.write_text("artifact-body", encoding="utf-8")
            status_path = root / "turn_status.json"
            status_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
            task_status_path = root / "task_runtime.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")

            def validator(path: Path) -> TurnFileResult:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return TurnFileResult(
                    status_path=str(path),
                    payload=payload,
                    artifact_paths={"artifact.txt": str(artifact_path)},
                    artifact_hashes={"artifact.txt": "sha256:test"},
                    validated_at="2026-04-17T00:00:00",
                )

            worker = FileContractReadyWorker(
                worker_id="file-contract-ready-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            result = worker.wait_for_turn_artifacts(
                contract=TurnFileContract(
                    turn_id="requirements_review_r1",
                    phase="需求评审",
                    status_path=status_path,
                    validator=validator,
                    quiet_window_sec=0.2,
                ),
                task_status_path=task_status_path,
                timeout_sec=1.0,
            )

            self.assertEqual(Path(result.status_path).resolve(), status_path.resolve())
            self.assertEqual(json.loads(task_status_path.read_text(encoding="utf-8")), {"status": "done"})
            self.assertEqual(worker.current_task_runtime_status, "done")
            self.assertGreaterEqual(worker.observe_count, 2)

    def test_wait_for_turn_artifacts_finishes_stable_routing_contract_even_when_agent_busy(self):
        class BusyFileContractWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="Thinking... (esc to cancel, 43s)\n*   Type your message or @path/to/file",
                    raw_log_delta="",
                    raw_log_tail="Thinking... (esc to cancel, 43s)\n*   Type your message or @path/to/file",
                    current_command="gemini",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-05-02T20:53:00",
                )

            def capture_visible(self, tail_lines=500):
                return "Thinking... (esc to cancel, 43s)"

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            artifact_path = root / "AGENTS.md"
            artifact_path.write_text("routing protocol", encoding="utf-8")
            status_path = root / "turn_status.json"
            task_status_path = root / "task_runtime.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")

            def validator(path: Path) -> TurnFileResult:
                if not path.exists():
                    path.write_text(json.dumps({"ok": True}), encoding="utf-8")
                payload = json.loads(path.read_text(encoding="utf-8"))
                return TurnFileResult(
                    status_path=str(path),
                    payload=payload,
                    artifact_paths={"agents": str(artifact_path)},
                    artifact_hashes={"agents": "sha256:agents"},
                    validated_at="2026-05-02T20:53:01",
                )

            worker = BusyFileContractWorker(
                worker_id="stable-routing-contract-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=root / "runtime",
            )
            worker.agent_started = True
            worker.current_task_runtime_status = "running"
            result = worker.wait_for_turn_artifacts(
                contract=TurnFileContract(
                    turn_id="create_routing_layer_1",
                    phase="routing_layer_create",
                    status_path=status_path,
                    validator=validator,
                    quiet_window_sec=0.0,
                    kind="routing_file_contract",
                ),
                task_status_path=task_status_path,
                timeout_sec=1.0,
            )

            self.assertEqual(Path(result.status_path).resolve(), status_path.resolve())
            self.assertEqual(json.loads(task_status_path.read_text(encoding="utf-8")), {"status": "done"})
            self.assertEqual(worker.current_task_runtime_status, "done")

    def test_wait_for_turn_artifacts_finishes_stable_review_contract_even_when_agent_busy(self):
        class BusyReviewContractWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="Build · Big Pickle\nesc interrupt",
                    raw_log_delta="",
                    raw_log_tail="Build · Big Pickle\nesc interrupt",
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-05-03T21:38:00",
                )

            def get_agent_state(self, observation=None):
                return AgentRuntimeState.BUSY

            def capture_visible(self, tail_lines=500):
                return "Build · Big Pickle"

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            review_md_path = root / "需求A_代码评审记录_测试工程师.md"
            review_json_path = root / "需求A_评审记录_测试工程师.json"
            task_status_path = root / "task_runtime.json"
            review_md_path.write_text("", encoding="utf-8")
            review_json_path.write_text(
                json.dumps([{"task_name": "M1-T1", "review_pass": True}], ensure_ascii=False),
                encoding="utf-8",
            )
            time.sleep(0.01)
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            review_md_path.write_text("", encoding="utf-8")
            review_json_path.write_text(
                json.dumps([{"task_name": "M1-T1", "review_pass": True}], ensure_ascii=False),
                encoding="utf-8",
            )

            def validator(path: Path) -> TurnFileResult:
                payload = json.loads(path.read_text(encoding="utf-8"))
                matched = payload[0]
                if matched["task_name"] != "M1-T1" or matched["review_pass"] is not True:
                    raise ValueError("review payload mismatch")
                if review_md_path.read_text(encoding="utf-8").strip():
                    raise ValueError("review md must be empty on pass")
                return TurnFileResult(
                    status_path=str(path.resolve()),
                    payload={"task_name": "M1-T1", "review_pass": True},
                    artifact_paths={
                        "review_md": str(review_md_path.resolve()),
                        "review_json": str(review_json_path.resolve()),
                    },
                    artifact_hashes={
                        "review_md": "sha256:empty",
                        "review_json": "sha256:pass",
                    },
                    validated_at="2026-05-03T21:38:00",
                )

            worker = BusyReviewContractWorker(
                worker_id="stable-review-contract-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="opencode/big-pickle"),
                runtime_root=root / "runtime",
            )
            worker.agent_started = True
            worker.current_task_runtime_status = "running"
            result = worker.wait_for_turn_artifacts(
                contract=TurnFileContract(
                    turn_id="development_review_M1-T1_R2",
                    phase="任务开发",
                    status_path=review_json_path,
                    validator=validator,
                    quiet_window_sec=0.0,
                    kind="review_round",
                ),
                task_status_path=task_status_path,
                timeout_sec=1.0,
            )

            self.assertEqual(Path(result.status_path).resolve(), review_json_path.resolve())
            self.assertEqual(json.loads(task_status_path.read_text(encoding="utf-8")), {"status": "done"})
            self.assertEqual(worker.current_task_runtime_status, "done")

    def test_wait_for_turn_artifacts_rejects_stale_review_contract_when_agent_ready(self):
        class ReadyReviewContractWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="› ready",
                    raw_log_delta="",
                    raw_log_tail="› ready",
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-05-03T21:39:00",
                )

            def get_agent_state(self, observation=None):
                return AgentRuntimeState.READY

            def capture_visible(self, tail_lines=500):
                return "› ready"

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            review_md_path = root / "需求A_代码评审记录_测试工程师.md"
            review_json_path = root / "需求A_评审记录_测试工程师.json"
            task_status_path = root / "task_runtime.json"
            review_md_path.write_text("", encoding="utf-8")
            review_json_path.write_text(
                json.dumps([{"task_name": "M1-T1", "review_pass": True}], ensure_ascii=False),
                encoding="utf-8",
            )
            time.sleep(0.01)
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")

            def validator(path: Path) -> TurnFileResult:
                payload = json.loads(path.read_text(encoding="utf-8"))
                matched = payload[0]
                if matched["task_name"] != "M1-T1" or matched["review_pass"] is not True:
                    raise ValueError("review payload mismatch")
                if review_md_path.read_text(encoding="utf-8").strip():
                    raise ValueError("review md must be empty on pass")
                return TurnFileResult(
                    status_path=str(path.resolve()),
                    payload={"task_name": "M1-T1", "review_pass": True},
                    artifact_paths={
                        "review_md": str(review_md_path.resolve()),
                        "review_json": str(review_json_path.resolve()),
                    },
                    artifact_hashes={
                        "review_md": "sha256:empty",
                        "review_json": "sha256:pass",
                    },
                    validated_at="2026-05-03T21:39:00",
                )

            worker = ReadyReviewContractWorker(
                worker_id="stale-review-contract-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="opencode/big-pickle"),
                runtime_root=root / "runtime",
            )
            worker.agent_started = True
            worker.current_task_runtime_status = "running"

            with self.assertRaisesRegex(RuntimeError, "stale_review_round_artifacts"):
                worker.wait_for_turn_artifacts(
                    contract=TurnFileContract(
                        turn_id="development_review_M1-T1_R2",
                        phase="任务开发",
                        status_path=review_json_path,
                        validator=validator,
                        quiet_window_sec=0.0,
                        kind="review_round",
                    ),
                    task_status_path=task_status_path,
                    timeout_sec=1.0,
                )

            self.assertEqual(json.loads(task_status_path.read_text(encoding="utf-8")), {"status": "running"})

    def test_wait_for_turn_artifacts_raises_contract_violation_after_done_with_invalid_files(self):
        class InvalidFileContractWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="› Continue working in @filename",
                    raw_log_delta="",
                    raw_log_tail="› Continue working in @filename",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-20T00:00:00",
                )

            def capture_visible(self, tail_lines=500):
                return "› Continue working in @filename"

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            status_path = root / "turn_status.json"
            status_path.write_text(json.dumps({"task_name": "需求评审", "review_pass": False}), encoding="utf-8")
            task_status_path = root / "task_runtime.json"
            task_status_path.write_text('{"status": "done"}', encoding="utf-8")

            def validator(path: Path) -> TurnFileResult:
                raise ValueError("审核器未通过，但评审 markdown 为空")

            worker = InvalidFileContractWorker(
                worker_id="invalid-file-contract-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            with mock.patch("tmux_core.runtime.tmux_runtime.TURN_ARTIFACT_POST_DONE_GRACE_SEC", 0.0):
                with self.assertRaisesRegex(RuntimeError, "turn artifacts contract violation after task completion"):
                    worker.wait_for_turn_artifacts(
                        contract=TurnFileContract(
                            turn_id="requirements_review_r2",
                            phase="需求评审",
                            status_path=status_path,
                            validator=validator,
                            quiet_window_sec=0.0,
                        ),
                        task_status_path=task_status_path,
                        timeout_sec=2.0,
                    )

    def test_wait_for_turn_artifacts_fails_fast_when_probe_reports_session_loss(self):
        class SessionLostWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="",
                    raw_log_delta="",
                    raw_log_tail="",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=False,
                    log_mtime=0.0,
                    observed_at="2026-04-24T00:00:00",
                    pane_title="",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            status_path = root / "turn_status.json"
            task_status_path = root / "task_runtime.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            worker = SessionLostWorker(
                worker_id="turn-session-lost-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"

            with self.assertRaisesRegex(RuntimeError, "tmux pane exited"):
                worker.wait_for_turn_artifacts(
                    contract=TurnFileContract(
                        turn_id="turn-session-lost",
                        phase="phase",
                        status_path=status_path,
                        validator=lambda path: (_ for _ in ()).throw(FileNotFoundError(path)),
                    ),
                    task_status_path=task_status_path,
                    timeout_sec=1.0,
                )

    def test_wait_for_turn_artifacts_raises_contract_violation_when_runtime_stalls_without_files(self):
        class StalledArtifactWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="审查中",
                    raw_log_delta="",
                    raw_log_tail="审查中",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-25T00:00:00",
                    pane_title="TmuxCodingTeam",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            status_path = root / "review.json"
            task_status_path = root / "task_runtime.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")

            def validator(path: Path) -> TurnFileResult:
                raise FileNotFoundError(path)

            worker = StalledArtifactWorker(
                worker_id="stalled-artifact-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            worker.current_task_runtime_status = "running"
            worker._last_terminal_change_monotonic = time.monotonic() - 60.0
            worker.terminal_recently_changed = False

            with mock.patch("T02_tmux_agents.TASK_CONTRACT_STALL_IDLE_SEC", 0.0):
                with self.assertRaisesRegex(RuntimeError, TURN_ARTIFACT_CONTRACT_ERROR_PREFIX):
                    worker.wait_for_turn_artifacts(
                        contract=TurnFileContract(
                            turn_id="requirements_review_stalled",
                            phase="需求评审",
                            status_path=status_path,
                            validator=validator,
                            quiet_window_sec=0.0,
                        ),
                        task_status_path=task_status_path,
                        timeout_sec=1.0,
                    )

    def test_wait_for_turn_artifacts_raises_contract_violation_when_invalid_file_stalls_running_task(self):
        class StalledInvalidArtifactWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="审核通过\nAsk anything...",
                    raw_log_delta="",
                    raw_log_tail="审核通过\nAsk anything...",
                    current_command="opencode",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-05-03T09:35:00",
                    pane_title="TmuxCodingTeam",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            status_path = root / "review.json"
            status_path.write_text("[]", encoding="utf-8")
            task_status_path = root / "task_runtime.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")

            def validator(path: Path) -> TurnFileResult:
                raise ValueError("review payload must include task_name")

            worker = StalledInvalidArtifactWorker(
                worker_id="stalled-invalid-artifact-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="opencode/minimax-m2.5-free"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            worker.current_task_runtime_status = "running"
            worker._last_terminal_change_monotonic = time.monotonic() - 60.0
            worker.terminal_recently_changed = False

            with mock.patch("T02_tmux_agents.TASK_CONTRACT_STALL_IDLE_SEC", 0.0):
                with self.assertRaisesRegex(RuntimeError, r"runtime_stalled"):
                    worker.wait_for_turn_artifacts(
                        contract=TurnFileContract(
                            turn_id="task_split_review_stalled_invalid",
                            phase="任务拆分评审",
                            status_path=status_path,
                            validator=validator,
                            quiet_window_sec=0.0,
                        ),
                        task_status_path=task_status_path,
                        timeout_sec=1.0,
                    )

    def test_wait_for_turn_artifacts_stalls_on_stable_invalid_file_even_when_terminal_changes(self):
        class BusyInvalidArtifactWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.observe_count = 0

            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                self.observe_count += 1
                self.agent_state = AgentRuntimeState.BUSY
                now = time.monotonic()
                self._last_terminal_change_monotonic = now
                self.terminal_recently_changed = True
                return WorkerObservation(
                    visible_text=f"Build · Big Pickle {self.observe_count}",
                    raw_log_delta="",
                    raw_log_tail=f"Build · Big Pickle {self.observe_count}",
                    current_command="opencode",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-05-03T22:20:00",
                    pane_title="OC | review",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            status_path = root / "review.json"
            status_path.write_text("[]", encoding="utf-8")
            task_status_path = root / "task_runtime.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")

            def validator(path: Path) -> TurnFileResult:
                raise ValueError("review payload must include task_name")

            worker = BusyInvalidArtifactWorker(
                worker_id="busy-invalid-artifact-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="opencode/big-pickle"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            worker.current_task_runtime_status = "running"

            with mock.patch("T02_tmux_agents.TASK_CONTRACT_STALL_IDLE_SEC", 0.0):
                with self.assertRaisesRegex(RuntimeError, r"runtime_stalled"):
                    worker.wait_for_turn_artifacts(
                        contract=TurnFileContract(
                            turn_id="development_review_stalled_invalid",
                            phase="任务开发",
                            status_path=status_path,
                            validator=validator,
                            quiet_window_sec=0.0,
                        ),
                        task_status_path=task_status_path,
                        timeout_sec=1.0,
                    )

    def test_wait_for_turn_artifacts_checks_liveness_after_valid_files_before_done(self):
        class SessionLostWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="",
                    raw_log_delta="",
                    raw_log_tail="",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=False,
                    log_mtime=0.0,
                    observed_at="2026-04-24T00:00:00",
                    pane_title="",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            status_path = root / "turn_status.json"
            status_path.write_text('{"ok": true}', encoding="utf-8")
            task_status_path = root / "task_runtime.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")

            def validator(path: Path) -> TurnFileResult:
                return TurnFileResult(
                    status_path=str(path),
                    payload={"ok": True},
                    artifact_paths={},
                    artifact_hashes={},
                    validated_at="2026-04-24T00:00:00",
                )

            worker = SessionLostWorker(
                worker_id="turn-valid-session-lost-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            with self.assertRaisesRegex(RuntimeError, "tmux pane exited"):
                worker.wait_for_turn_artifacts(
                    contract=TurnFileContract(
                        turn_id="turn-valid-session-lost",
                        phase="phase",
                        status_path=status_path,
                        validator=validator,
                        quiet_window_sec=0.0,
                    ),
                    task_status_path=task_status_path,
                    timeout_sec=1.0,
                )

    def test_wait_for_turn_artifacts_does_not_treat_ready_terminal_as_completion_for_invalid_files(self):
        class ReadyInvalidFileContractWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="› Continue working in @filename",
                    raw_log_delta="",
                    raw_log_tail="› Continue working in @filename",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-20T00:00:00",
                    pane_title="TmuxCodingTeam",
                )

            def _can_finalize_turn_artifacts_without_helper(self, observation):
                return True

            def capture_visible(self, tail_lines=500):
                return "› Continue working in @filename"

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            expected_status_path = root / "需求A_评审记录_架构师-柳土獐.json"
            task_status_path = root / "task_runtime.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")

            def validator(path: Path) -> TurnFileResult:
                raise FileNotFoundError(f"缺少审核 JSON 文件: {path}")

            worker = ReadyInvalidFileContractWorker(
                worker_id="ready-invalid-file-contract-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.agent_started = True
            with mock.patch("tmux_core.runtime.tmux_runtime.TURN_ARTIFACT_POST_DONE_GRACE_SEC", 0.0):
                with self.assertRaises(TimeoutError):
                    worker.wait_for_turn_artifacts(
                        contract=TurnFileContract(
                            turn_id="development_review_M4-T1_架构师",
                            phase="任务开发",
                            status_path=expected_status_path,
                            validator=validator,
                            quiet_window_sec=0.0,
                            kind="review_round",
                        ),
                        task_status_path=task_status_path,
                        timeout_sec=1.0,
                    )

    def test_wait_for_turn_artifacts_allows_late_files_after_done_before_grace_expires(self):
        class LateArtifactWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.observe_count = 0

            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                self.observe_count += 1
                if self.observe_count >= 3:
                    status_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
                return WorkerObservation(
                    visible_text="processing",
                    raw_log_delta="delta",
                    raw_log_tail="tail",
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-20T00:00:00",
                )

            def capture_visible(self, tail_lines=500):
                return "visible"

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            artifact_path = root / "artifact.txt"
            artifact_path.write_text("artifact-body", encoding="utf-8")
            status_path = root / "turn_status.json"
            status_path.write_text(json.dumps({"ok": False}), encoding="utf-8")
            task_status_path = root / "task_runtime.json"
            task_status_path.write_text('{"status": "done"}', encoding="utf-8")

            def validator(path: Path) -> TurnFileResult:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not payload.get("ok"):
                    raise ValueError("not ready")
                return TurnFileResult(
                    status_path=str(path),
                    payload=payload,
                    artifact_paths={"artifact.txt": str(artifact_path)},
                    artifact_hashes={"artifact.txt": "sha256:test"},
                    validated_at="2026-04-20T00:00:00",
                )

            worker = LateArtifactWorker(
                worker_id="late-artifact-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            with mock.patch("tmux_core.runtime.tmux_runtime.TURN_ARTIFACT_POST_DONE_GRACE_SEC", 2.0):
                result = worker.wait_for_turn_artifacts(
                    contract=TurnFileContract(
                        turn_id="task_split_review_r1",
                        phase="任务拆分",
                        status_path=status_path,
                        validator=validator,
                        quiet_window_sec=0.0,
                    ),
                    task_status_path=task_status_path,
                    timeout_sec=3.0,
                )

        self.assertEqual(Path(result.status_path).resolve(), status_path.resolve())
        self.assertGreaterEqual(worker.observe_count, 3)

    def test_wait_for_turn_artifacts_keeps_waiting_when_shell_returns_after_done(self):
        class ShellAfterDoneWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.observe_count = 0

            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                self.observe_count += 1
                if self.observe_count >= 3:
                    status_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
                return WorkerObservation(
                    visible_text="❯",
                    raw_log_delta="delta",
                    raw_log_tail="tail",
                    current_command="zsh",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-20T00:00:00",
                )

            def capture_visible(self, tail_lines=500):
                return "❯"

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            artifact_path = root / "artifact.txt"
            artifact_path.write_text("artifact-body", encoding="utf-8")
            status_path = root / "turn_status.json"
            status_path.write_text(json.dumps({"ok": False}), encoding="utf-8")
            task_status_path = root / "task_runtime.json"
            task_status_path.write_text('{"status": "done"}', encoding="utf-8")

            def validator(path: Path) -> TurnFileResult:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not payload.get("ok"):
                    raise ValueError("not ready")
                return TurnFileResult(
                    status_path=str(path),
                    payload=payload,
                    artifact_paths={"artifact.txt": str(artifact_path)},
                    artifact_hashes={"artifact.txt": "sha256:test"},
                    validated_at="2026-04-20T00:00:00",
                )

            worker = ShellAfterDoneWorker(
                worker_id="shell-after-done-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            with mock.patch("tmux_core.runtime.tmux_runtime.TURN_ARTIFACT_POST_DONE_GRACE_SEC", 2.0):
                result = worker.wait_for_turn_artifacts(
                    contract=TurnFileContract(
                        turn_id="requirements_review_r3",
                        phase="需求评审",
                        status_path=status_path,
                        validator=validator,
                        quiet_window_sec=0.0,
                    ),
                    task_status_path=task_status_path,
                    timeout_sec=3.0,
                )

        self.assertEqual(Path(result.status_path).resolve(), status_path.resolve())
        self.assertGreaterEqual(worker.observe_count, 3)






    def test_observe_tolerates_tmux_display_message_race(self):
        class RaceBackend(TmuxBackend):
            def has_session(self, session_name: str) -> bool:
                return True

            def target_exists(self, target_name: str) -> bool:
                return True

            def capture_visible(self, target: str, *, tail_lines: int = 500) -> str:
                return "› ready"

            def display_message(self, target: str, expression: str) -> str:
                raise subprocess.CalledProcessError(1, ["tmux", "display-message"])

            def tail_raw_log(
                    self,
                    raw_log_path: str | Path,
                    *,
                    last_offset: int = 0,
                    tail_bytes: int = 24000,
            ) -> tuple[str, str, int, float]:
                return "", "", last_offset, 0.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="observe-race-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
                existing_session_name="demo-session",
                existing_pane_id="%1",
                backend=RaceBackend(),
            )
            observation = worker.observe()
            self.assertTrue(observation.session_exists)
            self.assertEqual("", observation.current_command)
            self.assertEqual("› ready", observation.visible_text)
            self.assertEqual("", observation.pane_title)

    def test_observe_uses_10000_line_default_tail(self):
        class ObserveWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.tail_lines_seen: list[int] = []

            def session_exists(self) -> bool:
                return True

            def target_exists(self, target=None):  # noqa: ANN001
                _ = target
                return True

            def capture_visible(self, tail_lines=500):  # noqa: ANN001
                self.tail_lines_seen.append(tail_lines)
                return "› ready"

            def pane_dead(self) -> bool:
                return False

            def pane_current_command(self) -> str:
                return "codex"

            def pane_current_path(self) -> str:
                return str(self.work_dir)

            def pane_title(self) -> str:
                return ""

            def tail_raw_log(self, *, tail_bytes=24000):  # noqa: ANN001
                _ = tail_bytes
                return "", "", 0, 0.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = ObserveWorker(
                worker_id="observe-default-tail-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"

            worker.observe()

        self.assertEqual(worker.tail_lines_seen, [10000])

    def test_run_turn_with_completion_contract_uses_file_protocol_not_stdout_tokens(self):
        class CompletionContractWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.sent_prompts = []

            def _append_transcript(self, title, body):
                return None

            def ensure_agent_ready(self, timeout_sec=60.0):
                self.pane_id = "%1"
                self.agent_ready = True
                self.agent_started = True
                self.agent_state = AgentRuntimeState.READY
                self.current_command = "claude"
                self.current_path = str(self.work_dir)

            def target_exists(self, target=None):
                return True

            def capture_visible(self, tail_lines=500):
                return "visible"

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                delta = self.sent_prompts[-1] if self.sent_prompts else ""
                return WorkerObservation(
                    visible_text=f"processing\n{delta}".strip(),
                    raw_log_delta=delta,
                    raw_log_tail=f"processing\n{delta}".strip(),
                    current_command="claude",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-12T00:00:00",
                )

            def _send_text(self, text, enter_count=None):
                self.sent_prompts.append(text)
                artifact_path = self.work_dir / "artifact.json"
                artifact_path.write_text('{"ready": true}', encoding="utf-8")
                status_payload = {
                    "schema_version": "1.0",
                    "turn_id": "audit_routing_layer_1",
                    "phase": "routing_layer_audit",
                    "status": "done",
                    "written_at": "2026-04-12T00:00:00+08:00",
                }
                contract_path.write_text(json.dumps(status_payload, ensure_ascii=False), encoding="utf-8")
                write_task_status(self.current_task_status_path, status="done")

            def _wait_for_turn_reply(self, **kwargs):
                raise AssertionError("completion_contract path should not use stdout reply waiting")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            contract_path = root / "turn_status.json"
            artifact_path = root / "artifact.json"

            def validator(path: Path) -> TurnFileResult:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return TurnFileResult(
                    status_path=str(path),
                    payload=payload,
                    artifact_paths={"artifact.json": str(artifact_path)},
                    artifact_hashes={"artifact.json": "sha256:test"},
                    validated_at="2026-04-12T00:00:01",
                )

            worker = CompletionContractWorker(
                worker_id="completion-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="claude", model="haiku"),
                runtime_root=root / "runtime",
            )
            result = worker.run_turn(
                label="audit_routing_layer_1",
                prompt="write audit files only",
                completion_contract=TurnFileContract(
                    turn_id="audit_routing_layer_1",
                    phase="routing_layer_audit",
                    status_path=contract_path,
                    validator=validator,
                    quiet_window_sec=1.0,
                ),
                timeout_sec=2.0,
            )
            self.assertTrue(result.ok)
            self.assertEqual(len(worker.sent_prompts), 1)
            self.assertIn("write audit files only", worker.sent_prompts[0])
            self.assertNotIn(worker.current_task_status_path, worker.sent_prompts[0])
            parsed = json.loads(result.clean_output)
            self.assertEqual(Path(parsed["status_path"]).resolve(), contract_path.resolve())
            self.assertEqual(Path(parsed["artifact_paths"]["artifact.json"]).resolve(), artifact_path.resolve())

    def test_run_turn_with_completion_contract_does_not_finalize_after_unobserved_prompt_submission_timeout(self):
        class PromptTimeoutCompletionWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.sent_prompts = []

            def _append_transcript(self, title, body):
                return None

            def ensure_agent_ready(self, timeout_sec=60.0):
                self.pane_id = "%1"
                self.agent_ready = True
                self.agent_started = True
                self.agent_state = AgentRuntimeState.READY
                self.current_command = "node"
                self.current_path = str(self.work_dir)

            def target_exists(self, target=None):
                return True

            def capture_visible(self, tail_lines=500):
                return "You've reached your usage limit for this period. retrying in 14m"

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="146.1K ctrl+p commands",
                    raw_log_delta="",
                    raw_log_tail="146.1K ctrl+p commands",
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-24T00:00:00",
                    pane_title="OC | review",
                )

            def _send_text(self, text, enter_count=None):
                self.sent_prompts.append(text)
                review_md.write_text("", encoding="utf-8")
                review_json.write_text(
                    json.dumps([{"task_name": "M2-T3", "review_pass": True}], ensure_ascii=False),
                    encoding="utf-8",
                )

            def _wait_for_prompt_submission(self, **kwargs):
                raise TimeoutError("等待智能体确认收到 prompt 超时")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            review_json = root / "review.json"
            review_md = root / "review.md"

            def validator(path: Path) -> TurnFileResult:
                payload = json.loads(path.read_text(encoding="utf-8"))
                matched = payload[0]
                if matched["task_name"] != "M2-T3" or matched["review_pass"] is not True:
                    raise ValueError("review not complete")
                if review_md.read_text(encoding="utf-8").strip():
                    raise ValueError("review markdown should be empty when passed")
                return TurnFileResult(
                    status_path=str(path),
                    payload={"task_name": "M2-T3", "review_pass": True},
                    artifact_paths={"review_json": str(review_json), "review_md": str(review_md)},
                    artifact_hashes={
                        "review_json": "sha256:json",
                        "review_md": "sha256:md",
                    },
                    validated_at="2026-04-24T00:00:00",
                )

            worker = PromptTimeoutCompletionWorker(
                worker_id="prompt-timeout-completion-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="kimi-code/kimi-for-coding"),
                runtime_root=root / "runtime",
            )
            result = worker.run_turn(
                label="development_review_init_M2-T3_审核员_round_1",
                prompt="write review files only",
                completion_contract=TurnFileContract(
                    turn_id="development_review_M2-T3_审核员",
                    phase="任务开发",
                    status_path=review_json,
                    validator=validator,
                    quiet_window_sec=0.0,
                    kind="review_round",
                    tracked_artifacts={"review_json": review_json, "review_md": review_md},
                ),
                timeout_sec=1.0,
            )

            self.assertFalse(result.ok)
            self.assertEqual(len(worker.sent_prompts), 2)
            self.assertEqual(json.loads(Path(worker.current_task_status_path).read_text(encoding="utf-8")), {"status": "running"})
            self.assertIn("等待智能体确认收到 prompt 超时", result.clean_output)


    def test_validate_task_result_file_requires_required_artifacts_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            worker = TmuxBatchWorker(
                worker_id="validate-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            ask_human = root / "ask_human.md"
            ask_human.write_text("问题\n", encoding="utf-8")
            result_path = root / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "turn_id": "requirements_review_human_feedback",
                        "phase": "requirements_review_human_feedback",
                        "task_kind": "a03_human_feedback",
                        "status": "completed",
                        "summary": "ok",
                        "artifacts": {},
                        "artifact_hashes": {},
                        "written_at": "2026-04-16T00:00:00+00:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            contract = TaskResultContract(
                turn_id="requirements_review_human_feedback",
                phase="requirements_review_human_feedback",
                task_kind="a03_human_feedback",
                mode="a03_human_feedback",
                expected_statuses=("completed",),
                required_artifacts={"ask_human": ask_human},
            )
            with self.assertRaises(ValueError):
                worker._validate_task_result_file(contract=contract, result_path=result_path)

    def test_wait_for_task_result_requires_done_task_status_before_success(self):
        class WaitResultWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.track_calls = 0

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="› ready",
                    raw_log_delta="delta",
                    raw_log_tail="› ready",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-16T00:00:00",
                )

            def _maybe_send_task_completion_nudge(self, **kwargs):
                return False

            def _track_task_completion_signal(self, *, task_status_path, status_done_seen):
                self.track_calls += 1
                return self.track_calls >= 2

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            ask_human = root / "ask_human.md"
            ask_human.write_text("答复\n", encoding="utf-8")
            result_path = root / "result.json"
            worker = WaitResultWorker(
                worker_id="wait-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            result_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "turn_id": "requirements_review_human_feedback",
                        "phase": "requirements_review_human_feedback",
                        "task_kind": "a03_human_feedback",
                        "status": "completed",
                        "summary": "ok",
                        "artifacts": {
                            "ask_human": str(ask_human.resolve()),
                        },
                        "artifact_hashes": {
                            str(ask_human.resolve()): "sha256:" + __import__("hashlib").sha256(
                                ask_human.read_bytes()
                            ).hexdigest(),
                        },
                        "written_at": "2026-04-16T00:00:00+00:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            contract = TaskResultContract(
                turn_id="requirements_review_human_feedback",
                phase="requirements_review_human_feedback",
                task_kind="a03_human_feedback",
                mode="a03_human_feedback",
                expected_statuses=("completed",),
                required_artifacts={"ask_human": ask_human},
            )
            result = worker.wait_for_task_result(
                contract=contract,
                task_status_path=root / "task_status.json",
                result_path=result_path,
                timeout_sec=2.0,
            )
            self.assertEqual(result.payload["status"], "completed")
            self.assertGreaterEqual(worker.track_calls, 2)

    def test_wait_for_task_result_raises_contract_violation_when_runtime_stalls_without_result(self):
        class StalledResultWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="处理中",
                    raw_log_delta="",
                    raw_log_tail="处理中",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-25T00:00:00",
                    pane_title="TmuxCodingTeam",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            ask_human = root / "ask_human.md"
            result_path = root / "result.json"
            task_status_path = root / "task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            worker = StalledResultWorker(
                worker_id="stalled-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            worker.current_task_runtime_status = "running"
            worker._last_terminal_change_monotonic = time.monotonic() - 60.0
            worker.terminal_recently_changed = False
            contract = TaskResultContract(
                turn_id="requirements_review_human_feedback",
                phase="requirements_review_human_feedback",
                task_kind="a03_human_feedback",
                mode="a03_human_feedback",
                expected_statuses=("completed",),
                required_artifacts={"ask_human": ask_human},
            )

            with mock.patch("T02_tmux_agents.TASK_CONTRACT_STALL_IDLE_SEC", 0.0):
                with self.assertRaisesRegex(RuntimeError, TASK_RESULT_CONTRACT_ERROR_PREFIX):
                    worker.wait_for_task_result(
                        contract=contract,
                        task_status_path=task_status_path,
                        result_path=result_path,
                        timeout_sec=1.0,
                    )

    def test_wait_for_task_result_marks_task_done_when_result_stable_and_agent_ready(self):
        class ResultReadyWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.observe_count = 0

            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                self.observe_count += 1
                self.agent_state = AgentRuntimeState.READY
                return WorkerObservation(
                    visible_text="› ready",
                    raw_log_delta="",
                    raw_log_tail="› ready",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-16T00:00:00",
                    pane_title="TmuxCodingTeam",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            ask_human = root / "ask_human.md"
            ask_human.write_text("答复\n", encoding="utf-8")
            result_path = root / "result.json"
            task_status_path = root / "task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            result_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "turn_id": "requirements_review_human_feedback",
                        "phase": "requirements_review_human_feedback",
                        "task_kind": "a03_human_feedback",
                        "status": "completed",
                        "summary": "ok",
                        "artifacts": {
                            "ask_human": str(ask_human.resolve()),
                        },
                        "artifact_hashes": {
                            str(ask_human.resolve()): "sha256:" + __import__("hashlib").sha256(
                                ask_human.read_bytes()
                            ).hexdigest(),
                        },
                        "written_at": "2026-04-16T00:00:00+00:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            worker = ResultReadyWorker(
                worker_id="result-ready-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            contract = TaskResultContract(
                turn_id="requirements_review_human_feedback",
                phase="requirements_review_human_feedback",
                task_kind="a03_human_feedback",
                mode="a03_human_feedback",
                expected_statuses=("completed",),
                required_artifacts={"ask_human": ask_human},
            )

            result = worker.wait_for_task_result(
                contract=contract,
                task_status_path=task_status_path,
                result_path=result_path,
                timeout_sec=1.0,
            )

            self.assertEqual(result.payload["status"], "completed")
            self.assertEqual(json.loads(task_status_path.read_text(encoding="utf-8")), {"status": "done"})
            self.assertEqual(worker.current_task_runtime_status, "done")

    def test_wait_for_task_result_finalizes_from_contract_when_ready_without_helper(self):
        class ContractReadyWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def capture_visible(self, tail_lines=200):
                return "› Continue working in @filename"

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="› Continue working in @filename",
                    raw_log_delta="",
                    raw_log_tail="",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-16T00:00:03",
                    pane_title="TmuxCodingTeam",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            requirements_clear = root / "需求澄清.md"
            requirements_clear.write_text("已澄清\n", encoding="utf-8")
            task_status_path = root / "task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            result_path = root / "result.json"
            worker = ContractReadyWorker(
                worker_id="contract-ready-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            contract = TaskResultContract(
                turn_id="requirements_review_ba_resume",
                phase="requirements_review_ba_resume",
                task_kind="a03_ba_resume",
                mode="a03_ba_resume",
                expected_statuses=("ready",),
                optional_artifacts={"requirements_clear": requirements_clear},
            )

            result = worker.wait_for_task_result(
                contract=contract,
                task_status_path=task_status_path,
                result_path=result_path,
                timeout_sec=1.0,
            )

            self.assertEqual(result.payload["status"], "ready")
            self.assertEqual(json.loads(task_status_path.read_text(encoding="utf-8")), {"status": "done"})
            self.assertTrue(result_path.exists())

    def test_wait_for_task_result_finalizes_a07_ready_hitl_as_ready_when_ask_human_empty(self):
        class ContractReadyWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="› Continue working in @filename",
                    raw_log_delta="",
                    raw_log_tail="› Continue working in @filename",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-16T00:00:03",
                    pane_title="TmuxCodingTeam",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            ask_human = root / "与人类交流.md"
            ask_human.write_text("", encoding="utf-8")
            task_status_path = root / "task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            result_path = root / "result.json"
            worker = ContractReadyWorker(
                worker_id="a07-ready-hitl-ready-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            contract = TaskResultContract(
                turn_id="a07_developer_human_reply",
                phase="a07_developer_human_reply",
                task_kind="a07_developer_human_reply",
                mode="a07_developer_human_reply",
                expected_statuses=("ready", "hitl"),
                optional_artifacts={"ask_human": ask_human},
                outcome_artifacts={
                    "ready": {"forbids": ("ask_human",)},
                    "hitl": {"requires": ("ask_human",)},
                },
            )

            result = worker.wait_for_task_result(
                contract=contract,
                task_status_path=task_status_path,
                result_path=result_path,
                timeout_sec=1.0,
            )

            self.assertEqual(result.payload["status"], "ready")
            self.assertEqual(json.loads(task_status_path.read_text(encoding="utf-8")), {"status": "done"})
            self.assertTrue(result_path.exists())

    def test_wait_for_task_result_does_not_finalize_a07_hitl_while_busy(self):
        class BusyContractWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="working",
                    raw_log_delta="",
                    raw_log_tail="working",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-16T00:00:03",
                    pane_title="⠋ TmuxCodingTeam",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_status_path = root / "task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            time.sleep(0.02)
            ask_human = root / "与人类交流.md"
            ask_human.write_text("请补充结算金额边界\n", encoding="utf-8")
            result_path = root / "result.json"
            worker = BusyContractWorker(
                worker_id="a07-ready-hitl-hitl-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            contract = TaskResultContract(
                turn_id="a07_developer_init",
                phase="a07_developer_init",
                task_kind="a07_developer_init",
                mode="a07_developer_init",
                expected_statuses=("ready", "hitl"),
                optional_artifacts={"ask_human": ask_human},
                outcome_artifacts={
                    "ready": {"forbids": ("ask_human",)},
                    "hitl": {"requires": ("ask_human",)},
                },
            )

            with self.assertRaisesRegex(TimeoutError, "等待任务结果超时"):
                worker.wait_for_task_result(
                    contract=contract,
                    task_status_path=task_status_path,
                    result_path=result_path,
                    timeout_sec=0.6,
                )

            self.assertEqual(json.loads(task_status_path.read_text(encoding="utf-8")), {"status": "running"})
            self.assertFalse(result_path.exists())

    def test_wait_for_task_result_finalizes_a07_ready_hitl_as_hitl_when_ready(self):
        class ContractReadyWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="› Continue working in @filename",
                    raw_log_delta="",
                    raw_log_tail="› Continue working in @filename",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-16T00:00:03",
                    pane_title="TmuxCodingTeam",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_status_path = root / "task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            time.sleep(0.02)
            ask_human = root / "与人类交流.md"
            ask_human.write_text("请补充结算金额边界\n", encoding="utf-8")
            result_path = root / "result.json"
            worker = ContractReadyWorker(
                worker_id="a07-ready-hitl-hitl-ready-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            contract = TaskResultContract(
                turn_id="a07_developer_init",
                phase="a07_developer_init",
                task_kind="a07_developer_init",
                mode="a07_developer_init",
                expected_statuses=("ready", "hitl"),
                optional_artifacts={"ask_human": ask_human},
                outcome_artifacts={
                    "ready": {"forbids": ("ask_human",)},
                    "hitl": {"requires": ("ask_human",)},
                },
            )

            result = worker.wait_for_task_result(
                contract=contract,
                task_status_path=task_status_path,
                result_path=result_path,
                timeout_sec=1.0,
            )

            self.assertEqual(result.payload["status"], "hitl")
            self.assertEqual(result.payload["artifacts"]["ask_human"], str(ask_human.resolve()))
            self.assertEqual(json.loads(task_status_path.read_text(encoding="utf-8")), {"status": "done"})
            self.assertTrue(result_path.exists())

    def test_wait_for_task_result_does_not_finalize_completed_from_stale_artifact_without_helper(self):
        class ContractReadyWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def capture_visible(self, tail_lines=200):
                return "› Continue working in @filename"

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="› Continue working in @filename",
                    raw_log_delta="",
                    raw_log_tail="",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-16T00:00:03",
                    pane_title="TmuxCodingTeam",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            developer_output = root / "开发记录.md"
            developer_output.write_text("旧开发记录\n", encoding="utf-8")
            time.sleep(0.02)
            task_status_path = root / "task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            result_path = root / "result.json"
            worker = ContractReadyWorker(
                worker_id="contract-ready-completed-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            contract = TaskResultContract(
                turn_id="a07_developer_task_complete",
                phase="a07_developer_task_complete",
                task_kind="a07_developer_task_complete",
                mode="a07_developer_task_complete",
                expected_statuses=("completed",),
                required_artifacts={"developer_output": developer_output},
            )

            with self.assertRaisesRegex(TimeoutError, "等待任务结果超时"):
                worker.wait_for_task_result(
                    contract=contract,
                    task_status_path=task_status_path,
                    result_path=result_path,
                    timeout_sec=0.6,
                )

            self.assertFalse(result_path.exists())
            self.assertEqual(json.loads(task_status_path.read_text(encoding="utf-8")), {"status": "running"})

    def test_wait_for_task_result_finalizes_completed_from_fresh_file_contract_when_ready(self):
        class ContractReadyWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def capture_visible(self, tail_lines=200):
                return "› Continue working in @filename"

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="› Continue working in @filename",
                    raw_log_delta="",
                    raw_log_tail="",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-16T00:00:03",
                    pane_title="TmuxCodingTeam",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_status_path = root / "task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            time.sleep(0.02)
            developer_output = root / "开发记录.md"
            developer_output.write_text("本轮开发记录\n", encoding="utf-8")
            result_path = root / "result.json"
            worker = ContractReadyWorker(
                worker_id="contract-ready-fresh-completed-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            contract = TaskResultContract(
                turn_id="a07_developer_task_complete",
                phase="a07_developer_task_complete",
                task_kind="a07_developer_task_complete",
                mode="a07_developer_task_complete",
                expected_statuses=("completed",),
                required_artifacts={"developer_output": developer_output},
            )

            result = worker.wait_for_task_result(
                contract=contract,
                task_status_path=task_status_path,
                result_path=result_path,
                timeout_sec=1.0,
            )

            self.assertEqual(result.payload["status"], "completed")
            self.assertEqual(json.loads(task_status_path.read_text(encoding="utf-8")), {"status": "done"})
            self.assertTrue(result_path.exists())

    def test_wait_for_task_result_does_not_finalize_fresh_completed_artifact_while_busy(self):
        class BusyContractWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="working",
                    raw_log_delta="",
                    raw_log_tail="working",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-16T00:00:03",
                    pane_title="⠋ TmuxCodingTeam",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_status_path = root / "task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            time.sleep(0.02)
            developer_output = root / "开发记录.md"
            developer_output.write_text("本轮开发记录\n", encoding="utf-8")
            result_path = root / "result.json"
            worker = BusyContractWorker(
                worker_id="contract-busy-fresh-completed-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            contract = TaskResultContract(
                turn_id="a07_developer_task_complete",
                phase="a07_developer_task_complete",
                task_kind="a07_developer_task_complete",
                mode="a07_developer_task_complete",
                expected_statuses=("completed",),
                required_artifacts={"developer_output": developer_output},
            )

            with self.assertRaisesRegex(TimeoutError, "等待任务结果超时"):
                worker.wait_for_task_result(
                    contract=contract,
                    task_status_path=task_status_path,
                    result_path=result_path,
                    timeout_sec=0.6,
                )

            self.assertEqual(json.loads(task_status_path.read_text(encoding="utf-8")), {"status": "running"})
            self.assertFalse(result_path.exists())

    def test_wait_for_task_result_fails_contract_fast_when_ready_without_artifact(self):
        class ReadyMissingArtifactWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="› ready",
                    raw_log_delta="",
                    raw_log_tail="› ready",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-16T00:00:03",
                    pane_title="TmuxCodingTeam",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_status_path = root / "task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            developer_output = root / "开发记录.md"
            result_path = root / "result.json"
            worker = ReadyMissingArtifactWorker(
                worker_id="ready-missing-contract-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            contract = TaskResultContract(
                turn_id="a07_developer_task_complete",
                phase="a07_developer_task_complete",
                task_kind="a07_developer_task_complete",
                mode="a07_developer_task_complete",
                expected_statuses=("completed",),
                required_artifacts={"developer_output": developer_output},
            )

            with mock.patch("tmux_core.runtime.tmux_runtime.TASK_RESULT_READY_MISSING_GRACE_SEC", 0.0):
                with self.assertRaisesRegex(RuntimeError, "task result contract violation"):
                    worker.wait_for_task_result(
                        contract=contract,
                        task_status_path=task_status_path,
                        result_path=result_path,
                        timeout_sec=1.0,
                    )

    def test_wait_for_task_result_does_not_auto_ready_force_hitl_without_ask_human(self):
        class ReadyMissingHitlWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="› ready",
                    raw_log_delta="",
                    raw_log_tail="› ready",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-16T00:00:03",
                    pane_title="TmuxCodingTeam",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_status_path = root / "task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            ask_human = root / "与人类交流.md"
            ask_human.write_text("", encoding="utf-8")
            result_path = root / "result.json"
            worker = ReadyMissingHitlWorker(
                worker_id="force-hitl-missing-ask-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            contract = TaskResultContract(
                turn_id="a07_developer_review_limit_force_hitl",
                phase="a07_developer_review_limit_force_hitl",
                task_kind="a07_developer_review_limit_force_hitl",
                mode="a07_developer_review_limit_force_hitl",
                expected_statuses=("hitl",),
                optional_artifacts={"ask_human": ask_human},
                outcome_artifacts={"hitl": {"requires": ("ask_human",)}},
            )

            with mock.patch("tmux_core.runtime.tmux_runtime.TASK_RESULT_READY_MISSING_GRACE_SEC", 0.0):
                with self.assertRaisesRegex(RuntimeError, "task result contract violation"):
                    worker.wait_for_task_result(
                        contract=contract,
                        task_status_path=task_status_path,
                        result_path=result_path,
                        timeout_sec=1.0,
                    )

            self.assertFalse(result_path.exists())
            self.assertEqual(json.loads(task_status_path.read_text(encoding="utf-8")), {"status": "running"})

    def test_wait_for_task_result_ready_contract_checks_liveness_before_materializing(self):
        class BusyWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="",
                    raw_log_delta="",
                    raw_log_tail="",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-24T00:00:00",
                    pane_title="⠋ TmuxCodingTeam",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            requirements_clear = root / "需求澄清.md"
            requirements_clear.write_text("已澄清\n", encoding="utf-8")
            task_status_path = root / "task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            result_path = root / "result.json"
            worker = BusyWorker(
                worker_id="ready-busy-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            with self.assertRaises(TimeoutError):
                worker.wait_for_task_result(
                    contract=TaskResultContract(
                        turn_id="requirements_review_ba_resume",
                        phase="requirements_review_ba_resume",
                        task_kind="a03_ba_resume",
                        mode="a03_ba_resume",
                        expected_statuses=("ready",),
                        optional_artifacts={"requirements_clear": requirements_clear},
                    ),
                    task_status_path=task_status_path,
                    result_path=result_path,
                    timeout_sec=1.0,
                )

    def test_wait_for_task_result_fails_fast_when_probe_reports_shell_exit(self):
        class ShellExitWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="shell",
                    raw_log_delta="",
                    raw_log_tail="shell",
                    current_command="zsh",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-24T00:00:00",
                    pane_title="",
                )

            def capture_visible(self, tail_lines=500):
                return "shell"

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            developer_output = root / "开发记录.md"
            developer_output.write_text("旧开发记录\n", encoding="utf-8")
            task_status_path = root / "task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            result_path = root / "result.json"
            worker = ShellExitWorker(
                worker_id="shell-exit-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            with self.assertRaisesRegex(RuntimeError, "agent exited back to shell"):
                worker.wait_for_task_result(
                    contract=TaskResultContract(
                        turn_id="a07_developer_task_complete",
                        phase="a07_developer_task_complete",
                        task_kind="a07_developer_task_complete",
                        mode="a07_developer_task_complete",
                        expected_statuses=("completed",),
                        required_artifacts={"developer_output": developer_output},
                    ),
                    task_status_path=task_status_path,
                    result_path=result_path,
                    timeout_sec=1.0,
                )

    def test_wait_for_task_result_checks_liveness_after_valid_result_before_done(self):
        class ShellExitWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="shell",
                    raw_log_delta="",
                    raw_log_tail="shell",
                    current_command="zsh",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-24T00:00:00",
                    pane_title="",
                )

            def capture_visible(self, tail_lines=500):
                return "shell"

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            developer_output = root / "开发记录.md"
            developer_output.write_text("开发完成\n", encoding="utf-8")
            task_status_path = root / "task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            result_path = root / "result.json"
            contract = TaskResultContract(
                turn_id="a07_developer_task_complete",
                phase="a07_developer_task_complete",
                task_kind="a07_developer_task_complete",
                mode="a07_developer_task_complete",
                expected_statuses=("completed",),
                required_artifacts={"developer_output": developer_output},
            )
            finalize_task_result(contract=contract, result_path=result_path, task_status_path=None)
            worker = ShellExitWorker(
                worker_id="valid-result-shell-exit-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            with self.assertRaisesRegex(RuntimeError, "agent exited back to shell"):
                worker.wait_for_task_result(
                    contract=contract,
                    task_status_path=task_status_path,
                    result_path=result_path,
                    timeout_sec=1.0,
                )

    def test_wait_for_task_result_materializes_ready_result_from_ready_state_without_helper(self):
        class ReadyReplyWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="\n".join(
                        [
                            "• 准备完毕",
                            "",
                            "",
                            "› Use /skills to list available skills",
                            "",
                            "  gpt-5.4-mini low · ~/Desktop/my_test",
                        ]
                    ),
                    raw_log_delta="",
                    raw_log_tail="• 准备完毕",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-16T00:00:03",
                    pane_title="TmuxCodingTeam",
                )

            def _maybe_send_task_completion_nudge(self, **kwargs):
                return False

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            requirements_clear = root / "需求澄清.md"
            requirements_clear.write_text("已澄清\n", encoding="utf-8")
            task_status_path = root / "task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            result_path = root / "result.json"
            worker = ReadyReplyWorker(
                worker_id="ready-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            contract = TaskResultContract(
                turn_id="requirements_review_ba_resume",
                phase="requirements_review_ba_resume",
                task_kind="a03_ba_resume",
                mode="a03_ba_resume",
                expected_statuses=("ready",),
                optional_artifacts={"requirements_clear": requirements_clear},
            )
            result = worker.wait_for_task_result(
                contract=contract,
                task_status_path=task_status_path,
                result_path=result_path,
                timeout_sec=1.0,
            )
            self.assertEqual(result.payload["status"], "ready")
            self.assertEqual(result.payload["summary"], "需求分析师已进入需求评审准备态")
            self.assertEqual(json.loads(task_status_path.read_text(encoding="utf-8")), {"status": "done"})
            self.assertEqual(
                result.payload["artifacts"]["requirements_clear"],
                str(requirements_clear.resolve()),
            )

    def test_run_turn_ready_only_finalizes_claude_ready_after_surface_change_without_prompt_echo(self):
        class ReadyOnlyClaudeWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def ensure_agent_ready(self, timeout_sec=60.0):
                self.agent_started = True
                self.agent_ready = True
                self.wrapper_state = WrapperState.READY
                self.current_command = "claude"
                self.current_path = str(self.work_dir)
                self.last_pane_title = "✳ Ready"

            def _send_text(self, text, enter_count=None):
                self.sent_text = text

            def _wait_for_prompt_submission(self, *, prompt, timeout_sec):
                raise AssertionError("READY-only turn must not wait for prompt echo")

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                prompt_sent = bool(getattr(self, "sent_text", ""))
                surface = (
                    "\n".join(
                        [
                            "⏺ 完成",
                            "",
                            "────────────────────────────────────────────────────────────────────────────────",
                            "❯",
                            "⏵⏵ bypass permissions on",
                        ]
                    )
                    if prompt_sent
                    else "❯"
                )
                return WorkerObservation(
                    visible_text=surface,
                    raw_log_delta="",
                    raw_log_tail=surface,
                    current_command="claude",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-24T00:00:03",
                    pane_title="✳ Understand service independence refactoring architecture",
                )

            def capture_visible(self, tail_lines=500):
                return "⏺ 完成\n❯"

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            requirements_clear = root / "需求澄清.md"
            requirements_clear.write_text("已澄清\n", encoding="utf-8")
            worker = ReadyOnlyClaudeWorker(
                worker_id="ready-only-claude-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="claude", model="sonnet"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            contract = TaskResultContract(
                turn_id="a08_reviewer_init",
                phase="a08_reviewer_init",
                task_kind="a08_reviewer_init",
                mode="a08_reviewer_init",
                expected_statuses=("ready",),
                optional_artifacts={"requirements_clear": requirements_clear},
            )

            result = worker.run_turn(
                label="overall_review_reviewer_init_architect",
                prompt="只返回 完成",
                result_contract=contract,
                timeout_sec=1.0,
            )

            self.assertTrue(result.ok)
            result_path = Path(worker.current_task_result_path)
            self.assertTrue(result_path.exists())
            self.assertEqual(json.loads(result_path.read_text(encoding="utf-8"))["status"], "ready")
            self.assertEqual(json.loads(Path(worker.current_task_status_path).read_text(encoding="utf-8")), {"status": "done"})

    def test_run_turn_ready_only_finalizes_after_busy_to_ready_without_prompt_echo(self):
        class BusyThenReadyWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.after_submit_observes = 0

            def target_exists(self, target=None):
                return True

            def ensure_agent_ready(self, timeout_sec=60.0):
                self.agent_started = True
                self.agent_ready = True
                self.wrapper_state = WrapperState.READY
                self.current_command = "codex"
                self.current_path = str(self.work_dir)
                self.last_pane_title = "TmuxCodingTeam"

            def _send_text(self, text, enter_count=None):
                self.sent_text = text

            def _wait_for_prompt_submission(self, *, prompt, timeout_sec):
                raise AssertionError("READY-only turn must not wait for prompt echo")

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                prompt_sent = bool(getattr(self, "sent_text", ""))
                if not prompt_sent:
                    pane_title = "TmuxCodingTeam"
                    surface = "› Continue"
                elif self.after_submit_observes == 0:
                    self.after_submit_observes += 1
                    pane_title = "⠋ TmuxCodingTeam"
                    surface = "working"
                else:
                    pane_title = "TmuxCodingTeam"
                    surface = "› Continue"
                return WorkerObservation(
                    visible_text=surface,
                    raw_log_delta="",
                    raw_log_tail=surface,
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-24T00:00:03",
                    pane_title=pane_title,
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            requirements_clear = root / "需求澄清.md"
            requirements_clear.write_text("已澄清\n", encoding="utf-8")
            worker = BusyThenReadyWorker(
                worker_id="busy-then-ready-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            contract = TaskResultContract(
                turn_id="a08_reviewer_init",
                phase="a08_reviewer_init",
                task_kind="a08_reviewer_init",
                mode="a08_reviewer_init",
                expected_statuses=("ready",),
                optional_artifacts={"requirements_clear": requirements_clear},
            )

            result = worker.run_turn(
                label="overall_review_reviewer_init_tester",
                prompt="初始化",
                result_contract=contract,
                timeout_sec=1.0,
            )

            self.assertTrue(result.ok)
            self.assertEqual(json.loads(Path(worker.current_task_status_path).read_text(encoding="utf-8")), {"status": "done"})

    def test_run_turn_ready_only_does_not_finalize_unchanged_ready_surface(self):
        class UnchangedReadyWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def ensure_agent_ready(self, timeout_sec=60.0):
                self.agent_started = True
                self.agent_ready = True
                self.wrapper_state = WrapperState.READY
                self.current_command = "codex"
                self.current_path = str(self.work_dir)
                self.last_pane_title = "TmuxCodingTeam"

            def _send_text(self, text, enter_count=None):
                self.sent_text = text

            def _wait_for_prompt_submission(self, *, prompt, timeout_sec):
                raise AssertionError("READY-only turn must not wait for prompt echo")

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="› Continue",
                    raw_log_delta="",
                    raw_log_tail="› Continue",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-24T00:00:03",
                    pane_title="TmuxCodingTeam",
                )

            def capture_visible(self, tail_lines=500):
                return "› Continue"

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            requirements_clear = root / "需求澄清.md"
            requirements_clear.write_text("已澄清\n", encoding="utf-8")
            worker = UnchangedReadyWorker(
                worker_id="unchanged-ready-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            contract = TaskResultContract(
                turn_id="a08_reviewer_init",
                phase="a08_reviewer_init",
                task_kind="a08_reviewer_init",
                mode="a08_reviewer_init",
                expected_statuses=("ready",),
                optional_artifacts={"requirements_clear": requirements_clear},
            )

            result = worker.run_turn(
                label="overall_review_reviewer_init_architect",
                prompt="初始化",
                result_contract=contract,
                timeout_sec=0.05,
            )

            self.assertFalse(result.ok)
            self.assertFalse(Path(worker.current_task_result_path).exists())
            self.assertEqual(json.loads(Path(worker.current_task_status_path).read_text(encoding="utf-8")), {"status": "running"})
            self.assertIn("等待 READY-only 任务结果超时", result.clean_output)

    def test_wait_for_task_result_materializes_ready_result_even_when_terminal_recently_changed(self):
        class NoisyReadyReplyWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                self.agent_state = AgentRuntimeState.READY
                self.terminal_recently_changed = True
                return WorkerObservation(
                    visible_text="\n".join(
                        [
                            "• 完成",
                            "",
                            "",
                            "› Run /review on my current changes",
                            "",
                            "  gpt-5.4-mini low · ~/Desktop/my_test",
                        ]
                    ),
                    raw_log_delta="",
                    raw_log_tail="• 完成",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-19T00:00:03",
                    pane_title="my-project",
                )

            def _maybe_send_task_completion_nudge(self, **kwargs):
                return False

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            requirements_clear = root / "需求澄清.md"
            requirements_clear.write_text("已澄清\n", encoding="utf-8")
            task_status_path = root / "task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            result_path = root / "result.json"
            worker = NoisyReadyReplyWorker(
                worker_id="noisy-ready-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            contract = TaskResultContract(
                turn_id="a05_ba_init",
                phase="a05_ba_init",
                task_kind="a05_ba_init",
                mode="a05_ba_init",
                expected_statuses=("ready",),
                optional_artifacts={"requirements_clear": requirements_clear},
            )
            result = worker.wait_for_task_result(
                contract=contract,
                task_status_path=task_status_path,
                result_path=result_path,
                timeout_sec=1.0,
            )
            self.assertEqual(result.payload["status"], "ready")
            self.assertEqual(result.payload["summary"], "需求分析师已完成详细设计初始化")
            self.assertEqual(json.loads(task_status_path.read_text(encoding="utf-8")), {"status": "done"})
            self.assertEqual(
                result.payload["artifacts"]["requirements_clear"],
                str(requirements_clear.resolve()),
            )

    def test_get_agent_state_keeps_detector_ready_when_terminal_recently_changed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="ready-surface-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            worker.terminal_recently_changed = True
            observation = WorkerObservation(
                visible_text="› Use /skills to list available skills",
                raw_log_delta="",
                raw_log_tail="› Use /skills to list available skills",
                current_command="codex",
                current_path=str(Path(tmp_dir).resolve()),
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-25T23:53:03",
                pane_title="tmux-api-v3",
            )

            self.assertEqual(worker.get_agent_state(observation), AgentRuntimeState.READY)

    def test_wait_for_task_result_ignores_stale_terminal_text_from_previous_turn(self):
        class StaleDoneReplyWorker(TmuxBatchWorker):
            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="\n".join(
                        [
                            "• 完成",
                            "",
                            "",
                            "› Explain this codebase",
                            "",
                            "  gpt-5.4-mini low · ~/Desktop/my_test",
                        ]
                    ),
                    raw_log_delta="",
                    raw_log_tail="",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-20T00:00:03",
                    pane_title="TmuxCodingTeam",
                )

            def _maybe_send_task_completion_nudge(self, **kwargs):
                return False

            def capture_visible(self, tail_lines=500):
                return "• 完成"

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            detailed_design = root / "详细设计.md"
            task_status_path = root / "task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            result_path = root / "result.json"
            worker = StaleDoneReplyWorker(
                worker_id="stale-done-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            contract = TaskResultContract(
                turn_id="a05_detailed_design_generate",
                phase="a05_detailed_design_generate",
                task_kind="a05_detailed_design_generate",
                mode="a05_detailed_design_generate",
                expected_statuses=("completed",),
                required_artifacts={"detailed_design": detailed_design},
            )
            with self.assertRaises(TimeoutError):
                worker.wait_for_task_result(
                    contract=contract,
                    task_status_path=task_status_path,
                    result_path=result_path,
                    timeout_sec=0.1,
                    baseline_visible="\n".join(
                        [
                            "• 完成",
                            "",
                            "",
                            "› Explain this codebase",
                            "",
                            "  gpt-5.4-mini low · ~/Desktop/my_test",
                        ]
                    ),
                )

    def test_wait_for_task_result_ignores_terminal_text_seen_only_in_baseline_raw_log_tail(self):
        class StaleTailDoneReplyWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                surface = "\n".join(
                    [
                        "• 完成",
                        "",
                        "",
                        "› Explain this codebase",
                        "",
                        "  gpt-5.4-mini low · ~/Desktop/my_test",
                    ]
                )
                return WorkerObservation(
                    visible_text=surface,
                    raw_log_delta="",
                    raw_log_tail=surface,
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-20T00:00:03",
                    pane_title="TmuxCodingTeam",
                )

            def _maybe_send_task_completion_nudge(self, **kwargs):
                return False

            def capture_visible(self, tail_lines=500):
                return "• 完成"

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            detailed_design = root / "详细设计.md"
            task_status_path = root / "task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            result_path = root / "result.json"
            worker = StaleTailDoneReplyWorker(
                worker_id="stale-tail-done-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            contract = TaskResultContract(
                turn_id="a05_detailed_design_generate",
                phase="a05_detailed_design_generate",
                task_kind="a05_detailed_design_generate",
                mode="a05_detailed_design_generate",
                expected_statuses=("completed",),
                required_artifacts={"detailed_design": detailed_design},
            )
            with self.assertRaises(TimeoutError):
                worker.wait_for_task_result(
                    contract=contract,
                    task_status_path=task_status_path,
                    result_path=result_path,
                    timeout_sec=0.1,
                    baseline_visible="› Explain this codebase\n  gpt-5.4-mini low · ~/Desktop/my_test",
                    baseline_raw_log_tail="\n".join(
                        [
                            "• 完成",
                            "",
                            "",
                            "› Explain this codebase",
                            "",
                            "  gpt-5.4-mini low · ~/Desktop/my_test",
                        ]
                    ),
                )

    def test_wait_for_task_result_ignores_fresh_terminal_text_without_result_file(self):
        class FreshDoneReplyWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="\n".join(
                        [
                            "• 完成",
                            "",
                            "",
                            "› Explain this codebase",
                            "",
                            "  gpt-5.4-mini low · ~/Desktop/my_test",
                        ]
                    ),
                    raw_log_delta="• 完成",
                    raw_log_tail="• 完成",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-20T00:00:03",
                    pane_title="TmuxCodingTeam",
                )

            def _maybe_send_task_completion_nudge(self, **kwargs):
                return False

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            detailed_design = root / "详细设计.md"
            detailed_design.write_text("设计正文\n", encoding="utf-8")
            task_status_path = root / "task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            result_path = root / "result.json"
            worker = FreshDoneReplyWorker(
                worker_id="fresh-done-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            contract = TaskResultContract(
                turn_id="a05_detailed_design_generate",
                phase="a05_detailed_design_generate",
                task_kind="a05_detailed_design_generate",
                mode="a05_detailed_design_generate",
                expected_statuses=("completed",),
                required_artifacts={"detailed_design": detailed_design},
            )
            with self.assertRaises(TimeoutError):
                worker.wait_for_task_result(
                    contract=contract,
                    task_status_path=task_status_path,
                    result_path=result_path,
                    timeout_sec=1.0,
                    baseline_visible="\n".join(
                        [
                            "• 完成",
                            "",
                            "",
                            "› Explain this codebase",
                            "",
                            "  gpt-5.4-mini low · ~/Desktop/my_test",
                        ]
                    ),
                )
            self.assertFalse(result_path.exists())
            self.assertEqual(json.loads(task_status_path.read_text(encoding="utf-8")), {"status": "running"})

    def test_wait_for_task_result_finalizes_opencode_ready_with_extreme_narrow_footer(self):
        class ExtremeWrappedFooterReadyWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                surface = "\n".join(
                    [
                        "完成",
                        "",
                        "74.8K (29%) · $0.08  ctrl+p",
                        "command",
                        "s",
                    ]
                )
                return WorkerObservation(
                    visible_text=surface,
                    raw_log_delta="",
                    raw_log_tail=surface,
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-23T00:00:03",
                    pane_title="OC | demo",
                )

            def _maybe_send_task_completion_nudge(self, **kwargs):
                return False

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            requirements_clear = root / "需求澄清.md"
            requirements_clear.write_text("已澄清\n", encoding="utf-8")
            task_status_path = root / "task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            result_path = root / "result.json"
            worker = ExtremeWrappedFooterReadyWorker(
                worker_id="opencode-wrapped-ready-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="default"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            contract = TaskResultContract(
                turn_id="a06_ba_init",
                phase="a06_ba_init",
                task_kind="a06_ba_init",
                mode="a06_ba_init",
                expected_statuses=("ready",),
                optional_artifacts={"requirements_clear": requirements_clear},
            )
            result = worker.wait_for_task_result(
                contract=contract,
                task_status_path=task_status_path,
                result_path=result_path,
                timeout_sec=1.0,
            )
            self.assertEqual(result.payload["status"], "ready")
            self.assertEqual(result.payload["summary"], "任务拆分阶段智能体已完成初始化")
            self.assertEqual(json.loads(task_status_path.read_text(encoding="utf-8")), {"status": "done"})

    def test_wait_for_task_result_finalizes_claude_ready_with_footer_noise(self):
        class ClaudeFooterReadyWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                surface = "\n".join(
                    [
                        "⏺ 完成",
                        "",
                        "────────────────────────────────────────────────────────────────────────────────",
                        "❯",
                        "⏵⏵ bypass permissions on     ✗ Auto-update failed · Try claude doctor or npm …",
                        "(·oo·) Crumpet",
                    ]
                )
                return WorkerObservation(
                    visible_text=surface,
                    raw_log_delta="⏺ 完成",
                    raw_log_tail=surface,
                    current_command="claude",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-23T00:00:03",
                    pane_title="✳ Understand CashCost feature requirements",
                )

            def _maybe_send_task_completion_nudge(self, **kwargs):
                return False

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            requirements_clear = root / "需求澄清.md"
            requirements_clear.write_text("已澄清\n", encoding="utf-8")
            task_status_path = root / "task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            result_path = root / "result.json"
            worker = ClaudeFooterReadyWorker(
                worker_id="claude-footer-ready-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="claude", model="sonnet"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            contract = TaskResultContract(
                turn_id="a08_reviewer_init",
                phase="a08_reviewer_init",
                task_kind="a08_reviewer_init",
                mode="a08_reviewer_init",
                expected_statuses=("ready",),
                optional_artifacts={"requirements_clear": requirements_clear},
            )
            result = worker.wait_for_task_result(
                contract=contract,
                task_status_path=task_status_path,
                result_path=result_path,
                timeout_sec=1.0,
            )
            self.assertEqual(result.payload["status"], "ready")
            self.assertEqual(result.payload["summary"], "智能体已完成复核阶段初始化")
            self.assertEqual(json.loads(task_status_path.read_text(encoding="utf-8")), {"status": "done"})

    def test_wait_for_task_result_opencode_uses_raw_log_when_message_extract_fails(self):
        class ExtractionFailingOpenCodeReadyWorker(TmuxBatchWorker):
            def target_exists(self, target=None):
                return True

            def _extract_last_message(self, visible_text: str) -> str:
                raise ValueError("No assistant response found in terminal output")

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                surface = "\n".join(
                    [
                        "完成",
                        "",
                        "74.8K (29%) · $0.08  ctrl+p",
                        "command",
                        "s",
                    ]
                )
                return WorkerObservation(
                    visible_text=surface,
                    raw_log_delta="完成",
                    raw_log_tail=surface,
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-23T00:00:03",
                    pane_title="OC | demo",
                )

            def _maybe_send_task_completion_nudge(self, **kwargs):
                return False

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            requirements_clear = root / "需求澄清.md"
            requirements_clear.write_text("已澄清\n", encoding="utf-8")
            task_status_path = root / "task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            result_path = root / "result.json"
            worker = ExtractionFailingOpenCodeReadyWorker(
                worker_id="opencode-fallback-ready-result-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="default"),
                runtime_root=root / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            contract = TaskResultContract(
                turn_id="a08_reviewer_init",
                phase="a08_reviewer_init",
                task_kind="a08_reviewer_init",
                mode="a08_reviewer_init",
                expected_statuses=("ready",),
                optional_artifacts={"requirements_clear": requirements_clear},
            )
            result = worker.wait_for_task_result(
                contract=contract,
                task_status_path=task_status_path,
                result_path=result_path,
                timeout_sec=1.0,
            )
            self.assertEqual(result.payload["status"], "ready")
            self.assertEqual(result.payload["summary"], "智能体已完成复核阶段初始化")
            self.assertEqual(json.loads(task_status_path.read_text(encoding="utf-8")), {"status": "done"})

    def test_wait_for_prompt_submission_rejects_opencode_footer_delta_without_prompt_or_processing(self):
        class ExtremeWrappedFooterPromptWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.observe_calls = 0

            def observe(self, *, tail_lines=320, tail_bytes=24000):
                self.observe_calls += 1
                surface = "\n".join(
                    [
                        "完成",
                        "",
                        "74.8K (29%) · $0.08  ctrl+p",
                        "command",
                        "s",
                    ]
                )
                observation = WorkerObservation(
                    visible_text=surface,
                    raw_log_delta="完成",
                    raw_log_tail=surface,
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-23T00:00:03",
                    pane_title="OC | demo",
                )
                self.agent_state = self.detector.classify_agent_state(observation)
                return observation

        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch("tmux_core.runtime.tmux_runtime.time.sleep", return_value=None):
            worker = ExtremeWrappedFooterPromptWorker(
                worker_id="opencode-wrapped-ready-prompt-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="default"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            with self.assertRaises(TimeoutError):
                worker._wait_for_prompt_submission(prompt="analyze", timeout_sec=0.01)

        self.assertGreaterEqual(worker.observe_calls, 1)

    def test_source_mentions_prompt_accepts_multiple_wrapped_fragments(self):
        prompt = "\n".join(
            [
                "## 任务上下文",
                "你已进入代码实现预研阶段，需要交叉验证需求文档和现有代码。",
                "## 核心预研任务",
                "链路追踪、兼容性评估和物理环境对齐必须在准备就绪前完成。",
                "## 约束事项",
                "严禁在预研阶段修改任何源代码或配置文件。",
            ]
        )
        source = "\n".join(
            [
                "你已进入代码实现预研阶段，需要交叉验证需求文档和现有代码。",
                "链路追踪、兼容性评估和物理环境对齐必须在准备就绪前完成。",
                "严禁在预研阶段修改任何源代码或配置文件。",
            ]
        )

        self.assertTrue(TmuxBatchWorker._source_mentions_prompt(source, prompt))  # noqa: SLF001

    def test_source_mentions_prompt_rejects_single_fragment_match(self):
        prompt = "\n".join(
            [
                "## 任务上下文",
                "你已进入代码实现预研阶段，需要交叉验证需求文档和现有代码。",
                "## 核心预研任务",
                "链路追踪、兼容性评估和物理环境对齐必须在准备就绪前完成。",
                "## 约束事项",
                "严禁在预研阶段修改任何源代码或配置文件。",
            ]
        )
        source = "你已进入代码实现预研阶段，需要交叉验证需求文档和现有代码。"

        self.assertFalse(TmuxBatchWorker._source_mentions_prompt(source, prompt))  # noqa: SLF001

    def test_wait_for_prompt_submission_accepts_opencode_opaque_tui_delta_as_processing(self):
        class OpaqueTuiPromptWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.observe_calls = 0

            def observe(self, *, tail_lines=320, tail_bytes=24000):
                self.observe_calls += 1
                surface = "■■⬝⬝⬝⬝⬝■■■■⬝⬝⬝"
                observation = WorkerObservation(
                    visible_text=surface,
                    raw_log_delta=surface,
                    raw_log_tail=surface,
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-23T00:00:03",
                    pane_title="OC | demo",
                )
                self.agent_state = self.detector.classify_agent_state(observation)
                return observation

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = OpaqueTuiPromptWorker(
                worker_id="opencode-opaque-tui-prompt-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="kimi-code/kimi-for-coding"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.agent_state = AgentRuntimeState.READY
            worker.pane_id = "%1"
            worker.agent_started = True
            observation = worker._wait_for_prompt_submission(prompt="analyze", timeout_sec=1.0)

        self.assertEqual(observation.current_command, "node")
        self.assertEqual(worker.agent_state, AgentRuntimeState.BUSY)
        self.assertEqual(worker.observe_calls, 1)

    def test_wait_for_prompt_submission_rejects_processing_from_unknown_without_prompt_marker(self):
        class UnknownProcessingPromptWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.observe_calls = 0

            def observe(self, *, tail_lines=320, tail_bytes=24000):
                self.observe_calls += 1
                surface = "■■⬝⬝⬝⬝⬝■■■■⬝⬝⬝"
                observation = WorkerObservation(
                    visible_text=surface,
                    raw_log_delta=surface,
                    raw_log_tail=surface,
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-23T00:00:03",
                    pane_title="OC | demo",
                )
                self.agent_state = self.detector.classify_agent_state(observation)
                return observation

        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch("tmux_core.runtime.tmux_runtime.time.sleep", return_value=None):
            worker = UnknownProcessingPromptWorker(
                worker_id="opencode-unknown-processing-prompt-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="default"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            with self.assertRaises(TimeoutError):
                worker._wait_for_prompt_submission(prompt="analyze", timeout_sec=0.01)

        self.assertGreaterEqual(worker.observe_calls, 1)

    def test_opencode_lightweight_probe_upgrades_processing_to_full_observe_for_ready(self):
        class OpenCodeReadyProbeWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.observe_calls = 0

            def observe(self, *, tail_lines=320, tail_bytes=24000):
                self.observe_calls += 1
                surface = "Ask anything...\nctrl+p commands"
                observation = WorkerObservation(
                    visible_text=surface,
                    raw_log_delta="",
                    raw_log_tail=surface,
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-23T00:00:03",
                    pane_title="OpenCode",
                )
                self.agent_state = self.get_agent_state(observation)
                self.wrapper_state = self._infer_wrapper_state(
                    current_command=observation.current_command,
                    visible_text=observation.visible_text,
                    raw_log_tail=observation.raw_log_tail,
                )
                return observation

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = OpenCodeReadyProbeWorker(
                worker_id="opencode-ready-probe-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="default"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            worker.agent_state = AgentRuntimeState.BUSY
            worker.wrapper_state = WrapperState.NOT_READY

            observation = worker._probe_agent_liveness_for_file_wait()

        self.assertEqual(worker.get_agent_state(observation), AgentRuntimeState.READY)
        self.assertIn(worker.agent_state, {AgentRuntimeState.READY, AgentRuntimeState.READY})
        self.assertEqual(worker.observe_calls, 1)

    def test_opencode_base_probe_and_passive_health_upgrade_processing_to_full_observe(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="opencode-base-ready-probe-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="default"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            worker.agent_state = AgentRuntimeState.BUSY
            observation = WorkerObservation(
                visible_text="Ask anything...\nctrl+p commands",
                raw_log_delta="",
                raw_log_tail="Ask anything...\nctrl+p commands",
                current_command="node",
                current_path=tmp_dir,
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-23T00:00:03",
                pane_title="OpenCode",
            )
            with mock.patch.object(worker, "observe", return_value=observation) as observe:
                self.assertIs(worker._probe_agent_liveness_for_file_wait(), observation)
                self.assertIs(worker._capture_passive_observation(), observation)

        self.assertEqual([call.kwargs.get("tail_bytes") for call in observe.call_args_list], [12000, 12000])

    def test_loaded_opencode_ready_state_stays_ready_on_lightweight_health_refresh(self):
        class LoadedOpenCodeReadyWorker(TmuxBatchWorker):
            def observe(self, *, tail_lines=320, tail_bytes=24000):
                raise AssertionError("READY OpenCode passive health should not require full raw-log observe")

            def _capture_lightweight_observation(self):
                return WorkerObservation(
                    visible_text="",
                    raw_log_delta="",
                    raw_log_tail="",
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-05-06T12:24:53",
                    pane_title="OC | 代码评审测试工程师角色初始化",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runtime_root = root / "runtime"
            runtime_dir = runtime_root / "development-review-08fdcab4"
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "worker_id": "development-review-测试工程师",
                        "session_name": "测试工程师-天寿星",
                        "pane_id": "%24",
                        "work_dir": str(root),
                        "config": {"vendor": "opencode", "model": "opencode/minimax-m2.5-free", "reasoning_effort": "high"},
                        "agent_state": "READY",
                        "agent_started": True,
                        "agent_ready": True,
                        "health_status": "alive",
                        "health_note": "alive",
                        "current_command": "node",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            worker = LoadedOpenCodeReadyWorker(
                worker_id="development-review-测试工程师",
                work_dir=root,
                config=AgentRunConfig(vendor="opencode", model="opencode/minimax-m2.5-free"),
                runtime_root=runtime_root,
                existing_runtime_dir=runtime_dir,
                existing_session_name="测试工程师-天寿星",
                existing_pane_id="%24",
            )
            snapshot = worker.refresh_health(notify_on_change=False)

        self.assertEqual(worker.agent_state, AgentRuntimeState.READY)
        self.assertEqual(snapshot.agent_state, AgentRuntimeState.READY.value)

    def test_running_task_probe_uses_full_observe_for_vendor_state_patterns(self):
        class RunningTaskProbeWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.observe_calls = 0

            def observe(self, *, tail_lines=320, tail_bytes=24000):
                self.observe_calls += 1
                return WorkerObservation(
                    visible_text="› Continue with the current task",
                    raw_log_delta="delta",
                    raw_log_tail="› Continue with the current task",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-25T00:00:03",
                    pane_title="TmuxCodingTeam",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = RunningTaskProbeWorker(
                worker_id="running-task-probe-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            worker.agent_state = AgentRuntimeState.READY
            worker.current_task_runtime_status = "running"

            observation = worker._probe_agent_liveness_for_file_wait()

        self.assertEqual(worker.observe_calls, 1)
        self.assertEqual(observation.current_command, "codex")

    def test_capture_pane_liveness_returns_missing_when_target_is_gone(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="missing-target-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            with mock.patch.object(worker, "session_exists", return_value=True), mock.patch.object(
                worker,
                "target_exists",
                return_value=False,
            ):
                self.assertEqual(worker._capture_pane_liveness_snapshot(), (False, "", "", "", False))

    def test_ensure_agent_ready_does_not_reuse_processing_phase(self):
        class ProcessingWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.wait_called = 0

            def target_exists(self, target=None):
                return True

            def pane_dead(self):
                return False

            def pane_current_command(self):
                return "codex"

            def _wait_for_agent_ready(self, timeout_sec=60.0):
                self.wait_called += 1
                self.agent_ready = True
                self.agent_state = AgentRuntimeState.READY

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = ProcessingWorker(
                worker_id="processing-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_ready = True
            worker.agent_state = AgentRuntimeState.BUSY
            worker.ensure_agent_ready(timeout_sec=0.1)
            self.assertEqual(worker.wait_called, 1)

    def test_has_ever_launched_ignores_metadata_only_state_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="metadata-only-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.state_path.parent.mkdir(parents=True, exist_ok=True)
            worker.state_path.write_text("{}", encoding="utf-8")

            self.assertFalse(worker.has_ever_launched())

    def test_runtime_prelaunch_helper_treats_active_no_pane_state_as_starting_candidate(self):
        payload = {
            "status": "running",
            "result_status": "running",
            "agent_state": "DEAD",
            "agent_started": False,
            "pane_id": "",
            "health_status": "missing_session",
            "workflow_stage": "pending",
        }

        self.assertFalse(worker_state_has_launch_evidence(payload))
        self.assertTrue(worker_state_is_prelaunch_active(payload))

    def test_runtime_prelaunch_helper_treats_session_created_pane_as_starting_candidate(self):
        payload = {
            "status": "ready",
            "result_status": "running",
            "agent_state": "STARTING",
            "agent_started": False,
            "pane_id": "%7",
            "health_status": "alive",
            "workflow_stage": "pending",
            "note": "session_created",
        }

        self.assertFalse(worker_state_has_launch_evidence(payload))
        self.assertTrue(worker_state_is_prelaunch_active(payload))

    def test_runtime_prelaunch_helper_keeps_launched_missing_session_as_not_prelaunch(self):
        self.assertFalse(
            worker_state_is_prelaunch_active(
                {
                    "status": "running",
                    "result_status": "running",
                    "agent_state": "DEAD",
                    "agent_started": True,
                    "pane_id": "",
                    "health_status": "missing_session",
                }
            )
        )
        self.assertFalse(
            worker_state_is_prelaunch_active(
                {
                    "status": "failed",
                    "result_status": "failed",
                    "agent_state": "DEAD",
                    "agent_started": False,
                    "pane_id": "",
                    "health_status": "missing_session",
                }
            )
        )

    def test_ensure_agent_ready_launches_metadata_only_worker_without_pane(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="metadata-only-launch-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.state_path.parent.mkdir(parents=True, exist_ok=True)
            worker.state_path.write_text("{}", encoding="utf-8")

            with mock.patch.object(worker, "launch_agent") as launch_agent:
                worker.ensure_agent_ready(timeout_sec=0.1)

        launch_agent.assert_called_once_with(timeout_sec=0.1)
        self.assertEqual(worker.agent_state, AgentRuntimeState.STARTING)

    def test_ensure_agent_ready_blocks_launched_worker_when_pane_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="launched-missing-pane-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.agent_started = True
            worker.state_path.parent.mkdir(parents=True, exist_ok=True)
            worker.state_path.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "tmux pane missing"):
                worker.ensure_agent_ready(timeout_sec=0.1)

        self.assertTrue(is_worker_death_error("原因: tmux pane missing"))

    def test_wait_for_agent_ready_accepts_codex_work_dir_title(self):
        class ReadyTitleWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.observe_count = 0

            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                self.observe_count += 1
                pane_title = "⠋ tmux-api-v3" if self.observe_count == 1 else "tmux-api-v3"
                return WorkerObservation(
                    visible_text="› Write tests for @filename",
                    raw_log_delta="",
                    raw_log_tail="› Write tests for @filename",
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at=f"2026-04-21T00:00:0{self.observe_count}",
                    pane_title=pane_title,
                )

            def capture_visible(self, tail_lines=500):
                return "› Write tests for @filename"

        with tempfile.TemporaryDirectory(prefix="codex-ready-") as tmp_dir:
            work_dir = Path(tmp_dir) / "tmux-api-v3"
            work_dir.mkdir()
            worker = ReadyTitleWorker(
                worker_id="ready-title-worker",
                work_dir=work_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"

            with mock.patch("tmux_core.runtime.tmux_runtime.time.sleep", return_value=None):
                worker._wait_for_agent_ready(timeout_sec=1.0)

            self.assertTrue(worker.agent_started)
            self.assertTrue(worker.agent_ready)
            self.assertEqual(worker.wrapper_state, WrapperState.READY)
            self.assertEqual(worker.last_pane_title, "tmux-api-v3")
            self.assertEqual(worker.observe_count, 3)
            state = worker.read_state()
            self.assertEqual(state["agent_state"], AgentRuntimeState.READY.value)
            self.assertEqual(state["agent_ready"], True)
            self.assertEqual(state["pane_title"], "tmux-api-v3")

    def test_ensure_agent_ready_syncs_codex_ready_title_to_state_file(self):
        class FastReadyWorker(TmuxBatchWorker):
            def session_exists(self):
                return True

            def target_exists(self, target=None):
                return True

            def _ensure_health_supervisor_started(self):
                return None

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="› Continue with the task",
                    raw_log_delta="",
                    raw_log_tail="› Continue with the task",
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-05-08T18:10:00",
                    pane_title="DRL_PM",
                )

        with tempfile.TemporaryDirectory(prefix="codex-fast-ready-") as tmp_dir:
            work_dir = Path(tmp_dir) / "DRL_PM"
            work_dir.mkdir()
            worker = FastReadyWorker(
                worker_id="fast-ready-worker",
                work_dir=work_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.5"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            worker.agent_ready = False
            worker.agent_state = AgentRuntimeState.BUSY
            worker.wrapper_state = WrapperState.NOT_READY

            worker.ensure_agent_ready(timeout_sec=0.1)

            state = worker.read_state()
            self.assertTrue(worker.agent_ready)
            self.assertEqual(worker.agent_state, AgentRuntimeState.READY)
            self.assertEqual(worker.wrapper_state, WrapperState.READY)
            self.assertEqual(worker.last_pane_title, "DRL_PM")
            self.assertEqual(state["agent_state"], AgentRuntimeState.READY.value)
            self.assertEqual(state["agent_ready"], True)
            self.assertEqual(state["current_command"], "node")
            self.assertEqual(state["current_path"], str(worker.work_dir))
            self.assertEqual(state["pane_title"], "DRL_PM")

    def test_wait_for_agent_ready_rejects_codex_visible_prompt_with_busy_title(self):
        class VisibleReadyWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.observe_count = 0

            def session_exists(self):
                return True

            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                self.observe_count += 1
                return WorkerObservation(
                    visible_text="› Write tests for @filename",
                    raw_log_delta="",
                    raw_log_tail="› Write tests for @filename",
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at=f"2026-04-21T00:01:0{self.observe_count}",
                    pane_title="⠋ tmux-api-v3",
                )

            def capture_visible(self, tail_lines=500):
                return "› Write tests for @filename"

        with tempfile.TemporaryDirectory(prefix="codex-visible-ready-") as tmp_dir:
            work_dir = Path(tmp_dir) / "tmux-api-v3"
            work_dir.mkdir()
            worker = VisibleReadyWorker(
                worker_id="visible-ready-worker",
                work_dir=work_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"

            with mock.patch("tmux_core.runtime.tmux_runtime.time.monotonic", side_effect=[0.0, 0.1, 0.2, 1.2]), \
                    mock.patch("tmux_core.runtime.tmux_runtime.time.sleep", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "Timed out waiting for agent ready"):
                    worker._wait_for_agent_ready(timeout_sec=1.0)

            self.assertFalse(worker.agent_ready)
            self.assertEqual(worker.wrapper_state, WrapperState.NOT_READY)
            self.assertEqual(worker.observe_count, 2)


    def test_codex_boot_prompt_handler_debounces_repeated_enter(self):
        class CodexBootWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.keys: list[str] = []

            def send_special_key(self, key: str) -> None:
                self.keys.append(key)

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = CodexBootWorker(
                worker_id="boot-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            trust_prompt = """
Do you trust the contents of this directory?
› 1. Yes, continue
2. No, quit
Press enter to continue
"""
            self.assertTrue(worker._maybe_handle_codex_boot_prompt(trust_prompt))
            self.assertFalse(worker._maybe_handle_codex_boot_prompt(trust_prompt))
            self.assertEqual(worker.keys, ["Enter"])

    def test_gemini_boot_prompt_handler_accepts_trust_folder_prompt(self):
        class GeminiBootWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.keys: list[str] = []

            def send_special_key(self, key: str) -> None:
                self.keys.append(key)

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = GeminiBootWorker(
                worker_id="boot-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            trust_prompt = """
Do you trust the files in this folder?
1. Trust folder
2. Trust parent folder
3. Don't trust
"""
            self.assertTrue(worker._maybe_handle_gemini_boot_prompt(trust_prompt))
            self.assertFalse(worker._maybe_handle_gemini_boot_prompt(trust_prompt))
            self.assertEqual(worker.keys, ["Enter"])

    def test_wait_for_turn_reply_does_not_require_full_pane_stability_after_token(self):
        class DynamicPaneWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.visible_count = 0

            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                self.visible_count += 1
                return WorkerObservation(
                    visible_text=f"frame-{self.visible_count}",
                    raw_log_delta="done\n[[ACX_TURN:test1234:DONE]]\n[[ROUTING_CREATE:DONE]]",
                    raw_log_tail="done\n[[ACX_TURN:test1234:DONE]]\n[[ROUTING_CREATE:DONE]]",
                    current_command="gemini",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-12T00:00:03",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = DynamicPaneWorker(
                worker_id="gemini-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            reply = worker._wait_for_turn_reply(
                baseline_reply="",
                baseline_visible="baseline",
                turn_token="[[ACX_TURN:test1234:DONE]]",
                required_tokens=["[[ROUTING_CREATE:DONE]]"],
                timeout_sec=0.2,
            )
            self.assertIn("[[ROUTING_CREATE:DONE]]", reply)

    def test_wait_for_turn_reply_requires_done_status(self):
        class DynamicPaneWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.observe_count = 0

            def target_exists(self, target=None):
                return True

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                self.observe_count += 1
                if self.observe_count == 1:
                    task_status_path.write_text('{"status": "running"}', encoding="utf-8")
                else:
                    task_status_path.write_text('{"status": "done"}', encoding="utf-8")
                return WorkerObservation(
                    visible_text=f"frame-{self.observe_count}",
                    raw_log_delta="delta",
                    raw_log_tail="answer\n[[ACX_TURN:test1234:DONE]]\n[[ROUTING_CREATE:DONE]]",
                    current_command="gemini",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-12T00:00:03",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            task_status_path = Path(tmp_dir) / "task_runtime.json"
            worker = DynamicPaneWorker(
                worker_id="gemini-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            reply = worker._wait_for_turn_reply(
                baseline_reply="",
                baseline_visible="baseline",
                turn_token="[[ACX_TURN:test1234:DONE]]",
                required_tokens=["[[ROUTING_CREATE:DONE]]"],
                task_status_path=task_status_path,
                timeout_sec=1.0,
            )
            self.assertIn("[[ROUTING_CREATE:DONE]]", reply)
            self.assertGreaterEqual(worker.observe_count, 2)

    def test_wait_for_turn_reply_accepts_required_token_after_done_without_turn_token(self):
        class ReadyWorker(TmuxBatchWorker):
            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="\n".join(
                        [
                            "• 准备完毕",
                            "",
                            "",
                            "› Use /skills to list available skills",
                            "",
                            "  gpt-5.4 high fast · ~/Desktop/KevinGit/My_C_Tools",
                        ]
                    ),
                    raw_log_delta="",
                    raw_log_tail="helper finished",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-16T00:00:03",
                    pane_title="My_C_Tools",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            task_status_path = Path(tmp_dir) / "task_runtime.json"
            task_status_path.write_text('{"status": "done"}', encoding="utf-8")
            worker = ReadyWorker(
                worker_id="codex-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            reply = worker._wait_for_turn_reply(
                baseline_reply="",
                baseline_visible="baseline",
                turn_token="[[ACX_TURN:test1234:DONE]]",
                required_tokens=["准备完毕"],
                task_status_path=task_status_path,
                timeout_sec=0.2,
            )
            self.assertEqual(reply, "准备完毕")

    def test_extract_reply_from_observation_normalizes_pass_audit_to_token_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="gemini-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            observation = WorkerObservation(
                visible_text="",
                raw_log_delta="",
                raw_log_tail="\n".join(
                    [
                        "⠼ Thinking... (esc to cancel, 14s)   ? for shortcuts",
                        "*   Type your message or @path/to/file",
                        "~/Desktop/KevinGit/PyFinance main no sandbox gemini-3-flash-preview",
                        "[[ACX_TURN:f565aae0:DONE]]",
                        "[[ROUTING_AUDIT:PASS]]",
                    ]
                ),
                current_command="gemini",
                current_path=str(Path(tmp_dir)),
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
            reply = worker._extract_reply_from_observation(
                observation,
                turn_token="[[ACX_TURN:f565aae0:DONE]]",
                required_tokens=["[[ROUTING_AUDIT:PASS]]", "[[ROUTING_AUDIT:REVISE]]"],
            )
            self.assertEqual(reply, "[[ROUTING_AUDIT:PASS]]")

    def test_extract_reply_from_observation_preserves_unexpected_pass_content_for_validation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="gemini-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            observation = WorkerObservation(
                visible_text="",
                raw_log_delta="",
                raw_log_tail="\n".join(
                    [
                        "summary: mostly usable, but here is extra prose",
                        "[[ACX_TURN:f565aae0:DONE]]",
                        "[[ROUTING_AUDIT:PASS]]",
                    ]
                ),
                current_command="gemini",
                current_path=str(Path(tmp_dir)),
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
            reply = worker._extract_reply_from_observation(
                observation,
                turn_token="[[ACX_TURN:f565aae0:DONE]]",
                required_tokens=["[[ROUTING_AUDIT:PASS]]", "[[ROUTING_AUDIT:REVISE]]"],
            )
            self.assertIn("summary: mostly usable", reply)
            self.assertTrue(reply.endswith("[[ROUTING_AUDIT:PASS]]"))

    def test_extract_reply_from_observation_preserves_unexpected_pass_content_after_turn_token(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="gemini-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            observation = WorkerObservation(
                visible_text="",
                raw_log_delta="",
                raw_log_tail="\n".join(
                    [
                        "[[ACX_TURN:f565aae0:DONE]]",
                        "summary: extra prose after turn token",
                        "[[ROUTING_AUDIT:PASS]]",
                    ]
                ),
                current_command="gemini",
                current_path=str(Path(tmp_dir)),
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
            reply = worker._extract_reply_from_observation(
                observation,
                turn_token="[[ACX_TURN:f565aae0:DONE]]",
                required_tokens=["[[ROUTING_AUDIT:PASS]]", "[[ROUTING_AUDIT:REVISE]]"],
            )
            self.assertIn("summary: extra prose after turn token", reply)
            self.assertTrue(reply.endswith("[[ROUTING_AUDIT:PASS]]"))

    def test_extract_reply_from_observation_uses_last_audit_token_in_current_turn(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="gemini-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            observation = WorkerObservation(
                visible_text="",
                raw_log_delta="",
                raw_log_tail="\n".join(
                    [
                        "- verdict: usable_but_drift_prone",
                        "- recommendation: minor_structural_fixes",
                        "[[ACX_TURN:f565aae0:DONE]]",
                        "[[ROUTING_AUDIT:PASS]]",
                        "- finding: severity=high | title=late correction | problem=late | impact=late | evidence=docs/x | direction=use final token",
                        "[[ROUTING_AUDIT:REVISE]]",
                    ]
                ),
                current_command="gemini",
                current_path=str(Path(tmp_dir)),
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
            reply = worker._extract_reply_from_observation(
                observation,
                turn_token="[[ACX_TURN:f565aae0:DONE]]",
                required_tokens=["[[ROUTING_AUDIT:PASS]]", "[[ROUTING_AUDIT:REVISE]]"],
            )
            self.assertIn("- verdict: usable_but_drift_prone", reply)
            self.assertIn("- finding: severity=high | title=late correction", reply)
            self.assertTrue(reply.endswith("[[ROUTING_AUDIT:REVISE]]"))

    def test_extract_reply_from_observation_excludes_prompt_example_bullets_far_before_turn(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="gemini-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            filler = [f"filler line {index}" for index in range(170)]
            observation = WorkerObservation(
                visible_text="",
                raw_log_delta="",
                raw_log_tail="\n".join(
                    [
                        "- finding: severity=critical|high|medium|low | title=<short> | problem=<short> | impact=<short> | evidence=<path[:line]|...> | direction=<short structural fix>",
                        "- duplication: topic=<short> | owner=<file> | overlaps=<file1,file2,...> | value=necessary|wasteful | note=<short>",
                        *filler,
                        "- verdict: usable_but_drift_prone",
                        "- missing: area=docs_pitfalls | effect=missing risk registry | direction=restore file",
                        "- recommendation: minor_structural_fixes",
                        "[[ACX_TURN:test1234:DONE]]",
                        "[[ROUTING_AUDIT:REVISE]]",
                        "────────────────────────────────────────────────────────────────────────────────",
                    ]
                ),
                current_command="gemini",
                current_path=str(Path(tmp_dir)),
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
            reply = worker._extract_reply_from_observation(
                observation,
                turn_token="[[ACX_TURN:test1234:DONE]]",
                required_tokens=["[[ROUTING_AUDIT:PASS]]", "[[ROUTING_AUDIT:REVISE]]"],
            )
            self.assertIn("- verdict: usable_but_drift_prone", reply)
            self.assertIn("- missing: area=docs_pitfalls", reply)
            self.assertNotIn("severity=critical|high|medium|low | title=<short>", reply)
            self.assertNotIn("topic=<short> | owner=<file>", reply)
            self.assertTrue(reply.endswith("[[ROUTING_AUDIT:REVISE]]"))

    def test_extract_reply_from_observation_keeps_only_final_revise_audit_body(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="gemini-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            observation = WorkerObservation(
                visible_text="",
                raw_log_delta="",
                raw_log_tail="\n".join(
                    [
                        "⠼ Thinking... (esc to cancel, 14s)   ? for shortcuts",
                        "- verdict: usable_but_drift_prone",
                        "6. - missing",
                        "* - missing: area=escalation_conditions | effect=Agents may not know when to stop | direction=Add stop_and_verify_when.",
                        "- recommendation: minor_structural_fixes",
                        "- top_priority: Add escalation rule",
                        "[[ACX_TURN:f565aae0:DONE]]",
                        "[[ROUTING_AUDIT:REVISE]]",
                    ]
                ),
                current_command="gemini",
                current_path=str(Path(tmp_dir)),
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
            reply = worker._extract_reply_from_observation(
                observation,
                turn_token="[[ACX_TURN:f565aae0:DONE]]",
                required_tokens=["[[ROUTING_AUDIT:PASS]]", "[[ROUTING_AUDIT:REVISE]]"],
            )
            self.assertIn("- verdict: usable_but_drift_prone", reply)
            self.assertIn("- missing: area=escalation_conditions", reply)
            self.assertIn("- recommendation: minor_structural_fixes", reply)
            self.assertIn("[[ROUTING_AUDIT:REVISE]]", reply)
            self.assertNotIn("Thinking...", reply)

    def test_extract_reply_from_observation_keeps_revise_body_after_turn_token(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="gemini-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            observation = WorkerObservation(
                visible_text="",
                raw_log_delta="",
                raw_log_tail="\n".join(
                    [
                        "- verdict: usable_but_drift_prone",
                        "[[ACX_TURN:f565aae0:DONE]]",
                        "- missing: area=escalation_conditions | effect=Agents may not know when to stop | direction=Add stop_and_verify_when.",
                        "- recommendation: minor_structural_fixes",
                        "[[ROUTING_AUDIT:REVISE]]",
                    ]
                ),
                current_command="gemini",
                current_path=str(Path(tmp_dir)),
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
            reply = worker._extract_reply_from_observation(
                observation,
                turn_token="[[ACX_TURN:f565aae0:DONE]]",
                required_tokens=["[[ROUTING_AUDIT:PASS]]", "[[ROUTING_AUDIT:REVISE]]"],
            )
            self.assertIn("- verdict: usable_but_drift_prone", reply)
            self.assertIn("- missing: area=escalation_conditions", reply)
            self.assertIn("- recommendation: minor_structural_fixes", reply)
            self.assertTrue(reply.endswith("[[ROUTING_AUDIT:REVISE]]"))

    def test_extract_reply_from_observation_ignores_claude_footer_noise_after_revise_token(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="claude-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="claude", model="haiku"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            filler = [f"filler line {index}" for index in range(120)]
            observation = WorkerObservation(
                visible_text="",
                raw_log_delta="",
                raw_log_tail="\n".join(
                    [
                        "- finding: severity=critical|high|medium|low | title=<short> | problem=<short> | impact=<short> | evidence=<path[:line]|...> | direction=<short structural fix>",
                        *filler,
                        "- top_priority: <short>",
                        "- finding: severity=medium | title=Missing pitfalls registry | problem=docs/pitfalls.json is absent | impact=pitfalls become unresolvable | evidence=docs/pitfalls.json | direction=restore docs/pitfalls.json",
                        "- missing: area=Pitfall definitions source file | effect=P01-P05 unresolved | direction=Create docs/pitfalls.json",
                        "- recommendation: minor_structural_fixes",
                        "- top_priority: Restore docs/pitfalls.json",
                        "[[ACX_TURN:testclaude:DONE]]",
                        "[[ROUTING_AUDIT:REVISE]]",
                        "✳ Metamorphosing… (1m 44s · ↓ 1.0k tokens)",
                        "❯",
                        "⏵⏵bypasspermissionson ·escto✗Auto-updatefailed·Tryclaudedocto…",
                        "(·oo·)Crumpet",
                        "Metamorphosing…",
                        "✻ Baked for 1m 44s",
                        "✻ Churned for 1m 9s",
                        "✗ Auto-update faild · Tryclaude doctor or np…",
                        "· Precipitating… (43s · ↓ 1.5k tokens)",
                        "Precipitating…4",
                        "· Hashing… (46s · ↓ 1.2k tokens)",
                        "Hashing…",
                        "✽ Moseying… (38s · ↓ 1.3k tokens)",
                        "✢ Improvising… (38s · ↓ 1.4k tokens)",
                        "✻ Cogitated for 44s",
                        "✻ Sautéed for 46s",
                        "✻ Brewed for 39s",
                    ]
                ),
                current_command="claude",
                current_path=str(Path(tmp_dir)),
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
            reply = worker._extract_reply_from_observation(
                observation,
                turn_token="[[ACX_TURN:testclaude:DONE]]",
                required_tokens=["[[ROUTING_AUDIT:PASS]]", "[[ROUTING_AUDIT:REVISE]]"],
            )
            self.assertIn("- finding: severity=medium | title=Missing pitfalls registry", reply)
            self.assertIn("- missing: area=Pitfall definitions source file", reply)
            self.assertIn("- recommendation: minor_structural_fixes", reply)
            self.assertIn("- top_priority: Restore docs/pitfalls.json", reply)
            self.assertNotIn("title=<short>", reply)
            self.assertNotIn("<short>", reply)
            self.assertNotIn("Metamorphosing", reply)
            self.assertNotIn("Precipitating", reply)
            self.assertNotIn("Hashing", reply)
            self.assertNotIn("Moseying", reply)
            self.assertNotIn("Improvising", reply)
            self.assertNotIn("Cogitated", reply)
            self.assertNotIn("Sautéed", reply)
            self.assertNotIn("Brewed", reply)
            self.assertNotIn("bypasspermissionson", reply)
            self.assertTrue(reply.endswith("[[ROUTING_AUDIT:REVISE]]"))

    def test_extract_reply_from_observation_accepts_prefixed_turn_token_for_written_audit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="claude-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="claude", model="haiku"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            observation = WorkerObservation(
                visible_text="\n".join(
                    [
                        "⏺ Write(audit.json)",
                        "⎿ Wrote 7 lines to audit.json",
                        "⏺ [[ACX_TURN:testclaude:DONE]]",
                        "  [[ROUTING_AUDIT:WRITTEN]]",
                        "✻ Churned for 1m 24s",
                        "❯",
                    ]
                ),
                raw_log_delta="",
                raw_log_tail="",
                current_command="claude",
                current_path=str(Path(tmp_dir)),
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
            reply = worker._extract_reply_from_observation(
                observation,
                turn_token="[[ACX_TURN:testclaude:DONE]]",
                required_tokens=["[[ROUTING_AUDIT:WRITTEN]]"],
            )
            self.assertEqual(reply, "[[ROUTING_AUDIT:WRITTEN]]")

    def test_extract_reply_from_observation_truncates_claude_footer_after_turn_token(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="claude-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="claude", model="haiku"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            observation = WorkerObservation(
                visible_text="\n".join(
                    [
                        "❯ Reply with exactly `OK` on one line.",
                        "",
                        "Turn completion protocol:",
                        "- Output exactly `[[ACX_TURN:testclaude:DONE]]` on its own line before the final workflow token.",
                        "- If no workflow token is required, output `[[ACX_TURN:testclaude:DONE]]` as the final line.",
                        "",
                        "⏺ OK",
                        "",
                        "  [[ACX_TURN:testclaude:DONE]]",
                        "",
                        "────────────────────────────────────────────────────────────────────────────────",
                        "❯",
                        "⏵⏵ bypass permissions on     ✗ Auto-update failed · Try claude doctor or npm …",
                        "(·oo·) Crumpet",
                    ]
                ),
                raw_log_delta="",
                raw_log_tail="",
                current_command="claude",
                current_path=str(Path(tmp_dir)),
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
            reply = worker._extract_reply_from_observation(
                observation,
                turn_token="[[ACX_TURN:testclaude:DONE]]",
                required_tokens=[],
            )
            self.assertIn("OK", reply)
            self.assertNotIn("Crumpet", reply)
            self.assertNotIn("Auto-update failed", reply)

    def test_codex_extract_last_message_truncates_idle_prompt_footer(self):
        detector = CodexOutputDetector()
        output = "\n".join(
            [
                "• 准备完毕",
                "",
                "",
                "› Use /skills to list available skills",
                "",
                "  gpt-5.4 high fast · ~/Desktop/KevinGit/My_C_Tools",
            ]
        )
        self.assertEqual(detector.extract_last_message(output), "• 准备完毕")

    def test_codex_extract_last_message_truncates_idle_footer_after_assistant_label(self):
        detector = CodexOutputDetector()
        output = "\n".join(
            [
                "assistant:",
                "• 完成",
                "",
                "› Use /skills to list available skills",
                "",
                "  gpt-5.4 high fast · ~/Desktop/KevinGit/My_C_Tools",
            ]
        )
        self.assertEqual(detector.extract_last_message(output), "• 完成")

    def test_extract_reply_from_observation_accepts_symbol_prefixed_turn_token_for_written_audit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="gemini-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            observation = WorkerObservation(
                visible_text="\n".join(
                    [
                        "│ ✓  WriteFile Writing to audit.json",
                        "✦ [[ACX_TURN:testgemini:DONE]]",
                        "  [[ROUTING_AUDIT:WRITTEN]]",
                        "? for shortcuts",
                        "*   Type your message or @path/to/file",
                    ]
                ),
                raw_log_delta="",
                raw_log_tail="",
                current_command="gemini",
                current_path=str(Path(tmp_dir)),
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
            reply = worker._extract_reply_from_observation(
                observation,
                turn_token="[[ACX_TURN:testgemini:DONE]]",
                required_tokens=["[[ROUTING_AUDIT:WRITTEN]]"],
            )
            self.assertEqual(reply, "[[ROUTING_AUDIT:WRITTEN]]")

    def test_extract_reply_from_observation_strips_symbolic_tail_noise_after_written_token(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="gemini-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            observation = WorkerObservation(
                visible_text="",
                raw_log_delta="",
                raw_log_tail="\n".join(
                    [
                        "✦ [[ACX_TURN:testtail:DONE]]",
                        "[[ROUTING_AUDIT:WRITTEN]]",
                        "⠋ Working… (1m 3s · esc to cancel)",
                        "*   Type your message or @path/to/file",
                    ]
                ),
                current_command="gemini",
                current_path=str(Path(tmp_dir)),
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
            reply = worker._extract_reply_from_observation(
                observation,
                turn_token="[[ACX_TURN:testtail:DONE]]",
                required_tokens=["[[ROUTING_AUDIT:WRITTEN]]"],
            )
            self.assertEqual(reply, "[[ROUTING_AUDIT:WRITTEN]]")

    def test_extract_reply_from_observation_preserves_trailing_content_after_final_audit_token(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="gemini-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            observation = WorkerObservation(
                visible_text="",
                raw_log_delta="",
                raw_log_tail="\n".join(
                    [
                        "[[ACX_TURN:f565aae0:DONE]]",
                        "[[ROUTING_AUDIT:PASS]]",
                        "summary: trailing prose after final token",
                    ]
                ),
                current_command="gemini",
                current_path=str(Path(tmp_dir)),
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
            reply = worker._extract_reply_from_observation(
                observation,
                turn_token="[[ACX_TURN:f565aae0:DONE]]",
                required_tokens=["[[ROUTING_AUDIT:PASS]]", "[[ROUTING_AUDIT:REVISE]]"],
            )
            self.assertIn("[[ROUTING_AUDIT:PASS]]", reply)
            self.assertTrue(reply.endswith("summary: trailing prose after final token"))

    def test_extract_reply_from_observation_ignores_stale_previous_audit_turns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="gemini-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            observation = WorkerObservation(
                visible_text="",
                raw_log_delta="",
                raw_log_tail="\n".join(
                    [
                        "- verdict: weak",
                        "- finding: severity=high | title=old finding | problem=stale | impact=stale | evidence=docs/a | direction=ignore",
                        "[[ACX_TURN:oldturn:DONE]]",
                        "[[ROUTING_AUDIT:REVISE]]",
                        "- verdict: usable_but_drift_prone",
                        "- finding: severity=medium | title=current finding | problem=current | impact=current | evidence=docs/b | direction=fix current",
                        "- recommendation: minor_structural_fixes",
                        "[[ACX_TURN:newturn:DONE]]",
                        "[[ROUTING_AUDIT:REVISE]]",
                    ]
                ),
                current_command="gemini",
                current_path=str(Path(tmp_dir)),
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
            reply = worker._extract_reply_from_observation(
                observation,
                turn_token="[[ACX_TURN:newturn:DONE]]",
                required_tokens=["[[ROUTING_AUDIT:PASS]]", "[[ROUTING_AUDIT:REVISE]]"],
            )
            self.assertIn("current finding", reply)
            self.assertNotIn("old finding", reply)

    def test_extract_reply_from_observation_does_not_fallback_to_generic_for_partial_audit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="gemini-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            observation = WorkerObservation(
                visible_text="\n".join(
                    [
                        "assistant:",
                        "⠼ Thinking... (esc to cancel, 14s)",
                        "[[ACX_TURN:newturn:DONE]]",
                        "[[ROUTING_AUDIT:REVISE]]",
                    ]
                ),
                raw_log_delta="",
                raw_log_tail="\n".join(
                    [
                        "⠼ Thinking... (esc to cancel, 14s)",
                        "[[ACX_TURN:newturn:DONE]]",
                        "[[ROUTING_AUDIT:REVISE]]",
                    ]
                ),
                current_command="gemini",
                current_path=str(Path(tmp_dir)),
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-12T00:00:00",
            )
            reply = worker._extract_reply_from_observation(
                observation,
                turn_token="[[ACX_TURN:newturn:DONE]]",
                required_tokens=["[[ROUTING_AUDIT:PASS]]", "[[ROUTING_AUDIT:REVISE]]"],
            )
            self.assertEqual(reply, "")

    def test_tmux_backend_tail_raw_log_returns_delta_and_tail(self):
        backend = TmuxBackend()
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_log_path = Path(tmp_dir) / "worker.raw.log"
            raw_log_path.write_text("alpha\nbeta\n", encoding="utf-8")
            delta, tail, next_offset, _ = backend.tail_raw_log(raw_log_path, last_offset=0, tail_bytes=1024)
            self.assertIn("alpha", delta)
            self.assertIn("beta", tail)
            raw_log_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
            delta2, tail2, next_offset2, _ = backend.tail_raw_log(raw_log_path, last_offset=next_offset, tail_bytes=1024)
            self.assertEqual(delta2, "gamma\n")
            self.assertIn("gamma", tail2)
            self.assertGreater(next_offset2, next_offset)

    def test_read_text_tail_handles_missing_and_tail_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.txt"
            self.assertEqual(read_text_tail(path), "(文件不存在)")
            path.write_text("a\nb\nc\n", encoding="utf-8")
            self.assertEqual(read_text_tail(path, max_lines=2), "b\nc")

    def test_runtime_controller_uses_backend_for_session_ops(self):
        calls: list[tuple[str, str]] = []

        class FakeBackend:
            def has_session(self, session_name):
                calls.append(("has", session_name))
                return session_name == "demo"

            def attach_session(self, session_name):
                calls.append(("attach", session_name))

            def detach_session(self, session_name):
                calls.append(("detach", session_name))

            def kill_session(self, session_name):
                calls.append(("kill", session_name))

            def list_sessions(self):
                calls.append(("list", ""))
                return ["demo", "other"]

        controller = TmuxRuntimeController(FakeBackend())
        self.assertTrue(controller.session_exists("demo"))
        self.assertEqual(controller.list_sessions(), ["demo", "other"])
        controller.attach_session("demo")
        controller.detach_session("demo")
        controller.kill_session("demo")
        self.assertIn(("attach", "demo"), calls)
        self.assertIn(("detach", "demo"), calls)
        self.assertIn(("kill", "demo"), calls)
        self.assertIn(("list", ""), calls)

    def test_runtime_controller_rejects_matching_session_name_from_other_work_dir(self):
        class FakeBackend:
            def has_session(self, session_name):
                return session_name == "shared-session"

            def show_option(self, target, option_name):
                return ""

            def display_message(self, target, expression):
                return "/tmp/other-project"

        with tempfile.TemporaryDirectory() as tmp_dir:
            controller = TmuxRuntimeController(FakeBackend())
            self.assertFalse(
                controller.session_matches_context(
                    "shared-session",
                    work_dir=Path(tmp_dir) / "current-project",
                )
            )

    def test_runtime_controller_rejects_matching_session_name_from_other_runtime_identity(self):
        class FakeBackend:
            def __init__(self, runtime_dir: str):
                self.runtime_dir = runtime_dir

            def has_session(self, session_name):
                return session_name == "shared-session"

            def show_option(self, target, option_name):
                if option_name == "@tmux_runtime_dir":
                    return self.runtime_dir
                return ""

        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_a = Path(tmp_dir) / "project-a" / ".development_runtime" / "worker-a"
            runtime_b = Path(tmp_dir) / "project-b" / ".development_runtime" / "worker-b"
            controller = TmuxRuntimeController(FakeBackend(str(runtime_b)))
            self.assertFalse(
                controller.session_matches_context(
                    "shared-session",
                    runtime_dir=runtime_a,
                    work_dir=runtime_a.parent.parent,
                )
            )

    def test_worker_restart_and_kill_are_runtime_level_ops(self):
        class FakeBackend:
            def has_session(self, session_name):
                return True

            def kill_session(self, session_name):
                self.last_killed = session_name

            def run(self, *args, **kwargs):
                raise AssertionError("unexpected tmux run")

        with tempfile.TemporaryDirectory() as tmp_dir:
            backend = FakeBackend()
            worker = TmuxBatchWorker(
                worker_id="ops-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime",
                backend=backend,
            )
            worker.agent_ready = True
            session_name = worker.request_restart()
            self.assertEqual(session_name, worker.session_name)
            self.assertFalse(worker.agent_ready)
            self.assertTrue(worker.recoverable)
            self.assertEqual(worker.agent_state, AgentRuntimeState.STARTING)
            session_name = worker.request_kill()
            self.assertEqual(session_name, worker.session_name)
            self.assertFalse(worker.recoverable)
            self.assertEqual(worker.agent_state, AgentRuntimeState.DEAD)

    def test_cleanup_registered_tmux_workers_kills_live_sessions(self):
        class FakeBackend:
            def __init__(self):
                self.live_sessions = {"session-a", "session-b"}
                self.killed: list[str] = []

            def has_session(self, session_name):
                return session_name in self.live_sessions

            def kill_session(self, session_name):
                self.killed.append(session_name)
                self.live_sessions.discard(session_name)

            def run(self, *args, **kwargs):
                raise AssertionError("unexpected tmux run")

        with tempfile.TemporaryDirectory() as tmp_dir:
            backend = FakeBackend()
            worker_a = TmuxBatchWorker(
                worker_id="cleanup-a",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4"),
                runtime_root=Path(tmp_dir) / "runtime-a",
                backend=backend,
                existing_session_name="session-a",
            )
            worker_b = TmuxBatchWorker(
                worker_id="cleanup-b",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime-b",
                backend=backend,
                existing_session_name="session-b",
            )
            cleaned = cleanup_registered_tmux_workers(reason="unit_test")
            self.assertEqual(sorted(cleaned), ["session-a", "session-b"])
            self.assertEqual(sorted(backend.killed), ["session-a", "session-b"])
            self.assertFalse(worker_a.recoverable)
            self.assertFalse(worker_b.recoverable)

    def test_create_session_refuses_after_runtime_shutdown_requested(self):
        class FakeBackend:
            def __init__(self):
                self.created: list[str] = []

            def list_sessions(self):
                return []

            def has_session(self, session_name):
                return False

            def run(self, *args, **kwargs):
                return SimpleNamespace(stdout="", returncode=0)

            def create_session(self, session_name, work_dir, command):
                self.created.append(session_name)
                return "%1"

            def kill_session(self, session_name):
                raise AssertionError("no session should be created")

        clear_runtime_shutdown_request()
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                backend = FakeBackend()
                worker = TmuxBatchWorker(
                    worker_id="shutdown-create-worker",
                    work_dir=tmp_dir,
                    config=AgentRunConfig(vendor="codex", model="gpt-5.4"),
                    runtime_root=Path(tmp_dir) / "runtime",
                    backend=backend,
                )
                request_runtime_shutdown("unit_test")
                with self.assertRaises(RuntimeShutdownRequested):
                    worker.create_session()
                self.assertEqual(backend.created, [])
        finally:
            clear_runtime_shutdown_request()

    def test_create_session_kills_session_created_during_runtime_shutdown(self):
        class FakeBackend:
            def __init__(self):
                self.live_sessions: set[str] = set()
                self.created: list[str] = []
                self.killed: list[str] = []

            def list_sessions(self):
                return sorted(self.live_sessions)

            def has_session(self, session_name):
                return session_name in self.live_sessions

            def run(self, *args, **kwargs):
                return SimpleNamespace(stdout="", returncode=0)

            def create_session(self, session_name, work_dir, command):
                self.created.append(session_name)
                self.live_sessions.add(session_name)
                request_runtime_shutdown("unit_test")
                return "%1"

            def kill_session(self, session_name):
                self.killed.append(session_name)
                self.live_sessions.discard(session_name)

        clear_runtime_shutdown_request()
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                backend = FakeBackend()
                worker = TmuxBatchWorker(
                    worker_id="shutdown-create-race-worker",
                    work_dir=tmp_dir,
                    config=AgentRunConfig(vendor="codex", model="gpt-5.4"),
                    runtime_root=Path(tmp_dir) / "runtime",
                    backend=backend,
                )
                with self.assertRaises(RuntimeShutdownRequested):
                    worker.create_session()
                self.assertEqual(backend.created, [worker.session_name])
                self.assertEqual(backend.killed, [worker.session_name])
                self.assertEqual(backend.live_sessions, set())
                self.assertEqual(worker.pane_id, "")
        finally:
            clear_runtime_shutdown_request()

    def test_refresh_health_auto_relaunch_is_blocked(self):
        class RelaunchWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.ready_calls = 0

            def session_exists(self):
                return False if self.ready_calls == 0 else True

            def ensure_agent_ready(self, timeout_sec=60.0):
                self.ready_calls += 1
                self.agent_ready = True
                self.agent_state = AgentRuntimeState.READY
                self.last_heartbeat_at = "2026-04-12T00:00:00"

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = RelaunchWorker(
                worker_id="relaunch-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            snapshot = worker.refresh_health(auto_relaunch=True, relaunch_timeout_sec=0.1)
            self.assertIsInstance(snapshot, WorkerHealthSnapshot)
            self.assertEqual(snapshot.health_status, "unknown")
            self.assertEqual(snapshot.agent_state, AgentRuntimeState.STARTING.value)
            self.assertEqual(worker.ready_calls, 0)

    def test_refresh_health_auto_relaunch_keeps_opencode_blocked(self):
        class RelaunchWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.ready_calls = 0

            def session_exists(self):
                return False if self.ready_calls == 0 else True

            def ensure_agent_ready(self, timeout_sec=60.0):
                self.ready_calls += 1
                self.agent_ready = True
                self.agent_started = True
                self.agent_state = AgentRuntimeState.READY
                self.last_heartbeat_at = "2026-04-22T00:00:00"

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = RelaunchWorker(
                worker_id="opencode-health-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="default"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            snapshot = worker.refresh_health(auto_relaunch=True, relaunch_timeout_sec=0.1)
            self.assertEqual(snapshot.health_status, "unknown")
            self.assertEqual(snapshot.agent_state, AgentRuntimeState.STARTING.value)
            self.assertEqual(worker.ready_calls, 0)

    def test_opencode_output_detector_classifies_starting_ready_and_busy_surfaces(self):
        detector = OpenCodeOutputDetector()
        booting_phase = detector.classify_agent_state(
            WorkerObservation(
                visible_text="Performing one time database migration...\nDatabase migration complete.",
                raw_log_delta="",
                raw_log_tail="Performing one time database migration...\nDatabase migration complete.",
                current_command="node",
                current_path="/tmp/project",
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-22T00:00:00",
                pane_title="OpenCode",
            )
        )
        waiting_phase = detector.classify_agent_state(
            WorkerObservation(
                visible_text='Ask anything... "Fix a TODO in the codebase"\ntab agents  ctrl+p commands',
                raw_log_delta="",
                raw_log_tail='Ask anything... "Fix a TODO in the codebase"\ntab agents  ctrl+p commands',
                current_command="node",
                current_path="/tmp/project",
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-22T00:00:01",
                pane_title="OpenCode",
            )
        )
        processing_phase = detector.classify_agent_state(
            WorkerObservation(
                visible_text="Thinking: The user wants me to reply with exactly OK.\n■■■⬝⬝⬝⬝⬝  esc interrupt",
                raw_log_delta="",
                raw_log_tail="Thinking: The user wants me to reply with exactly OK.\n■■■⬝⬝⬝⬝⬝  esc interrupt",
                current_command="node",
                current_path="/tmp/project",
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-22T00:00:02",
                pane_title="OpenCode",
            )
        )
        footer_ready_state = detector.classify_agent_state(
            WorkerObservation(
                visible_text="OK\n\n10.9K  ctrl+p commands",
                raw_log_delta="",
                raw_log_tail="Thinking: The user wants me to reply with exactly OK.\nOK\n\n10.9K  ctrl+p commands",
                current_command="node",
                current_path="/tmp/project",
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-22T00:00:03",
                pane_title="OpenCode",
            )
        )
        wrapped_footer_ready_state = detector.classify_agent_state(
            WorkerObservation(
                visible_text="OK\n\n10.9K  ctrl+p\ncommands",
                raw_log_delta="",
                raw_log_tail="Thinking: The user wants me to reply with exactly OK.\nOK\n\n10.9K  ctrl+p\ncommands",
                current_command="node",
                current_path="/tmp/project",
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-22T00:00:04",
                pane_title="OpenCode",
            )
        )
        opaque_tui_processing_phase = detector.classify_agent_state(
            WorkerObservation(
                visible_text="■■⬝⬝⬝⬝⬝■■■■⬝⬝⬝",
                raw_log_delta="■■⬝⬝⬝⬝⬝■■■■⬝⬝⬝",
                raw_log_tail="■■⬝⬝⬝⬝⬝■■■■⬝⬝⬝",
                current_command="node",
                current_path="/tmp/project",
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-22T00:00:04",
                pane_title="OpenCode",
            )
        )
        self.assertEqual(booting_phase, AgentRuntimeState.STARTING)
        self.assertEqual(waiting_phase, AgentRuntimeState.READY)
        self.assertEqual(processing_phase, AgentRuntimeState.BUSY)
        self.assertEqual(footer_ready_state, AgentRuntimeState.READY)
        self.assertEqual(wrapped_footer_ready_state, AgentRuntimeState.READY)
        self.assertEqual(opaque_tui_processing_phase, AgentRuntimeState.BUSY)
        narrow_processing_phase = detector.classify_agent_state(
            WorkerObservation(
                visible_text="Thinking: The user wants me to reply with exactly OK.\nesc\ninterrupt\n10.9K ctrl+p\ncommand\ns",
                raw_log_delta="",
                raw_log_tail="Thinking: The user wants me to reply with exactly OK.\nesc\ninterrupt\n10.9K ctrl+p\ncommand\ns",
                current_command="node",
                current_path="/tmp/project",
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-22T00:00:05",
                pane_title="OpenCode",
            )
        )
        extreme_wrapped_footer_ready_state = detector.classify_agent_state(
            WorkerObservation(
                visible_text="OK\n\n10.9Kctrl+p\ncommand\ns",
                raw_log_delta="",
                raw_log_tail="Thinking: The user wants me to reply with exactly OK.\nOK\n\n10.9Kctrl+p\ncommand\ns",
                current_command="node",
                current_path="/tmp/project",
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-22T00:00:06",
                pane_title="OpenCode",
            )
        )
        self.assertEqual(narrow_processing_phase, AgentRuntimeState.BUSY)
        self.assertEqual(extreme_wrapped_footer_ready_state, AgentRuntimeState.READY)

    def test_wait_for_agent_ready_supports_opencode_visible_ready_without_title_ready(self):
        class OpenCodeReadyWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.observe_calls = 0

            def observe(self, tail_lines=220):
                self.observe_calls += 1
                return WorkerObservation(
                    visible_text='Ask anything... "Fix a TODO in the codebase"\ntab agents  ctrl+p commands',
                    raw_log_delta="",
                    raw_log_tail='Ask anything... "Fix a TODO in the codebase"\ntab agents  ctrl+p commands',
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-22T00:00:00",
                    pane_title="OpenCode",
                )

        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch("tmux_core.runtime.tmux_runtime.time.sleep", return_value=None):
            worker = OpenCodeReadyWorker(
                worker_id="opencode-ready-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="default"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker._wait_for_agent_ready(timeout_sec=1.0)
            self.assertTrue(worker.agent_ready)
            self.assertTrue(worker.agent_started)
            self.assertEqual(worker.wrapper_state, WrapperState.READY)
            self.assertEqual(worker.current_command, "node")
            self.assertGreaterEqual(worker.observe_calls, 2)

    def test_opencode_get_agent_state_uses_recent_log_for_extreme_narrow_ready_footer(self):
        class OpenCodeStateWorker(TmuxBatchWorker):
            def target_exists(self, target=None):  # noqa: ARG002
                return True

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = OpenCodeStateWorker(
                worker_id="opencode-narrow-ready-state",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="default"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.agent_started = True
            worker.pane_id = "%1"
            observation = WorkerObservation(
                visible_text="10.9Kctrl+p\ncommand\ns",
                raw_log_delta="",
                raw_log_tail="Created hello.txt with the content hi.\n\n10.9K  ctrl+p commands",
                current_command="node",
                current_path=str(worker.work_dir),
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-22T00:00:07",
                pane_title="OC | demo",
            )

            self.assertEqual(worker.get_agent_state(observation), AgentRuntimeState.READY)

    def test_opencode_get_agent_state_without_observation_accepts_ready_wrapper_state(self):
        class OpenCodeCachedStateWorker(TmuxBatchWorker):
            def session_exists(self):
                return True

            def target_exists(self, target=None):  # noqa: ARG002
                return True

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = OpenCodeCachedStateWorker(
                worker_id="opencode-ready-cached",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="default"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.agent_started = True
            worker.pane_id = "%1"
            worker.current_command = "node"
            worker.current_path = str(worker.work_dir)
            worker.agent_state = AgentRuntimeState.READY
            worker.last_pane_title = "OpenCode"

            self.assertEqual(worker.get_agent_state().value, AgentRuntimeState.READY.value)

    def test_opencode_get_agent_state_without_observation_accepts_cached_ready_state(self):
        class OpenCodeCachedPhaseWorker(TmuxBatchWorker):
            def session_exists(self):
                return True

            def target_exists(self, target=None):  # noqa: ARG002
                return True

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = OpenCodeCachedPhaseWorker(
                worker_id="opencode-ready-phase",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="default"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.agent_started = True
            worker.pane_id = "%1"
            worker.current_command = "node"
            worker.current_path = str(worker.work_dir)
            worker.agent_state = AgentRuntimeState.READY
            worker.last_pane_title = "OpenCode"

            self.assertEqual(worker.get_agent_state().value, AgentRuntimeState.READY.value)

    def test_opencode_empty_observation_uses_busy_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="opencode-empty-observation",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="default"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            observation = WorkerObservation(
                visible_text="",
                raw_log_delta="",
                raw_log_tail="",
                current_command="node",
                current_path=tmp_dir,
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-24T00:00:00",
                pane_title="",
            )

            worker.agent_state = AgentRuntimeState.READY
            self.assertEqual(worker.get_agent_state(observation), AgentRuntimeState.READY)
            worker.agent_state = AgentRuntimeState.STARTING
            self.assertEqual(worker.get_agent_state(observation), AgentRuntimeState.BUSY)

    def test_worker_does_not_restore_legacy_provider_phase_from_state_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_root = Path(tmp_dir) / "runtime"
            worker = TmuxBatchWorker(
                worker_id="phase-state-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=runtime_root,
            )
            worker.state_path.write_text(
                json.dumps(
                    {
                        "session_name": worker.session_name,
                        "pane_id": "%1",
                        "provider_phase": "waiting_input",
                        "agent_started": True,
                        "current_command": "codex",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            restored = TmuxBatchWorker(
                worker_id="phase-state-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                existing_runtime_dir=worker.runtime_dir,
                existing_session_name=worker.session_name,
                existing_pane_id="%1",
                runtime_root=runtime_root,
            )

            self.assertEqual(restored.agent_state, AgentRuntimeState.STARTING)
            self.assertTrue(restored.agent_started)

    def test_passive_health_refresh_updates_state_without_consuming_terminal_output_or_raw_log(self):
        class PassiveHealthWorker(TmuxBatchWorker):
            def session_exists(self):
                return True

            def target_exists(self, target=None):
                return True

            def capture_visible(self, tail_lines=500):
                raise AssertionError("passive health refresh should not capture terminal output")

            def pane_current_command(self):
                return "codex"

            def pane_current_path(self):
                return str(self.work_dir)

            def pane_title(self):
                return "TmuxCodingTeam"

            def pane_dead(self):
                return False

            def tail_raw_log(self, *, tail_bytes=24000):
                raise AssertionError("passive health refresh should not consume raw log")

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = PassiveHealthWorker(
                worker_id="passive-health-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            worker.last_log_offset = 123
            worker._write_state(WorkerStatus.READY, note="seed")
            previous_updated_at = str(worker.read_state().get("updated_at", ""))

            snapshot = worker.refresh_health()
            state = worker.read_state()

            self.assertEqual(snapshot.health_status, "alive")
            self.assertEqual(snapshot.health_note, "alive")
            self.assertEqual(state["health_status"], "alive")
            self.assertEqual(state["health_note"], "alive")
            self.assertEqual(state["current_command"], "codex")
            self.assertEqual(state["current_path"], str(worker.work_dir))
            self.assertEqual(str(state.get("updated_at", "")), str(state.get("last_heartbeat_at", "")))
            self.assertTrue(str(state.get("updated_at", "")))
            self.assertEqual(worker.last_log_offset, 123)

    def test_prelaunch_worker_without_session_stays_starting_during_turn_start(self):
        class PendingSessionWorker(TmuxBatchWorker):
            def session_exists(self):
                return False

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = PendingSessionWorker(
                worker_id="pending-session-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="opencode/big-pickle"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.current_task_runtime_status = "running"

            self.assertEqual(worker.get_agent_state(), AgentRuntimeState.STARTING)
            worker._write_state(  # noqa: SLF001
                WorkerStatus.RUNNING,
                note="turn:review",
                extra={
                    "result_status": "running",
                    "current_task_runtime_status": "running",
                },
            )
            snapshot = worker.refresh_health()
            state = worker.read_state()

        self.assertEqual(snapshot.agent_state, AgentRuntimeState.STARTING.value)
        self.assertEqual(snapshot.health_status, "unknown")
        self.assertEqual(snapshot.health_note, "launch pending")
        self.assertEqual(state["agent_state"], AgentRuntimeState.STARTING.value)
        self.assertEqual(state["health_status"], "unknown")
        self.assertEqual(state["health_note"], "launch pending")

    def test_session_created_worker_with_missing_session_stays_starting_before_agent_launch(self):
        class PendingSessionCreatedWorker(TmuxBatchWorker):
            def session_exists(self):
                return False

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = PendingSessionCreatedWorker(
                worker_id="pending-session-created-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="opencode/big-pickle"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%7"
            worker._write_state(  # noqa: SLF001
                WorkerStatus.READY,
                note="session_created",
                extra={
                    "result_status": "running",
                    "workflow_stage": "pending",
                },
            )
            snapshot = worker.refresh_health()
            state = worker.read_state()

        self.assertEqual(snapshot.agent_state, AgentRuntimeState.STARTING.value)
        self.assertEqual(snapshot.health_status, "unknown")
        self.assertEqual(snapshot.health_note, "launch pending")
        self.assertEqual(state["agent_state"], AgentRuntimeState.STARTING.value)
        self.assertEqual(state["health_status"], "unknown")
        self.assertEqual(state["health_note"], "launch pending")

    def test_missing_session_after_launch_still_reports_dead(self):
        class MissingLaunchedSessionWorker(TmuxBatchWorker):
            def session_exists(self):
                return False

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = MissingLaunchedSessionWorker(
                worker_id="missing-launched-session-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="opencode", model="opencode/big-pickle"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            snapshot = worker.refresh_health()
            state = worker.read_state()

        self.assertEqual(snapshot.agent_state, AgentRuntimeState.DEAD.value)
        self.assertEqual(snapshot.health_status, "missing_session")
        self.assertEqual(state["agent_state"], AgentRuntimeState.DEAD.value)
        self.assertEqual(state["health_status"], "missing_session")

    def test_lightweight_liveness_probe_handles_missing_target_and_display_failures(self):
        class MissingSessionWorker(TmuxBatchWorker):
            def session_exists(self):
                return False

        class MissingTargetWorker(TmuxBatchWorker):
            def session_exists(self):
                return True

            def target_exists(self, target=None):
                raise subprocess.CalledProcessError(1, "tmux")

        class DisplayFailureWorker(TmuxBatchWorker):
            def session_exists(self):
                return True

            def target_exists(self, target=None):
                return True

            def pane_current_command(self):
                raise subprocess.CalledProcessError(1, "tmux")

            def pane_current_path(self):
                raise subprocess.CalledProcessError(1, "tmux")

            def pane_title(self):
                raise subprocess.CalledProcessError(1, "tmux")

            def pane_dead(self):
                raise subprocess.CalledProcessError(1, "tmux")

        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_session = MissingSessionWorker(
                worker_id="missing-session-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            missing_session.pane_id = "%1"
            self.assertEqual(missing_session._capture_pane_liveness_snapshot(), (False, "", "", "", False))  # noqa: SLF001

            missing = MissingTargetWorker(
                worker_id="missing-target-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            missing.pane_id = "%1"
            self.assertEqual(missing._capture_pane_liveness_snapshot(), (False, "", "", "", False))  # noqa: SLF001

            display = DisplayFailureWorker(
                worker_id="display-failure-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            display.pane_id = "%1"
            display.current_command = "codex"
            display.current_path = tmp_dir
            display.last_pane_title = "TmuxCodingTeam"
            self.assertEqual(
                display._capture_pane_liveness_snapshot(),  # noqa: SLF001
                (True, "codex", tmp_dir, "TmuxCodingTeam", False),
            )
            self.assertEqual(display._capture_passive_observation().pane_title, "TmuxCodingTeam")  # noqa: SLF001

    def test_file_wait_liveness_probe_skips_until_probe_interval(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="probe-skip-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            observation, last_probe = worker._maybe_probe_agent_liveness_for_file_wait(  # noqa: SLF001
                last_probe_monotonic=time.monotonic(),
                status_done_seen=False,
            )

        self.assertIsNone(observation)
        self.assertGreater(last_probe, 0)

    def test_passive_health_refresh_notifies_runtime_state_change_when_health_changes(self):
        class PassiveHealthWorker(TmuxBatchWorker):
            def session_exists(self):
                return True

            def target_exists(self, target=None):
                return True

            def capture_visible(self, tail_lines=500):
                return """
› Continue with the current task
  gpt-5.4 high · ~/Desktop/KevinGit/My_C_Tools
"""

            def pane_current_command(self):
                return "codex"

            def pane_current_path(self):
                return str(self.work_dir)

            def pane_dead(self):
                return False

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = PassiveHealthWorker(
                worker_id="passive-health-notify-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            worker._write_state(WorkerStatus.READY, note="seed")

            with mock.patch("T02_tmux_agents._notify_runtime_state_changed_best_effort") as notifier:
                worker.refresh_health()

            notifier.assert_called_once_with()

    def test_passive_health_refresh_does_not_classify_auth_error_from_terminal_text(self):
        class PassiveAuthErrorWorker(TmuxBatchWorker):
            def session_exists(self):
                return True

            def target_exists(self, target=None):
                return True

            def capture_visible(self, tail_lines=500):
                raise AssertionError("passive health refresh should not capture terminal output")

            def pane_current_command(self):
                return "codex"

            def pane_current_path(self):
                return str(self.work_dir)

            def pane_title(self):
                return "TmuxCodingTeam"

            def pane_dead(self):
                return False

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = PassiveAuthErrorWorker(
                worker_id="passive-auth-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            worker._write_state(WorkerStatus.READY, note="seed")

            snapshot = worker.refresh_health()
            state = worker.read_state()

            self.assertEqual(snapshot.health_status, "alive")
            self.assertEqual(snapshot.health_note, "alive")
            self.assertEqual(state["health_status"], "alive")
            self.assertEqual(state["health_note"], "alive")

    def test_codex_busy_title_overrides_visible_ready_prompt(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="codex-stale-title-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            observation = WorkerObservation(
                visible_text="› Explain this codebase\n  gpt-5.4 xhigh · ~/project",
                raw_log_delta="",
                raw_log_tail="",
                current_command="node",
                current_path=tmp_dir,
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-26T10:40:00",
                pane_title=f"⠼ {worker.work_dir.name}",
            )

            self.assertEqual(worker.get_agent_state(observation), AgentRuntimeState.BUSY)

    def test_codex_busy_title_wins_over_visible_ready_while_task_running(self):
        with tempfile.TemporaryDirectory(prefix="tmux-api-v3-dev-parent-") as tmp_dir:
            work_dir = Path(tmp_dir) / "tmux-api-v3-dev"
            work_dir.mkdir()
            worker = TmuxBatchWorker(
                worker_id="codex-running-busy-title-worker",
                work_dir=work_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            worker.current_task_runtime_status = "running"
            observation = WorkerObservation(
                visible_text="› Explain this codebase\n  gpt-5.4 xhigh · ~/project",
                raw_log_delta="",
                raw_log_tail="",
                current_command="node",
                current_path=str(work_dir),
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-26T10:40:00",
                pane_title=f"⠼ {worker.work_dir.name}",
            )

            self.assertEqual(worker.get_agent_state(observation), AgentRuntimeState.BUSY)

    def test_codex_passive_health_keeps_busy_title_without_visible_probe(self):
        class StaleBusyTitleWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.capture_calls = 0

            def session_exists(self):
                return True

            def target_exists(self, target=None):
                return True

            def capture_visible(self, tail_lines=500):
                self.capture_calls += 1
                return "› Explain this codebase\n  gpt-5.4 xhigh · ~/project"

            def pane_current_command(self):
                return "node"

            def pane_current_path(self):
                return str(self.work_dir)

            def pane_title(self):
                return f"⠼ {self.work_dir.name}"

            def pane_dead(self):
                return False

            def tail_raw_log(self, *, tail_bytes=24000):
                raise AssertionError("passive Codex title-only status should not consume raw log")

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = StaleBusyTitleWorker(
                worker_id="codex-passive-stale-title-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            worker.agent_state = AgentRuntimeState.BUSY
            worker.current_command = "node"
            worker.last_pane_title = f"⠼ {worker.work_dir.name}"
            worker.last_log_offset = 77
            worker._write_state(WorkerStatus.READY, note="seed")

            snapshot = worker.refresh_health()
            state = worker.read_state()

        self.assertEqual(snapshot.agent_state, AgentRuntimeState.BUSY.value)
        self.assertEqual(state["agent_state"], AgentRuntimeState.BUSY.value)
        self.assertEqual(worker.capture_calls, 0)
        self.assertEqual(worker.last_log_offset, 77)

    def test_claude_passive_health_uses_ready_title_prefix_when_title_is_task_name(self):
        class TaskTitleClaudeWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.capture_calls = 0

            def session_exists(self):
                return True

            def target_exists(self, target=None):
                return True

            def capture_visible(self, tail_lines=500):
                self.capture_calls += 1
                return """
⏺ Bash(sleep 2 && echo claude-ready-regression-done)
  ⎿  claude-ready-regression-done

⏺ 命令执行成功，输出为：claude-ready-regression-done

✻ Brewed for 20s

────────────────────────────────────────────────────────────────────────────────
❯
────────────────────────────────────────────────────────────────────────────────
  ⏵⏵ bypass permissions on (shift+tab to cycle)
""".strip()

            def pane_current_command(self):
                return "claude.exe"

            def pane_current_path(self):
                return str(self.work_dir)

            def pane_title(self):
                return "✳ Execute command and report results"

            def pane_dead(self):
                return False

            def tail_raw_log(self, *, tail_bytes=24000):
                raise AssertionError("passive Claude ready recovery should not consume raw log")

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TaskTitleClaudeWorker(
                worker_id="claude-passive-task-title-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="claude", model="sonnet"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            worker.agent_state = AgentRuntimeState.BUSY
            worker.current_command = "claude.exe"
            worker.last_pane_title = "✳ Execute command and report results"
            worker.last_log_offset = 88
            worker._write_state(WorkerStatus.READY, note="seed")

            snapshot = worker.refresh_health()
            state = worker.read_state()

        self.assertEqual(snapshot.agent_state, AgentRuntimeState.READY.value)
        self.assertEqual(state["agent_state"], AgentRuntimeState.READY.value)
        self.assertEqual(worker.capture_calls, 0)
        self.assertEqual(worker.last_log_offset, 88)

    def test_passive_health_snapshot_maps_busy_and_dead_agent_states(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="passive-phase-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            busy = worker._build_passive_health_snapshot(  # noqa: SLF001
                WorkerObservation(
                    visible_text="",
                    raw_log_delta="",
                    raw_log_tail="",
                    current_command="codex",
                    current_path=tmp_dir,
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-24T00:00:00",
                    pane_title="⠋ TmuxCodingTeam",
                )
            )
            self.assertEqual(busy.agent_state, AgentRuntimeState.BUSY.value)

            dead = worker._build_passive_health_snapshot(  # noqa: SLF001
                WorkerObservation(
                    visible_text="",
                    raw_log_delta="",
                    raw_log_tail="",
                    current_command="",
                    current_path=tmp_dir,
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-24T00:00:00",
                    pane_title="",
                )
            )
            self.assertEqual(dead.agent_state, AgentRuntimeState.DEAD.value)

    def test_run_turn_keeps_ready_agent_state_on_non_death_contract_error(self):
        class ContractErrorWorker(TmuxBatchWorker):
            def session_exists(self):
                return True

            def target_exists(self, target=None):  # noqa: ANN001, ARG002
                return True

            def capture_visible(self, tail_lines=200):  # noqa: ANN001, ARG002
                return "• 准备就绪\n\n› Summarize recent commits"

            def ensure_agent_ready(self, timeout_sec=60.0):  # noqa: ANN001, ARG002
                self.agent_started = True
                self.agent_ready = True
                self.agent_state = AgentRuntimeState.READY
                self.current_command = "node"

            def observe(self, *, tail_lines=120, tail_bytes=24000):  # noqa: ANN001, ARG002
                self.agent_started = True
                self.agent_state = AgentRuntimeState.READY
                self.current_command = "node"
                return WorkerObservation(
                    visible_text="• 准备就绪\n\n› Summarize recent commits",
                    raw_log_delta="",
                    raw_log_tail="• 准备就绪",
                    current_command="node",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-24T00:00:00",
                    pane_title=self.work_dir.name,
                )

            def _send_text(self, text, enter_count=None):  # noqa: ANN001, ARG002
                return None

            def _wait_for_prompt_submission(self, *, prompt, timeout_sec):  # noqa: ANN001, ARG002
                return self.observe()

            def wait_for_task_result(self, **kwargs):  # noqa: ANN003
                raise RuntimeError(f"{TASK_RESULT_CONTRACT_ERROR_PREFIX}: missing result.json")

            def _build_passive_health_snapshot(self, observation=None):  # noqa: ANN001, ARG002
                return TmuxAgentsTests._health_snapshot(agent_state=AgentRuntimeState.READY.value)

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = ContractErrorWorker(
                worker_id="contract-error-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker.pane_id = "%1"
            worker.agent_started = True
            worker.agent_ready = True
            worker.agent_state = AgentRuntimeState.READY
            worker.current_command = "node"
            contract = TaskResultContract(
                turn_id="a07_developer_init",
                phase="a07_developer_init",
                task_kind="a07_developer_init",
                mode="a07_developer_init",
                expected_statuses=("ready", "hitl"),
                optional_artifacts={"ask_human": Path(tmp_dir) / "ask.md"},
            )

            result = worker.run_turn(
                label="development_developer_init",
                prompt="初始化",
                result_contract=contract,
                timeout_sec=0.1,
            )
            state = worker.read_state()

        self.assertFalse(result.ok)
        self.assertEqual(state["status"], WorkerStatus.FAILED.value)
        self.assertEqual(state["result_status"], "failed")
        self.assertEqual(state["agent_state"], AgentRuntimeState.READY.value)
        self.assertTrue(state["agent_ready"])
        self.assertEqual(state["health_status"], "alive")

    def test_agent_state_detector_returns_ready_without_provider_debounce(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="gemini-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            observation = WorkerObservation(
                visible_text="Type your message or @path/to/file",
                raw_log_delta="",
                raw_log_tail="Type your message or @path/to/file",
                current_command="gemini",
                current_path=tmp_dir,
                pane_dead=False,
                session_exists=True,
                log_mtime=0.0,
                observed_at="2026-04-24T00:00:00",
                pane_title="",
            )

            self.assertEqual(worker.detector.classify_agent_state(observation), AgentRuntimeState.READY)

    def test_launch_coordinator_backoff_doubles_on_failure_and_resets_on_success(self):
        LaunchCoordinator._stagger_by_vendor.clear()
        self.assertEqual(LaunchCoordinator.current_stagger(Vendor.GEMINI), 2.0)
        LaunchCoordinator.record_launch_result(Vendor.GEMINI, success=False)
        self.assertEqual(LaunchCoordinator.current_stagger(Vendor.GEMINI), 4.0)
        LaunchCoordinator.record_launch_result(Vendor.GEMINI, success=False)
        self.assertEqual(LaunchCoordinator.current_stagger(Vendor.GEMINI), 8.0)
        LaunchCoordinator.record_launch_result(Vendor.GEMINI, success=True)
        self.assertEqual(LaunchCoordinator.current_stagger(Vendor.GEMINI), 2.0)

    def test_run_turn_retries_once_after_timeout(self):
        class RetryOnceWorker(TmuxBatchWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.wait_calls = 0
                self.ready_calls = 0

            def _write_state(self, status, *, note, extra=None):
                return None

            def _append_transcript(self, title, body):
                return None

            def ensure_agent_ready(self, timeout_sec=60.0):
                self.pane_id = "%1"
                self.agent_ready = True
                self.ready_calls += 1

            def observe(self, *, tail_lines=500, tail_bytes=24000):
                return WorkerObservation(
                    visible_text="visible",
                    raw_log_delta="",
                    raw_log_tail="",
                    current_command="codex",
                    current_path=str(self.work_dir),
                    pane_dead=False,
                    session_exists=True,
                    log_mtime=0.0,
                    observed_at="2026-04-12T00:00:00",
                )

            def target_exists(self, target=None):
                return True

            def _send_text(self, text, enter_count=None):
                return None

            def _wait_for_turn_reply(self, **kwargs):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    raise TimeoutError("timed out once")
                return "ok\n[[DONE]]"

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = RetryOnceWorker(
                worker_id="retry-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            result = worker.run_turn(label="retry_case", prompt="hello", required_tokens=["[[DONE]]"], timeout_sec=0.05)
            self.assertTrue(result.ok)
            self.assertEqual(worker.wait_calls, 2)
            self.assertGreaterEqual(worker.ready_calls, 2)

    def test_build_turn_prompt_does_not_embed_runtime_context_header(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="plain-prompt-worker",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="gemini", model="flash"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            task_status_path = Path(tmp_dir) / "task_runtime.json"
            submitted = worker._build_turn_prompt(
                "hello world",
                "[[ACX_TURN:test:DONE]]",
                ["[[DONE]]"],
                task_status_path=task_status_path,
                complete_task_command="/tmp/complete_task --status done",
                include_turn_protocol=True,
            )
            self.assertIn("hello world", submitted)
            self.assertNotIn("[Agent Runtime Context]", submitted)
            self.assertNotIn(str(task_status_path.resolve()), submitted)
            self.assertNotIn("complete_task", submitted)




    def test_worker_session_names_are_reserved_across_local_initialization(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_root = Path(tmp_dir) / "runtime"
            worker_a = TmuxBatchWorker(
                worker_id="requirements-analyst",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=runtime_root,
            )
            worker_b = TmuxBatchWorker(
                worker_id="requirements-analyst",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=runtime_root,
            )
            self.assertNotEqual(worker_a.session_name, worker_b.session_name)

    def test_session_name_reservation_is_released_after_session_creation(self):
        class FakeBackend:
            def __init__(self):
                self.live_sessions: set[str] = set()

            def list_sessions(self):
                return list(self.live_sessions)

            def has_session(self, session_name):
                return session_name in self.live_sessions

            def run(self, *args, **kwargs):  # noqa: ANN003
                _ = args
                _ = kwargs
                return subprocess.CompletedProcess(["tmux"], 0, "", "")

            def create_session(self, session_name, work_dir, command):
                self.live_sessions.add(session_name)
                return "%1"

        with tempfile.TemporaryDirectory() as tmp_dir:
            backend = FakeBackend()
            worker = TmuxBatchWorker(
                worker_id="requirements-analyst",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
                backend=backend,
            )
            self.assertTrue(worker._session_name_reserved)
            worker._tmux = lambda *args, **kwargs: subprocess.CompletedProcess(["tmux"], 0, "", "")
            worker._start_pipe_logging = lambda: None
            worker._ensure_health_supervisor_started = lambda: None
            worker._refresh_health_state_nonintrusive = lambda: None
            worker._log_event = lambda *args, **kwargs: None
            worker._write_state = lambda *args, **kwargs: None
            worker.create_session()
            self.assertFalse(worker._session_name_reserved)

    def test_create_session_sets_tmux_history_limit_to_10000(self):
        class FakeBackend:
            def __init__(self):
                self.live_sessions: set[str] = set()
                self.run_calls: list[tuple[str, ...]] = []

            def list_sessions(self):
                return list(self.live_sessions)

            def has_session(self, session_name):
                return session_name in self.live_sessions

            def run(self, *args, **kwargs):  # noqa: ANN003
                _ = kwargs
                self.run_calls.append(tuple(str(arg) for arg in args))
                return subprocess.CompletedProcess(["tmux", *args], 0, "", "")

            def create_session(self, session_name, work_dir, command):
                _ = work_dir
                _ = command
                self.live_sessions.add(session_name)
                return "%1"

        with tempfile.TemporaryDirectory() as tmp_dir:
            backend = FakeBackend()
            worker = TmuxBatchWorker(
                worker_id="requirements-analyst",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
                backend=backend,
            )
            tmux_calls: list[tuple[str, ...]] = []
            worker._tmux = lambda *args, **kwargs: (  # noqa: ARG005
                tmux_calls.append(tuple(str(arg) for arg in args)),
                subprocess.CompletedProcess(["tmux", *args], 0, "", ""),
            )[1]
            worker._start_pipe_logging = lambda: None
            worker._ensure_health_supervisor_started = lambda: None
            worker._refresh_health_state_nonintrusive = lambda: None
            worker._log_event = lambda *args, **kwargs: None
            worker._write_state = lambda *args, **kwargs: None

            worker.create_session()

        self.assertIn(("set-option", "-g", "history-limit", "10000"), backend.run_calls)
        self.assertIn(("set-option", "-t", worker.session_name, "history-limit", "10000"), tmux_calls)

    def test_session_name_reservation_is_released_when_session_creation_fails(self):
        class FakeBackend:
            def list_sessions(self):
                return []

            def has_session(self, session_name):
                return False

            def run(self, *args, **kwargs):  # noqa: ANN003
                _ = args
                _ = kwargs
                return subprocess.CompletedProcess(["tmux"], 0, "", "")

            def create_session(self, session_name, work_dir, command):
                raise RuntimeError("tmux create failed")

        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="requirements-analyst",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
                backend=FakeBackend(),
            )
            self.assertTrue(worker._session_name_reserved)
            with self.assertRaisesRegex(RuntimeError, "tmux create failed"):
                worker.create_session()
            self.assertFalse(worker._session_name_reserved)

    def test_create_session_renames_on_precheck_conflict_without_killing_existing_session(self):
        class FakeBackend:
            def __init__(self):
                self.live_sessions: set[str] = set()
                self.kill_calls: list[str] = []
                self.create_calls: list[str] = []

            def list_sessions(self):
                return list(self.live_sessions)

            def has_session(self, session_name):
                return session_name in self.live_sessions

            def run(self, *args, **kwargs):  # noqa: ANN003
                _ = args
                _ = kwargs
                return subprocess.CompletedProcess(["tmux"], 0, "", "")

            def create_session(self, session_name, work_dir, command):
                _ = work_dir
                _ = command
                self.create_calls.append(session_name)
                if session_name in self.live_sessions:
                    raise subprocess.CalledProcessError(
                        1,
                        ["tmux", "new-session"],
                        stderr=f"duplicate session: {session_name}",
                    )
                self.live_sessions.add(session_name)
                return "%2"

            def kill_session(self, session_name):
                self.kill_calls.append(session_name)

        with tempfile.TemporaryDirectory() as tmp_dir:
            backend = FakeBackend()
            worker = TmuxBatchWorker(
                worker_id="requirements-analyst",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
                backend=backend,
            )
            original_name = worker.session_name
            backend.live_sessions.add(original_name)
            events: list[tuple[str, dict[str, object]]] = []
            worker._tmux = lambda *args, **kwargs: subprocess.CompletedProcess(["tmux"], 0, "", "")
            worker._start_pipe_logging = lambda: None
            worker._ensure_health_supervisor_started = lambda: None
            worker._refresh_health_state_nonintrusive = lambda: None
            worker._write_state = lambda *args, **kwargs: None
            worker._log_event = lambda event, **payload: events.append((event, payload))

            pane_id = worker.create_session()

            self.assertEqual(pane_id, "%2")
            self.assertNotEqual(worker.session_name, original_name)
            self.assertIn(original_name, backend.live_sessions)
            self.assertIn(worker.session_name, backend.live_sessions)
            self.assertEqual(backend.kill_calls, [])
            self.assertTrue(any(event == "session_name_conflict_retry" for event, _ in events))
            self.assertFalse(worker._session_name_reserved)

    def test_create_session_conflict_retry_fails_fast_after_retry_limit(self):
        class AlwaysConflictBackend:
            def __init__(self):
                self.kill_calls: list[str] = []
                self.create_calls: list[str] = []

            def list_sessions(self):
                return []

            def has_session(self, session_name):
                _ = session_name
                return True

            def create_session(self, session_name, work_dir, command):
                _ = work_dir
                _ = command
                self.create_calls.append(session_name)
                raise AssertionError("create_session should not be called when pre-check always conflicts")

            def kill_session(self, session_name):
                self.kill_calls.append(session_name)

        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch(
            "tmux_core.runtime.tmux_runtime.SESSION_NAME_CREATE_MAX_RETRIES",
            2,
        ):
            backend = AlwaysConflictBackend()
            worker = TmuxBatchWorker(
                worker_id="requirements-analyst",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
                backend=backend,
            )
            worker._log_event = lambda *args, **kwargs: None

            with self.assertRaisesRegex(RuntimeError, "conflict_key="):
                worker.create_session()
            self.assertEqual(backend.kill_calls, [])
            self.assertFalse(worker._session_name_reserved)

    def test_session_name_reservation_uses_global_lease_before_create(self):
        import tmux_core.runtime.tmux_runtime as tmux_runtime

        class FakeBackend:
            def list_sessions(self):
                return []

        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch.object(
            tmux_runtime,
            "_SESSION_NAME_LEASE_ROOT",
            Path(tmp_dir) / "leases",
        ):
            backend = FakeBackend()
            worker_a = TmuxBatchWorker(
                worker_id="requirements-analyst",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime-a",
                backend=backend,
            )
            first_name = worker_a.session_name
            tmux_runtime._RESERVED_SESSION_NAMES.clear()  # noqa: SLF001

            worker_b = TmuxBatchWorker(
                worker_id="requirements-analyst",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime-b",
                backend=backend,
            )

            self.assertNotEqual(worker_b.session_name, first_name)
            occupied = set(list_occupied_tmux_session_names(backend=backend))
            self.assertIn(first_name, occupied)
            self.assertIn(worker_b.session_name, occupied)
            worker_a._release_session_name_reservation()  # noqa: SLF001
            worker_b._release_session_name_reservation()  # noqa: SLF001

    def test_mark_awaiting_reconfiguration_rewrites_failed_state_to_running(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = TmuxBatchWorker(
                worker_id="requirements-analyst",
                work_dir=tmp_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5.4-mini"),
                runtime_root=Path(tmp_dir) / "runtime",
            )
            worker._write_state(WorkerStatus.FAILED, note="error:init")  # noqa: SLF001

            worker.mark_awaiting_reconfiguration(reason_text="需要重新选择模型")

            state = worker.read_state()
            self.assertEqual(state["status"], WorkerStatus.RUNNING.value)
            self.assertEqual(state["result_status"], "running")
            self.assertEqual(state["note"], "awaiting_reconfig")
            self.assertEqual(state["health_status"], "awaiting_reconfig")
            self.assertEqual(state["health_note"], "需要重新选择模型")



if __name__ == "__main__":
    unittest.main()
