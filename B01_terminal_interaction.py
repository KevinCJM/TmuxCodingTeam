# -*- encoding: utf-8 -*-
"""
@File: B01_terminal_interaction.py
@Modify Time: 2026/4/12
@Author: Kevin-Chen
@Descriptions: AGENT初始化阶段的终端交互控制层
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tmux_core.runtime.vendor_catalog import get_default_model_for_vendor
from A01_Routing_LayerPlanning import (
    CliRequest,
    DEFAULT_MODEL_BY_VENDOR,
    EMPTY_PROJECT_ROUTING_SKIP_MESSAGE,
    build_parser,
    determine_exit_code,
    display_status_label,
    normalize_run_init_choice,
    normalize_vendor_choice,
    prompt_confirmation,
    prompt_effort,
    prompt_model,
    prompt_project_dir,
    prompt_target_dirs,
    prompt_vendor,
    render_noop_summary,
    render_preflight_summary,
)
from T02_tmux_agents import (
    AgentRunConfig,
    HealthSupervisor,
    TmuxBatchWorker,
    TmuxRuntimeController,
    cleanup_registered_tmux_workers,
)
from T09_terminal_ops import (
    maybe_launch_tui,
    message,
    prompt_command_line,
    prompt_with_default,
    prompt_yes_no as terminal_prompt_yes_no,
)
from T03_agent_init_workflow import (
    BatchInitResult,
    DirectoryInitResult,
    LiveWorkerHandle,
    RunStore,
    TargetSelection,
    build_batch_result,
    cleanup_routing_stage_artifacts,
    determine_batch_worker_count,
    kill_run_tmux_sessions,
    load_existing_run,
    missing_routing_layer_files,
    project_has_business_files,
    prepare_live_workers,
    required_routing_layer_paths,
    resolve_existing_directory,
    resolve_target_selection,
    run_directory_initialization_with_worker,
    should_skip_routing_init_for_empty_project,
)


ACTIVE_ROUTING_WORKFLOW_STAGES = {"create_running", "audit_running", "refine_running"}
PRELAUNCH_AGENT_STATES = {"", "STARTING", "DEAD"}


@dataclass(frozen=True)
class ControlCommand:
    action: str
    argument: str = ""


@dataclass(frozen=True)
class SessionStatusRow:
    index: int
    work_dir: str
    session_name: str
    status: str
    workflow_stage: str
    agent_state: str
    health_status: str
    retry_count: int
    session_exists: bool
    note: str
    forced: bool


@dataclass(frozen=True)
class WorkerTarget:
    work_dir: str
    session_name: str
    transcript_path: str
    live_handle: LiveWorkerHandle | None


def _is_prelaunch_active_worker(entry: object) -> bool:
    workflow_stage = str(getattr(entry, "workflow_stage", "") or "").strip()
    result_status = str(getattr(entry, "result_status", "") or "").strip()
    agent_state = str(getattr(entry, "agent_state", "") or "").strip().upper()
    agent_started = bool(getattr(entry, "agent_started", False))
    pane_id = str(getattr(entry, "pane_id", "") or "").strip()
    return (
        workflow_stage in ACTIVE_ROUTING_WORKFLOW_STAGES
        and result_status in {"pending", "ready", "running"}
        and agent_state in PRELAUNCH_AGENT_STATES
        and not agent_started
        and not pane_id
    )


def _collect_snapshot_paths(node: object) -> list[str]:
    flattened: list[str] = []
    if isinstance(node, dict):
        for value in node.values():
            flattened.extend(_collect_snapshot_paths(value))
        return flattened
    if isinstance(node, (list, tuple, set)):
        for value in node:
            flattened.extend(_collect_snapshot_paths(value))
        return flattened
    if node is None:
        return flattened
    text = str(node).strip()
    if text:
        flattened.append(text)
    return flattened


def _read_turn_artifact_bundle(turn_status_path: str) -> dict[str, object]:
    resolved = Path(turn_status_path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        return {"artifact_paths": [], "question_path": "", "answer_path": ""}
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"artifact_paths": [], "question_path": "", "answer_path": ""}
    if not isinstance(payload, dict):
        return {"artifact_paths": [], "question_path": "", "answer_path": ""}
    artifacts = payload.get("artifacts", {})
    artifact_paths = [item for item in _collect_snapshot_paths(artifacts) if Path(item).exists()]
    question_path = str(artifacts.get("question", "")).strip() if isinstance(artifacts, dict) else ""
    answer_path = str(artifacts.get("record", "")).strip() if isinstance(artifacts, dict) else ""
    stage_status_path = str(artifacts.get("stage_status", "")).strip() if isinstance(artifacts, dict) else ""
    if stage_status_path:
        stage_status_file = Path(stage_status_path).expanduser().resolve()
        if stage_status_file.exists() and stage_status_file.is_file():
            try:
                stage_payload = json.loads(stage_status_file.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                stage_payload = {}
            if isinstance(stage_payload, dict):
                question_path = str(stage_payload.get("question_path", question_path)).strip()
                answer_path = str(stage_payload.get("record_path", answer_path)).strip()
                output_path = str(stage_payload.get("output_path", "")).strip()
                if output_path:
                    artifact_paths.append(output_path)
                if question_path:
                    artifact_paths.append(question_path)
                if answer_path:
                    artifact_paths.append(answer_path)
    deduped_paths: list[str] = []
    for item in artifact_paths:
        resolved_item = str(Path(item).expanduser().resolve())
        if resolved_item not in deduped_paths:
            deduped_paths.append(resolved_item)
    return {
        "artifact_paths": deduped_paths,
        "question_path": question_path,
        "answer_path": answer_path,
    }


def prompt_yes_no(prompt_text: str, default: bool = False) -> bool:
    return normalize_run_init_choice("yes" if terminal_prompt_yes_no(prompt_text, default) else "no")


def collect_b01_request(args: argparse.Namespace) -> CliRequest:
    parameter_mode = any(
        [
            args.target_dir,
            args.vendor,
            args.model,
            args.effort,
            args.proxy_port,
            args.run_init,
        ]
    )
    project_dir = (
        str(resolve_existing_directory(args.project_dir))
        if args.project_dir
        else prompt_project_dir("")
    )
    target_dirs = tuple(args.target_dir or ())
    project_missing_files = tuple(missing_routing_layer_files(project_dir))
    explicit_run_init_yes = bool(args.run_init and normalize_run_init_choice(args.run_init))
    should_skip_empty_project = should_skip_routing_init_for_empty_project(
        project_dir,
        project_missing_files=project_missing_files,
        target_dirs=target_dirs,
        explicit_run_init_yes=explicit_run_init_yes,
    )
    if project_missing_files:
        requested_run_init = not should_skip_empty_project
    else:
        requested_run_init = (
            normalize_run_init_choice(args.run_init)
            if args.run_init
            else (True if parameter_mode else prompt_yes_no("是否需要生成项目路由层", True))
        )
    run_init = requested_run_init
    if project_missing_files and not should_skip_empty_project and (
        (args.run_init and not normalize_run_init_choice(args.run_init))
        or (not args.run_init and not parameter_mode)
    ):
        message("当前项目路由层文件缺失, 强制执行路由初始化")

    if run_init and not args.target_dir and not parameter_mode:
        target_dirs = prompt_target_dirs(())
    if not run_init:
        target_dirs = tuple()

    if run_init:
        vendor = normalize_vendor_choice(args.vendor) if args.vendor else prompt_vendor("codex")
        model_default = get_default_model_for_vendor(vendor)
        model = args.model or prompt_model(vendor, model_default)
        reasoning_effort = args.effort or ("high" if parameter_mode else prompt_effort(vendor, model, "high"))

        if args.proxy_port:
            proxy_port = args.proxy_port
        elif parameter_mode:
            proxy_port = ""
        else:
            use_proxy = prompt_yes_no("是否使用代理端口", False)
            proxy_port = prompt_with_default("代理端口或完整代理 URL", "", allow_empty=False) if use_proxy else ""
    else:
        vendor = normalize_vendor_choice(args.vendor or "codex")
        model = args.model or get_default_model_for_vendor(vendor)
        reasoning_effort = args.effort or "high"
        proxy_port = args.proxy_port or ""

    return CliRequest(
        project_dir=project_dir,
        target_dirs=target_dirs,
        vendor=vendor,
        model=model,
        reasoning_effort=reasoning_effort,
        proxy_port=proxy_port,
        run_init=run_init,
        max_refine_rounds=max(int(args.max_refine_rounds or 3), 1),
        auto_confirm=bool(args.yes),
    )


def parse_control_command(text: str) -> ControlCommand:
    raw = str(text or "").strip()
    if not raw:
        return ControlCommand("status")
    parts = raw.split(maxsplit=1)
    action = parts[0].strip().lower()
    argument = parts[1].strip() if len(parts) > 1 else ""
    aliases = {
        "ls": "status",
        "list": "status",
        "show": "status",
        "open": "attach",
        "tail": "transcript",
        "log": "transcript",
        "logs": "transcript",
        "quit": "exit",
    }
    return ControlCommand(aliases.get(action, action), argument)


def render_control_help() -> str:
    return "\n".join(
        [
            "可用命令:",
            "  status                           查看所有 tmux 会话状态",
            "  attach <编号|session_name>       attach 到指定 tmux 会话",
            "  transcript <编号|session_name>   查看指定会话 transcript 尾部",
            "  detach <编号|session_name>       从 B01 侧 detach 指定会话上的 client",
            "  restart <编号|session_name>      人工请求重启当前 tmux 会话",
            "  kill <编号|session_name>         杀掉当前 tmux 会话并关闭自动恢复",
            "  retry <编号|session_name>        对 failed/stale_failed 目录重新创建 worker 并重跑",
            "  wait                             阻塞等待全部目录执行完成",
            "  resume <run_id>                  当前 run 完成后切换到已存在 run",
            "  help                             显示帮助",
            "  exit                             全部完成后退出控制台",
            "",
            "attach 后如果要返回 B01，可在 tmux 内按 Ctrl-b d。",
        ]
    )


class AgentInitControlCenter:
    def __init__(
            self,
            *,
            selection: TargetSelection,
            config: AgentRunConfig,
            max_refine_rounds: int,
            run_store: RunStore,
            live_workers: Sequence[LiveWorkerHandle],
            initial_results: dict[str, DirectoryInitResult],
            max_workers: int | None = None,
            worker_factory=TmuxBatchWorker,
            resume_mode: bool = False,
            runtime_controller: TmuxRuntimeController | None = None,
    ) -> None:
        self.selection = selection
        self.config = config
        self.max_refine_rounds = max_refine_rounds
        self.run_store = run_store
        self.run_id = self.run_store.manifest.run_id
        self.run_root = self.run_store.run_root
        self.live_workers = list(live_workers)
        self.handle_by_dir = {item.work_dir: item for item in self.live_workers}
        self.results_by_dir: dict[str, DirectoryInitResult] = dict(initial_results)
        self.worker_factory = worker_factory
        self.resume_mode = resume_mode
        self.tmux_runtime = runtime_controller or TmuxRuntimeController()

        self.executor = ThreadPoolExecutor(
            max_workers=determine_batch_worker_count([item.work_dir for item in self.live_workers],
                                                     max_workers=max_workers)
            if self.live_workers
            else 1
        )
        self.futures: dict[Future[DirectoryInitResult], LiveWorkerHandle] = {}
        self.started = False
        self.supervisor = HealthSupervisor(
            self.refresh_worker_health,
            interval_sec=2.0,
            thread_name=f"health-{self.run_id}",
        )
        self._supervisor_stopped = False
        self._requirements_placeholder_entered = False
        self.supervisor.start()

    @classmethod
    def create_new(
            cls,
            *,
            selection: TargetSelection,
            config: AgentRunConfig,
            max_refine_rounds: int,
            runtime_root: str | Path | None = None,
            max_workers: int | None = None,
            worker_factory=TmuxBatchWorker,
    ) -> "AgentInitControlCenter":
        run_store, live_workers, immediate_results = prepare_live_workers(
            selection=selection,
            config=config,
            runtime_root=runtime_root,
            worker_factory=worker_factory,
        )
        results_by_dir: dict[str, DirectoryInitResult] = {}
        for skipped_dir in selection.skipped_dirs:
            results_by_dir[skipped_dir] = DirectoryInitResult(
                work_dir=skipped_dir,
                forced=False,
                status="skipped",
                rounds_used=0,
            )
        for item in immediate_results:
            results_by_dir[item.work_dir] = item
        return cls(
            selection=selection,
            config=config,
            max_refine_rounds=max_refine_rounds,
            run_store=run_store,
            live_workers=live_workers,
            initial_results=results_by_dir,
            max_workers=max_workers,
            worker_factory=worker_factory,
            resume_mode=False,
            runtime_controller=TmuxRuntimeController(),
        )

    @classmethod
    def from_existing_run(
            cls,
            *,
            run_id: str,
            project_dir: str | Path | None = None,
            runtime_root: str | Path | None = None,
            max_refine_rounds: int = 3,
            max_workers: int | None = None,
            worker_factory=TmuxBatchWorker,
    ) -> "AgentInitControlCenter":
        run_store, selection, config, live_workers, results_by_dir = load_existing_run(
            run_id=run_id,
            project_dir=project_dir,
            runtime_root=runtime_root,
            worker_factory=worker_factory,
        )
        return cls(
            selection=selection,
            config=config,
            max_refine_rounds=max_refine_rounds,
            run_store=run_store,
            live_workers=live_workers,
            initial_results=results_by_dir,
            max_workers=max_workers,
            worker_factory=worker_factory,
            resume_mode=True,
            runtime_controller=TmuxRuntimeController(),
        )

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        if self.live_workers:
            self.run_store.set_status("running")
        for handle in self.live_workers:
            if handle.work_dir in self.results_by_dir:
                continue
            self._submit_handle(handle)

    def _submit_handle(self, handle: LiveWorkerHandle) -> None:
        future = self.executor.submit(
            run_directory_initialization_with_worker,
            worker=handle.worker,
            forced=handle.forced,
            max_refine_rounds=self.max_refine_rounds,
            run_store=self.run_store,
            resume_state=handle.resume_state,
        )
        self.futures[future] = handle

    def _harvest_done_futures(self) -> None:
        for future, handle in list(self.futures.items()):
            if not future.done():
                continue
            try:
                result = future.result()
            except Exception as error:
                result = DirectoryInitResult(
                    work_dir=handle.work_dir,
                    forced=handle.forced,
                    status="failed",
                    rounds_used=0,
                    missing_before=[],
                    missing_after=[],
                    failure_reason=f"batch_future_exception: {error}",
                    runtime_dir=str(handle.worker.runtime_dir),
                    session_name=handle.worker.session_name,
                )
                self.run_store.update_worker_result(result)
            self.results_by_dir[result.work_dir] = result
            del self.futures[future]

    def _entry_by_dir(self, work_dir: str):
        return self.run_store.ensure_worker(work_dir=work_dir)

    def _refresh_entry_from_state_file(self, work_dir: str):
        entry = self._entry_by_dir(work_dir)
        state_path = str(getattr(entry, "state_path", "") or "").strip()
        if not state_path:
            return entry
        try:
            updated = self.run_store.update_worker_state_from_file(
                work_dir,
                state_path,
                preserve_workflow_fields=True,
            )
        except Exception:
            return entry
        return updated or entry

    def _resolve_work_dir(self, argument: str) -> str:
        if not argument:
            raise ValueError("请提供会话编号或 session_name")
        candidate = str(argument).strip()
        if candidate.isdigit():
            index = int(candidate)
            if index < 1 or index > len(self.selection.selected_dirs):
                raise ValueError(f"编号越界: {index}")
            return self.selection.selected_dirs[index - 1]

        for handle in self.live_workers:
            if handle.session_name == candidate:
                return handle.work_dir
        for entry in self.run_store.manifest.workers:
            if entry.session_name == candidate:
                return entry.work_dir
        raise ValueError(f"未找到会话: {argument}")

    def get_target(self, argument: str) -> WorkerTarget:
        work_dir = self._resolve_work_dir(argument)
        handle = self.handle_by_dir.get(work_dir)
        entry = self._entry_by_dir(work_dir)
        transcript_path = entry.transcript_path or (str(handle.worker.transcript_path) if handle else "")
        session_name = entry.session_name or (handle.session_name if handle else "")
        return WorkerTarget(
            work_dir=work_dir,
            session_name=session_name,
            transcript_path=transcript_path,
            live_handle=handle,
        )

    def all_done(self) -> bool:
        self._harvest_done_futures()
        return not self.futures

    def pending_work_count(self) -> int:
        return sum(1 for work_dir in self.selection.selected_dirs if work_dir not in self.results_by_dir)

    def refresh_worker_health(self) -> None:
        for handle in list(self.live_workers):
            if handle.work_dir in self.results_by_dir:
                continue
            entry = self._entry_by_dir(handle.work_dir)
            if _is_prelaunch_active_worker(entry):
                self.run_store.sync_worker_snapshot(
                    handle.work_dir,
                    preserve_workflow_fields=True,
                    agent_state="STARTING",
                    health_status="unknown",
                    health_note="launch pending",
                    session_name=entry.session_name or handle.worker.session_name,
                    runtime_dir=entry.runtime_dir or str(handle.worker.runtime_dir),
                    pane_id=entry.pane_id,
                )
                continue
            snapshot = handle.worker.refresh_health(
                auto_relaunch=False,
                relaunch_timeout_sec=30.0,
            )

            self.run_store.sync_worker_snapshot(
                handle.work_dir,
                state_path=handle.worker.state_path,
                preserve_workflow_fields=True,
                agent_state=snapshot.agent_state,
                health_status=snapshot.health_status,
                health_note=snapshot.health_note,
                last_heartbeat_at=snapshot.last_heartbeat_at,
                last_log_offset=snapshot.last_log_offset,
                current_command=snapshot.current_command,
                current_path=snapshot.current_path,
                session_name=snapshot.session_name,
                runtime_dir=str(handle.worker.runtime_dir),
                pane_id=snapshot.pane_id,
            )

    def build_status_rows(self) -> list[SessionStatusRow]:
        self._harvest_done_futures()
        rows: list[SessionStatusRow] = []
        for index, work_dir in enumerate(self.selection.selected_dirs, start=1):
            finished = self.results_by_dir.get(work_dir)
            entry = self._refresh_entry_from_state_file(work_dir)
            handle = self.handle_by_dir.get(work_dir)
            if finished is not None:
                status_value = entry.result_status if entry.result_status else display_status_label(finished)
                note = finished.failure_reason or finished.last_audit_token or entry.note or "completed"
                rows.append(
                    SessionStatusRow(
                        index=index,
                        work_dir=work_dir,
                        session_name=entry.session_name or finished.session_name or (
                            handle.session_name if handle else ""),
                        status=status_value,
                        workflow_stage=entry.workflow_stage,
                        agent_state=entry.agent_state,
                        health_status=entry.health_status,
                        retry_count=entry.retry_count,
                        session_exists=self.tmux_runtime.session_exists(entry.session_name),
                        note=note,
                        forced=entry.forced if entry else bool(handle.forced if handle else finished.forced),
                    )
                )
                continue

            if handle is None:
                rows.append(
                    SessionStatusRow(
                        index=index,
                        work_dir=work_dir,
                        session_name=entry.session_name,
                        status=entry.result_status or "pending",
                        workflow_stage=entry.workflow_stage,
                        agent_state=entry.agent_state,
                        health_status=entry.health_status,
                        retry_count=entry.retry_count,
                        session_exists=self.tmux_runtime.session_exists(entry.session_name),
                        note=entry.note or entry.health_note or "pending",
                        forced=entry.forced,
                    )
                )
                continue

            rows.append(
                SessionStatusRow(
                    index=index,
                    work_dir=work_dir,
                    session_name=entry.session_name or handle.session_name,
                    status=entry.result_status if entry.result_status != "pending" else (
                                entry.workflow_stage or "pending"),
                    workflow_stage=entry.workflow_stage,
                    agent_state=entry.agent_state,
                    health_status=entry.health_status,
                    retry_count=entry.retry_count,
                    session_exists=self.tmux_runtime.session_exists(entry.session_name or handle.session_name),
                    note=entry.note or entry.workflow_stage or "pending",
                    forced=entry.forced or handle.forced,
                )
            )
        return rows

    def build_worker_snapshots(self) -> list[dict[str, object]]:
        snapshots: list[dict[str, object]] = []
        for row in self.build_status_rows():
            entry = self._entry_by_dir(row.work_dir)
            artifact_bundle = _read_turn_artifact_bundle(entry.current_turn_status_path)
            artifact_paths = list(artifact_bundle["artifact_paths"]) if artifact_bundle["artifact_paths"] else []
            if not artifact_paths and entry.result_status == "passed":
                artifact_paths = [str(path.resolve()) for path in required_routing_layer_paths(row.work_dir) if path.exists()]
            snapshots.append(
                {
                    "index": row.index,
                    "work_dir": row.work_dir,
                    "session_name": row.session_name,
                    "status": row.status,
                    "workflow_stage": row.workflow_stage,
                    "agent_state": row.agent_state,
                    "health_status": row.health_status,
                    "current_task_runtime_status": str(getattr(entry, "current_task_runtime_status", "") or "").strip(),
                    "retry_count": row.retry_count,
                    "session_exists": row.session_exists,
                    "note": row.note,
                    "forced": row.forced,
                    "transcript_path": entry.transcript_path,
                    "turn_status_path": entry.current_turn_status_path,
                    "question_path": str(artifact_bundle.get("question_path", "")).strip(),
                    "answer_path": str(artifact_bundle.get("answer_path", "")).strip(),
                    "artifact_paths": artifact_paths,
                    "last_heartbeat_at": str(getattr(entry, "last_heartbeat_at", "") or "").strip(),
                }
            )
        return snapshots

    def render_status(self) -> str:
        lines = [
            f"run_id: {self.run_id}",
            f"runtime_dir: {self.run_root}",
            f"vendor: {self.config.vendor.value}",
            f"model: {self.config.model}",
            f"reasoning_effort: {self.config.reasoning_effort}",
            f"proxy_url: {self.config.proxy_url or '(none)'}",
            "sessions:",
        ]
        for row in self.build_status_rows():
            lines.append(
                f"  {row.index}. {row.status} | {row.session_name} | {row.work_dir} | "
                f"stage={row.workflow_stage} | state={row.agent_state} | health={row.health_status} | "
                f"retry={row.retry_count} | session_exists={row.session_exists} | note={row.note}"
            )
        if self.selection.skipped_dirs:
            lines.append("skipped:")
            for path in self.selection.skipped_dirs:
                lines.append(f"  - {path}")
        return "\n".join(lines)

    def get_handle(self, argument: str) -> LiveWorkerHandle:
        target = self.get_target(argument)
        if target.live_handle is None:
            raise ValueError("该目录没有可 attach 的 tmux 会话")
        return target.live_handle

    def attach(self, argument: str) -> None:
        target = self.get_target(argument)
        if not target.session_name:
            raise RuntimeError("tmux 会话尚未创建")
        self.tmux_runtime.attach_session(target.session_name)

    def show_transcript(self, argument: str) -> str:
        target = self.get_target(argument)
        if not target.transcript_path:
            raise RuntimeError("transcript 路径不存在")
        if target.live_handle is not None:
            return target.live_handle.worker.show_transcript_tail(max_lines=60)
        return self.tmux_runtime.read_transcript_tail(target.transcript_path, max_lines=60)

    def detach(self, argument: str) -> str:
        target = self.get_target(argument)
        if not target.session_name:
            raise RuntimeError("tmux 会话尚未创建")
        return self.tmux_runtime.detach_session(target.session_name)

    def restart_worker(self, argument: str) -> str:
        target = self.get_target(argument)
        if target.live_handle is None:
            raise RuntimeError("该目录当前没有 live worker，不能 restart；请使用 retry。")
        entry = self._entry_by_dir(target.work_dir)
        session_name = target.live_handle.worker.request_restart()
        self.run_store.update_worker_binding(target.work_dir, recoverable=True, health_note="manual_restart_requested")
        self.run_store.append_event("manual_restart", work_dir=target.work_dir,
                                    session_name=session_name or entry.session_name)
        return session_name

    def kill_worker(self, argument: str) -> str:
        target = self.get_target(argument)
        entry = self._entry_by_dir(target.work_dir)
        if target.live_handle is not None:
            session_name = target.live_handle.worker.request_kill()
        else:
            session_name = self.tmux_runtime.kill_session(target.session_name, missing_ok=True)
        self.run_store.update_worker_binding(target.work_dir, recoverable=False, health_note="manual_kill_requested")
        self.run_store.append_event("manual_kill", work_dir=target.work_dir,
                                    session_name=session_name or entry.session_name)
        return session_name

    def retry_worker(self, argument: str) -> str:
        work_dir = self._resolve_work_dir(argument)
        entry = self._entry_by_dir(work_dir)
        if entry.result_status not in {"failed", "stale_failed"}:
            raise RuntimeError("只有 failed/stale_failed 的目录才允许 retry")
        if entry.session_name:
            self.tmux_runtime.kill_session(entry.session_name, missing_ok=True)
        worker = self.worker_factory(
            worker_id=Path(work_dir).name or "project",
            work_dir=work_dir,
            config=self.config,
            runtime_root=self.run_root,
        )
        handle = LiveWorkerHandle(
            work_dir=work_dir,
            forced=entry.forced,
            worker=worker,
            resume_state={},
        )
        self.handle_by_dir[work_dir] = handle
        self.live_workers = [item for item in self.live_workers if item.work_dir != work_dir] + [handle]
        self.results_by_dir.pop(work_dir, None)
        metadata = worker.runtime_metadata()
        self.run_store.update_worker_binding(
            work_dir,
            session_name=metadata["session_name"],
            runtime_dir=metadata["runtime_dir"],
            pane_id=metadata["pane_id"],
            workflow_stage="pending",
            workflow_round=0,
            result_status="pending",
            agent_state="DEAD",
            retry_count=entry.retry_count + 1,
            recoverable=True,
            health_status="unknown",
            health_note="manual_retry",
            current_turn_id="",
            current_turn_phase="",
            current_turn_status_path="",
            note="manual_retry",
            log_path=metadata["log_path"],
            raw_log_path=metadata["raw_log_path"],
            state_path=metadata["state_path"],
            transcript_path=metadata["transcript_path"],
        )
        self.run_store.append_event("manual_retry", work_dir=work_dir, session_name=metadata["session_name"])
        self._submit_handle(handle)
        return metadata["session_name"]

    def wait_until_complete(self) -> BatchInitResult:
        self.start()
        while self.futures:
            self._harvest_done_futures()
            if not self.futures:
                break
            time.sleep(0.5)
        return build_batch_result(
            run_store=self.run_store,
            selection=self.selection,
            config=self.config,
            results_by_dir=self.results_by_dir,
        )

    def can_switch_runs(self) -> bool:
        return self.all_done()

    def close(self) -> None:
        if not self._supervisor_stopped:
            self.supervisor.stop()
            self._supervisor_stopped = True
        self.executor.shutdown(wait=False, cancel_futures=False)

    def cleanup_routing_tmux_sessions(self) -> list[str]:
        return kill_run_tmux_sessions(
            run_store=self.run_store,
            runtime_controller=self.tmux_runtime,
        )

    def transition_to_requirements_phase(self, batch_result: BatchInitResult) -> str:
        if any(item.status == "failed" for item in batch_result.results):
            return "\n".join(
                [
                    "路由层初始化存在失败目录，当前不进入需求录入阶段。",
                    "请继续使用 attach / transcript / retry 排查后再推进。",
                ]
            )

        if self._requirements_placeholder_entered:
            return "\n".join(
                [
                    "路由层配置完成",
                    "进入需求录入阶段（占位）",
                    "下一步请运行: python3 A02_RequirementIntake.py",
                ]
            )

        killed_sessions = self.cleanup_routing_tmux_sessions()
        if not self._supervisor_stopped:
            self.supervisor.stop()
            self._supervisor_stopped = True
        self._requirements_placeholder_entered = True
        self.run_store.append_event(
            "requirements_stage_placeholder_entered",
            run_id=batch_result.run_id,
            killed_session_count=len(killed_sessions),
        )
        cleanup_result = cleanup_routing_stage_artifacts(batch_result=batch_result)
        lines = ["路由层配置完成"]
        if killed_sessions:
            lines.append(f"已清理路由层 tmux 会话: {len(killed_sessions)}")
        else:
            lines.append("路由层 tmux 会话已清理")
        if cleanup_result.removed_intermediate_count:
            lines.append(f"已清理阶段中间文件: {cleanup_result.removed_intermediate_count}")
        if cleanup_result.removed_runtime_count:
            lines.append(f"已清理阶段运行目录: {cleanup_result.removed_runtime_count}")
        lines.append("进入需求录入阶段（占位）")
        lines.append("下一步请运行: python3 A02_RequirementIntake.py")
        return "\n".join(lines)


def run_terminal_control_loop(control_center: AgentInitControlCenter) -> BatchInitResult:
    control_center.start()
    message(render_control_help())
    message()
    message(control_center.render_status())

    final_result: BatchInitResult | None = None

    def announce_stage_transition() -> None:
        nonlocal final_result
        if final_result is None:
            return
        message()
        message(control_center.transition_to_requirements_phase(final_result))

    while True:
        if control_center.all_done() and final_result is None:
            final_result = control_center.wait_until_complete()
            message("\n全部目录执行完成。\n")
            message(render_control_help())
            message()
            message(control_center.render_status())
            message()
            from A01_Routing_LayerPlanning import format_batch_summary

            message(format_batch_summary(final_result))
            announce_stage_transition()

        command = parse_control_command(prompt_command_line("B01>", ""))
        if command.action == "help":
            message(render_control_help())
            continue
        if command.action == "status":
            message(control_center.render_status())
            continue
        if command.action == "attach":
            try:
                control_center.attach(command.argument)
            except Exception as error:
                message(f"attach 失败: {error}")
            continue
        if command.action == "transcript":
            try:
                message(control_center.show_transcript(command.argument))
            except Exception as error:
                message(f"读取 transcript 失败: {error}")
            continue
        if command.action == "detach":
            try:
                session_name = control_center.detach(command.argument)
                message(f"已请求 detach: {session_name}")
            except Exception as error:
                message(f"detach 失败: {error}")
            continue
        if command.action == "restart":
            try:
                session_name = control_center.restart_worker(command.argument)
                final_result = None
                message(f"已请求 restart: {session_name}")
            except Exception as error:
                message(f"restart 失败: {error}")
            continue
        if command.action == "kill":
            try:
                session_name = control_center.kill_worker(command.argument)
                message(f"已请求 kill: {session_name}")
            except Exception as error:
                message(f"kill 失败: {error}")
            continue
        if command.action == "retry":
            try:
                session_name = control_center.retry_worker(command.argument)
                final_result = None
                message(f"已重新提交 worker: {session_name}")
            except Exception as error:
                message(f"retry 失败: {error}")
            continue
        if command.action == "wait":
            if final_result is None:
                try:
                    final_result = control_center.wait_until_complete()
                    from A01_Routing_LayerPlanning import format_batch_summary

                    message(format_batch_summary(final_result))
                    announce_stage_transition()
                except Exception as error:
                    message(error)
            else:
                message("全部目录已完成。")
            continue
        if command.action == "resume":
            if not control_center.can_switch_runs():
                message("当前 run 尚未完成，不能切换到其他 run。")
                continue
            if not command.argument:
                message("请提供 run_id。")
                continue
            project_dir = str(getattr(getattr(control_center, "selection", None), "project_dir", "") or "").strip()
            if not project_dir:
                message("resume 失败: 当前只支持恢复当前项目内的 routing run，缺少 project_dir。")
                continue
            try:
                next_center = AgentInitControlCenter.from_existing_run(
                    run_id=command.argument,
                    project_dir=project_dir,
                    max_refine_rounds=control_center.max_refine_rounds,
                )
            except Exception as error:
                message(f"resume 失败: {error}")
                continue
            control_center.close()
            control_center = next_center
            final_result = None
            control_center.start()
            message(f"已切换到 run: {control_center.run_id}")
            message(control_center.render_status())
            continue
        if command.action == "exit":
            if final_result is None:
                message("任务仍在运行。当前实现不支持后台脱离，请先使用 wait，或 attach 到某个会话观察执行。")
                continue
            control_center.close()
            return final_result
        message(f"未知命令: {command.action}")


def main(argv: Sequence[str] | None = None) -> int:
    redirected, launch = maybe_launch_tui(argv, route="control", action="control.b01.open")
    if redirected:
        return int(launch)
    parser = build_parser()
    args = parser.parse_args(list(launch))
    if getattr(args, "resume_run", ""):
        project_dir = (
            str(resolve_existing_directory(args.project_dir))
            if args.project_dir
            else prompt_project_dir("")
        )
        control_center = AgentInitControlCenter.from_existing_run(
            run_id=args.resume_run,
            project_dir=project_dir,
            max_refine_rounds=max(int(args.max_refine_rounds or 3), 1),
        )
        batch_result = run_terminal_control_loop(control_center)
        return determine_exit_code(batch_result)

    request = collect_b01_request(args)
    config = AgentRunConfig(
        vendor=request.vendor,
        model=request.model,
        reasoning_effort=request.reasoning_effort,
        proxy_url=request.proxy_port,
    )
    selection = resolve_target_selection(
        project_dir=request.project_dir,
        target_dirs=request.target_dirs,
        run_init=request.run_init,
    )

    if not selection.should_run:
        from A01_Routing_LayerPlanning import render_requirements_stage_placeholder

        if selection.project_missing_files and not project_has_business_files(selection.project_dir):
            message(EMPTY_PROJECT_ROUTING_SKIP_MESSAGE)
        else:
            message("当前项目路由层已完备，跳过路由初始化。")
        message(render_requirements_stage_placeholder([]))
        return 0

    preflight_summary = render_preflight_summary(request, config, selection)
    force_confirmation = bool(selection.project_missing_files)
    if not request.auto_confirm and not prompt_confirmation(preflight_summary, force_yes=force_confirmation):
        message("已取消执行。")
        return 0

    control_center = AgentInitControlCenter.create_new(
        selection=selection,
        config=config,
        max_refine_rounds=request.max_refine_rounds,
    )
    batch_result = run_terminal_control_loop(control_center)
    return determine_exit_code(batch_result)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        cleaned_sessions = cleanup_registered_tmux_workers(reason="keyboard_interrupt")
        if cleaned_sessions:
            message(f"\n已清理 tmux 会话: {', '.join(cleaned_sessions)}")
        raise SystemExit(130)
