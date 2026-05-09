from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

from tmux_core.runtime.tmux_runtime import DEFAULT_COMMAND_TIMEOUT_SEC, worker_state_is_prelaunch_active
from tmux_core.stage_kernel.agent_intervention import render_worker_intervention_summary

READY_STABILIZATION_GRACE_SEC = 10.0
READY_STARTUP_STABILIZATION_GRACE_SEC = 60.0
READY_STABILIZATION_POLL_SEC = 0.25

TMain = TypeVar("TMain")
TReviewer = TypeVar("TReviewer")
TResult = TypeVar("TResult")


@dataclass
class WorkerReadyCheckFailed(RuntimeError):
    role_label: str
    worker: object | None
    state_name: str
    reason_text: str
    stage_label: str = "阶段调度"

    def __post_init__(self) -> None:
        RuntimeError.__init__(
            self,
            render_worker_intervention_summary(
                stage_label=self.stage_label,
                role_label=self.role_label,
                worker=self.worker,
                reason_text=self.reason_text,
            ),
        )


def _resolve_worker(owner: object | None):
    if owner is None:
        return None
    worker = getattr(owner, "worker", None)
    return worker if worker is not None else owner


def _state_name(worker: object) -> str:
    refresh_health = getattr(worker, "refresh_health", None)
    if callable(refresh_health):
        try:
            snapshot = refresh_health(notify_on_change=False)
            state_name = str(getattr(snapshot, "agent_state", "") or "").strip().upper()
            if state_name:
                return state_name
        except Exception:
            pass
    observe = getattr(worker, "observe", None)
    get_state = getattr(worker, "get_agent_state", None)
    if callable(observe) and callable(get_state):
        try:
            observation = observe(tail_lines=120)
            state = get_state(observation)
            state_name = str(getattr(state, "value", state) or "").strip().upper()
            if state_name:
                return state_name
        except Exception:
            pass
    if not callable(get_state):
        return ""
    state = get_state()
    return str(getattr(state, "value", state) or "").strip().upper()


def _read_worker_state(worker: object) -> dict[str, object]:
    read_state = getattr(worker, "read_state", None)
    if callable(read_state):
        try:
            state = read_state()
            return state if isinstance(state, dict) else {}
        except Exception:
            return {}
    return {}


def _current_turn_completed(worker: object) -> bool:
    state = _read_worker_state(worker)
    status = str(state.get("status", getattr(worker, "status", "")) or "").strip().lower()
    result_status = str(state.get("result_status", getattr(worker, "result_status", "")) or "").strip().lower()
    runtime_status = str(
        state.get("current_task_runtime_status", getattr(worker, "current_task_runtime_status", "")) or ""
    ).strip().lower()
    return status == "succeeded" and result_status == "succeeded" and runtime_status == "done"


def _current_task_running(worker: object) -> bool:
    state = _read_worker_state(worker)
    runtime_status = str(
        state.get("current_task_runtime_status", getattr(worker, "current_task_runtime_status", "")) or ""
    ).strip().lower()
    return runtime_status == "running"


def _worker_failed_or_stale(worker: object) -> bool:
    state = _read_worker_state(worker)
    if worker_state_is_prelaunch_active(state or worker):
        return False
    status = str(state.get("status", getattr(worker, "status", "")) or "").strip().lower()
    result_status = str(state.get("result_status", getattr(worker, "result_status", "")) or "").strip().lower()
    health_status = str(state.get("health_status", getattr(worker, "health_status", "")) or "").strip().lower()
    return status in {"failed", "stale_failed", "error"} or result_status in {"failed", "error"} or health_status in {
        "provider_runtime_error",
        "dead",
        "missing_session",
    }


def _failed_worker_is_still_active(worker: object, state_name: str) -> bool:
    if str(state_name or "").strip().upper() not in {"BUSY", "STARTING"}:
        return False
    state = _read_worker_state(worker)
    health_status = str(state.get("health_status", getattr(worker, "health_status", "")) or "").strip().lower()
    if health_status in {"dead", "missing_session", "pane_dead", "provider_runtime_error"}:
        return False
    if hasattr(worker, "terminal_recently_changed"):
        return bool(getattr(worker, "terminal_recently_changed", False))
    return bool(state.get("terminal_recently_changed", False))


def _state_allows_ready_stabilization(worker: object, state_name: str) -> bool:
    if str(state_name or "").strip().upper() not in {"BUSY", "STARTING"}:
        return False
    state = _read_worker_state(worker)
    health_status = str(state.get("health_status", getattr(worker, "health_status", "")) or "").strip().lower()
    return health_status not in {"dead", "missing_session", "pane_dead", "provider_runtime_error"}


def _ready_stabilization_grace_sec(worker: object, state_name: str) -> float:
    normalized = str(state_name or "").strip().upper()
    if normalized in {"BUSY", "STARTING"} and not _current_task_running(worker):
        return max(READY_STABILIZATION_GRACE_SEC, READY_STARTUP_STABILIZATION_GRACE_SEC)
    return READY_STABILIZATION_GRACE_SEC


def _wait_for_ready_stabilization(
    worker: object,
    *,
    grace_sec: float = READY_STABILIZATION_GRACE_SEC,
    poll_sec: float = READY_STABILIZATION_POLL_SEC,
) -> str:
    deadline = time.monotonic() + max(0.0, grace_sec)
    state_name = _state_name(worker)
    while state_name != "READY" and _state_allows_ready_stabilization(worker, state_name):
        if time.monotonic() >= deadline:
            break
        time.sleep(max(0.0, poll_sec))
        state_name = _state_name(worker)
    return state_name


def _restart_failed_worker_before_ready_wait(worker: object, *, timeout_sec: float) -> bool:
    _ = timeout_sec
    return False


def _ensure_worker_ready(
    worker: object | None,
    *,
    role_label: str,
    timeout_sec: float,
    allow_completed_nonready: bool = False,
) -> None:
    if worker is None:
        return
    ensure_ready = getattr(worker, "ensure_agent_ready", None)
    if not callable(ensure_ready):
        return
    if allow_completed_nonready and _current_turn_completed(worker):
        return
    state_name = _state_name(worker)
    if _worker_failed_or_stale(worker) and state_name != "READY" and not _failed_worker_is_still_active(worker, state_name):
        raise WorkerReadyCheckFailed(
            role_label=role_label,
            worker=worker,
            state_name=state_name,
            reason_text="上一轮执行失败或状态异常，系统不会自动重启该智能体。请人工进入 tmux 检查后再继续。",
        )
    if state_name != "READY":
        try:
            ensure_ready(timeout_sec=timeout_sec)
        except Exception as error:  # noqa: BLE001
            raise WorkerReadyCheckFailed(
                role_label=role_label,
                worker=worker,
                state_name=_state_name(worker),
                reason_text=f"等待智能体 READY 时发生异常: {error}",
            ) from error
    if allow_completed_nonready and _current_turn_completed(worker):
        return
    state_name = _state_name(worker)
    if state_name != "READY":
        state_name = _wait_for_ready_stabilization(worker, grace_sec=_ready_stabilization_grace_sec(worker, state_name))
    if state_name != "READY":
        raise WorkerReadyCheckFailed(
            role_label=role_label,
            worker=worker,
            state_name=state_name,
            reason_text=f"{role_label} 未进入 READY 状态（当前状态: {state_name or 'UNKNOWN'}）",
        )


def ensure_main_ready(
    main_owner: object | None,
    reviewers: Sequence[object] = (),
    *,
    main_label: str = "主工作智能体",
    reviewer_label_getter: Callable[[object, int], str] | None = None,
    timeout_sec: float = DEFAULT_COMMAND_TIMEOUT_SEC,
    allow_completed_nonready: bool = False,
) -> None:
    _ensure_worker_ready(
        _resolve_worker(main_owner),
        role_label=main_label,
        timeout_sec=timeout_sec,
        allow_completed_nonready=allow_completed_nonready,
    )
    for index, reviewer in enumerate(reviewers, start=1):
        label = reviewer_label_getter(reviewer, index) if reviewer_label_getter is not None else f"审核智能体 {index}"
        _ensure_worker_ready(
            _resolve_worker(reviewer),
            role_label=label,
            timeout_sec=timeout_sec,
            allow_completed_nonready=allow_completed_nonready,
        )


def ensure_reviewers_ready(
    main_owner: object | None,
    reviewers: Sequence[object],
    *,
    main_label: str = "主工作智能体",
    reviewer_label_getter: Callable[[object, int], str] | None = None,
    timeout_sec: float = DEFAULT_COMMAND_TIMEOUT_SEC,
    allow_completed_nonready: bool = False,
) -> None:
    ensure_main_ready(
        main_owner,
        reviewers,
        main_label=main_label,
        reviewer_label_getter=reviewer_label_getter,
        timeout_sec=timeout_sec,
        allow_completed_nonready=allow_completed_nonready,
    )


def run_main_phase(
    main_owner: TMain,
    *,
    reviewers: Sequence[object] = (),
    run_phase: Callable[[TMain], TResult],
    owner_getter: Callable[[TResult], object] | None = None,
    main_label: str = "主工作智能体",
    reviewer_label_getter: Callable[[object, int], str] | None = None,
    timeout_sec: float = DEFAULT_COMMAND_TIMEOUT_SEC,
) -> TResult:
    ensure_main_ready(
        main_owner,
        reviewers,
        main_label=main_label,
        reviewer_label_getter=reviewer_label_getter,
        timeout_sec=timeout_sec,
    )
    result = run_phase(main_owner)
    updated_owner = owner_getter(result) if owner_getter is not None else result
    ensure_main_ready(
        updated_owner,
        reviewers,
        main_label=main_label,
        reviewer_label_getter=reviewer_label_getter,
        timeout_sec=timeout_sec,
        allow_completed_nonready=True,
    )
    return result


def run_reviewer_phase(
    main_owner: object | None,
    reviewers: Sequence[TReviewer],
    *,
    run_phase: Callable[[Sequence[TReviewer]], list[TReviewer]],
    main_label: str = "主工作智能体",
    reviewer_label_getter: Callable[[object, int], str] | None = None,
    timeout_sec: float = DEFAULT_COMMAND_TIMEOUT_SEC,
) -> list[TReviewer]:
    ensure_reviewers_ready(
        main_owner,
        reviewers,
        main_label=main_label,
        reviewer_label_getter=reviewer_label_getter,
        timeout_sec=timeout_sec,
    )
    updated_reviewers = run_phase(reviewers)
    ensure_reviewers_ready(
        main_owner,
        updated_reviewers,
        main_label=main_label,
        reviewer_label_getter=reviewer_label_getter,
        timeout_sec=timeout_sec,
        allow_completed_nonready=True,
    )
    return updated_reviewers
