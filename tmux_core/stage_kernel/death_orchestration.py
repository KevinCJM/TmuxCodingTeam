from __future__ import annotations

from typing import Callable, Sequence, TypeVar

from tmux_core.runtime.tmux_runtime import (
    DEFAULT_COMMAND_TIMEOUT_SEC,
    is_worker_death_error,
    worker_state_is_prelaunch_active,
)
from tmux_core.stage_kernel.agent_intervention import (
    AGENT_INTERVENTION_RECHECK,
    AGENT_INTERVENTION_RECREATE,
    AGENT_INTERVENTION_WORKER_DEAD,
    request_worker_manual_intervention,
)
from tmux_core.stage_kernel.role_orchestration import WorkerReadyCheckFailed, ensure_main_ready

TMain = TypeVar("TMain")
TReviewer = TypeVar("TReviewer")
TResult = TypeVar("TResult")


def _resolve_worker(owner: object | None):
    if owner is None:
        return None
    worker = getattr(owner, "worker", None)
    return worker if worker is not None else owner


def _state_name(worker: object | None) -> str:
    if worker is None:
        return ""
    get_state = getattr(worker, "get_agent_state", None)
    if not callable(get_state):
        return ""
    state = get_state()
    return str(getattr(state, "value", state) or "").strip().upper()


def _is_dead(owner: object | None) -> bool:
    worker = _resolve_worker(owner)
    if worker_state_is_prelaunch_active(worker):
        return False
    return _has_ever_launched(worker) and _state_name(worker) == "DEAD"


def _has_ever_launched(worker: object | None) -> bool:
    if worker is None:
        return False
    has_ever_launched = getattr(worker, "has_ever_launched", None)
    if callable(has_ever_launched):
        try:
            return bool(has_ever_launched())
        except Exception:
            return False
    if bool(getattr(worker, "agent_started", False)):
        return True
    if str(getattr(worker, "pane_id", "") or "").strip():
        return True
    return False


def _read_worker_state(worker: object | None) -> dict[str, object]:
    if worker is None:
        return {}
    read_state = getattr(worker, "read_state", None)
    if not callable(read_state):
        return {}
    try:
        state = read_state()
    except Exception:
        return {}
    return state if isinstance(state, dict) else {}


def _reviewer_ready_check_should_be_skipped(reviewer: object | None) -> bool:
    worker = _resolve_worker(reviewer)
    state = _read_worker_state(worker)
    if worker_state_is_prelaunch_active(state or worker):
        return False
    status = str(state.get("status", getattr(worker, "status", "")) or "").strip().lower()
    result_status = str(state.get("result_status", getattr(worker, "result_status", "")) or "").strip().lower()
    health_status = str(state.get("health_status", getattr(worker, "health_status", "")) or "").strip().lower()
    return status in {"failed", "stale_failed", "error"} or result_status in {"failed", "error"} or health_status in {
        "dead",
        "missing_session",
        "pane_dead",
        "provider_runtime_error",
    }


def _filter_reviewers_for_final_ready_check(
    reviewers: Sequence[TReviewer],
    *,
    reviewer_label_getter: Callable[[TReviewer, int], str] | None = None,
    notify: Callable[[str], None] | None = None,
) -> list[TReviewer]:
    selected: list[TReviewer] = []
    for index, reviewer in enumerate(reviewers, start=1):
        if not _reviewer_ready_check_should_be_skipped(reviewer):
            selected.append(reviewer)
            continue
        if notify is not None:
            label = reviewer_label_getter(reviewer, index) if reviewer_label_getter is not None else f"审核智能体 {index}"
            notify(f"{label} 当前处于失败/异常状态，阶段末尾将忽略该审核智能体。")
    return selected


def drop_dead_reviewers(
    reviewers: Sequence[TReviewer],
    *,
    replace_reviewer: Callable[[TReviewer, int], TReviewer | None] | None = None,
    reviewer_label_getter: Callable[[TReviewer, int], str] | None = None,
    notify: Callable[[str], None] | None = None,
) -> list[TReviewer]:
    survivors: list[TReviewer] = []
    for index, reviewer in enumerate(reviewers, start=1):
        if not _is_dead(reviewer):
            survivors.append(reviewer)
            continue
        if replace_reviewer is not None:
            replacement = replace_reviewer(reviewer, index)
            if replacement is not None:
                survivors.append(replacement)
                if notify is not None:
                    label = reviewer_label_getter(reviewer, index) if reviewer_label_getter is not None else f"审核智能体 {index}"
                    notify(f"{label} 已死亡，已重建审核智能体继续当前阶段。")
                continue
        if notify is not None:
            label = reviewer_label_getter(reviewer, index) if reviewer_label_getter is not None else f"审核智能体 {index}"
            notify(f"{label} 已死亡，后续将忽略该审核智能体。")
    return survivors


def replace_dead_main(
    main_owner: TMain,
    *,
    replace_owner: Callable[[TMain], TMain],
) -> TMain:
    if not _is_dead(main_owner):
        return main_owner
    return replace_owner(main_owner)


def _main_ready_error_requires_replacement(
    error: Exception,
    main_owner: object | None,
    *,
    reviewers: Sequence[object] = (),
) -> bool:
    if _is_dead(main_owner):
        return True
    message = str(error or "").strip()
    if not message:
        return False
    lowered = message.lower()
    if not (
        is_worker_death_error(error)
        or "需要重新启动或重建" in message
        or "tmux pane missing" in lowered
    ):
        return False
    worker = _resolve_worker(main_owner)
    session_name = str(getattr(worker, "session_name", "") or "").strip()
    if not session_name:
        return not bool(reviewers)
    return session_name in message


def _ready_error_reason(error: Exception) -> str:
    if isinstance(error, WorkerReadyCheckFailed):
        return str(error.reason_text or "").strip() or str(error)
    return str(error or "").strip()


def _build_dead_ready_error(owner: object | None, *, role_label: str) -> WorkerReadyCheckFailed:
    worker = _resolve_worker(owner)
    return WorkerReadyCheckFailed(
        role_label=role_label,
        worker=worker,
        state_name="DEAD",
        reason_text=f"{role_label} 已死亡或已关闭，需要人工决定恢复方式。",
    )


def _request_ready_intervention(
    *,
    role_label: str,
    worker: object | None,
    error: Exception,
    allow_recreate: bool,
    allow_worker_dead: bool,
) -> str:
    return request_worker_manual_intervention(
        stage_label="阶段调度",
        role_label=role_label,
        worker=worker,
        reason_text=_ready_error_reason(error),
        allow_recreate=allow_recreate,
        allow_worker_dead=allow_worker_dead,
    )


def _ensure_main_ready_with_replacement(
    current_main: TMain,
    current_reviewers: Sequence[TReviewer],
    *,
    replace_dead_main_owner: Callable[[TMain], TMain],
    main_label: str,
    reviewer_label_getter: Callable[[TReviewer, int], str] | None,
    timeout_sec: float,
    allow_completed_nonready: bool = False,
) -> TMain:
    # Main-phase recovery only owns the main worker. Reviewers are filtered by
    # drop_dead_reviewers and then handled by reviewer-phase wrappers.
    while True:
        error: Exception | None = None
        if _is_dead(current_main):
            error = _build_dead_ready_error(current_main, role_label=main_label)
        else:
            try:
                ensure_main_ready(
                    current_main,
                    (),
                    main_label=main_label,
                    timeout_sec=timeout_sec,
                    allow_completed_nonready=allow_completed_nonready,
                )
                return current_main
            except RuntimeError as caught:
                if not isinstance(caught, WorkerReadyCheckFailed) and not _main_ready_error_requires_replacement(
                    caught,
                    current_main,
                    reviewers=current_reviewers,
                ):
                    raise
                error = caught
        decision = _request_ready_intervention(
            role_label=main_label,
            worker=_resolve_worker(current_main),
            error=error,
            allow_recreate=True,
            allow_worker_dead=False,
        )
        if decision == AGENT_INTERVENTION_RECREATE:
            current_main = replace_dead_main_owner(current_main)
            continue
        if decision == AGENT_INTERVENTION_RECHECK:
            continue


def run_main_phase_with_death_handling(
    main_owner: TMain,
    *,
    reviewers: Sequence[TReviewer] = (),
    run_phase: Callable[[TMain], TResult],
    replace_dead_main_owner: Callable[[TMain], TMain],
    owner_getter: Callable[[TResult], TMain] | None = None,
    main_label: str = "主工作智能体",
    reviewer_label_getter: Callable[[TReviewer, int], str] | None = None,
    replace_dead_reviewer: Callable[[TReviewer, int], TReviewer | None] | None = None,
    notify: Callable[[str], None] | None = None,
    timeout_sec: float = DEFAULT_COMMAND_TIMEOUT_SEC,
) -> tuple[TResult, list[TReviewer], TMain]:
    current_main = main_owner
    current_reviewers = drop_dead_reviewers(
        reviewers,
        replace_reviewer=replace_dead_reviewer,
        reviewer_label_getter=reviewer_label_getter,
        notify=notify,
    )
    current_main = _ensure_main_ready_with_replacement(
        current_main,
        current_reviewers,
        replace_dead_main_owner=replace_dead_main_owner,
        main_label=main_label,
        reviewer_label_getter=reviewer_label_getter,
        timeout_sec=timeout_sec,
    )
    result = run_phase(current_main)
    updated_main = owner_getter(result) if owner_getter is not None else result
    current_reviewers = drop_dead_reviewers(
        current_reviewers,
        replace_reviewer=replace_dead_reviewer,
        reviewer_label_getter=reviewer_label_getter,
        notify=notify,
    )
    updated_main = _ensure_main_ready_with_replacement(
        updated_main,
        current_reviewers,
        replace_dead_main_owner=replace_dead_main_owner,
        main_label=main_label,
        reviewer_label_getter=reviewer_label_getter,
        timeout_sec=timeout_sec,
        allow_completed_nonready=True,
    )
    return result, current_reviewers, updated_main


def run_reviewer_phase_with_death_handling(
    main_owner: TMain,
    reviewers: Sequence[TReviewer],
    *,
    run_phase: Callable[[Sequence[TReviewer]], list[TReviewer]],
    replace_dead_main_owner: Callable[[TMain], TMain],
    main_label: str = "主工作智能体",
    reviewer_label_getter: Callable[[TReviewer, int], str] | None = None,
    replace_dead_reviewer: Callable[[TReviewer, int], TReviewer | None] | None = None,
    notify: Callable[[str], None] | None = None,
    timeout_sec: float = DEFAULT_COMMAND_TIMEOUT_SEC,
) -> tuple[list[TReviewer], TMain]:
    current_main = main_owner
    current_reviewers = drop_dead_reviewers(
        reviewers,
        replace_reviewer=replace_dead_reviewer,
        reviewer_label_getter=reviewer_label_getter,
        notify=notify,
    )
    # Reviewer startup is intentionally lazy here: stage-specific reviewer turn
    # wrappers know how to recreate or drop a reviewer after provider/auth/ready
    # failures, while this shared layer only guarantees the main owner is ready.
    current_main = _ensure_main_ready_with_replacement(
        current_main,
        (),
        replace_dead_main_owner=replace_dead_main_owner,
        main_label=main_label,
        reviewer_label_getter=reviewer_label_getter,
        timeout_sec=timeout_sec,
    )
    updated_reviewers = run_phase(current_reviewers)
    updated_reviewers = drop_dead_reviewers(
        updated_reviewers,
        replace_reviewer=replace_dead_reviewer,
        reviewer_label_getter=reviewer_label_getter,
        notify=notify,
    )
    current_main = _ensure_main_ready_with_replacement(
        current_main,
        updated_reviewers,
        replace_dead_main_owner=replace_dead_main_owner,
        main_label=main_label,
        reviewer_label_getter=reviewer_label_getter,
        timeout_sec=timeout_sec,
        allow_completed_nonready=True,
    )
    ready_reviewers: list[TReviewer] = []
    for index, reviewer in enumerate(updated_reviewers, start=1):
        label = reviewer_label_getter(reviewer, index) if reviewer_label_getter is not None else f"审核智能体 {index}"
        current_reviewer = reviewer
        while True:
            error: Exception | None = None
            if _is_dead(current_reviewer):
                error = _build_dead_ready_error(current_reviewer, role_label=label)
            else:
                try:
                    ensure_main_ready(
                        current_reviewer,
                        (),
                        main_label=label,
                        timeout_sec=timeout_sec,
                        allow_completed_nonready=True,
                    )
                    ready_reviewers.append(current_reviewer)
                    break
                except RuntimeError as caught:
                    if not isinstance(caught, WorkerReadyCheckFailed):
                        raise
                    error = caught
            decision = _request_ready_intervention(
                role_label=label,
                worker=_resolve_worker(current_reviewer),
                error=error,
                allow_recreate=replace_dead_reviewer is not None,
                allow_worker_dead=True,
            )
            if decision == AGENT_INTERVENTION_WORKER_DEAD:
                if notify is not None:
                    notify(f"{label} 已按死亡处理，后续将忽略该审核智能体。")
                break
            if decision == AGENT_INTERVENTION_RECREATE and replace_dead_reviewer is not None:
                replacement = replace_dead_reviewer(current_reviewer, index)
                if replacement is None:
                    if notify is not None:
                        notify(f"{label} 重新创建失败，后续将忽略该审核智能体。")
                    break
                current_reviewer = replacement
                continue
            if decision == AGENT_INTERVENTION_RECHECK:
                continue
    return ready_reviewers, current_main


def ensure_active_reviewers(reviewers: Sequence[object], *, stage_label: str) -> None:
    if reviewers:
        return
    raise RuntimeError(f"{stage_label} 的审核智能体已全部死亡，无法继续当前阶段。")
