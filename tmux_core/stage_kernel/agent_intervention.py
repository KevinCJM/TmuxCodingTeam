from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Mapping, Sequence

from T09_terminal_ops import message, prompt_select_option

AGENT_INTERVENTION_RECHECK = "recheck_after_manual_intervention"
AGENT_INTERVENTION_RECREATE = "recreate_after_manual_intervention"
AGENT_INTERVENTION_WORKER_DEAD = "worker_dead_after_manual_intervention"


def _session_name(worker: object | None) -> str:
    return str(getattr(worker, "session_name", "") or "").strip()


def _worker_state(worker: object | None) -> str:
    if worker is None:
        return "unknown"
    read_state = getattr(worker, "read_state", None)
    if callable(read_state):
        try:
            state = read_state()
        except Exception:
            state = {}
        if isinstance(state, Mapping):
            agent_state = str(state.get("agent_state", "") or "").strip()
            status = str(state.get("status", "") or "").strip()
            health_status = str(state.get("health_status", "") or "").strip()
            summary = "/".join(item for item in (status, agent_state, health_status) if item)
            if summary:
                return summary
    get_agent_state = getattr(worker, "get_agent_state", None)
    if callable(get_agent_state):
        try:
            state = get_agent_state()
            return str(getattr(state, "value", state) or "").strip() or "unknown"
        except Exception:
            return "unknown"
    return "unknown"


def _attach_command(worker: object | None) -> str:
    session = _session_name(worker)
    return f"tmux attach -t {session}" if session else ""


def _mark_awaiting_manual(worker: object | None, *, reason_text: str) -> None:
    marker = getattr(worker, "mark_awaiting_reconfiguration", None)
    if not callable(marker):
        return
    try:
        marker(reason_text=reason_text)
    except Exception:
        return


def render_worker_intervention_summary(
    *,
    stage_label: str,
    role_label: str,
    worker: object | None,
    reason_text: str,
    target_paths: Sequence[str | Path] = (),
) -> str:
    lines = [
        f"{stage_label or '当前阶段'} 需要人工介入",
        f"角色: {role_label or '智能体'}",
        f"状态: {_worker_state(worker)}",
    ]
    session = _session_name(worker)
    if session:
        lines.append(f"会话: {session}")
    attach = _attach_command(worker)
    if attach:
        lines.append(f"进入会话: {attach}")
    reason = str(reason_text or "").strip()
    if reason:
        lines.append(f"原因: {reason}")
    paths = [str(Path(item).expanduser().resolve()) for item in target_paths if str(item or "").strip()]
    if paths:
        lines.append("需要检查的文件:")
        lines.extend(f"- {item}" for item in paths)
    return "\n".join(lines)


def request_worker_manual_intervention(
    *,
    stage_label: str,
    role_label: str,
    worker: object | None,
    reason_text: str,
    target_paths: Sequence[str | Path] = (),
    progress: object | None = None,
    allow_recreate: bool = False,
    allow_worker_dead: bool = True,
) -> str:
    role_text = str(role_label or "").strip() or "智能体"
    stage_text = str(stage_label or "").strip() or "当前阶段"
    summary = render_worker_intervention_summary(
        stage_label=stage_text,
        role_label=role_text,
        worker=worker,
        reason_text=reason_text,
        target_paths=target_paths,
    )
    _mark_awaiting_manual(worker, reason_text=summary)
    message(summary)
    set_phase = getattr(progress, "set_phase", None)
    if callable(set_phase):
        set_phase(f"{stage_text} / 等待人工介入 | {role_text}")
    suspended = getattr(progress, "suspended", None)
    context = suspended() if callable(suspended) else nullcontext()
    options: list[tuple[str, str]] = [
        (AGENT_INTERVENTION_RECHECK, "我已进入 tmux/修正文件，重新检查"),
    ]
    if allow_recreate:
        options.append((AGENT_INTERVENTION_RECREATE, "重新创建该智能体"))
    if allow_worker_dead:
        options.append((AGENT_INTERVENTION_WORKER_DEAD, "智能体已死亡或已关闭，按死亡处理"))
    with context:
        return prompt_select_option(
            title=f"HITL: {role_text} 需要人工介入",
            options=tuple(options),
            default_value=AGENT_INTERVENTION_RECHECK,
            prompt_text="请选择恢复方式",
            is_hitl=True,
            extra_payload={
                "recovery_kind": "agent_manual_intervention",
                "stage_label": stage_text,
                "role_label": role_text,
                "session_name": _session_name(worker),
                "worker_state": _worker_state(worker),
                "attach_command": _attach_command(worker),
                "target_paths": [str(Path(item).expanduser().resolve()) for item in target_paths if str(item or "").strip()],
                "reason_text": str(reason_text or "").strip(),
            },
        )


def request_file_noncompliance_intervention(
    *,
    stage_label: str,
    role_label: str,
    worker: object | None,
    reason_text: str,
    attempts_used: int,
    target_paths: Sequence[str | Path] = (),
    progress: object | None = None,
    allow_recreate: bool = False,
) -> str:
    reason = (
        f"指定文件连续 {attempts_used} 次修复后仍不符合要求。\n"
        f"{str(reason_text or '').strip()}"
    ).strip()
    return request_worker_manual_intervention(
        stage_label=stage_label,
        role_label=role_label,
        worker=worker,
        reason_text=reason,
        target_paths=target_paths,
        progress=progress,
        allow_recreate=allow_recreate,
    )
