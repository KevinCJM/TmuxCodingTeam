# -*- encoding: utf-8 -*-
"""
@File: T02_tmux_agents.py
@Modify Time: 2026/4/12
@Author: Kevin-Chen
@Descriptions: tmux + 多厂商 coding agent 的长会话运行时
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
import threading
import time
import uuid
import weakref
import contextlib
from datetime import datetime
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence
from contextlib import contextmanager
from urllib.parse import urlparse
from tmux_core.runtime.vendor_catalog import LaunchResolution, resolve_launch
from tmux_core.runtime.contracts import (
    TASK_RESULT_COMPLETED,
    TASK_RESULT_HITL,
    TASK_RESULT_READY,
    TASK_STATUS_DONE,
    TASK_STATUS_RUNNING,
    TaskResultContract,
    TaskResultFile,
    TurnFileContract,
    TurnFileResult,
    build_missing_task_result_finalization_candidate,
    finalize_task_result,
    read_task_result_payload,
    read_task_status,
    validate_task_result_file,
    validate_turn_file_artifact_rules,
    write_task_status,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / ".agent_init_runtime"
DEFAULT_COMMAND_TIMEOUT_SEC = 60 * 20
TURN_START_BUSY_PROBE_TIMEOUT_SEC = 8.0
DEFAULT_PROXY_HOST = "127.0.0.1"
TMUX_HISTORY_LIMIT_LINES = 10000
DEFAULT_CAPTURE_TAIL_LINES = 10000
RESULT_CONTRACT_PRE_SUBMIT_CAPTURE_TAIL_LINES = 240
SESSION_NAME_CREATE_MAX_RETRIES = 8
TERMINAL_ACTIVITY_IDLE_WINDOW_SEC = 1.5
TURN_ARTIFACT_POST_DONE_GRACE_SEC = 10.0
TASK_RESULT_POST_DONE_GRACE_SEC = 10.0
TASK_RESULT_READY_MISSING_GRACE_SEC = 2.0
STALE_BUSY_WITHOUT_CONTRACT_REASON_PREFIX = "stale_busy_without_contract"
TASK_CONTRACT_STALL_IDLE_SEC = 45.0
FILE_CONTRACT_POLL_INTERVAL_SEC = 0.5
ACTIVE_AGENT_PROBE_INTERVAL_SEC = 2.0
POST_DONE_AGENT_PROBE_INTERVAL_SEC = 5.0
READY_HEALTH_INTERVAL_SEC = 15.0
IDLE_HEALTH_INTERVAL_SEC = 30.0
IDLE_HEALTH_AFTER_SEC = 60.0
PRELAUNCH_ACTIVE_RESULT_STATUSES = {"pending", "ready", "running"}
PRELAUNCH_WORKFLOW_STAGES = {"audit_running", "create_running", "pending", "refine_running", "starting"}
TERMINAL_WORKER_RESULT_STATUSES = {
    "completed",
    "done",
    "error",
    "failed",
    "passed",
    "skipped",
    "stale_failed",
    "succeeded",
}
COMPLETED_WORKER_RESULT_STATUSES = {
    "completed",
    "done",
    "passed",
    "skipped",
    "succeeded",
}
WORKER_DEATH_ERROR_MARKERS = (
    "tmux pane died",
    "tmux pane exited",
    "tmux pane missing",
    "agent exited back to shell",
    "missing_session",
    "pane_dead",
)
PROVIDER_AUTH_ERROR_MARKERS = (
    "api error: 401",
    "401 invalid access token",
    "401 unauthorized",
    "invalid access token",
    "token expired",
    "access token expired",
    "token has expired",
)
PROVIDER_RUNTIME_ERROR_MARKERS = (
    "sse read timed out",
    "stream read timed out",
    "stream timed out",
    "read timed out",
    "request timed out",
    "rate limit",
    "quota exceeded",
    "backend unavailable",
    "service unavailable",
    "upstream timeout",
    "gateway timeout",
)
PROMPT_DISPATCH_TIMEOUT_ERROR_MARKERS = (
    "load-buffer",
    "paste-buffer",
    "send-keys",
)
AGENT_READY_TIMEOUT_ERROR_MARKERS = (
    "timed out waiting for agent ready",
)
TURN_ARTIFACT_CONTRACT_ERROR_PREFIX = "turn artifacts contract violation after task completion"
TASK_RESULT_CONTRACT_ERROR_PREFIX = "task result contract violation after task completion"
TIMEOUT_EXIT_CODE = -1
GENERIC_ERROR_EXIT_CODE = 1
RUNTIME_SHUTDOWN_ERROR_MARKER = "tmux runtime shutdown requested"
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
SHELL_COMMANDS = {"bash", "fish", "sh", "zsh"}
BRAILLE_SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
CLAUDE_READY_TITLE = "✳ Claude Code"
CLAUDE_BUSY_TITLE_CHARS = f"{BRAILLE_SPINNER_CHARS}⠐⠂·✶✻✽✢"
_RUNTIME_STATE_NOTIFY_DEBOUNCE_SEC = 0.15
_runtime_state_notify_lock = threading.Lock()
_runtime_state_notify_timer: threading.Timer | None = None
_runtime_state_notify_pending = False
_runtime_state_notify_running = False


def _truthy_runtime_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if not text:
        return False
    if text in {"0", "false", "no", "none", "null", "off"}:
        return False
    return True


def _worker_state_payload(source: Mapping[str, object] | object | None) -> dict[str, object]:
    if source is None:
        return {}
    if isinstance(source, Mapping):
        return dict(source)
    payload: dict[str, object] = {}
    read_state = getattr(source, "read_state", None)
    if callable(read_state):
        try:
            state = read_state()
            if isinstance(state, Mapping):
                payload.update(dict(state))
        except Exception:
            pass
    for field_name in (
        "agent_ready",
        "agent_started",
        "agent_state",
        "current_task_runtime_status",
        "health_status",
        "pane_id",
        "result_status",
        "status",
        "workflow_stage",
    ):
        if field_name not in payload and hasattr(source, field_name):
            payload[field_name] = getattr(source, field_name)
    return payload


def worker_state_has_launch_evidence(source: Mapping[str, object] | object | None) -> bool:
    payload = _worker_state_payload(source)
    if _truthy_runtime_flag(payload.get("agent_started")) or _truthy_runtime_flag(payload.get("agent_ready")):
        return True
    if not str(payload.get("pane_id", "") or "").strip():
        return False
    if _worker_state_is_session_created_prelaunch(payload):
        return False
    return True


def _worker_state_is_session_created_prelaunch(payload: Mapping[str, object]) -> bool:
    if _truthy_runtime_flag(payload.get("agent_started")) or _truthy_runtime_flag(payload.get("agent_ready")):
        return False
    if str(payload.get("note", "") or "").strip().lower() != "session_created":
        return False
    agent_state = str(payload.get("agent_state", "") or "").strip().upper()
    if agent_state not in {"", "STARTING", "DEAD"}:
        return False
    workflow_stage = str(payload.get("workflow_stage", "") or "").strip().lower()
    if workflow_stage and workflow_stage not in PRELAUNCH_WORKFLOW_STAGES:
        return False
    statuses = {
        str(payload.get(field_name, "") or "").strip().lower()
        for field_name in ("status", "result_status")
    }
    statuses.discard("")
    return bool(statuses & PRELAUNCH_ACTIVE_RESULT_STATUSES) and not bool(statuses & TERMINAL_WORKER_RESULT_STATUSES)


def worker_state_is_prelaunch_active(source: Mapping[str, object] | object | None) -> bool:
    payload = _worker_state_payload(source)
    if not payload or worker_state_has_launch_evidence(payload):
        return False
    agent_state = str(payload.get("agent_state", "") or "").strip().upper()
    if agent_state not in {"", "STARTING", "DEAD"}:
        return False
    workflow_stage = str(payload.get("workflow_stage", "") or "").strip().lower()
    note = str(payload.get("note", "") or "").strip().lower()
    health_note = str(payload.get("health_note", "") or "").strip().lower()
    if workflow_stage and workflow_stage not in PRELAUNCH_WORKFLOW_STAGES:
        return False
    if not workflow_stage and (not note or note in {"missing_session", "pane_dead", "tmux session missing"}):
        return False
    if workflow_stage == "pending" and not note and health_note in {"missing_session", "pane_dead", "tmux session missing"}:
        return False
    statuses = {
        str(payload.get(field_name, "") or "").strip().lower()
        for field_name in ("status", "result_status")
    }
    statuses.discard("")
    if not statuses:
        return False
    if statuses & TERMINAL_WORKER_RESULT_STATUSES:
        return False
    return bool(statuses & PRELAUNCH_ACTIVE_RESULT_STATUSES)


def _worker_state_payload_has_completed_status(payload: Mapping[str, object]) -> bool:
    statuses = {
        str(payload.get(field_name, "") or "").strip().lower()
        for field_name in ("status", "result_status")
    }
    runtime_status = str(payload.get("current_task_runtime_status", "") or "").strip().lower()
    return runtime_status == TASK_STATUS_DONE or bool(statuses & COMPLETED_WORKER_RESULT_STATUSES)


BRAILLE_SPINNER_PREFIX_RE = re.compile(rf"^[{BRAILLE_SPINNER_CHARS}]")
CLAUDE_BUSY_TITLE_PREFIX_RE = re.compile(rf"^[{re.escape(CLAUDE_BUSY_TITLE_CHARS)}]\s+")
CLAUDE_READY_PATTERNS = (
    r"^\s*❯(?:[\s\u00a0].*)?$",
)
CLAUDE_BUSY_PATTERNS = (
    r"\bNesting…",
    r"\bNesting\.\.\.",
    r"\bRunning…",
    r"\bRunning\.\.\.",
    r"\bthinking with\b",
    r"\besc to interrupt\b",
    r"⎿\s+Running",
    r"^\s*(?:·|✶|✻|✽|✢|✳)\s+\S.*(?:thinking|tokens|Nesting)",
)
CODEX_TRUST_PROMPT_PATTERNS = (
    r"allow Codex to work in this folder",
    r"Do you trust the contents of this directory\?",
)
CODEX_READY_PATTERNS = (
    r"^\s*(?:❯|›|codex>)\s*$",
    r"^\s*[›❯]\s+\S.*$",
    r"^\s*[›❯]\s+.*@filename.*$",
    r"^\s*[›❯]\s+.*@path/to/file.*$",
)
CODEX_READY_FOOTER_PATTERNS = (
    r"^\s*\S[^\n]*\s+·\s+(?:~|/|\.)",
)
CODEX_MODEL_SELECTION_PROMPT_PATTERNS = (
    r"Introducing GPT-5\.4",
    r"Choose how you'd like Codex to proceed",
    r"Try new model",
    r"Use existing model",
)
CODEX_UPDATE_NOTICE_PATTERNS = (
    r"Update available!",
    r"Update now",
    r"Skip until next version",
    r"Press enter to continue",
)
CODEX_STARTING_PATTERNS = (
    r"Starting MCP servers",
    r"MCP servers \(\d+/\d+\)",
)
GEMINI_READY_PATTERNS = (
    r"Type your message",
    r"@path/to/file",
)
GEMINI_TRUST_PROMPT_PATTERNS = (
    r"Do you trust the files in this folder\?",
    r"Trust folder",
    r"Trust parent folder",
    r"Don't trust",
)
GEMINI_NOT_READY_PATTERNS = (
    r"Waiting for authentication",
    r"Press Esc or Ctrl\+C to cancel",
)
GEMINI_BUSY_PATTERNS = (
    r"Working…",
    r"Working\.\.\.",
    r"Thinking…",
    r"Thinking\.\.\.",
)
GEMINI_INPUT_BOX_PATTERNS = (
    r"Type your message or @path/to/file",
    r"^│\s*>",
    r"^>$",
)
OPENCODE_READY_PROMPT_PATTERNS = (
    r"Ask anything\.\.\.",
)
OPENCODE_READY_FOOTER_PATTERNS = (
    r"ctrl\+p commands",
)
OPENCODE_BUSY_PATTERNS = (
    r"\besc interrupt\b",
)
OPENCODE_STARTING_PATTERNS = (
    r"Performing one time database migration",
    r"Database migration complete",
)
OPENCODE_READY_PROMPT_COMPACT_PATTERNS = (
    r"askanything\.\.\.",
)
OPENCODE_READY_FOOTER_COMPACT_PATTERNS = (
    r"ctrl\+pcommands",
)
OPENCODE_BUSY_COMPACT_PATTERNS = (
    r"escinterrupt",
)
OPENCODE_STARTING_COMPACT_PATTERNS = (
    r"performingonetimedatabasemigration",
    r"databasemigrationcomplete",
)
OPENCODE_FOOTER_PATTERNS = (
    r"ctrl\+p commands$",
    r"^tab agents\b",
    r"Build\s+·",
    r"^[╹▀]+$",
)


def _codex_effective_recent_surface(text: str, *, max_lines: int = 120) -> str:
    lines = str(text or "").splitlines()[-max_lines:]
    if not lines:
        return ""
    footer_index = -1
    for index, line in enumerate(lines):
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in CODEX_READY_FOOTER_PATTERNS):
            footer_index = index
    if footer_index < 0:
        return "\n".join(lines)
    prompt_index = -1
    for index in range(footer_index, -1, -1):
        if any(re.search(pattern, lines[index], re.IGNORECASE | re.MULTILINE) for pattern in CODEX_READY_PATTERNS):
            prompt_index = index
            break
    start_index = prompt_index if prompt_index >= 0 else max(0, footer_index - 20)
    return "\n".join(lines[start_index:])


def _codex_surface_indicates_ready_prompt(text: str) -> bool:
    surface = str(text or "")
    if not surface.strip():
        return False
    if not any(re.search(pattern, surface, re.IGNORECASE | re.MULTILINE) for pattern in CODEX_READY_PATTERNS):
        return False
    return not any(
        re.search(pattern, surface, re.IGNORECASE)
        for pattern in (
            *CODEX_STARTING_PATTERNS,
            *CODEX_TRUST_PROMPT_PATTERNS,
            *CODEX_MODEL_SELECTION_PROMPT_PATTERNS,
        )
    )


def _codex_surface_indicates_ready_input(text: str) -> bool:
    surface = str(text or "")
    if not _codex_surface_indicates_ready_prompt(surface):
        return False
    return any(re.search(pattern, surface, re.IGNORECASE | re.MULTILINE) for pattern in CODEX_READY_FOOTER_PATTERNS)


def _claude_title_indicates_ready(pane_title: str) -> bool:
    title = str(pane_title or "").strip()
    return bool(re.match(r"^✳(?:[\s\u00a0]|$)", title))


def _claude_title_indicates_busy(pane_title: str) -> bool:
    title = str(pane_title or "").strip()
    if not title or _claude_title_indicates_ready(title):
        return False
    return bool(CLAUDE_BUSY_TITLE_PREFIX_RE.match(title))


GEMINI_FOOTER_PATTERNS = (
    r"^\?\s+for shortcuts$",
    r"^YOLO\b",
    r"^[▀▄]+$",
    r"^workspace \(/directory\)",
)
AUDIT_STRUCTURED_PREFIXES = (
    "- verdict:",
    "- file_role:",
    "- finding:",
    "- duplication:",
    "- boundary_conflict:",
    "- missing:",
    "- recommendation:",
    "- top_priority:",
)
RUNTIME_NOISE_PATTERNS = (
    r"Thinking\.\.\.",
    r"^\?\s+for shortcuts$",
    r"^YOLO\b",
    r"^YOLO 模式",
    r"Type your message(?:\s+or\s+@path/to/file)?",
    r"输入您的消息(?:\s*或\s*@\s*文件路径)?",
    r"@path/to/file",
    r"^workspace \(/directory\)",
    r"^[─━]{5,}$",
    r"^[▀▄]+$",
    r"^❯$",
    r"^[·✳✽✢]?\s*[A-Za-z]+…(?:\s*\(.*\))?$",
    r"^\S+…(?:\s*\(.*\))?$",
    r"^[A-Za-z]+…\d*$",
    r"^✻\s*.+ for .*$",
    r"^✳\s*Metamorphosing.*$",
    r"^Metamorphosing….*$",
    r"^[·✳✽]?\s*(?:Precipitating|Metamorphosing|Moseying)….*$",
    r"^✻\s*(?:Baked|Cogitated|Churned|Brewed) for.*$",
    r"^\(·oo·\).*$",
    r"^⏵⏵.*$",
    r"^✗\s*Auto-update.*$",
    r"^(?:~|/)\S+\s+.+$",
    r"^(?:~|/).+\s{2,}.+$",
    r"^(?:gemini|claude|codex|opencode)(?:[-_.a-z0-9]+)?$",
)

_LIVE_WORKERS: "weakref.WeakSet[TmuxBatchWorker]" = weakref.WeakSet()
_LIVE_WORKERS_LOCK = threading.RLock()
_RESERVED_SESSION_NAMES: set[str] = set()
_RESERVED_SESSION_NAMES_LOCK = threading.RLock()
_SESSION_NAME_LEASE_ROOT = Path(tempfile.gettempdir()) / (
    f"tmux-tmux-session-leases-{getattr(os, 'getuid', lambda: 0)()}"
)
_SESSION_NAME_LEASE_LOCK_PATH = _SESSION_NAME_LEASE_ROOT / ".lock"
_SESSION_NAME_LEASE_PROCESS_LOCK = threading.RLock()
_SESSION_NAME_LEASE_LOCK_DEPTH = 0
TMUX_IDENTITY_RUNTIME_DIR_OPTION = "@tmux_runtime_dir"
TMUX_IDENTITY_WORK_DIR_OPTION = "@tmux_work_dir"
TMUX_IDENTITY_REQUIREMENT_NAME_OPTION = "@tmux_requirement_name"
TMUX_IDENTITY_WORKFLOW_ACTION_OPTION = "@tmux_workflow_action"
TMUX_IDENTITY_WORKER_ID_OPTION = "@tmux_worker_id"
SESSION_CONSTELLATION_NAMES: tuple[str, ...] = (
    "角木蛟",
    "亢金龙",
    "氐土貉",
    "房日兔",
    "心月狐",
    "尾火虎",
    "箕水豹",
    "斗木獬",
    "牛金牛",
    "女土蝠",
    "虚日鼠",
    "危月燕",
    "室火猪",
    "壁水貐",
    "奎木狼",
    "娄金狗",
    "胃土雉",
    "昴日鸡",
    "毕月乌",
    "觜火猴",
    "参水猿",
    "井木犴",
    "鬼金羊",
    "柳土獐",
    "星日马",
    "张月鹿",
    "翼火蛇",
    "轸水蚓",
    "天魁星",
    "天罡星",
    "天机星",
    "天闲星",
    "天勇星",
    "天雄星",
    "天猛星",
    "天威星",
    "天英星",
    "天贵星",
    "天富星",
    "天满星",
    "天孤星",
    "天伤星",
    "天立星",
    "天捷星",
    "天暗星",
    "天佑星",
    "天空星",
    "天速星",
    "天异星",
    "天杀星",
    "天微星",
    "天究星",
    "天退星",
    "天寿星",
    "天剑星",
    "天平星",
    "天罪星",
    "天损星",
    "天败星",
    "天牢星",
    "天慧星",
    "天暴星",
    "天哭星",
    "天巧星",
    "地魁星",
    "地煞星",
    "地勇星",
    "地杰星",
    "地雄星",
    "地威星",
    "地英星",
    "地奇星",
    "地猛星",
    "地文星",
    "地正星",
    "地辟星",
    "地阖星",
    "地强星",
    "地暗星",
    "地轴星",
    "地会星",
    "地佐星",
    "地佑星",
    "地灵星",
    "地兽星",
    "地微星",
    "地慧星",
    "地暴星",
    "地默星",
    "地猖星",
    "地狂星",
    "地飞星",
    "地走星",
    "地巧星",
    "地明星",
    "地进星",
    "地退星",
    "地满星",
    "地遂星",
    "地周星",
    "地隐星",
    "地异星",
    "地理星",
    "地俊星",
    "地乐星",
    "地捷星",
    "地速星",
    "地镇星",
)
_SESSION_ALLOWED_CHARS_RE = re.compile(r"[^A-Za-z0-9\u4e00-\u9fff-]+")
_SESSION_ROLE_REVIEWER_RE = re.compile(r"^requirements-review-r(\d+)$")
_SESSION_ROLE_DETAILED_DESIGN_REVIEWER_RE = re.compile(r"^detailed-design-review-(.+)$")
_SESSION_ROLE_TASK_SPLIT_REVIEWER_RE = re.compile(r"^task-split-review-(.+)$")
_SESSION_ROLE_DEVELOPMENT_REVIEWER_RE = re.compile(r"^development-review-(.+)$")
_DUPLICATE_SESSION_ERROR_MARKERS = (
    "duplicate session",
    "session already exists",
    "session exists",
    "already exists",
)


class Vendor(str, Enum):
    CODEX = "codex"
    CLAUDE = "claude"
    GEMINI = "gemini"
    OPENCODE = "opencode"


class WorkerStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class AgentRuntimeState(str, Enum):
    DEAD = "DEAD"
    STARTING = "STARTING"
    READY = "READY"
    BUSY = "BUSY"


class WrapperState(str, Enum):
    READY = "READY"
    NOT_READY = "NOT_READY"


def _register_live_worker(worker: "TmuxBatchWorker") -> None:
    with _LIVE_WORKERS_LOCK:
        _LIVE_WORKERS.add(worker)


def list_registered_tmux_workers() -> list["TmuxBatchWorker"]:
    with _LIVE_WORKERS_LOCK:
        return list(_LIVE_WORKERS)


def list_tmux_session_names(*, backend: Any | None = None) -> tuple[str, ...]:
    runtime_backend = backend or TmuxBackend()
    return tuple(sorted(_list_backend_session_names(runtime_backend)))


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def _session_name_lease_path(session_name: str) -> Path:
    digest = hashlib.sha1(str(session_name).encode("utf-8")).hexdigest()
    return _SESSION_NAME_LEASE_ROOT / f"{digest}.json"


@contextmanager
def _session_name_lease_lock():
    global _SESSION_NAME_LEASE_LOCK_DEPTH
    with _SESSION_NAME_LEASE_PROCESS_LOCK:
        if _SESSION_NAME_LEASE_LOCK_DEPTH > 0:
            _SESSION_NAME_LEASE_LOCK_DEPTH += 1
            try:
                yield
            finally:
                _SESSION_NAME_LEASE_LOCK_DEPTH -= 1
            return
        _SESSION_NAME_LEASE_ROOT.mkdir(parents=True, exist_ok=True)
        _SESSION_NAME_LEASE_LOCK_PATH.touch(exist_ok=True)
        with _SESSION_NAME_LEASE_LOCK_PATH.open("r+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            _SESSION_NAME_LEASE_LOCK_DEPTH = 1
            try:
                yield
            finally:
                _SESSION_NAME_LEASE_LOCK_DEPTH = 0
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_session_name_lease(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _active_session_name_leases_locked() -> set[str]:
    active: set[str] = set()
    if not _SESSION_NAME_LEASE_ROOT.exists():
        return active
    for lease_path in _SESSION_NAME_LEASE_ROOT.glob("*.json"):
        payload = _read_session_name_lease(lease_path)
        if payload is None:
            with contextlib.suppress(Exception):
                lease_path.unlink()
            continue
        session_name = str(payload.get("session_name", "")).strip()
        owner_pid = int(payload.get("owner_pid", 0) or 0)
        if not session_name:
            with contextlib.suppress(Exception):
                lease_path.unlink()
            continue
        if owner_pid and not _pid_exists(owner_pid):
            with contextlib.suppress(Exception):
                lease_path.unlink()
            continue
        active.add(session_name)
    return active


def _write_session_name_lease_locked(*, session_name: str, worker_id: str, work_dir: str | Path) -> None:
    lease_path = _session_name_lease_path(session_name)
    _atomic_write_json(
        lease_path,
        {
            "session_name": session_name,
            "worker_id": str(worker_id or "").strip(),
            "work_dir": str(Path(work_dir).expanduser().resolve()),
            "owner_pid": os.getpid(),
            "created_at": _now_iso(),
        },
    )


def _release_session_name_lease_locked(session_name: str) -> None:
    session_name_text = str(session_name or "").strip()
    if not session_name_text:
        return
    lease_path = _session_name_lease_path(session_name_text)
    if not lease_path.exists():
        return
    payload = _read_session_name_lease(lease_path)
    if payload is not None and str(payload.get("session_name", "")).strip() != session_name_text:
        return
    with contextlib.suppress(Exception):
        lease_path.unlink()


def list_occupied_tmux_session_names(
        *,
        backend: Any | None = None,
        additional_session_names: Sequence[str] = (),
) -> tuple[str, ...]:
    occupied = {str(name).strip() for name in additional_session_names if str(name).strip()}
    occupied.update(_list_backend_session_names(backend))
    with _session_name_lease_lock():
        occupied.update(_active_session_name_leases_locked())
    for worker in list_registered_tmux_workers():
        session_name = str(getattr(worker, "session_name", "") or "").strip()
        if session_name:
            occupied.add(session_name)
    return tuple(sorted(occupied))


def cleanup_registered_tmux_workers(*, reason: str = "process_exit") -> list[str]:
    cleaned_sessions: list[str] = []
    for worker in list_registered_tmux_workers():
        try:
            if not worker.session_exists():
                continue
            session_name = worker.request_kill()
            if session_name:
                cleaned_sessions.append(session_name)
                worker._log_event("process_cleanup_kill", reason=reason, session_name=session_name)
        except Exception:
            continue
    return sorted(set(cleaned_sessions))


@dataclass(frozen=True)
class WorkerObservation:
    visible_text: str
    raw_log_delta: str
    raw_log_tail: str
    current_command: str
    current_path: str
    pane_dead: bool
    session_exists: bool
    log_mtime: float
    observed_at: str
    pane_title: str = ""


@dataclass(frozen=True)
class WorkerHealthSnapshot:
    session_exists: bool
    health_status: str
    health_note: str
    last_heartbeat_at: str
    last_log_offset: int
    current_command: str
    current_path: str
    pane_id: str
    session_name: str
    agent_state: str = AgentRuntimeState.DEAD.value
    pane_title: str = ""


class TmuxBackend:
    def run(
            self,
            *args: str,
            input_text: str | None = None,
            timeout_sec: float = 10.0,
            check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["tmux", *args],
            check=check,
            text=True,
            capture_output=True,
            input=input_text,
            timeout=timeout_sec,
        )

    def has_session(self, session_name: str) -> bool:
        result = self.run("has-session", "-t", session_name, check=False)
        return result.returncode == 0

    def list_sessions(self) -> list[str]:
        result = self.run("list-sessions", "-F", "#S", check=False)
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def create_session(self, session_name: str, work_dir: Path, command: str) -> str:
        result = self.run(
            "new-session",
            "-d",
            "-P",
            "-F",
            "#{pane_id}",
            "-s",
            session_name,
            "-c",
            str(work_dir),
            command,
        )
        return result.stdout.strip()

    def kill_session(self, session_name: str) -> None:
        self.run("kill-session", "-t", session_name)

    def attach_session(self, session_name: str) -> None:
        subprocess.run(["tmux", "attach-session", "-t", session_name], check=True)

    def detach_session(self, session_name: str) -> None:
        self.run("detach-client", "-s", session_name)

    def target_exists(self, target_name: str) -> bool:
        result = self.run("list-panes", "-t", target_name, check=False)
        return result.returncode == 0

    def display_message(self, target: str, expression: str) -> str:
        return self.run("display-message", "-p", "-t", target, expression).stdout.strip()

    def show_option(self, target: str, option_name: str) -> str:
        result = self.run("show-options", "-qv", "-t", target, option_name, check=False)
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    def capture_visible(self, target: str, *, tail_lines: int = DEFAULT_CAPTURE_TAIL_LINES) -> str:
        return self.run(
            "capture-pane",
            "-J",
            "-p",
            "-t",
            target,
            "-S",
            f"-{tail_lines}",
            timeout_sec=15.0,
        ).stdout

    def pipe_log(self, target: str, raw_log_path: Path) -> None:
        command = f"cat >> {shlex.quote(str(raw_log_path))}"
        self.run("pipe-pane", "-t", target, "-o", command)

    def send_key(self, target: str, key: str) -> None:
        self.run("send-keys", "-t", target, key)

    def send_text(self, target: str, text: str, *, submit_count: int) -> None:
        buffer_name = f"acx_{uuid.uuid4().hex[:8]}"
        prompt_file_path = ""
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".tmux-prompt") as prompt_file:
                prompt_file.write(text)
                prompt_file_path = prompt_file.name
            self.run("load-buffer", "-b", buffer_name, prompt_file_path, timeout_sec=30.0)
            self.run("paste-buffer", "-p", "-b", buffer_name, "-t", target, timeout_sec=30.0)
            time.sleep(0.3)
            for index in range(submit_count):
                if index > 0:
                    time.sleep(0.5)
                self.run("send-keys", "-t", target, "Enter", timeout_sec=15.0)
        finally:
            if prompt_file_path:
                with contextlib.suppress(OSError):
                    os.unlink(prompt_file_path)
            subprocess.run(["tmux", "delete-buffer", "-b", buffer_name], check=False, capture_output=True)

    def tail_raw_log(
            self,
            raw_log_path: str | Path,
            *,
            last_offset: int = 0,
            tail_bytes: int = 24000,
    ) -> tuple[str, str, int, float]:
        path = Path(raw_log_path)
        if not path.exists():
            return "", "", 0, 0.0
        size = path.stat().st_size
        start = min(max(last_offset, 0), size)
        with path.open("rb") as file:
            file.seek(start)
            delta_bytes = file.read()
            tail_start = max(size - tail_bytes, 0)
            file.seek(tail_start)
            tail_data = file.read()
        mtime = path.stat().st_mtime
        return (
            delta_bytes.decode("utf-8", errors="replace"),
            tail_data.decode("utf-8", errors="replace"),
            size,
            mtime,
        )


class LaunchCoordinator:
    _vendor_locks: dict[str, threading.Lock] = {}
    _stagger_by_vendor: dict[str, float] = {}
    _guard = threading.Lock()
    base_stagger_sec = 2.0
    max_stagger_sec = 10.0

    def __init__(self, runtime_root: str | Path) -> None:
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        self.lock_root = self.runtime_root / "_locks"
        self.lock_root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _get_vendor_lock(cls, vendor: Vendor) -> threading.Lock:
        with cls._guard:
            return cls._vendor_locks.setdefault(vendor.value, threading.Lock())

    @classmethod
    def current_stagger(cls, vendor: Vendor) -> float:
        return cls._stagger_by_vendor.get(vendor.value, cls.base_stagger_sec)

    @classmethod
    def record_launch_result(cls, vendor: Vendor, *, success: bool) -> None:
        if success:
            cls._stagger_by_vendor[vendor.value] = cls.base_stagger_sec
            return
        previous = cls._stagger_by_vendor.get(vendor.value, cls.base_stagger_sec)
        cls._stagger_by_vendor[vendor.value] = min(previous * 2, cls.max_stagger_sec)

    @contextmanager
    def startup_slot(self, vendor: Vendor):
        vendor_lock = self._get_vendor_lock(vendor)
        lock_path = self.lock_root / f"launch_{vendor.value}.lock"
        with vendor_lock:
            with lock_path.open("a+", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    time.sleep(self.current_stagger(vendor))
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class HealthSupervisor:
    def __init__(
            self,
            refresh_callback: Callable[[], WorkerHealthSnapshot | None],
            *,
            interval_sec: float = 2.0,
            ready_interval_sec: float = READY_HEALTH_INTERVAL_SEC,
            idle_interval_sec: float = IDLE_HEALTH_INTERVAL_SEC,
            idle_after_sec: float = IDLE_HEALTH_AFTER_SEC,
            thread_name: str = "tmux-health",
    ) -> None:
        self.refresh_callback = refresh_callback
        self.interval_sec = interval_sec
        self.ready_interval_sec = ready_interval_sec
        self.idle_interval_sec = idle_interval_sec
        self.idle_after_sec = idle_after_sec
        self._next_interval_sec = interval_sec
        self._last_state_key = ""
        self._last_state_since = time.monotonic()
        self._terminal_snapshot_count = 0
        self._stopped = False
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=thread_name, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._stopped = True

    def is_alive(self) -> bool:
        return self._thread.is_alive() and not self._stop_event.is_set()

    def stopped(self) -> bool:
        return self._stopped or (self._stop_event.is_set() and not self._thread.is_alive())

    def _update_next_interval(self, snapshot: WorkerHealthSnapshot | None) -> bool:
        if snapshot is None:
            self._next_interval_sec = self.interval_sec
            return False
        agent_state = str(snapshot.agent_state or "").strip().upper()
        health_status = str(snapshot.health_status or "").strip().lower()
        state_key = f"{agent_state}:{health_status}"
        now = time.monotonic()
        if state_key != self._last_state_key:
            self._last_state_key = state_key
            self._last_state_since = now
        if agent_state == AgentRuntimeState.DEAD.value or health_status in {"missing_session", "pane_dead"}:
            self._terminal_snapshot_count += 1
            self._next_interval_sec = self.interval_sec
            return self._terminal_snapshot_count >= 2
        self._terminal_snapshot_count = 0
        if agent_state == AgentRuntimeState.READY.value:
            stable_for = now - self._last_state_since
            self._next_interval_sec = self.idle_interval_sec if stable_for >= self.idle_after_sec else self.ready_interval_sec
            return False
        self._next_interval_sec = self.interval_sec
        return False

    def _run(self) -> None:
        while not self._stop_event.wait(self._next_interval_sec):
            try:
                should_stop = self._update_next_interval(self.refresh_callback())
            except Exception:
                self._next_interval_sec = self.interval_sec
                continue
            if should_stop:
                self._stop_event.set()
        self._stopped = True


class TmuxRuntimeController:
    def __init__(self, backend: TmuxBackend | None = None) -> None:
        self.backend = backend or TmuxBackend()

    def session_exists(self, session_name: str) -> bool:
        return bool(session_name) and self.backend.has_session(session_name)

    def session_matches_context(
        self,
        session_name: str,
        *,
        runtime_dir: str | Path = "",
        work_dir: str | Path = "",
        requirement_name: str = "",
        workflow_action: str = "",
    ) -> bool:
        return _tmux_session_matches_context(
            self.backend,
            session_name,
            runtime_dir=runtime_dir,
            work_dir=work_dir,
            requirement_name=requirement_name,
            workflow_action=workflow_action,
        )

    def session_matches_worker_state(self, session_name: str, state: Mapping[str, Any], state_path: str | Path) -> bool:
        return self.session_matches_context(
            session_name,
            runtime_dir=Path(state_path).expanduser().resolve().parent,
            work_dir=str(state.get("work_dir") or state.get("project_dir") or "").strip(),
            requirement_name=str(state.get("requirement_name", "") or "").strip(),
            workflow_action=str(state.get("workflow_action", "") or "").strip(),
        )

    def worker_identity_for_runtime_dir(self, runtime_dir: str | Path) -> dict[str, object]:
        runtime_dir_text = _resolved_path_text(runtime_dir)
        if not runtime_dir_text:
            return {}
        for session_name in sorted(_list_backend_session_names(self.backend)):
            actual_runtime_dir = _backend_show_option(self.backend, session_name, TMUX_IDENTITY_RUNTIME_DIR_OPTION)
            if not actual_runtime_dir or not _same_resolved_path(actual_runtime_dir, runtime_dir_text):
                continue
            work_dir = _backend_show_option(self.backend, session_name, TMUX_IDENTITY_WORK_DIR_OPTION)
            pane_id = ""
            with contextlib.suppress(Exception):
                pane_id = str(self.backend.display_message(session_name, "#{pane_id}") or "").strip()
            return {
                "session_name": session_name,
                "session_exists": True,
                "pane_id": pane_id,
                "runtime_dir": actual_runtime_dir,
                "worker_id": _backend_show_option(self.backend, session_name, TMUX_IDENTITY_WORKER_ID_OPTION),
                "work_dir": work_dir,
                "project_dir": work_dir,
                "requirement_name": _backend_show_option(self.backend, session_name, TMUX_IDENTITY_REQUIREMENT_NAME_OPTION),
                "workflow_action": _backend_show_option(self.backend, session_name, TMUX_IDENTITY_WORKFLOW_ACTION_OPTION),
            }
        return {}

    def list_sessions(self) -> list[str]:
        return self.backend.list_sessions()

    def attach_session(self, session_name: str) -> None:
        if not self.session_exists(session_name):
            raise RuntimeError(f"tmux 会话尚未创建: {session_name}")
        self.backend.attach_session(session_name)

    def detach_session(self, session_name: str) -> str:
        if not session_name:
            raise RuntimeError("tmux 会话尚未创建")
        self.backend.detach_session(session_name)
        return session_name

    def kill_session(self, session_name: str, *, missing_ok: bool = True) -> str:
        if not session_name:
            if missing_ok:
                return ""
            raise RuntimeError("tmux 会话尚未创建")
        if not self.session_exists(session_name):
            if missing_ok:
                return session_name
            raise RuntimeError(f"tmux 会话尚未创建: {session_name}")
        self.backend.kill_session(session_name)
        return session_name

    def read_transcript_tail(self, transcript_path: str | Path, *, max_lines: int = 60) -> str:
        return read_text_tail(transcript_path, max_lines=max_lines)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_suffix = f".tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
    tmp_path = target.with_suffix(target.suffix + tmp_suffix)
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(target)


def is_worker_death_error(error: BaseException | str) -> bool:
    message = str(error or "").strip().lower()
    if not message:
        return False
    return any(marker in message for marker in WORKER_DEATH_ERROR_MARKERS)


def is_provider_auth_error(error: BaseException | str) -> bool:
    message = str(error or "").strip().lower()
    if not message:
        return False
    return any(marker in message for marker in PROVIDER_AUTH_ERROR_MARKERS)


def is_provider_runtime_error(error: BaseException | str) -> bool:
    message = str(error or "").strip().lower()
    if not message:
        return False
    return any(marker in message for marker in PROVIDER_RUNTIME_ERROR_MARKERS)


def is_agent_ready_timeout_error(error: BaseException | str) -> bool:
    message = str(error or "").strip().lower()
    if not message:
        return False
    return any(marker in message for marker in AGENT_READY_TIMEOUT_ERROR_MARKERS)


def is_prompt_dispatch_timeout_error(error: BaseException | str) -> bool:
    message = str(error or "").strip().lower()
    if not message or "timed out" not in message:
        return False
    return any(marker in message for marker in PROMPT_DISPATCH_TIMEOUT_ERROR_MARKERS)


def is_stale_busy_without_contract_error(error: BaseException | str) -> bool:
    message = str(error or "").strip().lower()
    if not message:
        return False
    return STALE_BUSY_WITHOUT_CONTRACT_REASON_PREFIX in message


class RuntimeShutdownRequested(RuntimeError):
    """Raised inside worker code when the owning backend is shutting down."""


_RUNTIME_SHUTDOWN_REQUESTED = threading.Event()
_RUNTIME_SHUTDOWN_REASON_LOCK = threading.Lock()
_RUNTIME_SHUTDOWN_REASON = ""


def request_runtime_shutdown(reason: str = "") -> None:
    global _RUNTIME_SHUTDOWN_REASON
    reason_text = str(reason or "").strip() or "shutdown"
    with _RUNTIME_SHUTDOWN_REASON_LOCK:
        _RUNTIME_SHUTDOWN_REASON = reason_text
    _RUNTIME_SHUTDOWN_REQUESTED.set()


def clear_runtime_shutdown_request() -> None:
    global _RUNTIME_SHUTDOWN_REASON
    _RUNTIME_SHUTDOWN_REQUESTED.clear()
    with _RUNTIME_SHUTDOWN_REASON_LOCK:
        _RUNTIME_SHUTDOWN_REASON = ""


def runtime_shutdown_requested() -> bool:
    return _RUNTIME_SHUTDOWN_REQUESTED.is_set()


def raise_if_runtime_shutdown_requested(context: str = "") -> None:
    if not runtime_shutdown_requested():
        return
    with _RUNTIME_SHUTDOWN_REASON_LOCK:
        reason_text = _RUNTIME_SHUTDOWN_REASON
    context_text = str(context or "").strip()
    details = f"{RUNTIME_SHUTDOWN_ERROR_MARKER}: {reason_text or 'shutdown'}"
    if context_text:
        details = f"{details} while {context_text}"
    raise RuntimeShutdownRequested(details)


def is_runtime_shutdown_error(error: BaseException | str) -> bool:
    if isinstance(error, RuntimeShutdownRequested):
        return True
    message = str(error or "").strip().lower()
    return RUNTIME_SHUTDOWN_ERROR_MARKER in message


def is_turn_artifact_contract_error(error: BaseException | str) -> bool:
    message = str(error or "").strip().lower()
    if not message:
        return False
    return TURN_ARTIFACT_CONTRACT_ERROR_PREFIX in message


def is_task_result_contract_error(error: BaseException | str) -> bool:
    message = str(error or "").strip().lower()
    if not message:
        return False
    return TASK_RESULT_CONTRACT_ERROR_PREFIX in message


def try_resume_worker(worker: "TmuxBatchWorker", *, timeout_sec: float = 60.0) -> bool:
    if worker is None:
        return False

    def _read_state() -> dict[str, object]:
        read_state = getattr(worker, "read_state", None)
        if not callable(read_state):
            return {}
        try:
            payload = read_state()
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _session_exists() -> bool:
        session_exists = getattr(worker, "session_exists", None)
        if not callable(session_exists):
            return False
        try:
            return bool(session_exists())
        except Exception:
            return False

    def _target_exists() -> bool:
        target_exists = getattr(worker, "target_exists", None)
        if not callable(target_exists):
            return True
        try:
            return bool(target_exists())
        except Exception:
            return False

    def _observe_once() -> WorkerObservation | None:
        observe = getattr(worker, "observe", None)
        if not callable(observe):
            return None
        try:
            return observe(tail_lines=120, tail_bytes=12000)
        except Exception:
            return None

    def _current_agent_state(observation: WorkerObservation | None = None) -> AgentRuntimeState | None:
        get_agent_state = getattr(worker, "get_agent_state", None)
        if callable(get_agent_state):
            try:
                if observation is None:
                    return get_agent_state()
                return get_agent_state(observation)
            except TypeError:
                try:
                    return get_agent_state()
                except Exception:
                    return None
            except Exception:
                return None
        state = _read_state()
        agent_state_text = str(state.get("agent_state", "") or "").strip().upper()
        if not agent_state_text:
            return None
        with contextlib.suppress(Exception):
            return AgentRuntimeState[agent_state_text]
        return None

    def _normalize_ready_state(observation: WorkerObservation | None = None) -> None:
        write_state = getattr(worker, "_write_state", None)
        if not callable(write_state):
            return
        current_command = ""
        current_path = ""
        observed_at = ""
        pane_title = ""
        if observation is not None:
            current_command = str(observation.current_command or "").strip()
            current_path = str(observation.current_path or "").strip()
            observed_at = str(observation.observed_at or "").strip()
            pane_title = str(observation.pane_title or "").strip()
        extra = {
            "dispatch_state": "",
            "dispatch_reason": "",
            "health_status": "alive",
            "health_note": "auto_resumed",
            "agent_ready": True,
            "agent_alive": True,
            "agent_started": True,
            "agent_state": AgentRuntimeState.READY.value,
            "result_status": "ready",
            "status": WorkerStatus.READY.value,
            "current_task_runtime_status": "",
        }
        if current_command:
            extra["current_command"] = current_command
        if current_path:
            extra["current_path"] = current_path
        if observed_at:
            extra["last_heartbeat_at"] = observed_at
        if pane_title:
            extra["pane_title"] = pane_title
        with contextlib.suppress(Exception):
            write_state(WorkerStatus.READY, note="auto_resume_ready", extra=extra)

    state = _read_state()
    dispatch_reason = str(state.get("dispatch_reason", "") or "").strip()
    dispatch_state = str(state.get("dispatch_state", "") or "").strip().lower()
    note = str(state.get("note", "") or "").strip().lower()
    health_status = str(state.get("health_status", "") or "").strip().lower()
    health_note = str(state.get("health_note", "") or "").strip()
    current_command_state = str(state.get("current_command", "") or "").strip()

    if dispatch_reason.startswith(f"{STALE_BUSY_WITHOUT_CONTRACT_REASON_PREFIX}:"):
        return False
    if note == "awaiting_reconfig" or health_status in {
        "awaiting_reconfig",
        "provider_auth_error",
        "provider_runtime_error",
        "dead",
        "missing_session",
    }:
        return False
    if is_provider_auth_error(health_note) or is_provider_runtime_error(health_note):
        return False

    if not _session_exists() or not _target_exists():
        return False

    deadline = time.monotonic() + max(0.0, timeout_sec)
    last_observation: WorkerObservation | None = None
    while time.monotonic() <= deadline:
        observation = _observe_once()
        if observation is not None:
            last_observation = observation
            if not observation.session_exists or observation.pane_dead:
                return False
            current_command = str(observation.current_command or "").strip()
            if current_command in SHELL_COMMANDS:
                return False
            agent_state = _current_agent_state(observation)
            if agent_state == AgentRuntimeState.READY:
                _normalize_ready_state(observation)
                return True
            if agent_state not in {AgentRuntimeState.BUSY, AgentRuntimeState.STARTING}:
                return False
        else:
            agent_state = _current_agent_state()
            if agent_state == AgentRuntimeState.READY:
                if current_command_state and current_command_state not in SHELL_COMMANDS:
                    _normalize_ready_state(last_observation)
                    return True
                return False
            if agent_state not in {AgentRuntimeState.BUSY, AgentRuntimeState.STARTING, None}:
                return False
        time.sleep(FILE_CONTRACT_POLL_INTERVAL_SEC)
    return False


def load_worker_from_state_path(
        state_path: str | Path,
        *,
        backend: TmuxBackend | None = None,
) -> "TmuxBatchWorker" | None:
    path = Path(state_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    work_dir = str(payload.get("work_dir", "")).strip()
    config_payload = payload.get("config", {})
    if not work_dir or not isinstance(config_payload, dict):
        return None
    vendor = str(config_payload.get("vendor", "")).strip()
    model = str(config_payload.get("model", "")).strip()
    if not vendor or not model:
        return None
    worker_id = (
        str(payload.get("worker_id", "")).strip()
        or str(payload.get("raw_worker_id", "")).strip()
        or path.parent.name.rsplit("-", 1)[0]
        or "worker"
    )
    try:
        return TmuxBatchWorker(
            worker_id=worker_id,
            work_dir=work_dir,
            config=AgentRunConfig(
                vendor=vendor,
                model=model,
                reasoning_effort=str(config_payload.get("reasoning_effort", "high")).strip() or "high",
                proxy_url=str(config_payload.get("proxy_url", "")).strip(),
            ),
            runtime_root=path.parent.parent,
            existing_runtime_dir=path.parent,
            existing_session_name=str(payload.get("session_name", "")).strip(),
            existing_pane_id=str(payload.get("pane_id", "")).strip(),
            backend=backend,
        )
    except Exception:
        return None


def _slugify(text: str, max_len: int = 40) -> str:
    raw = str(text or "").strip()
    value = re.sub(r"[^a-zA-Z0-9]+", "-", raw).strip("-").lower()
    if not value:
        if raw:
            value = f"worker-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:8]}"
        else:
            value = "worker"
    return value[:max_len].strip("-") or "worker"


def _sanitize_session_fragment(text: str, *, fallback: str) -> str:
    value = str(text or "").strip()
    if not value:
        value = fallback
    value = re.sub(r"[\s/\\]+", "-", value)
    value = _SESSION_ALLOWED_CHARS_RE.sub("-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or fallback


def _sanitize_task_runtime_fragment(text: str, *, fallback: str, max_len: int) -> str:
    raw = str(text or "").strip()
    value = _sanitize_session_fragment(raw, fallback=fallback)
    if len(value) <= max_len:
        return value or fallback
    digest = hashlib.sha1((raw or value).encode("utf-8")).hexdigest()[:8]
    suffix = f"-{digest}"
    head_len = max(max_len - len(suffix), 1)
    head = value[:head_len].strip("-")
    return f"{head}{suffix}" if head else digest[:max_len]


def _worker_role_key(worker_id: str, work_dir: str | Path) -> str:
    worker_key = str(worker_id or "").strip().lower()
    if not worker_key:
        return "generic-executor"
    if worker_key == "requirements-analyst":
        return worker_key
    if worker_key == "requirements-notion-reader":
        return worker_key
    if worker_key == "requirements-review-analyst":
        return worker_key
    if worker_key == "detailed-design-analyst":
        return worker_key
    if worker_key == "task-split-analyst":
        return worker_key
    if worker_key == "development-developer":
        return worker_key
    if _SESSION_ROLE_REVIEWER_RE.fullmatch(worker_key):
        return worker_key
    if _SESSION_ROLE_DETAILED_DESIGN_REVIEWER_RE.fullmatch(worker_key):
        return worker_key
    if _SESSION_ROLE_TASK_SPLIT_REVIEWER_RE.fullmatch(worker_key):
        return worker_key
    if _SESSION_ROLE_DEVELOPMENT_REVIEWER_RE.fullmatch(worker_key):
        return worker_key
    work_dir_path = Path(work_dir).expanduser().resolve()
    work_dir_name = work_dir_path.name.strip().lower()
    work_dir_slug = _slugify(work_dir_name, max_len=48)
    if worker_key in {"project", work_dir_name, work_dir_slug}:
        return "routing-initializer"
    return "generic-executor"


def _worker_role_label(role_key: str) -> str:
    if role_key == "requirements-analyst":
        return "分析师"
    if role_key == "requirements-notion-reader":
        return "需求录入员"
    if role_key == "requirements-review-analyst":
        return "需求分析师"
    if role_key == "detailed-design-analyst":
        return "需求分析师"
    if role_key == "task-split-analyst":
        return "需求分析师"
    if role_key == "development-developer":
        return "开发工程师"
    reviewer_match = _SESSION_ROLE_REVIEWER_RE.fullmatch(role_key)
    if reviewer_match:
        return "审核器"
    detailed_design_match = _SESSION_ROLE_DETAILED_DESIGN_REVIEWER_RE.fullmatch(role_key)
    if detailed_design_match:
        return _sanitize_session_fragment(detailed_design_match.group(1), fallback="审核员")
    task_split_match = _SESSION_ROLE_TASK_SPLIT_REVIEWER_RE.fullmatch(role_key)
    if task_split_match:
        return _sanitize_session_fragment(task_split_match.group(1), fallback="审核员")
    development_match = _SESSION_ROLE_DEVELOPMENT_REVIEWER_RE.fullmatch(role_key)
    if development_match:
        return _sanitize_session_fragment(development_match.group(1), fallback="审核员")
    if role_key == "routing-initializer":
        return "路由器"
    return "执行者"


def _flush_runtime_state_changed_notification() -> None:
    global _runtime_state_notify_pending, _runtime_state_notify_running, _runtime_state_notify_timer
    with _runtime_state_notify_lock:
        if not _runtime_state_notify_pending:
            _runtime_state_notify_timer = None
            return
        if _runtime_state_notify_running:
            _runtime_state_notify_timer = None
            return
        _runtime_state_notify_pending = False
        _runtime_state_notify_timer = None
        _runtime_state_notify_running = True
    try:
        from T09_terminal_ops import notify_runtime_state_changed
    except Exception:
        pass
    else:
        try:
            notify_runtime_state_changed()
        except Exception:
            pass
    finally:
        with _runtime_state_notify_lock:
            _runtime_state_notify_running = False
            should_reschedule = _runtime_state_notify_pending and _runtime_state_notify_timer is None
            if not should_reschedule:
                return
            timer = threading.Timer(
                max(float(_RUNTIME_STATE_NOTIFY_DEBOUNCE_SEC), 0.0),
                _flush_runtime_state_changed_notification,
            )
            timer.daemon = True
            _runtime_state_notify_timer = timer
            timer.start()


def _notify_runtime_state_changed_best_effort() -> None:
    global _runtime_state_notify_pending, _runtime_state_notify_timer
    with _runtime_state_notify_lock:
        _runtime_state_notify_pending = True
        if _runtime_state_notify_timer is not None or _runtime_state_notify_running:
            return
        timer = threading.Timer(
            max(float(_RUNTIME_STATE_NOTIFY_DEBOUNCE_SEC), 0.0),
            _flush_runtime_state_changed_notification,
        )
        timer.daemon = True
        _runtime_state_notify_timer = timer
        timer.start()


def _preferred_constellation_index(work_dir: str | Path, role_key: str) -> int:
    stable_key = f"{Path(work_dir).expanduser().resolve()}::{role_key}"
    digest = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % len(SESSION_CONSTELLATION_NAMES)


def _resolved_path_text(path: str | Path) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).expanduser().resolve())
    except Exception:
        return text


def _same_resolved_path(left: str | Path, right: str | Path) -> bool:
    left_text = _resolved_path_text(left)
    right_text = _resolved_path_text(right)
    return bool(left_text and right_text and left_text == right_text)


def _path_is_same_or_under(candidate: str | Path, root: str | Path) -> bool:
    candidate_text = _resolved_path_text(candidate)
    root_text = _resolved_path_text(root)
    if not candidate_text or not root_text:
        return False
    candidate_path = Path(candidate_text)
    root_path = Path(root_text)
    return candidate_path == root_path or root_path in candidate_path.parents


def _list_backend_session_names(backend: Any | None) -> set[str]:
    list_sessions = getattr(backend, "list_sessions", None)
    if not callable(list_sessions):
        return set()
    try:
        return {str(name).strip() for name in list_sessions() if str(name).strip()}
    except Exception:
        return set()


def _exception_message(error: Exception) -> str:
    parts = [str(error or "")]
    for field_name in ("stderr", "stdout"):
        value = getattr(error, field_name, "")
        if value:
            parts.append(str(value))
    return " ".join(part for part in parts if part).strip()


def _backend_show_option(backend: Any, target: str, option_name: str) -> str:
    show_option = getattr(backend, "show_option", None)
    if callable(show_option):
        try:
            return str(show_option(target, option_name) or "").strip()
        except Exception:
            return ""
    run = getattr(backend, "run", None)
    if not callable(run):
        return ""
    try:
        result = run("show-options", "-qv", "-t", target, option_name, check=False)
    except Exception:
        return ""
    if getattr(result, "returncode", 1) != 0:
        return ""
    return str(getattr(result, "stdout", "") or "").strip()


def _backend_current_path(backend: Any, target: str) -> str:
    display_message = getattr(backend, "display_message", None)
    if not callable(display_message):
        return ""
    try:
        return str(display_message(target, "#{pane_current_path}") or "").strip()
    except Exception:
        return ""


def _tmux_session_matches_context(
    backend: Any,
    session_name: str,
    *,
    runtime_dir: str | Path = "",
    work_dir: str | Path = "",
    requirement_name: str = "",
    workflow_action: str = "",
) -> bool:
    session_name_text = str(session_name or "").strip()
    if not session_name_text:
        return False
    has_session = getattr(backend, "has_session", None)
    if not callable(has_session):
        has_session = getattr(backend, "session_exists", None)
    if not callable(has_session):
        return False
    try:
        if not bool(has_session(session_name_text)):
            return False
    except Exception:
        return False

    expected_runtime_dir = _resolved_path_text(runtime_dir)
    expected_work_dir = _resolved_path_text(work_dir)
    expected_requirement_name = str(requirement_name or "").strip()
    expected_workflow_action = str(workflow_action or "").strip()

    actual_runtime_dir = _backend_show_option(backend, session_name_text, TMUX_IDENTITY_RUNTIME_DIR_OPTION)
    if actual_runtime_dir:
        return bool(expected_runtime_dir and _same_resolved_path(actual_runtime_dir, expected_runtime_dir))

    actual_work_dir = _backend_show_option(backend, session_name_text, TMUX_IDENTITY_WORK_DIR_OPTION)
    if actual_work_dir and expected_work_dir and not _same_resolved_path(actual_work_dir, expected_work_dir):
        return False

    actual_requirement_name = _backend_show_option(backend, session_name_text, TMUX_IDENTITY_REQUIREMENT_NAME_OPTION)
    if actual_requirement_name and expected_requirement_name and actual_requirement_name != expected_requirement_name:
        return False

    actual_workflow_action = _backend_show_option(backend, session_name_text, TMUX_IDENTITY_WORKFLOW_ACTION_OPTION)
    if actual_workflow_action and expected_workflow_action and actual_workflow_action != expected_workflow_action:
        return False

    if actual_work_dir or actual_requirement_name or actual_workflow_action:
        return True

    current_path = _backend_current_path(backend, session_name_text)
    if current_path and expected_work_dir:
        return _path_is_same_or_under(current_path, expected_work_dir)
    return True


def _occupied_session_names(backend: Any | None = None) -> set[str]:
    return set(list_occupied_tmux_session_names(backend=backend))


def _reserve_session_name(
        *,
        worker_id: str,
        work_dir: str | Path,
        vendor: Vendor,
        instance_id: str = "",
        backend: Any | None = None,
) -> str:
    del instance_id
    with _session_name_lease_lock():
        occupied = _list_backend_session_names(backend)
        occupied.update(_active_session_name_leases_locked())
        for worker in list_registered_tmux_workers():
            session_name = str(getattr(worker, "session_name", "") or "").strip()
            if session_name:
                occupied.add(session_name)
        with _RESERVED_SESSION_NAMES_LOCK:
            occupied.update(_RESERVED_SESSION_NAMES)
        session_name = build_session_name(
            worker_id,
            Path(work_dir),
            vendor,
            occupied_session_names=sorted(occupied),
        )
        _write_session_name_lease_locked(
            session_name=session_name,
            worker_id=worker_id,
            work_dir=work_dir,
        )
        with _RESERVED_SESSION_NAMES_LOCK:
            _RESERVED_SESSION_NAMES.add(session_name)
        return session_name


def _release_reserved_session_name(session_name: str) -> None:
    session_name_text = str(session_name or "").strip()
    with _session_name_lease_lock():
        _release_session_name_lease_locked(session_name_text)
        with _RESERVED_SESSION_NAMES_LOCK:
            _RESERVED_SESSION_NAMES.discard(session_name_text)


def _build_prefixed_sha256(path: str | Path) -> str:
    target = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with target.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def clean_ansi(text: str) -> str:
    return re.sub(
        r"\x1b\[[0-9;?]*[A-Za-z]"
        r"|\x1b\]8;[^\x1b]*\x1b\\"
        r"|\x1b\][^\x07]*\x07"
        r"|\x1b\][^\x1b]*\x1b\\"
        r"|\x1b[()][A-Z0-9]"
        r"|\x1b[\x20-\x2f]*[\x40-\x7e]",
        "",
        str(text or ""),
    )


def _normalize_opencode_surface(text: str) -> str:
    return re.sub(r"\s+", " ", clean_ansi(text)).strip()


def _compact_opencode_surface(text: str) -> str:
    normalized = _normalize_opencode_surface(text)
    return re.sub(r"[\s\u2500-\u28ff]+", "", normalized)


def _matches_opencode_surface(
        normalized_text: str,
        compact_text: str,
        *,
        patterns: Sequence[str],
        compact_patterns: Sequence[str],
) -> bool:
    return any(re.search(pattern, normalized_text, re.IGNORECASE | re.MULTILINE) for pattern in patterns) or any(
        re.search(pattern, compact_text, re.IGNORECASE | re.MULTILINE) for pattern in compact_patterns
    )


def _classify_opencode_surface_state(
        *,
        visible_text: str,
        recent_log: str,
        current_command: str,
) -> AgentRuntimeState:
    normalized_visible = _normalize_opencode_surface(visible_text)
    normalized_recent = _normalize_opencode_surface(recent_log)
    compact_visible = _compact_opencode_surface(visible_text)
    compact_recent = _compact_opencode_surface(recent_log)
    if _matches_opencode_surface(
            normalized_visible,
            compact_visible,
            patterns=OPENCODE_BUSY_PATTERNS,
            compact_patterns=OPENCODE_BUSY_COMPACT_PATTERNS,
    ):
        return AgentRuntimeState.BUSY
    if _matches_opencode_surface(
            normalized_visible,
            compact_visible,
            patterns=OPENCODE_READY_PROMPT_PATTERNS,
            compact_patterns=OPENCODE_READY_PROMPT_COMPACT_PATTERNS,
    ):
        return AgentRuntimeState.READY
    if current_command and _matches_opencode_surface(
            normalized_visible,
            compact_visible,
            patterns=OPENCODE_READY_FOOTER_PATTERNS,
            compact_patterns=OPENCODE_READY_FOOTER_COMPACT_PATTERNS,
    ):
        return AgentRuntimeState.READY
    if _matches_opencode_surface(
            normalized_visible or normalized_recent,
            compact_visible or compact_recent,
            patterns=OPENCODE_STARTING_PATTERNS,
            compact_patterns=OPENCODE_STARTING_COMPACT_PATTERNS,
    ):
        return AgentRuntimeState.STARTING
    if _matches_opencode_surface(
            normalized_recent,
            compact_recent,
            patterns=OPENCODE_READY_PROMPT_PATTERNS,
            compact_patterns=OPENCODE_READY_PROMPT_COMPACT_PATTERNS,
    ):
        return AgentRuntimeState.READY
    if current_command and _matches_opencode_surface(
            normalized_recent,
            compact_recent,
            patterns=OPENCODE_READY_FOOTER_PATTERNS,
            compact_patterns=OPENCODE_READY_FOOTER_COMPACT_PATTERNS,
    ):
        return AgentRuntimeState.READY
    if not current_command:
        return AgentRuntimeState.STARTING
    if normalized_visible or normalized_recent or compact_visible or compact_recent:
        return AgentRuntimeState.BUSY
    return AgentRuntimeState.BUSY


def read_text_tail(path: str | Path, max_lines: int = 40) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return "(文件不存在)"
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:]).strip() or "(文件为空)"


def _canonical_audit_line(line: str) -> str:
    text = clean_ansi(line).strip()
    if not text or text.startswith("[[ACX_TURN:"):
        return ""
    for prefix in AUDIT_STRUCTURED_PREFIXES:
        marker_index = text.find(prefix)
        if marker_index >= 0:
            return text[marker_index:].strip()
    return ""


def _is_prompt_example_audit_line(line: str) -> bool:
    text = clean_ansi(line).strip()
    if not text:
        return False
    return bool(re.search(r"<[^>]+>", text))


def is_runtime_noise_line(line: str) -> bool:
    text = clean_ansi(line).strip()
    if not text:
        return True
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in RUNTIME_NOISE_PATTERNS)


def _last_seen_protocol_token(text: str, allowed_tokens: Sequence[str]) -> str:
    allowed = set(allowed_tokens)
    for line in reversed(str(text or "").splitlines()):
        candidate = _extract_protocol_token_from_line(line, allowed)
        if candidate:
            return candidate
    return ""


def _extract_protocol_token_from_line(text: str, allowed_tokens: Sequence[str]) -> str:
    cleaned = clean_ansi(text).strip()
    if not cleaned:
        return ""
    for token in allowed_tokens:
        index = cleaned.find(token)
        if index < 0:
            continue
        prefix = cleaned[:index].strip()
        suffix = cleaned[index + len(token):].strip()
        if suffix:
            continue
        if prefix and not _is_symbolic_protocol_prefix(prefix):
            continue
        return token
    return ""


def _is_symbolic_protocol_prefix(prefix: str) -> bool:
    text = str(prefix or "").strip()
    if not text:
        return True
    if len(text) > 8:
        return False
    for ch in text:
        if ch.isalnum() or ch in "<>{}`[]/\\'\"":
            return False
        if "\u4e00" <= ch <= "\u9fff":
            return False
    return True


def _is_turn_token_line(text: str) -> bool:
    cleaned = clean_ansi(text).strip()
    match = re.search(r"\[\[ACX_TURN:[^:\]]+:DONE\]\]", cleaned)
    if not match:
        return False
    prefix = cleaned[: match.start()].strip()
    suffix = cleaned[match.end():].strip()
    return not suffix and _is_symbolic_protocol_prefix(prefix)


def normalize_effort(effort: str | None) -> str:
    value = str(effort or "high").strip().lower()
    allowed = {"low", "medium", "high", "xhigh", "max"}
    if value not in allowed:
        raise ValueError(f"不支持的推理强度: {effort}")
    return value


def normalize_vendor(vendor: str | Vendor) -> Vendor:
    if isinstance(vendor, Vendor):
        return vendor
    value = str(vendor or "").strip().lower()
    try:
        return Vendor(value)
    except ValueError as error:
        raise ValueError(f"不支持的厂商: {vendor}") from error


def normalize_proxy_url(proxy_value: str | int | None) -> str:
    if proxy_value is None:
        return ""
    text = str(proxy_value).strip()
    if not text:
        return ""
    if text.isdigit():
        return f"http://{DEFAULT_PROXY_HOST}:{text}"
    if re.fullmatch(r"[A-Za-z0-9_.-]+:\d+", text):
        return f"http://{text}"
    if re.fullmatch(r"\d+\.\d+\.\d+\.\d+:\d+", text):
        return f"http://{text}"
    if re.match(r"^(?:https?|socks5h?|socks4)://", text):
        return text
    raise ValueError(f"无法解析代理端口/地址: {proxy_value}")


def build_proxy_env(proxy_url: str) -> dict[str, str]:
    if not proxy_url:
        return {}
    parsed = urlparse(proxy_url)
    http_proxy = proxy_url
    all_proxy = proxy_url
    if parsed.scheme in {"http", "https"} and parsed.hostname and parsed.port:
        all_proxy = f"socks5h://{parsed.hostname}:{parsed.port}"
    return {
        "HTTP_PROXY": http_proxy,
        "HTTPS_PROXY": http_proxy,
        "http_proxy": http_proxy,
        "https_proxy": http_proxy,
        "ALL_PROXY": all_proxy,
        "all_proxy": all_proxy,
    }


def build_reasoning_note(
        vendor: Vendor,
        effort: str,
        *,
        model: str = "",
        resolution: LaunchResolution | None = None,
) -> str:
    resolved = resolution or resolve_launch(vendor.value, model, effort)
    parts = [
        f"reasoning_effort={resolved.normalized_effort}",
        f"reasoning_mode={resolved.reasoning_control_mode}",
        f"catalog_source={resolved.catalog_source_kind}",
    ]
    if resolved.native_reasoning_level:
        parts.append(f"native_reasoning={resolved.native_reasoning_level}")
    if vendor == Vendor.CLAUDE and resolved.native_reasoning_level:
        parts.append(f"claude_effort={resolved.native_reasoning_level}")
    if vendor == Vendor.GEMINI and resolved.reasoning_control_mode == "model_family_routing":
        parts.append(f"gemini_model_family={resolved.resolved_model}")
    if vendor == Vendor.OPENCODE:
        parts.append(f"opencode_model={resolved.resolved_model}")
        if resolved.resolved_variant:
            parts.append(f"opencode_variant={resolved.resolved_variant}")
    return "; ".join(parts)


def build_prompt_header(vendor: Vendor, model: str, effort: str) -> str:
    resolution = resolve_launch(vendor.value, model, effort)
    note = build_reasoning_note(vendor, effort, model=model, resolution=resolution)
    return (
        "[Agent Runtime Context]\n"
        f"- vendor: {vendor.value}\n"
        f"- model: {resolution.resolved_model}\n"
        f"- {note}\n"
        "- execution_mode: tmux_interactive_conversation\n"
        "- keep_scope_strict: true\n"
    )


class BaseOutputDetector:
    @staticmethod
    def _contains_any(text: str, patterns: Sequence[str]) -> bool:
        return any(re.search(pattern, text, re.IGNORECASE | re.MULTILINE) for pattern in patterns)

    @staticmethod
    def _has_turn_token(text: str) -> bool:
        return bool(re.search(r"\[\[ACX_TURN:[^:\]]+:DONE\]\]", text))

    def observation_text(self, observation: WorkerObservation) -> str:
        return clean_ansi("\n".join(part for part in [observation.raw_log_tail, observation.visible_text] if part))

    @staticmethod
    def current_visible_text(observation: WorkerObservation) -> str:
        return clean_ansi(observation.visible_text or "")

    @staticmethod
    def recent_log_text(observation: WorkerObservation, *, max_lines: int = 120) -> str:
        lines = clean_ansi(observation.raw_log_tail or "").splitlines()
        return "\n".join(lines[-max_lines:])

    def classify_agent_state(self, observation: WorkerObservation) -> AgentRuntimeState:
        text = self.observation_text(observation)
        if observation.pane_dead:
            return AgentRuntimeState.DEAD
        if not observation.session_exists:
            return AgentRuntimeState.DEAD
        if observation.current_command in SHELL_COMMANDS:
            return AgentRuntimeState.DEAD
        if self._has_turn_token(text):
            return AgentRuntimeState.READY
        if observation.current_command:
            return AgentRuntimeState.BUSY
        return AgentRuntimeState.STARTING

    @staticmethod
    def _split_blocks(text: str) -> list[str]:
        blocks: list[str] = []
        current_lines: list[str] = []
        for line in str(text or "").splitlines():
            if not line.strip():
                if current_lines:
                    blocks.append("\n".join(current_lines).strip())
                    current_lines = []
                continue
            current_lines.append(line.rstrip())
        if current_lines:
            blocks.append("\n".join(current_lines).strip())
        return [block for block in blocks if block]

    @staticmethod
    def _is_shell_prompt_line(line: str) -> bool:
        stripped = line.strip()
        return bool(re.search(r"[$%#]\s*$", stripped))

    def _should_skip_line(self, line: str) -> bool:
        skip_patterns = (
            r"^\? for shortcuts",
            r"^context left",
            r"^\d+%\s+left",
            r"^Tip:",
            r"^Press (?:ESC|Esc|esc)",
            r"^Use the arrow keys",
            r"^Select an option",
            r"^[│┌┐└┘╭╮╰╯╷╵─═]+$",
            r"^╭─",
            r"^╰─",
            r"^│\s*$",
        )
        return any(re.search(pattern, line, re.IGNORECASE) for pattern in skip_patterns) or self._is_shell_prompt_line(
            line)

    def extract_last_message(self, output: str) -> str:
        clean_output = clean_ansi(output)
        lines: list[str] = []
        for line in clean_output.splitlines():
            normalized = line.strip()
            if not normalized:
                lines.append("")
                continue
            if self._should_skip_line(normalized):
                continue
            lines.append(line.rstrip())
        blocks = self._split_blocks("\n".join(lines))
        if not blocks:
            raise ValueError("No assistant response found in terminal output")
        return blocks[-1]


class CodexOutputDetector(BaseOutputDetector):
    def classify_agent_state(self, observation: WorkerObservation) -> AgentRuntimeState:
        base_state = super().classify_agent_state(observation)
        if base_state == AgentRuntimeState.DEAD:
            return base_state
        surface = "\n".join(
            part for part in (observation.visible_text, observation.raw_log_tail) if str(part or "").strip()
        )
        effective_surface = _codex_effective_recent_surface(surface)
        title = str(observation.pane_title or "").strip()
        if not title:
            return AgentRuntimeState.STARTING
        if BRAILLE_SPINNER_PREFIX_RE.match(title):
            return AgentRuntimeState.BUSY
        if _codex_surface_indicates_ready_input(effective_surface):
            return AgentRuntimeState.READY
        return AgentRuntimeState.STARTING

    def extract_last_message(self, output: str) -> str:
        clean_output = clean_ansi(output)
        assistant_matches = list(
            re.finditer(r"^(?:assistant|codex|agent)\s*:\s*", clean_output, re.IGNORECASE | re.MULTILINE)
        )
        if assistant_matches:
            last_match = assistant_matches[-1]
            content = clean_output[last_match.end():]
            idle_match = re.search(r"^\s*(?:❯|›|codex>)(?:\s|$)", content, re.MULTILINE)
            if idle_match:
                content = content[:idle_match.start()]
            return super().extract_last_message(content)
        idle_match = re.search(r"^\s*(?:❯|›|codex>)(?:\s|$)", clean_output, re.MULTILINE)
        if idle_match:
            clean_output = clean_output[:idle_match.start()]
        return super().extract_last_message(clean_output)


class ClaudeOutputDetector(BaseOutputDetector):
    def classify_agent_state(self, observation: WorkerObservation) -> AgentRuntimeState:
        visible_text = self.current_visible_text(observation)
        recent_log = self.recent_log_text(observation)
        text = self.observation_text(observation)
        if observation.pane_dead:
            return AgentRuntimeState.DEAD
        if not observation.session_exists:
            return AgentRuntimeState.DEAD
        if observation.current_command in SHELL_COMMANDS:
            return AgentRuntimeState.DEAD
        if _claude_title_indicates_busy(observation.pane_title):
            return AgentRuntimeState.BUSY
        if _claude_title_indicates_ready(observation.pane_title):
            return AgentRuntimeState.READY
        if self._contains_any(visible_text or recent_log or text, CLAUDE_BUSY_PATTERNS):
            return AgentRuntimeState.BUSY
        if self._has_turn_token(text):
            return AgentRuntimeState.READY
        if self._contains_any(visible_text or recent_log or text, CLAUDE_READY_PATTERNS):
            return AgentRuntimeState.READY
        return AgentRuntimeState.BUSY if observation.current_command else AgentRuntimeState.STARTING

    def extract_last_message(self, output: str) -> str:
        clean_output = clean_ansi(output)
        lines = clean_output.splitlines()
        content_lines: list[str] = []
        for line in lines:
            normalized = line.strip()
            if normalized.startswith("❯"):
                continue
            content_lines.append(line)
        return super().extract_last_message("\n".join(content_lines))


class GeminiOutputDetector(BaseOutputDetector):
    def classify_agent_state(self, observation: WorkerObservation) -> AgentRuntimeState:
        visible_text = self.current_visible_text(observation)
        recent_log = self.recent_log_text(observation)
        text = self.observation_text(observation)
        base_state = super().classify_agent_state(observation)
        if base_state != AgentRuntimeState.BUSY:
            return base_state
        if self._contains_any(visible_text, GEMINI_TRUST_PROMPT_PATTERNS):
            return AgentRuntimeState.STARTING
        if self._contains_any(visible_text, GEMINI_NOT_READY_PATTERNS):
            return AgentRuntimeState.STARTING
        if self._contains_any(visible_text, GEMINI_BUSY_PATTERNS):
            return AgentRuntimeState.BUSY
        if self._contains_any(visible_text, GEMINI_INPUT_BOX_PATTERNS) or self._contains_any(
            visible_text,
            GEMINI_READY_PATTERNS,
        ):
            return AgentRuntimeState.READY
        if self._contains_any(recent_log or text, GEMINI_TRUST_PROMPT_PATTERNS):
            return AgentRuntimeState.STARTING
        if self._contains_any(recent_log or text, GEMINI_NOT_READY_PATTERNS):
            return AgentRuntimeState.STARTING
        if self._contains_any(recent_log or text, GEMINI_BUSY_PATTERNS):
            return AgentRuntimeState.BUSY
        if self._contains_any(recent_log or text, GEMINI_INPUT_BOX_PATTERNS) or self._contains_any(
            recent_log or text,
            GEMINI_READY_PATTERNS,
        ):
            return AgentRuntimeState.READY
        return AgentRuntimeState.STARTING if observation.current_command else AgentRuntimeState.DEAD

    def extract_last_message(self, output: str) -> str:
        clean_output = clean_ansi(output)
        lines = clean_output.splitlines()
        input_box_start = len(lines)
        for index in range(len(lines) - 1, -1, -1):
            normalized = lines[index].strip()
            if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in GEMINI_INPUT_BOX_PATTERNS):
                input_box_start = index
                break
        content_lines: list[str] = []
        for line in lines[:input_box_start]:
            normalized = line.strip()
            if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in GEMINI_FOOTER_PATTERNS):
                continue
            content_lines.append(line)
        return super().extract_last_message("\n".join(content_lines))


class OpenCodeOutputDetector(BaseOutputDetector):
    def classify_agent_state(self, observation: WorkerObservation) -> AgentRuntimeState:
        base_state = super().classify_agent_state(observation)
        if base_state != AgentRuntimeState.BUSY:
            return base_state
        return _classify_opencode_surface_state(
            visible_text=self.current_visible_text(observation),
            recent_log=self.recent_log_text(observation),
            current_command=observation.current_command,
        )

    def extract_last_message(self, output: str) -> str:
        clean_output = clean_ansi(output)
        lines: list[str] = []
        for line in clean_output.splitlines():
            normalized = line.strip()
            if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in OPENCODE_FOOTER_PATTERNS):
                continue
            if re.search(r"Thinking:", normalized, re.IGNORECASE):
                continue
            lines.append(line)
        return super().extract_last_message("\n".join(lines))


def build_output_detector(vendor: Vendor) -> BaseOutputDetector:
    if vendor == Vendor.CODEX:
        return CodexOutputDetector()
    if vendor == Vendor.CLAUDE:
        return ClaudeOutputDetector()
    if vendor == Vendor.GEMINI:
        return GeminiOutputDetector()
    if vendor == Vendor.OPENCODE:
        return OpenCodeOutputDetector()
    raise ValueError(f"不支持的厂商: {vendor}")


@dataclass(frozen=True)
class AgentRuntimeClassifierContext:
    vendor: Vendor
    agent_started: bool
    cached_state: AgentRuntimeState
    pane_id: str
    expected_current_commands: tuple[str, ...]
    task_running: bool = False
    pre_submit_ready_probe: bool = False
    title_ready: bool = False
    title_busy: bool = False


def _agent_command_running(current_command: str, expected_current_commands: Sequence[str]) -> bool:
    command = str(current_command or "").strip()
    if command in expected_current_commands:
        return True
    return bool(command) and command not in SHELL_COMMANDS


def classify_agent_runtime_state(
        observation: WorkerObservation,
        *,
        context: AgentRuntimeClassifierContext,
        detector: BaseOutputDetector,
) -> AgentRuntimeState:
    current_command = str(observation.current_command or "").strip()
    if not observation.session_exists or observation.pane_dead or not str(context.pane_id or "").strip():
        return AgentRuntimeState.DEAD
    if not context.agent_started and current_command in SHELL_COMMANDS:
        return AgentRuntimeState.STARTING
    if not _agent_command_running(current_command, context.expected_current_commands):
        return AgentRuntimeState.DEAD
    if not context.agent_started:
        if context.vendor == Vendor.CODEX:
            if context.title_ready:
                return AgentRuntimeState.READY
            if context.title_busy:
                return detector.classify_agent_state(observation)
        if context.title_ready:
            return AgentRuntimeState.READY
        if context.title_busy:
            return AgentRuntimeState.BUSY
        if context.vendor == Vendor.GEMINI:
            surface = "\n".join(
                part for part in (observation.visible_text, observation.raw_log_tail) if str(part or "").strip()
            )
            if surface.strip():
                surface_state = detector.classify_agent_state(observation)
                if surface_state in {AgentRuntimeState.READY, AgentRuntimeState.BUSY}:
                    return surface_state
        else:
            surface = "\n".join(
                part for part in (observation.visible_text, observation.raw_log_tail) if str(part or "").strip()
            )
        if context.vendor == Vendor.OPENCODE and (
                re.search(r"\bBuild\s*·", _normalize_opencode_surface(surface), re.IGNORECASE)
        ):
            surface_state = _classify_opencode_surface_state(
                visible_text=observation.visible_text,
                recent_log=observation.raw_log_tail,
                current_command=current_command,
            )
            if surface_state in {AgentRuntimeState.READY, AgentRuntimeState.BUSY}:
                return surface_state
        return AgentRuntimeState.STARTING
    if context.pre_submit_ready_probe and not context.task_running:
        if context.title_ready:
            return AgentRuntimeState.READY
        surface = "\n".join(
            part for part in (observation.visible_text, observation.raw_log_tail) if str(part or "").strip()
        )
        if context.vendor == Vendor.CODEX:
            if _codex_surface_indicates_ready_input(_codex_effective_recent_surface(surface)):
                return AgentRuntimeState.READY
        elif context.vendor == Vendor.CLAUDE:
            if (
                    not BaseOutputDetector._contains_any(surface, CLAUDE_BUSY_PATTERNS)
                    and BaseOutputDetector._contains_any(surface, CLAUDE_READY_PATTERNS)
            ):
                return AgentRuntimeState.READY
        elif context.vendor == Vendor.GEMINI:
            if (
                    not BaseOutputDetector._contains_any(surface, GEMINI_BUSY_PATTERNS)
                    and (
                        BaseOutputDetector._contains_any(surface, GEMINI_INPUT_BOX_PATTERNS)
                        or BaseOutputDetector._contains_any(surface, GEMINI_READY_PATTERNS)
                    )
            ):
                return AgentRuntimeState.READY
        elif context.vendor == Vendor.OPENCODE:
            surface_state = _classify_opencode_surface_state(
                visible_text=observation.visible_text,
                recent_log=observation.raw_log_tail,
                current_command=current_command,
            )
            if surface_state == AgentRuntimeState.READY:
                return AgentRuntimeState.READY
    if context.vendor == Vendor.CODEX:
        if context.title_busy:
            return AgentRuntimeState.BUSY
        if context.title_ready:
            detected_state = detector.classify_agent_state(observation)
            if context.agent_started and detected_state == AgentRuntimeState.STARTING:
                return AgentRuntimeState.BUSY
            return detected_state
    if context.title_ready:
        return AgentRuntimeState.READY
    if context.title_busy:
        return AgentRuntimeState.BUSY
    if (
            context.vendor == Vendor.OPENCODE
            and not (str(observation.visible_text or "").strip() or str(observation.raw_log_tail or "").strip())
            and context.cached_state in {AgentRuntimeState.READY, AgentRuntimeState.BUSY}
    ):
        return context.cached_state
    return detector.classify_agent_state(observation)


@dataclass(frozen=True)
class AgentRunConfig:
    vendor: Vendor
    model: str
    reasoning_effort: str = "high"
    proxy_url: str = ""
    extra_args: tuple[str, ...] = ()
    resolved_model: str = ""
    resolved_variant: str = ""
    reasoning_control_mode: str = ""
    catalog_source_kind: str = ""
    catalog_confidence: str = ""
    native_reasoning_level: str = ""
    supports_reasoning: bool = False
    resolution_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "vendor", normalize_vendor(self.vendor))
        object.__setattr__(self, "reasoning_effort", normalize_effort(self.reasoning_effort))
        object.__setattr__(self, "proxy_url", normalize_proxy_url(self.proxy_url))
        object.__setattr__(self, "model", str(self.model or "").strip())
        if not self.model:
            raise ValueError("model 不能为空")
        resolution = resolve_launch(self.vendor.value, self.model, self.reasoning_effort)
        object.__setattr__(self, "resolved_model", resolution.resolved_model)
        object.__setattr__(self, "resolved_variant", resolution.resolved_variant)
        object.__setattr__(self, "reasoning_control_mode", resolution.reasoning_control_mode)
        object.__setattr__(self, "catalog_source_kind", resolution.catalog_source_kind)
        object.__setattr__(self, "catalog_confidence", resolution.confidence)
        object.__setattr__(self, "native_reasoning_level", resolution.native_reasoning_level)
        object.__setattr__(self, "supports_reasoning", resolution.supports_reasoning)
        object.__setattr__(self, "resolution_notes", resolution.notes)

    def with_prompt_header(self, prompt: str) -> str:
        header = build_prompt_header(self.vendor, self.model, self.reasoning_effort)
        return f"{header}\n\n{str(prompt or '').strip()}".strip()

    def to_summary(self) -> dict[str, str]:
        resolution = resolve_launch(self.vendor.value, self.model, self.reasoning_effort)
        return {
            "vendor": self.vendor.value,
            "model": self.model,
            "resolved_model": self.resolved_model,
            "resolved_variant": self.resolved_variant,
            "reasoning_effort": self.reasoning_effort,
            "reasoning_control_mode": self.reasoning_control_mode,
            "catalog_source_kind": self.catalog_source_kind,
            "proxy_url": self.proxy_url,
            "reasoning_note": build_reasoning_note(self.vendor, self.reasoning_effort, model=self.model, resolution=resolution),
        }

    def expected_current_commands(self) -> tuple[str, ...]:
        return {
            Vendor.CODEX: ("codex", "node"),
            Vendor.CLAUDE: ("claude", "claude.exe", "node"),
            Vendor.GEMINI: ("gemini", "node"),
            Vendor.OPENCODE: ("opencode", "node"),
        }[self.vendor]

    def submit_enter_count(self) -> int:
        return 2 if self.vendor == Vendor.CODEX else 1

    def build_launch_command(self, work_dir: Path) -> str:
        resolution = resolve_launch(self.vendor.value, self.model, self.reasoning_effort)
        args: list[str] = []
        if self.vendor == Vendor.CODEX:
            args = [
                "codex",
                "--model",
                resolution.resolved_model,
                "--config",
                f'model_reasoning_effort="{resolution.native_reasoning_level or "high"}"',
                "--sandbox",
                "danger-full-access",
                "--ask-for-approval",
                "never",
                "--cd",
                str(work_dir),
                "--no-alt-screen",
            ]
        elif self.vendor == Vendor.CLAUDE:
            args = [
                "claude",
                "--model",
                resolution.resolved_model,
                "--permission-mode",
                "bypassPermissions",
                "--effort",
                resolution.native_reasoning_level or "high",
            ]
        elif self.vendor == Vendor.GEMINI:
            args = [
                "gemini",
                "--model",
                resolution.resolved_model,
                "--approval-mode",
                "yolo",
            ]
        elif self.vendor == Vendor.OPENCODE:
            args = [
                "opencode",
                str(work_dir),
                "--pure",
                "--model",
                resolution.resolved_model,
            ]
        else:
            raise ValueError(f"不支持的厂商: {self.vendor}")

        args.extend(self.extra_args)
        return " ".join(shlex.quote(item) for item in args)


@dataclass
class CommandResult:
    label: str
    command: str
    exit_code: int
    raw_output: str
    clean_output: str
    started_at: str
    finished_at: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


@dataclass
class WorkerResult:
    worker_id: str
    session_name: str
    pane_id: str
    runtime_dir: str
    work_dir: str
    config: dict[str, str]
    status: str
    commands: list[CommandResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["commands"] = [asdict(item) for item in self.commands]
        return payload


class TmuxBatchWorker:
    def __init__(
            self,
            *,
            worker_id: str,
            work_dir: str | Path,
            config: AgentRunConfig,
            runtime_root: str | Path | None = None,
            existing_runtime_dir: str | Path | None = None,
            existing_session_name: str = "",
            existing_pane_id: str = "",
            backend: TmuxBackend | None = None,
            launch_coordinator: LaunchCoordinator | None = None,
            runtime_metadata: Mapping[str, object] | None = None,
    ) -> None:
        reserved_session_name = ""
        self._session_name_reserved = False
        self.worker_id = str(worker_id or "").strip() or "worker"
        self.runtime_worker_id = _slugify(self.worker_id, max_len=48)
        self.work_dir = Path(work_dir).expanduser().resolve()
        if not self.work_dir.is_dir():
            raise FileNotFoundError(f"工作目录不存在: {self.work_dir}")
        self.config = config
        self.backend = backend or TmuxBackend()
        self.detector = build_output_detector(self.config.vendor)
        self.runtime_root = Path(runtime_root or DEFAULT_RUNTIME_ROOT).expanduser().resolve()
        self.launch_coordinator = launch_coordinator or LaunchCoordinator(self.runtime_root)
        if existing_runtime_dir:
            self.runtime_dir = Path(existing_runtime_dir).expanduser().resolve()
            self.runtime_dir.mkdir(parents=True, exist_ok=True)
            self.instance_id = self.runtime_dir.name.rsplit("-", 1)[
                -1] if "-" in self.runtime_dir.name else uuid.uuid4().hex[:8]
        else:
            self.instance_id = uuid.uuid4().hex[:8]
            self.runtime_dir = self.runtime_root / f"{self.runtime_worker_id}-{self.instance_id}"
            self.runtime_dir.mkdir(parents=True, exist_ok=True)
        if existing_session_name:
            self.session_name = existing_session_name
        else:
            reserved_session_name = _reserve_session_name(
                worker_id=self.worker_id,
                work_dir=self.work_dir,
                vendor=self.config.vendor,
                instance_id=self.instance_id,
                backend=self.backend,
            )
            self.session_name = reserved_session_name
        self.log_path = self.runtime_dir / "worker.log"
        self.raw_log_path = self.runtime_dir / "worker.raw.log"
        self.state_path = self.runtime_dir / "worker.state.json"
        self.transcript_path = self.runtime_dir / "transcript.md"
        self.pane_id = existing_pane_id
        self.send_lock = threading.Lock()
        self.state_lock = threading.RLock()
        self.results: list[CommandResult] = []
        self.health_supervisor: HealthSupervisor | None = None
        self.agent_ready = False
        self.agent_started = False
        self.recoverable = True
        self.last_reply = ""
        self.last_log_offset = 0
        self.last_pane_title = ""
        self.current_command = ""
        self.current_path = ""
        self.last_heartbeat_at = ""
        self.agent_state = AgentRuntimeState.STARTING
        self.wrapper_state = WrapperState.NOT_READY
        self.current_task_status_path = ""
        self.current_task_result_path = ""
        self.current_task_runtime_status = ""
        self.dispatch_state = ""
        self.dispatch_reason = ""
        self._runtime_metadata: dict[str, object] = {
            key: value
            for key, value in dict(runtime_metadata or {}).items()
            if str(key).strip()
        }
        self.last_terminal_signature = ""
        self.last_terminal_changed_at = ""
        self.terminal_recently_changed = False
        self._last_terminal_change_monotonic = 0.0
        self._last_boot_action_signature = ""
        self._last_boot_action_at = 0.0
        self.launch_command = self.config.build_launch_command(self.work_dir)
        if self.state_path.exists():
            existing_state = self.read_state()
            self.pane_id = existing_pane_id or str(existing_state.get("pane_id", self.pane_id))
            self.session_name = existing_session_name or str(existing_state.get("session_name", self.session_name))
            if reserved_session_name and self.session_name != reserved_session_name:
                _release_reserved_session_name(reserved_session_name)
                reserved_session_name = ""
            self.recoverable = bool(existing_state.get("recoverable", True))
            self.last_reply = str(existing_state.get("last_reply", ""))
            self.last_log_offset = int(existing_state.get("last_log_offset", 0))
            self.last_pane_title = str(existing_state.get("pane_title", ""))
            self.current_command = str(existing_state.get("current_command", ""))
            self.current_path = str(existing_state.get("current_path", ""))
            self.last_heartbeat_at = str(existing_state.get("last_heartbeat_at", ""))
            self.current_task_status_path = str(existing_state.get("current_task_status_path", ""))
            self.current_task_result_path = str(existing_state.get("current_task_result_path", ""))
            self.current_task_runtime_status = str(existing_state.get("current_task_runtime_status", ""))
            self.dispatch_state = str(existing_state.get("dispatch_state", ""))
            self.dispatch_reason = str(existing_state.get("dispatch_reason", ""))
            self.last_terminal_signature = str(existing_state.get("last_terminal_signature", ""))
            self.last_terminal_changed_at = str(existing_state.get("last_terminal_changed_at", ""))
            self.terminal_recently_changed = bool(existing_state.get("terminal_recently_changed", False))
            self.agent_started = bool(existing_state.get("agent_started", existing_state.get("agent_ready", False)))
            existing_agent_state = str(existing_state.get("agent_state", "")).strip().upper()
            if existing_agent_state in {state.value for state in AgentRuntimeState}:
                self.agent_state = AgentRuntimeState(existing_agent_state)
            self.agent_ready = self.agent_state == AgentRuntimeState.READY
            self.wrapper_state = WrapperState.READY if self.agent_ready else WrapperState.NOT_READY
            for key in ("project_dir", "requirement_name", "workflow_action", "stage_seq", "run_id"):
                if key in existing_state and key not in self._runtime_metadata:
                    self._runtime_metadata[key] = existing_state.get(key)
        self._session_name_reserved = bool(reserved_session_name)
        _register_live_worker(self)

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self._release_session_name_reservation()

    def _tmux(self, *args: str, input_text: str | None = None, timeout_sec: float = 10.0) -> \
    subprocess.CompletedProcess[str]:
        return self.backend.run(*args, input_text=input_text, timeout_sec=timeout_sec, check=True)

    def session_exists(self) -> bool:
        return _tmux_session_matches_context(
            self.backend,
            self.session_name,
            runtime_dir=self.runtime_dir,
            work_dir=self.work_dir,
            requirement_name=str(self._runtime_metadata.get("requirement_name", "") or "").strip(),
            workflow_action=str(self._runtime_metadata.get("workflow_action", "") or "").strip(),
        )

    def _set_tmux_identity_options(self) -> None:
        identity = {
            TMUX_IDENTITY_RUNTIME_DIR_OPTION: str(self.runtime_dir),
            TMUX_IDENTITY_WORK_DIR_OPTION: str(self.work_dir),
            TMUX_IDENTITY_REQUIREMENT_NAME_OPTION: str(self._runtime_metadata.get("requirement_name", "") or "").strip(),
            TMUX_IDENTITY_WORKFLOW_ACTION_OPTION: str(self._runtime_metadata.get("workflow_action", "") or "").strip(),
            TMUX_IDENTITY_WORKER_ID_OPTION: self.worker_id,
        }
        for option_name, option_value in identity.items():
            self._tmux("set-option", "-t", self.session_name, option_name, option_value)

    def attach_session(self) -> None:
        if not self.session_exists():
            raise RuntimeError(f"tmux session 不存在: {self.session_name}")
        self.backend.attach_session(self.session_name)

    def detach_session(self) -> str:
        if not self.session_name:
            raise RuntimeError("tmux 会话尚未创建")
        if not self.session_exists():
            raise RuntimeError(f"tmux 会话尚未创建: {self.session_name}")
        self.backend.detach_session(self.session_name)
        return self.session_name

    def show_transcript_tail(self, *, max_lines: int = 60) -> str:
        return read_text_tail(self.transcript_path, max_lines=max_lines)

    def read_state(self) -> dict[str, object]:
        with self.state_lock:
            if not self.state_path.exists():
                return {}
            raw_text = self.state_path.read_text(encoding="utf-8")
            try:
                payload = json.loads(raw_text)
                return payload if isinstance(payload, dict) else {}
            except Exception:
                text = str(raw_text or "").lstrip()
                if not text:
                    return {}
                try:
                    payload, _ = json.JSONDecoder().raw_decode(text)
                except Exception:
                    return {}
                if not isinstance(payload, dict):
                    return {}
                try:
                    _atomic_write_json(self.state_path, payload)
                except Exception:
                    pass
                return payload

    def runtime_metadata(self) -> dict[str, str]:
        payload = {
            "worker_id": self.worker_id,
            "runtime_worker_id": self.runtime_worker_id,
            "session_name": self.session_name,
            "pane_id": self.pane_id,
            "runtime_dir": str(self.runtime_dir),
            "work_dir": str(self.work_dir),
            "log_path": str(self.log_path),
            "raw_log_path": str(self.raw_log_path),
            "state_path": str(self.state_path),
            "transcript_path": str(self.transcript_path),
        }
        for key, value in self._runtime_metadata.items():
            payload[str(key)] = "" if value is None else str(value)
        return payload

    def set_runtime_metadata(self, **metadata: object) -> None:
        if not metadata:
            return
        normalized = {
            str(key): value
            for key, value in metadata.items()
            if str(key).strip()
        }
        self._runtime_metadata.update(normalized)
        if not self.state_path.exists():
            return
        with self.state_lock:
            payload = self.read_state()
            payload.update(normalized)
            _atomic_write_json(self.state_path, payload)
        _notify_runtime_state_changed_best_effort()

    def target_exists(self, target: str | None = None) -> bool:
        target_name = target or self.pane_id
        if not target_name:
            return False
        return self.backend.target_exists(target_name)

    def pane_current_command(self) -> str:
        return self.backend.display_message(self.pane_id, "#{pane_current_command}")

    def pane_current_path(self) -> str:
        return self.backend.display_message(self.pane_id, "#{pane_current_path}")

    def pane_title(self) -> str:
        return self.backend.display_message(self.pane_id, "#{pane_title}")

    def pane_dead(self) -> bool:
        return self.backend.display_message(self.pane_id, "#{pane_dead}") == "1"

    def capture_visible(self, tail_lines: int = DEFAULT_CAPTURE_TAIL_LINES) -> str:
        return self.backend.capture_visible(self.pane_id, tail_lines=tail_lines)

    def _diagnostic_visible_tail(self, tail_lines: int = 200) -> str:
        try:
            return clean_ansi(self.capture_visible(tail_lines))[-4000:]
        except Exception as error:  # noqa: BLE001
            return f"(unable to capture tmux pane: {error})"

    def _capture_pane_snapshot(self, *, tail_lines: int) -> tuple[bool, str, str, str, str, bool]:
        session_exists = self.session_exists()
        if not session_exists or not self.pane_id:
            return False, "", "", "", "", False
        try:
            if not self.target_exists():
                return False, "", "", "", "", False
            visible_text = clean_ansi(self.capture_visible(tail_lines))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False, "", "", "", "", False
        def _display_fallback(getter: Callable[[], str], previous: str = "") -> str:
            try:
                value = getter()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                value = ""
            return str(value or previous or "").strip()
        try:
            pane_dead = self.pane_dead()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pane_dead = False
        current_command = _display_fallback(self.pane_current_command, self.current_command)
        current_path = _display_fallback(self.pane_current_path, self.current_path)
        pane_title = _display_fallback(self.pane_title, self.last_pane_title)
        return True, visible_text, current_command, current_path, pane_title, pane_dead

    def _capture_pane_liveness_snapshot(self) -> tuple[bool, str, str, str, bool]:
        session_exists = self.session_exists()
        if not session_exists or not self.pane_id:
            return False, "", "", "", False
        try:
            if not self.target_exists():
                return False, "", "", "", False
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False, "", "", "", False

        def _display_fallback(getter: Callable[[], str], previous: str = "") -> str:
            try:
                value = getter()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                value = ""
            return str(value or previous or "").strip()

        try:
            pane_dead = self.pane_dead()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pane_dead = False
        current_command = _display_fallback(self.pane_current_command, self.current_command)
        current_path = _display_fallback(self.pane_current_path, self.current_path)
        pane_title = _display_fallback(self.pane_title, self.last_pane_title)
        return True, current_command, current_path, pane_title, pane_dead

    def _capture_lightweight_observation(self) -> WorkerObservation:
        observed_at = _now_iso()
        session_exists, current_command, current_path, pane_title, pane_dead = self._capture_pane_liveness_snapshot()
        self.last_pane_title = pane_title or self.last_pane_title
        self.current_command = current_command or self.current_command
        self.current_path = current_path or self.current_path
        self.last_heartbeat_at = observed_at
        return WorkerObservation(
            visible_text="",
            raw_log_delta="",
            raw_log_tail="",
            pane_title=pane_title,
            current_command=current_command,
            current_path=current_path,
            pane_dead=pane_dead,
            session_exists=session_exists,
            log_mtime=0.0,
            observed_at=observed_at,
        )

    def _capture_visible_observation_without_raw_log(self, *, tail_lines: int = DEFAULT_CAPTURE_TAIL_LINES) -> WorkerObservation:
        observed_at = _now_iso()
        session_exists, visible_text, current_command, current_path, pane_title, pane_dead = self._capture_pane_snapshot(
            tail_lines=tail_lines
        )
        self.last_pane_title = pane_title or self.last_pane_title
        self.current_command = current_command or self.current_command
        self.current_path = current_path or self.current_path
        self.last_heartbeat_at = observed_at
        terminal_surface = "\n".join(part for part in [pane_title, visible_text] if part)
        self._update_terminal_activity(terminal_surface, observed_at=observed_at)
        return WorkerObservation(
            visible_text=visible_text,
            raw_log_delta="",
            raw_log_tail="",
            pane_title=pane_title,
            current_command=current_command,
            current_path=current_path,
            pane_dead=pane_dead,
            session_exists=session_exists,
            log_mtime=0.0,
            observed_at=observed_at,
        )

    def _build_shell_bootstrap_command(self) -> str:
        env_parts = [
            f"{key}={shlex.quote(value)}"
            for key, value in build_proxy_env(self.config.proxy_url).items()
        ]
        shell_path = os.environ.get("SHELL", "/bin/zsh")
        if env_parts:
            return f"env {' '.join(env_parts)} {shlex.quote(shell_path)} -il"
        return f"{shlex.quote(shell_path)} -il"

    def _log_event(self, event: str, **payload: object) -> None:
        entry = {"at": _now_iso(), "event": event, **payload}
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _ensure_health_supervisor_started(self) -> None:
        if self.health_supervisor is not None and self.health_supervisor.is_alive():
            return
        if self.health_supervisor is not None:
            self.health_supervisor.stop()
            self.health_supervisor = None
        self.health_supervisor = HealthSupervisor(
            self._refresh_health_state_nonintrusive,
            interval_sec=2.0,
            ready_interval_sec=READY_HEALTH_INTERVAL_SEC,
            idle_interval_sec=IDLE_HEALTH_INTERVAL_SEC,
            idle_after_sec=IDLE_HEALTH_AFTER_SEC,
            thread_name=f"worker-health-{self.instance_id}",
        )
        self.health_supervisor.start()

    def _stop_health_supervisor(self) -> None:
        if self.health_supervisor is None:
            return
        supervisor = self.health_supervisor
        self.health_supervisor = None
        supervisor.stop()

    def _release_session_name_reservation(self) -> None:
        if not self._session_name_reserved:
            return
        _release_reserved_session_name(self.session_name)
        self._session_name_reserved = False

    def _abort_session_create_if_runtime_shutdown_requested(self, context: str) -> None:
        try:
            raise_if_runtime_shutdown_requested(context)
        except RuntimeShutdownRequested:
            with contextlib.suppress(Exception):
                self._stop_health_supervisor()
            if self.session_name:
                with contextlib.suppress(Exception):
                    self.backend.kill_session(self.session_name)
            self.pane_id = ""
            self.agent_ready = False
            self.agent_started = False
            self.wrapper_state = WrapperState.NOT_READY
            self.agent_state = AgentRuntimeState.DEAD
            self._release_session_name_reservation()
            raise

    def mark_awaiting_reconfiguration(self, *, reason_text: str) -> None:
        self.agent_ready = False
        self.agent_state = AgentRuntimeState.STARTING
        self.wrapper_state = WrapperState.NOT_READY
        self._write_state(
            WorkerStatus.RUNNING,
            note="awaiting_reconfig",
            extra={
                "result_status": "running",
                "health_status": "awaiting_reconfig",
                "health_note": str(reason_text or "").strip(),
                "agent_state": AgentRuntimeState.STARTING.value,
            },
        )

    def mark_provider_runtime_error(self, *, reason_text: str) -> None:
        reason = str(reason_text or "").strip()
        self.agent_ready = False
        self.agent_state = AgentRuntimeState.STARTING
        self.wrapper_state = WrapperState.NOT_READY
        self._write_state(
            WorkerStatus.RUNNING,
            note="provider_runtime_error",
            extra={
                "result_status": "running",
                "health_status": "provider_runtime_error",
                "health_note": reason,
                "last_provider_error": reason,
                "agent_state": AgentRuntimeState.STARTING.value,
            },
        )

    def _is_session_name_conflict_error(self, error: Exception) -> bool:
        try:
            if self.session_exists():
                return True
        except Exception:
            pass
        message = _exception_message(error).lower()
        return any(marker in message for marker in _DUPLICATE_SESSION_ERROR_MARKERS)

    def _reserve_conflict_retry_session_name(self, *, retry_count: int, conflict_source: str) -> None:
        old_session_name = self.session_name
        self._release_session_name_reservation()
        new_session_name = _reserve_session_name(
            worker_id=self.worker_id,
            work_dir=self.work_dir,
            vendor=self.config.vendor,
            instance_id=self.instance_id,
            backend=self.backend,
        )
        self.session_name = new_session_name
        self._session_name_reserved = True
        self._log_event(
            "session_name_conflict_retry",
            conflict_source=conflict_source,
            retry_count=retry_count,
            old_session_name=old_session_name,
            new_session_name=new_session_name,
        )

    def _raise_session_name_conflict(self, *, retries: int, conflict_source: str, error: Exception | None = None) -> None:
        conflict_key = f"{self.work_dir}::{self.worker_id}"
        message = (
            "tmux session 命名冲突，自动重试已达上限: "
            f"conflict_key={conflict_key}, session_name={self.session_name}, "
            f"retries={retries}, source={conflict_source}"
        )
        if error is None:
            raise RuntimeError(message)
        raise RuntimeError(message) from error

    def create_session(self) -> str:
        raise_if_runtime_shutdown_requested("creating tmux session")
        self._reset_terminal_activity()
        self.agent_started = False
        self.agent_ready = False
        self.wrapper_state = WrapperState.NOT_READY
        self.last_pane_title = ""
        self.current_task_status_path = ""
        self.current_task_result_path = ""
        self.current_task_runtime_status = ""
        self._stop_health_supervisor()
        retry_count = 0
        max_retries = SESSION_NAME_CREATE_MAX_RETRIES
        while True:
            raise_if_runtime_shutdown_requested("creating tmux session")
            if self.session_exists():
                if retry_count >= max_retries:
                    self._release_session_name_reservation()
                    self._raise_session_name_conflict(retries=retry_count, conflict_source="pre-check")
                retry_count += 1
                self._reserve_conflict_retry_session_name(
                    retry_count=retry_count,
                    conflict_source="pre-check",
                )
                continue
            try:
                self.backend.run(
                    "set-option",
                    "-g",
                    "history-limit",
                    str(TMUX_HISTORY_LIMIT_LINES),
                    check=False,
                )
                self.pane_id = self.backend.create_session(
                    self.session_name,
                    self.work_dir,
                    self._build_shell_bootstrap_command(),
                )
                self._abort_session_create_if_runtime_shutdown_requested("creating tmux session")
                break
            except Exception as error:
                if not self._is_session_name_conflict_error(error):
                    self._release_session_name_reservation()
                    raise
                if retry_count >= max_retries:
                    self._release_session_name_reservation()
                    self._raise_session_name_conflict(
                        retries=retry_count,
                        conflict_source="create-failure",
                        error=error,
                    )
                retry_count += 1
                self._reserve_conflict_retry_session_name(
                    retry_count=retry_count,
                    conflict_source="create-failure",
                )
        self._release_session_name_reservation()
        self._tmux("set-option", "-t", self.session_name, "history-limit", str(TMUX_HISTORY_LIMIT_LINES))
        self._tmux("set-option", "-t", self.session_name, "allow-rename", "off")
        self._tmux("set-window-option", "-t", f"{self.session_name}:0", "automatic-rename", "off")
        self._set_tmux_identity_options()
        self._abort_session_create_if_runtime_shutdown_requested("creating tmux session")
        self._log_event(
            "session_created",
            pane_id=self.pane_id,
            session_name=self.session_name,
            session_name_retry_count=retry_count,
        )
        self._write_session_created_state_fast()
        self._start_pipe_logging()
        self._ensure_health_supervisor_started()
        return self.pane_id

    def _start_pipe_logging(self) -> None:
        self.log_path.write_text("", encoding="utf-8")
        self.raw_log_path.write_text("", encoding="utf-8")
        self.backend.pipe_log(self.pane_id, self.raw_log_path)
        self._log_event("pipe_log_started", raw_log_path=str(self.raw_log_path))

    def tail_raw_log(self, *, tail_bytes: int = 24000) -> tuple[str, str, int, float]:
        delta, tail, next_offset, log_mtime = self.backend.tail_raw_log(
            self.raw_log_path,
            last_offset=self.last_log_offset,
            tail_bytes=tail_bytes,
        )
        self.last_log_offset = next_offset
        return delta, tail, next_offset, log_mtime

    def observe(self, *, tail_lines: int = DEFAULT_CAPTURE_TAIL_LINES, tail_bytes: int = 24000) -> WorkerObservation:
        observed_at = _now_iso()
        session_exists, visible_text, current_command, current_path, pane_title, pane_dead = self._capture_pane_snapshot(
            tail_lines=tail_lines
        )
        raw_log_delta, raw_log_tail, _, log_mtime = self.tail_raw_log(tail_bytes=tail_bytes)
        self.last_pane_title = pane_title or self.last_pane_title
        self.current_command = current_command or self.current_command
        self.current_path = current_path or self.current_path
        self.last_heartbeat_at = observed_at
        terminal_surface = "\n".join(part for part in [pane_title, visible_text or clean_ansi(raw_log_tail)] if part)
        self._update_terminal_activity(terminal_surface, observed_at=observed_at)
        observation = WorkerObservation(
            visible_text=visible_text,
            raw_log_delta=clean_ansi(raw_log_delta),
            raw_log_tail=clean_ansi(raw_log_tail),
            pane_title=pane_title,
            current_command=current_command,
            current_path=current_path,
            pane_dead=pane_dead,
            session_exists=session_exists,
            log_mtime=log_mtime,
            observed_at=observed_at,
        )
        self.agent_state = self.get_agent_state(observation)
        self.wrapper_state = self._infer_wrapper_state(
            current_command=observation.current_command,
            visible_text=observation.visible_text,
            raw_log_tail=observation.raw_log_tail,
        )
        return observation

    def _write_session_created_state_fast(self) -> None:
        observed_at = _now_iso()
        with self.state_lock:
            previous = self.read_state()
            payload: dict[str, object] = {
                "worker_id": self.worker_id,
                "runtime_worker_id": self.runtime_worker_id,
                "session_name": self.session_name,
                "pane_id": self.pane_id,
                "work_dir": str(self.work_dir),
                "status": WorkerStatus.READY.value,
                "note": "session_created",
                "updated_at": observed_at,
                "config": self.config.to_summary(),
                "log_path": str(self.log_path),
                "raw_log_path": str(self.raw_log_path),
                "transcript_path": str(self.transcript_path),
                "agent_ready": False,
                "agent_started": False,
                "agent_alive": False,
                "agent_state": AgentRuntimeState.STARTING.value,
                "last_reply": self.last_reply,
                "state_revision": int(previous.get("state_revision", 0)) + 1,
                "last_writer": "TmuxBatchWorker",
                "workflow_stage": str(previous.get("workflow_stage", "pending")),
                "workflow_round": int(previous.get("workflow_round", 0)),
                "health_status": "alive",
                "health_note": "session_created",
                "retry_count": int(previous.get("retry_count", 0)),
                "last_log_offset": self.last_log_offset,
                "auto_recovery_mode": str(previous.get("auto_recovery_mode", "standard")),
                "recoverable": self.recoverable,
                "result_status": str(previous.get("result_status", WorkerStatus.READY.value)),
                "pane_title": self.last_pane_title,
                "current_command": self.current_command,
                "current_path": self.current_path or str(self.work_dir),
                "last_turn_token": str(previous.get("last_turn_token", "")),
                "last_prompt_hash": str(previous.get("last_prompt_hash", "")),
                "last_heartbeat_at": observed_at,
                "current_turn_id": str(previous.get("current_turn_id", "")),
                "current_turn_phase": str(previous.get("current_turn_phase", "")),
                "current_turn_status_path": str(previous.get("current_turn_status_path", "")),
                "current_task_status_path": self.current_task_status_path or str(previous.get("current_task_status_path", "")),
                "current_task_result_path": self.current_task_result_path or str(previous.get("current_task_result_path", "")),
                "current_task_runtime_status": self.current_task_runtime_status or str(previous.get("current_task_runtime_status", "")),
                "dispatch_state": self.dispatch_state or str(previous.get("dispatch_state", "")),
                "dispatch_reason": self.dispatch_reason or str(previous.get("dispatch_reason", "")),
                "last_terminal_signature": self.last_terminal_signature,
                "last_terminal_changed_at": self.last_terminal_changed_at,
                "terminal_recently_changed": self.terminal_recently_changed,
            }
            payload.update(self._runtime_metadata)
            _atomic_write_json(self.state_path, payload)
        self._log_event("state_changed", status=WorkerStatus.READY.value, note="session_created")
        _notify_runtime_state_changed_best_effort()

    def _write_state(self, status: WorkerStatus, *, note: str, extra: dict[str, object] | None = None) -> None:
        with self.state_lock:
            previous = self.read_state()
            agent_alive = self.is_agent_alive()
            agent_state = self.get_agent_state().value
            extra_payload = dict(extra or {})
            result_status = str(previous.get("result_status", "pending"))
            current_task_runtime_status = self.current_task_runtime_status or str(previous.get("current_task_runtime_status", ""))
            effective_result_status = str(extra_payload.get("result_status", result_status) or "").strip().lower()
            effective_runtime_status = str(
                extra_payload.get("current_task_runtime_status", current_task_runtime_status) or ""
            ).strip().lower()
            completed_success = status == WorkerStatus.SUCCEEDED or effective_result_status in COMPLETED_WORKER_RESULT_STATUSES
            completed_runtime = effective_runtime_status == TASK_STATUS_DONE
            if status == WorkerStatus.READY and agent_state == AgentRuntimeState.READY.value:
                if "result_status" not in extra_payload and result_status in {"running", "pending"}:
                    result_status = WorkerStatus.READY.value
                if "current_task_runtime_status" not in extra_payload and current_task_runtime_status == TASK_STATUS_RUNNING:
                    current_task_runtime_status = ""
                    self.current_task_runtime_status = ""
            if completed_success:
                result_status = str(extra_payload.get("result_status", WorkerStatus.SUCCEEDED.value) or WorkerStatus.SUCCEEDED.value)
                current_task_runtime_status = str(
                    extra_payload.get("current_task_runtime_status", current_task_runtime_status) or ""
                ).strip()
                if not current_task_runtime_status or current_task_runtime_status == TASK_STATUS_RUNNING:
                    current_task_runtime_status = TASK_STATUS_DONE
                self.current_task_runtime_status = current_task_runtime_status
            elif completed_runtime:
                current_task_runtime_status = TASK_STATUS_DONE
                self.current_task_runtime_status = TASK_STATUS_DONE
            if completed_success or completed_runtime:
                agent_state = AgentRuntimeState.READY.value
                self.agent_state = AgentRuntimeState.READY
                self.agent_ready = True
                self.agent_started = True
                extra_payload["agent_ready"] = True
                extra_payload["agent_started"] = True
                extra_payload["agent_state"] = AgentRuntimeState.READY.value
                extra_payload["current_task_runtime_status"] = current_task_runtime_status
                if completed_success:
                    extra_payload.setdefault("result_status", result_status)
            payload: dict[str, object] = {
                "worker_id": self.worker_id,
                "runtime_worker_id": self.runtime_worker_id,
                "session_name": self.session_name,
                "pane_id": self.pane_id,
                "work_dir": str(self.work_dir),
                "status": status.value,
                "note": note,
                "updated_at": _now_iso(),
                "config": self.config.to_summary(),
                "log_path": str(self.log_path),
                "raw_log_path": str(self.raw_log_path),
                "transcript_path": str(self.transcript_path),
                "agent_ready": agent_state == AgentRuntimeState.READY.value,
                "agent_started": self.agent_started,
                "agent_alive": agent_alive,
                "agent_state": agent_state,
                "last_reply": self.last_reply,
                "state_revision": int(previous.get("state_revision", 0)) + 1,
                "last_writer": "TmuxBatchWorker",
                "workflow_stage": str(previous.get("workflow_stage", "pending")),
                "workflow_round": int(previous.get("workflow_round", 0)),
                "health_status": str(previous.get("health_status", "unknown")),
                "health_note": str(previous.get("health_note", "")),
                "retry_count": int(previous.get("retry_count", 0)),
                "last_log_offset": self.last_log_offset,
                "auto_recovery_mode": str(previous.get("auto_recovery_mode", "standard")),
                "recoverable": self.recoverable,
                "result_status": result_status,
                "pane_title": self.last_pane_title,
                "current_command": self.current_command or str(previous.get("current_command", "")),
                "current_path": self.current_path or str(previous.get("current_path", "")),
                "last_turn_token": str(previous.get("last_turn_token", "")),
                "last_prompt_hash": str(previous.get("last_prompt_hash", "")),
                "last_heartbeat_at": self.last_heartbeat_at or str(previous.get("last_heartbeat_at", "")),
                "current_turn_id": str(previous.get("current_turn_id", "")),
                "current_turn_phase": str(previous.get("current_turn_phase", "")),
                "current_turn_status_path": str(previous.get("current_turn_status_path", "")),
                "current_task_status_path": self.current_task_status_path or str(previous.get("current_task_status_path", "")),
                "current_task_result_path": self.current_task_result_path or str(previous.get("current_task_result_path", "")),
                "current_task_runtime_status": current_task_runtime_status,
                "dispatch_state": self.dispatch_state or str(previous.get("dispatch_state", "")),
                "dispatch_reason": self.dispatch_reason or str(previous.get("dispatch_reason", "")),
                "last_terminal_signature": self.last_terminal_signature,
                "last_terminal_changed_at": self.last_terminal_changed_at,
                "terminal_recently_changed": self.terminal_recently_changed,
            }
            for key in ("project_dir", "requirement_name", "workflow_action", "stage_seq", "run_id"):
                if key in previous and key not in self._runtime_metadata:
                    self._runtime_metadata[key] = previous.get(key)
            payload.update(self._runtime_metadata)
            if extra_payload:
                payload.update(extra_payload)
            _atomic_write_json(self.state_path, payload)
        self._log_event("state_changed", status=status.value, note=note)
        _notify_runtime_state_changed_best_effort()

    def _capture_passive_observation(self, *, tail_lines: int = 120) -> WorkerObservation:
        if self.config.vendor == Vendor.OPENCODE and self.agent_state == AgentRuntimeState.BUSY:
            return self.observe(tail_lines=tail_lines, tail_bytes=12000)
        observation = self._capture_lightweight_observation()
        if self._should_capture_visible_for_passive_health(observation):
            return self._capture_visible_observation_without_raw_log(tail_lines=tail_lines)
        return observation

    def _should_capture_visible_for_passive_health(self, observation: WorkerObservation) -> bool:
        if not observation.session_exists or observation.pane_dead:
            return False
        current_command = observation.current_command or self.current_command
        if not self._agent_running(current_command):
            return False
        if self.config.vendor == Vendor.OPENCODE:
            title = str(observation.pane_title or "").strip()
            return (
                    not self.agent_started
                    or self.agent_state in {AgentRuntimeState.BUSY, AgentRuntimeState.STARTING}
                    or title == "OpenCode"
            )
        if self.config.vendor == Vendor.GEMINI:
            return not self.agent_started or self.agent_state in {AgentRuntimeState.BUSY, AgentRuntimeState.STARTING}
        if not self.agent_started:
            return False
        if self.config.vendor == Vendor.CODEX:
            if self._title_indicates_busy(observation.pane_title):
                return False
            return (
                self.agent_state == AgentRuntimeState.BUSY
                or self.current_task_runtime_status == TASK_STATUS_RUNNING
                or self.dispatch_state == "submitted"
            )
        if self.config.vendor == Vendor.CLAUDE:
            if self._title_indicates_ready(observation.pane_title):
                return False
            title = str(observation.pane_title or "").strip()
            if title.startswith("✳ "):
                return True
            return self.agent_state in {AgentRuntimeState.BUSY, AgentRuntimeState.STARTING}
        return False

    def _build_passive_health_snapshot(self, observation: WorkerObservation | None = None) -> WorkerHealthSnapshot:
        passive_observation = observation or self._capture_passive_observation()
        observed_at = passive_observation.observed_at
        session_exists = passive_observation.session_exists
        current_command = passive_observation.current_command
        current_path = passive_observation.current_path
        pane_dead = passive_observation.pane_dead
        if not session_exists:
            if self._is_prelaunch_without_session():
                self.agent_state = AgentRuntimeState.STARTING
                return WorkerHealthSnapshot(
                    session_exists=False,
                    health_status="unknown",
                    health_note="launch pending",
                    agent_state=AgentRuntimeState.STARTING.value,
                    pane_title=self.last_pane_title,
                    last_heartbeat_at=observed_at,
                    last_log_offset=self.last_log_offset,
                    current_command=self.current_command,
                    current_path=self.current_path,
                    pane_id=self.pane_id,
                    session_name=self.session_name,
                )
            self.agent_state = AgentRuntimeState.DEAD
            return WorkerHealthSnapshot(
                session_exists=False,
                health_status="missing_session",
                health_note="missing_session",
                agent_state=AgentRuntimeState.DEAD.value,
                pane_title=self.last_pane_title,
                last_heartbeat_at=observed_at,
                last_log_offset=self.last_log_offset,
                current_command=self.current_command,
                current_path=self.current_path,
                pane_id=self.pane_id,
                session_name=self.session_name,
            )
        observed_agent_state = self.get_agent_state(passive_observation)
        if (
                self.current_task_runtime_status == TASK_STATUS_RUNNING
                and self.dispatch_state == "submitted"
                and self.agent_started
                and self.is_agent_alive(passive_observation)
                and observed_agent_state != AgentRuntimeState.READY
        ):
            agent_state = AgentRuntimeState.BUSY
        else:
            agent_state = observed_agent_state
        if self.current_task_runtime_status == TASK_STATUS_DONE:
            agent_state = AgentRuntimeState.READY
        if agent_state == AgentRuntimeState.READY:
            if self.current_task_runtime_status == TASK_STATUS_RUNNING:
                self.current_task_runtime_status = ""
            if self.dispatch_state == "submitted":
                self.dispatch_state = ""
                self.dispatch_reason = ""
        self.agent_state = agent_state
        if agent_state in {AgentRuntimeState.READY, AgentRuntimeState.BUSY}:
            self.agent_started = True
        self.agent_ready = agent_state == AgentRuntimeState.READY
        health_status = "pane_dead" if pane_dead else "alive"
        health_note = "pane_dead" if pane_dead else "alive"
        return WorkerHealthSnapshot(
            session_exists=True,
            health_status=health_status,
            health_note=health_note,
            agent_state=agent_state.value,
            pane_title=passive_observation.pane_title,
            last_heartbeat_at=observed_at,
            last_log_offset=self.last_log_offset,
            current_command=current_command or self.current_command,
            current_path=current_path or self.current_path,
            pane_id=self.pane_id,
            session_name=self.session_name,
        )

    def _refresh_health_state_nonintrusive(self, *, notify_on_change: bool = True) -> WorkerHealthSnapshot:
        observation = self._capture_passive_observation()
        snapshot = self._build_passive_health_snapshot(observation)
        health_changed = False
        with self.state_lock:
            previous = self.read_state()
            previous_completed = _worker_state_payload_has_completed_status(previous)
            if previous_completed and snapshot.agent_state in {
                AgentRuntimeState.BUSY.value,
                AgentRuntimeState.STARTING.value,
            }:
                self.agent_state = AgentRuntimeState.READY
                self.agent_ready = True
                self.agent_started = True
                if str(previous.get("current_task_runtime_status", "") or "").strip().lower() == TASK_STATUS_DONE:
                    self.current_task_runtime_status = TASK_STATUS_DONE
                snapshot = WorkerHealthSnapshot(
                    session_exists=snapshot.session_exists,
                    health_status=snapshot.health_status,
                    health_note=snapshot.health_note,
                    last_heartbeat_at=snapshot.last_heartbeat_at,
                    last_log_offset=snapshot.last_log_offset,
                    current_command=snapshot.current_command,
                    current_path=snapshot.current_path,
                    pane_id=snapshot.pane_id,
                    session_name=snapshot.session_name,
                    agent_state=AgentRuntimeState.READY.value,
                    pane_title=snapshot.pane_title,
                )
            clear_stale_runtime_markers = (
                snapshot.agent_state == AgentRuntimeState.READY.value
                and not previous_completed
                and (
                    str(previous.get("current_task_runtime_status", "")) == TASK_STATUS_RUNNING
                    or str(previous.get("dispatch_state", "")) == "submitted"
                )
            )
            health_changed = (
                str(previous.get("health_status", "unknown")) != snapshot.health_status
                or str(previous.get("health_note", "")) != snapshot.health_note
                or str(previous.get("agent_state", "")) != snapshot.agent_state
                or str(previous.get("current_command", "")) != snapshot.current_command
                or str(previous.get("current_path", "")) != snapshot.current_path
                or str(previous.get("pane_title", "")) != snapshot.pane_title
            )
            normalize_completed_runtime_markers = previous_completed and (
                str(previous.get("agent_state", "")).strip().upper() != AgentRuntimeState.READY.value
                or str(previous.get("dispatch_state", "")).strip() == "submitted"
                or str(previous.get("current_task_runtime_status", "") or "").strip().lower() in {"", TASK_STATUS_RUNNING}
            )
            if health_changed or clear_stale_runtime_markers or normalize_completed_runtime_markers:
                payload = dict(previous)
                payload.update(self.runtime_metadata())
                payload.update(
                    {
                        "config": self.config.to_summary(),
                        "agent_alive": self.is_agent_alive(observation),
                        "agent_started": self.agent_started,
                        "agent_ready": snapshot.agent_state == AgentRuntimeState.READY.value,
                        "agent_state": snapshot.agent_state,
                        "health_status": snapshot.health_status,
                        "health_note": snapshot.health_note,
                        "pane_title": snapshot.pane_title,
                        "current_command": snapshot.current_command,
                        "current_path": snapshot.current_path,
                        "updated_at": snapshot.last_heartbeat_at or _now_iso(),
                        "last_heartbeat_at": snapshot.last_heartbeat_at,
                    }
                )
                if clear_stale_runtime_markers:
                    payload["current_task_runtime_status"] = ""
                    payload["dispatch_state"] = ""
                    payload["dispatch_reason"] = ""
                    if str(payload.get("result_status", "")) in {"running", "pending"}:
                        payload["result_status"] = WorkerStatus.READY.value
                    if str(payload.get("status", "")) in {"running", "pending"}:
                        payload["status"] = WorkerStatus.READY.value
                if previous_completed:
                    payload["agent_ready"] = True
                    payload["agent_started"] = True
                    payload["agent_state"] = AgentRuntimeState.READY.value
                    payload["dispatch_state"] = ""
                    payload["dispatch_reason"] = ""
                    if str(payload.get("current_task_runtime_status", "") or "").strip().lower() in {"", TASK_STATUS_RUNNING}:
                        payload["current_task_runtime_status"] = TASK_STATUS_DONE
                _atomic_write_json(self.state_path, payload)
        if health_changed and notify_on_change:
            _notify_runtime_state_changed_best_effort()
        return snapshot

    def request_restart(self) -> str:
        if self.session_exists():
            self._stop_health_supervisor()
            self.backend.kill_session(self.session_name)
        self.agent_ready = False
        self.agent_started = False
        self.wrapper_state = WrapperState.NOT_READY
        self.recoverable = True
        self.agent_state = AgentRuntimeState.STARTING
        self._reset_terminal_activity()
        self.last_pane_title = ""
        self.current_task_status_path = ""
        self.current_task_result_path = ""
        self.current_task_runtime_status = ""
        self._log_event("manual_restart_requested", session_name=self.session_name)
        return self.session_name

    def request_kill(self) -> str:
        if self.session_exists():
            self._stop_health_supervisor()
            self.backend.kill_session(self.session_name)
        self.agent_ready = False
        self.agent_started = False
        self.wrapper_state = WrapperState.NOT_READY
        self.recoverable = False
        self.agent_state = AgentRuntimeState.DEAD
        self._reset_terminal_activity()
        self.last_pane_title = ""
        self.current_task_status_path = ""
        self.current_task_result_path = ""
        self.current_task_runtime_status = ""
        self._log_event("manual_kill_requested", session_name=self.session_name)
        return self.session_name

    def refresh_health(
            self,
            *,
            auto_relaunch: bool = False,
            relaunch_timeout_sec: float = 30.0,
            notify_on_change: bool = True,
    ) -> WorkerHealthSnapshot:
        _ = relaunch_timeout_sec
        if auto_relaunch:
            self._log_event("auto_relaunch_blocked", reason="manual_intervention_required")
        return self._refresh_health_state_nonintrusive(notify_on_change=notify_on_change)

    def _append_transcript(self, title: str, body: str) -> None:
        with self.transcript_path.open("a", encoding="utf-8") as file:
            file.write(f"## {title}\n\n{body.rstrip()}\n\n")

    def _record_result(self, result: CommandResult, *, status: WorkerStatus, note: str,
                       extra: dict[str, object] | None = None) -> None:
        extra_payload = dict(extra or {})
        if status == WorkerStatus.SUCCEEDED:
            self.dispatch_state = ""
            self.dispatch_reason = ""
            extra_payload.setdefault("dispatch_state", "")
            extra_payload.setdefault("dispatch_reason", "")
        self.results.append(result)
        self._write_state(status, note=note, extra=extra_payload)
        self._append_transcript(f"{result.label} / output", f"```text\n{result.clean_output}\n```")

    def send_special_key(self, key: str) -> None:
        with self.send_lock:
            self.backend.send_key(self.pane_id, key)
        self._log_event("send_key", key=key)

    def _send_text(self, text: str, enter_count: int | None = None) -> None:
        raise_if_runtime_shutdown_requested("sending prompt to tmux worker")
        submit_count = enter_count if enter_count is not None else self.config.submit_enter_count()
        with self.send_lock:
            raise_if_runtime_shutdown_requested("sending prompt to tmux worker")
            self.backend.send_text(self.pane_id, text, submit_count=submit_count)
        self._log_event("send_text", submit_count=submit_count, size=len(text))

    @staticmethod
    def _normalize_prompt_text(prompt: str) -> str:
        return clean_ansi(str(prompt or "")).strip()

    @classmethod
    def _source_mentions_prompt(cls, source: str, prompt: str) -> bool:
        prompt_text = cls._normalize_prompt_text(prompt)
        if not prompt_text:
            return False
        source_text = cls._normalize_prompt_text(source)
        if not source_text:
            return False
        if prompt_text in source_text:
            return True

        source_flat = re.sub(r"\s+", " ", source_text).strip()
        prompt_flat = re.sub(r"\s+", " ", prompt_text).strip()
        if prompt_flat and prompt_flat in source_flat:
            return True

        # Long TUI prompts are often hard-wrapped by interactive CLIs, so the
        # full string is not recoverable from tmux output. Multiple stable
        # fragments are enough to prove that this prompt, not stale footer text,
        # reached the pane.
        fragments: list[str] = []
        for line in prompt_text.splitlines():
            line_flat = re.sub(r"\s+", " ", line).strip()
            if len(line_flat) < 16 or line_flat.startswith("[[ACX_TURN:"):
                continue
            fragments.append(line_flat[:48])
            if len(line_flat) > 96:
                fragments.append(line_flat[-48:])
        unique_fragments = list(dict.fromkeys(fragments))
        if len(unique_fragments) < 3:
            return False
        matched = 0
        for fragment in unique_fragments:
            if fragment in source_text or fragment in source_flat:
                matched += 1
                if matched >= 3:
                    return True
        return False

    @classmethod
    def _source_mentions_prompt_submission_marker(cls, source: str, prompt: str) -> bool:
        source_text = cls._normalize_prompt_text(source)
        if not source_text:
            return False
        prompt_text = cls._normalize_prompt_text(prompt)
        for marker in re.findall(r"\[\[ACX_TURN:[^\]]+:DONE\]\]", prompt_text):
            if marker and marker in source_text:
                return True
        return False

    def _wait_for_prompt_submission(
            self,
            *,
            prompt: str,
            timeout_sec: float,
    ) -> WorkerObservation:
        deadline = time.monotonic() + timeout_sec
        extra_enter_sent = False
        submit_started_at = time.monotonic()
        submission_observed = False
        initial_state = self.agent_state

        while time.monotonic() < deadline:
            raise_if_runtime_shutdown_requested("waiting for prompt submission")
            observation = self.observe(tail_lines=320)
            if not observation.session_exists:
                raise RuntimeError("tmux pane exited while waiting for prompt submission")
            if observation.pane_dead:
                raise RuntimeError(f"tmux pane died after sending prompt:\n{self.capture_visible(160)}")

            current_command = observation.current_command
            if current_command in SHELL_COMMANDS:
                self.agent_ready = False
                raise RuntimeError(f"agent exited back to shell after prompt submission:\n{observation.visible_text}")

            prompt_visible = self._source_mentions_prompt(observation.visible_text, prompt)
            prompt_in_delta = self._source_mentions_prompt(observation.raw_log_delta, prompt)
            marker_visible = self._source_mentions_prompt_submission_marker(observation.visible_text, prompt)
            marker_in_delta = self._source_mentions_prompt_submission_marker(observation.raw_log_delta, prompt)
            current_state = self.get_agent_state(observation)
            busy_transition = initial_state == AgentRuntimeState.READY and current_state == AgentRuntimeState.BUSY
            submission_observed = (
                submission_observed
                or prompt_visible
                or prompt_in_delta
                or marker_visible
                or marker_in_delta
                or busy_transition
            )

            if submission_observed and current_state in {AgentRuntimeState.READY, AgentRuntimeState.BUSY}:
                self.current_command = current_command
                self.current_path = observation.current_path
                self.last_heartbeat_at = observation.observed_at
                self._log_event("prompt_submitted", agent_state=current_state.value)
                return observation

            if (
                    not submission_observed
                    and not extra_enter_sent
                    and time.monotonic() - submit_started_at >= 3.0
                    and current_state in {AgentRuntimeState.READY, AgentRuntimeState.STARTING}
            ):
                self.send_special_key("Enter")
                extra_enter_sent = True
                self._log_event("prompt_extra_enter", agent_state=current_state.value)

            time.sleep(0.5)

        raise TimeoutError(f"等待智能体确认收到 prompt 超时:\n{self._diagnostic_visible_tail(200)}")

    def wait_for_turn_artifacts(
            self,
            *,
            contract: TurnFileContract,
            task_status_path: Path | None = None,
            timeout_sec: float,
    ) -> TurnFileResult:
        deadline = time.monotonic() + timeout_sec
        stable_signature: tuple[object, ...] | None = None
        stable_since_monotonic = 0.0
        invalid_signature: tuple[object, ...] | None = None
        invalid_since_monotonic = 0.0
        status_done_seen = task_status_path is None
        post_done_since_monotonic = time.monotonic() if status_done_seen else 0.0
        post_done_grace_sec = max(float(contract.quiet_window_sec), TURN_ARTIFACT_POST_DONE_GRACE_SEC)
        last_probe_monotonic = 0.0

        while time.monotonic() < deadline:
            raise_if_runtime_shutdown_requested("waiting for turn artifacts")
            previous_done_seen = status_done_seen
            status_done_seen = self._track_task_completion_signal(
                task_status_path=task_status_path,
                status_done_seen=status_done_seen,
            )
            if status_done_seen and not previous_done_seen and not post_done_since_monotonic:
                post_done_since_monotonic = time.monotonic()

            try:
                file_result = contract.validator(contract.status_path)
                validate_turn_file_artifact_rules(contract, file_result)
            except Exception as error:
                stable_signature = None
                stable_since_monotonic = 0.0
                invalid_state = "missing"
                if contract.status_path.exists():
                    try:
                        status_stat = contract.status_path.stat()
                        invalid_state = (
                            status_stat.st_size,
                            status_stat.st_mtime,
                        )
                    except OSError:
                        invalid_state = "stat_error"
                current_invalid_signature = (
                    invalid_state,
                    type(error).__name__,
                    str(error).strip(),
                )
                if current_invalid_signature == invalid_signature:
                    invalid_elapsed = (
                        time.monotonic() - invalid_since_monotonic
                        if invalid_since_monotonic
                        else 0.0
                    )
                else:
                    invalid_signature = current_invalid_signature
                    invalid_since_monotonic = time.monotonic()
                    invalid_elapsed = 0.0
                if status_done_seen:
                    if invalid_elapsed >= max(post_done_grace_sec, 0.0):
                        raise RuntimeError(
                            f"{TURN_ARTIFACT_CONTRACT_ERROR_PREFIX}: "
                            f"phase={contract.phase} status_path={contract.status_path} "
                            f"completion_source=task_status_done error={str(error).strip()}"
                        ) from error
                observation, last_probe_monotonic = self._maybe_probe_agent_liveness_for_file_wait(
                    last_probe_monotonic=last_probe_monotonic,
                    status_done_seen=status_done_seen,
                )
                if observation is not None:
                    if not observation.session_exists:
                        raise RuntimeError("tmux pane exited while waiting for turn artifacts")
                    if observation.pane_dead:
                        raise RuntimeError(f"tmux pane died while waiting for turn artifacts:\n{self._diagnostic_visible_tail(160)}")
                    if observation.current_command in SHELL_COMMANDS and not status_done_seen:
                        self.agent_ready = False
                        raise RuntimeError(
                            f"agent exited back to shell while waiting for turn artifacts:\n{self._diagnostic_visible_tail(160)}"
                        )
                    self._raise_if_stale_busy_without_contract(
                        observation=observation,
                        phase=contract.phase,
                        artifact_path=contract.status_path,
                        output_exists=contract.status_path.exists(),
                        status_done_seen=status_done_seen,
                        contract_error_prefix=TURN_ARTIFACT_CONTRACT_ERROR_PREFIX,
                    )
                    stalled, idle_elapsed = self._contract_wait_stalled(
                        observation=observation,
                        status_done_seen=status_done_seen,
                        output_exists=contract.status_path.exists(),
                        invalid_output_stable_sec=invalid_elapsed,
                    )
                    if stalled:
                        raise RuntimeError(
                            f"{TURN_ARTIFACT_CONTRACT_ERROR_PREFIX}: "
                            f"phase={contract.phase} status_path={contract.status_path} "
                            f"runtime_stalled idle_sec={idle_elapsed:.1f}"
                        ) from error
                time.sleep(FILE_CONTRACT_POLL_INTERVAL_SEC)
                continue

            status_stat = contract.status_path.stat()
            invalid_signature = None
            invalid_since_monotonic = 0.0
            signature = (
                status_stat.st_size,
                status_stat.st_mtime,
                tuple(sorted(file_result.artifact_hashes.items())),
            )
            if signature == stable_signature:
                stable_elapsed = time.monotonic() - stable_since_monotonic if stable_since_monotonic else 0.0
            else:
                stable_signature = signature
                stable_since_monotonic = time.monotonic()
                stable_elapsed = 0.0

            if stable_elapsed >= max(float(contract.quiet_window_sec), 0.0):
                fresh_for_task = self._turn_artifacts_are_fresh_for_task(
                    contract,
                    task_status_path=task_status_path,
                )
                if status_done_seen and not fresh_for_task:
                    raise RuntimeError(
                        f"{TURN_ARTIFACT_CONTRACT_ERROR_PREFIX}: "
                        f"phase={contract.phase} status_path={contract.status_path} "
                        f"stale_review_round_artifacts task_status_path={task_status_path}"
                    )
                if status_done_seen:
                    self._log_event(
                        "turn_artifacts_ready",
                        turn_id=contract.turn_id,
                        phase=contract.phase,
                        status_path=str(contract.status_path),
                    )
                    return file_result
                observation = self._probe_agent_liveness_for_file_wait()
                last_probe_monotonic = time.monotonic()
                if not observation.session_exists:
                    raise RuntimeError("tmux pane exited while waiting for turn artifacts")
                if observation.pane_dead:
                    raise RuntimeError(f"tmux pane died while waiting for turn artifacts:\n{self._diagnostic_visible_tail(160)}")
                if observation.current_command in SHELL_COMMANDS:
                    self.agent_ready = False
                    raise RuntimeError(
                        f"agent exited back to shell while waiting for turn artifacts:\n{self._diagnostic_visible_tail(160)}"
                    )
                agent_state = self.get_agent_state(observation)
                if not fresh_for_task:
                    if agent_state == AgentRuntimeState.READY:
                        raise RuntimeError(
                            f"{TURN_ARTIFACT_CONTRACT_ERROR_PREFIX}: "
                            f"phase={contract.phase} status_path={contract.status_path} "
                            f"stale_review_round_artifacts task_status_path={task_status_path}"
                        )
                    time.sleep(FILE_CONTRACT_POLL_INTERVAL_SEC)
                    continue
                if agent_state == AgentRuntimeState.READY:
                    if task_status_path is not None:
                        write_task_status(task_status_path, status=TASK_STATUS_DONE)
                    status_done_seen = True
                    self.current_task_runtime_status = TASK_STATUS_DONE
                    self.current_command = observation.current_command
                    self.current_path = observation.current_path
                    self.last_heartbeat_at = observation.observed_at
                    self._log_event(
                        "turn_artifacts_ready_from_agent_state",
                        turn_id=contract.turn_id,
                        phase=contract.phase,
                        status_path=str(contract.status_path),
                    )
                    return file_result
                if self._stable_turn_artifacts_can_finish_task(contract):
                    if task_status_path is not None:
                        write_task_status(task_status_path, status=TASK_STATUS_DONE)
                    self.current_task_runtime_status = TASK_STATUS_DONE
                    self.current_command = observation.current_command
                    self.current_path = observation.current_path
                    self.last_heartbeat_at = observation.observed_at
                    self._log_event(
                        "turn_artifacts_ready_from_stable_contract",
                        turn_id=contract.turn_id,
                        phase=contract.phase,
                        status_path=str(contract.status_path),
                        agent_state=agent_state.value,
                    )
                    return file_result
            observation, last_probe_monotonic = self._maybe_probe_agent_liveness_for_file_wait(
                last_probe_monotonic=last_probe_monotonic,
                status_done_seen=status_done_seen,
            )
            if observation is not None:
                if not observation.session_exists:
                    raise RuntimeError("tmux pane exited while waiting for turn artifacts")
                if observation.pane_dead:
                    raise RuntimeError(f"tmux pane died while waiting for turn artifacts:\n{self._diagnostic_visible_tail(160)}")
                if observation.current_command in SHELL_COMMANDS:
                    if not status_done_seen:
                        self.agent_ready = False
                        raise RuntimeError(
                            f"agent exited back to shell while waiting for turn artifacts:\n{self._diagnostic_visible_tail(160)}"
                        )
                    post_done_elapsed = (
                        time.monotonic() - post_done_since_monotonic
                        if post_done_since_monotonic
                        else 0.0
                    )
                    if post_done_elapsed >= post_done_grace_sec:
                        raise RuntimeError(
                            "task_status_done 后 artifacts 未稳定: "
                            f"phase={contract.phase} status_path={contract.status_path} "
                            f"current_command={observation.current_command}"
                        )
                self._raise_if_stale_busy_without_contract(
                    observation=observation,
                    phase=contract.phase,
                    artifact_path=contract.status_path,
                    output_exists=contract.status_path.exists(),
                    status_done_seen=status_done_seen,
                    contract_error_prefix=TURN_ARTIFACT_CONTRACT_ERROR_PREFIX,
                )
                stalled, idle_elapsed = self._contract_wait_stalled(
                    observation=observation,
                    status_done_seen=status_done_seen,
                    output_exists=contract.status_path.exists(),
                )
                if stalled:
                    raise RuntimeError(
                        f"{TURN_ARTIFACT_CONTRACT_ERROR_PREFIX}: "
                        f"phase={contract.phase} status_path={contract.status_path} "
                        f"runtime_stalled idle_sec={idle_elapsed:.1f}"
                    )
            time.sleep(FILE_CONTRACT_POLL_INTERVAL_SEC)

        raise TimeoutError(
            f"等待 turn 文件结果超时: phase={contract.phase} status_path={contract.status_path}\n"
            f"{self._diagnostic_visible_tail(200)}"
        )

    def _try_finalize_turn_artifacts_after_timeout(
            self,
            *,
            contract: TurnFileContract,
            task_status_path: Path | None,
            prompt_submission_observed: bool = False,
    ) -> TurnFileResult | None:
        if not prompt_submission_observed:
            return None
        try:
            file_result = contract.validator(contract.status_path)
            validate_turn_file_artifact_rules(contract, file_result)
            status_stat = contract.status_path.stat()
        except Exception:
            return None
        signature = (
            status_stat.st_size,
            status_stat.st_mtime,
            tuple(sorted(file_result.artifact_hashes.items())),
        )
        quiet_window = max(float(contract.quiet_window_sec), 0.0)
        if quiet_window:
            time.sleep(quiet_window)
        try:
            next_result = contract.validator(contract.status_path)
            validate_turn_file_artifact_rules(contract, next_result)
            next_stat = contract.status_path.stat()
        except Exception:
            return None
        next_signature = (
            next_stat.st_size,
            next_stat.st_mtime,
            tuple(sorted(next_result.artifact_hashes.items())),
        )
        if next_signature != signature:
            return None
        if not self._turn_artifacts_are_fresh_for_task(contract, task_status_path=task_status_path):
            return None
        if task_status_path is not None:
            write_task_status(task_status_path, status=TASK_STATUS_DONE)
        self.current_task_runtime_status = TASK_STATUS_DONE
        self._log_event(
            "turn_artifacts_ready_after_prompt_timeout",
            turn_id=contract.turn_id,
            phase=contract.phase,
            status_path=str(contract.status_path),
        )
        return next_result

    def _try_finalize_task_result_after_prompt_timeout(
            self,
            *,
            contract: TaskResultContract,
            task_status_path: Path | None,
            result_path: Path,
            baseline_visible: str,
            baseline_raw_log_tail: str,
            prompt_submission_observed: bool = False,
    ) -> TaskResultFile | None:
        if not prompt_submission_observed:
            return None
        try:
            return self.wait_for_task_result(
                contract=contract,
                task_status_path=task_status_path,
                result_path=result_path,
                timeout_sec=5.0,
                baseline_visible=baseline_visible,
                baseline_raw_log_tail=baseline_raw_log_tail,
            )
        except Exception:
            return None

    def _try_finalize_task_result_from_ready_agent_after_busy_timeout(
            self,
            *,
            contract: TaskResultContract,
            task_status_path: Path | None,
            result_path: Path,
            prompt_submission_observed: bool,
            observation: WorkerObservation | None = None,
    ) -> TaskResultFile | None:
        if not prompt_submission_observed or result_path.exists():
            return None
        candidate = build_missing_task_result_finalization_candidate(
            contract,
            task_status_path=task_status_path,
        )
        if candidate is None or not candidate.requires_agent_ready:
            return None
        decision_status = str(candidate.decision.status or "").strip()
        if observation is None:
            observation = self._probe_agent_liveness_for_file_wait()
        if not observation.session_exists:
            raise RuntimeError("tmux pane exited while finalizing task result after busy timeout")
        if observation.pane_dead:
            raise RuntimeError(f"tmux pane died while finalizing task result after busy timeout:\n{self._diagnostic_visible_tail(160)}")
        current_command = observation.current_command or self.current_command
        if current_command in SHELL_COMMANDS or not self._agent_running(current_command):
            return None
        self._raise_on_provider_runtime_error(
            observation,
            context=f"finalizing missing task result after busy timeout phase={contract.phase}",
        )
        agent_state = self.get_agent_state(observation)
        if agent_state == AgentRuntimeState.READY:
            ready_evidence = "agent_state"
        elif decision_status in {TASK_RESULT_READY, TASK_RESULT_HITL} and self._observation_indicates_ready_or_idle_surface(observation):
            ready_evidence = "idle_surface"
        else:
            return None
        try:
            result_file = finalize_task_result(
                contract=contract,
                result_path=result_path,
                task_status_path=task_status_path,
            )
        except Exception:
            return None
        self.current_task_runtime_status = TASK_STATUS_DONE
        self.current_command = current_command
        self.current_path = observation.current_path
        self.last_heartbeat_at = observation.observed_at
        self.agent_ready = True
        self.agent_started = True
        self.agent_state = AgentRuntimeState.READY
        self.wrapper_state = WrapperState.READY
        self._log_event(
            "task_result_ready_from_contract_after_busy_timeout",
            turn_id=contract.turn_id,
            phase=contract.phase,
            result_path=str(result_path),
            status=str(result_file.payload.get("status", "")),
            evidence=ready_evidence,
        )
        return result_file

    def _busy_agent_observation_after_turn_timeout(self) -> WorkerObservation | None:
        observation = self._probe_agent_liveness_for_file_wait()
        if not observation.session_exists:
            raise RuntimeError("tmux pane exited while waiting for timed-out turn")
        if observation.pane_dead:
            raise RuntimeError(f"tmux pane died while waiting for timed-out turn:\n{self._diagnostic_visible_tail(160)}")
        current_command = observation.current_command or self.current_command
        if current_command in SHELL_COMMANDS:
            self.agent_ready = False
            self.wrapper_state = WrapperState.NOT_READY
            return None
        if not self._agent_running(current_command):
            return None
        self._raise_on_provider_runtime_error(observation, context="waiting for timed-out turn")
        if self.get_agent_state(observation) != AgentRuntimeState.BUSY:
            return None
        self.agent_ready = False
        self.agent_started = True
        self.agent_state = AgentRuntimeState.BUSY
        self.wrapper_state = WrapperState.NOT_READY
        self.current_command = current_command
        self.current_path = observation.current_path or self.current_path
        self.last_heartbeat_at = observation.observed_at
        return observation

    def _record_turn_timeout_extended_for_busy_agent(
            self,
            *,
            label: str,
            attempt: int,
            timeout_sec: float,
            task_status_path: Path,
            observation: WorkerObservation,
    ) -> None:
        self._log_event(
            "turn_timeout_wait_extended_for_busy_agent",
            label=label,
            attempt=attempt,
            timeout_sec=timeout_sec,
            current_command=observation.current_command,
            pane_title=observation.pane_title,
        )
        self._write_state(
            WorkerStatus.RUNNING,
            note=f"still_running:{label}",
            extra={
                "label": label,
                "retry_count": attempt - 1,
                "result_status": "running",
                "agent_ready": False,
                "agent_state": AgentRuntimeState.BUSY.value,
                "current_command": observation.current_command,
                "current_path": observation.current_path,
                "current_task_status_path": str(task_status_path),
                "current_task_result_path": self.current_task_result_path,
                "current_task_runtime_status": self.current_task_runtime_status,
            },
        )

    def _wait_for_turn_artifacts_while_agent_busy_after_timeout(
            self,
            *,
            label: str,
            attempt: int,
            timeout_sec: float,
            contract: TurnFileContract,
            task_status_path: Path,
            prompt_submission_observed: bool,
    ) -> TurnFileResult | None:
        if not prompt_submission_observed:
            return None
        while True:
            raise_if_runtime_shutdown_requested("waiting for busy agent turn artifacts")
            observation = self._busy_agent_observation_after_turn_timeout()
            if observation is None:
                return None
            self._record_turn_timeout_extended_for_busy_agent(
                label=label,
                attempt=attempt,
                timeout_sec=timeout_sec,
                task_status_path=task_status_path,
                observation=observation,
            )
            try:
                return self.wait_for_turn_artifacts(
                    contract=contract,
                    task_status_path=task_status_path,
                    timeout_sec=timeout_sec,
                )
            except TimeoutError:
                file_result = self._try_finalize_turn_artifacts_after_timeout(
                    contract=contract,
                    task_status_path=task_status_path,
                    prompt_submission_observed=prompt_submission_observed,
                )
                if file_result is not None:
                    return file_result

    def _wait_for_task_result_while_agent_busy_after_timeout(
            self,
            *,
            label: str,
            attempt: int,
            timeout_sec: float,
            contract: TaskResultContract,
            task_status_path: Path,
            result_path: Path,
            baseline_visible: str,
            baseline_raw_log_tail: str,
            prompt_submission_observed: bool,
    ) -> TaskResultFile | None:
        if not prompt_submission_observed:
            return None
        while True:
            raise_if_runtime_shutdown_requested("waiting for busy agent task result")
            observation = self._busy_agent_observation_after_turn_timeout()
            if observation is None:
                return self._try_finalize_task_result_from_ready_agent_after_busy_timeout(
                    contract=contract,
                    task_status_path=task_status_path,
                    result_path=result_path,
                    prompt_submission_observed=prompt_submission_observed,
                )
            self._record_turn_timeout_extended_for_busy_agent(
                label=label,
                attempt=attempt,
                timeout_sec=timeout_sec,
                task_status_path=task_status_path,
                observation=observation,
            )
            try:
                return self.wait_for_task_result(
                    contract=contract,
                    task_status_path=task_status_path,
                    result_path=result_path,
                    timeout_sec=timeout_sec,
                    baseline_visible=baseline_visible,
                    baseline_raw_log_tail=baseline_raw_log_tail,
                )
            except TimeoutError:
                task_result = self._try_finalize_task_result_after_prompt_timeout(
                    contract=contract,
                    task_status_path=task_status_path,
                    result_path=result_path,
                    baseline_visible=baseline_visible,
                    baseline_raw_log_tail=baseline_raw_log_tail,
                    prompt_submission_observed=prompt_submission_observed,
                )
                if task_result is not None:
                    return task_result

    def _infer_prompt_submission_from_busy_agent_after_timeout(self) -> bool:
        probe_count = 10
        for probe_index in range(probe_count):
            observation = self._busy_agent_observation_after_turn_timeout()
            if observation is not None:
                self._log_event(
                    "prompt_submission_inferred_from_busy_agent",
                    current_command=observation.current_command,
                    pane_title=observation.pane_title,
                    probe_index=probe_index,
                )
                return True
            if probe_index + 1 < probe_count:
                time.sleep(0.5)
        self._log_event("prompt_submission_busy_probe_exhausted")
        return False

    def _turn_failure_runtime_state_extra(self, clean_output: str) -> dict[str, object]:
        if is_provider_runtime_error(clean_output):
            self.agent_ready = False
            self.agent_state = AgentRuntimeState.STARTING
            self.wrapper_state = WrapperState.NOT_READY
            return {
                "agent_ready": False,
                "agent_state": AgentRuntimeState.STARTING.value,
                "health_status": "provider_runtime_error",
                "health_note": clean_output,
                "last_provider_error": clean_output,
            }
        if is_worker_death_error(clean_output):
            self.agent_ready = False
            self.agent_state = AgentRuntimeState.DEAD
            self.wrapper_state = WrapperState.NOT_READY
            return {
                "agent_ready": False,
                "agent_state": AgentRuntimeState.DEAD.value,
            }
        if is_prompt_dispatch_timeout_error(clean_output):
            return {
                "dispatch_state": self.dispatch_state or "delayed",
                "dispatch_reason": self.dispatch_reason or f"prompt_dispatch_timeout:{clean_output}",
            }
        with contextlib.suppress(Exception):
            snapshot = self._build_passive_health_snapshot()
            agent_state = AgentRuntimeState(str(snapshot.agent_state or "").strip().upper())
            self.agent_state = agent_state
            self.agent_ready = agent_state == AgentRuntimeState.READY
            return {
                "agent_alive": agent_state != AgentRuntimeState.DEAD and bool(snapshot.session_exists),
                "agent_ready": self.agent_ready,
                "agent_state": agent_state.value,
                "health_status": snapshot.health_status,
                "health_note": snapshot.health_note,
                "pane_title": snapshot.pane_title,
                "current_command": snapshot.current_command,
                "current_path": snapshot.current_path,
                "last_heartbeat_at": snapshot.last_heartbeat_at,
            }
        agent_alive = False
        target_exists = False
        with contextlib.suppress(Exception):
            agent_alive = self.is_agent_alive()
        with contextlib.suppress(Exception):
            target_exists = bool(self.pane_id and self.target_exists())
        if agent_alive:
            fallback_state = self.agent_state if self.agent_state != AgentRuntimeState.DEAD else AgentRuntimeState.STARTING
        elif target_exists:
            fallback_state = AgentRuntimeState.STARTING
        else:
            fallback_state = AgentRuntimeState.DEAD
        self.agent_state = fallback_state
        self.agent_ready = fallback_state == AgentRuntimeState.READY
        return {
            "agent_ready": self.agent_ready,
            "agent_state": fallback_state.value,
        }

    def _observation_terminal_signature(self, observation: WorkerObservation) -> str:
        return self._build_terminal_signature("\n".join(
            str(part or "")
            for part in (
                observation.pane_title,
                observation.visible_text,
                observation.raw_log_tail,
            )
            if str(part or "").strip()
        ))

    def _provider_runtime_error_from_observation(self, observation: WorkerObservation) -> str:
        def _extract_provider_error(text: str) -> str:
            cleaned = clean_ansi(str(text or ""))
            if not cleaned.strip() or not is_provider_runtime_error(cleaned):
                return ""
            lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
            for line in reversed(lines[-80:]):
                if is_provider_runtime_error(line):
                    return line[-800:]
            return cleaned[-800:]

        delta_error = _extract_provider_error(observation.raw_log_delta)
        if delta_error:
            return delta_error

        surface_text = "\n".join(
            str(part or "")
            for part in (observation.visible_text, observation.raw_log_tail)
            if str(part or "").strip()
        )
        surface_error = _extract_provider_error(surface_text)
        if not surface_error:
            return ""

        ready_or_idle = False
        with contextlib.suppress(Exception):
            ready_or_idle = self.get_agent_state(observation, task_running_override=False) == AgentRuntimeState.READY
        if not ready_or_idle:
            with contextlib.suppress(Exception):
                ready_or_idle = self._observation_indicates_ready_or_idle_surface(observation)
        if ready_or_idle:
            return ""
        return surface_error

    def _raise_on_provider_runtime_error(self, observation: WorkerObservation, *, context: str) -> None:
        provider_error = self._provider_runtime_error_from_observation(observation)
        if not provider_error:
            return
        message = f"provider runtime error while {context}: {provider_error}"
        self.mark_provider_runtime_error(reason_text=message)
        raise RuntimeError(message)

    def _wait_for_ready_task_result_after_submit(
            self,
            *,
            contract: TaskResultContract,
            task_status_path: Path | None,
            result_path: Path,
            timeout_sec: float,
            prompt: str,
            baseline_observation: WorkerObservation,
    ) -> TaskResultFile:
        deadline = time.monotonic() + timeout_sec
        baseline_signature = self._observation_terminal_signature(baseline_observation)
        saw_busy_after_submit = False
        saw_submission_evidence = False
        ready_hits = 0
        status_done_seen = task_status_path is None

        while time.monotonic() < deadline:
            raise_if_runtime_shutdown_requested("waiting for ready task result")
            status_done_seen = self._track_task_completion_signal(
                task_status_path=task_status_path,
                status_done_seen=status_done_seen,
            )
            try:
                result_file = self._validate_task_result_file(
                    contract=contract,
                    result_path=result_path,
                )
                if status_done_seen:
                    return result_file
            except Exception:
                pass

            if saw_busy_after_submit and self.config.vendor != Vendor.OPENCODE:
                observation = self._probe_agent_liveness_for_file_wait()
            else:
                observation = self.observe(tail_lines=160, tail_bytes=12000)
            if not observation.session_exists:
                raise RuntimeError("tmux pane exited while waiting for ready task result")
            if observation.pane_dead:
                raise RuntimeError(f"tmux pane died while waiting for ready task result:\n{self._diagnostic_visible_tail(160)}")
            if observation.current_command in SHELL_COMMANDS:
                self.agent_ready = False
                raise RuntimeError(
                    f"agent exited back to shell while waiting for ready task result:\n{self._diagnostic_visible_tail(160)}"
                )
            self._raise_on_provider_runtime_error(
                observation,
                context=f"waiting for ready task result phase={contract.phase}",
            )

            agent_state = self.get_agent_state(observation)
            if agent_state == AgentRuntimeState.BUSY:
                saw_busy_after_submit = True
                ready_hits = 0
                result_file = self._try_finalize_task_result_from_ready_agent_after_busy_timeout(
                    contract=contract,
                    task_status_path=task_status_path,
                    result_path=result_path,
                    prompt_submission_observed=True,
                    observation=observation,
                )
                if result_file is not None:
                    return result_file
                self._raise_if_stale_busy_without_contract(
                    observation=observation,
                    phase=contract.phase,
                    artifact_path=result_path,
                    output_exists=result_path.exists(),
                    status_done_seen=status_done_seen,
                    contract_error_prefix=TASK_RESULT_CONTRACT_ERROR_PREFIX,
                )
                time.sleep(FILE_CONTRACT_POLL_INTERVAL_SEC)
                continue

            prompt_observed = (
                self._source_mentions_prompt(observation.visible_text, prompt)
                or self._source_mentions_prompt(observation.raw_log_delta, prompt)
                or self._source_mentions_prompt_submission_marker(observation.visible_text, prompt)
                or self._source_mentions_prompt_submission_marker(observation.raw_log_delta, prompt)
            )
            current_signature = self._observation_terminal_signature(observation)
            surface_changed = bool(current_signature) and current_signature != baseline_signature
            saw_submission_evidence = saw_submission_evidence or prompt_observed or surface_changed

            if agent_state == AgentRuntimeState.READY:
                ready_hits += 1
                if saw_busy_after_submit or prompt_observed or (saw_submission_evidence and ready_hits >= 2):
                    result_file = finalize_task_result(
                        contract=contract,
                        result_path=result_path,
                        task_status_path=task_status_path,
                    )
                    self.current_task_runtime_status = TASK_STATUS_DONE
                    self.current_command = observation.current_command
                    self.current_path = observation.current_path
                    self._log_event(
                        "task_result_ready_from_agent_state",
                        turn_id=contract.turn_id,
                        phase=contract.phase,
                        result_path=str(result_path),
                        status=str(result_file.payload.get("status", "")),
                        evidence=(
                            "busy_to_ready"
                            if saw_busy_after_submit
                            else "prompt_observed"
                            if prompt_observed
                            else "surface_changed"
                        ),
                    )
                    return result_file
            else:
                ready_hits = 0
            time.sleep(FILE_CONTRACT_POLL_INTERVAL_SEC)

        raise TimeoutError(
            f"等待 READY-only 任务结果超时: phase={contract.phase} result_path={result_path}\n"
            f"{self._diagnostic_visible_tail(200)}"
        )

    def wait_for_task_result(
            self,
            *,
            contract: TaskResultContract,
            task_status_path: Path | None,
            result_path: Path,
            timeout_sec: float,
            baseline_visible: str = "",
            baseline_raw_log_tail: str = "",
    ) -> TaskResultFile:
        deadline = time.monotonic() + timeout_sec
        stable_signature: tuple[object, ...] | None = None
        stable_hits = 0
        missing_contract_signature: tuple[object, ...] | None = None
        missing_contract_stable_hits = 0
        invalid_signature: tuple[object, ...] | None = None
        invalid_since_monotonic = 0.0
        ready_missing_signature: tuple[object, ...] | None = None
        ready_missing_since_monotonic = 0.0
        status_done_seen = task_status_path is None
        post_done_since_monotonic = time.monotonic() if status_done_seen else 0.0
        post_done_grace_sec = TASK_RESULT_POST_DONE_GRACE_SEC
        last_probe_monotonic = 0.0

        while time.monotonic() < deadline:
            raise_if_runtime_shutdown_requested("waiting for task result")
            previous_done_seen = status_done_seen
            status_done_seen = self._track_task_completion_signal(
                task_status_path=task_status_path,
                status_done_seen=status_done_seen,
            )
            if status_done_seen and not previous_done_seen and not post_done_since_monotonic:
                post_done_since_monotonic = time.monotonic()

            try:
                result_file = self._validate_task_result_file(
                    contract=contract,
                    result_path=result_path,
                )
            except Exception as error:
                stable_signature = None
                stable_hits = 0
                fresh_contract_signature = None
                if not result_path.exists():
                    fresh_contract_signature = self._fresh_completed_task_result_contract_signature(
                        contract=contract,
                        task_status_path=task_status_path,
                    )
                if fresh_contract_signature is not None:
                    if fresh_contract_signature == missing_contract_signature:
                        missing_contract_stable_hits += 1
                    else:
                        missing_contract_signature = fresh_contract_signature
                        missing_contract_stable_hits = 1
                    if missing_contract_stable_hits >= 2:
                        result_file = finalize_task_result(
                            contract=contract,
                            result_path=result_path,
                            task_status_path=task_status_path,
                        )
                        self.current_task_runtime_status = TASK_STATUS_DONE
                        self._log_event(
                            "task_result_ready_from_stable_contract",
                            turn_id=contract.turn_id,
                            phase=contract.phase,
                            result_path=str(result_path),
                            status=str(result_file.payload.get("status", "")),
                        )
                        return result_file
                else:
                    missing_contract_signature = None
                    missing_contract_stable_hits = 0
                if status_done_seen:
                    invalid_state = "missing"
                    if result_path.exists():
                        try:
                            result_stat = result_path.stat()
                            invalid_state = (result_stat.st_size, result_stat.st_mtime)
                        except OSError:
                            invalid_state = "stat_error"
                    current_invalid_signature = (
                        invalid_state,
                        type(error).__name__,
                        str(error).strip(),
                    )
                    if current_invalid_signature == invalid_signature:
                        invalid_elapsed = (
                            time.monotonic() - invalid_since_monotonic
                            if invalid_since_monotonic
                            else 0.0
                        )
                    else:
                        invalid_signature = current_invalid_signature
                        invalid_since_monotonic = time.monotonic()
                        invalid_elapsed = 0.0
                    if invalid_elapsed >= max(post_done_grace_sec, 0.0):
                        raise RuntimeError(
                            f"{TASK_RESULT_CONTRACT_ERROR_PREFIX}: "
                            f"phase={contract.phase} result_path={result_path} "
                            f"error={str(error).strip()}"
                        ) from error
                else:
                    invalid_signature = None
                    invalid_since_monotonic = 0.0
                if (
                    not result_path.exists()
                    and self._can_finalize_missing_task_result_from_contract(
                        contract=contract,
                        task_status_path=task_status_path,
                    )
                ):
                    observation, last_probe_monotonic = self._maybe_probe_agent_liveness_for_file_wait(
                        last_probe_monotonic=last_probe_monotonic,
                        status_done_seen=status_done_seen,
                        force=True,
                    )
                    if observation is not None and not observation.session_exists:
                        raise RuntimeError("tmux pane exited while waiting for task result")
                    if observation is not None and observation.pane_dead:
                        raise RuntimeError(f"tmux pane died while waiting for task result:\n{self._diagnostic_visible_tail(160)}")
                    if observation is not None:
                        self._raise_on_provider_runtime_error(
                            observation,
                            context=f"materializing missing task result phase={contract.phase}",
                        )
                        result_file = self._try_finalize_task_result_from_ready_agent_after_busy_timeout(
                            contract=contract,
                            task_status_path=task_status_path,
                            result_path=result_path,
                            prompt_submission_observed=True,
                            observation=observation,
                        )
                        if result_file is not None:
                            return result_file
                        self._raise_if_stale_busy_without_contract(
                            observation=observation,
                            phase=contract.phase,
                            artifact_path=result_path,
                            output_exists=result_path.exists(),
                            status_done_seen=status_done_seen,
                            contract_error_prefix=TASK_RESULT_CONTRACT_ERROR_PREFIX,
                        )
                    agent_ready = observation is not None and self.get_agent_state(observation) == AgentRuntimeState.READY
                    if not agent_ready:
                        time.sleep(FILE_CONTRACT_POLL_INTERVAL_SEC)
                        continue
                    try:
                        result_file = finalize_task_result(
                            contract=contract,
                            result_path=result_path,
                            task_status_path=task_status_path,
                        )
                    except Exception:
                        result_file = None
                    if result_file is not None:
                        self.current_task_runtime_status = TASK_STATUS_DONE
                        self.current_command = observation.current_command
                        self.current_path = observation.current_path
                        self._log_event(
                            "task_result_ready_from_contract",
                            turn_id=contract.turn_id,
                            phase=contract.phase,
                            result_path=str(result_path),
                            status=str(result_file.payload.get("status", "")),
                        )
                        return result_file
                observation, last_probe_monotonic = self._maybe_probe_agent_liveness_for_file_wait(
                    last_probe_monotonic=last_probe_monotonic,
                    status_done_seen=status_done_seen,
                )
                if observation is not None:
                    if not observation.session_exists:
                        raise RuntimeError("tmux pane exited while waiting for task result")
                    if observation.pane_dead:
                        raise RuntimeError(f"tmux pane died while waiting for task result:\n{self._diagnostic_visible_tail(160)}")
                    if observation.current_command in SHELL_COMMANDS and not status_done_seen:
                        self.agent_ready = False
                        raise RuntimeError(
                            f"agent exited back to shell while waiting for task result:\n{self._diagnostic_visible_tail(160)}"
                        )
                    self._raise_on_provider_runtime_error(
                        observation,
                        context=f"waiting for task result phase={contract.phase}",
                    )
                    result_file = self._try_finalize_task_result_from_ready_agent_after_busy_timeout(
                        contract=contract,
                        task_status_path=task_status_path,
                        result_path=result_path,
                        prompt_submission_observed=True,
                        observation=observation,
                    )
                    if result_file is not None:
                        return result_file
                    if (
                        not status_done_seen
                        and not result_path.exists()
                        and self.get_agent_state(observation) == AgentRuntimeState.READY
                    ):
                        current_ready_missing_signature = (
                            type(error).__name__,
                            str(error).strip(),
                            tuple(sorted(str(path) for path in contract.required_artifacts.values())),
                        )
                        if current_ready_missing_signature == ready_missing_signature:
                            ready_missing_elapsed = (
                                time.monotonic() - ready_missing_since_monotonic
                                if ready_missing_since_monotonic
                                else 0.0
                            )
                        else:
                            ready_missing_signature = current_ready_missing_signature
                            ready_missing_since_monotonic = time.monotonic()
                            ready_missing_elapsed = 0.0
                        if ready_missing_elapsed >= TASK_RESULT_READY_MISSING_GRACE_SEC:
                            raise RuntimeError(
                                f"{TASK_RESULT_CONTRACT_ERROR_PREFIX}: "
                                f"phase={contract.phase} result_path={result_path} "
                                f"agent_state=READY error={str(error).strip()}"
                            ) from error
                    else:
                        ready_missing_signature = None
                        ready_missing_since_monotonic = 0.0
                    self._raise_if_stale_busy_without_contract(
                        observation=observation,
                        phase=contract.phase,
                        artifact_path=result_path,
                        output_exists=result_path.exists(),
                        status_done_seen=status_done_seen,
                        contract_error_prefix=TASK_RESULT_CONTRACT_ERROR_PREFIX,
                    )
                    stalled, idle_elapsed = self._contract_wait_stalled(
                        observation=observation,
                        status_done_seen=status_done_seen,
                        output_exists=result_path.exists(),
                    )
                    if stalled:
                        raise RuntimeError(
                            f"{TASK_RESULT_CONTRACT_ERROR_PREFIX}: "
                            f"phase={contract.phase} result_path={result_path} "
                            f"runtime_stalled idle_sec={idle_elapsed:.1f}"
                        ) from error
                time.sleep(FILE_CONTRACT_POLL_INTERVAL_SEC)
                continue

            result_stat = result_path.stat()
            missing_contract_signature = None
            missing_contract_stable_hits = 0
            ready_missing_signature = None
            ready_missing_since_monotonic = 0.0
            invalid_signature = None
            invalid_since_monotonic = 0.0
            signature = (
                result_stat.st_size,
                result_stat.st_mtime,
                str(result_file.payload.get("status", "")),
                tuple(sorted(result_file.artifact_hashes.items())),
            )
            if signature == stable_signature:
                stable_hits += 1
            else:
                stable_signature = signature
                stable_hits = 1

            if stable_hits >= 2:
                if status_done_seen:
                    self._log_event(
                        "task_result_ready",
                        turn_id=contract.turn_id,
                        phase=contract.phase,
                        result_path=str(result_path),
                        status=str(result_file.payload.get("status", "")),
                    )
                    return result_file
                observation = self._probe_agent_liveness_for_file_wait()
                last_probe_monotonic = time.monotonic()
                if not observation.session_exists:
                    raise RuntimeError("tmux pane exited while waiting for task result")
                if observation.pane_dead:
                    raise RuntimeError(f"tmux pane died while waiting for task result:\n{self._diagnostic_visible_tail(160)}")
                if observation.current_command in SHELL_COMMANDS:
                    self.agent_ready = False
                    raise RuntimeError(
                        f"agent exited back to shell while waiting for task result:\n{self._diagnostic_visible_tail(160)}"
                    )
                if self.get_agent_state(observation) == AgentRuntimeState.READY:
                    if task_status_path is not None:
                        write_task_status(task_status_path, status=TASK_STATUS_DONE)
                    status_done_seen = True
                    self.current_task_runtime_status = TASK_STATUS_DONE
                    self.current_command = observation.current_command
                    self.current_path = observation.current_path
                    self.last_heartbeat_at = observation.observed_at
                    self._log_event(
                        "task_result_ready_from_agent_state",
                        turn_id=contract.turn_id,
                        phase=contract.phase,
                        result_path=str(result_path),
                        status=str(result_file.payload.get("status", "")),
                    )
                    return result_file
            observation, last_probe_monotonic = self._maybe_probe_agent_liveness_for_file_wait(
                last_probe_monotonic=last_probe_monotonic,
                status_done_seen=status_done_seen,
            )
            if observation is not None:
                if not observation.session_exists:
                    raise RuntimeError("tmux pane exited while waiting for task result")
                if observation.pane_dead:
                    raise RuntimeError(f"tmux pane died while waiting for task result:\n{self._diagnostic_visible_tail(160)}")
                self._raise_on_provider_runtime_error(
                    observation,
                    context=f"waiting for stable task result phase={contract.phase}",
                )
                if observation.current_command in SHELL_COMMANDS:
                    if not status_done_seen:
                        self.agent_ready = False
                        raise RuntimeError(
                            f"agent exited back to shell while waiting for task result:\n{self._diagnostic_visible_tail(160)}"
                        )
                    post_done_elapsed = (
                        time.monotonic() - post_done_since_monotonic
                        if post_done_since_monotonic
                        else 0.0
                    )
                    if post_done_elapsed >= post_done_grace_sec:
                        raise RuntimeError(
                            "task_status_done 后 result 未稳定: "
                            f"phase={contract.phase} result_path={result_path} "
                            f"current_command={observation.current_command}"
                        )
                self._raise_if_stale_busy_without_contract(
                    observation=observation,
                    phase=contract.phase,
                    artifact_path=result_path,
                    output_exists=result_path.exists(),
                    status_done_seen=status_done_seen,
                    contract_error_prefix=TASK_RESULT_CONTRACT_ERROR_PREFIX,
                )
                stalled, idle_elapsed = self._contract_wait_stalled(
                    observation=observation,
                    status_done_seen=status_done_seen,
                    output_exists=result_path.exists(),
                )
                if stalled:
                    raise RuntimeError(
                        f"{TASK_RESULT_CONTRACT_ERROR_PREFIX}: "
                        f"phase={contract.phase} result_path={result_path} "
                        f"runtime_stalled idle_sec={idle_elapsed:.1f}"
                    )
            time.sleep(FILE_CONTRACT_POLL_INTERVAL_SEC)

        raise TimeoutError(
            f"等待任务结果超时: phase={contract.phase} result_path={result_path}\n"
            f"{self._diagnostic_visible_tail(200)}"
        )

    def _wait_for_shell_ready(self, timeout_sec: float = 12.0) -> None:
        deadline = time.monotonic() + timeout_sec
        shell_context_stable_count = 0
        last_current_command = ""
        last_current_path = ""
        last_prompt_detected = False
        while time.monotonic() < deadline:
            raise_if_runtime_shutdown_requested("waiting for prompt output stability")
            observation = self.observe(tail_lines=120)
            if not observation.session_exists:
                raise RuntimeError("tmux pane exited before shell became ready")
            if observation.pane_dead:
                raise RuntimeError("tmux pane died before shell became ready")
            current_command = observation.current_command
            current_path = observation.current_path
            prompt_detected = self._shell_prompt_visible(observation)
            last_current_command = current_command
            last_current_path = current_path
            last_prompt_detected = prompt_detected
            if current_command in SHELL_COMMANDS and current_path == str(self.work_dir):
                if prompt_detected:
                    return
                shell_context_stable_count += 1
                if shell_context_stable_count >= 2:
                    return
            else:
                shell_context_stable_count = 0
            time.sleep(0.4)
        self._log_event(
            "shell_bootstrap_timeout",
            current_command=last_current_command,
            current_path=last_current_path,
            prompt_detected=last_prompt_detected,
        )
        raise RuntimeError(
            "Shell initialization timed out. "
            f"current_command={last_current_command or '(empty)'} "
            f"current_path={last_current_path or '(empty)'} "
            f"prompt_detected={'yes' if last_prompt_detected else 'no'}\n"
            f"{self.capture_visible(120)}"
        )

    def _shell_prompt_visible(self, observation: WorkerObservation) -> bool:
        combined_surface = "\n".join(
            part for part in (observation.visible_text, observation.raw_log_tail) if str(part or "").strip()
        )
        if not combined_surface.strip():
            return False
        for line in reversed(combined_surface.splitlines()):
            stripped = clean_ansi(line).strip()
            if not stripped:
                continue
            return bool(re.search(r"[$%#]\s*$", stripped))
        return False

    def _boot_action_allowed(self, action_signature: str, cooldown_sec: float = 3.0) -> bool:
        if (
            action_signature == self._last_boot_action_signature
            and time.monotonic() - self._last_boot_action_at < cooldown_sec
        ):
            return False
        self._last_boot_action_signature = action_signature
        self._last_boot_action_at = time.monotonic()
        return True

    def _maybe_handle_codex_boot_prompt(self, visible_text: str) -> bool:
        if self.config.vendor != Vendor.CODEX:
            return False
        recent_output = "\n".join(str(visible_text or "").splitlines()[-80:])
        effective_output = _codex_effective_recent_surface(recent_output, max_lines=80)
        if effective_output != recent_output and _codex_surface_indicates_ready_prompt(effective_output):
            return False
        if re.search(r"Press enter to continue", recent_output, re.IGNORECASE) and any(
                re.search(pattern, recent_output, re.IGNORECASE) for pattern in CODEX_TRUST_PROMPT_PATTERNS
        ):
            action_signature = f"codex-trust:{hashlib.sha1(recent_output.encode('utf-8')).hexdigest()[:12]}"
            if not self._boot_action_allowed(action_signature):
                return False
            self.send_special_key("Enter")
            return True
        if all(re.search(pattern, recent_output, re.IGNORECASE) for pattern in CODEX_UPDATE_NOTICE_PATTERNS):
            action_signature = f"codex-update:{hashlib.sha1(recent_output.encode('utf-8')).hexdigest()[:12]}"
            if not self._boot_action_allowed(action_signature):
                return False
            self.send_special_key("Down")
            time.sleep(0.1)
            self.send_special_key("Down")
            time.sleep(0.1)
            self.send_special_key("Enter")
            return True
        if all(re.search(pattern, recent_output, re.IGNORECASE) for pattern in CODEX_MODEL_SELECTION_PROMPT_PATTERNS):
            action_signature = f"codex-model:{hashlib.sha1(recent_output.encode('utf-8')).hexdigest()[:12]}"
            if not self._boot_action_allowed(action_signature):
                return False
            self.send_special_key("Down")
            time.sleep(0.1)
            self.send_special_key("Enter")
            return True
        return False

    def _maybe_handle_gemini_boot_prompt(self, visible_text: str) -> bool:
        if self.config.vendor != Vendor.GEMINI:
            return False
        recent_output = "\n".join(str(visible_text or "").splitlines()[-80:])
        if not all(re.search(pattern, recent_output, re.IGNORECASE) for pattern in GEMINI_TRUST_PROMPT_PATTERNS):
            return False
        action_signature = f"gemini-trust:{hashlib.sha1(recent_output.encode('utf-8')).hexdigest()[:12]}"
        if not self._boot_action_allowed(action_signature):
            return False
        self.send_special_key("Enter")
        return True

    def _task_runtime_dir(self) -> Path:
        path = self.runtime_dir / "task_runtime"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _build_task_runtime_basename(self, *, label: str, attempt: int) -> str:
        session_fragment = _sanitize_task_runtime_fragment(
            self.session_name or self.worker_id,
            fallback=_slugify(self.session_name or self.worker_id, max_len=72),
            max_len=72,
        )
        label_fragment = _sanitize_task_runtime_fragment(
            label,
            fallback=_slugify(label, max_len=32),
            max_len=32,
        )
        return f"{session_fragment}_{label_fragment}_attempt_{attempt}"

    def _build_task_status_path(self, *, label: str, attempt: int) -> Path:
        return self._task_runtime_dir() / f"{self._build_task_runtime_basename(label=label, attempt=attempt)}.json"

    def _build_task_result_path(self, *, label: str, attempt: int) -> Path:
        return self._task_runtime_dir() / f"{self._build_task_runtime_basename(label=label, attempt=attempt)}_result.json"

    @staticmethod
    def _write_task_result_file(path: str | Path, payload: dict[str, Any]) -> None:
        _ = write_task_status
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(target)

    def _validate_task_result_file(
            self,
            *,
            contract: TaskResultContract,
            result_path: Path,
    ) -> TaskResultFile:
        return validate_task_result_file(contract=contract, result_path=result_path)

    @staticmethod
    def _stable_turn_artifacts_can_finish_task(contract: TurnFileContract) -> bool:
        contract_kind = str(contract.kind or "").strip()
        return contract_kind in {"routing_file_contract", "review_round"}

    @staticmethod
    def _turn_artifacts_are_fresh_for_task(
            contract: TurnFileContract,
            *,
            task_status_path: Path | None,
    ) -> bool:
        if str(contract.kind or "").strip() != "review_round" or task_status_path is None:
            return True
        try:
            task_started_mtime_ns = Path(task_status_path).expanduser().resolve().stat().st_mtime_ns
        except OSError:
            return True
        try:
            return contract.status_path.expanduser().resolve().stat().st_mtime_ns >= task_started_mtime_ns
        except OSError:
            return False

    def _probe_agent_liveness_for_file_wait(self) -> WorkerObservation:
        if type(self).observe is not TmuxBatchWorker.observe:
            return self.observe(tail_lines=80, tail_bytes=0)
        if self.current_task_runtime_status == TASK_STATUS_RUNNING:
            return self.observe(tail_lines=80, tail_bytes=12000)
        if self.config.vendor == Vendor.OPENCODE and self.agent_state == AgentRuntimeState.BUSY:
            return self.observe(tail_lines=80, tail_bytes=12000)
        return self._capture_lightweight_observation()

    def _maybe_probe_agent_liveness_for_file_wait(
            self,
            *,
            last_probe_monotonic: float,
            status_done_seen: bool,
            force: bool = False,
    ) -> tuple[WorkerObservation | None, float]:
        now = time.monotonic()
        interval = POST_DONE_AGENT_PROBE_INTERVAL_SEC if status_done_seen else ACTIVE_AGENT_PROBE_INTERVAL_SEC
        if type(self).observe is not TmuxBatchWorker.observe:
            force = True
        if not force and last_probe_monotonic and now - last_probe_monotonic < interval:
            return None, last_probe_monotonic
        return self._probe_agent_liveness_for_file_wait(), now

    def _track_task_completion_signal(
            self,
            *,
            task_status_path: Path | None,
            status_done_seen: bool,
    ) -> bool:
        if task_status_path is None:
            return True
        if not status_done_seen:
            current_status = read_task_status(task_status_path)
            self.current_task_runtime_status = current_status
            if current_status == TASK_STATUS_DONE:
                self._log_event("task_status_done", task_status_path=str(task_status_path))
                return True
        return status_done_seen

    def _terminal_idle_elapsed_sec(self) -> float:
        if self.terminal_recently_changed:
            return 0.0
        if self._last_terminal_change_monotonic:
            return max(0.0, time.monotonic() - self._last_terminal_change_monotonic)
        last_changed_at = str(self.last_terminal_changed_at or "").strip()
        if not last_changed_at:
            return 0.0
        try:
            changed_at = datetime.fromisoformat(last_changed_at.replace("Z", "+00:00"))
        except ValueError:
            return 0.0
        return max(0.0, time.time() - changed_at.timestamp())

    def _contract_wait_stalled(
            self,
            *,
            observation: WorkerObservation,
            status_done_seen: bool,
            output_exists: bool,
            invalid_output_stable_sec: float = 0.0,
    ) -> tuple[bool, float]:
        if status_done_seen:
            return False, 0.0
        if output_exists and invalid_output_stable_sec < TASK_CONTRACT_STALL_IDLE_SEC:
            return False, 0.0
        if self.current_task_runtime_status != TASK_STATUS_RUNNING:
            return False, 0.0
        if observation.current_command in SHELL_COMMANDS or not self._agent_running(observation.current_command):
            return False, 0.0
        agent_state = self.get_agent_state(observation)
        idle_elapsed = self._terminal_idle_elapsed_sec()
        if output_exists and invalid_output_stable_sec >= TASK_CONTRACT_STALL_IDLE_SEC:
            if agent_state in {AgentRuntimeState.BUSY, AgentRuntimeState.STARTING}:
                if self.terminal_recently_changed or idle_elapsed < TASK_CONTRACT_STALL_IDLE_SEC:
                    return False, idle_elapsed
                return True, idle_elapsed
            return True, invalid_output_stable_sec
        if agent_state in {AgentRuntimeState.BUSY, AgentRuntimeState.STARTING}:
            return False, idle_elapsed
        if idle_elapsed < TASK_CONTRACT_STALL_IDLE_SEC:
            return False, idle_elapsed
        return True, idle_elapsed

    @staticmethod
    def _build_terminal_signature(terminal_text: str) -> str:
        normalized_lines = [line.rstrip() for line in clean_ansi(terminal_text).splitlines()]
        while normalized_lines and not normalized_lines[-1].strip():
            normalized_lines.pop()
        payload = "\n".join(normalized_lines[-160:]).strip()
        if not payload:
            return ""
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _reset_terminal_activity(self) -> None:
        self.last_terminal_signature = ""
        self.last_terminal_changed_at = ""
        self.terminal_recently_changed = False
        self._last_terminal_change_monotonic = 0.0

    def _update_terminal_activity(self, terminal_text: str, *, observed_at: str) -> None:
        now = time.monotonic()
        signature = self._build_terminal_signature(terminal_text)
        if signature != self.last_terminal_signature:
            self.last_terminal_signature = signature
            self.last_terminal_changed_at = observed_at if signature else ""
            self._last_terminal_change_monotonic = now if signature else 0.0
        elif signature and not self.last_terminal_changed_at:
            self.last_terminal_changed_at = observed_at
            self._last_terminal_change_monotonic = now
        elif signature and not self._last_terminal_change_monotonic and self.last_terminal_changed_at:
            try:
                last_changed_at = datetime.fromisoformat(str(self.last_terminal_changed_at).replace("Z", "+00:00"))
            except ValueError:
                last_changed_at = None
            if last_changed_at is not None:
                elapsed_sec = max(0.0, time.time() - last_changed_at.timestamp())
                self._last_terminal_change_monotonic = max(0.0, now - elapsed_sec)
        self.terminal_recently_changed = bool(signature) and bool(self._last_terminal_change_monotonic) and (
            now - self._last_terminal_change_monotonic < TERMINAL_ACTIVITY_IDLE_WINDOW_SEC
        )

    def _agent_running(self, current_command: str) -> bool:
        return _agent_command_running(current_command, self.config.expected_current_commands())

    def _codex_title_candidates(self) -> tuple[str, ...]:
        candidates = ["TmuxCodingTeam"]
        work_dir_name = self.work_dir.name.strip()
        if work_dir_name and work_dir_name not in candidates:
            candidates.append(work_dir_name)
        return tuple(candidates)

    def _title_indicates_ready(self, pane_title: str) -> bool:
        title = str(pane_title or "").strip()
        if not title:
            return False
        if self.config.vendor == Vendor.CODEX:
            return not bool(BRAILLE_SPINNER_PREFIX_RE.match(title))
        if self.config.vendor == Vendor.CLAUDE:
            return _claude_title_indicates_ready(title)
        if self.config.vendor == Vendor.GEMINI:
            return title.startswith("◇") and "Ready" in title
        return False

    def _title_indicates_busy(self, pane_title: str) -> bool:
        title = str(pane_title or "").strip()
        if not title:
            return False
        if self.config.vendor == Vendor.CODEX:
            return bool(BRAILLE_SPINNER_PREFIX_RE.match(title))
        if self.config.vendor == Vendor.CLAUDE:
            return _claude_title_indicates_busy(title)
        if self.config.vendor == Vendor.GEMINI:
            return title.startswith("✦") and "Working" in title
        return False

    def is_agent_alive(self, observation: WorkerObservation | None = None) -> bool:
        current_observation = observation
        if current_observation is None:
            session_exists = self.session_exists()
            if not session_exists:
                return False
            current_command = self.current_command
            pane_dead = False
            if not self.pane_id or not self.target_exists():
                return False
        else:
            session_exists = current_observation.session_exists
            current_command = current_observation.current_command or self.current_command
            pane_dead = current_observation.pane_dead
        if not session_exists or not self.pane_id:
            return False
        if pane_dead:
            return False
        normalized_command = str(current_command or "").strip()
        if not normalized_command or normalized_command in SHELL_COMMANDS:
            return False
        return True

    def is_agent_started(self) -> bool:
        return self.agent_started

    def has_ever_launched(self) -> bool:
        return worker_state_has_launch_evidence(self)

    def _is_prelaunch_without_session(self) -> bool:
        if self.agent_started:
            return False
        if str(self.pane_id or "").strip() and not worker_state_is_prelaunch_active(self.read_state()):
            return False
        try:
            return not self.session_exists()
        except Exception:  # noqa: BLE001
            return True

    def _is_prelaunch_shell_window(self, observation: WorkerObservation | None = None) -> bool:
        if self.agent_started:
            return False
        current_observation = observation
        if current_observation is None:
            if not self.session_exists():
                return False
            current_command = self.current_command
            pane_dead = False
            if not self.pane_id or not self.target_exists():
                return False
        else:
            if not current_observation.session_exists:
                return False
            current_command = current_observation.current_command or self.current_command
            pane_dead = current_observation.pane_dead
        if pane_dead:
            return False
        if not self.pane_id:
            return False
        return str(current_command or "").strip() in SHELL_COMMANDS

    def _classify_agent_state(
            self,
            observation: WorkerObservation | None = None,
            *,
            task_running_override: bool | None = None,
    ) -> AgentRuntimeState:
        if observation is None:
            if self._is_prelaunch_without_session():
                return AgentRuntimeState.STARTING
            if self._is_prelaunch_shell_window(observation):
                return AgentRuntimeState.STARTING
            if not self.is_agent_alive(observation):
                return AgentRuntimeState.DEAD
            if not self.agent_started:
                return AgentRuntimeState.STARTING
            return self.agent_state
        return classify_agent_runtime_state(
            observation,
            context=AgentRuntimeClassifierContext(
                vendor=self.config.vendor,
                agent_started=self.agent_started,
                cached_state=self.agent_state,
                pane_id=self.pane_id,
                expected_current_commands=self.config.expected_current_commands(),
                task_running=(
                    bool(task_running_override)
                    if task_running_override is not None
                    else self.current_task_runtime_status == TASK_STATUS_RUNNING
                ),
                pre_submit_ready_probe=task_running_override is not None,
                title_ready=self._title_indicates_ready(observation.pane_title),
                title_busy=self._title_indicates_busy(observation.pane_title),
            ),
            detector=self.detector,
        )

    def get_agent_state(
            self,
            observation: WorkerObservation | None = None,
            *,
            task_running_override: bool | None = None,
    ) -> AgentRuntimeState:
        return self._classify_agent_state(observation, task_running_override=task_running_override)

    @staticmethod
    def _can_finalize_task_result_from_contract_without_helper(contract: TaskResultContract) -> bool:
        expected_statuses = {
            str(status or "").strip()
            for status in contract.expected_statuses
            if str(status or "").strip()
        }
        if not expected_statuses or not expected_statuses <= {TASK_RESULT_READY}:
            return False
        candidate = build_missing_task_result_finalization_candidate(contract, task_status_path=None)
        return candidate is not None and candidate.requires_agent_ready

    @staticmethod
    def _fresh_completed_task_result_contract_signature(
        *,
        contract: TaskResultContract,
        task_status_path: Path | None,
    ) -> tuple[object, ...] | None:
        candidate = build_missing_task_result_finalization_candidate(
            contract,
            task_status_path=task_status_path,
        )
        if candidate is None or candidate.requires_agent_ready:
            return None
        return candidate.signature

    @staticmethod
    def _can_finalize_missing_task_result_from_contract(
        *,
        contract: TaskResultContract,
        task_status_path: Path | None,
    ) -> bool:
        return build_missing_task_result_finalization_candidate(
            contract,
            task_status_path=task_status_path,
        ) is not None

    def _visible_indicates_agent_starting(self, visible_text: str) -> bool:
        recent_output = "\n".join(str(visible_text or "").splitlines()[-120:])
        if not recent_output.strip():
            return False
        if self.config.vendor == Vendor.CODEX:
            recent_output = _codex_effective_recent_surface(recent_output)
            if _codex_surface_indicates_ready_prompt(recent_output):
                return False
            return any(
                re.search(pattern, recent_output, re.IGNORECASE)
                for pattern in (
                    *CODEX_TRUST_PROMPT_PATTERNS,
                    *CODEX_UPDATE_NOTICE_PATTERNS,
                    *CODEX_MODEL_SELECTION_PROMPT_PATTERNS,
                    *CODEX_STARTING_PATTERNS,
                )
            )
        if self.config.vendor == Vendor.GEMINI:
            return any(
                re.search(pattern, recent_output, re.IGNORECASE)
                for pattern in (
                    *GEMINI_TRUST_PROMPT_PATTERNS,
                    *GEMINI_NOT_READY_PATTERNS,
                )
            )
        if self.config.vendor == Vendor.OPENCODE:
            state = _classify_opencode_surface_state(
                visible_text=recent_output,
                recent_log="",
                current_command=self.current_command or "node",
            )
            return state == AgentRuntimeState.STARTING
        return False

    def _visible_indicates_agent_ready(
            self,
            visible_text: str,
            raw_log_tail: str = "",
            *,
            current_command: str = "",
    ) -> bool:
        recent_output = "\n".join(str(visible_text or raw_log_tail or "").splitlines()[-120:])
        if not recent_output.strip():
            return False
        if self.config.vendor == Vendor.CODEX:
            return False
        if self.config.vendor == Vendor.CLAUDE:
            if any(re.search(pattern, recent_output, re.IGNORECASE | re.MULTILINE) for pattern in CLAUDE_BUSY_PATTERNS):
                return False
            return any(re.search(pattern, recent_output, re.IGNORECASE | re.MULTILINE) for pattern in CLAUDE_READY_PATTERNS)
        if self.config.vendor == Vendor.GEMINI:
            if any(re.search(pattern, recent_output, re.IGNORECASE) for pattern in GEMINI_TRUST_PROMPT_PATTERNS + GEMINI_NOT_READY_PATTERNS):
                return False
            if any(re.search(pattern, recent_output, re.IGNORECASE) for pattern in GEMINI_BUSY_PATTERNS):
                return False
            return any(re.search(pattern, recent_output, re.IGNORECASE) for pattern in GEMINI_INPUT_BOX_PATTERNS + GEMINI_READY_PATTERNS)
        if self.config.vendor == Vendor.OPENCODE:
            state = _classify_opencode_surface_state(
                visible_text=recent_output,
                recent_log=raw_log_tail,
                current_command=current_command or self.current_command or "node",
            )
            return state == AgentRuntimeState.READY
        return False

    def _visible_ready_signature(self, observation: WorkerObservation) -> str:
        current_command = observation.current_command or self.current_command
        if not self._agent_running(current_command):
            return ""
        surface = observation.visible_text or observation.raw_log_tail
        if self._visible_indicates_agent_starting(surface):
            return ""
        if not self._visible_indicates_agent_ready(
            observation.visible_text,
            observation.raw_log_tail,
            current_command=current_command,
        ):
            return ""
        signature = self._build_terminal_signature("\n".join(
            part for part in (observation.pane_title, surface) if part
        ))
        return f"{self.config.vendor.value}-visible-ready:{signature or 'ready'}"

    def _observation_indicates_ready_or_idle_surface(self, observation: WorkerObservation) -> bool:
        current_command = observation.current_command or self.current_command
        if not self._agent_running(current_command):
            return False
        if self._title_indicates_ready(observation.pane_title):
            return True
        if self.config.vendor == Vendor.CODEX:
            surface = "\n".join(part for part in (observation.visible_text, observation.raw_log_tail) if str(part or "").strip())
            return _codex_surface_indicates_ready_input(_codex_effective_recent_surface(surface))
        return self._visible_indicates_agent_ready(
            observation.visible_text,
            observation.raw_log_tail,
            current_command=current_command,
        )

    def _stale_busy_without_contract_detail(
            self,
            *,
            observation: WorkerObservation,
            phase: str,
            artifact_path: str | Path,
            output_exists: bool,
            status_done_seen: bool,
    ) -> str:
        if status_done_seen or output_exists:
            return ""
        runtime_status = str(self.current_task_runtime_status or "").strip().lower()
        if runtime_status and runtime_status != TASK_STATUS_RUNNING:
            return ""
        current_command = str(observation.current_command or "").strip()
        if current_command in SHELL_COMMANDS or not self._agent_running(current_command):
            return ""
        agent_state = self.get_agent_state(observation)
        if agent_state not in {AgentRuntimeState.BUSY, AgentRuntimeState.STARTING}:
            return ""
        idle_elapsed = self._terminal_idle_elapsed_sec()
        if idle_elapsed < TASK_CONTRACT_STALL_IDLE_SEC:
            return ""
        if not self._observation_indicates_ready_or_idle_surface(observation):
            return ""
        return (
            f"phase={phase} artifact_path={artifact_path} "
            f"agent_state={agent_state.value} idle_sec={idle_elapsed:.1f} "
            f"current_command={current_command} pane_title={str(observation.pane_title or '').strip()}"
        )

    def _raise_if_stale_busy_without_contract(
            self,
            *,
            observation: WorkerObservation,
            phase: str,
            artifact_path: str | Path,
            output_exists: bool,
            status_done_seen: bool,
            contract_error_prefix: str,
    ) -> None:
        detail = self._stale_busy_without_contract_detail(
            observation=observation,
            phase=phase,
            artifact_path=artifact_path,
            output_exists=output_exists,
            status_done_seen=status_done_seen,
        )
        if not detail:
            return
        self.dispatch_state = "delayed"
        self.dispatch_reason = f"{STALE_BUSY_WITHOUT_CONTRACT_REASON_PREFIX}:{detail}"
        raise RuntimeError(f"{contract_error_prefix}: {self.dispatch_reason}")

    def _infer_wrapper_state(
            self,
            *,
            current_command: str,
            visible_text: str,
            raw_log_tail: str = "",
    ) -> WrapperState:
        if not self._agent_running(current_command):
            return WrapperState.NOT_READY
        if self.config.vendor != Vendor.CODEX and visible_text and self._visible_indicates_agent_starting(visible_text):
            return WrapperState.NOT_READY
        if self.terminal_recently_changed:
            return WrapperState.NOT_READY
        if self.agent_started and self._title_indicates_ready(self.last_pane_title):
            return WrapperState.READY
        if self.config.vendor == Vendor.OPENCODE and self.agent_started and self._visible_indicates_agent_ready(
                visible_text,
                raw_log_tail,
                current_command=current_command,
        ):
            return WrapperState.READY
        return WrapperState.NOT_READY

    def _mark_agent_ready_from_observation(
            self,
            observation: WorkerObservation,
            *,
            note: str = "agent_ready",
    ) -> None:
        current_command = observation.current_command or self.current_command
        self.agent_ready = True
        self.agent_started = True
        self.agent_state = AgentRuntimeState.READY
        self.wrapper_state = WrapperState.READY
        self.last_pane_title = observation.pane_title or self.last_pane_title
        self.current_command = current_command
        self.current_path = observation.current_path or self.current_path
        self.last_heartbeat_at = observation.observed_at or self.last_heartbeat_at
        previous = self.read_state()
        extra: dict[str, object] = {
            "agent_alive": self.is_agent_alive(observation),
            "agent_ready": True,
            "agent_state": AgentRuntimeState.READY.value,
            "current_command": current_command,
            "current_path": observation.current_path or self.current_path,
            "pane_title": self.last_pane_title,
            "last_heartbeat_at": self.last_heartbeat_at,
        }
        result_status = str(previous.get("result_status", "pending") or "").strip().lower()
        if result_status in {"", "running", "pending"}:
            extra["result_status"] = WorkerStatus.READY.value
        current_task_runtime_status = self.current_task_runtime_status or str(
            previous.get("current_task_runtime_status", "")
        )
        if current_task_runtime_status == TASK_STATUS_RUNNING:
            self.current_task_runtime_status = ""
            extra["current_task_runtime_status"] = ""
        self._write_state(
            WorkerStatus.READY,
            note=note,
            extra=extra,
        )

    def _mark_turn_submitted_busy(
            self,
            *,
            label: str,
            started_at: str,
            task_status_path: Path,
            result_path: Path,
            attempt: int,
            completion_contract: TurnFileContract | None,
    ) -> None:
        self.dispatch_state = "submitted"
        self.dispatch_reason = ""
        self.agent_ready = False
        self.agent_started = True
        self.agent_state = AgentRuntimeState.BUSY
        self.wrapper_state = WrapperState.NOT_READY
        self.current_task_runtime_status = TASK_STATUS_RUNNING
        self._write_state(
            WorkerStatus.RUNNING,
            note=f"submitted:{label}",
            extra={
                "label": label,
                "started_at": started_at,
                "phase": "submitted",
                "agent_ready": False,
                "agent_started": True,
                "agent_state": AgentRuntimeState.BUSY.value,
                "current_command": self.current_command,
                "current_path": self.current_path,
                "current_turn_id": completion_contract.turn_id if completion_contract else "",
                "current_turn_phase": completion_contract.phase if completion_contract else "",
                "current_turn_status_path": str(completion_contract.status_path) if completion_contract else "",
                "current_task_status_path": str(task_status_path),
                "current_task_result_path": str(result_path) if self.current_task_result_path else "",
                "current_task_runtime_status": TASK_STATUS_RUNNING,
                "dispatch_state": self.dispatch_state,
                "dispatch_reason": self.dispatch_reason,
                "result_status": "running",
                "retry_count": attempt - 1,
            },
        )

    def _wait_for_agent_ready(self, timeout_sec: float = 60.0) -> None:
        deadline = time.monotonic() + timeout_sec
        previous_ready_signature = ""
        stable_count = 0
        while time.monotonic() < deadline:
            raise_if_runtime_shutdown_requested("waiting for agent ready")
            observation = self.observe(tail_lines=220)
            if not observation.session_exists:
                raise RuntimeError("tmux pane exited while agent was starting")
            if observation.pane_dead:
                raise RuntimeError(f"tmux pane died while agent was starting:\n{self.capture_visible(120)}")

            current_command = observation.current_command
            visible = observation.raw_log_tail or observation.visible_text
            fallback_visible = observation.visible_text
            if (
                self._maybe_handle_codex_boot_prompt(visible)
                or self._maybe_handle_codex_boot_prompt(fallback_visible)
                or self._maybe_handle_gemini_boot_prompt(visible)
                or self._maybe_handle_gemini_boot_prompt(fallback_visible)
            ):
                time.sleep(0.6)
                previous_ready_signature = ""
                stable_count = 0
                continue

            if self._agent_running(current_command):
                ready_signature = observation.pane_title if self._title_indicates_ready(observation.pane_title) else ""
                if not ready_signature:
                    ready_signature = self._visible_ready_signature(observation)
                if ready_signature and ready_signature == previous_ready_signature:
                    stable_count += 1
                else:
                    stable_count = 1 if ready_signature else 0
                if stable_count >= 2 and ready_signature:
                    self._mark_agent_ready_from_observation(observation)
                    return
            elif current_command in SHELL_COMMANDS and previous_ready_signature:
                raise RuntimeError(f"agent exited back to shell while starting:\n{visible}")

            previous_ready_signature = ready_signature if self._agent_running(current_command) else ""
            time.sleep(0.5)

        raise RuntimeError(f"Timed out waiting for agent ready.\n{self.capture_visible(240)}")

    def launch_agent(self, timeout_sec: float = 60.0) -> None:
        last_error: Exception | None = None
        max_attempts = 1
        for attempt in range(1, max_attempts + 1):
            try:
                with self.launch_coordinator.startup_slot(self.config.vendor):
                    if not self.pane_id or not self.target_exists():
                        self.create_session()
                    else:
                        self._ensure_health_supervisor_started()
                    self._wait_for_shell_ready()
                    self.agent_state = AgentRuntimeState.STARTING
                    self._append_transcript("launch / command", f"```bash\n{self.launch_command}\n```")
                    self._log_event("launch_attempt", attempt=attempt, vendor=self.config.vendor.value)
                    self._send_text(self.launch_command, enter_count=1)
                    self._wait_for_agent_ready(timeout_sec=timeout_sec)
                self.launch_coordinator.record_launch_result(self.config.vendor, success=True)
                return
            except Exception as error:
                last_error = error
                self.launch_coordinator.record_launch_result(self.config.vendor, success=False)
                self.agent_state = AgentRuntimeState.DEAD
                self.agent_ready = False
                self.agent_started = False
                self.wrapper_state = WrapperState.NOT_READY
                self._log_event("launch_failed", attempt=attempt, error=str(error))
                self.mark_awaiting_reconfiguration(
                    reason_text=(
                        f"{self.session_name} 启动失败，系统不会自动重试或重建该智能体。\n"
                        f"请人工进入会话处理: tmux attach -t {self.session_name}\n"
                        f"原因: {error}"
                    )
                )
                if attempt >= max_attempts:
                    break
                if self.session_exists():
                    try:
                        self._stop_health_supervisor()
                        self.backend.kill_session(self.session_name)
                    except Exception:
                        pass
                self.pane_id = ""
        if last_error is not None:
            raise last_error

    def _raise_ready_relaunch_blocked(self, *, reason: str) -> None:
        reason_text = str(reason or "").strip() or "agent_not_ready"
        intervention_text = (
            f"检测到 {self.session_name} 需要重新启动或重建，但系统不会自动执行。\n"
            f"请人工进入会话处理: tmux attach -t {self.session_name}\n"
            f"原因: {reason_text}"
        )
        self.mark_awaiting_reconfiguration(reason_text=intervention_text)
        self._log_event(
            "ensure_ready_relaunch_blocked_manual_intervention_required",
            reason=reason_text,
        )
        raise RuntimeError(intervention_text)

    def ensure_agent_ready(self, timeout_sec: float = 60.0) -> None:
        if self.session_exists() and self.pane_id:
            self._ensure_health_supervisor_started()
        if not self.pane_id or not self.target_exists():
            if self.has_ever_launched():
                self._raise_ready_relaunch_blocked(reason="tmux pane missing")
            self.agent_state = AgentRuntimeState.STARTING
            self._log_event("ensure_ready_initial_launch", reason="missing_pane")
            self.launch_agent(timeout_sec=timeout_sec)
            return

        observation = self.observe(tail_lines=160)
        current_command = observation.current_command
        if self.get_agent_state(observation) == AgentRuntimeState.READY:
            self._mark_agent_ready_from_observation(observation)
            return

        if current_command in SHELL_COMMANDS:
            if self.has_ever_launched():
                self._raise_ready_relaunch_blocked(reason="agent exited back to shell")
            self.agent_state = AgentRuntimeState.STARTING
            self.agent_started = False
            self.agent_ready = False
            self.wrapper_state = WrapperState.NOT_READY
            self._log_event("ensure_ready_initial_launch", reason="shell_fallback")
            self.launch_agent(timeout_sec=timeout_sec)
            return

        self._wait_for_agent_ready(timeout_sec=timeout_sec)

    def _mark_turn_start_ready_from_observation(
            self,
            observation: WorkerObservation,
            *,
            label: str,
            delayed: bool,
    ) -> bool:
        if not observation.session_exists or observation.pane_dead:
            return False
        current_command = observation.current_command or self.current_command
        if current_command in SHELL_COMMANDS or not self._agent_running(current_command):
            return False
        self._raise_on_provider_runtime_error(observation, context="probing agent ready before turn")
        agent_state = self.get_agent_state(observation, task_running_override=False)
        if agent_state != AgentRuntimeState.READY and not self._observation_indicates_ready_or_idle_surface(observation):
            return False
        self._mark_agent_ready_from_observation(
            observation,
            note=f"turn_start_ready:{label}" if label else "turn_start_ready",
        )
        self._log_event(
            "turn_start_ready_from_idle_surface",
            label=label,
            delayed=delayed,
            current_command=current_command,
            pane_title=observation.pane_title,
        )
        return True

    def _try_mark_turn_start_ready_from_current_observation(self, *, label: str, delayed: bool) -> bool:
        if not self.pane_id:
            return False
        try:
            if not self.session_exists() or not self.target_exists():
                return False
            observation = self.observe(tail_lines=220)
        except Exception as error:  # noqa: BLE001
            self._log_event(
                "turn_start_ready_probe_failed",
                label=label,
                delayed=delayed,
                error=str(error),
            )
            return False
        return self._mark_turn_start_ready_from_observation(
            observation,
            label=label,
            delayed=delayed,
        )

    def _confirm_busy_agent_for_turn_start_ready_wait(
            self,
            *,
            error: Exception,
            timeout_sec: float,
            label: str = "",
    ) -> bool:
        observation = self._probe_agent_liveness_for_file_wait()
        if not observation.session_exists:
            raise RuntimeError("tmux pane exited while waiting for agent ready") from error
        if observation.pane_dead:
            raise RuntimeError(
                f"tmux pane died while waiting for agent ready:\n{self._diagnostic_visible_tail(160)}"
            ) from error
        current_command = observation.current_command or self.current_command
        if current_command in SHELL_COMMANDS:
            self.agent_ready = False
            self.wrapper_state = WrapperState.NOT_READY
            raise RuntimeError(
                f"agent exited back to shell while waiting for agent ready:\n{self._diagnostic_visible_tail(160)}"
            ) from error
        if not self._agent_running(current_command):
            raise error
        self._raise_on_provider_runtime_error(observation, context="waiting for agent ready before turn")
        if self._mark_turn_start_ready_from_observation(observation, label=label, delayed=True):
            return True
        agent_state = self.get_agent_state(observation, task_running_override=False)
        if agent_state != AgentRuntimeState.BUSY:
            raise error
        self.agent_ready = False
        self.agent_started = True
        self.agent_state = AgentRuntimeState.BUSY
        self.wrapper_state = WrapperState.NOT_READY
        self.current_command = current_command
        self.current_path = observation.current_path or self.current_path
        self.last_heartbeat_at = observation.observed_at
        self._log_event(
            "turn_start_ready_wait_extended_for_busy_agent",
            timeout_sec=timeout_sec,
            current_command=current_command,
            pane_title=observation.pane_title,
        )
        return False

    def _restart_stale_busy_agent_for_turn_start(self, *, timeout_sec: float, reason: str) -> None:
        _ = timeout_sec
        reason_text = str(reason or "").strip() or "stale_busy_agent"
        intervention_text = (
            f"检测到 {self.session_name} 上一轮仍处于异常 BUSY/失败状态。\n"
            "系统不会自动重启该智能体。\n"
            f"请人工进入会话处理: tmux attach -t {self.session_name}\n"
            f"原因: {reason_text}"
        )
        self.mark_awaiting_reconfiguration(reason_text=intervention_text)
        self._log_event(
            "turn_start_relaunch_blocked_manual_intervention_required",
            reason=reason_text,
        )
        raise RuntimeError(intervention_text)

    def _busy_agent_can_continue_waiting_for_turn_start(self, *, reason: str) -> bool:
        if not self.last_terminal_signature:
            return False
        idle_elapsed = self._terminal_idle_elapsed_sec()
        if not self.terminal_recently_changed and idle_elapsed >= TASK_CONTRACT_STALL_IDLE_SEC:
            return False
        self._log_event(
            "turn_start_busy_agent_still_active_waiting",
            reason=str(reason or "").strip() or "busy_agent_active",
            idle_sec=round(idle_elapsed, 3),
        )
        return True

    def _ensure_agent_ready_for_turn_start(
            self,
            *,
            timeout_sec: float,
            previous_task_runtime_status: str = "",
            previous_worker_status: str = "",
            initial_timeout_sec: float | None = None,
            label: str = "",
    ) -> None:
        ready_timeout = initial_timeout_sec if initial_timeout_sec is not None else 60.0
        self._log_event("ready_wait_start", label=label, timeout_sec=ready_timeout)
        try:
            if self._try_mark_turn_start_ready_from_current_observation(label=label, delayed=False):
                self._log_event("ready_wait_done", label=label, delayed=False)
                return
            if initial_timeout_sec is None:
                self.ensure_agent_ready()
            else:
                self.ensure_agent_ready(timeout_sec=initial_timeout_sec)
            self._log_event("ready_wait_done", label=label, delayed=False)
            return
        except Exception as error:
            if not is_agent_ready_timeout_error(error):
                raise
            self.dispatch_state = "delayed"
            self.dispatch_reason = f"ready_wait_timeout:{error}"
            self._write_state(
                WorkerStatus.RUNNING,
                note=f"dispatch_delayed:{label}" if label else "dispatch_delayed",
                extra={
                    "dispatch_state": self.dispatch_state,
                    "dispatch_reason": self.dispatch_reason,
                    "dispatch_timeout_sec": ready_timeout,
                },
            )
            self._log_event(
                "ready_wait_delayed",
                label=label,
                timeout_sec=ready_timeout,
                error=str(error),
            )
            current_error = error
        turn_start_deadline = time.monotonic() + max(float(timeout_sec), 0.0)
        while True:
            raise_if_runtime_shutdown_requested("waiting for busy agent before turn start")
            if self._confirm_busy_agent_for_turn_start_ready_wait(
                error=current_error,
                timeout_sec=timeout_sec,
                label=label,
            ):
                self._log_event("ready_wait_done", label=label, delayed=True)
                return
            previous_task_status = str(previous_task_runtime_status).strip().lower()
            previous_status = str(previous_worker_status).strip().lower()
            stale_reason = ""
            if previous_task_status == TASK_STATUS_DONE:
                stale_reason = "previous_task_done"
            elif previous_status in {"failed", "stale_failed", "error"}:
                stale_reason = "previous_worker_failed"
            if stale_reason and not self._busy_agent_can_continue_waiting_for_turn_start(reason=stale_reason):
                self._restart_stale_busy_agent_for_turn_start(timeout_sec=timeout_sec, reason=stale_reason)
            remaining_timeout = turn_start_deadline - time.monotonic()
            if remaining_timeout <= 0:
                raise current_error
            try:
                if self._try_mark_turn_start_ready_from_current_observation(label=label, delayed=True):
                    self._log_event("ready_wait_done", label=label, delayed=True)
                    return
                probe_timeout = max(
                    0.1,
                    min(
                        timeout_sec,
                        ready_timeout if ready_timeout > 0 else TURN_START_BUSY_PROBE_TIMEOUT_SEC,
                        TURN_START_BUSY_PROBE_TIMEOUT_SEC,
                    ),
                )
                self.ensure_agent_ready(timeout_sec=probe_timeout)
                self._log_event("ready_wait_done", label=label, delayed=True)
                return
            except Exception as error:
                if not is_agent_ready_timeout_error(error):
                    raise
                current_error = error

    def _build_turn_prompt(
            self,
            prompt: str,
            turn_token: str,
            required_tokens: Sequence[str],
            *,
            task_status_path: Path,
            complete_task_command: str | None = None,
            include_turn_protocol: bool,
    ) -> str:
        del task_status_path
        del complete_task_command
        sections = [str(prompt or "").strip()]
        if include_turn_protocol:
            if required_tokens:
                turn_protocol_prompt = f"""Turn completion protocol:
- Output exactly `{turn_token}` on its own line after the substantive answer.
- Keep the workflow-required token as the final workflow token before the runtime completion marker.
"""
            else:
                turn_protocol_prompt = f"""Turn completion protocol:
- Output exactly `{turn_token}` on its own line after the substantive answer.
- If no workflow token is required, `{turn_token}` must be the final workflow token before the runtime completion marker.
"""
            sections.append(turn_protocol_prompt)
        return "\n\n".join(part for part in sections if part)

    @staticmethod
    def _strip_turn_token(text: str, turn_token: str) -> str:
        lines = [line.rstrip() for line in str(text or "").splitlines()]
        kept = [line for line in lines if line.strip() != turn_token]
        return "\n".join(kept).strip()

    @staticmethod
    def _truncate_source_at_completion_token(source: str, turn_token: str, required_tokens: Sequence[str]) -> str:
        lines = clean_ansi(source).splitlines()
        token_index = -1
        for index, line in enumerate(lines):
            if _extract_protocol_token_from_line(line, [turn_token]) == turn_token:
                token_index = index
        if token_index < 0:
            return clean_ansi(source)
        completion_index = token_index
        if required_tokens:
            for index in range(token_index + 1, len(lines)):
                if _extract_protocol_token_from_line(lines[index], required_tokens):
                    completion_index = index
        return "\n".join(lines[:completion_index + 1])

    def _extract_last_message(self, visible_text: str) -> str:
        return self.detector.extract_last_message(visible_text)

    def _extract_audit_reply_from_source(
            self,
            source: str,
            *,
            turn_token: str,
            required_tokens: Sequence[str],
    ) -> str:
        clean_source = clean_ansi(source)
        if turn_token not in clean_source:
            return ""

        lines = clean_source.splitlines()
        allowed_tokens = set(required_tokens)
        turn_index = -1
        for index, line in enumerate(lines):
            if _extract_protocol_token_from_line(line, [turn_token]) == turn_token:
                turn_index = index
        if turn_index < 0:
            return ""

        token_positions = [
            index
            for index in range(turn_index + 1, len(lines))
            if _extract_protocol_token_from_line(lines[index], allowed_tokens)
        ]
        if not token_positions:
            return ""
        token_index = token_positions[-1]
        final_token = _extract_protocol_token_from_line(lines[token_index], allowed_tokens)
        if final_token == "[[ROUTING_AUDIT:WRITTEN]]":
            return "\n".join([turn_token, final_token])

        segment_start = 0
        for index in range(turn_index - 1, -1, -1):
            if _is_turn_token_line(lines[index]):
                segment_start = index + 1
                break

        tail_window_start = max(segment_start, turn_index - 60)

        def collect_structured_before_turn(max_gap: int = 60) -> list[str]:
            reverse_lines: list[str] = []
            reverse_seen: set[str] = set()
            collecting = False
            gap = 0
            for raw_line in reversed(lines[segment_start:turn_index]):
                canonical = _canonical_audit_line(raw_line)
                if canonical:
                    if _is_prompt_example_audit_line(canonical):
                        continue
                    if canonical not in reverse_seen:
                        reverse_lines.append(canonical)
                        reverse_seen.add(canonical)
                    collecting = True
                    gap = 0
                    continue
                text = clean_ansi(raw_line).strip()
                if is_runtime_noise_line(text):
                    continue
                if not collecting:
                    continue
                gap += 1
                if gap >= max_gap:
                    break
            reverse_lines.reverse()
            return reverse_lines

        unexpected_lines: list[str] = []
        seen_unexpected: set[str] = set()
        structured_lines: list[str] = []
        if final_token == "[[ROUTING_AUDIT:PASS]]":
            for line in list(lines[tail_window_start:turn_index]) + list(lines[turn_index + 1: token_index]):
                text = clean_ansi(line).strip()
                if is_runtime_noise_line(text):
                    continue
                if text in seen_unexpected:
                    continue
                unexpected_lines.append(text)
                seen_unexpected.add(text)
        else:
            structured_lines = collect_structured_before_turn()
            seen: set[str] = set(structured_lines)
            for line in lines[turn_index + 1: token_index]:
                canonical = _canonical_audit_line(line)
                if canonical:
                    if _is_prompt_example_audit_line(canonical):
                        continue
                    if canonical in seen:
                        continue
                    structured_lines.append(canonical)
                    seen.add(canonical)
                    continue
                text = clean_ansi(line).strip()
                if is_runtime_noise_line(text):
                    continue
                if text in seen_unexpected:
                    continue
                unexpected_lines.append(text)
                seen_unexpected.add(text)

        trailing_lines: list[str] = []
        seen_trailing: set[str] = set()
        for line in lines[token_index + 1:]:
            text = clean_ansi(line).strip()
            if is_runtime_noise_line(text):
                continue
            if text in seen_trailing:
                continue
            trailing_lines.append(text)
            seen_trailing.add(text)

        if final_token == "[[ROUTING_AUDIT:REVISE]]" and not structured_lines:
            return ""

        payload_lines = list(structured_lines)
        if final_token == "[[ROUTING_AUDIT:PASS]]":
            payload_lines.extend(unexpected_lines)
        payload_lines.append(turn_token)
        payload_lines.append(final_token)
        payload_lines.extend(trailing_lines)
        return "\n".join(payload_lines)

    def _extract_reply_from_observation(
            self,
            observation: WorkerObservation,
            *,
            turn_token: str,
            required_tokens: Sequence[str],
    ) -> str:
        candidate_sources = [observation.raw_log_tail, observation.visible_text]
        audit_turn = any(token.startswith("[[ROUTING_AUDIT:") for token in required_tokens)
        for source in candidate_sources:
            if not source or turn_token not in source:
                continue
            if audit_turn:
                reply = self._extract_audit_reply_from_source(
                    source,
                    turn_token=turn_token,
                    required_tokens=required_tokens,
                )
                if reply:
                    return self._strip_turn_token(reply, turn_token)
                continue
            truncated_source = self._truncate_source_at_completion_token(source, turn_token, required_tokens)
            try:
                reply = self._extract_last_message(truncated_source)
            except Exception:
                continue
            if reply.strip() == turn_token:
                blocks = BaseOutputDetector._split_blocks(truncated_source)
                if len(blocks) >= 2 and blocks[-1].strip() == turn_token:
                    reply = f"{blocks[-2]}\n{turn_token}"
            if turn_token not in reply:
                continue
            if required_tokens and not extract_final_protocol_token(reply, required_tokens):
                continue
            return self._strip_turn_token(reply, turn_token)
        return ""

    def _extract_required_token_reply_without_turn_token(
            self,
            observation: WorkerObservation,
            *,
            required_tokens: Sequence[str],
    ) -> str:
        if not required_tokens:
            return ""
        if self.get_agent_state(observation) != AgentRuntimeState.READY:
            return ""
        candidate_sources = [observation.visible_text, observation.raw_log_tail]
        for source in candidate_sources:
            if not source:
                continue
            try:
                reply = self.detector.extract_last_message(source)
            except Exception:
                continue
            lines = clean_ansi(reply).splitlines()
            for line in reversed(lines):
                token = _extract_protocol_token_from_line(line, required_tokens)
                if token:
                    return token
        return ""

    def _wait_for_turn_reply(
            self,
            *,
            baseline_reply: str,
            baseline_visible: str,
            turn_token: str,
            required_tokens: Sequence[str],
            task_status_path: Path | None = None,
            timeout_sec: float,
    ) -> str:
        deadline = time.monotonic() + timeout_sec
        resolved_reply = ""
        status_done_seen = task_status_path is None
        while time.monotonic() < deadline:
            raise_if_runtime_shutdown_requested("waiting for turn reply")
            observation = self.observe(tail_lines=DEFAULT_CAPTURE_TAIL_LINES)
            if not observation.session_exists:
                raise RuntimeError("tmux pane exited while waiting for reply")
            if observation.pane_dead:
                raise RuntimeError(f"tmux pane died while waiting for reply:\n{self.capture_visible(160)}")

            current_command = observation.current_command
            visible = observation.visible_text
            if current_command in SHELL_COMMANDS:
                self.agent_ready = False
                raise RuntimeError(f"agent exited back to shell during turn:\n{visible}")

            status_done_seen = self._track_task_completion_signal(
                task_status_path=task_status_path,
                status_done_seen=status_done_seen,
            )

            reply = self._extract_reply_from_observation(
                observation,
                turn_token=turn_token,
                required_tokens=required_tokens,
            )
            if reply:
                resolved_reply = reply

            if not resolved_reply and baseline_reply:
                time.sleep(0.4)
                continue
            if not resolved_reply and status_done_seen:
                fallback_reply = self._extract_required_token_reply_without_turn_token(
                    observation,
                    required_tokens=required_tokens,
                )
                if fallback_reply:
                    resolved_reply = fallback_reply
            if not resolved_reply:
                time.sleep(0.4)
                continue
            if not status_done_seen:
                time.sleep(0.4)
                continue
            self.current_command = current_command
            self.current_path = observation.current_path
            self.last_heartbeat_at = observation.observed_at
            self.last_reply = resolved_reply
            self.current_task_runtime_status = "done"
            return resolved_reply

        try:
            self.send_special_key("C-c")
        except Exception:
            pass
        self.agent_ready = False
        raise TimeoutError(f"等待智能体回复超时:\n{clean_ansi(self.capture_visible(200))[-4000:]}")

    def run_turn(
            self,
            *,
            label: str,
            prompt: str,
            required_tokens: Sequence[str] = (),
            completion_contract: TurnFileContract | None = None,
            result_contract: TaskResultContract | None = None,
            timeout_sec: float = DEFAULT_COMMAND_TIMEOUT_SEC,
            turn_start_timeout_sec: float | None = None,
            prompt_submit_timeout_sec: float | None = None,
            pre_submit_observation_tail_lines: int | None = None,
            pre_submit_observation_tail_bytes: int | None = None,
    ) -> CommandResult:
        raise_if_runtime_shutdown_requested(f"starting turn {label}")
        started_at = _now_iso()
        last_timeout: TimeoutError | None = None
        for attempt in range(1, 3):
            raise_if_runtime_shutdown_requested(f"starting turn {label}")
            previous_task_runtime_status = self.current_task_runtime_status
            previous_worker_status = str(self.read_state().get("status", "")).strip()
            turn_token = f"[[ACX_TURN:{uuid.uuid4().hex[:8]}:DONE]]"
            task_status_path = self._build_task_status_path(label=label, attempt=attempt)
            result_path = self._build_task_result_path(label=label, attempt=attempt)
            write_task_status(task_status_path, status=TASK_STATUS_RUNNING)
            if result_path.exists():
                result_path.unlink()
            self.current_task_status_path = str(task_status_path)
            self.current_task_result_path = str(result_path) if result_contract is not None else ""
            self.current_task_runtime_status = TASK_STATUS_RUNNING
            self.dispatch_state = "preparing"
            self.dispatch_reason = ""
            prompt_submission_observed = False
            submitted_prompt = self._build_turn_prompt(
                prompt,
                turn_token,
                required_tokens,
                task_status_path=task_status_path,
                include_turn_protocol=completion_contract is None and result_contract is None,
            )
            prompt_hash = hashlib.sha1(submitted_prompt.encode("utf-8")).hexdigest()[:12]
            self._append_transcript(f"{label} / prompt", f"```text\n{submitted_prompt}\n```")
            self._write_state(
                WorkerStatus.RUNNING,
                note=f"turn:{label}",
                extra={
                    "label": label,
                    "started_at": started_at,
                    "last_turn_token": turn_token,
                    "last_prompt_hash": prompt_hash,
                    "current_turn_id": completion_contract.turn_id if completion_contract else "",
                    "current_turn_phase": completion_contract.phase if completion_contract else "",
                    "current_turn_status_path": str(completion_contract.status_path) if completion_contract else "",
                    "current_task_status_path": str(task_status_path),
                    "current_task_result_path": self.current_task_result_path,
                    "current_task_runtime_status": TASK_STATUS_RUNNING,
                    "dispatch_state": self.dispatch_state,
                    "dispatch_reason": self.dispatch_reason,
                    "result_status": "running",
                    "retry_count": attempt - 1,
                },
            )
            self._log_event(
                "dispatch_start",
                label=label,
                attempt=attempt,
                turn_start_timeout_sec=turn_start_timeout_sec,
                prompt_submit_timeout_sec=prompt_submit_timeout_sec,
            )

            try:
                self._ensure_agent_ready_for_turn_start(
                    timeout_sec=timeout_sec,
                    previous_task_runtime_status=previous_task_runtime_status,
                    previous_worker_status=previous_worker_status,
                    initial_timeout_sec=turn_start_timeout_sec,
                    label=label,
                )
                raise_if_runtime_shutdown_requested(f"submitting turn {label}")
                needs_pre_submit_observation = (
                    result_contract is not None
                    or (completion_contract is None and result_contract is None)
                )
                baseline_observation: WorkerObservation | None = None
                baseline_visible = ""
                baseline_raw_log_tail = ""
                baseline_reply = self.last_reply
                if needs_pre_submit_observation:
                    default_observe_tail_lines = (
                        RESULT_CONTRACT_PRE_SUBMIT_CAPTURE_TAIL_LINES
                        if result_contract is not None
                        else DEFAULT_CAPTURE_TAIL_LINES
                    )
                    observe_tail_lines = (
                        default_observe_tail_lines
                        if pre_submit_observation_tail_lines is None
                        else max(int(pre_submit_observation_tail_lines), 1)
                    )
                    observe_tail_bytes = (
                        24000
                        if pre_submit_observation_tail_bytes is None
                        else max(int(pre_submit_observation_tail_bytes), 1)
                    )
                    observe_started_monotonic = time.monotonic()
                    self._log_event(
                        "pre_submit_observe_start",
                        label=label,
                        tail_lines=observe_tail_lines,
                        tail_bytes=observe_tail_bytes,
                    )
                    baseline_observation = self.observe(
                        tail_lines=observe_tail_lines,
                        tail_bytes=observe_tail_bytes,
                    )
                    self._log_event(
                        "pre_submit_observe_done",
                        label=label,
                        tail_lines=observe_tail_lines,
                        tail_bytes=observe_tail_bytes,
                        elapsed_ms=round((time.monotonic() - observe_started_monotonic) * 1000, 3),
                    )
                    baseline_visible = baseline_observation.visible_text
                    baseline_raw_log_tail = baseline_observation.raw_log_tail
                    baseline_reply = self.last_reply
                self.dispatch_state = "submitting"
                self.dispatch_reason = ""
                self._write_state(
                    WorkerStatus.RUNNING,
                    note=f"turn:{label}",
                    extra={
                        "label": label,
                        "started_at": started_at,
                        "phase": "submit",
                        "current_command": self.current_command,
                        "current_path": self.current_path,
                        "current_task_status_path": str(task_status_path),
                        "current_task_result_path": self.current_task_result_path,
                        "current_task_runtime_status": TASK_STATUS_RUNNING,
                        "dispatch_state": self.dispatch_state,
                        "dispatch_reason": self.dispatch_reason,
                        "retry_count": attempt - 1,
                    },
                )
                self._log_event("send_text_start", label=label, size=len(submitted_prompt))
                try:
                    self._send_text(submitted_prompt)
                except subprocess.TimeoutExpired as error:
                    self.dispatch_state = "delayed"
                    self.dispatch_reason = f"prompt_dispatch_timeout:{error}"
                    self._write_state(
                        WorkerStatus.RUNNING,
                        note=f"dispatch_delayed:{label}",
                        extra={
                            "dispatch_state": self.dispatch_state,
                            "dispatch_reason": self.dispatch_reason,
                        },
                    )
                    self._log_event("prompt_dispatch_timeout", label=label, timeout_sec=getattr(error, "timeout", 0.0))
                    raise TimeoutError(str(error)) from error
                self._log_event("send_text_done", label=label, size=len(submitted_prompt))
                self._mark_turn_submitted_busy(
                    label=label,
                    started_at=started_at,
                    task_status_path=task_status_path,
                    result_path=result_path,
                    attempt=attempt,
                    completion_contract=completion_contract,
                )
                prompt_confirmation_timeout = min(
                    timeout_sec,
                    prompt_submit_timeout_sec if prompt_submit_timeout_sec is not None else 20.0,
                )
                if completion_contract is not None:
                    try:
                        self._wait_for_prompt_submission(prompt=submitted_prompt, timeout_sec=prompt_confirmation_timeout)
                        self._log_event("prompt_confirm_done", label=label, timeout_sec=prompt_confirmation_timeout)
                        prompt_submission_observed = True
                    except TimeoutError as error:
                        self.dispatch_state = "delayed"
                        self.dispatch_reason = f"prompt_confirm_timeout:{error}"
                        self._write_state(
                            WorkerStatus.RUNNING,
                            note=f"dispatch_delayed:{label}",
                            extra={
                                "dispatch_state": self.dispatch_state,
                                "dispatch_reason": self.dispatch_reason,
                                "dispatch_timeout_sec": prompt_confirmation_timeout,
                            },
                        )
                        self._log_event("prompt_confirm_timeout", label=label, timeout_sec=prompt_confirmation_timeout)
                        prompt_submission_observed = True
                    file_result = self.wait_for_turn_artifacts(
                        contract=completion_contract,
                        task_status_path=task_status_path,
                        timeout_sec=timeout_sec,
                    )
                    reply = json.dumps(
                        {
                            "status_path": file_result.status_path,
                            "artifact_paths": file_result.artifact_paths,
                            "artifact_hashes": file_result.artifact_hashes,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                elif result_contract is not None:
                    if self._can_finalize_task_result_from_contract_without_helper(result_contract):
                        if baseline_observation is None:
                            raise AssertionError("result contract turn requires pre-submit baseline observation")
                        task_result = self._wait_for_ready_task_result_after_submit(
                            contract=result_contract,
                            task_status_path=task_status_path,
                            result_path=result_path,
                            timeout_sec=timeout_sec,
                            prompt=submitted_prompt,
                            baseline_observation=baseline_observation,
                        )
                    else:
                        try:
                            self._wait_for_prompt_submission(prompt=submitted_prompt, timeout_sec=prompt_confirmation_timeout)
                            self._log_event("prompt_confirm_done", label=label, timeout_sec=prompt_confirmation_timeout)
                            prompt_submission_observed = True
                        except TimeoutError as error:
                            self.dispatch_state = "delayed"
                            self.dispatch_reason = f"prompt_confirm_timeout:{error}"
                            self._write_state(
                                WorkerStatus.RUNNING,
                                note=f"dispatch_delayed:{label}",
                                extra={
                                    "dispatch_state": self.dispatch_state,
                                    "dispatch_reason": self.dispatch_reason,
                                    "dispatch_timeout_sec": prompt_confirmation_timeout,
                                },
                            )
                            self._log_event("prompt_confirm_timeout", label=label, timeout_sec=prompt_confirmation_timeout)
                            prompt_submission_observed = True
                        task_result = self.wait_for_task_result(
                            contract=result_contract,
                            task_status_path=task_status_path,
                            result_path=result_path,
                            timeout_sec=timeout_sec,
                            baseline_visible=baseline_visible,
                            baseline_raw_log_tail=baseline_raw_log_tail,
                        )
                    reply = json.dumps(task_result.payload, ensure_ascii=False, indent=2)
                else:
                    reply = self._wait_for_turn_reply(
                        baseline_reply=baseline_reply,
                        baseline_visible=baseline_visible,
                        turn_token=turn_token,
                        required_tokens=required_tokens,
                        task_status_path=task_status_path,
                        timeout_sec=timeout_sec,
                    )
                finished_at = _now_iso()
                self.current_task_runtime_status = read_task_status(task_status_path)
                self.agent_ready = True
                self.agent_started = True
                self.agent_state = AgentRuntimeState.READY
                self.dispatch_state = ""
                self.dispatch_reason = ""
                self.wrapper_state = (
                    WrapperState.READY
                    if self._title_indicates_ready(self.last_pane_title)
                    else WrapperState.NOT_READY
                )
                result = CommandResult(
                    label=label,
                    command=submitted_prompt,
                    exit_code=0,
                    raw_output=reply,
                    clean_output=reply,
                    started_at=started_at,
                    finished_at=finished_at,
                )
                self._record_result(
                    result,
                    status=WorkerStatus.SUCCEEDED,
                    note=f"done:{label}",
                    extra={
                        "label": label,
                        "result_status": "succeeded",
                        "retry_count": attempt - 1,
                        "current_turn_id": completion_contract.turn_id if completion_contract else "",
                        "current_turn_phase": completion_contract.phase if completion_contract else "",
                        "current_turn_status_path": str(completion_contract.status_path) if completion_contract else "",
                        "current_task_status_path": str(task_status_path),
                        "current_task_result_path": self.current_task_result_path,
                        "current_task_runtime_status": self.current_task_runtime_status,
                        "dispatch_state": self.dispatch_state,
                        "dispatch_reason": self.dispatch_reason,
                    },
                )
                return result
            except TimeoutError as error:
                if (
                        not prompt_submission_observed
                        and (completion_contract is not None or result_contract is not None)
                ):
                    prompt_submission_observed = self._infer_prompt_submission_from_busy_agent_after_timeout()
                if completion_contract is not None:
                    file_result = self._try_finalize_turn_artifacts_after_timeout(
                        contract=completion_contract,
                        task_status_path=task_status_path,
                        prompt_submission_observed=prompt_submission_observed,
                    )
                    if file_result is not None:
                        reply = json.dumps(
                            {
                                "status_path": file_result.status_path,
                                "artifact_paths": file_result.artifact_paths,
                                "artifact_hashes": file_result.artifact_hashes,
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                        finished_at = _now_iso()
                        result = CommandResult(
                            label=label,
                            command=submitted_prompt,
                            exit_code=0,
                            raw_output=reply,
                            clean_output=reply,
                            started_at=started_at,
                            finished_at=finished_at,
                        )
                        self._record_result(
                            result,
                            status=WorkerStatus.SUCCEEDED,
                            note=f"done:{label}",
                            extra={
                                "label": label,
                                "result_status": "succeeded",
                                "retry_count": attempt - 1,
                                "current_turn_id": completion_contract.turn_id,
                                "current_turn_phase": completion_contract.phase,
                                "current_turn_status_path": str(completion_contract.status_path),
                                "current_task_status_path": str(task_status_path),
                                "current_task_result_path": self.current_task_result_path,
                                "current_task_runtime_status": self.current_task_runtime_status,
                            },
                        )
                        return result
                if result_contract is not None:
                    task_result = self._try_finalize_task_result_after_prompt_timeout(
                        contract=result_contract,
                        task_status_path=task_status_path,
                        result_path=result_path,
                        baseline_visible=locals().get("baseline_visible", ""),
                        baseline_raw_log_tail=locals().get("baseline_raw_log_tail", ""),
                        prompt_submission_observed=prompt_submission_observed,
                    )
                    if task_result is not None:
                        reply = json.dumps(task_result.payload, ensure_ascii=False, indent=2)
                        finished_at = _now_iso()
                        result = CommandResult(
                            label=label,
                            command=submitted_prompt,
                            exit_code=0,
                            raw_output=reply,
                            clean_output=reply,
                            started_at=started_at,
                            finished_at=finished_at,
                        )
                        self._record_result(
                            result,
                            status=WorkerStatus.SUCCEEDED,
                            note=f"done:{label}",
                            extra={
                                "label": label,
                                "result_status": "succeeded",
                                "retry_count": attempt - 1,
                                "current_task_status_path": str(task_status_path),
                                "current_task_result_path": self.current_task_result_path,
                                "current_task_runtime_status": self.current_task_runtime_status,
                            },
                        )
                        return result
                if completion_contract is not None:
                    file_result = self._wait_for_turn_artifacts_while_agent_busy_after_timeout(
                        label=label,
                        attempt=attempt,
                        timeout_sec=timeout_sec,
                        contract=completion_contract,
                        task_status_path=task_status_path,
                        prompt_submission_observed=prompt_submission_observed,
                    )
                    if file_result is not None:
                        reply = json.dumps(
                            {
                                "status_path": file_result.status_path,
                                "artifact_paths": file_result.artifact_paths,
                                "artifact_hashes": file_result.artifact_hashes,
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                        finished_at = _now_iso()
                        result = CommandResult(
                            label=label,
                            command=submitted_prompt,
                            exit_code=0,
                            raw_output=reply,
                            clean_output=reply,
                            started_at=started_at,
                            finished_at=finished_at,
                        )
                        self._record_result(
                            result,
                            status=WorkerStatus.SUCCEEDED,
                            note=f"done:{label}",
                            extra={
                                "label": label,
                                "result_status": "succeeded",
                                "retry_count": attempt - 1,
                                "current_turn_id": completion_contract.turn_id,
                                "current_turn_phase": completion_contract.phase,
                                "current_turn_status_path": str(completion_contract.status_path),
                                "current_task_status_path": str(task_status_path),
                                "current_task_result_path": self.current_task_result_path,
                                "current_task_runtime_status": self.current_task_runtime_status,
                            },
                        )
                        return result
                if result_contract is not None:
                    task_result = self._wait_for_task_result_while_agent_busy_after_timeout(
                        label=label,
                        attempt=attempt,
                        timeout_sec=timeout_sec,
                        contract=result_contract,
                        task_status_path=task_status_path,
                        result_path=result_path,
                        baseline_visible=locals().get("baseline_visible", ""),
                        baseline_raw_log_tail=locals().get("baseline_raw_log_tail", ""),
                        prompt_submission_observed=prompt_submission_observed,
                    )
                    if task_result is not None:
                        reply = json.dumps(task_result.payload, ensure_ascii=False, indent=2)
                        finished_at = _now_iso()
                        result = CommandResult(
                            label=label,
                            command=submitted_prompt,
                            exit_code=0,
                            raw_output=reply,
                            clean_output=reply,
                            started_at=started_at,
                            finished_at=finished_at,
                        )
                        self._record_result(
                            result,
                            status=WorkerStatus.SUCCEEDED,
                            note=f"done:{label}",
                            extra={
                                "label": label,
                                "result_status": "succeeded",
                                "retry_count": attempt - 1,
                                "current_task_status_path": str(task_status_path),
                                "current_task_result_path": self.current_task_result_path,
                                "current_task_runtime_status": self.current_task_runtime_status,
                            },
                        )
                        return result
                last_timeout = error
                self.agent_ready = False
                self.agent_state = AgentRuntimeState.STARTING
                self.current_task_runtime_status = read_task_status(task_status_path)
                if self.current_task_runtime_status == TASK_STATUS_RUNNING:
                    self.current_task_runtime_status = ""
                if attempt < 2:
                    self._log_event("turn_timeout_retry", label=label, attempt=attempt)
                    self._write_state(
                        WorkerStatus.RUNNING,
                        note=f"retry:{label}",
                        extra={
                            "label": label,
                            "retry_count": attempt,
                            "result_status": "running",
                            "current_task_status_path": str(task_status_path),
                            "current_task_result_path": self.current_task_result_path,
                            "current_task_runtime_status": self.current_task_runtime_status,
                        },
                    )
                    self.ensure_agent_ready(timeout_sec=timeout_sec)
                    continue
                finished_at = _now_iso()
                clean_output = str(error).strip()
                timeout_extra = {
                    "label": label,
                    "timeout_sec": timeout_sec,
                    "result_status": "failed",
                    "retry_count": attempt,
                    "current_task_status_path": str(task_status_path),
                    "current_task_result_path": self.current_task_result_path,
                    "current_task_runtime_status": self.current_task_runtime_status,
                    "dispatch_state": self.dispatch_state,
                    "dispatch_reason": self.dispatch_reason,
                }
                if is_provider_runtime_error(clean_output):
                    self.agent_ready = False
                    self.agent_state = AgentRuntimeState.STARTING
                    self.wrapper_state = WrapperState.NOT_READY
                    timeout_extra.update(
                        {
                            "health_status": "provider_runtime_error",
                            "health_note": clean_output,
                            "last_provider_error": clean_output,
                            "agent_state": AgentRuntimeState.STARTING.value,
                        }
                    )
                result = CommandResult(
                    label=label,
                    command=submitted_prompt,
                    exit_code=TIMEOUT_EXIT_CODE,
                    raw_output=clean_output,
                    clean_output=clean_output,
                    started_at=started_at,
                    finished_at=finished_at,
                )
                self._record_result(
                    result,
                    status=WorkerStatus.FAILED,
                    note=f"timeout:{label}",
                    extra=timeout_extra,
                )
                return result
            except RuntimeShutdownRequested:
                self.current_task_runtime_status = read_task_status(task_status_path)
                if self.current_task_runtime_status == TASK_STATUS_RUNNING:
                    self.current_task_runtime_status = ""
                raise
            except Exception as error:
                finished_at = _now_iso()
                current_visible = clean_ansi(self.capture_visible(200)) if self.pane_id and self.target_exists() else ""
                clean_output = "\n".join(part for part in [str(error).strip(), current_visible.strip()] if part).strip()
                runtime_state_extra = self._turn_failure_runtime_state_extra(clean_output)
                self.current_task_runtime_status = read_task_status(task_status_path)
                if self.current_task_runtime_status == TASK_STATUS_RUNNING:
                    self.current_task_runtime_status = ""
                error_extra = {
                    "label": label,
                    "result_status": "failed",
                    "retry_count": attempt - 1,
                    "current_task_status_path": str(task_status_path),
                    "current_task_result_path": self.current_task_result_path,
                    "current_task_runtime_status": self.current_task_runtime_status,
                    "dispatch_state": self.dispatch_state,
                    "dispatch_reason": self.dispatch_reason,
                }
                error_extra.update(runtime_state_extra)
                result = CommandResult(
                    label=label,
                    command=submitted_prompt,
                    exit_code=GENERIC_ERROR_EXIT_CODE,
                    raw_output=clean_output,
                    clean_output=clean_output,
                    started_at=started_at,
                    finished_at=finished_at,
                )
                self._record_result(
                    result,
                    status=WorkerStatus.FAILED,
                    note=f"error:{label}",
                    extra=error_extra,
                )
                return result

        raise AssertionError(f"unreachable turn state for {label}: {last_timeout}")

    def collect_result(self) -> WorkerResult:
        status = WorkerStatus.READY.value
        if self.results:
            status = WorkerStatus.SUCCEEDED.value if all(
                item.ok for item in self.results) else WorkerStatus.FAILED.value
        return WorkerResult(
            worker_id=self.worker_id,
            session_name=self.session_name,
            pane_id=self.pane_id,
            runtime_dir=str(self.runtime_dir),
            work_dir=str(self.work_dir),
            config=self.config.to_summary(),
            status=status,
            commands=list(self.results),
        )


def extract_final_protocol_token(text: str, allowed_tokens: Sequence[str]) -> str:
    lines = [clean_ansi(line).strip() for line in str(text or "").splitlines() if line.strip()]
    while lines and is_runtime_noise_line(lines[-1]):
        lines.pop()
    if not lines:
        return ""
    return _extract_protocol_token_from_line(lines[-1], allowed_tokens)


def build_session_name(
        worker_id: str,
        work_dir: Path,
        vendor: Vendor,
        instance_id: str = "",
        *,
        occupied_session_names: Sequence[str] | None = None,
) -> str:
    del vendor
    del instance_id
    role_key = _worker_role_key(worker_id, work_dir)
    role_label = _sanitize_session_fragment(_worker_role_label(role_key), fallback="执行者")
    occupied = {str(name).strip() for name in occupied_session_names or () if str(name).strip()}
    preferred_index = _preferred_constellation_index(work_dir, role_key)
    for offset in range(len(SESSION_CONSTELLATION_NAMES)):
        constellation = _sanitize_session_fragment(
            SESSION_CONSTELLATION_NAMES[(preferred_index + offset) % len(SESSION_CONSTELLATION_NAMES)],
            fallback="天魁星",
        )
        candidate = f"{role_label}-{constellation}"
        if candidate not in occupied:
            return candidate
    base_constellation = _sanitize_session_fragment(SESSION_CONSTELLATION_NAMES[preferred_index], fallback="天魁星")
    base_name = f"{role_label}-{base_constellation}"
    suffix = 2
    candidate = f"{base_name}-{suffix}"
    while candidate in occupied:
        suffix += 1
        candidate = f"{base_name}-{suffix}"
    return candidate
