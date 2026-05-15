# -*- encoding: utf-8 -*-
"""
@File: T09_terminal_ops.py
@Modify Time: 2026/4/16
@Author: Kevin-Chen
@Descriptions: 统一的高层终端交互抽象门面，支持 legacy CLI 与 bridge UI
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import signal
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager, Mapping, Protocol, Sequence, TextIO

from T06_terminal_progress import (
    SingleLineSpinnerMonitor as LegacySingleLineSpinnerMonitor,
    TERMINAL_SPINNER_FRAMES,
)


PROMPT_BACK_VALUE = "__tmux_back__"
_PROMPT_METADATA: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("tmux_prompt_metadata", default={})


class PromptBackRequested(Exception):
    """Raised when a bridge client requests returning to the previous setup prompt."""


_EXTERNAL_PROCESS_SIGINT_WAIT_SEC = 30.0
_EXTERNAL_PROCESS_SIGTERM_WAIT_SEC = 10.0


def _current_prompt_metadata() -> dict[str, Any]:
    return dict(_PROMPT_METADATA.get({}))


def _apply_prompt_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = _current_prompt_metadata()
    for key, value in metadata.items():
        if key not in payload:
            payload[key] = value
    return payload


def _raise_if_prompt_back(response_payload: Mapping[str, Any], request_payload: Mapping[str, Any]) -> None:
    back_value = str(request_payload.get("back_value", PROMPT_BACK_VALUE) or PROMPT_BACK_VALUE)
    if not bool(request_payload.get("allow_back", False)):
        return
    if str(response_payload.get("value", "")) == back_value:
        raise PromptBackRequested()


@contextlib.contextmanager
def prompt_metadata(**metadata: Any) -> ContextManager[None]:
    previous = _current_prompt_metadata()
    next_metadata = {**previous}
    for key, value in metadata.items():
        if value is None:
            continue
        next_metadata[str(key)] = value
    token = _PROMPT_METADATA.set(next_metadata)
    try:
        yield
    finally:
        _PROMPT_METADATA.reset(token)


class ProgressMonitor(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def _display_line(self, line: str) -> None: ...


class TerminalUI(Protocol):
    def message(self, *objects: object, sep: str = " ", end: str = "\n", flush: bool = False) -> None: ...

    def prompt_text(self, prompt_text: str, default: str = "", allow_empty: bool = False) -> str: ...

    def prompt_select(
        self,
        *,
        title: str,
        options: Sequence[tuple[str, str]],
        default_value: str,
        prompt_text: str = "请选择",
        preview_path: str | Path | None = None,
        preview_title: str = "",
        is_hitl: bool = False,
        extra_payload: Mapping[str, Any] | None = None,
    ) -> str: ...

    def prompt_multiline(
        self,
        *,
        title: str,
        empty_retry_message: str = "输入不能为空，请重试。",
        question_path: str | Path | None = None,
        answer_path: str | Path | None = None,
        is_hitl: bool = False,
    ) -> str: ...

    def clear_pending_tty_input(self) -> None: ...

    def notify_runtime_state_changed(self) -> None: ...

    def notify_stage_action_changed(self, action: str) -> None: ...

    def create_progress_monitor(
        self,
        *,
        frame_builder: Callable[[int], str],
        stream: TextIO | None = None,
        interval_sec: float = 0.2,
    ) -> ProgressMonitor: ...

    def attach_external_process(
        self,
        command: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> int: ...


class StdioTerminalUI:
    def message(self, *objects: object, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        print(*objects, sep=sep, end=end, flush=flush)

    def prompt_text(self, prompt_text: str, default: str = "", allow_empty: bool = False) -> str:
        suffix = f" [{default}]" if default else ""
        while True:
            value = input(f"{prompt_text}{suffix}: ").strip()
            if value:
                return value
            if default:
                return default
            if allow_empty:
                return ""
            self.message("输入不能为空，请重试。")

    def prompt_select(
        self,
        *,
        title: str,
        options: Sequence[tuple[str, str]],
        default_value: str,
        prompt_text: str = "请选择",
        preview_path: str | Path | None = None,
        preview_title: str = "",
        is_hitl: bool = False,
        extra_payload: Mapping[str, Any] | None = None,
    ) -> str:
        _ = preview_path
        _ = preview_title
        _ = is_hitl
        _ = extra_payload
        normalized = [(str(value), str(label)) for value, label in options]
        if not normalized:
            raise ValueError("选项不能为空")
        self.message(title)
        for index, (_, label) in enumerate(normalized, start=1):
            self.message(f"  {index}. {label}")
        default_prompt = default_value if default_value else str(1)
        while True:
            candidate = self.prompt_text(prompt_text, default_prompt, allow_empty=bool(default_prompt)).strip()
            if candidate.isdigit():
                selected_index = int(candidate)
                if 1 <= selected_index <= len(normalized):
                    return normalized[selected_index - 1][0]
            for value, label in normalized:
                if candidate in {value, label}:
                    return value
            self.message(f"不支持的选择: {candidate}")

    def prompt_multiline(
        self,
        *,
        title: str,
        empty_retry_message: str = "输入不能为空，请重试。",
        question_path: str | Path | None = None,
        answer_path: str | Path | None = None,
        is_hitl: bool = False,
    ) -> str:
        _ = question_path
        _ = answer_path
        _ = is_hitl
        while True:
            self.message(title)
            self.message("输入完成后单独输入 END 或 EOF 提交。")
            lines: list[str] = []
            while True:
                try:
                    line = input()
                except EOFError:
                    line = "EOF"
                if line in {"END", "EOF"}:
                    break
                lines.append(line)
            content = "\n".join(lines).strip()
            if content:
                return content
            self.message(empty_retry_message)

    def clear_pending_tty_input(self) -> None:
        return None

    def notify_runtime_state_changed(self) -> None:
        return None

    def notify_stage_action_changed(self, action: str) -> None:
        _ = action
        return None

    def create_progress_monitor(
        self,
        *,
        frame_builder: Callable[[int], str],
        stream: TextIO | None = None,
        interval_sec: float = 0.2,
    ) -> ProgressMonitor:
        return LegacySingleLineSpinnerMonitor(
            frame_builder=frame_builder,
            stream=stream,
            interval_sec=interval_sec,
        )

    def attach_external_process(
        self,
        command: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> int:
        merged_env = dict(os.environ)
        merged_env.update(dict(env or {}))
        child = subprocess.Popen(list(command), cwd=cwd, env=merged_env)
        try:
            return int(child.wait())
        except KeyboardInterrupt:
            try:
                child.send_signal(signal.SIGINT)
            except Exception:
                return 130
            try:
                child.wait(timeout=_EXTERNAL_PROCESS_SIGINT_WAIT_SEC)
                return 130
            except subprocess.TimeoutExpired:
                pass
            try:
                child.terminate()
                child.wait(timeout=_EXTERNAL_PROCESS_SIGTERM_WAIT_SEC)
                return 130
            except Exception:
                pass
            with contextlib.suppress(Exception):
                child.kill()
            with contextlib.suppress(Exception):
                child.wait()
            return 130


@dataclass
class BridgePromptRequest:
    prompt_type: str
    payload: dict[str, Any]


class BridgeProgressMonitor:
    def __init__(
        self,
        *,
        monitor_id: str,
        emit_event: Callable[[str, dict[str, Any]], None],
        frame_builder: Callable[[int], str],
        interval_sec: float,
        action: str = "",
        stage_seq: int = 0,
    ) -> None:
        self.monitor_id = monitor_id
        self.emit_event = emit_event
        self.frame_builder = frame_builder
        self.interval_sec = interval_sec
        self.action = str(action or "").strip()
        self.stage_seq = max(int(stage_seq or 0), 0)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._tick = 0
        self._started = False
        self._last_frame = ""

    def _payload(self, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.monitor_id,
            "action": self.action,
            "stage_seq": self.stage_seq,
        }
        if extra:
            payload.update(dict(extra))
        return payload

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.emit_event("progress.start", self._payload())
        self._display_line(self.frame_builder(self._tick))
        self._tick += 1
        self._thread = threading.Thread(target=self._run_loop, name=f"bridge-progress-{self.monitor_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.emit_event("progress.stop", self._payload())
        self._thread = None
        self._stop_event = threading.Event()
        self._started = False

    def _display_line(self, line: str) -> None:
        self.emit_event("progress.update", self._payload({"line": line}))
        self._last_frame = line

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            frame = self.frame_builder(self._tick)
            self._tick += 1
            if frame != self._last_frame:
                self._display_line(frame)
            if self._stop_event.wait(self.interval_sec):
                break


class BridgeTerminalUI:
    def __init__(
        self,
        *,
        emit_event: Callable[[str, dict[str, Any]], None],
        request_prompt: Callable[[BridgePromptRequest], dict[str, Any]],
        external_process_runner: Callable[[Sequence[str], str | None, Mapping[str, str] | None], int] | None = None,
        state_change_notifier: Callable[[], None] | None = None,
        stage_change_notifier: Callable[[str], None] | None = None,
        progress_context_provider: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self._emit_event = emit_event
        self._request_prompt = request_prompt
        self._external_process_runner = external_process_runner
        self._state_change_notifier = state_change_notifier
        self._stage_change_notifier = stage_change_notifier
        self._progress_context_provider = progress_context_provider

    def message(self, *objects: object, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        text = sep.join(str(item) for item in objects) + end
        self._emit_event("log.append", {"text": text, "flush": flush})

    def prompt_text(self, prompt_text: str, default: str = "", allow_empty: bool = False) -> str:
        while True:
            request_payload = _apply_prompt_metadata(
                {
                    "prompt_text": prompt_text,
                    "default": default,
                    "allow_empty": allow_empty,
                }
            )
            payload = self._request_prompt(
                BridgePromptRequest(
                    prompt_type="text",
                    payload=request_payload,
                )
            )
            _raise_if_prompt_back(payload, request_payload)
            value = str(payload.get("value", ""))
            if allow_empty or value.strip():
                return value
            self.message("输入不能为空，请重试。")

    def prompt_select(
        self,
        *,
        title: str,
        options: Sequence[tuple[str, str]],
        default_value: str,
        prompt_text: str = "请选择",
        preview_path: str | Path | None = None,
        preview_title: str = "",
        is_hitl: bool = False,
        extra_payload: Mapping[str, Any] | None = None,
    ) -> str:
        request_payload: dict[str, Any] = _apply_prompt_metadata(
            {
                "title": title,
                "options": [{"value": value, "label": label} for value, label in options],
                "default_value": default_value,
                "prompt_text": prompt_text,
                "preview_path": str(Path(preview_path).expanduser().resolve()) if preview_path else "",
                "preview_title": str(preview_title or "").strip(),
            }
        )
        if extra_payload:
            protected_keys = set(request_payload)
            for key, value in extra_payload.items():
                key_text = str(key)
                if key_text not in protected_keys:
                    request_payload[key_text] = value
        if is_hitl:
            request_payload["is_hitl"] = True
        payload = self._request_prompt(
            BridgePromptRequest(
                prompt_type="select",
                payload=request_payload,
            )
        )
        _raise_if_prompt_back(payload, request_payload)
        value = str(payload.get("value", default_value))
        allowed = {item[0] for item in options}
        if value not in allowed:
            raise ValueError(f"不支持的选择: {value}")
        return value

    def prompt_multiline(
        self,
        *,
        title: str,
        empty_retry_message: str = "输入不能为空，请重试。",
        question_path: str | Path | None = None,
        answer_path: str | Path | None = None,
        is_hitl: bool = False,
    ) -> str:
        while True:
            request_payload = _apply_prompt_metadata(
                {
                    "title": title,
                    "empty_retry_message": empty_retry_message,
                    "question_path": str(Path(question_path).expanduser().resolve()) if question_path else "",
                    "answer_path": str(Path(answer_path).expanduser().resolve()) if answer_path else "",
                    "is_hitl": bool(is_hitl),
                }
            )
            payload = self._request_prompt(
                BridgePromptRequest(
                    prompt_type="multiline",
                    payload=request_payload,
                )
            )
            _raise_if_prompt_back(payload, request_payload)
            value = str(payload.get("value", ""))
            if value.strip():
                return value
            self.message(empty_retry_message)

    def clear_pending_tty_input(self) -> None:
        return None

    def notify_runtime_state_changed(self) -> None:
        if self._state_change_notifier is None:
            return None
        self._state_change_notifier()

    def notify_stage_action_changed(self, action: str) -> None:
        if self._stage_change_notifier is None:
            return None
        self._stage_change_notifier(str(action).strip())

    def create_progress_monitor(
        self,
        *,
        frame_builder: Callable[[int], str],
        stream: TextIO | None = None,
        interval_sec: float = 0.2,
    ) -> ProgressMonitor:
        context = dict(self._progress_context_provider() or {}) if self._progress_context_provider is not None else {}
        return BridgeProgressMonitor(
            monitor_id=uuid.uuid4().hex,
            emit_event=self._emit_event,
            frame_builder=frame_builder,
            interval_sec=interval_sec,
            action=str(context.get("action", "") or "").strip(),
            stage_seq=int(context.get("stage_seq", 0) or 0),
        )

    def attach_external_process(
        self,
        command: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> int:
        if self._external_process_runner is None:
            raise RuntimeError("BridgeTerminalUI 未配置 attach_external_process")
        return int(self._external_process_runner(command, cwd, env))


class SingleLineSpinnerMonitor:
    def __init__(
        self,
        *,
        frame_builder: Callable[[int], str],
        stream: TextIO | None = None,
        interval_sec: float = 0.2,
    ) -> None:
        self._monitor = get_terminal_ui().create_progress_monitor(
            frame_builder=frame_builder,
            stream=stream,
            interval_sec=interval_sec,
        )
        self._last_frame = ""
        self._last_line_width = 0

    def start(self) -> None:
        self._monitor.start()
        self._sync_compat_fields()

    def stop(self) -> None:
        self._monitor.stop()
        self._sync_compat_fields()

    def _display_line(self, line: str) -> None:
        self._monitor._display_line(line)
        self._last_frame = line
        self._last_line_width = len(line)
        self._sync_compat_fields()

    def _sync_compat_fields(self) -> None:
        self._last_frame = getattr(self._monitor, "_last_frame", self._last_frame)
        self._last_line_width = getattr(self._monitor, "_last_line_width", self._last_line_width)


_DEFAULT_UI = StdioTerminalUI()
_CURRENT_UI: TerminalUI = _DEFAULT_UI
_UI_LOCK = threading.Lock()


def set_terminal_ui(ui: TerminalUI) -> None:
    global _CURRENT_UI
    with _UI_LOCK:
        _CURRENT_UI = ui


def get_terminal_ui() -> TerminalUI:
    with _UI_LOCK:
        return _CURRENT_UI


def terminal_ui_is_interactive() -> bool:
    ui = get_terminal_ui()
    if isinstance(ui, BridgeTerminalUI):
        return True
    stdin = sys.stdin
    return bool(getattr(stdin, "isatty", lambda: False)())


@contextlib.contextmanager
def use_terminal_ui(ui: TerminalUI) -> ContextManager[None]:
    previous = get_terminal_ui()
    set_terminal_ui(ui)
    try:
        yield
    finally:
        set_terminal_ui(previous)


def message(*objects: object, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
    get_terminal_ui().message(*objects, sep=sep, end=end, flush=flush)


def prompt_with_default(prompt_text: str, default: str = "", allow_empty: bool = False) -> str:
    return get_terminal_ui().prompt_text(prompt_text, default, allow_empty)


def prompt_select_option(
    *,
    title: str,
    options: Sequence[tuple[str, str]],
    default_value: str,
    prompt_text: str = "请选择",
    preview_path: str | Path | None = None,
    preview_title: str = "",
    is_hitl: bool = False,
    extra_payload: Mapping[str, Any] | None = None,
) -> str:
    terminal_ui = get_terminal_ui()
    try:
        return terminal_ui.prompt_select(
            title=title,
            options=options,
            default_value=default_value,
            prompt_text=prompt_text,
            preview_path=preview_path,
            preview_title=preview_title,
            is_hitl=is_hitl,
            extra_payload=extra_payload,
        )
    except TypeError:
        if is_hitl or extra_payload:
            raise
        return terminal_ui.prompt_select(
            title=title,
            options=options,
            default_value=default_value,
            prompt_text=prompt_text,
            preview_path=preview_path,
            preview_title=preview_title,
        )


def prompt_yes_no(
    prompt_text: str,
    default: bool = False,
    *,
    preview_path: str | Path | None = None,
    preview_title: str = "",
) -> bool:
    value = prompt_select_option(
        title=f"{prompt_text} (yes/no)",
        options=(("yes", "yes"), ("no", "no")),
        default_value="yes" if default else "no",
        prompt_text=prompt_text,
        preview_path=preview_path,
        preview_title=preview_title,
    )
    return value == "yes"


def prompt_positive_int(prompt_text: str, default: int = 1) -> int:
    while True:
        value = prompt_with_default(prompt_text, str(default))
        if value.isdigit() and int(value) >= 1:
            return int(value)
        message("请输入大于等于 1 的整数。")


def prompt_command_line(prompt_text: str = "输入命令", default: str = "") -> str:
    return prompt_with_default(prompt_text, default, allow_empty=True)


def collect_multiline_input(
    *,
    title: str,
    empty_retry_message: str = "输入不能为空，请重试。",
    question_path: str | Path | None = None,
    answer_path: str | Path | None = None,
    is_hitl: bool = False,
) -> str:
    return get_terminal_ui().prompt_multiline(
        title=title,
        empty_retry_message=empty_retry_message,
        question_path=question_path,
        answer_path=answer_path,
        is_hitl=is_hitl,
    )


def clear_pending_tty_input() -> None:
    get_terminal_ui().clear_pending_tty_input()


def notify_runtime_state_changed() -> None:
    get_terminal_ui().notify_runtime_state_changed()


def notify_stage_action_changed(action: str) -> None:
    get_terminal_ui().notify_stage_action_changed(action)


def attach_external_process(
    command: Sequence[str],
    *,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    return get_terminal_ui().attach_external_process(command, cwd=cwd, env=env)


def split_legacy_cli_flag(argv: Sequence[str] | None) -> tuple[bool, list[str]]:
    args = list(sys.argv[1:] if argv is None else argv)
    filtered: list[str] = []
    legacy_cli = False
    for item in args:
        if item in {"--legacy-cli", "--no-tui"}:
            legacy_cli = True
            continue
        filtered.append(str(item))
    return legacy_cli, filtered


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def tui_package_dir() -> Path:
    return repo_root() / "packages" / "tui"


def _missing_tui_dependency_names() -> list[str]:
    package_dir = tui_package_dir()
    expected = {
        "solid-js": package_dir / "node_modules" / "solid-js" / "package.json",
        "@opentui/solid": package_dir / "node_modules" / "@opentui" / "solid" / "package.json",
    }
    return [name for name, package_json in expected.items() if not package_json.exists()]


def ensure_tui_dependencies_installed() -> None:
    package_dir = tui_package_dir()
    package_json = package_dir / "package.json"
    if not package_json.exists():
        raise RuntimeError(f"缺少 OpenTUI package.json: {package_json}")
    missing = _missing_tui_dependency_names()
    if not missing:
        return
    message(f"检测到 OpenTUI 依赖缺失，正在执行 bun install --frozen-lockfile: {', '.join(missing)}")
    try:
        completed = subprocess.run(
            ["bun", "install", "--frozen-lockfile"],
            cwd=str(package_dir),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("OpenTUI 依赖缺失且未找到 bun；请先安装 Bun 或使用 --no-tui") from error
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"OpenTUI 依赖安装失败: {details or f'退出码 {completed.returncode}'}")
    remaining = _missing_tui_dependency_names()
    if remaining:
        raise RuntimeError(f"OpenTUI 依赖安装后仍缺少: {', '.join(remaining)}")


def maybe_launch_tui(
    argv: Sequence[str] | None,
    *,
    route: str,
    action: str,
) -> tuple[bool, int | list[str]]:
    no_tui, filtered = split_legacy_cli_flag(argv)
    if any(item in {"-h", "--help"} for item in filtered):
        return False, filtered
    if not no_tui and sys.stdin.isatty() and sys.stdout.isatty():
        ensure_tui_dependencies_installed()
        command = [
            str(repo_root() / "scripts" / "tmux-tui"),
            "--route",
            route,
            "--action",
            action,
        ]
        if filtered:
            command.extend(["--argv-json", json.dumps(filtered, ensure_ascii=False)])
        return True, attach_external_process(command, cwd=str(repo_root()))
    return False, filtered


__all__ = [
    "BridgePromptRequest",
    "BridgeTerminalUI",
    "PROMPT_BACK_VALUE",
    "ProgressMonitor",
    "PromptBackRequested",
    "SingleLineSpinnerMonitor",
    "StdioTerminalUI",
    "TERMINAL_SPINNER_FRAMES",
    "TerminalUI",
    "attach_external_process",
    "clear_pending_tty_input",
    "collect_multiline_input",
    "get_terminal_ui",
    "message",
    "maybe_launch_tui",
    "notify_runtime_state_changed",
    "notify_stage_action_changed",
    "repo_root",
    "tui_package_dir",
    "ensure_tui_dependencies_installed",
    "prompt_command_line",
    "prompt_metadata",
    "prompt_positive_int",
    "prompt_select_option",
    "prompt_with_default",
    "prompt_yes_no",
    "set_terminal_ui",
    "split_legacy_cli_flag",
    "terminal_ui_is_interactive",
    "use_terminal_ui",
]
