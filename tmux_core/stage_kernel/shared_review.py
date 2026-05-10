from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tmux_core.runtime.vendor_catalog import get_default_model_for_vendor
from A01_Routing_LayerPlanning import (
    DEFAULT_MODEL_BY_VENDOR,
    normalize_effort_choice,
    normalize_model_choice,
    normalize_vendor_choice,
    prompt_effort,
    prompt_model,
    prompt_vendor,
)
from tmux_core.runtime.contracts import TurnFileContract
from tmux_core.runtime.tmux_runtime import (
    AgentRunConfig,
    TmuxBatchWorker,
    Vendor,
    is_agent_ready_timeout_error,
    is_provider_auth_error,
    is_provider_runtime_error,
    is_worker_death_error,
)
from tmux_core.stage_kernel.agent_intervention import (
    AGENT_INTERVENTION_RECHECK,
    AGENT_INTERVENTION_WORKER_DEAD,
    request_worker_manual_intervention,
)
from T09_terminal_ops import (
    PROMPT_BACK_VALUE,
    PromptBackRequested,
    SingleLineSpinnerMonitor,
    TERMINAL_SPINNER_FRAMES,
    collect_multiline_input,
    message,
    prompt_metadata,
    prompt_select_option,
    prompt_positive_int as terminal_prompt_positive_int,
    prompt_with_default,
    prompt_yes_no as terminal_prompt_yes_no,
)
from T12_requirements_common import (
    DEFAULT_REQUIREMENTS_CLARIFICATION_EFFORT,
    DEFAULT_REQUIREMENTS_CLARIFICATION_VENDOR,
    DEFAULT_REQUIREMENTS_CLARIFICATION_MODEL,
    stdin_is_interactive,
)

DEFAULT_REVIEWER_COUNT = 1
MAX_REVIEWER_REPAIR_ATTEMPTS = 2
DEFAULT_STAGE_REVIEW_MAX_ROUNDS = 5
REVIEWER_CONSECUTIVE_FAILURE_RECONFIG_THRESHOLD = 2
AGENT_READY_TIMEOUT_RETRY = AGENT_INTERVENTION_RECHECK
AGENT_READY_TIMEOUT_SKIP = AGENT_INTERVENTION_WORKER_DEAD


@dataclass(frozen=True)
class ReviewAgentSelection:
    vendor: str
    model: str
    reasoning_effort: str
    proxy_url: str


@dataclass(frozen=True)
class InvalidAgentSelection:
    name: str
    source: str
    raw_spec: object
    error: str


@dataclass(frozen=True)
class ReviewerRuntime:
    reviewer_name: str
    selection: ReviewAgentSelection
    worker: TmuxBatchWorker
    review_md_path: Path
    review_json_path: Path
    contract: TurnFileContract
    failure_streak: int = 0
    last_failure_reason: str = ""


@dataclass(frozen=True)
class ReviewAgentHandoff:
    reviewer_key: str
    role_name: str
    role_prompt: str
    selection: ReviewAgentSelection
    worker: TmuxBatchWorker


@dataclass(frozen=True)
class StageAgentConfig:
    main: ReviewAgentSelection | None = None
    reviewers: dict[str, ReviewAgentSelection] | None = None
    reviewer_order: tuple[str, ...] = ()
    invalid_main: InvalidAgentSelection | None = None
    invalid_reviewers: dict[str, InvalidAgentSelection] = field(default_factory=dict)

    def reviewer_selection(self, reviewer_key: str) -> ReviewAgentSelection | None:
        selections = self.reviewers or {}
        return selections.get(str(reviewer_key or "").strip())


@dataclass
class ReviewRoundPolicy:
    max_rounds: int | None
    quota_count: int = 0
    initial_review_done: bool = False

    def record_review_attempt(self) -> None:
        self.quota_count += 1
        self.initial_review_done = True

    def should_escalate_before_next_review(self) -> bool:
        return self.max_rounds is not None and self.quota_count >= self.max_rounds

    def reset_after_hitl(self) -> None:
        self.quota_count = 0


@dataclass(frozen=True)
class ReviewLimitHitlConfig:
    stage_label: str
    artifact_label: str
    primary_output_path: str | Path
    ask_human_path: str | Path
    hitl_record_path: str | Path
    merged_review_path: str | Path
    output_summary_path: str | Path
    continue_output_label: str


@dataclass(frozen=True)
class ReviewLimitHitlResult:
    owner: object
    rounds_used: int
    post_hitl_continue_completed: bool = False


def _review_worker_state(worker: TmuxBatchWorker | None) -> dict[str, object]:
    if worker is None:
        return {}
    reader = getattr(worker, "read_state", None)
    if not callable(reader):
        return {}
    try:
        state = reader()
    except Exception:
        return {}
    return dict(state) if isinstance(state, Mapping) else {}


AGENT_CONFIG_ERROR_MARKERS = (
    "不支持的推理强度",
    "不支持的厂商",
    "不支持的模型",
    "模型不能为空",
    "model 不能为空",
    "未安装",
    "没有可用模型",
    "无法选择模型",
    "unsupported vendor",
    "unsupported reasoning effort",
    "does not support normalized effort",
    "model unavailable",
    "model cannot be empty",
    "model is required",
    "not installed",
    "no available model",
    "scanned catalog",
    "vendor catalog",
)


def is_agent_config_error(error: Exception | BaseException | str) -> bool:
    text = str(error or "").strip()
    if not text:
        return False
    lowered = text.lower()
    return any(marker in text or marker in lowered for marker in AGENT_CONFIG_ERROR_MARKERS)


def describe_reviewer_failure_reason(
    error: Exception | BaseException | str,
    worker: TmuxBatchWorker | None = None,
) -> str:
    message_text = str(error or "").strip()
    lowered = message_text.lower()
    state = _review_worker_state(worker)
    health_status = str(state.get("health_status", "")).strip().lower()
    if is_agent_config_error(error):
        first_line = message_text.splitlines()[0].strip() if message_text else ""
        return first_line[:160] if first_line else "智能体模型配置不可用"
    if is_provider_auth_error(error) or worker_has_provider_auth_error(worker):
        return "模型认证已失效"
    if is_provider_runtime_error(error) or worker_has_provider_runtime_error(worker):
        return "模型服务出现临时运行错误"
    if is_agent_ready_timeout_error(error) or "shell initialization timed out" in lowered:
        return "启动超时，未能进入可输入状态"
    if "agent exited back to shell" in lowered:
        return "agent 进程已退出并返回 shell"
    if is_worker_death_error(error):
        if health_status == "missing_session":
            return "tmux 会话已丢失"
        if health_status == "pane_dead":
            return "tmux pane 已退出"
        if not getattr(worker, "session_exists", lambda: True)():
            return "tmux 会话已丢失"
        return "智能体进程已死亡或退出"
    first_line = message_text.splitlines()[0].strip() if message_text else ""
    return first_line[:160] if first_line else "未知原因"


def note_reviewer_failure(
    reviewer: ReviewerRuntime,
    *,
    reason_text: str,
) -> ReviewerRuntime:
    next_streak = max(int(getattr(reviewer, "failure_streak", 0) or 0), 0) + 1
    return replace(
        reviewer,
        failure_streak=next_streak,
        last_failure_reason=str(reason_text or "").strip(),
    )


def carry_reviewer_failure_state(
    reviewer: ReviewerRuntime,
    *,
    previous: ReviewerRuntime,
) -> ReviewerRuntime:
    return replace(
        reviewer,
        failure_streak=max(int(getattr(previous, "failure_streak", 0) or 0), 0),
        last_failure_reason=str(getattr(previous, "last_failure_reason", "") or "").strip(),
    )


def reviewer_requires_manual_model_reconfiguration(reviewer: ReviewerRuntime) -> bool:
    return int(getattr(reviewer, "failure_streak", 0) or 0) >= REVIEWER_CONSECUTIVE_FAILURE_RECONFIG_THRESHOLD


def build_reviewer_failure_reconfiguration_reason(
    reviewer: ReviewerRuntime,
    *,
    role_label: str,
    failure_reason: str,
) -> str:
    streak = max(int(getattr(reviewer, "failure_streak", 0) or 0), 0)
    return (
        f"检测到{role_label}连续 {streak} 次死亡/失败。\n"
        f"最近一次原因：{str(failure_reason or '').strip() or '未知原因'}\n"
        "需要重新选择模型后继续当前阶段。"
    )


class ReviewStageProgress:
    def __init__(self, *, initial_phase: str = "评审准备中") -> None:
        self._phase = initial_phase
        self._active = False
        self._monitor = SingleLineSpinnerMonitor(
            frame_builder=self._render_line,
            interval_sec=0.2,
        )

    def _render_line(self, tick: int) -> str:
        spinner = TERMINAL_SPINNER_FRAMES[tick % len(TERMINAL_SPINNER_FRAMES)]
        return f"{spinner} {self._phase}"

    def set_phase(self, phase: str, *, start: bool = True) -> None:
        self._phase = str(phase).strip() or "评审中"
        if start:
            self.start()

    def start(self) -> None:
        if self._active:
            return
        self._monitor.start()
        self._active = True

    def stop(self) -> None:
        if not self._active:
            return
        self._monitor.stop()
        self._active = False

    def suspended(self):
        was_active = self._active
        self.stop()
        if not was_active:
            return nullcontext()

        progress = self

        class _ResumeContext:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                progress.start()
                return False

        return _ResumeContext()


_ACTIVE_REVIEW_PROGRESS: ReviewStageProgress | None = None


def resolve_review_progress(progress: ReviewStageProgress | None = None) -> ReviewStageProgress | None:
    return progress if progress is not None else _ACTIVE_REVIEW_PROGRESS


def prompt_proxy_url(default: str = "", *, role_label: str = "") -> str:
    role_text = str(role_label or "").strip()
    prompt_text = "输入代理端口或完整代理 URL（可留空）"
    if role_text:
        prompt_text = f"为 {role_text} {prompt_text}"
    return prompt_with_default(prompt_text, default, allow_empty=True)


def prompt_positive_int(
    prompt_text: str,
    default: int = 1,
    *,
    progress: ReviewStageProgress | None = None,
    allow_back: bool = False,
    stage_key: str = "",
    stage_step_index: int = 0,
) -> int:
    progress = resolve_review_progress(progress)
    with progress.suspended() if progress is not None else nullcontext():
        with prompt_metadata(
            allow_back=allow_back,
            back_value=PROMPT_BACK_VALUE,
            stage_key=stage_key,
            stage_step_index=stage_step_index,
        ):
            return terminal_prompt_positive_int(prompt_text, default)


def _parse_spec_text(value: str, *, source: str) -> dict[str, str]:
    text = str(value or "").strip()
    if not text:
        return {}
    fields: dict[str, str] = {}
    if "=" not in text and ":" in text:
        parts = [part.strip() for part in text.split(":")]
        names = ("vendor", "model", "effort", "proxy")
        for index, part in enumerate(parts[: len(names)]):
            if part:
                fields[names[index]] = part
        return fields
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise RuntimeError(f"{source} 配置项必须是 key=value: {item}")
        key, raw = item.split("=", 1)
        key_text = key.strip().replace("-", "_").lower()
        if not key_text:
            raise RuntimeError(f"{source} 存在空 key: {item}")
        fields[key_text] = raw.strip()
    return fields


def _coerce_agent_spec_fields(spec: object, *, source: str) -> dict[str, str]:
    if spec is None:
        return {}
    if isinstance(spec, str):
        return _parse_spec_text(spec, source=source)
    if isinstance(spec, Mapping):
        return {
            str(key).strip().replace("-", "_").lower(): str(value).strip()
            for key, value in spec.items()
            if str(key).strip() and value is not None and str(value).strip()
        }
    raise RuntimeError(f"{source} 必须是字符串或对象")


def parse_agent_selection_spec(
    spec: object,
    *,
    default_name: str = "",
    default_vendor: str = DEFAULT_REQUIREMENTS_CLARIFICATION_VENDOR,
    default_model: str = "",
    default_reasoning_effort: str = DEFAULT_REQUIREMENTS_CLARIFICATION_EFFORT,
    source: str = "agent",
) -> tuple[str, ReviewAgentSelection]:
    fields = _coerce_agent_spec_fields(spec, source=source)
    raw_vendor = fields.get("vendor") or default_vendor
    vendor = normalize_vendor_choice(raw_vendor)
    model_default = default_model if default_model and vendor == default_vendor else get_default_model_for_vendor(vendor)
    model = normalize_model_choice(vendor, fields.get("model") or model_default)
    effort = normalize_effort_choice(
        vendor,
        model,
        fields.get("effort") or fields.get("reasoning_effort") or default_reasoning_effort,
    )
    proxy_url = (
        fields.get("proxy_url")
        or fields.get("proxy")
        or fields.get("proxy_port")
        or fields.get("port")
        or ""
    )
    name = (
        fields.get("name")
        or fields.get("key")
        or fields.get("role")
        or fields.get("reviewer")
        or default_name
    )
    return (
        str(name or "").strip(),
        ReviewAgentSelection(
            vendor=vendor,
            model=model,
            reasoning_effort=effort,
            proxy_url=str(proxy_url or "").strip(),
        ),
    )


def _agent_spec_name(spec: object, *, default_name: str, source: str) -> str:
    try:
        fields = _coerce_agent_spec_fields(spec, source=source)
    except Exception:
        return str(default_name or "").strip()
    return str(
        fields.get("name")
        or fields.get("key")
        or fields.get("role")
        or fields.get("reviewer")
        or default_name
        or ""
    ).strip()


def _load_agent_config_payload(path_value: object) -> dict[str, Any]:
    text = str(path_value or "").strip()
    if not text:
        return {}
    path = Path(text).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"--agent-config 读取失败: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"--agent-config 根节点必须是 JSON 对象: {path}")
    return payload


def _config_reviewer_specs(payload: Mapping[str, Any]) -> list[object]:
    for key in ("reviewers", "reviewer_agents", "reviewer_agent"):
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            return list(value)
        return [value]
    return []


def _stage_config_payload(payload: Mapping[str, Any], stage_key: str) -> Mapping[str, Any]:
    key = str(stage_key or "").strip()
    if not key:
        return {}
    stages = payload.get("stages")
    if not isinstance(stages, Mapping):
        return {}
    stage_payload = stages.get(key)
    if not isinstance(stage_payload, Mapping):
        return {}
    return stage_payload


def resolve_stage_agent_config(
    args: object,
    *,
    stage_key: str = "",
    default_reviewer_names: Sequence[str] = (),
) -> StageAgentConfig:
    config_payload = _load_agent_config_payload(getattr(args, "agent_config", ""))
    stage_payload = _stage_config_payload(config_payload, stage_key)
    main_spec = stage_payload.get("main") or stage_payload.get("main_agent") or config_payload.get("main") or config_payload.get("main_agent")
    cli_main = getattr(args, "main_agent", "")
    if str(cli_main or "").strip():
        main_spec = cli_main
    main_selection: ReviewAgentSelection | None = None
    invalid_main: InvalidAgentSelection | None = None
    if main_spec:
        try:
            _, main_selection = parse_agent_selection_spec(main_spec, source="main-agent")
        except Exception as error:  # noqa: BLE001
            invalid_main = InvalidAgentSelection(
                name=_agent_spec_name(main_spec, default_name="main", source="main-agent"),
                source="main-agent",
                raw_spec=main_spec,
                error=str(error),
            )

    reviewer_specs: list[object] = _config_reviewer_specs(stage_payload) if stage_payload else []
    if not reviewer_specs:
        reviewer_specs = _config_reviewer_specs(config_payload)
    cli_reviewers = list(getattr(args, "reviewer_agent", []) or [])
    if cli_reviewers:
        reviewer_specs = cli_reviewers

    reviewers: dict[str, ReviewAgentSelection] = {}
    invalid_reviewers: dict[str, InvalidAgentSelection] = {}
    reviewer_order: list[str] = []
    default_names = [str(item).strip() for item in default_reviewer_names if str(item).strip()]
    for index, reviewer_spec in enumerate(reviewer_specs):
        default_name = default_names[index] if index < len(default_names) else f"R{index + 1}"
        source = f"reviewer-agent[{index + 1}]"
        reviewer_name = _agent_spec_name(reviewer_spec, default_name=default_name, source=source)
        if not reviewer_name:
            reviewer_name = default_name
        reviewer_order.append(reviewer_name)
        try:
            _, selection = parse_agent_selection_spec(
                reviewer_spec,
                default_name=default_name,
                source=source,
            )
        except Exception as error:  # noqa: BLE001
            invalid_reviewers[reviewer_name] = InvalidAgentSelection(
                name=reviewer_name,
                source=source,
                raw_spec=reviewer_spec,
                error=str(error),
            )
            continue
        reviewers[reviewer_name] = selection
    return StageAgentConfig(
        main=main_selection,
        reviewers=reviewers,
        reviewer_order=tuple(reviewer_order),
        invalid_main=invalid_main,
        invalid_reviewers=invalid_reviewers,
    )


def prompt_review_agent_selection(
    default_vendor: str = DEFAULT_REQUIREMENTS_CLARIFICATION_VENDOR,
    default_model: str = "",
    default_reasoning_effort: str = DEFAULT_REQUIREMENTS_CLARIFICATION_EFFORT,
    default_proxy_url: str = "",
    *,
    role_label: str = "",
    progress: ReviewStageProgress | None = None,
    allow_back_first_step: bool = False,
    stage_key: str = "agent_selection",
) -> ReviewAgentSelection:
    progress = resolve_review_progress(progress)
    while True:
        try:
            vendor = default_vendor
            model = default_model
            reasoning_effort = default_reasoning_effort
            proxy_url = default_proxy_url
            step = 0
            while step < 4:
                try:
                    with progress.suspended() if progress is not None else nullcontext():
                        with prompt_metadata(
                            allow_back=allow_back_first_step if step == 0 else True,
                            back_value=PROMPT_BACK_VALUE,
                            stage_key=stage_key,
                            stage_step_index=step,
                        ):
                            if step == 0:
                                vendor = prompt_vendor(vendor or default_vendor, role_label=role_label)
                                if model:
                                    try:
                                        normalize_model_choice(vendor, model)
                                    except ValueError:
                                        model = ""
                                step = 1
                                continue
                            if step == 1:
                                preferred_model = model if model and vendor == default_vendor else get_default_model_for_vendor(vendor)
                                model = prompt_model(vendor, preferred_model, role_label=role_label)
                                step = 2
                                continue
                            if step == 2:
                                reasoning_effort = prompt_effort(vendor, model, reasoning_effort or default_reasoning_effort, role_label=role_label)
                                step = 3
                                continue
                            if step == 3:
                                proxy_url = prompt_proxy_url(proxy_url, role_label=role_label)
                                step = 4
                                continue
                except PromptBackRequested:
                    if step == 0:
                        continue
                    step -= 1
            return ReviewAgentSelection(
                vendor=vendor,
                model=model,
                reasoning_effort=reasoning_effort,
                proxy_url=proxy_url,
            )
        except Exception as error:  # noqa: BLE001
            if not is_agent_config_error(error):
                raise
            if not stdin_is_interactive():
                raise
            role_text = str(role_label or "").strip() or "智能体"
            message(f"{role_text} 模型配置不可用: {error}\n请重新选择厂商、模型和推理强度。")


def resolve_agent_run_config_with_recovery(
    selection: ReviewAgentSelection,
    *,
    role_label: str,
    progress: ReviewStageProgress | None = None,
    reason_text: str = "",
) -> tuple[ReviewAgentSelection, AgentRunConfig]:
    progress = resolve_review_progress(progress)
    current_selection = selection
    role_text = str(role_label or "").strip() or "智能体"
    while True:
        try:
            config = AgentRunConfig(
                vendor=current_selection.vendor,
                model=current_selection.model,
                reasoning_effort=current_selection.reasoning_effort,
                proxy_url=current_selection.proxy_url,
            )
            return current_selection, config
        except Exception as error:  # noqa: BLE001
            if not is_agent_config_error(error):
                raise
            prompt_is_patched = getattr(prompt_review_agent_selection, "__module__", __name__) != __name__
            if not stdin_is_interactive() and not prompt_is_patched:
                raise RuntimeError(f"{role_text} 模型配置不可用: {error}；当前环境无法交互重新选择模型。") from error
            message(
                str(reason_text or "").strip()
                or f"{role_text} 模型配置不可用: {error}\n请重新选择模型配置后继续当前阶段。"
            )
            current_selection = prompt_review_agent_selection(
                default_vendor=current_selection.vendor,
                default_model=current_selection.model,
                default_reasoning_effort=current_selection.reasoning_effort,
                default_proxy_url=current_selection.proxy_url,
                role_label=role_text,
                progress=progress,
            )
            message(render_review_agent_selection(f"{role_text} 新配置", current_selection))


def render_review_agent_selection(title: str, selection: ReviewAgentSelection) -> str:
    return "\n".join(
        [
            title,
            f"vendor: {selection.vendor}",
            f"model: {selection.model}",
            f"reasoning_effort: {selection.reasoning_effort}",
            f"proxy_url: {selection.proxy_url or '(none)'}",
        ]
    )


def collect_reviewer_agent_selections(
    *,
    project_dir: str | Path,
    reviewer_specs: Sequence[object],
    display_name_resolver: Callable[[str | Path, object, Sequence[str]], str],
    progress: ReviewStageProgress | None = None,
    skip_reviewer_keys: Sequence[str] = (),
    reserved_session_names: Sequence[str] = (),
    allow_back_first_prompt: bool = False,
    stage_key: str = "reviewer_selection",
) -> dict[str, ReviewAgentSelection]:
    selections: dict[str, ReviewAgentSelection] = {}
    predicted_session_names: set[str] = {str(name).strip() for name in reserved_session_names if str(name).strip()}
    skip_keys = {str(item).strip() for item in skip_reviewer_keys if str(item).strip()}
    interactive = stdin_is_interactive()
    next_allow_back = bool(allow_back_first_prompt)
    for reviewer_spec in reviewer_specs:
        reviewer_key = str(
            getattr(reviewer_spec, "reviewer_key", "") or getattr(reviewer_spec, "role_name", "")
        ).strip()
        if not reviewer_key or reviewer_key in skip_keys:
            continue
        reviewer_display_name = display_name_resolver(project_dir, reviewer_spec, sorted(predicted_session_names))
        predicted_session_names.add(reviewer_display_name)
        if interactive:
            selection = prompt_review_agent_selection(
                DEFAULT_REQUIREMENTS_CLARIFICATION_VENDOR,
                default_model=DEFAULT_REQUIREMENTS_CLARIFICATION_MODEL,
                default_reasoning_effort=DEFAULT_REQUIREMENTS_CLARIFICATION_EFFORT,
                default_proxy_url="",
                role_label=reviewer_display_name,
                progress=progress,
                allow_back_first_step=next_allow_back,
                stage_key=stage_key,
            )
            next_allow_back = False
            message(render_review_agent_selection(f"{reviewer_display_name} 配置", selection))
        else:
            selection = ReviewAgentSelection(
                vendor=DEFAULT_REQUIREMENTS_CLARIFICATION_VENDOR,
                model=DEFAULT_REQUIREMENTS_CLARIFICATION_MODEL,
                reasoning_effort=DEFAULT_REQUIREMENTS_CLARIFICATION_EFFORT,
                proxy_url="",
            )
        selections[reviewer_key] = selection
    return selections


def prompt_yes_no_choice(
    prompt_text: str,
    default: bool = False,
    *,
    progress: ReviewStageProgress | None = None,
    preview_path: str | Path | None = None,
    preview_title: str = "",
    allow_back: bool = False,
    stage_key: str = "",
    stage_step_index: int = 0,
) -> bool:
    progress = resolve_review_progress(progress)
    with progress.suspended() if progress is not None else nullcontext():
        with prompt_metadata(
            allow_back=allow_back,
            back_value=PROMPT_BACK_VALUE,
            stage_key=stage_key,
            stage_step_index=stage_step_index,
        ):
            return terminal_prompt_yes_no(
                prompt_text,
                default,
                preview_path=preview_path,
                preview_title=preview_title,
            )


def prompt_replacement_review_agent_selection(
    *,
    reason_text: str,
    previous_selection: ReviewAgentSelection,
    force_model_change: bool,
    role_label: str,
    progress: ReviewStageProgress | None = None,
) -> ReviewAgentSelection | None:
    progress = resolve_review_progress(progress)
    message(reason_text)
    if not prompt_yes_no_choice(f"是否创建新的{role_label}继续当前阶段", True, progress=progress):
        return None
    while True:
        selection = prompt_review_agent_selection(
            default_vendor=previous_selection.vendor,
            default_model=previous_selection.model,
            default_reasoning_effort=previous_selection.reasoning_effort,
            default_proxy_url=previous_selection.proxy_url,
            role_label=role_label,
            progress=progress,
        )
        if not force_model_change or (
            selection.vendor != previous_selection.vendor
            or selection.model != previous_selection.model
        ):
            return selection
        message("新的智能体必须切换 vendor 或 model，当前选择与旧智能体完全相同，请重新选择。")


def prompt_required_replacement_review_agent_selection(
    *,
    reason_text: str,
    previous_selection: ReviewAgentSelection,
    force_model_change: bool,
    role_label: str,
    progress: ReviewStageProgress | None = None,
) -> ReviewAgentSelection:
    progress = resolve_review_progress(progress)
    prompt_is_patched = getattr(prompt_review_agent_selection, "__module__", __name__) != __name__
    if not stdin_is_interactive() and not prompt_is_patched:
        raise RuntimeError(f"{role_label} 需要重新配置智能体，但当前环境无法交互选择厂商/模型。")
    message(reason_text)
    while True:
        selection = prompt_review_agent_selection(
            default_vendor=previous_selection.vendor,
            default_model=previous_selection.model,
            default_reasoning_effort=previous_selection.reasoning_effort,
            default_proxy_url=previous_selection.proxy_url,
            role_label=role_label,
            progress=progress,
        )
        if not force_model_change or (
            selection.vendor != previous_selection.vendor
            or selection.model != previous_selection.model
        ):
            message(render_review_agent_selection(f"重新创建{role_label}", selection))
            return selection
        message("新的智能体必须切换 vendor 或 model，当前选择与旧智能体完全相同，请重新选择。")


def render_tmux_start_summary(role_name: str, worker: TmuxBatchWorker) -> str:
    return "\n".join(
        [
            f"{role_name} 已创建",
            f"runtime_dir: {worker.runtime_dir}",
            f"session_name: {worker.session_name}",
            "首次执行任务时会等待 READY；启动失败将进入阶段恢复逻辑。",
            "可使用以下命令进入会话:",
            f"  tmux attach -t {worker.session_name}",
        ]
    )


def worker_has_provider_auth_error(worker: TmuxBatchWorker | None) -> bool:
    if worker is None:
        return False
    try:
        state = worker.read_state()
    except Exception:
        state = {}
    health_status = str(state.get("health_status", "")).strip().lower()
    health_note = str(state.get("health_note", "")).strip().lower()
    return health_status == "provider_auth_error" or is_provider_auth_error(health_note)


def worker_has_provider_runtime_error(worker: TmuxBatchWorker | None) -> bool:
    if worker is None:
        return False
    try:
        state = worker.read_state()
    except Exception:
        state = {}
    health_status = str(state.get("health_status", "")).strip().lower()
    health_note = str(state.get("health_note", "")).strip().lower()
    last_provider_error = str(state.get("last_provider_error", "")).strip().lower()
    return (
        health_status == "provider_runtime_error"
        or is_provider_runtime_error(health_note)
        or is_provider_runtime_error(last_provider_error)
    )


def is_recoverable_startup_failure(error: Exception, worker: TmuxBatchWorker | None = None) -> bool:
    message_text = str(error or "").strip().lower()
    if is_agent_config_error(error):
        return True
    if is_provider_auth_error(error) or worker_has_provider_auth_error(worker):
        return True
    if is_provider_runtime_error(error) or worker_has_provider_runtime_error(worker):
        return True
    if is_agent_ready_timeout_error(error):
        return True
    if "shell initialization timed out" in message_text:
        return True
    if "agent exited back to shell while starting" in message_text:
        return True
    return False


def mark_worker_awaiting_reconfiguration(
    worker: TmuxBatchWorker | None,
    *,
    reason_text: str,
) -> None:
    if worker is None:
        return
    marker = getattr(worker, "mark_awaiting_reconfiguration", None)
    if not callable(marker):
        return
    try:
        marker(reason_text=reason_text)
    except Exception:
        return


def prompt_agent_ready_timeout_recovery(
    worker: TmuxBatchWorker | None,
    *,
    role_label: str,
    can_skip: bool,
    progress: ReviewStageProgress | None = None,
    reason_text: str = "",
    allow_recreate: bool = False,
    target_paths: Sequence[str | Path] = (),
) -> str:
    role_text = str(role_label or "").strip() or "智能体"
    session_name = str(getattr(worker, "session_name", "") or "").strip()
    reason = str(reason_text or "").strip() or (
        f"{session_name or role_text}启动超时，未能进入可输入状态。\n"
        "请先手动更换模型或处理该 AGENT，再选择恢复动作。"
    )
    return request_worker_manual_intervention(
        stage_label="智能体启动超时",
        role_label=session_name or role_text,
        worker=worker,
        reason_text=reason,
        target_paths=target_paths,
        progress=progress,
        allow_recreate=allow_recreate,
        allow_worker_dead=can_skip,
    )


def ensure_empty_file(file_path: str | Path) -> Path:
    target = Path(file_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")
    return target


def ensure_review_artifacts(md_path: str | Path, json_path: str | Path) -> tuple[Path, Path]:
    review_md = ensure_empty_file(md_path)
    review_json = Path(json_path).expanduser().resolve()
    review_json.parent.mkdir(parents=True, exist_ok=True)
    review_json.write_text("[]", encoding="utf-8")
    return review_md, review_json


def collect_review_limit_hitl_response(
    question_path: str | Path,
    *,
    stage_label: str,
    hitl_round: int,
    answer_path: str | Path | None = None,
    progress: ReviewStageProgress | None = None,
) -> str:
    question_file = Path(question_path).expanduser().resolve()
    question_text = question_file.read_text(encoding="utf-8").strip() if question_file.exists() else ""
    message()
    message(f"{stage_label} 第 {hitl_round} 轮，需要人工补充信息")
    message(f"问题文档: {question_file}")
    message(question_text or "(问题文档为空)")
    if progress is not None:
        progress.set_phase(f"{stage_label} / 等待 HITL")
    with progress.suspended() if progress is not None else nullcontext():
        return collect_multiline_input(
            title=f"{stage_label} HITL 第 {hitl_round} 轮回复",
            empty_retry_message="回复不能为空，请重新输入。",
            question_path=question_file,
            answer_path=answer_path,
            is_hitl=True,
        )


def collect_auto_review_limit_hitl_response(
    question_path: str | Path,
    *,
    stage_label: str,
    hitl_round: int,
) -> str:
    question_file = Path(question_path).expanduser().resolve()
    question_text = question_file.read_text(encoding="utf-8").strip() if question_file.exists() else ""
    message()
    message(f"{stage_label} 第 {hitl_round} 轮 已由 --yes 自动回复，继续非交互流程。")
    return (
        f"{stage_label} 第 {hitl_round} 轮自动回复：当前以 --yes 非交互模式运行。"
        "请采用最保守的修正路径：严格按已有原始需求、澄清记录和评审记录补齐遗漏，"
        "删除或改写无来源依据的扩展内容，不新增需求、不跳过评审、不再等待人工输入。"
        "若问题文档提供多个方案，优先选择“原样同步/最小修正/无扩展”的方案；"
        "将本轮自动决策和假设追加到 HITL 记录，随后清空问题文档并继续当前阶段。"
        "\n\n[自动回复所依据的问题文档]\n"
        f"{question_text or '(问题文档为空)'}"
    )


def parse_review_max_rounds(value: object, *, source: str, default: int = DEFAULT_STAGE_REVIEW_MAX_ROUNDS) -> int | None:
    text = str(value or "").strip()
    if not text:
        return int(default)
    if text.lower() == "infinite":
        return None
    try:
        parsed = int(text)
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"{source} 必须是正整数或 infinite") from error
    if parsed <= 0:
        raise RuntimeError(f"{source} 必须是正整数或 infinite")
    return parsed


def prompt_review_max_rounds(
    *,
    default: int = DEFAULT_STAGE_REVIEW_MAX_ROUNDS,
    progress: ReviewStageProgress | None = None,
    prompt_text: str = "输入最大审核轮次（输入 infinite 表示不设上限）",
    source: str = "最大审核轮次",
    allow_back: bool = False,
    stage_key: str = "",
    stage_step_index: int = 0,
) -> int | None:
    progress = resolve_review_progress(progress)
    with progress.suspended() if progress is not None else nullcontext():
        while True:
            with prompt_metadata(
                allow_back=allow_back,
                back_value=PROMPT_BACK_VALUE,
                stage_key=stage_key,
                stage_step_index=stage_step_index,
            ):
                value = prompt_with_default(prompt_text, str(default)).strip()
            try:
                return parse_review_max_rounds(value, source=source, default=default)
            except RuntimeError as error:
                message(str(error))


def render_review_limit_force_hitl_prompt(
    *,
    config: ReviewLimitHitlConfig,
    review_limit: int,
    review_rounds_used: int,
    hitl_record_md: str | Path,
    extra_inputs: Sequence[str | Path] = (),
) -> str:
    merged_review_md = str(Path(config.merged_review_path).expanduser().resolve())
    ask_human_md = str(Path(config.ask_human_path).expanduser().resolve())
    output_md = str(Path(config.primary_output_path).expanduser().resolve())
    hitl_record_md = str(Path(hitl_record_md).expanduser().resolve())
    feedback_md = str(Path(config.output_summary_path).expanduser().resolve())
    extra_input_list = [str(Path(item).expanduser().resolve()) for item in extra_inputs]
    input_lines = "\n".join(f"- 《{item}》" for item in [merged_review_md, output_md, hitl_record_md, *extra_input_list])
    return f"""## 任务目标
当前《{merged_review_md}》对应的评审已累计 {review_rounds_used} 轮，达到上限 {review_limit}。
你现在必须停止继续自修，改为发起一次强制 HITL，请人类给出新的决策信息。

## 必读输入
{input_lines}

## 强制执行步骤
1. 阅读并去重《{merged_review_md}》中的多轮评审意见。
2. 结合《{output_md}》与《{hitl_record_md}》，总结“为什么多轮仍未通过”。
3. 只保留必须由人类拍板的缺口，覆盖写入《{ask_human_md}》。
4. 不要继续尝试闭环该问题，不要声称已完成继续工作。

## 《{ask_human_md}》固定结构
- [多轮未通过原因]
- [仍未闭环的问题]
- [需要人类决策]
- [可选方案与影响]
- [继续工作后将修改的产物]

## 输出约束
- 允许修改：《{ask_human_md}》、可选更新《{hitl_record_md}》。
- 禁止修改：《{output_md}》与其他业务产物。
- 若《{ask_human_md}》为空，视为失败。
- 只允许返回 `HITL`。
"""


def render_review_limit_human_reply_prompt(
    *,
    config: ReviewLimitHitlConfig,
    human_msg: str,
    hitl_record_md: str | Path,
    extra_inputs: Sequence[str | Path] = (),
) -> str:
    ask_human_md = str(Path(config.ask_human_path).expanduser().resolve())
    output_md = str(Path(config.primary_output_path).expanduser().resolve())
    hitl_record_md = str(Path(hitl_record_md).expanduser().resolve())
    feedback_md = str(Path(config.output_summary_path).expanduser().resolve())
    extra_input_list = [str(Path(item).expanduser().resolve()) for item in extra_inputs]
    input_lines = "\n".join(f"- 《{item}》" for item in [output_md, hitl_record_md, *extra_input_list])
    return f"""## 任务目标
你上一轮因评审超过上限触发了 HITL。现在人类已经回复，请先同步人类信息，再继续当前阶段工作。

## 人类回复
[HUMAN MSG START]
{human_msg}
[HUMAN MSG END]

## 必读输入
{input_lines}

## 执行步骤
1. 解析人类回复，区分有效信息、噪音、冲突修订。
2. 以追加 / 拦截 / 覆写规则同步《{hitl_record_md}》。
3. 若信息仍不足，继续覆盖写入《{ask_human_md}》并返回 `HITL`。
4. 若信息足够，继续当前阶段工作，必须更新《{output_md}》。
5. 如有必要，同时更新《{feedback_md}》说明本轮处理结果。

## 输出约束
- 如果仍需人类介入：必须写《{ask_human_md}》，只返回 `HITL`。
- 如果信息已足够：必须清空《{ask_human_md}》，并完成《{config.continue_output_label}》对应产物更新，只返回 `修改完成`。
- 禁止输出其他文本。
"""


def run_review_limit_hitl_cycle(
    *,
    stage_label: str,
    ask_human_path: str | Path,
    hitl_record_path: str | Path,
    initial_turn: Callable[[], object],
    human_reply_turn: Callable[[str], object],
    human_input_provider: Callable[[Path, int], str] | None = None,
    progress: ReviewStageProgress | None = None,
    max_hitl_rounds: int = 8,
    on_hitl_question: Callable[[int, Path], None] | None = None,
    on_hitl_answer: Callable[[int, str, Path], None] | None = None,
) -> ReviewLimitHitlResult:
    ask_human_file = Path(ask_human_path).expanduser().resolve()
    hitl_record_file = Path(hitl_record_path).expanduser().resolve()
    owner = initial_turn()
    if not ask_human_file.exists() or not ask_human_file.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"{stage_label} 超限后未生成有效《{ask_human_file.name}》")
    post_hitl_continue_completed = False

    def _invoke_callback(callback: Callable[..., object] | None, callback_name: str, *args: object) -> None:
        if callback is None:
            return
        try:
            callback(*args)
        except Exception as error:  # noqa: BLE001
            try:
                message(f"警告：评审超限 HITL 回调失败 callback={callback_name} error={error}")
            except Exception:
                pass

    for hitl_round in range(1, max_hitl_rounds + 1):
        if not ask_human_file.read_text(encoding="utf-8").strip():
            return ReviewLimitHitlResult(
                owner=owner,
                rounds_used=hitl_round - 1,
                post_hitl_continue_completed=post_hitl_continue_completed,
            )
        _invoke_callback(on_hitl_question, "on_hitl_question", hitl_round, ask_human_file)
        if human_input_provider is not None:
            human_msg = human_input_provider(ask_human_file, hitl_round)
        else:
            human_msg = collect_review_limit_hitl_response(
                ask_human_file,
                stage_label=stage_label,
                hitl_round=hitl_round,
                answer_path=hitl_record_file,
                progress=progress,
            )
        _invoke_callback(on_hitl_answer, "on_hitl_answer", hitl_round, human_msg, hitl_record_file)
        owner = human_reply_turn(human_msg)
        post_hitl_continue_completed = True
    raise RuntimeError(f"{stage_label} HITL 轮次超过上限: {max_hitl_rounds}")
