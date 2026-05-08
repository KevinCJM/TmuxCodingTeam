# -*- encoding: utf-8 -*-
"""
@File: A01_Routing_LayerPlanning.py
@Modify Time: 2026/4/12
@Author: Kevin-Chen
@Descriptions: AGENT初始化阶段 CLI 入口
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tmux_core.runtime.vendor_catalog import (
    LEGACY_DEFAULT_MODEL_BY_VENDOR,
    VENDOR_ORDER as CATALOG_VENDOR_ORDER,
    get_default_model_for_vendor,
    get_model_choices,
    get_normalized_effort_choices,
    get_vendor_inventory,
)
from T02_tmux_agents import (
    AgentRunConfig,
    TmuxRuntimeController,
    Vendor,
    build_session_name,
    cleanup_registered_tmux_workers,
    list_registered_tmux_workers,
    worker_state_is_prelaunch_active,
)
from T03_agent_init_workflow import (
    BatchInitResult,
    RoutingCleanupResult,
    DirectoryInitResult,
    LiveWorkerHandle,
    RunStore,
    cleanup_routing_stage_artifacts,
    kill_run_tmux_sessions,
    project_has_business_files,
    routing_layer_readiness_issues,
    resolve_existing_directory,
    resolve_target_selection,
    run_batch_initialization,
    should_skip_routing_init_for_empty_project,
)
from T09_terminal_ops import (
    PROMPT_BACK_VALUE,
    PromptBackRequested,
    SingleLineSpinnerMonitor,
    TERMINAL_SPINNER_FRAMES,
    message,
    maybe_launch_tui,
    prompt_metadata,
    prompt_select_option,
    prompt_with_default,
)


VENDOR_CHOICES = CATALOG_VENDOR_ORDER
VENDOR_ALIASES = {
    "1": "codex",
    "2": "claude",
    "3": "gemini",
    "4": "opencode",
    "claude code": "claude",
    "claude-code": "claude",
    "open code": "opencode",
    "open-code": "opencode",
}
DEFAULT_MODEL_BY_VENDOR = dict(LEGACY_DEFAULT_MODEL_BY_VENDOR)
EFFORT_CHOICES = ("low", "medium", "high", "xhigh", "max")
PROXY_PRESET_CHOICES = ("", "10900", "7890")
RUN_INIT_CHOICES = ("yes", "no")
EMPTY_PROJECT_ROUTING_SKIP_MESSAGE = "当前项目未检测到业务文件，跳过路由层初始化。"


@dataclass(frozen=True)
class CliRequest:
    project_dir: str
    target_dirs: tuple[str, ...]
    vendor: str
    model: str
    reasoning_effort: str
    proxy_port: str
    run_init: bool
    max_refine_rounds: int
    auto_confirm: bool


@dataclass(frozen=True)
class RoutingStageResult:
    project_dir: str
    skipped: bool
    exit_code: int
    batch_result: BatchInitResult | None = None
    killed_sessions: tuple[str, ...] = ()
    cleanup_result: RoutingCleanupResult | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AGENT初始化阶段：tmux + coding agent 路由层初始化")
    parser.add_argument("--project-dir", help="项目工作目录")
    # Workflow entry forwards a shared argv shape to every stage.
    # Routing init does not use requirement_name, but must accept it for compatibility.
    parser.add_argument("--requirement-name", default="", help=argparse.SUPPRESS)
    parser.add_argument("--reuse-existing-original-requirement", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--allow-project-dir-back", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--target-dir", action="append", default=[], help="额外目标目录，可重复传入")
    parser.add_argument("--vendor", help="厂商: codex|claude|gemini|opencode")
    parser.add_argument("--model", help="模型名称")
    parser.add_argument("--effort", help="推理强度")
    parser.add_argument("--proxy-port", default="", help="代理端口或完整代理 URL")
    parser.add_argument("--run-init", choices=RUN_INIT_CHOICES, help="是否执行 AGENT初始化: yes|no")
    parser.add_argument("--max-refine-rounds", type=int, default=3, help="最大 refine 轮数")
    parser.add_argument("--resume-run", default="", help="恢复已有 run_id，仅供 B01 控制台使用")
    parser.add_argument("--yes", action="store_true", help="跳过最终确认")
    parser.add_argument("--no-tui", action="store_true", help="显式禁用 OpenTUI")
    parser.add_argument("--legacy-cli", action="store_true", help="使用旧版 Python CLI，不跳转 OpenTUI")
    return parser


def normalize_vendor_choice(value: str | None) -> str:
    text = str(value or "").strip().lower()
    text = VENDOR_ALIASES.get(text, text)
    if text not in VENDOR_CHOICES:
        raise ValueError(f"不支持的厂商: {value}")
    return text


def normalize_model_choice(vendor: str, value: str | None) -> str:
    normalized_vendor = normalize_vendor_choice(vendor)
    inventory = get_vendor_inventory(normalized_vendor)
    models = tuple(item.model_id for item in get_model_choices(normalized_vendor))
    text = str(value or "").strip()
    if not text:
        raise ValueError("模型不能为空")
    if text.isdigit():
        index = int(text)
        if 1 <= index <= len(models):
            return models[index - 1]
    if normalized_vendor == "opencode" and text == "default":
        resolved_default = get_default_model_for_vendor(normalized_vendor)
        if resolved_default:
            return resolved_default
    if text in models:
        return text
    if not inventory.installed:
        raise ValueError(f"{normalized_vendor} 未安装，无法选择模型")
    available = ", ".join(models[:12])
    raise ValueError(f"{normalized_vendor} 不支持的模型: {value}; 已扫描模型: {available or 'none'}")


def normalize_effort_choice(vendor: str, model: str, value: str | None) -> str:
    normalized_vendor = normalize_vendor_choice(vendor)
    normalized_model = normalize_model_choice(normalized_vendor, model)
    allowed = get_normalized_effort_choices(normalized_vendor, normalized_model)
    text = str(value or "").strip().lower()
    if text.isdigit():
        index = int(text)
        if 1 <= index <= len(allowed):
            return allowed[index - 1]
    if text in allowed:
        return text
    raise ValueError(f"{normalized_model} 不支持的推理强度: {value}")


def normalize_run_init_choice(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "yes").strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return True
    if text in {"no", "n", "false", "0"}:
        return False
    raise ValueError(f"无法解析 run-init 选项: {value}")


def split_target_dirs_text(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def prompt_project_dir(default: str = "") -> str:
    last_error = ""
    while True:
        with prompt_metadata(error_message=last_error):
            candidate = str(prompt_with_default("输入项目工作目录", default)).strip()
        default = candidate
        try:
            if not Path(candidate).expanduser().is_absolute():
                raise ValueError("请输入绝对路径")
            return str(resolve_existing_directory(candidate))
        except Exception as error:
            last_error = f"目录无效: {error}"
            message(last_error)


def prompt_target_dirs(defaults: Sequence[str] = ()) -> tuple[str, ...]:
    default_text = ",".join(defaults)
    raw_value = prompt_with_default("输入额外目标目录，多个用逗号分隔，可留空", default_text, allow_empty=True)
    return tuple(split_target_dirs_text(raw_value))


def _role_scoped_text(text: str, role_label: str = "") -> str:
    role = str(role_label or "").strip()
    if not role:
        return text
    return f"为 {role} {text}"


def _predict_routing_role_label(
        *,
        project_dir: str | Path,
        target_dirs: Sequence[str],
        run_init: bool,
) -> str:
    try:
        selection = resolve_target_selection(
            project_dir=str(resolve_existing_directory(project_dir)),
            target_dirs=tuple(target_dirs),
            run_init=run_init,
        )
        focus_dir = selection.selected_dirs[0] if selection.selected_dirs else selection.project_dir
        focus_path = Path(focus_dir).expanduser().resolve()
        occupied_session_names: set[str] = set()
        try:
            occupied_session_names.update(TmuxRuntimeController().list_sessions())
        except Exception:
            pass
        for worker in list_registered_tmux_workers():
            session_name = str(getattr(worker, "session_name", "") or "").strip()
            if session_name:
                occupied_session_names.add(session_name)
        return build_session_name(
            focus_path.name or "project",
            focus_path,
            Vendor.CODEX,
            occupied_session_names=sorted(occupied_session_names),
        )
    except Exception:
        return "路由器"


def prompt_vendor(default: str = "codex", *, role_label: str = "") -> str:
    options = []
    installed_defaults = [vendor for vendor in VENDOR_CHOICES if get_vendor_inventory(vendor).installed]
    actual_default = normalize_vendor_choice(default if default in VENDOR_CHOICES else "codex")
    if actual_default not in installed_defaults and installed_defaults:
        actual_default = installed_defaults[0]
    for vendor in VENDOR_CHOICES:
        inventory = get_vendor_inventory(vendor)
        label = (
            f"{vendor} | installed={'yes' if inventory.installed else 'no'}"
            f" | default={inventory.default_model or 'unknown'}"
            f" | models={len(inventory.models)}"
        )
        options.append((vendor, label))
    scoped_title = _role_scoped_text("选择厂商", role_label)
    candidate = prompt_select_option(
        title=scoped_title,
        options=options,
        default_value=actual_default,
        prompt_text=scoped_title,
    )
    return normalize_vendor_choice(candidate)


def prompt_model(vendor: str, default: str | None = None, *, role_label: str = "") -> str:
    normalized_vendor = normalize_vendor_choice(vendor)
    inventory = get_vendor_inventory(normalized_vendor)
    if not inventory.installed:
        raise ValueError(f"{normalized_vendor} 未安装，无法选择模型")
    models = get_model_choices(normalized_vendor)
    if not models:
        raise ValueError(f"{normalized_vendor} 没有可用模型")
    model_ids = tuple(item.model_id for item in models)
    actual_default = ""
    for candidate in (
        default,
        get_default_model_for_vendor(normalized_vendor),
        DEFAULT_MODEL_BY_VENDOR.get(normalized_vendor),
        model_ids[0] if model_ids else "",
    ):
        if not str(candidate or "").strip():
            continue
        try:
            actual_default = normalize_model_choice(normalized_vendor, candidate)
            break
        except ValueError:
            continue
    if not actual_default and model_ids:
        actual_default = model_ids[0]
    options = [
        (
            item.model_id,
            f"{item.model_id}"
            f" | efforts: {'/'.join(item.reasoning.normalized_reasoning_levels or ('high',))}"
            f"{' | synthetic' if item.synthetic else ''}"
            f"{' | fallback' if item.source_kind == 'legacy_fallback' else ''}",
        )
        for item in models
    ]
    scoped_title = _role_scoped_text(f"选择 {normalized_vendor} 模型", role_label)
    candidate = prompt_select_option(
        title=scoped_title,
        options=options,
        default_value=actual_default,
        prompt_text=scoped_title,
    )
    return normalize_model_choice(normalized_vendor, candidate)


def prompt_effort(vendor: str, model: str, default: str = "high", *, role_label: str = "") -> str:
    normalized_vendor = normalize_vendor_choice(vendor)
    normalized_model = normalize_model_choice(normalized_vendor, model)
    allowed = get_normalized_effort_choices(normalized_vendor, normalized_model)
    try:
        actual_default = normalize_effort_choice(normalized_vendor, normalized_model, default)
    except ValueError:
        actual_default = "high" if "high" in allowed else allowed[0]
    scoped_title = _role_scoped_text(f"选择 {normalized_model} 推理强度", role_label)
    candidate = prompt_select_option(
        title=scoped_title,
        options=[(effort, effort) for effort in allowed],
        default_value=actual_default,
        prompt_text=scoped_title,
    )
    return normalize_effort_choice(normalized_vendor, normalized_model, candidate)


def prompt_proxy_port(default: str = "", *, role_label: str = "") -> str:
    normalized_default = str(default or "").strip()
    has_preset_default = normalized_default in PROXY_PRESET_CHOICES
    default_value = normalized_default if has_preset_default else "__custom__"
    options = (
        ("", "(none)"),
        ("10900", "10900"),
        ("7890", "7890"),
        ("__custom__", "自定义输入"),
    )
    scoped_title = _role_scoped_text("选择代理端口", role_label)
    candidate = prompt_select_option(
        title=f"{scoped_title}:",
        options=options,
        default_value=default_value,
        prompt_text=scoped_title,
    )
    if candidate != "__custom__":
        return candidate
    return prompt_with_default("输入代理端口或完整代理 URL（可留空）", normalized_default, allow_empty=True)


def prompt_run_init(default: bool = True) -> bool:
    candidate = prompt_select_option(
        title="是否执行 AGENT初始化:",
        options=(("yes", "yes"), ("no", "no")),
        default_value="yes" if default else "no",
        prompt_text="是否执行 AGENT初始化",
    )
    return normalize_run_init_choice(candidate)


def prompt_confirmation(summary_text: str, *, force_yes: bool = False) -> bool:
    message("\n执行摘要:")
    message(summary_text)
    if not force_yes:
        decision = prompt_select_option(
            title="确认开始执行?",
            options=(("yes", "yes"), ("no", "no")),
            default_value="yes",
            prompt_text="确认开始执行",
        )
        return normalize_run_init_choice(decision)

    while True:
        decision = prompt_select_option(
            title="确认开始执行? (yes only)",
            options=(("yes", "yes"), ("no", "no")),
            default_value="yes",
            prompt_text="确认开始执行",
        )
        if normalize_run_init_choice(decision):
            return True
        message("当前项目路由层文件缺失，必须执行初始化。请输入 yes 继续。")


def _routing_prompt_step(step_index: int, *, allow_back: bool):
    return prompt_metadata(
        allow_back=allow_back,
        back_value=PROMPT_BACK_VALUE,
        stage_key="routing",
        stage_step_index=step_index,
    )


def _collect_interactive_cli_request(
        args: argparse.Namespace,
        *,
        initial_request: CliRequest | None = None,
        start_step: int = 0,
) -> CliRequest:
    max_refine_rounds = int(args.max_refine_rounds or (initial_request.max_refine_rounds if initial_request else 3))
    if max_refine_rounds < 1:
        raise ValueError("max-refine-rounds 必须 >= 1")

    project_dir = str(args.project_dir or (initial_request.project_dir if initial_request else "")).strip()
    target_dirs = tuple(args.target_dir or (initial_request.target_dirs if initial_request else ()))
    run_init = bool(initial_request.run_init) if initial_request else True
    vendor = str(args.vendor or (initial_request.vendor if initial_request else "codex")).strip() or "codex"
    model = str(args.model or (initial_request.model if initial_request else "")).strip()
    reasoning_effort = str(args.effort or (initial_request.reasoning_effort if initial_request else "high")).strip() or "high"
    proxy_port = str(args.proxy_port or (initial_request.proxy_port if initial_request else "")).strip()
    project_missing_files: tuple[str, ...] = ()
    allow_project_dir_back = bool(getattr(args, "allow_project_dir_back", False) and project_dir)
    first_prompt_step = 0 if (not project_dir or allow_project_dir_back) else 1
    initial_step = 1 if project_dir else 0
    step = max(first_prompt_step, int(start_step or initial_step))

    while step < 7:
        try:
            if step == 0:
                with _routing_prompt_step(0, allow_back=False):
                    project_dir = prompt_project_dir(project_dir)
                project_missing_files = tuple(routing_layer_readiness_issues(project_dir))
                step = 1
                continue
            if step == 1:
                project_missing_files = tuple(routing_layer_readiness_issues(project_dir))
                if project_missing_files:
                    should_skip_empty_project = should_skip_routing_init_for_empty_project(
                        project_dir,
                        project_missing_files=project_missing_files,
                        target_dirs=target_dirs,
                        explicit_run_init_yes=bool(args.run_init and normalize_run_init_choice(args.run_init)),
                    )
                    run_init = not should_skip_empty_project
                    if not should_skip_empty_project and not args.run_init:
                        message("当前项目路由层文件缺失, 强制执行路由初始化")
                else:
                    with _routing_prompt_step(1, allow_back=step > first_prompt_step):
                        run_init = prompt_run_init(run_init)
                step = 2
                continue
            if step == 2:
                if run_init:
                    with _routing_prompt_step(2, allow_back=step > first_prompt_step):
                        target_dirs = prompt_target_dirs(target_dirs)
                else:
                    target_dirs = tuple()
                step = 3
                continue
            if step == 3:
                if not run_init:
                    vendor = normalize_vendor_choice(vendor or "codex")
                    model = normalize_model_choice(vendor, model or get_default_model_for_vendor(vendor))
                    reasoning_effort = normalize_effort_choice(vendor, model, reasoning_effort or "high")
                    proxy_port = ""
                    step = 7
                    continue
                routing_role_label = _predict_routing_role_label(project_dir=project_dir, target_dirs=target_dirs, run_init=run_init)
                with _routing_prompt_step(3, allow_back=step > first_prompt_step):
                    vendor = prompt_vendor(vendor or "codex", role_label=routing_role_label)
                if model:
                    try:
                        normalize_model_choice(vendor, model)
                    except ValueError:
                        model = ""
                step = 4
                continue
            if step == 4:
                routing_role_label = _predict_routing_role_label(project_dir=project_dir, target_dirs=target_dirs, run_init=run_init)
                model_default = model or get_default_model_for_vendor(vendor)
                with _routing_prompt_step(4, allow_back=step > first_prompt_step):
                    model = prompt_model(vendor, model_default, role_label=routing_role_label)
                step = 5
                continue
            if step == 5:
                routing_role_label = _predict_routing_role_label(project_dir=project_dir, target_dirs=target_dirs, run_init=run_init)
                with _routing_prompt_step(5, allow_back=step > first_prompt_step):
                    reasoning_effort = prompt_effort(vendor, model, reasoning_effort or "high", role_label=routing_role_label)
                step = 6
                continue
            if step == 6:
                routing_role_label = _predict_routing_role_label(project_dir=project_dir, target_dirs=target_dirs, run_init=run_init)
                with _routing_prompt_step(6, allow_back=step > first_prompt_step):
                    proxy_port = prompt_proxy_port(proxy_port, role_label=routing_role_label)
                step = 7
                continue
        except PromptBackRequested:
            if step == first_prompt_step:
                continue
            if step == 2 and project_missing_files and not args.run_init:
                step = 0
            else:
                step = max(first_prompt_step, step - 1)

    return CliRequest(
        project_dir=project_dir,
        target_dirs=target_dirs,
        vendor=vendor,
        model=model,
        reasoning_effort=reasoning_effort,
        proxy_port=proxy_port,
        run_init=run_init,
        max_refine_rounds=max_refine_rounds,
        auto_confirm=bool(args.yes),
    )


def collect_cli_request(
        args: argparse.Namespace,
        *,
        initial_request: CliRequest | None = None,
        start_step: int = 0,
) -> CliRequest:
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
    if not parameter_mode:
        return _collect_interactive_cli_request(args, initial_request=initial_request, start_step=start_step)
    project_dir = (
        str(resolve_existing_directory(args.project_dir))
        if args.project_dir
        else prompt_project_dir("")
    )
    target_dirs = tuple(args.target_dir or ())
    project_missing_files = tuple(routing_layer_readiness_issues(project_dir))
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
            else (True if parameter_mode else prompt_run_init(True))
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
        routing_role_label = _predict_routing_role_label(
            project_dir=project_dir,
            target_dirs=target_dirs,
            run_init=run_init,
        )
        vendor = normalize_vendor_choice(args.vendor) if args.vendor else prompt_vendor("codex", role_label=routing_role_label)
        model_default = get_default_model_for_vendor(vendor)
        model = normalize_model_choice(
            vendor,
            args.model or prompt_model(vendor, model_default, role_label=routing_role_label),
        )
        reasoning_effort = normalize_effort_choice(
            vendor,
            model,
            args.effort or (
                "high" if parameter_mode else prompt_effort(vendor, model, "high", role_label=routing_role_label)
            ),
        )
        proxy_port = args.proxy_port or ("" if parameter_mode else prompt_proxy_port("", role_label=routing_role_label))
    else:
        vendor = normalize_vendor_choice(args.vendor or "codex")
        model = normalize_model_choice(vendor, args.model or get_default_model_for_vendor(vendor))
        reasoning_effort = normalize_effort_choice(vendor, model, args.effort or "high")
        proxy_port = args.proxy_port or ""
    max_refine_rounds = int(args.max_refine_rounds or 3)
    if max_refine_rounds < 1:
        raise ValueError("max-refine-rounds 必须 >= 1")

    return CliRequest(
        project_dir=project_dir,
        target_dirs=target_dirs,
        vendor=vendor,
        model=model,
        reasoning_effort=reasoning_effort,
        proxy_port=proxy_port,
        run_init=run_init,
        max_refine_rounds=max_refine_rounds,
        auto_confirm=bool(args.yes),
    )


def prepare_batch_request(request: CliRequest) -> tuple[AgentRunConfig, object]:
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
    return config, selection


def render_preflight_summary(request: CliRequest, config: AgentRunConfig, selection) -> str:
    lines = [
        f"project_dir: {selection.project_dir}",
        f"target_dirs: {', '.join(request.target_dirs) if request.target_dirs else '(none)'}",
        f"selected_dirs: {', '.join(selection.selected_dirs) if selection.selected_dirs else '(none)'}",
        f"skipped_dirs: {', '.join(selection.skipped_dirs) if selection.skipped_dirs else '(none)'}",
        f"forced_dirs: {', '.join(selection.forced_dirs) if selection.forced_dirs else '(none)'}",
        f"vendor: {config.vendor.value}",
        f"model: {config.model}",
        f"reasoning_effort: {config.reasoning_effort}",
        f"proxy_url: {config.proxy_url or '(none)'}",
        f"max_refine_rounds: {request.max_refine_rounds}",
    ]
    if selection.project_missing_files:
        lines.append(f"project_missing_files: {', '.join(selection.project_missing_files)}")
    return "\n".join(lines)


def render_noop_summary(request: CliRequest, config: AgentRunConfig, selection) -> str:
    lines = [
        "无需执行 AGENT初始化。",
        f"project_dir: {selection.project_dir}",
        f"vendor: {config.vendor.value}",
        f"model: {config.model}",
        f"reasoning_effort: {config.reasoning_effort}",
        f"proxy_url: {config.proxy_url or '(none)'}",
    ]
    if selection.skipped_dirs:
        lines.append(f"skipped_dirs: {', '.join(selection.skipped_dirs)}")
    return "\n".join(lines)


def display_status_label(result) -> str:
    if result.status == "failed":
        return "failed"
    if result.status == "skipped":
        return "skipped"
    if result.forced:
        return "forced"
    return result.status


def format_batch_summary(batch_result: BatchInitResult) -> str:
    lines = [
        f"run_id: {batch_result.run_id}",
        f"runtime_dir: {batch_result.runtime_dir}",
        f"vendor: {batch_result.config['vendor']}",
        f"model: {batch_result.config['model']}",
        f"reasoning_effort: {batch_result.config['reasoning_effort']}",
        f"proxy_url: {batch_result.config['proxy_url'] or '(none)'}",
        "directories:",
    ]
    for item in batch_result.results:
        label = display_status_label(item)
        line = f"- {item.work_dir}: {label}"
        if item.failure_reason:
            line += f" | failure={item.failure_reason}"
        if item.last_audit_summary:
            compact = item.last_audit_summary.replace("\n", " | ")
            line += f" | audit={compact}"
        lines.append(line)
    return "\n".join(lines)


def determine_exit_code(batch_result: BatchInitResult) -> int:
    return 1 if any(item.status == "failed" for item in batch_result.results) else 0


def failed_batch_results(batch_result: BatchInitResult) -> list[DirectoryInitResult]:
    return [item for item in batch_result.results if item.status == "failed"]


def render_routing_failure_summary(
    batch_result: BatchInitResult,
    killed_sessions: Sequence[str],
    cleanup_result: RoutingCleanupResult | None = None,
) -> str:
    cleanup_result = cleanup_result or RoutingCleanupResult()
    lines = ["路由层初始化未完全通过"]
    failures = failed_batch_results(batch_result)
    if failures:
        lines.append("失败目录:")
        for item in failures:
            reason = item.failure_reason or item.last_audit_summary or item.status
            line = f"- {item.work_dir}: {reason}"
            lines.append(line)
    if killed_sessions:
        lines.append(f"已清理路由层 tmux 会话: {len(killed_sessions)}")
    else:
        lines.append("路由层 tmux 会话已清理")
    if cleanup_result.removed_intermediate_count:
        lines.append(f"已清理阶段中间文件: {cleanup_result.removed_intermediate_count}")
    if cleanup_result.removed_runtime_count:
        lines.append(f"已清理阶段运行目录: {cleanup_result.removed_runtime_count}")
    lines.append("请修复失败目录后重新发起路由初始化。")
    return "\n".join(lines)


def render_requirements_stage_placeholder(
    killed_sessions: Sequence[str],
    cleanup_result: RoutingCleanupResult | None = None,
) -> str:
    cleanup_result = cleanup_result or RoutingCleanupResult()
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


def render_runtime_start_summary(
    *,
    run_store: RunStore,
    live_workers: Sequence[LiveWorkerHandle],
    immediate_results: Sequence[DirectoryInitResult],
) -> str:
    lines = [
        "路由层初始化已启动",
        f"run_id: {run_store.manifest.run_id}",
        f"runtime_dir: {run_store.manifest.runtime_dir}",
    ]
    if live_workers:
        lines.append("tmux sessions:")
        for handle in live_workers:
            forced_text = " | forced" if handle.forced else ""
            lines.append(f"- {handle.session_name} | {handle.work_dir}{forced_text}")
        lines.append("可使用以下命令进入某个会话:")
        for handle in live_workers:
            lines.append(f"  tmux attach -t {handle.session_name}")
    else:
        lines.append("tmux sessions: (none)")
    if immediate_results:
        lines.append("preflight failures:")
        for item in immediate_results:
            lines.append(f"- {item.work_dir}: {item.failure_reason or item.status}")
    return "\n".join(lines)


def summarize_live_result_counts(run_store: RunStore) -> dict[str, int]:
    counts = {"pending": 0, "running": 0, "passed": 0, "failed": 0, "skipped": 0}
    for entry in run_store.manifest.workers:
        status = str(entry.result_status or "").strip() or "pending"
        if status in {"passed", "failed", "skipped", "stale_failed"}:
            if status == "stale_failed":
                counts["failed"] += 1
            else:
                counts[status] += 1
            continue
        if status == "pending" and entry.workflow_stage == "pending":
            counts["pending"] += 1
        else:
            counts["running"] += 1
    return counts


def render_live_progress_frame(
    *,
    run_store: RunStore,
    selection,
    tick: int = 0,
) -> str:
    counts = summarize_live_result_counts(run_store)
    spinner = TERMINAL_SPINNER_FRAMES[tick % len(TERMINAL_SPINNER_FRAMES)]
    lines = [
        f"路由层初始化运行中... {spinner}",
        f"run_id: {run_store.manifest.run_id}",
        f"runtime_dir: {run_store.manifest.runtime_dir}",
        (
            "counts: "
            f"pending={counts['pending']} "
            f"running={counts['running']} "
            f"passed={counts['passed']} "
            f"failed={counts['failed']} "
            f"skipped={counts['skipped']}"
        ),
        "sessions:",
    ]
    entry_by_dir = {item.work_dir: item for item in run_store.manifest.workers}
    for index, work_dir in enumerate(selection.selected_dirs, start=1):
        entry = entry_by_dir.get(work_dir)
        if entry is None:
            lines.append(f"  {index}. pending | {work_dir} | stage=pending | phase=unknown | health=unknown | note=pending")
            continue
        status = str(entry.result_status or "").strip() or "pending"
        if status not in {"passed", "failed", "skipped", "stale_failed"}:
            status = "running" if entry.workflow_stage != "pending" else "pending"
        if status == "stale_failed":
            status = "failed"
        workflow_stage = entry.workflow_stage or "pending"
        agent_state = "STARTING" if worker_state_is_prelaunch_active(entry) else entry.agent_state or "DEAD"
        health_status = entry.health_status or "unknown"
        note = entry.note or workflow_stage or "pending"
        session_name = entry.session_name or "(preparing)"
        lines.append(
            f"  {index}. {status} | {session_name} | {work_dir} | "
            f"stage={workflow_stage} | state={agent_state} | health={health_status} | note={note}"
        )
    if selection.skipped_dirs:
        lines.append("skipped:")
        for skipped_dir in selection.skipped_dirs:
            lines.append(f"  - {skipped_dir}")
    return "\n".join(lines)


def render_live_progress_line(
    *,
    run_store: RunStore,
    selection,
    tick: int = 0,
) -> str:
    for entry in run_store.manifest.workers:
        state_path = str(getattr(entry, "state_path", "") or "").strip()
        if not state_path:
            continue
        try:
            run_store.update_worker_state_from_file(
                entry.work_dir,
                state_path,
                preserve_workflow_fields=True,
            )
        except Exception:
            continue
    counts = summarize_live_result_counts(run_store)
    spinner = TERMINAL_SPINNER_FRAMES[tick % len(TERMINAL_SPINNER_FRAMES)]
    entry_by_dir = {item.work_dir: item for item in run_store.manifest.workers}
    focus_dirs = list(selection.selected_dirs) or [selection.project_dir]
    focus_dir = focus_dirs[tick % len(focus_dirs)]
    entry = entry_by_dir.get(focus_dir)
    if entry is None:
        focus_label = f"{Path(focus_dir).name}:pending/unknown"
        note = "pending"
    else:
        workflow_stage = entry.workflow_stage or "pending"
        agent_state = "STARTING" if worker_state_is_prelaunch_active(entry) else entry.agent_state or "DEAD"
        focus_label = f"{Path(focus_dir).name}:{workflow_stage}/{agent_state}"
        note = entry.note or workflow_stage
    return (
        f"{spinner} 路由层初始化中"
        f" | pending={counts['pending']}"
        f" running={counts['running']}"
        f" passed={counts['passed']}"
        f" failed={counts['failed']}"
        f" skipped={counts['skipped']}"
        f" | {focus_label}"
        f" | {note}"
    )


class TerminalProgressMonitor:
    def __init__(
        self,
        *,
        run_id: str,
        runtime_root: str | Path,
        selection,
        stream=None,
        interval_sec: float = 0.2,
    ) -> None:
        self.run_id = run_id
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        self.selection = selection
        self.stream = stream or sys.stdout
        self.interval_sec = interval_sec
        self._monitor = SingleLineSpinnerMonitor(
            frame_builder=self._build_frame,
            stream=self.stream,
            interval_sec=self.interval_sec,
        )
        self._last_frame = ""
        self._last_line_width = 0
        self._isatty = bool(getattr(self.stream, "isatty", lambda: False)())

    def start(self) -> None:
        self._monitor.start()

    def stop(self) -> None:
        self._monitor.stop()

    def _display_line(self, line: str) -> None:
        self._monitor._display_line(line)
        self._last_frame = self._monitor._last_frame
        self._last_line_width = self._monitor._last_line_width

    def _build_frame(self, tick: int) -> str:
        try:
            run_store = RunStore.load(run_id=self.run_id, runtime_root=self.runtime_root)
            return render_live_progress_line(
                run_store=run_store,
                selection=self.selection,
                tick=tick,
            )
        except Exception as error:  # noqa: BLE001
            return (
                f"{TERMINAL_SPINNER_FRAMES[tick % len(TERMINAL_SPINNER_FRAMES)]} "
                f"路由层初始化中 | 状态读取失败: {error}"
            )


def run_routing_stage(argv: Sequence[str] | None = None) -> RoutingStageResult:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "resume_run", ""):
        parser.error("A01 不支持 --resume-run，请改用 B01_terminal_interaction.py")
    request: CliRequest | None = None
    while True:
        request = collect_cli_request(args, initial_request=request, start_step=6 if request is not None else 0)
        config, selection = prepare_batch_request(request)

        if not selection.should_run:
            if selection.project_missing_files and not project_has_business_files(selection.project_dir):
                message(EMPTY_PROJECT_ROUTING_SKIP_MESSAGE)
            else:
                message("当前项目路由层已完备，跳过路由初始化。")
            message(render_requirements_stage_placeholder([]))
            return RoutingStageResult(
                project_dir=request.project_dir,
                skipped=True,
                exit_code=0,
                cleanup_result=RoutingCleanupResult(),
            )

        preflight_summary = render_preflight_summary(request, config, selection)
        force_confirmation = bool(selection.project_missing_files)
        try:
            with _routing_prompt_step(7, allow_back=not request.auto_confirm):
                confirmed = request.auto_confirm or prompt_confirmation(preflight_summary, force_yes=force_confirmation)
        except PromptBackRequested:
            continue
        if not confirmed:
            message("已取消执行。")
            return RoutingStageResult(
                project_dir=request.project_dir,
                skipped=True,
                exit_code=0,
                cleanup_result=RoutingCleanupResult(),
            )
        break

    progress_monitor: TerminalProgressMonitor | None = None

    def handle_workers_prepared(run_store: RunStore, live_workers, immediate_results) -> None:
        nonlocal progress_monitor
        message(
            render_runtime_start_summary(
                run_store=run_store,
                live_workers=live_workers,
                immediate_results=immediate_results,
            ),
            flush=True,
        )
        if live_workers:
            progress_monitor = TerminalProgressMonitor(
                run_id=run_store.manifest.run_id,
                runtime_root=Path(run_store.manifest.runtime_dir).parent,
                selection=selection,
            )
            progress_monitor.start()

    try:
        batch_result = run_batch_initialization(
            selection=selection,
            config=config,
            max_refine_rounds=request.max_refine_rounds,
            on_workers_prepared=handle_workers_prepared,
        )
    finally:
        if progress_monitor is not None:
            progress_monitor.stop()
    message(format_batch_summary(batch_result))
    run_store = RunStore.load(
        run_id=batch_result.run_id,
        runtime_root=Path(batch_result.runtime_dir).parent,
    )
    killed_sessions = kill_run_tmux_sessions(run_store=run_store)
    cleanup_result = cleanup_routing_stage_artifacts(batch_result=batch_result)
    exit_code = determine_exit_code(batch_result)
    if exit_code == 0:
        message(render_requirements_stage_placeholder(killed_sessions, cleanup_result))
    else:
        message(render_routing_failure_summary(batch_result, killed_sessions, cleanup_result))
    return RoutingStageResult(
        project_dir=request.project_dir,
        skipped=False,
        exit_code=exit_code,
        batch_result=batch_result,
        killed_sessions=tuple(killed_sessions),
        cleanup_result=cleanup_result,
    )


def main(argv: Sequence[str] | None = None) -> int:
    redirected, launch = maybe_launch_tui(argv, route="routing", action="stage.a01.start")
    if redirected:
        return int(launch)
    return run_routing_stage(list(launch)).exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        cleaned_sessions = cleanup_registered_tmux_workers(reason="keyboard_interrupt")
        if cleaned_sessions:
            message(f"\n已清理 tmux 会话: {', '.join(cleaned_sessions)}")
        raise SystemExit(130)
