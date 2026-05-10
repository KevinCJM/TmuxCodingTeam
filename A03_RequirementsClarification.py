# -*- encoding: utf-8 -*-
"""
@File: A03_RequirementsClarification.py
@Modify Time: 2026/4/17
@Author: Kevin-Chen
@Descriptions: 需求澄清阶段
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

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
from Prompt_03_RequirementsClarification import (
    REQUIREMENTS_STATUS_OK,
    REQUIREMENTS_STATUS_SCHEMA_VERSION,
    fintech_ba,
    hitl_bck,
    requirements_understand,
    resume_requirements_understand,
)
from T01_tools import get_markdown_content
from T02_tmux_agents import (
    DEFAULT_COMMAND_TIMEOUT_SEC,
    AgentRunConfig,
    TmuxBatchWorker,
    cleanup_registered_tmux_workers,
    is_agent_ready_timeout_error,
    is_provider_auth_error,
    is_worker_death_error,
    worker_state_is_prelaunch_active,
)
from T05_hitl_runtime import HitlPromptContext, run_hitl_agent_loop
from tmux_core.stage_kernel.shared_review import is_agent_config_error
from tmux_core.stage_kernel.requirement_concurrency import requirement_concurrency_lock
from tmux_core.stage_kernel.stage_audit import (
    StageAuditRunContext,
    append_stage_audit_record,
    begin_stage_audit_run,
    record_before_cleanup,
)
from T08_pre_development import mark_requirement_clarification_completed
from T09_terminal_ops import (
    PROMPT_BACK_VALUE,
    PromptBackRequested,
    SingleLineSpinnerMonitor,
    TERMINAL_SPINNER_FRAMES,
    maybe_launch_tui,
    message,
    prompt_metadata,
    terminal_ui_is_interactive,
    prompt_yes_no as terminal_prompt_yes_no,
)
from T12_requirements_common import (
    DEFAULT_REQUIREMENTS_CLARIFICATION_EFFORT,
    DEFAULT_REQUIREMENTS_CLARIFICATION_MODEL,
    DEFAULT_REQUIREMENTS_CLARIFICATION_VENDOR,
    RequirementsAnalystHandoff,
    build_requirements_clarification_paths,
    clear_requirements_human_exchange_file,
    cleanup_runtime_paths,
    ensure_requirements_hitl_record_file,
    prompt_project_dir,
    prompt_requirement_name_selection,
    prompt_with_default,
    resolve_existing_directory,
)


REQUIREMENTS_CLARIFICATION_TURN_PHASE = "requirements_clarification"
REQUIREMENTS_CLARIFICATION_STAGE_NAME = "requirements_clarification"
REQUIREMENTS_RUNTIME_ROOT_NAME = ".requirements_clarification_runtime"
PLACEHOLDER_NEXT_STEP = "下一步进入需求评审阶段（待接入）"
AUTO_HITL_RESPONSE_TEXT = """自动澄清（--yes）：
本轮为非交互全流程测试，不再等待人工补充。请基于原始需求、已有澄清记录和验收示例做最小、保守、可测试的默认决策；把所有默认假设写入澄清记录和需求澄清文档，并继续完成需求澄清。除非原始需求完全无法实现，否则不要再次发起 HITL。"""


@dataclass(frozen=True)
class RequirementsClarificationStageResult:
    project_dir: str
    requirement_name: str
    requirements_clear_path: str
    cleanup_paths: tuple[str, ...] = ()
    ba_handoff: RequirementsAnalystHandoff | None = None


RequirementsStageResult = RequirementsClarificationStageResult
RequirementsAnalysisResult = RequirementsClarificationStageResult


@dataclass(frozen=True)
class RequirementsClarificationAgentSelection:
    vendor: str
    model: str
    reasoning_effort: str
    proxy_url: str


RequirementsAnalysisAgentSelection = RequirementsClarificationAgentSelection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="需求澄清阶段")
    parser.add_argument("--project-dir", help="项目目录")
    parser.add_argument("--requirement-name", help="需求名称")
    parser.add_argument("--allow-previous-stage-back", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--vendor", help="需求澄清阶段厂商: codex|claude|gemini|opencode")
    parser.add_argument("--model", help="需求澄清阶段模型名称")
    parser.add_argument("--effort", help="需求澄清阶段推理强度")
    parser.add_argument("--proxy-url", default="", help="需求澄清阶段代理端口或完整代理 URL")
    parser.add_argument("--overwrite", action="store_true", help="存在需求澄清时不直接复用，而是重新执行核验")
    parser.add_argument("--yes", action="store_true", help="跳过非关键确认")
    parser.add_argument("--no-tui", action="store_true", help="显式禁用 OpenTUI")
    parser.add_argument("--legacy-cli", action="store_true", help="使用旧版 Python CLI，不跳转 OpenTUI")
    return parser


def prompt_proxy_url(default: str = "") -> str:
    return prompt_with_default("输入代理端口或完整代理 URL（可留空）", default, allow_empty=True)


def _clarification_prompt_step(step_index: int, *, allow_back: bool):
    return prompt_metadata(
        allow_back=allow_back,
        back_value=PROMPT_BACK_VALUE,
        stage_key="requirements_clarification",
        stage_step_index=step_index,
    )


def prompt_yes_no(prompt_text: str, default: bool = False) -> bool:
    return terminal_prompt_yes_no(prompt_text, default)


def stdin_is_interactive() -> bool:
    return terminal_ui_is_interactive()


def collect_auto_requirements_hitl_response(question_path: str | Path, hitl_round: int = 0) -> str:
    question_file = Path(question_path).expanduser().resolve()
    question_text = get_markdown_content(question_file).strip()
    round_label = f"第 {hitl_round} 轮" if hitl_round else "本轮"
    message(f"A03 HITL {round_label} 已由 --yes 自动回复，继续非交互流程。")
    if question_text:
        return f"{AUTO_HITL_RESPONSE_TEXT}\n\n原问题文档摘要：\n{question_text}"
    return AUTO_HITL_RESPONSE_TEXT


def has_existing_requirements_clarification(project_dir: str | Path, requirement_name: str) -> bool:
    _, requirements_clear_path, _, _ = build_requirements_clarification_paths(project_dir, requirement_name)
    return bool(get_markdown_content(requirements_clear_path).strip())


def should_reuse_existing_requirements_clarification(
        project_dir: str | Path,
        requirement_name: str,
        *,
        overwrite: bool,
        interactive: bool,
        allow_back: bool = False,
) -> bool:
    _, requirements_clear_path, _, _ = build_requirements_clarification_paths(project_dir, requirement_name)
    if not get_markdown_content(requirements_clear_path).strip():
        return False
    if not interactive:
        return not overwrite
    message(f"检测项目内已有需求澄清: {requirements_clear_path.name}")
    with _clarification_prompt_step(0, allow_back=allow_back):
        return prompt_yes_no("是否直接复用已有的需求澄清并跳入需求评审阶段", True)


def reuse_existing_requirements_clarification(project_dir: str | Path, requirement_name: str) -> RequirementsClarificationStageResult:
    _, requirements_clear_path, _, _ = build_requirements_clarification_paths(project_dir, requirement_name)
    if not get_markdown_content(requirements_clear_path).strip():
        raise RuntimeError(f"缺少可复用的需求澄清文档: {requirements_clear_path}")
    ensure_requirements_hitl_record_file(project_dir, requirement_name)
    return RequirementsClarificationStageResult(
        project_dir=str(resolve_existing_directory(project_dir)),
        requirement_name=requirement_name,
        requirements_clear_path=str(requirements_clear_path.resolve()),
    )


def render_agent_boot_progress_line(*, tick: int) -> str:
    spinner = TERMINAL_SPINNER_FRAMES[tick % len(TERMINAL_SPINNER_FRAMES)]
    return f"{spinner} 智能体启动中..."


def render_requirements_clarification_tmux_start_summary(worker: TmuxBatchWorker) -> str:
    return "\n".join(
        [
            "需求澄清智能体已启动",
            f"runtime_dir: {worker.runtime_dir}",
            f"session_name: {worker.session_name}",
            "可使用以下命令进入会话:",
            f"  tmux attach -t {worker.session_name}",
        ]
    )


def render_requirements_clarification_progress_line(*, worker: TmuxBatchWorker, requirement_name: str, tick: int) -> str:
    try:
        state = worker.read_state()
    except Exception:  # noqa: BLE001
        state = {}
    workflow_stage = str(state.get("workflow_stage") or state.get("current_turn_phase") or "starting").strip() or "starting"
    agent_state = str(state.get("agent_state", "")).strip().upper()
    if worker_state_is_prelaunch_active(state):
        agent_state = "STARTING"
    elif agent_state not in {"DEAD", "STARTING", "READY", "BUSY"}:
        provider_phase = str(state.get("provider_phase", "")).strip().lower()
        wrapper_state = str(state.get("wrapper_state", "")).strip().upper()
        if wrapper_state == "READY" or provider_phase in {"waiting_input", "idle_ready", "completed_response"}:
            agent_state = "READY"
        elif provider_phase:
            agent_state = "BUSY"
        else:
            agent_state = "DEAD"
    health_status = str(state.get("health_status", "unknown")).strip() or "unknown"
    note = str(state.get("note", "")).strip() or workflow_stage
    status = str(state.get("status", "running")).strip() or "running"
    spinner = TERMINAL_SPINNER_FRAMES[tick % len(TERMINAL_SPINNER_FRAMES)]
    return (
        f"{spinner} 需求澄清中"
        f" | {requirement_name}:{status}/{agent_state}"
        f" | health={health_status}"
        f" | {note}"
    )


def collect_requirements_clarification_agent_selection(args: argparse.Namespace) -> RequirementsClarificationAgentSelection:
    interactive = stdin_is_interactive()
    auto_confirm = bool(getattr(args, "yes", False))
    vendor_value = str(getattr(args, "vendor", "") or "").strip()
    proxy_url = str(getattr(args, "proxy_url", "") or "").strip()
    allow_previous_stage_back = bool(getattr(args, "allow_previous_stage_back", False))
    try:
        model_value = str(getattr(args, "model", "") or "").strip()
        effort_value = str(getattr(args, "effort", "") or "").strip()
        if interactive:
            if auto_confirm and vendor_value and model_value and effort_value:
                vendor = normalize_vendor_choice(vendor_value)
                model = normalize_model_choice(vendor, model_value)
                reasoning_effort = normalize_effort_choice(vendor, model, effort_value)
            else:
                vendor = normalize_vendor_choice(vendor_value) if vendor_value else DEFAULT_REQUIREMENTS_CLARIFICATION_VENDOR
                model = model_value
                reasoning_effort = effort_value or DEFAULT_REQUIREMENTS_CLARIFICATION_EFFORT
                first_prompt_step = 0 if not vendor_value else (1 if not model_value else (2 if not effort_value else 3))
                step = first_prompt_step
                while step < 4:
                    try:
                        if step == 0:
                            with _clarification_prompt_step(
                                0,
                                allow_back=(step > first_prompt_step) or (step == first_prompt_step and allow_previous_stage_back),
                            ):
                                vendor = prompt_vendor(vendor)
                            if model:
                                try:
                                    normalize_model_choice(vendor, model)
                                except ValueError:
                                    model = ""
                            step = 1
                            continue
                        if step == 1:
                            if model_value and step < first_prompt_step:
                                model = normalize_model_choice(vendor, model_value)
                            else:
                                with _clarification_prompt_step(
                                    1,
                                    allow_back=(step > first_prompt_step) or (step == first_prompt_step and allow_previous_stage_back),
                                ):
                                    model = prompt_model(vendor, model or get_default_model_for_vendor(vendor))
                            step = 2
                            continue
                        if step == 2:
                            if effort_value and step < first_prompt_step:
                                reasoning_effort = normalize_effort_choice(vendor, model, effort_value)
                            else:
                                with _clarification_prompt_step(
                                    2,
                                    allow_back=(step > first_prompt_step) or (step == first_prompt_step and allow_previous_stage_back),
                                ):
                                    reasoning_effort = prompt_effort(vendor, model, reasoning_effort or DEFAULT_REQUIREMENTS_CLARIFICATION_EFFORT)
                            step = 3
                            continue
                        if step == 3:
                            with _clarification_prompt_step(
                                3,
                                allow_back=(step > first_prompt_step) or (step == first_prompt_step and allow_previous_stage_back),
                            ):
                                proxy_url = prompt_proxy_url(proxy_url)
                            step = 4
                            continue
                    except PromptBackRequested:
                        if step == first_prompt_step:
                            if allow_previous_stage_back:
                                raise
                            continue
                        step = max(first_prompt_step, step - 1)
        else:
            if not vendor_value:
                raise RuntimeError("需求澄清阶段需要选择厂商；非交互模式请传入 --vendor、--model、--effort。")
            vendor = normalize_vendor_choice(vendor_value)
            if not model_value:
                raise RuntimeError("需求澄清阶段需要选择模型；非交互模式请传入 --vendor、--model、--effort。")
            model = normalize_model_choice(vendor, model_value)
            if not effort_value:
                raise RuntimeError("需求澄清阶段需要选择推理强度；非交互模式请传入 --vendor、--model、--effort。")
            reasoning_effort = normalize_effort_choice(vendor, model, effort_value)
    except Exception as error:  # noqa: BLE001
        if not interactive or not is_agent_config_error(error):
            raise
        message(f"需求分析师模型配置不可用: {error}\n请重新选择厂商、模型和推理强度。")
        vendor = prompt_vendor(DEFAULT_REQUIREMENTS_CLARIFICATION_VENDOR)
        model = prompt_model(vendor, get_default_model_for_vendor(vendor))
        reasoning_effort = prompt_effort(vendor, model, DEFAULT_REQUIREMENTS_CLARIFICATION_EFFORT)
        proxy_url = prompt_proxy_url(proxy_url)

    vendor = normalize_vendor_choice(vendor)
    model = normalize_model_choice(vendor, model)
    reasoning_effort = normalize_effort_choice(vendor, model, reasoning_effort)
    return RequirementsClarificationAgentSelection(
        vendor=vendor,
        model=model,
        reasoning_effort=reasoning_effort,
        proxy_url=proxy_url,
    )


def render_requirements_clarification_stage_start(selection: RequirementsClarificationAgentSelection) -> str:
    return "\n".join(
        [
            "进入需求澄清阶段（需求分析师）",
            f"vendor: {selection.vendor}",
            f"model: {selection.model}",
            f"reasoning_effort: {selection.reasoning_effort}",
            f"proxy_url: {selection.proxy_url or '(none)'}",
        ]
    )


def prompt_recreate_requirements_clarification_agent(
        *,
        reason_text: str,
        requirement_name: str,
        current_vendor: str,
        current_model: str,
        current_reasoning_effort: str,
        current_proxy_url: str,
        force_model_change: bool,
) -> RequirementsClarificationAgentSelection | None:
    if not stdin_is_interactive():
        return None
    message(reason_text)
    if not prompt_yes_no("是否创建新的需求分析师继续当前阶段", True):
        return None
    while True:
        vendor = prompt_vendor(current_vendor)
        model = prompt_model(vendor, current_model if vendor == current_vendor else get_default_model_for_vendor(vendor))
        reasoning_effort = prompt_effort(vendor, model, current_reasoning_effort)
        proxy_url = prompt_proxy_url(current_proxy_url)
        if (not force_model_change) or vendor != current_vendor or model != current_model:
            selection = RequirementsClarificationAgentSelection(
                vendor=vendor,
                model=model,
                reasoning_effort=reasoning_effort,
                proxy_url=proxy_url,
            )
            message(render_requirements_clarification_stage_start(selection))
            return selection
        message("需要更换模型，请选择与当前不同的厂商或模型。")


def load_json_object(file_path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON 文件必须是对象")
    return payload


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


def run_requirements_clarification(
        project_dir: str | Path,
        requirement_name: str,
        *,
        vendor: str = DEFAULT_REQUIREMENTS_CLARIFICATION_VENDOR,
        model: str = DEFAULT_REQUIREMENTS_CLARIFICATION_MODEL,
        reasoning_effort: str = DEFAULT_REQUIREMENTS_CLARIFICATION_EFFORT,
        proxy_url: str = "",
        resume_existing: bool = False,
        preserve_ba_worker: bool = False,
        human_input_provider: Callable[..., str] | None = None,
        audit_context: StageAuditRunContext | None = None,
) -> RequirementsClarificationStageResult:
    project_root = resolve_existing_directory(project_dir)
    runtime_root = project_root / REQUIREMENTS_RUNTIME_ROOT_NAME
    original_requirement_path, requirements_clear_path, ask_human_path, hitl_record_path = (
        build_requirements_clarification_paths(project_root, requirement_name)
    )
    hitl_record_path = ensure_requirements_hitl_record_file(project_root, requirement_name)
    if not original_requirement_path.exists() or not original_requirement_path.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"缺少原始需求文档: {original_requirement_path}")

    current_vendor = vendor
    current_model = model
    current_reasoning_effort = reasoning_effort
    current_proxy_url = proxy_url
    current_resume_existing = bool(resume_existing)
    keep_worker_alive = False
    worker: TmuxBatchWorker | None = None
    runtime_dir = runtime_root

    progress_monitor = SingleLineSpinnerMonitor(
        frame_builder=lambda tick: render_requirements_clarification_progress_line(
            worker=worker,
            requirement_name=requirement_name,
            tick=tick,
        ),
        interval_sec=0.2,
    )
    boot_progress_monitor = SingleLineSpinnerMonitor(
        frame_builder=lambda tick: render_agent_boot_progress_line(tick=tick),
        interval_sec=0.2,
    )
    boot_progress_active = False
    progress_active = False

    def start_boot_progress() -> None:
        nonlocal boot_progress_active
        if boot_progress_active:
            return
        boot_progress_monitor.start()
        boot_progress_active = True

    def stop_boot_progress() -> None:
        nonlocal boot_progress_active
        if not boot_progress_active:
            return
        boot_progress_monitor.stop()
        boot_progress_active = False

    def start_progress() -> None:
        nonlocal progress_active
        if progress_active:
            return
        stop_boot_progress()
        progress_monitor.start()
        progress_active = True

    def stop_progress() -> None:
        nonlocal progress_active
        if not progress_active:
            return
        progress_monitor.stop()
        progress_active = False

    def handle_worker_started(live_worker: TmuxBatchWorker) -> None:
        stop_boot_progress()
        message(render_requirements_clarification_tmux_start_summary(live_worker))

    try:
        while True:
            try:
                worker = TmuxBatchWorker(
                    worker_id="requirements-analyst",
                    work_dir=project_root,
                    config=AgentRunConfig(
                        vendor=current_vendor,
                        model=current_model,
                        reasoning_effort=current_reasoning_effort,
                        proxy_url=current_proxy_url,
                    ),
                    runtime_root=runtime_root,
                )
            except Exception as error:  # noqa: BLE001
                stop_progress()
                stop_boot_progress()
                if not is_agent_config_error(error):
                    raise
                selection = prompt_recreate_requirements_clarification_agent(
                    reason_text=f"需求分析师模型配置不可用: {error}\n请重新选择模型后继续当前阶段。",
                    requirement_name=requirement_name,
                    current_vendor=current_vendor,
                    current_model=current_model,
                    current_reasoning_effort=current_reasoning_effort,
                    current_proxy_url=current_proxy_url,
                    force_model_change=False,
                )
                if selection is not None:
                    current_vendor = selection.vendor
                    current_model = selection.model
                    current_reasoning_effort = selection.reasoning_effort
                    current_proxy_url = selection.proxy_url
                    current_resume_existing = current_resume_existing or bool(get_markdown_content(requirements_clear_path).strip())
                    keep_worker_alive = False
                    continue
                raise RuntimeError("需求分析师模型配置不可用，且用户未重新选择模型") from error
            runtime_dir = worker.runtime_dir
            stage_status_path = runtime_dir / "requirements_clarification_status.json"
            turns_root = runtime_dir / "turns"

            def initial_prompt_builder(context: HitlPromptContext) -> str:
                prompt_builder = resume_requirements_understand if current_resume_existing else requirements_understand
                return prompt_builder(
                    fintech_ba,
                    original_requirement_md=str(original_requirement_path.resolve()),
                    requirements_clear_md=str(Path(context.output_path).resolve()),
                    ask_human_md=str(Path(context.question_path).resolve()),
                    hitl_record_md=str(Path(context.record_path).resolve()),
                )

            def hitl_prompt_builder(human_msg: str, context: HitlPromptContext) -> str:
                return hitl_bck(
                    human_msg,
                    original_requirement_md=str(original_requirement_path.resolve()),
                    hitl_record_md=str(Path(context.record_path).resolve()),
                    requirements_clear_md=str(Path(context.output_path).resolve()),
                    ask_human_md=str(Path(context.question_path).resolve()),
                )

            def audit_before_question_clear(context: HitlPromptContext) -> None:
                record_before_cleanup(
                    audit_context,
                    {"ask_human": context.question_path},
                    metadata={"trigger": "hitl_question_clear"},
                    hitl_round_index=context.hitl_round,
                )

            def audit_hitl_question(context: HitlPromptContext, _decision: object) -> None:
                append_stage_audit_record(
                    audit_context,
                    event_type="hitl_question",
                    source_paths={"ask_human": context.question_path},
                    hitl_round_index=context.hitl_round,
                )

            def audit_hitl_answer(context: HitlPromptContext, human_msg: str, human_history_path: Path) -> None:
                source_paths: dict[str, str | Path | None] = {
                    "human_answer": "",
                    "hitl_record": context.record_path,
                }
                if human_history_path:
                    source_paths["runtime_human_answer"] = human_history_path
                append_stage_audit_record(
                    audit_context,
                    event_type="hitl_answer",
                    source_paths=source_paths,
                    metadata={
                        "human_answer_source": "runtime_payload",
                        "runtime_human_answer": str(Path(human_history_path).expanduser().resolve()) if human_history_path else "",
                    },
                    hitl_round_index=context.hitl_round,
                    snapshot_overrides={"human_answer": human_msg},
                )

            try:
                hitl_loop_kwargs: dict[str, object] = {}
                if human_input_provider is not None:
                    hitl_loop_kwargs["human_input_provider"] = human_input_provider
                loop_result = run_hitl_agent_loop(
                    worker=worker,
                    stage_name=REQUIREMENTS_CLARIFICATION_STAGE_NAME,
                    output_path=requirements_clear_path,
                    question_path=ask_human_path,
                    record_path=hitl_record_path,
                    stage_status_path=stage_status_path,
                    turns_root=turns_root,
                    initial_prompt_builder=initial_prompt_builder,
                    hitl_prompt_builder=hitl_prompt_builder,
                    label_prefix="requirements_clarification",
                    turn_phase=REQUIREMENTS_CLARIFICATION_TURN_PHASE,
                    on_worker_starting=lambda live_worker: start_boot_progress(),
                    on_worker_started=handle_worker_started,
                    on_agent_turn_started=lambda context, live_worker: start_progress(),
                    on_agent_turn_finished=lambda context, live_worker: stop_progress(),
                    on_before_question_clear=audit_before_question_clear,
                    on_hitl_question=audit_hitl_question,
                    on_hitl_answer=audit_hitl_answer,
                    timeout_sec=DEFAULT_COMMAND_TIMEOUT_SEC,
                    **hitl_loop_kwargs,
                )
                if str(loop_result.decision.payload.get("status", "")).strip() != REQUIREMENTS_STATUS_OK:
                    raise RuntimeError(loop_result.decision.summary or "需求澄清未完成闭环")
                if not requirements_clear_path.exists():
                    raise RuntimeError("需求澄清未生成需求澄清文档")
                if not requirements_clear_path.read_text(encoding="utf-8").strip():
                    raise RuntimeError("需求澄清文档为空")
                handoff = None
                cleanup_paths: tuple[str, ...] = (str(ask_human_path.resolve()),)
                if preserve_ba_worker:
                    keep_worker_alive = True
                    handoff = RequirementsAnalystHandoff(
                        worker=worker,
                        vendor=worker.config.vendor.value,
                        model=worker.config.model,
                        reasoning_effort=worker.config.reasoning_effort,
                        proxy_url=worker.config.proxy_url,
                    )
                else:
                    cleanup_paths = (
                        str(ask_human_path.resolve()),
                        str(Path(runtime_dir).expanduser().resolve()),
                        str(Path(runtime_root).expanduser().resolve()),
                    )
                append_stage_audit_record(
                    audit_context,
                    event_type="clarification_updated",
                    source_paths={"requirements_clear": requirements_clear_path},
                )
                return RequirementsClarificationStageResult(
                    project_dir=str(project_root),
                    requirement_name=requirement_name,
                    requirements_clear_path=str(requirements_clear_path.resolve()),
                    cleanup_paths=cleanup_paths,
                    ba_handoff=handoff,
                )
            except Exception as error:  # noqa: BLE001
                stop_progress()
                stop_boot_progress()
                auth_error = is_provider_auth_error(error) or worker_has_provider_auth_error(worker)
                ready_timeout_error = is_agent_ready_timeout_error(error)
                if auth_error:
                    if not keep_worker_alive:
                        try:
                            worker.request_kill()
                        except Exception:
                            pass
                    selection = prompt_recreate_requirements_clarification_agent(
                        reason_text=f"检测到需求分析师仍在 agent 界面，但模型认证已失效: {requirement_name}\n需要更换模型后继续当前阶段。",
                        requirement_name=requirement_name,
                        current_vendor=current_vendor,
                        current_model=current_model,
                        current_reasoning_effort=current_reasoning_effort,
                        current_proxy_url=current_proxy_url,
                        force_model_change=True,
                    )
                    if selection is not None:
                        current_vendor = selection.vendor
                        current_model = selection.model
                        current_reasoning_effort = selection.reasoning_effort
                        current_proxy_url = selection.proxy_url
                        current_resume_existing = current_resume_existing or bool(get_markdown_content(requirements_clear_path).strip())
                        keep_worker_alive = False
                        continue
                    raise RuntimeError("需求分析师认证已失效，且用户未更换模型") from error
                if ready_timeout_error:
                    if not keep_worker_alive:
                        try:
                            worker.request_kill()
                        except Exception:
                            pass
                    selection = prompt_recreate_requirements_clarification_agent(
                        reason_text=f"需求分析师启动超时，未能进入可输入状态: {requirement_name}\n请重新选择模型后继续当前阶段。",
                        requirement_name=requirement_name,
                        current_vendor=current_vendor,
                        current_model=current_model,
                        current_reasoning_effort=current_reasoning_effort,
                        current_proxy_url=current_proxy_url,
                        force_model_change=True,
                    )
                    if selection is not None:
                        current_vendor = selection.vendor
                        current_model = selection.model
                        current_reasoning_effort = selection.reasoning_effort
                        current_proxy_url = selection.proxy_url
                        current_resume_existing = current_resume_existing or bool(get_markdown_content(requirements_clear_path).strip())
                        keep_worker_alive = False
                        continue
                    raise RuntimeError("需求分析师启动超时，且用户未更换模型") from error
                if is_worker_death_error(error):
                    if not keep_worker_alive:
                        try:
                            worker.request_kill()
                        except Exception:
                            pass
                    selection = prompt_recreate_requirements_clarification_agent(
                        reason_text=f"检测到需求分析师已死亡: {requirement_name}\n需要更换模型后继续当前阶段。",
                        requirement_name=requirement_name,
                        current_vendor=current_vendor,
                        current_model=current_model,
                        current_reasoning_effort=current_reasoning_effort,
                        current_proxy_url=current_proxy_url,
                        force_model_change=True,
                    )
                    if selection is not None:
                        current_vendor = selection.vendor
                        current_model = selection.model
                        current_reasoning_effort = selection.reasoning_effort
                        current_proxy_url = selection.proxy_url
                        current_resume_existing = current_resume_existing or bool(get_markdown_content(requirements_clear_path).strip())
                        keep_worker_alive = False
                        continue
                raise
    finally:
        stop_progress()
        stop_boot_progress()
        if worker is not None and not keep_worker_alive:
            try:
                worker.request_kill()
            except Exception:
                pass


def collect_request(args: argparse.Namespace) -> tuple[str, str]:
    project_dir = (
        str(resolve_existing_directory(args.project_dir))
        if args.project_dir
        else prompt_project_dir("")
    )
    if args.requirement_name:
        requirement_name = str(args.requirement_name).strip()
    else:
        requirement_name = prompt_requirement_name_selection(project_dir, "").requirement_name
    clear_requirements_human_exchange_file(project_dir, requirement_name)
    return project_dir, requirement_name


def run_requirements_clarification_stage(
        argv: Sequence[str] | None = None,
        *,
        preserve_ba_worker: bool = False,
) -> RequirementsClarificationStageResult:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_dir, requirement_name = collect_request(args)
    human_input_provider = collect_auto_requirements_hitl_response if bool(args.yes) else None
    lock_context = requirement_concurrency_lock(
        project_dir,
        requirement_name,
        action="stage.a03.start",
    )
    lock_context.__enter__()
    audit_context: StageAuditRunContext | None = None
    try:
        audit_context = begin_stage_audit_run(
            project_dir,
            requirement_name,
            "A03",
            metadata={
                "trigger": "run_requirements_clarification_stage",
                "argv": list(argv or []),
                "args": vars(args),
            },
        )
        _, requirements_clear_path, ask_human_path, hitl_record_path = build_requirements_clarification_paths(
            project_dir,
            requirement_name,
        )
        record_before_cleanup(
            audit_context,
            {"ask_human": ask_human_path},
            metadata={"trigger": "clear_requirements_human_exchange_file"},
        )
        clear_requirements_human_exchange_file(project_dir, requirement_name)
        if has_existing_requirements_clarification(project_dir, requirement_name):
            if should_reuse_existing_requirements_clarification(
                    project_dir,
                    requirement_name,
                    overwrite=bool(args.overwrite),
                    interactive=stdin_is_interactive(),
                    allow_back=bool(getattr(args, "allow_previous_stage_back", False)),
            ):
                message("复用已有的需求澄清，直接进入需求评审阶段")
                result = reuse_existing_requirements_clarification(project_dir, requirement_name)
                append_stage_audit_record(
                    audit_context,
                    event_type="clarification_updated",
                    source_paths={"requirements_clear": requirements_clear_path},
                )
                mark_requirement_clarification_completed(project_dir, requirement_name)
                append_stage_audit_record(
                    audit_context,
                    event_type="stage_passed",
                    source_paths={
                        "requirements_clear": requirements_clear_path,
                        "hitl_record": hitl_record_path,
                    },
                )
                return result
            message("不直接复用已有需求澄清，将启动需求分析师基于现有澄清继续核验")
            selection = collect_requirements_clarification_agent_selection(args)
            message(render_requirements_clarification_stage_start(selection))
            result = run_requirements_clarification(
                project_dir,
                requirement_name,
                vendor=selection.vendor,
                model=selection.model,
                reasoning_effort=selection.reasoning_effort,
                proxy_url=selection.proxy_url,
                resume_existing=True,
                preserve_ba_worker=preserve_ba_worker,
                human_input_provider=human_input_provider,
                audit_context=audit_context,
            )
        else:
            message("执行摘要: 未检测到可复用的需求澄清，需要启动需求分析师智能体执行需求澄清；请为需求分析师选择厂商、模型、推理强度、代理端口。")
            selection = collect_requirements_clarification_agent_selection(args)
            message(render_requirements_clarification_stage_start(selection))
            result = run_requirements_clarification(
                project_dir,
                requirement_name,
                vendor=selection.vendor,
                model=selection.model,
                reasoning_effort=selection.reasoning_effort,
                proxy_url=selection.proxy_url,
                resume_existing=False,
                preserve_ba_worker=preserve_ba_worker,
                human_input_provider=human_input_provider,
                audit_context=audit_context,
            )
        mark_requirement_clarification_completed(project_dir, requirement_name)
        append_stage_audit_record(
            audit_context,
            event_type="stage_passed",
            source_paths={
                "requirements_clear": requirements_clear_path,
                "hitl_record": hitl_record_path,
            },
        )
        cleanup_paths = result.cleanup_paths
        if cleanup_paths:
            cleanup_runtime_paths(cleanup_paths)
            cleanup_paths = ()
        return RequirementsClarificationStageResult(
            project_dir=result.project_dir,
            requirement_name=result.requirement_name,
            requirements_clear_path=result.requirements_clear_path,
            cleanup_paths=cleanup_paths,
            ba_handoff=result.ba_handoff,
        )
    except Exception as error:  # noqa: BLE001
        append_stage_audit_record(
            audit_context,
            event_type="stage_failed",
            source_paths={
                "requirements_clear": requirements_clear_path if "requirements_clear_path" in locals() else "",
                "hitl_record": hitl_record_path if "hitl_record_path" in locals() else "",
            },
            metadata={"error": str(error)},
        )
        raise
    finally:
        lock_context.__exit__(None, None, None)


def main(argv: Sequence[str] | None = None) -> int:
    redirected, launch = maybe_launch_tui(argv, route="requirements", action="stage.a03.start")
    if redirected:
        return int(launch)
    try:
        result = run_requirements_clarification_stage(list(launch), preserve_ba_worker=False)
    except Exception as error:  # noqa: BLE001
        message(error)
        return 1

    message("需求澄清完成")
    message(result.requirements_clear_path)
    message(PLACEHOLDER_NEXT_STEP)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        cleaned_sessions = cleanup_registered_tmux_workers(reason="keyboard_interrupt")
        if cleaned_sessions:
            message(f"\n已清理 tmux 会话: {', '.join(cleaned_sessions)}")
        raise SystemExit(130)
