# -*- encoding: utf-8 -*-
"""
@File: A07_Development.py
@Modify Time: 2026/4/21
@Author: Kevin-Chen
@Descriptions: 任务开发阶段
"""

from __future__ import annotations

import argparse
import json
import shutil
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from A01_Routing_LayerPlanning import (
    DEFAULT_MODEL_BY_VENDOR,
    normalize_effort_choice,
    normalize_model_choice,
    normalize_vendor_choice,
)
from tmux_core.prompt_contracts.common import check_develop_job, check_reviewer_job
from tmux_core.prompt_contracts.development import (
    fintech_developer_role,
    human_reply,
    init_code_reviewer,
    init_developer,
    refine_code,
    re_review_code,
    reviewer_review_code,
    start_develop,
)
from tmux_core.runtime.contracts import (
    TaskResultContract,
    TurnFileContract,
    TurnFileResult,
    normalize_review_status_payload,
)
from tmux_core.runtime.hitl import build_prefixed_sha256
from tmux_core.runtime.tmux_runtime import (
    DEFAULT_COMMAND_TIMEOUT_SEC,
    LaunchCoordinator,
    TmuxBatchWorker,
    Vendor,
    build_session_name,
    cleanup_registered_tmux_workers,
    is_agent_ready_timeout_error,
    is_task_result_contract_error,
    is_worker_death_error,
    is_turn_artifact_contract_error,
    is_provider_auth_error,
    list_registered_tmux_workers,
    list_tmux_session_names,
    list_occupied_tmux_session_names,
    try_resume_worker,
)
from tmux_core.stage_kernel.detailed_design import collect_ba_agent_selection
from tmux_core.stage_kernel.reviewer_orchestration import (
    repair_reviewer_round_outputs,
    run_parallel_reviewer_round,
    shutdown_stage_workers,
)
from tmux_core.stage_kernel.prompt_turns import build_prompt_task_turn
from tmux_core.stage_kernel.death_orchestration import (
    ensure_active_reviewers,
    run_main_phase_with_death_handling,
    run_reviewer_phase_with_death_handling,
)
from tmux_core.stage_kernel.agent_intervention import (
    AGENT_INTERVENTION_RECREATE,
    AGENT_INTERVENTION_WORKER_DEAD,
    request_worker_manual_intervention,
)
from tmux_core.stage_kernel.requirement_concurrency import requirement_concurrency_lock
from tmux_core.stage_kernel.runtime_scope_cleanup import cleanup_runtime_dirs_by_scope
from tmux_core.stage_kernel.stage_audit import (
    StageAuditRunContext,
    append_stage_audit_record,
    begin_stage_audit_run,
    record_before_cleanup,
)
from tmux_core.stage_kernel.shared_review import (
    AGENT_READY_TIMEOUT_RETRY,
    AGENT_READY_TIMEOUT_SKIP,
    MAX_REVIEWER_REPAIR_ATTEMPTS,
    ReviewLimitHitlConfig,
    build_reviewer_failure_reconfiguration_reason,
    carry_reviewer_failure_state,
    describe_reviewer_failure_reason,
    ReviewRoundPolicy,
    ReviewAgentHandoff,
    ReviewAgentSelection,
    ReviewStageProgress,
    ReviewerRuntime,
    collect_auto_review_limit_hitl_response,
    ensure_empty_file,
    ensure_review_artifacts,
    is_recoverable_startup_failure,
    mark_worker_awaiting_reconfiguration,
    note_reviewer_failure,
    parse_review_max_rounds,
    prompt_agent_ready_timeout_recovery,
    prompt_positive_int,
    prompt_required_replacement_review_agent_selection,
    prompt_review_max_rounds,
    prompt_replacement_review_agent_selection,
    prompt_review_agent_selection,
    render_review_limit_force_hitl_prompt,
    render_review_limit_human_reply_prompt,
    render_review_agent_selection,
    render_tmux_start_summary,
    resolve_agent_run_config_with_recovery,
    resolve_stage_agent_config,
    reviewer_requires_manual_model_reconfiguration,
    run_review_limit_hitl_cycle,
    worker_has_provider_auth_error,
    worker_has_provider_runtime_error,
)
from tmux_core.stage_kernel.turn_output_goals import (
    CompletionTurnGoal,
    OutcomeGoal,
    RepairPromptContext,
    TaskTurnGoal,
    run_completion_turn_with_repair,
    run_task_result_turn_with_repair,
)
from tmux_core.stage_kernel.task_split import run_task_split_stage
from T01_tools import (
    get_first_false_task,
    get_markdown_content,
    is_file_empty,
    is_task_progress_json,
    task_done,
)
from T08_pre_development import ensure_pre_development_task_record
from T09_terminal_ops import (
    PROMPT_BACK_VALUE,
    BridgeTerminalUI,
    PromptBackRequested,
    collect_multiline_input,
    get_terminal_ui,
    maybe_launch_tui,
    message,
    prompt_metadata,
)
from T12_requirements_common import (
    DEFAULT_REQUIREMENTS_CLARIFICATION_EFFORT,
    DEFAULT_REQUIREMENTS_CLARIFICATION_MODEL,
    DEFAULT_REQUIREMENTS_CLARIFICATION_VENDOR,
    build_requirements_clarification_paths,
    prompt_project_dir,
    prompt_requirement_name_selection,
    prompt_with_default,
    sanitize_requirement_name,
    stdin_is_interactive,
)


DEVELOPMENT_RUNTIME_ROOT_NAME = ".development_runtime"
MAX_DEVELOPMENT_HITL_ROUNDS = 8
MAX_DEVELOPMENT_REVIEW_ROUNDS = 5
MAX_DEVELOPER_METADATA_REPAIR_ATTEMPTS = 2
DEFAULT_DEVELOPER_MAX_TURNS = 15
PLACEHOLDER_NEXT_STEP = "下一步进入测试阶段（待接入）"

DEFAULT_DEVELOPMENT_REVIEWER_PROMPTS: dict[str, str] = {
    "需求分析师": "你是代码评审中的需求分析师。重点检查任务边界、业务规则、人类澄清和任务描述是否被准确落地，禁止越界开发。",
    "测试工程师": "你是代码评审中的测试工程师。重点检查边界条件、异常处理、可测试性、回归风险和遗漏场景。",
    "审核员": "你是代码评审中的审核员。重点检查需求对齐、最小改动原则、契约兼容和非任务性改动。",
    "架构师": "你是代码评审中的架构师。重点检查系统边界、依赖影响、分层约束、接口兼容和隐藏副作用。",
}
DEFAULT_DEVELOPMENT_REVIEWER_ROLE_NAMES: tuple[str, ...] = tuple(DEFAULT_DEVELOPMENT_REVIEWER_PROMPTS.keys())


class DevelopmentStageLaunchCoordinator(LaunchCoordinator):
    @classmethod
    def current_stagger(cls, vendor: Vendor) -> float:  # noqa: ARG003
        return 0.0

    @classmethod
    def record_launch_result(cls, vendor: Vendor, *, success: bool) -> None:  # noqa: ARG003
        return None

    def startup_slot(self, vendor: Vendor):  # noqa: ARG002
        return nullcontext()


@dataclass(frozen=True)
class DevelopmentReviewerSpec:
    role_name: str
    role_prompt: str
    reviewer_key: str = ""


@dataclass(frozen=True)
class DeveloperPlan:
    selection: ReviewAgentSelection
    role_prompt: str


@dataclass(frozen=True)
class DeveloperRuntime:
    selection: ReviewAgentSelection
    worker: TmuxBatchWorker
    role_prompt: str


@dataclass(frozen=True)
class DevelopmentAgentHandoff:
    selection: ReviewAgentSelection
    role_prompt: str
    worker: TmuxBatchWorker


class _ReadyTimeoutSkipBudget:
    def __init__(self, reviewer_count: int) -> None:
        self._remaining = max(int(reviewer_count) - 1, 0)
        self._lock = threading.Lock()

    def reserve(self) -> bool:
        with self._lock:
            if self._remaining <= 0:
                return False
            self._remaining -= 1
            return True

    def release(self) -> None:
        with self._lock:
            self._remaining += 1


@dataclass(frozen=True)
class DevelopmentStageResult:
    project_dir: str
    requirement_name: str
    task_md_path: str
    task_json_path: str
    merged_review_path: str
    completed: bool
    cleanup_paths: tuple[str, ...] = ()
    developer_handoff: DevelopmentAgentHandoff | None = None
    reviewer_handoff: tuple[ReviewAgentHandoff, ...] = ()


@dataclass
class DeveloperTurnPolicy:
    max_turns: int | None
    turns_used: int = 0

    def record_turn(self) -> None:
        self.turns_used += 1

    def should_recreate_before_next_task(self) -> bool:
        return self.max_turns is not None and self.turns_used >= self.max_turns

    def reset(self) -> None:
        self.turns_used = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="任务开发阶段")
    parser.add_argument("--project-dir", help="项目目录")
    parser.add_argument("--requirement-name", help="需求名称")
    parser.add_argument("--allow-previous-stage-back", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--vendor", help="开发工程师厂商: codex|claude|gemini|opencode")
    parser.add_argument("--model", help="开发工程师模型名称")
    parser.add_argument("--effort", help="开发工程师推理强度")
    parser.add_argument("--proxy-url", default="", help="开发工程师代理端口或完整代理 URL")
    parser.add_argument("--developer-role-prompt", default="", help="开发工程师自定义角色定义提示词")
    parser.add_argument("--developer-max-turns", default=None, help="开发工程师最大对话轮数；传 infinite 表示不重建，默认 15")
    parser.add_argument("--review-max-rounds", default="", help="代码评审最多重试几轮；传 infinite 表示不设上限")
    parser.add_argument("--subagent-num", type=int, default=None, help="开发工程师自检使用的 subagent 数量")
    parser.add_argument("--reviewer-agent", action="append", default=[], help="审核智能体模型配置: name=<key>,vendor=...,model=...,effort=...,proxy=...")
    parser.add_argument("--reviewer-role", action="append", default=[], help="重复传入以覆盖代码评审角色列表")
    parser.add_argument("--reviewer-role-prompt", action="append", default=[], help="重复传入以覆盖对应角色提示词")
    parser.add_argument("--yes", action="store_true", help="跳过非关键确认")
    parser.add_argument("--no-tui", action="store_true", help="显式禁用 OpenTUI")
    parser.add_argument("--legacy-cli", action="store_true", help="使用旧版 Python CLI，不跳转 OpenTUI")
    return parser


def _consume_stage_back(allow_previous_stage_back: bool, will_prompt: bool) -> tuple[bool, bool]:
    allow_back = bool(allow_previous_stage_back and will_prompt)
    if will_prompt:
        return allow_back, False
    return False, bool(allow_previous_stage_back)


def build_development_paths(project_dir: str | Path, requirement_name: str) -> dict[str, Path]:
    project_root = Path(project_dir).expanduser().resolve()
    safe_name = sanitize_requirement_name(requirement_name)
    original_requirement_path, requirements_clear_path, ask_human_path, hitl_record_path = build_requirements_clarification_paths(
        project_root,
        requirement_name,
    )
    legacy_question_path = project_root / f"{safe_name}_向人类提问.md"
    if legacy_question_path.exists() and legacy_question_path.is_file():
        ask_human_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_text = legacy_question_path.read_text(encoding="utf-8")
        current_text = ask_human_path.read_text(encoding="utf-8") if ask_human_path.exists() else ""
        if legacy_text.strip() and not current_text.strip():
            ask_human_path.write_text(legacy_text, encoding="utf-8")
        legacy_question_path.unlink()
    return {
        "project_root": project_root,
        "original_requirement_path": original_requirement_path,
        "requirements_clear_path": requirements_clear_path,
        "ask_human_path": ask_human_path,
        "hitl_record_path": hitl_record_path,
        "detailed_design_path": project_root / f"{safe_name}_详细设计.md",
        "task_md_path": project_root / f"{safe_name}_任务单.md",
        "task_json_path": project_root / f"{safe_name}_任务单.json",
        "developer_output_path": project_root / f"{safe_name}_工程师开发内容.md",
        "merged_review_path": project_root / f"{safe_name}_代码评审记录.md",
    }


def build_development_runtime_root(project_dir: str | Path, requirement_name: str = "") -> Path:
    project_root = Path(project_dir).expanduser().resolve()
    runtime_root = project_root / DEVELOPMENT_RUNTIME_ROOT_NAME
    safe_requirement = sanitize_requirement_name(requirement_name) if str(requirement_name).strip() else ""
    return runtime_root / safe_requirement if safe_requirement else runtime_root


def build_reviewer_artifact_paths(project_dir: str | Path, requirement_name: str, reviewer_name: str) -> tuple[Path, Path]:
    project_root = Path(project_dir).expanduser().resolve()
    safe_name = sanitize_requirement_name(requirement_name)
    artifact_agent_name = sanitize_requirement_name(reviewer_name)
    review_md_path = project_root / f"{safe_name}_代码评审记录_{artifact_agent_name}.md"
    review_json_path = project_root / f"{safe_name}_评审记录_{artifact_agent_name}.json"
    return review_md_path, review_json_path


def cleanup_existing_development_artifacts(
    paths: dict[str, Path],
    requirement_name: str,
    audit_context: StageAuditRunContext | None = None,
) -> tuple[str, ...]:
    project_root = paths["project_root"]
    safe_name = sanitize_requirement_name(requirement_name)
    removed: list[str] = []
    review_json_candidates: list[Path] = []
    review_md_candidates: list[Path] = []
    for pattern in (
        f"{safe_name}_评审记录_*.json",
        f"{safe_name}_代码评审记录_*.md",
    ):
        for candidate in project_root.glob(pattern):
            if candidate.is_file():
                if candidate.suffix == ".json":
                    review_json_candidates.append(candidate)
                else:
                    review_md_candidates.append(candidate)
    if audit_context is not None:
        record_before_cleanup(
            audit_context,
            {
                "ask_human": paths["ask_human_path"],
                "developer_output": paths["developer_output_path"],
                "merged_review": paths["merged_review_path"],
            },
            metadata={"trigger": "cleanup_existing_development_artifacts"},
            reviewer_markdown_paths=review_md_candidates,
            reviewer_json_paths=review_json_candidates,
        )
    for candidate in (*review_json_candidates, *review_md_candidates):
        if candidate.is_file():
            candidate.unlink()
            removed.append(str(candidate.resolve()))
    for candidate in (
        paths["ask_human_path"],
        paths["developer_output_path"],
        paths["merged_review_path"],
    ):
        if candidate.exists() and candidate.is_file():
            candidate.write_text("", encoding="utf-8")
            removed.append(str(candidate.resolve()))
    return tuple(dict.fromkeys(removed))


def cleanup_stale_development_runtime_state(
    project_dir: str | Path,
    requirement_name: str,
) -> tuple[str, ...]:
    runtime_root = Path(project_dir).expanduser().resolve() / DEVELOPMENT_RUNTIME_ROOT_NAME
    if not runtime_root.exists() or not runtime_root.is_dir():
        return ()
    return cleanup_runtime_dirs_by_scope(
        runtime_root=runtime_root,
        project_dir=project_dir,
        requirement_name=requirement_name,
        workflow_action="stage.a07.start",
    )


def build_developer_worker_id() -> str:
    return "development-developer"


def build_development_reviewer_worker_id(role_name: str) -> str:
    return f"development-review-{str(role_name).strip()}"


def _predict_worker_display_name(
    *,
    project_dir: str | Path,
    worker_id: str,
    occupied_session_names: Sequence[str] = (),
) -> str:
    occupied = {str(name).strip() for name in occupied_session_names if str(name).strip()}
    for session_name in list_tmux_session_names():
        name = str(session_name or "").strip()
        if name:
            occupied.add(name)
    for worker in list_registered_tmux_workers():
        session_name = str(getattr(worker, "session_name", "") or "").strip()
        if session_name:
            occupied.add(session_name)
    occupied.update(
        list_occupied_tmux_session_names(
            additional_session_names=sorted(occupied),
        )
    )
    return build_session_name(
        worker_id,
        Path(project_dir).expanduser().resolve(),
        Vendor.CODEX,
        occupied_session_names=sorted(occupied),
    )


def build_task_split_stage_argv(args: argparse.Namespace, *, project_dir: str, requirement_name: str) -> list[str]:
    argv = ["--project-dir", project_dir, "--requirement-name", requirement_name]
    if getattr(args, "yes", False):
        argv.append("--yes")
    if getattr(args, "no_tui", False):
        argv.append("--no-tui")
    if getattr(args, "legacy_cli", False):
        argv.append("--legacy-cli")
    return argv


def ensure_development_inputs(
    args: argparse.Namespace,
    *,
    project_dir: str,
    requirement_name: str,
) -> dict[str, Path]:
    paths = build_development_paths(project_dir, requirement_name)
    if (not get_markdown_content(paths["task_md_path"]).strip()) or (not is_task_progress_json(paths["task_json_path"])):
        message(f"缺少任务拆分产物，先自动执行任务拆分阶段: {paths['task_md_path'].name}")
        run_task_split_stage(build_task_split_stage_argv(args, project_dir=project_dir, requirement_name=requirement_name))
    paths = build_development_paths(project_dir, requirement_name)
    ensure_pre_development_task_record(project_dir, requirement_name)
    if not get_markdown_content(paths["requirements_clear_path"]).strip():
        raise RuntimeError(f"缺少需求澄清文档: {paths['requirements_clear_path']}")
    if not get_markdown_content(paths["detailed_design_path"]).strip():
        raise RuntimeError(f"缺少详细设计文档: {paths['detailed_design_path']}")
    if not get_markdown_content(paths["task_md_path"]).strip():
        raise RuntimeError(f"缺少任务单文档: {paths['task_md_path']}")
    if not is_task_progress_json(paths["task_json_path"]):
        raise RuntimeError(f"缺少合法任务单 JSON: {paths['task_json_path']}")
    return paths


def _default_development_prompt_for_role(role_name: str) -> str:
    return DEFAULT_DEVELOPMENT_REVIEWER_PROMPTS.get(str(role_name).strip(), "")


def _finalize_reviewer_specs(specs: Sequence[DevelopmentReviewerSpec]) -> list[DevelopmentReviewerSpec]:
    role_counts = Counter(str(item.role_name).strip() for item in specs)
    role_seen: Counter[str] = Counter()
    finalized: list[DevelopmentReviewerSpec] = []
    for item in specs:
        role_name = str(item.role_name).strip()
        role_prompt = str(item.role_prompt).strip()
        if not role_name:
            raise RuntimeError("审核智能体角色定义不能为空。")
        if not role_prompt:
            raise RuntimeError(f"{role_name} 的角色定义提示词不能为空。")
        reviewer_key = str(item.reviewer_key).strip()
        if not reviewer_key:
            role_seen[role_name] += 1
            reviewer_key = f"{role_name}#{role_seen[role_name]}" if role_counts[role_name] > 1 else role_name
        finalized.append(
            DevelopmentReviewerSpec(
                role_name=role_name,
                role_prompt=role_prompt,
                reviewer_key=reviewer_key,
            )
        )
    return finalized


def _review_progress_context(progress: ReviewStageProgress | None):
    return progress.suspended() if progress is not None else nullcontext()


def _prompt_reviewer_select(
    *,
    title: str,
    options: Sequence[tuple[str, str]],
    default_value: str,
    prompt_text: str,
    progress: ReviewStageProgress | None = None,
    allow_back: bool = False,
    stage_key: str = "",
    stage_step_index: int = 0,
) -> str:
    from T09_terminal_ops import prompt_select_option

    with _review_progress_context(progress):
        with prompt_metadata(
            allow_back=allow_back,
            back_value=PROMPT_BACK_VALUE,
            stage_key=stage_key,
            stage_step_index=stage_step_index,
        ):
            return prompt_select_option(
                title=title,
                options=options,
                default_value=default_value,
                prompt_text=prompt_text,
            )


def _prompt_reviewer_text(
    prompt_text: str,
    *,
    default: str = "",
    allow_empty: bool = False,
    progress: ReviewStageProgress | None = None,
    allow_back: bool = False,
    stage_key: str = "",
    stage_step_index: int = 0,
) -> str:
    with _review_progress_context(progress):
        with prompt_metadata(
            allow_back=allow_back,
            back_value=PROMPT_BACK_VALUE,
            stage_key=stage_key,
            stage_step_index=stage_step_index,
        ):
            return prompt_with_default(prompt_text, default, allow_empty=allow_empty).strip()


def _collect_single_interactive_reviewer_spec(
    *,
    index: int,
    default_roles: Sequence[str],
    default_role_name: str,
    progress: ReviewStageProgress | None,
    stage_key: str,
) -> DevelopmentReviewerSpec:
    role_source = "default"
    role_name = default_role_name
    prompt_source = "default"
    role_prompt = ""
    step = 0
    while True:
        try:
            if step == 0:
                role_source = _prompt_reviewer_select(
                    title=f"第 {index} 个审核智能体 - 角色定义来源",
                    options=(("default", "默认角色"), ("custom", "自定义角色")),
                    default_value=role_source or "default",
                    prompt_text=f"选择第 {index} 个审核智能体的角色定义来源",
                    progress=progress,
                    allow_back=True,
                    stage_key=stage_key,
                    stage_step_index=1,
                )
                step = 1
                continue
            if step == 1:
                if role_source == "default":
                    role_name = _prompt_reviewer_select(
                        title=f"第 {index} 个审核智能体 - 选择默认角色定义",
                        options=tuple((item, item) for item in default_roles),
                        default_value=role_name or default_role_name,
                        prompt_text=f"选择第 {index} 个审核智能体的默认角色定义",
                        progress=progress,
                        allow_back=True,
                        stage_key=stage_key,
                        stage_step_index=2,
                    )
                else:
                    role_name = _prompt_reviewer_text(
                        f"输入第 {index} 个审核智能体的自定义角色定义",
                        default="" if role_name == default_role_name else role_name,
                        progress=progress,
                        allow_back=True,
                        stage_key=stage_key,
                        stage_step_index=2,
                    )
                step = 2
                continue
            default_prompt = _default_development_prompt_for_role(role_name)
            if default_prompt:
                if step == 2:
                    prompt_source = _prompt_reviewer_select(
                        title=f"第 {index} 个审核智能体 - 角色定义提示词来源",
                        options=(("default", "默认提示词"), ("custom", "自定义提示词")),
                        default_value=prompt_source or "default",
                        prompt_text=f"选择第 {index} 个审核智能体的角色定义提示词来源",
                        progress=progress,
                        allow_back=True,
                        stage_key=stage_key,
                        stage_step_index=3,
                    )
                    if prompt_source == "default":
                        return DevelopmentReviewerSpec(role_name=role_name, role_prompt=default_prompt)
                    step = 3
                    continue
                if step == 3:
                    role_prompt = _prompt_reviewer_text(
                        f"输入第 {index} 个审核智能体的自定义角色定义提示词",
                        default=role_prompt or default_prompt,
                        progress=progress,
                        allow_back=True,
                        stage_key=stage_key,
                        stage_step_index=4,
                    )
                    return DevelopmentReviewerSpec(role_name=role_name, role_prompt=role_prompt)
            else:
                role_prompt = _prompt_reviewer_text(
                    f"输入第 {index} 个审核智能体的自定义角色定义提示词",
                    default=role_prompt,
                    progress=progress,
                    allow_back=True,
                    stage_key=stage_key,
                    stage_step_index=3,
                )
                return DevelopmentReviewerSpec(role_name=role_name, role_prompt=role_prompt)
        except PromptBackRequested:
            if step == 0:
                raise
            step -= 1


def collect_interactive_reviewer_specs(
    *,
    progress: ReviewStageProgress | None = None,
    allow_back_first_prompt: bool = False,
    stage_key: str = "development_reviewer_specs",
) -> list[DevelopmentReviewerSpec]:
    if progress is not None:
        progress.set_phase("任务开发 / 配置审核器")
    reviewer_count = prompt_positive_int(
        "请输入代码评审智能体数量",
        len(DEFAULT_DEVELOPMENT_REVIEWER_ROLE_NAMES),
        progress=progress,
        allow_back=allow_back_first_prompt,
        stage_key=stage_key,
        stage_step_index=0,
    )
    collected_specs: list[DevelopmentReviewerSpec] = []
    default_roles = list(DEFAULT_DEVELOPMENT_REVIEWER_ROLE_NAMES)
    index = 1
    while index <= reviewer_count:
        default_role_name = default_roles[(index - 1) % len(default_roles)]
        try:
            collected_specs.append(
                _collect_single_interactive_reviewer_spec(
                    index=index,
                    default_roles=default_roles,
                    default_role_name=default_role_name,
                    progress=progress,
                    stage_key=stage_key,
                )
            )
            index += 1
        except PromptBackRequested:
            if not collected_specs:
                reviewer_count = prompt_positive_int(
                    "请输入代码评审智能体数量",
                    reviewer_count,
                    progress=progress,
                    allow_back=allow_back_first_prompt,
                    stage_key=stage_key,
                    stage_step_index=0,
                )
                index = 1
                continue
            collected_specs.pop()
            index -= 1
    return _finalize_reviewer_specs(collected_specs)


def resolve_reviewer_specs(
    args: argparse.Namespace,
    *,
    progress: ReviewStageProgress | None = None,
    allow_back_first_prompt: bool = False,
) -> list[DevelopmentReviewerSpec]:
    agent_config = resolve_stage_agent_config(args)
    role_names = [str(item).strip() for item in getattr(args, "reviewer_role", []) if str(item).strip()]
    prompt_values = [str(item).strip() for item in getattr(args, "reviewer_role_prompt", []) if str(item).strip()]
    if agent_config.reviewer_order and not role_names and not prompt_values:
        default_roles = list(DEFAULT_DEVELOPMENT_REVIEWER_ROLE_NAMES)
        specs: list[DevelopmentReviewerSpec] = []
        for index, reviewer_key in enumerate(agent_config.reviewer_order):
            role_name = reviewer_key if reviewer_key in DEFAULT_DEVELOPMENT_REVIEWER_PROMPTS else default_roles[index % len(default_roles)]
            specs.append(
                DevelopmentReviewerSpec(
                    role_name=role_name,
                    role_prompt=_default_development_prompt_for_role(role_name),
                    reviewer_key=reviewer_key,
                )
            )
        return _finalize_reviewer_specs(specs)
    if stdin_is_interactive() and not role_names and not prompt_values:
        return collect_interactive_reviewer_specs(
            progress=progress,
            allow_back_first_prompt=allow_back_first_prompt,
        )
    if role_names and prompt_values and len(prompt_values) != len(role_names):
        raise RuntimeError("--reviewer-role-prompt 数量必须与 --reviewer-role 数量一致。")
    if not role_names:
        role_names = list(DEFAULT_DEVELOPMENT_REVIEWER_ROLE_NAMES)
        if prompt_values and len(prompt_values) != len(role_names):
            raise RuntimeError("--reviewer-role-prompt 数量必须与默认角色数量一致。")
    specs: list[DevelopmentReviewerSpec] = []
    for index, role_name in enumerate(role_names):
        default_prompt = _default_development_prompt_for_role(role_name)
        if prompt_values:
            role_prompt = prompt_values[index]
        elif default_prompt:
            role_prompt = default_prompt
        else:
            raise RuntimeError(f"自定义角色 {role_name} 必须同时提供对应的角色定义提示词。")
        specs.append(DevelopmentReviewerSpec(role_name=role_name, role_prompt=role_prompt))
    return _finalize_reviewer_specs(specs)


def resolve_developer_role_prompt(
    args: argparse.Namespace,
    *,
    progress: ReviewStageProgress | None = None,
    allow_back: bool = False,
) -> str:
    value = str(getattr(args, "developer_role_prompt", "") or "").strip()
    if value:
        return value
    if bool(getattr(args, "yes", False)):
        return fintech_developer_role
    if not stdin_is_interactive():
        return fintech_developer_role
    from tmux_core.stage_kernel.shared_review import prompt_yes_no_choice

    if prompt_yes_no_choice(
        "开发工程师是否使用默认角色定义提示词",
        True,
        progress=progress,
        allow_back=allow_back,
        stage_key="development",
        stage_step_index=0,
    ):
        return fintech_developer_role
    return _prompt_reviewer_text(
        "输入开发工程师的自定义角色定义提示词",
        default=fintech_developer_role,
        progress=progress,
    )


def _reviewer_default_selection() -> ReviewAgentSelection:
    return ReviewAgentSelection(
        vendor=DEFAULT_REQUIREMENTS_CLARIFICATION_VENDOR,
        model=DEFAULT_REQUIREMENTS_CLARIFICATION_MODEL,
        reasoning_effort=DEFAULT_REQUIREMENTS_CLARIFICATION_EFFORT,
        proxy_url="",
    )


def _developer_display_name(*, project_dir: str | Path) -> str:
    return _predict_worker_display_name(project_dir=project_dir, worker_id=build_developer_worker_id())


def create_developer_runtime(
    *,
    project_dir: str | Path,
    requirement_name: str = "",
    selection: ReviewAgentSelection,
    role_prompt: str,
    launch_coordinator: LaunchCoordinator | None = None,
    run_id: str = "",
) -> DeveloperRuntime:
    project_root = Path(project_dir).expanduser().resolve()
    selection, config = resolve_agent_run_config_with_recovery(
        selection,
        role_label=_developer_display_name(project_dir=project_root),
    )
    worker = TmuxBatchWorker(
        worker_id=build_developer_worker_id(),
        work_dir=project_root,
        config=config,
        runtime_root=build_development_runtime_root(project_root, requirement_name),
        launch_coordinator=launch_coordinator,
        runtime_metadata={
            "project_dir": str(project_root),
            "requirement_name": str(requirement_name).strip(),
            "workflow_action": "stage.a07.start",
            "run_id": str(run_id or "").strip(),
            "agent_role": "developer",
            "role_prompt": str(role_prompt or "").strip(),
        },
    )
    message(render_tmux_start_summary(str(worker.session_name).strip() or "开发工程师", worker))
    return DeveloperRuntime(selection=selection, worker=worker, role_prompt=role_prompt)


def recreate_developer_runtime(
    *,
    project_dir: str | Path,
    requirement_name: str = "",
    developer: DeveloperRuntime,
    progress: ReviewStageProgress | None = None,
    launch_coordinator: LaunchCoordinator | None = None,
    required_reconfiguration: bool = False,
    reason_text: str = "",
) -> DeveloperRuntime | None:
    scope_requirement_name = (
        str(requirement_name).strip()
        or str(getattr(developer.worker, "_runtime_metadata", {}).get("requirement_name", "") or "").strip()
    )
    if required_reconfiguration and not stdin_is_interactive():
        raise RuntimeError("开发工程师需要重新配置智能体，但当前环境无法交互选择厂商/模型。")
    if not required_reconfiguration and not stdin_is_interactive():
        raise RuntimeError("开发工程师已死亡，需要人工选择厂商/模型/推理/代理后才能重建。")
    developer_name = str(developer.worker.session_name or "开发工程师").strip() or "开发工程师"
    selection = (
        prompt_required_replacement_review_agent_selection(
            reason_text=reason_text or f"检测到{developer_name}不可继续使用，需要更换模型后继续当前阶段。",
            previous_selection=developer.selection,
            force_model_change=True,
            role_label=developer_name,
            progress=progress,
        )
        if required_reconfiguration
        else prompt_replacement_review_agent_selection(
            reason_text=f"检测到{developer_name}已死亡，需要重建开发工程师后继续当前阶段。",
            previous_selection=developer.selection,
            force_model_change=True,
            role_label=developer_name,
            progress=progress,
        )
    )
    if selection is None:
        return None
    return create_developer_runtime(
        project_dir=project_dir,
        requirement_name=scope_requirement_name,
        selection=selection,
        role_prompt=developer.role_prompt,
        launch_coordinator=launch_coordinator,
    )


def resolve_developer_plan(
    args: argparse.Namespace,
    *,
    project_dir: str | Path,
    progress: ReviewStageProgress | None = None,
    allow_back_first_prompt: bool = False,
) -> DeveloperPlan:
    role_prompt_prompted = stdin_is_interactive() and not str(getattr(args, "developer_role_prompt", "") or "").strip()
    role_prompt = resolve_developer_role_prompt(
        args,
        progress=progress,
        allow_back=allow_back_first_prompt and role_prompt_prompted,
    )
    selection = collect_ba_agent_selection(
        args,
        role_label=_developer_display_name(project_dir=project_dir),
        allow_back_first_step=allow_back_first_prompt and not role_prompt_prompted,
        stage_key="development_main",
    )
    message(render_review_agent_selection("开发工程师 配置", selection))
    return DeveloperPlan(selection=selection, role_prompt=role_prompt)


def prepare_developer_runtime(
    args: argparse.Namespace,
    *,
    project_dir: str | Path,
    requirement_name: str = "",
    progress: ReviewStageProgress | None = None,
) -> DeveloperRuntime:
    plan = resolve_developer_plan(args, project_dir=project_dir, progress=progress)
    return create_developer_runtime(
        project_dir=project_dir,
        requirement_name=requirement_name,
        selection=plan.selection,
        role_prompt=plan.role_prompt,
    )


def build_developer_init_prompt(paths: dict[str, Path], *, role_prompt: str) -> str:
    return init_developer(
        role_prompt,
        init_prompt="",
        ask_human_md=str(paths["ask_human_path"].resolve()),
        hitl_record_md=str(paths["hitl_record_path"].resolve()),
        requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
        detailed_design_md=str(paths["detailed_design_path"].resolve()),
        task_split_md=str(paths["task_md_path"].resolve()),
    )


def build_developer_human_reply_prompt(paths: dict[str, Path], *, human_msg: str) -> str:
    return human_reply(
        human_msg,
        ask_human_md=str(paths["ask_human_path"].resolve()),
        hitl_record_md=str(paths["hitl_record_path"].resolve()),
        requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
        detailed_design_md=str(paths["detailed_design_path"].resolve()),
        task_split_md=str(paths["task_md_path"].resolve()),
    )


def _parse_developer_max_turns(raw_value: str, *, source: str) -> int | None:
    text = str(raw_value).strip()
    lowered = text.lower()
    if lowered in {"infinite", "inf", "unlimited", "none", "无限", "不重建"}:
        return None
    if text.isdigit():
        value = int(text)
        if value <= 0:
            raise RuntimeError(f"{source} 必须是正整数，或输入 infinite 表示不重建。")
        return value
    raise RuntimeError(f"{source} 必须是正整数，或输入 infinite 表示不重建。")


def resolve_developer_max_turns(
    args: argparse.Namespace,
    *,
    progress: ReviewStageProgress | None = None,
    allow_back: bool = False,
) -> int | None:
    explicit = getattr(args, "developer_max_turns", None)
    if explicit is not None and str(explicit).strip():
        return _parse_developer_max_turns(str(explicit), source="--developer-max-turns")
    if not stdin_is_interactive():
        return DEFAULT_DEVELOPER_MAX_TURNS
    if progress is not None:
        progress.set_phase("任务开发 / 配置开发工程师最大对话轮数")
    with progress.suspended() if progress is not None else nullcontext():
        while True:
            with prompt_metadata(
                allow_back=allow_back,
                back_value=PROMPT_BACK_VALUE,
                stage_key="development",
                stage_step_index=3,
            ):
                value = prompt_with_default(
                    "输入开发工程师最大对话轮数（输入 infinite 表示不重建）",
                    str(DEFAULT_DEVELOPER_MAX_TURNS),
                ).strip()
            try:
                return _parse_developer_max_turns(value, source="开发工程师最大对话轮数")
            except RuntimeError as error:
                message(str(error))


def resolve_review_max_rounds(
    args: argparse.Namespace,
    *,
    progress: ReviewStageProgress | None = None,
    allow_back: bool = False,
) -> int | None:
    explicit = getattr(args, "review_max_rounds", "")
    if str(explicit or "").strip():
        return parse_review_max_rounds(
            explicit,
            source="--review-max-rounds",
            default=MAX_DEVELOPMENT_REVIEW_ROUNDS,
        )
    if not stdin_is_interactive():
        return MAX_DEVELOPMENT_REVIEW_ROUNDS
    if progress is not None and hasattr(progress, "set_phase"):
        progress.set_phase("任务开发 / 配置最大审核轮次")
    return prompt_review_max_rounds(
        default=MAX_DEVELOPMENT_REVIEW_ROUNDS,
        progress=progress,
        allow_back=allow_back,
        stage_key="development",
        stage_step_index=4,
    )


def build_development_review_limit_hitl_config(paths: dict[str, Path]) -> ReviewLimitHitlConfig:
    return ReviewLimitHitlConfig(
        stage_label="任务开发评审超限",
        artifact_label="代码评审",
        primary_output_path=paths["developer_output_path"],
        ask_human_path=paths["ask_human_path"],
        hitl_record_path=paths["hitl_record_path"],
        merged_review_path=paths["merged_review_path"],
        output_summary_path=paths["developer_output_path"],
        continue_output_label="工程师开发内容.md",
    )


def build_development_review_limit_force_hitl_prompt(
    *,
    paths: dict[str, Path],
    task_name: str,
    review_msg: str,
    review_limit: int,
    review_rounds_used: int,
) -> str:
    _ = task_name
    return render_review_limit_force_hitl_prompt(
        config=build_development_review_limit_hitl_config(paths),
        review_limit=review_limit,
        review_rounds_used=review_rounds_used,
        hitl_record_md=paths["hitl_record_path"],
        extra_inputs=(
            paths["requirements_clear_path"],
            paths["detailed_design_path"],
            paths["task_md_path"],
            paths["task_json_path"],
        ),
    ) + f"\n## 当前评审记录\n[REVIEW MSG START]\n{review_msg}\n[REVIEW MSG END]\n"


def build_development_review_limit_human_reply_prompt(
    *,
    paths: dict[str, Path],
    task_name: str,
    review_msg: str,
    human_msg: str,
) -> str:
    _ = task_name
    return render_review_limit_human_reply_prompt(
        config=build_development_review_limit_hitl_config(paths),
        human_msg=human_msg,
        hitl_record_md=paths["hitl_record_path"],
        extra_inputs=(
            paths["requirements_clear_path"],
            paths["detailed_design_path"],
            paths["task_md_path"],
            paths["task_json_path"],
        ),
    ) + f"\n## 当前评审记录\n[REVIEW MSG START]\n{review_msg}\n[REVIEW MSG END]\n"


def build_reviewer_init_prompt(paths: dict[str, Path], *, reviewer_spec: DevelopmentReviewerSpec) -> str:
    return init_code_reviewer(
        reviewer_spec.role_prompt,
        init_prompt="",
        hitl_record_md=str(paths["hitl_record_path"].resolve()),
        requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
        detailed_design_md=str(paths["detailed_design_path"].resolve()),
        task_split_md=str(paths["task_md_path"].resolve()),
    )


def build_developer_init_result_contract(paths: dict[str, Path], *, mode: str) -> TaskResultContract:
    return TaskResultContract(
        turn_id=mode,
        phase=mode,
        task_kind=mode,
        mode=mode,
        expected_statuses=("ready", "hitl"),
        stage_name="任务开发",
        optional_artifacts={
            "ask_human": paths["ask_human_path"],
            "hitl_record": paths["hitl_record_path"],
            "requirements_clear": paths["requirements_clear_path"],
            "detailed_design": paths["detailed_design_path"],
            "task_md": paths["task_md_path"],
            "task_json": paths["task_json_path"],
        },
        outcome_artifacts={
            "ready": {
                "optional": ("hitl_record", "requirements_clear", "detailed_design", "task_md", "task_json"),
                "forbids": ("ask_human",),
            },
            "hitl": {
                "requires": ("ask_human",),
                "optional": ("hitl_record", "requirements_clear", "detailed_design", "task_md", "task_json"),
            },
        },
    )


def build_developer_review_feedback_result_contract(paths: dict[str, Path]) -> TaskResultContract:
    return build_developer_review_feedback_contract(paths)


def build_developer_review_feedback_contract(
    paths: dict[str, Path],
    *,
    mode: str = "a07_developer_review_feedback",
) -> TaskResultContract:
    expected_statuses = ("hitl", "completed")
    outcome_artifacts = {
        "hitl": {
            "requires": ("ask_human",),
            "optional": ("hitl_record", "requirements_clear", "detailed_design", "task_md", "task_json"),
        },
        "completed": {
            "requires": ("developer_output",),
            "optional": ("hitl_record", "requirements_clear", "detailed_design", "task_md", "task_json"),
        },
    }
    if mode == "a07_developer_review_limit_force_hitl":
        expected_statuses = ("hitl",)
        outcome_artifacts = {
            "hitl": {
                "requires": ("ask_human",),
                "optional": ("hitl_record", "requirements_clear", "detailed_design", "task_md", "task_json"),
            },
        }
    return TaskResultContract(
        turn_id=mode,
        phase=mode,
        task_kind=mode,
        mode=mode,
        expected_statuses=expected_statuses,
        stage_name="任务开发",
        optional_artifacts={
            "ask_human": paths["ask_human_path"],
            "hitl_record": paths["hitl_record_path"],
            "developer_output": paths["developer_output_path"],
            "requirements_clear": paths["requirements_clear_path"],
            "detailed_design": paths["detailed_design_path"],
            "task_md": paths["task_md_path"],
            "task_json": paths["task_json_path"],
        },
        outcome_artifacts=outcome_artifacts,
    )


def _build_developer_turn_goal(paths: dict[str, Path], contract: TaskResultContract, *, task_name: str = "") -> TaskTurnGoal | None:
    if contract.mode in {"a07_developer_init", "a07_developer_human_reply"}:
        return TaskTurnGoal(
            goal_id=contract.mode,
            outcomes={
                "ready": OutcomeGoal(status="ready"),
                "hitl": OutcomeGoal(status="hitl", required_aliases=("ask_human",)),
            },
        )
    if contract.mode in {"a07_developer_task_complete", "a07_developer_refine"}:
        return TaskTurnGoal(
            goal_id=contract.mode,
            outcomes={"completed": OutcomeGoal(status="completed", required_aliases=("developer_output",))},
            repair_prompt_builder=lambda context: check_develop_job(
                paths["developer_output_path"],
                task_name or "当前任务",
                task_split_md=str(paths["task_md_path"].resolve()),
                what_just_dev=str(paths["developer_output_path"].resolve()),
            ),
        )
    if contract.mode == "a07_developer_review_limit_force_hitl":
        return TaskTurnGoal(
            goal_id=contract.mode,
            outcomes={"hitl": OutcomeGoal(status="hitl", required_aliases=("ask_human",))},
        )
    if contract.mode in {"a07_developer_review_feedback", "a07_developer_review_limit_human_reply"}:
        return TaskTurnGoal(
            goal_id=contract.mode,
            outcomes={
                "hitl": OutcomeGoal(status="hitl", required_aliases=("ask_human",)),
                "completed": OutcomeGoal(status="completed", required_aliases=("developer_output",)),
            },
        )
    if contract.mode == "a07_reviewer_init":
        return TaskTurnGoal(goal_id=contract.mode, outcomes={"ready": OutcomeGoal(status="ready")})
    return None


def build_reviewer_init_result_contract(paths: dict[str, Path]) -> TaskResultContract:
    return TaskResultContract(
        turn_id="a07_reviewer_init",
        phase="a07_reviewer_init",
        task_kind="a07_reviewer_init",
        mode="a07_reviewer_init",
        expected_statuses=("ready",),
        stage_name="任务开发",
        optional_artifacts={
            "requirements_clear": paths["requirements_clear_path"],
            "detailed_design": paths["detailed_design_path"],
            "task_md": paths["task_md_path"],
            "task_json": paths["task_json_path"],
            "hitl_record": paths["hitl_record_path"],
        },
    )


def build_developer_task_complete_result_contract(paths: dict[str, Path]) -> TaskResultContract:
    return TaskResultContract(
        turn_id="a07_developer_task_complete",
        phase="a07_developer_task_complete",
        task_kind="a07_developer_task_complete",
        mode="a07_developer_task_complete",
        expected_statuses=("completed",),
        stage_name="任务开发",
        required_artifacts={"developer_output": paths["developer_output_path"]},
        optional_artifacts={
            "task_md": paths["task_md_path"],
            "task_json": paths["task_json_path"],
        },
    )


def build_developer_refine_result_contract(paths: dict[str, Path]) -> TaskResultContract:
    return TaskResultContract(
        turn_id="a07_developer_refine",
        phase="a07_developer_refine",
        task_kind="a07_developer_refine",
        mode="a07_developer_refine",
        expected_statuses=("completed",),
        stage_name="任务开发",
        required_artifacts={"developer_output": paths["developer_output_path"]},
        optional_artifacts={
            "task_md": paths["task_md_path"],
            "task_json": paths["task_json_path"],
        },
    )


def _parse_result_payload(clean_output: str) -> dict[str, object]:
    payload = json.loads(clean_output)
    if not isinstance(payload, dict):
        raise RuntimeError("结构化结果必须是 JSON 对象")
    return payload


def _run_developer_result_turn(
    developer: DeveloperRuntime,
    *,
    label: str,
    prompt: str,
    result_contract: TaskResultContract,
    paths: dict[str, Path] | None = None,
    task_name: str = "",
    turn_goal: TaskTurnGoal | None = None,
    turn_policy: DeveloperTurnPolicy | None = None,
    replace_dead_developer=None,
    progress: ReviewStageProgress | None = None,
) -> tuple[DeveloperRuntime, dict[str, object]]:
    current_developer = developer
    while True:
        try:
            payload = run_task_result_turn_with_repair(
                worker=current_developer.worker,
                label=label,
                prompt=prompt,
                result_contract=result_contract,
                parse_result_payload=_parse_result_payload,
                turn_goal=turn_goal or (
                    _build_developer_turn_goal(paths, result_contract, task_name=task_name)
                    if paths is not None else None
                ),
                timeout_sec=DEFAULT_COMMAND_TIMEOUT_SEC,
                stage_label="任务开发",
                role_label=str(current_developer.worker.session_name or "开发工程师").strip() or "开发工程师",
                task_name=task_name,
            )
            if turn_policy is not None:
                turn_policy.record_turn()
            return current_developer, payload
        except Exception as error:  # noqa: BLE001
            if is_agent_ready_timeout_error(error):
                developer_display_name = str(current_developer.worker.session_name or "开发工程师").strip() or "开发工程师"
                prompt_agent_ready_timeout_recovery(
                    current_developer.worker,
                    role_label=developer_display_name,
                    can_skip=False,
                    progress=progress,
                    reason_text=(
                        f"{developer_display_name}启动超时，未能进入可输入状态。\n"
                        "请先手动更换模型，再继续尝试当前 turn。"
                    ),
                )
                continue
            if is_worker_death_error(error) and replace_dead_developer is not None:
                current_developer = replace_dead_developer(current_developer, error)
                continue
            if is_recoverable_startup_failure(error, current_developer.worker) and replace_dead_developer is not None:
                current_developer = replace_dead_developer(current_developer, error)
                continue
            if is_task_result_contract_error(error):
                raise
            if try_resume_worker(current_developer.worker, timeout_sec=60.0):
                continue
            raise RuntimeError(f"{current_developer.worker.session_name or '开发工程师'} 执行失败") from error


def build_placeholder_reviewer_contract(review_json_path: Path) -> TurnFileContract:
    def validator(status_path: Path) -> TurnFileResult:
        return TurnFileResult(
            status_path=str(status_path.resolve()),
            payload={"task_name": "", "review_pass": True},
            artifact_paths={},
            artifact_hashes={},
            validated_at="0",
        )

    return TurnFileContract(
        turn_id="development_review_placeholder",
        phase="任务开发",
        status_path=review_json_path,
        validator=validator,
    )


def build_reviewer_completion_contract(
    *,
    reviewer_name: str,
    task_name: str,
    review_md_path: Path,
    review_json_path: Path,
) -> TurnFileContract:
    def validator(status_path: Path) -> TurnFileResult:
        if not status_path.exists():
            raise FileNotFoundError(f"缺少审核 JSON 文件: {status_path}")
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        matched_item = normalize_review_status_payload(
            payload,
            task_name=task_name,
            source=status_path.name,
        )
        review_pass = matched_item["review_pass"]
        review_md_empty = is_file_empty(review_md_path)
        if review_pass and not review_md_empty:
            raise ValueError(f"{reviewer_name} 已审核通过，但 {review_md_path.name} 不为空")
        if (not review_pass) and review_md_empty:
            raise ValueError(f"{reviewer_name} 未通过，但 {review_md_path.name} 为空")
        artifact_paths = {
            "review_md": str(review_md_path.resolve()),
            "review_json": str(review_json_path.resolve()),
        }
        artifact_hashes = {
            str(review_md_path.resolve()): build_prefixed_sha256(review_md_path),
            str(review_json_path.resolve()): build_prefixed_sha256(review_json_path),
        }
        return TurnFileResult(
            status_path=str(status_path.resolve()),
            payload={"task_name": task_name, "review_pass": review_pass},
            artifact_paths=artifact_paths,
            artifact_hashes=artifact_hashes,
            validated_at=str(status_path.stat().st_mtime),
        )

    return TurnFileContract(
        turn_id=f"development_review_{sanitize_requirement_name(task_name)}_{sanitize_requirement_name(reviewer_name)}",
        phase="任务开发",
        status_path=review_json_path,
        validator=validator,
        kind="review_round",
        tracked_artifacts={
            "review_md": review_md_path,
            "review_json": review_json_path,
        },
    )


def _build_reviewer_turn_goal() -> CompletionTurnGoal:
    return CompletionTurnGoal(
        goal_id="a07_reviewer_round",
        outcomes={
            "review_pass": OutcomeGoal(status="review_pass", required_aliases=("review_json",), forbidden_aliases=("review_md",)),
            "review_fail": OutcomeGoal(status="review_fail", required_aliases=("review_json", "review_md")),
        },
        repair_prompt_builder=_build_reviewer_protocol_repair_prompt,
    )


def _infer_reviewer_artifact_name_from_context(context: RepairPromptContext) -> str:
    role_label = str(context.role_label or "").strip()
    if role_label:
        return role_label
    review_json = str(context.artifact_paths.get("review_json", "") or "").strip()
    if not review_json:
        return ""
    path = Path(review_json)
    safe_requirement = sanitize_requirement_name(context.requirement_name) if context.requirement_name else ""
    prefix = f"{safe_requirement}_评审记录_" if safe_requirement else ""
    suffix = ".json"
    name = path.name
    if prefix and name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix):-len(suffix)]
    if name.endswith(suffix):
        return name[:-len(suffix)]
    return name


def _build_reviewer_protocol_repair_prompt(context: RepairPromptContext) -> str:
    reviewer_artifact_name = _infer_reviewer_artifact_name_from_context(context)
    review_json = str(context.artifact_paths.get("review_json", "") or "").strip()
    review_md = str(context.artifact_paths.get("review_md", "") or "").strip()
    task_name = str(context.task_name or "").strip()
    if not reviewer_artifact_name or not review_json or not task_name:
        from tmux_core.stage_kernel.turn_output_goals import build_default_completion_repair_prompt

        return build_default_completion_repair_prompt(context)

    review_json_path = Path(review_json).expanduser().resolve()
    safe_requirement = sanitize_requirement_name(context.requirement_name) if context.requirement_name else ""
    if safe_requirement:
        json_pattern = f"{safe_requirement}_评审记录_*.json"
        md_pattern = f"{safe_requirement}_代码评审记录_*.md"
    else:
        json_pattern = review_json_path.name.replace(reviewer_artifact_name, "*")
        md_pattern = Path(review_md).name.replace(reviewer_artifact_name, "*") if review_md else "代码评审记录_*.md"
    prompts = check_reviewer_job(
        [reviewer_artifact_name],
        directory=review_json_path.parent,
        task_name=task_name,
        json_pattern=json_pattern,
        md_pattern=md_pattern,
    )
    prompt = str(prompts.get(reviewer_artifact_name, "") or "").strip()
    if not prompt:
        from tmux_core.stage_kernel.turn_output_goals import build_default_completion_repair_prompt

        prompt = build_default_completion_repair_prompt(context)
    exact_paths = [
        "",
        "## 精确文件路径约束",
        f"- 必须写入 JSON: `{review_json_path}`",
    ]
    if review_md:
        exact_paths.append(f"- 必须写入/清空 Markdown: `{Path(review_md).expanduser().resolve()}`")
    exact_paths.extend(
        [
            "- 不要创建同名变体文件。",
            "- 不要在角色名与星宿名之间插入额外空格。",
        ]
    )
    return prompt + "\n" + "\n".join(exact_paths)


def _reviewer_has_materialized_outputs(reviewer: ReviewerRuntime, task_name: str) -> bool:
    try:
        payload = json.loads(reviewer.review_json_path.read_text(encoding="utf-8"))
    except Exception:
        payload = None
    try:
        normalize_review_status_payload(
            payload,
            task_name=task_name,
            source=str(reviewer.review_json_path),
        )
        return True
    except ValueError:
        pass
    return not is_file_empty(reviewer.review_md_path)


def _reviewer_artifact_signature(reviewer: ReviewerRuntime) -> tuple[object, ...]:
    signatures: list[object] = []
    for path in (reviewer.review_md_path, reviewer.review_json_path):
        if not path.exists():
            signatures.append(("missing", str(path.resolve())))
            continue
        stat = path.stat()
        signatures.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    return tuple(signatures)


def create_reviewer_runtime(
    *,
    project_dir: str | Path,
    requirement_name: str,
    reviewer_spec: DevelopmentReviewerSpec,
    selection: ReviewAgentSelection,
    launch_coordinator: LaunchCoordinator | None = None,
    run_id: str = "",
) -> ReviewerRuntime:
    reviewer_identity = str(reviewer_spec.reviewer_key or reviewer_spec.role_name).strip()
    runtime_root = build_development_runtime_root(project_dir, requirement_name)
    reviewer_display_name = _predict_worker_display_name(
        project_dir=project_dir,
        worker_id=build_development_reviewer_worker_id(reviewer_spec.role_name),
    )
    selection, config = resolve_agent_run_config_with_recovery(
        selection,
        role_label=reviewer_display_name,
    )
    worker = TmuxBatchWorker(
        worker_id=build_development_reviewer_worker_id(reviewer_spec.role_name),
        work_dir=Path(project_dir).expanduser().resolve(),
        config=config,
        runtime_root=runtime_root,
        launch_coordinator=launch_coordinator,
        runtime_metadata={
            "project_dir": str(Path(project_dir).expanduser().resolve()),
            "requirement_name": str(requirement_name).strip(),
            "workflow_action": "stage.a07.start",
            "run_id": str(run_id or "").strip(),
            "agent_role": "reviewer",
            "role_name": str(reviewer_spec.role_name).strip(),
            "reviewer_key": reviewer_identity,
            "role_prompt": str(reviewer_spec.role_prompt or "").strip(),
        },
    )
    review_md_path, review_json_path = build_reviewer_artifact_paths(
        project_dir,
        requirement_name,
        str(worker.session_name).strip() or reviewer_spec.role_name,
    )
    ensure_review_artifacts(review_md_path, review_json_path)
    message(render_tmux_start_summary(str(worker.session_name).strip() or reviewer_spec.role_name, worker))
    return ReviewerRuntime(
        reviewer_name=reviewer_identity,
        selection=selection,
        worker=worker,
        review_md_path=review_md_path,
        review_json_path=review_json_path,
        contract=build_placeholder_reviewer_contract(review_json_path),
    )


def build_reviewer_workers(
    args: argparse.Namespace,
    *,
    project_dir: str | Path,
    requirement_name: str,
    reviewer_specs: Sequence[DevelopmentReviewerSpec],
    reviewer_selections_by_name: dict[str, ReviewAgentSelection] | None = None,
    progress: ReviewStageProgress | None = None,
    launch_coordinator: LaunchCoordinator | None = None,
) -> list[ReviewerRuntime]:
    reviewers: list[ReviewerRuntime] = []
    predicted_session_names: set[str] = set()
    interactive = stdin_is_interactive()
    if progress is not None:
        progress.set_phase("任务开发 / 启动审核器")
    agent_config = resolve_stage_agent_config(args)
    for reviewer_spec in reviewer_specs:
        reviewer_display_name = _predict_worker_display_name(
            project_dir=project_dir,
            worker_id=build_development_reviewer_worker_id(reviewer_spec.role_name),
            occupied_session_names=sorted(predicted_session_names),
        )
        predicted_session_names.add(reviewer_display_name)
        reviewer_key = str(reviewer_spec.reviewer_key or reviewer_spec.role_name).strip()
        selection = (reviewer_selections_by_name or {}).get(reviewer_key) or agent_config.reviewer_selection(reviewer_key)
        if selection is None and interactive:
            selection = prompt_review_agent_selection(
                DEFAULT_REQUIREMENTS_CLARIFICATION_VENDOR,
                default_model=DEFAULT_REQUIREMENTS_CLARIFICATION_MODEL,
                default_reasoning_effort=DEFAULT_REQUIREMENTS_CLARIFICATION_EFFORT,
                default_proxy_url="",
                role_label=reviewer_display_name,
                progress=progress,
            )
            message(render_review_agent_selection(f"{reviewer_display_name} 配置", selection))
        elif selection is None:
            selection = _reviewer_default_selection()
        reviewers.append(
            create_reviewer_runtime(
                project_dir=project_dir,
                requirement_name=requirement_name,
                reviewer_spec=reviewer_spec,
                selection=selection,
                launch_coordinator=launch_coordinator,
            )
        )
    return reviewers


def collect_reviewer_agent_selections(
    *,
    project_dir: str | Path,
    reviewer_specs: Sequence[DevelopmentReviewerSpec],
    reserved_session_names: Sequence[str] = (),
    progress: ReviewStageProgress | None = None,
    allow_back_first_prompt: bool = False,
    stage_key: str = "development_reviewer_selection",
) -> dict[str, ReviewAgentSelection]:
    selections: dict[str, ReviewAgentSelection] = {}
    predicted_session_names: set[str] = {str(name).strip() for name in reserved_session_names if str(name).strip()}
    interactive = stdin_is_interactive()
    if progress is not None:
        progress.set_phase("任务开发 / 配置审核器模型")
    next_allow_back = bool(allow_back_first_prompt)
    for reviewer_spec in reviewer_specs:
        reviewer_key = str(reviewer_spec.reviewer_key or reviewer_spec.role_name).strip()
        reviewer_display_name = _predict_worker_display_name(
            project_dir=project_dir,
            worker_id=build_development_reviewer_worker_id(reviewer_spec.role_name),
            occupied_session_names=sorted(predicted_session_names),
        )
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
            selection = _reviewer_default_selection()
        selections[reviewer_key] = selection
    return selections


def _run_parallel_reviewer_initialization(
    reviewers: Sequence[ReviewerRuntime],
    *,
    project_dir: str | Path,
    requirement_name: str,
    paths: dict[str, Path],
    reviewer_specs_by_name: dict[str, DevelopmentReviewerSpec],
    progress: ReviewStageProgress | None = None,
) -> list[ReviewerRuntime]:
    reviewer_list = list(reviewers)
    if not reviewer_list:
        return reviewer_list
    if progress is not None:
        progress.set_phase("任务开发 / 初始化审核器")
    skip_budget = _ReadyTimeoutSkipBudget(len(reviewer_list))

    def run_initialization(reviewer: ReviewerRuntime) -> ReviewerRuntime | None:
        return _run_single_reviewer_initialization(
            reviewer,
            project_dir=project_dir,
            requirement_name=requirement_name,
            paths=paths,
            reviewer_specs_by_name=reviewer_specs_by_name,
            progress=progress,
            can_skip_ready_timeout=True,
            ready_timeout_skip_budget=skip_budget,
        )

    return run_parallel_reviewer_round(
        reviewer_list,
        key_func=lambda reviewer: reviewer.reviewer_name,
        run_turn=run_initialization,
        error_prefix="任务开发审核智能体初始化失败:",
    )


def _run_single_reviewer_initialization(
    reviewer: ReviewerRuntime,
    *,
    project_dir: str | Path,
    requirement_name: str,
    paths: dict[str, Path],
    reviewer_specs_by_name: dict[str, DevelopmentReviewerSpec],
    progress: ReviewStageProgress | None = None,
    can_skip_ready_timeout: bool = True,
    ready_timeout_skip_budget: _ReadyTimeoutSkipBudget | None = None,
) -> ReviewerRuntime | None:
    init_contract = build_reviewer_init_result_contract(paths)
    current_reviewer = reviewer
    while True:
        reviewer_spec = reviewer_specs_by_name[current_reviewer.reviewer_name]
        try:
            run_task_result_turn_with_repair(
                worker=current_reviewer.worker,
                label=f"development_reviewer_init_{sanitize_requirement_name(current_reviewer.reviewer_name)}",
                prompt=build_reviewer_init_prompt(paths, reviewer_spec=reviewer_spec),
                result_contract=init_contract,
                parse_result_payload=lambda text: _parse_result_payload(text) if str(text).strip() else {},
                turn_goal=_build_developer_turn_goal(paths, init_contract),
                timeout_sec=DEFAULT_COMMAND_TIMEOUT_SEC,
                stage_label="任务开发",
                role_label=str(current_reviewer.worker.session_name or current_reviewer.reviewer_name).strip() or current_reviewer.reviewer_name,
            )
            return current_reviewer
        except Exception as error:  # noqa: BLE001
            reviewer_display_name = str(current_reviewer.worker.session_name or current_reviewer.reviewer_name).strip() or current_reviewer.reviewer_name
            if is_worker_death_error(error):
                if _worker_appears_live_for_reviewer_recovery(current_reviewer.worker):
                    message(f"{reviewer_display_name} 当前仍存活但初始化失败，当前阶段将忽略该审核智能体。")
                    return None
                failure_reason = describe_reviewer_failure_reason(error, current_reviewer.worker)
                failed_reviewer = note_reviewer_failure(current_reviewer, reason_text=failure_reason)
                if reviewer_requires_manual_model_reconfiguration(failed_reviewer):
                    reason_text = build_reviewer_failure_reconfiguration_reason(
                        failed_reviewer,
                        role_label=reviewer_display_name,
                        failure_reason=failure_reason,
                    )
                    mark_worker_awaiting_reconfiguration(current_reviewer.worker, reason_text=reason_text)
                    replacement = recreate_development_reviewer_runtime(
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        reviewer=failed_reviewer,
                        reviewer_spec=reviewer_spec,
                        progress=progress,
                        force_model_change=True,
                        required_reconfiguration=True,
                        reason_text=reason_text,
                    )
                    if replacement is None:
                        raise RuntimeError(f"{reviewer_display_name} 初始化失败，且未能重建审核智能体") from error
                    current_reviewer = replacement
                    continue
                replacement = recreate_development_reviewer_runtime(
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    reviewer=failed_reviewer,
                    reviewer_spec=reviewer_spec,
                    progress=progress,
                    force_model_change=False,
                )
                if replacement is None:
                    message(f"{reviewer_display_name} 已死亡，当前阶段将忽略该审核智能体。")
                    return None
                current_reviewer = carry_reviewer_failure_state(replacement, previous=failed_reviewer)
                continue
            if is_agent_ready_timeout_error(error):
                effective_can_skip = bool(can_skip_ready_timeout)
                reserved_skip = False
                if ready_timeout_skip_budget is not None:
                    effective_can_skip = ready_timeout_skip_budget.reserve()
                    reserved_skip = effective_can_skip
                try:
                    choice = prompt_agent_ready_timeout_recovery(
                        current_reviewer.worker,
                        role_label=reviewer_display_name,
                        can_skip=effective_can_skip,
                        progress=progress,
                        reason_text=(
                            f"{reviewer_display_name}启动超时，未能进入可输入状态。\n"
                            + ("请先手动更换模型，或关闭并跳过该审核智能体。" if effective_can_skip else "请先手动更换模型，再继续尝试当前 turn。")
                        ),
                    )
                except Exception:
                    if reserved_skip:
                        ready_timeout_skip_budget.release()
                    raise
                if choice == AGENT_READY_TIMEOUT_SKIP and effective_can_skip:
                    with suppress(Exception):
                        current_reviewer.worker.request_kill()
                    message(f"{reviewer_display_name} 已关闭，当前阶段将跳过该审核智能体。")
                    return None
                if reserved_skip:
                    ready_timeout_skip_budget.release()
                if choice == AGENT_READY_TIMEOUT_RETRY:
                    continue
                continue
            if is_recoverable_startup_failure(error, current_reviewer.worker):
                reason_text = (
                    f"检测到{reviewer_display_name}仍在 agent 界面，但模型认证已失效。\n需要更换模型后继续当前阶段。"
                    if is_provider_auth_error(error) or worker_has_provider_auth_error(current_reviewer.worker)
                    else f"检测到{reviewer_display_name}的模型服务出现临时运行错误。\n需要更换或重启模型后继续当前阶段。"
                    if worker_has_provider_runtime_error(current_reviewer.worker)
                    else f"{reviewer_display_name}启动超时，未能进入可输入状态。\n需要更换模型后继续当前阶段。"
                )
                mark_worker_awaiting_reconfiguration(current_reviewer.worker, reason_text=reason_text)
                replacement = recreate_development_reviewer_runtime(
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    reviewer=current_reviewer,
                    reviewer_spec=reviewer_spec,
                    progress=progress,
                    force_model_change=True,
                    required_reconfiguration=True,
                    reason_text=reason_text,
                )
                if replacement is None:
                    raise RuntimeError(f"{reviewer_display_name} 初始化失败，且未能重建审核智能体") from error
                current_reviewer = replacement
                continue
            raise


def initialize_development_workers(
    developer: DeveloperRuntime,
    *,
    paths: dict[str, Path],
    reviewers: Sequence[ReviewerRuntime],
    reviewer_specs_by_name: dict[str, DevelopmentReviewerSpec],
    project_dir: str | Path = ".",
    requirement_name: str = "",
    initialize_developer: bool = True,
    initialize_reviewers: bool = True,
    progress: ReviewStageProgress | None = None,
    turn_policy: DeveloperTurnPolicy | None = None,
    replace_dead_developer=None,
    replace_dead_developer_for_init=None,
    audit_context: StageAuditRunContext | None = None,
) -> tuple[DeveloperRuntime, list[ReviewerRuntime]]:
    current_developer = developer
    reviewer_list = list(reviewers)
    if not initialize_developer and not initialize_reviewers:
        return current_developer, reviewer_list
    if progress is not None:
        progress.set_phase("任务开发 / 初始化中")
    if initialize_developer:
        developer_payload: dict[str, object] | None = None
        current_developer, developer_payload = _run_developer_result_turn(
            current_developer,
            label="development_developer_init",
            prompt=build_developer_init_prompt(paths, role_prompt=current_developer.role_prompt),
            result_contract=build_developer_init_result_contract(paths, mode="a07_developer_init"),
            paths=paths,
            turn_policy=turn_policy,
            replace_dead_developer=replace_dead_developer_for_init or replace_dead_developer,
            progress=progress,
        )
        if developer_payload is None:
            raise RuntimeError("开发工程师初始化未返回有效结果")
        current_developer = run_developer_hitl_loop(
            current_developer,
            paths=paths,
            initial_payload=developer_payload,
            progress=progress,
            turn_policy=turn_policy,
            replace_dead_developer=replace_dead_developer,
            audit_context=audit_context,
        )
    if initialize_reviewers:
        reviewer_list = _run_parallel_reviewer_initialization(
            reviewer_list,
            project_dir=project_dir,
            requirement_name=requirement_name,
            paths=paths,
            reviewer_specs_by_name=reviewer_specs_by_name,
            progress=progress,
        )
    return current_developer, reviewer_list


def _collect_development_hitl_response(
    question_path: str | Path,
    *,
    hitl_round: int,
    answer_path: str | Path | None = None,
    progress: ReviewStageProgress | None = None,
) -> str:
    question_file = Path(question_path).expanduser().resolve()
    question_text = get_markdown_content(question_file).strip()
    if not isinstance(get_terminal_ui(), BridgeTerminalUI):
        message()
        message(f"任务开发阶段 HITL 第 {hitl_round} 轮，需要人工补充信息")
        message(f"问题文档: {question_file}")
        message(question_text or "(问题文档为空)")
    if progress is not None:
        progress.set_phase("任务开发 / 等待 HITL")
    with progress.suspended() if progress is not None else nullcontext():
        return collect_multiline_input(
            title=f"HITL 第 {hitl_round} 轮回复",
            empty_retry_message="回复不能为空，请重新输入。",
            question_path=question_file,
            answer_path=answer_path,
            is_hitl=True,
        )


def run_developer_hitl_loop(
    developer: DeveloperRuntime,
    *,
    paths: dict[str, Path],
    initial_payload: dict[str, object],
    progress: ReviewStageProgress | None = None,
    turn_policy: DeveloperTurnPolicy | None = None,
    replace_dead_developer=None,
    audit_context: StageAuditRunContext | None = None,
) -> DeveloperRuntime:
    current_developer = developer
    payload = initial_payload
    for hitl_round in range(1, MAX_DEVELOPMENT_HITL_ROUNDS + 1):
        if str(payload.get("status", "")).strip() == "ready" and not get_markdown_content(paths["ask_human_path"]).strip():
            return current_developer
        if not get_markdown_content(paths["ask_human_path"]).strip():
            return current_developer
        if audit_context is not None:
            append_stage_audit_record(
                audit_context,
                event_type="hitl_question",
                source_paths={"ask_human": paths["ask_human_path"]},
                hitl_round_index=hitl_round,
                metadata={"trigger": "developer_init_hitl"},
            )
        human_msg = _collect_development_hitl_response(
            paths["ask_human_path"],
            hitl_round=hitl_round,
            answer_path=paths["hitl_record_path"],
            progress=progress,
        )
        if audit_context is not None:
            append_stage_audit_record(
                audit_context,
                event_type="hitl_answer",
                source_paths={
                    "human_answer": "",
                    "hitl_record": paths["hitl_record_path"],
                },
                hitl_round_index=hitl_round,
                metadata={
                    "trigger": "developer_init_hitl",
                    "human_answer_source": "runtime_payload",
                },
                snapshot_overrides={"human_answer": human_msg},
            )
            record_before_cleanup(
                audit_context,
                {"ask_human": paths["ask_human_path"]},
                metadata={"trigger": "developer_init_hitl_reply_prepare"},
                hitl_round_index=hitl_round,
            )
        ensure_empty_file(paths["ask_human_path"])
        current_developer, payload = _run_developer_result_turn(
            current_developer,
            label=f"development_developer_hitl_reply_round_{hitl_round}",
            prompt=build_developer_human_reply_prompt(paths, human_msg=human_msg),
            result_contract=build_developer_init_result_contract(paths, mode="a07_developer_human_reply"),
            paths=paths,
            turn_policy=turn_policy,
            replace_dead_developer=replace_dead_developer,
            progress=progress,
        )
    raise RuntimeError(f"任务开发 HITL 轮次超过上限: {MAX_DEVELOPMENT_HITL_ROUNDS}")


def run_development_review_limit_hitl_loop(
    developer: DeveloperRuntime,
    *,
    paths: dict[str, Path],
    task_name: str,
    review_msg: str,
    review_limit: int,
    review_rounds_used: int,
    progress: ReviewStageProgress | None = None,
    turn_policy: DeveloperTurnPolicy | None = None,
    replace_dead_developer=None,
    apply_human_review_override: Callable[[str], None] | None = None,
    human_input_provider=None,
    audit_context: StageAuditRunContext | None = None,
) -> object:
    if audit_context is not None:
        record_before_cleanup(
            audit_context,
            {"ask_human": paths["ask_human_path"]},
            metadata={"trigger": "development_review_limit_hitl_prepare"},
            task_name=task_name,
        )
    ensure_empty_file(paths["ask_human_path"])

    def audit_hitl_question(hitl_round: int, ask_human_file: Path) -> None:
        if audit_context is None:
            return
        append_stage_audit_record(
            audit_context,
            event_type="hitl_question",
            source_paths={"ask_human": ask_human_file},
            hitl_round_index=hitl_round,
            task_name=task_name,
            metadata={"trigger": "development_review_limit_hitl"},
        )

    def audit_hitl_answer(hitl_round: int, human_msg: str, hitl_record_file: Path) -> None:
        if audit_context is None:
            return
        append_stage_audit_record(
            audit_context,
            event_type="hitl_answer",
            source_paths={
                "human_answer": "",
                "hitl_record": hitl_record_file,
            },
            hitl_round_index=hitl_round,
            task_name=task_name,
            metadata={
                "trigger": "development_review_limit_hitl",
                "human_answer_source": "runtime_payload",
            },
            snapshot_overrides={"human_answer": human_msg},
        )

    def initial_turn() -> object:
        nonlocal developer
        developer, _ = _run_developer_result_turn(
            developer,
            label=f"development_review_limit_hitl_{sanitize_requirement_name(task_name)}",
            prompt=build_development_review_limit_force_hitl_prompt(
                paths=paths,
                task_name=task_name,
                review_msg=review_msg,
                review_limit=review_limit,
                review_rounds_used=review_rounds_used,
            ),
            result_contract=build_developer_review_feedback_contract(
                paths,
                mode="a07_developer_review_limit_force_hitl",
            ),
            paths=paths,
            task_name=task_name,
            turn_policy=turn_policy,
            replace_dead_developer=replace_dead_developer,
            progress=progress,
        )
        return developer

    def human_reply_turn(human_msg: str) -> object:
        nonlocal developer
        if (
            apply_human_review_override is not None
            and _development_review_human_override_requested(human_msg, task_name=task_name)
        ):
            apply_human_review_override(human_msg)
            ensure_empty_file(paths["ask_human_path"])
            return developer
        developer, _ = _run_developer_result_turn(
            developer,
            label=f"development_review_limit_human_reply_{sanitize_requirement_name(task_name)}",
            prompt=build_development_review_limit_human_reply_prompt(
                paths=paths,
                task_name=task_name,
                review_msg=review_msg,
                human_msg=human_msg,
            ),
            result_contract=build_developer_review_feedback_contract(
                paths,
                mode="a07_developer_review_limit_human_reply",
            ),
            paths=paths,
            task_name=task_name,
            turn_policy=turn_policy,
            replace_dead_developer=replace_dead_developer,
            progress=progress,
        )
        return developer

    result = run_review_limit_hitl_cycle(
        stage_label="任务开发评审超限",
        ask_human_path=paths["ask_human_path"],
        hitl_record_path=paths["hitl_record_path"],
        initial_turn=initial_turn,
        human_reply_turn=human_reply_turn,
        human_input_provider=human_input_provider,
        progress=progress,
        max_hitl_rounds=MAX_DEVELOPMENT_HITL_ROUNDS,
        on_hitl_question=audit_hitl_question,
        on_hitl_answer=audit_hitl_answer,
    )
    return result


def _run_reviewer_turn_with_resume(
    reviewer: ReviewerRuntime,
    *,
    task_name: str,
    label: str,
    prompt: str,
) -> ReviewerRuntime | None:
    while True:
        if _reviewer_has_materialized_outputs(reviewer, task_name):
            return reviewer
        ensure_review_artifacts(reviewer.review_md_path, reviewer.review_json_path)
        baseline_signature = _reviewer_artifact_signature(reviewer)
        try:
            run_completion_turn_with_repair(
                worker=reviewer.worker,
                label=label,
                prompt=prompt,
                completion_contract=build_reviewer_completion_contract(
                    reviewer_name=reviewer.reviewer_name,
                    task_name=task_name,
                    review_md_path=reviewer.review_md_path,
                    review_json_path=reviewer.review_json_path,
                ),
                turn_goal=_build_reviewer_turn_goal(),
                timeout_sec=DEFAULT_COMMAND_TIMEOUT_SEC,
                stage_label="任务开发",
                role_label=str(reviewer.worker.session_name or reviewer.reviewer_name).strip() or reviewer.reviewer_name,
                task_name=task_name,
            )
            return reviewer
        except Exception as error:  # noqa: BLE001
            if is_turn_artifact_contract_error(error):
                return reviewer
            if _reviewer_has_materialized_outputs(reviewer, task_name) and _reviewer_artifact_signature(reviewer) != baseline_signature:
                return reviewer
            if is_worker_death_error(error):
                message(f"{reviewer.worker.session_name or reviewer.reviewer_name} 已死亡，当前阶段将忽略该审核智能体。")
                return None
            if try_resume_worker(reviewer.worker, timeout_sec=60.0):
                continue
            message(f"{reviewer.worker.session_name or reviewer.reviewer_name} 无法继续，当前阶段将忽略该审核智能体。")
            return None


def recreate_development_reviewer_runtime(
    *,
    project_dir: str | Path,
    requirement_name: str,
    reviewer: ReviewerRuntime,
    reviewer_spec: DevelopmentReviewerSpec,
    progress: ReviewStageProgress | None = None,
    force_model_change: bool,
    required_reconfiguration: bool = False,
    reason_text: str = "",
) -> ReviewerRuntime | None:
    reviewer_display_name = str(reviewer.worker.session_name or reviewer.reviewer_name).strip() or reviewer.reviewer_name
    if force_model_change:
        if required_reconfiguration and not stdin_is_interactive():
            raise RuntimeError("审核智能体需要重新配置，但当前环境无法交互选择厂商/模型。")
        if not required_reconfiguration and not stdin_is_interactive():
            return None
        selection = (
            prompt_required_replacement_review_agent_selection(
                reason_text=reason_text or f"检测到{reviewer_display_name}不可继续使用，需要更换模型后重建审核智能体。",
                previous_selection=reviewer.selection,
                force_model_change=True,
                role_label=reviewer_display_name,
                progress=progress,
            )
            if required_reconfiguration
            else prompt_replacement_review_agent_selection(
                reason_text=f"检测到{reviewer_display_name}不可继续使用，需要更换模型后重建审核智能体。",
                previous_selection=reviewer.selection,
                force_model_change=True,
                role_label=reviewer_display_name,
                progress=progress,
            )
        )
        if selection is None:
            return None
    else:
        if not stdin_is_interactive():
            return None
        selection = prompt_replacement_review_agent_selection(
            reason_text=f"检测到{reviewer_display_name}已死亡，需要由人类决定是否重建审核智能体。",
            previous_selection=reviewer.selection,
            force_model_change=False,
            role_label=reviewer_display_name,
            progress=progress,
        )
        if selection is None:
            return None
    replacement = create_reviewer_runtime(
        project_dir=project_dir,
        requirement_name=requirement_name,
        reviewer_spec=reviewer_spec,
        selection=selection,
        launch_coordinator=getattr(reviewer.worker, "launch_coordinator", None),
    )
    if replacement.review_md_path != reviewer.review_md_path and reviewer.review_md_path.exists():
        reviewer.review_md_path.unlink()
    if replacement.review_json_path != reviewer.review_json_path and reviewer.review_json_path.exists():
        reviewer.review_json_path.unlink()
    with suppress(Exception):
        reviewer.worker.request_kill()
    runtime_dir_text = str(getattr(reviewer.worker, "runtime_dir", "") or "").strip()
    if runtime_dir_text:
        runtime_dir = Path(runtime_dir_text).expanduser().resolve()
        if runtime_dir.exists():
            shutil.rmtree(runtime_dir, ignore_errors=True)
    return replacement


def _worker_appears_live_for_reviewer_recovery(worker: object | None) -> bool:
    if worker is None:
        return False
    session_exists = getattr(worker, "session_exists", None)
    if callable(session_exists):
        try:
            if not session_exists():
                return False
        except Exception:
            pass
    read_state = getattr(worker, "read_state", None)
    if callable(read_state):
        try:
            state = read_state()
        except Exception:
            state = {}
        if isinstance(state, dict):
            agent_state = str(state.get("agent_state", "") or "").strip().upper()
            health_status = str(state.get("health_status", "") or "").strip().lower()
            if agent_state in {"READY", "BUSY", "STARTING"} and health_status not in {
                "dead",
                "missing_session",
                "pane_dead",
            }:
                return True
    get_agent_state = getattr(worker, "get_agent_state", None)
    if callable(get_agent_state):
        try:
            state = get_agent_state()
            return str(getattr(state, "value", state) or "").strip().upper() in {"READY", "BUSY", "STARTING"}
        except Exception:
            return False
    return False


def run_reviewer_turn_with_recreation(
    reviewer: ReviewerRuntime,
    *,
    project_dir: str | Path,
    requirement_name: str,
    task_name: str,
    reviewer_spec: DevelopmentReviewerSpec,
    paths: dict[str, Path],
    reviewer_specs_by_name: dict[str, DevelopmentReviewerSpec],
    label: str,
    prompt: str = "",
    prompt_builder=None,
    progress: ReviewStageProgress | None = None,
    can_skip_ready_timeout: bool = True,
    ready_timeout_skip_budget: _ReadyTimeoutSkipBudget | None = None,
    allow_existing_outputs: bool = True,
) -> ReviewerRuntime | None:
    current_reviewer = reviewer
    while True:
        if allow_existing_outputs and _reviewer_has_materialized_outputs(current_reviewer, task_name):
            return current_reviewer
        ensure_review_artifacts(current_reviewer.review_md_path, current_reviewer.review_json_path)
        baseline_signature = _reviewer_artifact_signature(current_reviewer)
        current_prompt = prompt_builder(current_reviewer) if prompt_builder is not None else prompt
        try:
            run_completion_turn_with_repair(
                worker=current_reviewer.worker,
                label=label,
                prompt=current_prompt,
                completion_contract=build_reviewer_completion_contract(
                    reviewer_name=current_reviewer.reviewer_name,
                    task_name=task_name,
                    review_md_path=current_reviewer.review_md_path,
                    review_json_path=current_reviewer.review_json_path,
                ),
                turn_goal=_build_reviewer_turn_goal(),
                timeout_sec=DEFAULT_COMMAND_TIMEOUT_SEC,
                stage_label="任务开发",
                role_label=str(current_reviewer.worker.session_name or current_reviewer.reviewer_name).strip() or current_reviewer.reviewer_name,
                task_name=task_name,
                requirement_name=requirement_name,
            )
            return current_reviewer
        except Exception as error:  # noqa: BLE001
            if is_turn_artifact_contract_error(error):
                return current_reviewer
            if _reviewer_has_materialized_outputs(current_reviewer, task_name) and _reviewer_artifact_signature(current_reviewer) != baseline_signature:
                return current_reviewer
            reviewer_display_name = str(current_reviewer.worker.session_name or current_reviewer.reviewer_name).strip() or current_reviewer.reviewer_name
            auth_error = is_provider_auth_error(error) or worker_has_provider_auth_error(current_reviewer.worker)
            provider_runtime_error = worker_has_provider_runtime_error(current_reviewer.worker)
            ready_timeout_error = is_agent_ready_timeout_error(error)
            if ready_timeout_error:
                effective_can_skip = bool(can_skip_ready_timeout)
                reserved_skip = False
                if ready_timeout_skip_budget is not None:
                    effective_can_skip = ready_timeout_skip_budget.reserve()
                    reserved_skip = effective_can_skip
                try:
                    choice = prompt_agent_ready_timeout_recovery(
                        current_reviewer.worker,
                        role_label=reviewer_display_name,
                        can_skip=effective_can_skip,
                        progress=progress,
                        reason_text=(
                            f"{reviewer_display_name}启动超时，未能进入可输入状态。\n"
                            + ("请先手动更换模型，或关闭并跳过该审核智能体。" if effective_can_skip else "请先手动更换模型，再继续尝试当前 turn。")
                        ),
                    )
                except Exception:
                    if reserved_skip:
                        ready_timeout_skip_budget.release()
                    raise
                if choice == AGENT_READY_TIMEOUT_SKIP and effective_can_skip:
                    with suppress(Exception):
                        current_reviewer.worker.request_kill()
                    message(f"{reviewer_display_name} 已关闭，当前阶段将跳过该审核智能体。")
                    return None
                if reserved_skip:
                    ready_timeout_skip_budget.release()
                if choice == AGENT_READY_TIMEOUT_RETRY:
                    continue
                continue
            if auth_error or provider_runtime_error:
                reason_text = (
                    f"检测到{reviewer_display_name}仍在 agent 界面，但模型认证已失效。\n需要更换模型后继续当前阶段。"
                    if auth_error
                    else f"检测到{reviewer_display_name}的模型服务出现临时运行错误。\n需要更换或重启模型后继续当前阶段。"
                )
                mark_worker_awaiting_reconfiguration(current_reviewer.worker, reason_text=reason_text)
                replacement = recreate_development_reviewer_runtime(
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    reviewer=current_reviewer,
                    reviewer_spec=reviewer_spec,
                    progress=progress,
                    force_model_change=True,
                    required_reconfiguration=True,
                    reason_text=reason_text,
                )
                if replacement is None:
                    raise RuntimeError(f"{reviewer_display_name} 无法继续，且未能重建审核智能体") from error
                initialized = _run_single_reviewer_initialization(
                    replacement,
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    paths=paths,
                    reviewer_specs_by_name=reviewer_specs_by_name,
                    progress=progress,
                    can_skip_ready_timeout=can_skip_ready_timeout,
                    ready_timeout_skip_budget=ready_timeout_skip_budget,
                )
                if initialized is None:
                    raise RuntimeError(f"{reviewer_display_name} 重建后初始化失败") from error
                current_reviewer = initialized
                continue
            if is_worker_death_error(error):
                if _worker_appears_live_for_reviewer_recovery(current_reviewer.worker):
                    message(f"{reviewer_display_name} 当前仍存活但本轮执行异常，当前阶段将忽略该审核智能体。")
                    return None
                failure_reason = describe_reviewer_failure_reason(error, current_reviewer.worker)
                failed_reviewer = note_reviewer_failure(current_reviewer, reason_text=failure_reason)
                if reviewer_requires_manual_model_reconfiguration(failed_reviewer):
                    reason_text = build_reviewer_failure_reconfiguration_reason(
                        failed_reviewer,
                        role_label=reviewer_display_name,
                        failure_reason=failure_reason,
                    )
                    mark_worker_awaiting_reconfiguration(current_reviewer.worker, reason_text=reason_text)
                    replacement = recreate_development_reviewer_runtime(
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        reviewer=failed_reviewer,
                        reviewer_spec=reviewer_spec,
                        progress=progress,
                        force_model_change=True,
                        required_reconfiguration=True,
                        reason_text=reason_text,
                    )
                else:
                    replacement = recreate_development_reviewer_runtime(
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        reviewer=failed_reviewer,
                        reviewer_spec=reviewer_spec,
                        progress=progress,
                        force_model_change=False,
                    )
                if replacement is None:
                    message(f"{reviewer_display_name} 已死亡，当前阶段将忽略该审核智能体。")
                    return None
                if not reviewer_requires_manual_model_reconfiguration(failed_reviewer):
                    replacement = carry_reviewer_failure_state(replacement, previous=failed_reviewer)
                initialized = _run_single_reviewer_initialization(
                    replacement,
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    paths=paths,
                    reviewer_specs_by_name=reviewer_specs_by_name,
                    progress=progress,
                    can_skip_ready_timeout=can_skip_ready_timeout,
                    ready_timeout_skip_budget=ready_timeout_skip_budget,
                )
                if initialized is None:
                    message(f"{reviewer_display_name} 已死亡，重建后初始化失败，当前阶段将忽略该审核智能体。")
                    return None
                current_reviewer = initialized
                continue
            if try_resume_worker(current_reviewer.worker, timeout_sec=60.0):
                continue
            if _worker_appears_live_for_reviewer_recovery(current_reviewer.worker):
                message(f"{reviewer_display_name} 当前仍存活但上一轮执行异常，当前阶段将忽略该审核智能体。")
                return None
            replacement = recreate_development_reviewer_runtime(
                project_dir=project_dir,
                requirement_name=requirement_name,
                reviewer=current_reviewer,
                reviewer_spec=reviewer_spec,
                progress=progress,
                force_model_change=False,
            )
            if replacement is None:
                message(f"{reviewer_display_name} 未重建，当前阶段将忽略该审核智能体。")
                return None
            initialized = _run_single_reviewer_initialization(
                replacement,
                project_dir=project_dir,
                requirement_name=requirement_name,
                paths=paths,
                reviewer_specs_by_name=reviewer_specs_by_name,
                progress=progress,
                can_skip_ready_timeout=can_skip_ready_timeout,
                ready_timeout_skip_budget=ready_timeout_skip_budget,
            )
            if initialized is None:
                raise RuntimeError(f"{reviewer_display_name} 重建后初始化失败") from error
            current_reviewer = initialized


def _run_parallel_reviewers(
    reviewers: Sequence[ReviewerRuntime],
    *,
    project_dir: str | Path,
    requirement_name: str,
    paths: dict[str, Path],
    reviewer_specs_by_name: dict[str, DevelopmentReviewerSpec],
    task_name: str,
    round_index: int,
    prompt_builder,
    label_prefix: str,
    progress: ReviewStageProgress | None = None,
    allow_existing_outputs: bool = True,
) -> list[ReviewerRuntime]:
    if progress is not None:
        progress.set_phase(f"任务开发 / {task_name} 评审第 {round_index} 轮")
    reviewer_list = list(reviewers)
    skip_budget = _ReadyTimeoutSkipBudget(len(reviewer_list))

    def run_review_turn(reviewer: ReviewerRuntime) -> ReviewerRuntime | None:
        return run_reviewer_turn_with_recreation(
            reviewer,
            project_dir=project_dir,
            requirement_name=requirement_name,
            task_name=task_name,
            reviewer_spec=reviewer_specs_by_name[reviewer.reviewer_name],
            paths=paths,
            reviewer_specs_by_name=reviewer_specs_by_name,
            label=f"{label_prefix}_{sanitize_requirement_name(reviewer.reviewer_name)}_round_{round_index}",
            prompt_builder=prompt_builder,
            progress=progress,
            can_skip_ready_timeout=True,
            ready_timeout_skip_budget=skip_budget,
            allow_existing_outputs=allow_existing_outputs,
        )

    return run_parallel_reviewer_round(
        reviewer_list,
        key_func=lambda reviewer: reviewer.reviewer_name,
        run_turn=run_review_turn,
        error_prefix=f"{task_name} 代码评审智能体执行失败:",
    )


def repair_reviewer_outputs(
    reviewers: Sequence[ReviewerRuntime],
    *,
    paths: dict[str, Path],
    reviewer_specs_by_name: dict[str, DevelopmentReviewerSpec],
    project_dir: str | Path,
    requirement_name: str,
    task_name: str,
    round_index: int,
    progress: ReviewStageProgress | None = None,
) -> list[ReviewerRuntime]:
    json_pattern = f"{sanitize_requirement_name(requirement_name)}_评审记录_*.json"
    md_pattern = f"{sanitize_requirement_name(requirement_name)}_代码评审记录_*.md"
    reviewer_list = list(reviewers)
    skip_budget = _ReadyTimeoutSkipBudget(len(reviewer_list))

    def run_fix_turn(reviewer: ReviewerRuntime, fix_prompt: str, repair_attempt: int) -> ReviewerRuntime | None:
        return run_reviewer_turn_with_recreation(
            reviewer,
            project_dir=project_dir,
            requirement_name=requirement_name,
            task_name=task_name,
            reviewer_spec=reviewer_specs_by_name[reviewer.reviewer_name],
            paths=paths,
            reviewer_specs_by_name=reviewer_specs_by_name,
            label=f"development_review_fix_{sanitize_requirement_name(task_name)}_{sanitize_requirement_name(reviewer.reviewer_name)}_round_{round_index}_attempt_{repair_attempt}",
            prompt=fix_prompt,
            progress=progress,
            can_skip_ready_timeout=True,
            ready_timeout_skip_budget=skip_budget,
        )

    return repair_reviewer_round_outputs(
        reviewer_list,
        key_func=lambda reviewer: reviewer.reviewer_name,
        artifact_name_func=lambda reviewer: str(reviewer.worker.session_name).strip() or reviewer.reviewer_name,
        check_job=lambda reviewer_names: check_reviewer_job(
            reviewer_names,
            directory=project_dir,
            task_name=task_name,
            json_pattern=json_pattern,
            md_pattern=md_pattern,
        ),
        run_fix_turn=run_fix_turn,
        max_attempts=MAX_REVIEWER_REPAIR_ATTEMPTS,
        error_prefix=f"{task_name} 代码评审智能体修复输出失败:",
        final_error="代码评审智能体多次修复后仍未按协议更新文档",
        stage_label="任务开发",
        progress=progress,
        recreate_reviewer=lambda reviewer: recreate_development_reviewer_runtime(
            project_dir=project_dir,
            requirement_name=requirement_name,
            reviewer=reviewer,
            reviewer_spec=reviewer_specs_by_name[reviewer.reviewer_name],
            progress=progress,
            force_model_change=False,
        ),
    )


def prepare_review_round_artifacts(
    paths: dict[str, Path],
    reviewers: Sequence[ReviewerRuntime],
    *,
    task_name: str = "",
    audit_context: StageAuditRunContext | None = None,
    review_round_index: int | None = None,
    preserve_existing_outputs: bool = True,
) -> None:
    if audit_context is not None:
        record_before_cleanup(
            audit_context,
            {"merged_review": paths["merged_review_path"]},
            metadata={"trigger": "prepare_review_round_artifacts"},
            reviewer_markdown_paths=[reviewer.review_md_path for reviewer in reviewers],
            reviewer_json_paths=[reviewer.review_json_path for reviewer in reviewers],
            review_round_index=review_round_index,
            task_name=task_name,
        )
    ensure_empty_file(paths["merged_review_path"])
    for reviewer in reviewers:
        if preserve_existing_outputs and task_name and _reviewer_has_materialized_outputs(reviewer, task_name):
            continue
        ensure_review_artifacts(reviewer.review_md_path, reviewer.review_json_path)


def _active_reviewer_files(reviewers: Sequence[ReviewerRuntime]) -> tuple[list[str], list[str]]:
    json_files = [str(reviewer.review_json_path.resolve()) for reviewer in reviewers]
    md_files = [str(reviewer.review_md_path.resolve()) for reviewer in reviewers]
    return json_files, md_files


def _reviewer_audit_metadata(reviewers: Sequence[ReviewerRuntime], *, trigger: str = "") -> dict[str, object]:
    agent_names: list[str] = []
    session_names: list[str] = []
    for reviewer in reviewers:
        reviewer_name = str(getattr(reviewer, "reviewer_name", "") or "").strip()
        if reviewer_name:
            agent_names.append(reviewer_name)
        session_name = str(getattr(getattr(reviewer, "worker", None), "session_name", "") or "").strip()
        if session_name:
            session_names.append(session_name)
    return {
        "trigger": trigger,
        "agent_names": agent_names,
        "session_names": session_names,
    }


def _development_review_human_override_requested(human_msg: str, *, task_name: str) -> bool:
    normalized = " ".join(str(human_msg or "").lower().split())
    if not normalized:
        return False

    normalized_task = str(task_name or "").strip().lower()
    task_scoped = (
        bool(normalized_task and normalized_task in normalized)
        or "当前任务" in normalized
        or "本次" in normalized
        or "该任务" in normalized
    )
    if not task_scoped:
        return False

    pass_markers = (
        "人工验收通过",
        "人工确认通过",
        "人工已验收",
        "人工已通过",
        "人类已使用",
        "免检通过",
        "强行闭环",
        "强制通过",
        "视为通过",
        "按通过处理",
        "按通过继续",
        "不阻塞",
        "不再阻塞",
        "human verified pass",
    )
    json_pass_marker = "review_pass" in normalized and (
        "true" in normalized
        or "置为 true" in normalized
        or "设为 true" in normalized
        or "统一置为 true" in normalized
    )
    if not (any(marker in normalized for marker in pass_markers) or json_pass_marker):
        return False

    context_markers = (
        "评审环境",
        "工具链",
        "误报",
        "false positive",
        "py_compile",
        "无法复现",
        "不可复现",
        "no module named",
        "pytest",
        "忽略该失败评审",
        "忽略该类重复失败评审",
        "禁止修改",
        "不要修改",
        "不新增需求",
        "非行动项",
        "范围外",
        "不再要求评审",
        "不再针对",
    )
    return any(marker in normalized for marker in context_markers)


def _mark_review_task_passed(review_json_path: str | Path, *, task_name: str) -> None:
    target = Path(review_json_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    task_key = str(task_name or "").strip()
    records: list[object] = []
    if target.exists() and target.read_text(encoding="utf-8").strip():
        payload = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = [payload]

    found = False
    for record in records:
        if isinstance(record, dict) and str(record.get("task_name", "")).strip() == task_key:
            record["review_pass"] = True
            found = True
    if not found:
        records.append({"task_name": task_key, "review_pass": True})

    target.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def apply_development_review_human_override(
    *,
    task_name: str,
    reviewer_workers: Sequence[ReviewerRuntime],
    merged_review_path: str | Path,
) -> None:
    for reviewer in reviewer_workers:
        _mark_review_task_passed(reviewer.review_json_path, task_name=task_name)
        ensure_empty_file(reviewer.review_md_path)
    ensure_empty_file(merged_review_path)


def _replace_dead_developer(
    developer: DeveloperRuntime,
    *,
    project_dir: str | Path,
    requirement_name: str = "",
    progress: ReviewStageProgress | None = None,
    launch_coordinator: LaunchCoordinator | None = None,
    error: Exception | None = None,
) -> DeveloperRuntime:
    developer_name = str(developer.worker.session_name or "开发工程师").strip() or "开发工程师"
    if error is not None and is_agent_ready_timeout_error(error):
        reason_text = (
            f"{developer_name}启动超时，未能进入可输入状态。\n"
            "请先手动更换模型，再继续尝试当前 turn。"
        )
        prompt_agent_ready_timeout_recovery(
            developer.worker,
            role_label=developer_name,
            can_skip=False,
            progress=progress,
            reason_text=reason_text,
        )
        return developer
    startup_reconfigure = bool(error is not None and is_recoverable_startup_failure(error, developer.worker))
    if startup_reconfigure:
        reason_text = (
            f"检测到{developer_name}仍在 agent 界面，但模型认证已失效。\n需要更换模型后继续当前阶段。"
            if is_provider_auth_error(error) or worker_has_provider_auth_error(developer.worker)
            else f"检测到{developer_name}的模型服务出现临时运行错误。\n需要更换或重启模型后继续当前阶段。"
            if worker_has_provider_runtime_error(developer.worker)
            else f"{developer_name}启动超时，未能进入可输入状态。\n需要更换模型后继续当前阶段。"
        )
        mark_worker_awaiting_reconfiguration(developer.worker, reason_text=reason_text)
    replacement = recreate_developer_runtime(
        project_dir=project_dir,
        requirement_name=requirement_name,
        developer=developer,
        progress=progress,
        launch_coordinator=launch_coordinator,
        required_reconfiguration=True,
        reason_text=(
            reason_text
            if startup_reconfigure
            else f"{developer_name}已死亡，必须重新选择厂商/模型/推理/代理后从当前阶段继续。"
        ),
    )
    if replacement is None:
        raise RuntimeError(f"{developer_name} 已死亡，且未能重建开发工程师")
    return replacement


def _bootstrap_developer_runtime(
    developer: DeveloperRuntime,
    *,
    paths: dict[str, Path],
    reviewer_specs_by_name: dict[str, DevelopmentReviewerSpec],
    project_dir: str | Path,
    requirement_name: str = "",
    progress: ReviewStageProgress | None = None,
    turn_policy: DeveloperTurnPolicy | None = None,
    launch_coordinator: LaunchCoordinator | None = None,
) -> DeveloperRuntime:
    bootstrapped_developer, _ = initialize_development_workers(
        developer,
        project_dir=project_dir,
        requirement_name=requirement_name,
        paths=paths,
        reviewers=(),
        reviewer_specs_by_name=reviewer_specs_by_name,
        initialize_developer=True,
        initialize_reviewers=False,
        progress=progress,
        turn_policy=turn_policy,
        replace_dead_developer=lambda active_developer, error: _replace_dead_developer_with_bootstrap(
            active_developer,
            paths=paths,
            reviewer_specs_by_name=reviewer_specs_by_name,
            project_dir=project_dir,
            requirement_name=requirement_name,
            progress=progress,
            turn_policy=turn_policy,
            launch_coordinator=launch_coordinator,
            error=error,
        ),
        replace_dead_developer_for_init=lambda active_developer, error: _replace_dead_developer(
            active_developer,
            project_dir=project_dir,
            requirement_name=requirement_name,
            progress=progress,
            launch_coordinator=launch_coordinator,
            error=error,
        ),
    )
    return bootstrapped_developer


def _replace_dead_developer_with_bootstrap(
    developer: DeveloperRuntime,
    *,
    paths: dict[str, Path],
    reviewer_specs_by_name: dict[str, DevelopmentReviewerSpec],
    project_dir: str | Path,
    requirement_name: str = "",
    progress: ReviewStageProgress | None = None,
    turn_policy: DeveloperTurnPolicy | None = None,
    launch_coordinator: LaunchCoordinator | None = None,
    error: Exception | None = None,
) -> DeveloperRuntime:
    replacement = _replace_dead_developer(
        developer,
        project_dir=project_dir,
        requirement_name=requirement_name,
        progress=progress,
        launch_coordinator=launch_coordinator,
        error=error,
    )
    return _bootstrap_developer_runtime(
        replacement,
        paths=paths,
        reviewer_specs_by_name=reviewer_specs_by_name,
        project_dir=project_dir,
        requirement_name=requirement_name,
        progress=progress,
        turn_policy=turn_policy,
        launch_coordinator=launch_coordinator,
    )


def resolve_subagent_num(
    args: argparse.Namespace,
    *,
    progress: ReviewStageProgress | None = None,
    allow_back: bool = False,
) -> int:
    explicit = getattr(args, "subagent_num", None)
    if explicit is not None:
        if int(explicit) < 0:
            raise RuntimeError("--subagent-num 必须是大于等于 0 的整数。")
        return int(explicit)
    if not stdin_is_interactive():
        return 0
    if progress is not None:
        progress.set_phase("任务开发 / 配置 subagent 数量")
    with progress.suspended() if progress is not None else nullcontext():
        while True:
            with prompt_metadata(
                allow_back=allow_back,
                back_value=PROMPT_BACK_VALUE,
                stage_key="development",
                stage_step_index=5,
            ):
                value = prompt_with_default("输入开发工程师可用的 subagent 数量", "0")
            if value.isdigit():
                return int(value)
            message("请输入大于等于 0 的整数。")


def ensure_developer_metadata(
    developer: DeveloperRuntime,
    *,
    paths: dict[str, Path],
    task_name: str,
    label_prefix: str,
    progress: ReviewStageProgress | None = None,
    turn_policy: DeveloperTurnPolicy | None = None,
    replace_dead_developer=None,
) -> tuple[DeveloperRuntime, str]:
    current_developer = developer
    for attempt in range(0, MAX_DEVELOPER_METADATA_REPAIR_ATTEMPTS + 1):
        code_change = get_markdown_content(paths["developer_output_path"]).strip()
        reminder_prompt = check_develop_job(
            paths["developer_output_path"],
            task_name,
            task_split_md=str(paths["task_md_path"].resolve()),
            what_just_dev=str(paths["developer_output_path"].resolve()),
        )
        if not reminder_prompt and code_change:
            return current_developer, code_change
        if attempt >= MAX_DEVELOPER_METADATA_REPAIR_ATTEMPTS:
            break
        current_developer, _ = _run_developer_result_turn(
            current_developer,
            label=f"{label_prefix}_metadata_repair_attempt_{attempt + 1}",
            prompt=reminder_prompt,
            result_contract=build_developer_task_complete_result_contract(paths),
            paths=paths,
            task_name=task_name,
            turn_policy=turn_policy,
            replace_dead_developer=replace_dead_developer,
            progress=progress,
        )
    raise RuntimeError(f"{task_name} 开发完成后，开发工程师仍未按协议更新《{paths['developer_output_path'].name}》")


def develop_current_task(
    developer: DeveloperRuntime,
    *,
    paths: dict[str, Path],
    task_name: str,
    subagent_num: int,
    progress: ReviewStageProgress | None = None,
    turn_policy: DeveloperTurnPolicy | None = None,
    replace_dead_developer=None,
    audit_context: StageAuditRunContext | None = None,
) -> tuple[DeveloperRuntime, str]:
    if audit_context is not None:
        record_before_cleanup(
            audit_context,
            {"developer_output": paths["developer_output_path"]},
            metadata={"trigger": "develop_current_task_prepare"},
            task_name=task_name,
        )
    ensure_empty_file(paths["developer_output_path"])
    if progress is not None:
        progress.set_phase(f"任务开发 / 开发中 | {task_name}")
    built_turn = build_prompt_task_turn(
        start_develop,
        task_name,
        hitl_record_md=str(paths["hitl_record_path"].resolve()),
        requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
        detailed_design_md=str(paths["detailed_design_path"].resolve()),
        task_split_md=str(paths["task_md_path"].resolve()),
        what_just_dev=str(paths["developer_output_path"].resolve()),
        sub_agent_num=subagent_num,
        stage_name="任务开发",
    )
    if built_turn.task_result_contract is None or built_turn.task_turn_goal is None:
        raise AssertionError("A07 developer prompt contract was not generated")
    current_developer, _ = _run_developer_result_turn(
        developer,
        label=f"development_start_{sanitize_requirement_name(task_name)}",
        prompt=built_turn.prompt,
        result_contract=built_turn.task_result_contract,
        paths=paths,
        task_name=task_name,
        turn_goal=built_turn.task_turn_goal,
        turn_policy=turn_policy,
        replace_dead_developer=replace_dead_developer,
        progress=progress,
    )
    result = ensure_developer_metadata(
        current_developer,
        paths=paths,
        task_name=task_name,
        label_prefix=f"development_start_{sanitize_requirement_name(task_name)}",
        progress=progress,
        turn_policy=turn_policy,
        replace_dead_developer=replace_dead_developer,
    )
    if audit_context is not None:
        append_stage_audit_record(
            audit_context,
            event_type="developer_output",
            source_paths={"developer_output": paths["developer_output_path"]},
            task_name=task_name,
            metadata={"trigger": "develop_current_task"},
        )
    return result


def refine_current_task(
    developer: DeveloperRuntime,
    *,
    paths: dict[str, Path],
    task_name: str,
    review_msg: str,
    progress: ReviewStageProgress | None = None,
    turn_policy: DeveloperTurnPolicy | None = None,
    replace_dead_developer=None,
    audit_context: StageAuditRunContext | None = None,
    review_round_index: int | None = None,
) -> tuple[DeveloperRuntime, str]:
    if audit_context is not None:
        record_before_cleanup(
            audit_context,
            {"developer_output": paths["developer_output_path"]},
            metadata={"trigger": "refine_current_task_prepare"},
            review_round_index=review_round_index,
            task_name=task_name,
        )
    ensure_empty_file(paths["developer_output_path"])
    if progress is not None:
        progress.set_phase(f"任务开发 / 修订中 | {task_name}")
    built_turn = build_prompt_task_turn(
        refine_code,
        review_msg,
        task_name,
        detailed_design_md=str(paths["detailed_design_path"].resolve()),
        what_just_dev=str(paths["developer_output_path"].resolve()),
        stage_name="任务开发",
    )
    if built_turn.task_result_contract is None or built_turn.task_turn_goal is None:
        raise AssertionError("A07 developer refine prompt contract was not generated")
    current_developer, _ = _run_developer_result_turn(
        developer,
        label=f"development_refine_{sanitize_requirement_name(task_name)}",
        prompt=built_turn.prompt,
        result_contract=built_turn.task_result_contract,
        paths=paths,
        task_name=task_name,
        turn_goal=built_turn.task_turn_goal,
        turn_policy=turn_policy,
        replace_dead_developer=replace_dead_developer,
        progress=progress,
    )
    result = ensure_developer_metadata(
        current_developer,
        paths=paths,
        task_name=task_name,
        label_prefix=f"development_refine_{sanitize_requirement_name(task_name)}",
        progress=progress,
        turn_policy=turn_policy,
        replace_dead_developer=replace_dead_developer,
    )
    if audit_context is not None:
        append_stage_audit_record(
            audit_context,
            event_type="developer_output",
            source_paths={"developer_output": paths["developer_output_path"]},
            review_round_index=review_round_index,
            task_name=task_name,
            metadata={"trigger": "refine_current_task"},
        )
        append_stage_audit_record(
            audit_context,
            event_type="change_after_review",
            source_paths={
                "developer_output": paths["developer_output_path"],
                "merged_review": paths["merged_review_path"],
            },
            review_round_index=review_round_index,
            task_name=task_name,
            metadata={"trigger": "refine_current_task"},
        )
    return result


def run_first_task_with_parallel_reviewer_init(
    developer: DeveloperRuntime,
    reviewers: Sequence[ReviewerRuntime],
    *,
    paths: dict[str, Path],
    reviewer_specs_by_name: dict[str, DevelopmentReviewerSpec],
    task_name: str,
    subagent_num: int,
    project_dir: str | Path = ".",
    requirement_name: str = "",
    progress: ReviewStageProgress | None = None,
    turn_policy: DeveloperTurnPolicy | None = None,
    replace_dead_developer=None,
    audit_context: StageAuditRunContext | None = None,
) -> tuple[DeveloperRuntime, str, list[ReviewerRuntime]]:
    reviewer_list = list(reviewers)
    if not reviewer_list:
        current_developer, code_change = develop_current_task(
            developer,
            paths=paths,
            task_name=task_name,
            subagent_num=subagent_num,
            progress=progress,
            turn_policy=turn_policy,
            replace_dead_developer=replace_dead_developer,
            audit_context=audit_context,
        )
        return current_developer, code_change, reviewer_list

    results: dict[str, object] = {}
    errors: dict[str, Exception] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_map = {
            executor.submit(
                develop_current_task,
                developer,
                paths=paths,
                task_name=task_name,
                subagent_num=subagent_num,
                progress=progress,
                turn_policy=turn_policy,
                replace_dead_developer=replace_dead_developer,
                audit_context=audit_context,
            ): "developer_task",
            executor.submit(
                initialize_development_workers,
                developer,
                project_dir=project_dir,
                requirement_name=requirement_name,
                paths=paths,
                reviewers=reviewer_list,
                reviewer_specs_by_name=reviewer_specs_by_name,
                initialize_developer=False,
                initialize_reviewers=True,
                progress=None,
                turn_policy=turn_policy,
                replace_dead_developer=replace_dead_developer,
            ): "reviewer_init",
        }
        for future in as_completed(future_map):
            branch = future_map[future]
            try:
                results[branch] = future.result()
            except Exception as error:  # noqa: BLE001
                errors[branch] = error
    if errors:
        if len(errors) == 1:
            raise next(iter(errors.values()))
        summary = "\n".join(f"{branch}: {error}" for branch, error in errors.items())
        raise RuntimeError(f"{task_name} 首轮开发与审核器初始化并行执行失败:\n{summary}")
    current_developer, code_change = results["developer_task"]  # type: ignore[misc]
    _, initialized_reviewers = results["reviewer_init"]  # type: ignore[misc]
    return current_developer, code_change, list(initialized_reviewers)


def recreate_development_workers(
    developer: DeveloperRuntime,
    reviewers: Sequence[ReviewerRuntime],
    *,
    project_dir: str | Path,
    requirement_name: str,
    paths: dict[str, Path],
    reviewer_specs_by_name: dict[str, DevelopmentReviewerSpec],
    progress: ReviewStageProgress | None = None,
    turn_policy: DeveloperTurnPolicy | None = None,
    launch_coordinator: LaunchCoordinator | None = None,
) -> tuple[DeveloperRuntime, list[ReviewerRuntime], tuple[str, ...]]:
    if progress is not None:
        progress.set_phase("任务开发 / 重建智能体")
    reviewer_selections = {
        reviewer.reviewer_name: reviewer.selection
        for reviewer in reviewers
    }
    cleanup_paths = list(
        _shutdown_workers(
            developer,
            reviewers,
            project_dir=project_dir,
            requirement_name=requirement_name,
            cleanup_runtime=True,
        )
    )
    cleanup_paths.extend(cleanup_existing_development_artifacts(paths, requirement_name))
    recreated_developer = create_developer_runtime(
        project_dir=project_dir,
        requirement_name=requirement_name,
        selection=developer.selection,
        role_prompt=developer.role_prompt,
        launch_coordinator=launch_coordinator,
    )
    recreated_reviewers: list[ReviewerRuntime] = []
    for reviewer in reviewers:
        reviewer_spec = reviewer_specs_by_name[reviewer.reviewer_name]
        recreated_reviewers.append(
            create_reviewer_runtime(
                project_dir=project_dir,
                requirement_name=requirement_name,
                reviewer_spec=reviewer_spec,
                selection=reviewer_selections[reviewer.reviewer_name],
                launch_coordinator=launch_coordinator,
            )
        )
    if turn_policy is not None:
        turn_policy.reset()
    recreated_developer = _bootstrap_developer_runtime(
        recreated_developer,
        paths=paths,
        reviewer_specs_by_name=reviewer_specs_by_name,
        project_dir=project_dir,
        requirement_name=requirement_name,
        progress=progress,
        turn_policy=turn_policy,
        launch_coordinator=launch_coordinator,
    )
    return recreated_developer, recreated_reviewers, tuple(dict.fromkeys(cleanup_paths))


def _export_developer_handoff(developer: DeveloperRuntime | None) -> DevelopmentAgentHandoff | None:
    if developer is None:
        return None
    return DevelopmentAgentHandoff(
        selection=developer.selection,
        role_prompt=developer.role_prompt,
        worker=developer.worker,
    )


def _export_reviewer_handoff(
    reviewers: Sequence[ReviewerRuntime],
    *,
    reviewer_specs_by_name: dict[str, DevelopmentReviewerSpec],
) -> tuple[ReviewAgentHandoff, ...]:
    handoffs: list[ReviewAgentHandoff] = []
    for reviewer in reviewers:
        reviewer_spec = reviewer_specs_by_name.get(reviewer.reviewer_name)
        if reviewer_spec is None:
            continue
        handoffs.append(
            ReviewAgentHandoff(
                reviewer_key=reviewer.reviewer_name,
                role_name=reviewer_spec.role_name,
                role_prompt=reviewer_spec.role_prompt,
                selection=reviewer.selection,
                worker=reviewer.worker,
            )
        )
    return tuple(handoffs)


def _shutdown_workers(
    developer: DeveloperRuntime | None,
    reviewers: Sequence[ReviewerRuntime],
    *,
    project_dir: str | Path,
    requirement_name: str,
    cleanup_runtime: bool,
    preserve_developer: bool = False,
    preserve_reviewer_keys: Sequence[str] = (),
) -> tuple[str, ...]:
    return shutdown_stage_workers(
        developer,
        reviewers,
        cleanup_runtime=cleanup_runtime,
        preserve_ba_worker=preserve_developer,
        preserve_reviewer_keys=preserve_reviewer_keys,
        runtime_root_filter=build_development_runtime_root(project_dir, requirement_name),
    )


def run_development_stage(
    argv: Sequence[str] | None = None,
    *,
    preserve_workers: bool = False,
) -> DevelopmentStageResult:
    parser = build_parser()
    args = parser.parse_args(argv)
    allow_previous_stage_back = bool(getattr(args, "allow_previous_stage_back", False))
    project_dir = str(Path(args.project_dir).expanduser().resolve()) if args.project_dir else prompt_project_dir("")
    requirement_name = str(args.requirement_name).strip() if args.requirement_name else prompt_requirement_name_selection(project_dir, "").requirement_name

    lock_context = requirement_concurrency_lock(
        project_dir,
        requirement_name,
        action="stage.a07.start",
    )
    lock_context.__enter__()
    progress = ReviewStageProgress(initial_phase="任务开发准备中")
    developer: DeveloperRuntime | None = None
    reviewer_workers: list[ReviewerRuntime] = []
    cleanup_records: list[str] = []
    audit_context: StageAuditRunContext | None = None
    current_task_name = ""
    try:
        audit_context = begin_stage_audit_run(
            project_dir,
            requirement_name,
            "A07",
            metadata={
                "trigger": "run_development_stage",
                "argv": list(argv or []),
                "args": vars(args),
            },
        )
        paths = ensure_development_inputs(args, project_dir=project_dir, requirement_name=requirement_name)
        cleanup_records.extend(cleanup_stale_development_runtime_state(project_dir, requirement_name))
        cleanup_records.extend(cleanup_existing_development_artifacts(paths, requirement_name, audit_context))
        launch_coordinator = DevelopmentStageLaunchCoordinator(build_development_runtime_root(project_dir, requirement_name))

        next_task = get_first_false_task(paths["task_json_path"])
        if next_task is None:
            append_stage_audit_record(
                audit_context,
                event_type="stage_passed",
                source_paths={
                    "task_json": paths["task_json_path"],
                    "developer_output": paths["developer_output_path"],
                    "merged_review": paths["merged_review_path"],
                },
            )
            return DevelopmentStageResult(
                project_dir=project_dir,
                requirement_name=requirement_name,
                task_md_path=str(paths["task_md_path"].resolve()),
                task_json_path=str(paths["task_json_path"].resolve()),
                merged_review_path=str(paths["merged_review_path"].resolve()),
                completed=True,
                cleanup_paths=(),
                developer_handoff=None,
                reviewer_handoff=(),
            )

        developer_plan_prompted = stdin_is_interactive() and (
            not str(getattr(args, "developer_role_prompt", "") or "").strip()
            or not any(str(getattr(args, key, "") or "").strip() for key in ("vendor", "model", "effort", "proxy_url"))
        )
        developer_plan_allow_back, allow_previous_stage_back = _consume_stage_back(
            allow_previous_stage_back,
            developer_plan_prompted,
        )
        reviewer_specs_prompted = stdin_is_interactive() and not any(
            str(item).strip() for item in [*getattr(args, "reviewer_role", []), *getattr(args, "reviewer_role_prompt", [])]
        ) and not resolve_stage_agent_config(args).reviewer_order
        reviewer_specs_allow_back, allow_previous_stage_back = _consume_stage_back(
            allow_previous_stage_back,
            reviewer_specs_prompted,
        )
        while True:
            developer_plan = resolve_developer_plan(
                args,
                project_dir=project_dir,
                progress=progress,
                allow_back_first_prompt=developer_plan_allow_back,
            )
            try:
                reviewer_specs = resolve_reviewer_specs(
                    args,
                    progress=progress,
                    allow_back_first_prompt=reviewer_specs_allow_back or developer_plan_prompted,
                )
                break
            except PromptBackRequested:
                if not developer_plan_prompted:
                    raise
                continue
        agent_config = resolve_stage_agent_config(args)
        reviewer_selection_allow_back, allow_previous_stage_back = _consume_stage_back(
            allow_previous_stage_back,
            stdin_is_interactive() and not bool(agent_config.reviewers) and bool(reviewer_specs),
        )
        reviewer_selections_by_name = agent_config.reviewers or collect_reviewer_agent_selections(
            project_dir=project_dir,
            reviewer_specs=reviewer_specs,
            reserved_session_names=(_developer_display_name(project_dir=project_dir),),
            progress=progress,
            allow_back_first_prompt=reviewer_selection_allow_back,
        )
        developer_max_turns_allow_back, allow_previous_stage_back = _consume_stage_back(
            allow_previous_stage_back,
            stdin_is_interactive() and getattr(args, "developer_max_turns", None) in {None, ""},
        )
        developer_turn_policy = DeveloperTurnPolicy(
            resolve_developer_max_turns(args, progress=progress, allow_back=developer_max_turns_allow_back)
        )
        review_round_allow_back, allow_previous_stage_back = _consume_stage_back(
            allow_previous_stage_back,
            stdin_is_interactive() and not str(getattr(args, "review_max_rounds", "") or "").strip(),
        )
        review_round_limit = resolve_review_max_rounds(
            args,
            progress=progress,
            allow_back=review_round_allow_back,
        )
        subagent_allow_back, allow_previous_stage_back = _consume_stage_back(
            allow_previous_stage_back,
            stdin_is_interactive() and getattr(args, "subagent_num", None) is None,
        )
        subagent_num = resolve_subagent_num(args, progress=progress, allow_back=subagent_allow_back)
        developer = create_developer_runtime(
            project_dir=project_dir,
            requirement_name=requirement_name,
            selection=developer_plan.selection,
            role_prompt=developer_plan.role_prompt,
            launch_coordinator=launch_coordinator,
        )
        set_runtime_metadata = getattr(developer.worker, "set_runtime_metadata", None)
        if callable(set_runtime_metadata):
            set_runtime_metadata(project_dir=project_dir, requirement_name=requirement_name, workflow_action="stage.a07.start")
        reviewer_specs_by_name = {str(item.reviewer_key or item.role_name).strip(): item for item in reviewer_specs}
        reviewer_label_getter = lambda reviewer, index: str(getattr(getattr(reviewer, "worker", None), "session_name", "") or getattr(reviewer, "reviewer_name", "") or f"代码审核智能体 {index}")  # noqa: E731
        replace_dead_developer_raw = lambda owner: _replace_dead_developer(  # noqa: E731
            owner,
            project_dir=project_dir,
            requirement_name=requirement_name,
            progress=progress,
            launch_coordinator=launch_coordinator,
        )
        replace_dead_developer_owner = lambda owner: _replace_dead_developer_with_bootstrap(  # noqa: E731
            owner,
            paths=paths,
            reviewer_specs_by_name=reviewer_specs_by_name,
            project_dir=project_dir,
            requirement_name=requirement_name,
            progress=progress,
            turn_policy=developer_turn_policy,
            launch_coordinator=launch_coordinator,
        )
        def replace_dead_reviewer(reviewer, _index):  # noqa: ANN001
            reviewer_spec = reviewer_specs_by_name.get(reviewer.reviewer_name)
            if reviewer_spec is None:
                return reviewer
            replacement = recreate_development_reviewer_runtime(
                project_dir=project_dir,
                requirement_name=requirement_name,
                reviewer=reviewer,
                reviewer_spec=reviewer_spec,
                progress=progress,
                force_model_change=False,
            )
            if replacement is None:
                return None
            return _run_single_reviewer_initialization(
                replacement,
                project_dir=project_dir,
                requirement_name=requirement_name,
                paths=paths,
                reviewer_specs_by_name=reviewer_specs_by_name,
                progress=progress,
                can_skip_ready_timeout=len(reviewer_workers) > 1,
            )
        reviewers_built = False
        reviewers_initialized = False
        _, reviewer_workers, developer = run_main_phase_with_death_handling(
            developer,
            reviewers=(),
            run_phase=lambda current_developer: initialize_development_workers(
                current_developer,
                project_dir=project_dir,
                requirement_name=requirement_name,
                paths=paths,
                reviewers=(),
                reviewer_specs_by_name=reviewer_specs_by_name,
                initialize_developer=True,
                initialize_reviewers=False,
                progress=progress,
                turn_policy=developer_turn_policy,
                replace_dead_developer=lambda active_developer, error: _replace_dead_developer_with_bootstrap(
                    active_developer,
                    paths=paths,
                    reviewer_specs_by_name=reviewer_specs_by_name,
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    progress=progress,
                    turn_policy=developer_turn_policy,
                    launch_coordinator=launch_coordinator,
                    error=error,
                ),
                replace_dead_developer_for_init=lambda active_developer, error: _replace_dead_developer(
                    active_developer,
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    progress=progress,
                    launch_coordinator=launch_coordinator,
                    error=error,
                ),
                audit_context=audit_context,
            )[0],
            replace_dead_main_owner=replace_dead_developer_owner,
            main_label="开发工程师",
            reviewer_label_getter=reviewer_label_getter,
            notify=message,
        )
        review_round_policy = ReviewRoundPolicy(review_round_limit)

        while next_task is not None:
            current_task_name = str(next_task)
            if not reviewers_built:
                reviewer_workers = build_reviewer_workers(
                    args,
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    reviewer_specs=reviewer_specs,
                    reviewer_selections_by_name=reviewer_selections_by_name,
                    progress=progress,
                    launch_coordinator=launch_coordinator,
                )
                reviewers_built = True
            if not reviewers_initialized:
                developer, code_change, reviewer_workers = run_first_task_with_parallel_reviewer_init(
                    developer,
                    reviewer_workers,
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    paths=paths,
                    reviewer_specs_by_name=reviewer_specs_by_name,
                    task_name=next_task,
                    subagent_num=subagent_num,
                    progress=progress,
                    turn_policy=developer_turn_policy,
                    replace_dead_developer=lambda active_developer, error: _replace_dead_developer_with_bootstrap(
                        active_developer,
                        paths=paths,
                        reviewer_specs_by_name=reviewer_specs_by_name,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        progress=progress,
                        turn_policy=developer_turn_policy,
                        launch_coordinator=launch_coordinator,
                        error=error,
                    ),
                    audit_context=audit_context,
                )
                reviewers_initialized = True
            else:
                (developer, code_change), reviewer_workers, developer = run_main_phase_with_death_handling(
                    developer,
                    reviewers=reviewer_workers,
                    run_phase=lambda current_developer: develop_current_task(
                        current_developer,
                        paths=paths,
                            task_name=next_task,
                            subagent_num=subagent_num,
                            progress=progress,
                            turn_policy=developer_turn_policy,
                            replace_dead_developer=lambda active_developer, error: _replace_dead_developer_with_bootstrap(
                                active_developer,
                                paths=paths,
                                reviewer_specs_by_name=reviewer_specs_by_name,
                                project_dir=project_dir,
                                requirement_name=requirement_name,
                                progress=progress,
                                turn_policy=developer_turn_policy,
                                launch_coordinator=launch_coordinator,
                                error=error,
                            ),
                            audit_context=audit_context,
                        ),
                        owner_getter=lambda result: result[0],
                    replace_dead_main_owner=replace_dead_developer_owner,
                    main_label="开发工程师",
                    reviewer_label_getter=reviewer_label_getter,
                    notify=message,
                )

            round_index = 1
            post_hitl_continue_completed = False
            while True:
                if not review_round_policy.initial_review_done:
                    prepare_review_round_artifacts(
                        paths,
                        reviewer_workers,
                        task_name=next_task,
                        audit_context=audit_context,
                        review_round_index=round_index,
                    )
                    reviewer_workers, developer = run_reviewer_phase_with_death_handling(
                        developer,
                        reviewer_workers,
                        run_phase=lambda active_reviewers: _run_parallel_reviewers(
                            active_reviewers,
                            project_dir=project_dir,
                            requirement_name=requirement_name,
                            paths=paths,
                            reviewer_specs_by_name=reviewer_specs_by_name,
                            task_name=next_task,
                            round_index=round_index,
                            prompt_builder=lambda reviewer: reviewer_review_code(
                                next_task,
                                code_change,
                                task_split_md=str(paths["task_md_path"].resolve()),
                                detailed_design_md=str(paths["detailed_design_path"].resolve()),
                                review_md=str(reviewer.review_md_path.resolve()),
                                review_json=str(reviewer.review_json_path.resolve()),
                            ),
                            label_prefix=f"development_review_init_{sanitize_requirement_name(next_task)}",
                            progress=progress,
                        ),
                        replace_dead_main_owner=replace_dead_developer_owner,
                        replace_dead_reviewer=replace_dead_reviewer,
                        main_label="开发工程师",
                        reviewer_label_getter=reviewer_label_getter,
                        notify=message,
                    )
                else:
                    review_msg = get_markdown_content(paths["merged_review_path"]).strip()
                    if not review_msg:
                        raise RuntimeError(f"{next_task} 代码评审未通过，但《{paths['merged_review_path'].name}》为空")
                    if not post_hitl_continue_completed:
                        (developer, code_change), reviewer_workers, developer = run_main_phase_with_death_handling(
                            developer,
                            reviewers=reviewer_workers,
                            run_phase=lambda current_developer: refine_current_task(
                                current_developer,
                                paths=paths,
                                task_name=next_task,
                                review_msg=review_msg,
                                progress=progress,
                                turn_policy=developer_turn_policy,
                                replace_dead_developer=lambda active_developer, error: _replace_dead_developer_with_bootstrap(
                                    active_developer,
                                    paths=paths,
                                    reviewer_specs_by_name=reviewer_specs_by_name,
                                    project_dir=project_dir,
                                    requirement_name=requirement_name,
                                    progress=progress,
                                    turn_policy=developer_turn_policy,
                                    launch_coordinator=launch_coordinator,
                                    error=error,
                                ),
                                audit_context=audit_context,
                                review_round_index=round_index,
                            ),
                            owner_getter=lambda result: result[0],
                            replace_dead_main_owner=replace_dead_developer_owner,
                            replace_dead_reviewer=replace_dead_reviewer,
                            main_label="开发工程师",
                            reviewer_label_getter=reviewer_label_getter,
                            notify=message,
                        )
                    post_hitl_continue_completed = False
                    prepare_review_round_artifacts(
                        paths,
                        reviewer_workers,
                        task_name=next_task,
                        audit_context=audit_context,
                        review_round_index=round_index,
                        preserve_existing_outputs=False,
                    )
                    reviewer_workers, developer = run_reviewer_phase_with_death_handling(
                        developer,
                        reviewer_workers,
                        run_phase=lambda active_reviewers: _run_parallel_reviewers(
                            active_reviewers,
                            project_dir=project_dir,
                            requirement_name=requirement_name,
                            paths=paths,
                            reviewer_specs_by_name=reviewer_specs_by_name,
                            task_name=next_task,
                            round_index=round_index,
                            prompt_builder=lambda reviewer: re_review_code(
                                next_task,
                                review_msg,
                                code_change,
                                task_split_md=str(paths["task_md_path"].resolve()),
                                detailed_design_md=str(paths["detailed_design_path"].resolve()),
                                review_md=str(reviewer.review_md_path.resolve()),
                                review_json=str(reviewer.review_json_path.resolve()),
                            ),
                            label_prefix=f"development_review_again_{sanitize_requirement_name(next_task)}",
                            progress=progress,
                            allow_existing_outputs=False,
                        ),
                        replace_dead_main_owner=replace_dead_developer_owner,
                        replace_dead_reviewer=replace_dead_reviewer,
                        main_label="开发工程师",
                        reviewer_label_getter=reviewer_label_getter,
                        notify=message,
                    )

                review_round_policy.record_review_attempt()
                reviewer_workers, developer = run_reviewer_phase_with_death_handling(
                    developer,
                    reviewer_workers,
                    run_phase=lambda active_reviewers: repair_reviewer_outputs(
                        active_reviewers,
                        paths=paths,
                        reviewer_specs_by_name=reviewer_specs_by_name,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        task_name=next_task,
                        round_index=round_index,
                        progress=progress,
                    ),
                    replace_dead_main_owner=replace_dead_developer_owner,
                    replace_dead_reviewer=replace_dead_reviewer,
                    main_label="开发工程师",
                    reviewer_label_getter=reviewer_label_getter,
                    notify=message,
                )
                ensure_active_reviewers(reviewer_workers, stage_label="任务开发")
                review_json_files, review_md_files = _active_reviewer_files(reviewer_workers)
                passed = task_done(
                    directory=project_dir,
                    file_path=paths["task_json_path"],
                    task_name=next_task,
                    json_pattern=f"{sanitize_requirement_name(requirement_name)}_评审记录_*.json",
                    md_pattern=f"{sanitize_requirement_name(requirement_name)}_代码评审记录_*.md",
                    md_output_name=paths["merged_review_path"].name,
                    json_files=review_json_files,
                    md_files=review_md_files,
                )
                append_stage_audit_record(
                    audit_context,
                    event_type="review_merged",
                    source_paths={"merged_review": paths["merged_review_path"]},
                    reviewer_markdown_paths=review_md_files,
                    reviewer_json_paths=review_json_files,
                    review_round_index=round_index,
                    task_name=next_task,
                    metadata=_reviewer_audit_metadata(reviewer_workers, trigger="task_done"),
                )
                if passed:
                    append_stage_audit_record(
                        audit_context,
                        event_type="task_passed",
                        source_paths={
                            "task_json": paths["task_json_path"],
                            "developer_output": paths["developer_output_path"],
                            "merged_review": paths["merged_review_path"],
                        },
                        review_round_index=round_index,
                        task_name=next_task,
                    )
                    break
                review_msg = get_markdown_content(paths["merged_review_path"]).strip()
                if not review_msg:
                    raise RuntimeError(f"{next_task} 代码评审未通过，但《{paths['merged_review_path'].name}》为空")
                if review_round_policy.should_escalate_before_next_review():
                    if review_round_policy.max_rounds is None:
                        raise RuntimeError("review_round_policy 配置错误：无限轮次不应触发超限 HITL")
                    hitl_result, reviewer_workers, developer = run_main_phase_with_death_handling(
                        developer,
                        reviewers=reviewer_workers,
                        run_phase=lambda current_developer: run_development_review_limit_hitl_loop(
                            current_developer,
                            paths=paths,
                            task_name=next_task,
                            review_msg=review_msg,
                            review_limit=review_round_policy.max_rounds,
                            review_rounds_used=review_round_policy.quota_count,
                            progress=progress,
                            turn_policy=developer_turn_policy,
                            replace_dead_developer=lambda active_developer, error: _replace_dead_developer_with_bootstrap(
                                active_developer,
                                paths=paths,
                                reviewer_specs_by_name=reviewer_specs_by_name,
                                project_dir=project_dir,
                                requirement_name=requirement_name,
                                progress=progress,
                                turn_policy=developer_turn_policy,
                                launch_coordinator=launch_coordinator,
                                error=error,
                            ),
                            apply_human_review_override=lambda human_msg: apply_development_review_human_override(
                                task_name=next_task,
                                reviewer_workers=reviewer_workers,
                                merged_review_path=paths["merged_review_path"],
                            ),
                            human_input_provider=(
                                lambda question_path, hitl_round: collect_auto_review_limit_hitl_response(
                                    question_path,
                                    stage_label="任务开发评审超限",
                                    hitl_round=hitl_round,
                                )
                            ) if bool(getattr(args, "yes", False)) else None,
                            audit_context=audit_context,
                        ),
                        owner_getter=lambda result: result.owner,
                        replace_dead_main_owner=replace_dead_developer_owner,
                        main_label="开发工程师",
                        reviewer_label_getter=reviewer_label_getter,
                        notify=message,
                    )
                    post_hitl_continue_completed = bool(getattr(hitl_result, "post_hitl_continue_completed", False))
                    review_round_policy.reset_after_hitl()
                    review_json_files, review_md_files = _active_reviewer_files(reviewer_workers)
                    passed = task_done(
                        directory=project_dir,
                        file_path=paths["task_json_path"],
                        task_name=next_task,
                        json_pattern=f"{sanitize_requirement_name(requirement_name)}_评审记录_*.json",
                        md_pattern=f"{sanitize_requirement_name(requirement_name)}_代码评审记录_*.md",
                        md_output_name=paths["merged_review_path"].name,
                        json_files=review_json_files,
                        md_files=review_md_files,
                    )
                    append_stage_audit_record(
                        audit_context,
                        event_type="review_merged",
                        source_paths={"merged_review": paths["merged_review_path"]},
                        reviewer_markdown_paths=review_md_files,
                        reviewer_json_paths=review_json_files,
                        review_round_index=round_index,
                        task_name=next_task,
                        metadata=_reviewer_audit_metadata(reviewer_workers, trigger="post_hitl_task_done"),
                    )
                    if passed:
                        append_stage_audit_record(
                            audit_context,
                            event_type="task_passed",
                            source_paths={
                                "task_json": paths["task_json_path"],
                                "developer_output": paths["developer_output_path"],
                                "merged_review": paths["merged_review_path"],
                            },
                            review_round_index=round_index,
                            task_name=next_task,
                        )
                        break
                round_index += 1

            next_task = get_first_false_task(paths["task_json_path"])
            if (
                next_task is not None
                and developer is not None
                and reviewer_workers
                and developer_turn_policy.should_recreate_before_next_task()
            ):
                decision = request_worker_manual_intervention(
                    stage_label="任务开发",
                    role_label=str(developer.worker.session_name or "开发工程师").strip() or "开发工程师",
                    worker=developer.worker,
                    reason_text=(
                        f"开发工程师已达到最大对话轮数 {developer_turn_policy.max_turns}。"
                        "系统不会自动重建开发与评审智能体。请人工进入 tmux 交涉或关闭智能体，"
                        f"确认可以继续后再进入任务 {next_task}。"
                    ),
                    target_paths=(paths["task_json_path"], paths["developer_output_path"]),
                    progress=progress,
                    allow_recreate=True,
                    allow_worker_dead=False,
                )
                if decision in {AGENT_INTERVENTION_RECREATE, AGENT_INTERVENTION_WORKER_DEAD}:
                    developer = replace_dead_developer_owner(developer)
                    reviewers_built = False
                    reviewers_initialized = False
                developer_turn_policy.reset()
            review_round_policy = ReviewRoundPolicy(review_round_limit)

        current_task_name = ""
        result_developer_handoff = _export_developer_handoff(developer) if preserve_workers else None
        result_reviewer_handoff = _export_reviewer_handoff(
            reviewer_workers,
            reviewer_specs_by_name=reviewer_specs_by_name,
        ) if preserve_workers else ()
        cleanup_records.extend(_shutdown_workers(
            developer,
            reviewer_workers,
            project_dir=project_dir,
            requirement_name=requirement_name,
            cleanup_runtime=True,
            preserve_developer=preserve_workers,
            preserve_reviewer_keys=[item.reviewer_key for item in result_reviewer_handoff],
        ))
        append_stage_audit_record(
            audit_context,
            event_type="stage_passed",
            source_paths={
                "task_json": paths["task_json_path"],
                "developer_output": paths["developer_output_path"],
                "merged_review": paths["merged_review_path"],
            },
        )
        return DevelopmentStageResult(
            project_dir=project_dir,
            requirement_name=requirement_name,
            task_md_path=str(paths["task_md_path"].resolve()),
            task_json_path=str(paths["task_json_path"].resolve()),
            merged_review_path=str(paths["merged_review_path"].resolve()),
            completed=True,
            cleanup_paths=tuple(dict.fromkeys(cleanup_records)),
            developer_handoff=result_developer_handoff,
            reviewer_handoff=result_reviewer_handoff,
        )
    except Exception as error:
        failed_paths = paths if "paths" in locals() else build_development_paths(project_dir, requirement_name)
        if current_task_name:
            append_stage_audit_record(
                audit_context,
                event_type="task_failed",
                source_paths={
                    "task_json": failed_paths["task_json_path"],
                    "developer_output": failed_paths["developer_output_path"],
                    "merged_review": failed_paths["merged_review_path"],
                    "ask_human": failed_paths["ask_human_path"],
                    "hitl_record": failed_paths["hitl_record_path"],
                },
                task_name=current_task_name,
                metadata={"error": str(error)},
            )
        append_stage_audit_record(
            audit_context,
            event_type="stage_failed",
            source_paths={
                "task_json": failed_paths["task_json_path"],
                "developer_output": failed_paths["developer_output_path"],
                "merged_review": failed_paths["merged_review_path"],
            },
            metadata={"error": str(error)},
        )
        _shutdown_workers(
            developer,
            reviewer_workers,
            project_dir=project_dir,
            requirement_name=requirement_name,
            cleanup_runtime=False,
        )
        raise
    finally:
        progress.stop()
        lock_context.__exit__(None, None, None)


def main(argv: Sequence[str] | None = None) -> int:
    redirected, launch = maybe_launch_tui(argv, route="development", action="stage.a07.start")
    if redirected:
        return int(launch)
    try:
        result = run_development_stage(list(launch))
    except Exception as error:  # noqa: BLE001
        message(error)
        return 1

    message("任务开发完成")
    message(result.task_md_path)
    message(result.task_json_path)
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
