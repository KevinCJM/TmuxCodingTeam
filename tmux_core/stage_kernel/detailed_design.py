# -*- encoding: utf-8 -*-
"""
@File: A05_DetailedDesign.py
@Modify Time: 2026/4/18
@Author: Kevin-Chen
@Descriptions: 详细设计阶段
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from tmux_core.runtime.vendor_catalog import get_default_model_for_vendor
from A01_Routing_LayerPlanning import (
    DEFAULT_MODEL_BY_VENDOR,
    normalize_effort_choice,
    normalize_model_choice,
    normalize_vendor_choice,
)
from A03_RequirementsClarification import run_requirements_clarification_stage
from tmux_core.prompt_contracts.common import check_reviewer_job
from tmux_core.prompt_contracts.detailed_design import (
    again_review_detailed_design,
    create_detailed_design_ba,
    detailed_design,
    hitl_relpy,
    modify_detailed_design,
    review_detailed_design,
)
from tmux_core.runtime.contracts import (
    TASK_STATUS_DONE,
    TaskResultContract,
    TurnFileContract,
    TurnFileResult,
    finalize_task_result,
    normalize_review_status_payload,
    resolve_task_result_decision,
    write_task_status,
)
from tmux_core.runtime.hitl import build_prefixed_sha256
from tmux_core.runtime.tmux_runtime import (
    CommandResult,
    DEFAULT_COMMAND_TIMEOUT_SEC,
    TmuxBatchWorker,
    Vendor,
    WorkerStatus,
    build_session_name,
    cleanup_registered_tmux_workers,
    is_agent_ready_timeout_error,
    is_provider_auth_error,
    is_turn_artifact_contract_error,
    is_worker_death_error,
    list_registered_tmux_workers,
    list_tmux_session_names,
    list_occupied_tmux_session_names,
)
from tmux_core.stage_kernel.reviewer_orchestration import (
    repair_reviewer_round_outputs,
    run_parallel_reviewer_round,
    shutdown_stage_workers,
)
from tmux_core.stage_kernel.death_orchestration import (
    ensure_active_reviewers,
    run_main_phase_with_death_handling,
    run_reviewer_phase_with_death_handling,
)
from tmux_core.stage_kernel.requirement_concurrency import requirement_concurrency_lock
from tmux_core.stage_kernel.runtime_scope_cleanup import cleanup_runtime_dirs_by_scope
from tmux_core.stage_kernel.stage_audit import (
    StageAuditRunContext,
    append_stage_audit_record,
    begin_stage_audit_run,
    record_before_cleanup,
)
from tmux_core.stage_kernel.agent_intervention import (
    AGENT_INTERVENTION_RECHECK,
    AGENT_INTERVENTION_RECREATE,
    AGENT_INTERVENTION_WORKER_DEAD,
    request_worker_manual_intervention,
)
from tmux_core.stage_kernel.shared_review import (
    ReviewLimitHitlConfig,
    ReviewRoundPolicy,
    MAX_REVIEWER_REPAIR_ATTEMPTS,
    ReviewAgentHandoff,
    ReviewAgentSelection,
    ReviewStageProgress,
    ReviewerRuntime,
    collect_auto_review_limit_hitl_response,
    collect_reviewer_agent_selections,
    ensure_empty_file,
    ensure_review_artifacts,
    mark_reviewer_turn_succeeded_from_materialized_outputs,
    mark_worker_awaiting_reconfiguration,
    parse_review_max_rounds,
    prompt_positive_int,
    prompt_required_replacement_review_agent_selection,
    prompt_review_max_rounds,
    prompt_replacement_review_agent_selection,
    prompt_review_agent_selection,
    prompt_yes_no_choice,
    render_review_limit_force_hitl_prompt,
    render_review_limit_human_reply_prompt,
    render_review_agent_selection,
    render_tmux_start_summary,
    is_agent_config_error,
    reviewer_artifact_signature,
    reviewer_outputs_satisfy_contract,
    reviewer_worker_needs_terminal_success_normalization,
    resolve_agent_run_config_with_recovery,
    resolve_stage_agent_config,
    run_review_limit_hitl_cycle,
    worker_has_provider_auth_error,
    worker_has_provider_runtime_error,
)
from tmux_core.stage_kernel.turn_output_goals import (
    CompletionTurnGoal,
    OutcomeGoal,
    TaskTurnGoal,
    run_completion_turn_with_repair,
    run_task_result_turn_with_repair,
)
from T01_tools import get_markdown_content, is_file_empty, task_done
from T08_pre_development import (
    build_pre_development_task_record_path,
    ensure_pre_development_task_record,
    mark_detailed_design_completed,
    update_pre_development_task_status,
)
from T09_terminal_ops import (
    PROMPT_BACK_VALUE,
    BridgeTerminalUI,
    PromptBackRequested,
    collect_multiline_input,
    get_terminal_ui,
    maybe_launch_tui,
    message,
    prompt_metadata,
    prompt_select_option,
    prompt_with_default,
)
from T12_requirements_common import (
    DEFAULT_REQUIREMENTS_CLARIFICATION_EFFORT,
    DEFAULT_REQUIREMENTS_CLARIFICATION_MODEL,
    DEFAULT_REQUIREMENTS_CLARIFICATION_VENDOR,
    RequirementsAnalystHandoff,
    build_requirements_clarification_paths,
    prompt_project_dir,
    prompt_requirement_name_selection,
    sanitize_requirement_name,
    stdin_is_interactive,
)


DETAILED_DESIGN_TASK_NAME = "详细设计"
DETAILED_DESIGN_RUNTIME_ROOT_NAME = ".detailed_design_runtime"
MAX_DETAILED_DESIGN_REVIEW_ROUNDS = 5
MAX_DETAILED_DESIGN_HITL_ROUNDS = 8
PLACEHOLDER_NEXT_STEP = "下一步进入任务拆分阶段（待接入）"

DEFAULT_REVIEWER_PROMPTS: dict[str, str] = {
    "开发工程师": "你是详细设计审计中的开发工程师。重点检查实现可行性、模块边界、代码落点、最小改动路径、是否引入非必要重构。",
    "测试工程师": "你是详细设计审计中的测试工程师。重点检查边界条件、异常处理、可测试性、验收覆盖、回归风险与遗漏场景。",
    "架构师": "你是详细设计审计中的架构师。重点检查系统边界、依赖影响、契约兼容、扩展约束，以及是否保持最小改动。",
    "审核员": "你是详细设计审计中的审核员。重点检查需求对齐、文档一致性、契约兼容、越界内容，以及四问原则是否被违反。",
}
DEFAULT_REVIEWER_ROLE_NAMES: tuple[str, ...] = tuple(DEFAULT_REVIEWER_PROMPTS.keys())
ExistingDetailedDesignMode = Literal["skip", "review_existing", "rerun"]


@dataclass(frozen=True)
class DetailedDesignReviewerSpec:
    role_name: str
    role_prompt: str
    reviewer_key: str = ""


@dataclass(frozen=True)
class DetailedDesignStageResult:
    project_dir: str
    requirement_name: str
    detailed_design_path: str
    merged_review_path: str
    passed: bool
    cleanup_paths: tuple[str, ...] = ()
    ba_handoff: RequirementsAnalystHandoff | None = None
    reviewer_handoff: tuple[ReviewAgentHandoff, ...] = ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="详细设计阶段")
    parser.add_argument("--project-dir", help="项目目录")
    parser.add_argument("--requirement-name", help="需求名称")
    parser.add_argument("--allow-previous-stage-back", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--vendor", help="需求分析师厂商: codex|claude|gemini|opencode")
    parser.add_argument("--model", help="需求分析师模型名称")
    parser.add_argument("--effort", help="需求分析师推理强度")
    parser.add_argument("--proxy-url", default="", help="需求分析师代理端口或完整代理 URL")
    parser.add_argument("--reuse-review-ba", action="store_true", help="优先复用需求评审阶段的需求分析师")
    parser.add_argument("--rebuild-review-ba", action="store_true", help="强制重建需求分析师")
    parser.add_argument("--review-max-rounds", default="", help="详设评审最多重试几轮；传 infinite 表示不设上限")
    parser.add_argument("--reviewer-agent", action="append", default=[], help="审核智能体模型配置: name=<key>,vendor=...,model=...,effort=...,proxy=...")
    parser.add_argument("--reviewer-role", action="append", default=[], help="重复传入以覆盖详设评审角色列表")
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


def build_detailed_design_paths(project_dir: str | Path, requirement_name: str) -> dict[str, Path]:
    project_root = Path(project_dir).expanduser().resolve()
    safe_name = sanitize_requirement_name(requirement_name)
    original_requirement_path, requirements_clear_path, ask_human_path, hitl_record_path = (
        build_requirements_clarification_paths(project_root, requirement_name)
    )
    return {
        "project_root": project_root,
        "original_requirement_path": original_requirement_path,
        "requirements_clear_path": requirements_clear_path,
        "ask_human_path": ask_human_path,
        "hitl_record_path": hitl_record_path,
        "pre_development_path": build_pre_development_task_record_path(project_root, requirement_name),
        "detailed_design_path": project_root / f"{safe_name}_详细设计.md",
        "merged_review_path": project_root / f"{safe_name}_详设评审记录.md",
        "ba_feedback_path": project_root / f"{safe_name}_需求分析师反馈.md",
    }


def build_detailed_design_runtime_root(project_dir: str | Path, requirement_name: str = "") -> Path:
    project_root = Path(project_dir).expanduser().resolve()
    safe_requirement = sanitize_requirement_name(requirement_name) if str(requirement_name or "").strip() else ""
    runtime_root = project_root / DETAILED_DESIGN_RUNTIME_ROOT_NAME
    return runtime_root / safe_requirement if safe_requirement else runtime_root


def _infer_detailed_design_runtime_scope(worker: object, project_dir: str | Path) -> str:
    try:
        worker_root = Path(getattr(worker, "runtime_root", "")).expanduser().resolve()
        legacy_root = Path(project_dir).expanduser().resolve() / DETAILED_DESIGN_RUNTIME_ROOT_NAME
    except Exception:
        return ""
    if worker_root.parent == legacy_root:
        return worker_root.name
    return ""


def build_reviewer_artifact_paths(project_dir: str | Path, requirement_name: str, reviewer_name: str) -> tuple[Path, Path]:
    project_root = Path(project_dir).expanduser().resolve()
    safe_name = sanitize_requirement_name(requirement_name)
    artifact_agent_name = sanitize_requirement_name(reviewer_name)
    review_md_path = project_root / f"{safe_name}_详设评审记录_{artifact_agent_name}.md"
    review_json_path = project_root / f"{safe_name}_评审记录_{artifact_agent_name}.json"
    return review_md_path, review_json_path


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
            default=MAX_DETAILED_DESIGN_REVIEW_ROUNDS,
        )
    if not stdin_is_interactive():
        return MAX_DETAILED_DESIGN_REVIEW_ROUNDS
    if progress is not None and hasattr(progress, "set_phase"):
        progress.set_phase("详细设计 / 配置最大审核轮次")
    return prompt_review_max_rounds(
        default=MAX_DETAILED_DESIGN_REVIEW_ROUNDS,
        progress=progress,
        allow_back=allow_back,
        stage_key="detailed_design",
        stage_step_index=1,
    )


def build_detailed_design_review_limit_hitl_config(paths: dict[str, Path]) -> ReviewLimitHitlConfig:
    return ReviewLimitHitlConfig(
        stage_label="详细设计评审超限",
        artifact_label="详设评审",
        primary_output_path=paths["detailed_design_path"],
        ask_human_path=paths["ask_human_path"],
        hitl_record_path=paths["hitl_record_path"],
        merged_review_path=paths["merged_review_path"],
        output_summary_path=paths["ba_feedback_path"],
        continue_output_label="详细设计.md",
    )


def build_detailed_design_review_limit_force_hitl_prompt(
    *,
    paths: dict[str, Path],
    review_msg: str,
    review_limit: int,
    review_rounds_used: int,
) -> str:
    return render_review_limit_force_hitl_prompt(
        config=build_detailed_design_review_limit_hitl_config(paths),
        review_limit=review_limit,
        review_rounds_used=review_rounds_used,
        hitl_record_md=paths["hitl_record_path"],
        extra_inputs=(
            paths["original_requirement_path"],
            paths["requirements_clear_path"],
        ),
    ) + f"\n## 当前评审记录\n[REVIEW MSG START]\n{review_msg}\n[REVIEW MSG END]\n"


def build_detailed_design_review_limit_human_reply_prompt(
    *,
    paths: dict[str, Path],
    review_msg: str,
    human_msg: str,
) -> str:
    return render_review_limit_human_reply_prompt(
        config=build_detailed_design_review_limit_hitl_config(paths),
        human_msg=human_msg,
        hitl_record_md=paths["hitl_record_path"],
        extra_inputs=(
            paths["original_requirement_path"],
            paths["requirements_clear_path"],
        ),
    ) + f"\n## 当前评审记录\n[REVIEW MSG START]\n{review_msg}\n[REVIEW MSG END]\n"


def cleanup_existing_detailed_design_artifacts(
    paths: dict[str, Path],
    requirement_name: str,
    *,
    clear_detailed_design: bool = True,
    audit_context: StageAuditRunContext | None = None,
) -> tuple[str, ...]:
    project_root = paths["project_root"]
    safe_name = sanitize_requirement_name(requirement_name)
    removed: list[str] = []
    review_json_candidates: list[Path] = []
    review_md_candidates: list[Path] = []
    for pattern in (
        f"{safe_name}_评审记录_*.json",
        f"{safe_name}_详设评审记录_*.md",
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
                "merged_review": paths["merged_review_path"],
                "ba_feedback": paths["ba_feedback_path"],
                "ask_human": paths["ask_human_path"],
                "detailed_design": paths["detailed_design_path"],
            },
            metadata={
                "trigger": "cleanup_existing_detailed_design_artifacts",
                "clear_detailed_design": bool(clear_detailed_design),
            },
            reviewer_markdown_paths=review_md_candidates,
            reviewer_json_paths=review_json_candidates,
        )
    for candidate in (*review_json_candidates, *review_md_candidates):
        if candidate.is_file():
            candidate.unlink()
            removed.append(str(candidate.resolve()))
    for candidate in (
        paths["merged_review_path"],
        paths["ba_feedback_path"],
        paths["ask_human_path"],
    ):
        if candidate.exists() and candidate.is_file():
            candidate.write_text("", encoding="utf-8")
            removed.append(str(candidate.resolve()))
    if clear_detailed_design and paths["detailed_design_path"].exists() and paths["detailed_design_path"].is_file():
        paths["detailed_design_path"].write_text("", encoding="utf-8")
        removed.append(str(paths["detailed_design_path"].resolve()))
    return tuple(dict.fromkeys(removed))


def cleanup_stale_detailed_design_runtime_state(
    project_dir: str | Path,
    requirement_name: str,
) -> tuple[str, ...]:
    runtime_root = Path(project_dir).expanduser().resolve() / DETAILED_DESIGN_RUNTIME_ROOT_NAME
    if not runtime_root.exists() or not runtime_root.is_dir():
        return ()
    return cleanup_runtime_dirs_by_scope(
        runtime_root=runtime_root,
        project_dir=project_dir,
        requirement_name=requirement_name,
        workflow_action="stage.a05.start",
    )


def build_detailed_design_reviewer_worker_id(role_name: str) -> str:
    return f"detailed-design-review-{str(role_name).strip()}"


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


def _detailed_design_ba_display_name(
    *,
    project_dir: str | Path,
    handoff: RequirementsAnalystHandoff | None = None,
) -> str:
    session_name = str(getattr(getattr(handoff, "worker", None), "session_name", "") or "").strip()
    if session_name:
        return session_name
    return _predict_worker_display_name(
        project_dir=project_dir,
        worker_id="detailed-design-analyst",
    )


def _reviewer_artifact_agent_name(reviewer: ReviewerRuntime) -> str:
    worker = getattr(reviewer, "worker", None)
    session_name = str(getattr(worker, "session_name", "") or "").strip()
    reviewer_name = str(getattr(reviewer, "reviewer_name", "") or "").strip()
    return session_name or reviewer_name


def _predict_reviewer_display_name(
    *,
    project_dir: str | Path,
    role_name: str,
    occupied_session_names: Sequence[str] = (),
) -> str:
    return _predict_worker_display_name(
        project_dir=project_dir,
        worker_id=build_detailed_design_reviewer_worker_id(role_name),
        occupied_session_names=occupied_session_names,
    )


def _is_live_ba_handoff(handoff: RequirementsAnalystHandoff | None) -> bool:
    if handoff is None:
        return False
    session_name = str(getattr(handoff.worker, "session_name", "") or "").strip()
    if not session_name:
        return False
    session_exists = getattr(handoff.worker, "session_exists", None)
    if callable(session_exists):
        try:
            return bool(session_exists())
        except Exception:
            return False
    return True


def _reviewer_default_selection() -> ReviewAgentSelection:
    return ReviewAgentSelection(
        vendor=DEFAULT_REQUIREMENTS_CLARIFICATION_VENDOR,
        model=DEFAULT_REQUIREMENTS_CLARIFICATION_MODEL,
        reasoning_effort=DEFAULT_REQUIREMENTS_CLARIFICATION_EFFORT,
        proxy_url="",
    )


def _reviewer_spec_identity(reviewer_spec: DetailedDesignReviewerSpec) -> str:
    return str(reviewer_spec.reviewer_key or reviewer_spec.role_name).strip()


def _default_prompt_for_role(role_name: str) -> str:
    return DEFAULT_REVIEWER_PROMPTS.get(
        role_name,
        "",
    )


def _finalize_reviewer_specs(specs: Sequence[DetailedDesignReviewerSpec]) -> list[DetailedDesignReviewerSpec]:
    role_counts = Counter(str(item.role_name).strip() for item in specs if str(item.role_name).strip())
    role_seen: Counter[str] = Counter()
    finalized: list[DetailedDesignReviewerSpec] = []
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
            if role_counts[role_name] > 1:
                reviewer_key = f"{role_name}#{role_seen[role_name]}"
            else:
                reviewer_key = role_name
        finalized.append(
            DetailedDesignReviewerSpec(
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
) -> DetailedDesignReviewerSpec:
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
                    options=(
                        ("default", "默认角色"),
                        ("custom", "自定义角色"),
                    ),
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
            default_prompt = _default_prompt_for_role(role_name)
            if default_prompt:
                if step == 2:
                    prompt_source = _prompt_reviewer_select(
                        title=f"第 {index} 个审核智能体 - 角色定义提示词来源",
                        options=(
                            ("default", "默认提示词"),
                            ("custom", "自定义提示词"),
                        ),
                        default_value=prompt_source or "default",
                        prompt_text=f"选择第 {index} 个审核智能体的角色定义提示词来源",
                        progress=progress,
                        allow_back=True,
                        stage_key=stage_key,
                        stage_step_index=3,
                    )
                    if prompt_source == "default":
                        return DetailedDesignReviewerSpec(role_name=role_name, role_prompt=default_prompt)
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
                    return DetailedDesignReviewerSpec(role_name=role_name, role_prompt=role_prompt)
            else:
                role_prompt = _prompt_reviewer_text(
                    f"输入第 {index} 个审核智能体的自定义角色定义提示词",
                    default=role_prompt,
                    progress=progress,
                    allow_back=True,
                    stage_key=stage_key,
                    stage_step_index=3,
                )
                return DetailedDesignReviewerSpec(role_name=role_name, role_prompt=role_prompt)
        except PromptBackRequested:
            if step == 0:
                raise
            step -= 1


def collect_interactive_reviewer_specs(
    *,
    progress: ReviewStageProgress | None = None,
    allow_back_first_prompt: bool = False,
    stage_key: str = "detailed_design_reviewer_specs",
) -> list[DetailedDesignReviewerSpec]:
    if progress is not None:
        progress.set_phase("详细设计 / 配置审核器")
    reviewer_count = prompt_positive_int(
        "请输入详细设计审核智能体数量",
        len(DEFAULT_REVIEWER_ROLE_NAMES),
        progress=progress,
        allow_back=allow_back_first_prompt,
        stage_key=stage_key,
        stage_step_index=0,
    )
    collected_specs: list[DetailedDesignReviewerSpec] = []
    default_roles = list(DEFAULT_REVIEWER_ROLE_NAMES)
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
                    "请输入详细设计审核智能体数量",
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
) -> list[DetailedDesignReviewerSpec]:
    agent_config = resolve_stage_agent_config(args)
    role_names = [str(item).strip() for item in getattr(args, "reviewer_role", []) if str(item).strip()]
    prompt_values = [str(item).strip() for item in getattr(args, "reviewer_role_prompt", []) if str(item).strip()]
    if agent_config.reviewer_order and not role_names and not prompt_values:
        default_roles = list(DEFAULT_REVIEWER_ROLE_NAMES)
        specs: list[DetailedDesignReviewerSpec] = []
        for index, reviewer_key in enumerate(agent_config.reviewer_order):
            role_name = reviewer_key if reviewer_key in DEFAULT_REVIEWER_PROMPTS else default_roles[index % len(default_roles)]
            specs.append(
                DetailedDesignReviewerSpec(
                    role_name=role_name,
                    role_prompt=_default_prompt_for_role(role_name),
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
        role_names = list(DEFAULT_REVIEWER_ROLE_NAMES)
        if prompt_values and len(prompt_values) != len(role_names):
            raise RuntimeError("--reviewer-role-prompt 数量必须与默认角色数量一致。")
    specs: list[DetailedDesignReviewerSpec] = []
    for index, role_name in enumerate(role_names):
        default_prompt = _default_prompt_for_role(role_name)
        if prompt_values:
            if len(prompt_values) == len(role_names):
                role_prompt = prompt_values[index]
            else:
                role_prompt = prompt_values[index]
        elif default_prompt:
            role_prompt = default_prompt
        else:
            raise RuntimeError(f"自定义角色 {role_name} 必须同时提供对应的角色定义提示词。")
        specs.append(DetailedDesignReviewerSpec(role_name=role_name, role_prompt=role_prompt))
    return _finalize_reviewer_specs(specs)


def collect_ba_agent_selection(
    args: argparse.Namespace,
    *,
    role_label: str,
    allow_back_first_step: bool = False,
    stage_key: str = "detailed_design_main",
) -> ReviewAgentSelection:
    interactive = stdin_is_interactive()
    vendor_value = str(getattr(args, "vendor", "") or "").strip()
    model_value = str(getattr(args, "model", "") or "").strip()
    effort_value = str(getattr(args, "effort", "") or "").strip()
    proxy_value = str(getattr(args, "proxy_url", "") or "").strip()
    if interactive and not any((vendor_value, model_value, effort_value, proxy_value)):
        return prompt_review_agent_selection(
            DEFAULT_REQUIREMENTS_CLARIFICATION_VENDOR,
            default_model=DEFAULT_REQUIREMENTS_CLARIFICATION_MODEL,
            default_reasoning_effort=DEFAULT_REQUIREMENTS_CLARIFICATION_EFFORT,
            default_proxy_url="",
            role_label=role_label,
            allow_back_first_step=allow_back_first_step,
            stage_key=stage_key,
        )
    try:
        vendor = normalize_vendor_choice(vendor_value or DEFAULT_REQUIREMENTS_CLARIFICATION_VENDOR)
        model = normalize_model_choice(vendor, model_value or get_default_model_for_vendor(vendor))
        reasoning_effort = normalize_effort_choice(vendor, model, effort_value or DEFAULT_REQUIREMENTS_CLARIFICATION_EFFORT)
    except Exception as error:  # noqa: BLE001
        if not interactive or not is_agent_config_error(error):
            raise
        message(f"{role_label} 模型配置不可用: {error}\n请重新选择厂商、模型和推理强度。")
        return prompt_review_agent_selection(
            DEFAULT_REQUIREMENTS_CLARIFICATION_VENDOR,
            default_model=DEFAULT_REQUIREMENTS_CLARIFICATION_MODEL,
            default_reasoning_effort=DEFAULT_REQUIREMENTS_CLARIFICATION_EFFORT,
            default_proxy_url=proxy_value,
            role_label=role_label,
            allow_back_first_step=allow_back_first_step,
            stage_key=stage_key,
        )
    return ReviewAgentSelection(
        vendor=vendor,
        model=model,
        reasoning_effort=reasoning_effort,
        proxy_url=proxy_value,
    )


def build_clarification_stage_argv(args: argparse.Namespace, *, project_dir: str, requirement_name: str) -> list[str]:
    argv = ["--project-dir", project_dir, "--requirement-name", requirement_name]
    if getattr(args, "yes", False):
        argv.append("--yes")
    if getattr(args, "no_tui", False):
        argv.append("--no-tui")
    if getattr(args, "legacy_cli", False):
        argv.append("--legacy-cli")
    interactive = stdin_is_interactive()
    if not interactive:
        vendor = normalize_vendor_choice(str(getattr(args, "vendor", "") or "").strip() or DEFAULT_REQUIREMENTS_CLARIFICATION_VENDOR)
        model = normalize_model_choice(vendor, str(getattr(args, "model", "") or "").strip() or get_default_model_for_vendor(vendor))
        effort = normalize_effort_choice(vendor, model, str(getattr(args, "effort", "") or "").strip() or DEFAULT_REQUIREMENTS_CLARIFICATION_EFFORT)
        argv.extend(["--vendor", vendor, "--model", model, "--effort", effort])
        proxy_url = str(getattr(args, "proxy_url", "") or "").strip()
        if proxy_url:
            argv.extend(["--proxy-url", proxy_url])
    return argv


def ensure_detailed_design_inputs(args: argparse.Namespace, *, project_dir: str, requirement_name: str) -> dict[str, Path]:
    paths = build_detailed_design_paths(project_dir, requirement_name)
    if not get_markdown_content(paths["original_requirement_path"]).strip():
        raise RuntimeError(f"缺少原始需求文档: {paths['original_requirement_path']}")
    if not get_markdown_content(paths["requirements_clear_path"]).strip():
        message(f"缺少需求澄清文档，先自动执行需求澄清阶段: {paths['requirements_clear_path'].name}")
        clarification_argv = build_clarification_stage_argv(
            args,
            project_dir=project_dir,
            requirement_name=requirement_name,
        )
        run_requirements_clarification_stage(clarification_argv, preserve_ba_worker=False)
    paths = build_detailed_design_paths(project_dir, requirement_name)
    if not get_markdown_content(paths["requirements_clear_path"]).strip():
        raise RuntimeError(f"缺少需求澄清文档: {paths['requirements_clear_path']}")
    ensure_pre_development_task_record(project_dir, requirement_name)
    return paths


def decide_existing_detailed_design_mode(
    args: argparse.Namespace,
    *,
    paths: dict[str, Path],
    progress: ReviewStageProgress | None = None,
    allow_back: bool = False,
) -> ExistingDetailedDesignMode:
    if bool(getattr(args, "yes", False)) or not stdin_is_interactive():
        return "rerun"
    detailed_design_text = get_markdown_content(paths["detailed_design_path"]).strip()
    if not detailed_design_text:
        return "rerun"
    if progress is not None:
        progress.set_phase("详细设计 / 已有文档处理")
    with _review_progress_context(progress):
        with prompt_metadata(
            allow_back=allow_back,
            back_value=PROMPT_BACK_VALUE,
            stage_key="detailed_design",
            stage_step_index=0,
        ):
            return prompt_select_option(
                title=f"检测到已存在《{paths['detailed_design_path'].name}》",
                options=(
                    ("skip", "跳过详细设计阶段"),
                    ("review_existing", "直接评审现有详细设计"),
                    ("rerun", "从头重跑并重新生成详细设计"),
                ),
                default_value="rerun",
                prompt_text="请选择处理方式",
                preview_path=paths["detailed_design_path"],
                preview_title="现有详细设计",
            )


def decide_ba_strategy(
    args: argparse.Namespace,
    *,
    project_dir: str | Path,
    ba_handoff: RequirementsAnalystHandoff | None,
    allow_back: bool = False,
) -> str:
    if getattr(args, "rebuild_review_ba", False):
        return "rebuild"
    if getattr(args, "reuse_review_ba", False):
        return "reuse"
    if ba_handoff is None:
        return "rebuild"
    if not stdin_is_interactive() or bool(getattr(args, "yes", False)):
        return "reuse"
    if prompt_yes_no_choice(
        "是否复用需求评审阶段的需求分析师继续详细设计阶段",
        True,
        allow_back=allow_back,
        stage_key="detailed_design",
        stage_step_index=2,
    ):
        return "reuse"
    return "rebuild"


def create_design_ba_handoff(
    *,
    project_dir: str | Path,
    requirement_name: str = "",
    selection: ReviewAgentSelection,
) -> RequirementsAnalystHandoff:
    project_root = Path(project_dir).expanduser().resolve()
    selection, config = resolve_agent_run_config_with_recovery(
        selection,
        role_label=_detailed_design_ba_display_name(project_dir=project_root),
    )
    worker = TmuxBatchWorker(
        worker_id="detailed-design-analyst",
        work_dir=project_root,
        config=config,
        runtime_root=build_detailed_design_runtime_root(project_root, requirement_name),
        runtime_metadata={
            "project_dir": str(project_root),
            "requirement_name": str(requirement_name or "").strip(),
            "workflow_action": "stage.a05.start",
        },
    )
    message(render_tmux_start_summary(str(worker.session_name).strip() or "需求分析师", worker))
    return RequirementsAnalystHandoff(
        worker=worker,
        vendor=selection.vendor,
        model=selection.model,
        reasoning_effort=selection.reasoning_effort,
        proxy_url=selection.proxy_url,
    )


def prepare_design_ba_handoff(
    args: argparse.Namespace,
    *,
    project_dir: str | Path,
    requirement_name: str = "",
    ba_handoff: RequirementsAnalystHandoff | None,
    allow_back_first_prompt: bool = False,
) -> tuple[RequirementsAnalystHandoff, bool]:
    strategy_prompted = (
        ba_handoff is not None
        and stdin_is_interactive()
        and not bool(getattr(args, "yes", False))
        and not getattr(args, "rebuild_review_ba", False)
        and not getattr(args, "reuse_review_ba", False)
    )
    strategy = decide_ba_strategy(
        args,
        project_dir=project_dir,
        ba_handoff=ba_handoff,
        allow_back=allow_back_first_prompt and strategy_prompted,
    )
    if strategy == "reuse" and _is_live_ba_handoff(ba_handoff):
        message("复用需求评审阶段的需求分析师继续生成详细设计")
        return ba_handoff, False
    if strategy == "reuse":
        message("请求复用需求评审阶段的需求分析师，但当前没有可复用的 live worker，将回退为重建需求分析师")
    role_label = _detailed_design_ba_display_name(project_dir=project_dir)
    selection = collect_ba_agent_selection(
        args,
        role_label=role_label,
        allow_back_first_step=allow_back_first_prompt and not strategy_prompted,
    )
    message(render_review_agent_selection("进入详细设计阶段（需求分析师）", selection))
    create_kwargs: dict[str, object] = {"project_dir": project_dir, "selection": selection}
    if str(requirement_name or "").strip():
        create_kwargs["requirement_name"] = requirement_name
    new_handoff = create_design_ba_handoff(**create_kwargs)
    if ba_handoff is not None and strategy in {"reuse", "rebuild"}:
        try:
            ba_handoff.worker.request_kill()
        except Exception:
            pass
    return new_handoff, True


def build_detailed_design_init_prompt(paths: dict[str, Path]) -> str:
    return create_detailed_design_ba(
        original_requirement_md=str(paths["original_requirement_path"].resolve()),
        requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
        hitl_record_md=str(paths["hitl_record_path"].resolve()),
    )


def build_detailed_design_prompt(paths: dict[str, Path]) -> str:
    return detailed_design(
        original_requirement_md=str(paths["original_requirement_path"].resolve()),
        requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
        hitl_record_md=str(paths["hitl_record_path"].resolve()),
        detail_design_md=str(paths["detailed_design_path"].resolve()),
    )


def build_ba_init_result_contract(paths: dict[str, Path]) -> TaskResultContract:
    return TaskResultContract(
        turn_id="a05_ba_init",
        phase="a05_ba_init",
        task_kind="a05_ba_init",
        mode="a05_ba_init",
        expected_statuses=("ready",),
        stage_name=DETAILED_DESIGN_TASK_NAME,
        optional_artifacts={
            "original_requirement": paths["original_requirement_path"],
            "requirements_clear": paths["requirements_clear_path"],
            "hitl_record": paths["hitl_record_path"],
        },
    )


def build_detailed_design_generate_result_contract(paths: dict[str, Path]) -> TaskResultContract:
    return TaskResultContract(
        turn_id="a05_detailed_design_generate",
        phase="a05_detailed_design_generate",
        task_kind="a05_detailed_design_generate",
        mode="a05_detailed_design_generate",
        expected_statuses=("completed",),
        stage_name=DETAILED_DESIGN_TASK_NAME,
        required_artifacts={
            "detailed_design": paths["detailed_design_path"],
        },
        optional_artifacts={
            "original_requirement": paths["original_requirement_path"],
            "requirements_clear": paths["requirements_clear_path"],
            "hitl_record": paths["hitl_record_path"],
        },
    )


def build_detailed_design_feedback_result_contract(paths: dict[str, Path]) -> TaskResultContract:
    return build_detailed_design_feedback_contract(paths)


def build_detailed_design_feedback_contract(
    paths: dict[str, Path],
    *,
    mode: str = "a05_detailed_design_feedback",
) -> TaskResultContract:
    expected_statuses = ("hitl", "completed")
    outcome_artifacts = {
        "hitl": {
            "requires": ("ask_human",),
            "optional": ("hitl_record", "requirements_clear"),
            "forbids": ("ba_feedback",),
        },
        "completed": {
            "requires": ("ba_feedback", "detailed_design"),
            "optional": ("hitl_record", "requirements_clear"),
        },
    }
    if mode == "a05_detailed_design_review_limit_force_hitl":
        expected_statuses = ("hitl",)
        outcome_artifacts = {
            "hitl": {
                "requires": ("ask_human",),
                "optional": ("hitl_record", "requirements_clear"),
            },
        }
    return TaskResultContract(
        turn_id=mode,
        phase=mode,
        task_kind=mode,
        mode=mode,
        expected_statuses=expected_statuses,
        stage_name=DETAILED_DESIGN_TASK_NAME,
        optional_artifacts={
            "ask_human": paths["ask_human_path"],
            "ba_feedback": paths["ba_feedback_path"],
            "detailed_design": paths["detailed_design_path"],
            "hitl_record": paths["hitl_record_path"],
            "requirements_clear": paths["requirements_clear_path"],
        },
        outcome_artifacts=outcome_artifacts,
    )


def _build_ba_turn_goal(contract: TaskResultContract) -> TaskTurnGoal | None:
    if contract.mode == "a05_ba_init":
        return TaskTurnGoal(goal_id=contract.mode, outcomes={"ready": OutcomeGoal(status="ready")})
    if contract.mode == "a05_detailed_design_generate":
        return TaskTurnGoal(
            goal_id=contract.mode,
            outcomes={"completed": OutcomeGoal(status="completed", required_aliases=("detailed_design",))},
        )
    if contract.mode == "a05_detailed_design_review_limit_force_hitl":
        return TaskTurnGoal(
            goal_id=contract.mode,
            outcomes={"hitl": OutcomeGoal(status="hitl", required_aliases=("ask_human",))},
        )
    if contract.mode in {"a05_detailed_design_feedback", "a05_detailed_design_review_limit_human_reply"}:
        return TaskTurnGoal(
            goal_id=contract.mode,
            outcomes={
                "hitl": OutcomeGoal(status="hitl", required_aliases=("ask_human",), forbidden_aliases=("ba_feedback",)),
                "completed": OutcomeGoal(
                    status="completed",
                    required_aliases=("ba_feedback", "detailed_design"),
                ),
            },
        )
    return None


def _parse_result_payload(clean_output: str) -> dict[str, object]:
    try:
        payload = json.loads(clean_output)
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"未识别到结构化结果 JSON: {clean_output!r}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("结构化结果必须是 JSON 对象")
    return payload


A05_BA_FEEDBACK_MODES = {"a05_detailed_design_feedback", "a05_detailed_design_review_limit_human_reply"}


def _scope_detailed_design_worker(worker: object, *, project_dir: str | Path, requirement_name: str = "") -> None:
    set_runtime_metadata = getattr(worker, "set_runtime_metadata", None)
    if not callable(set_runtime_metadata):
        return
    with suppress(Exception):
        set_runtime_metadata(
            project_dir=str(Path(project_dir).expanduser().resolve()),
            requirement_name=str(requirement_name or "").strip(),
            workflow_action="stage.a05.start",
        )


def _detailed_design_task_result_decision(result_contract: TaskResultContract):
    if result_contract.mode not in A05_BA_FEEDBACK_MODES:
        return None
    try:
        return resolve_task_result_decision(result_contract)
    except Exception:
        return None


def _task_result_decision_payload(decision) -> dict[str, object]:
    return {
        "status": str(getattr(decision, "status", "") or "").strip(),
        "summary": str(getattr(decision, "summary", "") or "").strip(),
        "artifacts": dict(getattr(decision, "artifacts", {}) or {}),
        "artifact_hashes": dict(getattr(decision, "artifact_hashes", {}) or {}),
    }


def _mark_ba_turn_succeeded_from_materialized_outputs(
    handoff: RequirementsAnalystHandoff,
    *,
    label: str,
    result_contract: TaskResultContract,
    decision,
) -> None:
    worker = handoff.worker
    state: dict[str, object] = {}
    read_state = getattr(worker, "read_state", None)
    if callable(read_state):
        with suppress(Exception):
            raw_state = read_state()
            if isinstance(raw_state, dict):
                state = raw_state
    task_status_path_text = str(state.get("current_task_status_path", "") or "").strip()
    result_path_text = str(
        state.get("current_task_result_path", "") or getattr(worker, "current_task_result_path", "") or ""
    ).strip()
    if task_status_path_text:
        with suppress(Exception):
            write_task_status(Path(task_status_path_text).expanduser().resolve(), status=TASK_STATUS_DONE)
    if result_path_text:
        with suppress(Exception):
            finalize_task_result(
                contract=result_contract,
                result_path=Path(result_path_text).expanduser().resolve(),
                task_status_path=Path(task_status_path_text).expanduser().resolve() if task_status_path_text else None,
            )
    extra = {
        "label": label,
        "result_status": "succeeded",
        "current_task_status_path": task_status_path_text,
        "current_task_result_path": result_path_text,
        "current_task_runtime_status": TASK_STATUS_DONE,
        "agent_started": True,
        "agent_ready": True,
        "agent_state": "READY",
        "health_status": "alive",
        "health_note": "alive",
    }
    record_result = getattr(worker, "_record_result", None)
    if callable(record_result):
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        record_result(
            CommandResult(
                label=label,
                command="",
                exit_code=0,
                raw_output=json.dumps(_task_result_decision_payload(decision), ensure_ascii=False),
                clean_output=json.dumps(_task_result_decision_payload(decision), ensure_ascii=False),
                started_at=timestamp,
                finished_at=timestamp,
            ),
            status=WorkerStatus.SUCCEEDED,
            note=f"done:{label}",
            extra=extra,
        )
        return
    write_state = getattr(worker, "_write_state", None)
    if callable(write_state):
        with suppress(Exception):
            write_state(WorkerStatus.SUCCEEDED, note=f"done:{label}", extra=extra)


def _run_ba_turn(
    handoff: RequirementsAnalystHandoff,
    *,
    label: str,
    prompt: str,
    result_contract: TaskResultContract,
    turn_goal: TaskTurnGoal | None = None,
) -> dict[str, object]:
    return run_task_result_turn_with_repair(
        worker=handoff.worker,
        label=label,
        prompt=prompt,
        result_contract=result_contract,
        parse_result_payload=_parse_result_payload,
        turn_goal=turn_goal or _build_ba_turn_goal(result_contract),
        timeout_sec=DEFAULT_COMMAND_TIMEOUT_SEC,
        stage_label=DETAILED_DESIGN_TASK_NAME,
        role_label=str(getattr(handoff.worker, "session_name", "") or "需求分析师"),
    )


def recreate_design_ba_handoff(
    *,
    project_dir: str | Path,
    requirement_name: str = "",
    previous_handoff: RequirementsAnalystHandoff,
    progress: ReviewStageProgress | None = None,
    required_reconfiguration: bool = False,
    reason_text: str = "",
) -> RequirementsAnalystHandoff | None:
    if required_reconfiguration and not stdin_is_interactive():
        raise RuntimeError("需求分析师需要重新配置智能体，但当前环境无法交互选择厂商/模型。")
    if not required_reconfiguration and not stdin_is_interactive():
        return None
    ba_display_name = _detailed_design_ba_display_name(project_dir=project_dir, handoff=previous_handoff)
    selection_input = ReviewAgentSelection(
        previous_handoff.vendor,
        previous_handoff.model,
        previous_handoff.reasoning_effort,
        previous_handoff.proxy_url,
    )
    selection = (
        prompt_required_replacement_review_agent_selection(
            reason_text=reason_text or f"检测到{ba_display_name}不可继续使用，需要重建需求分析师后继续当前阶段。",
            previous_selection=selection_input,
            force_model_change=True,
            role_label=ba_display_name,
            progress=progress,
        )
        if required_reconfiguration
        else prompt_replacement_review_agent_selection(
            reason_text=f"检测到{ba_display_name}不可继续使用，需要重建需求分析师后继续当前阶段。",
            previous_selection=selection_input,
            force_model_change=True,
            role_label=ba_display_name,
            progress=progress,
        )
    )
    if selection is None:
        return None
    return create_design_ba_handoff(project_dir=project_dir, requirement_name=requirement_name, selection=selection)


def run_ba_turn_with_recovery(
    handoff: RequirementsAnalystHandoff,
    *,
    project_dir: str | Path,
    requirement_name: str = "",
    label: str,
    prompt: str,
    result_contract: TaskResultContract,
    initialize_on_replacement: bool,
    paths: dict[str, Path],
    progress: ReviewStageProgress | None = None,
) -> tuple[RequirementsAnalystHandoff, dict[str, object]]:
    current_handoff = handoff
    needs_initialize = False
    while True:
        _scope_detailed_design_worker(
            current_handoff.worker,
            project_dir=project_dir,
            requirement_name=requirement_name,
        )
        try:
            if needs_initialize:
                _run_ba_turn(
                    current_handoff,
                    label=f"{label}_reinit",
                    prompt=build_detailed_design_init_prompt(paths),
                    result_contract=build_ba_init_result_contract(paths),
                )
                needs_initialize = False
            payload = _run_ba_turn(
                current_handoff,
                label=label,
                prompt=prompt,
                result_contract=result_contract,
            )
            decision = _detailed_design_task_result_decision(result_contract)
            if decision is not None:
                _mark_ba_turn_succeeded_from_materialized_outputs(
                    current_handoff,
                    label=label,
                    result_contract=result_contract,
                    decision=decision,
                )
                payload = _task_result_decision_payload(decision)
            return current_handoff, payload
        except Exception as error:  # noqa: BLE001
            decision = _detailed_design_task_result_decision(result_contract)
            if decision is not None:
                _mark_ba_turn_succeeded_from_materialized_outputs(
                    current_handoff,
                    label=label,
                    result_contract=result_contract,
                    decision=decision,
                )
                return current_handoff, _task_result_decision_payload(decision)
            ba_display_name = _detailed_design_ba_display_name(project_dir=project_dir, handoff=current_handoff)
            auth_error = is_provider_auth_error(error) or worker_has_provider_auth_error(current_handoff.worker)
            provider_runtime_error = worker_has_provider_runtime_error(current_handoff.worker)
            ready_timeout_error = is_agent_ready_timeout_error(error)
            if auth_error or provider_runtime_error or ready_timeout_error:
                effective_requirement_name = requirement_name or _infer_detailed_design_runtime_scope(
                    current_handoff.worker,
                    project_dir,
                )
                reason_text = (
                    f"检测到{ba_display_name}仍在 agent 界面，但模型认证已失效。\n需要更换模型后继续当前阶段。"
                    if auth_error
                    else f"检测到{ba_display_name}的模型服务出现临时运行错误。\n需要更换或重启模型后继续当前阶段。"
                    if provider_runtime_error
                    else f"{ba_display_name}启动超时，未能进入可输入状态。\n需要更换模型后继续当前阶段。"
                )
                mark_worker_awaiting_reconfiguration(current_handoff.worker, reason_text=reason_text)
                replacement = recreate_design_ba_handoff(
                    project_dir=project_dir,
                    requirement_name=effective_requirement_name,
                    previous_handoff=current_handoff,
                    progress=progress,
                    required_reconfiguration=True,
                    reason_text=reason_text,
                )
                if replacement is None:
                    raise RuntimeError(f"{ba_display_name} 无法继续，且未能重建需求分析师") from error
                current_handoff = replacement
                needs_initialize = initialize_on_replacement
                continue
            if is_worker_death_error(error):
                effective_requirement_name = requirement_name or _infer_detailed_design_runtime_scope(
                    current_handoff.worker,
                    project_dir,
                )
                replacement = recreate_design_ba_handoff(
                    project_dir=project_dir,
                    requirement_name=effective_requirement_name,
                    previous_handoff=current_handoff,
                    progress=progress,
                    required_reconfiguration=True,
                    reason_text=f"{ba_display_name}已死亡，必须重新选择厂商/模型/推理/代理后从当前阶段继续。",
                )
                if replacement is None:
                    raise RuntimeError(f"{ba_display_name} 已死亡，且未能重建需求分析师") from error
                current_handoff = replacement
                needs_initialize = initialize_on_replacement
                continue
            raise


def generate_detailed_design_document(
    handoff: RequirementsAnalystHandoff,
    *,
    project_dir: str | Path,
    paths: dict[str, Path],
    initialize_first: bool,
    progress: ReviewStageProgress | None = None,
) -> RequirementsAnalystHandoff:
    current_handoff = handoff
    if progress is not None:
        progress.set_phase("详细设计 / 生成中")
    if initialize_first:
        current_handoff = initialize_detailed_design_ba(
            current_handoff,
            project_dir=project_dir,
            paths=paths,
            progress=progress,
        )
    current_handoff, _ = run_ba_turn_with_recovery(
        current_handoff,
        project_dir=project_dir,
        label="generate_detailed_design",
        prompt=build_detailed_design_prompt(paths),
        result_contract=build_detailed_design_generate_result_contract(paths),
        initialize_on_replacement=True,
        paths=paths,
        progress=progress,
    )
    if not get_markdown_content(paths["detailed_design_path"]).strip():
        raise RuntimeError("详细设计文档为空，未生成有效《详细设计.md》")
    return current_handoff


def initialize_detailed_design_ba(
    handoff: RequirementsAnalystHandoff,
    *,
    project_dir: str | Path,
    paths: dict[str, Path],
    progress: ReviewStageProgress | None = None,
) -> RequirementsAnalystHandoff:
    current_handoff, _ = run_ba_turn_with_recovery(
        handoff,
        project_dir=project_dir,
        label="detailed_design_ba_init",
        prompt=build_detailed_design_init_prompt(paths),
        result_contract=build_ba_init_result_contract(paths),
        initialize_on_replacement=False,
        paths=paths,
        progress=progress,
    )
    return current_handoff


def _discard_unused_ba_handoff(ba_handoff: RequirementsAnalystHandoff | None) -> None:
    if ba_handoff is None:
        return
    try:
        ba_handoff.worker.request_kill()
    except Exception:
        pass


def build_reviewer_completion_contract(
    *,
    reviewer_name: str,
    review_md_path: Path,
    review_json_path: Path,
) -> TurnFileContract:
    def validator(status_path: Path) -> TurnFileResult:
        if not status_path.exists():
            raise FileNotFoundError(f"缺少审核 JSON 文件: {status_path}")
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        matched_item = normalize_review_status_payload(
            payload,
            task_name=DETAILED_DESIGN_TASK_NAME,
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
            payload={"task_name": DETAILED_DESIGN_TASK_NAME, "review_pass": review_pass},
            artifact_paths=artifact_paths,
            artifact_hashes=artifact_hashes,
            validated_at=str(status_path.stat().st_mtime),
        )

    return TurnFileContract(
        turn_id=f"detailed_design_review_{reviewer_name}",
        phase=DETAILED_DESIGN_TASK_NAME,
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
        goal_id="a05_reviewer_round",
        outcomes={
            "review_pass": OutcomeGoal(status="review_pass", required_aliases=("review_json",), forbidden_aliases=("review_md",)),
            "review_fail": OutcomeGoal(status="review_fail", required_aliases=("review_json", "review_md")),
        },
    )


def _reviewer_outputs_satisfy_contract(reviewer: ReviewerRuntime) -> bool:
    return reviewer_outputs_satisfy_contract(reviewer)


def _reviewer_artifact_signature(reviewer: ReviewerRuntime) -> tuple[object, ...]:
    return reviewer_artifact_signature(reviewer)


def _mark_reviewer_turn_succeeded_from_materialized_outputs(
        reviewer: ReviewerRuntime,
        *,
        label: str,
) -> None:
    mark_reviewer_turn_succeeded_from_materialized_outputs(
        reviewer,
        label=label,
        task_name=DETAILED_DESIGN_TASK_NAME,
    )


def create_reviewer_runtime(
    *,
    project_dir: str | Path,
    requirement_name: str,
    reviewer_spec: DetailedDesignReviewerSpec,
    selection: ReviewAgentSelection,
) -> ReviewerRuntime:
    reviewer_identity = _reviewer_spec_identity(reviewer_spec)
    runtime_root = build_detailed_design_runtime_root(project_dir, requirement_name)
    reviewer_display_name = _predict_reviewer_display_name(
        project_dir=project_dir,
        role_name=reviewer_spec.role_name,
    )
    selection, config = resolve_agent_run_config_with_recovery(
        selection,
        role_label=reviewer_display_name,
    )
    worker = TmuxBatchWorker(
        worker_id=build_detailed_design_reviewer_worker_id(reviewer_spec.role_name),
        work_dir=Path(project_dir).expanduser().resolve(),
        config=config,
        runtime_root=runtime_root,
        runtime_metadata={
            "project_dir": str(Path(project_dir).expanduser().resolve()),
            "requirement_name": str(requirement_name).strip(),
            "workflow_action": "stage.a05.start",
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
        contract=build_reviewer_completion_contract(
            reviewer_name=reviewer_identity,
            review_md_path=review_md_path,
            review_json_path=review_json_path,
        ),
    )


def recreate_reviewer_runtime(
    *,
    project_dir: str | Path,
    requirement_name: str,
    reviewer: ReviewerRuntime,
    reviewer_spec: DetailedDesignReviewerSpec,
    progress: ReviewStageProgress | None = None,
    required_reconfiguration: bool = False,
    reason_text: str = "",
) -> ReviewerRuntime | None:
    if required_reconfiguration and not stdin_is_interactive():
        raise RuntimeError("审核智能体需要重新配置，但当前环境无法交互选择厂商/模型。")
    if not required_reconfiguration and not stdin_is_interactive():
        return None
    reviewer_display_name = _reviewer_artifact_agent_name(reviewer)
    selection = (
        prompt_required_replacement_review_agent_selection(
            reason_text=reason_text or f"检测到{reviewer_display_name}不可继续使用，需要重建审核智能体后继续当前阶段。",
            previous_selection=reviewer.selection,
            force_model_change=True,
            role_label=reviewer_display_name,
            progress=progress,
        )
        if required_reconfiguration
        else prompt_replacement_review_agent_selection(
            reason_text=f"检测到{reviewer_display_name}不可继续使用，需要重建审核智能体后继续当前阶段。",
            previous_selection=reviewer.selection,
            force_model_change=True,
            role_label=reviewer_display_name,
            progress=progress,
        )
    )
    if selection is None:
        return None
    replacement = create_reviewer_runtime(
        project_dir=project_dir,
        requirement_name=requirement_name,
        reviewer_spec=reviewer_spec,
        selection=selection,
    )
    if replacement.review_md_path != reviewer.review_md_path and reviewer.review_md_path.exists():
        reviewer.review_md_path.unlink()
    if replacement.review_json_path != reviewer.review_json_path and reviewer.review_json_path.exists():
        reviewer.review_json_path.unlink()
    return replacement

def run_reviewer_turn_with_recreation(
    reviewer: ReviewerRuntime,
    *,
    project_dir: str | Path,
    requirement_name: str,
    reviewer_spec: DetailedDesignReviewerSpec,
    label: str,
    prompt: str,
    progress: ReviewStageProgress | None = None,
) -> ReviewerRuntime | None:
    current_reviewer = reviewer
    while True:
        baseline_satisfies_contract = _reviewer_outputs_satisfy_contract(current_reviewer)
        ensure_review_artifacts(current_reviewer.review_md_path, current_reviewer.review_json_path)
        baseline_signature = _reviewer_artifact_signature(current_reviewer)
        try:
            run_completion_turn_with_repair(
                worker=current_reviewer.worker,
                label=label,
                prompt=prompt,
                completion_contract=current_reviewer.contract,
                turn_goal=_build_reviewer_turn_goal(),
                timeout_sec=DEFAULT_COMMAND_TIMEOUT_SEC,
                stage_label=DETAILED_DESIGN_TASK_NAME,
                role_label=_reviewer_artifact_agent_name(current_reviewer),
            )
            if _reviewer_outputs_satisfy_contract(current_reviewer):
                _mark_reviewer_turn_succeeded_from_materialized_outputs(
                    current_reviewer,
                    label=label,
                )
            return current_reviewer
        except Exception as error:  # noqa: BLE001
            reviewer_display_name = _reviewer_artifact_agent_name(current_reviewer)
            auth_error = is_provider_auth_error(error) or worker_has_provider_auth_error(current_reviewer.worker)
            provider_runtime_error = worker_has_provider_runtime_error(current_reviewer.worker)
            ready_timeout_error = is_agent_ready_timeout_error(error)
            if (
                _reviewer_outputs_satisfy_contract(current_reviewer)
                and (
                    not baseline_satisfies_contract
                    or _reviewer_artifact_signature(current_reviewer) != baseline_signature
                    or reviewer_worker_needs_terminal_success_normalization(current_reviewer)
                )
            ):
                _mark_reviewer_turn_succeeded_from_materialized_outputs(
                    current_reviewer,
                    label=label,
                )
                return current_reviewer
            if is_turn_artifact_contract_error(error):
                return current_reviewer
            if auth_error or provider_runtime_error or ready_timeout_error:
                reason_text = (
                    f"检测到{reviewer_display_name}仍在 agent 界面，但模型认证已失效。\n需要更换模型后继续当前阶段。"
                    if auth_error
                    else f"检测到{reviewer_display_name}的模型服务出现临时运行错误。\n需要更换或重启模型后继续当前阶段。"
                    if provider_runtime_error
                    else f"{reviewer_display_name}启动超时，未能进入可输入状态。\n需要更换模型后继续当前阶段。"
                )
                decision = request_worker_manual_intervention(
                    stage_label="详细设计",
                    role_label=reviewer_display_name,
                    worker=current_reviewer.worker,
                    reason_text=reason_text,
                    target_paths=(current_reviewer.review_md_path, current_reviewer.review_json_path),
                    progress=progress,
                    allow_recreate=True,
                    allow_worker_dead=True,
                )
                if decision == AGENT_INTERVENTION_RECHECK:
                    continue
                if decision == AGENT_INTERVENTION_WORKER_DEAD:
                    with suppress(Exception):
                        current_reviewer.worker.request_kill()
                    message(f"{reviewer_display_name} 已按死亡处理，当前阶段将忽略该审核智能体。")
                    return None
                if decision != AGENT_INTERVENTION_RECREATE:
                    continue
                mark_worker_awaiting_reconfiguration(current_reviewer.worker, reason_text=reason_text)
                replacement = recreate_reviewer_runtime(
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    reviewer=current_reviewer,
                    reviewer_spec=reviewer_spec,
                    progress=progress,
                    required_reconfiguration=True,
                    reason_text=reason_text,
                )
                if replacement is None:
                    message(f"{reviewer_display_name} 重新创建失败，请重新选择恢复方式。")
                    continue
                current_reviewer = replacement
                continue
            if is_worker_death_error(error):
                if not stdin_is_interactive():
                    replacement = recreate_reviewer_runtime(
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        reviewer=current_reviewer,
                        reviewer_spec=reviewer_spec,
                        progress=progress,
                    )
                    if replacement is None:
                        message(f"{reviewer_display_name} 已按死亡处理，当前阶段将忽略该审核智能体。")
                        return None
                    current_reviewer = replacement
                    continue
                decision = request_worker_manual_intervention(
                    stage_label="详细设计",
                    role_label=reviewer_display_name,
                    worker=current_reviewer.worker,
                    reason_text=f"{reviewer_display_name} 执行失败或已死亡。\n{str(error)}",
                    target_paths=(current_reviewer.review_md_path, current_reviewer.review_json_path),
                    progress=progress,
                    allow_recreate=True,
                    allow_worker_dead=True,
                )
                if decision == AGENT_INTERVENTION_RECHECK:
                    continue
                if decision == AGENT_INTERVENTION_WORKER_DEAD:
                    with suppress(Exception):
                        current_reviewer.worker.request_kill()
                    message(f"{reviewer_display_name} 已按死亡处理，当前阶段将忽略该审核智能体。")
                    return None
                if decision != AGENT_INTERVENTION_RECREATE:
                    continue
                replacement = recreate_reviewer_runtime(
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    reviewer=current_reviewer,
                    reviewer_spec=reviewer_spec,
                    progress=progress,
                )
                if replacement is None:
                    message(f"{reviewer_display_name} 重新创建失败，请重新选择恢复方式。")
                    continue
                current_reviewer = replacement
                continue
            raise


def build_reviewer_workers(
    args: argparse.Namespace,
    *,
    project_dir: str | Path,
    requirement_name: str,
    reviewer_specs: Sequence[DetailedDesignReviewerSpec],
    reviewer_selections_by_name: dict[str, ReviewAgentSelection] | None = None,
    progress: ReviewStageProgress | None = None,
) -> list[ReviewerRuntime]:
    if progress is not None:
        progress.set_phase("详细设计 / 启动审核器")
    reviewers: list[ReviewerRuntime] = []
    predicted_session_names: set[str] = set()
    interactive = stdin_is_interactive()
    agent_config = resolve_stage_agent_config(args)
    for reviewer_spec in reviewer_specs:
        reviewer_display_name = _predict_reviewer_display_name(
            project_dir=project_dir,
            role_name=reviewer_spec.role_name,
            occupied_session_names=sorted(predicted_session_names),
        )
        predicted_session_names.add(reviewer_display_name)
        reviewer_key = _reviewer_spec_identity(reviewer_spec)
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
            )
        )
    return reviewers


def _run_parallel_reviewers(
    reviewers: Sequence[ReviewerRuntime],
    *,
    reviewer_specs_by_name: dict[str, DetailedDesignReviewerSpec],
    project_dir: str | Path,
    requirement_name: str,
    round_index: int,
    prompt_builder,
    label_prefix: str,
    progress: ReviewStageProgress | None = None,
) -> list[ReviewerRuntime]:
    if progress is not None:
        progress.set_phase(f"详细设计评审第 {round_index} 轮")
    return run_parallel_reviewer_round(
        reviewers,
        key_func=lambda reviewer: reviewer.reviewer_name,
        run_turn=lambda reviewer: run_reviewer_turn_with_recreation(
            reviewer,
            project_dir=project_dir,
            requirement_name=requirement_name,
            reviewer_spec=reviewer_specs_by_name[reviewer.reviewer_name],
            label=f"{label_prefix}_{sanitize_requirement_name(reviewer.reviewer_name)}_round_{round_index}",
            prompt=prompt_builder(reviewer, reviewer_specs_by_name[reviewer.reviewer_name]),
            progress=progress,
        ),
        error_prefix="详设审核智能体执行失败:",
    )


def repair_reviewer_outputs(
    reviewers: Sequence[ReviewerRuntime],
    *,
    reviewer_specs_by_name: dict[str, DetailedDesignReviewerSpec],
    project_dir: str | Path,
    requirement_name: str,
    round_index: int,
    progress: ReviewStageProgress | None = None,
) -> list[ReviewerRuntime]:
    json_pattern = f"{sanitize_requirement_name(requirement_name)}_评审记录_*.json"
    md_pattern = f"{sanitize_requirement_name(requirement_name)}_详设评审记录_*.md"
    repaired_reviewers = repair_reviewer_round_outputs(
        reviewers,
        key_func=lambda reviewer: reviewer.reviewer_name,
        artifact_name_func=_reviewer_artifact_agent_name,
        check_job=lambda reviewer_names: check_reviewer_job(
            reviewer_names,
            directory=project_dir,
            task_name=DETAILED_DESIGN_TASK_NAME,
            json_pattern=json_pattern,
            md_pattern=md_pattern,
        ),
        run_fix_turn=lambda reviewer, fix_prompt, repair_attempt: run_reviewer_turn_with_recreation(
            reviewer,
            project_dir=project_dir,
            requirement_name=requirement_name,
            reviewer_spec=reviewer_specs_by_name[reviewer.reviewer_name],
            label=f"detailed_design_review_fix_{sanitize_requirement_name(reviewer.reviewer_name)}_round_{round_index}_attempt_{repair_attempt}",
            prompt=fix_prompt,
            progress=progress,
        ),
        max_attempts=MAX_REVIEWER_REPAIR_ATTEMPTS,
        error_prefix="详设审核智能体修复输出失败:",
        final_error="详设审核智能体多次修复后仍未按协议更新文档",
        stage_label=DETAILED_DESIGN_TASK_NAME,
        progress=progress,
        recreate_reviewer=lambda reviewer: _replace_dead_detailed_design_reviewer(
            reviewer,
            project_dir=project_dir,
            requirement_name=requirement_name,
            reviewer_specs_by_name=reviewer_specs_by_name,
            progress=progress,
        ),
    )
    for reviewer in repaired_reviewers:
        if _reviewer_outputs_satisfy_contract(reviewer):
            _mark_reviewer_turn_succeeded_from_materialized_outputs(
                reviewer,
                label=f"detailed_design_repair_verified_{reviewer.reviewer_name}_round_{round_index}",
            )
    return repaired_reviewers


def _collect_design_hitl_response(
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
        message(f"详细设计阶段 HITL 第 {hitl_round} 轮，需要人工补充信息")
        message(f"问题文档: {question_file}")
        message(question_text or "(问题文档为空)")
    with progress.suspended() if progress is not None else nullcontext():
        return collect_multiline_input(
            title=f"HITL 第 {hitl_round} 轮回复",
            empty_retry_message="回复不能为空，请重新输入。",
            question_path=question_file,
            answer_path=answer_path,
            is_hitl=True,
        )
def _read_required_detailed_design_ba_feedback(paths: dict[str, Path]) -> str:
    ba_feedback = get_markdown_content(paths["ba_feedback_path"]).strip()
    if ba_feedback:
        return ba_feedback
    raise RuntimeError(f"详细设计评审未通过，但 {paths['ba_feedback_path'].name} 为空")


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


def _replace_dead_detailed_design_ba(
    handoff: RequirementsAnalystHandoff,
    *,
    project_dir: str | Path,
    progress: ReviewStageProgress | None = None,
) -> RequirementsAnalystHandoff:
    replacement = recreate_design_ba_handoff(
        project_dir=project_dir,
        previous_handoff=handoff,
        progress=progress,
        required_reconfiguration=True,
        reason_text="主工作智能体已死亡，必须重新选择厂商/模型/推理/代理后从当前阶段继续。",
    )
    if replacement is None:
        ba_display_name = _detailed_design_ba_display_name(project_dir=project_dir, handoff=handoff)
        raise RuntimeError(f"{ba_display_name} 已死亡，且未能重建需求分析师")
    return replacement


def _replace_dead_detailed_design_reviewer(
    reviewer: ReviewerRuntime,
    *,
    project_dir: str | Path,
    requirement_name: str,
    reviewer_specs_by_name: dict[str, DetailedDesignReviewerSpec],
    progress: ReviewStageProgress | None = None,
) -> ReviewerRuntime | None:
    reviewer_spec = reviewer_specs_by_name.get(reviewer.reviewer_name)
    if reviewer_spec is None:
        return reviewer
    return recreate_reviewer_runtime(
        project_dir=project_dir,
        requirement_name=requirement_name,
        reviewer=reviewer,
        reviewer_spec=reviewer_spec,
        progress=progress,
    )


def _export_reviewer_handoff(
    reviewers: Sequence[ReviewerRuntime],
    *,
    reviewer_specs_by_name: dict[str, DetailedDesignReviewerSpec],
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


def run_ba_modify_loop(
    handoff: RequirementsAnalystHandoff,
    *,
    project_dir: str | Path,
    paths: dict[str, Path],
    review_msg: str,
    progress: ReviewStageProgress | None = None,
    audit_context: StageAuditRunContext | None = None,
    review_round_index: int | None = None,
) -> RequirementsAnalystHandoff:
    current_handoff = handoff
    if audit_context is not None:
        record_before_cleanup(
            audit_context,
            {
                "ask_human": paths["ask_human_path"],
                "ba_feedback": paths["ba_feedback_path"],
            },
            metadata={"trigger": "detailed_design_modify_prepare"},
            review_round_index=review_round_index,
        )
    ensure_empty_file(paths["ask_human_path"])
    ensure_empty_file(paths["ba_feedback_path"])
    feedback_result_contract = build_detailed_design_feedback_result_contract(paths)
    current_handoff, _ = run_ba_turn_with_recovery(
        current_handoff,
        project_dir=project_dir,
        label="modify_detailed_design",
        prompt=modify_detailed_design(
            review_msg,
            original_requirement_md=str(paths["original_requirement_path"].resolve()),
            requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
            hitl_record_md=str(paths["hitl_record_path"].resolve()),
            detail_design_md=str(paths["detailed_design_path"].resolve()),
            ask_human_md=str(paths["ask_human_path"].resolve()),
            what_just_change=str(paths["ba_feedback_path"].resolve()),
        ),
        result_contract=feedback_result_contract,
        initialize_on_replacement=True,
        paths=paths,
        progress=progress,
    )
    for hitl_round in range(1, MAX_DETAILED_DESIGN_HITL_ROUNDS + 1):
        if not get_markdown_content(paths["ask_human_path"]).strip():
            return current_handoff
        if audit_context is not None:
            append_stage_audit_record(
                audit_context,
                event_type="hitl_question",
                source_paths={"ask_human": paths["ask_human_path"]},
                hitl_round_index=hitl_round,
                review_round_index=review_round_index,
                metadata={"trigger": "detailed_design_modify_hitl"},
            )
        human_msg = _collect_design_hitl_response(
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
                review_round_index=review_round_index,
                metadata={
                    "trigger": "detailed_design_modify_hitl",
                    "human_answer_source": "runtime_payload",
                },
                snapshot_overrides={"human_answer": human_msg},
            )
            record_before_cleanup(
                audit_context,
                {"ask_human": paths["ask_human_path"]},
                metadata={"trigger": "detailed_design_hitl_reply_prepare"},
                hitl_round_index=hitl_round,
                review_round_index=review_round_index,
            )
        ensure_empty_file(paths["ask_human_path"])
        current_handoff, _ = run_ba_turn_with_recovery(
            current_handoff,
            project_dir=project_dir,
            label=f"detailed_design_hitl_reply_round_{hitl_round}",
            prompt=hitl_relpy(
                human_msg,
                review_msg,
                hitl_record_md=str(paths["hitl_record_path"].resolve()),
                detail_design_md=str(paths["detailed_design_path"].resolve()),
                ask_human_md=str(paths["ask_human_path"].resolve()),
                what_just_change=str(paths["ba_feedback_path"].resolve()),
            ),
            result_contract=feedback_result_contract,
            initialize_on_replacement=True,
            paths=paths,
            progress=progress,
        )
    raise RuntimeError(f"详细设计 HITL 轮次超过上限: {MAX_DETAILED_DESIGN_HITL_ROUNDS}")


def run_detailed_design_review_limit_hitl_loop(
    handoff: RequirementsAnalystHandoff,
    *,
    project_dir: str | Path,
    paths: dict[str, Path],
    review_msg: str,
    review_limit: int,
    review_rounds_used: int,
    progress: ReviewStageProgress | None = None,
    human_input_provider=None,
    audit_context: StageAuditRunContext | None = None,
) -> object:
    current_handoff = handoff
    if audit_context is not None:
        record_before_cleanup(
            audit_context,
            {
                "ask_human": paths["ask_human_path"],
                "ba_feedback": paths["ba_feedback_path"],
            },
            metadata={"trigger": "detailed_design_review_limit_hitl_prepare"},
        )
    ensure_empty_file(paths["ask_human_path"])
    ensure_empty_file(paths["ba_feedback_path"])

    def audit_hitl_question(hitl_round: int, ask_human_file: Path) -> None:
        if audit_context is None:
            return
        append_stage_audit_record(
            audit_context,
            event_type="hitl_question",
            source_paths={"ask_human": ask_human_file},
            hitl_round_index=hitl_round,
            metadata={"trigger": "detailed_design_review_limit_hitl"},
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
            metadata={
                "trigger": "detailed_design_review_limit_hitl",
                "human_answer_source": "runtime_payload",
            },
            snapshot_overrides={"human_answer": human_msg},
        )

    def initial_turn() -> object:
        nonlocal current_handoff
        current_handoff, _ = run_ba_turn_with_recovery(
            current_handoff,
            project_dir=project_dir,
            label="detailed_design_review_limit_hitl",
            prompt=build_detailed_design_review_limit_force_hitl_prompt(
                paths=paths,
                review_msg=review_msg,
                review_limit=review_limit,
                review_rounds_used=review_rounds_used,
            ),
            result_contract=build_detailed_design_feedback_contract(
                paths,
                mode="a05_detailed_design_review_limit_force_hitl",
            ),
            initialize_on_replacement=True,
            paths=paths,
            progress=progress,
        )
        return current_handoff

    def human_reply_turn(human_msg: str) -> object:
        nonlocal current_handoff
        current_handoff, _ = run_ba_turn_with_recovery(
            current_handoff,
            project_dir=project_dir,
            label="detailed_design_review_limit_human_reply",
            prompt=build_detailed_design_review_limit_human_reply_prompt(
                paths=paths,
                review_msg=review_msg,
                human_msg=human_msg,
            ),
            result_contract=build_detailed_design_feedback_contract(
                paths,
                mode="a05_detailed_design_review_limit_human_reply",
            ),
            initialize_on_replacement=True,
            paths=paths,
            progress=progress,
        )
        return current_handoff

    result = run_review_limit_hitl_cycle(
        stage_label="详细设计评审超限",
        ask_human_path=paths["ask_human_path"],
        hitl_record_path=paths["hitl_record_path"],
        initial_turn=initial_turn,
        human_reply_turn=human_reply_turn,
        human_input_provider=human_input_provider,
        progress=progress,
        max_hitl_rounds=MAX_DETAILED_DESIGN_HITL_ROUNDS,
        on_hitl_question=audit_hitl_question,
        on_hitl_answer=audit_hitl_answer,
    )
    return result


def _shutdown_workers(
    ba_handoff: RequirementsAnalystHandoff | None,
    reviewers: Sequence[ReviewerRuntime],
    *,
    project_dir: str | Path,
    requirement_name: str = "",
    cleanup_runtime: bool,
    preserve_ba_worker: bool = False,
    preserve_reviewer_keys: Sequence[str] = (),
) -> tuple[str, ...]:
    return shutdown_stage_workers(
        ba_handoff,
        reviewers,
        cleanup_runtime=cleanup_runtime,
        preserve_ba_worker=preserve_ba_worker,
        preserve_reviewer_keys=preserve_reviewer_keys,
        runtime_root_filter=build_detailed_design_runtime_root(project_dir, requirement_name),
    )


def run_detailed_design_stage(
    argv: Sequence[str] | None = None,
    *,
    ba_handoff: RequirementsAnalystHandoff | None = None,
    preserve_workers: bool = False,
) -> DetailedDesignStageResult:
    parser = build_parser()
    args = parser.parse_args(argv)
    allow_previous_stage_back = bool(getattr(args, "allow_previous_stage_back", False))
    if args.project_dir:
        project_dir = str(Path(args.project_dir).expanduser().resolve())
    else:
        project_dir = prompt_project_dir("")
    if args.requirement_name:
        requirement_name = str(args.requirement_name).strip()
    else:
        requirement_name = prompt_requirement_name_selection(project_dir, "").requirement_name

    progress = ReviewStageProgress()
    paths = ensure_detailed_design_inputs(args, project_dir=project_dir, requirement_name=requirement_name)
    active_ba_handoff: RequirementsAnalystHandoff | None = None
    pending_discard_ba_handoff: RequirementsAnalystHandoff | None = None
    reviewer_workers: list[ReviewerRuntime] = []
    cleanup_paths: tuple[str, ...] = ()
    reviewer_specs_by_name: dict[str, DetailedDesignReviewerSpec] = {}
    audit_context: StageAuditRunContext | None = None
    lock_context = requirement_concurrency_lock(
        project_dir,
        requirement_name,
        action="stage.a05.start",
    )
    lock_context.__enter__()
    try:
        audit_context = begin_stage_audit_run(
            project_dir,
            requirement_name,
            "A05",
            metadata={
                "trigger": "run_detailed_design_stage",
                "argv": list(argv or []),
                "args": vars(args),
            },
        )
        existing_design_mode = decide_existing_detailed_design_mode(
            args,
            paths=paths,
            progress=progress,
            allow_back=_consume_stage_back(
                allow_previous_stage_back,
                stdin_is_interactive() and not bool(getattr(args, "yes", False)) and bool(get_markdown_content(paths["detailed_design_path"]).strip()),
            )[0],
        )
        _, allow_previous_stage_back = _consume_stage_back(
            allow_previous_stage_back,
            stdin_is_interactive() and not bool(getattr(args, "yes", False)) and bool(get_markdown_content(paths["detailed_design_path"]).strip()),
        )
        if existing_design_mode == "skip":
            message(f"检测到已存在《{paths['detailed_design_path'].name}》")
            message("用户选择跳过详细设计阶段")
            update_pre_development_task_status(project_dir, requirement_name, task_key="详细设计", completed=True)
            message("已将详细设计标记为完成，继续后续阶段")
            append_stage_audit_record(
                audit_context,
                event_type="stage_passed",
                source_paths={
                    "merged_review": paths["merged_review_path"],
                    "ba_feedback": paths["ba_feedback_path"],
                    "detailed_design": paths["detailed_design_path"],
                },
            )
            return DetailedDesignStageResult(
                project_dir=project_dir,
                requirement_name=requirement_name,
                detailed_design_path=str(paths["detailed_design_path"].resolve()),
                merged_review_path=str(paths["merged_review_path"].resolve()),
                passed=True,
                cleanup_paths=(),
                ba_handoff=None,
                reviewer_handoff=(),
            )
        if existing_design_mode == "review_existing":
            message(f"检测到已存在《{paths['detailed_design_path'].name}》")
            message("将直接评审现有详细设计；若评审未通过，再进入修改流程")
        update_pre_development_task_status(project_dir, requirement_name, task_key="详细设计", completed=False)
        cleanup_stale_detailed_design_runtime_state(project_dir, requirement_name)
        cleanup_existing_detailed_design_artifacts(
            paths,
            requirement_name,
            clear_detailed_design=existing_design_mode == "rerun",
            audit_context=audit_context,
        )
        reviewer_label_getter = lambda reviewer, index: _reviewer_artifact_agent_name(reviewer) or f"详设审核智能体 {index}"  # noqa: E731
        review_round_allow_back, allow_previous_stage_back = _consume_stage_back(
            allow_previous_stage_back,
            stdin_is_interactive() and not str(getattr(args, "review_max_rounds", "") or "").strip(),
        )
        review_round_limit = resolve_review_max_rounds(args, progress=progress, allow_back=review_round_allow_back)
        review_round_policy = ReviewRoundPolicy(review_round_limit)
        if existing_design_mode == "rerun":
            ba_prompted = (
                stdin_is_interactive()
                and (
                    (
                        ba_handoff is not None
                        and not bool(getattr(args, "yes", False))
                        and not getattr(args, "rebuild_review_ba", False)
                        and not getattr(args, "reuse_review_ba", False)
                    )
                    or not any(
                        str(getattr(args, key, "") or "").strip()
                        for key in ("vendor", "model", "effort", "proxy_url")
                    )
                )
            )
            ba_allow_back, allow_previous_stage_back = _consume_stage_back(allow_previous_stage_back, ba_prompted)
            active_ba_handoff, created_new_ba = prepare_design_ba_handoff(
                args,
                project_dir=project_dir,
                requirement_name=requirement_name,
                ba_handoff=ba_handoff,
                allow_back_first_prompt=ba_allow_back,
            )
            reviewer_specs_prompted = stdin_is_interactive() and not any(
                str(item).strip() for item in [*getattr(args, "reviewer_role", []), *getattr(args, "reviewer_role_prompt", [])]
            ) and not resolve_stage_agent_config(args).reviewer_order
            reviewer_specs_allow_back, allow_previous_stage_back = _consume_stage_back(
                allow_previous_stage_back,
                reviewer_specs_prompted,
            )
            reviewer_specs = resolve_reviewer_specs(
                args,
                progress=progress,
                allow_back_first_prompt=reviewer_specs_allow_back,
            )
            reviewer_specs_by_name = {_reviewer_spec_identity(item): item for item in reviewer_specs}
            agent_config = resolve_stage_agent_config(args)
            reviewer_selection_allow_back, allow_previous_stage_back = _consume_stage_back(
                allow_previous_stage_back,
                stdin_is_interactive() and not bool(agent_config.reviewers),
            )
            reviewer_selections_by_name = agent_config.reviewers or collect_reviewer_agent_selections(
                project_dir=project_dir,
                reviewer_specs=reviewer_specs,
                display_name_resolver=lambda current_project_dir, reviewer_spec, occupied_session_names: _predict_reviewer_display_name(
                    project_dir=current_project_dir,
                    role_name=reviewer_spec.role_name,
                    occupied_session_names=occupied_session_names,
                ),
                progress=progress,
                allow_back_first_prompt=reviewer_selection_allow_back,
                stage_key="detailed_design_reviewer_selection",
            )
            _, reviewer_workers, active_ba_handoff = run_main_phase_with_death_handling(
                active_ba_handoff,
                reviewers=(),
                run_phase=lambda handoff: generate_detailed_design_document(
                    handoff,
                    project_dir=project_dir,
                    paths=paths,
                    initialize_first=created_new_ba,
                    progress=progress,
                ),
                replace_dead_main_owner=lambda owner: _replace_dead_detailed_design_ba(
                    owner,
                    project_dir=project_dir,
                    progress=progress,
                ),
                main_label="详细设计需求分析师",
                reviewer_label_getter=reviewer_label_getter,
                notify=message,
            )
            reviewer_workers = build_reviewer_workers(
                args,
                project_dir=project_dir,
                requirement_name=requirement_name,
                reviewer_specs=reviewer_specs,
                reviewer_selections_by_name=reviewer_selections_by_name,
                progress=progress,
            )
        else:
            pending_discard_ba_handoff = ba_handoff
            reviewer_specs_prompted = stdin_is_interactive() and not any(
                str(item).strip() for item in [*getattr(args, "reviewer_role", []), *getattr(args, "reviewer_role_prompt", [])]
            ) and not resolve_stage_agent_config(args).reviewer_order
            reviewer_specs_allow_back, allow_previous_stage_back = _consume_stage_back(
                allow_previous_stage_back,
                reviewer_specs_prompted,
            )
            reviewer_specs = resolve_reviewer_specs(
                args,
                progress=progress,
                allow_back_first_prompt=reviewer_specs_allow_back,
            )
            reviewer_specs_by_name = {_reviewer_spec_identity(item): item for item in reviewer_specs}
            agent_config = resolve_stage_agent_config(args)
            reviewer_selection_allow_back, allow_previous_stage_back = _consume_stage_back(
                allow_previous_stage_back,
                stdin_is_interactive() and not bool(agent_config.reviewers),
            )
            reviewer_selections_by_name = agent_config.reviewers or collect_reviewer_agent_selections(
                project_dir=project_dir,
                reviewer_specs=reviewer_specs,
                display_name_resolver=lambda current_project_dir, reviewer_spec, occupied_session_names: _predict_reviewer_display_name(
                    project_dir=current_project_dir,
                    role_name=reviewer_spec.role_name,
                    occupied_session_names=occupied_session_names,
                ),
                progress=progress,
                allow_back_first_prompt=reviewer_selection_allow_back,
                stage_key="detailed_design_reviewer_selection",
            )
            reviewer_workers = build_reviewer_workers(
                args,
                project_dir=project_dir,
                requirement_name=requirement_name,
                reviewer_specs=reviewer_specs,
                reviewer_selections_by_name=reviewer_selections_by_name,
                progress=progress,
            )

        def initial_prompt_builder(reviewer: ReviewerRuntime, reviewer_spec: DetailedDesignReviewerSpec) -> str:
            return review_detailed_design(
                reviewer_spec.role_prompt,
                init_prompt="",
                task_name=DETAILED_DESIGN_TASK_NAME,
                original_requirement_md=str(paths["original_requirement_path"].resolve()),
                requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
                hitl_record_md=str(paths["hitl_record_path"].resolve()),
                detail_design_md=str(paths["detailed_design_path"].resolve()),
                detail_design_review_md=str(reviewer.review_md_path.resolve()),
                detail_design_review_json=str(reviewer.review_json_path.resolve()),
            )

        round_index = 1
        post_hitl_continue_completed = False
        while True:
            if not review_round_policy.initial_review_done:
                reviewer_workers, active_ba_handoff = run_reviewer_phase_with_death_handling(
                    active_ba_handoff,
                    reviewer_workers,
                    run_phase=lambda reviewers: _run_parallel_reviewers(
                        reviewers,
                        reviewer_specs_by_name=reviewer_specs_by_name,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        round_index=round_index,
                        prompt_builder=initial_prompt_builder,
                        label_prefix="detailed_design_review_init",
                        progress=progress,
                    ),
                    replace_dead_main_owner=lambda owner: _replace_dead_detailed_design_ba(
                        owner,
                        project_dir=project_dir,
                        progress=progress,
                    ),
                    replace_dead_reviewer=lambda reviewer, _index: _replace_dead_detailed_design_reviewer(
                        reviewer,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        reviewer_specs_by_name=reviewer_specs_by_name,
                        progress=progress,
                    ),
                    main_label="详细设计需求分析师",
                    reviewer_label_getter=reviewer_label_getter,
                    notify=message,
                )
                reviewer_workers, active_ba_handoff = run_reviewer_phase_with_death_handling(
                    active_ba_handoff,
                    reviewer_workers,
                    run_phase=lambda reviewers: repair_reviewer_outputs(
                        reviewers,
                        reviewer_specs_by_name=reviewer_specs_by_name,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        round_index=round_index,
                        progress=progress,
                    ),
                    replace_dead_main_owner=lambda owner: _replace_dead_detailed_design_ba(
                        owner,
                        project_dir=project_dir,
                        progress=progress,
                    ),
                    replace_dead_reviewer=lambda reviewer, _index: _replace_dead_detailed_design_reviewer(
                        reviewer,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        reviewer_specs_by_name=reviewer_specs_by_name,
                        progress=progress,
                    ),
                    main_label="详细设计需求分析师",
                    reviewer_label_getter=reviewer_label_getter,
                    notify=message,
                )
            else:
                review_msg = get_markdown_content(paths["merged_review_path"]).strip()
                if not review_msg:
                    raise RuntimeError("详设评审未通过，但合并后的详设评审记录为空")
                if active_ba_handoff is None:
                    if pending_discard_ba_handoff is not None:
                        _discard_unused_ba_handoff(pending_discard_ba_handoff)
                        pending_discard_ba_handoff = None
                    active_ba_handoff, _ = prepare_design_ba_handoff(
                        args,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        ba_handoff=None,
                    )
                    _, reviewer_workers, active_ba_handoff = run_main_phase_with_death_handling(
                        active_ba_handoff,
                        reviewers=reviewer_workers,
                        run_phase=lambda handoff: initialize_detailed_design_ba(
                            handoff,
                            project_dir=project_dir,
                            paths=paths,
                            progress=progress,
                        ),
                        replace_dead_main_owner=lambda owner: _replace_dead_detailed_design_ba(
                            owner,
                            project_dir=project_dir,
                            progress=progress,
                        ),
                        main_label="详细设计需求分析师",
                        reviewer_label_getter=reviewer_label_getter,
                        notify=message,
                    )
                if not post_hitl_continue_completed:
                    _, reviewer_workers, active_ba_handoff = run_main_phase_with_death_handling(
                        active_ba_handoff,
                        reviewers=reviewer_workers,
                        run_phase=lambda handoff: run_ba_modify_loop(
                            handoff,
                            project_dir=project_dir,
                            paths=paths,
                            review_msg=review_msg,
                            progress=progress,
                            audit_context=audit_context,
                            review_round_index=round_index,
                        ),
                        replace_dead_main_owner=lambda owner: _replace_dead_detailed_design_ba(
                            owner,
                            project_dir=project_dir,
                            progress=progress,
                        ),
                        main_label="详细设计需求分析师",
                        reviewer_label_getter=reviewer_label_getter,
                        notify=message,
                    )
                post_hitl_continue_completed = False
                ba_reply = _read_required_detailed_design_ba_feedback(paths)
                append_stage_audit_record(
                    audit_context,
                    event_type="feedback_written",
                    source_paths={
                        "ba_feedback": paths["ba_feedback_path"],
                        "merged_review": paths["merged_review_path"],
                    },
                    review_round_index=round_index,
                    metadata={"trigger": "detailed_design_feedback"},
                )

                def again_prompt_builder(reviewer: ReviewerRuntime, reviewer_spec: DetailedDesignReviewerSpec) -> str:
                    return again_review_detailed_design(
                        ba_reply,
                        task_name=DETAILED_DESIGN_TASK_NAME,
                        original_requirement_md=str(paths["original_requirement_path"].resolve()),
                        requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
                        hitl_record_md=str(paths["hitl_record_path"].resolve()),
                        detail_design_md=str(paths["detailed_design_path"].resolve()),
                        detail_design_review_md=str(reviewer.review_md_path.resolve()),
                        detail_design_review_json=str(reviewer.review_json_path.resolve()),
                    )

                reviewer_workers, active_ba_handoff = run_reviewer_phase_with_death_handling(
                    active_ba_handoff,
                    reviewer_workers,
                    run_phase=lambda reviewers: _run_parallel_reviewers(
                        reviewers,
                        reviewer_specs_by_name=reviewer_specs_by_name,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        round_index=round_index,
                        prompt_builder=again_prompt_builder,
                        label_prefix="detailed_design_review_again",
                        progress=progress,
                    ),
                    replace_dead_main_owner=lambda owner: _replace_dead_detailed_design_ba(
                        owner,
                        project_dir=project_dir,
                        progress=progress,
                    ),
                    replace_dead_reviewer=lambda reviewer, _index: _replace_dead_detailed_design_reviewer(
                        reviewer,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        reviewer_specs_by_name=reviewer_specs_by_name,
                        progress=progress,
                    ),
                    main_label="详细设计需求分析师",
                    reviewer_label_getter=reviewer_label_getter,
                    notify=message,
                )
                reviewer_workers, active_ba_handoff = run_reviewer_phase_with_death_handling(
                    active_ba_handoff,
                    reviewer_workers,
                    run_phase=lambda reviewers: repair_reviewer_outputs(
                        reviewers,
                        reviewer_specs_by_name=reviewer_specs_by_name,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        round_index=round_index,
                        progress=progress,
                    ),
                    replace_dead_main_owner=lambda owner: _replace_dead_detailed_design_ba(
                        owner,
                        project_dir=project_dir,
                        progress=progress,
                    ),
                    replace_dead_reviewer=lambda reviewer, _index: _replace_dead_detailed_design_reviewer(
                        reviewer,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        reviewer_specs_by_name=reviewer_specs_by_name,
                        progress=progress,
                    ),
                    main_label="详细设计需求分析师",
                    reviewer_label_getter=reviewer_label_getter,
                    notify=message,
                )

            review_round_policy.record_review_attempt()
            ensure_active_reviewers(reviewer_workers, stage_label="详细设计")
            review_json_files, review_md_files = _active_reviewer_files(reviewer_workers)
            passed = task_done(
                directory=project_dir,
                file_path=paths["pre_development_path"],
                task_name=DETAILED_DESIGN_TASK_NAME,
                json_pattern=f"{sanitize_requirement_name(requirement_name)}_评审记录_*.json",
                md_pattern=f"{sanitize_requirement_name(requirement_name)}_详设评审记录_*.md",
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
                metadata=_reviewer_audit_metadata(reviewer_workers, trigger="task_done"),
            )
            if passed:
                if pending_discard_ba_handoff is not None:
                    _discard_unused_ba_handoff(pending_discard_ba_handoff)
                    pending_discard_ba_handoff = None
                mark_detailed_design_completed(project_dir, requirement_name)
                append_stage_audit_record(
                    audit_context,
                    event_type="stage_passed",
                    source_paths={
                        "merged_review": paths["merged_review_path"],
                        "ba_feedback": paths["ba_feedback_path"],
                        "detailed_design": paths["detailed_design_path"],
                    },
                    review_round_index=round_index,
                )
                result_ba_handoff = active_ba_handoff if preserve_workers else None
                result_reviewer_handoff = _export_reviewer_handoff(
                    reviewer_workers,
                    reviewer_specs_by_name=reviewer_specs_by_name,
                ) if preserve_workers else ()
                cleanup_paths = _shutdown_workers(
                    active_ba_handoff,
                    reviewer_workers,
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    cleanup_runtime=True,
                    preserve_ba_worker=preserve_workers,
                    preserve_reviewer_keys=[item.reviewer_key for item in result_reviewer_handoff],
                )
                return DetailedDesignStageResult(
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    detailed_design_path=str(paths["detailed_design_path"].resolve()),
                    merged_review_path=str(paths["merged_review_path"].resolve()),
                    passed=True,
                    cleanup_paths=cleanup_paths,
                    ba_handoff=result_ba_handoff,
                    reviewer_handoff=result_reviewer_handoff,
                )
            review_msg = get_markdown_content(paths["merged_review_path"]).strip()
            if not review_msg:
                raise RuntimeError("详设评审未通过，但合并后的详设评审记录为空")
            if review_round_policy.should_escalate_before_next_review():
                if review_round_policy.max_rounds is None:
                    raise RuntimeError("review_round_policy 配置错误：无限轮次不应触发超限 HITL")
                if active_ba_handoff is None:
                    if pending_discard_ba_handoff is not None:
                        _discard_unused_ba_handoff(pending_discard_ba_handoff)
                        pending_discard_ba_handoff = None
                    active_ba_handoff, _ = prepare_design_ba_handoff(
                        args,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        ba_handoff=None,
                    )
                    _, reviewer_workers, active_ba_handoff = run_main_phase_with_death_handling(
                        active_ba_handoff,
                        reviewers=reviewer_workers,
                        run_phase=lambda handoff: initialize_detailed_design_ba(
                            handoff,
                            project_dir=project_dir,
                            paths=paths,
                            progress=progress,
                        ),
                        replace_dead_main_owner=lambda owner: _replace_dead_detailed_design_ba(
                            owner,
                            project_dir=project_dir,
                            progress=progress,
                        ),
                        main_label="详细设计需求分析师",
                        reviewer_label_getter=reviewer_label_getter,
                        notify=message,
                    )
                hitl_result, reviewer_workers, active_ba_handoff = run_main_phase_with_death_handling(
                    active_ba_handoff,
                    reviewers=reviewer_workers,
                    run_phase=lambda handoff: run_detailed_design_review_limit_hitl_loop(
                        handoff,
                        project_dir=project_dir,
                        paths=paths,
                        review_msg=review_msg,
                        review_limit=review_round_policy.max_rounds,
                        review_rounds_used=review_round_policy.quota_count,
                        progress=progress,
                        human_input_provider=(
                            lambda question_path, hitl_round: collect_auto_review_limit_hitl_response(
                                question_path,
                                stage_label="详细设计评审超限",
                                hitl_round=hitl_round,
                            )
                        ) if bool(getattr(args, "yes", False)) else None,
                        audit_context=audit_context,
                    ),
                    owner_getter=lambda result: result.owner,
                    replace_dead_main_owner=lambda owner: _replace_dead_detailed_design_ba(
                        owner,
                        project_dir=project_dir,
                        progress=progress,
                    ),
                    main_label="详细设计需求分析师",
                    reviewer_label_getter=reviewer_label_getter,
                    notify=message,
                )
                post_hitl_continue_completed = bool(getattr(hitl_result, "post_hitl_continue_completed", False))
                review_round_policy.reset_after_hitl()
            round_index += 1
    except Exception as error:
        append_stage_audit_record(
            audit_context,
            event_type="stage_failed",
            source_paths={
                "merged_review": paths["merged_review_path"],
                "ba_feedback": paths["ba_feedback_path"],
                "detailed_design": paths["detailed_design_path"],
            },
            metadata={"error": str(error)},
        )
        if pending_discard_ba_handoff is not None:
            _discard_unused_ba_handoff(pending_discard_ba_handoff)
        _shutdown_workers(
            active_ba_handoff,
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
    redirected, launch = maybe_launch_tui(argv, route="design", action="stage.a05.start")
    if redirected:
        return int(launch)
    try:
        result = run_detailed_design_stage(list(launch))
    except Exception as error:  # noqa: BLE001
        message(error)
        return 1

    message("详细设计完成")
    message(result.detailed_design_path)
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
