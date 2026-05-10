# -*- encoding: utf-8 -*-
"""
@File: A06_TaskSplit.py
@Modify Time: 2026/4/20
@Author: Kevin-Chen
@Descriptions: 任务拆分阶段
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections import Counter
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
from tmux_core.prompt_contracts.common import check_reviewer_job
from tmux_core.prompt_contracts.task_split import (
    again_review_task,
    create_task_split_ba,
    modify_task,
    re_task_md_to_json,
    review_task,
    task_md_to_json,
    task_split,
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
    TmuxBatchWorker,
    Vendor,
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
from tmux_core.stage_kernel.detailed_design import (
    DetailedDesignReviewerSpec,
    build_detailed_design_paths,
    collect_ba_agent_selection,
    run_detailed_design_stage,
    resolve_reviewer_specs as resolve_design_reviewer_specs,
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
    MAX_REVIEWER_REPAIR_ATTEMPTS,
    ReviewLimitHitlConfig,
    ReviewRoundPolicy,
    ReviewAgentHandoff,
    ReviewAgentSelection,
    ReviewStageProgress,
    ReviewerRuntime,
    collect_auto_review_limit_hitl_response,
    collect_reviewer_agent_selections,
    ensure_empty_file,
    ensure_review_artifacts,
    mark_worker_awaiting_reconfiguration,
    parse_review_max_rounds,
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
    collect_review_limit_hitl_response,
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
from T01_tools import get_markdown_content, is_file_empty, is_standard_task_initial_json, is_task_progress_json, task_done
from T08_pre_development import (
    build_pre_development_task_record_path,
    ensure_pre_development_task_record,
    mark_task_split_completed,
    update_pre_development_task_status,
)
from T09_terminal_ops import PROMPT_BACK_VALUE, maybe_launch_tui, message, prompt_metadata, prompt_select_option
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


TASK_SPLIT_TASK_NAME = "任务拆分"
TASK_SPLIT_RUNTIME_ROOT_NAME = ".task_split_runtime"
MAX_TASK_SPLIT_REVIEW_ROUNDS = 5
MAX_TASK_SPLIT_HITL_ROUNDS = 8
MAX_TASK_SPLIT_JSON_REPAIR_ATTEMPTS = 2
PLACEHOLDER_NEXT_STEP = "下一步进入任务开发阶段（待接入）"
TASK_SPLIT_BA_ROLE_DESC = "你是任务拆分阶段的需求分析师，负责将详细设计转换为可执行任务单，并在评审后对任务单做最小化修订。"

TaskSplitReviewerSpec = DetailedDesignReviewerSpec
ExistingTaskSplitMode = Literal["skip", "review_existing", "rerun"]


@dataclass(frozen=True)
class TaskSplitStageResult:
    project_dir: str
    requirement_name: str
    task_md_path: str
    task_json_path: str
    merged_review_path: str
    passed: bool
    cleanup_paths: tuple[str, ...] = ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="任务拆分阶段")
    parser.add_argument("--project-dir", help="项目目录")
    parser.add_argument("--requirement-name", help="需求名称")
    parser.add_argument("--allow-previous-stage-back", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--vendor", help="需求分析师厂商: codex|claude|gemini|opencode")
    parser.add_argument("--model", help="需求分析师模型名称")
    parser.add_argument("--effort", help="需求分析师推理强度")
    parser.add_argument("--proxy-url", default="", help="需求分析师代理端口或完整代理 URL")
    parser.add_argument("--review-max-rounds", default="", help="任务拆分评审最多重试几轮；传 infinite 表示不设上限")
    parser.add_argument("--reviewer-agent", action="append", default=[], help="审核智能体模型配置: name=<key>,vendor=...,model=...,effort=...,proxy=...")
    parser.add_argument("--reviewer-role", action="append", default=[], help="重复传入以覆盖任务拆分评审角色列表")
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


def build_task_split_paths(project_dir: str | Path, requirement_name: str) -> dict[str, Path]:
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
        "task_md_path": project_root / f"{safe_name}_任务单.md",
        "task_json_path": project_root / f"{safe_name}_任务单.json",
        "merged_review_path": project_root / f"{safe_name}_任务单评审记录.md",
        "ba_feedback_path": project_root / f"{safe_name}_需求分析师反馈.md",
    }


def build_task_split_runtime_root(project_dir: str | Path, requirement_name: str = "") -> Path:
    project_root = Path(project_dir).expanduser().resolve()
    safe_requirement = sanitize_requirement_name(requirement_name) if str(requirement_name or "").strip() else ""
    runtime_root = project_root / TASK_SPLIT_RUNTIME_ROOT_NAME
    return runtime_root / safe_requirement if safe_requirement else runtime_root


def _infer_task_split_runtime_scope(worker: object, project_dir: str | Path) -> str:
    try:
        worker_root = Path(getattr(worker, "runtime_root", "")).expanduser().resolve()
        legacy_root = Path(project_dir).expanduser().resolve() / TASK_SPLIT_RUNTIME_ROOT_NAME
    except Exception:
        return ""
    if worker_root.parent == legacy_root:
        return worker_root.name
    return ""


def build_reviewer_artifact_paths(project_dir: str | Path, requirement_name: str, reviewer_name: str) -> tuple[Path, Path]:
    project_root = Path(project_dir).expanduser().resolve()
    safe_name = sanitize_requirement_name(requirement_name)
    artifact_agent_name = sanitize_requirement_name(reviewer_name)
    review_md_path = project_root / f"{safe_name}_任务单评审记录_{artifact_agent_name}.md"
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
            default=MAX_TASK_SPLIT_REVIEW_ROUNDS,
        )
    if not stdin_is_interactive():
        return MAX_TASK_SPLIT_REVIEW_ROUNDS
    if progress is not None and hasattr(progress, "set_phase"):
        progress.set_phase("任务拆分 / 配置最大审核轮次")
    return prompt_review_max_rounds(
        default=MAX_TASK_SPLIT_REVIEW_ROUNDS,
        progress=progress,
        allow_back=allow_back,
        stage_key="task_split",
        stage_step_index=1,
    )


def build_task_split_review_limit_hitl_config(paths: dict[str, Path]) -> ReviewLimitHitlConfig:
    return ReviewLimitHitlConfig(
        stage_label="任务拆分评审超限",
        artifact_label="任务单评审",
        primary_output_path=paths["task_md_path"],
        ask_human_path=paths["ask_human_path"],
        hitl_record_path=paths["hitl_record_path"],
        merged_review_path=paths["merged_review_path"],
        output_summary_path=paths["ba_feedback_path"],
        continue_output_label="任务单.md",
    )


def build_task_split_review_limit_force_hitl_prompt(
    *,
    paths: dict[str, Path],
    review_msg: str,
    review_limit: int,
    review_rounds_used: int,
) -> str:
    return render_review_limit_force_hitl_prompt(
        config=build_task_split_review_limit_hitl_config(paths),
        review_limit=review_limit,
        review_rounds_used=review_rounds_used,
        hitl_record_md=paths["hitl_record_path"],
        extra_inputs=(
            paths["original_requirement_path"],
            paths["requirements_clear_path"],
            paths["detailed_design_path"],
        ),
    ) + f"\n## 当前评审记录\n[REVIEW MSG START]\n{review_msg}\n[REVIEW MSG END]\n"


def build_task_split_review_limit_human_reply_prompt(
    *,
    paths: dict[str, Path],
    review_msg: str,
    human_msg: str,
) -> str:
    return render_review_limit_human_reply_prompt(
        config=build_task_split_review_limit_hitl_config(paths),
        human_msg=human_msg,
        hitl_record_md=paths["hitl_record_path"],
        extra_inputs=(
            paths["original_requirement_path"],
            paths["requirements_clear_path"],
            paths["detailed_design_path"],
        ),
    ) + f"\n## 当前评审记录\n[REVIEW MSG START]\n{review_msg}\n[REVIEW MSG END]\n"


def build_task_split_human_reply_prompt(
    *,
    paths: dict[str, Path],
    review_msg: str,
    human_msg: str,
) -> str:
    return build_task_split_review_limit_human_reply_prompt(
        paths=paths,
        review_msg=review_msg,
        human_msg=human_msg,
    )


def build_task_split_reviewer_worker_id(role_name: str) -> str:
    return f"task-split-review-{str(role_name).strip()}"


def cleanup_existing_task_split_artifacts(
    paths: dict[str, Path],
    requirement_name: str,
    *,
    clear_task_md: bool = True,
    clear_task_json: bool = True,
    audit_context: StageAuditRunContext | None = None,
) -> tuple[str, ...]:
    project_root = paths["project_root"]
    safe_name = sanitize_requirement_name(requirement_name)
    removed: list[str] = []
    review_json_candidates: list[Path] = []
    review_md_candidates: list[Path] = []
    for pattern in (
        f"{safe_name}_评审记录_*.json",
        f"{safe_name}_任务单评审记录_*.md",
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
                "task_md": paths["task_md_path"],
                "task_json": paths["task_json_path"],
            },
            metadata={
                "trigger": "cleanup_existing_task_split_artifacts",
                "clear_task_md": bool(clear_task_md),
                "clear_task_json": bool(clear_task_json),
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
    for candidate, should_clear in (
        (paths["task_md_path"], clear_task_md),
        (paths["task_json_path"], clear_task_json),
    ):
        if should_clear and candidate.exists() and candidate.is_file():
            candidate.write_text("", encoding="utf-8")
            removed.append(str(candidate.resolve()))
    return tuple(dict.fromkeys(removed))


def _task_md_content_hash(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    if not get_markdown_content(path).strip():
        return ""
    return build_prefixed_sha256(path)


def cleanup_stale_task_split_runtime_state(
    project_dir: str | Path,
    requirement_name: str,
) -> tuple[str, ...]:
    runtime_root = Path(project_dir).expanduser().resolve() / TASK_SPLIT_RUNTIME_ROOT_NAME
    if not runtime_root.exists() or not runtime_root.is_dir():
        return ()
    return cleanup_runtime_dirs_by_scope(
        runtime_root=runtime_root,
        project_dir=project_dir,
        requirement_name=requirement_name,
        workflow_action="stage.a06.start",
    )


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


def _task_split_ba_display_name(
    *,
    project_dir: str | Path,
    handoff: RequirementsAnalystHandoff | None = None,
) -> str:
    session_name = str(getattr(getattr(handoff, "worker", None), "session_name", "") or "").strip()
    if session_name:
        return session_name
    return _predict_worker_display_name(project_dir=project_dir, worker_id="task-split-analyst")


def _reviewer_artifact_agent_name(reviewer: ReviewerRuntime) -> str:
    worker = getattr(reviewer, "worker", None)
    session_name = str(getattr(worker, "session_name", "") or "").strip()
    reviewer_name = str(getattr(reviewer, "reviewer_name", "") or "").strip()
    return session_name or reviewer_name


def _reviewer_spec_identity(reviewer_spec: TaskSplitReviewerSpec) -> str:
    return str(reviewer_spec.reviewer_key or reviewer_spec.role_name).strip()


def _reviewer_default_selection() -> ReviewAgentSelection:
    return ReviewAgentSelection(
        vendor=DEFAULT_REQUIREMENTS_CLARIFICATION_VENDOR,
        model=DEFAULT_REQUIREMENTS_CLARIFICATION_MODEL,
        reasoning_effort=DEFAULT_REQUIREMENTS_CLARIFICATION_EFFORT,
        proxy_url="",
    )


def _is_live_ba_handoff(handoff: RequirementsAnalystHandoff | None) -> bool:
    if handoff is None:
        return False
    get_state = getattr(handoff.worker, "get_agent_state", None)
    if callable(get_state):
        with contextlib.suppress(Exception):
            state = get_state()
            state_name = str(getattr(state, "value", state) or "").strip().upper()
            if state_name == "DEAD":
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


def _is_live_reviewer_handoff(handoff: ReviewAgentHandoff) -> bool:
    get_state = getattr(handoff.worker, "get_agent_state", None)
    if callable(get_state):
        with contextlib.suppress(Exception):
            state = get_state()
            state_name = str(getattr(state, "value", state) or "").strip().upper()
            if state_name == "DEAD":
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


def build_task_split_stage_argv(args: argparse.Namespace, *, project_dir: str, requirement_name: str) -> list[str]:
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


def ensure_task_split_inputs(
    args: argparse.Namespace,
    *,
    project_dir: str,
    requirement_name: str,
    ba_handoff: RequirementsAnalystHandoff | None,
    reviewer_handoff: Sequence[ReviewAgentHandoff],
) -> tuple[dict[str, Path], RequirementsAnalystHandoff | None, tuple[ReviewAgentHandoff, ...]]:
    paths = build_task_split_paths(project_dir, requirement_name)
    active_ba_handoff = ba_handoff
    active_reviewer_handoff = tuple(reviewer_handoff)
    if not get_markdown_content(paths["detailed_design_path"]).strip():
        message(f"缺少详细设计文档，先自动执行详细设计阶段: {paths['detailed_design_path'].name}")
        design_result = run_detailed_design_stage(
            build_task_split_stage_argv(args, project_dir=project_dir, requirement_name=requirement_name),
            ba_handoff=ba_handoff,
            preserve_workers=True,
        )
        active_ba_handoff = design_result.ba_handoff
        active_reviewer_handoff = tuple(design_result.reviewer_handoff)
    paths = build_task_split_paths(project_dir, requirement_name)
    if not get_markdown_content(paths["original_requirement_path"]).strip():
        raise RuntimeError(f"缺少原始需求文档: {paths['original_requirement_path']}")
    if not get_markdown_content(paths["requirements_clear_path"]).strip():
        raise RuntimeError(f"缺少需求澄清文档: {paths['requirements_clear_path']}")
    if not get_markdown_content(paths["detailed_design_path"]).strip():
        raise RuntimeError(f"缺少详细设计文档: {paths['detailed_design_path']}")
    ensure_pre_development_task_record(project_dir, requirement_name)
    return paths, active_ba_handoff, active_reviewer_handoff


def decide_existing_task_split_mode(
    args: argparse.Namespace,
    *,
    paths: dict[str, Path],
    progress: ReviewStageProgress | None = None,
    allow_back: bool = False,
) -> ExistingTaskSplitMode:
    if bool(getattr(args, "yes", False)) or not stdin_is_interactive():
        return "rerun"
    task_md_text = get_markdown_content(paths["task_md_path"]).strip()
    if not task_md_text:
        return "rerun"
    has_valid_task_json = is_task_progress_json(paths["task_json_path"])
    if progress is not None:
        progress.set_phase("任务拆分 / 已有文档处理")
    with progress.suspended() if progress is not None else contextlib.nullcontext():
        if has_valid_task_json:
            with prompt_metadata(
                allow_back=allow_back,
                back_value=PROMPT_BACK_VALUE,
                stage_key="task_split",
                stage_step_index=0,
            ):
                return prompt_select_option(
                    title=f"检测到已存在《{paths['task_md_path'].name}》与《{paths['task_json_path'].name}》",
                    options=(
                        ("skip", "跳过任务拆分阶段"),
                        ("review_existing", "直接评审现有任务单"),
                        ("rerun", "从头重跑并重新生成任务单"),
                    ),
                    default_value="rerun",
                    prompt_text="请选择处理方式",
                    preview_path=paths["task_md_path"],
                    preview_title="现有任务单",
                )
        with prompt_metadata(
            allow_back=allow_back,
            back_value=PROMPT_BACK_VALUE,
            stage_key="task_split",
            stage_step_index=0,
        ):
            return prompt_select_option(
                title=f"检测到已存在《{paths['task_md_path'].name}》，但《{paths['task_json_path'].name}》缺失或无效",
                options=(
                    ("review_existing", "直接评审现有任务单"),
                    ("rerun", "从头重跑并重新生成任务单"),
                ),
                default_value="review_existing",
                prompt_text="请选择处理方式",
                preview_path=paths["task_md_path"],
                preview_title="现有任务单",
            )


def create_task_split_ba_handoff(
    *,
    project_dir: str | Path,
    requirement_name: str = "",
    selection: ReviewAgentSelection,
) -> RequirementsAnalystHandoff:
    project_root = Path(project_dir).expanduser().resolve()
    selection, config = resolve_agent_run_config_with_recovery(
        selection,
        role_label=_task_split_ba_display_name(project_dir=project_root),
    )
    worker = TmuxBatchWorker(
        worker_id="task-split-analyst",
        work_dir=project_root,
        config=config,
        runtime_root=build_task_split_runtime_root(project_root, requirement_name),
        runtime_metadata={
            "project_dir": str(project_root),
            "requirement_name": str(requirement_name or "").strip(),
            "workflow_action": "stage.a06.start",
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


def prepare_task_split_ba_handoff(
    args: argparse.Namespace,
    *,
    project_dir: str | Path,
    ba_handoff: RequirementsAnalystHandoff | None,
    allow_back_first_prompt: bool = False,
) -> tuple[RequirementsAnalystHandoff, bool]:
    if _is_live_ba_handoff(ba_handoff):
        message("复用上一阶段的需求分析师继续生成任务单")
        return ba_handoff, False
    role_label = _task_split_ba_display_name(project_dir=project_dir)
    selection = collect_ba_agent_selection(
        args,
        role_label=role_label,
        allow_back_first_step=allow_back_first_prompt,
        stage_key="task_split_main",
    )
    message(render_review_agent_selection("进入任务拆分阶段（需求分析师）", selection))
    create_kwargs: dict[str, object] = {"project_dir": project_dir, "selection": selection}
    requirement_name = str(getattr(args, "requirement_name", "") or "").strip()
    if requirement_name:
        create_kwargs["requirement_name"] = requirement_name
    return create_task_split_ba_handoff(**create_kwargs), True


def build_task_split_init_prompt(paths: dict[str, Path], *, role_desc: str = TASK_SPLIT_BA_ROLE_DESC) -> str:
    return create_task_split_ba(
        role_desc,
        original_requirement_md=str(paths["original_requirement_path"].resolve()),
        requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
        hitl_record_md=str(paths["hitl_record_path"].resolve()),
        detail_design_md=str(paths["detailed_design_path"].resolve()),
    )


def build_task_split_prompt(paths: dict[str, Path]) -> str:
    return task_split(
        task_md=str(paths["task_md_path"].resolve()),
        detail_design_md=str(paths["detailed_design_path"].resolve()),
        hitl_record_md=str(paths["hitl_record_path"].resolve()),
        original_requirement_md=str(paths["original_requirement_path"].resolve()),
        requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
    )


def build_ba_init_result_contract(paths: dict[str, Path]) -> TaskResultContract:
    return TaskResultContract(
        turn_id="a06_ba_init",
        phase="a06_ba_init",
        task_kind="a06_ba_init",
        mode="a06_ba_init",
        expected_statuses=("ready",),
        stage_name=TASK_SPLIT_TASK_NAME,
        optional_artifacts={
            "original_requirement": paths["original_requirement_path"],
            "requirements_clear": paths["requirements_clear_path"],
            "hitl_record": paths["hitl_record_path"],
            "detailed_design": paths["detailed_design_path"],
        },
    )


def build_reviewer_init_result_contract(paths: dict[str, Path]) -> TaskResultContract:
    return TaskResultContract(
        turn_id="a06_reviewer_init",
        phase="a06_reviewer_init",
        task_kind="a06_reviewer_init",
        mode="a06_reviewer_init",
        expected_statuses=("ready",),
        stage_name=TASK_SPLIT_TASK_NAME,
        optional_artifacts={
            "original_requirement": paths["original_requirement_path"],
            "requirements_clear": paths["requirements_clear_path"],
            "hitl_record": paths["hitl_record_path"],
            "detailed_design": paths["detailed_design_path"],
        },
    )


def build_task_split_generate_result_contract(paths: dict[str, Path]) -> TaskResultContract:
    return TaskResultContract(
        turn_id="a06_task_split_generate",
        phase="a06_task_split_generate",
        task_kind="a06_task_split_generate",
        mode="a06_task_split_generate",
        expected_statuses=("completed",),
        stage_name=TASK_SPLIT_TASK_NAME,
        required_artifacts={"task_md": paths["task_md_path"]},
        optional_artifacts={
            "original_requirement": paths["original_requirement_path"],
            "requirements_clear": paths["requirements_clear_path"],
            "hitl_record": paths["hitl_record_path"],
            "detailed_design": paths["detailed_design_path"],
        },
    )


def build_task_split_feedback_result_contract(paths: dict[str, Path]) -> TaskResultContract:
    return build_task_split_feedback_contract(paths)


def build_task_split_feedback_contract(
    paths: dict[str, Path],
    *,
    mode: str = "a06_task_split_feedback",
) -> TaskResultContract:
    expected_statuses = ("hitl", "completed")
    outcome_artifacts = {
        "hitl": {
            "requires": ("ask_human",),
            "optional": ("hitl_record",),
            "forbids": ("ba_feedback",),
        },
        "completed": {
            "requires": ("ba_feedback", "task_md"),
            "optional": ("hitl_record",),
        },
    }
    if mode == "a06_task_split_review_limit_force_hitl":
        expected_statuses = ("hitl",)
        outcome_artifacts = {
            "hitl": {
                "requires": ("ask_human",),
                "optional": ("hitl_record",),
            },
        }
    return TaskResultContract(
        turn_id=mode,
        phase=mode,
        task_kind=mode,
        mode=mode,
        expected_statuses=expected_statuses,
        stage_name=TASK_SPLIT_TASK_NAME,
        optional_artifacts={
            "ask_human": paths["ask_human_path"],
            "ba_feedback": paths["ba_feedback_path"],
            "task_md": paths["task_md_path"],
            "hitl_record": paths["hitl_record_path"],
        },
        outcome_artifacts=outcome_artifacts,
    )


def _build_ba_turn_goal(contract: TaskResultContract) -> TaskTurnGoal | None:
    if contract.mode in {"a06_ba_init", "a06_reviewer_init"}:
        return TaskTurnGoal(goal_id=contract.mode, outcomes={"ready": OutcomeGoal(status="ready")})
    if contract.mode == "a06_task_split_generate":
        return TaskTurnGoal(
            goal_id=contract.mode,
            outcomes={"completed": OutcomeGoal(status="completed", required_aliases=("task_md",))},
        )
    if contract.mode == "a06_task_split_json_generate":
        return TaskTurnGoal(
            goal_id=contract.mode,
            outcomes={"completed": OutcomeGoal(status="completed", required_aliases=("task_json",))},
        )
    if contract.mode == "a06_task_split_review_limit_force_hitl":
        return TaskTurnGoal(
            goal_id=contract.mode,
            outcomes={"hitl": OutcomeGoal(status="hitl", required_aliases=("ask_human",))},
        )
    if contract.mode in {"a06_task_split_feedback", "a06_task_split_review_limit_human_reply"}:
        return TaskTurnGoal(
            goal_id=contract.mode,
            outcomes={
                "hitl": OutcomeGoal(status="hitl", required_aliases=("ask_human",), forbidden_aliases=("ba_feedback",)),
                "completed": OutcomeGoal(status="completed", required_aliases=("ba_feedback", "task_md")),
            },
        )
    return None


def build_task_split_json_result_contract(paths: dict[str, Path]) -> TaskResultContract:
    return TaskResultContract(
        turn_id="a06_task_split_json_generate",
        phase="a06_task_split_json_generate",
        task_kind="a06_task_split_json_generate",
        mode="a06_task_split_json_generate",
        expected_statuses=("completed",),
        stage_name=TASK_SPLIT_TASK_NAME,
        required_artifacts={"task_json": paths["task_json_path"]},
        optional_artifacts={"task_md": paths["task_md_path"]},
    )


def _parse_result_payload(clean_output: str) -> dict[str, object]:
    try:
        payload = json.loads(clean_output)
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"未识别到结构化结果 JSON: {clean_output!r}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("结构化结果必须是 JSON 对象")
    return payload


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
        stage_label=TASK_SPLIT_TASK_NAME,
        role_label=str(getattr(handoff.worker, "session_name", "") or "需求分析师"),
    )


def recreate_task_split_ba_handoff(
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
    ba_display_name = _task_split_ba_display_name(project_dir=project_dir, handoff=previous_handoff)
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
    return create_task_split_ba_handoff(project_dir=project_dir, requirement_name=requirement_name, selection=selection)


def run_ba_turn_with_recovery(
    handoff: RequirementsAnalystHandoff,
    *,
    project_dir: str | Path,
    label: str,
    prompt: str,
    result_contract: TaskResultContract,
    initialize_on_replacement: bool,
    paths: dict[str, Path],
    init_role_desc: str = TASK_SPLIT_BA_ROLE_DESC,
    progress: ReviewStageProgress | None = None,
) -> tuple[RequirementsAnalystHandoff, dict[str, object]]:
    current_handoff = handoff
    needs_initialize = False
    while True:
        try:
            if needs_initialize:
                _run_ba_turn(
                    current_handoff,
                    label=f"{label}_reinit",
                    prompt=build_task_split_init_prompt(paths, role_desc=init_role_desc),
                    result_contract=build_ba_init_result_contract(paths),
                )
                needs_initialize = False
            payload = _run_ba_turn(
                current_handoff,
                label=label,
                prompt=prompt,
                result_contract=result_contract,
            )
            return current_handoff, payload
        except Exception as error:  # noqa: BLE001
            ba_display_name = _task_split_ba_display_name(project_dir=project_dir, handoff=current_handoff)
            auth_error = is_provider_auth_error(error) or worker_has_provider_auth_error(current_handoff.worker)
            provider_runtime_error = worker_has_provider_runtime_error(current_handoff.worker)
            ready_timeout_error = is_agent_ready_timeout_error(error)
            if auth_error or provider_runtime_error or ready_timeout_error:
                effective_requirement_name = requirement_name or _infer_task_split_runtime_scope(
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
                replacement = recreate_task_split_ba_handoff(
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
                effective_requirement_name = requirement_name or _infer_task_split_runtime_scope(
                    current_handoff.worker,
                    project_dir,
                )
                replacement = recreate_task_split_ba_handoff(
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


def generate_task_split_document(
    handoff: RequirementsAnalystHandoff,
    *,
    project_dir: str | Path,
    paths: dict[str, Path],
    initialize_first: bool,
    progress: ReviewStageProgress | None = None,
) -> RequirementsAnalystHandoff:
    current_handoff = handoff
    if initialize_first:
        current_handoff, _ = run_ba_turn_with_recovery(
            current_handoff,
            project_dir=project_dir,
            label="task_split_ba_init",
            prompt=build_task_split_init_prompt(paths),
            result_contract=build_ba_init_result_contract(paths),
            initialize_on_replacement=False,
            paths=paths,
            progress=progress,
        )
    if progress is not None:
        progress.set_phase("任务拆分 / 生成中")
    current_handoff, _ = run_ba_turn_with_recovery(
        current_handoff,
        project_dir=project_dir,
        label="generate_task_split",
        prompt=build_task_split_prompt(paths),
        result_contract=build_task_split_generate_result_contract(paths),
        initialize_on_replacement=True,
        paths=paths,
        progress=progress,
    )
    if not get_markdown_content(paths["task_md_path"]).strip():
        raise RuntimeError("任务单为空，未生成有效《任务单.md》")
    return current_handoff


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
            task_name=TASK_SPLIT_TASK_NAME,
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
            payload={"task_name": TASK_SPLIT_TASK_NAME, "review_pass": review_pass},
            artifact_paths=artifact_paths,
            artifact_hashes=artifact_hashes,
            validated_at=str(status_path.stat().st_mtime),
        )

    return TurnFileContract(
        turn_id=f"task_split_review_{reviewer_name}",
        phase=TASK_SPLIT_TASK_NAME,
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
        goal_id="a06_reviewer_round",
        outcomes={
            "review_pass": OutcomeGoal(status="review_pass", required_aliases=("review_json",), forbidden_aliases=("review_md",)),
            "review_fail": OutcomeGoal(status="review_fail", required_aliases=("review_json", "review_md")),
        },
    )


def _reviewer_has_materialized_outputs(reviewer: ReviewerRuntime) -> bool:
    try:
        payload = json.loads(reviewer.review_json_path.read_text(encoding="utf-8"))
    except Exception:
        payload = None
    try:
        normalize_review_status_payload(
            payload,
            task_name=TASK_SPLIT_TASK_NAME,
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
    reviewer_spec: TaskSplitReviewerSpec,
    selection: ReviewAgentSelection,
) -> ReviewerRuntime:
    reviewer_identity = _reviewer_spec_identity(reviewer_spec)
    runtime_root = build_task_split_runtime_root(project_dir, requirement_name)
    reviewer_display_name = _predict_worker_display_name(
        project_dir=project_dir,
        worker_id=build_task_split_reviewer_worker_id(reviewer_spec.role_name),
    )
    selection, config = resolve_agent_run_config_with_recovery(
        selection,
        role_label=reviewer_display_name,
    )
    worker = TmuxBatchWorker(
        worker_id=build_task_split_reviewer_worker_id(reviewer_spec.role_name),
        work_dir=Path(project_dir).expanduser().resolve(),
        config=config,
        runtime_root=runtime_root,
        runtime_metadata={
            "project_dir": str(Path(project_dir).expanduser().resolve()),
            "requirement_name": str(requirement_name).strip(),
            "workflow_action": "stage.a06.start",
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


def recreate_task_split_reviewer_runtime(
    *,
    project_dir: str | Path,
    requirement_name: str,
    reviewer: ReviewerRuntime,
    reviewer_spec: TaskSplitReviewerSpec,
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


def bind_reviewer_runtime_from_handoff(
    *,
    project_dir: str | Path,
    requirement_name: str,
    handoff: ReviewAgentHandoff,
) -> ReviewerRuntime:
    reviewer_name = str(getattr(handoff.worker, "session_name", "") or "").strip() or handoff.role_name
    set_runtime_metadata = getattr(handoff.worker, "set_runtime_metadata", None)
    if callable(set_runtime_metadata):
        set_runtime_metadata(
            project_dir=str(Path(project_dir).expanduser().resolve()),
            requirement_name=str(requirement_name).strip(),
            workflow_action="stage.a06.start",
    )
    review_md_path, review_json_path = build_reviewer_artifact_paths(project_dir, requirement_name, reviewer_name)
    ensure_review_artifacts(review_md_path, review_json_path)
    return ReviewerRuntime(
        reviewer_name=handoff.reviewer_key,
        selection=handoff.selection,
        worker=handoff.worker,
        review_md_path=review_md_path,
        review_json_path=review_json_path,
        contract=build_reviewer_completion_contract(
            reviewer_name=handoff.reviewer_key,
            review_md_path=review_md_path,
            review_json_path=review_json_path,
        ),
    )


def resolve_reviewer_specs(
    args: argparse.Namespace,
    *,
    reviewer_handoff: Sequence[ReviewAgentHandoff],
    progress: ReviewStageProgress | None = None,
    allow_back_first_prompt: bool = False,
) -> list[TaskSplitReviewerSpec]:
    if reviewer_handoff:
        return [
            TaskSplitReviewerSpec(
                role_name=item.role_name,
                role_prompt=item.role_prompt,
                reviewer_key=item.reviewer_key,
            )
            for item in reviewer_handoff
        ]
    return list(
        resolve_design_reviewer_specs(
            args,
            progress=progress,
            allow_back_first_prompt=allow_back_first_prompt,
        )
    )


def build_reviewer_workers(
    args: argparse.Namespace,
    *,
    project_dir: str | Path,
    requirement_name: str,
    reviewer_specs: Sequence[TaskSplitReviewerSpec],
    reviewer_handoff: Sequence[ReviewAgentHandoff],
    reviewer_selections_by_name: dict[str, ReviewAgentSelection] | None = None,
    progress: ReviewStageProgress | None = None,
) -> tuple[list[ReviewerRuntime], list[ReviewerRuntime]]:
    if progress is not None:
        progress.set_phase("任务拆分 / 启动审核器")
    reviewers: list[ReviewerRuntime] = []
    newly_created_reviewers: list[ReviewerRuntime] = []
    predicted_session_names: set[str] = set()
    interactive = stdin_is_interactive()
    live_handoffs_by_key = {
        item.reviewer_key: item
        for item in reviewer_handoff
        if _is_live_reviewer_handoff(item)
    }
    if live_handoffs_by_key:
        message("复用仍存活的详细设计审核智能体继续审核任务单")
    if reviewer_handoff and len(live_handoffs_by_key) != len(reviewer_handoff):
        message("部分详细设计审核智能体已失效，仅重建失效的任务拆分审核智能体")
    agent_config = resolve_stage_agent_config(args)
    for reviewer_spec in reviewer_specs:
        reviewer_key = _reviewer_spec_identity(reviewer_spec)
        live_handoff = live_handoffs_by_key.get(reviewer_key)
        if live_handoff is not None:
            reviewers.append(
                bind_reviewer_runtime_from_handoff(
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    handoff=live_handoff,
                )
            )
            continue
        reviewer_display_name = _predict_worker_display_name(
            project_dir=project_dir,
            worker_id=build_task_split_reviewer_worker_id(reviewer_spec.role_name),
            occupied_session_names=sorted(predicted_session_names),
        )
        predicted_session_names.add(reviewer_display_name)
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
        reviewer = create_reviewer_runtime(
            project_dir=project_dir,
            requirement_name=requirement_name,
            reviewer_spec=reviewer_spec,
            selection=selection,
        )
        reviewers.append(reviewer)
        newly_created_reviewers.append(reviewer)
    return reviewers, newly_created_reviewers


def _run_reviewer_result_turn(
    reviewer: ReviewerRuntime,
    *,
    label: str,
    prompt: str,
    result_contract: TaskResultContract,
    project_dir: str | Path = ".",
    requirement_name: str = "",
    reviewer_spec: TaskSplitReviewerSpec | None = None,
    progress: ReviewStageProgress | None = None,
) -> ReviewerRuntime | None:
    current_reviewer = reviewer
    while True:
        ensure_review_artifacts(current_reviewer.review_md_path, current_reviewer.review_json_path)
        try:
            run_task_result_turn_with_repair(
                worker=current_reviewer.worker,
                label=label,
                prompt=prompt,
                result_contract=result_contract,
                parse_result_payload=_parse_result_payload,
                turn_goal=TaskTurnGoal(
                    goal_id=f"{result_contract.mode}_reviewer_init",
                    outcomes={"ready": OutcomeGoal(status="ready")},
                ),
                timeout_sec=DEFAULT_COMMAND_TIMEOUT_SEC,
                stage_label=TASK_SPLIT_TASK_NAME,
                role_label=_reviewer_artifact_agent_name(current_reviewer),
            )
            return current_reviewer
        except Exception as error:  # noqa: BLE001
            reviewer_display_name = _reviewer_artifact_agent_name(current_reviewer)
            auth_error = is_provider_auth_error(error) or worker_has_provider_auth_error(current_reviewer.worker)
            provider_runtime_error = worker_has_provider_runtime_error(current_reviewer.worker)
            ready_timeout_error = is_agent_ready_timeout_error(error)
            if is_worker_death_error(error):
                decision = request_worker_manual_intervention(
                    stage_label=TASK_SPLIT_TASK_NAME,
                    role_label=reviewer_display_name,
                    worker=current_reviewer.worker,
                    reason_text=f"{reviewer_display_name} 初始化失败或已死亡。\n{str(error)}",
                    target_paths=(current_reviewer.review_md_path, current_reviewer.review_json_path),
                    progress=progress,
                    allow_recreate=True,
                    allow_worker_dead=True,
                )
                if decision == AGENT_INTERVENTION_RECHECK:
                    continue
                if decision == AGENT_INTERVENTION_WORKER_DEAD:
                    with contextlib.suppress(Exception):
                        current_reviewer.worker.request_kill()
                    message(f"{reviewer_display_name} 已按死亡处理，当前阶段将忽略该审核智能体。")
                    return None
                if decision != AGENT_INTERVENTION_RECREATE:
                    continue
                replacement = recreate_task_split_reviewer_runtime(
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    reviewer=current_reviewer,
                    reviewer_spec=reviewer_spec or TaskSplitReviewerSpec(
                        role_name=current_reviewer.reviewer_name,
                        role_prompt="",
                        reviewer_key=current_reviewer.reviewer_name,
                    ),
                    progress=progress,
                )
                if replacement is None:
                    message(f"{reviewer_display_name} 重新创建失败，请重新选择恢复方式。")
                    continue
                current_reviewer = replacement
                continue
            if auth_error or provider_runtime_error or ready_timeout_error:
                reason_text = (
                    f"检测到{reviewer_display_name}仍在 agent 界面，但模型认证已失效。\n需要更换模型后继续当前阶段。"
                    if auth_error
                    else f"检测到{reviewer_display_name}的模型服务出现临时运行错误。\n需要更换或重启模型后继续当前阶段。"
                    if provider_runtime_error
                    else f"{reviewer_display_name}启动超时，未能进入可输入状态。\n需要更换模型后继续当前阶段。"
                )
                decision = request_worker_manual_intervention(
                    stage_label=TASK_SPLIT_TASK_NAME,
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
                    with contextlib.suppress(Exception):
                        current_reviewer.worker.request_kill()
                    message(f"{reviewer_display_name} 已按死亡处理，当前阶段将忽略该审核智能体。")
                    return None
                if decision != AGENT_INTERVENTION_RECREATE:
                    continue
                mark_worker_awaiting_reconfiguration(current_reviewer.worker, reason_text=reason_text)
                replacement = recreate_task_split_reviewer_runtime(
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    reviewer=current_reviewer,
                    reviewer_spec=reviewer_spec or TaskSplitReviewerSpec(
                        role_name=current_reviewer.reviewer_name,
                        role_prompt="",
                        reviewer_key=current_reviewer.reviewer_name,
                    ),
                    progress=progress,
                    required_reconfiguration=True,
                    reason_text=reason_text,
                )
                if replacement is None:
                    message(f"{reviewer_display_name} 重新创建失败，请重新选择恢复方式。")
                    continue
                current_reviewer = replacement
                continue
            raise RuntimeError(f"{reviewer_display_name} 初始化失败") from error


def _run_reviewer_turn_with_resume(
    reviewer: ReviewerRuntime,
    *,
    label: str,
    prompt: str,
    project_dir: str | Path = ".",
    requirement_name: str = "",
    reviewer_spec: TaskSplitReviewerSpec | None = None,
    progress: ReviewStageProgress | None = None,
) -> ReviewerRuntime | None:
    current_reviewer = reviewer
    while True:
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
                stage_label=TASK_SPLIT_TASK_NAME,
                role_label=_reviewer_artifact_agent_name(current_reviewer),
            )
            return current_reviewer
        except Exception as error:  # noqa: BLE001
            if is_turn_artifact_contract_error(error):
                return current_reviewer
            if (
                _reviewer_has_materialized_outputs(current_reviewer)
                and _reviewer_artifact_signature(current_reviewer) != baseline_signature
            ):
                return current_reviewer
            reviewer_display_name = _reviewer_artifact_agent_name(current_reviewer)
            auth_error = is_provider_auth_error(error) or worker_has_provider_auth_error(current_reviewer.worker)
            provider_runtime_error = worker_has_provider_runtime_error(current_reviewer.worker)
            ready_timeout_error = is_agent_ready_timeout_error(error)
            if is_worker_death_error(error):
                decision = request_worker_manual_intervention(
                    stage_label=TASK_SPLIT_TASK_NAME,
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
                    with contextlib.suppress(Exception):
                        current_reviewer.worker.request_kill()
                    message(f"{reviewer_display_name} 已按死亡处理，当前阶段将忽略该审核智能体。")
                    return None
                if decision != AGENT_INTERVENTION_RECREATE:
                    continue
                replacement = recreate_task_split_reviewer_runtime(
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    reviewer=current_reviewer,
                    reviewer_spec=reviewer_spec or TaskSplitReviewerSpec(
                        role_name=current_reviewer.reviewer_name,
                        role_prompt="",
                        reviewer_key=current_reviewer.reviewer_name,
                    ),
                    progress=progress,
                )
                if replacement is None:
                    message(f"{reviewer_display_name} 重新创建失败，请重新选择恢复方式。")
                    continue
                current_reviewer = replacement
                continue
            if auth_error or provider_runtime_error or ready_timeout_error:
                reason_text = (
                    f"检测到{reviewer_display_name}仍在 agent 界面，但模型认证已失效。\n需要更换模型后继续当前阶段。"
                    if auth_error
                    else f"检测到{reviewer_display_name}的模型服务出现临时运行错误。\n需要更换或重启模型后继续当前阶段。"
                    if provider_runtime_error
                    else f"{reviewer_display_name}启动超时，未能进入可输入状态。\n需要更换模型后继续当前阶段。"
                )
                decision = request_worker_manual_intervention(
                    stage_label=TASK_SPLIT_TASK_NAME,
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
                    with contextlib.suppress(Exception):
                        current_reviewer.worker.request_kill()
                    message(f"{reviewer_display_name} 已按死亡处理，当前阶段将忽略该审核智能体。")
                    return None
                if decision != AGENT_INTERVENTION_RECREATE:
                    continue
                mark_worker_awaiting_reconfiguration(current_reviewer.worker, reason_text=reason_text)
                replacement = recreate_task_split_reviewer_runtime(
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    reviewer=current_reviewer,
                    reviewer_spec=reviewer_spec or TaskSplitReviewerSpec(
                        role_name=current_reviewer.reviewer_name,
                        role_prompt="",
                        reviewer_key=current_reviewer.reviewer_name,
                    ),
                    progress=progress,
                    required_reconfiguration=True,
                    reason_text=reason_text,
                )
                if replacement is None:
                    message(f"{reviewer_display_name} 重新创建失败，请重新选择恢复方式。")
                    continue
                current_reviewer = replacement
                continue
            raise


def initialize_task_split_workers(
    ba_handoff: RequirementsAnalystHandoff,
    *,
    project_dir: str | Path,
    paths: dict[str, Path],
    initialize_ba: bool,
    reviewers: Sequence[ReviewerRuntime],
    reviewer_specs_by_name: dict[str, TaskSplitReviewerSpec],
    initialize_reviewers: bool,
    requirement_name: str = "",
    progress: ReviewStageProgress | None = None,
) -> tuple[RequirementsAnalystHandoff, list[ReviewerRuntime]]:
    current_handoff = ba_handoff
    reviewer_list = list(reviewers)
    if not initialize_ba and not initialize_reviewers:
        return current_handoff, reviewer_list
    if progress is not None:
        progress.set_phase("任务拆分 / 初始化中")
    if initialize_ba:
        current_handoff, _ = run_ba_turn_with_recovery(
            current_handoff,
            project_dir=project_dir,
            label="task_split_ba_init",
            prompt=build_task_split_init_prompt(paths),
            result_contract=build_ba_init_result_contract(paths),
            initialize_on_replacement=False,
            paths=paths,
            progress=progress,
        )
    if initialize_reviewers:
        if progress is not None:
            progress.set_phase("任务拆分 / 初始化审核器")
        reviewer_list = run_parallel_reviewer_round(
            reviewer_list,
            key_func=lambda reviewer: reviewer.reviewer_name,
            run_turn=lambda reviewer: _run_reviewer_result_turn(
                reviewer,
                project_dir=project_dir,
                requirement_name=requirement_name,
                label=f"task_split_reviewer_init_{sanitize_requirement_name(reviewer.reviewer_name)}",
                prompt=build_task_split_init_prompt(
                    paths,
                    role_desc=reviewer_specs_by_name[reviewer.reviewer_name].role_prompt,
                ),
                result_contract=build_reviewer_init_result_contract(paths),
                reviewer_spec=reviewer_specs_by_name[reviewer.reviewer_name],
                progress=progress,
            ),
            error_prefix="任务拆分审核智能体初始化失败:",
        )
    return current_handoff, reviewer_list


def _run_parallel_reviewers(
    reviewers: Sequence[ReviewerRuntime],
    *,
    reviewer_specs_by_name: dict[str, TaskSplitReviewerSpec],
    project_dir: str | Path,
    requirement_name: str,
    round_index: int,
    prompt_builder,
    label_prefix: str,
    progress: ReviewStageProgress | None = None,
) -> list[ReviewerRuntime]:
    if progress is not None:
        progress.set_phase(f"任务拆分评审第 {round_index} 轮")
    return run_parallel_reviewer_round(
        reviewers,
        key_func=lambda reviewer: reviewer.reviewer_name,
        run_turn=lambda reviewer: _run_reviewer_turn_with_resume(
            reviewer,
            project_dir=project_dir,
            requirement_name=requirement_name,
            reviewer_spec=reviewer_specs_by_name[reviewer.reviewer_name],
            label=f"{label_prefix}_{sanitize_requirement_name(reviewer.reviewer_name)}_round_{round_index}",
            prompt=prompt_builder(reviewer, reviewer_specs_by_name[reviewer.reviewer_name]),
            progress=progress,
        ),
        error_prefix="任务拆分审核智能体执行失败:",
    )


def repair_reviewer_outputs(
    reviewers: Sequence[ReviewerRuntime],
    *,
    reviewer_specs_by_name: dict[str, TaskSplitReviewerSpec],
    project_dir: str | Path,
    requirement_name: str,
    round_index: int,
    progress: ReviewStageProgress | None = None,
) -> list[ReviewerRuntime]:
    json_pattern = f"{sanitize_requirement_name(requirement_name)}_评审记录_*.json"
    md_pattern = f"{sanitize_requirement_name(requirement_name)}_任务单评审记录_*.md"
    return repair_reviewer_round_outputs(
        reviewers,
        key_func=lambda reviewer: reviewer.reviewer_name,
        artifact_name_func=_reviewer_artifact_agent_name,
        check_job=lambda reviewer_names: check_reviewer_job(
            reviewer_names,
            directory=project_dir,
            task_name=TASK_SPLIT_TASK_NAME,
            json_pattern=json_pattern,
            md_pattern=md_pattern,
        ),
        run_fix_turn=lambda reviewer, fix_prompt, repair_attempt: _run_reviewer_turn_with_resume(
            reviewer,
            project_dir=project_dir,
            requirement_name=requirement_name,
            reviewer_spec=reviewer_specs_by_name[reviewer.reviewer_name],
            label=f"task_split_review_fix_{sanitize_requirement_name(reviewer.reviewer_name)}_round_{round_index}_attempt_{repair_attempt}",
            prompt=fix_prompt,
        ),
        max_attempts=MAX_REVIEWER_REPAIR_ATTEMPTS,
        error_prefix="任务拆分审核智能体修复输出失败:",
        final_error="任务拆分审核智能体多次修复后仍未按协议更新文档",
        stage_label=TASK_SPLIT_TASK_NAME,
        progress=progress,
        recreate_reviewer=lambda reviewer: _replace_dead_task_split_reviewer(
            reviewer,
            project_dir=project_dir,
            requirement_name=requirement_name,
            reviewer_specs_by_name=reviewer_specs_by_name,
            progress=progress,
        ),
    )


def _read_required_task_split_ba_feedback(paths: dict[str, Path]) -> str:
    ba_feedback = get_markdown_content(paths["ba_feedback_path"]).strip()
    if ba_feedback:
        return ba_feedback
    raise RuntimeError(f"任务拆分评审未通过，但 {paths['ba_feedback_path'].name} 为空")


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


def _replace_dead_task_split_ba(
    handoff: RequirementsAnalystHandoff,
    *,
    project_dir: str | Path,
    progress: ReviewStageProgress | None = None,
) -> RequirementsAnalystHandoff:
    replacement = recreate_task_split_ba_handoff(
        project_dir=project_dir,
        previous_handoff=handoff,
        progress=progress,
        required_reconfiguration=True,
        reason_text="主工作智能体已死亡，必须重新选择厂商/模型/推理/代理后从当前阶段继续。",
    )
    if replacement is None:
        ba_display_name = _task_split_ba_display_name(project_dir=project_dir, handoff=handoff)
        raise RuntimeError(f"{ba_display_name} 已死亡，且未能重建需求分析师")
    return replacement


def _replace_dead_task_split_reviewer(
    reviewer: ReviewerRuntime,
    *,
    project_dir: str | Path,
    requirement_name: str,
    reviewer_specs_by_name: dict[str, TaskSplitReviewerSpec],
    progress: ReviewStageProgress | None = None,
) -> ReviewerRuntime | None:
    reviewer_spec = reviewer_specs_by_name.get(reviewer.reviewer_name)
    if reviewer_spec is None:
        return reviewer
    return recreate_task_split_reviewer_runtime(
        project_dir=project_dir,
        requirement_name=requirement_name,
        reviewer=reviewer,
        reviewer_spec=reviewer_spec,
        progress=progress,
    )


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
                "task_json": paths["task_json_path"],
            },
            metadata={"trigger": "task_split_modify_prepare"},
            review_round_index=review_round_index,
        )
    ensure_empty_file(paths["ask_human_path"])
    ensure_empty_file(paths["ba_feedback_path"])
    ensure_empty_file(paths["task_json_path"])
    feedback_result_contract = build_task_split_feedback_result_contract(paths)
    if progress is not None:
        progress.set_phase("任务拆分 / 按评审修订")
    current_handoff, _ = run_ba_turn_with_recovery(
        current_handoff,
        project_dir=project_dir,
        label="modify_task_split",
        prompt=modify_task(
            review_msg,
            task_md=str(paths["task_md_path"].resolve()),
            what_just_change=str(paths["ba_feedback_path"].resolve()),
        ),
        result_contract=feedback_result_contract,
        initialize_on_replacement=True,
        paths=paths,
        progress=progress,
    )
    for hitl_round in range(1, MAX_TASK_SPLIT_HITL_ROUNDS + 1):
        if not get_markdown_content(paths["ask_human_path"]).strip():
            return current_handoff
        if audit_context is not None:
            append_stage_audit_record(
                audit_context,
                event_type="hitl_question",
                source_paths={"ask_human": paths["ask_human_path"]},
                hitl_round_index=hitl_round,
                review_round_index=review_round_index,
                metadata={"trigger": "task_split_modify_hitl"},
            )
        human_msg = collect_review_limit_hitl_response(
            paths["ask_human_path"],
            stage_label=TASK_SPLIT_TASK_NAME,
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
                    "trigger": "task_split_modify_hitl",
                    "human_answer_source": "runtime_payload",
                },
                snapshot_overrides={"human_answer": human_msg},
            )
            record_before_cleanup(
                audit_context,
                {"ask_human": paths["ask_human_path"]},
                metadata={"trigger": "task_split_hitl_reply_prepare"},
                hitl_round_index=hitl_round,
                review_round_index=review_round_index,
            )
        ensure_empty_file(paths["ask_human_path"])
        if progress is not None:
            progress.set_phase("任务拆分 / 处理 HITL 回复")
        current_handoff, _ = run_ba_turn_with_recovery(
            current_handoff,
            project_dir=project_dir,
            label=f"task_split_hitl_human_reply_round_{hitl_round}",
            prompt=build_task_split_human_reply_prompt(
                paths=paths,
                review_msg=review_msg,
                human_msg=human_msg,
            ),
            result_contract=feedback_result_contract,
            initialize_on_replacement=True,
            paths=paths,
            progress=progress,
        )
    raise RuntimeError(f"任务拆分 HITL 轮次超过上限: {MAX_TASK_SPLIT_HITL_ROUNDS}")


def run_task_split_review_limit_hitl_loop(
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
            metadata={"trigger": "task_split_review_limit_hitl_prepare"},
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
            metadata={"trigger": "task_split_review_limit_hitl"},
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
                "trigger": "task_split_review_limit_hitl",
                "human_answer_source": "runtime_payload",
            },
            snapshot_overrides={"human_answer": human_msg},
        )

    def initial_turn() -> object:
        nonlocal current_handoff
        current_handoff, _ = run_ba_turn_with_recovery(
            current_handoff,
            project_dir=project_dir,
            label="task_split_review_limit_hitl",
            prompt=build_task_split_review_limit_force_hitl_prompt(
                paths=paths,
                review_msg=review_msg,
                review_limit=review_limit,
                review_rounds_used=review_rounds_used,
            ),
            result_contract=build_task_split_feedback_contract(
                paths,
                mode="a06_task_split_review_limit_force_hitl",
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
            label="task_split_review_limit_human_reply",
            prompt=build_task_split_human_reply_prompt(
                paths=paths,
                review_msg=review_msg,
                human_msg=human_msg,
            ),
            result_contract=build_task_split_feedback_contract(
                paths,
                mode="a06_task_split_review_limit_human_reply",
            ),
            initialize_on_replacement=True,
            paths=paths,
            progress=progress,
        )
        return current_handoff

    result = run_review_limit_hitl_cycle(
        stage_label="任务拆分评审超限",
        ask_human_path=paths["ask_human_path"],
        hitl_record_path=paths["hitl_record_path"],
        initial_turn=initial_turn,
        human_reply_turn=human_reply_turn,
        human_input_provider=human_input_provider,
        progress=progress,
        max_hitl_rounds=MAX_TASK_SPLIT_HITL_ROUNDS,
        on_hitl_question=audit_hitl_question,
        on_hitl_answer=audit_hitl_answer,
    )
    return result


def generate_task_split_json(
    handoff: RequirementsAnalystHandoff,
    *,
    project_dir: str | Path,
    paths: dict[str, Path],
    progress: ReviewStageProgress | None = None,
    audit_context: StageAuditRunContext | None = None,
) -> RequirementsAnalystHandoff:
    current_handoff = handoff
    if audit_context is not None:
        record_before_cleanup(
            audit_context,
            {"task_json": paths["task_json_path"]},
            metadata={"trigger": "generate_task_split_json_prepare"},
        )
    ensure_empty_file(paths["task_json_path"])
    if progress is not None:
        progress.set_phase("任务拆分 / 生成任务单 JSON")
    prompts = [
        task_md_to_json(
            task_md=str(paths["task_md_path"].resolve()),
            task_json=str(paths["task_json_path"].resolve()),
        ),
        re_task_md_to_json(
            task_md=str(paths["task_md_path"].resolve()),
            task_json=str(paths["task_json_path"].resolve()),
        ),
    ]
    last_error: Exception | None = None
    for attempt in range(1, MAX_TASK_SPLIT_JSON_REPAIR_ATTEMPTS + 1):
        if audit_context is not None:
            record_before_cleanup(
                audit_context,
                {"task_json": paths["task_json_path"]},
                metadata={"trigger": "generate_task_split_json_attempt", "attempt": attempt},
            )
        ensure_empty_file(paths["task_json_path"])
        prompt = prompts[0] if attempt == 1 else prompts[1]
        try:
            current_handoff, _ = run_ba_turn_with_recovery(
                current_handoff,
                project_dir=project_dir,
                label=f"generate_task_split_json_attempt_{attempt}",
                prompt=prompt,
                result_contract=build_task_split_json_result_contract(paths),
                initialize_on_replacement=True,
                paths=paths,
                progress=progress,
            )
        except Exception as error:  # noqa: BLE001
            last_error = error
            continue
        if is_standard_task_initial_json(paths["task_json_path"]):
            return current_handoff
        last_error = RuntimeError("任务单 JSON 未通过结构校验")
    raise RuntimeError("任务单 JSON 生成失败") from last_error


def _shutdown_workers(
    ba_handoff: RequirementsAnalystHandoff | None,
    reviewers: Sequence[ReviewerRuntime],
    *,
    project_dir: str | Path,
    requirement_name: str = "",
    cleanup_runtime: bool,
) -> tuple[str, ...]:
    return shutdown_stage_workers(
        ba_handoff,
        reviewers,
        cleanup_runtime=cleanup_runtime,
    )


def run_task_split_stage(
    argv: Sequence[str] | None = None,
    *,
    ba_handoff: RequirementsAnalystHandoff | None = None,
    reviewer_handoff: Sequence[ReviewAgentHandoff] | None = None,
) -> TaskSplitStageResult:
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

    progress = ReviewStageProgress(initial_phase="任务拆分准备中")
    active_ba_handoff = ba_handoff
    active_reviewer_handoff = tuple(reviewer_handoff or ())
    reviewer_workers: list[ReviewerRuntime] = []
    cleanup_paths: tuple[str, ...] = ()
    audit_context: StageAuditRunContext | None = None
    lock_context = requirement_concurrency_lock(
        project_dir,
        requirement_name,
        action="stage.a06.start",
    )
    lock_context.__enter__()
    try:
        audit_context = begin_stage_audit_run(
            project_dir,
            requirement_name,
            "A06",
            metadata={
                "trigger": "run_task_split_stage",
                "argv": list(argv or []),
                "args": vars(args),
            },
        )
        paths, active_ba_handoff, active_reviewer_handoff = ensure_task_split_inputs(
            args,
            project_dir=project_dir,
            requirement_name=requirement_name,
            ba_handoff=active_ba_handoff,
            reviewer_handoff=active_reviewer_handoff,
        )
        existing_task_prompted = stdin_is_interactive() and not bool(getattr(args, "yes", False)) and bool(get_markdown_content(paths["task_md_path"]).strip())
        existing_task_allow_back, allow_previous_stage_back = _consume_stage_back(
            allow_previous_stage_back,
            existing_task_prompted,
        )
        existing_task_split_mode = decide_existing_task_split_mode(
            args,
            paths=paths,
            progress=progress,
            allow_back=existing_task_allow_back,
        )
        initial_task_md_hash = _task_md_content_hash(paths["task_md_path"]) if existing_task_split_mode == "review_existing" else ""
        if existing_task_split_mode == "skip":
            message(f"检测到已存在《{paths['task_md_path'].name}》与《{paths['task_json_path'].name}》")
            message("用户选择跳过任务拆分阶段")
            cleanup_paths = cleanup_stale_task_split_runtime_state(project_dir, requirement_name)
            update_pre_development_task_status(project_dir, requirement_name, task_key="任务拆分", completed=True)
            message("已将任务拆分标记为完成，继续后续阶段")
            append_stage_audit_record(
                audit_context,
                event_type="stage_passed",
                source_paths={
                    "merged_review": paths["merged_review_path"],
                    "ba_feedback": paths["ba_feedback_path"],
                    "task_md": paths["task_md_path"],
                    "task_json": paths["task_json_path"],
                },
            )
            return TaskSplitStageResult(
                project_dir=project_dir,
                requirement_name=requirement_name,
                task_md_path=str(paths["task_md_path"].resolve()),
                task_json_path=str(paths["task_json_path"].resolve()),
                merged_review_path=str(paths["merged_review_path"].resolve()),
                passed=True,
                cleanup_paths=cleanup_paths,
            )
        if existing_task_split_mode == "review_existing":
            message(f"检测到已存在《{paths['task_md_path'].name}》")
            message("将直接评审现有任务单；若评审未通过，再进入修改流程")
        update_pre_development_task_status(project_dir, requirement_name, task_key="任务拆分", completed=False)
        cleanup_stale_task_split_runtime_state(project_dir, requirement_name)
        cleanup_existing_task_split_artifacts(
            paths,
            requirement_name,
            clear_task_md=existing_task_split_mode == "rerun",
            clear_task_json=existing_task_split_mode == "rerun",
            audit_context=audit_context,
        )
        reviewer_label_getter = lambda reviewer, index: _reviewer_artifact_agent_name(reviewer) or f"任务拆分审核智能体 {index}"  # noqa: E731
        review_round_allow_back, allow_previous_stage_back = _consume_stage_back(
            allow_previous_stage_back,
            stdin_is_interactive() and not str(getattr(args, "review_max_rounds", "") or "").strip(),
        )
        review_round_limit = resolve_review_max_rounds(args, progress=progress, allow_back=review_round_allow_back)
        review_round_policy = ReviewRoundPolicy(review_round_limit)
        reviewer_specs_prompted = stdin_is_interactive() and not reviewer_handoff and not any(
            str(item).strip() for item in [*getattr(args, "reviewer_role", []), *getattr(args, "reviewer_role_prompt", [])]
        ) and not resolve_stage_agent_config(args).reviewer_order
        reviewer_specs_allow_back, allow_previous_stage_back = _consume_stage_back(
            allow_previous_stage_back,
            reviewer_specs_prompted,
        )
        reviewer_specs = resolve_reviewer_specs(
            args,
            reviewer_handoff=active_reviewer_handoff,
            progress=progress,
            allow_back_first_prompt=reviewer_specs_allow_back,
        )
        reviewer_specs_by_name = {_reviewer_spec_identity(item): item for item in reviewer_specs}
        live_reviewer_keys = tuple(
            item.reviewer_key
            for item in active_reviewer_handoff
            if _is_live_reviewer_handoff(item)
        )
        agent_config = resolve_stage_agent_config(args)
        reviewer_selection_allow_back, allow_previous_stage_back = _consume_stage_back(
            allow_previous_stage_back,
            stdin_is_interactive() and not bool(agent_config.reviewers) and bool(reviewer_specs),
        )
        reviewer_selections_by_name = agent_config.reviewers or collect_reviewer_agent_selections(
            project_dir=project_dir,
            reviewer_specs=reviewer_specs,
            display_name_resolver=lambda current_project_dir, reviewer_spec, occupied_session_names: _predict_worker_display_name(
                project_dir=current_project_dir,
                worker_id=build_task_split_reviewer_worker_id(reviewer_spec.role_name),
                occupied_session_names=occupied_session_names,
            ),
            progress=progress,
            skip_reviewer_keys=live_reviewer_keys,
            allow_back_first_prompt=reviewer_selection_allow_back,
            stage_key="task_split_reviewer_selection",
        )
        created_new_ba = False
        if existing_task_split_mode == "rerun":
            ba_prompted = stdin_is_interactive() and not _is_live_ba_handoff(active_ba_handoff) and not any(
                str(getattr(args, key, "") or "").strip()
                for key in ("vendor", "model", "effort", "proxy_url")
            )
            ba_allow_back, allow_previous_stage_back = _consume_stage_back(allow_previous_stage_back, ba_prompted)
            active_ba_handoff, created_new_ba = prepare_task_split_ba_handoff(
                args,
                project_dir=project_dir,
                ba_handoff=active_ba_handoff,
                allow_back_first_prompt=ba_allow_back,
            )
            if created_new_ba:
                _, reviewer_workers, active_ba_handoff = run_main_phase_with_death_handling(
                    active_ba_handoff,
                    reviewers=(),
                    run_phase=lambda handoff: initialize_task_split_workers(
                        handoff,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        paths=paths,
                        initialize_ba=True,
                        reviewers=(),
                        reviewer_specs_by_name=reviewer_specs_by_name,
                        initialize_reviewers=False,
                        progress=progress,
                    )[0],
                    replace_dead_main_owner=lambda owner: _replace_dead_task_split_ba(
                        owner,
                        project_dir=project_dir,
                        progress=progress,
                    ),
                    main_label="任务拆分需求分析师",
                    reviewer_label_getter=reviewer_label_getter,
                    notify=message,
                )
        reviewer_workers: list[ReviewerRuntime] = []
        new_reviewer_workers: list[ReviewerRuntime] = []
        if existing_task_split_mode == "rerun":
            _, reviewer_workers, active_ba_handoff = run_main_phase_with_death_handling(
                active_ba_handoff,
                reviewers=reviewer_workers,
                run_phase=lambda handoff: generate_task_split_document(
                    handoff,
                    project_dir=project_dir,
                    paths=paths,
                    initialize_first=False,
                    progress=progress,
                ),
                replace_dead_main_owner=lambda owner: _replace_dead_task_split_ba(
                    owner,
                    project_dir=project_dir,
                    progress=progress,
                ),
                main_label="任务拆分需求分析师",
                reviewer_label_getter=reviewer_label_getter,
                notify=message,
            )

        def initial_prompt_builder(reviewer: ReviewerRuntime, reviewer_spec: TaskSplitReviewerSpec) -> str:
            return review_task(
                reviewer_spec.role_prompt,
                task_name=TASK_SPLIT_TASK_NAME,
                original_requirement_md=str(paths["original_requirement_path"].resolve()),
                requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
                hitl_record_md=str(paths["hitl_record_path"].resolve()),
                task_md=str(paths["task_md_path"].resolve()),
                detail_design_md=str(paths["detailed_design_path"].resolve()),
                task_review_md=str(reviewer.review_md_path.resolve()),
                task_review_json=str(reviewer.review_json_path.resolve()),
            )

        round_index = 1
        post_hitl_continue_completed = False
        while True:
            if not review_round_policy.initial_review_done:
                reviewer_workers, new_reviewer_workers = build_reviewer_workers(
                    args,
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    reviewer_specs=reviewer_specs,
                    reviewer_handoff=active_reviewer_handoff,
                    reviewer_selections_by_name=reviewer_selections_by_name,
                    progress=progress,
                )
                if new_reviewer_workers:
                    created_reviewer_keys = {item.reviewer_name for item in new_reviewer_workers}
                    new_reviewer_workers, active_ba_handoff = run_reviewer_phase_with_death_handling(
                        active_ba_handoff,
                        new_reviewer_workers,
                        run_phase=lambda active_reviewers: initialize_task_split_workers(
                            active_ba_handoff,
                            project_dir=project_dir,
                            requirement_name=requirement_name,
                            paths=paths,
                            initialize_ba=False,
                            reviewers=active_reviewers,
                            reviewer_specs_by_name=reviewer_specs_by_name,
                            initialize_reviewers=True,
                            progress=progress,
                        )[1],
                        replace_dead_main_owner=lambda owner: _replace_dead_task_split_ba(
                            owner,
                            project_dir=project_dir,
                            progress=progress,
                        ),
                        replace_dead_reviewer=lambda reviewer, _index: _replace_dead_task_split_reviewer(
                            reviewer,
                            project_dir=project_dir,
                            requirement_name=requirement_name,
                            reviewer_specs_by_name=reviewer_specs_by_name,
                            progress=progress,
                        ),
                        main_label="任务拆分需求分析师",
                        reviewer_label_getter=reviewer_label_getter,
                        notify=message,
                    )
                    reviewer_workers = [item for item in reviewer_workers if item.reviewer_name not in created_reviewer_keys]
                    reviewer_workers.extend(new_reviewer_workers)
                reviewer_workers, active_ba_handoff = run_reviewer_phase_with_death_handling(
                    active_ba_handoff,
                    reviewer_workers,
                    run_phase=lambda active_reviewers: _run_parallel_reviewers(
                        active_reviewers,
                        reviewer_specs_by_name=reviewer_specs_by_name,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        round_index=round_index,
                        prompt_builder=initial_prompt_builder,
                        label_prefix="task_split_review_init",
                        progress=progress,
                    ),
                    replace_dead_main_owner=lambda owner: _replace_dead_task_split_ba(
                        owner,
                        project_dir=project_dir,
                        progress=progress,
                    ),
                    replace_dead_reviewer=lambda reviewer, _index: _replace_dead_task_split_reviewer(
                        reviewer,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        reviewer_specs_by_name=reviewer_specs_by_name,
                        progress=progress,
                    ),
                    main_label="任务拆分需求分析师",
                    reviewer_label_getter=reviewer_label_getter,
                    notify=message,
                )
                reviewer_workers, active_ba_handoff = run_reviewer_phase_with_death_handling(
                    active_ba_handoff,
                    reviewer_workers,
                    run_phase=lambda active_reviewers: repair_reviewer_outputs(
                        active_reviewers,
                        reviewer_specs_by_name=reviewer_specs_by_name,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        round_index=round_index,
                        progress=progress,
                    ),
                    replace_dead_main_owner=lambda owner: _replace_dead_task_split_ba(
                        owner,
                        project_dir=project_dir,
                        progress=progress,
                    ),
                    replace_dead_reviewer=lambda reviewer, _index: _replace_dead_task_split_reviewer(
                        reviewer,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        reviewer_specs_by_name=reviewer_specs_by_name,
                        progress=progress,
                    ),
                    main_label="任务拆分需求分析师",
                    reviewer_label_getter=reviewer_label_getter,
                    notify=message,
                )
            else:
                review_msg = get_markdown_content(paths["merged_review_path"]).strip()
                if not review_msg:
                    raise RuntimeError("任务拆分评审未通过，但合并后的任务单评审记录为空")
                if active_ba_handoff is None or not _is_live_ba_handoff(active_ba_handoff):
                    active_ba_handoff, created_new_ba = prepare_task_split_ba_handoff(
                        args,
                        project_dir=project_dir,
                        ba_handoff=active_ba_handoff,
                    )
                    if created_new_ba:
                        _, reviewer_workers, active_ba_handoff = run_main_phase_with_death_handling(
                            active_ba_handoff,
                            reviewers=reviewer_workers,
                            run_phase=lambda handoff: initialize_task_split_workers(
                                handoff,
                                project_dir=project_dir,
                                requirement_name=requirement_name,
                                paths=paths,
                                initialize_ba=True,
                                reviewers=reviewer_workers,
                                reviewer_specs_by_name=reviewer_specs_by_name,
                                initialize_reviewers=False,
                                progress=progress,
                            )[0],
                            replace_dead_main_owner=lambda owner: _replace_dead_task_split_ba(
                                owner,
                                project_dir=project_dir,
                                progress=progress,
                            ),
                            main_label="任务拆分需求分析师",
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
                        replace_dead_main_owner=lambda owner: _replace_dead_task_split_ba(
                            owner,
                            project_dir=project_dir,
                            progress=progress,
                        ),
                        main_label="任务拆分需求分析师",
                        reviewer_label_getter=reviewer_label_getter,
                        notify=message,
                    )
                post_hitl_continue_completed = False
                ba_reply = _read_required_task_split_ba_feedback(paths)
                append_stage_audit_record(
                    audit_context,
                    event_type="feedback_written",
                    source_paths={
                        "ba_feedback": paths["ba_feedback_path"],
                        "merged_review": paths["merged_review_path"],
                    },
                    review_round_index=round_index,
                    metadata={"trigger": "task_split_feedback"},
                )

                def again_prompt_builder(reviewer: ReviewerRuntime, reviewer_spec: TaskSplitReviewerSpec) -> str:
                    del reviewer_spec
                    return again_review_task(
                        ba_reply,
                        task_name=TASK_SPLIT_TASK_NAME,
                        task_md=str(paths["task_md_path"].resolve()),
                        task_review_md=str(reviewer.review_md_path.resolve()),
                        task_review_json=str(reviewer.review_json_path.resolve()),
                    )

                reviewer_workers, active_ba_handoff = run_reviewer_phase_with_death_handling(
                    active_ba_handoff,
                    reviewer_workers,
                    run_phase=lambda active_reviewers: _run_parallel_reviewers(
                        active_reviewers,
                        reviewer_specs_by_name=reviewer_specs_by_name,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        round_index=round_index,
                        prompt_builder=again_prompt_builder,
                        label_prefix="task_split_review_again",
                        progress=progress,
                    ),
                    replace_dead_main_owner=lambda owner: _replace_dead_task_split_ba(
                        owner,
                        project_dir=project_dir,
                        progress=progress,
                    ),
                    replace_dead_reviewer=lambda reviewer, _index: _replace_dead_task_split_reviewer(
                        reviewer,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        reviewer_specs_by_name=reviewer_specs_by_name,
                        progress=progress,
                    ),
                    main_label="任务拆分需求分析师",
                    reviewer_label_getter=reviewer_label_getter,
                    notify=message,
                )
                reviewer_workers, active_ba_handoff = run_reviewer_phase_with_death_handling(
                    active_ba_handoff,
                    reviewer_workers,
                    run_phase=lambda active_reviewers: repair_reviewer_outputs(
                        active_reviewers,
                        reviewer_specs_by_name=reviewer_specs_by_name,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        round_index=round_index,
                        progress=progress,
                    ),
                    replace_dead_main_owner=lambda owner: _replace_dead_task_split_ba(
                        owner,
                        project_dir=project_dir,
                        progress=progress,
                    ),
                    replace_dead_reviewer=lambda reviewer, _index: _replace_dead_task_split_reviewer(
                        reviewer,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        reviewer_specs_by_name=reviewer_specs_by_name,
                        progress=progress,
                    ),
                    main_label="任务拆分需求分析师",
                    reviewer_label_getter=reviewer_label_getter,
                    notify=message,
                )

            review_round_policy.record_review_attempt()
            ensure_active_reviewers(reviewer_workers, stage_label="任务拆分")
            review_json_files, review_md_files = _active_reviewer_files(reviewer_workers)
            passed = task_done(
                directory=project_dir,
                file_path=paths["pre_development_path"],
                task_name=TASK_SPLIT_TASK_NAME,
                json_pattern=f"{sanitize_requirement_name(requirement_name)}_评审记录_*.json",
                md_pattern=f"{sanitize_requirement_name(requirement_name)}_任务单评审记录_*.md",
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
                current_task_md_hash = _task_md_content_hash(paths["task_md_path"])
                should_regenerate_task_json = (
                    existing_task_split_mode != "review_existing"
                    or current_task_md_hash != initial_task_md_hash
                    or not is_standard_task_initial_json(paths["task_json_path"])
                )
                try:
                    if should_regenerate_task_json:
                        if active_ba_handoff is None or not _is_live_ba_handoff(active_ba_handoff):
                            active_ba_handoff, created_new_ba = prepare_task_split_ba_handoff(
                                args,
                                project_dir=project_dir,
                                ba_handoff=active_ba_handoff,
                            )
                            if created_new_ba:
                                _, reviewer_workers, active_ba_handoff = run_main_phase_with_death_handling(
                                    active_ba_handoff,
                                    reviewers=reviewer_workers,
                                    run_phase=lambda handoff: initialize_task_split_workers(
                                        handoff,
                                        project_dir=project_dir,
                                        requirement_name=requirement_name,
                                        paths=paths,
                                        initialize_ba=True,
                                        reviewers=reviewer_workers,
                                        reviewer_specs_by_name=reviewer_specs_by_name,
                                        initialize_reviewers=False,
                                        progress=progress,
                                    )[0],
                                    replace_dead_main_owner=lambda owner: _replace_dead_task_split_ba(
                                        owner,
                                        project_dir=project_dir,
                                        progress=progress,
                                    ),
                                    main_label="任务拆分需求分析师",
                                    reviewer_label_getter=reviewer_label_getter,
                                    notify=message,
                                )
                        _, reviewer_workers, active_ba_handoff = run_main_phase_with_death_handling(
                            active_ba_handoff,
                            reviewers=reviewer_workers,
                            run_phase=lambda handoff: generate_task_split_json(
                                handoff,
                                project_dir=project_dir,
                                paths=paths,
                                progress=progress,
                                audit_context=audit_context,
                            ),
                            replace_dead_main_owner=lambda owner: _replace_dead_task_split_ba(
                                owner,
                                project_dir=project_dir,
                                progress=progress,
                            ),
                            main_label="任务拆分需求分析师",
                            reviewer_label_getter=reviewer_label_getter,
                            notify=message,
                        )
                except Exception:
                    update_pre_development_task_status(project_dir, requirement_name, task_key="任务拆分", completed=False)
                    raise
                mark_task_split_completed(project_dir, requirement_name)
                append_stage_audit_record(
                    audit_context,
                    event_type="stage_passed",
                    source_paths={
                        "merged_review": paths["merged_review_path"],
                        "ba_feedback": paths["ba_feedback_path"],
                        "task_md": paths["task_md_path"],
                        "task_json": paths["task_json_path"],
                    },
                    review_round_index=round_index,
                )
                cleanup_paths = _shutdown_workers(
                    active_ba_handoff,
                    reviewer_workers,
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    cleanup_runtime=True,
                )
                return TaskSplitStageResult(
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    task_md_path=str(paths["task_md_path"].resolve()),
                    task_json_path=str(paths["task_json_path"].resolve()),
                    merged_review_path=str(paths["merged_review_path"].resolve()),
                    passed=True,
                    cleanup_paths=cleanup_paths,
                )
            review_msg = get_markdown_content(paths["merged_review_path"]).strip()
            if not review_msg:
                raise RuntimeError("任务拆分评审未通过，但合并后的任务单评审记录为空")
            if review_round_policy.should_escalate_before_next_review():
                if review_round_policy.max_rounds is None:
                    raise RuntimeError("review_round_policy 配置错误：无限轮次不应触发超限 HITL")
                hitl_result, reviewer_workers, active_ba_handoff = run_main_phase_with_death_handling(
                    active_ba_handoff,
                    reviewers=reviewer_workers,
                    run_phase=lambda handoff: run_task_split_review_limit_hitl_loop(
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
                                stage_label="任务拆分评审超限",
                                hitl_round=hitl_round,
                            )
                        ) if bool(getattr(args, "yes", False)) else None,
                        audit_context=audit_context,
                    ),
                    owner_getter=lambda result: result.owner,
                    replace_dead_main_owner=lambda owner: _replace_dead_task_split_ba(
                        owner,
                        project_dir=project_dir,
                        progress=progress,
                    ),
                    replace_dead_reviewer=lambda reviewer, _index: _replace_dead_task_split_reviewer(
                        reviewer,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        reviewer_specs_by_name=reviewer_specs_by_name,
                        progress=progress,
                    ),
                    main_label="任务拆分需求分析师",
                    reviewer_label_getter=reviewer_label_getter,
                    notify=message,
                )
                post_hitl_continue_completed = bool(getattr(hitl_result, "post_hitl_continue_completed", False))
                review_round_policy.reset_after_hitl()
            round_index += 1
    except Exception as error:
        stage_failed_paths = paths if "paths" in locals() else build_task_split_paths(project_dir, requirement_name)
        append_stage_audit_record(
            audit_context,
            event_type="stage_failed",
            source_paths={
                "merged_review": stage_failed_paths["merged_review_path"],
                "ba_feedback": stage_failed_paths["ba_feedback_path"],
                "task_md": stage_failed_paths["task_md_path"],
                "task_json": stage_failed_paths["task_json_path"],
            },
            metadata={"error": str(error)},
        )
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
    redirected, launch = maybe_launch_tui(argv, route="task-split", action="stage.a06.start")
    if redirected:
        return int(launch)
    try:
        result = run_task_split_stage(list(launch))
    except Exception as error:  # noqa: BLE001
        message(error)
        return 1

    message("任务拆分完成")
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
