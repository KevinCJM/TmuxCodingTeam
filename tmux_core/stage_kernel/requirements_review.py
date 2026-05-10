# -*- encoding: utf-8 -*-
"""
@File: A04_RequirementsReview.py
@Modify Time: 2026/4/16
@Author: Kevin-Chen
@Descriptions: 需求评审阶段
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from A01_Routing_LayerPlanning import (
    DEFAULT_MODEL_BY_VENDOR,
    prompt_effort,
    prompt_model,
    prompt_vendor,
)
from tmux_core.prompt_contracts.common import check_reviewer_job
from tmux_core.prompt_contracts.requirements_clarification import REQUIREMENTS_STATUS_OK, hitl_bck
from tmux_core.prompt_contracts.requirements_review import (
    human_feed_bck,
    requirements_review_init,
    requirements_review_reply,
    resume_ba,
    review_feedback,
)
from tmux_core.runtime.contracts import (
    TaskResultContract,
    TurnFileContract,
    TurnFileResult,
    normalize_review_status_payload,
)
from tmux_core.runtime.hitl import HitlPromptContext, build_prefixed_sha256, run_hitl_agent_loop
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
from tmux_core.stage_kernel.reviewer_orchestration import (
    repair_reviewer_round_outputs,
    run_parallel_reviewer_round,
    shutdown_stage_workers,
)
from tmux_core.stage_kernel.requirement_concurrency import requirement_concurrency_lock
from tmux_core.stage_kernel.runtime_scope_cleanup import cleanup_runtime_dirs_by_scope
from tmux_core.stage_kernel.stage_audit import (
    StageAuditRunContext,
    append_stage_audit_record,
    begin_stage_audit_run,
    record_before_cleanup,
)
from tmux_core.stage_kernel.death_orchestration import (
    ensure_active_reviewers,
    run_main_phase_with_death_handling,
    run_reviewer_phase_with_death_handling,
)
from tmux_core.stage_kernel import shared_review
from tmux_core.stage_kernel.agent_intervention import (
    AGENT_INTERVENTION_RECHECK,
    AGENT_INTERVENTION_RECREATE,
    AGENT_INTERVENTION_WORKER_DEAD,
    request_worker_manual_intervention,
)
from tmux_core.stage_kernel.shared_review import (
    DEFAULT_REVIEWER_COUNT,
    MAX_REVIEWER_REPAIR_ATTEMPTS,
    ReviewLimitHitlConfig,
    ReviewRoundPolicy,
    ReviewAgentSelection,
    ReviewerRuntime,
    collect_auto_review_limit_hitl_response,
    ensure_empty_file,
    ensure_review_artifacts,
    is_recoverable_startup_failure,
    mark_worker_awaiting_reconfiguration,
    parse_review_max_rounds,
    prompt_required_replacement_review_agent_selection as shared_prompt_required_replacement_review_agent_selection,
    render_review_limit_force_hitl_prompt,
    render_review_limit_human_reply_prompt,
    render_review_agent_selection,
    render_tmux_start_summary,
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
from T01_tools import (
    check_task_exists,
    get_markdown_content,
    is_file_empty,
    task_done,
)
from T09_terminal_ops import (
    BridgeTerminalUI,
    PromptBackRequested,
    SingleLineSpinnerMonitor,
    TERMINAL_SPINNER_FRAMES,
    collect_multiline_input,
    get_terminal_ui,
    message,
    maybe_launch_tui,
    prompt_positive_int as terminal_prompt_positive_int,
    prompt_with_default,
    prompt_yes_no as terminal_prompt_yes_no,
)
from T08_pre_development import (
    build_pre_development_task_record_path,
    ensure_pre_development_task_record,
    update_pre_development_task_status,
)
from T12_requirements_common import (
    DEFAULT_REQUIREMENTS_CLARIFICATION_EFFORT,
    DEFAULT_REQUIREMENTS_CLARIFICATION_MODEL,
    DEFAULT_REQUIREMENTS_CLARIFICATION_VENDOR,
    RequirementsAnalystHandoff,
    build_requirements_clarification_paths,
    ensure_requirements_hitl_record_file,
    prompt_project_dir,
    prompt_requirement_name_selection,
    sanitize_requirement_name,
    stdin_is_interactive,
)


REQUIREMENTS_REVIEW_TASK_NAME = "需求评审"
REQUIREMENTS_REVIEW_RUNTIME_ROOT_NAME = ".requirements_review_runtime"
MAX_REVIEW_ROUNDS = 5
REVIEW_CLARIFICATION_STAGE_NAME = "requirements_clarification"
REVIEW_CLARIFICATION_TURN_PHASE = "requirements_clarification"

@dataclass(frozen=True)
class RequirementsReviewStageResult:
    project_dir: str
    requirement_name: str
    merged_review_path: str
    rounds_used: int
    passed: bool
    cleanup_paths: tuple[str, ...] = ()
    ba_handoff: RequirementsAnalystHandoff | None = None


class _SkipToDetailedDesign(RuntimeError):
    def __init__(self, handoff: RequirementsAnalystHandoff | None) -> None:
        super().__init__("skip_to_detailed_design")
        self.handoff = handoff


def _sync_shared_review_bindings() -> None:
    shared_review.DEFAULT_MODEL_BY_VENDOR = DEFAULT_MODEL_BY_VENDOR
    shared_review.prompt_effort = prompt_effort
    shared_review.prompt_model = prompt_model
    shared_review.prompt_proxy_url = prompt_proxy_url
    shared_review.prompt_vendor = prompt_vendor
    shared_review.SingleLineSpinnerMonitor = SingleLineSpinnerMonitor
    shared_review.TERMINAL_SPINNER_FRAMES = TERMINAL_SPINNER_FRAMES
    shared_review.message = message
    shared_review.prompt_with_default = prompt_with_default
    shared_review.terminal_prompt_positive_int = terminal_prompt_positive_int
    shared_review.terminal_prompt_yes_no = terminal_prompt_yes_no


class ReviewStageProgress(shared_review.ReviewStageProgress):
    def __init__(self, *, initial_phase: str = "需求评审准备中") -> None:
        _sync_shared_review_bindings()
        super().__init__(initial_phase=initial_phase)


def _resolve_review_progress(progress: ReviewStageProgress | None = None) -> ReviewStageProgress | None:
    return shared_review.resolve_review_progress(progress)


def prompt_positive_int(
        prompt_text: str,
        default: int = 1,
        *,
        progress: ReviewStageProgress | None = None,
        allow_back: bool = False,
        stage_key: str = "",
        stage_step_index: int = 0,
) -> int:
    _sync_shared_review_bindings()
    return shared_review.prompt_positive_int(
        prompt_text,
        default,
        progress=progress,
        allow_back=allow_back,
        stage_key=stage_key,
        stage_step_index=stage_step_index,
    )


def prompt_proxy_url(default: str = "", *, role_label: str = "") -> str:
    role_text = str(role_label or "").strip()
    prompt_text = "输入代理端口或完整代理 URL（可留空）"
    if role_text:
        prompt_text = f"为 {role_text} {prompt_text}"
    return prompt_with_default(prompt_text, default, allow_empty=True)


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
    _sync_shared_review_bindings()
    return shared_review.prompt_review_agent_selection(
        default_vendor=default_vendor,
        default_model=default_model,
        default_reasoning_effort=default_reasoning_effort,
        default_proxy_url=default_proxy_url,
        role_label=role_label,
        progress=progress,
        allow_back_first_step=allow_back_first_step,
        stage_key=stage_key,
    )


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
    _sync_shared_review_bindings()
    return shared_review.prompt_yes_no_choice(
        prompt_text,
        default,
        progress=progress,
        preview_path=preview_path,
        preview_title=preview_title,
        allow_back=allow_back,
        stage_key=stage_key,
        stage_step_index=stage_step_index,
    )


def prompt_review_max_rounds(
        *,
        default: int = MAX_REVIEW_ROUNDS,
        progress: ReviewStageProgress | None = None,
        allow_back: bool = False,
        stage_key: str = "",
        stage_step_index: int = 0,
) -> int | None:
    _sync_shared_review_bindings()
    return shared_review.prompt_review_max_rounds(
        default=default,
        progress=progress,
        allow_back=allow_back,
        stage_key=stage_key,
        stage_step_index=stage_step_index,
    )


def prompt_replacement_review_agent_selection(
        *,
        reason_text: str,
        previous_selection: ReviewAgentSelection,
        force_model_change: bool,
        role_label: str,
        progress: ReviewStageProgress | None = None,
) -> ReviewAgentSelection | None:
    progress = _resolve_review_progress(progress)
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
        if (
                not force_model_change
                or selection.vendor != previous_selection.vendor
                or selection.model != previous_selection.model
        ):
            message(render_review_agent_selection(f"重新创建{role_label}", selection))
            return selection
        message("需要更换模型，请选择与当前不同的厂商或模型。")


def prompt_required_replacement_review_agent_selection(
        *,
        reason_text: str,
        previous_selection: ReviewAgentSelection,
        force_model_change: bool,
        role_label: str,
        progress: ReviewStageProgress | None = None,
) -> ReviewAgentSelection:
    progress = _resolve_review_progress(progress)
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
        if (
                not force_model_change
                or selection.vendor != previous_selection.vendor
                or selection.model != previous_selection.model
        ):
            message(render_review_agent_selection(f"重新创建{role_label}", selection))
            return selection
        message("需要更换模型，请选择与当前不同的厂商或模型。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="需求评审阶段")
    parser.add_argument("--project-dir", help="项目目录")
    parser.add_argument("--requirement-name", help="需求名称")
    parser.add_argument("--allow-previous-stage-back", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--review-max-rounds", default="", help="需求评审最多重试几轮；传 infinite 表示不设上限")
    parser.add_argument("--reviewer-agent", action="append", default=[], help="审核智能体模型配置: name=R1,vendor=...,model=...,effort=...,proxy=...")
    parser.add_argument("--yes", action="store_true", help="跳过非关键确认")
    parser.add_argument("--no-tui", action="store_true", help="显式禁用 OpenTUI")
    parser.add_argument("--legacy-cli", action="store_true", help="使用旧版 Python CLI，不跳转 OpenTUI")
    return parser


def _consume_stage_back(allow_previous_stage_back: bool, will_prompt: bool) -> tuple[bool, bool]:
    allow_back = bool(allow_previous_stage_back and will_prompt)
    if will_prompt:
        return allow_back, False
    return False, bool(allow_previous_stage_back)


def build_requirements_review_paths(
        project_dir: str | Path,
        requirement_name: str,
        *,
        ensure_hitl_record: bool = True,
) -> dict[str, Path]:
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
        "hitl_record_path": (
            ensure_requirements_hitl_record_file(project_root, requirement_name)
            if ensure_hitl_record
            else hitl_record_path
        ),
        "pre_development_path": build_pre_development_task_record_path(project_root, requirement_name),
        "merged_review_path": project_root / f"{safe_name}_需求评审记录.md",
        "ba_feedback_path": project_root / f"{safe_name}_需求分析师反馈.md",
    }


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
            default=MAX_REVIEW_ROUNDS,
        )
    if not stdin_is_interactive():
        return MAX_REVIEW_ROUNDS
    if progress is not None and hasattr(progress, "set_phase"):
        progress.set_phase("需求评审 / 配置最大审核轮次")
    return prompt_review_max_rounds(
        default=MAX_REVIEW_ROUNDS,
        progress=progress,
        allow_back=allow_back,
        stage_key="requirements_review",
        stage_step_index=0,
    )


def build_requirements_review_limit_hitl_config(paths: dict[str, Path]) -> ReviewLimitHitlConfig:
    return ReviewLimitHitlConfig(
        stage_label="需求评审超限",
        artifact_label="需求评审",
        primary_output_path=paths["requirements_clear_path"],
        ask_human_path=paths["ask_human_path"],
        hitl_record_path=paths["hitl_record_path"],
        merged_review_path=paths["merged_review_path"],
        output_summary_path=paths["ba_feedback_path"],
        continue_output_label="需求澄清.md",
    )


def build_requirements_review_limit_force_hitl_prompt(
    *,
    paths: dict[str, Path],
    review_msg: str,
    review_limit: int,
    review_rounds_used: int,
) -> str:
    return render_review_limit_force_hitl_prompt(
        config=build_requirements_review_limit_hitl_config(paths),
        review_limit=review_limit,
        review_rounds_used=review_rounds_used,
        hitl_record_md=paths["hitl_record_path"],
        extra_inputs=(paths["original_requirement_path"],),
    ) + f"\n## 当前评审记录\n[REVIEW MSG START]\n{review_msg}\n[REVIEW MSG END]\n"


def build_requirements_review_limit_human_reply_prompt(
    *,
    paths: dict[str, Path],
    review_msg: str,
    human_msg: str,
) -> str:
    return render_review_limit_human_reply_prompt(
        config=build_requirements_review_limit_hitl_config(paths),
        human_msg=human_msg,
        hitl_record_md=paths["hitl_record_path"],
        extra_inputs=(paths["original_requirement_path"],),
    ) + f"\n## 当前评审记录\n[REVIEW MSG START]\n{review_msg}\n[REVIEW MSG END]\n"


def build_reviewer_artifact_paths(project_dir: str | Path, requirement_name: str, reviewer_name: str) -> tuple[Path, Path]:
    project_root = Path(project_dir).expanduser().resolve()
    safe_name = sanitize_requirement_name(requirement_name)
    artifact_agent_name = sanitize_requirement_name(reviewer_name)
    review_md_path = project_root / f"{safe_name}_需求评审记录_{artifact_agent_name}.md"
    review_json_path = project_root / f"{safe_name}_评审记录_{artifact_agent_name}.json"
    return review_md_path, review_json_path


def _predict_review_worker_display_name(
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


def _review_ba_display_name(
        *,
        project_dir: str | Path,
        handoff: RequirementsAnalystHandoff | None = None,
) -> str:
    session_name = str(getattr(getattr(handoff, "worker", None), "session_name", "") or "").strip()
    if session_name:
        return session_name
    return _predict_review_worker_display_name(
        project_dir=project_dir,
        worker_id="requirements-review-analyst",
    )


def _reviewer_artifact_agent_name(reviewer: ReviewerRuntime) -> str:
    worker = getattr(reviewer, "worker", None)
    session_name = str(getattr(worker, "session_name", "") or "").strip()
    reviewer_name = str(getattr(reviewer, "reviewer_name", "") or "").strip()
    return session_name or reviewer_name


def _predict_reviewer_display_name(
        *,
        project_dir: str | Path,
        reviewer_name: str,
        occupied_session_names: Sequence[str] = (),
) -> str:
    return _predict_review_worker_display_name(
        project_dir=project_dir,
        worker_id=f"requirements-review-{reviewer_name.lower()}",
        occupied_session_names=occupied_session_names,
    )


def create_reviewer_runtime(
        *,
        project_dir: str | Path,
        requirement_name: str,
        reviewer_name: str,
        selection: ReviewAgentSelection,
) -> ReviewerRuntime:
    runtime_root = Path(project_dir).expanduser().resolve() / REQUIREMENTS_REVIEW_RUNTIME_ROOT_NAME
    selection, config = resolve_agent_run_config_with_recovery(
        selection,
        role_label=_predict_reviewer_display_name(project_dir=project_dir, reviewer_name=reviewer_name),
    )
    worker = TmuxBatchWorker(
        worker_id=f"requirements-review-{reviewer_name.lower()}",
        work_dir=Path(project_dir).expanduser().resolve(),
        config=config,
        runtime_root=runtime_root,
        runtime_metadata={
            "project_dir": str(Path(project_dir).expanduser().resolve()),
            "requirement_name": str(requirement_name).strip(),
            "workflow_action": "stage.a04.start",
        },
    )
    review_md_path, review_json_path = build_reviewer_artifact_paths(
        project_dir,
        requirement_name,
        str(worker.session_name).strip() or reviewer_name,
    )
    ensure_review_artifacts(review_md_path, review_json_path)
    message(render_tmux_start_summary(str(worker.session_name).strip() or f"审核器 {reviewer_name}", worker))
    return ReviewerRuntime(
        reviewer_name=reviewer_name,
        selection=selection,
        worker=worker,
        review_md_path=review_md_path,
        review_json_path=review_json_path,
        contract=build_reviewer_completion_contract(
            requirement_name=requirement_name,
            reviewer_name=reviewer_name,
            review_md_path=review_md_path,
            review_json_path=review_json_path,
        ),
    )


def cleanup_existing_review_artifacts(
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
            f"{safe_name}_需求评审记录_*.md",
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
            },
            metadata={"trigger": "cleanup_existing_review_artifacts"},
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
    ):
        if candidate.exists() and candidate.is_file():
            candidate.unlink()
            removed.append(str(candidate.resolve()))
    return tuple(removed)


def cleanup_stale_review_runtime_state(
        project_dir: str | Path,
        requirement_name: str,
        *,
        preserve_workers: Sequence[TmuxBatchWorker] = (),
) -> tuple[str, ...]:
    runtime_root = Path(project_dir).expanduser().resolve() / REQUIREMENTS_REVIEW_RUNTIME_ROOT_NAME
    if not runtime_root.exists() or not runtime_root.is_dir():
        return ()

    preserve_runtime_dirs = {
        Path(worker.runtime_dir).expanduser().resolve()
        for worker in preserve_workers
        if str(getattr(worker, "runtime_dir", "")).strip()
    }
    preserve_session_names = {
        str(getattr(worker, "session_name", "")).strip()
        for worker in preserve_workers
        if str(getattr(worker, "session_name", "")).strip()
    }
    return cleanup_runtime_dirs_by_scope(
        runtime_root=runtime_root,
        project_dir=project_dir,
        requirement_name=requirement_name,
        workflow_action="stage.a04.start",
        preserve_runtime_dirs=tuple(preserve_runtime_dirs),
        preserve_session_names=tuple(preserve_session_names),
    )


def build_reviewer_completion_contract(
        *,
        requirement_name: str,
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
            task_name=REQUIREMENTS_REVIEW_TASK_NAME,
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
            payload={"task_name": REQUIREMENTS_REVIEW_TASK_NAME, "review_pass": review_pass},
            artifact_paths=artifact_paths,
            artifact_hashes=artifact_hashes,
            validated_at=str(status_path.stat().st_mtime),
        )

    return TurnFileContract(
        turn_id=f"requirements_review_{reviewer_name}",
        phase=REQUIREMENTS_REVIEW_TASK_NAME,
        status_path=review_json_path,
        validator=validator,
        kind="review_round",
        tracked_artifacts={
            "review_md": review_md_path,
            "review_json": review_json_path,
        },
    )


def build_ba_resume_result_contract(paths: dict[str, Path]) -> TaskResultContract:
    return TaskResultContract(
        turn_id="requirements_review_ba_resume",
        phase="requirements_review_ba_resume",
        task_kind="a03_ba_resume",
        mode="a03_ba_resume",
        expected_statuses=("ready",),
        stage_name=REQUIREMENTS_REVIEW_TASK_NAME,
        optional_artifacts={
            "original_requirement": paths["original_requirement_path"],
            "requirements_clear": paths["requirements_clear_path"],
            "hitl_record": paths["hitl_record_path"],
        },
    )


def build_ba_human_feedback_result_contract(paths: dict[str, Path]) -> TaskResultContract:
    return TaskResultContract(
        turn_id="requirements_review_human_feedback",
        phase="requirements_review_human_feedback",
        task_kind="a03_human_feedback",
        mode="a03_human_feedback",
        expected_statuses=("completed",),
        stage_name=REQUIREMENTS_REVIEW_TASK_NAME,
        required_artifacts={
            "ask_human": paths["ask_human_path"],
        },
        optional_artifacts={
            "requirements_clear": paths["requirements_clear_path"],
            "hitl_record": paths["hitl_record_path"],
        },
    )


def build_ba_review_feedback_result_contract(paths: dict[str, Path]) -> TaskResultContract:
    return build_requirements_feedback_result_contract(paths)


def build_requirements_feedback_result_contract(
    paths: dict[str, Path],
    *,
    mode: str = "a03_ba_feedback",
) -> TaskResultContract:
    expected_statuses = ("hitl", "completed")
    outcome_artifacts = {
        "hitl": {
            "requires": ("ask_human",),
            "optional": ("hitl_record",),
            "forbids": ("ba_feedback",),
        },
        "completed": {
            "requires": ("ba_feedback", "requirements_clear"),
            "optional": ("hitl_record",),
        },
    }
    if mode == "a03_ba_review_limit_force_hitl":
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
        stage_name=REQUIREMENTS_REVIEW_TASK_NAME,
        optional_artifacts={
            "ask_human": paths["ask_human_path"],
            "ba_feedback": paths["ba_feedback_path"],
            "requirements_clear": paths["requirements_clear_path"],
            "hitl_record": paths["hitl_record_path"],
        },
        outcome_artifacts=outcome_artifacts,
    )


def _build_ba_turn_goal(contract: TaskResultContract) -> TaskTurnGoal | None:
    if contract.mode == "a03_ba_resume":
        return TaskTurnGoal(
            goal_id=contract.mode,
            outcomes={"ready": OutcomeGoal(status="ready")},
        )
    if contract.mode == "a03_human_feedback":
        return TaskTurnGoal(
            goal_id=contract.mode,
            outcomes={"completed": OutcomeGoal(status="completed", required_aliases=("ask_human",))},
        )
    if contract.mode == "a03_ba_review_limit_force_hitl":
        return TaskTurnGoal(
            goal_id=contract.mode,
            outcomes={"hitl": OutcomeGoal(status="hitl", required_aliases=("ask_human",))},
        )
    if contract.mode in {"a03_ba_feedback", "a03_ba_review_limit_human_reply"}:
        return TaskTurnGoal(
            goal_id=contract.mode,
            outcomes={
                "hitl": OutcomeGoal(status="hitl", required_aliases=("ask_human",), forbidden_aliases=("ba_feedback",)),
                "completed": OutcomeGoal(
                    status="completed",
                    required_aliases=("ba_feedback", "requirements_clear"),
                ),
            },
        )
    return None


def _build_reviewer_turn_goal() -> CompletionTurnGoal:
    return CompletionTurnGoal(
        goal_id="requirements_review_reviewer_round",
        outcomes={
            "review_pass": OutcomeGoal(status="review_pass", required_aliases=("review_json",), forbidden_aliases=("review_md",)),
            "review_fail": OutcomeGoal(status="review_fail", required_aliases=("review_json", "review_md")),
        },
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
        stage_label=REQUIREMENTS_REVIEW_TASK_NAME,
        role_label=str(getattr(handoff.worker, "session_name", "") or "需求分析师"),
    )


def _run_reviewer_turn(
        reviewer: ReviewerRuntime,
        *,
        label: str,
        prompt: str,
) -> None:
    ensure_review_artifacts(reviewer.review_md_path, reviewer.review_json_path)
    run_completion_turn_with_repair(
        worker=reviewer.worker,
        label=label,
        prompt=prompt,
        completion_contract=reviewer.contract,
        turn_goal=_build_reviewer_turn_goal(),
        timeout_sec=DEFAULT_COMMAND_TIMEOUT_SEC,
        stage_label=REQUIREMENTS_REVIEW_TASK_NAME,
        role_label=_reviewer_artifact_agent_name(reviewer),
    )


def run_ba_turn_with_recreation(
        handoff: RequirementsAnalystHandoff,
        *,
        project_dir: str | Path,
        label: str,
        prompt: str,
        result_contract: TaskResultContract,
        progress: ReviewStageProgress | None = None,
) -> tuple[RequirementsAnalystHandoff, dict[str, object]]:
    progress = _resolve_review_progress(progress)
    current_handoff = handoff
    while True:
        try:
            payload = _run_ba_turn(
                current_handoff,
                label=label,
                prompt=prompt,
                result_contract=result_contract,
            )
            return current_handoff, payload
        except Exception as error:  # noqa: BLE001
            auth_error = is_provider_auth_error(error) or worker_has_provider_auth_error(current_handoff.worker)
            provider_runtime_error = worker_has_provider_runtime_error(current_handoff.worker)
            ready_timeout_error = is_agent_ready_timeout_error(error)
            ba_display_name = _review_ba_display_name(project_dir=project_dir, handoff=current_handoff)
            if auth_error or provider_runtime_error:
                reason_text = (
                    f"检测到{ba_display_name}仍在 agent 界面，但模型认证已失效。\n需要更换模型后继续当前阶段。"
                    if auth_error
                    else f"检测到{ba_display_name}的模型服务出现临时运行错误。\n需要更换或重启模型后继续当前阶段。"
                )
                mark_worker_awaiting_reconfiguration(current_handoff.worker, reason_text=reason_text)
                selection = prompt_required_replacement_review_agent_selection(
                    reason_text=reason_text,
                    previous_selection=ReviewAgentSelection(
                        current_handoff.vendor,
                        current_handoff.model,
                        current_handoff.reasoning_effort,
                        current_handoff.proxy_url,
                    ),
                    force_model_change=True,
                    role_label=ba_display_name,
                    progress=progress,
                )
                selection, config = resolve_agent_run_config_with_recovery(
                    selection,
                    role_label=ba_display_name,
                    progress=progress,
                )
                current_handoff = RequirementsAnalystHandoff(
                    worker=TmuxBatchWorker(
                        worker_id="requirements-review-analyst",
                        work_dir=Path(project_dir).expanduser().resolve(),
                        config=config,
                        runtime_root=Path(project_dir).expanduser().resolve() / REQUIREMENTS_REVIEW_RUNTIME_ROOT_NAME,
                    ),
                    vendor=selection.vendor,
                    model=selection.model,
                    reasoning_effort=selection.reasoning_effort,
                    proxy_url=selection.proxy_url,
                )
                message(render_tmux_start_summary(str(current_handoff.worker.session_name).strip() or ba_display_name, current_handoff.worker))
                continue
            if ready_timeout_error:
                reason_text = f"{ba_display_name}启动超时，未能进入可输入状态。\n需要更换模型后继续当前阶段。"
                mark_worker_awaiting_reconfiguration(current_handoff.worker, reason_text=reason_text)
                selection = prompt_required_replacement_review_agent_selection(
                    reason_text=reason_text,
                    previous_selection=ReviewAgentSelection(
                        current_handoff.vendor,
                        current_handoff.model,
                        current_handoff.reasoning_effort,
                        current_handoff.proxy_url,
                    ),
                    force_model_change=True,
                    role_label=ba_display_name,
                    progress=progress,
                )
                selection, config = resolve_agent_run_config_with_recovery(
                    selection,
                    role_label=ba_display_name,
                    progress=progress,
                )
                current_handoff = RequirementsAnalystHandoff(
                    worker=TmuxBatchWorker(
                        worker_id="requirements-review-analyst",
                        work_dir=Path(project_dir).expanduser().resolve(),
                        config=config,
                        runtime_root=Path(project_dir).expanduser().resolve() / REQUIREMENTS_REVIEW_RUNTIME_ROOT_NAME,
                    ),
                    vendor=selection.vendor,
                    model=selection.model,
                    reasoning_effort=selection.reasoning_effort,
                    proxy_url=selection.proxy_url,
                )
                message(render_tmux_start_summary(str(current_handoff.worker.session_name).strip() or ba_display_name, current_handoff.worker))
                continue
            if is_worker_death_error(error):
                replacement = recreate_ba_handoff(
                    project_dir=project_dir,
                    previous_handoff=current_handoff,
                    progress=progress,
                )
                if replacement is None:
                    raise RuntimeError(f"{ba_display_name}已死亡，且用户未创建新的{ba_display_name}") from error
                current_handoff = replacement
                continue
            raise


def run_reviewer_turn_with_recreation(
        reviewer: ReviewerRuntime,
        *,
        project_dir: str | Path,
        requirement_name: str,
        label: str,
        prompt: str,
        progress: ReviewStageProgress | None = None,
) -> ReviewerRuntime | None:
    progress = _resolve_review_progress(progress)
    current_reviewer = reviewer
    while True:
        try:
            _run_reviewer_turn(
                current_reviewer,
                label=label,
                prompt=prompt,
            )
            return current_reviewer
        except Exception as error:  # noqa: BLE001
            auth_error = is_provider_auth_error(error) or worker_has_provider_auth_error(current_reviewer.worker)
            provider_runtime_error = worker_has_provider_runtime_error(current_reviewer.worker)
            ready_timeout_error = is_agent_ready_timeout_error(error)
            reviewer_display_name = _reviewer_artifact_agent_name(current_reviewer)
            if is_turn_artifact_contract_error(error):
                return current_reviewer
            if auth_error or provider_runtime_error:
                reason_text = (
                    f"检测到{reviewer_display_name}仍在 agent 界面，但模型认证已失效。\n需要更换模型后继续当前阶段。"
                    if auth_error
                    else f"检测到{reviewer_display_name}的模型服务出现临时运行错误。\n需要更换或重启模型后继续当前阶段。"
                )
                decision = request_worker_manual_intervention(
                    stage_label="需求评审",
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
                selection = prompt_required_replacement_review_agent_selection(
                    reason_text=reason_text,
                    previous_selection=current_reviewer.selection,
                    force_model_change=True,
                    role_label=reviewer_display_name,
                    progress=progress,
                )
                current_reviewer = create_reviewer_runtime(
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    reviewer_name=current_reviewer.reviewer_name,
                    selection=selection,
                )
                continue
            if ready_timeout_error:
                reason_text = f"{reviewer_display_name}启动超时，未能进入可输入状态。\n需要更换模型后继续当前阶段。"
                decision = request_worker_manual_intervention(
                    stage_label="需求评审",
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
                selection = prompt_required_replacement_review_agent_selection(
                    reason_text=reason_text,
                    previous_selection=current_reviewer.selection,
                    force_model_change=True,
                    role_label=reviewer_display_name,
                    progress=progress,
                )
                current_reviewer = create_reviewer_runtime(
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    reviewer_name=current_reviewer.reviewer_name,
                    selection=selection,
                )
                continue
            if is_worker_death_error(error):
                decision = request_worker_manual_intervention(
                    stage_label="需求评审",
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
                    progress=progress,
                )
                if replacement is None:
                    message(f"{reviewer_display_name} 重新创建失败，请重新选择恢复方式。")
                    continue
                current_reviewer = replacement
                continue
            raise


def ensure_review_stage_inputs(paths: dict[str, Path], requirement_name: str) -> None:
    if not get_markdown_content(paths["original_requirement_path"]).strip():
        raise RuntimeError(f"缺少原始需求文档: {paths['original_requirement_path']}")
    if not get_markdown_content(paths["requirements_clear_path"]).strip():
        raise RuntimeError(f"缺少需求澄清文档: {paths['requirements_clear_path']}")
    ensure_pre_development_task_record(paths["project_root"], requirement_name)


def prepare_ba_handoff(
        *,
        project_dir: str | Path,
        requirement_name: str,
        ba_handoff: RequirementsAnalystHandoff | None,
        paths: dict[str, Path],
        progress: ReviewStageProgress | None = None,
) -> tuple[RequirementsAnalystHandoff, tuple[str, ...]]:
    progress = _resolve_review_progress(progress)
    if ba_handoff is not None:
        return ba_handoff, ()

    if progress is not None:
        progress.set_phase("需求评审准备中")
    message("当前没有可复用的需求分析师，将新建需求分析师处理评审反馈")
    handoff = _create_review_ba_handoff(
        project_dir=project_dir,
        selection_title="进入需求评审阶段（需求分析师）",
        progress=progress,
    )
    handoff, payload = run_ba_turn_with_recreation(
        handoff,
        project_dir=project_dir,
        label="resume_requirements_review_ba",
        prompt=resume_ba(
            original_requirement_md=str(paths["original_requirement_path"].resolve()),
            requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
            hitl_record_md=str(paths["hitl_record_path"].resolve()),
        ),
        result_contract=build_ba_resume_result_contract(paths),
    )
    if str(payload.get("status", "")).strip() != "ready":
        raise RuntimeError("需求分析师未按要求进入需求评审准备态")
    return (
        handoff,
        (str(handoff.worker.runtime_dir), str((Path(project_dir).expanduser().resolve() / REQUIREMENTS_REVIEW_RUNTIME_ROOT_NAME).resolve())),
    )

def _create_review_ba_handoff(
        *,
        project_dir: str | Path,
        selection_title: str,
        progress: ReviewStageProgress | None = None,
) -> RequirementsAnalystHandoff:
    progress = _resolve_review_progress(progress)
    ba_display_name = _review_ba_display_name(project_dir=project_dir)
    selection = prompt_review_agent_selection(
        DEFAULT_REQUIREMENTS_CLARIFICATION_VENDOR,
        role_label=ba_display_name,
        progress=progress,
    )
    message(render_review_agent_selection(selection_title, selection))
    selection, config = resolve_agent_run_config_with_recovery(
        selection,
        role_label=ba_display_name,
        progress=progress,
    )
    worker = TmuxBatchWorker(
        worker_id="requirements-review-analyst",
        work_dir=Path(project_dir).expanduser().resolve(),
        config=config,
        runtime_root=Path(project_dir).expanduser().resolve() / REQUIREMENTS_REVIEW_RUNTIME_ROOT_NAME,
        runtime_metadata={
            "project_dir": str(Path(project_dir).expanduser().resolve()),
            "workflow_action": "stage.a04.start",
        },
    )
    message(render_tmux_start_summary(str(worker.session_name).strip() or ba_display_name, worker))
    return RequirementsAnalystHandoff(
        worker=worker,
        vendor=selection.vendor,
        model=selection.model,
        reasoning_effort=selection.reasoning_effort,
        proxy_url=selection.proxy_url,
    )


def recreate_ba_handoff(
        *,
        project_dir: str | Path,
        previous_handoff: RequirementsAnalystHandoff,
        progress: ReviewStageProgress | None = None,
) -> RequirementsAnalystHandoff | None:
    progress = _resolve_review_progress(progress)
    ba_display_name = _review_ba_display_name(project_dir=project_dir, handoff=previous_handoff)
    selection = prompt_required_replacement_review_agent_selection(
        reason_text=f"检测到{ba_display_name}已死亡，且 resume 失败。\n需要更换模型后继续当前阶段。",
        previous_selection=ReviewAgentSelection(
            previous_handoff.vendor,
            previous_handoff.model,
            previous_handoff.reasoning_effort,
            previous_handoff.proxy_url,
        ),
        force_model_change=True,
        role_label=ba_display_name,
        progress=progress,
    )
    selection, config = resolve_agent_run_config_with_recovery(
        selection,
        role_label=ba_display_name,
        progress=progress,
    )
    worker = TmuxBatchWorker(
        worker_id="requirements-review-analyst",
        work_dir=Path(project_dir).expanduser().resolve(),
        config=config,
        runtime_root=Path(project_dir).expanduser().resolve() / REQUIREMENTS_REVIEW_RUNTIME_ROOT_NAME,
        runtime_metadata={
            "project_dir": str(Path(project_dir).expanduser().resolve()),
            "workflow_action": "stage.a04.start",
        },
    )
    message(render_tmux_start_summary(str(worker.session_name).strip() or ba_display_name, worker))
    return RequirementsAnalystHandoff(
        worker=worker,
        vendor=selection.vendor,
        model=selection.model,
        reasoning_effort=selection.reasoning_effort,
        proxy_url=selection.proxy_url,
    )


def recreate_reviewer_runtime(
        *,
        project_dir: str | Path,
        requirement_name: str,
        reviewer: ReviewerRuntime,
        progress: ReviewStageProgress | None = None,
) -> ReviewerRuntime | None:
    progress = _resolve_review_progress(progress)
    selection = prompt_replacement_review_agent_selection(
        reason_text=f"检测到审核器 {reviewer.reviewer_name} 已死亡，且 resume 失败。\n需要更换模型后继续当前阶段。",
        previous_selection=reviewer.selection,
        force_model_change=True,
        role_label=f"审核器 {reviewer.reviewer_name}",
        progress=progress,
    )
    if selection is None:
        return None
    replacement = create_reviewer_runtime(
        project_dir=project_dir,
        requirement_name=requirement_name,
        reviewer_name=reviewer.reviewer_name,
        selection=selection,
    )
    if replacement.review_md_path != reviewer.review_md_path and reviewer.review_md_path.exists():
        reviewer.review_md_path.unlink()
    if replacement.review_json_path != reviewer.review_json_path and reviewer.review_json_path.exists():
        reviewer.review_json_path.unlink()
    return replacement


def _active_reviewer_files(reviewers: Sequence[ReviewerRuntime]) -> tuple[list[str], list[str]]:
    review_json_files = [str(reviewer.review_json_path.resolve()) for reviewer in reviewers]
    review_md_files = [str(reviewer.review_md_path.resolve()) for reviewer in reviewers]
    return review_json_files, review_md_files


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


def _replace_dead_review_ba(
        handoff: RequirementsAnalystHandoff,
        *,
        project_dir: str | Path,
        progress: ReviewStageProgress | None = None,
) -> RequirementsAnalystHandoff:
    replacement = recreate_ba_handoff(
        project_dir=project_dir,
        previous_handoff=handoff,
        progress=progress,
    )
    if replacement is None:
        ba_display_name = _review_ba_display_name(project_dir=project_dir, handoff=handoff)
        raise RuntimeError(f"{ba_display_name} 已死亡，且用户未创建新的{ba_display_name}")
    return replacement


def _replace_dead_requirements_review_reviewer(
        reviewer: ReviewerRuntime,
        *,
        project_dir: str | Path,
        requirement_name: str,
        progress: ReviewStageProgress | None = None,
) -> ReviewerRuntime | None:
    return recreate_reviewer_runtime(
        project_dir=project_dir,
        requirement_name=requirement_name,
        reviewer=reviewer,
        progress=progress,
    )


def run_human_check_loop(
        *,
        handoff: RequirementsAnalystHandoff | None,
        paths: dict[str, Path],
        requirement_name: str,
        progress: ReviewStageProgress | None = None,
        allow_previous_stage_back: bool = False,
        auto_confirm: bool = False,
) -> RequirementsAnalystHandoff | None:
    progress = _resolve_review_progress(progress)
    message("进入需求评审阶段")
    message(f"请先阅读需求澄清文档: {paths['requirements_clear_path']}")
    current_handoff = handoff
    if auto_confirm:
        message("--yes 已跳过需求评审前人工建议确认，继续执行需求评审。")
        if current_handoff is None:
            message("当前没有可复用的需求分析师，将新建需求分析师处理后续需求评审")
            current_handoff = _create_review_ba_handoff(
                project_dir=paths["project_root"],
                selection_title="进入需求评审阶段（需求分析师）",
                progress=progress,
            )
        return current_handoff
    while True:
        if progress is not None:
            progress.set_phase("等待人工审核")
        if not prompt_yes_no_choice(
                "是否向需求分析师提出建议或问题",
                False,
                progress=progress,
                preview_path=paths["requirements_clear_path"],
                preview_title="需求澄清文档",
                allow_back=allow_previous_stage_back,
                stage_key="requirements_review",
                stage_step_index=0,
        ):
            try:
                skip_review = prompt_yes_no_choice(
                        "是否跳过需求评审阶段",
                        False,
                        progress=progress,
                        preview_path=paths["requirements_clear_path"],
                        preview_title="需求澄清文档",
                        allow_back=True,
                        stage_key="requirements_review",
                        stage_step_index=1,
                )
            except PromptBackRequested:
                continue
            if skip_review:
                raise _SkipToDetailedDesign(current_handoff)
            if current_handoff is None:
                message("当前没有可复用的需求分析师，将新建需求分析师处理后续需求评审")
                current_handoff = _create_review_ba_handoff(
                    project_dir=paths["project_root"],
                    selection_title="进入需求评审阶段（需求分析师）",
                    progress=progress,
                )
            return current_handoff
        with progress.suspended() if progress is not None else nullcontext():
            human_msg = collect_multiline_input(
                title="请输入给需求分析师的问题或建议",
                empty_retry_message="内容不能为空，请重新输入。",
            )
        ensure_empty_file(paths["ask_human_path"])
        reuse_existing_handoff = current_handoff is not None
        if current_handoff is None:
            if progress is not None:
                progress.set_phase("需求评审 / 处理人类建议")
            current_handoff = _create_review_ba_handoff(
                project_dir=paths["project_root"],
                selection_title="按人类建议启动需求分析师",
                progress=progress,
            )
        if reuse_existing_handoff:
            initial_prompt = human_feed_bck(
                human_msg,
                ask_human_md=str(paths["ask_human_path"].resolve()),
                requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
                hitl_record_md=str(paths["hitl_record_path"].resolve()),
            )
        else:
            initial_prompt = resume_ba(
                human_msg=human_msg,
                original_requirement_md=str(paths["original_requirement_path"].resolve()),
                ask_human_md=str(paths["ask_human_path"].resolve()),
                requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
                hitl_record_md=str(paths["hitl_record_path"].resolve()),
            )
        current_handoff = _run_review_clarification_continuation(
            handoff=current_handoff,
            paths=paths,
            initial_prompt=initial_prompt,
            label_prefix="requirements_review_human_audit",
            progress=progress,
        )
        response = get_markdown_content(paths["ask_human_path"]).strip()
        if response:
            message("需求分析师回复:")
            message(response)
        else:
            message(f"需求分析师已更新需求澄清文档: {paths['requirements_clear_path']}")


def _collect_review_hitl_response(
        question_path: str | Path,
        *,
        hitl_round: int,
        answer_path: str | Path | None = None,
        progress: ReviewStageProgress | None = None,
) -> str:
    progress = _resolve_review_progress(progress)
    question_file = Path(question_path).expanduser().resolve()
    question_text = get_markdown_content(question_file).strip()
    if not isinstance(get_terminal_ui(), BridgeTerminalUI):
        message()
        message(f"需求评审阶段 HITL 第 {hitl_round} 轮，需要人工补充信息")
        message(f"问题文档: {question_file}")
        message(question_text or "(问题文档为空)")
    if progress is not None:
        progress.set_phase("需求评审 / 等待 HITL")
    with progress.suspended() if progress is not None else nullcontext():
        return collect_multiline_input(
            title=f"HITL 第 {hitl_round} 轮回复",
            empty_retry_message="回复不能为空，请重新输入。",
            question_path=question_file,
            answer_path=answer_path,
            is_hitl=True,
        )


def _default_review_reply(paths: dict[str, Path]) -> str:
    return f"需求分析师已更新《{paths['requirements_clear_path'].name}》，请基于最新需求澄清重新审核。"


def _run_review_clarification_continuation(
        *,
        handoff: RequirementsAnalystHandoff,
        paths: dict[str, Path],
        initial_prompt: str,
        label_prefix: str,
        progress: ReviewStageProgress | None = None,
) -> RequirementsAnalystHandoff:
    progress = _resolve_review_progress(progress)
    current_handoff = handoff
    stage_status_path = current_handoff.worker.runtime_dir / f"{label_prefix}_clarification_status.json"
    turns_root = current_handoff.worker.runtime_dir / f"{label_prefix}_clarification_turns"
    fresh_completion_paths = [
        paths["requirements_clear_path"],
        paths["hitl_record_path"],
    ]
    ba_feedback_path = paths.get("ba_feedback_path")
    if isinstance(ba_feedback_path, Path):
        fresh_completion_paths.append(ba_feedback_path)

    def initial_prompt_builder(context: HitlPromptContext) -> str:
        return initial_prompt

    def hitl_prompt_builder(human_msg: str, context: HitlPromptContext) -> str:
        return hitl_bck(
            human_msg,
            original_requirement_md=str(paths["original_requirement_path"].resolve()),
            hitl_record_md=str(paths["hitl_record_path"].resolve()),
            requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
            ask_human_md=str(paths["ask_human_path"].resolve()),
        )

    def replace_dead_worker(current_worker: object, error: BaseException) -> object:
        del current_worker
        nonlocal current_handoff
        current_handoff = _replace_dead_review_ba(
            current_handoff,
            project_dir=paths["project_root"],
            progress=progress,
        )
        return current_handoff.worker

    if progress is not None:
        progress.set_phase("需求评审 / 澄清中")
    loop_result = run_hitl_agent_loop(
        worker=current_handoff.worker,
        stage_name=REVIEW_CLARIFICATION_STAGE_NAME,
        output_path=paths["requirements_clear_path"],
        question_path=paths["ask_human_path"],
        record_path=paths["hitl_record_path"],
        stage_status_path=stage_status_path,
        turns_root=turns_root,
        initial_prompt_builder=initial_prompt_builder,
        hitl_prompt_builder=hitl_prompt_builder,
        label_prefix=label_prefix,
        turn_phase=REVIEW_CLARIFICATION_TURN_PHASE,
        human_input_provider=lambda question_path, hitl_round: _collect_review_hitl_response(
            question_path,
            hitl_round=hitl_round,
            answer_path=paths["hitl_record_path"],
            progress=progress,
        ),
        on_worker_starting=lambda live_worker: progress.set_phase("需求评审 / 澄清中") if progress is not None else None,
        on_agent_turn_started=lambda context, live_worker: progress.set_phase(
            f"需求评审 / 澄清中 | HITL 第 {context.hitl_round} 轮"
        ) if progress is not None else None,
        replace_dead_worker=replace_dead_worker,
        timeout_sec=DEFAULT_COMMAND_TIMEOUT_SEC,
        fresh_completion_paths=fresh_completion_paths,
        fresh_completion_start_round=2,
    )
    if str(loop_result.decision.payload.get("status", "")).strip() != REQUIREMENTS_STATUS_OK:
        raise RuntimeError("需求分析师未完成需求澄清闭环")
    if not get_markdown_content(paths["requirements_clear_path"]).strip():
        raise RuntimeError("需求澄清未生成有效《需求澄清.md》")
    return current_handoff


def build_reviewer_workers(
        *,
        args: argparse.Namespace | None = None,
        project_dir: str | Path,
        requirement_name: str,
        progress: ReviewStageProgress | None = None,
        allow_back_first_prompt: bool = False,
) -> list[ReviewerRuntime]:
    progress = _resolve_review_progress(progress)
    if progress is not None:
        progress.set_phase("启动审核器中")
    agent_config = resolve_stage_agent_config(args or argparse.Namespace(reviewer_agent=[]))
    if agent_config.reviewer_order:
        reviewer_names = list(agent_config.reviewer_order)
    else:
        reviewer_count = prompt_positive_int(
            "请输入审核器数量",
            DEFAULT_REVIEWER_COUNT,
            progress=progress,
            allow_back=allow_back_first_prompt,
            stage_key="requirements_review",
            stage_step_index=1,
        )
        reviewer_names = [f"R{index}" for index in range(1, reviewer_count + 1)]
    reviewers: list[ReviewerRuntime] = []
    predicted_session_names: set[str] = set()
    next_allow_back = bool(allow_back_first_prompt and agent_config.reviewer_order)
    for reviewer_name in reviewer_names:
        reviewer_display_name = _predict_reviewer_display_name(
            project_dir=project_dir,
            reviewer_name=reviewer_name,
            occupied_session_names=predicted_session_names,
        )
        predicted_session_names.add(reviewer_display_name)
        selection = agent_config.reviewer_selection(reviewer_name)
        if selection is None:
            message(f"配置审核器 {reviewer_display_name}")
            selection = prompt_review_agent_selection(
                DEFAULT_REQUIREMENTS_CLARIFICATION_VENDOR,
                role_label=reviewer_display_name,
                progress=progress,
                allow_back_first_step=next_allow_back,
                stage_key="requirements_review_reviewer_selection",
            )
            next_allow_back = False
            message(render_review_agent_selection(f"审核器 {reviewer_display_name} 配置", selection))
        reviewers.append(
            create_reviewer_runtime(
                project_dir=project_dir,
                requirement_name=requirement_name,
                reviewer_name=reviewer_name,
                selection=selection,
            )
        )
    return reviewers


def _run_parallel_reviewers(
        reviewers: Sequence[ReviewerRuntime],
        *,
        project_dir: str | Path,
        requirement_name: str,
        round_index: int,
        prompt_builder: Callable[[ReviewerRuntime], str],
        label_prefix: str,
        progress: ReviewStageProgress | None = None,
) -> list[ReviewerRuntime]:
    progress = _resolve_review_progress(progress)
    if progress is not None:
        progress.set_phase(f"需求评审第 {round_index} 轮")
    return run_parallel_reviewer_round(
        reviewers,
        key_func=lambda reviewer: reviewer.reviewer_name,
        run_turn=lambda reviewer: run_reviewer_turn_with_recreation(
            reviewer,
            project_dir=project_dir,
            requirement_name=requirement_name,
            label=f"{label_prefix}_{reviewer.reviewer_name}_round_{round_index}",
            prompt=prompt_builder(reviewer),
            progress=progress,
        ),
        error_prefix="审核器执行失败:",
    )


def repair_reviewer_outputs(
        reviewers: Sequence[ReviewerRuntime],
        *,
        project_dir: str | Path,
        requirement_name: str,
        round_index: int,
        progress: ReviewStageProgress | None = None,
) -> list[ReviewerRuntime]:
    progress = _resolve_review_progress(progress)
    if progress is not None:
        progress.set_phase(f"需求评审第 {round_index} 轮")
    json_pattern = f"{sanitize_requirement_name(requirement_name)}_评审记录_*.json"
    md_pattern = f"{sanitize_requirement_name(requirement_name)}_需求评审记录_*.md"
    return repair_reviewer_round_outputs(
        reviewers,
        key_func=lambda reviewer: reviewer.reviewer_name,
        artifact_name_func=_reviewer_artifact_agent_name,
        check_job=lambda reviewer_names: check_reviewer_job(
            reviewer_names,
            directory=project_dir,
            task_name=REQUIREMENTS_REVIEW_TASK_NAME,
            json_pattern=json_pattern,
            md_pattern=md_pattern,
        ),
        run_fix_turn=lambda reviewer, fix_prompt, repair_attempt: run_reviewer_turn_with_recreation(
            reviewer,
            project_dir=project_dir,
            requirement_name=requirement_name,
            label=f"requirements_review_fix_{reviewer.reviewer_name}_round_{round_index}_attempt_{repair_attempt}",
            prompt=fix_prompt,
            progress=progress,
        ),
        max_attempts=MAX_REVIEWER_REPAIR_ATTEMPTS,
        error_prefix="审核器修复输出失败:",
        final_error="审核器多次修复后仍未按协议更新文档",
        stage_label=REQUIREMENTS_REVIEW_TASK_NAME,
        progress=progress,
        recreate_reviewer=lambda reviewer: _replace_dead_requirements_review_reviewer(
            reviewer,
            project_dir=project_dir,
            requirement_name=requirement_name,
            progress=progress,
        ),
    )


def _run_review_feedback_loop(
        *,
        handoff: RequirementsAnalystHandoff | None,
        reviewers: Sequence[ReviewerRuntime],
        paths: dict[str, Path],
        requirement_name: str,
        round_index: int,
        progress: ReviewStageProgress | None = None,
        skip_ba_feedback: bool = False,
        audit_context: StageAuditRunContext | None = None,
) -> tuple[RequirementsAnalystHandoff, list[ReviewerRuntime]]:
    progress = _resolve_review_progress(progress)
    reviewer_label_getter = lambda reviewer, index: _reviewer_artifact_agent_name(reviewer) or f"审核智能体 {index}"  # noqa: E731
    review_msg = get_markdown_content(paths["merged_review_path"]).strip()
    if not review_msg:
        raise RuntimeError("评审未通过，但合并后的需求评审记录为空")
    current_handoff = handoff
    if current_handoff is None:
        current_handoff, _ = prepare_ba_handoff(
            project_dir=paths["project_root"],
            requirement_name=requirement_name,
            ba_handoff=None,
            paths=paths,
            progress=progress,
        )
    if not skip_ba_feedback:
        if audit_context is not None:
            record_before_cleanup(
                audit_context,
                {
                    "ask_human": paths["ask_human_path"],
                    "ba_feedback": paths["ba_feedback_path"],
                },
                metadata={"trigger": "requirements_review_feedback_prepare"},
                review_round_index=round_index,
            )
        ensure_empty_file(paths["ask_human_path"])
        ensure_empty_file(paths["ba_feedback_path"])
        _, reviewers, current_handoff = run_main_phase_with_death_handling(
            current_handoff,
            reviewers=reviewers,
            run_phase=lambda active_handoff: _run_review_clarification_continuation(
                handoff=active_handoff,
                paths=paths,
                initial_prompt=review_feedback(
                    review_msg,
                    original_requirement_md=str(paths["original_requirement_path"].resolve()),
                    ask_human_md=str(paths["ask_human_path"].resolve()),
                    hitl_record_md=str(paths["hitl_record_path"].resolve()),
                    requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
                    what_just_change=str(paths["ba_feedback_path"].resolve()),
                ),
                label_prefix=f"requirements_review_feedback_round_{round_index}",
                progress=progress,
            ),
            replace_dead_main_owner=lambda owner: _replace_dead_review_ba(
                owner,
                project_dir=paths["project_root"],
                progress=progress,
            ),
            main_label="需求评审需求分析师",
            reviewer_label_getter=reviewer_label_getter,
            notify=message,
        )
    ba_reply = get_markdown_content(paths["ba_feedback_path"]).strip()
    if not ba_reply:
        raise RuntimeError(f"需求分析师反馈为空: {paths['ba_feedback_path']}")
    if audit_context is not None:
        append_stage_audit_record(
            audit_context,
            event_type="feedback_written",
            source_paths={
                "ba_feedback": paths["ba_feedback_path"],
                "merged_review": paths["merged_review_path"],
            },
            review_round_index=round_index,
            metadata={"trigger": "run_requirements_review_feedback_loop"},
        )

    def prompt_builder(reviewer: ReviewerRuntime) -> str:
        return requirements_review_reply(
            ba_reply,
            REQUIREMENTS_REVIEW_TASK_NAME,
            requirement_review_md=str(reviewer.review_md_path.resolve()),
            requirement_review_json=str(reviewer.review_json_path.resolve()),
            requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
        )

    reviewer_list, current_handoff = run_reviewer_phase_with_death_handling(
        current_handoff,
        reviewers,
        run_phase=lambda active_reviewers: _run_parallel_reviewers(
            active_reviewers,
            project_dir=paths["project_root"],
            requirement_name=requirement_name,
            round_index=round_index,
            prompt_builder=prompt_builder,
            label_prefix="requirements_review_reply",
        ),
        replace_dead_main_owner=lambda owner: _replace_dead_review_ba(
            owner,
            project_dir=paths["project_root"],
            progress=progress,
        ),
        replace_dead_reviewer=lambda reviewer, _index: _replace_dead_requirements_review_reviewer(
            reviewer,
            project_dir=paths["project_root"],
            requirement_name=requirement_name,
            progress=progress,
        ),
        main_label="需求评审需求分析师",
        reviewer_label_getter=reviewer_label_getter,
        notify=message,
    )
    reviewer_list, current_handoff = run_reviewer_phase_with_death_handling(
        current_handoff,
        reviewer_list,
        run_phase=lambda active_reviewers: repair_reviewer_outputs(
            active_reviewers,
            project_dir=paths["project_root"],
            requirement_name=requirement_name,
            round_index=round_index,
        ),
        replace_dead_main_owner=lambda owner: _replace_dead_review_ba(
            owner,
            project_dir=paths["project_root"],
            progress=progress,
        ),
        replace_dead_reviewer=lambda reviewer, _index: _replace_dead_requirements_review_reviewer(
            reviewer,
            project_dir=paths["project_root"],
            requirement_name=requirement_name,
            progress=progress,
        ),
        main_label="需求评审需求分析师",
        reviewer_label_getter=reviewer_label_getter,
        notify=message,
    )
    return current_handoff, reviewer_list


def run_requirements_review_limit_hitl_loop(
        handoff: RequirementsAnalystHandoff | None,
        *,
        reviewers: Sequence[ReviewerRuntime],
        paths: dict[str, Path],
        requirement_name: str,
        review_msg: str,
        review_limit: int,
        review_rounds_used: int,
        progress: ReviewStageProgress | None = None,
        human_input_provider=None,
        audit_context: StageAuditRunContext | None = None,
) -> tuple[RequirementsAnalystHandoff, list[ReviewerRuntime], bool]:
    progress = _resolve_review_progress(progress)
    reviewer_label_getter = lambda reviewer, index: _reviewer_artifact_agent_name(reviewer) or f"审核智能体 {index}"  # noqa: E731
    current_handoff = handoff
    if current_handoff is None:
        current_handoff, _ = prepare_ba_handoff(
            project_dir=paths["project_root"],
            requirement_name=requirement_name,
            ba_handoff=None,
            paths=paths,
            progress=progress,
        )
    if audit_context is not None:
        record_before_cleanup(
            audit_context,
            {
                "ask_human": paths["ask_human_path"],
                "ba_feedback": paths["ba_feedback_path"],
            },
            metadata={"trigger": "requirements_review_limit_hitl_prepare"},
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
            metadata={"trigger": "requirements_review_limit_hitl"},
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
                "trigger": "requirements_review_limit_hitl",
                "human_answer_source": "runtime_payload",
            },
            snapshot_overrides={"human_answer": human_msg},
        )

    def initial_turn() -> object:
        nonlocal current_handoff, reviewers
        _, _, current_handoff = run_main_phase_with_death_handling(
            current_handoff,
            reviewers=(),
            run_phase=lambda active_handoff: run_ba_turn_with_recreation(
                active_handoff,
                project_dir=paths["project_root"],
                label="requirements_review_limit_hitl",
                prompt=build_requirements_review_limit_force_hitl_prompt(
                    paths=paths,
                    review_msg=review_msg,
                    review_limit=review_limit,
                    review_rounds_used=review_rounds_used,
                ),
                result_contract=build_requirements_feedback_result_contract(
                    paths,
                    mode="a03_ba_review_limit_force_hitl",
                ),
                progress=progress,
            )[0],
            replace_dead_main_owner=lambda owner: _replace_dead_review_ba(
                owner,
                project_dir=paths["project_root"],
                progress=progress,
            ),
            main_label="需求评审需求分析师",
            reviewer_label_getter=reviewer_label_getter,
            notify=message,
        )
        return current_handoff

    def human_reply_turn(human_msg: str) -> object:
        nonlocal current_handoff, reviewers
        _, _, current_handoff = run_main_phase_with_death_handling(
            current_handoff,
            reviewers=(),
            run_phase=lambda active_handoff: run_ba_turn_with_recreation(
                active_handoff,
                project_dir=paths["project_root"],
                label="requirements_review_limit_human_reply",
                prompt=build_requirements_review_limit_human_reply_prompt(
                    paths=paths,
                    review_msg=review_msg,
                    human_msg=human_msg,
                ),
                result_contract=build_requirements_feedback_result_contract(
                    paths,
                    mode="a03_ba_review_limit_human_reply",
                ),
                progress=progress,
            )[0],
            replace_dead_main_owner=lambda owner: _replace_dead_review_ba(
                owner,
                project_dir=paths["project_root"],
                progress=progress,
            ),
            main_label="需求评审需求分析师",
            reviewer_label_getter=reviewer_label_getter,
            notify=message,
        )
        return current_handoff

    result = run_review_limit_hitl_cycle(
        stage_label="需求评审超限",
        ask_human_path=paths["ask_human_path"],
        hitl_record_path=paths["hitl_record_path"],
        initial_turn=initial_turn,
        human_reply_turn=human_reply_turn,
        human_input_provider=human_input_provider,
        progress=progress,
        max_hitl_rounds=8,
        on_hitl_question=audit_hitl_question,
        on_hitl_answer=audit_hitl_answer,
    )
    return current_handoff, list(reviewers), result.post_hitl_continue_completed


def _shutdown_workers(
        ba_handoff: RequirementsAnalystHandoff | None,
        reviewers: Sequence[ReviewerRuntime],
        *,
        cleanup_runtime: bool,
        preserve_ba_worker: bool = False,
) -> tuple[str, ...]:
    return shutdown_stage_workers(
        ba_handoff,
        reviewers,
        cleanup_runtime=cleanup_runtime,
        preserve_ba_worker=preserve_ba_worker,
    )


def run_requirements_review_stage(
        argv: Sequence[str] | None = None,
        *,
        ba_handoff: RequirementsAnalystHandoff | None = None,
        preserve_ba_worker: bool = False,
) -> RequirementsReviewStageResult:
    parser = build_parser()
    args = parser.parse_args(argv)
    allow_previous_stage_back = bool(getattr(args, "allow_previous_stage_back", False))
    project_dir = str(Path(args.project_dir).expanduser().resolve()) if args.project_dir else prompt_project_dir("")
    if args.requirement_name:
        requirement_name = str(args.requirement_name).strip()
    else:
        requirement_name = prompt_requirement_name_selection(project_dir, "").requirement_name

    paths = build_requirements_review_paths(project_dir, requirement_name)
    ensure_review_stage_inputs(paths, requirement_name)
    ensure_pre_development_task_record(project_dir, requirement_name)
    update_pre_development_task_status(project_dir, requirement_name, task_key="需求评审", completed=False)

    lock_context = requirement_concurrency_lock(
        project_dir,
        requirement_name,
        action="stage.a04.start",
    )
    lock_context.__enter__()
    progress = ReviewStageProgress()
    shared_review._ACTIVE_REVIEW_PROGRESS = progress
    active_ba_handoff: RequirementsAnalystHandoff | None = None
    reviewer_workers: list[ReviewerRuntime] = []
    cleanup_paths: tuple[str, ...] = ()
    audit_context: StageAuditRunContext | None = None
    try:
        audit_context = begin_stage_audit_run(
            project_dir,
            requirement_name,
            "A04",
            metadata={
                "trigger": "run_requirements_review_stage",
                "argv": list(argv or []),
                "args": vars(args),
            },
        )
        reviewer_label_getter = lambda reviewer, index: _reviewer_artifact_agent_name(reviewer) or f"审核智能体 {index}"  # noqa: E731
        preserved_workers: tuple[TmuxBatchWorker, ...] = ()
        if ba_handoff is not None:
            runtime_root = Path(project_dir).expanduser().resolve() / REQUIREMENTS_REVIEW_RUNTIME_ROOT_NAME
            if Path(ba_handoff.worker.runtime_root).expanduser().resolve() == runtime_root:
                preserved_workers = (ba_handoff.worker,)
        try:
            active_ba_handoff = run_human_check_loop(
                handoff=ba_handoff,
                paths=paths,
                requirement_name=requirement_name,
                allow_previous_stage_back=allow_previous_stage_back,
                auto_confirm=bool(getattr(args, "yes", False)),
            )
            allow_previous_stage_back = False
        except _SkipToDetailedDesign as skip_to_design:
            cleanup_paths = cleanup_existing_review_artifacts(paths, requirement_name, audit_context) + _shutdown_workers(
                skip_to_design.handoff,
                (),
                cleanup_runtime=True,
                preserve_ba_worker=False,
            )
            return RequirementsReviewStageResult(
                project_dir=project_dir,
                requirement_name=requirement_name,
                merged_review_path=str(paths["merged_review_path"].resolve()),
                rounds_used=0,
                passed=False,
                cleanup_paths=cleanup_paths,
                ba_handoff=None,
            )
        review_round_allow_back, allow_previous_stage_back = _consume_stage_back(
            allow_previous_stage_back,
            stdin_is_interactive() and not str(getattr(args, "review_max_rounds", "") or "").strip(),
        )
        review_round_limit = resolve_review_max_rounds(args, progress=progress, allow_back=review_round_allow_back)
        review_round_policy = ReviewRoundPolicy(review_round_limit)
        cleanup_stale_review_runtime_state(
            project_dir,
            requirement_name,
            preserve_workers=preserved_workers,
        )
        cleanup_existing_review_artifacts(paths, requirement_name, audit_context)
        agent_config = resolve_stage_agent_config(args or argparse.Namespace(reviewer_agent=[]))
        reviewer_allow_back, allow_previous_stage_back = _consume_stage_back(
            allow_previous_stage_back,
            stdin_is_interactive() and (
                not agent_config.reviewer_order
                or any(agent_config.reviewer_selection(name) is None for name in agent_config.reviewer_order)
            ),
        )
        reviewer_workers = build_reviewer_workers(
            args=args,
            project_dir=project_dir,
            requirement_name=requirement_name,
            allow_back_first_prompt=reviewer_allow_back,
        )

        def initial_prompt_builder(reviewer: ReviewerRuntime) -> str:
            return requirements_review_init(
                init_prompt="",
                original_requirement_md=str(paths["original_requirement_path"].resolve()),
                hitl_record_md=str(paths["hitl_record_path"].resolve()),
                requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
                requirement_review_md=str(reviewer.review_md_path.resolve()),
                requirement_review_json=str(reviewer.review_json_path.resolve()),
            )

        round_index = 1
        post_hitl_continue_completed = False
        while True:
            if not review_round_policy.initial_review_done:
                reviewer_workers, active_ba_handoff = run_reviewer_phase_with_death_handling(
                    active_ba_handoff,
                    reviewer_workers,
                    run_phase=lambda active_reviewers: _run_parallel_reviewers(
                        active_reviewers,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        round_index=round_index,
                        prompt_builder=initial_prompt_builder,
                        label_prefix="requirements_review_init",
                    ),
                    replace_dead_main_owner=lambda owner: _replace_dead_review_ba(
                        owner,
                        project_dir=project_dir,
                        progress=progress,
                    ),
                    replace_dead_reviewer=lambda reviewer, _index: _replace_dead_requirements_review_reviewer(
                        reviewer,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        progress=progress,
                    ),
                    main_label="需求评审需求分析师",
                    reviewer_label_getter=reviewer_label_getter,
                    notify=message,
                )
                reviewer_workers, active_ba_handoff = run_reviewer_phase_with_death_handling(
                    active_ba_handoff,
                    reviewer_workers,
                    run_phase=lambda active_reviewers: repair_reviewer_outputs(
                        active_reviewers,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        round_index=round_index,
                    ),
                    replace_dead_main_owner=lambda owner: _replace_dead_review_ba(
                        owner,
                        project_dir=project_dir,
                        progress=progress,
                    ),
                    replace_dead_reviewer=lambda reviewer, _index: _replace_dead_requirements_review_reviewer(
                        reviewer,
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        progress=progress,
                    ),
                    main_label="需求评审需求分析师",
                    reviewer_label_getter=reviewer_label_getter,
                    notify=message,
                )
            else:
                active_ba_handoff, reviewer_workers = _run_review_feedback_loop(
                    handoff=active_ba_handoff,
                    reviewers=reviewer_workers,
                    paths=paths,
                    requirement_name=requirement_name,
                    round_index=round_index,
                    skip_ba_feedback=post_hitl_continue_completed,
                    audit_context=audit_context,
                )
                post_hitl_continue_completed = False

            review_round_policy.record_review_attempt()
            ensure_active_reviewers(reviewer_workers, stage_label="需求评审")
            review_json_files, review_md_files = _active_reviewer_files(reviewer_workers)
            passed = task_done(
                directory=project_dir,
                file_path=paths["pre_development_path"],
                task_name=REQUIREMENTS_REVIEW_TASK_NAME,
                json_pattern=f"{sanitize_requirement_name(requirement_name)}_评审记录_*.json",
                md_pattern=f"{sanitize_requirement_name(requirement_name)}_需求评审记录_*.md",
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
                append_stage_audit_record(
                    audit_context,
                    event_type="stage_passed",
                    source_paths={
                        "merged_review": paths["merged_review_path"],
                        "ba_feedback": paths["ba_feedback_path"],
                        "requirements_clear": paths["requirements_clear_path"],
                    },
                    review_round_index=round_index,
                )
                result_ba_handoff = active_ba_handoff if preserve_ba_worker else None
                cleanup_paths = _shutdown_workers(
                    active_ba_handoff,
                    reviewer_workers,
                    cleanup_runtime=True,
                    preserve_ba_worker=preserve_ba_worker,
                )
                return RequirementsReviewStageResult(
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    merged_review_path=str(paths["merged_review_path"].resolve()),
                    rounds_used=round_index,
                    passed=True,
                    cleanup_paths=cleanup_paths,
                    ba_handoff=result_ba_handoff,
                )
            review_msg = get_markdown_content(paths["merged_review_path"]).strip()
            if not review_msg:
                raise RuntimeError("评审未通过，但合并后的需求评审记录为空")
            if review_round_policy.should_escalate_before_next_review():
                if review_round_policy.max_rounds is None:
                    raise RuntimeError("review_round_policy 配置错误：无限轮次不应触发超限 HITL")
                active_ba_handoff, reviewer_workers, post_hitl_continue_completed = run_requirements_review_limit_hitl_loop(
                    active_ba_handoff,
                    reviewers=reviewer_workers,
                    paths=paths,
                    requirement_name=requirement_name,
                    review_msg=review_msg,
                    review_limit=review_round_policy.max_rounds,
                    review_rounds_used=review_round_policy.quota_count,
                    progress=progress,
                    human_input_provider=(
                        lambda question_path, hitl_round: collect_auto_review_limit_hitl_response(
                            question_path,
                            stage_label="需求评审超限",
                            hitl_round=hitl_round,
                        )
                    ) if bool(getattr(args, "yes", False)) else None,
                    audit_context=audit_context,
                )
                review_round_policy.reset_after_hitl()
            round_index += 1
    except Exception as error:
        append_stage_audit_record(
            audit_context,
            event_type="stage_failed",
            source_paths={
                "merged_review": paths["merged_review_path"],
                "ba_feedback": paths["ba_feedback_path"],
                "requirements_clear": paths["requirements_clear_path"],
            },
            metadata={"error": str(error)},
        )
        _shutdown_workers(
            active_ba_handoff,
            reviewer_workers,
            cleanup_runtime=False,
            preserve_ba_worker=preserve_ba_worker,
        )
        raise
    finally:
        progress.stop()
        shared_review._ACTIVE_REVIEW_PROGRESS = None
        lock_context.__exit__(None, None, None)


def main(argv: Sequence[str] | None = None) -> int:
    redirected, launch = maybe_launch_tui(argv, route="review", action="stage.a04.start")
    if redirected:
        return int(launch)
    try:
        result = run_requirements_review_stage(list(launch))
    except Exception as error:  # noqa: BLE001
        message(error)
        return 1
    message("需求评审完成")
    message(result.merged_review_path)
    message("下一步进入详细设计阶段（待接入）")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        cleaned_sessions = cleanup_registered_tmux_workers(reason="keyboard_interrupt")
        if cleaned_sessions:
            message(f"\n已清理 tmux 会话: {', '.join(cleaned_sessions)}")
        raise SystemExit(130)
