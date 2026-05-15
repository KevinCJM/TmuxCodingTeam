# -*- encoding: utf-8 -*-
"""
@File: A08_OverallReview.py
@Modify Time: 2026/4/23
@Author: Kevin-Chen
@Descriptions: 复核阶段
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tmux_core.prompt_contracts.common import check_develop_job, check_reviewer_job
from tmux_core.prompt_contracts.overall_review import (
    init_reviewer,
    refine_all_code,
    review_all_code,
    review_all_code_again,
)
from tmux_core.runtime.contracts import TaskResultContract, TurnFileContract
from tmux_core.runtime.tmux_runtime import (
    DEFAULT_COMMAND_TIMEOUT_SEC,
    cleanup_registered_tmux_workers,
    is_agent_ready_timeout_error,
    is_provider_auth_error,
    is_task_result_contract_error,
    is_turn_artifact_contract_error,
    is_worker_death_error,
    load_worker_from_state_path,
    try_resume_worker,
)
from tmux_core.stage_kernel.development import (
    DEVELOPMENT_RUNTIME_ROOT_NAME,
    DevelopmentAgentHandoff,
    DevelopmentReviewerSpec,
    DeveloperPlan,
    DeveloperRuntime,
    _active_reviewer_files,
    _build_reviewer_turn_goal,
    _developer_display_name,
    _parse_result_payload,
    _predict_worker_display_name,
    _reviewer_artifact_signature,
    _reviewer_has_materialized_outputs,
    build_developer_worker_id,
    build_development_runtime_root,
    build_development_reviewer_worker_id,
    build_development_paths,
    build_placeholder_reviewer_contract,
    build_reviewer_completion_contract,
    create_developer_runtime,
    create_reviewer_runtime,
    prepare_review_round_artifacts,
    recreate_developer_runtime,
    recreate_development_reviewer_runtime,
    resolve_developer_plan,
    resolve_reviewer_specs as resolve_development_reviewer_specs,
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
from tmux_core.stage_kernel.shared_review import (
    MAX_REVIEWER_REPAIR_ATTEMPTS,
    ReviewRoundPolicy,
    ReviewAgentHandoff,
    ReviewAgentSelection,
    ReviewStageProgress,
    ReviewerRuntime,
    build_reviewer_failure_reconfiguration_reason,
    carry_reviewer_failure_state,
    collect_reviewer_agent_selections,
    describe_reviewer_failure_reason,
    ensure_empty_file,
    ensure_review_artifacts,
    is_recoverable_startup_failure,
    mark_worker_awaiting_reconfiguration,
    note_reviewer_failure,
    parse_review_max_rounds,
    prompt_review_max_rounds,
    resolve_stage_agent_config,
    reviewer_requires_manual_model_reconfiguration,
    worker_has_provider_auth_error,
    worker_has_provider_runtime_error,
)
from tmux_core.stage_kernel.agent_intervention import (
    AGENT_INTERVENTION_RECHECK,
    AGENT_INTERVENTION_RECREATE,
    AGENT_INTERVENTION_WORKER_DEAD,
    request_worker_manual_intervention,
)
from tmux_core.stage_kernel.turn_output_goals import (
    OutcomeGoal,
    TaskTurnGoal,
    run_completion_turn_with_repair,
    run_task_result_turn_with_repair,
)
from T01_tools import get_first_false_task, get_markdown_content, is_task_progress_json, task_done
from T09_terminal_ops import BridgeTerminalUI, get_terminal_ui, maybe_launch_tui, message
from T12_requirements_common import (
    prompt_project_dir,
    prompt_requirement_name_selection,
    sanitize_requirement_name,
    stdin_is_interactive,
)


OVERALL_REVIEW_TASK_NAME = "全面复核"
PLACEHOLDER_NEXT_STEP = "下一步进入测试阶段（功能测试 + 全面回归，待接入）"
MAX_OVERALL_REVIEW_ROUNDS = 5


@dataclass(frozen=True)
class OverallReviewStageResult:
    project_dir: str
    requirement_name: str
    merged_review_path: str
    developer_output_path: str
    state_path: str
    completed: bool
    cleanup_paths: tuple[str, ...] = ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="复核阶段")
    parser.add_argument("--project-dir", help="项目目录")
    parser.add_argument("--requirement-name", help="需求名称")
    parser.add_argument("--allow-previous-stage-back", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--vendor", help="开发工程师厂商: codex|claude|gemini|opencode")
    parser.add_argument("--model", help="开发工程师模型名称")
    parser.add_argument("--effort", help="开发工程师推理强度")
    parser.add_argument("--proxy-url", default="", help="开发工程师代理端口或完整代理 URL")
    parser.add_argument("--developer-role-prompt", default="", help="开发工程师自定义角色定义提示词")
    parser.add_argument("--reviewer-agent", action="append", default=[], help="审核智能体模型配置: name=<key>,vendor=...,model=...,effort=...,proxy=...")
    parser.add_argument("--reviewer-role", action="append", default=[], help="重复传入以覆盖复核角色列表")
    parser.add_argument("--reviewer-role-prompt", action="append", default=[], help="重复传入以覆盖对应角色提示词")
    parser.add_argument("--review-max-rounds", default="", help="整体复核最多重试几轮；传 infinite 表示不设上限")
    parser.add_argument("--yes", action="store_true", help="跳过非关键确认")
    parser.add_argument("--no-tui", action="store_true", help="显式禁用 OpenTUI")
    parser.add_argument("--legacy-cli", action="store_true", help="使用旧版 Python CLI，不跳转 OpenTUI")
    return parser


def resolve_overall_review_max_rounds(
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
            default=MAX_OVERALL_REVIEW_ROUNDS,
        )
    if not stdin_is_interactive():
        return MAX_OVERALL_REVIEW_ROUNDS
    if progress is not None and hasattr(progress, "set_phase"):
        progress.set_phase("整体复核 / 配置最大审核轮次")
    return prompt_review_max_rounds(
        default=MAX_OVERALL_REVIEW_ROUNDS,
        progress=progress,
        allow_back=allow_back,
        stage_key="overall_review",
        stage_step_index=3,
    )


def _review_line_is_nonblocking_ambiguity(line: str) -> bool:
    stripped = str(line or "").strip()
    if not stripped:
        return True
    if stripped in {"---", "***"}:
        return True
    if "[Ambiguity]" in stripped or "[疑问]" in stripped or "[歧义]" in stripped:
        return True
    return False


def _review_text_is_only_nonblocking_ambiguities(text: str) -> bool:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return False
    return all(_review_line_is_nonblocking_ambiguity(line) for line in lines)


def _consume_stage_back(allow_previous_stage_back: bool, will_prompt: bool) -> tuple[bool, bool]:
    allow_back = bool(allow_previous_stage_back and will_prompt)
    if will_prompt:
        return allow_back, False
    return False, bool(allow_previous_stage_back)


def build_overall_review_paths(project_dir: str | Path, requirement_name: str) -> dict[str, Path]:
    paths = build_development_paths(project_dir, requirement_name)
    project_root = Path(project_dir).expanduser().resolve()
    safe_name = sanitize_requirement_name(requirement_name)
    paths["merged_review_path"] = project_root / f"{safe_name}_整体代码复核记录.md"
    paths["state_path"] = project_root / f"{safe_name}_复核阶段状态.json"
    return paths


def build_overall_review_reviewer_artifact_paths(
    project_dir: str | Path,
    requirement_name: str,
    reviewer_name: str,
) -> tuple[Path, Path]:
    project_root = Path(project_dir).expanduser().resolve()
    safe_name = sanitize_requirement_name(requirement_name)
    artifact_agent_name = sanitize_requirement_name(reviewer_name)
    review_md_path = project_root / f"{safe_name}_整体代码复核记录_{artifact_agent_name}.md"
    review_json_path = project_root / f"{safe_name}_整体复核记录_{artifact_agent_name}.json"
    return review_md_path, review_json_path


def cleanup_existing_overall_review_artifacts(
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
        f"{safe_name}_整体复核记录_*.json",
        f"{safe_name}_整体代码复核记录_*.md",
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
            {"overall_review": paths["merged_review_path"]},
            metadata={"trigger": "cleanup_existing_overall_review_artifacts"},
            reviewer_markdown_paths=review_md_candidates,
            reviewer_json_paths=review_json_candidates,
        )
    for candidate in (*review_json_candidates, *review_md_candidates):
        if candidate.is_file():
            candidate.unlink()
            removed.append(str(candidate.resolve()))
    merged_review_path = paths["merged_review_path"]
    if merged_review_path.exists() and merged_review_path.is_file():
        merged_review_path.write_text("", encoding="utf-8")
        removed.append(str(merged_review_path.resolve()))
    return tuple(dict.fromkeys(removed))


def _write_overall_review_state(state_path: str | Path, *, passed: bool) -> None:
    path = Path(state_path).expanduser().resolve()
    path.write_text(json.dumps({"passed": bool(passed)}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_overall_review_state(state_path: str | Path) -> dict[str, object]:
    path = Path(state_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def overall_review_passed(state_path: str | Path) -> bool:
    return bool(load_overall_review_state(state_path).get("passed", False))


def cleanup_stale_overall_review_runtime_state(
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
        workflow_action="stage.a08.start",
    )


def ensure_overall_review_inputs(
    *,
    project_dir: str,
    requirement_name: str,
) -> dict[str, Path]:
    paths = build_overall_review_paths(project_dir, requirement_name)
    if not get_markdown_content(paths["original_requirement_path"]).strip():
        raise RuntimeError(f"缺少原始需求文档: {paths['original_requirement_path']}")
    if not get_markdown_content(paths["requirements_clear_path"]).strip():
        raise RuntimeError(f"缺少需求澄清文档: {paths['requirements_clear_path']}")
    if not get_markdown_content(paths["detailed_design_path"]).strip():
        raise RuntimeError(f"缺少详细设计文档: {paths['detailed_design_path']}")
    if not get_markdown_content(paths["task_md_path"]).strip():
        raise RuntimeError(f"进入复核阶段前缺少任务单文档: {paths['task_md_path']}")
    if not is_task_progress_json(paths["task_json_path"]):
        raise RuntimeError(f"进入复核阶段前缺少合法任务单 JSON: {paths['task_json_path']}")
    next_task = get_first_false_task(paths["task_json_path"])
    if next_task is not None:
        raise RuntimeError(f"进入复核阶段前，任务单 JSON 必须全部完成；当前未完成任务: {next_task}")
    return paths


def _is_live_developer_handoff(handoff: DevelopmentAgentHandoff | None) -> bool:
    if handoff is None:
        return False
    if not _worker_reusable_for_handoff(getattr(handoff, "worker", None)):
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
    if not _worker_reusable_for_handoff(getattr(handoff, "worker", None)):
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


def _load_runtime_state_payload(state_path: str | Path) -> dict[str, object]:
    path = Path(state_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _runtime_payload_reusable_for_handoff(payload: dict[str, object]) -> bool:
    status = str(payload.get("status") or payload.get("result_status") or "").strip().lower()
    health_status = str(payload.get("health_status", "")).strip().lower()
    note = str(payload.get("note", "")).strip().lower()
    agent_state = str(payload.get("agent_state", "")).strip().upper()
    task_runtime_status = str(payload.get("current_task_runtime_status", "")).strip().lower()
    active_execution = agent_state == "BUSY" or task_runtime_status == "running"
    if (note == "awaiting_reconfig" or health_status == "awaiting_reconfig") and not active_execution:
        return False
    if status in {"failed", "stale_failed", "error"} and not active_execution:
        return False
    if health_status in {"provider_auth_error", "provider_runtime_error", "missing_session", "pane_dead", "dead"}:
        return False
    if agent_state == "DEAD" and not active_execution:
        return False
    return True


def _worker_state_payload(worker) -> dict[str, object]:
    if worker is None:
        return {}
    reader = getattr(worker, "read_state", None)
    if not callable(reader):
        return {}
    with contextlib.suppress(Exception):
        payload = reader()
        if isinstance(payload, dict):
            return dict(payload)
    return {}


def _worker_reusable_for_handoff(worker) -> bool:
    if worker is None:
        return False
    state_payload = _worker_state_payload(worker)
    if state_payload and not _runtime_payload_reusable_for_handoff(state_payload):
        return False
    return True


def _selection_from_runtime_state(payload: dict[str, object]) -> ReviewAgentSelection | None:
    config_payload = payload.get("config", {})
    if not isinstance(config_payload, dict):
        return None
    vendor = str(config_payload.get("vendor", "")).strip()
    model = str(config_payload.get("model", "")).strip()
    if not vendor or not model:
        return None
    return ReviewAgentSelection(
        vendor=vendor,
        model=model,
        reasoning_effort=str(config_payload.get("reasoning_effort", "high")).strip() or "high",
        proxy_url=str(config_payload.get("proxy_url", "")).strip(),
    )


def _workflow_action_priority(action: str) -> int:
    normalized = str(action or "").strip()
    if normalized == "stage.a08.start":
        return 2
    if normalized == "stage.a07.start":
        return 1
    return 0


def _worker_runtime_metadata(worker) -> dict[str, str]:
    if worker is None:
        return {}
    runtime_metadata = getattr(worker, "runtime_metadata", None)
    if callable(runtime_metadata):
        with contextlib.suppress(Exception):
            payload = runtime_metadata()
            if isinstance(payload, dict):
                return {
                    str(key): "" if value is None else str(value)
                    for key, value in payload.items()
                    if str(key).strip()
                }
    raw_metadata = getattr(worker, "_runtime_metadata", {})
    if isinstance(raw_metadata, dict):
        return {
            str(key): "" if value is None else str(value)
            for key, value in raw_metadata.items()
            if str(key).strip()
        }
    return {}


def _handoff_workflow_action(handoff, *, default: str = "stage.a07.start") -> str:
    worker = getattr(handoff, "worker", None)
    metadata = _worker_runtime_metadata(worker)
    return str(metadata.get("workflow_action", "")).strip() or default


def _handoff_updated_at(handoff) -> str:
    worker = getattr(handoff, "worker", None)
    metadata = _worker_runtime_metadata(worker)
    return str(metadata.get("updated_at", "")).strip()


def _prefer_handoff(current, candidate, *, current_default_action: str = "stage.a07.start", candidate_default_action: str = "stage.a07.start"):
    if current is None:
        return candidate
    if candidate is None:
        return current
    current_priority = _workflow_action_priority(_handoff_workflow_action(current, default=current_default_action))
    candidate_priority = _workflow_action_priority(_handoff_workflow_action(candidate, default=candidate_default_action))
    if candidate_priority != current_priority:
        return candidate if candidate_priority > current_priority else current
    current_updated_at = _handoff_updated_at(current)
    candidate_updated_at = _handoff_updated_at(candidate)
    if current_updated_at and candidate_updated_at and candidate_updated_at > current_updated_at:
        return candidate
    return current


def _merge_preferred_reviewer_handoffs(
    explicit_handoffs: Sequence[ReviewAgentHandoff],
    discovered_handoffs: Sequence[ReviewAgentHandoff],
) -> tuple[ReviewAgentHandoff, ...]:
    merged: dict[str, ReviewAgentHandoff] = {}
    for handoff in explicit_handoffs:
        reviewer_key = str(handoff.reviewer_key).strip()
        if reviewer_key:
            merged[reviewer_key] = handoff
    for handoff in discovered_handoffs:
        reviewer_key = str(handoff.reviewer_key).strip()
        if not reviewer_key:
            continue
        merged[reviewer_key] = _prefer_handoff(
            merged.get(reviewer_key),
            handoff,
        )
    return tuple(merged.values())


def _workflow_action_label(action: str) -> str:
    normalized = str(action or "").strip()
    if normalized == "stage.a08.start":
        return "复核阶段"
    if normalized == "stage.a07.start":
        return "任务开发阶段"
    return normalized or "未知阶段"


def _render_live_handoff_source_summary(
    developer_handoff: DevelopmentAgentHandoff | None,
    reviewer_handoffs: Sequence[ReviewAgentHandoff],
) -> str:
    parts: list[str] = []
    if developer_handoff is not None:
        parts.append(f"开发工程师来源: {_workflow_action_label(_handoff_workflow_action(developer_handoff))}")
    if reviewer_handoffs:
        counts: dict[str, int] = {}
        for handoff in reviewer_handoffs:
            label = _workflow_action_label(_handoff_workflow_action(handoff))
            counts[label] = counts.get(label, 0) + 1
        reviewer_source_text = ", ".join(f"{label}{count}个" for label, count in sorted(counts.items()))
        parts.append(f"审核器来源: {reviewer_source_text}")
    return "；".join(parts)


def discover_live_development_handoffs(
    project_dir: str | Path,
    requirement_name: str,
) -> tuple[DevelopmentAgentHandoff | None, tuple[ReviewAgentHandoff, ...]]:
    runtime_root = Path(project_dir).expanduser().resolve() / DEVELOPMENT_RUNTIME_ROOT_NAME
    if not runtime_root.exists() or not runtime_root.is_dir():
        return None, ()
    resolved_project_dir = str(Path(project_dir).expanduser().resolve())
    candidate_developer: tuple[str, DevelopmentAgentHandoff] | None = None
    reviewer_candidates: dict[str, tuple[str, ReviewAgentHandoff]] = {}
    for state_path in sorted(runtime_root.glob("**/worker.state.json")):
        try:
            if "_locks" in state_path.relative_to(runtime_root).parts:
                continue
        except ValueError:
            pass
        payload = _load_runtime_state_payload(state_path)
        if not payload:
            continue
        if not _runtime_payload_reusable_for_handoff(payload):
            continue
        workflow_action = str(payload.get("workflow_action", "")).strip()
        if workflow_action not in {"stage.a07.start", "stage.a08.start"}:
            continue
        if str(payload.get("project_dir", payload.get("work_dir", ""))).strip() != resolved_project_dir:
            continue
        if str(payload.get("requirement_name", "")).strip() != str(requirement_name).strip():
            continue
        selection = _selection_from_runtime_state(payload)
        if selection is None:
            continue
        worker = load_worker_from_state_path(state_path)
        if worker is None:
            continue
        session_exists = getattr(worker, "session_exists", None)
        if not callable(session_exists):
            continue
        try:
            if not session_exists():
                continue
        except Exception:
            continue
        updated_at = str(payload.get("updated_at", "")).strip()
        candidate_sort_key = f"{_workflow_action_priority(workflow_action)}|{updated_at}"
        worker_id = str(payload.get("worker_id", "")).strip()
        if worker_id == build_developer_worker_id():
            candidate = DevelopmentAgentHandoff(
                selection=selection,
                role_prompt=str(payload.get("role_prompt", "")).strip(),
                worker=worker,
            )
            if not _is_live_developer_handoff(candidate):
                continue
            if candidate_developer is None or candidate_sort_key >= candidate_developer[0]:
                candidate_developer = (candidate_sort_key, candidate)
            continue
        if not worker_id.startswith("development-review-"):
            continue
        reviewer_key = (
            str(payload.get("reviewer_key", "")).strip()
            or str(payload.get("role_name", "")).strip()
            or worker_id.removeprefix("development-review-")
        )
        role_name = str(payload.get("role_name", "")).strip() or reviewer_key
        if not reviewer_key:
            continue
        candidate = ReviewAgentHandoff(
            reviewer_key=reviewer_key,
            role_name=role_name,
            role_prompt=str(payload.get("role_prompt", "")).strip(),
            selection=selection,
            worker=worker,
        )
        if not _is_live_reviewer_handoff(candidate):
            continue
        existing = reviewer_candidates.get(reviewer_key)
        if existing is None or candidate_sort_key >= existing[0]:
            reviewer_candidates[reviewer_key] = (candidate_sort_key, candidate)
    developer_handoff = candidate_developer[1] if candidate_developer is not None else None
    reviewer_handoffs = tuple(reviewer_candidates[key][1] for key in sorted(reviewer_candidates))
    return developer_handoff, reviewer_handoffs


def _set_worker_stage_metadata(
    worker,
    *,
    project_dir: str | Path,
    requirement_name: str,
) -> None:
    set_runtime_metadata = getattr(worker, "set_runtime_metadata", None)
    if callable(set_runtime_metadata):
        set_runtime_metadata(
            project_dir=str(Path(project_dir).expanduser().resolve()),
            requirement_name=str(requirement_name).strip(),
            workflow_action="stage.a08.start",
        )


def bind_developer_runtime_from_handoff(
    *,
    project_dir: str | Path,
    requirement_name: str,
    handoff: DevelopmentAgentHandoff,
) -> DeveloperRuntime:
    _set_worker_stage_metadata(handoff.worker, project_dir=project_dir, requirement_name=requirement_name)
    return DeveloperRuntime(
        selection=handoff.selection,
        worker=handoff.worker,
        role_prompt=handoff.role_prompt,
    )


def bind_reviewer_runtime_from_handoff(
    *,
    project_dir: str | Path,
    requirement_name: str,
    handoff: ReviewAgentHandoff,
) -> ReviewerRuntime:
    reviewer_name = str(getattr(handoff.worker, "session_name", "") or "").strip() or handoff.role_name
    _set_worker_stage_metadata(handoff.worker, project_dir=project_dir, requirement_name=requirement_name)
    review_md_path, review_json_path = build_overall_review_reviewer_artifact_paths(project_dir, requirement_name, reviewer_name)
    ensure_review_artifacts(review_md_path, review_json_path)
    reviewer = ReviewerRuntime(
        reviewer_name=handoff.reviewer_key,
        selection=handoff.selection,
        worker=handoff.worker,
        review_md_path=review_md_path,
        review_json_path=review_json_path,
        contract=build_placeholder_reviewer_contract(review_json_path),
        failure_streak=getattr(handoff, "failure_streak", 0) or 0,
        last_failure_reason=str(getattr(handoff, "last_failure_reason", "") or "").strip(),
    )
    return normalize_overall_review_reviewer_runtime(
        reviewer,
        project_dir=project_dir,
        requirement_name=requirement_name,
    )


def normalize_overall_review_reviewer_runtime(
    reviewer: ReviewerRuntime,
    *,
    project_dir: str | Path,
    requirement_name: str,
) -> ReviewerRuntime:
    artifact_reviewer_name = str(getattr(reviewer.worker, "session_name", "") or "").strip() or reviewer.reviewer_name
    review_md_path, review_json_path = build_overall_review_reviewer_artifact_paths(
        project_dir,
        requirement_name,
        artifact_reviewer_name,
    )
    ensure_review_artifacts(review_md_path, review_json_path)
    current_status_path = Path(getattr(reviewer.contract, "status_path", review_json_path)).expanduser().resolve()
    if (
        reviewer.review_md_path.resolve() == review_md_path.resolve()
        and reviewer.review_json_path.resolve() == review_json_path.resolve()
        and current_status_path == review_json_path.resolve()
    ):
        return reviewer
    return ReviewerRuntime(
        reviewer_name=reviewer.reviewer_name,
        selection=reviewer.selection,
        worker=reviewer.worker,
        review_md_path=review_md_path,
        review_json_path=review_json_path,
        contract=build_placeholder_reviewer_contract(review_json_path),
        failure_streak=getattr(reviewer, "failure_streak", 0) or 0,
        last_failure_reason=str(getattr(reviewer, "last_failure_reason", "") or "").strip(),
    )


def _overall_reviewer_display_name(reviewer: ReviewerRuntime) -> str:
    return str(reviewer.worker.session_name or reviewer.reviewer_name).strip() or reviewer.reviewer_name


def _overall_reviewer_target_paths(
    reviewer: ReviewerRuntime,
    paths: dict[str, Path] | None = None,
) -> tuple[Path, ...]:
    targets: list[Path] = [reviewer.review_md_path, reviewer.review_json_path]
    if paths:
        for key in ("task_json_path", "developer_output_path", "state_path"):
            target = paths.get(key)
            if target is not None:
                targets.append(target)
    return tuple(targets)


def _kill_overall_reviewer_best_effort(reviewer: ReviewerRuntime) -> None:
    with contextlib.suppress(Exception):
        reviewer.worker.request_kill()


def _prompt_overall_reviewer_recovery(
    reviewer: ReviewerRuntime,
    *,
    reason_text: str,
    paths: dict[str, Path] | None = None,
    progress: ReviewStageProgress | None = None,
) -> str:
    return request_worker_manual_intervention(
        stage_label="复核阶段",
        role_label=_overall_reviewer_display_name(reviewer),
        worker=reviewer.worker,
        reason_text=reason_text,
        target_paths=_overall_reviewer_target_paths(reviewer, paths),
        progress=progress,
        allow_recreate=True,
        allow_worker_dead=True,
    )


def _recreate_overall_reviewer_from_hitl(
    *,
    project_dir: str | Path,
    requirement_name: str,
    reviewer: ReviewerRuntime,
    reviewer_spec: DevelopmentReviewerSpec,
    progress: ReviewStageProgress | None,
    force_model_change: bool,
    required_reconfiguration: bool = False,
    reason_text: str = "",
    carry_failure_from: ReviewerRuntime | None = None,
) -> ReviewerRuntime | None:
    replacement = recreate_development_reviewer_runtime(
        project_dir=project_dir,
        requirement_name=requirement_name,
        reviewer=reviewer,
        reviewer_spec=reviewer_spec,
        progress=progress,
        force_model_change=force_model_change,
        required_reconfiguration=required_reconfiguration,
        reason_text=reason_text,
        reuse_existing_selection=not force_model_change,
    )
    if replacement is None:
        message(f"{_overall_reviewer_display_name(reviewer)} 重新创建失败，请重新选择恢复方式。")
        return None
    _set_worker_stage_metadata(replacement.worker, project_dir=project_dir, requirement_name=requirement_name)
    if carry_failure_from is not None and not force_model_change:
        replacement = carry_reviewer_failure_state(replacement, previous=carry_failure_from)
    return normalize_overall_review_reviewer_runtime(
        replacement,
        project_dir=project_dir,
        requirement_name=requirement_name,
    )


def resolve_overall_review_reviewer_specs(
    args: argparse.Namespace,
    *,
    reviewer_handoff: Sequence[ReviewAgentHandoff],
    progress: ReviewStageProgress | None = None,
    allow_back_first_prompt: bool = False,
) -> list[DevelopmentReviewerSpec]:
    if reviewer_handoff:
        return [
            DevelopmentReviewerSpec(
                role_name=item.role_name,
                role_prompt=item.role_prompt,
                reviewer_key=item.reviewer_key,
            )
            for item in reviewer_handoff
        ]
    return list(
        resolve_development_reviewer_specs(
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
    reviewer_specs: Sequence[DevelopmentReviewerSpec],
    reviewer_handoff: Sequence[ReviewAgentHandoff],
    reviewer_selections_by_name: dict[str, ReviewAgentSelection] | None = None,
    progress: ReviewStageProgress | None = None,
) -> list[ReviewerRuntime]:
    if progress is not None:
        progress.set_phase("复核阶段 / 启动审核器")
    reviewers: list[ReviewerRuntime] = []
    predicted_session_names: set[str] = set()
    live_handoffs_by_key = {
        item.reviewer_key: item
        for item in reviewer_handoff
        if _is_live_reviewer_handoff(item)
    }
    if live_handoffs_by_key:
        message("复用仍存活的任务开发审核智能体进入复核阶段")
    for reviewer_spec in reviewer_specs:
        reviewer_key = str(reviewer_spec.reviewer_key or reviewer_spec.role_name).strip()
        live_handoff = live_handoffs_by_key.get(reviewer_key)
        if live_handoff is not None:
            reviewers.append(
                normalize_overall_review_reviewer_runtime(
                    bind_reviewer_runtime_from_handoff(
                        project_dir=project_dir,
                        requirement_name=requirement_name,
                        handoff=live_handoff,
                    ),
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                )
            )
            continue
        reviewer_display_name = _predict_worker_display_name(
            project_dir=project_dir,
            worker_id=build_development_reviewer_worker_id(reviewer_spec.role_name),
            occupied_session_names=sorted(predicted_session_names),
        )
        predicted_session_names.add(reviewer_display_name)
        selection = (reviewer_selections_by_name or {}).get(reviewer_key)
        if selection is None:
            raise RuntimeError(f"缺少复核审核智能体 {reviewer_key} 的模型配置")
        reviewer = create_reviewer_runtime(
            project_dir=project_dir,
            requirement_name=requirement_name,
            reviewer_spec=reviewer_spec,
            selection=selection,
        )
        _set_worker_stage_metadata(reviewer.worker, project_dir=project_dir, requirement_name=requirement_name)
        reviewers.append(
            normalize_overall_review_reviewer_runtime(
                reviewer,
                project_dir=project_dir,
                requirement_name=requirement_name,
            )
        )
    return reviewers


def _parse_lenient_result_payload(text: str) -> dict[str, object]:
    stripped = str(text or "").strip()
    if not stripped or stripped in {"完成", "修改完成", "审核通过", "未通过"}:
        return {}
    try:
        return _parse_result_payload(stripped)
    except Exception:
        return {}


def build_overall_review_init_result_contract(paths: dict[str, Path], *, mode: str) -> TaskResultContract:
    return TaskResultContract(
        turn_id=mode,
        phase=mode,
        task_kind=mode,
        mode=mode,
        expected_statuses=("ready",),
        stage_name="复核阶段",
        optional_artifacts={
            "original_requirement": paths["original_requirement_path"],
            "requirements_clear": paths["requirements_clear_path"],
            "hitl_record": paths["hitl_record_path"],
            "detailed_design": paths["detailed_design_path"],
            "task_md": paths["task_md_path"],
            "task_json": paths["task_json_path"],
        },
    )


def build_overall_review_refine_result_contract(paths: dict[str, Path]) -> TaskResultContract:
    return TaskResultContract(
        turn_id="a08_developer_refine_all_code",
        phase="a08_developer_refine_all_code",
        task_kind="a08_developer_refine_all_code",
        mode="a08_developer_refine_all_code",
        expected_statuses=("completed",),
        stage_name="复核阶段",
        required_artifacts={"developer_output": paths["developer_output_path"]},
        optional_artifacts={
            "original_requirement": paths["original_requirement_path"],
            "requirements_clear": paths["requirements_clear_path"],
            "hitl_record": paths["hitl_record_path"],
            "detailed_design": paths["detailed_design_path"],
            "task_md": paths["task_md_path"],
            "task_json": paths["task_json_path"],
        },
    )


def build_overall_review_metadata_repair_result_contract(paths: dict[str, Path]) -> TaskResultContract:
    return build_overall_review_refine_result_contract(paths)


def build_overall_review_reviewer_completion_contract(
    *,
    reviewer_name: str,
    task_name: str,
    review_md_path: Path,
    review_json_path: Path,
) -> TurnFileContract:
    base_contract = build_reviewer_completion_contract(
        reviewer_name=reviewer_name,
        task_name=task_name,
        review_md_path=review_md_path,
        review_json_path=review_json_path,
    )
    return TurnFileContract(
        turn_id=f"overall_review_{sanitize_requirement_name(task_name)}_{sanitize_requirement_name(reviewer_name)}",
        phase="复核阶段",
        status_path=review_json_path,
        validator=base_contract.validator,
        quiet_window_sec=base_contract.quiet_window_sec,
        kind=base_contract.kind,
        tracked_artifacts=base_contract.tracked_artifacts,
        artifact_rules=base_contract.artifact_rules,
        outcome_artifacts=base_contract.outcome_artifacts,
    )


_OVERALL_REVIEW_ACTIVE_CODE_SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".sh",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".swift",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".toml",
    ".yaml",
    ".yml",
}

_OVERALL_REVIEW_CONTEXT_EXCLUDED_DIRS = {
    ".detailed_design_runtime",
    ".development_runtime",
    ".pytest_cache",
    ".requirements_clarification_runtime",
    ".requirements_review_runtime",
    ".routing_init_runtime",
    ".task_split_runtime",
    ".tmux_stage_locks",
    ".tmux_workflow",
    ".venv",
    ".git",
    "__pycache__",
    "node_modules",
    "venv",
}
_OVERALL_REVIEW_CONTEXT_ROOT_EXCLUDED_DIRS = {"data", "docs", "logs", "vendor"}

_OVERALL_REVIEW_DEFAULT_CONTEXT_DIRS = ("src", "tests", "configs", "scripts")
_OVERALL_REVIEW_ROOT_CONTEXT_FILES = (
    "AGENTS.md",
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "tox.ini",
    "pytest.ini",
    "mypy.ini",
    "package.json",
    "tsconfig.json",
)


def _overall_review_rel_is_excluded(rel_path: Path) -> bool:
    parts = rel_path.parts
    if not parts:
        return False
    if parts[0] in _OVERALL_REVIEW_CONTEXT_ROOT_EXCLUDED_DIRS:
        return True
    return any(part in _OVERALL_REVIEW_CONTEXT_EXCLUDED_DIRS for part in parts)


def _overall_review_rel_is_active_file(rel_path: Path) -> bool:
    if _overall_review_rel_is_excluded(rel_path):
        return False
    if rel_path.suffix.lower() not in _OVERALL_REVIEW_ACTIVE_CODE_SUFFIXES:
        return False
    if rel_path.name.startswith(".") and rel_path.suffix.lower() not in {".toml", ".yaml", ".yml"}:
        return False
    return True


def _safe_repo_map_ref(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    rel = Path(text)
    if rel.is_absolute() or any(part == ".." for part in rel.parts):
        return None
    if _overall_review_rel_is_excluded(rel):
        return None
    return rel


def _repo_map_owned_path_refs(project_root: Path) -> tuple[Path, ...]:
    repo_map_path = project_root / "docs" / "repo_map.json"
    if not repo_map_path.exists() or not repo_map_path.is_file():
        return ()
    try:
        payload = json.loads(repo_map_path.read_text(encoding="utf-8"))
    except Exception:
        return ()
    refs: list[Path] = []
    modules = payload.get("modules", [])
    if not isinstance(modules, list):
        return ()
    for module in modules:
        if not isinstance(module, dict):
            continue
        owned_paths = module.get("owned_paths", [])
        if not isinstance(owned_paths, list):
            continue
        for item in owned_paths:
            if not isinstance(item, dict):
                continue
            rel = _safe_repo_map_ref(item.get("ref") or item.get("path"))
            if rel is not None:
                refs.append(rel)
    return tuple(dict.fromkeys(refs))


def _overall_review_context_seed_paths(project_root: Path) -> tuple[Path, ...]:
    seeds: list[Path] = []
    seeds.extend(_repo_map_owned_path_refs(project_root))
    seeds.extend(Path(item) for item in _OVERALL_REVIEW_DEFAULT_CONTEXT_DIRS)
    seeds.extend(Path(item) for item in _OVERALL_REVIEW_ROOT_CONTEXT_FILES)
    return tuple(dict.fromkeys(seed for seed in seeds if not _overall_review_rel_is_excluded(seed)))


def _iter_root_active_files(project_root: Path):
    try:
        with os.scandir(project_root) as entries_iter:
            entries = sorted(entries_iter, key=lambda entry: entry.name)
    except OSError:
        return
    for entry in entries:
        try:
            if not entry.is_file(follow_symlinks=False):
                continue
            rel = Path(entry.name)
            if _overall_review_rel_is_active_file(rel):
                yield rel
        except OSError:
            continue


def _iter_active_files_under_seed(project_root: Path, seed: Path):
    target = project_root / seed
    if not target.exists():
        return
    if target.is_file():
        try:
            rel = target.relative_to(project_root)
        except ValueError:
            return
        if _overall_review_rel_is_active_file(rel):
            yield rel
        return
    if not target.is_dir():
        return
    pending = [target]
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as entries_iter:
                entries = sorted(entries_iter, key=lambda entry: entry.name)
        except OSError:
            continue
        dirs: list[Path] = []
        for entry in entries:
            path = Path(entry.path)
            try:
                rel = path.relative_to(project_root)
            except ValueError:
                continue
            if _overall_review_rel_is_excluded(rel):
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    dirs.append(path)
                    continue
                if entry.is_file(follow_symlinks=False) and _overall_review_rel_is_active_file(rel):
                    yield rel
            except OSError:
                continue
        pending.extend(reversed(dirs))


def _discover_overall_review_active_files(project_dir: str | Path, *, limit: int = 40) -> tuple[str, ...]:
    root = Path(project_dir).expanduser().resolve()
    if not root.exists():
        return ()
    discovered: list[str] = []
    seen: set[str] = set()
    for rel_iter in (
        _iter_root_active_files(root),
        *(_iter_active_files_under_seed(root, seed) for seed in _overall_review_context_seed_paths(root)),
    ):
        for rel in rel_iter:
            text = rel.as_posix()
            if text in seen:
                continue
            seen.add(text)
            discovered.append(text)
            if len(discovered) >= limit:
                return tuple(discovered)
    return tuple(discovered)


def _build_overall_review_active_code_context(project_dir: str | Path) -> str:
    root = Path(project_dir).expanduser().resolve()
    files = _discover_overall_review_active_files(root)
    if not files:
        return (
            "\n\n## 当前代码事实补充\n"
            f"- 当前项目根目录: `{root}`\n"
            "- 未自动发现常见代码/配置文件；如 routing docs 与实际目录不一致，必须以当前目录实际文件检查结果为准。\n"
            "- routing docs 只用于导航，不能仅凭 `repo_map.json` 的历史 scope_notes 判定代码不存在。\n"
        )
    file_lines = "\n".join(f"- `{item}`" for item in files)
    return (
        "\n\n## 当前代码事实补充\n"
        f"- 当前项目根目录: `{root}`\n"
        "- 以下是在当前项目目录内发现的活跃代码/配置入口，复核时必须实际检查这些文件。\n"
        "- routing docs 只用于导航，不能仅凭 `repo_map.json` 的历史 scope_notes 判定业务代码不存在。\n"
        f"{file_lines}\n"
    )


def _resolve_overall_review_active_code_context(
    project_dir: str | Path | None,
    active_code_context: str | None = None,
) -> str:
    if active_code_context is not None:
        return active_code_context
    if project_dir is None:
        return ""
    return _build_overall_review_active_code_context(project_dir)


def build_overall_review_init_prompt(
    paths: dict[str, Path],
    *,
    project_dir: str | Path | None = None,
    active_code_context: str | None = None,
) -> str:
    prompt = init_reviewer(
        hitl_record_md=str(paths["hitl_record_path"].resolve()),
        requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
        original_requirement_md=str(paths["original_requirement_path"].resolve()),
        detailed_design_md=str(paths["detailed_design_path"].resolve()),
        task_split_md=str(paths["task_md_path"].resolve()),
    )
    prompt += _resolve_overall_review_active_code_context(project_dir, active_code_context)
    return prompt


def build_overall_review_turn_goal(*, mode: str, paths: dict[str, Path]) -> TaskTurnGoal:
    if mode in {"a08_reviewer_init", "a08_developer_init"}:
        return TaskTurnGoal(
            goal_id=mode,
            outcomes={"ready": OutcomeGoal(status="ready")},
        )
    if mode == "a08_developer_refine_all_code":
        return TaskTurnGoal(
            goal_id=mode,
            outcomes={"completed": OutcomeGoal(status="completed", required_aliases=("developer_output",))},
            repair_prompt_builder=lambda _context: check_develop_job(
                paths["developer_output_path"],
                OVERALL_REVIEW_TASK_NAME,
                task_split_md=str(paths["task_md_path"].resolve()),
                what_just_dev=str(paths["developer_output_path"].resolve()),
            ),
        )
    raise RuntimeError(f"未知复核阶段 turn goal: {mode}")


def _replace_dead_overall_review_developer(
    developer: DeveloperRuntime,
    *,
    project_dir: str | Path,
    requirement_name: str,
    progress: ReviewStageProgress | None = None,
    error: Exception | None = None,
) -> DeveloperRuntime:
    developer_name = str(developer.worker.session_name or "开发工程师").strip() or "开发工程师"
    startup_reconfigure = bool(error and is_recoverable_startup_failure(error, developer.worker))
    replacement = recreate_developer_runtime(
        project_dir=project_dir,
        requirement_name=requirement_name,
        developer=developer,
        progress=progress,
        reason_text=(
            str(error or "")
            if startup_reconfigure
            else f"{developer_name}已死亡，必须重新选择厂商/模型/推理/代理后从当前阶段继续。"
        ),
        required_reconfiguration=True,
    )
    if replacement is None:
        raise RuntimeError(f"{developer_name} 无法继续，且未能重建开发工程师")
    _set_worker_stage_metadata(replacement.worker, project_dir=project_dir, requirement_name=requirement_name)
    return replacement


def _run_overall_review_developer_turn(
    developer: DeveloperRuntime,
    *,
    project_dir: str | Path,
    requirement_name: str,
    label: str,
    prompt: str,
    result_contract: TaskResultContract,
    paths: dict[str, Path],
    progress: ReviewStageProgress | None = None,
) -> DeveloperRuntime:
    current_developer = developer
    while True:
        try:
            run_task_result_turn_with_repair(
                worker=current_developer.worker,
                label=label,
                prompt=prompt,
                result_contract=result_contract,
                repair_result_contract=(
                    build_overall_review_metadata_repair_result_contract(paths)
                    if result_contract.mode == "a08_developer_refine_all_code"
                    else None
                ),
                parse_result_payload=_parse_lenient_result_payload,
                turn_goal=build_overall_review_turn_goal(mode=result_contract.mode, paths=paths),
                timeout_sec=DEFAULT_COMMAND_TIMEOUT_SEC,
                stage_label="复核阶段",
                role_label=str(current_developer.worker.session_name or "开发工程师").strip() or "开发工程师",
                task_name=OVERALL_REVIEW_TASK_NAME,
                requirement_name=requirement_name,
            )
            return current_developer
        except Exception as error:  # noqa: BLE001
            if is_worker_death_error(error) or is_recoverable_startup_failure(error, current_developer.worker):
                current_developer = _replace_dead_overall_review_developer(
                    current_developer,
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    progress=progress,
                    error=error,
                )
                continue
            if is_task_result_contract_error(error):
                raise
            if try_resume_worker(current_developer.worker, timeout_sec=60.0):
                continue
            raise RuntimeError(f"{current_developer.worker.session_name or '开发工程师'} 执行失败") from error


def initialize_overall_review_developer(
    developer: DeveloperRuntime,
    *,
    project_dir: str | Path,
    requirement_name: str,
    paths: dict[str, Path],
    progress: ReviewStageProgress | None = None,
    active_code_context: str | None = None,
) -> DeveloperRuntime:
    if progress is not None:
        progress.set_phase("复核阶段 / 初始化开发工程师")
    return _run_overall_review_developer_turn(
        developer,
        project_dir=project_dir,
        requirement_name=requirement_name,
        label="overall_review_developer_init",
        prompt=build_overall_review_init_prompt(
            paths,
            project_dir=project_dir,
            active_code_context=active_code_context,
        ),
        result_contract=build_overall_review_init_result_contract(paths, mode="a08_developer_init"),
        paths=paths,
        progress=progress,
    )


def refine_overall_review_code(
    developer: DeveloperRuntime,
    *,
    project_dir: str | Path,
    requirement_name: str,
    paths: dict[str, Path],
    review_msg: str,
    progress: ReviewStageProgress | None = None,
    audit_context: StageAuditRunContext | None = None,
    review_round_index: int | None = None,
) -> tuple[DeveloperRuntime, str]:
    if audit_context is not None:
        record_before_cleanup(
            audit_context,
            {"developer_output": paths["developer_output_path"]},
            metadata={"trigger": "refine_overall_review_code_prepare"},
            review_round_index=review_round_index,
        )
    ensure_empty_file(paths["developer_output_path"])
    if progress is not None:
        progress.set_phase("复核阶段 / 开发工程师修订代码")
    current_developer = _run_overall_review_developer_turn(
        developer,
        project_dir=project_dir,
        requirement_name=requirement_name,
        label="overall_review_refine_all_code",
        prompt=refine_all_code(
            review_msg,
            hitl_record_md=str(paths["hitl_record_path"].resolve()),
            requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
            original_requirement_md=str(paths["original_requirement_path"].resolve()),
            detailed_design_md=str(paths["detailed_design_path"].resolve()),
            what_just_dev=str(paths["developer_output_path"].resolve()),
        ),
        result_contract=build_overall_review_refine_result_contract(paths),
        paths=paths,
        progress=progress,
    )
    code_change = get_markdown_content(paths["developer_output_path"]).strip()
    reminder_prompt = check_develop_job(
        paths["developer_output_path"],
        OVERALL_REVIEW_TASK_NAME,
        task_split_md=str(paths["task_md_path"].resolve()),
        what_just_dev=str(paths["developer_output_path"].resolve()),
    )
    if reminder_prompt or not code_change:
        current_developer = _run_overall_review_developer_turn(
            current_developer,
            project_dir=project_dir,
            requirement_name=requirement_name,
            label="overall_review_refine_all_code_metadata_repair",
            prompt=reminder_prompt or "请按协议补全工程师开发内容.md，然后只返回 修改完成",
            result_contract=build_overall_review_metadata_repair_result_contract(paths),
            paths=paths,
            progress=progress,
        )
        code_change = get_markdown_content(paths["developer_output_path"]).strip()
    if not code_change:
        raise RuntimeError(f"复核阶段修订完成后，开发工程师仍未按协议更新《{paths['developer_output_path'].name}》")
    if audit_context is not None:
        append_stage_audit_record(
            audit_context,
            event_type="change_after_overall_review",
            source_paths={
                "developer_output": paths["developer_output_path"],
                "overall_review": paths["merged_review_path"],
            },
            review_round_index=review_round_index,
            metadata={"trigger": "refine_overall_review_code"},
        )
    return current_developer, code_change


def _run_single_overall_review_reviewer_init(
    reviewer: ReviewerRuntime,
    *,
    project_dir: str | Path,
    requirement_name: str,
    paths: dict[str, Path],
    reviewer_specs_by_name: dict[str, DevelopmentReviewerSpec],
    progress: ReviewStageProgress | None = None,
    active_code_context: str | None = None,
) -> ReviewerRuntime | None:
    init_contract = build_overall_review_init_result_contract(paths, mode="a08_reviewer_init")
    current_reviewer = reviewer
    while True:
        current_reviewer = normalize_overall_review_reviewer_runtime(
            current_reviewer,
            project_dir=project_dir,
            requirement_name=requirement_name,
        )
        reviewer_spec = reviewer_specs_by_name[current_reviewer.reviewer_name]
        try:
            run_task_result_turn_with_repair(
                worker=current_reviewer.worker,
                label=f"overall_review_reviewer_init_{sanitize_requirement_name(current_reviewer.reviewer_name)}",
                prompt=build_overall_review_init_prompt(
                    paths,
                    project_dir=project_dir,
                    active_code_context=active_code_context,
                ),
                result_contract=init_contract,
                parse_result_payload=_parse_lenient_result_payload,
                turn_goal=build_overall_review_turn_goal(mode=init_contract.mode, paths=paths),
                timeout_sec=DEFAULT_COMMAND_TIMEOUT_SEC,
                stage_label="复核阶段",
                role_label=str(current_reviewer.worker.session_name or current_reviewer.reviewer_name).strip() or current_reviewer.reviewer_name,
                task_name=OVERALL_REVIEW_TASK_NAME,
                requirement_name=requirement_name,
            )
            return current_reviewer
        except Exception as error:  # noqa: BLE001
            reviewer_display_name = str(current_reviewer.worker.session_name or current_reviewer.reviewer_name).strip() or current_reviewer.reviewer_name
            if is_worker_death_error(error):
                failure_reason = describe_reviewer_failure_reason(error, current_reviewer.worker)
                failed_reviewer = note_reviewer_failure(current_reviewer, reason_text=failure_reason)
                reason_text = f"{reviewer_display_name} 初始化失败或已死亡。\n{failure_reason or str(error)}".strip()
                decision = _prompt_overall_reviewer_recovery(
                    failed_reviewer,
                    reason_text=reason_text,
                    paths=paths,
                    progress=progress,
                )
                if decision == AGENT_INTERVENTION_RECHECK:
                    current_reviewer = failed_reviewer
                    continue
                if decision == AGENT_INTERVENTION_WORKER_DEAD:
                    _kill_overall_reviewer_best_effort(failed_reviewer)
                    message(f"{reviewer_display_name} 已按死亡处理，当前阶段将忽略该审核智能体。")
                    return None
                if decision != AGENT_INTERVENTION_RECREATE:
                    continue
                force_model_change = reviewer_requires_manual_model_reconfiguration(failed_reviewer)
                if force_model_change:
                    reason_text = build_reviewer_failure_reconfiguration_reason(
                        failed_reviewer,
                        role_label=reviewer_display_name,
                        failure_reason=failure_reason,
                    )
                    mark_worker_awaiting_reconfiguration(current_reviewer.worker, reason_text=reason_text)
                replacement = _recreate_overall_reviewer_from_hitl(
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    reviewer=failed_reviewer,
                    reviewer_spec=reviewer_spec,
                    progress=progress,
                    force_model_change=force_model_change,
                    required_reconfiguration=force_model_change,
                    reason_text=reason_text,
                    carry_failure_from=failed_reviewer,
                )
                if replacement is None:
                    current_reviewer = failed_reviewer
                else:
                    current_reviewer = replacement
                continue
            if is_recoverable_startup_failure(error, current_reviewer.worker):
                reason_text = (
                    f"检测到{reviewer_display_name}仍在 agent 界面，但模型认证已失效。\n需要更换模型后继续当前阶段。"
                    if is_provider_auth_error(error) or worker_has_provider_auth_error(current_reviewer.worker)
                    else f"检测到{reviewer_display_name}的模型服务出现临时运行错误。\n需要更换或重启模型后继续当前阶段。"
                    if worker_has_provider_runtime_error(current_reviewer.worker)
                    else f"{reviewer_display_name}启动超时，未能进入可输入状态。\n需要更换模型后继续当前阶段。"
                )
                decision = _prompt_overall_reviewer_recovery(
                    current_reviewer,
                    reason_text=reason_text,
                    paths=paths,
                    progress=progress,
                )
                if decision == AGENT_INTERVENTION_RECHECK:
                    continue
                if decision == AGENT_INTERVENTION_WORKER_DEAD:
                    _kill_overall_reviewer_best_effort(current_reviewer)
                    message(f"{reviewer_display_name} 已按死亡处理，当前阶段将忽略该审核智能体。")
                    return None
                if decision != AGENT_INTERVENTION_RECREATE:
                    continue
                mark_worker_awaiting_reconfiguration(current_reviewer.worker, reason_text=reason_text)
                replacement = _recreate_overall_reviewer_from_hitl(
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    reviewer=current_reviewer,
                    reviewer_spec=reviewer_spec,
                    progress=progress,
                    force_model_change=True,
                    required_reconfiguration=True,
                    reason_text=reason_text,
                )
                if replacement is not None:
                    current_reviewer = replacement
                continue
            raise


def initialize_overall_review_reviewers(
    reviewers: Sequence[ReviewerRuntime],
    *,
    project_dir: str | Path,
    requirement_name: str,
    paths: dict[str, Path],
    reviewer_specs_by_name: dict[str, DevelopmentReviewerSpec],
    progress: ReviewStageProgress | None = None,
    active_code_context: str | None = None,
) -> list[ReviewerRuntime]:
    if progress is not None:
        progress.set_phase("复核阶段 / 初始化审核器")
    return run_parallel_reviewer_round(
        reviewers,
        key_func=lambda reviewer: reviewer.reviewer_name,
        run_turn=lambda reviewer: _run_single_overall_review_reviewer_init(
            reviewer,
            project_dir=project_dir,
            requirement_name=requirement_name,
            paths=paths,
            reviewer_specs_by_name=reviewer_specs_by_name,
            progress=progress,
            active_code_context=active_code_context,
        ),
        error_prefix="复核阶段审核智能体初始化失败:",
    )


def run_overall_review_turn_with_recreation(
    reviewer: ReviewerRuntime,
    *,
    project_dir: str | Path,
    requirement_name: str,
    reviewer_spec: DevelopmentReviewerSpec,
    paths: dict[str, Path],
    reviewer_specs_by_name: dict[str, DevelopmentReviewerSpec],
    label: str,
    prompt: str,
    progress: ReviewStageProgress | None = None,
) -> ReviewerRuntime | None:
    current_reviewer = reviewer
    while True:
        current_reviewer = normalize_overall_review_reviewer_runtime(
            current_reviewer,
            project_dir=project_dir,
            requirement_name=requirement_name,
        )
        baseline_signature = _reviewer_artifact_signature(current_reviewer)
        try:
            run_completion_turn_with_repair(
                worker=current_reviewer.worker,
                label=label,
                prompt=prompt,
                completion_contract=build_overall_review_reviewer_completion_contract(
                    reviewer_name=current_reviewer.reviewer_name,
                    task_name=OVERALL_REVIEW_TASK_NAME,
                    review_md_path=current_reviewer.review_md_path,
                    review_json_path=current_reviewer.review_json_path,
                ),
                turn_goal=_build_reviewer_turn_goal(),
                timeout_sec=DEFAULT_COMMAND_TIMEOUT_SEC,
                stage_label="复核阶段",
                role_label=str(current_reviewer.worker.session_name or current_reviewer.reviewer_name).strip() or current_reviewer.reviewer_name,
                task_name=OVERALL_REVIEW_TASK_NAME,
                requirement_name=requirement_name,
            )
            return current_reviewer
        except Exception as error:  # noqa: BLE001
            if is_turn_artifact_contract_error(error):
                return current_reviewer
            if (
                _reviewer_has_materialized_outputs(current_reviewer, OVERALL_REVIEW_TASK_NAME)
                and _reviewer_artifact_signature(current_reviewer) != baseline_signature
            ):
                return current_reviewer
            reviewer_display_name = str(current_reviewer.worker.session_name or current_reviewer.reviewer_name).strip() or current_reviewer.reviewer_name
            auth_error = is_provider_auth_error(error) or worker_has_provider_auth_error(current_reviewer.worker)
            provider_runtime_error = worker_has_provider_runtime_error(current_reviewer.worker)
            ready_timeout_error = is_agent_ready_timeout_error(error)
            if auth_error or provider_runtime_error or ready_timeout_error:
                reason_text = (
                    f"检测到{reviewer_display_name}仍在 agent 界面，但模型认证已失效。\n需要更换模型后继续当前阶段。"
                    if auth_error
                    else f"检测到{reviewer_display_name}的模型服务出现临时运行错误。\n需要更换或重启模型后继续当前阶段。"
                    if provider_runtime_error
                    else f"{reviewer_display_name}启动超时，未能进入可输入状态。\n需要更换模型后继续当前阶段。"
                )
                decision = _prompt_overall_reviewer_recovery(
                    current_reviewer,
                    reason_text=reason_text,
                    paths=paths,
                    progress=progress,
                )
                if decision == AGENT_INTERVENTION_RECHECK:
                    continue
                if decision == AGENT_INTERVENTION_WORKER_DEAD:
                    _kill_overall_reviewer_best_effort(current_reviewer)
                    message(f"{reviewer_display_name} 已按死亡处理，当前阶段将忽略该审核智能体。")
                    return None
                if decision != AGENT_INTERVENTION_RECREATE:
                    continue
                mark_worker_awaiting_reconfiguration(current_reviewer.worker, reason_text=reason_text)
                replacement = _recreate_overall_reviewer_from_hitl(
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
                    continue
                initialized = _run_single_overall_review_reviewer_init(
                    replacement,
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    paths=paths,
                    reviewer_specs_by_name=reviewer_specs_by_name,
                    progress=progress,
                )
                if initialized is None:
                    return None
                current_reviewer = initialized
                continue
            if is_worker_death_error(error):
                failure_reason = describe_reviewer_failure_reason(error, current_reviewer.worker)
                failed_reviewer = note_reviewer_failure(current_reviewer, reason_text=failure_reason)
                reason_text = f"{reviewer_display_name} 本轮执行失败或已死亡。\n{failure_reason or str(error)}".strip()
                decision = _prompt_overall_reviewer_recovery(
                    failed_reviewer,
                    reason_text=reason_text,
                    paths=paths,
                    progress=progress,
                )
                if decision == AGENT_INTERVENTION_RECHECK:
                    current_reviewer = failed_reviewer
                    continue
                if decision == AGENT_INTERVENTION_WORKER_DEAD:
                    _kill_overall_reviewer_best_effort(failed_reviewer)
                    message(f"{reviewer_display_name} 已按死亡处理，当前阶段将忽略该审核智能体。")
                    return None
                if decision != AGENT_INTERVENTION_RECREATE:
                    continue
                force_model_change = reviewer_requires_manual_model_reconfiguration(failed_reviewer)
                if force_model_change:
                    reason_text = build_reviewer_failure_reconfiguration_reason(
                        failed_reviewer,
                        role_label=reviewer_display_name,
                        failure_reason=failure_reason,
                    )
                    mark_worker_awaiting_reconfiguration(current_reviewer.worker, reason_text=reason_text)
                replacement = _recreate_overall_reviewer_from_hitl(
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    reviewer=failed_reviewer,
                    reviewer_spec=reviewer_spec,
                    progress=progress,
                    force_model_change=force_model_change,
                    required_reconfiguration=force_model_change,
                    reason_text=reason_text,
                    carry_failure_from=failed_reviewer,
                )
                if replacement is None:
                    current_reviewer = failed_reviewer
                    continue
                initialized = _run_single_overall_review_reviewer_init(
                    replacement,
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    paths=paths,
                    reviewer_specs_by_name=reviewer_specs_by_name,
                    progress=progress,
                )
                if initialized is None:
                    return None
                current_reviewer = initialized
                continue
            if try_resume_worker(current_reviewer.worker, timeout_sec=60.0):
                continue
            decision = _prompt_overall_reviewer_recovery(
                current_reviewer,
                reason_text=f"{reviewer_display_name} 无法继续执行当前复核任务。",
                paths=paths,
                progress=progress,
            )
            if decision == AGENT_INTERVENTION_RECHECK:
                continue
            if decision == AGENT_INTERVENTION_WORKER_DEAD:
                _kill_overall_reviewer_best_effort(current_reviewer)
                message(f"{reviewer_display_name} 已按死亡处理，当前阶段将忽略该审核智能体。")
                return None
            if decision != AGENT_INTERVENTION_RECREATE:
                continue
            replacement = _recreate_overall_reviewer_from_hitl(
                project_dir=project_dir,
                requirement_name=requirement_name,
                reviewer=current_reviewer,
                reviewer_spec=reviewer_spec,
                progress=progress,
                force_model_change=False,
            )
            if replacement is None:
                continue
            initialized = _run_single_overall_review_reviewer_init(
                replacement,
                project_dir=project_dir,
                requirement_name=requirement_name,
                paths=paths,
                reviewer_specs_by_name=reviewer_specs_by_name,
                progress=progress,
            )
            if initialized is None:
                return None
            current_reviewer = initialized


def _run_parallel_overall_reviewers(
    reviewers: Sequence[ReviewerRuntime],
    *,
    project_dir: str | Path,
    requirement_name: str,
    paths: dict[str, Path],
    reviewer_specs_by_name: dict[str, DevelopmentReviewerSpec],
    label_prefix: str,
    prompt_builder,
    round_index: int,
    progress: ReviewStageProgress | None = None,
) -> list[ReviewerRuntime]:
    if progress is not None:
        progress.set_phase(f"复核阶段 / 第 {round_index} 轮审核")
    return run_parallel_reviewer_round(
        reviewers,
        key_func=lambda reviewer: reviewer.reviewer_name,
        run_turn=lambda reviewer: run_overall_review_turn_with_recreation(
            reviewer,
            project_dir=project_dir,
            requirement_name=requirement_name,
            reviewer_spec=reviewer_specs_by_name[reviewer.reviewer_name],
            paths=paths,
            reviewer_specs_by_name=reviewer_specs_by_name,
            label=f"{label_prefix}_{sanitize_requirement_name(reviewer.reviewer_name)}_round_{round_index}",
            prompt=prompt_builder(reviewer),
            progress=progress,
        ),
        error_prefix="复核阶段审核智能体执行失败:",
    )


def repair_overall_review_outputs(
    reviewers: Sequence[ReviewerRuntime],
    *,
    project_dir: str | Path,
    requirement_name: str,
    paths: dict[str, Path],
    reviewer_specs_by_name: dict[str, DevelopmentReviewerSpec],
    round_index: int,
    progress: ReviewStageProgress | None = None,
) -> list[ReviewerRuntime]:
    safe_name = sanitize_requirement_name(requirement_name)
    json_pattern = f"{safe_name}_整体复核记录_*.json"
    md_pattern = f"{safe_name}_整体代码复核记录_*.md"
    return repair_reviewer_round_outputs(
        reviewers,
        key_func=lambda reviewer: reviewer.reviewer_name,
        artifact_name_func=lambda reviewer: str(reviewer.worker.session_name or reviewer.reviewer_name).strip() or reviewer.reviewer_name,
        check_job=lambda reviewer_names: check_reviewer_job(
            reviewer_names,
            directory=project_dir,
            task_name=OVERALL_REVIEW_TASK_NAME,
            json_pattern=json_pattern,
            md_pattern=md_pattern,
        ),
        run_fix_turn=lambda reviewer, fix_prompt, repair_attempt: run_overall_review_turn_with_recreation(
            reviewer,
            project_dir=project_dir,
            requirement_name=requirement_name,
            reviewer_spec=reviewer_specs_by_name[reviewer.reviewer_name],
            paths=paths,
            reviewer_specs_by_name=reviewer_specs_by_name,
            label=f"overall_review_fix_{sanitize_requirement_name(reviewer.reviewer_name)}_round_{round_index}_attempt_{repair_attempt}",
            prompt=fix_prompt,
            progress=progress,
        ),
        max_attempts=MAX_REVIEWER_REPAIR_ATTEMPTS,
        error_prefix="复核阶段审核智能体修复输出失败:",
        final_error="复核阶段审核智能体多次修复后仍未按协议更新文档",
        stage_label=OVERALL_REVIEW_TASK_NAME,
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


def _shutdown_workers(
    developer: DeveloperRuntime | None,
    reviewers: Sequence[ReviewerRuntime],
    *,
    project_dir: str | Path,
    requirement_name: str,
    cleanup_runtime: bool,
) -> tuple[str, ...]:
    return shutdown_stage_workers(
        developer,
        reviewers,
        cleanup_runtime=cleanup_runtime,
        runtime_root_filter=build_development_runtime_root(project_dir, requirement_name),
    )


def run_overall_review_stage(
    argv: Sequence[str] | None = None,
    *,
    developer_handoff: DevelopmentAgentHandoff | None = None,
    reviewer_handoff: Sequence[ReviewAgentHandoff] = (),
) -> OverallReviewStageResult:
    parser = build_parser()
    args = parser.parse_args(argv)
    allow_previous_stage_back = bool(getattr(args, "allow_previous_stage_back", False))
    project_dir = str(Path(args.project_dir).expanduser().resolve()) if args.project_dir else prompt_project_dir("")
    requirement_name = str(args.requirement_name).strip() if args.requirement_name else prompt_requirement_name_selection(project_dir, "").requirement_name

    lock_context = requirement_concurrency_lock(
        project_dir,
        requirement_name,
        action="stage.a08.start",
    )
    lock_context.__enter__()
    progress = ReviewStageProgress(initial_phase="复核阶段准备中")
    developer: DeveloperRuntime | None = None
    reviewers: list[ReviewerRuntime] = []
    cleanup_paths: list[str] = []
    audit_context: StageAuditRunContext | None = None
    try:
        audit_context = begin_stage_audit_run(
            project_dir,
            requirement_name,
            "A08",
            metadata={
                "trigger": "run_overall_review_stage",
                "argv": list(argv or []),
                "args": vars(args),
            },
        )
        paths = ensure_overall_review_inputs(project_dir=project_dir, requirement_name=requirement_name)
        active_code_context = _build_overall_review_active_code_context(project_dir)
        explicit_live_developer_handoff = developer_handoff if _is_live_developer_handoff(developer_handoff) else None
        explicit_live_reviewer_handoffs = tuple(item for item in reviewer_handoff if _is_live_reviewer_handoff(item))
        discovered_developer_handoff, discovered_reviewer_handoffs = discover_live_development_handoffs(
            project_dir=project_dir,
            requirement_name=requirement_name,
        )
        live_developer_handoff = _prefer_handoff(explicit_live_developer_handoff, discovered_developer_handoff)
        live_reviewer_handoffs = _merge_preferred_reviewer_handoffs(
            explicit_live_reviewer_handoffs,
            tuple(item for item in discovered_reviewer_handoffs if _is_live_reviewer_handoff(item)),
        )
        if live_developer_handoff is not None or live_reviewer_handoffs:
            message("检测到当前需求下仍存活的开发/复核智能体，将直接复用进入复核阶段")
            source_summary = _render_live_handoff_source_summary(live_developer_handoff, live_reviewer_handoffs)
            if source_summary:
                message(source_summary)
        effective_reviewer_handoffs = live_reviewer_handoffs or tuple(reviewer_handoff)

        developer_plan: DeveloperPlan | None = None
        if live_developer_handoff is not None:
            developer = bind_developer_runtime_from_handoff(
                project_dir=project_dir,
                requirement_name=requirement_name,
                handoff=live_developer_handoff,
            )
        else:
            developer_plan_prompted = stdin_is_interactive() and (
                not str(getattr(args, "developer_role_prompt", "") or "").strip()
                or not any(str(getattr(args, key, "") or "").strip() for key in ("vendor", "model", "effort", "proxy_url"))
            )
            developer_plan_allow_back, allow_previous_stage_back = _consume_stage_back(
                allow_previous_stage_back,
                developer_plan_prompted,
            )
            developer_plan = resolve_developer_plan(
                args,
                project_dir=project_dir,
                progress=progress,
                allow_back_first_prompt=developer_plan_allow_back,
            )

        reviewer_specs_prompted = stdin_is_interactive() and not effective_reviewer_handoffs and not any(
            str(item).strip() for item in [*getattr(args, "reviewer_role", []), *getattr(args, "reviewer_role_prompt", [])]
        ) and not resolve_stage_agent_config(args).reviewer_order
        reviewer_specs_allow_back, allow_previous_stage_back = _consume_stage_back(
            allow_previous_stage_back,
            reviewer_specs_prompted,
        )
        reviewer_specs = resolve_overall_review_reviewer_specs(
            args,
            reviewer_handoff=effective_reviewer_handoffs,
            progress=progress,
            allow_back_first_prompt=reviewer_specs_allow_back,
        )
        reviewer_specs_by_name = {str(item.reviewer_key or item.role_name).strip(): item for item in reviewer_specs}
        agent_config = resolve_stage_agent_config(args)
        reviewer_selections_by_name = dict(agent_config.reviewers or {})
        live_reviewer_keys = {str(item.reviewer_key).strip() for item in live_reviewer_handoffs if str(item.reviewer_key).strip()}
        missing_reviewer_specs = [
            item for item in reviewer_specs
            if str(item.reviewer_key or item.role_name).strip()
            and str(item.reviewer_key or item.role_name).strip() not in live_reviewer_keys
            and str(item.reviewer_key or item.role_name).strip() not in reviewer_selections_by_name
        ]
        if missing_reviewer_specs:
            reviewer_selection_allow_back, allow_previous_stage_back = _consume_stage_back(
                allow_previous_stage_back,
                stdin_is_interactive() and not bool(reviewer_selections_by_name),
            )
            reviewer_selections_by_name.update(
                collect_reviewer_agent_selections(
                    project_dir=project_dir,
                    reviewer_specs=missing_reviewer_specs,
                    display_name_resolver=lambda current_project_dir, reviewer_spec, occupied_session_names: _predict_worker_display_name(
                        project_dir=current_project_dir,
                        worker_id=build_development_reviewer_worker_id(getattr(reviewer_spec, "role_name", "")),
                        occupied_session_names=occupied_session_names,
                    ),
                    progress=progress,
                    reserved_session_names=(
                        _developer_display_name(project_dir=project_dir),
                        *(str(item.worker.session_name or "").strip() for item in live_reviewer_handoffs if str(item.worker.session_name or "").strip()),
                    ),
                    allow_back_first_prompt=reviewer_selection_allow_back,
                    stage_key="overall_review_reviewer_selection",
                )
            )
        review_round_allow_back, allow_previous_stage_back = _consume_stage_back(
            allow_previous_stage_back,
            stdin_is_interactive() and not str(getattr(args, "review_max_rounds", "") or "").strip(),
        )
        review_round_policy = ReviewRoundPolicy(
            resolve_overall_review_max_rounds(
                args,
                progress=progress,
                allow_back=review_round_allow_back,
            )
        )
        if live_developer_handoff is None and not live_reviewer_handoffs:
            cleanup_paths.extend(cleanup_runtime_dirs_by_scope(
                runtime_root=Path(project_dir).expanduser().resolve() / DEVELOPMENT_RUNTIME_ROOT_NAME,
                project_dir=project_dir,
                requirement_name=requirement_name,
                workflow_action="stage.a07.start",
            ))
            cleanup_paths.extend(cleanup_stale_overall_review_runtime_state(project_dir, requirement_name))
        cleanup_paths.extend(cleanup_existing_overall_review_artifacts(paths, requirement_name, audit_context))
        _write_overall_review_state(paths["state_path"], passed=False)
        append_stage_audit_record(
            audit_context,
            event_type="developer_output",
            source_paths={"developer_output": paths["developer_output_path"]},
            metadata={"trigger": "overall_review_initial_input"},
        )
        reviewers = build_reviewer_workers(
            args,
            project_dir=project_dir,
            requirement_name=requirement_name,
            reviewer_specs=reviewer_specs,
            reviewer_handoff=effective_reviewer_handoffs,
            reviewer_selections_by_name=reviewer_selections_by_name,
            progress=progress,
        )
        reviewers = initialize_overall_review_reviewers(
            reviewers,
            project_dir=project_dir,
            requirement_name=requirement_name,
            paths=paths,
            reviewer_specs_by_name=reviewer_specs_by_name,
            progress=progress,
            active_code_context=active_code_context,
        )

        developer_initialized = False
        round_index = 1
        previous_review_msg = ""
        code_change_msg = ""
        while True:
            prepare_review_round_artifacts(
                paths,
                reviewers,
                audit_context=audit_context,
                review_round_index=round_index,
            )
            if not previous_review_msg:
                reviewers = _run_parallel_overall_reviewers(
                    reviewers,
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    paths=paths,
                    reviewer_specs_by_name=reviewer_specs_by_name,
                    label_prefix="overall_review_initial",
                    prompt_builder=lambda reviewer: review_all_code(
                            reviewer_specs_by_name[reviewer.reviewer_name].role_prompt,
                            OVERALL_REVIEW_TASK_NAME,
                            hitl_record_md=str(paths["hitl_record_path"].resolve()),
                            requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
                            original_requirement_md=str(paths["original_requirement_path"].resolve()),
                            detailed_design_md=str(paths["detailed_design_path"].resolve()),
                            task_split_md=str(paths["task_md_path"].resolve()),
                            review_md=str(reviewer.review_md_path.resolve()),
                            review_json=str(reviewer.review_json_path.resolve()),
                        )
                        + active_code_context,
                    round_index=round_index,
                    progress=progress,
                )
            else:
                reviewers = _run_parallel_overall_reviewers(
                    reviewers,
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    paths=paths,
                    reviewer_specs_by_name=reviewer_specs_by_name,
                    label_prefix="overall_review_again",
                    prompt_builder=lambda reviewer: review_all_code_again(
                            previous_review_msg,
                            code_change_msg,
                            OVERALL_REVIEW_TASK_NAME,
                            hitl_record_md=str(paths["hitl_record_path"].resolve()),
                            requirements_clear_md=str(paths["requirements_clear_path"].resolve()),
                            original_requirement_md=str(paths["original_requirement_path"].resolve()),
                            detailed_design_md=str(paths["detailed_design_path"].resolve()),
                            review_md=str(reviewer.review_md_path.resolve()),
                            review_json=str(reviewer.review_json_path.resolve()),
                        )
                        + active_code_context,
                    round_index=round_index,
                    progress=progress,
                )
            reviewers = repair_overall_review_outputs(
                reviewers,
                project_dir=project_dir,
                requirement_name=requirement_name,
                paths=paths,
                reviewer_specs_by_name=reviewer_specs_by_name,
                round_index=round_index,
                progress=progress,
            )
            review_round_policy.record_review_attempt()
            review_json_files, review_md_files = _active_reviewer_files(reviewers)
            passed = task_done(
                directory=project_dir,
                file_path=paths["task_json_path"],
                task_name=OVERALL_REVIEW_TASK_NAME,
                json_pattern=f"{sanitize_requirement_name(requirement_name)}_整体复核记录_*.json",
                md_pattern=f"{sanitize_requirement_name(requirement_name)}_整体代码复核记录_*.md",
                md_output_name=paths["merged_review_path"].name,
                json_files=review_json_files,
                md_files=review_md_files,
            )
            append_stage_audit_record(
                audit_context,
                event_type="overall_review_merged",
                source_paths={"overall_review": paths["merged_review_path"]},
                reviewer_markdown_paths=review_md_files,
                reviewer_json_paths=review_json_files,
                review_round_index=round_index,
                metadata=_reviewer_audit_metadata(reviewers, trigger="task_done"),
            )
            if passed:
                _write_overall_review_state(paths["state_path"], passed=True)
                break
            previous_review_msg = get_markdown_content(paths["merged_review_path"]).strip()
            if not previous_review_msg:
                raise RuntimeError(f"{OVERALL_REVIEW_TASK_NAME} 未通过，但《{paths['merged_review_path'].name}》为空")
            if _review_text_is_only_nonblocking_ambiguities(previous_review_msg):
                message("整体复核仅剩非阻断歧义，已保留复核记录并继续完成。")
                _write_overall_review_state(paths["state_path"], passed=True)
                break
            if review_round_policy.should_escalate_before_next_review():
                if review_round_policy.max_rounds is None:
                    raise RuntimeError("review_round_policy 配置错误：无限轮次不应触发整体复核超限")
                raise RuntimeError(
                    f"整体复核超过最大审核轮次 {review_round_policy.max_rounds}，仍未通过:\n"
                    f"{previous_review_msg}"
                )
            if developer is None:
                if developer_plan is None:
                    developer_plan = resolve_developer_plan(args, project_dir=project_dir, progress=progress)
                developer = create_developer_runtime(
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    selection=developer_plan.selection,
                    role_prompt=developer_plan.role_prompt,
                )
                _set_worker_stage_metadata(developer.worker, project_dir=project_dir, requirement_name=requirement_name)
            if not developer_initialized:
                developer = initialize_overall_review_developer(
                    developer,
                    project_dir=project_dir,
                    requirement_name=requirement_name,
                    paths=paths,
                    progress=progress,
                    active_code_context=active_code_context,
                )
                developer_initialized = True
            developer, code_change_msg = refine_overall_review_code(
                developer,
                project_dir=project_dir,
                requirement_name=requirement_name,
                paths=paths,
                review_msg=previous_review_msg,
                progress=progress,
                audit_context=audit_context,
                review_round_index=round_index,
            )
            round_index += 1

        cleanup_paths.extend(
            _shutdown_workers(
                developer,
                reviewers,
                project_dir=project_dir,
                requirement_name=requirement_name,
                cleanup_runtime=True,
            )
        )
        append_stage_audit_record(
            audit_context,
            event_type="stage_passed",
            source_paths={
                "overall_review": paths["merged_review_path"],
                "developer_output": paths["developer_output_path"],
                "state": paths["state_path"],
            },
        )
        return OverallReviewStageResult(
            project_dir=project_dir,
            requirement_name=requirement_name,
            merged_review_path=str(paths["merged_review_path"].resolve()),
            developer_output_path=str(paths["developer_output_path"].resolve()),
            state_path=str(paths["state_path"].resolve()),
            completed=True,
            cleanup_paths=tuple(dict.fromkeys(cleanup_paths)),
        )
    except Exception as error:
        failed_paths = paths if "paths" in locals() else build_overall_review_paths(project_dir, requirement_name)
        _write_overall_review_state(failed_paths["state_path"], passed=False)
        append_stage_audit_record(
            audit_context,
            event_type="stage_failed",
            source_paths={
                "overall_review": failed_paths["merged_review_path"],
                "developer_output": failed_paths["developer_output_path"],
                "state": failed_paths["state_path"],
            },
            metadata={"error": str(error)},
        )
        _shutdown_workers(
            developer,
            reviewers,
            project_dir=project_dir,
            requirement_name=requirement_name,
            cleanup_runtime=False,
        )
        raise
    finally:
        progress.stop()
        lock_context.__exit__(None, None, None)


def main(argv: Sequence[str] | None = None) -> int:
    redirected, launch = maybe_launch_tui(argv, route="overall-review", action="stage.a08.start")
    if redirected:
        return int(launch)
    try:
        result = run_overall_review_stage(list(launch))
    except Exception as error:  # noqa: BLE001
        message(error)
        return 1

    message("复核阶段完成")
    message(result.merged_review_path)
    message(result.state_path)
    message(PLACEHOLDER_NEXT_STEP)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        cleaned_sessions = cleanup_registered_tmux_workers(reason="keyboard_interrupt")
        if cleaned_sessions and not isinstance(get_terminal_ui(), BridgeTerminalUI):
            message(f"\n已清理 tmux 会话: {', '.join(cleaned_sessions)}")
        raise SystemExit(130)
