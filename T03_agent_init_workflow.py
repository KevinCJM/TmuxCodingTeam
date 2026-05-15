# -*- encoding: utf-8 -*-
"""
@File: T03_agent_init_workflow.py
@Modify Time: 2026/4/12
@Author: Kevin-Chen
@Descriptions: AGENT初始化阶段的目录解析、运行时持久化与路由层初始化工作流
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from Prompt_01_RoutingLayerPlanning import (
    build_audit_prompt as build_stage_audit_prompt,
    build_create_prompt as build_stage_create_prompt,
    build_refine_prompt as build_stage_refine_prompt,
)
from T02_tmux_agents import (
    AgentRuntimeState,
    AgentRunConfig,
    CommandResult,
    TurnFileContract,
    TurnFileResult,
    TmuxBatchWorker,
    TmuxRuntimeController,
    clean_ansi,
    is_runtime_noise_line,
    worker_state_is_prelaunch_active,
)


ROUTING_LAYER_REQUIRED_FILES = (
    "AGENTS.md",
    "docs/repo_map.json",
    "docs/task_routes.json",
    "docs/pitfalls.json",
)
ROUTING_AUDIT_RECORD_FILE = "路由层审核记录.md"
ROUTING_AUDIT_STATUS_FILE = "audit.json"
ROUTING_AUDIT_PASS_TOKEN = "[[ROUTING_AUDIT:PASS]]"
ROUTING_AUDIT_REVISE_TOKEN = "[[ROUTING_AUDIT:REVISE]]"
ROUTING_AUDIT_STATUS_PASS = "审核通过"
ROUTING_AUDIT_STATUS_FAIL = "审核未通过"
TURN_STATUS_FILE = "turn_status.json"
TURN_STATUS_SCHEMA_VERSION = "1.0"
PHASE_ROUTING_LAYER_CREATE = "routing_layer_create"
PHASE_ROUTING_LAYER_AUDIT = "routing_layer_audit"
PHASE_ROUTING_LAYER_REFINE = "routing_layer_refine"
ROUTING_RUNTIME_ROOT_NAME = ".routing_init_runtime"
RUN_MANIFEST_VERSION = 1
FINAL_RESULT_STATUSES = {"passed", "failed", "skipped", "stale_failed"}
ACTIVE_ROUTING_WORKFLOW_STAGES = {"create_running", "audit_running", "refine_running"}
PRELAUNCH_AGENT_STATES = {"", AgentRuntimeState.STARTING.value, AgentRuntimeState.DEAD.value}
BUSINESS_FILE_IGNORED_DIR_NAMES = {
    ".agent_init_runtime",
    ".benchmarks",
    ".cache",
    ".codex_tmp",
    ".detailed_design_runtime",
    ".development_runtime",
    ".eggs",
    ".git",
    ".hypothesis",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".requirements_analysis_runtime",
    ".requirements_clarification_runtime",
    ".requirements_intake_runtime",
    ".requirements_review_runtime",
    ".routing_init_runtime",
    ".ruff_cache",
    ".tmp",
    ".tmux_stage_locks",
    ".tmux_workflow",
    ".tox",
    ".venv",
    "_tui_e2e_projects",
    "__pycache__",
    "codex-disk-incident",
    "ENV",
    "env",
    "node_modules",
    "venv",
}
BUSINESS_FILE_IGNORED_FILE_NAMES = {
    ".DS_Store",
    ".editorconfig",
    ".gitattributes",
    ".gitignore",
}


def _now_iso() -> str:
    return __import__("time").strftime("%Y-%m-%dT%H:%M:%S")


def resolve_existing_directory(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"目录不存在: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"不是目录: {path}")
    return path


def resolve_target_path(project_dir: Path, target: str | Path) -> Path:
    candidate = Path(target).expanduser()
    if not candidate.is_absolute():
        candidate = project_dir / candidate
    return resolve_existing_directory(candidate)


def required_routing_layer_paths(work_dir: str | Path) -> list[Path]:
    root = resolve_existing_directory(work_dir)
    return [root / relative_path for relative_path in ROUTING_LAYER_REQUIRED_FILES]


def build_routing_runtime_root(project_dir: str | Path) -> Path:
    return resolve_existing_directory(project_dir) / ROUTING_RUNTIME_ROOT_NAME


def resolve_routing_runtime_root(
    *,
    project_dir: str | Path | None = None,
    runtime_root: str | Path | None = None,
) -> Path:
    if runtime_root is not None:
        return Path(runtime_root).expanduser().resolve()
    if project_dir is None or not str(project_dir).strip():
        raise ValueError("当前只支持恢复当前项目内的 routing run；请传入 project_dir 或 runtime_root。")
    return build_routing_runtime_root(project_dir)


def list_routing_run_manifest_paths(
    *,
    project_dir: str | Path | None = None,
    runtime_root: str | Path | None = None,
) -> list[Path]:
    root = resolve_routing_runtime_root(project_dir=project_dir, runtime_root=runtime_root)
    if not root.exists() or not root.is_dir():
        return []
    return sorted(root.glob("run_*/manifest.json"), key=lambda item: item.stat().st_mtime, reverse=True)


def missing_routing_layer_files(work_dir: str | Path) -> list[str]:
    root = resolve_existing_directory(work_dir)
    missing: list[str] = []
    for file_path in required_routing_layer_paths(root):
        relative_path = str(file_path.relative_to(root))
        if not file_path.exists():
            missing.append(relative_path)
            continue
        if not file_path.is_file():
            missing.append(relative_path)
            continue
        if file_path.stat().st_size == 0:
            missing.append(relative_path)
    return missing


def project_has_business_files(work_dir: str | Path) -> bool:
    root = resolve_existing_directory(work_dir)

    def visit(directory: Path) -> bool:
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError:
            return True
        for entry in entries:
            name = entry.name
            if entry.is_symlink():
                if name in BUSINESS_FILE_IGNORED_DIR_NAMES or name in BUSINESS_FILE_IGNORED_FILE_NAMES:
                    continue
                return True
            if entry.is_dir():
                if name in BUSINESS_FILE_IGNORED_DIR_NAMES:
                    continue
                if visit(entry):
                    return True
                continue
            if entry.is_file():
                if name in BUSINESS_FILE_IGNORED_FILE_NAMES:
                    continue
                return True
            if name not in BUSINESS_FILE_IGNORED_DIR_NAMES and name not in BUSINESS_FILE_IGNORED_FILE_NAMES:
                return True
        return False

    return visit(root)


def should_skip_routing_init_for_empty_project(
        work_dir: str | Path,
        *,
        project_missing_files: Sequence[str] = (),
        target_dirs: Sequence[str | Path] = (),
        explicit_run_init_yes: bool = False,
) -> bool:
    return (
        bool(project_missing_files)
        and not tuple(target_dirs or ())
        and not explicit_run_init_yes
        and not project_has_business_files(work_dir)
    )


def _load_routing_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path.relative_to(path.parents[1])} JSON 非法: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path.relative_to(path.parents[1])} 根节点必须是 object")
    return payload


def routing_layer_artifact_errors(work_dir: str | Path) -> list[str]:
    root = resolve_existing_directory(work_dir)
    missing = missing_routing_layer_files(root)
    if missing:
        return [f"缺少路由层文件: {item}" for item in missing]

    repo_map_path = root / "docs" / "repo_map.json"
    task_routes_path = root / "docs" / "task_routes.json"
    pitfalls_path = root / "docs" / "pitfalls.json"
    errors: list[str] = []
    try:
        repo_map = _load_routing_json(repo_map_path)
        task_routes = _load_routing_json(task_routes_path)
        pitfalls = _load_routing_json(pitfalls_path)
    except ValueError as error:
        return [str(error)]

    modules = repo_map.get("modules")
    if not isinstance(modules, list) or not modules:
        errors.append("docs/repo_map.json 缺少非空 modules[]")
        module_ids: set[str] = set()
    else:
        module_ids = {
            str(item.get("id", "")).strip()
            for item in modules
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        }
        if not module_ids:
            errors.append("docs/repo_map.json modules[] 缺少有效 id")

    routes = task_routes.get("routes")
    if not isinstance(routes, list) or not routes:
        errors.append("docs/task_routes.json 缺少非空 routes[]")
        routes = []

    pitfall_items = pitfalls.get("pitfalls")
    if not isinstance(pitfall_items, list):
        errors.append("docs/pitfalls.json 缺少 pitfalls[]")
        pitfall_ids: set[str] = set()
    else:
        pitfall_ids = {
            str(item.get("id", "")).strip()
            for item in pitfall_items
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        }

    for route in routes:
        if not isinstance(route, dict):
            errors.append("docs/task_routes.json routes[] 项必须是 object")
            continue
        route_id = str(route.get("id", "")).strip() or "unknown_route"
        for module_id in route.get("first_read_modules", ()) or ():
            module_id_text = str(module_id).strip()
            if module_id_text and module_id_text not in module_ids:
                errors.append(f"route {route_id} first_read_modules 引用不存在模块: {module_id_text}")
        for pitfall_id in route.get("pitfall_ids", ()) or ():
            pitfall_id_text = str(pitfall_id).strip()
            if pitfall_id_text and pitfall_id_text not in pitfall_ids:
                errors.append(f"route {route_id} pitfall_ids 引用不存在风险: {pitfall_id_text}")
    return errors


def validate_routing_layer_artifacts(work_dir: str | Path) -> None:
    errors = routing_layer_artifact_errors(work_dir)
    if errors:
        raise ValueError("; ".join(errors))


def routing_layer_readiness_issues(work_dir: str | Path) -> list[str]:
    return routing_layer_artifact_errors(work_dir)


def has_complete_routing_layer(work_dir: str | Path) -> bool:
    try:
        validate_routing_layer_artifacts(work_dir)
    except Exception:
        return False
    return True


@dataclass(frozen=True)
class TargetSelection:
    project_dir: str
    selected_dirs: tuple[str, ...]
    skipped_dirs: tuple[str, ...]
    forced_dirs: tuple[str, ...]
    project_missing_files: tuple[str, ...]

    @property
    def should_run(self) -> bool:
        return bool(self.selected_dirs)

    @property
    def project_is_forced(self) -> bool:
        return self.project_dir in self.forced_dirs


def resolve_target_selection(
    *,
    project_dir: str | Path,
    target_dirs: Sequence[str | Path] = (),
    run_init: bool = True,
) -> TargetSelection:
    project_root = resolve_existing_directory(project_dir)
    ordered_candidates: list[Path] = []
    seen: set[str] = set()

    def add_candidate(path_value: Path) -> None:
        key = str(path_value)
        if key in seen:
            return
        seen.add(key)
        ordered_candidates.append(path_value)

    add_candidate(project_root)
    for target in target_dirs:
        add_candidate(resolve_target_path(project_root, target))

    project_missing = tuple(routing_layer_readiness_issues(project_root))
    skipped_dirs: list[str] = []
    selected_dirs: list[str] = []
    forced_dirs: list[str] = []

    if not run_init:
        if project_missing and project_has_business_files(project_root):
            selected_dirs = [str(project_root)]
            forced_dirs = [str(project_root)]
            skipped_dirs = [
                str(candidate)
                for candidate in ordered_candidates
                if candidate != project_root
            ]
        else:
            skipped_dirs = [str(candidate) for candidate in ordered_candidates]
    else:
        selected_dirs = [str(candidate) for candidate in ordered_candidates]
        if project_missing:
            forced_dirs.append(str(project_root))

    return TargetSelection(
        project_dir=str(project_root),
        selected_dirs=tuple(selected_dirs),
        skipped_dirs=tuple(skipped_dirs),
        forced_dirs=tuple(forced_dirs),
        project_missing_files=project_missing,
    )


def build_create_prompt() -> str:
    return build_stage_create_prompt()


def routing_audit_record_path(work_dir: str | Path) -> Path:
    return resolve_existing_directory(work_dir) / ROUTING_AUDIT_RECORD_FILE


def routing_audit_status_path(work_dir: str | Path) -> Path:
    return resolve_existing_directory(work_dir) / ROUTING_AUDIT_STATUS_FILE


def routing_turns_root(runtime_dir: str | Path) -> Path:
    root = Path(runtime_dir).expanduser().resolve() / "turns"
    root.mkdir(parents=True, exist_ok=True)
    return root


def routing_turn_dir(runtime_dir: str | Path, turn_id: str) -> Path:
    return routing_turns_root(runtime_dir) / str(turn_id).strip()


def routing_turn_status_path(runtime_dir: str | Path, turn_id: str) -> Path:
    return routing_turn_dir(runtime_dir, turn_id) / TURN_STATUS_FILE


def reset_turn_runtime_dir(runtime_dir: str | Path, turn_id: str) -> Path:
    turn_dir = routing_turn_dir(runtime_dir, turn_id)
    if turn_dir.exists():
        shutil.rmtree(turn_dir)
    turn_dir.mkdir(parents=True, exist_ok=True)
    return turn_dir


def _relative_path_within_root(root: Path, candidate: str | Path) -> str:
    path = Path(candidate).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (root / path).resolve()
    if root not in (resolved, *resolved.parents):
        raise ValueError(f"artifact path 超出当前工作目录: {resolved}")
    return str(resolved.relative_to(root))


def _flatten_artifact_paths(root: Path, artifacts: object) -> list[str]:
    flattened: list[str] = []
    if isinstance(artifacts, dict):
        values = artifacts.values()
    elif isinstance(artifacts, (list, tuple, set)):
        values = artifacts
    else:
        values = [artifacts]
    for value in values:
        if isinstance(value, dict):
            flattened.extend(_flatten_artifact_paths(root, value))
            continue
        if isinstance(value, (list, tuple, set)):
            flattened.extend(_flatten_artifact_paths(root, value))
            continue
        if value is None:
            continue
        flattened.append(_relative_path_within_root(root, value))
    return flattened


def sha256_file(file_path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_prefixed_sha256(file_path: str | Path) -> str:
    return f"sha256:{sha256_file(file_path)}"


def _parse_iso_timestamp(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        raise ValueError("written_at 不能为空")
    datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value


@dataclass(frozen=True)
class TurnStatusDecision:
    turn_id: str
    phase: str
    status_path: str
    payload: dict[str, object]
    artifact_paths: dict[str, str]
    artifact_hashes: dict[str, str]
    written_at: str


def capture_artifact_hashes(
    work_dir: str | Path,
    required_artifacts: Sequence[str],
) -> dict[str, str]:
    root = resolve_existing_directory(work_dir)
    artifact_hashes: dict[str, str] = {}
    for relative_path in sorted({str(item) for item in required_artifacts}):
        file_path = (root / relative_path).resolve()
        if not file_path.exists() or not file_path.is_file():
            continue
        artifact_hashes[relative_path] = build_prefixed_sha256(file_path)
    return artifact_hashes


def _has_artifact_change(
    current_hashes: dict[str, str],
    baseline_hashes: dict[str, str] | None,
) -> bool:
    baseline = dict(baseline_hashes or {})
    if not baseline:
        return True
    for relative_path, current_hash in current_hashes.items():
        if baseline.get(relative_path) != current_hash:
            return True
    return False


def _validate_turn_status_file(
    status_path: str | Path,
    *,
    expected_turn_id: str,
    expected_phase: str,
    work_dir: str | Path,
    required_artifacts: Sequence[str],
) -> TurnStatusDecision:
    path = Path(status_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"缺少 turn status 文件: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version != TURN_STATUS_SCHEMA_VERSION:
        raise ValueError(f"turn_status schema_version 非法: {schema_version!r}")
    turn_id = str(payload.get("turn_id", "")).strip()
    if turn_id != expected_turn_id:
        raise ValueError(f"turn_id 非预期: {turn_id!r} != {expected_turn_id!r}")
    phase = str(payload.get("phase", "")).strip()
    if phase != expected_phase:
        raise ValueError(f"phase 非预期: {phase!r} != {expected_phase!r}")
    status = str(payload.get("status", "")).strip().lower()
    if status != "done":
        raise ValueError(f"turn_status.status 非法: {status!r}")
    written_at = _parse_iso_timestamp(payload.get("written_at", ""))
    root = resolve_existing_directory(work_dir)
    raw_artifacts = payload.get("artifacts", {})
    if not isinstance(raw_artifacts, dict):
        raise ValueError("turn_status.artifacts 必须是对象")
    normalized_paths = _flatten_artifact_paths(root, raw_artifacts)
    required_set = {str(item) for item in required_artifacts}
    if not required_set.issubset(set(normalized_paths)):
        raise ValueError(f"turn_status.artifacts 未覆盖所有必需文件: {sorted(required_set - set(normalized_paths))}")
    raw_hashes = payload.get("artifact_hashes", {})
    if not isinstance(raw_hashes, dict):
        raise ValueError("turn_status.artifact_hashes 必须是对象")
    validated_paths: dict[str, str] = {}
    validated_hashes: dict[str, str] = {}
    for relative_path in sorted(required_set):
        file_path = (root / relative_path).resolve()
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"缺少必需结果文件: {file_path}")
        expected_hash = str(raw_hashes.get(relative_path, "")).strip()
        actual_hash = build_prefixed_sha256(file_path)
        if expected_hash != actual_hash:
            raise ValueError(f"artifact_hash mismatch: {relative_path} -> {expected_hash!r} != {actual_hash!r}")
        validated_paths[relative_path] = str(file_path)
        validated_hashes[relative_path] = actual_hash
    if required_set == set(ROUTING_LAYER_REQUIRED_FILES):
        validate_routing_layer_artifacts(root)
    return TurnStatusDecision(
        turn_id=turn_id,
        phase=phase,
        status_path=str(path),
        payload=payload,
        artifact_paths=validated_paths,
        artifact_hashes=validated_hashes,
        written_at=written_at,
    )


def _default_turn_status_artifacts(required_artifacts: Sequence[str]) -> dict[str, object]:
    normalized = tuple(str(item) for item in required_artifacts)
    if normalized == ROUTING_LAYER_REQUIRED_FILES:
        return {"routing_layer_files": list(normalized)}
    if normalized == (ROUTING_AUDIT_STATUS_FILE, ROUTING_AUDIT_RECORD_FILE):
        return {
            "audit_status": ROUTING_AUDIT_STATUS_FILE,
            "review_record": ROUTING_AUDIT_RECORD_FILE,
        }
    if len(normalized) == 1:
        return {"primary_artifact": normalized[0]}
    return {"artifacts": list(normalized)}


def _write_json_atomic(path: str | Path, payload: dict[str, object]) -> Path:
    target_path = Path(path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f".{target_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return target_path


def _build_turn_status_payload(
    *,
    turn_id: str,
    phase: str,
    work_dir: str | Path,
    required_artifacts: Sequence[str],
) -> dict[str, object]:
    root = resolve_existing_directory(work_dir)
    artifacts = _default_turn_status_artifacts(required_artifacts)
    artifact_paths = _flatten_artifact_paths(root, artifacts)
    artifact_hashes: dict[str, str] = {}
    for relative_path in artifact_paths:
        file_path = (root / relative_path).resolve()
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"缺少必需结果文件: {file_path}")
        artifact_hashes[relative_path] = build_prefixed_sha256(file_path)
    return {
        "schema_version": TURN_STATUS_SCHEMA_VERSION,
        "turn_id": turn_id,
        "phase": phase,
        "status": "done",
        "artifacts": artifacts,
        "artifact_hashes": artifact_hashes,
        "written_at": _now_iso(),
    }


def _materialize_turn_status_file(
    status_path: str | Path,
    *,
    turn_id: str,
    phase: str,
    work_dir: str | Path,
    required_artifacts: Sequence[str],
) -> Path:
    payload = _build_turn_status_payload(
        turn_id=turn_id,
        phase=phase,
        work_dir=work_dir,
        required_artifacts=required_artifacts,
    )
    return _write_json_atomic(status_path, payload)


def build_turn_file_contract(
    *,
    runtime_dir: str | Path,
    work_dir: str | Path,
    turn_id: str,
    phase: str,
    required_artifacts: Sequence[str],
    quiet_window_sec: float = 1.0,
    baseline_artifact_hashes: dict[str, str] | None = None,
    require_artifact_change: bool = False,
) -> TurnFileContract:
    status_path = routing_turn_status_path(runtime_dir, turn_id)

    def validator(path: Path) -> TurnFileResult:
        validation_error: Exception | None = None
        if Path(path).expanduser().resolve().exists():
            try:
                decision = _validate_turn_status_file(
                    path,
                    expected_turn_id=turn_id,
                    expected_phase=phase,
                    work_dir=work_dir,
                    required_artifacts=required_artifacts,
                )
            except Exception as error:  # noqa: BLE001
                validation_error = error
            else:
                if require_artifact_change and not _has_artifact_change(
                    decision.artifact_hashes,
                    baseline_artifact_hashes,
                ):
                    validation_error = ValueError("required_artifact_change_missing")
                else:
                    return TurnFileResult(
                        status_path=decision.status_path,
                        payload=decision.payload,
                        artifact_paths=decision.artifact_paths,
                        artifact_hashes=decision.artifact_hashes,
                        validated_at=_now_iso(),
                    )

        try:
            materialized_path = _materialize_turn_status_file(
                path,
                turn_id=turn_id,
                phase=phase,
                work_dir=work_dir,
                required_artifacts=required_artifacts,
            )
        except Exception:
            if validation_error is not None:
                raise validation_error
            raise

        decision = _validate_turn_status_file(
            materialized_path,
            expected_turn_id=turn_id,
            expected_phase=phase,
            work_dir=work_dir,
            required_artifacts=required_artifacts,
        )
        if require_artifact_change and not _has_artifact_change(
            decision.artifact_hashes,
            baseline_artifact_hashes,
        ):
            raise ValueError("required_artifact_change_missing")
        return TurnFileResult(
            status_path=decision.status_path,
            payload=decision.payload,
            artifact_paths=decision.artifact_paths,
            artifact_hashes=decision.artifact_hashes,
            validated_at=_now_iso(),
        )

    return TurnFileContract(
        turn_id=turn_id,
        phase=phase,
        status_path=status_path,
        validator=validator,
        quiet_window_sec=quiet_window_sec,
        kind="routing_file_contract",
    )


def reset_routing_audit_artifacts(work_dir: str | Path) -> None:
    for artifact_path in (
        routing_audit_record_path(work_dir),
        routing_audit_status_path(work_dir),
    ):
        if artifact_path.exists():
            artifact_path.unlink()


@dataclass(frozen=True)
class RoutingAuditDecision:
    status: str
    record_path: str
    record_text: str
    audit_round: int
    payload: dict[str, object]


def load_routing_audit_decision(
    work_dir: str | Path,
    *,
    expected_round: int,
) -> RoutingAuditDecision:
    root = resolve_existing_directory(work_dir)
    status_path = routing_audit_status_path(root)
    if not status_path.exists():
        raise FileNotFoundError(f"缺少审核状态文件: {status_path}")
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version != "1.0":
        raise ValueError(f"audit.json schema_version 非法: {schema_version!r}")
    stage = str(payload.get("stage", "")).strip()
    if stage != PHASE_ROUTING_LAYER_AUDIT:
        raise ValueError(f"audit.json stage 非法: {stage!r}")
    status = str(payload.get("status", "")).strip()
    if status not in {ROUTING_AUDIT_STATUS_PASS, ROUTING_AUDIT_STATUS_FAIL}:
        raise ValueError(f"audit.json status 非法: {status!r}")
    audit_round = int(payload.get("audit_round", 0))
    if audit_round != int(expected_round):
        raise ValueError(f"audit.json audit_round 非预期: {audit_round} != {expected_round}")
    review_record_path = str(payload.get("review_record_path", ROUTING_AUDIT_RECORD_FILE)).strip() or ROUTING_AUDIT_RECORD_FILE
    record_path = (root / review_record_path).resolve() if not Path(review_record_path).is_absolute() else Path(review_record_path).resolve()
    if root not in (record_path, *record_path.parents):
        raise ValueError(f"review_record_path 超出当前工作目录: {record_path}")
    if not record_path.exists() or not record_path.is_file():
        raise FileNotFoundError(f"缺少审核记录文件: {record_path}")
    record_text = record_path.read_text(encoding="utf-8").strip()
    if not record_text:
        raise ValueError("审核记录文件为空")
    return RoutingAuditDecision(
        status=status,
        record_path=str(record_path),
        record_text=record_text,
        audit_round=audit_round,
        payload=payload,
    )


def build_audit_prompt(*, audit_round: int) -> str:
    return build_stage_audit_prompt(
        audit_round=audit_round,
        routing_audit_status_file=ROUTING_AUDIT_STATUS_FILE,
        routing_audit_record_file=ROUTING_AUDIT_RECORD_FILE,
        routing_audit_status_pass=ROUTING_AUDIT_STATUS_PASS,
        routing_audit_status_fail=ROUTING_AUDIT_STATUS_FAIL,
    )


def build_refine_prompt(
    audit_record_path: str | Path,
) -> str:
    return build_stage_refine_prompt(
        audit_record_path,
        routing_audit_status_file=ROUTING_AUDIT_STATUS_FILE,
    )


def extract_protocol_token(text: str, allowed_tokens: Sequence[str]) -> str:
    lines = [clean_ansi(line).strip() for line in str(text or "").splitlines() if line.strip()]
    while lines and is_runtime_noise_line(lines[-1]):
        lines.pop()
    if not lines:
        return ""
    final_line = lines[-1]
    return final_line if final_line in set(allowed_tokens) else ""


def summarize_audit_output(audit_text: str, *, max_lines: int = 12) -> str:
    lines = [line.rstrip() for line in str(audit_text or "").splitlines() if line.strip()]
    return "\n".join(lines[:max_lines]).strip()


def normalize_audit_output(audit_text: str, audit_token: str) -> str:
    lines = [line.strip() for line in str(audit_text or "").splitlines() if line.strip()]
    if audit_token == ROUTING_AUDIT_PASS_TOKEN:
        return ROUTING_AUDIT_PASS_TOKEN
    if audit_token != ROUTING_AUDIT_REVISE_TOKEN:
        return "\n".join(lines).strip()

    normalized_lines: list[str] = []
    seen: set[str] = set()
    prefixes = (
        "- verdict:",
        "- file_role:",
        "- finding:",
        "- duplication:",
        "- boundary_conflict:",
        "- missing:",
        "- recommendation:",
        "- top_priority:",
    )
    for raw_line in lines:
        for prefix in prefixes:
            marker_index = raw_line.find(prefix)
            if marker_index < 0:
                continue
            canonical = raw_line[marker_index:].strip()
            if canonical in seen:
                break
            normalized_lines.append(canonical)
            seen.add(canonical)
            break
    if normalized_lines:
        normalized_lines.append(ROUTING_AUDIT_REVISE_TOKEN)
        return "\n".join(normalized_lines)
    return ROUTING_AUDIT_REVISE_TOKEN


def prepare_revise_audit_output(audit_text: str) -> str:
    lines = [clean_ansi(line).strip() for line in str(audit_text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    token_positions = [
        index
        for index, line in enumerate(lines)
        if line in {ROUTING_AUDIT_PASS_TOKEN, ROUTING_AUDIT_REVISE_TOKEN}
    ]
    if not token_positions:
        return ""
    token_index = token_positions[-1]
    final_token = lines[token_index]
    if final_token != ROUTING_AUDIT_REVISE_TOKEN:
        return ""
    body_lines: list[str] = []
    for raw_line in lines[:token_index]:
        if raw_line in {ROUTING_AUDIT_PASS_TOKEN, ROUTING_AUDIT_REVISE_TOKEN}:
            continue
        if is_runtime_noise_line(raw_line):
            continue
        body_lines.append(raw_line)
    if not body_lines:
        return ""
    return "\n".join(body_lines + [ROUTING_AUDIT_REVISE_TOKEN])


def audit_pass_has_extra_text(audit_text: str) -> bool:
    lines = [line.strip() for line in str(audit_text or "").splitlines() if line.strip()]
    if not lines:
        return False
    return lines != [ROUTING_AUDIT_PASS_TOKEN]


def audit_output_requires_revise(audit_text: str) -> bool:
    lines = [line.strip() for line in str(audit_text or "").splitlines() if line.strip()]
    if not lines:
        return True
    if lines == [ROUTING_AUDIT_PASS_TOKEN]:
        return False

    decision_tokens = {ROUTING_AUDIT_PASS_TOKEN, ROUTING_AUDIT_REVISE_TOKEN}
    content_lines = [line for line in lines if line not in decision_tokens]
    structured_prefixes = (
        "- verdict:",
        "- file_role:",
        "- finding:",
        "- duplication:",
        "- boundary_conflict:",
        "- missing:",
        "- recommendation:",
        "- top_priority:",
    )
    structured_lines = [line for line in content_lines if line.startswith(structured_prefixes)]
    if not structured_lines:
        return False

    verdict = ""
    recommendation = ""
    for line in structured_lines:
        if line.startswith("- verdict:"):
            verdict = line.split(":", 1)[1].strip().lower()
        elif line.startswith("- recommendation:"):
            recommendation = line.split(":", 1)[1].strip().lower()

    if verdict != "strong":
        return True
    if recommendation != "use_as_is":
        return True

    for line in structured_lines:
        if line.startswith("- finding:"):
            return True
        if line.startswith("- missing:"):
            return True
        if line.startswith("- boundary_conflict:"):
            return True
        if line.startswith("- duplication:") and "value=wasteful" in line.lower():
            return True
    return False


@dataclass
class DirectoryInitResult:
    work_dir: str
    forced: bool
    status: str
    rounds_used: int
    runtime_dir: str = ""
    session_name: str = ""
    missing_before: list[str] = field(default_factory=list)
    missing_after: list[str] = field(default_factory=list)
    failure_reason: str = ""
    last_audit_token: str = ""
    last_audit_summary: str = ""
    commands: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalize_agent_state_from_runtime_state(state: dict[str, object], fallback: str = AgentRuntimeState.DEAD.value) -> str:
    if worker_state_is_prelaunch_active(state):
        return AgentRuntimeState.STARTING.value
    candidate = str(state.get("agent_state", "")).strip().upper()
    if candidate in {item.value for item in AgentRuntimeState}:
        return candidate
    current_command = str(state.get("current_command", "")).strip()
    provider_phase = str(state.get("provider_phase", "")).strip().lower()
    wrapper_state = str(state.get("wrapper_state", "")).strip().upper()
    started = bool(state.get("agent_started", state.get("agent_ready", False)))
    if not started and provider_phase in {"waiting_input", "idle_ready", "completed_response"}:
        started = True
    alive = bool(state.get("agent_alive", False))
    if not alive:
        alive = bool(current_command) and current_command not in {"bash", "fish", "sh", "zsh"}
    if not alive:
        return AgentRuntimeState.DEAD.value
    if not started:
        return AgentRuntimeState.STARTING.value
    if wrapper_state == AgentRuntimeState.READY.value or provider_phase in {"waiting_input", "idle_ready", "completed_response"}:
        return AgentRuntimeState.READY.value
    return AgentRuntimeState.BUSY.value


def _is_active_prelaunch_state_update(
    entry: "WorkerManifestEntry",
    state: dict[str, object],
    normalized_agent_state: str,
) -> bool:
    workflow_stage = str(entry.workflow_stage or "").strip()
    result_status = str(entry.result_status or "").strip()
    payload = dict(state)
    payload.setdefault("agent_state", normalized_agent_state)
    payload.setdefault("workflow_stage", workflow_stage)
    payload.setdefault("result_status", result_status)
    payload.setdefault("status", result_status)
    payload.setdefault("pane_id", entry.pane_id)
    payload.setdefault("agent_started", entry.agent_started)
    return (
        workflow_stage in ACTIVE_ROUTING_WORKFLOW_STAGES | {"pending"}
        and normalized_agent_state in PRELAUNCH_AGENT_STATES
        and worker_state_is_prelaunch_active(payload)
    )


def _entry_has_active_routing_turn(entry: "WorkerManifestEntry", state: dict[str, object] | None = None) -> bool:
    payload = dict(state or {})
    workflow_stage = str(payload.get("workflow_stage", entry.workflow_stage) or "").strip()
    result_status = str(payload.get("result_status", entry.result_status) or "").strip()
    current_task_runtime_status = str(
        payload.get("current_task_runtime_status", entry.current_task_runtime_status) or ""
    ).strip().lower()
    current_turn_status_path = str(
        payload.get("current_turn_status_path", entry.current_turn_status_path) or ""
    ).strip()
    if workflow_stage not in ACTIVE_ROUTING_WORKFLOW_STAGES:
        return False
    if result_status not in {"running", "pending"}:
        return False
    return current_task_runtime_status == "running" or bool(current_turn_status_path)


def _normalize_active_routing_entry_display_state(entry: "WorkerManifestEntry") -> None:
    if (
        str(entry.agent_state or "").strip().upper() == AgentRuntimeState.READY.value
        and _entry_has_active_routing_turn(entry)
    ):
        entry.agent_state = AgentRuntimeState.BUSY.value


@dataclass
class WorkerManifestEntry:
    work_dir: str
    session_name: str = ""
    runtime_dir: str = ""
    pane_id: str = ""
    forced: bool = False
    workflow_stage: str = "pending"
    workflow_round: int = 0
    result_status: str = "pending"
    agent_state: str = AgentRuntimeState.DEAD.value
    agent_alive: bool = False
    agent_started: bool = False
    retry_count: int = 0
    last_turn_token: str = ""
    last_prompt_hash: str = ""
    last_heartbeat_at: str = ""
    last_log_offset: int = 0
    current_command: str = ""
    current_path: str = ""
    current_turn_id: str = ""
    current_turn_phase: str = ""
    current_turn_status_path: str = ""
    current_turn_baseline_hashes: dict[str, str] = field(default_factory=dict)
    current_task_runtime_status: str = ""
    recoverable: bool = True
    health_status: str = "unknown"
    health_note: str = ""
    note: str = ""
    failure_reason: str = ""
    last_audit_token: str = ""
    last_audit_summary: str = ""
    last_audit_output: str = ""
    missing_before: list[str] = field(default_factory=list)
    missing_after: list[str] = field(default_factory=list)
    log_path: str = ""
    raw_log_path: str = ""
    state_path: str = ""
    transcript_path: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class RunManifest:
    manifest_version: int
    run_id: str
    runtime_dir: str
    project_dir: str
    selection: dict[str, object]
    config: dict[str, str]
    status: str
    created_at: str
    updated_at: str
    workers: list[WorkerManifestEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["workers"] = [item.to_dict() for item in self.workers]
        return payload


@dataclass
class BatchInitResult:
    run_id: str
    runtime_dir: str
    selection: TargetSelection
    config: dict[str, str]
    results: list[DirectoryInitResult]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "runtime_dir": self.runtime_dir,
            "selection": asdict(self.selection),
            "config": self.config,
            "results": [item.to_dict() for item in self.results],
        }


@dataclass(frozen=True)
class RoutingCleanupResult:
    removed_intermediate_files: tuple[str, ...] = ()
    removed_runtime_dirs: tuple[str, ...] = ()

    @property
    def removed_intermediate_count(self) -> int:
        return len(self.removed_intermediate_files)

    @property
    def removed_runtime_count(self) -> int:
        return len(self.removed_runtime_dirs)


@dataclass(frozen=True)
class LiveWorkerHandle:
    work_dir: str
    forced: bool
    worker: TmuxBatchWorker
    resume_state: dict[str, object] = field(default_factory=dict)

    @property
    def session_name(self) -> str:
        return self.worker.session_name


class RunStore:
    def __init__(self, *, run_root: str | Path, manifest: RunManifest) -> None:
        self.run_root = Path(run_root).expanduser().resolve()
        self.manifest = manifest
        self.manifest_path = self.run_root / "manifest.json"
        self.events_path = self.run_root / "events.jsonl"
        self._lock = threading.Lock()

    @classmethod
    def create(
        cls,
        *,
        selection: TargetSelection,
        config: AgentRunConfig,
        runtime_root: str | Path | None = None,
        run_id: str | None = None,
    ) -> "RunStore":
        actual_run_id, run_root = create_batch_runtime(
            run_id=run_id,
            project_dir=selection.project_dir,
            runtime_root=runtime_root,
        )
        manifest = RunManifest(
            manifest_version=RUN_MANIFEST_VERSION,
            run_id=actual_run_id,
            runtime_dir=str(run_root),
            project_dir=selection.project_dir,
            selection=asdict(selection),
            config=config.to_summary(),
            status="created",
            created_at=_now_iso(),
            updated_at=_now_iso(),
            workers=[],
        )
        store = cls(run_root=run_root, manifest=manifest)
        store.write_manifest()
        store.append_event("run_created", project_dir=selection.project_dir)
        return store

    @classmethod
    def load(
        cls,
        *,
        run_id: str,
        project_dir: str | Path | None = None,
        runtime_root: str | Path | None = None,
    ) -> "RunStore":
        run_root = resolve_routing_runtime_root(project_dir=project_dir, runtime_root=runtime_root) / run_id
        manifest_path = run_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"未找到 run manifest: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        workers = [
            WorkerManifestEntry(
                work_dir=str(item.get("work_dir", "")),
                session_name=str(item.get("session_name", "")),
                runtime_dir=str(item.get("runtime_dir", "")),
                pane_id=str(item.get("pane_id", "")),
                forced=bool(item.get("forced", False)),
                workflow_stage=str(item.get("workflow_stage", "pending")),
                workflow_round=int(item.get("workflow_round", 0)),
                result_status=str(item.get("result_status", "pending")),
                agent_state=_normalize_agent_state_from_runtime_state(dict(item), str(item.get("agent_state", AgentRuntimeState.DEAD.value))),
                agent_alive=bool(item.get("agent_alive", False)),
                agent_started=bool(item.get("agent_started", item.get("agent_ready", False))),
                retry_count=int(item.get("retry_count", 0)),
                last_turn_token=str(item.get("last_turn_token", "")),
                last_prompt_hash=str(item.get("last_prompt_hash", "")),
                last_heartbeat_at=str(item.get("last_heartbeat_at", "")),
                last_log_offset=int(item.get("last_log_offset", 0)),
                current_command=str(item.get("current_command", "")),
                current_path=str(item.get("current_path", "")),
                current_turn_id=str(item.get("current_turn_id", "")),
                current_turn_phase=str(item.get("current_turn_phase", "")),
                current_turn_status_path=str(item.get("current_turn_status_path", "")),
                current_turn_baseline_hashes=dict(item.get("current_turn_baseline_hashes", {})),
                current_task_runtime_status=str(item.get("current_task_runtime_status", "")),
                recoverable=bool(item.get("recoverable", True)),
                health_status=str(item.get("health_status", "unknown")),
                health_note=str(item.get("health_note", "")),
                note=str(item.get("note", "")),
                failure_reason=str(item.get("failure_reason", "")),
                last_audit_token=str(item.get("last_audit_token", "")),
                last_audit_summary=str(item.get("last_audit_summary", "")),
                last_audit_output=str(item.get("last_audit_output", "")),
                missing_before=list(item.get("missing_before", [])),
                missing_after=list(item.get("missing_after", [])),
                log_path=str(item.get("log_path", "")),
                raw_log_path=str(item.get("raw_log_path", "")),
                state_path=str(item.get("state_path", "")),
                transcript_path=str(item.get("transcript_path", "")),
            )
            for item in payload.get("workers", [])
            if isinstance(item, dict)
        ]
        for worker in workers:
            _normalize_active_routing_entry_display_state(worker)
        manifest = RunManifest(
            manifest_version=int(payload.get("manifest_version", RUN_MANIFEST_VERSION)),
            run_id=str(payload["run_id"]),
            runtime_dir=str(payload["runtime_dir"]),
            project_dir=str(payload["project_dir"]),
            selection=dict(payload["selection"]),
            config=dict(payload["config"]),
            status=str(payload.get("status", "created")),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
            workers=workers,
        )
        return cls(run_root=run_root, manifest=manifest)

    def selection(self) -> TargetSelection:
        raw = self.manifest.selection
        return TargetSelection(
            project_dir=str(raw["project_dir"]),
            selected_dirs=tuple(raw.get("selected_dirs", ())),
            skipped_dirs=tuple(raw.get("skipped_dirs", ())),
            forced_dirs=tuple(raw.get("forced_dirs", ())),
            project_missing_files=tuple(raw.get("project_missing_files", ())),
        )

    def config_object(self) -> AgentRunConfig:
        return AgentRunConfig(
            vendor=self.manifest.config["vendor"],
            model=self.manifest.config["model"],
            reasoning_effort=self.manifest.config["reasoning_effort"],
            proxy_url=self.manifest.config.get("proxy_url", ""),
        )

    def write_manifest(self) -> Path:
        with self._lock:
            self.manifest.updated_at = _now_iso()
            tmp_path = self.manifest_path.with_name(
                f"{self.manifest_path.name}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
            )
            try:
                tmp_path.write_text(
                    json.dumps(self.manifest.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                tmp_path.replace(self.manifest_path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
        return self.manifest_path

    def append_event(self, event_type: str, **payload: object) -> None:
        event = {"type": event_type, "at": _now_iso(), **payload}
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def set_status(self, status: str) -> None:
        self.manifest.status = status
        self.write_manifest()

    def _get_worker(self, work_dir: str) -> WorkerManifestEntry | None:
        for item in self.manifest.workers:
            if item.work_dir == work_dir:
                return item
        return None

    def ensure_worker(self, *, work_dir: str, forced: bool = False) -> WorkerManifestEntry:
        entry = self._get_worker(work_dir)
        if entry is not None:
            return entry
        entry = WorkerManifestEntry(work_dir=work_dir, forced=forced)
        self.manifest.workers.append(entry)
        return entry

    @staticmethod
    def _apply_state_to_entry(
        entry: WorkerManifestEntry,
        state: dict[str, object],
        *,
        preserve_workflow_fields: bool = False,
    ) -> WorkerManifestEntry:
        entry.session_name = str(state.get("session_name", entry.session_name))
        entry.runtime_dir = str(state.get("runtime_dir", entry.runtime_dir))
        entry.pane_id = str(state.get("pane_id", entry.pane_id))
        if not preserve_workflow_fields:
            entry.workflow_stage = str(state.get("workflow_stage", entry.workflow_stage))
            entry.workflow_round = int(state.get("workflow_round", entry.workflow_round))
            entry.result_status = str(state.get("result_status", entry.result_status))
        normalized_agent_state = _normalize_agent_state_from_runtime_state(state, entry.agent_state)
        active_prelaunch_update = preserve_workflow_fields and _is_active_prelaunch_state_update(
            entry,
            state,
            normalized_agent_state,
        )
        entry.agent_state = (
            AgentRuntimeState.STARTING.value
            if active_prelaunch_update
            else normalized_agent_state
        )
        entry.agent_alive = bool(state.get("agent_alive", entry.agent_alive))
        entry.agent_started = bool(state.get("agent_started", state.get("agent_ready", entry.agent_started)))
        entry.retry_count = int(state.get("retry_count", entry.retry_count))
        entry.last_turn_token = str(state.get("last_turn_token", entry.last_turn_token))
        entry.last_prompt_hash = str(state.get("last_prompt_hash", entry.last_prompt_hash))
        entry.last_heartbeat_at = str(state.get("last_heartbeat_at", entry.last_heartbeat_at))
        entry.last_log_offset = int(state.get("last_log_offset", entry.last_log_offset))
        entry.current_command = str(state.get("current_command", entry.current_command))
        entry.current_path = str(state.get("current_path", entry.current_path))
        entry.current_turn_id = str(state.get("current_turn_id", entry.current_turn_id))
        entry.current_turn_phase = str(state.get("current_turn_phase", entry.current_turn_phase))
        entry.current_turn_status_path = str(state.get("current_turn_status_path", entry.current_turn_status_path))
        entry.current_task_runtime_status = str(state.get("current_task_runtime_status", entry.current_task_runtime_status))
        baseline_hashes = state.get("current_turn_baseline_hashes", entry.current_turn_baseline_hashes)
        if isinstance(baseline_hashes, dict):
            entry.current_turn_baseline_hashes = {
                str(key): str(value) for key, value in baseline_hashes.items()
            }
        entry.recoverable = bool(state.get("recoverable", entry.recoverable))
        entry.health_status = str(state.get("health_status", entry.health_status))
        entry.health_note = str(state.get("health_note", entry.health_note))
        if active_prelaunch_update:
            entry.health_status = "unknown"
            if entry.health_note in {"", "missing_session", "pane_dead", "tmux session missing"}:
                entry.health_note = "launch pending"
        if not preserve_workflow_fields:
            entry.note = str(state.get("note", entry.note))
        entry.log_path = str(state.get("log_path", entry.log_path))
        entry.raw_log_path = str(state.get("raw_log_path", entry.raw_log_path))
        entry.state_path = str(state.get("state_path", entry.state_path))
        entry.transcript_path = str(state.get("transcript_path", entry.transcript_path))
        _normalize_active_routing_entry_display_state(entry)
        return entry

    def update_worker_binding(self, work_dir: str, **fields: object) -> WorkerManifestEntry:
        entry = self.ensure_worker(work_dir=work_dir, forced=bool(fields.get("forced", False)))
        for key, value in fields.items():
            if hasattr(entry, key):
                setattr(entry, key, value)
        _normalize_active_routing_entry_display_state(entry)
        self.write_manifest()
        return entry

    def update_worker_state(
        self,
        work_dir: str,
        state: dict[str, object],
        *,
        preserve_workflow_fields: bool = False,
    ) -> WorkerManifestEntry:
        entry = self.ensure_worker(work_dir=work_dir)
        self._apply_state_to_entry(entry, state, preserve_workflow_fields=preserve_workflow_fields)
        self.write_manifest()
        return entry

    def update_worker_state_from_file(
        self,
        work_dir: str,
        state_path: str | Path,
        *,
        preserve_workflow_fields: bool = False,
    ) -> WorkerManifestEntry | None:
        path = Path(state_path)
        if not path.exists():
            return None
        state = json.loads(path.read_text(encoding="utf-8"))
        if not state.get("runtime_dir"):
            state["runtime_dir"] = str(path.parent)
        return self.update_worker_state(work_dir, state, preserve_workflow_fields=preserve_workflow_fields)

    def sync_worker_snapshot(
        self,
        work_dir: str,
        *,
        state_path: str | Path | None = None,
        preserve_workflow_fields: bool = False,
        **fields: object,
    ) -> WorkerManifestEntry:
        entry = self.ensure_worker(work_dir=work_dir, forced=bool(fields.get("forced", False)))
        if state_path:
            path = Path(state_path)
            if path.exists():
                state = json.loads(path.read_text(encoding="utf-8"))
                if not state.get("runtime_dir"):
                    state["runtime_dir"] = str(path.parent)
                self._apply_state_to_entry(entry, state, preserve_workflow_fields=preserve_workflow_fields)
        for key, value in fields.items():
            if hasattr(entry, key):
                setattr(entry, key, value)
        _normalize_active_routing_entry_display_state(entry)
        self.write_manifest()
        return entry

    def update_worker_result(
        self,
        result: DirectoryInitResult,
        *,
        result_status_override: str | None = None,
    ) -> WorkerManifestEntry:
        entry = self.ensure_worker(work_dir=result.work_dir, forced=result.forced)
        entry.session_name = result.session_name or entry.session_name
        entry.runtime_dir = result.runtime_dir or entry.runtime_dir
        entry.result_status = result_status_override or result.status
        entry.workflow_stage = "completed"
        entry.workflow_round = result.rounds_used
        entry.failure_reason = result.failure_reason
        entry.last_audit_token = result.last_audit_token
        entry.last_audit_summary = result.last_audit_summary
        entry.last_audit_output = ""
        entry.missing_before = list(result.missing_before)
        entry.missing_after = list(result.missing_after)
        entry.note = result.failure_reason or result.last_audit_token or result.status
        entry.current_turn_id = ""
        entry.current_turn_phase = ""
        entry.current_turn_status_path = ""
        entry.current_turn_baseline_hashes = {}
        entry.current_task_runtime_status = ""
        self.write_manifest()
        self.append_event(
            "result_finalized",
            work_dir=result.work_dir,
            result_status=entry.result_status,
            failure_reason=result.failure_reason,
        )
        return entry


def _command_to_dict(result: CommandResult) -> dict[str, object]:
    return {
        "label": result.label,
        "command": result.command,
        "exit_code": result.exit_code,
        "clean_output": result.clean_output,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
    }


def _command_failure_reason(prefix: str, result: CommandResult) -> str:
    detail = clean_ansi(result.clean_output or result.raw_output or "").strip()
    if not detail:
        return prefix
    detail = " ".join(detail.split())
    if len(detail) > 240:
        detail = f"{detail[:237]}..."
    return f"{prefix}: {detail}"


def create_batch_runtime(
    run_id: str | None = None,
    *,
    project_dir: str | Path | None = None,
    runtime_root: str | Path | None = None,
) -> tuple[str, Path]:
    actual_run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
    base_root = resolve_routing_runtime_root(project_dir=project_dir, runtime_root=runtime_root)
    run_root = base_root / actual_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    return actual_run_id, run_root


def _result_from_manifest_entry(entry: WorkerManifestEntry) -> DirectoryInitResult:
    status = "failed" if entry.result_status == "stale_failed" else entry.result_status
    if status not in {"passed", "failed", "skipped"}:
        status = "failed"
    return DirectoryInitResult(
        work_dir=entry.work_dir,
        forced=entry.forced,
        status=status,
        rounds_used=entry.workflow_round,
        runtime_dir=entry.runtime_dir,
        session_name=entry.session_name,
        missing_before=list(entry.missing_before),
        missing_after=list(entry.missing_after),
        failure_reason=entry.failure_reason or ("stale_failed" if entry.result_status == "stale_failed" else ""),
        last_audit_token=entry.last_audit_token,
        last_audit_summary=entry.last_audit_summary,
    )


def prepare_live_workers(
    *,
    selection: TargetSelection,
    config: AgentRunConfig,
    runtime_root: str | Path | None = None,
    run_id: str | None = None,
    worker_factory: Callable[..., TmuxBatchWorker] = TmuxBatchWorker,
    run_store: RunStore | None = None,
) -> tuple[RunStore, list[LiveWorkerHandle], list[DirectoryInitResult]]:
    store = run_store or RunStore.create(
        selection=selection,
        config=config,
        runtime_root=runtime_root,
        run_id=run_id,
    )
    live_workers: list[LiveWorkerHandle] = []
    immediate_results: list[DirectoryInitResult] = []

    for target_dir in selection.selected_dirs:
        try:
            worker = worker_factory(
                worker_id=Path(target_dir).name or "project",
                work_dir=target_dir,
                config=config,
                runtime_root=store.run_root,
            )
        except Exception as error:
            failed = DirectoryInitResult(
                work_dir=target_dir,
                forced=target_dir in selection.forced_dirs,
                status="failed",
                rounds_used=0,
                missing_before=missing_routing_layer_files(target_dir),
                missing_after=missing_routing_layer_files(target_dir),
                failure_reason=f"worker_init_exception: {error}",
            )
            store.update_worker_result(failed)
            immediate_results.append(failed)
            continue

        handle = LiveWorkerHandle(
            work_dir=target_dir,
            forced=target_dir in selection.forced_dirs,
            worker=worker,
            resume_state={},
        )
        live_workers.append(handle)
        metadata = worker.runtime_metadata()
        store.update_worker_binding(
            target_dir,
            session_name=metadata["session_name"],
            runtime_dir=metadata["runtime_dir"],
            pane_id=metadata["pane_id"],
            forced=handle.forced,
            workflow_stage="pending",
            workflow_round=0,
            result_status="pending",
            agent_state=AgentRuntimeState.STARTING.value,
            agent_alive=False,
            agent_started=False,
            recoverable=True,
            current_turn_id="",
            current_turn_phase="",
            current_turn_status_path="",
            log_path=metadata["log_path"],
            raw_log_path=metadata["raw_log_path"],
            state_path=metadata["state_path"],
            transcript_path=metadata["transcript_path"],
            note="worker_prepared",
        )
        store.append_event(
            "worker_created",
            work_dir=target_dir,
            session_name=handle.session_name,
            runtime_dir=metadata["runtime_dir"],
        )

    return store, live_workers, immediate_results


def load_existing_run(
    *,
    run_id: str,
    project_dir: str | Path | None = None,
    runtime_root: str | Path | None = None,
    worker_factory: Callable[..., TmuxBatchWorker] = TmuxBatchWorker,
) -> tuple[RunStore, TargetSelection, AgentRunConfig, list[LiveWorkerHandle], dict[str, DirectoryInitResult]]:
    store = RunStore.load(run_id=run_id, project_dir=project_dir, runtime_root=runtime_root)
    selection = store.selection()
    config = store.config_object()
    live_workers: list[LiveWorkerHandle] = []
    results_by_dir: dict[str, DirectoryInitResult] = {
        skipped_dir: DirectoryInitResult(
            work_dir=skipped_dir,
            forced=False,
            status="skipped",
            rounds_used=0,
            missing_before=missing_routing_layer_files(skipped_dir),
            missing_after=missing_routing_layer_files(skipped_dir),
        )
        for skipped_dir in selection.skipped_dirs
    }

    for entry in list(store.manifest.workers):
        if entry.result_status in FINAL_RESULT_STATUSES:
            results_by_dir[entry.work_dir] = _result_from_manifest_entry(entry)
            continue

        worker = worker_factory(
            worker_id=Path(entry.work_dir).name or "project",
            work_dir=entry.work_dir,
            config=config,
            runtime_root=store.run_root,
            existing_runtime_dir=entry.runtime_dir,
            existing_session_name=entry.session_name,
            existing_pane_id=entry.pane_id,
        )
        if entry.state_path:
            store.update_worker_state_from_file(
                entry.work_dir,
                entry.state_path,
                preserve_workflow_fields=True,
            )
            entry = store.ensure_worker(work_dir=entry.work_dir)

        if entry.session_name and worker.session_exists():
            live_workers.append(
                LiveWorkerHandle(
                    work_dir=entry.work_dir,
                    forced=entry.forced,
                    worker=worker,
                    resume_state={
                        "workflow_stage": entry.workflow_stage,
                        "workflow_round": entry.workflow_round,
                        "result_status": entry.result_status,
                        "last_audit_token": entry.last_audit_token,
                        "last_audit_summary": entry.last_audit_summary,
                        "last_audit_output": entry.last_audit_output,
                        "current_turn_id": entry.current_turn_id,
                        "current_turn_phase": entry.current_turn_phase,
                        "current_turn_status_path": entry.current_turn_status_path,
                        "current_turn_baseline_hashes": dict(entry.current_turn_baseline_hashes),
                        "note": entry.note,
                    },
                )
            )
            continue

        stale_result = DirectoryInitResult(
            work_dir=entry.work_dir,
            forced=entry.forced,
            status="failed",
            rounds_used=entry.workflow_round,
            runtime_dir=entry.runtime_dir,
            session_name=entry.session_name,
            missing_before=list(entry.missing_before),
            missing_after=list(entry.missing_after),
            failure_reason="stale_failed",
            last_audit_token=entry.last_audit_token,
            last_audit_summary=entry.last_audit_summary,
        )
        store.update_worker_result(stale_result, result_status_override="stale_failed")
        results_by_dir[entry.work_dir] = stale_result

    return store, selection, config, live_workers, results_by_dir


def run_directory_initialization_with_worker(
    *,
    worker: TmuxBatchWorker,
    forced: bool = False,
    max_refine_rounds: int = 3,
    run_store: RunStore | None = None,
    resume_state: dict[str, object] | None = None,
) -> DirectoryInitResult:
    target_dir = resolve_existing_directory(worker.work_dir)
    missing_before = missing_routing_layer_files(target_dir)
    resumed_state = dict(resume_state or {})
    rounds_used = max(int(resumed_state.get("workflow_round", 0) or 0), 0)
    last_audit_output = str(resumed_state.get("last_audit_output", "") or "")
    last_audit_token = str(resumed_state.get("last_audit_token", "") or "")
    current_stage = str(resumed_state.get("workflow_stage", "pending") or "pending")
    current_turn_id = str(resumed_state.get("current_turn_id", "") or "")
    current_turn_phase = str(resumed_state.get("current_turn_phase", "") or "")
    current_turn_status_path = str(resumed_state.get("current_turn_status_path", "") or "")
    current_turn_baseline_hashes = {
        str(key): str(value)
        for key, value in dict(resumed_state.get("current_turn_baseline_hashes", {}) or {}).items()
    }

    def create_turn_id() -> str:
        return "create_routing_layer_1"

    def audit_turn_id(round_index: int) -> str:
        return f"audit_routing_layer_{int(round_index)}"

    def refine_turn_id(round_index: int) -> str:
        return f"refine_routing_layer_{int(round_index)}"

    def build_contract(
        turn_id: str,
        phase: str,
        required_artifacts: Sequence[str],
        *,
        baseline_artifact_hashes: dict[str, str] | None = None,
        require_artifact_change: bool = False,
    ) -> TurnFileContract:
        return build_turn_file_contract(
            runtime_dir=worker.runtime_dir,
            work_dir=target_dir,
            turn_id=turn_id,
            phase=phase,
            required_artifacts=required_artifacts,
            baseline_artifact_hashes=baseline_artifact_hashes,
            require_artifact_change=require_artifact_change,
        )

    def validate_contract(contract: TurnFileContract) -> TurnFileResult:
        return contract.validator(contract.status_path)

    def sync_state(workflow_stage: str, *, note: str = "", result_status: str = "running") -> None:
        if run_store is None:
            return
        run_store.sync_worker_snapshot(
            str(target_dir),
            state_path=worker.state_path,
            preserve_workflow_fields=True,
            workflow_stage=workflow_stage,
            workflow_round=rounds_used,
            result_status=result_status,
            forced=forced,
            note=note or workflow_stage,
            session_name=worker.session_name,
            runtime_dir=str(worker.runtime_dir),
            pane_id=worker.pane_id,
            last_audit_token=last_audit_token,
            last_audit_summary=summarize_audit_output(last_audit_output),
            last_audit_output=last_audit_output,
            current_turn_id=current_turn_id,
            current_turn_phase=current_turn_phase,
            current_turn_status_path=current_turn_status_path,
            current_turn_baseline_hashes=dict(current_turn_baseline_hashes),
        )
        run_store.append_event(
            "state_changed",
            work_dir=str(target_dir),
            workflow_stage=workflow_stage,
            result_status=result_status,
            note=note or workflow_stage,
        )

    def fail(reason: str) -> DirectoryInitResult:
        collected = worker.collect_result()
        result = DirectoryInitResult(
            work_dir=str(target_dir),
            forced=forced,
            status="failed",
            rounds_used=rounds_used,
            runtime_dir=collected.runtime_dir,
            session_name=collected.session_name,
            missing_before=missing_before,
            missing_after=missing_routing_layer_files(target_dir),
            failure_reason=reason,
            last_audit_token=last_audit_token,
            last_audit_summary=summarize_audit_output(last_audit_output),
            commands=[_command_to_dict(item) for item in collected.commands],
        )
        if run_store is not None:
            run_store.update_worker_result(result)
        return result

    def finalize_pass_result() -> DirectoryInitResult:
        collected = worker.collect_result()
        missing_after = missing_routing_layer_files(target_dir)
        if missing_after:
            return fail("required_files_missing_after_pass")
        try:
            validate_routing_layer_artifacts(target_dir)
        except ValueError as error:
            return fail(f"routing_layer_artifacts_invalid: {error}")
        result = DirectoryInitResult(
            work_dir=str(target_dir),
            forced=forced,
            status="passed",
            rounds_used=rounds_used,
            runtime_dir=collected.runtime_dir,
            session_name=collected.session_name,
            missing_before=missing_before,
            missing_after=missing_after,
            last_audit_token=last_audit_token,
            last_audit_summary=summarize_audit_output(last_audit_output),
            commands=[_command_to_dict(item) for item in collected.commands],
        )
        if run_store is not None:
            run_store.update_worker_result(result)
        return result

    def consume_audit_files(round_index: int, *, source: str) -> str | DirectoryInitResult:
        nonlocal last_audit_output, last_audit_token, rounds_used, current_turn_id, current_turn_phase, current_turn_status_path, current_turn_baseline_hashes
        try:
            audit_decision = load_routing_audit_decision(target_dir, expected_round=round_index)
        except Exception as error:
            return fail(f"audit_status_invalid: {error}")

        last_audit_output = audit_decision.record_text
        last_audit_token = (
            ROUTING_AUDIT_PASS_TOKEN if audit_decision.status == ROUTING_AUDIT_STATUS_PASS else ROUTING_AUDIT_REVISE_TOKEN
        )
        if last_audit_token == ROUTING_AUDIT_PASS_TOKEN:
            current_turn_id = ""
            current_turn_phase = ""
            current_turn_status_path = ""
            current_turn_baseline_hashes = {}
            sync_state("completed", note=f"{source}:audit_pass", result_status="passed")
            return finalize_pass_result()

        if last_audit_token != ROUTING_AUDIT_REVISE_TOKEN:
            return fail("audit_status_invalid")
        if not last_audit_output.strip():
            return fail("audit_revise_body_missing")
        if rounds_used >= max_refine_rounds:
            return fail("max_refine_rounds_reached")
        rounds_used += 1
        current_turn_id = ""
        current_turn_phase = ""
        current_turn_status_path = ""
        current_turn_baseline_hashes = {}
        sync_state("refine_pending", note=f"{source}:audit_revise")
        return "refine"

    def run_create_step() -> str | DirectoryInitResult:
        nonlocal current_turn_id, current_turn_phase, current_turn_status_path, current_turn_baseline_hashes
        current_turn_id = create_turn_id()
        current_turn_phase = PHASE_ROUTING_LAYER_CREATE
        current_turn_baseline_hashes = capture_artifact_hashes(target_dir, ROUTING_LAYER_REQUIRED_FILES)
        contract = build_contract(
            current_turn_id,
            current_turn_phase,
            ROUTING_LAYER_REQUIRED_FILES,
            baseline_artifact_hashes=current_turn_baseline_hashes,
            require_artifact_change=bool(current_turn_baseline_hashes),
        )
        reset_turn_runtime_dir(worker.runtime_dir, current_turn_id)
        current_turn_status_path = str(contract.status_path)
        sync_state("create_running", note="create_routing_layer")
        create_result = worker.run_turn(
            label="create_routing_layer",
            prompt=build_create_prompt(),
            completion_contract=contract,
        )
        if run_store is not None:
            run_store.update_worker_state_from_file(
                str(target_dir),
                worker.state_path,
                preserve_workflow_fields=True,
            )
            run_store.append_event("turn_finished", work_dir=str(target_dir), label="create_routing_layer")
        if not create_result.ok:
            return fail(_command_failure_reason("create_command_failed", create_result))
        try:
            validate_contract(contract)
        except Exception as error:
            return fail(f"create_turn_status_invalid: {error}")
        current_turn_id = ""
        current_turn_phase = ""
        current_turn_status_path = ""
        current_turn_baseline_hashes = {}
        sync_state("audit_pending", note="create_completed")
        return "audit"

    def run_audit_step(round_index: int) -> str | DirectoryInitResult:
        nonlocal last_audit_output, last_audit_token, rounds_used, current_turn_id, current_turn_phase, current_turn_status_path, current_turn_baseline_hashes
        current_turn_id = audit_turn_id(round_index)
        current_turn_phase = PHASE_ROUTING_LAYER_AUDIT
        current_turn_baseline_hashes = {}
        contract = build_contract(
            current_turn_id,
            current_turn_phase,
            (ROUTING_AUDIT_STATUS_FILE, ROUTING_AUDIT_RECORD_FILE),
        )
        reset_turn_runtime_dir(worker.runtime_dir, current_turn_id)
        current_turn_status_path = str(contract.status_path)
        reset_routing_audit_artifacts(target_dir)
        sync_state("audit_running", note=f"audit_routing_layer_{round_index}")
        audit_result = worker.run_turn(
            label=f"audit_routing_layer_{round_index}",
            prompt=build_audit_prompt(
                audit_round=round_index,
            ),
            completion_contract=contract,
        )
        if run_store is not None:
            run_store.update_worker_state_from_file(
                str(target_dir),
                worker.state_path,
                preserve_workflow_fields=True,
            )
            run_store.append_event(
                "turn_finished",
                work_dir=str(target_dir),
                label=f"audit_routing_layer_{round_index}",
            )
        if not audit_result.ok:
            return fail(_command_failure_reason("audit_command_failed", audit_result))
        try:
            validate_contract(contract)
        except Exception as error:
            return fail(f"audit_turn_status_invalid: {error}")
        return consume_audit_files(round_index, source="fresh")

    def run_refine_step() -> str | DirectoryInitResult:
        nonlocal current_turn_id, current_turn_phase, current_turn_status_path, current_turn_baseline_hashes
        current_round = max(rounds_used, 1)
        try:
            audit_decision = load_routing_audit_decision(target_dir, expected_round=current_round)
        except Exception as error:
            return fail(f"audit_status_invalid: {error}")
        audit_record = Path(audit_decision.record_path)
        current_turn_id = refine_turn_id(current_round)
        current_turn_phase = PHASE_ROUTING_LAYER_REFINE
        current_turn_baseline_hashes = capture_artifact_hashes(target_dir, ROUTING_LAYER_REQUIRED_FILES)
        contract = build_contract(
            current_turn_id,
            current_turn_phase,
            ROUTING_LAYER_REQUIRED_FILES,
            baseline_artifact_hashes=current_turn_baseline_hashes,
            require_artifact_change=bool(current_turn_baseline_hashes),
        )
        reset_turn_runtime_dir(worker.runtime_dir, current_turn_id)
        current_turn_status_path = str(contract.status_path)
        sync_state("refine_running", note=f"refine_routing_layer_{current_round}")
        refine_result = worker.run_turn(
            label=f"refine_routing_layer_{current_round}",
            prompt=build_refine_prompt(
                audit_record,
            ),
            completion_contract=contract,
        )
        if run_store is not None:
            run_store.update_worker_state_from_file(
                str(target_dir),
                worker.state_path,
                preserve_workflow_fields=True,
            )
            run_store.append_event(
                "turn_finished",
                work_dir=str(target_dir),
                label=f"refine_routing_layer_{current_round}",
            )
        if not refine_result.ok:
            return fail(_command_failure_reason("refine_command_failed", refine_result))
        try:
            validate_contract(contract)
        except Exception as error:
            return fail(f"refine_turn_status_invalid: {error}")
        current_turn_id = ""
        current_turn_phase = ""
        current_turn_status_path = ""
        current_turn_baseline_hashes = {}
        sync_state("audit_pending", note="refine_completed")
        return "audit"

    def try_resume_completed_step() -> str | DirectoryInitResult | None:
        nonlocal current_turn_id, current_turn_phase, current_turn_status_path, current_turn_baseline_hashes
        if not current_turn_id or not current_turn_phase or not current_turn_status_path:
            return None

        if current_turn_phase == PHASE_ROUTING_LAYER_CREATE:
            contract = build_contract(
                current_turn_id,
                current_turn_phase,
                ROUTING_LAYER_REQUIRED_FILES,
                baseline_artifact_hashes=current_turn_baseline_hashes,
                require_artifact_change=bool(current_turn_baseline_hashes),
            )
            status_path = Path(current_turn_status_path).expanduser().resolve()
            if status_path != contract.status_path:
                contract = TurnFileContract(
                    turn_id=contract.turn_id,
                    phase=contract.phase,
                    status_path=status_path,
                    validator=contract.validator,
                    quiet_window_sec=contract.quiet_window_sec,
                )
            try:
                validate_contract(contract)
            except Exception:
                return None
            current_turn_id = ""
            current_turn_phase = ""
            current_turn_status_path = ""
            current_turn_baseline_hashes = {}
            sync_state("audit_pending", note="resume:create_completed")
            return "audit"

        if current_turn_phase == PHASE_ROUTING_LAYER_AUDIT:
            contract = build_contract(
                current_turn_id,
                current_turn_phase,
                (ROUTING_AUDIT_STATUS_FILE, ROUTING_AUDIT_RECORD_FILE),
            )
            status_path = Path(current_turn_status_path).expanduser().resolve()
            if status_path != contract.status_path:
                contract = TurnFileContract(
                    turn_id=contract.turn_id,
                    phase=contract.phase,
                    status_path=status_path,
                    validator=contract.validator,
                    quiet_window_sec=contract.quiet_window_sec,
                )
            try:
                validate_contract(contract)
            except Exception:
                return None
            return consume_audit_files(rounds_used + 1, source="resume")

        if current_turn_phase == PHASE_ROUTING_LAYER_REFINE:
            contract = build_contract(
                current_turn_id,
                current_turn_phase,
                ROUTING_LAYER_REQUIRED_FILES,
                baseline_artifact_hashes=current_turn_baseline_hashes,
                require_artifact_change=bool(current_turn_baseline_hashes),
            )
            status_path = Path(current_turn_status_path).expanduser().resolve()
            if status_path != contract.status_path:
                contract = TurnFileContract(
                    turn_id=contract.turn_id,
                    phase=contract.phase,
                    status_path=status_path,
                    validator=contract.validator,
                    quiet_window_sec=contract.quiet_window_sec,
                )
            try:
                validate_contract(contract)
            except Exception:
                return None
            current_turn_id = ""
            current_turn_phase = ""
            current_turn_status_path = ""
            current_turn_baseline_hashes = {}
            sync_state("audit_pending", note="resume:refine_completed")
            return "audit"
        return None

    try:
        resumed_step = try_resume_completed_step()
        if isinstance(resumed_step, DirectoryInitResult):
            return resumed_step
        if isinstance(resumed_step, str):
            next_action = resumed_step
        elif current_stage in {"audit_pending", "audit_running"}:
            next_action = "audit"
        elif current_stage in {"refine_pending", "refine_running"}:
            rounds_used = max(rounds_used, 1)
            next_action = "refine"
        else:
            rounds_used = 0
            next_action = "create"

        while True:
            if next_action == "create":
                next_step = run_create_step()
            elif next_action == "audit":
                next_step = run_audit_step(rounds_used + 1)
            else:
                next_step = run_refine_step()

            if isinstance(next_step, DirectoryInitResult):
                return next_step
            next_action = next_step
    except Exception as error:
        return fail(f"workflow_exception: {error}")


def run_directory_initialization(
    *,
    work_dir: str | Path,
    config: AgentRunConfig,
    runtime_root: str | Path,
    forced: bool = False,
    max_refine_rounds: int = 3,
    worker_factory: Callable[..., TmuxBatchWorker] = TmuxBatchWorker,
) -> DirectoryInitResult:
    target_dir = resolve_existing_directory(work_dir)
    try:
        worker = worker_factory(
            worker_id=target_dir.name or "project",
            work_dir=target_dir,
            config=config,
            runtime_root=runtime_root,
        )
    except Exception as error:
        return DirectoryInitResult(
            work_dir=str(target_dir),
            forced=forced,
            status="failed",
            rounds_used=0,
            missing_before=missing_routing_layer_files(target_dir),
            missing_after=missing_routing_layer_files(target_dir),
            failure_reason=f"worker_init_exception: {error}",
        )
    return run_directory_initialization_with_worker(
        worker=worker,
        forced=forced,
        max_refine_rounds=max_refine_rounds,
    )


def has_overlapping_scope_paths(paths: Sequence[str | Path]) -> bool:
    resolved_paths = [resolve_existing_directory(path) for path in paths]
    for index, left in enumerate(resolved_paths):
        for right in resolved_paths[index + 1:]:
            if left == right:
                continue
            if right.is_relative_to(left) or left.is_relative_to(right):
                return True
    return False


def determine_batch_worker_count(
    selected_dirs: Sequence[str | Path],
    max_workers: int | None = None,
) -> int:
    return max_workers or min(len(selected_dirs), 4) or 1


def build_batch_result(
    *,
    run_store: RunStore,
    selection: TargetSelection,
    config: AgentRunConfig,
    results_by_dir: dict[str, DirectoryInitResult],
) -> BatchInitResult:
    ordered_results = [results_by_dir[item] for item in (*selection.selected_dirs, *selection.skipped_dirs)]
    batch_result = BatchInitResult(
        run_id=run_store.manifest.run_id,
        runtime_dir=str(run_store.run_root.resolve()),
        selection=selection,
        config=config.to_summary(),
        results=ordered_results,
    )
    run_store.set_status("completed")
    run_store.append_event("run_completed", failed=any(item.status == "failed" for item in ordered_results))
    return batch_result


def kill_run_tmux_sessions(
    *,
    run_store: RunStore,
    runtime_controller: TmuxRuntimeController | None = None,
) -> list[str]:
    controller = runtime_controller or TmuxRuntimeController()
    killed_sessions: list[str] = []
    seen: set[str] = set()
    for entry in run_store.manifest.workers:
        session_name = str(entry.session_name or "").strip()
        if not session_name or session_name in seen:
            continue
        seen.add(session_name)
        killed_sessions.append(controller.kill_session(session_name, missing_ok=True))
    run_store.append_event(
        "routing_tmux_cleanup",
        session_names=[item for item in killed_sessions if item],
        count=len([item for item in killed_sessions if item]),
    )
    return [item for item in killed_sessions if item]


def cleanup_routing_stage_artifacts(
    *,
    batch_result: BatchInitResult,
) -> RoutingCleanupResult:
    if any(item.status == "failed" for item in batch_result.results):
        return RoutingCleanupResult()

    removed_files: list[str] = []
    removed_runtime_dirs: list[str] = []
    unique_dirs: list[str] = []
    for candidate in (*batch_result.selection.selected_dirs, *batch_result.selection.skipped_dirs):
        if candidate not in unique_dirs:
            unique_dirs.append(candidate)

    for work_dir in unique_dirs:
        root = Path(work_dir).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            continue
        for file_name in (ROUTING_AUDIT_STATUS_FILE, ROUTING_AUDIT_RECORD_FILE):
            file_path = root / file_name
            if not file_path.exists():
                continue
            if file_path.is_file():
                file_path.unlink()
                removed_files.append(str(file_path))

    runtime_dir = Path(batch_result.runtime_dir).expanduser().resolve()
    if runtime_dir.exists() and runtime_dir.is_dir():
        shutil.rmtree(runtime_dir)
        removed_runtime_dirs.append(str(runtime_dir))
        runtime_parent = runtime_dir.parent
        if runtime_parent.exists() and runtime_parent.is_dir() and not any(runtime_parent.iterdir()):
            runtime_parent.rmdir()
            removed_runtime_dirs.append(str(runtime_parent))

    return RoutingCleanupResult(
        removed_intermediate_files=tuple(removed_files),
        removed_runtime_dirs=tuple(removed_runtime_dirs),
    )


def run_batch_initialization(
    *,
    selection: TargetSelection,
    config: AgentRunConfig,
    runtime_root: str | Path | None = None,
    max_refine_rounds: int = 3,
    max_workers: int | None = None,
    worker_factory: Callable[..., TmuxBatchWorker] = TmuxBatchWorker,
    on_workers_prepared: Callable[[RunStore, Sequence[LiveWorkerHandle], Sequence[DirectoryInitResult]], None] | None = None,
) -> BatchInitResult:
    run_store, live_workers, immediate_results = prepare_live_workers(
        selection=selection,
        config=config,
        runtime_root=runtime_root,
        worker_factory=worker_factory,
    )
    if on_workers_prepared is not None:
        on_workers_prepared(run_store, tuple(live_workers), tuple(immediate_results))

    results_by_dir: dict[str, DirectoryInitResult] = {}
    for skipped_dir in selection.skipped_dirs:
        results_by_dir[skipped_dir] = DirectoryInitResult(
            work_dir=skipped_dir,
            forced=False,
            status="skipped",
            rounds_used=0,
            missing_before=missing_routing_layer_files(skipped_dir),
            missing_after=missing_routing_layer_files(skipped_dir),
        )
    for item in immediate_results:
        results_by_dir[item.work_dir] = item

    if live_workers:
        worker_count = determine_batch_worker_count([item.work_dir for item in live_workers], max_workers=max_workers)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    run_directory_initialization_with_worker,
                    worker=handle.worker,
                    forced=handle.forced,
                    max_refine_rounds=max_refine_rounds,
                    run_store=run_store,
                    resume_state=handle.resume_state,
                ): handle.work_dir
                for handle in live_workers
            }
            for future in as_completed(future_map):
                target_dir = future_map[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = DirectoryInitResult(
                        work_dir=target_dir,
                        forced=target_dir in selection.forced_dirs,
                        status="failed",
                        rounds_used=0,
                        missing_before=missing_routing_layer_files(target_dir),
                        missing_after=missing_routing_layer_files(target_dir),
                        failure_reason=f"batch_future_exception: {error}",
                    )
                    run_store.update_worker_result(result)
                results_by_dir[result.work_dir] = result

    return build_batch_result(
        run_store=run_store,
        selection=selection,
        config=config,
        results_by_dir=results_by_dir,
    )
