from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TASK_RESULT_SCHEMA_VERSION = "1.0"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_DONE = "done"
TASK_RESULT_READY = "ready"
TASK_RESULT_COMPLETED = "completed"
TASK_RESULT_HITL = "hitl"
TASK_RESULT_ERROR = "error"
TASK_RESULT_REVIEW_PASS = "review_pass"
TASK_RESULT_REVIEW_FAIL = "review_fail"

_MODE_SPECIFIC_TASK_RESULT_MODES = {
    "a02_notion_intake",
    "a02_requirement_intake",
    "a02_requirements_analysis",
    "a03_requirements_clarification",
    "a03_ba_resume",
    "a03_reviewer_round",
    "a03_ba_feedback",
    "a03_ba_review_limit_human_reply",
    "a03_ba_review_limit_force_hitl",
    "a03_human_feedback",
    "a05_ba_init",
    "a05_detailed_design_generate",
    "a05_detailed_design_feedback",
    "a05_detailed_design_review_limit_human_reply",
    "a05_detailed_design_review_limit_force_hitl",
    "a06_ba_init",
    "a06_reviewer_init",
    "a06_task_split_generate",
    "a06_task_split_feedback",
    "a06_task_split_review_limit_human_reply",
    "a06_task_split_review_limit_force_hitl",
    "a06_task_split_json_generate",
    "a07_developer_init",
    "a07_developer_human_reply",
    "a07_reviewer_init",
    "a07_developer_task_complete",
    "a07_developer_refine",
    "a07_developer_review_feedback",
    "a07_developer_review_limit_human_reply",
    "a07_developer_review_limit_force_hitl",
    "a08_developer_init",
    "a08_reviewer_init",
    "a08_developer_refine_all_code",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve_optional_path(path: str | Path | None) -> Path | None:
    text = str(path or "").strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(target)
    return target


def _build_prefixed_sha256(path: str | Path) -> str:
    target = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with target.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def snapshot_file_fingerprint(path: str | Path) -> dict[str, object]:
    target = Path(path).expanduser().resolve()
    if not target.exists() or not target.is_file():
        return {"exists": False, "path": str(target)}
    stat = target.stat()
    return {
        "exists": True,
        "path": str(target),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _build_prefixed_sha256(target),
    }


def _artifact_rule(contract_rules: dict[str, object], alias: str) -> dict[str, object]:
    raw = contract_rules.get(alias, {})
    return raw if isinstance(raw, dict) else {}


def _baseline_from_rule(rule: dict[str, object]) -> dict[str, object]:
    baseline = rule.get("baseline", {})
    return baseline if isinstance(baseline, dict) else {}


def _fingerprints_match(left: dict[str, object], right: dict[str, object]) -> bool:
    if not bool(left.get("exists")) or not bool(right.get("exists")):
        return bool(left.get("exists")) == bool(right.get("exists"))
    left_hash = str(left.get("sha256", "")).strip()
    right_hash = str(right.get("sha256", "")).strip()
    return bool(left_hash) and left_hash == right_hash


def _enforce_artifact_change_rule(
        *,
        alias: str,
        path: Path,
        rule: dict[str, object],
) -> None:
    change = str(rule.get("change", "none") or "none").strip()
    if change in {"", "none", "may_change", "must_exist_nonempty"}:
        return
    baseline = _baseline_from_rule(rule)
    current = snapshot_file_fingerprint(path)
    if change == "must_change":
        if _fingerprints_match(current, baseline):
            raise ValueError(f"artifact 未变化: {alias} -> {path}")
        return
    if change == "must_not_change":
        if not _fingerprints_match(current, baseline):
            raise ValueError(f"artifact 不应变化: {alias} -> {path}")
        return
    raise ValueError(f"未知 artifact change policy: {alias} -> {change}")


def _read_text(path: str | Path | None) -> str:
    target = _resolve_optional_path(path)
    if target is None or not target.exists() or not target.is_file():
        return ""
    return target.read_text(encoding="utf-8").strip()


def _iter_review_status_items(payload: object) -> tuple[dict[str, Any], ...]:
    if isinstance(payload, dict):
        return (payload,)
    if isinstance(payload, list):
        return tuple(item for item in payload if isinstance(item, dict))
    return ()


def normalize_review_status_payload(
        payload: object,
        *,
        task_name: str,
        source: str = "审核 JSON",
) -> dict[str, Any]:
    task_name_text = str(task_name or "").strip()
    source_text = str(source or "").strip() or "审核 JSON"
    if not task_name_text:
        raise ValueError(f"{source_text} 缺少 task_name")
    if not isinstance(payload, (list, dict)):
        raise ValueError(f"{source_text} 必须是 list 或单对象 dict")
    matched_item: dict[str, Any] | None = None
    for item in _iter_review_status_items(payload):
        if str(item.get("task_name", "")).strip() == task_name_text:
            matched_item = item
            break
    if matched_item is None:
        raise ValueError(f"{source_text} 缺少 {task_name_text} 状态项")
    review_pass = matched_item.get("review_pass")
    if not isinstance(review_pass, bool):
        raise ValueError(f"{source_text} 中 {task_name_text}.review_pass 必须是 bool")
    return matched_item


@dataclass(frozen=True)
class TurnFileResult:
    status_path: str
    payload: dict[str, object]
    artifact_paths: dict[str, str]
    artifact_hashes: dict[str, str]
    validated_at: str


@dataclass(frozen=True)
class TurnFileContract:
    turn_id: str
    phase: str
    status_path: Path
    validator: Any
    quiet_window_sec: float = 1.0
    kind: str = ""
    tracked_artifacts: dict[str, Path] = field(default_factory=dict)
    artifact_rules: dict[str, object] = field(default_factory=dict)
    outcome_artifacts: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status_path", Path(self.status_path).expanduser().resolve())
        object.__setattr__(
            self,
            "tracked_artifacts",
            {
                str(alias).strip(): Path(path).expanduser().resolve()
                for alias, path in self.tracked_artifacts.items()
                if str(alias).strip()
            },
        )
        object.__setattr__(
            self,
            "artifact_rules",
            {str(alias).strip(): rule for alias, rule in self.artifact_rules.items() if str(alias).strip()},
        )
        object.__setattr__(self, "outcome_artifacts", _normalize_outcome_artifacts(self.outcome_artifacts))


@dataclass(frozen=True)
class TaskResultFile:
    result_path: str
    payload: dict[str, object]
    artifact_paths: dict[str, str]
    artifact_hashes: dict[str, str]
    validated_at: str


@dataclass(frozen=True)
class TaskResultContract:
    turn_id: str
    phase: str
    task_kind: str
    mode: str
    expected_statuses: tuple[str, ...]
    stage_name: str = ""
    turn_status_path: Path | None = None
    stage_status_path: Path | None = None
    required_artifacts: dict[str, Path] = field(default_factory=dict)
    optional_artifacts: dict[str, Path] = field(default_factory=dict)
    artifact_rules: dict[str, object] = field(default_factory=dict)
    outcome_artifacts: dict[str, object] = field(default_factory=dict)
    retry_policy: dict[str, object] = field(default_factory=dict)
    resume_policy: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "turn_status_path", _resolve_optional_path(self.turn_status_path))
        object.__setattr__(self, "stage_status_path", _resolve_optional_path(self.stage_status_path))
        object.__setattr__(
            self,
            "required_artifacts",
            {key: Path(value).expanduser().resolve() for key, value in self.required_artifacts.items()},
        )
        object.__setattr__(
            self,
            "optional_artifacts",
            {key: Path(value).expanduser().resolve() for key, value in self.optional_artifacts.items()},
        )
        object.__setattr__(
            self,
            "expected_statuses",
            tuple(str(item).strip() for item in self.expected_statuses if str(item).strip()),
        )
        object.__setattr__(
            self,
            "artifact_rules",
            {str(alias).strip(): rule for alias, rule in self.artifact_rules.items() if str(alias).strip()},
        )
        object.__setattr__(self, "outcome_artifacts", _normalize_outcome_artifacts(self.outcome_artifacts))


@dataclass(frozen=True)
class TaskResultDecision:
    status: str
    summary: str
    artifacts: dict[str, str]
    artifact_hashes: dict[str, str]


@dataclass(frozen=True)
class MissingTaskResultFinalizationCandidate:
    decision: TaskResultDecision
    signature: tuple[object, ...]
    requires_agent_ready: bool


@dataclass(frozen=True)
class TaskResultObservation:
    observed_status: str
    present_aliases: tuple[str, ...]
    missing_required_aliases: tuple[str, ...]
    empty_required_aliases: tuple[str, ...]
    artifact_paths: dict[str, str]
    last_validation_error: str


@dataclass(frozen=True)
class CompletionObservation:
    observed_status: str
    present_aliases: tuple[str, ...]
    missing_required_aliases: tuple[str, ...]
    empty_required_aliases: tuple[str, ...]
    artifact_paths: dict[str, str]
    last_validation_error: str


def write_task_status(path: str | Path, *, status: str) -> Path:
    return _write_json_atomic(path, {"status": status})


def read_task_status(path: str | Path) -> str:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return ""
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if payload == {"status": TASK_STATUS_RUNNING}:
        return TASK_STATUS_RUNNING
    if payload == {"status": TASK_STATUS_DONE}:
        return TASK_STATUS_DONE
    return ""


def read_task_result_payload(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_task_result_payload(path: str | Path, payload: dict[str, Any]) -> Path:
    return _write_json_atomic(path, payload)


def _normalize_alias_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _normalize_outcome_artifacts(raw: dict[str, object]) -> dict[str, dict[str, tuple[str, ...]]]:
    normalized: dict[str, dict[str, tuple[str, ...]]] = {}
    for status, value in raw.items():
        status_key = str(status).strip()
        if not status_key or not isinstance(value, dict):
            continue
        normalized[status_key] = {
            "requires": _normalize_alias_tuple(value.get("requires", ())),
            "optional": _normalize_alias_tuple(value.get("optional", ())),
            "forbids": _normalize_alias_tuple(value.get("forbids", ())),
        }
    return normalized


def _contract_artifact_paths(contract: TaskResultContract) -> dict[str, Path]:
    return {**contract.required_artifacts, **contract.optional_artifacts}


def _outcome_artifact_spec(contract: TaskResultContract, status: str) -> dict[str, tuple[str, ...]]:
    spec = contract.outcome_artifacts.get(str(status).strip(), {})
    return spec if isinstance(spec, dict) else {}


def _outcome_required_aliases(contract: TaskResultContract, status: str) -> tuple[str, ...]:
    spec = _outcome_artifact_spec(contract, status)
    return tuple(spec.get("requires", ())) if spec else tuple(contract.required_artifacts)


def _outcome_optional_aliases(contract: TaskResultContract, status: str) -> tuple[str, ...]:
    spec = _outcome_artifact_spec(contract, status)
    return tuple(spec.get("optional", ())) if spec else tuple(contract.optional_artifacts)


def _outcome_forbidden_aliases(contract: TaskResultContract, status: str) -> tuple[str, ...]:
    spec = _outcome_artifact_spec(contract, status)
    return tuple(spec.get("forbids", ())) if spec else ()


def _outcome_collection_aliases(contract: TaskResultContract, status: str) -> tuple[str, ...]:
    aliases: list[str] = []
    for alias in _outcome_required_aliases(contract, status) + _outcome_optional_aliases(contract, status):
        if alias not in aliases:
            aliases.append(alias)
    if aliases:
        return tuple(aliases)
    return tuple(_contract_artifact_paths(contract))


def collect_contract_artifacts(
        contract: TaskResultContract,
        *,
        aliases: tuple[str, ...] | None = None,
        required_aliases: tuple[str, ...] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    artifacts: dict[str, str] = {}
    artifact_hashes: dict[str, str] = {}
    empty_allowed_required_aliases: set[str] = set()
    if contract.mode == "a03_reviewer_round":
        empty_allowed_required_aliases.add("review_md")
    all_artifacts = _contract_artifact_paths(contract)
    selected_aliases = tuple(dict.fromkeys(aliases)) if aliases is not None else tuple(all_artifacts)
    required_set = set(required_aliases) if required_aliases is not None else set(contract.required_artifacts)
    for alias in selected_aliases:
        artifact_path = all_artifacts.get(alias)
        if artifact_path is None:
            if alias in required_set:
                raise FileNotFoundError(f"缺少必填 artifact path: {alias}")
            continue
        resolved = artifact_path.expanduser().resolve()
        if not resolved.exists() or not resolved.is_file():
            if alias in required_set:
                raise FileNotFoundError(f"缺少必填 artifact: {resolved}")
            continue
        text = resolved.read_text(encoding="utf-8").strip()
        if not text:
            if alias in required_set and alias not in empty_allowed_required_aliases:
                raise ValueError(f"缺少必填 artifact 内容: {resolved}")
            continue
        _enforce_artifact_change_rule(
            alias=alias,
            path=resolved,
            rule=_artifact_rule(contract.artifact_rules, alias),
        )
        artifacts[alias] = str(resolved)
        artifact_hashes[str(resolved)] = _build_prefixed_sha256(resolved)
    return artifacts, artifact_hashes


def _expected_status_set(contract: TaskResultContract) -> set[str]:
    return {str(status or "").strip() for status in contract.expected_statuses if str(status or "").strip()}


def _required_aliases_for_decision(contract: TaskResultContract, status: str) -> tuple[str, ...]:
    required_aliases: tuple[str, ...] = ()
    if contract.outcome_artifacts:
        required_aliases = _outcome_required_aliases(contract, status)
    if not required_aliases:
        required_aliases = tuple(contract.required_artifacts)
    return required_aliases


def build_missing_task_result_finalization_candidate(
    contract: TaskResultContract,
    *,
    task_status_path: str | Path | None = None,
) -> MissingTaskResultFinalizationCandidate | None:
    expected_statuses = _expected_status_set(contract)
    if not expected_statuses:
        return None
    try:
        decision = resolve_task_result_decision(contract)
    except Exception:
        return None
    if decision.status not in expected_statuses:
        return None
    if decision.status == TASK_RESULT_READY:
        return MissingTaskResultFinalizationCandidate(
            decision=decision,
            requires_agent_ready=True,
            signature=(
                decision.status,
                "agent_ready",
                tuple(sorted(decision.artifact_hashes.items())),
            ),
        )
    if decision.status not in {TASK_RESULT_COMPLETED, TASK_RESULT_HITL}:
        return None

    required_aliases = _required_aliases_for_decision(contract, decision.status)
    if not required_aliases or task_status_path is None:
        return None
    try:
        task_started_mtime_ns = Path(task_status_path).expanduser().resolve().stat().st_mtime_ns
    except OSError:
        return None

    all_artifacts = _contract_artifact_paths(contract)
    artifact_signatures: list[tuple[object, ...]] = []
    saw_fresh_required_artifact = False
    for alias in required_aliases:
        artifact_path_text = str(decision.artifacts.get(alias, "")).strip()
        artifact_path = Path(artifact_path_text).expanduser().resolve() if artifact_path_text else all_artifacts.get(alias)
        if artifact_path is None:
            return None
        try:
            resolved_path = artifact_path.expanduser().resolve()
            artifact_stat = resolved_path.stat()
            artifact_text = resolved_path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not artifact_text:
            return None
        if artifact_stat.st_mtime_ns >= task_started_mtime_ns:
            saw_fresh_required_artifact = True
        artifact_signatures.append(
            (
                alias,
                str(resolved_path),
                artifact_stat.st_size,
                artifact_stat.st_mtime_ns,
                decision.artifact_hashes.get(str(resolved_path), ""),
            )
        )
    if not saw_fresh_required_artifact:
        return None
    return MissingTaskResultFinalizationCandidate(
        decision=decision,
        requires_agent_ready=True,
        signature=(decision.status, tuple(sorted(artifact_signatures))),
    )


def _turn_file_status_from_result(result: TurnFileResult | None) -> str:
    if result is None:
        return ""
    review_pass = result.payload.get("review_pass")
    if isinstance(review_pass, bool):
        return TASK_RESULT_REVIEW_PASS if review_pass else TASK_RESULT_REVIEW_FAIL
    return str(result.payload.get("status", "")).strip()


def _turn_file_selected_aliases(contract: TurnFileContract, status: str) -> tuple[str, ...]:
    if not status or not contract.outcome_artifacts:
        return tuple(contract.tracked_artifacts)
    aliases: list[str] = []
    spec = contract.outcome_artifacts.get(status, {})
    if not isinstance(spec, dict):
        return tuple(contract.tracked_artifacts)
    for alias in tuple(spec.get("requires", ())) + tuple(spec.get("optional", ())):
        if alias not in aliases:
            aliases.append(alias)
    return tuple(aliases)


def validate_turn_file_artifact_rules(contract: TurnFileContract, result: TurnFileResult | None = None) -> None:
    status = _turn_file_status_from_result(result)
    if status and contract.outcome_artifacts:
        for alias in _outcome_forbidden_aliases(contract, status):
            path = contract.tracked_artifacts.get(alias)
            if path is not None and _read_text(path):
                raise ValueError(f"turn artifact 包含 forbidden artifact: {alias} -> {path}")
    for alias in _turn_file_selected_aliases(contract, status):
        path = contract.tracked_artifacts.get(alias)
        if path is None:
            continue
        rule = _artifact_rule(contract.artifact_rules, alias)
        if not rule:
            continue
        _enforce_artifact_change_rule(alias=alias, path=path.expanduser().resolve(), rule=rule)


def _build_result_payload(contract: TaskResultContract, result: TaskResultDecision) -> dict[str, Any]:
    return {
        "schema_version": TASK_RESULT_SCHEMA_VERSION,
        "turn_id": contract.turn_id,
        "phase": contract.phase,
        "task_kind": contract.task_kind,
        "status": result.status,
        "summary": result.summary,
        "artifacts": result.artifacts,
        "artifact_hashes": result.artifact_hashes,
        "written_at": _now_iso(),
    }


def _generic_task_summary(contract: TaskResultContract, status: str) -> str:
    if status == TASK_RESULT_COMPLETED:
        return f"{contract.mode} 已按文件契约完成"
    if status == TASK_RESULT_HITL:
        return f"{contract.mode} 已按文件契约进入 HITL"
    if status == TASK_RESULT_READY:
        return f"{contract.mode} 已按文件契约准备就绪"
    if status in {TASK_RESULT_REVIEW_PASS, TASK_RESULT_REVIEW_FAIL}:
        return f"{contract.mode} 已按文件契约完成评审"
    return f"{contract.mode} 已按文件契约产出 {status}"


def _resolve_outcome_scoped_decision(contract: TaskResultContract) -> TaskResultDecision | None:
    if not contract.outcome_artifacts:
        return None
    if set(contract.expected_statuses) <= {TASK_RESULT_REVIEW_PASS, TASK_RESULT_REVIEW_FAIL}:
        return None
    all_artifacts = _contract_artifact_paths(contract)
    _, present_aliases, _ = _collect_observed_artifact_texts(all_artifacts)
    validation_errors: list[str] = []
    for status in contract.expected_statuses:
        spec = _outcome_artifact_spec(contract, status)
        if not spec:
            continue
        required_aliases = _outcome_required_aliases(contract, status)
        forbidden_aliases = _outcome_forbidden_aliases(contract, status)
        missing = tuple(alias for alias in required_aliases if alias not in present_aliases)
        if missing:
            continue
        forbidden_present = tuple(alias for alias in forbidden_aliases if alias in present_aliases)
        if forbidden_present:
            validation_errors.append(f"{status} 分支存在 forbidden artifact: {', '.join(forbidden_present)}")
            continue
        try:
            artifacts, artifact_hashes = collect_contract_artifacts(
                contract,
                aliases=_outcome_collection_aliases(contract, status),
                required_aliases=required_aliases,
            )
        except Exception as error:  # noqa: BLE001
            validation_errors.append(str(error))
            continue
        return TaskResultDecision(
            status=status,
            summary=_generic_task_summary(contract, status),
            artifacts=artifacts,
            artifact_hashes=artifact_hashes,
        )

    if validation_errors:
        raise ValueError("; ".join(validation_errors))
    return None


def resolve_task_result_decision(contract: TaskResultContract) -> TaskResultDecision:
    mode = contract.mode
    outcome_decision = _resolve_outcome_scoped_decision(contract)
    if outcome_decision is not None:
        return outcome_decision
    all_artifacts = _contract_artifact_paths(contract)
    artifacts, artifact_hashes = collect_contract_artifacts(contract)

    if mode == "a03_ba_resume":
        return TaskResultDecision(
            status=TASK_RESULT_READY,
            summary="需求分析师已进入需求评审准备态",
            artifacts=artifacts,
            artifact_hashes=artifact_hashes,
        )

    if mode == "a05_ba_init":
        return TaskResultDecision(
            status=TASK_RESULT_READY,
            summary="需求分析师已完成详细设计初始化",
            artifacts=artifacts,
            artifact_hashes=artifact_hashes,
        )

    if mode == "a06_ba_init":
        return TaskResultDecision(
            status=TASK_RESULT_READY,
            summary="任务拆分阶段智能体已完成初始化",
            artifacts=artifacts,
            artifact_hashes=artifact_hashes,
        )

    if mode == "a06_reviewer_init":
        return TaskResultDecision(
            status=TASK_RESULT_READY,
            summary="任务拆分阶段审核智能体已完成初始化",
            artifacts=artifacts,
            artifact_hashes=artifact_hashes,
        )

    if mode == "a05_detailed_design_generate":
        if _read_text(all_artifacts.get("detailed_design")):
            return TaskResultDecision(
                status=TASK_RESULT_COMPLETED,
                summary="需求分析师已生成详细设计文档",
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
            )
        raise ValueError("详细设计文档未生成有效内容")

    if mode in {"a05_detailed_design_feedback", "a05_detailed_design_review_limit_human_reply"}:
        ask_human_text = _read_text(all_artifacts.get("ask_human"))
        ba_feedback_text = _read_text(all_artifacts.get("ba_feedback"))
        detailed_design_text = _read_text(all_artifacts.get("detailed_design"))
        if ask_human_text:
            return TaskResultDecision(
                status=TASK_RESULT_HITL,
                summary="需求分析师需要继续向人类澄清详细设计问题",
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
            )
        if ba_feedback_text and detailed_design_text:
            return TaskResultDecision(
                status=TASK_RESULT_COMPLETED,
                summary="需求分析师已完成详细设计修订",
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
            )
        raise ValueError("详细设计反馈未生成有效 HITL 问题或修订结果")

    if mode == "a05_detailed_design_review_limit_force_hitl":
        if _read_text(all_artifacts.get("ask_human")):
            return TaskResultDecision(
                status=TASK_RESULT_HITL,
                summary="需求分析师需要继续向人类澄清详细设计问题",
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
            )
        raise ValueError("详细设计评审超限后未写入《与人类交流.md》")

    if mode == "a06_task_split_generate":
        if _read_text(all_artifacts.get("task_md")):
            return TaskResultDecision(
                status=TASK_RESULT_COMPLETED,
                summary="需求分析师已生成任务单文档",
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
            )
        raise ValueError("任务单文档未生成有效内容")

    if mode in {"a06_task_split_feedback", "a06_task_split_review_limit_human_reply"}:
        ask_human_text = _read_text(all_artifacts.get("ask_human"))
        ba_feedback_text = _read_text(all_artifacts.get("ba_feedback"))
        task_md_text = _read_text(all_artifacts.get("task_md"))
        if ask_human_text:
            return TaskResultDecision(
                status=TASK_RESULT_HITL,
                summary="需求分析师需要继续向人类澄清任务拆分问题",
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
            )
        if ba_feedback_text and task_md_text:
            return TaskResultDecision(
                status=TASK_RESULT_COMPLETED,
                summary="需求分析师已完成任务单修订",
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
            )
        raise ValueError("任务拆分反馈未生成有效修订结果")

    if mode == "a06_task_split_review_limit_force_hitl":
        if _read_text(all_artifacts.get("ask_human")):
            return TaskResultDecision(
                status=TASK_RESULT_HITL,
                summary="需求分析师需要继续向人类澄清任务拆分问题",
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
            )
        raise ValueError("任务拆分评审超限后未写入《与人类交流.md》")

    if mode == "a06_task_split_json_generate":
        from T01_tools import is_standard_task_initial_json

        task_json_path = _resolve_optional_path(all_artifacts.get("task_json"))
        if task_json_path is None or not is_standard_task_initial_json(task_json_path):
            raise ValueError("任务单 JSON 未生成有效初始结构")
        return TaskResultDecision(
            status=TASK_RESULT_COMPLETED,
            summary="需求分析师已生成任务单 JSON",
            artifacts=artifacts,
            artifact_hashes=artifact_hashes,
        )

    if mode in {"a07_developer_init", "a07_developer_human_reply"}:
        if _read_text(all_artifacts.get("ask_human")):
            return TaskResultDecision(
                status=TASK_RESULT_HITL,
                summary="开发工程师需要人类补充阻断信息",
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
            )
        return TaskResultDecision(
            status=TASK_RESULT_READY,
            summary="开发工程师已完成预研并准备就绪",
            artifacts=artifacts,
            artifact_hashes=artifact_hashes,
        )

    if mode == "a07_reviewer_init":
        return TaskResultDecision(
            status=TASK_RESULT_READY,
            summary="代码评审智能体已完成初始化",
            artifacts=artifacts,
            artifact_hashes=artifact_hashes,
        )

    if mode in {"a08_developer_init", "a08_reviewer_init"}:
        return TaskResultDecision(
            status=TASK_RESULT_READY,
            summary="智能体已完成复核阶段初始化",
            artifacts=artifacts,
            artifact_hashes=artifact_hashes,
        )

    if mode in {"a07_developer_task_complete", "a07_developer_refine"}:
        developer_output_text = _read_text(all_artifacts.get("developer_output"))
        if developer_output_text:
            summary = "开发工程师已完成当前任务实现" if mode == "a07_developer_task_complete" else "开发工程师已完成当前任务修订"
            return TaskResultDecision(
                status=TASK_RESULT_COMPLETED,
                summary=summary,
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
            )
        raise ValueError("开发工程师未生成有效开发元数据")

    if mode == "a08_developer_refine_all_code":
        developer_output_text = _read_text(all_artifacts.get("developer_output"))
        if developer_output_text:
            return TaskResultDecision(
                status=TASK_RESULT_COMPLETED,
                summary="开发工程师已完成复核阶段代码修订",
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
            )
        raise ValueError("开发工程师未生成有效复核阶段开发元数据")

    if mode in {"a07_developer_review_feedback", "a07_developer_review_limit_human_reply"}:
        ask_human_text = _read_text(all_artifacts.get("ask_human"))
        developer_output_text = _read_text(all_artifacts.get("developer_output"))
        if ask_human_text:
            return TaskResultDecision(
                status=TASK_RESULT_HITL,
                summary="开发工程师在评审超限后需要继续等待人类决策",
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
            )
        if developer_output_text:
            return TaskResultDecision(
                status=TASK_RESULT_COMPLETED,
                summary="开发工程师已根据人类反馈继续完成任务开发",
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
            )
        raise ValueError("开发工程师评审反馈未生成有效 HITL 问题或开发结果")

    if mode == "a07_developer_review_limit_force_hitl":
        if _read_text(all_artifacts.get("ask_human")):
            return TaskResultDecision(
                status=TASK_RESULT_HITL,
                summary="开发工程师在评审超限后需要继续等待人类决策",
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
            )
        raise ValueError("开发阶段评审超限后未写入《与人类交流.md》")

    if mode == "a03_reviewer_round":
        review_json = _resolve_optional_path(all_artifacts.get("review_json"))
        review_md = _resolve_optional_path(all_artifacts.get("review_md"))
        if review_json is None or review_md is None:
            raise ValueError("a03_reviewer_round 缺少 review_json/review_md")
        payload = json.loads(review_json.read_text(encoding="utf-8"))
        matched_item = normalize_review_status_payload(payload, task_name="需求评审", source=str(review_json))
        review_pass = matched_item["review_pass"]
        review_md_empty = not review_md.read_text(encoding="utf-8").strip()
        if review_pass and not review_md_empty:
            raise ValueError("审核通过时 reviewer md 必须为空")
        if (not review_pass) and review_md_empty:
            raise ValueError("审核未通过时 reviewer md 必须非空")
        status = TASK_RESULT_REVIEW_PASS if review_pass else TASK_RESULT_REVIEW_FAIL
        summary = "审核器已完成需求评审" if review_pass else "审核器已写入未通过评审意见"
        return TaskResultDecision(
            status=status,
            summary=summary,
            artifacts=artifacts,
            artifact_hashes=artifact_hashes,
        )

    if mode in {"a03_ba_feedback", "a03_ba_review_limit_human_reply"}:
        if _read_text(all_artifacts.get("ask_human")):
            return TaskResultDecision(
                status=TASK_RESULT_HITL,
                summary="需求分析师需要继续向人类澄清",
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
            )
        if _read_text(all_artifacts.get("ba_feedback")):
            return TaskResultDecision(
                status=TASK_RESULT_COMPLETED,
                summary="需求分析师已完成评审反馈修订",
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
            )
        raise ValueError("需求分析师反馈未生成 HITL 问题或修改反馈")

    if mode == "a03_ba_review_limit_force_hitl":
        if _read_text(all_artifacts.get("ask_human")):
            return TaskResultDecision(
                status=TASK_RESULT_HITL,
                summary="需求分析师需要继续向人类澄清",
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
            )
        raise ValueError("需求评审超限后未写入《与人类交流.md》")

    if mode == "a03_human_feedback":
        if _read_text(all_artifacts.get("ask_human")):
            return TaskResultDecision(
                status=TASK_RESULT_COMPLETED,
                summary="需求分析师已完成对人类反馈的处理",
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
            )
        raise ValueError("需求分析师未写入《与人类交流.md》")

    if mode in {"a02_requirements_analysis", "a03_requirements_clarification"}:
        requirements_clear_text = _read_text(all_artifacts.get("requirements_clear"))
        ask_human_text = _read_text(all_artifacts.get("ask_human"))
        hitl_record_text = _read_text(all_artifacts.get("hitl_record"))
        if requirements_clear_text:
            return TaskResultDecision(
                status=TASK_RESULT_COMPLETED,
                summary="需求澄清已完成",
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
            )
        if ask_human_text and hitl_record_text:
            return TaskResultDecision(
                status=TASK_RESULT_HITL,
                summary="需求澄清需要继续 HITL",
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
            )
        raise ValueError("需求澄清未生成有效结果文件")

    if mode in {"a02_notion_intake", "a02_requirement_intake"}:
        original_requirement_text = _read_text(all_artifacts.get("original_requirement"))
        ask_human_text = _read_text(all_artifacts.get("ask_human"))
        if original_requirement_text:
            return TaskResultDecision(
                status=TASK_RESULT_COMPLETED,
                summary="Notion 需求录入完成",
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
            )
        if ask_human_text:
            return TaskResultDecision(
                status=TASK_RESULT_HITL,
                summary="Notion 需求录入需要继续 HITL",
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
            )
        raise ValueError("Notion 需求录入未生成有效结果文件")

    if len(contract.expected_statuses) == 1:
        status = contract.expected_statuses[0]
        return TaskResultDecision(
            status=status,
            summary=_generic_task_summary(contract, status),
            artifacts=artifacts,
            artifact_hashes=artifact_hashes,
        )

    raise ValueError(f"不支持的 contract mode: {mode}")


def _collect_observed_artifact_texts(artifact_map: dict[str, Path]) -> tuple[dict[str, str], set[str], set[str]]:
    resolved_paths: dict[str, str] = {}
    present_aliases: set[str] = set()
    empty_aliases: set[str] = set()
    for alias, path in artifact_map.items():
        resolved = Path(path).expanduser().resolve()
        resolved_paths[alias] = str(resolved)
        if not resolved.exists() or not resolved.is_file():
            continue
        text = resolved.read_text(encoding="utf-8").strip()
        if text:
            present_aliases.add(alias)
        else:
            empty_aliases.add(alias)
    return resolved_paths, present_aliases, empty_aliases


def _infer_observed_task_status(contract: TaskResultContract, present_aliases: set[str]) -> str:
    try:
        outcome_decision = _resolve_outcome_scoped_decision(contract)
    except Exception:
        outcome_decision = None
    if outcome_decision is not None:
        return outcome_decision.status
    mode = contract.mode
    if mode in {"a03_ba_review_limit_force_hitl", "a05_detailed_design_review_limit_force_hitl", "a06_task_split_review_limit_force_hitl", "a07_developer_review_limit_force_hitl"}:
        return TASK_RESULT_HITL if "ask_human" in present_aliases else ""
    if mode in {"a03_ba_feedback", "a03_ba_review_limit_human_reply", "a05_detailed_design_feedback", "a05_detailed_design_review_limit_human_reply", "a06_task_split_feedback", "a06_task_split_review_limit_human_reply", "a07_developer_review_feedback", "a07_developer_review_limit_human_reply"}:
        if "ask_human" in present_aliases:
            return TASK_RESULT_HITL
        if {"ba_feedback", "requirements_clear"} <= present_aliases:
            return TASK_RESULT_COMPLETED
        if {"ba_feedback", "detailed_design"} <= present_aliases:
            return TASK_RESULT_COMPLETED
        if {"ba_feedback", "task_md"} <= present_aliases:
            return TASK_RESULT_COMPLETED
        if "developer_output" in present_aliases:
            return TASK_RESULT_COMPLETED
        return ""
    if mode in {"a05_detailed_design_generate"}:
        return TASK_RESULT_COMPLETED if "detailed_design" in present_aliases else ""
    if mode in {"a06_task_split_generate"}:
        return TASK_RESULT_COMPLETED if "task_md" in present_aliases else ""
    if mode in {"a06_task_split_json_generate"}:
        return TASK_RESULT_COMPLETED if "task_json" in present_aliases else ""
    if mode in {"a07_developer_task_complete", "a07_developer_refine"}:
        return TASK_RESULT_COMPLETED if "developer_output" in present_aliases else ""
    if mode in {"a03_ba_resume", "a05_ba_init", "a06_ba_init", "a06_reviewer_init", "a07_reviewer_init"}:
        return TASK_RESULT_READY
    if mode in {"a07_developer_init", "a07_developer_human_reply"}:
        if "ask_human" in present_aliases:
            return TASK_RESULT_HITL
        return TASK_RESULT_READY
    if mode == "a03_human_feedback":
        return TASK_RESULT_COMPLETED if "ask_human" in present_aliases else ""
    if len(contract.expected_statuses) == 1:
        required = set(contract.required_artifacts)
        if required <= present_aliases:
            return contract.expected_statuses[0]
        if not required and contract.expected_statuses[0] == TASK_RESULT_READY:
            return TASK_RESULT_READY
    return ""


def observe_task_result_state(
    contract: TaskResultContract,
    result_path: str | Path,
) -> TaskResultObservation:
    target = Path(result_path).expanduser().resolve()
    all_artifacts = {**contract.required_artifacts, **contract.optional_artifacts}
    artifact_paths, present_aliases, empty_aliases = _collect_observed_artifact_texts(all_artifacts)
    observed_status = ""
    validation_error = ""
    try:
        validated = validate_task_result_file(contract=contract, result_path=target)
        observed_status = str(validated.payload.get("status", "")).strip()
    except Exception as error:  # noqa: BLE001
        validation_error = str(error).strip()
        observed_status = _infer_observed_task_status(contract, present_aliases)
    missing_required = tuple(
        alias
        for alias in contract.required_artifacts
        if alias not in present_aliases and alias not in empty_aliases
    )
    empty_required = tuple(alias for alias in contract.required_artifacts if alias in empty_aliases)
    return TaskResultObservation(
        observed_status=observed_status,
        present_aliases=tuple(sorted(present_aliases)),
        missing_required_aliases=missing_required,
        empty_required_aliases=empty_required,
        artifact_paths=artifact_paths,
        last_validation_error=validation_error,
    )


def observe_completion_state(contract: TurnFileContract) -> CompletionObservation:
    tracked = dict(contract.tracked_artifacts)
    if "status_path" not in tracked:
        tracked["status_path"] = contract.status_path
    artifact_paths, present_aliases, empty_aliases = _collect_observed_artifact_texts(tracked)
    observed_status = ""
    validation_error = ""
    try:
        result = contract.validator(contract.status_path)
        validate_turn_file_artifact_rules(contract, result)
        review_pass = result.payload.get("review_pass")
        if isinstance(review_pass, bool):
            observed_status = TASK_RESULT_REVIEW_PASS if review_pass else TASK_RESULT_REVIEW_FAIL
    except Exception as error:  # noqa: BLE001
        validation_error = str(error).strip()
        review_json_path = tracked.get("review_json")
        if review_json_path is not None and review_json_path.exists():
            try:
                payload = json.loads(review_json_path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
            for item in _iter_review_status_items(payload):
                review_pass = item.get("review_pass")
                if isinstance(review_pass, bool):
                    observed_status = TASK_RESULT_REVIEW_PASS if review_pass else TASK_RESULT_REVIEW_FAIL
                    break
    required_aliases: tuple[str, ...]
    if contract.kind == "review_round":
        required_aliases = ("review_json",)
    else:
        required_aliases = tuple(sorted(alias for alias in tracked if alias != "status_path"))
    missing_required = tuple(alias for alias in required_aliases if alias not in present_aliases and alias not in empty_aliases)
    empty_required = tuple(alias for alias in required_aliases if alias in empty_aliases)
    return CompletionObservation(
        observed_status=observed_status,
        present_aliases=tuple(sorted(present_aliases)),
        missing_required_aliases=missing_required,
        empty_required_aliases=empty_required,
        artifact_paths=artifact_paths,
        last_validation_error=validation_error,
    )


def validate_task_result_file(
    *,
    contract: TaskResultContract,
    result_path: str | Path,
) -> TaskResultFile:
    target = Path(result_path).expanduser().resolve()
    payload = read_task_result_payload(target)
    if not payload:
        raise FileNotFoundError(f"缺少 result.json: {target}")
    if str(payload.get("schema_version", "")).strip() != TASK_RESULT_SCHEMA_VERSION:
        raise ValueError("result.json schema_version 非法")
    if str(payload.get("turn_id", "")).strip() != contract.turn_id:
        raise ValueError("result.json turn_id 非法")
    if str(payload.get("phase", "")).strip() != contract.phase:
        raise ValueError("result.json phase 非法")
    if str(payload.get("task_kind", "")).strip() != contract.task_kind:
        raise ValueError("result.json task_kind 非法")
    status = str(payload.get("status", "")).strip()
    if contract.expected_statuses and status not in contract.expected_statuses:
        raise ValueError(f"result.json status 非法: {status}")
    if not isinstance(payload.get("summary", ""), str):
        raise ValueError("result.json summary 必须是字符串")
    artifacts = payload.get("artifacts", {})
    artifact_hashes = payload.get("artifact_hashes", {})
    if not isinstance(artifacts, dict):
        raise ValueError("result.json artifacts 必须是对象")
    if not isinstance(artifact_hashes, dict):
        raise ValueError("result.json artifact_hashes 必须是对象")
    all_artifacts = _contract_artifact_paths(contract)
    required_aliases = _outcome_required_aliases(contract, status)
    if not contract.outcome_artifacts:
        required_aliases = tuple(contract.required_artifacts)
    for alias in required_aliases:
        required_path = all_artifacts.get(alias)
        if required_path is None:
            raise ValueError(f"result.json contract 缺少 artifact path: {alias}")
        resolved_required = str(required_path.resolve())
        actual_path = str(artifacts.get(alias, "")).strip()
        if not actual_path:
            raise ValueError(f"result.json 缺少必填 artifact: {alias}")
        if str(Path(actual_path).expanduser().resolve()) != resolved_required:
            raise ValueError(f"result.json artifact 路径非法: {alias}")
    for alias in _outcome_forbidden_aliases(contract, status):
        actual_path = str(artifacts.get(alias, "")).strip()
        if actual_path and _read_text(actual_path):
            raise ValueError(f"result.json 包含 forbidden artifact: {alias}")
    for artifact_path in artifacts.values():
        resolved = Path(str(artifact_path)).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"result.json 引用的文件不存在: {resolved}")
        expected_hash = str(artifact_hashes.get(str(resolved), "")).strip()
        if not expected_hash:
            raise ValueError(f"result.json 缺少 artifact_hashes: {resolved}")
        if expected_hash != _build_prefixed_sha256(resolved):
            raise ValueError(f"result.json artifact_hashes 不匹配: {resolved}")
    return TaskResultFile(
        result_path=str(target),
        payload=payload,
        artifact_paths={str(key): str(value) for key, value in artifacts.items()},
        artifact_hashes={str(key): str(value) for key, value in artifact_hashes.items()},
        validated_at=_now_iso(),
    )


def materialize_task_result(
    *,
    contract: TaskResultContract,
    result_path: str | Path,
    status: str,
    summary: str,
    artifacts: dict[str, str] | None = None,
    artifact_hashes: dict[str, str] | None = None,
) -> TaskResultFile:
    if artifacts is None or artifact_hashes is None:
        artifacts, artifact_hashes = collect_contract_artifacts(contract)
    payload = {
        "schema_version": TASK_RESULT_SCHEMA_VERSION,
        "turn_id": contract.turn_id,
        "phase": contract.phase,
        "task_kind": contract.task_kind,
        "status": status,
        "summary": summary,
        "artifacts": artifacts,
        "artifact_hashes": artifact_hashes,
        "written_at": _now_iso(),
    }
    write_task_result_payload(result_path, payload)
    return validate_task_result_file(contract=contract, result_path=result_path)


def finalize_task_result(
    *,
    contract: TaskResultContract,
    result_path: str | Path,
    task_status_path: str | Path | None = None,
) -> TaskResultFile:
    decision = resolve_task_result_decision(contract)
    task_result = materialize_task_result(
        contract=contract,
        result_path=result_path,
        status=decision.status,
        summary=decision.summary,
        artifacts=decision.artifacts,
        artifact_hashes=decision.artifact_hashes,
    )
    if task_status_path is not None:
        write_task_status(task_status_path, status=TASK_STATUS_DONE)
    return task_result


__all__ = [
    "TASK_RESULT_COMPLETED",
    "TASK_RESULT_ERROR",
    "TASK_RESULT_HITL",
    "TASK_RESULT_READY",
    "TASK_RESULT_REVIEW_FAIL",
    "TASK_RESULT_REVIEW_PASS",
    "TASK_STATUS_DONE",
    "TASK_STATUS_RUNNING",
    "CompletionObservation",
    "MissingTaskResultFinalizationCandidate",
    "TaskResultContract",
    "TaskResultDecision",
    "TaskResultFile",
    "TaskResultObservation",
    "TurnFileContract",
    "TurnFileResult",
    "build_missing_task_result_finalization_candidate",
    "collect_contract_artifacts",
    "finalize_task_result",
    "materialize_task_result",
    "normalize_review_status_payload",
    "observe_completion_state",
    "observe_task_result_state",
    "read_task_result_payload",
    "read_task_status",
    "resolve_task_result_decision",
    "snapshot_file_fingerprint",
    "validate_task_result_file",
    "validate_turn_file_artifact_rules",
    "write_task_result_payload",
    "write_task_status",
]
