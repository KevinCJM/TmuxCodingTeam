from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tmux_core.runtime.contracts import (
    CompletionObservation,
    TaskResultContract,
    TaskResultObservation,
    TurnFileContract,
    observe_completion_state,
    observe_task_result_state,
    read_task_result_payload,
)
from tmux_core.runtime.tmux_runtime import (
    DEFAULT_COMMAND_TIMEOUT_SEC,
    TASK_RESULT_CONTRACT_ERROR_PREFIX,
    TURN_ARTIFACT_CONTRACT_ERROR_PREFIX,
    TmuxBatchWorker,
    is_task_result_contract_error,
    is_turn_artifact_contract_error,
)
from tmux_core.stage_kernel.agent_intervention import (
    AGENT_INTERVENTION_WORKER_DEAD,
    request_file_noncompliance_intervention,
)

DEFAULT_TURN_REPAIR_ATTEMPTS = 2


@dataclass(frozen=True)
class OutcomeGoal:
    status: str
    required_aliases: tuple[str, ...] = ()
    optional_aliases: tuple[str, ...] = ()
    forbidden_aliases: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class RepairPromptContext:
    turn_label: str
    stage_label: str
    role_label: str
    observed_status: str
    expected_status: str
    missing_aliases: tuple[str, ...]
    forbidden_aliases: tuple[str, ...]
    present_aliases: tuple[str, ...]
    last_validation_error: str
    artifact_paths: dict[str, str]
    task_name: str = ""
    requirement_name: str = ""


@dataclass(frozen=True)
class TaskTurnGoal:
    goal_id: str
    outcomes: dict[str, OutcomeGoal]
    max_repair_attempts: int = DEFAULT_TURN_REPAIR_ATTEMPTS
    repair_prompt_builder: Callable[[RepairPromptContext], str] | None = None
    allow_repair_on_contract_failure: bool = True


@dataclass(frozen=True)
class CompletionTurnGoal:
    goal_id: str
    outcomes: dict[str, OutcomeGoal]
    max_repair_attempts: int = DEFAULT_TURN_REPAIR_ATTEMPTS
    repair_prompt_builder: Callable[[RepairPromptContext], str] | None = None


@dataclass(frozen=True)
class GoalValidation:
    valid: bool
    expected_status: str
    missing_aliases: tuple[str, ...]
    forbidden_aliases: tuple[str, ...]
    message: str


_INTERNAL_TASK_RESULT_MARKERS = (
    ".development_runtime",
    ".agent_init_runtime",
    ".routing_init_runtime",
    "task_runtime",
    "_result.json",
)


def _is_internal_task_result_error(message: str) -> bool:
    text = str(message or "")
    if not text:
        return False
    if "result.json" not in text and "_result.json" not in text:
        return False
    return "缺少 result.json" in text or "result_path=" in text or any(marker in text for marker in _INTERNAL_TASK_RESULT_MARKERS)


def _sanitize_task_validation_error(message: str) -> str:
    if _is_internal_task_result_error(message):
        return "内部任务结果尚未由 runtime 生成；请只检查上方允许修改的业务文件。"
    return str(message or "").strip()


def _choose_expected_status(
    outcomes: dict[str, OutcomeGoal],
    observed_status: str,
    present_aliases: tuple[str, ...],
) -> str:
    if observed_status in outcomes:
        return observed_status
    if len(outcomes) == 1:
        return next(iter(outcomes))
    present = set(present_aliases)
    best_status = ""
    best_score = -1
    tied = False
    for status, outcome in outcomes.items():
        score = len(present.intersection(outcome.required_aliases))
        if score > best_score:
            best_status = status
            best_score = score
            tied = False
        elif score == best_score:
            tied = True
    if best_score > 0 and not tied:
        return best_status
    return ""


def _validate_observation(
    *,
    outcomes: dict[str, OutcomeGoal],
    observed_status: str,
    present_aliases: tuple[str, ...],
    last_validation_error: str,
) -> GoalValidation:
    expected_status = _choose_expected_status(outcomes, observed_status, present_aliases)
    if not expected_status:
        return GoalValidation(
            valid=False,
            expected_status="",
            missing_aliases=(),
            forbidden_aliases=(),
            message=last_validation_error or "未识别到符合本轮协议的结果分支",
        )
    outcome = outcomes[expected_status]
    present = set(present_aliases)
    missing_aliases = tuple(alias for alias in outcome.required_aliases if alias not in present)
    forbidden_aliases = tuple(alias for alias in outcome.forbidden_aliases if alias in present)
    if not missing_aliases and not forbidden_aliases:
        return GoalValidation(
            valid=True,
            expected_status=expected_status,
            missing_aliases=(),
            forbidden_aliases=(),
            message="",
        )
    details: list[str] = []
    if missing_aliases:
        details.append("缺少: " + ", ".join(missing_aliases))
    if forbidden_aliases:
        details.append("不应保留: " + ", ".join(forbidden_aliases))
    return GoalValidation(
        valid=False,
        expected_status=expected_status,
        missing_aliases=missing_aliases,
        forbidden_aliases=forbidden_aliases,
        message=last_validation_error or "；".join(details),
    )


def build_default_task_repair_prompt(context: RepairPromptContext) -> str:
    lines = [
        "上一轮已经结束，但你遗漏了本轮协议要求的产物。",
        f"阶段: {context.stage_label}",
        f"本轮标签: {context.turn_label}",
        f"当前角色: {context.role_label}",
    ]
    if context.task_name:
        lines.append(f"任务名: {context.task_name}")
    if context.requirement_name:
        lines.append(f"需求名: {context.requirement_name}")
    if context.expected_status:
        lines.append(f"期望结果分支: {context.expected_status}")
    if context.observed_status:
        lines.append(f"当前观察到的分支: {context.observed_status}")
    if context.missing_aliases:
        lines.append("缺失 alias: " + ", ".join(context.missing_aliases))
    if context.forbidden_aliases:
        lines.append("需要清空的 alias: " + ", ".join(context.forbidden_aliases))
    lines.append("允许修改的文件:")
    for alias, path in sorted(context.artifact_paths.items()):
        lines.append(f"- {alias}: {path}")
    validation_error = _sanitize_task_validation_error(context.last_validation_error)
    if validation_error:
        lines.append(f"上次校验错误: {validation_error}")
    lines.extend(
        [
            "只补齐上方允许修改文件中的缺失或错误内容，不要做无关修改。",
            "补齐后仍按本轮原协议返回，不要改终止 token 家族。",
        ]
    )
    return "\n".join(lines)


def build_default_completion_repair_prompt(context: RepairPromptContext) -> str:
    lines = [
        "上一轮评审输出未通过协议校验，需要你补齐或修正评审产物。",
        f"阶段: {context.stage_label}",
        f"本轮标签: {context.turn_label}",
        f"当前角色: {context.role_label}",
    ]
    if context.expected_status:
        lines.append(f"期望评审分支: {context.expected_status}")
    if context.observed_status:
        lines.append(f"当前观察到的分支: {context.observed_status}")
    if context.missing_aliases:
        lines.append("缺失 alias: " + ", ".join(context.missing_aliases))
    if context.forbidden_aliases:
        lines.append("需要清空的 alias: " + ", ".join(context.forbidden_aliases))
    lines.append("允许修改的文件:")
    for alias, path in sorted(context.artifact_paths.items()):
        lines.append(f"- {alias}: {path}")
    if context.last_validation_error:
        lines.append(f"上次校验错误: {context.last_validation_error}")
    lines.extend(
        [
            "只修正评审输出文件，不要重新审题，不要做无关修改。",
            "修正后仍按本轮原协议返回。",
        ]
    )
    return "\n".join(lines)


def _build_repair_context(
    *,
    turn_label: str,
    stage_label: str,
    role_label: str,
    task_name: str,
    requirement_name: str,
    artifact_paths: dict[str, str],
    present_aliases: tuple[str, ...],
    observed_status: str,
    last_validation_error: str,
    validation: GoalValidation,
) -> RepairPromptContext:
    return RepairPromptContext(
        turn_label=turn_label,
        stage_label=stage_label,
        role_label=role_label,
        observed_status=observed_status,
        expected_status=validation.expected_status,
        missing_aliases=validation.missing_aliases,
        forbidden_aliases=validation.forbidden_aliases,
        present_aliases=present_aliases,
        last_validation_error=last_validation_error,
        artifact_paths=artifact_paths,
        task_name=task_name,
        requirement_name=requirement_name,
    )


def _build_task_goal_error(
    *,
    turn_label: str,
    observation: TaskResultObservation,
    validation: GoalValidation,
) -> RuntimeError:
    expected = validation.expected_status or "unknown"
    detail = validation.message or observation.last_validation_error or "缺少必需结果文件"
    missing = ", ".join(validation.missing_aliases) if validation.missing_aliases else "-"
    return RuntimeError(
        f"{TASK_RESULT_CONTRACT_ERROR_PREFIX}: "
        f"turn={turn_label} expected_status={expected} missing_aliases={missing} error={detail}"
    )


def _build_completion_goal_error(
    *,
    turn_label: str,
    observation: CompletionObservation,
    validation: GoalValidation,
) -> RuntimeError:
    expected = validation.expected_status or "unknown"
    detail = validation.message or observation.last_validation_error or "评审产物未按协议更新"
    missing = ", ".join(validation.missing_aliases) if validation.missing_aliases else "-"
    return RuntimeError(
        f"{TURN_ARTIFACT_CONTRACT_ERROR_PREFIX}: "
        f"turn={turn_label} expected_status={expected} missing_aliases={missing} error={detail}"
    )


def run_task_result_turn_with_repair(
    *,
    worker: TmuxBatchWorker,
    label: str,
    prompt: str,
    result_contract: TaskResultContract,
    repair_result_contract: TaskResultContract | None = None,
    parse_result_payload: Callable[[str], dict[str, object]],
    turn_goal: TaskTurnGoal | None = None,
    timeout_sec: float = DEFAULT_COMMAND_TIMEOUT_SEC,
    turn_start_timeout_sec: float | None = None,
    prompt_submit_timeout_sec: float | None = None,
    pre_submit_observation_tail_lines: int | None = None,
    pre_submit_observation_tail_bytes: int | None = None,
    stage_label: str = "",
    role_label: str = "",
    task_name: str = "",
    requirement_name: str = "",
) -> dict[str, object]:
    repair_budget = turn_goal.max_repair_attempts if turn_goal is not None else 0
    repair_prompt_builder = (
        turn_goal.repair_prompt_builder
        if turn_goal is not None and turn_goal.repair_prompt_builder is not None
        else build_default_task_repair_prompt
    )
    current_prompt = prompt
    for repair_attempt in range(0, repair_budget + 1):
        current_label = label if repair_attempt == 0 else f"{label}_repair_{repair_attempt}"
        active_result_contract = (
            result_contract
            if repair_attempt == 0 or repair_result_contract is None
            else repair_result_contract
        )
        run_turn_kwargs: dict[str, object] = {
            "label": current_label,
            "prompt": current_prompt,
            "result_contract": active_result_contract,
            "timeout_sec": timeout_sec,
        }
        if turn_start_timeout_sec is not None:
            run_turn_kwargs["turn_start_timeout_sec"] = turn_start_timeout_sec
        if prompt_submit_timeout_sec is not None:
            run_turn_kwargs["prompt_submit_timeout_sec"] = prompt_submit_timeout_sec
        if pre_submit_observation_tail_lines is not None:
            run_turn_kwargs["pre_submit_observation_tail_lines"] = pre_submit_observation_tail_lines
        if pre_submit_observation_tail_bytes is not None:
            run_turn_kwargs["pre_submit_observation_tail_bytes"] = pre_submit_observation_tail_bytes
        result = worker.run_turn(**run_turn_kwargs)
        current_error: RuntimeError | None = None
        payload: dict[str, object] | None = None
        if result.ok:
            payload = parse_result_payload(result.clean_output)
        else:
            current_error = RuntimeError(result.clean_output or f"{current_label} 执行失败")

        current_task_result_path = str(getattr(worker, "current_task_result_path", "") or "").strip()
        if not current_task_result_path:
            if result.ok:
                return payload or {}
            raise current_error or RuntimeError(f"{current_label} 执行失败")
        result_path = Path(current_task_result_path).expanduser().resolve()
        observation = observe_task_result_state(active_result_contract, result_path)
        validation = (
            _validate_observation(
                outcomes=turn_goal.outcomes,
                observed_status=observation.observed_status,
                present_aliases=observation.present_aliases,
                last_validation_error=observation.last_validation_error,
            )
            if turn_goal is not None
            else GoalValidation(valid=True, expected_status="", missing_aliases=(), forbidden_aliases=(), message="")
        )
        internal_result_error = _is_internal_task_result_error(
            observation.last_validation_error or (str(current_error).strip() if current_error else "")
        )
        if validation.valid and internal_result_error and (
            current_error is None or is_task_result_contract_error(current_error)
        ):
            observed = observation.observed_status or validation.expected_status or "unknown"
            raise RuntimeError(
                f"{TASK_RESULT_CONTRACT_ERROR_PREFIX}: "
                f"turn={current_label} observed_status={observed} "
                "internal task result materialization missing"
            )
        if result.ok and validation.valid:
            return payload or {}
        if validation.valid and not observation.last_validation_error:
            return read_task_result_payload(result_path)
        if turn_goal is None:
            raise current_error or _build_task_goal_error(turn_label=current_label, observation=observation, validation=validation)
        if current_error is not None and not (
            is_task_result_contract_error(current_error) and turn_goal.allow_repair_on_contract_failure
        ):
            raise current_error
        if repair_attempt >= repair_budget:
            terminal_error = current_error or _build_task_goal_error(
                turn_label=current_label,
                observation=observation,
                validation=validation,
            )
            while True:
                target_paths = tuple(observation.artifact_paths.values()) or (str(result_path),)
                decision = request_file_noncompliance_intervention(
                    stage_label=stage_label or active_result_contract.stage_name or active_result_contract.phase,
                    role_label=role_label,
                    worker=worker,
                    reason_text=str(terminal_error),
                    attempts_used=repair_budget,
                    target_paths=target_paths,
                )
                if decision == AGENT_INTERVENTION_WORKER_DEAD:
                    raise RuntimeError(f"tmux pane died after manual file intervention: {terminal_error}") from terminal_error
                observation = observe_task_result_state(active_result_contract, result_path)
                validation = (
                    _validate_observation(
                        outcomes=turn_goal.outcomes,
                        observed_status=observation.observed_status,
                        present_aliases=observation.present_aliases,
                        last_validation_error=observation.last_validation_error,
                    )
                    if turn_goal is not None
                    else GoalValidation(valid=True, expected_status="", missing_aliases=(), forbidden_aliases=(), message="")
                )
                internal_result_error = _is_internal_task_result_error(observation.last_validation_error or "")
                if validation.valid and not internal_result_error:
                    return read_task_result_payload(result_path)
                terminal_error = _build_task_goal_error(
                    turn_label=current_label,
                    observation=observation,
                    validation=validation,
                )
        context = _build_repair_context(
            turn_label=current_label,
            stage_label=stage_label or active_result_contract.stage_name or active_result_contract.phase,
            role_label=role_label,
            task_name=task_name,
            requirement_name=requirement_name,
            artifact_paths=observation.artifact_paths,
            present_aliases=observation.present_aliases,
            observed_status=observation.observed_status,
            last_validation_error=observation.last_validation_error or (str(current_error).strip() if current_error else ""),
            validation=validation,
        )
        current_prompt = repair_prompt_builder(context)
    raise AssertionError(f"unreachable task turn repair state: {label}")


def run_completion_turn_with_repair(
    *,
    worker: TmuxBatchWorker,
    label: str,
    prompt: str,
    completion_contract: TurnFileContract,
    turn_goal: CompletionTurnGoal | None = None,
    timeout_sec: float = DEFAULT_COMMAND_TIMEOUT_SEC,
    turn_start_timeout_sec: float | None = None,
    prompt_submit_timeout_sec: float | None = None,
    pre_submit_observation_tail_lines: int | None = None,
    pre_submit_observation_tail_bytes: int | None = None,
    stage_label: str = "",
    role_label: str = "",
    task_name: str = "",
    requirement_name: str = "",
) -> None:
    repair_budget = turn_goal.max_repair_attempts if turn_goal is not None else 0
    repair_prompt_builder = (
        turn_goal.repair_prompt_builder
        if turn_goal is not None and turn_goal.repair_prompt_builder is not None
        else build_default_completion_repair_prompt
    )
    current_prompt = prompt
    for repair_attempt in range(0, repair_budget + 1):
        current_label = label if repair_attempt == 0 else f"{label}_repair_{repair_attempt}"
        if repair_attempt > 0 and turn_goal is not None:
            observation = observe_completion_state(completion_contract)
            validation = _validate_observation(
                outcomes=turn_goal.outcomes,
                observed_status=observation.observed_status,
                present_aliases=observation.present_aliases,
                last_validation_error=observation.last_validation_error,
            )
            if validation.valid and not observation.last_validation_error:
                return
        run_turn_kwargs: dict[str, object] = {
            "label": current_label,
            "prompt": current_prompt,
            "completion_contract": completion_contract,
            "timeout_sec": timeout_sec,
        }
        if turn_start_timeout_sec is not None:
            run_turn_kwargs["turn_start_timeout_sec"] = turn_start_timeout_sec
        if prompt_submit_timeout_sec is not None:
            run_turn_kwargs["prompt_submit_timeout_sec"] = prompt_submit_timeout_sec
        if pre_submit_observation_tail_lines is not None:
            run_turn_kwargs["pre_submit_observation_tail_lines"] = pre_submit_observation_tail_lines
        if pre_submit_observation_tail_bytes is not None:
            run_turn_kwargs["pre_submit_observation_tail_bytes"] = pre_submit_observation_tail_bytes
        result = worker.run_turn(**run_turn_kwargs)
        current_error = RuntimeError(result.clean_output or f"{current_label} 执行失败") if not result.ok else None
        observation = observe_completion_state(completion_contract)
        validation = (
            _validate_observation(
                outcomes=turn_goal.outcomes,
                observed_status=observation.observed_status,
                present_aliases=observation.present_aliases,
                last_validation_error=observation.last_validation_error,
            )
            if turn_goal is not None
            else GoalValidation(valid=True, expected_status="", missing_aliases=(), forbidden_aliases=(), message="")
        )
        if validation.valid and observation.last_validation_error:
            validation = GoalValidation(
                valid=False,
                expected_status=validation.expected_status,
                missing_aliases=validation.missing_aliases,
                forbidden_aliases=validation.forbidden_aliases,
                message=observation.last_validation_error,
            )
        if result.ok and validation.valid:
            return
        if validation.valid and not observation.last_validation_error:
            return
        if turn_goal is None:
            raise current_error or _build_completion_goal_error(
                turn_label=current_label,
                observation=observation,
                validation=validation,
            )
        if current_error is not None and not is_turn_artifact_contract_error(current_error):
            raise current_error
        if repair_attempt >= repair_budget:
            terminal_error = current_error or _build_completion_goal_error(
                turn_label=current_label,
                observation=observation,
                validation=validation,
            )
            while True:
                decision = request_file_noncompliance_intervention(
                    stage_label=stage_label or completion_contract.phase,
                    role_label=role_label,
                    worker=worker,
                    reason_text=str(terminal_error),
                    attempts_used=repair_budget,
                    target_paths=tuple(observation.artifact_paths.values()),
                )
                if decision == AGENT_INTERVENTION_WORKER_DEAD:
                    raise RuntimeError(f"tmux pane died after manual file intervention: {terminal_error}") from terminal_error
                observation = observe_completion_state(completion_contract)
                validation = (
                    _validate_observation(
                        outcomes=turn_goal.outcomes,
                        observed_status=observation.observed_status,
                        present_aliases=observation.present_aliases,
                        last_validation_error=observation.last_validation_error,
                    )
                    if turn_goal is not None
                    else GoalValidation(valid=True, expected_status="", missing_aliases=(), forbidden_aliases=(), message="")
                )
                if validation.valid and not observation.last_validation_error:
                    return
                terminal_error = _build_completion_goal_error(
                    turn_label=current_label,
                    observation=observation,
                    validation=validation,
                )
        context = _build_repair_context(
            turn_label=current_label,
            stage_label=stage_label or completion_contract.phase,
            role_label=role_label,
            task_name=task_name,
            requirement_name=requirement_name,
            artifact_paths=observation.artifact_paths,
            present_aliases=observation.present_aliases,
            observed_status=observation.observed_status,
            last_validation_error=observation.last_validation_error or (str(current_error).strip() if current_error else ""),
            validation=validation,
        )
        current_prompt = repair_prompt_builder(context)
    raise AssertionError(f"unreachable completion turn repair state: {label}")


__all__ = [
    "CompletionTurnGoal",
    "DEFAULT_TURN_REPAIR_ATTEMPTS",
    "OutcomeGoal",
    "RepairPromptContext",
    "TaskTurnGoal",
    "build_default_completion_repair_prompt",
    "build_default_task_repair_prompt",
    "run_completion_turn_with_repair",
    "run_task_result_turn_with_repair",
]
