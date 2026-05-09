from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Sequence, TypeVar

from tmux_core.runtime.tmux_runtime import (
    worker_state_has_launch_evidence,
    worker_state_is_prelaunch_active,
)
from tmux_core.stage_kernel.agent_intervention import (
    AGENT_INTERVENTION_RECREATE,
    AGENT_INTERVENTION_WORKER_DEAD,
    request_file_noncompliance_intervention,
)

TReviewer = TypeVar("TReviewer")


def _resolve_worker(owner: object | None):
    if owner is None:
        return None
    worker = getattr(owner, "worker", None)
    return worker if worker is not None else owner


def _owner_is_dead(owner: object | None) -> bool:
    worker = _resolve_worker(owner)
    if worker is None:
        return False
    if worker_state_is_prelaunch_active(worker) or not worker_state_has_launch_evidence(worker):
        return False
    get_agent_state = getattr(worker, "get_agent_state", None)
    if callable(get_agent_state):
        try:
            state = get_agent_state()
            if str(getattr(state, "value", state) or "").strip().upper() == "DEAD":
                return True
        except Exception:
            pass
    read_state = getattr(worker, "read_state", None)
    if callable(read_state):
        try:
            payload = read_state()
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            return (
                str(payload.get("agent_state", "") or "").strip().upper() == "DEAD"
                or str(payload.get("health_status", "") or "").strip().lower() in {"dead", "missing_session", "pane_dead"}
            )
    return False


def run_parallel_reviewer_round(
    reviewers: Sequence[TReviewer],
    *,
    key_func: Callable[[TReviewer], str],
    run_turn: Callable[[TReviewer], TReviewer | None],
    error_prefix: str,
) -> list[TReviewer]:
    reviewer_list = list(reviewers)
    if not reviewer_list:
        return reviewer_list
    reviewer_index = {key_func(item): index for index, item in enumerate(reviewer_list)}
    dropped_keys: set[str] = set()
    with ThreadPoolExecutor(max_workers=max(1, len(reviewer_list))) as executor:
        future_map = {
            executor.submit(run_turn, reviewer): key_func(reviewer)
            for reviewer in reviewer_list
        }
        errors: list[str] = []
        for future in as_completed(future_map):
            reviewer_key = future_map[future]
            try:
                result = future.result()
                if result is None:
                    dropped_keys.add(reviewer_key)
                    continue
                reviewer_list[reviewer_index[reviewer_key]] = result
            except Exception as error:  # noqa: BLE001
                errors.append(f"{reviewer_key}: {error}")
        if errors:
            raise RuntimeError(error_prefix + "\n" + "\n".join(errors))
    return [reviewer for reviewer in reviewer_list if key_func(reviewer) not in dropped_keys]


def repair_reviewer_round_outputs(
    reviewers: Sequence[TReviewer],
    *,
    key_func: Callable[[TReviewer], str],
    artifact_name_func: Callable[[TReviewer], str],
    check_job: Callable[[Sequence[str]], dict[str, str]],
    run_fix_turn: Callable[[TReviewer, str, int], TReviewer | None],
    max_attempts: int,
    error_prefix: str,
    final_error: str,
    stage_label: str = "",
    progress: object | None = None,
    recreate_reviewer: Callable[[TReviewer], TReviewer | None] | None = None,
) -> list[TReviewer]:
    reviewer_list = list(reviewers)
    if not reviewer_list:
        return reviewer_list
    reviewer_index = {key_func(item): index for index, item in enumerate(reviewer_list)}
    for repair_attempt in range(1, max_attempts + 1):
        prompts = check_job([artifact_name_func(item) for item in reviewer_list])
        if not prompts:
            return reviewer_list
        with ThreadPoolExecutor(max_workers=max(1, len(prompts))) as executor:
            future_map = {}
            for reviewer in reviewer_list:
                fix_prompt = prompts.get(artifact_name_func(reviewer))
                if not fix_prompt:
                    continue
                future_map[executor.submit(run_fix_turn, reviewer, fix_prompt, repair_attempt)] = key_func(reviewer)
            errors: list[str] = []
            dropped_keys: set[str] = set()
            for future in as_completed(future_map):
                reviewer_key = future_map[future]
                try:
                    result = future.result()
                    if result is None:
                        dropped_keys.add(reviewer_key)
                        continue
                    reviewer_list[reviewer_index[reviewer_key]] = result
                except Exception as error:  # noqa: BLE001
                    errors.append(f"{reviewer_key}: {error}")
            if errors:
                raise RuntimeError(error_prefix + "\n" + "\n".join(errors))
            if dropped_keys:
                reviewer_list = [item for item in reviewer_list if key_func(item) not in dropped_keys]
                reviewer_index = {key_func(item): index for index, item in enumerate(reviewer_list)}
                if not reviewer_list:
                    return reviewer_list
    prompts = check_job([artifact_name_func(item) for item in reviewer_list])
    while prompts:
        affected_reviewers = [
            item for item in reviewer_list
            if artifact_name_func(item) in prompts
        ]
        target_paths: list[str] = []
        for reviewer in affected_reviewers:
            for attr in ("review_md_path", "review_json_path"):
                path = getattr(reviewer, attr, "")
                if str(path or "").strip():
                    target_paths.append(str(Path(path).expanduser().resolve()))
        reason_text = "\n\n".join(
            f"[{name}]\n{prompt}" for name, prompt in prompts.items()
        ) or final_error
        representative = affected_reviewers[0] if affected_reviewers else (reviewer_list[0] if reviewer_list else None)
        decision = request_file_noncompliance_intervention(
            stage_label=stage_label or "审核阶段",
            role_label=artifact_name_func(representative) if representative is not None else "审核智能体",
            worker=_resolve_worker(representative),
            reason_text=reason_text,
            attempts_used=max_attempts,
            target_paths=target_paths,
            progress=progress,
            allow_recreate=recreate_reviewer is not None,
        )
        if decision == AGENT_INTERVENTION_RECREATE and recreate_reviewer is not None and representative is not None:
            representative_key = key_func(representative)
            replacement = recreate_reviewer(representative)
            if replacement is None:
                raise RuntimeError(f"{artifact_name_func(representative)} 重新创建失败，无法继续修复审核输出")
            if representative_key in reviewer_index:
                reviewer_list[reviewer_index[representative_key]] = replacement
            else:
                reviewer_list.append(replacement)
            reviewer_index = {key_func(item): index for index, item in enumerate(reviewer_list)}
            replacement_name = artifact_name_func(replacement)
            replacement_prompts = check_job([replacement_name])
            fix_prompt = replacement_prompts.get(replacement_name)
            if fix_prompt:
                fixed_reviewer = run_fix_turn(replacement, fix_prompt, max_attempts + 1)
                if fixed_reviewer is None:
                    reviewer_list = [item for item in reviewer_list if key_func(item) != key_func(replacement)]
                else:
                    reviewer_list[reviewer_index[key_func(replacement)]] = fixed_reviewer
                reviewer_index = {key_func(item): index for index, item in enumerate(reviewer_list)}
            prompts = check_job([artifact_name_func(item) for item in reviewer_list])
            continue
        if decision == AGENT_INTERVENTION_WORKER_DEAD:
            survivors = [item for item in reviewer_list if not _owner_is_dead(item)]
            if len(survivors) != len(reviewer_list):
                return survivors
            raise RuntimeError(f"tmux pane died after manual reviewer repair intervention: {final_error}")
        prompts = check_job([artifact_name_func(item) for item in reviewer_list])
    return reviewer_list


def shutdown_stage_workers(
    ba_handoff,
    reviewers: Sequence,
    *,
    cleanup_runtime: bool,
    preserve_ba_worker: bool = False,
    preserve_reviewer_keys: Sequence[str] = (),
    runtime_root_filter: str | Path | None = None,
) -> tuple[str, ...]:
    removed: list[str] = []
    seen_runtime_dirs: set[Path] = set()
    runtime_roots: set[Path] = set()
    preserved_reviewer_keys = {
        str(item).strip()
        for item in preserve_reviewer_keys
        if str(item).strip()
    }
    runtime_root_constraint = (
        Path(runtime_root_filter).expanduser().resolve()
        if runtime_root_filter is not None
        else None
    )
    for reviewer in reviewers:
        reviewer_key = str(getattr(reviewer, "reviewer_name", "") or "").strip()
        if reviewer_key and reviewer_key in preserved_reviewer_keys:
            continue
        reviewer_runtime_dir = Path(reviewer.worker.runtime_dir).expanduser().resolve()
        reviewer_runtime_root = Path(reviewer.worker.runtime_root).expanduser().resolve()
        if runtime_root_constraint is None or reviewer_runtime_root == runtime_root_constraint:
            if cleanup_runtime:
                try:
                    reviewer.worker.request_kill()
                except Exception:
                    pass
            seen_runtime_dirs.add(reviewer_runtime_dir)
            runtime_roots.add(reviewer_runtime_root)
    if ba_handoff is not None and not preserve_ba_worker:
        ba_runtime_dir = Path(ba_handoff.worker.runtime_dir).expanduser().resolve()
        ba_runtime_root = Path(ba_handoff.worker.runtime_root).expanduser().resolve()
        if runtime_root_constraint is None or ba_runtime_root == runtime_root_constraint:
            if cleanup_runtime:
                try:
                    ba_handoff.worker.request_kill()
                except Exception:
                    pass
            seen_runtime_dirs.add(ba_runtime_dir)
            runtime_roots.add(ba_runtime_root)
    if not cleanup_runtime:
        return ()
    for runtime_dir in seen_runtime_dirs:
        if runtime_dir.exists():
            shutil.rmtree(runtime_dir, ignore_errors=True)
            removed.append(str(runtime_dir))
    for runtime_root in runtime_roots:
        if runtime_root.exists() and runtime_root.is_dir() and not any(runtime_root.iterdir()):
            runtime_root.rmdir()
            removed.append(str(runtime_root))
    return tuple(removed)
