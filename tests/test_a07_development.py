from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from A07_Development import (
    A07_PRE_SUBMIT_OBSERVATION_TAIL_BYTES,
    A07_PRE_SUBMIT_OBSERVATION_TAIL_LINES,
    A07_REVIEWER_PROMPT_SUBMIT_TIMEOUT_SEC,
    A07_REVIEWER_TURN_START_TIMEOUT_SEC,
    DevelopmentReviewerSpec,
    DevelopmentStageResult,
    DeveloperPlan,
    DeveloperRuntime,
    _development_review_human_override_requested,
    _build_reviewer_protocol_repair_prompt,
    _build_reviewer_turn_goal,
    _shutdown_workers as shutdown_development_workers,
    _run_developer_result_turn,
    _run_parallel_reviewers,
    _replace_dead_developer,
    _run_single_reviewer_initialization,
    _replace_dead_developer_with_bootstrap,
    initialize_developer_with_parallel_reviewer_prelaunch,
    prelaunch_development_reviewers,
    build_reviewer_completion_contract,
    build_developer_review_feedback_result_contract,
    build_developer_init_prompt,
    build_development_paths,
    build_development_runtime_root,
    build_parser,
    build_reviewer_artifact_paths,
    build_reviewer_init_prompt,
    cleanup_stale_development_runtime_state,
    collect_interactive_reviewer_specs,
    create_developer_runtime,
    create_reviewer_runtime,
    develop_current_task,
    apply_development_review_human_override,
    ensure_developer_metadata,
    ensure_development_inputs,
    initialize_development_workers,
    prepare_review_round_artifacts,
    repair_reviewer_outputs,
    recreate_developer_runtime,
    recreate_development_workers,
    refine_current_task,
    recreate_development_reviewer_runtime,
    resolve_developer_max_turns,
    resolve_developer_role_prompt,
    resolve_review_max_rounds,
    run_development_stage,
    run_developer_hitl_loop,
    run_reviewer_turn_with_recreation,
    fintech_developer_role,
)
from tmux_core.runtime.contracts import TaskResultContract, TurnFileContract, TurnFileResult
from tmux_core.runtime.tmux_runtime import CommandResult, TURN_ARTIFACT_CONTRACT_ERROR_PREFIX
from tmux_core.prompt_contracts.spec import CHANGE_MUST_CHANGE
from tmux_core.stage_kernel.turn_output_goals import RepairPromptContext, run_completion_turn_with_repair
from tmux_core.stage_kernel.shared_review import (
    AGENT_READY_TIMEOUT_RETRY,
    AGENT_READY_TIMEOUT_SKIP,
    ReviewAgentSelection,
    ReviewerRuntime,
)
from tmux_core.stage_kernel.agent_intervention import (
    AGENT_INTERVENTION_RECHECK,
    AGENT_INTERVENTION_RECREATE,
    AGENT_INTERVENTION_WORKER_DEAD,
)


class _FakeWorker:
    def __init__(
        self,
        *,
        session_name: str,
        runtime_root: str | Path = "/tmp/runtime",
        runtime_dir: str | Path = "/tmp/runtime/worker",
        session_exists_value: bool = True,
    ) -> None:
        self.session_name = session_name
        self.runtime_root = Path(runtime_root)
        self.runtime_dir = Path(runtime_dir)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self._session_exists_value = session_exists_value
        self.killed = False

    def request_kill(self):
        self.killed = True
        return self.session_name

    def session_exists(self) -> bool:
        return self._session_exists_value


class _ReconfigurableWorker(_FakeWorker):
    def __init__(self, *, session_name: str):
        super().__init__(session_name=session_name)
        self.reconfig_reason = ""

    def mark_awaiting_reconfiguration(self, *, reason_text: str) -> None:
        self.reconfig_reason = reason_text


class _LiveReadyWorker(_FakeWorker):
    def read_state(self):
        return {
            "status": "failed",
            "result_status": "failed",
            "agent_state": "READY",
            "health_status": "alive",
        }

    def get_agent_state(self):
        return type("State", (), {"value": "READY"})()


class _RecordingStateWorker(_FakeWorker):
    def __init__(self, *, session_name: str, task_status_path: Path):
        super().__init__(session_name=session_name)
        self.task_status_path = task_status_path
        self.state_payload = {
            "status": "failed",
            "result_status": "failed",
            "agent_state": "READY",
            "health_status": "alive",
            "current_task_status_path": str(task_status_path),
            "current_turn_id": "development_review_M1-T1_审核员",
        }
        self.recorded_status = ""
        self.recorded_note = ""
        self.recorded_extra: dict[str, object] = {}

    def read_state(self):
        return dict(self.state_payload)

    def _write_state(self, status, *, note: str, extra=None):  # noqa: ANN001
        self.recorded_status = str(getattr(status, "value", status))
        self.recorded_note = note
        self.recorded_extra = dict(extra or {})
        self.state_payload.update({"status": self.recorded_status, "note": note, **self.recorded_extra})


class _ManualRecheckStateWorker(_RecordingStateWorker):
    def __init__(self, *, session_name: str, task_status_path: Path):
        super().__init__(session_name=session_name, task_status_path=task_status_path)
        self.write_history: list[tuple[str, str, dict[str, object]]] = []
        self.state_payload.update(
            {
                "note": "awaiting_reconfig",
                "health_status": "awaiting_reconfig",
                "health_note": "prompt confirm timeout",
                "dispatch_state": "delayed",
                "dispatch_reason": "prompt_confirm_timeout:slow prompt echo",
                "current_task_runtime_status": "running",
                "result_status": "failed",
            }
        )

    def _write_state(self, status, *, note: str, extra=None):  # noqa: ANN001
        status_text = str(getattr(status, "value", status))
        extra_payload = dict(extra or {})
        self.write_history.append((status_text, note, extra_payload))
        super()._write_state(status, note=note, extra=extra_payload)


class _FreshReviewerWorker:
    def __init__(self, *, session_name: str) -> None:
        self.session_name = session_name
        self.ensure_calls = 0
        self._launched = False

    def get_agent_state(self):
        state = "READY" if self._launched else "DEAD"
        return type("State", (), {"value": state})()

    def has_ever_launched(self) -> bool:
        return self._launched

    def ensure_agent_ready(self, timeout_sec: float = 0.0) -> None:
        _ = timeout_sec
        self.ensure_calls += 1
        self._launched = True


def _dummy_contract() -> TurnFileContract:
    def validator(path: Path) -> TurnFileResult:
        return TurnFileResult(
            status_path=str(path.resolve()),
            payload={"ok": True},
            artifact_paths={},
            artifact_hashes={},
            validated_at="0",
        )

    return TurnFileContract(
        turn_id="dummy",
        phase="dummy",
        status_path=Path("/tmp/dummy.json"),
        validator=validator,
    )


def _dummy_task_result_contract() -> TaskResultContract:
    return TaskResultContract(
        turn_id="dummy_task",
        phase="dummy",
        task_kind="dummy",
        mode="dummy",
        expected_statuses=("completed",),
    )


def _write_required_inputs(paths: dict[str, Path]) -> None:
    paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
    paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
    paths["detailed_design_path"].write_text("详细设计\n", encoding="utf-8")


class A07DevelopmentTests(unittest.TestCase):
    def test_code_review_reviewer_count_prompt_allows_previous_step_back(self):
        from T09_terminal_ops import BridgePromptRequest, BridgeTerminalUI, PROMPT_BACK_VALUE, PromptBackRequested, use_terminal_ui

        captured_requests: list[BridgePromptRequest] = []

        def request_prompt(request: BridgePromptRequest) -> dict[str, object]:
            captured_requests.append(request)
            return {"value": PROMPT_BACK_VALUE}

        ui = BridgeTerminalUI(emit_event=lambda *_args, **_kwargs: None, request_prompt=request_prompt)

        with use_terminal_ui(ui), self.assertRaises(PromptBackRequested):
            collect_interactive_reviewer_specs(allow_back_first_prompt=True)

        self.assertEqual(len(captured_requests), 1)
        self.assertEqual(captured_requests[0].prompt_type, "text")
        self.assertEqual(captured_requests[0].payload["prompt_text"], "请输入代码评审智能体数量")
        self.assertTrue(captured_requests[0].payload["allow_back"])
        self.assertEqual(captured_requests[0].payload["back_value"], PROMPT_BACK_VALUE)
        self.assertEqual(captured_requests[0].payload["stage_key"], "development_reviewer_specs")
        self.assertEqual(captured_requests[0].payload["stage_step_index"], 0)

    def test_code_review_reviewer_role_prompts_allow_back_after_count(self):
        from T09_terminal_ops import BridgePromptRequest, BridgeTerminalUI, PROMPT_BACK_VALUE, use_terminal_ui

        captured_requests: list[BridgePromptRequest] = []

        def request_prompt(request: BridgePromptRequest) -> dict[str, object]:
            captured_requests.append(request)
            if len(captured_requests) == 1:
                return {"value": "1"}
            if len(captured_requests) == 2:
                return {"value": "default"}
            raise RuntimeError("stop after default role prompt")

        ui = BridgeTerminalUI(emit_event=lambda *_args, **_kwargs: None, request_prompt=request_prompt)

        with use_terminal_ui(ui), self.assertRaisesRegex(RuntimeError, "stop after default role prompt"):
            collect_interactive_reviewer_specs()

        self.assertEqual(captured_requests[1].prompt_type, "select")
        self.assertEqual(captured_requests[1].payload["title"], "第 1 个审核智能体 - 角色定义来源")
        self.assertTrue(captured_requests[1].payload["allow_back"])
        self.assertEqual(captured_requests[1].payload["back_value"], PROMPT_BACK_VALUE)
        self.assertEqual(captured_requests[1].payload["stage_step_index"], 1)
        self.assertEqual(captured_requests[2].prompt_type, "select")
        self.assertEqual(captured_requests[2].payload["title"], "第 1 个审核智能体 - 选择默认角色定义")
        self.assertTrue(captured_requests[2].payload["allow_back"])
        self.assertEqual(captured_requests[2].payload["back_value"], PROMPT_BACK_VALUE)
        self.assertEqual(captured_requests[2].payload["stage_step_index"], 2)

    def test_repair_reviewer_outputs_accepts_progress_object(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            review_md_path, review_json_path = build_reviewer_artifact_paths(project_dir, "需求A", "测试工程师-天寿星")
            worker = _FakeWorker(
                session_name="测试工程师-天寿星",
                runtime_root=project_dir / ".runtime",
                runtime_dir=project_dir / ".runtime" / "worker",
            )
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("codex", "gpt", "high", ""),
                worker=worker,
                review_md_path=review_md_path,
                review_json_path=review_json_path,
                contract=_dummy_contract(),
            )
            progress = object()
            captured: dict[str, object] = {}

            def fake_repair(reviewer_list, **kwargs):
                captured["progress"] = kwargs.get("progress")
                return list(reviewer_list)

            with patch("A07_Development.repair_reviewer_round_outputs", side_effect=fake_repair):
                result = repair_reviewer_outputs(
                    [reviewer],
                    paths=paths,
                    reviewer_specs_by_name={"测试工程师": DevelopmentReviewerSpec(role_name="测试工程师", role_prompt="测试")},
                    project_dir=project_dir,
                    requirement_name="需求A",
                    task_name="M1-T3",
                    round_index=1,
                    progress=progress,
                )

        self.assertEqual(result, [reviewer])
        self.assertIs(captured["progress"], progress)

    def test_initialization_prompts_do_not_append_task_routing_assessment(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = build_development_paths(Path(tmp_dir), "需求A")
            reviewer_spec = DevelopmentReviewerSpec(
                role_name="测试工程师",
                role_prompt="审核角色",
                reviewer_key="测试工程师",
            )

            developer_prompt = build_developer_init_prompt(paths, role_prompt="开发角色")
            reviewer_prompt = build_reviewer_init_prompt(paths, reviewer_spec=reviewer_spec)

            for prompt in (developer_prompt, reviewer_prompt):
                self.assertIn("准备就绪", prompt)
                self.assertIn("禁止", prompt)
                self.assertNotIn("Task Routing Assessment", prompt)
                self.assertNotIn("Output the Task Routing Assessment first", prompt)

    def test_prepare_review_round_artifacts_precreates_empty_review_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            paths["merged_review_path"].write_text("stale", encoding="utf-8")
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="测试工程师-天英星"),
                review_md_path=project_dir / "需求A_代码评审记录_测试工程师.md",
                review_json_path=project_dir / "需求A_评审记录_测试工程师.json",
                contract=_dummy_contract(),
            )
            reviewer.review_md_path.write_text("stale", encoding="utf-8")
            reviewer.review_json_path.write_text('{"stale": true}', encoding="utf-8")

            prepare_review_round_artifacts(paths, [reviewer])

            self.assertEqual(paths["merged_review_path"].read_text(encoding="utf-8"), "")
            self.assertEqual(reviewer.review_md_path.read_text(encoding="utf-8"), "")
            self.assertEqual(json.loads(reviewer.review_json_path.read_text(encoding="utf-8")), [])

    def test_prepare_review_round_artifacts_preserves_existing_valid_task_review(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            paths["merged_review_path"].write_text("stale", encoding="utf-8")
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("opencode", "opencode/big-pickle", "high", ""),
                worker=_FakeWorker(session_name="测试工程师-天寿星"),
                review_md_path=project_dir / "需求A_代码评审记录_测试工程师-天寿星.md",
                review_json_path=project_dir / "需求A_评审记录_测试工程师-天寿星.json",
                contract=_dummy_contract(),
            )
            reviewer.review_md_path.write_text("", encoding="utf-8")
            reviewer.review_json_path.write_text(
                json.dumps([{"task_name": "M1-T3", "review_pass": True}], ensure_ascii=False),
                encoding="utf-8",
            )

            prepare_review_round_artifacts(paths, [reviewer], task_name="M1-T3")

            self.assertEqual(paths["merged_review_path"].read_text(encoding="utf-8"), "")
            self.assertEqual(json.loads(reviewer.review_json_path.read_text(encoding="utf-8")), [{"task_name": "M1-T3", "review_pass": True}])

    def test_prepare_review_round_artifacts_can_force_clear_existing_task_review(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            paths["merged_review_path"].write_text("stale", encoding="utf-8")
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("opencode", "opencode/big-pickle", "high", ""),
                worker=_FakeWorker(session_name="测试工程师-天寿星"),
                review_md_path=project_dir / "需求A_代码评审记录_测试工程师-天寿星.md",
                review_json_path=project_dir / "需求A_评审记录_测试工程师-天寿星.json",
                contract=_dummy_contract(),
            )
            reviewer.review_md_path.write_text("旧失败结论", encoding="utf-8")
            reviewer.review_json_path.write_text(
                json.dumps([{"task_name": "M1-T3", "review_pass": False}], ensure_ascii=False),
                encoding="utf-8",
            )

            prepare_review_round_artifacts(
                paths,
                [reviewer],
                task_name="M1-T3",
                preserve_existing_outputs=False,
            )

            self.assertEqual(paths["merged_review_path"].read_text(encoding="utf-8"), "")
            self.assertEqual(reviewer.review_md_path.read_text(encoding="utf-8"), "")
            self.assertEqual(json.loads(reviewer.review_json_path.read_text(encoding="utf-8")), [])

    def test_development_review_human_override_detector_requires_scope_and_context(self):
        self.assertTrue(
            _development_review_human_override_requested(
                "M3-T1 已人工验收通过；评审环境不可复现时按人工验收通过，review_pass 置为 true。",
                task_name="M3-T1",
            )
        )
        self.assertTrue(
            _development_review_human_override_requested(
                "这是评审误报。当前 task_counter.py 已通过 python3 -m py_compile。请忽略该失败评审，将 M3-T1 视为通过并继续。",
                task_name="M3-T1",
            )
        )
        self.assertTrue(
            _development_review_human_override_requested(
                "选择 A。继续沿用既有决策：该任务语法错误为评审误报，已通过 py_compile。请忽略该类重复失败评审，将 M3-T1 视为通过。",
                task_name="M3-T1",
            )
        )
        self.assertTrue(
            _development_review_human_override_requested(
                "M4-T2 评审中的 Unicode 新口径为本次范围外、非行动项，不新增需求，不阻塞当前任务，继续后续任务。",
                task_name="M4-T2",
            )
        )
        self.assertFalse(
            _development_review_human_override_requested(
                "已人工验收通过。",
                task_name="M3-T1",
            )
        )
        self.assertFalse(
            _development_review_human_override_requested(
                "M3-T2 已人工验收通过；评审环境不可复现。",
                task_name="M3-T1",
            )
        )

    def test_apply_development_review_human_override_marks_reviewers_passed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            merged_review_path = project_dir / "需求A_代码评审记录.md"
            merged_review_path.write_text("阻断意见", encoding="utf-8")
            reviewer_a = SimpleNamespace(
                review_md_path=project_dir / "需求A_代码评审记录_审核员A.md",
                review_json_path=project_dir / "需求A_评审记录_审核员A.json",
            )
            reviewer_b = SimpleNamespace(
                review_md_path=project_dir / "需求A_代码评审记录_审核员B.md",
                review_json_path=project_dir / "需求A_评审记录_审核员B.json",
            )
            reviewer_a.review_md_path.write_text("pytest 环境失败", encoding="utf-8")
            reviewer_b.review_md_path.write_text("旧意见", encoding="utf-8")
            reviewer_a.review_json_path.write_text(
                json.dumps([{"task_name": "M3-T1", "review_pass": False}], ensure_ascii=False),
                encoding="utf-8",
            )
            reviewer_b.review_json_path.write_text("[]", encoding="utf-8")

            apply_development_review_human_override(
                task_name="M3-T1",
                reviewer_workers=[reviewer_a, reviewer_b],
                merged_review_path=merged_review_path,
            )

            for reviewer in (reviewer_a, reviewer_b):
                self.assertEqual(reviewer.review_md_path.read_text(encoding="utf-8"), "")
                review_payload = json.loads(reviewer.review_json_path.read_text(encoding="utf-8"))
                self.assertIn({"task_name": "M3-T1", "review_pass": True}, review_payload)
            self.assertEqual(merged_review_path.read_text(encoding="utf-8"), "")

    def test_recreate_development_reviewer_runtime_retires_superseded_runtime(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            runtime_root = build_development_runtime_root(project_dir, "需求A")
            old_runtime_dir = runtime_root / "development-review-old"
            new_runtime_dir = runtime_root / "development-review-new"
            old_runtime_dir.mkdir(parents=True, exist_ok=True)
            new_runtime_dir.mkdir(parents=True, exist_ok=True)
            reviewer = ReviewerRuntime(
                reviewer_name="审核员",
                selection=ReviewAgentSelection("gemini", "flash", "high", "http://127.0.0.1:10900"),
                worker=_FakeWorker(
                    session_name="审核员-地辟星",
                    runtime_root=runtime_root,
                    runtime_dir=old_runtime_dir,
                ),
                review_md_path=project_dir / "需求A_代码评审记录_审核员.md",
                review_json_path=project_dir / "需求A_评审记录_审核员.json",
                contract=_dummy_contract(),
            )
            replacement = ReviewerRuntime(
                reviewer_name="审核员",
                selection=ReviewAgentSelection("gemini", "flash", "high", "http://127.0.0.1:10900"),
                worker=_FakeWorker(
                    session_name="审核员-地阖星",
                    runtime_root=runtime_root,
                    runtime_dir=new_runtime_dir,
                ),
                review_md_path=project_dir / "需求A_代码评审记录_审核员.md",
                review_json_path=project_dir / "需求A_评审记录_审核员.json",
                contract=_dummy_contract(),
            )

            with patch("A07_Development.stdin_is_interactive", return_value=True), patch(
                "A07_Development.prompt_replacement_review_agent_selection",
                return_value=reviewer.selection,
            ) as prompt_replace, patch("A07_Development.create_reviewer_runtime", return_value=replacement):
                returned = recreate_development_reviewer_runtime(
                    project_dir=project_dir,
                    requirement_name="需求A",
                    reviewer=reviewer,
                    reviewer_spec=DevelopmentReviewerSpec(
                        role_name="审核员",
                        role_prompt="复核视角",
                        reviewer_key="审核员",
                    ),
                    force_model_change=False,
                )
            self.assertIs(returned, replacement)
            prompt_replace.assert_called_once()
            self.assertTrue(reviewer.worker.killed)
            self.assertFalse(old_runtime_dir.exists())
            self.assertTrue(new_runtime_dir.exists())

    def test_reviewer_turn_rebuilds_prompt_after_replacement(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            old_reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("opencode", "opencode/big-pickle", "high", ""),
                worker=_FakeWorker(session_name="测试工程师-旧星"),
                review_md_path=project_dir / "需求A_代码评审记录_测试工程师-旧星.md",
                review_json_path=project_dir / "需求A_评审记录_测试工程师-旧星.json",
                contract=_dummy_contract(),
            )
            new_reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=old_reviewer.selection,
                worker=_FakeWorker(session_name="测试工程师-新星"),
                review_md_path=project_dir / "需求A_代码评审记录_测试工程师-新星.md",
                review_json_path=project_dir / "需求A_评审记录_测试工程师-新星.json",
                contract=_dummy_contract(),
            )
            prompts: list[str] = []

            def death_then_success(**kwargs):  # noqa: ANN001
                prompts.append(kwargs["prompt"])
                if len(prompts) == 1:
                    raise RuntimeError("tmux pane died")
                return {}

            with patch(
                "A07_Development.run_completion_turn_with_repair",
                side_effect=death_then_success,
            ), patch(
                "A07_Development.request_worker_manual_intervention",
                return_value=AGENT_INTERVENTION_RECREATE,
            ), patch(
                "A07_Development.recreate_development_reviewer_runtime",
                return_value=new_reviewer,
            ), patch(
                "A07_Development._run_single_reviewer_initialization",
                return_value=new_reviewer,
            ):
                result = run_reviewer_turn_with_recreation(
                    old_reviewer,
                    project_dir=project_dir,
                    requirement_name="需求A",
                    task_name="M3-T1",
                    reviewer_spec=DevelopmentReviewerSpec(
                        role_name="测试工程师",
                        role_prompt="测试视角",
                        reviewer_key="测试工程师",
                    ),
                    paths=paths,
                    reviewer_specs_by_name={
                        "测试工程师": DevelopmentReviewerSpec(
                            role_name="测试工程师",
                            role_prompt="测试视角",
                            reviewer_key="测试工程师",
                        )
                    },
                    label="development_review_again_M3-T1_测试工程师_round_2",
                    prompt_builder=lambda reviewer: f"write {reviewer.review_json_path.name}",
                )

        self.assertIs(result, new_reviewer)
        self.assertEqual(prompts, ["write 需求A_评审记录_测试工程师-旧星.json", "write 需求A_评审记录_测试工程师-新星.json"])

    def test_live_reviewer_non_death_failure_does_not_prompt_recreation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            reviewer_spec = DevelopmentReviewerSpec(
                role_name="测试工程师",
                role_prompt="测试视角",
                reviewer_key="测试工程师",
            )
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("opencode", "opencode/big-pickle", "high", ""),
                worker=_LiveReadyWorker(session_name="测试工程师-天寿星"),
                review_md_path=project_dir / "需求A_代码评审记录_测试工程师-天寿星.md",
                review_json_path=project_dir / "需求A_评审记录_测试工程师-天寿星.json",
                contract=_dummy_contract(),
            )

            with patch(
                "A07_Development.run_completion_turn_with_repair",
                side_effect=RuntimeError("协议输出不合规"),
            ), patch(
                "A07_Development.request_worker_manual_intervention",
                return_value=AGENT_INTERVENTION_WORKER_DEAD,
            ) as prompt_recovery, patch("A07_Development.recreate_development_reviewer_runtime") as recreate_runtime:
                result = run_reviewer_turn_with_recreation(
                    reviewer,
                    project_dir=project_dir,
                    requirement_name="需求A",
                    task_name="M3-T1",
                    reviewer_spec=reviewer_spec,
                    paths=paths,
                    reviewer_specs_by_name={"测试工程师": reviewer_spec},
                    label="development_review_M3-T1_测试工程师",
                )

        recreate_runtime.assert_not_called()
        prompt_recovery.assert_called_once()
        self.assertIsNone(result)

    def test_live_reviewer_initialization_failure_prompts_before_ignore(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            reviewer_spec = DevelopmentReviewerSpec(
                role_name="测试工程师",
                role_prompt="测试视角",
                reviewer_key="测试工程师",
            )
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("opencode", "opencode/big-pickle", "high", ""),
                worker=_LiveReadyWorker(session_name="测试工程师-天寿星"),
                review_md_path=project_dir / "需求A_代码评审记录_测试工程师-天寿星.md",
                review_json_path=project_dir / "需求A_评审记录_测试工程师-天寿星.json",
                contract=_dummy_contract(),
            )

            with patch(
                "A07_Development.run_task_result_turn_with_repair",
                side_effect=RuntimeError("tmux pane died"),
            ), patch(
                "A07_Development.request_worker_manual_intervention",
                return_value=AGENT_INTERVENTION_WORKER_DEAD,
            ) as prompt_recovery, patch("A07_Development.recreate_development_reviewer_runtime") as recreate_runtime:
                result = _run_single_reviewer_initialization(
                    reviewer,
                    project_dir=project_dir,
                    requirement_name="需求A",
                    paths=paths,
                    reviewer_specs_by_name={"测试工程师": reviewer_spec},
                )

        self.assertIsNone(result)
        self.assertTrue(reviewer.worker.killed)
        prompt_recovery.assert_called_once()
        recreate_runtime.assert_not_called()

    def test_single_reviewer_initialization_recheck_clears_awaiting_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            task_status_path = project_dir / "review_init_task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            reviewer_spec = DevelopmentReviewerSpec(
                role_name="测试工程师",
                role_prompt="测试视角",
                reviewer_key="测试工程师",
            )
            worker = _ManualRecheckStateWorker(session_name="测试工程师-天寿星", task_status_path=task_status_path)
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("opencode", "opencode/big-pickle", "high", ""),
                worker=worker,
                review_md_path=project_dir / "需求A_代码评审记录_测试工程师-天寿星.md",
                review_json_path=project_dir / "需求A_评审记录_测试工程师-天寿星.json",
                contract=_dummy_contract(),
            )

            with patch(
                "A07_Development.run_task_result_turn_with_repair",
                side_effect=[RuntimeError("tmux pane died"), {}],
            ) as run_turn, patch(
                "A07_Development._prompt_development_reviewer_recovery",
                return_value=AGENT_INTERVENTION_RECHECK,
            ) as prompt_recovery:
                result = _run_single_reviewer_initialization(
                    reviewer,
                    project_dir=project_dir,
                    requirement_name="需求A",
                    paths=paths,
                    reviewer_specs_by_name={"测试工程师": reviewer_spec},
                )

        self.assertIsNotNone(result)
        self.assertEqual(result.failure_streak, 0)
        self.assertEqual(run_turn.call_count, 2)
        prompt_recovery.assert_called_once()
        self.assertTrue(any(note == "manual_recheck" for _, note, _ in worker.write_history))
        self.assertEqual(worker.state_payload["note"], "manual_recheck")
        self.assertEqual(worker.state_payload["health_status"], "alive")
        self.assertEqual(worker.state_payload["dispatch_state"], "")
        self.assertEqual(worker.state_payload["dispatch_reason"], "")

    def test_single_reviewer_initialization_uses_fast_dispatch_budget(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            reviewer_spec = DevelopmentReviewerSpec(
                role_name="测试工程师",
                role_prompt="测试视角",
                reviewer_key="测试工程师",
            )
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("opencode", "opencode/big-pickle", "high", ""),
                worker=_LiveReadyWorker(session_name="测试工程师-天寿星"),
                review_md_path=project_dir / "需求A_代码评审记录_测试工程师-天寿星.md",
                review_json_path=project_dir / "需求A_评审记录_测试工程师-天寿星.json",
                contract=_dummy_contract(),
            )

            with patch("A07_Development.run_task_result_turn_with_repair", return_value={}) as run_turn:
                result = _run_single_reviewer_initialization(
                    reviewer,
                    project_dir=project_dir,
                    requirement_name="需求A",
                    paths=paths,
                    reviewer_specs_by_name={"测试工程师": reviewer_spec},
                )

        self.assertIs(result, reviewer)
        run_turn.assert_called_once()
        self.assertEqual(run_turn.call_args.kwargs["turn_start_timeout_sec"], A07_REVIEWER_TURN_START_TIMEOUT_SEC)
        self.assertEqual(
            run_turn.call_args.kwargs["prompt_submit_timeout_sec"],
            A07_REVIEWER_PROMPT_SUBMIT_TIMEOUT_SEC,
        )
        self.assertEqual(
            run_turn.call_args.kwargs["pre_submit_observation_tail_lines"],
            A07_PRE_SUBMIT_OBSERVATION_TAIL_LINES,
        )
        self.assertEqual(
            run_turn.call_args.kwargs["pre_submit_observation_tail_bytes"],
            A07_PRE_SUBMIT_OBSERVATION_TAIL_BYTES,
        )

    def test_prelaunch_development_reviewers_uses_fast_ready_timeout_and_logs_slow_reviewer(self):
        class SlowPrelaunchWorker(_FakeWorker):
            def __init__(self, *, session_name: str):
                super().__init__(session_name=session_name)
                self.ensure_timeouts: list[float] = []
                self.events: list[tuple[str, dict[str, object]]] = []

            def ensure_agent_ready(self, timeout_sec=60.0):
                self.ensure_timeouts.append(timeout_sec)
                raise RuntimeError("Timed out waiting for agent ready")

            def _log_event(self, event, **payload):  # noqa: ANN001
                self.events.append((event, payload))

        worker = SlowPrelaunchWorker(session_name="测试工程师-天寿星")
        reviewer = ReviewerRuntime(
            reviewer_name="测试工程师",
            selection=ReviewAgentSelection("opencode", "opencode/big-pickle", "high", ""),
            worker=worker,
            review_md_path=Path("/tmp/review.md"),
            review_json_path=Path("/tmp/review.json"),
            contract=_dummy_contract(),
        )
        notices: list[str] = []

        returned = prelaunch_development_reviewers([reviewer], notify=notices.append)

        self.assertEqual(returned, [reviewer])
        self.assertEqual(worker.ensure_timeouts, [A07_REVIEWER_TURN_START_TIMEOUT_SEC])
        self.assertEqual(worker.events[0][0], "reviewer_prelaunch_ready_timeout")
        self.assertEqual(worker.events[0][1]["reviewer_name"], "测试工程师")
        self.assertEqual(worker.events[0][1]["timeout_sec"], A07_REVIEWER_TURN_START_TIMEOUT_SEC)
        self.assertIn(f"timeout={A07_REVIEWER_TURN_START_TIMEOUT_SEC}s", notices[0])

    def test_initialize_developer_prelaunch_passes_fast_timeout(self):
        developer = DeveloperRuntime(
            selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
            worker=_ReconfigurableWorker(session_name="开发工程师-天魁星"),
            role_prompt="实现视角",
        )
        reviewer = ReviewerRuntime(
            reviewer_name="测试工程师",
            selection=ReviewAgentSelection("opencode", "opencode/big-pickle", "high", ""),
            worker=_FakeWorker(session_name="测试工程师-天寿星"),
            review_md_path=Path("/tmp/review.md"),
            review_json_path=Path("/tmp/review.json"),
            contract=_dummy_contract(),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            with patch(
                "A07_Development.initialize_development_workers",
                return_value=(developer, []),
            ), patch(
                "A07_Development.prelaunch_development_reviewers",
                return_value=[reviewer],
            ) as prelaunch:
                returned_developer, returned_reviewers = initialize_developer_with_parallel_reviewer_prelaunch(
                    developer,
                    project_dir=project_dir,
                    requirement_name="需求A",
                    paths=paths,
                    reviewers=[reviewer],
                    reviewer_specs_by_name={
                        "测试工程师": DevelopmentReviewerSpec(
                            role_name="测试工程师",
                            role_prompt="测试视角",
                            reviewer_key="测试工程师",
                        )
                    },
                )

        self.assertIs(returned_developer, developer)
        self.assertEqual(returned_reviewers, [reviewer])
        self.assertEqual(prelaunch.call_args.kwargs["timeout_sec"], A07_REVIEWER_TURN_START_TIMEOUT_SEC)

    def test_reviewer_turn_contract_error_accepts_late_valid_output_and_rewrites_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            task_status_path = project_dir / "review_task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            reviewer_spec = DevelopmentReviewerSpec(
                role_name="测试工程师",
                role_prompt="测试视角",
                reviewer_key="测试工程师",
            )
            worker = _RecordingStateWorker(session_name="测试工程师-天寿星", task_status_path=task_status_path)
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("opencode", "opencode/big-pickle", "high", ""),
                worker=worker,
                review_md_path=project_dir / "需求A_代码评审记录_测试工程师-天寿星.md",
                review_json_path=project_dir / "需求A_评审记录_测试工程师-天寿星.json",
                contract=_dummy_contract(),
            )

            def write_late_output(**kwargs):  # noqa: ANN001
                _ = kwargs
                reviewer.review_md_path.write_text("", encoding="utf-8")
                reviewer.review_json_path.write_text(
                    json.dumps([{"task_name": "M1-T1", "review_pass": True}], ensure_ascii=False),
                    encoding="utf-8",
                )
                raise RuntimeError(f"{TURN_ARTIFACT_CONTRACT_ERROR_PREFIX}: late output")

            with patch("A07_Development.run_completion_turn_with_repair", side_effect=write_late_output):
                result = run_reviewer_turn_with_recreation(
                    reviewer,
                    project_dir=project_dir,
                    requirement_name="需求A",
                    task_name="M1-T1",
                    reviewer_spec=reviewer_spec,
                    paths=paths,
                    reviewer_specs_by_name={"测试工程师": reviewer_spec},
                    label="development_review_M1-T1_测试工程师",
                    allow_existing_outputs=False,
                )
            task_status_payload = json.loads(task_status_path.read_text(encoding="utf-8"))

        self.assertIs(result, reviewer)
        self.assertEqual(worker.recorded_status, "succeeded")
        self.assertEqual(worker.recorded_extra["result_status"], "succeeded")
        self.assertEqual(worker.recorded_extra["current_task_runtime_status"], "done")
        self.assertEqual(task_status_payload, {"status": "done"})

    def test_reviewer_recheck_clears_awaiting_state_and_accepts_fixed_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            task_status_path = project_dir / "review_task_status.json"
            task_status_path.write_text('{"status": "running"}', encoding="utf-8")
            reviewer_spec = DevelopmentReviewerSpec(
                role_name="测试工程师",
                role_prompt="测试视角",
                reviewer_key="测试工程师",
            )
            worker = _ManualRecheckStateWorker(session_name="测试工程师-天寿星", task_status_path=task_status_path)
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("opencode", "opencode/big-pickle", "high", ""),
                worker=worker,
                review_md_path=project_dir / "需求A_代码评审记录_测试工程师-天寿星.md",
                review_json_path=project_dir / "需求A_评审记录_测试工程师-天寿星.json",
                contract=_dummy_contract(),
            )

            def fix_outputs_and_recheck(*args, **kwargs):  # noqa: ANN002, ANN003
                _ = args, kwargs
                reviewer.review_md_path.write_text("", encoding="utf-8")
                reviewer.review_json_path.write_text(
                    json.dumps([{"task_name": "M1-T2", "review_pass": True}], ensure_ascii=False),
                    encoding="utf-8",
                )
                return AGENT_INTERVENTION_RECHECK

            with patch(
                "A07_Development.run_completion_turn_with_repair",
                side_effect=RuntimeError("tmux pane died"),
            ) as run_turn, patch(
                "A07_Development._prompt_development_reviewer_recovery",
                side_effect=fix_outputs_and_recheck,
            ) as prompt_recovery:
                result = run_reviewer_turn_with_recreation(
                    reviewer,
                    project_dir=project_dir,
                    requirement_name="需求A",
                    task_name="M1-T2",
                    reviewer_spec=reviewer_spec,
                    paths=paths,
                    reviewer_specs_by_name={"测试工程师": reviewer_spec},
                    label="development_review_M1-T2_测试工程师",
                    allow_existing_outputs=False,
                )

        self.assertIsNotNone(result)
        self.assertEqual(result.failure_streak, 0)
        self.assertEqual(run_turn.call_count, 1)
        prompt_recovery.assert_called_once()
        self.assertTrue(any(note == "manual_recheck" for _, note, _ in worker.write_history))
        self.assertEqual(worker.state_payload["health_status"], "alive")
        self.assertEqual(worker.state_payload["dispatch_state"], "")
        self.assertEqual(worker.state_payload["dispatch_reason"], "")
        self.assertEqual(worker.recorded_status, "succeeded")

    def test_reviewer_recovery_reason_identifies_incomplete_failed_review_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            reviewer_spec = DevelopmentReviewerSpec(
                role_name="测试工程师",
                role_prompt="测试视角",
                reviewer_key="测试工程师",
            )
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("opencode", "opencode/big-pickle", "high", ""),
                worker=_LiveReadyWorker(session_name="测试工程师-天寿星"),
                review_md_path=project_dir / "需求A_代码评审记录_测试工程师-天寿星.md",
                review_json_path=project_dir / "需求A_评审记录_测试工程师-天寿星.json",
                contract=_dummy_contract(),
            )
            reasons: list[str] = []

            def write_incomplete_outputs(**kwargs):  # noqa: ANN001
                _ = kwargs
                reviewer.review_md_path.write_text("", encoding="utf-8")
                reviewer.review_json_path.write_text(
                    json.dumps([{"task_name": "M1-T5", "review_pass": False}], ensure_ascii=False),
                    encoding="utf-8",
                )
                raise RuntimeError("协议输出不合规")

            def capture_reason(reviewer_arg, *, reason_text, **kwargs):  # noqa: ANN001
                _ = reviewer_arg, kwargs
                reasons.append(reason_text)
                return AGENT_INTERVENTION_WORKER_DEAD

            with patch(
                "A07_Development.run_completion_turn_with_repair",
                side_effect=write_incomplete_outputs,
            ), patch("A07_Development.try_resume_worker", return_value=False), patch(
                "A07_Development._prompt_development_reviewer_recovery",
                side_effect=capture_reason,
            ):
                result = run_reviewer_turn_with_recreation(
                    reviewer,
                    project_dir=project_dir,
                    requirement_name="需求A",
                    task_name="M1-T5",
                    reviewer_spec=reviewer_spec,
                    paths=paths,
                    reviewer_specs_by_name={"测试工程师": reviewer_spec},
                    label="development_review_M1-T5_测试工程师",
                    allow_existing_outputs=False,
                )

        self.assertIsNone(result)
        self.assertEqual(len(reasons), 1)
        self.assertIn("评审文件不完整", reasons[0])
        self.assertIn("review_pass=false", reasons[0])
        self.assertIn("Markdown 为空", reasons[0])

    def test_reviewer_turn_reuses_existing_valid_output_without_rerun(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            reviewer_spec = DevelopmentReviewerSpec(
                role_name="测试工程师",
                role_prompt="测试视角",
                reviewer_key="测试工程师",
            )
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("opencode", "opencode/big-pickle", "high", ""),
                worker=_LiveReadyWorker(session_name="测试工程师-天寿星"),
                review_md_path=project_dir / "需求A_代码评审记录_测试工程师-天寿星.md",
                review_json_path=project_dir / "需求A_评审记录_测试工程师-天寿星.json",
                contract=_dummy_contract(),
            )
            reviewer.review_md_path.write_text("", encoding="utf-8")
            reviewer.review_json_path.write_text(
                json.dumps([{"task_name": "M1-T3", "review_pass": True}], ensure_ascii=False),
                encoding="utf-8",
            )

            with patch(
                "A07_Development.run_completion_turn_with_repair",
                side_effect=AssertionError("should not rerun completed reviewer"),
            ):
                result = run_reviewer_turn_with_recreation(
                    reviewer,
                    project_dir=project_dir,
                    requirement_name="需求A",
                    task_name="M1-T3",
                    reviewer_spec=reviewer_spec,
                    paths=paths,
                    reviewer_specs_by_name={"测试工程师": reviewer_spec},
                    label="development_review_M1-T3_测试工程师",
                )

        self.assertIs(result, reviewer)

    def test_reviewer_turn_can_force_rerun_existing_valid_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            reviewer_spec = DevelopmentReviewerSpec(
                role_name="测试工程师",
                role_prompt="测试视角",
                reviewer_key="测试工程师",
            )
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("opencode", "opencode/big-pickle", "high", ""),
                worker=_LiveReadyWorker(session_name="测试工程师-天寿星"),
                review_md_path=project_dir / "需求A_代码评审记录_测试工程师-天寿星.md",
                review_json_path=project_dir / "需求A_评审记录_测试工程师-天寿星.json",
                contract=_dummy_contract(),
            )
            reviewer.review_md_path.write_text("", encoding="utf-8")
            reviewer.review_json_path.write_text(
                json.dumps([{"task_name": "M1-T3", "review_pass": True}], ensure_ascii=False),
                encoding="utf-8",
            )

            with patch("A07_Development.run_completion_turn_with_repair", return_value={}) as run_turn:
                result = run_reviewer_turn_with_recreation(
                    reviewer,
                    project_dir=project_dir,
                    requirement_name="需求A",
                    task_name="M1-T3",
                    reviewer_spec=reviewer_spec,
                    paths=paths,
                    reviewer_specs_by_name={"测试工程师": reviewer_spec},
                    label="development_review_again_M1-T3_测试工程师_round_2",
                    allow_existing_outputs=False,
                )

            self.assertIs(result, reviewer)
            run_turn.assert_called_once()
            self.assertEqual(run_turn.call_args.kwargs["turn_start_timeout_sec"], A07_REVIEWER_TURN_START_TIMEOUT_SEC)
            self.assertEqual(
                run_turn.call_args.kwargs["prompt_submit_timeout_sec"],
                A07_REVIEWER_PROMPT_SUBMIT_TIMEOUT_SEC,
            )
            self.assertEqual(reviewer.review_md_path.read_text(encoding="utf-8"), "")
            self.assertEqual(json.loads(reviewer.review_json_path.read_text(encoding="utf-8")), [])

    def test_parallel_reviewer_round_fast_dispatch_submits_all_ready_reviewers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            expected_count = 6
            sent_calls: list[dict[str, object]] = []
            sent_lock = threading.Lock()
            all_sent = threading.Event()

            class ReadyReviewerWorker(_FakeWorker):
                def run_turn(self, **kwargs):  # noqa: ANN001
                    with sent_lock:
                        sent_calls.append({"session_name": self.session_name, **kwargs})
                        if len(sent_calls) == expected_count:
                            all_sent.set()
                    if not all_sent.wait(timeout=2.0):
                        raise AssertionError("reviewer prompts were not dispatched in parallel")
                    self._write_review_result(kwargs["completion_contract"], task_name="M1-T9")
                    return CommandResult(
                        label=str(kwargs["label"]),
                        command="",
                        exit_code=0,
                        raw_output="审核通过",
                        clean_output="审核通过",
                        started_at="2026-05-10T00:00:00",
                        finished_at="2026-05-10T00:00:01",
                    )

                @staticmethod
                def _write_review_result(contract: TurnFileContract, *, task_name: str) -> None:
                    review_md = contract.tracked_artifacts["review_md"]
                    review_md.write_text("", encoding="utf-8")
                    contract.status_path.write_text(
                        json.dumps([{"task_name": task_name, "review_pass": True}], ensure_ascii=False),
                        encoding="utf-8",
                    )

            reviewers: list[ReviewerRuntime] = []
            reviewer_specs_by_name: dict[str, DevelopmentReviewerSpec] = {}
            for index in range(expected_count):
                reviewer_name = f"审核员{index + 1}"
                reviewer_specs_by_name[reviewer_name] = DevelopmentReviewerSpec(
                    role_name=reviewer_name,
                    role_prompt="审核视角",
                    reviewer_key=reviewer_name,
                )
                reviewers.append(
                    ReviewerRuntime(
                        reviewer_name=reviewer_name,
                        selection=ReviewAgentSelection("opencode", "opencode/big-pickle", "high", ""),
                        worker=ReadyReviewerWorker(session_name=f"{reviewer_name}-星"),
                        review_md_path=project_dir / f"需求A_代码评审记录_{reviewer_name}-星.md",
                        review_json_path=project_dir / f"需求A_评审记录_{reviewer_name}-星.json",
                        contract=_dummy_contract(),
                    )
                )

            returned = _run_parallel_reviewers(
                reviewers,
                project_dir=project_dir,
                requirement_name="需求A",
                paths=paths,
                reviewer_specs_by_name=reviewer_specs_by_name,
                task_name="M1-T9",
                round_index=1,
                prompt_builder=lambda reviewer: f"review {reviewer.reviewer_name}",
                label_prefix="development_review_M1_T9",
                allow_existing_outputs=False,
            )

        self.assertEqual(len(returned), expected_count)
        self.assertEqual({str(call["session_name"]) for call in sent_calls}, {f"审核员{index + 1}-星" for index in range(expected_count)})
        self.assertTrue(all_sent.is_set())
        self.assertTrue(all(call["turn_start_timeout_sec"] == A07_REVIEWER_TURN_START_TIMEOUT_SEC for call in sent_calls))
        self.assertTrue(all(call["prompt_submit_timeout_sec"] == A07_REVIEWER_PROMPT_SUBMIT_TIMEOUT_SEC for call in sent_calls))

    def test_parallel_reviewer_round_slow_ready_reviewer_does_not_block_fast_reviewers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            fast_sent: list[str] = []
            fast_sent_lock = threading.Lock()
            fast_reviewers_sent = threading.Event()
            slow_started = threading.Event()
            release_slow = threading.Event()
            slow_worker_ref: dict[str, object] = {}

            class SlowAwareReviewerWorker(_FakeWorker):
                def __init__(self, *, session_name: str, slow: bool = False) -> None:
                    super().__init__(session_name=session_name)
                    self.slow = slow
                    self.state_payload: dict[str, object] = {}

                def read_state(self) -> dict[str, object]:
                    return dict(self.state_payload)

                def run_turn(self, **kwargs):  # noqa: ANN001
                    if self.slow:
                        self.state_payload.update(
                            {
                                "dispatch_state": "delayed",
                                "dispatch_reason": "ready_wait_timeout:simulated slow ready",
                            }
                        )
                        slow_started.set()
                        if not release_slow.wait(timeout=2.0):
                            raise AssertionError("slow reviewer was not released")
                    else:
                        with fast_sent_lock:
                            fast_sent.append(self.session_name)
                            if len(fast_sent) == 5:
                                fast_reviewers_sent.set()
                    self._write_review_result(kwargs["completion_contract"], task_name="M1-T10")
                    return CommandResult(
                        label=str(kwargs["label"]),
                        command="",
                        exit_code=0,
                        raw_output="审核通过",
                        clean_output="审核通过",
                        started_at="2026-05-10T00:00:00",
                        finished_at="2026-05-10T00:00:01",
                    )

                @staticmethod
                def _write_review_result(contract: TurnFileContract, *, task_name: str) -> None:
                    contract.tracked_artifacts["review_md"].write_text("", encoding="utf-8")
                    contract.status_path.write_text(
                        json.dumps([{"task_name": task_name, "review_pass": True}], ensure_ascii=False),
                        encoding="utf-8",
                    )

            reviewers: list[ReviewerRuntime] = []
            reviewer_specs_by_name: dict[str, DevelopmentReviewerSpec] = {}
            for index in range(6):
                reviewer_name = f"审核员{index + 1}"
                reviewer_specs_by_name[reviewer_name] = DevelopmentReviewerSpec(
                    role_name=reviewer_name,
                    role_prompt="审核视角",
                    reviewer_key=reviewer_name,
                )
                worker = SlowAwareReviewerWorker(session_name=f"{reviewer_name}-星", slow=index == 5)
                if worker.slow:
                    slow_worker_ref["worker"] = worker
                reviewers.append(
                    ReviewerRuntime(
                        reviewer_name=reviewer_name,
                        selection=ReviewAgentSelection("opencode", "opencode/big-pickle", "high", ""),
                        worker=worker,
                        review_md_path=project_dir / f"需求A_代码评审记录_{reviewer_name}-星.md",
                        review_json_path=project_dir / f"需求A_评审记录_{reviewer_name}-星.json",
                        contract=_dummy_contract(),
                    )
                )

            result_box: dict[str, object] = {}

            def run_round() -> None:
                try:
                    result_box["returned"] = _run_parallel_reviewers(
                        reviewers,
                        project_dir=project_dir,
                        requirement_name="需求A",
                        paths=paths,
                        reviewer_specs_by_name=reviewer_specs_by_name,
                        task_name="M1-T10",
                        round_index=1,
                        prompt_builder=lambda reviewer: f"review {reviewer.reviewer_name}",
                        label_prefix="development_review_M1_T10",
                        allow_existing_outputs=False,
                    )
                except Exception as error:  # noqa: BLE001
                    result_box["error"] = error

            thread = threading.Thread(target=run_round)
            thread.start()
            try:
                self.assertTrue(fast_reviewers_sent.wait(timeout=2.0))
                self.assertTrue(slow_started.wait(timeout=2.0))
                self.assertEqual(len(fast_sent), 5)
                self.assertTrue(thread.is_alive())
                slow_worker = slow_worker_ref["worker"]
                assert isinstance(slow_worker, SlowAwareReviewerWorker)
                self.assertEqual(slow_worker.read_state()["dispatch_state"], "delayed")
            finally:
                release_slow.set()
                thread.join(timeout=2.0)

            if "error" in result_box:
                raise result_box["error"]  # type: ignore[misc]

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(result_box["returned"]), 6)  # type: ignore[arg-type]

    def test_reviewer_protocol_repair_prompt_uses_check_reviewer_job_for_exact_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            expected_json = root / "需求A_评审记录_架构师-柳土獐.json"
            expected_md = root / "需求A_代码评审记录_架构师-柳土獐.md"
            expected_md.write_text("", encoding="utf-8")
            (root / "需求A_评审记录_架构师 - 柳土獐.json").write_text(
                json.dumps([{"task_name": "M4-T1", "review_pass": True}], ensure_ascii=False),
                encoding="utf-8",
            )
            context = RepairPromptContext(
                turn_label="development_review_init_M4-T1_架构师_round_1",
                stage_label="任务开发",
                role_label="架构师-柳土獐",
                observed_status="",
                expected_status="",
                missing_aliases=("review_json",),
                forbidden_aliases=(),
                present_aliases=(),
                last_validation_error=f"缺少审核 JSON 文件: {expected_json}",
                artifact_paths={
                    "review_json": str(expected_json),
                    "review_md": str(expected_md),
                },
                task_name="M4-T1",
                requirement_name="需求A",
            )

            prompt = _build_reviewer_protocol_repair_prompt(context)

        self.assertIn("协议违态提醒", prompt)
        self.assertIn("需求A_评审记录_架构师-柳土獐.json", prompt)
        self.assertIn(str(expected_json), prompt)
        self.assertIn("不要在角色名与星宿名之间插入额外空格", prompt)

    def test_reviewer_completion_repair_turn_uses_check_reviewer_job_prompt(self):
        class RepairPromptWorker:
            def __init__(self):
                self.prompts: list[str] = []

            def run_turn(self, *, label, prompt, completion_contract, timeout_sec):  # noqa: ANN001
                _ = completion_contract
                _ = timeout_sec
                self.prompts.append(prompt)
                if len(self.prompts) == 1:
                    error = (
                        f"{TURN_ARTIFACT_CONTRACT_ERROR_PREFIX}: "
                        "phase=任务开发 completion_source=task_status_done error=缺少审核 JSON 文件"
                    )
                    return CommandResult(label, prompt, 1, error, error, "start", "finish")
                expected_md.write_text("", encoding="utf-8")
                expected_json.write_text(
                    json.dumps([{"task_name": "M4-T1", "review_pass": True}], ensure_ascii=False),
                    encoding="utf-8",
                )
                return CommandResult(label, prompt, 0, "{}", "{}", "start", "finish")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            expected_json = root / "需求A_评审记录_架构师-柳土獐.json"
            expected_md = root / "需求A_代码评审记录_架构师-柳土獐.md"
            expected_md.write_text("", encoding="utf-8")
            (root / "需求A_评审记录_架构师 - 柳土獐.json").write_text(
                json.dumps([{"task_name": "M4-T1", "review_pass": True}], ensure_ascii=False),
                encoding="utf-8",
            )
            worker = RepairPromptWorker()

            run_completion_turn_with_repair(
                worker=worker,  # type: ignore[arg-type]
                label="development_review_init_M4-T1_架构师_round_1",
                prompt="原始审核提示",
                completion_contract=build_reviewer_completion_contract(
                    reviewer_name="架构师",
                    task_name="M4-T1",
                    review_md_path=expected_md,
                    review_json_path=expected_json,
                ),
                turn_goal=_build_reviewer_turn_goal(),
                stage_label="任务开发",
                role_label="架构师-柳土獐",
                task_name="M4-T1",
                requirement_name="需求A",
            )

        self.assertEqual(len(worker.prompts), 2)
        self.assertIn("协议违态提醒", worker.prompts[1])
        self.assertIn("需求A_评审记录_架构师-柳土獐.json", worker.prompts[1])
        self.assertIn("不要在角色名与星宿名之间插入额外空格", worker.prompts[1])

    def test_resolve_review_max_rounds_supports_default_and_infinite(self):
        args = build_parser().parse_args([])
        self.assertEqual(resolve_review_max_rounds(args), 5)

        args = build_parser().parse_args(["--review-max-rounds", "infinite"])
        self.assertIsNone(resolve_review_max_rounds(args))

    def test_resolve_review_max_rounds_rejects_invalid_cli_value(self):
        args = build_parser().parse_args(["--review-max-rounds", "abc"])

        with self.assertRaisesRegex(RuntimeError, "必须是正整数或 infinite"):
            resolve_review_max_rounds(args)

    def test_resolve_review_max_rounds_prompts_in_interactive_mode(self):
        args = build_parser().parse_args([])

        with patch("A07_Development.stdin_is_interactive", return_value=True), patch(
            "A07_Development.prompt_review_max_rounds",
            return_value=8,
        ) as prompt_mock:
            value = resolve_review_max_rounds(args, progress=object())

        self.assertEqual(value, 8)
        prompt_mock.assert_called_once()

    def test_resolve_developer_role_prompt_yes_uses_default_without_prompt(self):
        args = build_parser().parse_args(["--yes"])

        with patch(
            "tmux_core.stage_kernel.shared_review.prompt_yes_no_choice",
            side_effect=AssertionError("should not prompt"),
        ), patch(
            "A07_Development._prompt_reviewer_text",
            side_effect=AssertionError("should not prompt"),
        ):
            value = resolve_developer_role_prompt(args)

        self.assertEqual(value, fintech_developer_role)

    def test_build_developer_review_feedback_result_contract_uses_file_contract_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = build_development_paths(tmp_dir, "需求A")

            contract = build_developer_review_feedback_result_contract(paths)

        self.assertEqual(contract.expected_statuses, ("hitl", "completed"))
        self.assertIn("ask_human", contract.optional_artifacts)
        self.assertIn("developer_output", contract.optional_artifacts)

    def test_predict_worker_display_name_includes_tmux_sessions_in_occupied_pool(self):
        import A07_Development as development_module

        observed: dict[str, set[str]] = {}

        def fake_build_session_name(worker_id, work_dir, vendor, instance_id="", occupied_session_names=None):  # noqa: ANN001
            _ = worker_id
            _ = work_dir
            _ = vendor
            _ = instance_id
            observed["occupied"] = {str(item).strip() for item in occupied_session_names or () if str(item).strip()}
            return "开发工程师-天魁星"

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "A07_Development.build_session_name",
            side_effect=fake_build_session_name,
        ), patch(
            "A07_Development.list_tmux_session_names",
            return_value=["开发工程师-天暴星"],
        ), patch(
            "A07_Development.list_registered_tmux_workers",
            return_value=[],
        ):
            session_name = development_module._predict_worker_display_name(
                project_dir=tmpdir,
                worker_id="development-developer",
                occupied_session_names=("预占用-会话",),
            )

        self.assertEqual(session_name, "开发工程师-天魁星")
        self.assertIn("开发工程师-天暴星", observed["occupied"])
        self.assertIn("预占用-会话", observed["occupied"])

    def test_run_development_stage_prompts_models_before_review_limits(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            calls: list[str] = []

            def record(name: str, value):  # noqa: ANN001
                calls.append(name)
                return value

            with patch("A07_Development.ensure_development_inputs", return_value=paths), patch(
                "A07_Development.cleanup_stale_development_runtime_state",
                return_value=(),
            ), patch(
                "A07_Development.cleanup_existing_development_artifacts",
                return_value=(),
            ), patch(
                "A07_Development.resolve_developer_plan",
                side_effect=lambda *args, **kwargs: record(
                    "developer_plan",
                    DeveloperPlan(
                        selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                        role_prompt="实现视角",
                    ),
                ),
            ), patch(
                "A07_Development.resolve_reviewer_specs",
                side_effect=lambda *args, **kwargs: record("reviewer_specs", []),
            ), patch(
                "A07_Development.collect_reviewer_agent_selections",
                side_effect=lambda *args, **kwargs: record("reviewer_models", {}),
            ), patch(
                "A07_Development.resolve_developer_max_turns",
                side_effect=lambda *args, **kwargs: record("developer_max_turns", 15),
            ), patch(
                "A07_Development.resolve_review_max_rounds",
                side_effect=lambda *args, **kwargs: record("review_max_rounds", 5),
            ), patch(
                "A07_Development.resolve_subagent_num",
                side_effect=lambda *args, **kwargs: record("subagent_num", 0),
            ), patch(
                "A07_Development.create_developer_runtime",
                side_effect=RuntimeError("stop-after-order-check"),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop-after-order-check"):
                    run_development_stage(["--project-dir", str(project_dir), "--requirement-name", "需求A"])

        self.assertEqual(
            calls,
            [
                "developer_plan",
                "reviewer_specs",
                "reviewer_models",
                "developer_max_turns",
                "review_max_rounds",
                "subagent_num",
            ],
        )

    def test_resolve_developer_max_turns_defaults_to_15_in_noninteractive_mode(self):
        args = build_parser().parse_args([])

        with patch("A07_Development.stdin_is_interactive", return_value=False):
            value = resolve_developer_max_turns(args)

        self.assertEqual(value, 15)

    def test_resolve_developer_max_turns_accepts_infinite_cli_value(self):
        args = build_parser().parse_args(["--developer-max-turns", "infinite"])

        value = resolve_developer_max_turns(args)

        self.assertIsNone(value)

    def test_recreate_developer_runtime_noninteractive_reuses_existing_selection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            original = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            replacement = DeveloperRuntime(
                selection=original.selection,
                worker=_FakeWorker(session_name="开发工程师-天罡星"),
                role_prompt=original.role_prompt,
            )
            with patch("A07_Development.stdin_is_interactive", return_value=False), patch(
                "A07_Development.create_developer_runtime",
                return_value=replacement,
            ) as create_runtime, patch(
                "A07_Development.prompt_replacement_review_agent_selection",
            ) as prompt_replace:
                with self.assertRaises(RuntimeError) as raised:
                    recreate_developer_runtime(
                        project_dir=tmp_dir,
                        developer=original,
                        progress=None,
                    )

        self.assertIn("需要人工选择厂商/模型/推理/代理", str(raised.exception))
        prompt_replace.assert_not_called()
        create_runtime.assert_not_called()

    def test_run_development_stage_resolves_review_max_rounds_before_runtime_creation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            calls: list[str] = []

            with patch("A07_Development.ensure_development_inputs", return_value=paths), patch(
                "A07_Development.cleanup_stale_development_runtime_state",
                return_value=(),
            ), patch(
                "A07_Development.cleanup_existing_development_artifacts",
                return_value=(),
            ), patch(
                "A07_Development.resolve_developer_plan",
                return_value=DeveloperPlan(
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    role_prompt="实现视角",
                ),
            ), patch(
                "A07_Development.resolve_reviewer_specs",
                return_value=[],
            ), patch(
                "A07_Development.collect_reviewer_agent_selections",
                return_value={},
            ), patch(
                "A07_Development.resolve_developer_max_turns",
                return_value=15,
            ), patch(
                "A07_Development.resolve_review_max_rounds",
                side_effect=lambda *args, **kwargs: calls.append("review_max_rounds") or 5,
            ), patch(
                "A07_Development.resolve_subagent_num",
                return_value=0,
            ), patch(
                "A07_Development.create_developer_runtime",
                side_effect=lambda *args, **kwargs: calls.append("create_developer_runtime") or (_ for _ in ()).throw(RuntimeError("stop-after-order-check")),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop-after-order-check"):
                    run_development_stage(["--project-dir", str(project_dir), "--requirement-name", "需求A"])

        self.assertEqual(calls, ["review_max_rounds", "create_developer_runtime"])

    def test_build_development_paths_migrates_legacy_question_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            legacy_path = project_dir / "需求A_向人类提问.md"
            legacy_path.write_text("旧的人类问题\n", encoding="utf-8")

            paths = build_development_paths(project_dir, "需求A")
            migrated_text = paths["ask_human_path"].read_text(encoding="utf-8")
            legacy_exists = legacy_path.exists()

        self.assertFalse(legacy_exists)
        self.assertEqual(migrated_text, "旧的人类问题\n")

    def test_create_development_runtimes_accept_stage_local_launch_coordinator(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            sentinel = object()
            developer = create_developer_runtime(
                project_dir=project_dir,
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                role_prompt="实现视角",
                launch_coordinator=sentinel,  # type: ignore[arg-type]
            )
            reviewer = create_reviewer_runtime(
                project_dir=project_dir,
                requirement_name="需求A",
                reviewer_spec=DevelopmentReviewerSpec(role_name="测试工程师", role_prompt="测试视角", reviewer_key="测试工程师"),
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                launch_coordinator=sentinel,  # type: ignore[arg-type]
            )

        self.assertIs(developer.worker.launch_coordinator, sentinel)
        self.assertIs(reviewer.worker.launch_coordinator, sentinel)

    def test_initialize_development_workers_runs_reviewer_init_in_parallel(self):
        class ReviewerInitWorker(_FakeWorker):
            def __init__(self, *, session_name: str, barrier: threading.Barrier, active_lock: threading.Lock, active_state: dict[str, int]):
                super().__init__(session_name=session_name)
                self._barrier = barrier
                self._active_lock = active_lock
                self._active_state = active_state

            def run_turn(self, **kwargs):  # noqa: ANN003, ANN001
                with self._active_lock:
                    self._active_state["active"] += 1
                    self._active_state["max_active"] = max(
                        self._active_state["max_active"],
                        self._active_state["active"],
                    )
                try:
                    self._barrier.wait(timeout=1.0)
                finally:
                    with self._active_lock:
                        self._active_state["active"] -= 1
                return type("Result", (), {"ok": True, "clean_output": ""})()

        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            barrier = threading.Barrier(2)
            active_lock = threading.Lock()
            active_state = {"active": 0, "max_active": 0}
            reviewers = [
                ReviewerRuntime(
                    reviewer_name="测试工程师",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=ReviewerInitWorker(
                        session_name="测试工程师-天英星",
                        barrier=barrier,
                        active_lock=active_lock,
                        active_state=active_state,
                    ),
                    review_md_path=project_dir / "需求A_代码评审记录_测试工程师.md",
                    review_json_path=project_dir / "需求A_评审记录_测试工程师.json",
                    contract=_dummy_contract(),
                ),
                ReviewerRuntime(
                    reviewer_name="审核员",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=ReviewerInitWorker(
                        session_name="审核员-天机星",
                        barrier=barrier,
                        active_lock=active_lock,
                        active_state=active_state,
                    ),
                    review_md_path=project_dir / "需求A_代码评审记录_审核员.md",
                    review_json_path=project_dir / "需求A_评审记录_审核员.json",
                    contract=_dummy_contract(),
                ),
            ]

            _, initialized_reviewers = initialize_development_workers(
                developer,
                paths=paths,
                reviewers=reviewers,
                reviewer_specs_by_name={
                    "测试工程师": DevelopmentReviewerSpec(role_name="测试工程师", role_prompt="测试视角", reviewer_key="测试工程师"),
                    "审核员": DevelopmentReviewerSpec(role_name="审核员", role_prompt="审计视角", reviewer_key="审核员"),
                },
                initialize_developer=False,
                initialize_reviewers=True,
            )

        self.assertEqual(len(initialized_reviewers), 2)
        self.assertEqual(active_state["max_active"], 2)

    def test_recreate_development_workers_only_reinitializes_developer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星", runtime_root=project_dir / "runtime", runtime_dir=project_dir / "runtime" / "developer"),
                role_prompt="实现视角",
            )
            reviewers = [
                ReviewerRuntime(
                    reviewer_name="审核员",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="审核员-天机星", runtime_root=project_dir / "runtime", runtime_dir=project_dir / "runtime" / "reviewer"),
                    review_md_path=project_dir / "需求A_代码评审记录_审核员.md",
                    review_json_path=project_dir / "需求A_评审记录_审核员.json",
                    contract=_dummy_contract(),
                )
            ]
            recreated_developer = DeveloperRuntime(
                selection=developer.selection,
                worker=_FakeWorker(session_name="开发工程师-天罡星"),
                role_prompt="实现视角",
            )
            recreated_reviewer = ReviewerRuntime(
                reviewer_name="审核员",
                selection=reviewers[0].selection,
                worker=_FakeWorker(session_name="审核员-天平星"),
                review_md_path=project_dir / "需求A_代码评审记录_审核员_v2.md",
                review_json_path=project_dir / "需求A_评审记录_审核员_v2.json",
                contract=_dummy_contract(),
            )

            init_calls: list[tuple[bool, bool]] = []

            def fake_initialize(current_developer, **kwargs):  # noqa: ANN001
                init_calls.append((kwargs["initialize_developer"], kwargs["initialize_reviewers"]))
                return current_developer, list(kwargs["reviewers"])

            with patch("A07_Development.create_developer_runtime", return_value=recreated_developer), patch(
                "A07_Development.create_reviewer_runtime",
                return_value=recreated_reviewer,
            ), patch(
                "A07_Development.initialize_development_workers",
                side_effect=fake_initialize,
            ):
                updated_developer, updated_reviewers, _ = recreate_development_workers(
                    developer,
                    reviewers,
                    project_dir=project_dir,
                    requirement_name="需求A",
                    paths=paths,
                    reviewer_specs_by_name={"审核员": DevelopmentReviewerSpec(role_name="审核员", role_prompt="审计视角", reviewer_key="审核员")},
                )

        self.assertIs(updated_developer, recreated_developer)
        self.assertEqual(updated_reviewers, [recreated_reviewer])
        self.assertEqual(init_calls, [(True, False)])

    def test_recreate_development_workers_can_restart_stage_by_shutdown_then_init_all(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False, "M1-T2": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星", runtime_root=project_dir / "runtime", runtime_dir=project_dir / "runtime" / "developer"),
                role_prompt="实现视角",
            )
            reviewers = [
                ReviewerRuntime(
                    reviewer_name="审核员",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="审核员-天机星", runtime_root=project_dir / "runtime", runtime_dir=project_dir / "runtime" / "reviewer"),
                    review_md_path=project_dir / "需求A_代码评审记录_审核员.md",
                    review_json_path=project_dir / "需求A_评审记录_审核员.json",
                    contract=_dummy_contract(),
                )
            ]
            recreated_developer = DeveloperRuntime(
                selection=developer.selection,
                worker=_FakeWorker(session_name="开发工程师-天罡星"),
                role_prompt="实现视角",
            )
            recreated_reviewer = ReviewerRuntime(
                reviewer_name="审核员",
                selection=reviewers[0].selection,
                worker=_FakeWorker(session_name="审核员-天平星"),
                review_md_path=project_dir / "需求A_代码评审记录_审核员_v2.md",
                review_json_path=project_dir / "需求A_评审记录_审核员_v2.json",
                contract=_dummy_contract(),
            )
            call_order: list[str] = []

            def fake_shutdown(current_developer, current_reviewers, **kwargs):  # noqa: ANN001
                self.assertIs(current_developer, developer)
                self.assertEqual(list(current_reviewers), reviewers)
                self.assertTrue(kwargs["cleanup_runtime"])
                call_order.append("shutdown_all")
                return ("cleanup/runtime",)

            def fake_cleanup(current_paths, current_requirement_name, *args, **kwargs):  # noqa: ANN001
                self.assertEqual(current_paths, paths)
                self.assertEqual(current_requirement_name, "需求A")
                call_order.append("cleanup_artifacts")
                return ("cleanup/artifacts",)

            def fake_create_developer(**kwargs):  # noqa: ANN001
                self.assertIn("shutdown_all", call_order)
                call_order.append("create_developer")
                return recreated_developer

            def fake_create_reviewer(**kwargs):  # noqa: ANN001
                self.assertIn("shutdown_all", call_order)
                call_order.append("create_reviewer")
                return recreated_reviewer

            def fake_bootstrap(current_developer, **kwargs):  # noqa: ANN001
                self.assertIs(current_developer, recreated_developer)
                call_order.append("initialize_developer")
                return recreated_developer

            def fake_initialize(current_developer, **kwargs):  # noqa: ANN001
                self.assertIs(current_developer, recreated_developer)
                self.assertFalse(kwargs["initialize_developer"])
                self.assertTrue(kwargs["initialize_reviewers"])
                call_order.append("initialize_reviewers")
                return current_developer, list(kwargs["reviewers"])

            with patch("A07_Development._shutdown_workers", side_effect=fake_shutdown), patch(
                "A07_Development.cleanup_existing_development_artifacts",
                side_effect=fake_cleanup,
            ), patch(
                "A07_Development.create_developer_runtime",
                side_effect=fake_create_developer,
            ), patch(
                "A07_Development.create_reviewer_runtime",
                side_effect=fake_create_reviewer,
            ), patch(
                "A07_Development._bootstrap_developer_runtime",
                side_effect=fake_bootstrap,
            ), patch(
                "A07_Development.initialize_development_workers",
                side_effect=fake_initialize,
            ):
                updated_developer, updated_reviewers, cleanup_paths = recreate_development_workers(
                    developer,
                    reviewers,
                    project_dir=project_dir,
                    requirement_name="需求A",
                    paths=paths,
                    reviewer_specs_by_name={"审核员": DevelopmentReviewerSpec(role_name="审核员", role_prompt="审计视角", reviewer_key="审核员")},
                    initialize_reviewers=True,
                )

        self.assertIs(updated_developer, recreated_developer)
        self.assertEqual(updated_reviewers, [recreated_reviewer])
        self.assertEqual(cleanup_paths, ("cleanup/runtime", "cleanup/artifacts"))
        self.assertEqual(
            call_order,
            [
                "shutdown_all",
                "cleanup_artifacts",
                "create_developer",
                "create_reviewer",
                "initialize_developer",
                "initialize_reviewers",
            ],
        )

    def test_replace_dead_developer_with_bootstrap_reinitializes_replacement_before_retry(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            original = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            recreated = DeveloperRuntime(
                selection=original.selection,
                worker=_FakeWorker(session_name="开发工程师-天罡星"),
                role_prompt="实现视角",
            )
            bootstrapped = DeveloperRuntime(
                selection=original.selection,
                worker=_FakeWorker(session_name="开发工程师-天机星"),
                role_prompt="实现视角",
            )

            with patch("A07_Development._replace_dead_developer", return_value=recreated) as replace_raw, patch(
                "A07_Development.initialize_development_workers",
                return_value=(bootstrapped, []),
            ) as initialize_workers:
                result = _replace_dead_developer_with_bootstrap(
                    original,
                    paths=paths,
                    reviewer_specs_by_name={},
                    project_dir=project_dir,
                )

        self.assertIs(result, bootstrapped)
        replace_raw.assert_called_once()
        initialize_workers.assert_called_once()

    def test_run_single_reviewer_initialization_skips_reviewer_after_ready_timeout_close_choice(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_ReconfigurableWorker(session_name="测试工程师-天英星"),
                review_md_path=project_dir / "需求A_代码评审记录_测试工程师.md",
                review_json_path=project_dir / "需求A_评审记录_测试工程师.json",
                contract=_dummy_contract(),
            )

            attempts = {"count": 0}
            def always_timeout(**kwargs):  # noqa: ANN001
                attempts["count"] += 1
                raise RuntimeError("Timed out waiting for agent ready.\nmock screen")

            with patch(
                "A07_Development.run_task_result_turn_with_repair",
                side_effect=always_timeout,
            ), patch(
                "A07_Development.prompt_agent_ready_timeout_recovery",
                return_value=AGENT_READY_TIMEOUT_SKIP,
            ) as prompt_recovery, patch("A07_Development.recreate_development_reviewer_runtime") as recreate_runtime:
                result = _run_single_reviewer_initialization(
                    reviewer,
                    project_dir=project_dir,
                    requirement_name="需求A",
                    paths=paths,
                    reviewer_specs_by_name={
                        "测试工程师": DevelopmentReviewerSpec(
                            role_name="测试工程师",
                            role_prompt="测试视角",
                            reviewer_key="测试工程师",
                        ),
                    },
                    can_skip_ready_timeout=True,
                )

        self.assertIsNone(result)
        self.assertEqual(attempts["count"], 1)
        self.assertTrue(reviewer.worker.killed)
        prompt_recovery.assert_called_once()
        self.assertTrue(prompt_recovery.call_args.kwargs["can_skip"])
        self.assertTrue(prompt_recovery.call_args.kwargs["allow_recreate"])
        recreate_runtime.assert_not_called()

    def test_run_single_reviewer_initialization_last_reviewer_ready_timeout_only_retries(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_ReconfigurableWorker(session_name="测试工程师-天英星"),
                review_md_path=project_dir / "需求A_代码评审记录_测试工程师.md",
                review_json_path=project_dir / "需求A_评审记录_测试工程师.json",
                contract=_dummy_contract(),
            )
            attempts = {"count": 0}

            def timeout_then_success(**kwargs):  # noqa: ANN001
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise RuntimeError("Timed out waiting for agent ready.\nmock screen")
                return {}

            with patch(
                "A07_Development.run_task_result_turn_with_repair",
                side_effect=timeout_then_success,
            ), patch(
                "A07_Development.prompt_agent_ready_timeout_recovery",
                return_value=AGENT_READY_TIMEOUT_RETRY,
            ) as prompt_recovery, patch("A07_Development.recreate_development_reviewer_runtime") as recreate_runtime:
                result = _run_single_reviewer_initialization(
                    reviewer,
                    project_dir=project_dir,
                    requirement_name="需求A",
                    paths=paths,
                    reviewer_specs_by_name={
                        "测试工程师": DevelopmentReviewerSpec(
                            role_name="测试工程师",
                            role_prompt="测试视角",
                            reviewer_key="测试工程师",
                        ),
                    },
                    can_skip_ready_timeout=False,
                )

        self.assertIs(result, reviewer)
        self.assertEqual(attempts["count"], 2)
        self.assertFalse(reviewer.worker.killed)
        prompt_recovery.assert_called_once()
        self.assertFalse(prompt_recovery.call_args.kwargs["can_skip"])
        self.assertTrue(prompt_recovery.call_args.kwargs["allow_recreate"])
        recreate_runtime.assert_not_called()

    def test_run_single_reviewer_initialization_escalates_repeated_death_to_manual_reconfiguration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            reviewer = ReviewerRuntime(
                reviewer_name="架构师",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_ReconfigurableWorker(session_name="架构师-天究星"),
                review_md_path=project_dir / "需求A_代码评审记录_架构师.md",
                review_json_path=project_dir / "需求A_评审记录_架构师.json",
                contract=_dummy_contract(),
            )
            replacement1 = ReviewerRuntime(
                reviewer_name="架构师",
                selection=reviewer.selection,
                worker=_ReconfigurableWorker(session_name="架构师-天究星#2"),
                review_md_path=reviewer.review_md_path,
                review_json_path=reviewer.review_json_path,
                contract=_dummy_contract(),
            )
            replacement2 = ReviewerRuntime(
                reviewer_name="架构师",
                selection=ReviewAgentSelection("claude", "sonnet", "high", ""),
                worker=_ReconfigurableWorker(session_name="架构师-天究星#3"),
                review_md_path=reviewer.review_md_path,
                review_json_path=reviewer.review_json_path,
                contract=_dummy_contract(),
            )
            attempts = {"count": 0}

            def death_death_success(**kwargs):  # noqa: ANN001
                attempts["count"] += 1
                if attempts["count"] <= 2:
                    raise RuntimeError("tmux pane died")
                return {}

            with patch(
                "A07_Development.run_task_result_turn_with_repair",
                side_effect=death_death_success,
            ), patch(
                "A07_Development.request_worker_manual_intervention",
                return_value=AGENT_INTERVENTION_RECREATE,
            ), patch(
                "A07_Development.recreate_development_reviewer_runtime",
                side_effect=[replacement1, replacement2],
            ) as recreate_runtime:
                result = _run_single_reviewer_initialization(
                    reviewer,
                    project_dir=project_dir,
                    requirement_name="需求A",
                    paths=paths,
                    reviewer_specs_by_name={
                        "架构师": DevelopmentReviewerSpec(
                            role_name="架构师",
                            role_prompt="架构视角",
                            reviewer_key="架构师",
                        ),
                    },
                    can_skip_ready_timeout=False,
                )

        self.assertIs(result, replacement2)
        self.assertEqual(attempts["count"], 3)
        self.assertEqual(recreate_runtime.call_args_list[0].kwargs["force_model_change"], False)
        self.assertEqual(recreate_runtime.call_args_list[1].kwargs["force_model_change"], True)
        self.assertTrue(recreate_runtime.call_args_list[1].kwargs["required_reconfiguration"])
        self.assertIn("连续 2 次死亡/失败", recreate_runtime.call_args_list[1].kwargs["reason_text"])
        self.assertIn("智能体进程已死亡或退出", recreate_runtime.call_args_list[1].kwargs["reason_text"])

    def test_developer_ready_timeout_requires_manual_retry_and_continues_after_success(self):
        developer = DeveloperRuntime(
            selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
            worker=_ReconfigurableWorker(session_name="开发工程师-天魁星"),
            role_prompt="实现视角",
        )
        attempts = {"count": 0}

        def timeout_then_success(**kwargs):  # noqa: ANN001
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("Timed out waiting for agent ready.\nmock screen")
            return {"status": "completed"}

        with patch(
            "A07_Development.run_task_result_turn_with_repair",
            side_effect=timeout_then_success,
        ), patch(
            "A07_Development.prompt_agent_ready_timeout_recovery",
            return_value=AGENT_READY_TIMEOUT_RETRY,
        ) as prompt_recovery, patch("A07_Development.recreate_developer_runtime") as recreate_runtime:
            returned, payload = _run_developer_result_turn(
                developer,
                label="developer_ready_timeout_retry",
                prompt="请开发",
                result_contract=_dummy_task_result_contract(),
            )

        self.assertIs(returned, developer)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(attempts["count"], 2)
        prompt_recovery.assert_called_once()
        self.assertFalse(prompt_recovery.call_args.kwargs["can_skip"])
        recreate_runtime.assert_not_called()

    def test_developer_result_turn_uses_fast_dispatch_budget(self):
        developer = DeveloperRuntime(
            selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
            worker=_ReconfigurableWorker(session_name="开发工程师-天魁星"),
            role_prompt="实现视角",
        )

        with patch("A07_Development.run_task_result_turn_with_repair", return_value={"status": "completed"}) as run_turn:
            returned, payload = _run_developer_result_turn(
                developer,
                label="developer_fast_dispatch",
                prompt="请开发",
                result_contract=_dummy_task_result_contract(),
            )

        self.assertIs(returned, developer)
        self.assertEqual(payload["status"], "completed")
        run_turn.assert_called_once()
        self.assertEqual(run_turn.call_args.kwargs["turn_start_timeout_sec"], A07_REVIEWER_TURN_START_TIMEOUT_SEC)
        self.assertEqual(
            run_turn.call_args.kwargs["prompt_submit_timeout_sec"],
            A07_REVIEWER_PROMPT_SUBMIT_TIMEOUT_SEC,
        )
        self.assertEqual(
            run_turn.call_args.kwargs["pre_submit_observation_tail_lines"],
            A07_PRE_SUBMIT_OBSERVATION_TAIL_LINES,
        )
        self.assertEqual(
            run_turn.call_args.kwargs["pre_submit_observation_tail_bytes"],
            A07_PRE_SUBMIT_OBSERVATION_TAIL_BYTES,
        )

    def test_developer_ready_timeout_reopens_hitl_when_manual_retry_times_out_again(self):
        developer = DeveloperRuntime(
            selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
            worker=_ReconfigurableWorker(session_name="开发工程师-天魁星"),
            role_prompt="实现视角",
        )
        attempts = {"count": 0}

        def timeout_timeout_success(**kwargs):  # noqa: ANN001
            attempts["count"] += 1
            if attempts["count"] <= 2:
                raise RuntimeError("Timed out waiting for agent ready.\nmock screen")
            return {"status": "completed"}

        with patch(
            "A07_Development.run_task_result_turn_with_repair",
            side_effect=timeout_timeout_success,
        ), patch(
            "A07_Development.prompt_agent_ready_timeout_recovery",
            return_value=AGENT_READY_TIMEOUT_RETRY,
        ) as prompt_recovery:
            returned, payload = _run_developer_result_turn(
                developer,
                label="developer_ready_timeout_retry",
                prompt="请开发",
                result_contract=_dummy_task_result_contract(),
            )

        self.assertIs(returned, developer)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(attempts["count"], 3)
        self.assertEqual(prompt_recovery.call_count, 2)

    def test_developer_ready_timeout_uses_replacement_callback_when_available(self):
        developer = DeveloperRuntime(
            selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
            worker=_ReconfigurableWorker(session_name="开发工程师-天魁星"),
            role_prompt="实现视角",
        )
        replacement = DeveloperRuntime(
            selection=developer.selection,
            worker=_ReconfigurableWorker(session_name="开发工程师-天罡星"),
            role_prompt="实现视角",
        )
        attempts = {"count": 0}

        def timeout_then_success(**kwargs):  # noqa: ANN001
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("Timed out waiting for agent ready.\nmock screen")
            self.assertIs(kwargs["worker"], replacement.worker)
            return {"status": "completed"}

        with patch(
            "A07_Development.run_task_result_turn_with_repair",
            side_effect=timeout_then_success,
        ), patch(
            "A07_Development.prompt_agent_ready_timeout_recovery",
            side_effect=AssertionError("replacement callback should own HITL"),
        ), patch("A07_Development.recreate_developer_runtime") as recreate_runtime:
            returned, payload = _run_developer_result_turn(
                developer,
                label="developer_ready_timeout_recreate",
                prompt="请开发",
                result_contract=_dummy_task_result_contract(),
                replace_dead_developer=lambda owner, error: replacement,
            )

        self.assertIs(returned, replacement)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(attempts["count"], 2)
        recreate_runtime.assert_not_called()

    def test_replace_dead_developer_ready_timeout_allows_recreate(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_ReconfigurableWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            replacement = DeveloperRuntime(
                selection=developer.selection,
                worker=_ReconfigurableWorker(session_name="开发工程师-天罡星"),
                role_prompt="实现视角",
            )

            with patch(
                "A07_Development.prompt_agent_ready_timeout_recovery",
                return_value=AGENT_INTERVENTION_RECREATE,
            ) as prompt_recovery, patch(
                "A07_Development.recreate_developer_runtime",
                return_value=replacement,
            ) as recreate_runtime:
                result = _replace_dead_developer(
                    developer,
                    project_dir=project_dir,
                    requirement_name="需求A",
                    error=RuntimeError("Timed out waiting for agent ready.\nmock screen"),
                )

        self.assertIs(result, replacement)
        prompt_recovery.assert_called_once()
        self.assertFalse(prompt_recovery.call_args.kwargs["can_skip"])
        self.assertTrue(prompt_recovery.call_args.kwargs["allow_recreate"])
        recreate_runtime.assert_called_once()

    def test_recreate_development_workers_uses_bootstrap_helper(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            reviewers = [
                ReviewerRuntime(
                    reviewer_name="审核员",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="审核员-天机星"),
                    review_md_path=project_dir / "需求A_代码评审记录_审核员.md",
                    review_json_path=project_dir / "需求A_评审记录_审核员.json",
                    contract=_dummy_contract(),
                )
            ]
            recreated_developer = DeveloperRuntime(
                selection=developer.selection,
                worker=_FakeWorker(session_name="开发工程师-天罡星"),
                role_prompt="实现视角",
            )

            with patch("A07_Development.create_developer_runtime", return_value=recreated_developer), patch(
                "A07_Development.create_reviewer_runtime",
                return_value=reviewers[0],
            ), patch(
                "A07_Development._bootstrap_developer_runtime",
                return_value=recreated_developer,
            ) as bootstrap_helper:
                updated_developer, _, _ = recreate_development_workers(
                    developer,
                    reviewers,
                    project_dir=project_dir,
                    requirement_name="需求A",
                    paths=paths,
                    reviewer_specs_by_name={"审核员": DevelopmentReviewerSpec(role_name="审核员", role_prompt="审计视角", reviewer_key="审核员")},
                )

        self.assertIs(updated_developer, recreated_developer)
        bootstrap_helper.assert_called_once()

    def test_ensure_development_inputs_runs_a06_when_task_outputs_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            args = build_parser().parse_args(["--project-dir", str(project_dir), "--requirement-name", "需求A"])

            def fake_run_task_split_stage(argv):  # noqa: ANN001
                self.assertEqual(argv[:4], ["--project-dir", str(project_dir), "--requirement-name", "需求A"])
                paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
                paths["task_json_path"].write_text(
                    json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return object()

            with patch("A07_Development.run_task_split_stage", side_effect=fake_run_task_split_stage) as mocked_a06:
                resolved = ensure_development_inputs(
                    args,
                    project_dir=str(project_dir),
                    requirement_name="需求A",
                )

        mocked_a06.assert_called_once()
        self.assertEqual(resolved["task_md_path"].name, "需求A_任务单.md")
        self.assertEqual(resolved["task_json_path"].name, "需求A_任务单.json")

    def test_ensure_development_inputs_runs_a06_when_task_json_invalid(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": "invalid"}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            args = build_parser().parse_args(["--project-dir", str(project_dir), "--requirement-name", "需求A"])

            def fake_run_task_split_stage(argv):  # noqa: ANN001
                self.assertIn("--requirement-name", argv)
                paths["task_json_path"].write_text(
                    json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return object()

            with patch("A07_Development.run_task_split_stage", side_effect=fake_run_task_split_stage) as mocked_a06:
                resolved = ensure_development_inputs(
                    args,
                    project_dir=str(project_dir),
                    requirement_name="需求A",
                )

        mocked_a06.assert_called_once()
        self.assertEqual(resolved["task_json_path"].name, "需求A_任务单.json")

    def test_run_developer_hitl_loop_replies_until_ready(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = build_development_paths(tmp_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            paths["ask_human_path"].write_text("请确认字段映射\n", encoding="utf-8")
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )

            def fake_reply(current_developer, **kwargs):  # noqa: ANN001
                paths["ask_human_path"].write_text("", encoding="utf-8")
                return current_developer, {"status": "ready"}

            with patch("A07_Development._collect_development_hitl_response", return_value="已确认字段映射") as ask_human, patch(
                "A07_Development._run_developer_result_turn",
                side_effect=fake_reply,
            ) as reply_turn:
                returned = run_developer_hitl_loop(
                    developer,
                    paths=paths,
                    initial_payload={"status": "hitl"},
                )

        self.assertIs(returned, developer)
        ask_human.assert_called_once()
        reply_turn.assert_called_once()

    def test_ensure_developer_metadata_requests_repair_until_output_valid(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = build_development_paths(tmp_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )

            prompts: list[str] = []

            def fake_run_turn(current_developer, **kwargs):  # noqa: ANN001
                prompts.append(kwargs["prompt"])
                paths["developer_output_path"].write_text("- **完成任务**: `M1-T1`\n", encoding="utf-8")
                return current_developer, {"status": "completed"}

            with patch("A07_Development.check_develop_job", side_effect=["请补开发元数据", ""]), patch(
                "A07_Development._run_developer_result_turn",
                side_effect=fake_run_turn,
            ):
                returned_developer, code_change = ensure_developer_metadata(
                    developer,
                    paths=paths,
                    task_name="M1-T1",
                    label_prefix="development_start_M1_T1",
                )

        self.assertIs(returned_developer, developer)
        self.assertEqual(code_change.strip(), "- **完成任务**: `M1-T1`")
        self.assertEqual(prompts, ["请补开发元数据"])

    def test_develop_current_task_uses_prompt_contract_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = build_development_paths(tmp_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            captured: dict[str, object] = {}

            def fake_run_turn(current_developer, **kwargs):  # noqa: ANN001
                captured.update(kwargs)
                paths["developer_output_path"].write_text("- **完成任务**: `M1-T1`\n", encoding="utf-8")
                return current_developer, {"status": "completed"}

            with patch("A07_Development._run_developer_result_turn", side_effect=fake_run_turn), patch(
                "A07_Development.ensure_developer_metadata",
                return_value=(developer, "- **完成任务**: `M1-T1`"),
            ):
                returned, code_change = develop_current_task(
                    developer,
                    paths=paths,
                    task_name="M1-T1",
                    subagent_num=0,
                )

        self.assertIs(returned, developer)
        self.assertEqual(code_change, "- **完成任务**: `M1-T1`")
        contract = captured["result_contract"]
        self.assertIsInstance(contract, TaskResultContract)
        assert isinstance(contract, TaskResultContract)
        self.assertEqual(contract.outcome_artifacts["completed"]["requires"], ("developer_output",))
        self.assertEqual(contract.artifact_rules["developer_output"]["change"], CHANGE_MUST_CHANGE)
        self.assertIn("## 本轮文件契约", str(captured["prompt"]))

    def test_refine_current_task_uses_prompt_contract_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = build_development_paths(tmp_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            captured: dict[str, object] = {}

            def fake_run_turn(current_developer, **kwargs):  # noqa: ANN001
                captured.update(kwargs)
                paths["developer_output_path"].write_text("- **修订任务**: `M1-T1`\n", encoding="utf-8")
                return current_developer, {"status": "completed"}

            with patch("A07_Development._run_developer_result_turn", side_effect=fake_run_turn), patch(
                "A07_Development.ensure_developer_metadata",
                return_value=(developer, "- **修订任务**: `M1-T1`"),
            ):
                returned, code_change = refine_current_task(
                    developer,
                    paths=paths,
                    task_name="M1-T1",
                    review_msg="修复评审意见",
                )

        self.assertIs(returned, developer)
        self.assertEqual(code_change, "- **修订任务**: `M1-T1`")
        contract = captured["result_contract"]
        self.assertIsInstance(contract, TaskResultContract)
        assert isinstance(contract, TaskResultContract)
        self.assertEqual(contract.mode, "a07_developer_refine")
        self.assertEqual(contract.outcome_artifacts["completed"]["requires"], ("developer_output",))
        self.assertEqual(contract.artifact_rules["developer_output"]["change"], CHANGE_MUST_CHANGE)
        self.assertIn("## 本轮文件契约", str(captured["prompt"]))

    def test_run_development_stage_marks_current_task_true_after_review_pass(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            reviewers = [
                ReviewerRuntime(
                    reviewer_name="测试工程师",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="测试工程师-天英星"),
                    review_md_path=project_dir / "需求A_代码评审记录_测试工程师.md",
                    review_json_path=project_dir / "需求A_评审记录_测试工程师.json",
                    contract=_dummy_contract(),
                )
            ]

            def fake_task_done(**kwargs):  # noqa: ANN001
                paths["task_json_path"].write_text(
                    json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return True

            with patch("A07_Development.cleanup_stale_development_runtime_state", return_value=()), patch(
                "A07_Development.cleanup_existing_development_artifacts",
                return_value=(),
            ), patch(
                "A07_Development.resolve_developer_plan",
                return_value=DeveloperPlan(selection=developer.selection, role_prompt=developer.role_prompt),
            ), patch(
                "A07_Development.create_developer_runtime",
                return_value=developer,
            ), patch(
                "A07_Development.resolve_reviewer_specs",
                return_value=[DevelopmentReviewerSpec(role_name="测试工程师", role_prompt="测试视角", reviewer_key="测试工程师")],
            ), patch(
                "A07_Development.collect_reviewer_agent_selections",
                return_value={"测试工程师": reviewers[0].selection},
            ), patch(
                "A07_Development.build_reviewer_workers",
                return_value=reviewers,
            ), patch(
                "A07_Development.initialize_development_workers",
                return_value=(developer, reviewers),
            ), patch(
                "A07_Development.resolve_subagent_num",
                return_value=0,
            ), patch(
                "A07_Development.develop_current_task",
                return_value=(developer, "代码变更摘要"),
            ), patch(
                "A07_Development._run_parallel_reviewers",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ) as parallel_reviewers, patch(
                "A07_Development.repair_reviewer_outputs",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A07_Development.task_done",
                side_effect=fake_task_done,
            ) as task_done_mock, patch(
                "A07_Development._shutdown_workers",
                return_value=(),
            ):
                result = run_development_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A"],
                )

        self.assertIsInstance(result, DevelopmentStageResult)
        self.assertTrue(result.completed)
        parallel_reviewers.assert_called_once()
        task_done_mock.assert_called_once()

    def test_run_development_stage_exports_live_handoffs_when_preserve_workers_enabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            reviewers = [
                ReviewerRuntime(
                    reviewer_name="测试工程师",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="测试工程师-天英星"),
                    review_md_path=project_dir / "需求A_代码评审记录_测试工程师.md",
                    review_json_path=project_dir / "需求A_评审记录_测试工程师.json",
                    contract=_dummy_contract(),
                )
            ]

            def fake_task_done(**kwargs):  # noqa: ANN001
                paths["task_json_path"].write_text(
                    json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return True

            with patch("A07_Development.cleanup_stale_development_runtime_state", return_value=()), patch(
                "A07_Development.cleanup_existing_development_artifacts",
                return_value=(),
            ), patch(
                "A07_Development.resolve_developer_plan",
                return_value=DeveloperPlan(selection=developer.selection, role_prompt=developer.role_prompt),
            ), patch(
                "A07_Development.create_developer_runtime",
                return_value=developer,
            ), patch(
                "A07_Development.resolve_reviewer_specs",
                return_value=[DevelopmentReviewerSpec(role_name="测试工程师", role_prompt="测试视角", reviewer_key="测试工程师")],
            ), patch(
                "A07_Development.collect_reviewer_agent_selections",
                return_value={"测试工程师": reviewers[0].selection},
            ), patch(
                "A07_Development.build_reviewer_workers",
                return_value=reviewers,
            ), patch(
                "A07_Development.initialize_development_workers",
                return_value=(developer, reviewers),
            ), patch(
                "A07_Development.resolve_subagent_num",
                return_value=0,
            ), patch(
                "A07_Development.develop_current_task",
                return_value=(developer, "代码变更摘要"),
            ), patch(
                "A07_Development._run_parallel_reviewers",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A07_Development.repair_reviewer_outputs",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A07_Development.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A07_Development._shutdown_workers",
                return_value=(),
            ) as shutdown_mock:
                result = run_development_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A"],
                    preserve_workers=True,
                )

        self.assertTrue(result.completed)
        self.assertIsNotNone(result.developer_handoff)
        self.assertEqual(result.developer_handoff.worker, developer.worker)
        self.assertEqual(len(result.reviewer_handoff), 1)
        self.assertEqual(result.reviewer_handoff[0].reviewer_key, "测试工程师")
        self.assertTrue(shutdown_mock.call_args.kwargs["preserve_developer"])
        self.assertEqual(shutdown_mock.call_args.kwargs["preserve_reviewer_keys"], ["测试工程师"])

    def test_run_development_stage_keeps_fresh_reviewers_until_first_review(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            fresh_worker = _FreshReviewerWorker(session_name="测试工程师-天英星")
            reviewers = [
                ReviewerRuntime(
                    reviewer_name="测试工程师",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=fresh_worker,
                    review_md_path=project_dir / "需求A_代码评审记录_测试工程师.md",
                    review_json_path=project_dir / "需求A_评审记录_测试工程师.json",
                    contract=_dummy_contract(),
                )
            ]

            def fake_task_done(**kwargs):  # noqa: ANN001
                paths["task_json_path"].write_text(
                    json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return True

            with patch("A07_Development.cleanup_stale_development_runtime_state", return_value=()), patch(
                "A07_Development.cleanup_existing_development_artifacts",
                return_value=(),
            ), patch(
                "A07_Development.resolve_developer_plan",
                return_value=DeveloperPlan(selection=developer.selection, role_prompt=developer.role_prompt),
            ), patch(
                "A07_Development.create_developer_runtime",
                return_value=developer,
            ), patch(
                "A07_Development.resolve_reviewer_specs",
                return_value=[DevelopmentReviewerSpec(role_name="测试工程师", role_prompt="测试视角", reviewer_key="测试工程师")],
            ), patch(
                "A07_Development.collect_reviewer_agent_selections",
                return_value={"测试工程师": reviewers[0].selection},
            ), patch(
                "A07_Development.build_reviewer_workers",
                return_value=reviewers,
            ), patch(
                "A07_Development.initialize_development_workers",
                return_value=(developer, reviewers),
            ), patch(
                "A07_Development.resolve_subagent_num",
                return_value=0,
            ), patch(
                "A07_Development.develop_current_task",
                return_value=(developer, "代码变更摘要"),
            ), patch(
                "A07_Development._run_parallel_reviewers",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ) as parallel_reviewers, patch(
                "A07_Development.repair_reviewer_outputs",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A07_Development.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A07_Development._shutdown_workers",
                return_value=(),
            ):
                result = run_development_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A"],
                )

        self.assertTrue(result.completed)
        parallel_reviewers.assert_called_once()
        self.assertEqual(fresh_worker.ensure_calls, 1)

    def test_run_development_stage_builds_reviewers_before_first_task_and_overlaps_reviewer_init(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            reviewers = [
                ReviewerRuntime(
                    reviewer_name="测试工程师",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="测试工程师-天英星"),
                    review_md_path=project_dir / "需求A_代码评审记录_测试工程师.md",
                    review_json_path=project_dir / "需求A_评审记录_测试工程师.json",
                    contract=_dummy_contract(),
                )
            ]
            call_order: list[str] = []
            developer_plan = DeveloperPlan(selection=developer.selection, role_prompt=developer.role_prompt)
            developer_init_started = threading.Event()
            reviewer_prelaunch_started = threading.Event()
            reviewer_prelaunch_finished = threading.Event()
            develop_started = threading.Event()
            reviewer_init_started = threading.Event()
            develop_finished = threading.Event()
            reviewer_init_finished = threading.Event()

            def fake_initialize_workers(current_developer, **kwargs):  # noqa: ANN001
                call_order.append(
                    f"initialize_development_workers:{kwargs['initialize_developer']}:{kwargs['initialize_reviewers']}"
                )
                if kwargs["initialize_developer"]:
                    developer_init_started.set()
                    self.assertTrue(reviewer_prelaunch_started.wait(timeout=1.0))
                if kwargs["initialize_reviewers"]:
                    reviewer_init_started.set()
                    self.assertTrue(develop_started.wait(timeout=1.0))
                    reviewer_init_finished.set()
                return current_developer, list(kwargs["reviewers"])

            def fake_prelaunch_reviewers(reviewer_list, **kwargs):  # noqa: ANN001
                _ = kwargs
                call_order.append("prelaunch_development_reviewers")
                reviewer_prelaunch_started.set()
                self.assertTrue(developer_init_started.wait(timeout=1.0))
                reviewer_prelaunch_finished.set()
                return list(reviewer_list)

            def fake_develop(current_developer, **kwargs):  # noqa: ANN001
                call_order.append("develop_current_task")
                self.assertTrue(reviewer_prelaunch_finished.is_set())
                develop_started.set()
                self.assertTrue(reviewer_init_started.wait(timeout=1.0))
                develop_finished.set()
                return current_developer, "代码变更摘要"

            def fake_collect_developer_plan(*args, **kwargs):  # noqa: ANN001
                _ = args
                _ = kwargs
                call_order.append("resolve_developer_plan")
                return developer_plan

            def fake_collect_reviewer_selections(*args, **kwargs):  # noqa: ANN001
                _ = args
                _ = kwargs
                call_order.append("collect_reviewer_agent_selections")
                return {"测试工程师": reviewers[0].selection}

            def fake_create_developer_runtime(*args, **kwargs):  # noqa: ANN001
                _ = args
                _ = kwargs
                call_order.append("create_developer_runtime")
                return developer

            def fake_build_reviewers(*args, **kwargs):  # noqa: ANN001
                _ = args
                _ = kwargs
                call_order.append("build_reviewer_workers")
                return reviewers

            def fake_parallel_reviewers(reviewer_list, **kwargs):  # noqa: ANN001
                _ = kwargs
                self.assertTrue(develop_finished.is_set())
                self.assertTrue(reviewer_init_finished.is_set())
                call_order.append("_run_parallel_reviewers")
                return list(reviewer_list)

            def fake_task_done(**kwargs):  # noqa: ANN001
                paths["task_json_path"].write_text(
                    json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return True

            with patch("A07_Development.cleanup_stale_development_runtime_state", return_value=()), patch(
                "A07_Development.cleanup_existing_development_artifacts",
                return_value=(),
            ), patch(
                "A07_Development.resolve_developer_plan",
                side_effect=fake_collect_developer_plan,
            ), patch(
                "A07_Development.create_developer_runtime",
                side_effect=fake_create_developer_runtime,
            ), patch(
                "A07_Development.resolve_reviewer_specs",
                return_value=[DevelopmentReviewerSpec(role_name="测试工程师", role_prompt="测试视角", reviewer_key="测试工程师")],
            ), patch(
                "A07_Development.collect_reviewer_agent_selections",
                side_effect=fake_collect_reviewer_selections,
            ), patch(
                "A07_Development.build_reviewer_workers",
                side_effect=fake_build_reviewers,
            ), patch(
                "A07_Development.initialize_development_workers",
                side_effect=fake_initialize_workers,
            ), patch(
                "A07_Development.prelaunch_development_reviewers",
                side_effect=fake_prelaunch_reviewers,
            ), patch(
                "A07_Development.resolve_subagent_num",
                return_value=0,
            ), patch(
                "A07_Development.develop_current_task",
                side_effect=fake_develop,
            ), patch(
                "A07_Development._run_parallel_reviewers",
                side_effect=fake_parallel_reviewers,
            ), patch(
                "A07_Development.repair_reviewer_outputs",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A07_Development.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A07_Development._shutdown_workers",
                return_value=(),
            ):
                result = run_development_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A"],
                )

        self.assertTrue(result.completed)
        self.assertEqual(call_order[0:4], [
            "resolve_developer_plan",
            "collect_reviewer_agent_selections",
            "create_developer_runtime",
            "build_reviewer_workers",
        ])
        self.assertLess(call_order.index("build_reviewer_workers"), call_order.index("initialize_development_workers:True:False"))
        self.assertLess(call_order.index("build_reviewer_workers"), call_order.index("prelaunch_development_reviewers"))
        self.assertLess(call_order.index("initialize_development_workers:True:False"), call_order.index("develop_current_task"))
        self.assertLess(call_order.index("prelaunch_development_reviewers"), call_order.index("develop_current_task"))
        self.assertLess(call_order.index("develop_current_task"), call_order.index("initialize_development_workers:False:True"))
        self.assertLess(call_order.index("develop_current_task"), call_order.index("_run_parallel_reviewers"))
        self.assertLess(call_order.index("initialize_development_workers:False:True"), call_order.index("_run_parallel_reviewers"))

    def test_run_development_stage_resumes_from_first_false_task(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": True, "M1-T2": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            reviewers = [
                ReviewerRuntime(
                    reviewer_name="测试工程师",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="测试工程师-天英星"),
                    review_md_path=project_dir / "需求A_代码评审记录_测试工程师.md",
                    review_json_path=project_dir / "需求A_评审记录_测试工程师.json",
                    contract=_dummy_contract(),
                )
            ]
            seen_tasks: list[str] = []

            def fake_develop(current_developer, **kwargs):  # noqa: ANN001
                seen_tasks.append(kwargs["task_name"])
                kwargs["turn_policy"].record_turn()
                return current_developer, "代码变更摘要"

            def fake_task_done(**kwargs):  # noqa: ANN001
                paths["task_json_path"].write_text(
                    json.dumps({"M1": {"M1-T1": True, "M1-T2": True}}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return True

            with patch("A07_Development.cleanup_stale_development_runtime_state", return_value=()), patch(
                "A07_Development.cleanup_existing_development_artifacts",
                return_value=(),
            ), patch(
                "A07_Development.resolve_developer_plan",
                return_value=DeveloperPlan(selection=developer.selection, role_prompt=developer.role_prompt),
            ), patch(
                "A07_Development.create_developer_runtime",
                return_value=developer,
            ), patch(
                "A07_Development.resolve_reviewer_specs",
                return_value=[DevelopmentReviewerSpec(role_name="测试工程师", role_prompt="测试视角", reviewer_key="测试工程师")],
            ), patch(
                "A07_Development.collect_reviewer_agent_selections",
                return_value={"测试工程师": reviewers[0].selection},
            ), patch(
                "A07_Development.build_reviewer_workers",
                return_value=reviewers,
            ), patch(
                "A07_Development.initialize_development_workers",
                return_value=(developer, reviewers),
            ), patch(
                "A07_Development.resolve_subagent_num",
                return_value=0,
            ), patch(
                "A07_Development.develop_current_task",
                side_effect=fake_develop,
            ), patch(
                "A07_Development._run_parallel_reviewers",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A07_Development.repair_reviewer_outputs",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A07_Development.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A07_Development._shutdown_workers",
                return_value=(),
            ):
                result = run_development_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A"],
                )

        self.assertTrue(result.completed)
        self.assertEqual(seen_tasks, ["M1-T2"])

    def test_run_development_stage_shuts_down_when_parallel_reviewer_init_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            reviewers = [
                ReviewerRuntime(
                    reviewer_name="测试工程师",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="测试工程师-天英星"),
                    review_md_path=project_dir / "需求A_代码评审记录_测试工程师.md",
                    review_json_path=project_dir / "需求A_评审记录_测试工程师.json",
                    contract=_dummy_contract(),
                )
            ]

            def fake_initialize_workers(current_developer, **kwargs):  # noqa: ANN001
                if kwargs["initialize_reviewers"]:
                    raise RuntimeError("reviewer init failed")
                return current_developer, list(kwargs["reviewers"])

            with patch("A07_Development.cleanup_stale_development_runtime_state", return_value=()), patch(
                "A07_Development.cleanup_existing_development_artifacts",
                return_value=(),
            ), patch(
                "A07_Development.resolve_developer_plan",
                return_value=DeveloperPlan(selection=developer.selection, role_prompt=developer.role_prompt),
            ), patch(
                "A07_Development.create_developer_runtime",
                return_value=developer,
            ), patch(
                "A07_Development.resolve_reviewer_specs",
                return_value=[DevelopmentReviewerSpec(role_name="测试工程师", role_prompt="测试视角", reviewer_key="测试工程师")],
            ), patch(
                "A07_Development.collect_reviewer_agent_selections",
                return_value={"测试工程师": reviewers[0].selection},
            ), patch(
                "A07_Development.build_reviewer_workers",
                return_value=reviewers,
            ), patch(
                "A07_Development.initialize_development_workers",
                side_effect=fake_initialize_workers,
            ), patch(
                "A07_Development.resolve_subagent_num",
                return_value=0,
            ), patch(
                "A07_Development.develop_current_task",
                return_value=(developer, "代码变更摘要"),
            ), patch(
                "A07_Development._shutdown_workers",
                return_value=(),
            ) as shutdown_workers:
                with self.assertRaisesRegex(RuntimeError, "reviewer init failed"):
                    run_development_stage(
                        ["--project-dir", str(project_dir), "--requirement-name", "需求A"],
                    )

        shutdown_workers.assert_called_once()

    def test_run_development_stage_retries_failed_review_round_until_pass(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            reviewers = [
                ReviewerRuntime(
                    reviewer_name="审核员",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="审核员-天平星"),
                    review_md_path=project_dir / "需求A_代码评审记录_审核员.md",
                    review_json_path=project_dir / "需求A_评审记录_审核员.json",
                    contract=_dummy_contract(),
                )
            ]

            def fake_parallel(reviewers_arg, **kwargs):  # noqa: ANN001
                if kwargs["round_index"] == 1:
                    paths["merged_review_path"].write_text("请补充异常处理\n", encoding="utf-8")
                return list(reviewers_arg)

            task_done_calls = {"count": 0}

            def fake_task_done(**kwargs):  # noqa: ANN001
                task_done_calls["count"] += 1
                if task_done_calls["count"] >= 2:
                    paths["task_json_path"].write_text(
                        json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    return True
                paths["merged_review_path"].write_text("请补充异常处理\n", encoding="utf-8")
                return False

            with patch("A07_Development.cleanup_stale_development_runtime_state", return_value=()), patch(
                "A07_Development.cleanup_existing_development_artifacts",
                return_value=(),
            ), patch(
                "A07_Development.resolve_developer_plan",
                return_value=DeveloperPlan(selection=developer.selection, role_prompt=developer.role_prompt),
            ), patch(
                "A07_Development.create_developer_runtime",
                return_value=developer,
            ), patch(
                "A07_Development.resolve_reviewer_specs",
                return_value=[DevelopmentReviewerSpec(role_name="审核员", role_prompt="审计视角", reviewer_key="审核员")],
            ), patch(
                "A07_Development.collect_reviewer_agent_selections",
                return_value={"审核员": reviewers[0].selection},
            ), patch(
                "A07_Development.build_reviewer_workers",
                return_value=reviewers,
            ), patch(
                "A07_Development.initialize_development_workers",
                return_value=(developer, reviewers),
            ), patch(
                "A07_Development.resolve_subagent_num",
                return_value=0,
            ), patch(
                "A07_Development.develop_current_task",
                return_value=(developer, "首轮代码变更"),
            ), patch(
                "A07_Development._run_parallel_reviewers",
                side_effect=fake_parallel,
            ) as parallel_reviewers, patch(
                "A07_Development.repair_reviewer_outputs",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A07_Development.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A07_Development.refine_current_task",
                return_value=(developer, "修订后代码变更"),
            ) as refine_task, patch(
                "A07_Development._shutdown_workers",
                return_value=(),
            ):
                result = run_development_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A"],
                )

        self.assertTrue(result.completed)
        self.assertEqual(parallel_reviewers.call_count, 2)
        self.assertIs(parallel_reviewers.call_args_list[1].kwargs.get("allow_existing_outputs"), False)
        refine_task.assert_called_once()

    def test_run_development_stage_recreates_workers_before_next_task_when_turn_limit_reached(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False, "M1-T2": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            developer1 = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            developer2 = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天罡星"),
                role_prompt="实现视角",
            )
            reviewers1 = [
                ReviewerRuntime(
                    reviewer_name="审核员",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="审核员-天平星"),
                    review_md_path=project_dir / "需求A_代码评审记录_审核员.md",
                    review_json_path=project_dir / "需求A_评审记录_审核员.json",
                    contract=_dummy_contract(),
                )
            ]
            reviewers2 = [
                ReviewerRuntime(
                    reviewer_name="审核员",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="审核员-天机星"),
                    review_md_path=project_dir / "需求A_代码评审记录_审核员_v2.md",
                    review_json_path=project_dir / "需求A_评审记录_审核员_v2.json",
                    contract=_dummy_contract(),
                )
            ]
            used_developers: list[str] = []
            task_done_count = {"count": 0}

            def fake_develop(current_developer, **kwargs):  # noqa: ANN001
                used_developers.append(current_developer.worker.session_name)
                kwargs["turn_policy"].record_turn()
                return current_developer, f"代码变更-{kwargs['task_name']}"

            def fake_task_done(**kwargs):  # noqa: ANN001
                task_done_count["count"] += 1
                if task_done_count["count"] == 1:
                    paths["task_json_path"].write_text(
                        json.dumps({"M1": {"M1-T1": True, "M1-T2": False}}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                else:
                    paths["task_json_path"].write_text(
                        json.dumps({"M1": {"M1-T1": True, "M1-T2": True}}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                return True

            with patch("A07_Development.cleanup_stale_development_runtime_state", return_value=()), patch(
                "A07_Development.cleanup_existing_development_artifacts",
                return_value=(),
            ), patch(
                "A07_Development.resolve_developer_plan",
                return_value=DeveloperPlan(selection=developer1.selection, role_prompt=developer1.role_prompt),
            ), patch(
                "A07_Development.create_developer_runtime",
                return_value=developer1,
            ), patch(
                "A07_Development.resolve_reviewer_specs",
                return_value=[DevelopmentReviewerSpec(role_name="审核员", role_prompt="审计视角", reviewer_key="审核员")],
            ), patch(
                "A07_Development.collect_reviewer_agent_selections",
                return_value={"审核员": reviewers1[0].selection},
            ), patch(
                "A07_Development.build_reviewer_workers",
                return_value=reviewers1,
            ), patch(
                "A07_Development.initialize_development_workers",
                return_value=(developer1, reviewers1),
            ), patch(
                "A07_Development.resolve_subagent_num",
                return_value=0,
            ), patch(
                "A07_Development.develop_current_task",
                side_effect=fake_develop,
            ), patch(
                "A07_Development._run_parallel_reviewers",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A07_Development.repair_reviewer_outputs",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A07_Development.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A07_Development.recreate_development_workers",
                return_value=(developer2, reviewers2, ("cleanup/runtime1",)),
            ) as recreate_workers, patch(
                "A07_Development._shutdown_workers",
                return_value=(),
            ):
                result = run_development_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--developer-max-turns", "1"],
                )

        self.assertTrue(result.completed)
        recreate_workers.assert_called_once()
        self.assertTrue(recreate_workers.call_args.kwargs.get("initialize_reviewers"))
        self.assertEqual(used_developers, ["开发工程师-天魁星", "开发工程师-天罡星"])

    def test_run_development_stage_does_not_recreate_workers_when_turn_limit_is_infinite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False, "M1-T2": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            reviewers = [
                ReviewerRuntime(
                    reviewer_name="审核员",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="审核员-天平星"),
                    review_md_path=project_dir / "需求A_代码评审记录_审核员.md",
                    review_json_path=project_dir / "需求A_评审记录_审核员.json",
                    contract=_dummy_contract(),
                )
            ]
            used_developers: list[str] = []
            task_done_count = {"count": 0}

            def fake_develop(current_developer, **kwargs):  # noqa: ANN001
                used_developers.append(current_developer.worker.session_name)
                kwargs["turn_policy"].record_turn()
                return current_developer, f"代码变更-{kwargs['task_name']}"

            def fake_task_done(**kwargs):  # noqa: ANN001
                task_done_count["count"] += 1
                if task_done_count["count"] == 1:
                    paths["task_json_path"].write_text(
                        json.dumps({"M1": {"M1-T1": True, "M1-T2": False}}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                else:
                    paths["task_json_path"].write_text(
                        json.dumps({"M1": {"M1-T1": True, "M1-T2": True}}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                return True

            with patch("A07_Development.cleanup_stale_development_runtime_state", return_value=()), patch(
                "A07_Development.cleanup_existing_development_artifacts",
                return_value=(),
            ), patch(
                "A07_Development.resolve_developer_plan",
                return_value=DeveloperPlan(selection=developer.selection, role_prompt=developer.role_prompt),
            ), patch(
                "A07_Development.create_developer_runtime",
                return_value=developer,
            ), patch(
                "A07_Development.resolve_reviewer_specs",
                return_value=[DevelopmentReviewerSpec(role_name="审核员", role_prompt="审计视角", reviewer_key="审核员")],
            ), patch(
                "A07_Development.collect_reviewer_agent_selections",
                return_value={"审核员": reviewers[0].selection},
            ), patch(
                "A07_Development.build_reviewer_workers",
                return_value=reviewers,
            ), patch(
                "A07_Development.initialize_development_workers",
                return_value=(developer, reviewers),
            ), patch(
                "A07_Development.resolve_subagent_num",
                return_value=0,
            ), patch(
                "A07_Development.develop_current_task",
                side_effect=fake_develop,
            ), patch(
                "A07_Development._run_parallel_reviewers",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A07_Development.repair_reviewer_outputs",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A07_Development.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A07_Development.recreate_development_workers",
            ) as recreate_workers, patch(
                "A07_Development._shutdown_workers",
                return_value=(),
            ):
                result = run_development_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--developer-max-turns", "infinite"],
                )

        self.assertTrue(result.completed)
        recreate_workers.assert_not_called()
        self.assertEqual(used_developers, ["开发工程师-天魁星", "开发工程师-天魁星"])

    def test_run_development_stage_resolves_subagent_num_once_per_stage(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False, "M1-T2": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            reviewers = [
                ReviewerRuntime(
                    reviewer_name="审核员",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="审核员-天平星"),
                    review_md_path=project_dir / "需求A_代码评审记录_审核员.md",
                    review_json_path=project_dir / "需求A_评审记录_审核员.json",
                    contract=_dummy_contract(),
                )
            ]
            task_done_count = {"count": 0}

            def fake_task_done(**kwargs):  # noqa: ANN001
                task_done_count["count"] += 1
                if task_done_count["count"] == 1:
                    paths["task_json_path"].write_text(
                        json.dumps({"M1": {"M1-T1": True, "M1-T2": False}}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                else:
                    paths["task_json_path"].write_text(
                        json.dumps({"M1": {"M1-T1": True, "M1-T2": True}}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                return True

            with patch("A07_Development.cleanup_stale_development_runtime_state", return_value=()), patch(
                "A07_Development.cleanup_existing_development_artifacts",
                return_value=(),
            ), patch(
                "A07_Development.resolve_developer_plan",
                return_value=DeveloperPlan(selection=developer.selection, role_prompt=developer.role_prompt),
            ), patch(
                "A07_Development.create_developer_runtime",
                return_value=developer,
            ), patch(
                "A07_Development.resolve_reviewer_specs",
                return_value=[DevelopmentReviewerSpec(role_name="审核员", role_prompt="审计视角", reviewer_key="审核员")],
            ), patch(
                "A07_Development.collect_reviewer_agent_selections",
                return_value={"审核员": reviewers[0].selection},
            ), patch(
                "A07_Development.build_reviewer_workers",
                return_value=reviewers,
            ), patch(
                "A07_Development.initialize_development_workers",
                return_value=(developer, reviewers),
            ), patch(
                "A07_Development.resolve_subagent_num",
                return_value=0,
            ) as resolve_subagent_num, patch(
                "A07_Development.develop_current_task",
                side_effect=lambda current_developer, **kwargs: (current_developer, f"代码变更-{kwargs['task_name']}"),
            ), patch(
                "A07_Development._run_parallel_reviewers",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A07_Development.repair_reviewer_outputs",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A07_Development.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A07_Development._shutdown_workers",
                return_value=(),
            ):
                result = run_development_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A"],
                )

        self.assertTrue(result.completed)
        resolve_subagent_num.assert_called_once()

    def test_run_development_stage_returns_immediately_when_all_tasks_complete(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            result = run_development_stage(
                ["--project-dir", str(project_dir), "--requirement-name", "需求A"],
            )

        self.assertTrue(result.completed)

    def test_run_development_stage_does_not_short_circuit_completed_when_tasks_remain(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_development_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            with patch("A07_Development.cleanup_stale_development_runtime_state", return_value=()), patch(
                "A07_Development.cleanup_existing_development_artifacts",
                return_value=(),
            ), patch(
                "A07_Development.resolve_developer_plan",
                return_value=DeveloperPlan(
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    role_prompt="实现视角",
                ),
            ), patch(
                "A07_Development.resolve_reviewer_specs",
                return_value=[],
            ), patch(
                "A07_Development.collect_reviewer_agent_selections",
                return_value={},
            ), patch(
                "A07_Development.resolve_developer_max_turns",
                return_value=999,
            ), patch(
                "A07_Development.resolve_review_max_rounds",
                return_value=3,
            ), patch(
                "A07_Development.resolve_subagent_num",
                return_value=0,
            ), patch(
                "A07_Development.create_developer_runtime",
                side_effect=RuntimeError("developer runtime should start for unfinished task"),
            ):
                with self.assertRaisesRegex(RuntimeError, "developer runtime should start for unfinished task"):
                    run_development_stage(
                        ["--project-dir", str(project_dir), "--requirement-name", "需求A"],
                    )

    def test_development_runtime_root_is_scoped_by_requirement_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scoped_root = build_development_runtime_root(root, "需求A")
            developer = create_developer_runtime(
                project_dir=root,
                requirement_name="需求A",
                selection=ReviewAgentSelection(vendor="codex", model="gpt-5.4", reasoning_effort="high", proxy_url=""),
                role_prompt="role",
            )

        self.assertEqual(scoped_root, root.resolve() / ".development_runtime" / "需求A")
        self.assertEqual(Path(developer.worker.runtime_root), scoped_root)
        self.assertTrue(str(developer.worker.runtime_dir).startswith(str(scoped_root)))

    def test_shutdown_development_workers_removes_requirement_scoped_runtime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scoped_root = build_development_runtime_root(root, "需求A")
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(
                    session_name="开发工程师-天魁星",
                    runtime_root=scoped_root,
                    runtime_dir=scoped_root / "development-developer-aaaa",
                ),
                role_prompt="实现视角",
            )
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(
                    session_name="测试工程师-天英星",
                    runtime_root=scoped_root,
                    runtime_dir=scoped_root / "development-review-bbbb",
                ),
                review_md_path=root / "需求A_代码评审记录_测试工程师.md",
                review_json_path=root / "需求A_评审记录_测试工程师.json",
                contract=_dummy_contract(),
            )

            removed = shutdown_development_workers(
                developer,
                [reviewer],
                project_dir=root,
                requirement_name="需求A",
                cleanup_runtime=True,
            )

            self.assertTrue(developer.worker.killed)
            self.assertTrue(reviewer.worker.killed)
            self.assertFalse(developer.worker.runtime_dir.exists())
            self.assertFalse(reviewer.worker.runtime_dir.exists())
            self.assertFalse(scoped_root.exists())
            self.assertIn(str(developer.worker.runtime_dir.resolve()), removed)
            self.assertIn(str(reviewer.worker.runtime_dir.resolve()), removed)

    def test_cleanup_stale_development_runtime_state_scopes_by_requirement_and_keeps_live_legacy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_root = root / ".development_runtime"
            target_dir = runtime_root / "target-reviewer"
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "开发-当前需求",
                        "project_dir": str(root.resolve()),
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a07.start",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            other_requirement_dir = runtime_root / "other-requirement"
            other_requirement_dir.mkdir(parents=True, exist_ok=True)
            (other_requirement_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "开发-其他需求",
                        "project_dir": str(root.resolve()),
                        "requirement_name": "需求B",
                        "workflow_action": "stage.a07.start",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            legacy_live_dir = runtime_root / "legacy-live"
            legacy_live_dir.mkdir(parents=True, exist_ok=True)
            (legacy_live_dir / "worker.state.json").write_text(
                json.dumps({"session_name": "开发-遗留存活"}, ensure_ascii=False),
                encoding="utf-8",
            )
            legacy_dead_dir = runtime_root / "legacy-dead"
            legacy_dead_dir.mkdir(parents=True, exist_ok=True)
            (legacy_dead_dir / "worker.state.json").write_text(
                json.dumps({"session_name": "开发-遗留死亡", "agent_state": "DEAD"}, ensure_ascii=False),
                encoding="utf-8",
            )
            prelaunch_dir = runtime_root / "prelaunch-worker"
            prelaunch_dir.mkdir(parents=True, exist_ok=True)
            (prelaunch_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "开发-预启动",
                        "project_dir": str(root.resolve()),
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a07.start",
                        "status": "running",
                        "result_status": "running",
                        "workflow_stage": "pending",
                        "agent_state": "DEAD",
                        "agent_started": False,
                        "pane_id": "",
                        "health_status": "missing_session",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            killed_sessions: list[str] = []

            class FakeTmuxRuntimeController:
                def session_exists(self, session_name: str) -> bool:
                    return session_name in {"开发-遗留存活", "开发-其他需求"}

                def kill_session(self, session_name: str, *, missing_ok: bool = True):  # noqa: ANN001
                    killed_sessions.append(session_name)
                    return session_name

            with patch("tmux_core.stage_kernel.runtime_scope_cleanup.TmuxRuntimeController", FakeTmuxRuntimeController):
                removed = cleanup_stale_development_runtime_state(root, "需求A")

            self.assertFalse(target_dir.exists())
            self.assertFalse(legacy_dead_dir.exists())
            self.assertTrue(prelaunch_dir.exists())
            self.assertTrue(other_requirement_dir.exists())
            self.assertTrue(legacy_live_dir.exists())
            self.assertIn("开发-当前需求", killed_sessions)
            self.assertIn("开发-遗留死亡", killed_sessions)
            self.assertIn(str(target_dir.resolve()), removed)
            self.assertIn(str(legacy_dead_dir.resolve()), removed)

    def test_cleanup_stale_development_runtime_state_scans_requirement_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_root = root / ".development_runtime"
            target_dir = runtime_root / "需求A" / "development-developer-aaaa"
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "开发-当前需求",
                        "project_dir": str(root.resolve()),
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a07.start",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            other_dir = runtime_root / "需求B" / "development-developer-bbbb"
            other_dir.mkdir(parents=True, exist_ok=True)
            (other_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "开发-其他需求",
                        "project_dir": str(root.resolve()),
                        "requirement_name": "需求B",
                        "workflow_action": "stage.a07.start",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            class FakeTmuxRuntimeController:
                def session_exists(self, session_name: str) -> bool:
                    return True

                def kill_session(self, session_name: str, *, missing_ok: bool = True):  # noqa: ANN001
                    return session_name

            with patch("tmux_core.stage_kernel.runtime_scope_cleanup.TmuxRuntimeController", FakeTmuxRuntimeController):
                removed = cleanup_stale_development_runtime_state(root, "需求A")

            self.assertFalse(target_dir.exists())
            self.assertTrue(other_dir.exists())
            self.assertIn(str(target_dir.resolve()), removed)

    def test_cleanup_stale_development_runtime_state_preserves_lock_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_root = root / ".development_runtime"
            target_dir = runtime_root / "需求A" / "development-developer-aaaa"
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "开发-当前需求",
                        "project_dir": str(root.resolve()),
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a07.start",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            lock_child = runtime_root / "_locks" / "stage.a07.start"
            lock_child.mkdir(parents=True, exist_ok=True)

            class FakeTmuxRuntimeController:
                def session_exists(self, session_name: str) -> bool:
                    return True

                def kill_session(self, session_name: str, *, missing_ok: bool = True):  # noqa: ANN001
                    return session_name

            with patch("tmux_core.stage_kernel.runtime_scope_cleanup.TmuxRuntimeController", FakeTmuxRuntimeController):
                removed = cleanup_stale_development_runtime_state(root, "需求A")

            self.assertFalse(target_dir.exists())
            self.assertTrue(lock_child.exists())
            self.assertNotIn(str(lock_child.resolve()), removed)


if __name__ == "__main__":
    unittest.main()
