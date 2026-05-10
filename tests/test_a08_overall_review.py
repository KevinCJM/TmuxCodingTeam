from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from A07_Development import (
    DevelopmentAgentHandoff,
    DevelopmentReviewerSpec,
    DeveloperPlan,
    DeveloperRuntime,
    build_development_paths,
    build_development_runtime_root,
)
from A08_OverallReview import (
    OverallReviewStageResult,
    _build_overall_review_active_code_context,
    bind_reviewer_runtime_from_handoff,
    build_overall_review_reviewer_completion_contract,
    build_overall_review_metadata_repair_result_contract,
    build_overall_review_paths,
    build_overall_review_refine_result_contract,
    discover_live_development_handoffs,
    ensure_overall_review_inputs,
    normalize_overall_review_reviewer_runtime,
    refine_overall_review_code,
    resolve_overall_review_max_rounds,
    run_overall_review_stage,
    run_overall_review_turn_with_recreation,
    build_reviewer_workers,
    _run_overall_review_developer_turn,
    _shutdown_workers as shutdown_overall_review_workers,
)
from tmux_core.runtime.contracts import TaskResultContract, resolve_task_result_decision
from tmux_core.stage_kernel.shared_review import ReviewAgentHandoff, ReviewAgentSelection, ReviewerRuntime
from tmux_core.stage_kernel.agent_intervention import AGENT_INTERVENTION_RECREATE


class _FakeWorker:
    def __init__(
        self,
        *,
        session_name: str,
        runtime_root: str | Path = "/tmp/runtime",
        runtime_dir: str | Path = "/tmp/runtime/worker",
        session_exists_value: bool = True,
        runtime_metadata_payload: dict[str, object] | None = None,
        state_payload: dict[str, object] | None = None,
    ) -> None:
        self.session_name = session_name
        self.runtime_root = Path(runtime_root)
        self.runtime_dir = Path(runtime_dir)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self._session_exists_value = session_exists_value
        self._runtime_metadata = {
            str(key): value
            for key, value in dict(runtime_metadata_payload or {}).items()
            if str(key).strip()
        }
        self._state_payload = dict(state_payload or {})
        self.metadata_updates: list[dict[str, object]] = []
        self.killed = False

    def request_kill(self):
        self.killed = True
        return self.session_name

    def session_exists(self) -> bool:
        return self._session_exists_value

    def set_runtime_metadata(self, **kwargs) -> None:  # noqa: ANN003
        self.metadata_updates.append(dict(kwargs))
        self._runtime_metadata.update(dict(kwargs))
        self._state_payload.update(dict(kwargs))

    def runtime_metadata(self) -> dict[str, str]:
        payload = {
            "session_name": self.session_name,
            "runtime_root": str(self.runtime_root),
            "runtime_dir": str(self.runtime_dir),
        }
        for key, value in self._runtime_metadata.items():
            payload[str(key)] = "" if value is None else str(value)
        return payload

    def read_state(self) -> dict[str, object]:
        return dict(self._state_payload)


def _dummy_contract():
    return object()


def _write_required_inputs(paths: dict[str, Path]) -> None:
    paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
    paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
    paths["detailed_design_path"].write_text("详细设计\n", encoding="utf-8")
    paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")


class A08OverallReviewTests(unittest.TestCase):
    def test_resolve_overall_review_max_rounds_supports_default_and_infinite(self):
        self.assertEqual(resolve_overall_review_max_rounds(argparse.Namespace(review_max_rounds="")), 5)
        self.assertIsNone(resolve_overall_review_max_rounds(argparse.Namespace(review_max_rounds="infinite")))

    def test_resolve_overall_review_max_rounds_prompts_with_back_metadata(self):
        phases: list[str] = []
        progress = type("Progress", (), {"set_phase": lambda _self, phase: phases.append(phase)})()
        with patch("A08_OverallReview.stdin_is_interactive", return_value=True), patch(
            "A08_OverallReview.prompt_review_max_rounds",
            return_value=None,
        ) as prompt_mock:
            value = resolve_overall_review_max_rounds(
                argparse.Namespace(review_max_rounds=""),
                progress=progress,
                allow_back=True,
            )

        self.assertIsNone(value)
        self.assertEqual(phases, ["整体复核 / 配置最大审核轮次"])
        self.assertTrue(prompt_mock.call_args.kwargs["allow_back"])
        self.assertEqual(prompt_mock.call_args.kwargs["stage_key"], "overall_review")

    def test_build_overall_review_reviewer_completion_contract_uses_a08_semantics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            review_md_path = project_dir / "需求A_整体代码复核记录_测试工程师.md"
            review_json_path = project_dir / "需求A_整体复核记录_测试工程师.json"
            review_md_path.write_text("存在问题\n", encoding="utf-8")
            review_json_path.write_text(
                json.dumps([{"task_name": "全面复核", "review_pass": False}], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            contract = build_overall_review_reviewer_completion_contract(
                reviewer_name="测试工程师",
                task_name="全面复核",
                review_md_path=review_md_path,
                review_json_path=review_json_path,
            )

        self.assertEqual(contract.turn_id, "overall_review_全面复核_测试工程师")
        self.assertEqual(contract.phase, "复核阶段")
        self.assertEqual(contract.status_path, review_json_path.resolve())

    def test_normalize_overall_review_reviewer_runtime_rebinds_development_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            legacy_review_md = project_dir / "需求A_代码评审记录_测试工程师.md"
            legacy_review_json = project_dir / "需求A_评审记录_测试工程师.json"
            legacy_review_md.write_text("旧开发评审\n", encoding="utf-8")
            legacy_review_json.write_text("[]", encoding="utf-8")
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="测试工程师-天英星"),
                review_md_path=legacy_review_md,
                review_json_path=legacy_review_json,
                contract=_dummy_contract(),
            )

            normalized = normalize_overall_review_reviewer_runtime(
                reviewer,
                project_dir=project_dir,
                requirement_name="需求A",
            )

            self.assertTrue(normalized.review_md_path.exists())
            self.assertEqual(json.loads(normalized.review_json_path.read_text(encoding="utf-8")), [])
            self.assertTrue(legacy_review_md.exists())
            self.assertTrue(legacy_review_json.exists())

        self.assertEqual(normalized.review_md_path.name, "需求A_整体代码复核记录_测试工程师-天英星.md")
        self.assertEqual(normalized.review_json_path.name, "需求A_整体复核记录_测试工程师-天英星.json")

    def test_bind_reviewer_runtime_precreates_empty_review_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            handoff = ReviewAgentHandoff(
                reviewer_key="测试工程师",
                role_name="测试工程师",
                role_prompt="测试视角",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="测试工程师-天英星"),
            )

            reviewer = bind_reviewer_runtime_from_handoff(
                project_dir=project_dir,
                requirement_name="需求A",
                handoff=handoff,
            )

            self.assertEqual(reviewer.review_md_path.read_text(encoding="utf-8"), "")
            self.assertEqual(json.loads(reviewer.review_json_path.read_text(encoding="utf-8")), [])

    def test_build_reviewer_workers_rebinds_created_development_reviewer_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            legacy_reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="测试工程师-天英星"),
                review_md_path=project_dir / "需求A_代码评审记录_测试工程师.md",
                review_json_path=project_dir / "需求A_评审记录_测试工程师.json",
                contract=_dummy_contract(),
            )
            legacy_reviewer.review_md_path.write_text("", encoding="utf-8")
            legacy_reviewer.review_json_path.write_text("[]", encoding="utf-8")

            with patch("A08_OverallReview.create_reviewer_runtime", return_value=legacy_reviewer):
                reviewers = build_reviewer_workers(
                    argparse.Namespace(),
                    project_dir=project_dir,
                    requirement_name="需求A",
                    reviewer_specs=[DevelopmentReviewerSpec(role_name="测试工程师", role_prompt="测试视角", reviewer_key="测试工程师")],
                    reviewer_handoff=(),
                    reviewer_selections_by_name={"测试工程师": legacy_reviewer.selection},
                )

        self.assertEqual(len(reviewers), 1)
        self.assertEqual(reviewers[0].review_md_path.name, "需求A_整体代码复核记录_测试工程师-天英星.md")
        self.assertEqual(reviewers[0].review_json_path.name, "需求A_整体复核记录_测试工程师-天英星.json")

    def test_build_overall_review_paths_uses_distinct_review_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            development_paths = build_development_paths(tmp_dir, "需求A")
            overall_review_paths = build_overall_review_paths(tmp_dir, "需求A")

        self.assertEqual(overall_review_paths["developer_output_path"], development_paths["developer_output_path"])
        self.assertNotEqual(overall_review_paths["merged_review_path"], development_paths["merged_review_path"])
        self.assertIn("整体代码复核记录", overall_review_paths["merged_review_path"].name)

    def test_active_code_context_lists_project_code_despite_stale_routing_docs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            (project_dir / "docs").mkdir()
            (project_dir / "docs" / "repo_map.json").write_text(
                json.dumps({"scope": {"notes": ["No business logic or application source code is present"]}}),
                encoding="utf-8",
            )
            (project_dir / "text_stats.py").write_text("print('ok')\n", encoding="utf-8")

            context = _build_overall_review_active_code_context(project_dir)

        self.assertIn("text_stats.py", context)
        self.assertIn("不能仅凭 `repo_map.json`", context)

    def test_shutdown_overall_review_workers_removes_requirement_scoped_runtime(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            scoped_root = build_development_runtime_root(project_dir, "需求A")
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
                review_md_path=project_dir / "需求A_整体代码复核记录_测试工程师.md",
                review_json_path=project_dir / "需求A_整体复核记录_测试工程师.json",
                contract=_dummy_contract(),
            )

            removed = shutdown_overall_review_workers(
                developer,
                [reviewer],
                project_dir=project_dir,
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

    def test_metadata_repair_contract_uses_file_contract_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = build_overall_review_paths(tmp_dir, "需求A")
            normal_contract = build_overall_review_refine_result_contract(paths)
            repair_contract = build_overall_review_metadata_repair_result_contract(paths)

        self.assertEqual(repair_contract.mode, normal_contract.mode)
        self.assertEqual(repair_contract.expected_statuses, ("completed",))
        self.assertEqual(repair_contract.required_artifacts, normal_contract.required_artifacts)

    def test_overall_review_refine_contract_resolves_from_developer_output_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = build_overall_review_paths(tmp_dir, "需求A")
            contract = build_overall_review_refine_result_contract(paths)
            empty_contract = TaskResultContract(
                turn_id="a08_developer_refine_all_code",
                phase="a08_developer_refine_all_code",
                task_kind="a08_developer_refine_all_code",
                mode="a08_developer_refine_all_code",
                expected_statuses=("completed",),
                optional_artifacts={"developer_output": paths["developer_output_path"]},
            )
            with self.assertRaisesRegex(ValueError, "复核阶段开发元数据"):
                resolve_task_result_decision(empty_contract)

            paths["developer_output_path"].write_text("修订摘要\n", encoding="utf-8")
            decision = resolve_task_result_decision(contract)

        self.assertEqual(decision.status, "completed")
        self.assertIn("developer_output", decision.artifacts)

    def test_refine_overall_review_code_uses_repair_contract_only_for_metadata_repair(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = build_overall_review_paths(tmp_dir, "需求A")
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            calls = []

            def fake_run(current_developer, **kwargs):  # noqa: ANN001
                calls.append(kwargs)
                if kwargs["label"] == "overall_review_refine_all_code":
                    paths["developer_output_path"].write_text("修订摘要\n", encoding="utf-8")
                return current_developer

            with patch("A08_OverallReview._run_overall_review_developer_turn", side_effect=fake_run), patch(
                "A08_OverallReview.check_develop_job",
                return_value="请补全工程师开发内容元数据，然后只返回 任务完成",
            ):
                _, code_change = refine_overall_review_code(
                    developer,
                    project_dir=tmp_dir,
                    requirement_name="需求A",
                    paths=paths,
                    review_msg="复核意见",
                )

        self.assertEqual(code_change, "修订摘要")
        self.assertEqual(
            [call["label"] for call in calls],
            ["overall_review_refine_all_code", "overall_review_refine_all_code_metadata_repair"],
        )
        self.assertEqual(calls[0]["result_contract"].mode, "a08_developer_refine_all_code")
        self.assertEqual(calls[1]["result_contract"].mode, "a08_developer_refine_all_code")
        self.assertEqual(calls[0]["result_contract"].expected_statuses, ("completed",))
        self.assertEqual(calls[1]["result_contract"].expected_statuses, ("completed",))

    def test_overall_review_refine_turn_uses_repair_contract_for_internal_repair_attempts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = build_overall_review_paths(tmp_dir, "需求A")
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            calls = []

            def fake_run_task_result_turn_with_repair(**kwargs):  # noqa: ANN003
                calls.append(kwargs)
                return {}

            with patch(
                "A08_OverallReview.run_task_result_turn_with_repair",
                side_effect=fake_run_task_result_turn_with_repair,
            ):
                returned = _run_overall_review_developer_turn(
                    developer,
                    project_dir=tmp_dir,
                    requirement_name="需求A",
                    label="overall_review_refine_all_code",
                    prompt="请修复",
                    result_contract=build_overall_review_refine_result_contract(paths),
                    paths=paths,
                )

        self.assertIs(returned, developer)
        self.assertEqual(calls[0]["repair_result_contract"].mode, "a08_developer_refine_all_code")
        self.assertEqual(calls[0]["result_contract"].expected_statuses, ("completed",))
        self.assertEqual(calls[0]["repair_result_contract"].expected_statuses, ("completed",))

    def test_ensure_overall_review_inputs_requires_all_tasks_completed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = build_overall_review_paths(tmp_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "任务单 JSON 必须全部完成"):
                ensure_overall_review_inputs(project_dir=tmp_dir, requirement_name="需求A")

    def test_run_overall_review_stage_reuses_live_handoffs_and_marks_state_passed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_overall_review_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            reviewer_worker = _FakeWorker(session_name="测试工程师-天英星")
            reviewer_handoff = ReviewAgentHandoff(
                reviewer_key="测试工程师",
                role_name="测试工程师",
                role_prompt="测试视角",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=reviewer_worker,
            )

            def fake_task_done(**kwargs):  # noqa: ANN001
                paths["merged_review_path"].write_text("", encoding="utf-8")
                return True

            with patch("A08_OverallReview.initialize_overall_review_reviewers", side_effect=lambda reviewers, **kwargs: list(reviewers)), patch(
                "A08_OverallReview._run_parallel_overall_reviewers",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A08_OverallReview.repair_overall_review_outputs",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A08_OverallReview.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A08_OverallReview.create_developer_runtime",
            ) as create_developer_runtime_mock, patch(
                "A08_OverallReview._shutdown_workers",
                return_value=(),
            ):
                result = run_overall_review_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A"],
                    developer_handoff=DevelopmentAgentHandoff(
                        selection=developer.selection,
                        role_prompt=developer.role_prompt,
                        worker=developer.worker,
                    ),
                    reviewer_handoff=(reviewer_handoff,),
                )

            self.assertIsInstance(result, OverallReviewStageResult)
            self.assertTrue(result.completed)
            self.assertTrue(json.loads(Path(result.state_path).read_text(encoding="utf-8"))["passed"])
            create_developer_runtime_mock.assert_not_called()

    def test_run_overall_review_stage_creates_developer_after_failed_review(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_overall_review_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="测试工程师-天英星"),
                review_md_path=project_dir / "需求A_整体代码复核记录_测试工程师.md",
                review_json_path=project_dir / "需求A_整体复核记录_测试工程师.json",
                contract=_dummy_contract(),
            )
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            task_done_results = iter([False, True])

            def fake_task_done(**kwargs):  # noqa: ANN001
                passed = next(task_done_results)
                paths["merged_review_path"].write_text("" if passed else "存在复核问题\n", encoding="utf-8")
                return passed

            with patch(
                "A08_OverallReview.resolve_developer_plan",
                return_value=DeveloperPlan(selection=developer.selection, role_prompt=developer.role_prompt),
            ), patch(
                "A08_OverallReview.resolve_overall_review_reviewer_specs",
                return_value=[DevelopmentReviewerSpec(role_name="测试工程师", role_prompt="测试视角", reviewer_key="测试工程师")],
            ), patch(
                "A08_OverallReview.collect_reviewer_agent_selections",
                return_value={"测试工程师": reviewer.selection},
            ), patch(
                "A08_OverallReview.build_reviewer_workers",
                return_value=[reviewer],
            ), patch(
                "A08_OverallReview.initialize_overall_review_reviewers",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A08_OverallReview._run_parallel_overall_reviewers",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A08_OverallReview.repair_overall_review_outputs",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A08_OverallReview.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A08_OverallReview.create_developer_runtime",
                return_value=developer,
            ) as create_developer_runtime_mock, patch(
                "A08_OverallReview.initialize_overall_review_developer",
                return_value=developer,
            ) as initialize_developer_mock, patch(
                "A08_OverallReview.refine_overall_review_code",
                return_value=(developer, "修订摘要"),
            ) as refine_mock, patch(
                "A08_OverallReview._shutdown_workers",
                return_value=(),
            ):
                result = run_overall_review_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A"],
                )

            self.assertTrue(result.completed)
            create_developer_runtime_mock.assert_called_once()
            initialize_developer_mock.assert_called_once()
            refine_mock.assert_called_once()
            self.assertTrue(json.loads(Path(result.state_path).read_text(encoding="utf-8"))["passed"])

    def test_run_overall_review_stage_allows_ambiguity_only_findings_as_passed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_overall_review_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            reviewer_handoff = ReviewAgentHandoff(
                reviewer_key="测试工程师",
                role_name="测试工程师",
                role_prompt="测试视角",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="测试工程师-天英星"),
            )

            def fake_task_done(**kwargs):  # noqa: ANN001
                paths["merged_review_path"].write_text("- [Ambiguity] 任务单 Markdown 状态展示存在歧义\n", encoding="utf-8")
                return False

            with patch("A08_OverallReview.initialize_overall_review_reviewers", side_effect=lambda reviewers, **kwargs: list(reviewers)), patch(
                "A08_OverallReview._run_parallel_overall_reviewers",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A08_OverallReview.repair_overall_review_outputs",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A08_OverallReview.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A08_OverallReview.create_developer_runtime",
            ) as create_developer_runtime_mock, patch(
                "A08_OverallReview.refine_overall_review_code",
            ) as refine_mock, patch(
                "A08_OverallReview._shutdown_workers",
                return_value=(),
            ):
                result = run_overall_review_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A"],
                    developer_handoff=DevelopmentAgentHandoff(
                        selection=developer.selection,
                        role_prompt=developer.role_prompt,
                        worker=developer.worker,
                    ),
                    reviewer_handoff=(reviewer_handoff,),
                )

            self.assertTrue(result.completed)
            self.assertTrue(json.loads(Path(result.state_path).read_text(encoding="utf-8"))["passed"])
            create_developer_runtime_mock.assert_not_called()
            refine_mock.assert_not_called()

    def test_run_overall_review_stage_stops_at_review_max_rounds(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_overall_review_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            developer = DeveloperRuntime(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                role_prompt="实现视角",
            )
            reviewer_handoff = ReviewAgentHandoff(
                reviewer_key="测试工程师",
                role_name="测试工程师",
                role_prompt="测试视角",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="测试工程师-天英星"),
            )

            def fake_task_done(**kwargs):  # noqa: ANN001
                paths["merged_review_path"].write_text("- [Error] 缺少关键测试覆盖\n", encoding="utf-8")
                return False

            with patch("A08_OverallReview.initialize_overall_review_reviewers", side_effect=lambda reviewers, **kwargs: list(reviewers)), patch(
                "A08_OverallReview._run_parallel_overall_reviewers",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A08_OverallReview.repair_overall_review_outputs",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A08_OverallReview.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A08_OverallReview.create_developer_runtime",
            ) as create_developer_runtime_mock, patch(
                "A08_OverallReview.refine_overall_review_code",
            ) as refine_mock, patch(
                "A08_OverallReview._shutdown_workers",
                return_value=(),
            ):
                with self.assertRaisesRegex(RuntimeError, "整体复核超过最大审核轮次 1"):
                    run_overall_review_stage(
                        ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "1"],
                        developer_handoff=DevelopmentAgentHandoff(
                            selection=developer.selection,
                            role_prompt=developer.role_prompt,
                            worker=developer.worker,
                        ),
                        reviewer_handoff=(reviewer_handoff,),
                    )

            self.assertFalse(json.loads(paths["state_path"].read_text(encoding="utf-8"))["passed"])
            create_developer_runtime_mock.assert_not_called()
            refine_mock.assert_not_called()

    def test_run_overall_review_turn_with_recreation_rebinds_recreated_reviewer_runtime(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_overall_review_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            initial_reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="测试工程师-天英星"),
                review_md_path=project_dir / "需求A_代码评审记录_测试工程师.md",
                review_json_path=project_dir / "需求A_评审记录_测试工程师.json",
                contract=_dummy_contract(),
            )
            replacement = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=initial_reviewer.selection,
                worker=_FakeWorker(session_name="测试工程师-天英星"),
                review_md_path=project_dir / "需求A_代码评审记录_测试工程师.md",
                review_json_path=project_dir / "需求A_评审记录_测试工程师.json",
                contract=_dummy_contract(),
            )
            completion_paths = []

            def fake_run_completion_turn_with_repair(**kwargs):  # noqa: ANN003
                completion_paths.append(
                    (
                        kwargs["completion_contract"].turn_id,
                        kwargs["completion_contract"].phase,
                        kwargs["completion_contract"].status_path.name,
                        kwargs["completion_contract"].tracked_artifacts["review_md"].name,
                    )
                )
                if len(completion_paths) == 1:
                    raise RuntimeError("dead worker")
                return {}

            with patch(
                "A08_OverallReview.run_completion_turn_with_repair",
                side_effect=fake_run_completion_turn_with_repair,
            ), patch(
                "A08_OverallReview.is_worker_death_error",
                side_effect=lambda error: str(error) == "dead worker",
            ), patch(
                "A08_OverallReview.request_worker_manual_intervention",
                return_value=AGENT_INTERVENTION_RECREATE,
            ), patch(
                "A08_OverallReview.recreate_development_reviewer_runtime",
                return_value=replacement,
            ), patch(
                "A08_OverallReview._run_single_overall_review_reviewer_init",
                return_value=replacement,
            ):
                result = run_overall_review_turn_with_recreation(
                    initial_reviewer,
                    project_dir=project_dir,
                    requirement_name="需求A",
                    reviewer_spec=DevelopmentReviewerSpec(role_name="测试工程师", role_prompt="测试视角", reviewer_key="测试工程师"),
                    paths=paths,
                    reviewer_specs_by_name={"测试工程师": DevelopmentReviewerSpec(role_name="测试工程师", role_prompt="测试视角", reviewer_key="测试工程师")},
                    label="overall_review_initial_测试工程师_round_1",
                    prompt="执行复核",
                )

        self.assertIsNotNone(result)
        self.assertEqual(len(completion_paths), 2)
        self.assertEqual(completion_paths[0][0], "overall_review_全面复核_测试工程师")
        self.assertEqual(completion_paths[0][1], "复核阶段")
        self.assertEqual(completion_paths[0][2], "需求A_整体复核记录_测试工程师-天英星.json")
        self.assertEqual(completion_paths[0][3], "需求A_整体代码复核记录_测试工程师-天英星.md")
        self.assertEqual(completion_paths[1], completion_paths[0])

    def test_run_overall_review_turn_with_recreation_escalates_repeated_death_to_manual_reconfiguration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_overall_review_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            initial_reviewer = ReviewerRuntime(
                reviewer_name="架构师",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="架构师-天究星"),
                review_md_path=project_dir / "需求A_代码评审记录_架构师.md",
                review_json_path=project_dir / "需求A_评审记录_架构师.json",
                contract=_dummy_contract(),
            )
            replacement1 = ReviewerRuntime(
                reviewer_name="架构师",
                selection=initial_reviewer.selection,
                worker=_FakeWorker(session_name="架构师-天究星#2"),
                review_md_path=initial_reviewer.review_md_path,
                review_json_path=initial_reviewer.review_json_path,
                contract=_dummy_contract(),
            )
            replacement2 = ReviewerRuntime(
                reviewer_name="架构师",
                selection=ReviewAgentSelection("claude", "sonnet", "high", ""),
                worker=_FakeWorker(session_name="架构师-天究星#3"),
                review_md_path=initial_reviewer.review_md_path,
                review_json_path=initial_reviewer.review_json_path,
                contract=_dummy_contract(),
            )
            attempts = {"count": 0}

            def death_death_success(**kwargs):  # noqa: ANN003
                attempts["count"] += 1
                if attempts["count"] <= 2:
                    raise RuntimeError("tmux pane died")
                return {}

            with patch(
                "A08_OverallReview.run_completion_turn_with_repair",
                side_effect=death_death_success,
            ), patch(
                "A08_OverallReview.request_worker_manual_intervention",
                return_value=AGENT_INTERVENTION_RECREATE,
            ), patch(
                "A08_OverallReview.recreate_development_reviewer_runtime",
                side_effect=[replacement1, replacement2],
            ) as recreate_runtime, patch(
                "A08_OverallReview._run_single_overall_review_reviewer_init",
                side_effect=lambda reviewer, **kwargs: reviewer,
            ):
                result = run_overall_review_turn_with_recreation(
                    initial_reviewer,
                    project_dir=project_dir,
                    requirement_name="需求A",
                    reviewer_spec=DevelopmentReviewerSpec(role_name="架构师", role_prompt="架构视角", reviewer_key="架构师"),
                    paths=paths,
                    reviewer_specs_by_name={"架构师": DevelopmentReviewerSpec(role_name="架构师", role_prompt="架构视角", reviewer_key="架构师")},
                    label="overall_review_initial_架构师_round_1",
                    prompt="执行复核",
                )

        self.assertIsNotNone(result)
        self.assertEqual(result.worker.session_name, replacement2.worker.session_name)
        self.assertEqual(result.selection.vendor, replacement2.selection.vendor)
        self.assertEqual(attempts["count"], 3)
        self.assertEqual(recreate_runtime.call_args_list[0].kwargs["force_model_change"], False)
        self.assertEqual(recreate_runtime.call_args_list[1].kwargs["force_model_change"], True)
        self.assertTrue(recreate_runtime.call_args_list[1].kwargs["required_reconfiguration"])
        self.assertIn("连续 2 次死亡/失败", recreate_runtime.call_args_list[1].kwargs["reason_text"])
        self.assertIn("智能体进程已死亡或退出", recreate_runtime.call_args_list[1].kwargs["reason_text"])

    def test_discover_live_development_handoffs_reads_a07_runtime_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            resolved_project_dir = str(project_dir.resolve())
            runtime_root = project_dir / ".development_runtime"
            developer_state_path = runtime_root / "dev-worker" / "worker.state.json"
            reviewer_state_path = runtime_root / "reviewer-worker" / "worker.state.json"
            developer_state_path.parent.mkdir(parents=True, exist_ok=True)
            reviewer_state_path.parent.mkdir(parents=True, exist_ok=True)
            for state_path, payload in (
                (
                    developer_state_path,
                    {
                        "worker_id": "development-developer",
                        "session_name": "开发工程师-天魁星",
                        "work_dir": resolved_project_dir,
                        "project_dir": resolved_project_dir,
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a07.start",
                        "role_prompt": "实现视角",
                        "updated_at": "2026-04-23T10:00:00+08:00",
                        "config": {
                            "vendor": "codex",
                            "model": "gpt-5.4",
                            "reasoning_effort": "high",
                            "proxy_url": "",
                        },
                    },
                ),
                (
                    reviewer_state_path,
                    {
                        "worker_id": "development-review-测试工程师",
                        "session_name": "测试工程师-天英星",
                        "work_dir": resolved_project_dir,
                        "project_dir": resolved_project_dir,
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a07.start",
                        "role_prompt": "测试视角",
                        "role_name": "测试工程师",
                        "reviewer_key": "测试工程师",
                        "updated_at": "2026-04-23T10:00:01+08:00",
                        "config": {
                            "vendor": "codex",
                            "model": "gpt-5.4",
                            "reasoning_effort": "high",
                            "proxy_url": "",
                        },
                    },
                ),
            ):
                state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            fake_developer_worker = _FakeWorker(session_name="开发工程师-天魁星")
            fake_reviewer_worker = _FakeWorker(session_name="测试工程师-天英星")

            with patch(
                "A08_OverallReview.load_worker_from_state_path",
                side_effect=[fake_developer_worker, fake_reviewer_worker],
            ):
                developer_handoff, reviewer_handoff = discover_live_development_handoffs(project_dir, "需求A")

        self.assertIsNotNone(developer_handoff)
        self.assertEqual(developer_handoff.selection.vendor, "codex")
        self.assertEqual(developer_handoff.role_prompt, "实现视角")
        self.assertEqual(len(reviewer_handoff), 1)
        self.assertEqual(reviewer_handoff[0].reviewer_key, "测试工程师")
        self.assertEqual(reviewer_handoff[0].role_prompt, "测试视角")

    def test_discover_live_development_handoffs_prefers_stage_a08_runtime_over_a07(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            resolved_project_dir = str(project_dir.resolve())
            runtime_root = project_dir / ".development_runtime"
            payloads = {
                runtime_root / "a07-dev" / "worker.state.json": {
                    "worker_id": "development-developer",
                    "session_name": "开发工程师-天魁星",
                    "work_dir": resolved_project_dir,
                    "project_dir": resolved_project_dir,
                    "requirement_name": "需求A",
                    "workflow_action": "stage.a07.start",
                    "role_prompt": "A07 实现视角",
                    "updated_at": "2026-04-26T10:00:02+08:00",
                    "config": {
                        "vendor": "codex",
                        "model": "gpt-5.4",
                        "reasoning_effort": "high",
                        "proxy_url": "",
                    },
                },
                runtime_root / "a08-dev" / "worker.state.json": {
                    "worker_id": "development-developer",
                    "session_name": "开发工程师-地默星",
                    "work_dir": resolved_project_dir,
                    "project_dir": resolved_project_dir,
                    "requirement_name": "需求A",
                    "workflow_action": "stage.a08.start",
                    "role_prompt": "A08 实现视角",
                    "updated_at": "2026-04-26T10:00:01+08:00",
                    "config": {
                        "vendor": "claude",
                        "model": "sonnet",
                        "reasoning_effort": "high",
                        "proxy_url": "",
                    },
                },
                runtime_root / "a07-review" / "worker.state.json": {
                    "worker_id": "development-review-测试工程师",
                    "session_name": "测试工程师-天英星",
                    "work_dir": resolved_project_dir,
                    "project_dir": resolved_project_dir,
                    "requirement_name": "需求A",
                    "workflow_action": "stage.a07.start",
                    "role_prompt": "A07 测试视角",
                    "role_name": "测试工程师",
                    "reviewer_key": "测试工程师",
                    "updated_at": "2026-04-26T10:00:03+08:00",
                    "config": {
                        "vendor": "codex",
                        "model": "gpt-5.4",
                        "reasoning_effort": "high",
                        "proxy_url": "",
                    },
                },
                runtime_root / "a08-review" / "worker.state.json": {
                    "worker_id": "development-review-测试工程师",
                    "session_name": "测试工程师-张月鹿",
                    "work_dir": resolved_project_dir,
                    "project_dir": resolved_project_dir,
                    "requirement_name": "需求A",
                    "workflow_action": "stage.a08.start",
                    "role_prompt": "A08 测试视角",
                    "role_name": "测试工程师",
                    "reviewer_key": "测试工程师",
                    "updated_at": "2026-04-26T10:00:00+08:00",
                    "config": {
                        "vendor": "claude",
                        "model": "sonnet",
                        "reasoning_effort": "high",
                        "proxy_url": "",
                    },
                },
            }
            workers_by_path: dict[str, _FakeWorker] = {}
            for state_path, payload in payloads.items():
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                workers_by_path[str(state_path.resolve())] = _FakeWorker(
                    session_name=str(payload["session_name"]),
                    runtime_metadata_payload={
                        "workflow_action": payload["workflow_action"],
                        "updated_at": payload["updated_at"],
                    },
                )

            with patch(
                "A08_OverallReview.load_worker_from_state_path",
                side_effect=lambda state_path: workers_by_path[str(Path(state_path).resolve())],
            ):
                developer_handoff, reviewer_handoff = discover_live_development_handoffs(project_dir, "需求A")

        self.assertIsNotNone(developer_handoff)
        self.assertEqual(developer_handoff.worker.session_name, "开发工程师-地默星")
        self.assertEqual(developer_handoff.selection.vendor, "claude")
        self.assertEqual(len(reviewer_handoff), 1)
        self.assertEqual(reviewer_handoff[0].worker.session_name, "测试工程师-张月鹿")
        self.assertEqual(reviewer_handoff[0].selection.vendor, "claude")

    def test_discover_live_development_handoffs_reuses_failed_but_active_developer_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            resolved_project_dir = str(project_dir.resolve())
            runtime_root = project_dir / ".development_runtime"
            developer_state_path = runtime_root / "development-developer-31501633" / "worker.state.json"
            payload = {
                "worker_id": "development-developer",
                "session_name": "开发工程师-地默星",
                "work_dir": resolved_project_dir,
                "project_dir": resolved_project_dir,
                "requirement_name": "需求A",
                "workflow_action": "stage.a08.start",
                "role_prompt": "A08 实现视角",
                "updated_at": "2026-04-28T17:23:50+08:00",
                "status": "failed",
                "result_status": "failed",
                "note": "error:overall_review_refine_all_code_metadata_repair",
                "health_status": "alive",
                "agent_state": "READY",
                "current_task_runtime_status": "running",
                "config": {
                    "vendor": "codex",
                    "model": "gpt-5.4",
                    "reasoning_effort": "high",
                    "proxy_url": "",
                },
            }
            developer_state_path.parent.mkdir(parents=True, exist_ok=True)
            developer_state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            fake_developer_worker = _FakeWorker(
                session_name="开发工程师-地默星",
                runtime_metadata_payload={
                    "workflow_action": payload["workflow_action"],
                    "updated_at": payload["updated_at"],
                },
                state_payload=payload,
            )

            with patch(
                "A08_OverallReview.load_worker_from_state_path",
                return_value=fake_developer_worker,
            ):
                developer_handoff, reviewer_handoff = discover_live_development_handoffs(project_dir, "需求A")

        self.assertIsNotNone(developer_handoff)
        self.assertEqual(developer_handoff.worker.session_name, "开发工程师-地默星")
        self.assertEqual(developer_handoff.selection.vendor, "codex")
        self.assertEqual(reviewer_handoff, ())

    def test_discover_live_development_handoffs_reuses_awaiting_reconfig_but_active_reviewer_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            resolved_project_dir = str(project_dir.resolve())
            runtime_root = project_dir / ".development_runtime"
            reviewer_state_path = runtime_root / "development-review-0f01cc24" / "worker.state.json"
            payload = {
                "worker_id": "development-review-测试工程师",
                "session_name": "测试工程师-张月鹿",
                "work_dir": resolved_project_dir,
                "project_dir": resolved_project_dir,
                "requirement_name": "需求A",
                "workflow_action": "stage.a08.start",
                "role_prompt": "测试视角",
                "role_name": "测试工程师",
                "reviewer_key": "测试工程师",
                "updated_at": "2026-04-28T19:36:57+08:00",
                "status": "running",
                "result_status": "running",
                "note": "awaiting_reconfig",
                "health_status": "alive",
                "agent_state": "BUSY",
                "current_task_runtime_status": "running",
                "config": {
                    "vendor": "gemini",
                    "model": "flash",
                    "reasoning_effort": "high",
                    "proxy_url": "",
                },
            }
            reviewer_state_path.parent.mkdir(parents=True, exist_ok=True)
            reviewer_state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            fake_reviewer_worker = _FakeWorker(
                session_name="测试工程师-张月鹿",
                runtime_metadata_payload={
                    "workflow_action": payload["workflow_action"],
                    "updated_at": payload["updated_at"],
                },
                state_payload=payload,
            )

            with patch(
                "A08_OverallReview.load_worker_from_state_path",
                return_value=fake_reviewer_worker,
            ):
                developer_handoff, reviewer_handoff = discover_live_development_handoffs(project_dir, "需求A")

        self.assertIsNone(developer_handoff)
        self.assertEqual(len(reviewer_handoff), 1)
        self.assertEqual(reviewer_handoff[0].worker.session_name, "测试工程师-张月鹿")
        self.assertEqual(reviewer_handoff[0].selection.vendor, "gemini")

    def test_discover_live_development_handoffs_skips_awaiting_reconfig_reviewer_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            resolved_project_dir = str(project_dir.resolve())
            runtime_root = project_dir / ".development_runtime"
            payloads = {
                runtime_root / "stale-review" / "worker.state.json": {
                    "worker_id": "development-review-测试工程师",
                    "session_name": "测试工程师-旧会话",
                    "work_dir": resolved_project_dir,
                    "project_dir": resolved_project_dir,
                    "requirement_name": "需求A",
                    "workflow_action": "stage.a08.start",
                    "role_prompt": "旧测试视角",
                    "role_name": "测试工程师",
                    "reviewer_key": "测试工程师",
                    "updated_at": "2026-04-26T10:00:09+08:00",
                    "status": "running",
                    "result_status": "running",
                    "note": "awaiting_reconfig",
                    "health_status": "awaiting_reconfig",
                    "health_note": "需要重新选择模型",
                    "agent_state": "STARTING",
                    "config": {
                        "vendor": "codex",
                        "model": "gpt-5.4",
                        "reasoning_effort": "high",
                        "proxy_url": "",
                    },
                },
                runtime_root / "live-review" / "worker.state.json": {
                    "worker_id": "development-review-测试工程师",
                    "session_name": "测试工程师-新会话",
                    "work_dir": resolved_project_dir,
                    "project_dir": resolved_project_dir,
                    "requirement_name": "需求A",
                    "workflow_action": "stage.a08.start",
                    "role_prompt": "新测试视角",
                    "role_name": "测试工程师",
                    "reviewer_key": "测试工程师",
                    "updated_at": "2026-04-26T10:00:08+08:00",
                    "status": "ready",
                    "result_status": "ready",
                    "health_status": "alive",
                    "agent_state": "READY",
                    "config": {
                        "vendor": "claude",
                        "model": "sonnet",
                        "reasoning_effort": "high",
                        "proxy_url": "",
                    },
                },
            }
            workers_by_path: dict[str, _FakeWorker] = {}
            for state_path, payload in payloads.items():
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                workers_by_path[str(state_path.resolve())] = _FakeWorker(
                    session_name=str(payload["session_name"]),
                    runtime_metadata_payload={
                        "workflow_action": payload["workflow_action"],
                        "updated_at": payload["updated_at"],
                    },
                    state_payload={
                        "status": payload["status"],
                        "result_status": payload["result_status"],
                        "note": payload.get("note", ""),
                        "health_status": payload["health_status"],
                        "agent_state": payload["agent_state"],
                    },
                )

            with patch(
                "A08_OverallReview.load_worker_from_state_path",
                side_effect=lambda state_path: workers_by_path[str(Path(state_path).resolve())],
            ):
                developer_handoff, reviewer_handoff = discover_live_development_handoffs(project_dir, "需求A")

        self.assertIsNone(developer_handoff)
        self.assertEqual(len(reviewer_handoff), 1)
        self.assertEqual(reviewer_handoff[0].worker.session_name, "测试工程师-新会话")
        self.assertEqual(reviewer_handoff[0].selection.vendor, "claude")

    def test_run_overall_review_stage_uses_discovered_live_reviewer_handoff_for_specs_and_workers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_overall_review_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            developer_handoff = DevelopmentAgentHandoff(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                role_prompt="实现视角",
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
            )
            dead_reviewer_handoff = ReviewAgentHandoff(
                reviewer_key="测试工程师",
                role_name="测试工程师",
                role_prompt="已失效配置",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="测试工程师-已失效", session_exists_value=False),
            )
            live_reviewer_handoff = ReviewAgentHandoff(
                reviewer_key="测试工程师",
                role_name="测试工程师",
                role_prompt="发现到的存活配置",
                selection=ReviewAgentSelection("claude", "sonnet", "high", ""),
                worker=_FakeWorker(session_name="测试工程师-天英星"),
            )
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=live_reviewer_handoff.selection,
                worker=live_reviewer_handoff.worker,
                review_md_path=project_dir / "需求A_整体代码复核记录_测试工程师.md",
                review_json_path=project_dir / "需求A_整体复核记录_测试工程师.json",
                contract=_dummy_contract(),
            )

            def fake_task_done(**kwargs):  # noqa: ANN001
                paths["merged_review_path"].write_text("", encoding="utf-8")
                return True

            with patch(
                "A08_OverallReview.discover_live_development_handoffs",
                return_value=(None, (live_reviewer_handoff,)),
            ), patch(
                "A08_OverallReview.resolve_overall_review_reviewer_specs",
                return_value=[DevelopmentReviewerSpec(role_name="测试工程师", role_prompt="测试视角", reviewer_key="测试工程师")],
            ) as resolve_specs_mock, patch(
                "A08_OverallReview.collect_reviewer_agent_selections",
                return_value={},
            ), patch(
                "A08_OverallReview.build_reviewer_workers",
                return_value=[reviewer],
            ) as build_reviewer_workers_mock, patch(
                "A08_OverallReview.initialize_overall_review_reviewers",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A08_OverallReview._run_parallel_overall_reviewers",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A08_OverallReview.repair_overall_review_outputs",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A08_OverallReview.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A08_OverallReview._shutdown_workers",
                return_value=(),
            ):
                result = run_overall_review_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A"],
                    developer_handoff=developer_handoff,
                    reviewer_handoff=(dead_reviewer_handoff,),
                )

            self.assertTrue(result.completed)
            self.assertEqual(
                resolve_specs_mock.call_args.kwargs.get("reviewer_handoff"),
                (live_reviewer_handoff,),
            )
            self.assertEqual(
                build_reviewer_workers_mock.call_args.kwargs.get("reviewer_handoff"),
                (live_reviewer_handoff,),
            )

    def test_run_overall_review_stage_ignores_explicit_awaiting_reconfig_reviewer_handoff(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_overall_review_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            explicit_stale_reviewer_handoff = ReviewAgentHandoff(
                reviewer_key="测试工程师",
                role_name="测试工程师",
                role_prompt="等待重配的旧会话",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(
                    session_name="测试工程师-旧会话",
                    session_exists_value=True,
                    state_payload={
                        "status": "running",
                        "result_status": "running",
                        "note": "awaiting_reconfig",
                        "health_status": "awaiting_reconfig",
                        "health_note": "需要重新选择模型",
                        "agent_state": "STARTING",
                    },
                ),
            )
            live_reviewer_handoff = ReviewAgentHandoff(
                reviewer_key="测试工程师",
                role_name="测试工程师",
                role_prompt="可复用的新会话",
                selection=ReviewAgentSelection("claude", "sonnet", "high", ""),
                worker=_FakeWorker(session_name="测试工程师-新会话"),
            )
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=live_reviewer_handoff.selection,
                worker=live_reviewer_handoff.worker,
                review_md_path=project_dir / "需求A_整体代码复核记录_测试工程师.md",
                review_json_path=project_dir / "需求A_整体复核记录_测试工程师.json",
                contract=_dummy_contract(),
            )

            def fake_task_done(**kwargs):  # noqa: ANN001
                paths["merged_review_path"].write_text("", encoding="utf-8")
                return True

            with patch(
                "A08_OverallReview.discover_live_development_handoffs",
                return_value=(None, (live_reviewer_handoff,)),
            ), patch(
                "A08_OverallReview.resolve_overall_review_reviewer_specs",
                return_value=[DevelopmentReviewerSpec(role_name="测试工程师", role_prompt="测试视角", reviewer_key="测试工程师")],
            ), patch(
                "A08_OverallReview.collect_reviewer_agent_selections",
                return_value={},
            ), patch(
                "A08_OverallReview.build_reviewer_workers",
                return_value=[reviewer],
            ) as build_reviewer_workers_mock, patch(
                "A08_OverallReview.initialize_overall_review_reviewers",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A08_OverallReview._run_parallel_overall_reviewers",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A08_OverallReview.repair_overall_review_outputs",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A08_OverallReview.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A08_OverallReview._shutdown_workers",
                return_value=(),
            ):
                result = run_overall_review_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A"],
                    reviewer_handoff=(explicit_stale_reviewer_handoff,),
                )

        self.assertTrue(result.completed)
        self.assertEqual(
            build_reviewer_workers_mock.call_args.kwargs.get("reviewer_handoff"),
            (live_reviewer_handoff,),
        )

    def test_run_overall_review_stage_prefers_discovered_stage_a08_handoffs_without_reprompting(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_overall_review_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            explicit_a07_developer_handoff = DevelopmentAgentHandoff(
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                role_prompt="A07 实现视角",
                worker=_FakeWorker(
                    session_name="开发工程师-天魁星",
                    runtime_metadata_payload={"workflow_action": "stage.a07.start", "updated_at": "2026-04-26T10:00:02+08:00"},
                ),
            )
            explicit_a07_reviewer_handoff = ReviewAgentHandoff(
                reviewer_key="测试工程师",
                role_name="测试工程师",
                role_prompt="A07 测试视角",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(
                    session_name="测试工程师-天英星",
                    runtime_metadata_payload={"workflow_action": "stage.a07.start", "updated_at": "2026-04-26T10:00:03+08:00"},
                ),
            )
            live_a08_developer_handoff = DevelopmentAgentHandoff(
                selection=ReviewAgentSelection("claude", "sonnet", "high", ""),
                role_prompt="A08 实现视角",
                worker=_FakeWorker(
                    session_name="开发工程师-地默星",
                    runtime_metadata_payload={"workflow_action": "stage.a08.start", "updated_at": "2026-04-26T10:00:01+08:00"},
                ),
            )
            live_a08_reviewer_handoff = ReviewAgentHandoff(
                reviewer_key="测试工程师",
                role_name="测试工程师",
                role_prompt="A08 测试视角",
                selection=ReviewAgentSelection("claude", "sonnet", "high", ""),
                worker=_FakeWorker(
                    session_name="测试工程师-张月鹿",
                    runtime_metadata_payload={"workflow_action": "stage.a08.start", "updated_at": "2026-04-26T10:00:00+08:00"},
                ),
            )
            live_reviewer_runtime = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=live_a08_reviewer_handoff.selection,
                worker=live_a08_reviewer_handoff.worker,
                review_md_path=project_dir / "需求A_整体代码复核记录_测试工程师-张月鹿.md",
                review_json_path=project_dir / "需求A_整体复核记录_测试工程师-张月鹿.json",
                contract=_dummy_contract(),
            )
            bound_developer = DeveloperRuntime(
                selection=live_a08_developer_handoff.selection,
                worker=live_a08_developer_handoff.worker,
                role_prompt=live_a08_developer_handoff.role_prompt,
            )

            def fake_task_done(**kwargs):  # noqa: ANN001
                paths["merged_review_path"].write_text("", encoding="utf-8")
                return True

            with patch(
                "A08_OverallReview.discover_live_development_handoffs",
                return_value=(live_a08_developer_handoff, (live_a08_reviewer_handoff,)),
            ), patch(
                "A08_OverallReview.bind_developer_runtime_from_handoff",
                return_value=bound_developer,
            ) as bind_developer_mock, patch(
                "A08_OverallReview.resolve_overall_review_reviewer_specs",
                return_value=[DevelopmentReviewerSpec(role_name="测试工程师", role_prompt="测试视角", reviewer_key="测试工程师")],
            ), patch(
                "A08_OverallReview.collect_reviewer_agent_selections",
            ) as collect_reviewers_mock, patch(
                "A08_OverallReview.build_reviewer_workers",
                return_value=[live_reviewer_runtime],
            ) as build_reviewer_workers_mock, patch(
                "A08_OverallReview.initialize_overall_review_reviewers",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A08_OverallReview._run_parallel_overall_reviewers",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A08_OverallReview.repair_overall_review_outputs",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A08_OverallReview.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A08_OverallReview.resolve_developer_plan",
            ) as resolve_developer_plan_mock, patch(
                "A08_OverallReview.create_developer_runtime",
            ) as create_developer_runtime_mock, patch(
                "A08_OverallReview._shutdown_workers",
                return_value=(),
            ):
                result = run_overall_review_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A"],
                    developer_handoff=explicit_a07_developer_handoff,
                    reviewer_handoff=(explicit_a07_reviewer_handoff,),
                )

        self.assertTrue(result.completed)
        bind_kwargs = bind_developer_mock.call_args.kwargs
        self.assertEqual(bind_kwargs.get("handoff"), live_a08_developer_handoff)
        self.assertEqual(
            build_reviewer_workers_mock.call_args.kwargs.get("reviewer_handoff"),
            (live_a08_reviewer_handoff,),
        )
        collect_reviewers_mock.assert_not_called()
        resolve_developer_plan_mock.assert_not_called()
        create_developer_runtime_mock.assert_not_called()

    def test_run_overall_review_stage_keeps_explicit_live_stage_a08_reviewer_over_discovered_stale_a08(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_overall_review_paths(project_dir, "需求A")
            _write_required_inputs(paths)
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            explicit_live_reviewer_handoff = ReviewAgentHandoff(
                reviewer_key="测试工程师",
                role_name="测试工程师",
                role_prompt="当前轮新模型",
                selection=ReviewAgentSelection("claude", "sonnet", "high", ""),
                worker=_FakeWorker(
                    session_name="测试工程师-新会话",
                    runtime_metadata_payload={"workflow_action": "stage.a08.start"},
                    state_payload={
                        "status": "ready",
                        "result_status": "ready",
                        "health_status": "alive",
                        "agent_state": "READY",
                    },
                ),
            )
            discovered_stale_reviewer_handoff = ReviewAgentHandoff(
                reviewer_key="测试工程师",
                role_name="测试工程师",
                role_prompt="旧模型残留",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(
                    session_name="测试工程师-旧会话",
                    runtime_metadata_payload={"workflow_action": "stage.a08.start", "updated_at": "2026-04-26T10:00:09+08:00"},
                    state_payload={
                        "status": "ready",
                        "result_status": "ready",
                        "health_status": "alive",
                        "agent_state": "READY",
                    },
                ),
            )
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=explicit_live_reviewer_handoff.selection,
                worker=explicit_live_reviewer_handoff.worker,
                review_md_path=project_dir / "需求A_整体代码复核记录_测试工程师-新会话.md",
                review_json_path=project_dir / "需求A_整体复核记录_测试工程师-新会话.json",
                contract=_dummy_contract(),
            )

            def fake_task_done(**kwargs):  # noqa: ANN001
                paths["merged_review_path"].write_text("", encoding="utf-8")
                return True

            with patch(
                "A08_OverallReview.discover_live_development_handoffs",
                return_value=(None, (discovered_stale_reviewer_handoff,)),
            ), patch(
                "A08_OverallReview.resolve_overall_review_reviewer_specs",
                return_value=[DevelopmentReviewerSpec(role_name="测试工程师", role_prompt="测试视角", reviewer_key="测试工程师")],
            ), patch(
                "A08_OverallReview.collect_reviewer_agent_selections",
                return_value={},
            ) as collect_reviewers_mock, patch(
                "A08_OverallReview.build_reviewer_workers",
                return_value=[reviewer],
            ) as build_reviewer_workers_mock, patch(
                "A08_OverallReview.initialize_overall_review_reviewers",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A08_OverallReview._run_parallel_overall_reviewers",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A08_OverallReview.repair_overall_review_outputs",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A08_OverallReview.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A08_OverallReview._shutdown_workers",
                return_value=(),
            ):
                result = run_overall_review_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A"],
                    reviewer_handoff=(explicit_live_reviewer_handoff,),
                )

        self.assertTrue(result.completed)
        self.assertEqual(
            build_reviewer_workers_mock.call_args.kwargs.get("reviewer_handoff"),
            (explicit_live_reviewer_handoff,),
        )
        collect_reviewers_mock.assert_not_called()
