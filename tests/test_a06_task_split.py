from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, patch

from A06_TaskSplit import (
    ReviewAgentHandoff,
    ReviewAgentSelection,
    RequirementsAnalystHandoff,
    TaskSplitReviewerSpec,
    TaskSplitStageResult,
    _run_reviewer_result_turn,
    _run_reviewer_turn_with_resume,
    _shutdown_workers,
    build_parser,
    build_reviewer_init_result_contract,
    build_reviewer_workers,
    build_task_split_paths,
    cleanup_stale_task_split_runtime_state,
    decide_existing_task_split_mode,
    ensure_task_split_inputs,
    generate_task_split_json,
    initialize_task_split_workers,
    resolve_review_max_rounds,
    run_ba_modify_loop,
    run_task_split_stage,
)
from tmux_core.runtime.contracts import TurnFileContract, TurnFileResult
from tmux_core.stage_kernel.shared_review import ReviewerRuntime
from tmux_core.stage_kernel.agent_intervention import AGENT_INTERVENTION_RECREATE
from T08_pre_development import build_pre_development_task_record_path


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


def _dummy_contract() -> TurnFileContract:
    def validator(path: Path) -> TurnFileResult:
        return TurnFileResult(
            status_path=str(path),
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


class A06TaskSplitTests(unittest.TestCase):
    def test_shutdown_workers_cleans_reused_detailed_design_runtime_workers_on_success(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            detailed_runtime_root = project_dir / ".detailed_design_runtime" / "需求A"
            ba_worker = _FakeWorker(
                session_name="需求分析师-天佑星",
                runtime_root=detailed_runtime_root,
                runtime_dir=detailed_runtime_root / "ba",
            )
            reviewer_worker = _FakeWorker(
                session_name="审核员-地雄星",
                runtime_root=detailed_runtime_root,
                runtime_dir=detailed_runtime_root / "reviewer",
            )
            ba_handoff = RequirementsAnalystHandoff(
                worker=ba_worker,
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            reviewer = ReviewerRuntime(
                reviewer_name="审核员",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=reviewer_worker,
                review_md_path=project_dir / "需求A_任务单评审记录_审核员.md",
                review_json_path=project_dir / "需求A_评审记录_审核员.json",
                contract=_dummy_contract(),
            )

            removed = _shutdown_workers(
                ba_handoff,
                [reviewer],
                project_dir=project_dir,
                requirement_name="需求A",
                cleanup_runtime=True,
            )

            self.assertTrue(ba_worker.killed)
            self.assertTrue(reviewer_worker.killed)
            self.assertFalse(ba_worker.runtime_dir.exists())
            self.assertFalse(reviewer_worker.runtime_dir.exists())
            self.assertIn(str((detailed_runtime_root / "ba").resolve()), removed)
            self.assertIn(str((detailed_runtime_root / "reviewer").resolve()), removed)

    def test_shutdown_workers_keeps_runtime_for_failure_debug_when_cleanup_disabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            detailed_runtime_root = project_dir / ".detailed_design_runtime" / "需求A"
            ba_worker = _FakeWorker(
                session_name="需求分析师-天佑星",
                runtime_root=detailed_runtime_root,
                runtime_dir=detailed_runtime_root / "ba",
            )
            ba_handoff = RequirementsAnalystHandoff(
                worker=ba_worker,
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )

            removed = _shutdown_workers(
                ba_handoff,
                [],
                project_dir=project_dir,
                requirement_name="需求A",
                cleanup_runtime=False,
            )

            self.assertFalse(ba_worker.killed)
            self.assertTrue(ba_worker.runtime_dir.exists())
            self.assertEqual(removed, ())

    def test_resolve_review_max_rounds_supports_default_and_infinite(self):
        args = build_parser().parse_args([])
        self.assertEqual(resolve_review_max_rounds(args), 5)

        args = build_parser().parse_args(["--review-max-rounds", "infinite"])
        self.assertIsNone(resolve_review_max_rounds(args))

    def test_resolve_review_max_rounds_rejects_invalid_cli_value(self):
        args = build_parser().parse_args(["--review-max-rounds", "0"])

        with self.assertRaisesRegex(RuntimeError, "必须是正整数或 infinite"):
            resolve_review_max_rounds(args)

    def test_resolve_review_max_rounds_prompts_in_interactive_mode(self):
        args = build_parser().parse_args([])

        with patch("A06_TaskSplit.stdin_is_interactive", return_value=True), patch(
            "A06_TaskSplit.prompt_review_max_rounds",
            return_value=None,
        ) as prompt_mock:
            value = resolve_review_max_rounds(args, progress=object())

        self.assertIsNone(value)
        prompt_mock.assert_called_once()

    def test_predict_worker_display_name_includes_tmux_sessions_in_occupied_pool(self):
        import A06_TaskSplit as task_split_module

        observed: dict[str, set[str]] = {}

        def fake_build_session_name(worker_id, work_dir, vendor, instance_id="", occupied_session_names=None):  # noqa: ANN001
            _ = worker_id
            _ = work_dir
            _ = vendor
            _ = instance_id
            observed["occupied"] = {str(item).strip() for item in occupied_session_names or () if str(item).strip()}
            return "需求分析师-天佑星"

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "A06_TaskSplit.build_session_name",
            side_effect=fake_build_session_name,
        ), patch(
            "A06_TaskSplit.list_tmux_session_names",
            return_value=["审核员-天勇星"],
        ), patch(
            "A06_TaskSplit.list_registered_tmux_workers",
            return_value=[],
        ):
            session_name = task_split_module._predict_worker_display_name(
                project_dir=tmpdir,
                worker_id="task-split-analyst",
                occupied_session_names=("预占用-会话",),
            )

        self.assertEqual(session_name, "需求分析师-天佑星")
        self.assertIn("审核员-天勇星", observed["occupied"])
        self.assertIn("预占用-会话", observed["occupied"])

    def test_decide_existing_task_split_mode_supports_three_way_choice_when_task_json_valid(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_task_split_paths(project_dir, "需求A")
            paths["task_md_path"].write_text("已有任务单\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            args = build_parser().parse_args(["--project-dir", str(project_dir), "--requirement-name", "需求A"])

            with patch("A06_TaskSplit.stdin_is_interactive", return_value=True), patch(
                "A06_TaskSplit.prompt_select_option",
                return_value="review_existing",
            ) as prompt_mock:
                mode = decide_existing_task_split_mode(args, paths=paths)

        self.assertEqual(mode, "review_existing")
        self.assertEqual(
            tuple(item[0] for item in prompt_mock.call_args.kwargs["options"]),
            ("skip", "review_existing", "rerun"),
        )

    def test_decide_existing_task_split_mode_disallows_skip_when_task_json_missing_or_invalid(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_task_split_paths(project_dir, "需求A")
            paths["task_md_path"].write_text("已有任务单\n", encoding="utf-8")
            args = build_parser().parse_args(["--project-dir", str(project_dir), "--requirement-name", "需求A"])

            with patch("A06_TaskSplit.stdin_is_interactive", return_value=True), patch(
                "A06_TaskSplit.prompt_select_option",
                return_value="review_existing",
            ) as prompt_mock:
                mode = decide_existing_task_split_mode(args, paths=paths)

        self.assertEqual(mode, "review_existing")
        self.assertEqual(
            tuple(item[0] for item in prompt_mock.call_args.kwargs["options"]),
            ("review_existing", "rerun"),
        )

    def test_run_task_split_stage_skips_when_existing_task_outputs_are_confirmed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_task_split_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
            paths["detailed_design_path"].write_text("详细设计\n", encoding="utf-8")
            paths["task_md_path"].write_text("已有任务单\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            with patch("A06_TaskSplit.stdin_is_interactive", return_value=True), patch(
                "A06_TaskSplit.prompt_select_option",
                return_value="skip",
            ) as prompt_mock, patch(
                "A06_TaskSplit.resolve_review_max_rounds",
            ) as resolve_rounds_mock, patch(
                "A06_TaskSplit.cleanup_stale_task_split_runtime_state",
                return_value=("/tmp/task-split-runtime",),
            ) as cleanup_runtime, patch(
                "A06_TaskSplit.cleanup_existing_task_split_artifacts",
            ) as cleanup_artifacts, patch(
                "A06_TaskSplit.prepare_task_split_ba_handoff",
            ) as prepare_ba, patch(
                "A06_TaskSplit.build_reviewer_workers",
            ) as build_reviewers:
                result = run_task_split_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5"],
                )

            self.assertIsInstance(result, TaskSplitStageResult)
            self.assertTrue(result.passed)
            prompt_text = prompt_mock.call_args.kwargs["title"]
            self.assertIn("需求A_任务单.md", prompt_text)
            self.assertIn("需求A_任务单.json", prompt_text)
            cleanup_runtime.assert_called_once_with(str(project_dir.resolve()), "需求A")
            cleanup_artifacts.assert_not_called()
            prepare_ba.assert_not_called()
            build_reviewers.assert_not_called()
            resolve_rounds_mock.assert_not_called()
            self.assertEqual(result.cleanup_paths, ("/tmp/task-split-runtime",))
            pre_development_path = build_pre_development_task_record_path(project_dir, "需求A")
            payload = json.loads(pre_development_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["任务拆分"]["任务拆分"])

    def test_run_task_split_stage_reruns_when_existing_task_outputs_are_not_skipped(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_task_split_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
            paths["detailed_design_path"].write_text("详细设计\n", encoding="utf-8")
            paths["task_md_path"].write_text("已有任务单\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            ba_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(session_name="需求分析师-天佑星"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )

            def fake_generate_task_split_document(handoff, **kwargs):  # noqa: ANN001
                paths["task_md_path"].write_text("重新生成任务单\n", encoding="utf-8")
                return handoff

            def fake_generate_task_split_json(handoff, **kwargs):  # noqa: ANN001
                paths["task_json_path"].write_text(
                    json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return handoff

            with patch("A06_TaskSplit.stdin_is_interactive", return_value=True), patch(
                "A06_TaskSplit.prompt_select_option",
                return_value="rerun",
            ) as prompt_mock, patch(
                "A06_TaskSplit.cleanup_stale_task_split_runtime_state",
                return_value=(),
            ) as cleanup_runtime, patch(
                "A06_TaskSplit.cleanup_existing_task_split_artifacts",
                return_value=(),
            ) as cleanup_artifacts, patch(
                "A06_TaskSplit.prepare_task_split_ba_handoff",
                return_value=(ba_handoff, False),
            ), patch(
                "A06_TaskSplit.resolve_reviewer_specs",
                return_value=[],
            ), patch(
                "A06_TaskSplit.build_reviewer_workers",
                return_value=([], []),
            ), patch(
                "A06_TaskSplit.ensure_active_reviewers",
                return_value=None,
            ), patch(
                "A06_TaskSplit.initialize_task_split_workers",
                return_value=(ba_handoff, []),
            ), patch(
                "A06_TaskSplit.generate_task_split_document",
                side_effect=fake_generate_task_split_document,
            ) as generate_task_md, patch(
                "A06_TaskSplit.task_done",
                return_value=True,
            ), patch(
                "A06_TaskSplit.generate_task_split_json",
                side_effect=fake_generate_task_split_json,
            ) as generate_task_json, patch(
                "A06_TaskSplit._shutdown_workers",
                return_value=(),
            ):
                result = run_task_split_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5"],
                )

            self.assertTrue(result.passed)
            prompt_mock.assert_called_once()
            cleanup_runtime.assert_called_once()
            cleanup_artifacts.assert_called_once_with(
                paths,
                "需求A",
                clear_task_md=True,
                clear_task_json=True,
                audit_context=ANY,
            )
            generate_task_md.assert_called_once()
            generate_task_json.assert_called_once()
            self.assertEqual(paths["task_md_path"].read_text(encoding="utf-8"), "重新生成任务单\n")

    def test_run_task_split_stage_does_not_offer_skip_for_invalid_task_json_or_yes_mode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_task_split_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
            paths["detailed_design_path"].write_text("详细设计\n", encoding="utf-8")
            paths["task_md_path"].write_text("已有任务单\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": "invalid"}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            ba_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(session_name="需求分析师-天佑星"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )

            def fake_generate_task_split_document(handoff, **kwargs):  # noqa: ANN001
                paths["task_md_path"].write_text("重新生成任务单\n", encoding="utf-8")
                return handoff

            def fake_generate_task_split_json(handoff, **kwargs):  # noqa: ANN001
                paths["task_json_path"].write_text(
                    json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return handoff

            with patch("A06_TaskSplit.stdin_is_interactive", return_value=True), patch(
                "A06_TaskSplit.prompt_select_option",
            ) as prompt_mock, patch(
                "A06_TaskSplit.cleanup_stale_task_split_runtime_state",
                return_value=(),
            ) as cleanup_runtime, patch(
                "A06_TaskSplit.cleanup_existing_task_split_artifacts",
                return_value=(),
            ), patch(
                "A06_TaskSplit.prepare_task_split_ba_handoff",
                return_value=(ba_handoff, False),
            ), patch(
                "A06_TaskSplit.resolve_reviewer_specs",
                return_value=[],
            ), patch(
                "A06_TaskSplit.build_reviewer_workers",
                return_value=([], []),
            ), patch(
                "A06_TaskSplit.ensure_active_reviewers",
                return_value=None,
            ), patch(
                "A06_TaskSplit.initialize_task_split_workers",
                return_value=(ba_handoff, []),
            ), patch(
                "A06_TaskSplit.generate_task_split_document",
                side_effect=fake_generate_task_split_document,
            ), patch(
                "A06_TaskSplit.task_done",
                return_value=True,
            ), patch(
                "A06_TaskSplit.generate_task_split_json",
                side_effect=fake_generate_task_split_json,
            ), patch(
                "A06_TaskSplit._shutdown_workers",
                return_value=(),
            ):
                result = run_task_split_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5", "--yes"],
                )

            self.assertTrue(result.passed)
            prompt_mock.assert_not_called()
            cleanup_runtime.assert_called_once()

    def test_run_task_split_stage_review_existing_reuses_live_agents_and_keeps_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_task_split_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
            paths["detailed_design_path"].write_text("详细设计\n", encoding="utf-8")
            paths["task_md_path"].write_text("已有任务单\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            ba_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(session_name="需求分析师-天佑星"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            reviewer_handoff = (
                ReviewAgentHandoff(
                    reviewer_key="开发工程师",
                    role_name="开发工程师",
                    role_prompt="实现视角",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="开发工程师-天魁星"),
                ),
            )
            reviewers = [
                ReviewerRuntime(
                    reviewer_name="开发工程师",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=reviewer_handoff[0].worker,
                    review_md_path=project_dir / "需求A_任务单评审记录_开发工程师.md",
                    review_json_path=project_dir / "需求A_评审记录_开发工程师.json",
                    contract=_dummy_contract(),
                )
            ]

            def fake_task_done(**kwargs):  # noqa: ANN001
                Path(kwargs["directory"], kwargs["md_output_name"]).write_text("", encoding="utf-8")
                return True

            with patch("A06_TaskSplit.stdin_is_interactive", return_value=True), patch(
                "A06_TaskSplit.prompt_select_option",
                return_value="review_existing",
            ), patch(
                "A06_TaskSplit.cleanup_stale_task_split_runtime_state",
                return_value=(),
            ) as cleanup_runtime, patch(
                "A06_TaskSplit.cleanup_existing_task_split_artifacts",
                return_value=(),
            ) as cleanup_artifacts, patch(
                "A06_TaskSplit.prepare_task_split_ba_handoff",
            ) as prepare_ba, patch(
                "A06_TaskSplit.resolve_reviewer_specs",
                return_value=[TaskSplitReviewerSpec(role_name="开发工程师", role_prompt="实现视角", reviewer_key="开发工程师")],
            ), patch(
                "A06_TaskSplit.build_reviewer_workers",
                return_value=(reviewers, []),
            ) as build_reviewers, patch(
                "A06_TaskSplit._run_parallel_reviewers",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A06_TaskSplit.repair_reviewer_outputs",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A06_TaskSplit.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A06_TaskSplit.ensure_active_reviewers",
                return_value=None,
            ), patch(
                "A06_TaskSplit.generate_task_split_document",
            ) as generate_task_md, patch(
                "A06_TaskSplit.generate_task_split_json",
            ) as generate_task_json, patch(
                "A06_TaskSplit._shutdown_workers",
                return_value=(),
            ):
                result = run_task_split_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5"],
                    ba_handoff=ba_handoff,
                    reviewer_handoff=reviewer_handoff,
                )

            self.assertTrue(result.passed)
            cleanup_runtime.assert_called_once()
            cleanup_artifacts.assert_called_once_with(
                paths,
                "需求A",
                clear_task_md=False,
                clear_task_json=False,
                audit_context=ANY,
            )
            build_reviewers.assert_called_once()
            self.assertEqual(build_reviewers.call_args.kwargs["reviewer_handoff"], reviewer_handoff)
            prepare_ba.assert_not_called()
            generate_task_md.assert_not_called()
            generate_task_json.assert_not_called()

    def test_run_task_split_stage_review_existing_reuses_live_ba_on_failed_review_and_regenerates_json_after_modify(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_task_split_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
            paths["detailed_design_path"].write_text("详细设计\n", encoding="utf-8")
            paths["task_md_path"].write_text("已有任务单\n", encoding="utf-8")
            paths["task_json_path"].write_text(
                json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            ba_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(session_name="需求分析师-天佑星"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            reviewer_handoff = (
                ReviewAgentHandoff(
                    reviewer_key="开发工程师",
                    role_name="开发工程师",
                    role_prompt="实现视角",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="开发工程师-天魁星"),
                ),
            )
            reviewers = [
                ReviewerRuntime(
                    reviewer_name="开发工程师",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=reviewer_handoff[0].worker,
                    review_md_path=project_dir / "需求A_任务单评审记录_开发工程师.md",
                    review_json_path=project_dir / "需求A_评审记录_开发工程师.json",
                    contract=_dummy_contract(),
                )
            ]
            task_done_results = iter([False, True])

            def fake_task_done(**kwargs):  # noqa: ANN001
                passed = next(task_done_results)
                merged_review = Path(kwargs["directory"], kwargs["md_output_name"])
                merged_review.write_text("" if passed else "请补充分工边界\n", encoding="utf-8")
                return passed

            def fake_modify_loop(handoff, **kwargs):  # noqa: ANN001
                kwargs["paths"]["task_md_path"].write_text("优化后的任务单\n", encoding="utf-8")
                kwargs["paths"]["ba_feedback_path"].write_text("已根据评审优化任务单\n", encoding="utf-8")
                return handoff

            def fake_generate_task_split_json(handoff, **kwargs):  # noqa: ANN001
                kwargs["paths"]["task_json_path"].write_text(
                    json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return handoff

            with patch("A06_TaskSplit.stdin_is_interactive", return_value=True), patch(
                "A06_TaskSplit.prompt_select_option",
                return_value="review_existing",
            ), patch(
                "A06_TaskSplit.cleanup_stale_task_split_runtime_state",
                return_value=(),
            ), patch(
                "A06_TaskSplit.cleanup_existing_task_split_artifacts",
                return_value=(),
            ), patch(
                "A06_TaskSplit.prepare_task_split_ba_handoff",
            ) as prepare_ba, patch(
                "A06_TaskSplit.resolve_reviewer_specs",
                return_value=[TaskSplitReviewerSpec(role_name="开发工程师", role_prompt="实现视角", reviewer_key="开发工程师")],
            ), patch(
                "A06_TaskSplit.build_reviewer_workers",
                return_value=(reviewers, []),
            ), patch(
                "A06_TaskSplit._run_parallel_reviewers",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A06_TaskSplit.repair_reviewer_outputs",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A06_TaskSplit.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A06_TaskSplit.ensure_active_reviewers",
                return_value=None,
            ), patch(
                "A06_TaskSplit.run_ba_modify_loop",
                side_effect=fake_modify_loop,
            ) as modify_loop, patch(
                "A06_TaskSplit.generate_task_split_json",
                side_effect=fake_generate_task_split_json,
            ) as generate_task_json, patch(
                "A06_TaskSplit._shutdown_workers",
                return_value=(),
            ):
                result = run_task_split_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5"],
                    ba_handoff=ba_handoff,
                    reviewer_handoff=reviewer_handoff,
                )

            self.assertTrue(result.passed)
            prepare_ba.assert_not_called()
            modify_loop.assert_called_once()
            generate_task_json.assert_called_once()
            self.assertEqual(paths["task_md_path"].read_text(encoding="utf-8"), "优化后的任务单\n")

    def test_run_task_split_stage_review_existing_creates_ba_lazily_for_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_task_split_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
            paths["detailed_design_path"].write_text("详细设计\n", encoding="utf-8")
            paths["task_md_path"].write_text("已有任务单\n", encoding="utf-8")
            reviewer = ReviewerRuntime(
                reviewer_name="开发工程师",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="开发工程师-天魁星"),
                review_md_path=project_dir / "需求A_任务单评审记录_开发工程师.md",
                review_json_path=project_dir / "需求A_评审记录_开发工程师.json",
                contract=_dummy_contract(),
            )
            created_ba = RequirementsAnalystHandoff(
                worker=_FakeWorker(session_name="需求分析师-天佑星"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )

            def fake_task_done(**kwargs):  # noqa: ANN001
                Path(kwargs["directory"], kwargs["md_output_name"]).write_text("", encoding="utf-8")
                return True

            def fake_generate_task_split_json(handoff, **kwargs):  # noqa: ANN001
                kwargs["paths"]["task_json_path"].write_text(
                    json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return handoff

            with patch("A06_TaskSplit.stdin_is_interactive", return_value=True), patch(
                "A06_TaskSplit.prompt_select_option",
                return_value="review_existing",
            ), patch(
                "A06_TaskSplit.cleanup_stale_task_split_runtime_state",
                return_value=(),
            ), patch(
                "A06_TaskSplit.cleanup_existing_task_split_artifacts",
                return_value=(),
            ), patch(
                "A06_TaskSplit.prepare_task_split_ba_handoff",
                return_value=(created_ba, True),
            ) as prepare_ba, patch(
                "A06_TaskSplit.resolve_reviewer_specs",
                return_value=[TaskSplitReviewerSpec(role_name="开发工程师", role_prompt="实现视角", reviewer_key="开发工程师")],
            ), patch(
                "A06_TaskSplit.build_reviewer_workers",
                return_value=([reviewer], []),
            ), patch(
                "A06_TaskSplit._run_parallel_reviewers",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A06_TaskSplit.repair_reviewer_outputs",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A06_TaskSplit.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A06_TaskSplit.ensure_active_reviewers",
                return_value=None,
            ), patch(
                "A06_TaskSplit.initialize_task_split_workers",
                side_effect=lambda handoff, **kwargs: (handoff, list(kwargs["reviewers"])),
            ) as initialize_workers, patch(
                "A06_TaskSplit.generate_task_split_document",
            ) as generate_task_md, patch(
                "A06_TaskSplit.generate_task_split_json",
                side_effect=fake_generate_task_split_json,
            ) as generate_task_json, patch(
                "A06_TaskSplit._shutdown_workers",
                return_value=(),
            ):
                result = run_task_split_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5"],
                )

            self.assertTrue(result.passed)
            prepare_ba.assert_called_once()
            initialize_workers.assert_called_once()
            generate_task_md.assert_not_called()
            generate_task_json.assert_called_once()

    def test_run_task_split_stage_builds_reviewers_only_after_task_md_generation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_task_split_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
            paths["detailed_design_path"].write_text("详细设计\n", encoding="utf-8")
            ba_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(session_name="需求分析师-天佑星"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            reviewers = [
                ReviewerRuntime(
                    reviewer_name="开发工程师",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="开发工程师-天英星"),
                    review_md_path=project_dir / "需求A_任务单评审记录_开发工程师.md",
                    review_json_path=project_dir / "需求A_评审记录_开发工程师.json",
                    contract=_dummy_contract(),
                )
            ]
            call_order: list[str] = []

            def fake_generate_task_split_document(handoff, **kwargs):  # noqa: ANN001
                call_order.append("generate_task_split_document")
                paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
                return handoff

            def fake_build_reviewers(*args, **kwargs):  # noqa: ANN001
                _ = args
                _ = kwargs
                call_order.append("build_reviewer_workers")
                return reviewers, list(reviewers)

            def fake_initialize_workers(handoff, **kwargs):  # noqa: ANN001
                call_order.append(
                    f"initialize_task_split_workers:{kwargs['initialize_ba']}:{kwargs['initialize_reviewers']}"
                )
                return handoff, list(kwargs["reviewers"])

            def fake_run_parallel_reviewers(reviewer_list, **kwargs):  # noqa: ANN001
                _ = kwargs
                call_order.append("_run_parallel_reviewers")
                return list(reviewer_list)

            def fake_generate_task_split_json(handoff, **kwargs):  # noqa: ANN001
                _ = kwargs
                call_order.append("generate_task_split_json")
                paths["task_json_path"].write_text(
                    json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return handoff

            with patch("A06_TaskSplit.cleanup_stale_task_split_runtime_state", return_value=()), patch(
                "A06_TaskSplit.cleanup_existing_task_split_artifacts",
                return_value=(),
            ), patch(
                "A06_TaskSplit.prepare_task_split_ba_handoff",
                return_value=(ba_handoff, False),
            ), patch(
                "A06_TaskSplit.resolve_reviewer_specs",
                return_value=[TaskSplitReviewerSpec(role_name="开发工程师", role_prompt="实现视角", reviewer_key="开发工程师")],
            ), patch(
                "A06_TaskSplit.build_reviewer_workers",
                side_effect=fake_build_reviewers,
            ), patch(
                "A06_TaskSplit.initialize_task_split_workers",
                side_effect=fake_initialize_workers,
            ), patch(
                "A06_TaskSplit.generate_task_split_document",
                side_effect=fake_generate_task_split_document,
            ), patch(
                "A06_TaskSplit._run_parallel_reviewers",
                side_effect=fake_run_parallel_reviewers,
            ), patch(
                "A06_TaskSplit.repair_reviewer_outputs",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A06_TaskSplit.task_done",
                return_value=True,
            ), patch(
                "A06_TaskSplit.ensure_active_reviewers",
                return_value=None,
            ), patch(
                "A06_TaskSplit.generate_task_split_json",
                side_effect=fake_generate_task_split_json,
            ), patch(
                "A06_TaskSplit._shutdown_workers",
                return_value=(),
            ):
                result = run_task_split_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5"],
                )

        self.assertTrue(result.passed)
        self.assertEqual(
            call_order,
            [
                "generate_task_split_document",
                "build_reviewer_workers",
                "initialize_task_split_workers:False:True",
                "_run_parallel_reviewers",
                "generate_task_split_json",
            ],
        )

    def test_initialize_task_split_workers_runs_reviewer_init_in_parallel(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_task_split_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
            paths["detailed_design_path"].write_text("详细设计\n", encoding="utf-8")
            handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(session_name="需求分析师-天佑星"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            barrier = threading.Barrier(2)
            active_lock = threading.Lock()
            active_state = {"active": 0, "max_active": 0}
            reviewers = [
                ReviewerRuntime(
                    reviewer_name="开发工程师",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="开发工程师-天魁星"),
                    review_md_path=project_dir / "需求A_任务单评审记录_开发工程师.md",
                    review_json_path=project_dir / "需求A_评审记录_开发工程师.json",
                    contract=_dummy_contract(),
                ),
                ReviewerRuntime(
                    reviewer_name="测试工程师",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="测试工程师-天英星"),
                    review_md_path=project_dir / "需求A_任务单评审记录_测试工程师.md",
                    review_json_path=project_dir / "需求A_评审记录_测试工程师.json",
                    contract=_dummy_contract(),
                ),
            ]

            def fake_run_reviewer_result_turn(reviewer, **kwargs):  # noqa: ANN001
                _ = kwargs
                with active_lock:
                    active_state["active"] += 1
                    active_state["max_active"] = max(active_state["max_active"], active_state["active"])
                try:
                    barrier.wait(timeout=1.0)
                finally:
                    with active_lock:
                        active_state["active"] -= 1
                return reviewer

            with patch("A06_TaskSplit._run_reviewer_result_turn", side_effect=fake_run_reviewer_result_turn):
                _, initialized_reviewers = initialize_task_split_workers(
                    handoff,
                    project_dir=project_dir,
                    paths=paths,
                    initialize_ba=False,
                    reviewers=reviewers,
                    reviewer_specs_by_name={
                        "开发工程师": TaskSplitReviewerSpec(role_name="开发工程师", role_prompt="实现视角", reviewer_key="开发工程师"),
                        "测试工程师": TaskSplitReviewerSpec(role_name="测试工程师", role_prompt="测试视角", reviewer_key="测试工程师"),
                    },
                    initialize_reviewers=True,
                )

        self.assertEqual(len(initialized_reviewers), 2)
        self.assertEqual(active_state["max_active"], 2)

    def test_ensure_task_split_inputs_runs_a05_when_detailed_design_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_task_split_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
            ba_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(session_name="需求分析师-天佑星"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            reviewer_handoff = (
                ReviewAgentHandoff(
                    reviewer_key="开发工程师",
                    role_name="开发工程师",
                    role_prompt="实现视角",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="开发工程师-天魁星"),
                ),
            )
            args = build_parser().parse_args(["--project-dir", str(project_dir), "--requirement-name", "需求A"])

            def fake_run_detailed_design_stage(argv, ba_handoff=None, preserve_workers=False):  # noqa: ANN001
                self.assertIn("--project-dir", argv)
                self.assertEqual(ba_handoff.worker.session_name, "需求分析师-天佑星")
                self.assertTrue(preserve_workers)
                paths["detailed_design_path"].write_text("详细设计\n", encoding="utf-8")
                return type(
                    "Result",
                    (),
                    {
                        "ba_handoff": ba_handoff,
                        "reviewer_handoff": reviewer_handoff,
                    },
                )()

            with patch("A06_TaskSplit.run_detailed_design_stage", side_effect=fake_run_detailed_design_stage):
                resolved_paths, resolved_ba_handoff, resolved_reviewer_handoff = ensure_task_split_inputs(
                    args,
                    project_dir=str(project_dir),
                    requirement_name="需求A",
                    ba_handoff=ba_handoff,
                    reviewer_handoff=(),
                )

        self.assertEqual(resolved_paths["detailed_design_path"].name, "需求A_详细设计.md")
        self.assertIs(resolved_ba_handoff, ba_handoff)
        self.assertEqual(tuple(resolved_reviewer_handoff), reviewer_handoff)

    def test_build_reviewer_workers_reuses_live_handoff_without_prompting(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            reviewer_handoff = (
                ReviewAgentHandoff(
                    reviewer_key="开发工程师",
                    role_name="开发工程师",
                    role_prompt="实现视角",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="开发工程师-天魁星", runtime_root=Path(tmp_dir) / "runtime-1", runtime_dir=Path(tmp_dir) / "runtime-1" / "worker-1"),
                ),
                ReviewAgentHandoff(
                    reviewer_key="测试工程师",
                    role_name="测试工程师",
                    role_prompt="测试视角",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="测试工程师-天英星", runtime_root=Path(tmp_dir) / "runtime-2", runtime_dir=Path(tmp_dir) / "runtime-2" / "worker-1"),
                ),
            )
            specs = [
                TaskSplitReviewerSpec(role_name="开发工程师", role_prompt="实现视角", reviewer_key="开发工程师"),
                TaskSplitReviewerSpec(role_name="测试工程师", role_prompt="测试视角", reviewer_key="测试工程师"),
            ]
            args = build_parser().parse_args(["--project-dir", tmp_dir, "--requirement-name", "需求A"])

            with patch("A06_TaskSplit.prompt_review_agent_selection") as selection_prompt:
                reviewers, created_new = build_reviewer_workers(
                    args,
                    project_dir=tmp_dir,
                    requirement_name="需求A",
                    reviewer_specs=specs,
                    reviewer_handoff=reviewer_handoff,
                )

        selection_prompt.assert_not_called()
        self.assertEqual(created_new, [])
        self.assertEqual([item.reviewer_name for item in reviewers], ["开发工程师", "测试工程师"])
        self.assertEqual([item.worker.session_name for item in reviewers], ["开发工程师-天魁星", "测试工程师-天英星"])

    def test_build_reviewer_workers_reuses_live_handoff_and_only_rebuilds_dead_entries(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            reviewer_handoff = (
                ReviewAgentHandoff(
                    reviewer_key="开发工程师",
                    role_name="开发工程师",
                    role_prompt="实现视角",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="开发工程师-天魁星", runtime_root=Path(tmp_dir) / "runtime-1", runtime_dir=Path(tmp_dir) / "runtime-1" / "worker-1"),
                ),
                ReviewAgentHandoff(
                    reviewer_key="测试工程师",
                    role_name="测试工程师",
                    role_prompt="测试视角",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(
                        session_name="测试工程师-天英星",
                        runtime_root=Path(tmp_dir) / "runtime-2",
                        runtime_dir=Path(tmp_dir) / "runtime-2" / "worker-1",
                        session_exists_value=True,
                    ),
                ),
            )
            reviewer_handoff[1].worker.get_agent_state = lambda: type("State", (), {"value": "DEAD"})()
            specs = [
                TaskSplitReviewerSpec(role_name="开发工程师", role_prompt="实现视角", reviewer_key="开发工程师"),
                TaskSplitReviewerSpec(role_name="测试工程师", role_prompt="测试视角", reviewer_key="测试工程师"),
            ]
            args = build_parser().parse_args(["--project-dir", tmp_dir, "--requirement-name", "需求A"])

            with patch("A06_TaskSplit.prompt_review_agent_selection", return_value=ReviewAgentSelection("claude", "sonnet", "medium", "")):
                reviewers, created_new = build_reviewer_workers(
                    args,
                    project_dir=tmp_dir,
                    requirement_name="需求A",
                    reviewer_specs=specs,
                    reviewer_handoff=reviewer_handoff,
                )

        self.assertEqual([item.reviewer_name for item in reviewers], ["开发工程师", "测试工程师"])
        self.assertEqual(reviewers[0].worker.session_name, "开发工程师-天魁星")
        self.assertEqual(len(created_new), 1)
        self.assertEqual(created_new[0].reviewer_name, "测试工程师")

    def test_build_reviewer_workers_uses_precollected_selection_without_prompting(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            specs = [
                TaskSplitReviewerSpec(role_name="开发工程师", role_prompt="实现视角", reviewer_key="开发工程师"),
            ]
            args = build_parser().parse_args(["--project-dir", tmp_dir, "--requirement-name", "需求A"])
            preselected = {"开发工程师": ReviewAgentSelection("claude", "sonnet", "medium", "")}

            with patch("A06_TaskSplit.prompt_review_agent_selection") as selection_prompt:
                reviewers, created_new = build_reviewer_workers(
                    args,
                    project_dir=tmp_dir,
                    requirement_name="需求A",
                    reviewer_specs=specs,
                    reviewer_handoff=(),
                    reviewer_selections_by_name=preselected,
                )

        selection_prompt.assert_not_called()
        self.assertEqual(len(created_new), 1)
        self.assertEqual(reviewers[0].selection.vendor, "claude")
        self.assertEqual(reviewers[0].selection.model, "sonnet")

    def test_run_task_split_stage_collects_reviewer_models_before_generating_task_md(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_task_split_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
            paths["detailed_design_path"].write_text("详细设计\n", encoding="utf-8")
            ba_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(session_name="需求分析师-天佑星"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            specs = [
                TaskSplitReviewerSpec(role_name="开发工程师", role_prompt="实现视角", reviewer_key="开发工程师"),
            ]
            events: list[str] = []

            def fake_collect(**kwargs):  # noqa: ANN001
                events.append("collect")
                self.assertEqual(kwargs["reviewer_specs"], specs)
                return {"开发工程师": ReviewAgentSelection("claude", "sonnet", "medium", "")}

            def fake_generate_task_split_document(handoff, **kwargs):  # noqa: ANN001
                _ = handoff
                events.append("generate")
                kwargs["paths"]["task_md_path"].write_text("重新生成任务单\n", encoding="utf-8")
                return ba_handoff

            def fake_build_reviewers(*args, **kwargs):  # noqa: ANN001
                events.append("build_reviewers")
                self.assertEqual(kwargs["reviewer_selections_by_name"]["开发工程师"].vendor, "claude")
                return [], []

            def fake_generate_task_split_json(handoff, **kwargs):  # noqa: ANN001
                _ = handoff
                kwargs["paths"]["task_json_path"].write_text(
                    json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return ba_handoff

            with patch("A06_TaskSplit.cleanup_stale_task_split_runtime_state", return_value=()), patch(
                "A06_TaskSplit.cleanup_existing_task_split_artifacts",
                return_value=(),
            ), patch(
                "A06_TaskSplit.prepare_task_split_ba_handoff",
                return_value=(ba_handoff, False),
            ), patch(
                "A06_TaskSplit.resolve_reviewer_specs",
                return_value=specs,
            ), patch(
                "A06_TaskSplit.collect_reviewer_agent_selections",
                side_effect=fake_collect,
            ), patch(
                "A06_TaskSplit.generate_task_split_document",
                side_effect=fake_generate_task_split_document,
            ), patch(
                "A06_TaskSplit.build_reviewer_workers",
                side_effect=fake_build_reviewers,
            ), patch(
                "A06_TaskSplit.ensure_active_reviewers",
                return_value=None,
            ), patch(
                "A06_TaskSplit.task_done",
                return_value=True,
            ), patch(
                "A06_TaskSplit.generate_task_split_json",
                side_effect=fake_generate_task_split_json,
            ), patch(
                "A06_TaskSplit._shutdown_workers",
                return_value=(),
            ):
                result = run_task_split_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5"],
                )

        self.assertTrue(result.passed)
        self.assertEqual(events[:2], ["collect", "generate"])

    def test_run_task_split_stage_skips_live_handoffs_when_collecting_models(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_task_split_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
            paths["detailed_design_path"].write_text("详细设计\n", encoding="utf-8")
            ba_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(session_name="需求分析师-天佑星"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            reviewer_handoff = (
                ReviewAgentHandoff(
                    reviewer_key="开发工程师",
                    role_name="开发工程师",
                    role_prompt="实现视角",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="开发工程师-天魁星"),
                ),
            )
            observed_skip_keys: list[str] = []

            def fake_collect(**kwargs):  # noqa: ANN001
                observed_skip_keys[:] = list(kwargs["skip_reviewer_keys"])
                return {}

            def fake_generate_task_split_document(handoff, **kwargs):  # noqa: ANN001
                _ = handoff
                kwargs["paths"]["task_md_path"].write_text("重新生成任务单\n", encoding="utf-8")
                return ba_handoff

            def fake_generate_task_split_json(handoff, **kwargs):  # noqa: ANN001
                _ = handoff
                kwargs["paths"]["task_json_path"].write_text(
                    json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return ba_handoff

            with patch("A06_TaskSplit.cleanup_stale_task_split_runtime_state", return_value=()), patch(
                "A06_TaskSplit.cleanup_existing_task_split_artifacts",
                return_value=(),
            ), patch(
                "A06_TaskSplit.prepare_task_split_ba_handoff",
                return_value=(ba_handoff, False),
            ), patch(
                "A06_TaskSplit.collect_reviewer_agent_selections",
                side_effect=fake_collect,
            ), patch(
                "A06_TaskSplit.generate_task_split_document",
                side_effect=fake_generate_task_split_document,
            ), patch(
                "A06_TaskSplit.build_reviewer_workers",
                return_value=([], []),
            ), patch(
                "A06_TaskSplit.ensure_active_reviewers",
                return_value=None,
            ), patch(
                "A06_TaskSplit.task_done",
                return_value=True,
            ), patch(
                "A06_TaskSplit.generate_task_split_json",
                side_effect=fake_generate_task_split_json,
            ), patch(
                "A06_TaskSplit._shutdown_workers",
                return_value=(),
            ):
                result = run_task_split_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5"],
                    reviewer_handoff=reviewer_handoff,
                )

        self.assertTrue(result.passed)
        self.assertEqual(observed_skip_keys, ["开发工程师"])

    def test_run_ba_modify_loop_returns_immediately_when_no_hitl_needed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = build_task_split_paths(tmp_dir, "需求A")
            paths["task_md_path"].write_text("任务单\n", encoding="utf-8")
            handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(session_name="需求分析师-天佑星"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )

            def fake_run_ba_turn_with_recovery(*args, **kwargs):  # noqa: ANN001
                paths["ask_human_path"].write_text("", encoding="utf-8")
                paths["ba_feedback_path"].write_text("已补充任务拆分边界\n", encoding="utf-8")
                return handoff, {"status": "completed"}

            with patch("A06_TaskSplit.run_ba_turn_with_recovery", side_effect=fake_run_ba_turn_with_recovery) as turn_mock:
                returned = run_ba_modify_loop(
                    handoff,
                    project_dir=tmp_dir,
                    paths=paths,
                    review_msg="请补充拆分边界",
                )

        self.assertIs(returned, handoff)
        turn_mock.assert_called_once()

    def test_run_ba_modify_loop_routes_hitl_requests_into_hitl_reply_loop(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = build_task_split_paths(tmp_dir, "需求A")
            paths["task_md_path"].write_text("任务单\n", encoding="utf-8")
            handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(session_name="需求分析师-天佑星"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )

            call_count = {"value": 0}

            def fake_run_ba_turn_with_recovery(*args, **kwargs):  # noqa: ANN001
                call_count["value"] += 1
                if call_count["value"] == 1:
                    paths["ask_human_path"].write_text("请补充拆分边界\n", encoding="utf-8")
                    return handoff, {"status": "hitl"}
                paths["ask_human_path"].write_text("", encoding="utf-8")
                paths["ba_feedback_path"].write_text("已按人类建议补充拆分边界\n", encoding="utf-8")
                return handoff, {"status": "completed"}

            with patch("A06_TaskSplit.run_ba_turn_with_recovery", side_effect=fake_run_ba_turn_with_recovery) as turn_mock, patch(
                "A06_TaskSplit.collect_review_limit_hitl_response",
                return_value="请按方案 A 继续",
            ) as hitl_input_mock:
                returned = run_ba_modify_loop(
                    handoff,
                    project_dir=tmp_dir,
                    paths=paths,
                    review_msg="请补充拆分边界",
                )

        self.assertIs(returned, handoff)
        self.assertEqual(turn_mock.call_count, 2)
        hitl_input_mock.assert_called_once()

    def test_run_task_split_stage_round_two_reuses_reviewers_and_generates_task_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_task_split_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
            paths["detailed_design_path"].write_text("详细设计\n", encoding="utf-8")
            ba_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(session_name="需求分析师-天佑星", runtime_root=project_dir / "ba-runtime", runtime_dir=project_dir / "ba-runtime" / "worker"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            reviewers = [
                ReviewerRuntime(
                    reviewer_name="开发工程师",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="开发工程师-天魁星", runtime_root=project_dir / "r1", runtime_dir=project_dir / "r1" / "worker"),
                    review_md_path=project_dir / "需求A_任务单评审记录_开发工程师.md",
                    review_json_path=project_dir / "需求A_评审记录_开发工程师.json",
                    contract=_dummy_contract(),
                ),
                ReviewerRuntime(
                    reviewer_name="测试工程师",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="测试工程师-天英星", runtime_root=project_dir / "r2", runtime_dir=project_dir / "r2" / "worker"),
                    review_md_path=project_dir / "需求A_任务单评审记录_测试工程师.md",
                    review_json_path=project_dir / "需求A_评审记录_测试工程师.json",
                    contract=_dummy_contract(),
                ),
            ]
            reviewer_handoff = (
                ReviewAgentHandoff(
                    reviewer_key="开发工程师",
                    role_name="开发工程师",
                    role_prompt="实现视角",
                    selection=reviewers[0].selection,
                    worker=reviewers[0].worker,
                ),
                ReviewAgentHandoff(
                    reviewer_key="测试工程师",
                    role_name="测试工程师",
                    role_prompt="测试视角",
                    selection=reviewers[1].selection,
                    worker=reviewers[1].worker,
                ),
            )

            def fake_generate_task_split_document(handoff, **kwargs):  # noqa: ANN001
                paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
                return handoff

            def fake_run_parallel_reviewers(reviewers_arg, **kwargs):  # noqa: ANN001
                if kwargs["round_index"] == 1:
                    paths["merged_review_path"].write_text("请补充任务拆分边界\n", encoding="utf-8")
                return list(reviewers_arg)

            def fake_generate_task_split_json(handoff, **kwargs):  # noqa: ANN001
                paths["task_json_path"].write_text(
                    json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False),
                    encoding="utf-8",
                )
                return handoff

            with patch("A06_TaskSplit.cleanup_stale_task_split_runtime_state", return_value=()), patch(
                "A06_TaskSplit.prepare_task_split_ba_handoff",
                return_value=(ba_handoff, False),
            ), patch(
                "A06_TaskSplit.resolve_reviewer_specs",
                return_value=[
                    TaskSplitReviewerSpec(role_name="开发工程师", role_prompt="实现视角", reviewer_key="开发工程师"),
                    TaskSplitReviewerSpec(role_name="测试工程师", role_prompt="测试视角", reviewer_key="测试工程师"),
                ],
            ), patch(
                "A06_TaskSplit.build_reviewer_workers",
                return_value=(reviewers, False),
            ), patch(
                "A06_TaskSplit.initialize_task_split_workers",
                return_value=(ba_handoff, reviewers),
            ), patch(
                "A06_TaskSplit.generate_task_split_document",
                side_effect=fake_generate_task_split_document,
            ), patch(
                "A06_TaskSplit._run_parallel_reviewers",
                side_effect=fake_run_parallel_reviewers,
            ) as parallel_reviewers, patch(
                "A06_TaskSplit.repair_reviewer_outputs",
                side_effect=lambda reviewer_list, **kwargs: reviewer_list,
            ), patch(
                "A06_TaskSplit.task_done",
                side_effect=[False, True],
            ), patch(
                "A06_TaskSplit.run_ba_modify_loop",
                return_value=ba_handoff,
            ) as modify_loop, patch(
                "A06_TaskSplit._read_required_task_split_ba_feedback",
                return_value="已修订任务单",
            ), patch(
                "A06_TaskSplit.generate_task_split_json",
                side_effect=fake_generate_task_split_json,
            ) as generate_json, patch(
                "A06_TaskSplit._shutdown_workers",
                return_value=(),
            ):
                result = run_task_split_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5"],
                    ba_handoff=ba_handoff,
                    reviewer_handoff=reviewer_handoff,
                )

        self.assertIsInstance(result, TaskSplitStageResult)
        self.assertTrue(result.passed)
        self.assertEqual(parallel_reviewers.call_count, 2)
        modify_loop.assert_called_once()
        generate_json.assert_called_once()

    def test_generate_task_split_json_retries_with_repair_prompt_and_clears_stale_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = build_task_split_paths(tmp_dir, "需求A")
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(session_name="需求分析师-天佑星"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            prompts: list[str] = []
            stale_json_states: list[str] = []

            def fake_run_ba_turn_with_recovery(current_handoff, **kwargs):  # noqa: ANN001
                prompts.append(kwargs["prompt"])
                stale_json_states.append(paths["task_json_path"].read_text(encoding="utf-8"))
                if len(prompts) == 1:
                    paths["task_json_path"].write_text(
                        json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False),
                        encoding="utf-8",
                    )
                else:
                    paths["task_json_path"].write_text(
                        json.dumps({"M1": {"M1-T1": False}}, ensure_ascii=False),
                        encoding="utf-8",
                    )
                return current_handoff, {"status": "completed"}

            with patch("A06_TaskSplit.run_ba_turn_with_recovery", side_effect=fake_run_ba_turn_with_recovery):
                returned_handoff = generate_task_split_json(handoff, project_dir=tmp_dir, paths=paths)

        self.assertIs(returned_handoff, handoff)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(stale_json_states, ["", ""])
        self.assertIn("结构化数据转换专家", prompts[0])
        self.assertIn("重新解析", prompts[1])

    def test_run_reviewer_turn_with_resume_reuses_materialized_outputs_after_runtime_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(session_name="测试工程师-天英星"),
                review_md_path=project_dir / "需求A_任务单评审记录_测试工程师.md",
                review_json_path=project_dir / "需求A_评审记录_测试工程师.json",
                contract=_dummy_contract(),
            )
            reviewer.review_md_path.write_text("", encoding="utf-8")
            reviewer.review_json_path.write_text("[]", encoding="utf-8")

            def fake_run_turn(**kwargs):  # noqa: ANN001
                reviewer.review_md_path.write_text("- [Error] 缺少异常任务\n", encoding="utf-8")
                reviewer.review_json_path.write_text(
                    json.dumps([{"task_name": "任务拆分", "review_pass": False}], ensure_ascii=False),
                    encoding="utf-8",
                )
                return type("Result", (), {"ok": False, "clean_output": "runtime failed"})()

            reviewer.worker.run_turn = fake_run_turn

            returned = _run_reviewer_turn_with_resume(
                reviewer,
                label="task_split_review_init_测试工程师_round_1",
                prompt="review",
            )

            self.assertIs(returned, reviewer)

    def test_run_reviewer_result_turn_recovers_reviewer_init_after_ready_timeout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            reviewer = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_ReconfigurableWorker(session_name="测试工程师-天英星"),
                review_md_path=project_dir / "需求A_任务单评审记录_测试工程师.md",
                review_json_path=project_dir / "需求A_评审记录_测试工程师.json",
                contract=_dummy_contract(),
            )
            replacement = ReviewerRuntime(
                reviewer_name="测试工程师",
                selection=ReviewAgentSelection("codex", "gpt-5.4-mini", "high", ""),
                worker=_FakeWorker(session_name="测试工程师-天魁星"),
                review_md_path=reviewer.review_md_path,
                review_json_path=reviewer.review_json_path,
                contract=_dummy_contract(),
            )
            attempts = {"count": 0}

            def fake_run_task_result_turn_with_repair(**kwargs):  # noqa: ANN001
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise RuntimeError("Timed out waiting for agent ready.\nmock screen")
                return {}

            with patch(
                "A06_TaskSplit.run_task_result_turn_with_repair",
                side_effect=fake_run_task_result_turn_with_repair,
            ), patch(
                "A06_TaskSplit.request_worker_manual_intervention",
                return_value=AGENT_INTERVENTION_RECREATE,
            ), patch(
                "A06_TaskSplit.recreate_task_split_reviewer_runtime",
                return_value=replacement,
            ) as recreate_runtime:
                result = _run_reviewer_result_turn(
                    reviewer,
                    label="task_split_reviewer_init_测试工程师",
                    prompt="init",
                    result_contract=SimpleNamespace(mode="a06_reviewer_init"),
                    project_dir=project_dir,
                    requirement_name="需求A",
                    reviewer_spec=TaskSplitReviewerSpec(
                        role_name="测试工程师",
                        role_prompt="测试视角",
                        reviewer_key="测试工程师",
                    ),
                )

        self.assertIs(result, replacement)
        self.assertEqual(attempts["count"], 2)
        self.assertIn("启动超时", reviewer.worker.reconfig_reason)
        recreate_runtime.assert_called_once()

    def test_build_reviewer_init_result_contract_uses_reviewer_mode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = build_task_split_paths(tmp_dir, "需求A")

            contract = build_reviewer_init_result_contract(paths)

            self.assertEqual(contract.turn_id, "a06_reviewer_init")
            self.assertEqual(contract.mode, "a06_reviewer_init")

    def test_generate_task_split_json_raises_after_repair_attempts_exhausted(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = build_task_split_paths(tmp_dir, "需求A")
            paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
            handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(session_name="需求分析师-天佑星"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            prompts: list[str] = []
            stale_json_states: list[str] = []

            def fake_run_ba_turn_with_recovery(current_handoff, **kwargs):  # noqa: ANN001
                prompts.append(kwargs["prompt"])
                stale_json_states.append(paths["task_json_path"].read_text(encoding="utf-8"))
                paths["task_json_path"].write_text(
                    json.dumps({"M1": {"M1-T1": True}}, ensure_ascii=False),
                    encoding="utf-8",
                )
                return current_handoff, {"status": "completed"}

            with patch("A06_TaskSplit.run_ba_turn_with_recovery", side_effect=fake_run_ba_turn_with_recovery):
                with self.assertRaisesRegex(RuntimeError, "任务单 JSON 生成失败"):
                    generate_task_split_json(handoff, project_dir=tmp_dir, paths=paths)

        self.assertEqual(len(prompts), 2)
        self.assertEqual(stale_json_states, ["", ""])

    def test_run_task_split_stage_keeps_pre_development_status_false_when_task_json_generation_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            paths = build_task_split_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
            paths["detailed_design_path"].write_text("详细设计\n", encoding="utf-8")
            ba_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(
                    session_name="需求分析师-天佑星",
                    runtime_root=project_dir / "ba-runtime",
                    runtime_dir=project_dir / "ba-runtime" / "worker",
                ),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            reviewers = [
                ReviewerRuntime(
                    reviewer_name="开发工程师",
                    selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                    worker=_FakeWorker(session_name="开发工程师-天魁星"),
                    review_md_path=project_dir / "需求A_任务单评审记录_开发工程师.md",
                    review_json_path=project_dir / "需求A_评审记录_开发工程师.json",
                    contract=_dummy_contract(),
                )
            ]

            def fake_generate_task_split_document(handoff, **kwargs):  # noqa: ANN001
                paths["task_md_path"].write_text("任务单正文\n", encoding="utf-8")
                return handoff

            with patch("A06_TaskSplit.cleanup_stale_task_split_runtime_state", return_value=()), patch(
                "A06_TaskSplit.prepare_task_split_ba_handoff",
                return_value=(ba_handoff, False),
            ), patch(
                "A06_TaskSplit.resolve_reviewer_specs",
                return_value=[TaskSplitReviewerSpec(role_name="开发工程师", role_prompt="实现视角", reviewer_key="开发工程师")],
            ), patch(
                "A06_TaskSplit.build_reviewer_workers",
                return_value=(reviewers, False),
            ), patch(
                "A06_TaskSplit.initialize_task_split_workers",
                return_value=(ba_handoff, reviewers),
            ), patch(
                "A06_TaskSplit.generate_task_split_document",
                side_effect=fake_generate_task_split_document,
            ), patch(
                "A06_TaskSplit._run_parallel_reviewers",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A06_TaskSplit.repair_reviewer_outputs",
                side_effect=lambda reviewer_list, **kwargs: reviewer_list,
            ), patch(
                "A06_TaskSplit.task_done",
                return_value=True,
            ), patch(
                "A06_TaskSplit.generate_task_split_json",
                side_effect=RuntimeError("任务单 JSON 生成失败"),
            ), patch(
                "A06_TaskSplit._shutdown_workers",
                return_value=(),
            ):
                with self.assertRaisesRegex(RuntimeError, "任务单 JSON 生成失败"):
                    run_task_split_stage(
                        ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5"],
                        ba_handoff=ba_handoff,
                        reviewer_handoff=(),
                    )

            pre_development_path = build_pre_development_task_record_path(project_dir, "需求A")
            pre_development_payload = json.loads(pre_development_path.read_text(encoding="utf-8"))

        self.assertFalse(pre_development_payload["任务拆分"]["任务拆分"])

    def test_cleanup_stale_task_split_runtime_state_scopes_by_requirement_and_keeps_live_legacy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_root = root / ".task_split_runtime"
            target_dir = runtime_root / "target-reviewer"
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "任务拆分-当前需求",
                        "project_dir": str(root.resolve()),
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a06.start",
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
                        "session_name": "任务拆分-其他需求",
                        "project_dir": str(root.resolve()),
                        "requirement_name": "需求B",
                        "workflow_action": "stage.a06.start",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            legacy_live_dir = runtime_root / "legacy-live"
            legacy_live_dir.mkdir(parents=True, exist_ok=True)
            (legacy_live_dir / "worker.state.json").write_text(
                json.dumps({"session_name": "任务拆分-遗留存活"}, ensure_ascii=False),
                encoding="utf-8",
            )
            legacy_dead_dir = runtime_root / "legacy-dead"
            legacy_dead_dir.mkdir(parents=True, exist_ok=True)
            (legacy_dead_dir / "worker.state.json").write_text(
                json.dumps({"session_name": "任务拆分-遗留死亡", "agent_state": "DEAD"}, ensure_ascii=False),
                encoding="utf-8",
            )
            killed_sessions: list[str] = []

            class FakeTmuxRuntimeController:
                def session_exists(self, session_name: str) -> bool:
                    return session_name in {"任务拆分-遗留存活", "任务拆分-其他需求"}

                def kill_session(self, session_name: str, *, missing_ok: bool = True):  # noqa: ANN001
                    killed_sessions.append(session_name)
                    return session_name

            with patch("tmux_core.stage_kernel.runtime_scope_cleanup.TmuxRuntimeController", FakeTmuxRuntimeController):
                removed = cleanup_stale_task_split_runtime_state(root, "需求A")

            self.assertFalse(target_dir.exists())
            self.assertFalse(legacy_dead_dir.exists())
            self.assertTrue(other_requirement_dir.exists())
            self.assertTrue(legacy_live_dir.exists())
            self.assertIn("任务拆分-当前需求", killed_sessions)
            self.assertIn("任务拆分-遗留死亡", killed_sessions)
            self.assertIn(str(target_dir.resolve()), removed)
            self.assertIn(str(legacy_dead_dir.resolve()), removed)
