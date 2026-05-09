from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, patch

from tmux_core.runtime.contracts import resolve_task_result_decision

from A05_DetailedDesign import (
    DetailedDesignReviewerSpec,
    ReviewAgentSelection,
    RequirementsAnalystHandoff,
    _shutdown_workers,
    _run_parallel_reviewers,
    build_detailed_design_init_prompt,
    build_detailed_design_prompt,
    build_detailed_design_feedback_contract,
    build_parser,
    cleanup_stale_detailed_design_runtime_state,
    build_reviewer_workers,
    build_detailed_design_paths,
    collect_interactive_reviewer_specs,
    create_reviewer_runtime,
    generate_detailed_design_document,
    prepare_design_ba_handoff,
    resolve_review_max_rounds,
    resolve_reviewer_specs,
    run_ba_modify_loop,
    run_detailed_design_stage,
    run_reviewer_turn_with_recreation,
)
from T08_pre_development import build_pre_development_task_record_path


class _FakeWorker:
    def __init__(
        self,
        *,
        session_name: str = "需求分析师-天佑星",
        runtime_root: str | Path = "/tmp/runtime",
        runtime_dir: str | Path = "/tmp/runtime/worker",
        session_exists_value: bool = True,
    ) -> None:
        self.session_name = session_name
        self.runtime_root = Path(runtime_root)
        self.runtime_dir = Path(runtime_dir)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.killed = False
        self._session_exists_value = session_exists_value

    def request_kill(self):
        self.killed = True
        return self.session_name

    def session_exists(self) -> bool:
        return self._session_exists_value


class _FreshReviewerWorker:
    def __init__(self, *, session_name: str) -> None:
        self.session_name = session_name
        self.ensure_calls = 0
        self._launched = False

    def get_agent_state(self):
        state = "READY" if self._launched else "DEAD"
        return SimpleNamespace(value=state)

    def has_ever_launched(self) -> bool:
        return self._launched

    def ensure_agent_ready(self, timeout_sec: float = 0.0) -> None:
        _ = timeout_sec
        self.ensure_calls += 1
        self._launched = True


class A05DetailedDesignTests(unittest.TestCase):
    def test_detailed_design_reviewer_count_prompt_allows_previous_step_back(self):
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
        self.assertEqual(captured_requests[0].payload["prompt_text"], "请输入详细设计审核智能体数量")
        self.assertTrue(captured_requests[0].payload["allow_back"])
        self.assertEqual(captured_requests[0].payload["back_value"], PROMPT_BACK_VALUE)
        self.assertEqual(captured_requests[0].payload["stage_key"], "detailed_design_reviewer_specs")
        self.assertEqual(captured_requests[0].payload["stage_step_index"], 0)

    def test_detailed_design_reviewer_role_prompts_allow_back_after_count(self):
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

    def test_review_limit_force_hitl_contract_uses_outcome_scoped_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_detailed_design_paths(tmpdir, "需求A")
            paths["detailed_design_path"].write_text("旧详细设计\n", encoding="utf-8")
            paths["ask_human_path"].write_text("请决策\n", encoding="utf-8")

            contract = build_detailed_design_feedback_contract(
                paths,
                mode="a05_detailed_design_review_limit_force_hitl",
            )
            decision = resolve_task_result_decision(contract)

        self.assertEqual(decision.status, "hitl")
        self.assertIn("ask_human", decision.artifacts)
        self.assertNotIn("detailed_design", decision.artifacts)

    def test_feedback_contract_completed_branch_requires_design_and_feedback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_detailed_design_paths(tmpdir, "需求A")
            paths["detailed_design_path"].write_text("新详细设计\n", encoding="utf-8")
            paths["ba_feedback_path"].write_text("已处理评审\n", encoding="utf-8")

            contract = build_detailed_design_feedback_contract(paths)
            decision = resolve_task_result_decision(contract)

        self.assertEqual(decision.status, "completed")
        self.assertIn("detailed_design", decision.artifacts)
        self.assertIn("ba_feedback", decision.artifacts)

    def test_resolve_review_max_rounds_supports_default_and_infinite(self):
        args = build_parser().parse_args([])
        self.assertEqual(resolve_review_max_rounds(args), 5)

        args = build_parser().parse_args(["--review-max-rounds", "infinite"])
        self.assertIsNone(resolve_review_max_rounds(args))

    def test_resolve_review_max_rounds_rejects_invalid_cli_value(self):
        args = build_parser().parse_args(["--review-max-rounds", "-1"])

        with self.assertRaisesRegex(RuntimeError, "必须是正整数或 infinite"):
            resolve_review_max_rounds(args)

    def test_resolve_review_max_rounds_prompts_in_interactive_mode(self):
        args = build_parser().parse_args([])

        with patch("A05_DetailedDesign.stdin_is_interactive", return_value=True), patch(
            "A05_DetailedDesign.prompt_review_max_rounds",
            return_value=6,
        ) as prompt_mock:
            value = resolve_review_max_rounds(args, progress=object())

        self.assertEqual(value, 6)
        prompt_mock.assert_called_once()

    def test_predict_worker_display_name_includes_tmux_sessions_in_occupied_pool(self):
        import A05_DetailedDesign as design_module

        observed: dict[str, set[str]] = {}

        def fake_build_session_name(worker_id, work_dir, vendor, instance_id="", occupied_session_names=None):  # noqa: ANN001
            _ = worker_id
            _ = work_dir
            _ = vendor
            _ = instance_id
            observed["occupied"] = {str(item).strip() for item in occupied_session_names or () if str(item).strip()}
            return "需求分析师-天佑星"

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "A05_DetailedDesign.build_session_name",
            side_effect=fake_build_session_name,
        ), patch(
            "A05_DetailedDesign.list_tmux_session_names",
            return_value=["开发工程师-天孤星"],
        ), patch(
            "A05_DetailedDesign.list_registered_tmux_workers",
            return_value=[],
        ):
            session_name = design_module._predict_worker_display_name(
                project_dir=tmpdir,
                worker_id="detailed-design-analyst",
                occupied_session_names=("预占用-会话",),
            )

        self.assertEqual(session_name, "需求分析师-天佑星")
        self.assertIn("开发工程师-天孤星", observed["occupied"])
        self.assertIn("预占用-会话", observed["occupied"])

    def test_run_stage_skips_when_existing_detailed_design_is_confirmed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_detailed_design_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
            paths["detailed_design_path"].write_text("已有详细设计\n", encoding="utf-8")

            with patch("A05_DetailedDesign.stdin_is_interactive", return_value=True), patch(
                "A05_DetailedDesign.prompt_select_option",
                return_value="skip",
            ) as prompt_mock, patch(
                "A05_DetailedDesign.resolve_review_max_rounds",
            ) as resolve_rounds_mock, patch(
                "A05_DetailedDesign.cleanup_stale_detailed_design_runtime_state",
            ) as cleanup_runtime, patch(
                "A05_DetailedDesign.cleanup_existing_detailed_design_artifacts",
            ) as cleanup_artifacts, patch(
                "A05_DetailedDesign.prepare_design_ba_handoff",
            ) as prepare_ba, patch(
                "A05_DetailedDesign.generate_detailed_design_document",
            ) as generate_design, patch(
                "A05_DetailedDesign.build_reviewer_workers",
            ) as build_reviewers:
                result = run_detailed_design_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5"],
                )

            self.assertTrue(result.passed)
            self.assertIsNone(result.ba_handoff)
            self.assertEqual(result.reviewer_handoff, ())
            prompt_text = prompt_mock.call_args.kwargs["title"]
            self.assertIn("需求A_详细设计.md", prompt_text)
            cleanup_runtime.assert_not_called()
            cleanup_artifacts.assert_not_called()
            prepare_ba.assert_not_called()
            generate_design.assert_not_called()
            build_reviewers.assert_not_called()
            resolve_rounds_mock.assert_not_called()
            pre_development_path = build_pre_development_task_record_path(project_dir, "需求A")
            payload = json.loads(pre_development_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["详细设计"]["详细设计"])

    def test_run_stage_reruns_when_existing_detailed_design_is_selected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_detailed_design_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
            paths["detailed_design_path"].write_text("已有详细设计\n", encoding="utf-8")
            fake_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(runtime_root=project_dir / ".detailed_design_runtime", runtime_dir=project_dir / ".detailed_design_runtime" / "ba"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )

            def fake_generate(handoff, *, project_dir, paths, initialize_first, progress=None):  # noqa: ANN001
                _ = handoff
                _ = project_dir
                _ = initialize_first
                _ = progress
                paths["detailed_design_path"].write_text("重新生成的详细设计\n", encoding="utf-8")
                return fake_handoff

            def fake_task_done(**kwargs):  # noqa: ANN001
                Path(kwargs["directory"], kwargs["md_output_name"]).write_text("", encoding="utf-8")
                return True

            with patch("A05_DetailedDesign.stdin_is_interactive", return_value=True), patch(
                "A05_DetailedDesign.prompt_select_option",
                return_value="rerun",
            ) as prompt_mock, patch(
                "A05_DetailedDesign.cleanup_stale_detailed_design_runtime_state",
                return_value=(),
            ) as cleanup_runtime, patch(
                "A05_DetailedDesign.cleanup_existing_detailed_design_artifacts",
                return_value=(),
            ) as cleanup_artifacts, patch(
                "A05_DetailedDesign.prepare_design_ba_handoff",
                return_value=(fake_handoff, False),
            ), patch(
                "A05_DetailedDesign.generate_detailed_design_document",
                side_effect=fake_generate,
            ) as generate_design, patch(
                "A05_DetailedDesign.resolve_reviewer_specs",
                return_value=[],
            ), patch(
                "A05_DetailedDesign.build_reviewer_workers",
                return_value=[],
            ), patch(
                "A05_DetailedDesign.ensure_active_reviewers",
                return_value=None,
            ), patch(
                "A05_DetailedDesign.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A05_DetailedDesign.mark_detailed_design_completed",
                return_value=paths["pre_development_path"],
            ), patch(
                "A05_DetailedDesign._shutdown_workers",
                return_value=(),
            ):
                result = run_detailed_design_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5"],
                )

            self.assertTrue(result.passed)
            prompt_mock.assert_called_once()
            cleanup_runtime.assert_called_once()
            cleanup_artifacts.assert_called_once_with(paths, "需求A", clear_detailed_design=True, audit_context=ANY)
            generate_design.assert_called_once()
            self.assertEqual(paths["detailed_design_path"].read_text(encoding="utf-8"), "重新生成的详细设计\n")

    def test_run_stage_reviews_existing_detailed_design_without_regenerating(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_detailed_design_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
            paths["detailed_design_path"].write_text("已有详细设计\n", encoding="utf-8")
            fake_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(runtime_root=project_dir / ".detailed_design_runtime", runtime_dir=project_dir / ".detailed_design_runtime" / "ba"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )

            def fake_task_done(**kwargs):  # noqa: ANN001
                Path(kwargs["directory"], kwargs["md_output_name"]).write_text("", encoding="utf-8")
                return True

            with patch("A05_DetailedDesign.stdin_is_interactive", return_value=True), patch(
                "A05_DetailedDesign.prompt_select_option",
                return_value="review_existing",
            ), patch(
                "A05_DetailedDesign.cleanup_stale_detailed_design_runtime_state",
                return_value=(),
            ), patch(
                "A05_DetailedDesign.cleanup_existing_detailed_design_artifacts",
                return_value=(),
            ) as cleanup_artifacts, patch(
                "A05_DetailedDesign.prepare_design_ba_handoff",
                return_value=(fake_handoff, False),
            ) as prepare_ba, patch(
                "A05_DetailedDesign.initialize_detailed_design_ba",
                return_value=fake_handoff,
            ), patch(
                "A05_DetailedDesign.generate_detailed_design_document",
            ) as generate_design, patch(
                "A05_DetailedDesign.resolve_reviewer_specs",
                return_value=[],
            ), patch(
                "A05_DetailedDesign.build_reviewer_workers",
                return_value=[],
            ), patch(
                "A05_DetailedDesign.ensure_active_reviewers",
                return_value=None,
            ), patch(
                "A05_DetailedDesign.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A05_DetailedDesign.mark_detailed_design_completed",
                return_value=paths["pre_development_path"],
            ), patch(
                "A05_DetailedDesign._shutdown_workers",
                return_value=(),
            ):
                result = run_detailed_design_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5"],
                )

            self.assertTrue(result.passed)
            cleanup_artifacts.assert_called_once_with(paths, "需求A", clear_detailed_design=False, audit_context=ANY)
            prepare_ba.assert_not_called()
            generate_design.assert_not_called()
            self.assertEqual(paths["detailed_design_path"].read_text(encoding="utf-8"), "已有详细设计\n")

    def test_run_stage_review_existing_keeps_fresh_reviewers_until_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_detailed_design_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
            paths["detailed_design_path"].write_text("已有详细设计\n", encoding="utf-8")
            fresh_worker = _FreshReviewerWorker(session_name="审核员-地异星")
            reviewers = [
                SimpleNamespace(
                    reviewer_name="审核员",
                    worker=fresh_worker,
                    review_md_path=project_dir / "需求A_详设评审记录_审核员.md",
                    review_json_path=project_dir / "需求A_评审记录_审核员.json",
                )
            ]
            parallel_calls: list[int] = []
            observed_prompts: list[str] = []

            def fake_task_done(**kwargs):  # noqa: ANN001
                Path(kwargs["directory"], kwargs["md_output_name"]).write_text("", encoding="utf-8")
                return True

            def fake_parallel_reviewers(reviewer_list, *, round_index, prompt_builder, **kwargs):  # noqa: ANN001
                parallel_calls.append(round_index)
                observed_prompts.append(prompt_builder(reviewer_list[0], DetailedDesignReviewerSpec(role_name="审核员", role_prompt="一致性视角", reviewer_key="审核员")))
                return list(reviewer_list)

            with patch("A05_DetailedDesign.stdin_is_interactive", return_value=True), patch(
                "A05_DetailedDesign.prompt_select_option",
                return_value="review_existing",
            ), patch(
                "A05_DetailedDesign.cleanup_stale_detailed_design_runtime_state",
                return_value=(),
            ), patch(
                "A05_DetailedDesign.cleanup_existing_detailed_design_artifacts",
                return_value=(),
            ), patch(
                "A05_DetailedDesign.resolve_reviewer_specs",
                return_value=[DetailedDesignReviewerSpec(role_name="审核员", role_prompt="一致性视角", reviewer_key="审核员")],
            ), patch(
                "A05_DetailedDesign.build_reviewer_workers",
                return_value=reviewers,
            ), patch(
                "A05_DetailedDesign._run_parallel_reviewers",
                side_effect=fake_parallel_reviewers,
            ), patch(
                "A05_DetailedDesign.repair_reviewer_outputs",
                side_effect=lambda reviewer_list, **kwargs: list(reviewer_list),
            ), patch(
                "A05_DetailedDesign.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A05_DetailedDesign.mark_detailed_design_completed",
                return_value=paths["pre_development_path"],
            ), patch(
                "A05_DetailedDesign._shutdown_workers",
                return_value=(),
            ):
                result = run_detailed_design_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5"],
                )

        self.assertTrue(result.passed)
        self.assertEqual(parallel_calls, [1])
        self.assertEqual(len(observed_prompts), 1)
        self.assertNotIn("Task Routing Assessment", observed_prompts[0])
        self.assertNotIn("Output the Task Routing Assessment first", observed_prompts[0])
        self.assertEqual(fresh_worker.ensure_calls, 1)

    def test_run_stage_reviews_existing_detailed_design_then_modifies_if_review_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_detailed_design_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
            paths["detailed_design_path"].write_text("已有详细设计\n", encoding="utf-8")
            reviewer_specs = [
                DetailedDesignReviewerSpec(role_name="开发工程师", role_prompt="实现视角", reviewer_key="开发工程师"),
            ]
            reviewers = [
                SimpleNamespace(
                    reviewer_name="开发工程师",
                    review_md_path=project_dir / "开发工程师.md",
                    review_json_path=project_dir / "开发工程师.json",
                )
            ]
            fake_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(runtime_root=project_dir / ".detailed_design_runtime", runtime_dir=project_dir / ".detailed_design_runtime" / "ba"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            call_sequence: list[str] = []

            def fake_parallel(reviewers_arg, *, round_index, **kwargs):  # noqa: ANN001
                _ = kwargs
                call_sequence.append(f"review_round_{round_index}")
                return list(reviewers_arg)

            def fake_task_done(**kwargs):  # noqa: ANN001
                merged_path = Path(kwargs["directory"]) / kwargs["md_output_name"]
                review_count = sum(1 for item in call_sequence if item.startswith("review_round_"))
                if review_count == 1:
                    call_sequence.append("task_done_fail")
                    merged_path.write_text("第一轮未通过\n", encoding="utf-8")
                    return False
                call_sequence.append("task_done_pass")
                merged_path.write_text("", encoding="utf-8")
                return True

            def fake_run_ba_modify_loop(
                handoff,
                *,
                project_dir,
                paths,
                review_msg,
                progress=None,
                audit_context=None,
                review_round_index=None,
            ):  # noqa: ANN001
                _ = handoff
                _ = project_dir
                _ = progress
                self.assertEqual(review_msg, "第一轮未通过")
                call_sequence.append("modify")
                paths["detailed_design_path"].write_text("修改后的详细设计\n", encoding="utf-8")
                paths["ba_feedback_path"].write_text("已按建议修订\n", encoding="utf-8")
                paths["ask_human_path"].write_text("", encoding="utf-8")
                return fake_handoff

            def fake_prepare_ba(*args, **kwargs):  # noqa: ANN001
                call_sequence.append("prepare_ba")
                return fake_handoff, True

            def fake_initialize_ba(handoff, *, project_dir, paths, progress=None):  # noqa: ANN001
                _ = handoff
                _ = project_dir
                _ = paths
                _ = progress
                call_sequence.append("init_ba")
                return fake_handoff

            with patch("A05_DetailedDesign.stdin_is_interactive", return_value=True), patch(
                "A05_DetailedDesign.prompt_select_option",
                return_value="review_existing",
            ), patch(
                "A05_DetailedDesign.cleanup_stale_detailed_design_runtime_state",
                return_value=(),
            ), patch(
                "A05_DetailedDesign.cleanup_existing_detailed_design_artifacts",
                return_value=(),
            ), patch(
                "A05_DetailedDesign.prepare_design_ba_handoff",
                side_effect=fake_prepare_ba,
            ) as prepare_ba, patch(
                "A05_DetailedDesign.initialize_detailed_design_ba",
                side_effect=fake_initialize_ba,
            ), patch(
                "A05_DetailedDesign.generate_detailed_design_document",
            ) as generate_design, patch(
                "A05_DetailedDesign.resolve_reviewer_specs",
                return_value=reviewer_specs,
            ), patch(
                "A05_DetailedDesign.build_reviewer_workers",
                return_value=reviewers,
            ), patch(
                "A05_DetailedDesign._run_parallel_reviewers",
                side_effect=fake_parallel,
            ), patch(
                "A05_DetailedDesign.repair_reviewer_outputs",
                side_effect=lambda reviewers_arg, **kwargs: list(reviewers_arg),
            ), patch(
                "A05_DetailedDesign.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A05_DetailedDesign.run_ba_modify_loop",
                side_effect=fake_run_ba_modify_loop,
            ) as modify_loop, patch(
                "A05_DetailedDesign.mark_detailed_design_completed",
                return_value=paths["pre_development_path"],
            ), patch(
                "A05_DetailedDesign._shutdown_workers",
                return_value=(),
            ):
                result = run_detailed_design_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5"],
                )

            self.assertTrue(result.passed)
            generate_design.assert_not_called()
            prepare_ba.assert_called_once()
            modify_loop.assert_called_once()
            self.assertEqual(
                call_sequence,
                [
                    "review_round_1",
                    "task_done_fail",
                    "prepare_ba",
                    "init_ba",
                    "modify",
                    "review_round_2",
                    "task_done_pass",
                ],
            )
            self.assertEqual(paths["detailed_design_path"].read_text(encoding="utf-8"), "修改后的详细设计\n")

    def test_run_stage_review_existing_defers_discarding_incoming_ba_handoff_until_after_first_review(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_detailed_design_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
            paths["detailed_design_path"].write_text("已有详细设计\n", encoding="utf-8")
            incoming_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(
                    session_name="需求分析师-旧会话",
                    runtime_root=project_dir / ".detailed_design_runtime",
                    runtime_dir=project_dir / ".detailed_design_runtime" / "old-ba",
                ),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )

            def fake_task_done(**kwargs):  # noqa: ANN001
                self.assertFalse(incoming_handoff.worker.killed)
                Path(kwargs["directory"], kwargs["md_output_name"]).write_text("", encoding="utf-8")
                return True

            with patch("A05_DetailedDesign.stdin_is_interactive", return_value=True), patch(
                "A05_DetailedDesign.prompt_select_option",
                return_value="review_existing",
            ), patch(
                "A05_DetailedDesign.cleanup_stale_detailed_design_runtime_state",
                return_value=(),
            ), patch(
                "A05_DetailedDesign.cleanup_existing_detailed_design_artifacts",
                return_value=(),
            ), patch(
                "A05_DetailedDesign.prepare_design_ba_handoff",
            ) as prepare_ba, patch(
                "A05_DetailedDesign.generate_detailed_design_document",
            ), patch(
                "A05_DetailedDesign.resolve_reviewer_specs",
                return_value=[],
            ), patch(
                "A05_DetailedDesign.build_reviewer_workers",
                return_value=[],
            ), patch(
                "A05_DetailedDesign.ensure_active_reviewers",
                return_value=None,
            ), patch(
                "A05_DetailedDesign.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A05_DetailedDesign.mark_detailed_design_completed",
                return_value=paths["pre_development_path"],
            ), patch(
                "A05_DetailedDesign._shutdown_workers",
                return_value=(),
            ):
                result = run_detailed_design_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5"],
                    ba_handoff=incoming_handoff,
                )

            self.assertTrue(result.passed)
            prepare_ba.assert_not_called()
            self.assertTrue(incoming_handoff.worker.killed)

    def test_run_stage_review_existing_discards_incoming_ba_handoff_before_creating_new_ba(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_detailed_design_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
            paths["detailed_design_path"].write_text("已有详细设计\n", encoding="utf-8")
            incoming_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(
                    session_name="需求分析师-旧会话",
                    runtime_root=project_dir / ".detailed_design_runtime",
                    runtime_dir=project_dir / ".detailed_design_runtime" / "old-ba",
                ),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            fake_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(
                    session_name="需求分析师-新会话",
                    runtime_root=project_dir / ".detailed_design_runtime",
                    runtime_dir=project_dir / ".detailed_design_runtime" / "new-ba",
                ),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            reviewer_specs = [
                DetailedDesignReviewerSpec(role_name="开发工程师", role_prompt="实现视角", reviewer_key="开发工程师"),
            ]
            reviewers = [
                SimpleNamespace(
                    reviewer_name="开发工程师",
                    review_md_path=project_dir / "开发工程师.md",
                    review_json_path=project_dir / "开发工程师.json",
                )
            ]
            call_sequence: list[str] = []

            def fake_parallel(reviewers_arg, *, round_index, **kwargs):  # noqa: ANN001
                _ = kwargs
                if round_index == 1:
                    self.assertFalse(incoming_handoff.worker.killed)
                call_sequence.append(f"review_round_{round_index}")
                return list(reviewers_arg)

            def fake_task_done(**kwargs):  # noqa: ANN001
                merged_path = Path(kwargs["directory"]) / kwargs["md_output_name"]
                review_count = sum(1 for item in call_sequence if item.startswith("review_round_"))
                if review_count == 1:
                    merged_path.write_text("第一轮未通过\n", encoding="utf-8")
                    return False
                merged_path.write_text("", encoding="utf-8")
                return True

            def fake_prepare_ba(*args, **kwargs):  # noqa: ANN001
                _ = args
                _ = kwargs
                self.assertTrue(incoming_handoff.worker.killed)
                call_sequence.append("prepare_ba")
                return fake_handoff, True

            def fake_run_ba_modify_loop(
                handoff,
                *,
                project_dir,
                paths,
                review_msg,
                progress=None,
                audit_context=None,
                review_round_index=None,
            ):  # noqa: ANN001
                _ = handoff
                _ = project_dir
                _ = review_msg
                _ = progress
                paths["ba_feedback_path"].write_text("已按建议修订\n", encoding="utf-8")
                return fake_handoff

            with patch("A05_DetailedDesign.stdin_is_interactive", return_value=True), patch(
                "A05_DetailedDesign.prompt_select_option",
                return_value="review_existing",
            ), patch(
                "A05_DetailedDesign.cleanup_stale_detailed_design_runtime_state",
                return_value=(),
            ), patch(
                "A05_DetailedDesign.cleanup_existing_detailed_design_artifacts",
                return_value=(),
            ), patch(
                "A05_DetailedDesign.prepare_design_ba_handoff",
                side_effect=fake_prepare_ba,
            ), patch(
                "A05_DetailedDesign.initialize_detailed_design_ba",
                return_value=fake_handoff,
            ), patch(
                "A05_DetailedDesign.resolve_reviewer_specs",
                return_value=reviewer_specs,
            ), patch(
                "A05_DetailedDesign.build_reviewer_workers",
                return_value=reviewers,
            ), patch(
                "A05_DetailedDesign._run_parallel_reviewers",
                side_effect=fake_parallel,
            ), patch(
                "A05_DetailedDesign.repair_reviewer_outputs",
                side_effect=lambda reviewers_arg, **kwargs: list(reviewers_arg),
            ), patch(
                "A05_DetailedDesign.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A05_DetailedDesign.run_ba_modify_loop",
                side_effect=fake_run_ba_modify_loop,
            ), patch(
                "A05_DetailedDesign.mark_detailed_design_completed",
                return_value=paths["pre_development_path"],
            ), patch(
                "A05_DetailedDesign._shutdown_workers",
                return_value=(),
            ):
                result = run_detailed_design_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5"],
                    ba_handoff=incoming_handoff,
                )

            self.assertTrue(result.passed)
            self.assertEqual(call_sequence, ["review_round_1", "prepare_ba", "review_round_2"])

    def test_run_stage_does_not_offer_skip_in_yes_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_detailed_design_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
            paths["detailed_design_path"].write_text("已有详细设计\n", encoding="utf-8")
            fake_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(runtime_root=project_dir / ".detailed_design_runtime", runtime_dir=project_dir / ".detailed_design_runtime" / "ba"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )

            def fake_generate(handoff, *, project_dir, paths, initialize_first, progress=None):  # noqa: ANN001
                _ = handoff
                _ = project_dir
                _ = initialize_first
                _ = progress
                paths["detailed_design_path"].write_text("重新生成的详细设计\n", encoding="utf-8")
                return fake_handoff

            def fake_task_done(**kwargs):  # noqa: ANN001
                Path(kwargs["directory"], kwargs["md_output_name"]).write_text("", encoding="utf-8")
                return True

            with patch("A05_DetailedDesign.stdin_is_interactive", return_value=True), patch(
                "A05_DetailedDesign.prompt_yes_no_choice",
            ) as prompt_mock, patch(
                "A05_DetailedDesign.cleanup_stale_detailed_design_runtime_state",
                return_value=(),
            ) as cleanup_runtime, patch(
                "A05_DetailedDesign.cleanup_existing_detailed_design_artifacts",
                return_value=(),
            ), patch(
                "A05_DetailedDesign.prepare_design_ba_handoff",
                return_value=(fake_handoff, False),
            ), patch(
                "A05_DetailedDesign.generate_detailed_design_document",
                side_effect=fake_generate,
            ), patch(
                "A05_DetailedDesign.resolve_reviewer_specs",
                return_value=[],
            ), patch(
                "A05_DetailedDesign.build_reviewer_workers",
                return_value=[],
            ), patch(
                "A05_DetailedDesign.ensure_active_reviewers",
                return_value=None,
            ), patch(
                "A05_DetailedDesign.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A05_DetailedDesign.mark_detailed_design_completed",
                return_value=paths["pre_development_path"],
            ), patch(
                "A05_DetailedDesign._shutdown_workers",
                return_value=(),
            ):
                result = run_detailed_design_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5", "--yes"],
                )

            self.assertTrue(result.passed)
            prompt_mock.assert_not_called()
            cleanup_runtime.assert_called_once()

    def test_detailed_design_ba_prompts_do_not_reference_removed_completion_helper(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = build_detailed_design_paths(tmp_dir, "需求A")
            init_prompt = build_detailed_design_init_prompt(paths)
            design_prompt = build_detailed_design_prompt(paths)

        for prompt in (init_prompt, design_prompt):
            self.assertNotIn("completion helper", prompt)
            self.assertIn("仅输出 `完成`", prompt)

    def test_resolve_reviewer_specs_defaults_to_four_builtin_roles_in_noninteractive_mode(self):
        args = build_parser().parse_args([])

        with patch("A05_DetailedDesign.stdin_is_interactive", return_value=False):
            specs = resolve_reviewer_specs(args)

        self.assertEqual(
            [item.role_name for item in specs],
            ["开发工程师", "测试工程师", "架构师", "审核员"],
        )
        self.assertEqual(
            [item.reviewer_key for item in specs],
            ["开发工程师", "测试工程师", "架构师", "审核员"],
        )

    def test_resolve_reviewer_specs_uses_cli_values_without_interactive_collection(self):
        args = build_parser().parse_args(
            [
                "--reviewer-role",
                "开发工程师",
                "--reviewer-role",
                "审核员",
                "--reviewer-role-prompt",
                "实现视角",
                "--reviewer-role-prompt",
                "一致性视角",
            ]
        )

        with patch("A05_DetailedDesign.collect_interactive_reviewer_specs") as collect_mock:
            specs = resolve_reviewer_specs(args)

        collect_mock.assert_not_called()
        self.assertEqual(
            [(item.role_name, item.role_prompt) for item in specs],
            [("开发工程师", "实现视角"), ("审核员", "一致性视角")],
        )

    def test_resolve_reviewer_specs_requires_prompt_for_custom_cli_role(self):
        args = build_parser().parse_args(
            [
                "--reviewer-role",
                "性能审计师",
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "必须同时提供对应的角色定义提示词"):
            resolve_reviewer_specs(args)

    def test_collect_interactive_reviewer_specs_supports_default_and_custom_roles(self):
        select_values = iter(
            [
                "default",
                "架构师",
                "default",
                "custom",
            ]
        )
        text_values = iter(
            [
                "性能审计师",
                "检查性能瓶颈、资源开销和性能回归风险。",
            ]
        )

        def fake_prompt_select_option(**kwargs):  # noqa: ANN001
            return next(select_values)

        def fake_prompt_with_default(prompt_text, default="", allow_empty=False):  # noqa: ANN001
            _ = prompt_text
            _ = default
            _ = allow_empty
            return next(text_values)

        with patch("A05_DetailedDesign.prompt_positive_int", return_value=2), patch(
            "A05_DetailedDesign.prompt_select_option",
            side_effect=fake_prompt_select_option,
        ), patch(
            "A05_DetailedDesign.prompt_with_default",
            side_effect=fake_prompt_with_default,
        ):
            specs = collect_interactive_reviewer_specs()

        self.assertEqual(
            [(item.role_name, item.role_prompt, item.reviewer_key) for item in specs],
            [
                ("架构师", "你是详细设计审计中的架构师。重点检查系统边界、依赖影响、契约兼容、扩展约束，以及是否保持最小改动。", "架构师"),
                ("性能审计师", "检查性能瓶颈、资源开销和性能回归风险。", "性能审计师"),
            ],
        )

    def test_resolve_reviewer_specs_assigns_unique_keys_for_duplicate_roles(self):
        args = build_parser().parse_args(
            [
                "--reviewer-role",
                "开发工程师",
                "--reviewer-role",
                "开发工程师",
                "--reviewer-role-prompt",
                "实现视角A",
                "--reviewer-role-prompt",
                "实现视角B",
            ]
        )

        specs = resolve_reviewer_specs(args)

        self.assertEqual(
            [item.reviewer_key for item in specs],
            ["开发工程师#1", "开发工程师#2"],
        )

    def test_prepare_design_ba_handoff_reuses_live_review_ba_in_noninteractive_mode(self):
        handoff = RequirementsAnalystHandoff(
            worker=_FakeWorker(session_exists_value=True),
            vendor="codex",
            model="gpt-5.4",
            reasoning_effort="high",
            proxy_url="",
        )
        args = build_parser().parse_args(["--project-dir", "/tmp/project", "--requirement-name", "需求A"])

        with patch("A05_DetailedDesign.stdin_is_interactive", return_value=False):
            reused, created_new = prepare_design_ba_handoff(
                args,
                project_dir="/tmp/project",
                ba_handoff=handoff,
            )

        self.assertIs(reused, handoff)
        self.assertFalse(created_new)

    def test_prepare_design_ba_handoff_falls_back_to_rebuild_when_reuse_worker_is_not_live(self):
        old_handoff = RequirementsAnalystHandoff(
            worker=_FakeWorker(session_exists_value=False),
            vendor="codex",
            model="gpt-5.4",
            reasoning_effort="high",
            proxy_url="",
        )
        args = build_parser().parse_args(["--project-dir", "/tmp/project", "--requirement-name", "需求A", "--reuse-review-ba"])
        new_handoff = RequirementsAnalystHandoff(
            worker=_FakeWorker(session_name="需求分析师-天英星"),
            vendor="codex",
            model="gpt-5.4",
            reasoning_effort="high",
            proxy_url="",
        )

        with patch("A05_DetailedDesign.collect_ba_agent_selection", return_value=ReviewAgentSelection("codex", "gpt-5.4", "high", "")), patch(
            "A05_DetailedDesign.create_design_ba_handoff",
            return_value=new_handoff,
        ), patch("A05_DetailedDesign.message"):
            rebuilt, created_new = prepare_design_ba_handoff(
                args,
                project_dir="/tmp/project",
                ba_handoff=old_handoff,
            )

        self.assertIs(rebuilt, new_handoff)
        self.assertTrue(created_new)
        self.assertTrue(old_handoff.worker.killed)

    def test_prepare_design_ba_handoff_kills_old_worker_after_new_worker_is_created(self):
        old_handoff = RequirementsAnalystHandoff(
            worker=_FakeWorker(session_exists_value=True),
            vendor="codex",
            model="gpt-5.4",
            reasoning_effort="high",
            proxy_url="",
        )
        args = build_parser().parse_args(["--project-dir", "/tmp/project", "--requirement-name", "需求A", "--rebuild-review-ba"])
        new_handoff = RequirementsAnalystHandoff(
            worker=_FakeWorker(session_name="需求分析师-天英星"),
            vendor="codex",
            model="gpt-5.4",
            reasoning_effort="high",
            proxy_url="",
        )
        call_order: list[str] = []

        def fake_create_design_ba_handoff(*, project_dir, selection):  # noqa: ANN001
            self.assertFalse(old_handoff.worker.killed)
            call_order.append("create_new")
            return new_handoff

        with patch(
            "A05_DetailedDesign.collect_ba_agent_selection",
            return_value=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
        ), patch(
            "A05_DetailedDesign.create_design_ba_handoff",
            side_effect=fake_create_design_ba_handoff,
        ), patch("A05_DetailedDesign.message"):
            rebuilt, created_new = prepare_design_ba_handoff(
                args,
                project_dir="/tmp/project",
                ba_handoff=old_handoff,
            )

        self.assertEqual(call_order, ["create_new"])
        self.assertIs(rebuilt, new_handoff)
        self.assertTrue(created_new)
        self.assertTrue(old_handoff.worker.killed)

    def test_shutdown_workers_does_not_kill_reused_external_ba_on_a05_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            external_worker = _FakeWorker(
                session_name="分析师-参水猿",
                runtime_root=project_dir / ".requirements_clarification_runtime",
                runtime_dir=project_dir / ".requirements_clarification_runtime" / "ba",
            )
            handoff = RequirementsAnalystHandoff(
                worker=external_worker,
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )

            removed = _shutdown_workers(
                handoff,
                [],
                project_dir=project_dir,
                cleanup_runtime=False,
            )

        self.assertEqual(removed, ())
        self.assertFalse(external_worker.killed)

    def test_create_reviewer_runtime_uses_worker_session_name_for_detail_review_artifacts(self):
        class FakeTmuxWorker(_FakeWorker):
            def __init__(self, *, runtime_root, **kwargs):  # noqa: ANN001
                super().__init__(
                    session_name="开发工程师-天魁星",
                    runtime_root=runtime_root,
                    runtime_dir=Path(runtime_root) / "reviewer",
                )

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "A05_DetailedDesign.TmuxBatchWorker",
            FakeTmuxWorker,
        ), patch("sys.stdout", io.StringIO()):
            reviewer = create_reviewer_runtime(
                project_dir=tmpdir,
                requirement_name="需求A",
                reviewer_spec=DetailedDesignReviewerSpec(
                    role_name="开发工程师",
                    role_prompt="实现视角",
                ),
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
            )

        self.assertEqual(reviewer.worker.session_name, "开发工程师-天魁星")
        self.assertTrue(reviewer.review_md_path.name.endswith("详设评审记录_开发工程师-天魁星.md"))
        self.assertTrue(reviewer.review_json_path.name.endswith("评审记录_开发工程师-天魁星.json"))

    def test_create_reviewer_runtime_uses_reviewer_key_as_stable_runtime_identity(self):
        class FakeTmuxWorker(_FakeWorker):
            def __init__(self, *, runtime_root, **kwargs):  # noqa: ANN001
                super().__init__(
                    session_name="开发工程师-天魁星",
                    runtime_root=runtime_root,
                    runtime_dir=Path(runtime_root) / "reviewer",
                )

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "A05_DetailedDesign.TmuxBatchWorker",
            FakeTmuxWorker,
        ), patch("sys.stdout", io.StringIO()):
            reviewer = create_reviewer_runtime(
                project_dir=tmpdir,
                requirement_name="需求A",
                reviewer_spec=DetailedDesignReviewerSpec(
                    role_name="开发工程师",
                    role_prompt="实现视角",
                    reviewer_key="开发工程师#2",
                ),
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
            )

        self.assertEqual(reviewer.reviewer_name, "开发工程师#2")

    def test_build_reviewer_workers_supports_duplicate_role_specs(self):
        specs = [
            DetailedDesignReviewerSpec(role_name="开发工程师", role_prompt="实现视角A", reviewer_key="开发工程师#1"),
            DetailedDesignReviewerSpec(role_name="开发工程师", role_prompt="实现视角B", reviewer_key="开发工程师#2"),
        ]
        observed_role_labels: list[str] = []

        def fake_prompt_review_agent_selection(default_vendor, default_model="", default_reasoning_effort="high", default_proxy_url="", *, role_label="", progress=None):  # noqa: ANN001
            _ = default_vendor
            _ = default_model
            _ = default_reasoning_effort
            _ = default_proxy_url
            _ = progress
            observed_role_labels.append(role_label)
            return ReviewAgentSelection("codex", "gpt-5.4", "high", "")

        def fake_create_reviewer_runtime(*, project_dir, requirement_name, reviewer_spec, selection):  # noqa: ANN001
            runtime_root = Path(project_dir) / ".detailed_design_runtime"
            return SimpleNamespace(
                reviewer_name=reviewer_spec.reviewer_key,
                selection=selection,
                worker=_FakeWorker(
                    session_name=f"{reviewer_spec.role_name}-天魁星",
                    runtime_root=runtime_root,
                    runtime_dir=runtime_root / reviewer_spec.reviewer_key,
                ),
                review_md_path=Path(project_dir) / f"{reviewer_spec.reviewer_key}.md",
                review_json_path=Path(project_dir) / f"{reviewer_spec.reviewer_key}.json",
                contract=SimpleNamespace(),
            )

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "A05_DetailedDesign.stdin_is_interactive",
            return_value=True,
        ), patch(
            "A05_DetailedDesign.prompt_review_agent_selection",
            side_effect=fake_prompt_review_agent_selection,
        ), patch(
            "A05_DetailedDesign.create_reviewer_runtime",
            side_effect=fake_create_reviewer_runtime,
        ):
            reviewers = build_reviewer_workers(
                build_parser().parse_args([]),
                project_dir=tmpdir,
                requirement_name="需求A",
                reviewer_specs=specs,
            )

        self.assertEqual([item.reviewer_name for item in reviewers], ["开发工程师#1", "开发工程师#2"])
        self.assertEqual(len(set(observed_role_labels)), 2)

    def test_build_reviewer_workers_uses_precollected_selection_without_prompting(self):
        specs = [
            DetailedDesignReviewerSpec(role_name="开发工程师", role_prompt="实现视角", reviewer_key="开发工程师"),
        ]

        def fake_create_reviewer_runtime(*, project_dir, requirement_name, reviewer_spec, selection):  # noqa: ANN001
            runtime_root = Path(project_dir) / ".detailed_design_runtime"
            return SimpleNamespace(
                reviewer_name=reviewer_spec.reviewer_key,
                selection=selection,
                worker=_FakeWorker(
                    session_name=f"{reviewer_spec.role_name}-天魁星",
                    runtime_root=runtime_root,
                    runtime_dir=runtime_root / reviewer_spec.reviewer_key,
                ),
                review_md_path=Path(project_dir) / f"{reviewer_spec.reviewer_key}.md",
                review_json_path=Path(project_dir) / f"{reviewer_spec.reviewer_key}.json",
                contract=SimpleNamespace(),
            )

        preselected = {"开发工程师": ReviewAgentSelection("claude", "sonnet", "medium", "")}
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "A05_DetailedDesign.stdin_is_interactive",
            return_value=True,
        ), patch(
            "A05_DetailedDesign.prompt_review_agent_selection",
        ) as selection_prompt, patch(
            "A05_DetailedDesign.create_reviewer_runtime",
            side_effect=fake_create_reviewer_runtime,
        ):
            reviewers = build_reviewer_workers(
                build_parser().parse_args([]),
                project_dir=tmpdir,
                requirement_name="需求A",
                reviewer_specs=specs,
                reviewer_selections_by_name=preselected,
            )

        selection_prompt.assert_not_called()
        self.assertEqual(reviewers[0].selection.vendor, "claude")
        self.assertEqual(reviewers[0].selection.model, "sonnet")

    def test_run_stage_collects_reviewer_models_before_generating_design(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_detailed_design_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
            paths["detailed_design_path"].write_text("已有详细设计\n", encoding="utf-8")
            fake_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(runtime_root=project_dir / ".detailed_design_runtime", runtime_dir=project_dir / ".detailed_design_runtime" / "ba"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            specs = [
                DetailedDesignReviewerSpec(role_name="开发工程师", role_prompt="实现视角", reviewer_key="开发工程师"),
            ]
            events: list[str] = []

            def fake_collect(**kwargs):  # noqa: ANN001
                events.append("collect")
                self.assertEqual(kwargs["reviewer_specs"], specs)
                return {"开发工程师": ReviewAgentSelection("claude", "sonnet", "medium", "")}

            def fake_generate(handoff, *, project_dir, paths, initialize_first, progress=None):  # noqa: ANN001
                _ = handoff
                _ = project_dir
                _ = initialize_first
                _ = progress
                events.append("generate")
                paths["detailed_design_path"].write_text("重新生成的详细设计\n", encoding="utf-8")
                return fake_handoff

            def fake_build_reviewers(*args, **kwargs):  # noqa: ANN001
                events.append("build_reviewers")
                self.assertEqual(kwargs["reviewer_selections_by_name"]["开发工程师"].vendor, "claude")
                return []

            def fake_task_done(**kwargs):  # noqa: ANN001
                Path(kwargs["directory"], kwargs["md_output_name"]).write_text("", encoding="utf-8")
                return True

            with patch("A05_DetailedDesign.stdin_is_interactive", return_value=True), patch(
                "A05_DetailedDesign.prompt_select_option",
                return_value="rerun",
            ), patch(
                "A05_DetailedDesign.cleanup_stale_detailed_design_runtime_state",
                return_value=(),
            ), patch(
                "A05_DetailedDesign.cleanup_existing_detailed_design_artifacts",
                return_value=(),
            ), patch(
                "A05_DetailedDesign.prepare_design_ba_handoff",
                return_value=(fake_handoff, False),
            ), patch(
                "A05_DetailedDesign.resolve_reviewer_specs",
                return_value=specs,
            ), patch(
                "A05_DetailedDesign.collect_reviewer_agent_selections",
                side_effect=fake_collect,
            ), patch(
                "A05_DetailedDesign.generate_detailed_design_document",
                side_effect=fake_generate,
            ), patch(
                "A05_DetailedDesign.build_reviewer_workers",
                side_effect=fake_build_reviewers,
            ), patch(
                "A05_DetailedDesign.ensure_active_reviewers",
                return_value=None,
            ), patch(
                "A05_DetailedDesign.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A05_DetailedDesign.mark_detailed_design_completed",
                return_value=paths["pre_development_path"],
            ), patch(
                "A05_DetailedDesign._shutdown_workers",
                return_value=(),
            ):
                result = run_detailed_design_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5"],
                )

        self.assertTrue(result.passed)
        self.assertEqual(events[:2], ["collect", "generate"])

    def test_run_parallel_reviewers_uses_unique_reviewer_name_mapping_for_duplicate_roles(self):
        reviewers = [
            SimpleNamespace(reviewer_name="开发工程师#1"),
            SimpleNamespace(reviewer_name="开发工程师#2"),
        ]
        reviewer_specs_by_name = {
            "开发工程师#1": DetailedDesignReviewerSpec(
                role_name="开发工程师",
                role_prompt="实现视角A",
                reviewer_key="开发工程师#1",
            ),
            "开发工程师#2": DetailedDesignReviewerSpec(
                role_name="开发工程师",
                role_prompt="实现视角B",
                reviewer_key="开发工程师#2",
            ),
        }
        observed_prompts: dict[str, str] = {}

        def fake_run_reviewer_turn_with_recreation(reviewer_runtime, **kwargs):  # noqa: ANN001
            observed_prompts[reviewer_runtime.reviewer_name] = kwargs["prompt"]
            return reviewer_runtime

        with patch("A05_DetailedDesign.run_reviewer_turn_with_recreation", side_effect=fake_run_reviewer_turn_with_recreation):
            result = _run_parallel_reviewers(
                reviewers,
                reviewer_specs_by_name=reviewer_specs_by_name,
                project_dir="/tmp/project",
                requirement_name="需求A",
                round_index=1,
                prompt_builder=lambda reviewer, reviewer_spec: reviewer_spec.role_prompt,
                label_prefix="detailed_design_review_init",
            )

        self.assertEqual(len(result), 2)
        self.assertEqual(
            observed_prompts,
            {
                "开发工程师#1": "实现视角A",
                "开发工程师#2": "实现视角B",
            },
        )

    def test_run_reviewer_turn_with_recreation_returns_current_reviewer_on_contract_violation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reviewer = create_reviewer_runtime(
                project_dir=root,
                requirement_name="需求A",
                reviewer_spec=DetailedDesignReviewerSpec(
                    role_name="审核员",
                    role_prompt="一致性检查",
                    reviewer_key="审核员#1",
                ),
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
            )

            with patch.object(
                reviewer.worker,
                "run_turn",
                return_value=SimpleNamespace(
                    ok=False,
                    clean_output=(
                        "turn artifacts contract violation after task completion: "
                        "phase=详细设计 status_path=/tmp/review.json error=审核器未通过，但评审 markdown 为空"
                    ),
                ),
            ), patch(
                "A05_DetailedDesign.recreate_reviewer_runtime",
                side_effect=AssertionError("contract violation should enter repair loop instead of rebuilding reviewer"),
            ), patch(
                "tmux_core.stage_kernel.turn_output_goals.request_file_noncompliance_intervention",
                side_effect=RuntimeError("manual intervention requested"),
            ):
                with self.assertRaises(RuntimeError) as raised:
                    run_reviewer_turn_with_recreation(
                        reviewer,
                        project_dir=root,
                        requirement_name="需求A",
                        reviewer_spec=DetailedDesignReviewerSpec(
                            role_name="审核员",
                            role_prompt="一致性检查",
                            reviewer_key="审核员#1",
                        ),
                        label="review",
                        prompt="do review",
                    )

        self.assertIn("manual intervention requested", str(raised.exception))

    def test_run_stage_auto_runs_a03_when_requirements_clear_missing_and_initializes_new_ba(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_detailed_design_paths(project_dir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")

            clarification_calls: list[list[str]] = []
            generate_calls: list[bool] = []

            def fake_clarification_stage(argv, preserve_ba_worker=False):  # noqa: ANN001
                clarification_calls.append(list(argv))
                paths["requirements_clear_path"].write_text("需求澄清\n", encoding="utf-8")
                return SimpleNamespace(requirement_name="需求A")

            def fake_generate(handoff, *, project_dir, paths, initialize_first, progress=None):  # noqa: ANN001
                generate_calls.append(bool(initialize_first))
                paths["detailed_design_path"].write_text("详细设计正文\n", encoding="utf-8")
                return handoff

            def fake_task_done(**kwargs):  # noqa: ANN001
                Path(kwargs["directory"], kwargs["md_output_name"]).write_text("", encoding="utf-8")
                return True

            fake_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(runtime_root=project_dir / ".detailed_design_runtime", runtime_dir=project_dir / ".detailed_design_runtime" / "ba"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )

            with patch("A05_DetailedDesign.run_requirements_clarification_stage", side_effect=fake_clarification_stage), patch(
                "A05_DetailedDesign.cleanup_stale_detailed_design_runtime_state",
                return_value=(),
            ), patch(
                "A05_DetailedDesign.prepare_design_ba_handoff",
                return_value=(fake_handoff, True),
            ), patch(
                "A05_DetailedDesign.generate_detailed_design_document",
                side_effect=fake_generate,
            ), patch(
                "A05_DetailedDesign.build_reviewer_workers",
                return_value=[],
            ), patch(
                "A05_DetailedDesign.ensure_active_reviewers",
                return_value=None,
            ), patch(
                "A05_DetailedDesign.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A05_DetailedDesign._shutdown_workers",
                return_value=(),
            ), patch(
                "A05_DetailedDesign.mark_detailed_design_completed",
                return_value=paths["pre_development_path"],
            ):
                result = run_detailed_design_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5"],
                )

        self.assertTrue(clarification_calls)
        self.assertEqual(generate_calls, [True])
        self.assertTrue(result.passed)
        self.assertEqual(result.requirement_name, "需求A")

    def test_run_stage_configures_ba_and_generates_design_before_resolving_reviewers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_detailed_design_paths(project_dir, "需求A")
            call_order: list[str] = []
            fake_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(runtime_root=project_dir / ".detailed_design_runtime", runtime_dir=project_dir / ".detailed_design_runtime" / "ba"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )

            def fake_prepare(*args, **kwargs):  # noqa: ANN001
                call_order.append("prepare_ba")
                return fake_handoff, True

            def fake_generate(handoff, *, project_dir, paths, initialize_first, progress=None):  # noqa: ANN001
                _ = project_dir
                _ = initialize_first
                _ = progress
                call_order.append("generate_design")
                paths["detailed_design_path"].write_text("详细设计正文\n", encoding="utf-8")
                return handoff

            def fake_resolve(*args, **kwargs):  # noqa: ANN001
                call_order.append("resolve_reviewers")
                return []

            def fake_build(*args, **kwargs):  # noqa: ANN001
                call_order.append("build_reviewers")
                return []

            def fake_task_done(**kwargs):  # noqa: ANN001
                call_order.append("task_done")
                Path(kwargs["directory"], kwargs["md_output_name"]).write_text("", encoding="utf-8")
                return True

            with patch("A05_DetailedDesign.ensure_detailed_design_inputs", return_value=paths), patch(
                "A05_DetailedDesign.update_pre_development_task_status",
                return_value=None,
            ), patch(
                "A05_DetailedDesign.cleanup_stale_detailed_design_runtime_state",
                return_value=(),
            ), patch(
                "A05_DetailedDesign.cleanup_existing_detailed_design_artifacts",
                return_value=(),
            ), patch(
                "A05_DetailedDesign.prepare_design_ba_handoff",
                side_effect=fake_prepare,
            ), patch(
                "A05_DetailedDesign.generate_detailed_design_document",
                side_effect=fake_generate,
            ), patch(
                "A05_DetailedDesign.resolve_reviewer_specs",
                side_effect=fake_resolve,
            ), patch(
                "A05_DetailedDesign.build_reviewer_workers",
                side_effect=fake_build,
            ), patch(
                "A05_DetailedDesign.ensure_active_reviewers",
                return_value=None,
            ), patch(
                "A05_DetailedDesign.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A05_DetailedDesign.mark_detailed_design_completed",
                return_value=paths["pre_development_path"],
            ), patch(
                "A05_DetailedDesign._shutdown_workers",
                return_value=(),
            ):
                result = run_detailed_design_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5"],
                )

        self.assertTrue(result.passed)
        self.assertEqual(
            call_order[:4],
            ["prepare_ba", "resolve_reviewers", "generate_design", "build_reviewers"],
        )

    def test_generate_detailed_design_document_uses_result_contract_for_init_and_generate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_detailed_design_paths(project_dir, "需求A")
            for file_path in (
                paths["original_requirement_path"],
                paths["requirements_clear_path"],
                paths["hitl_record_path"],
            ):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("ready\n", encoding="utf-8")

            calls: list[tuple[str, str]] = []

            class _StructuredResultWorker:
                session_name = "需求分析师-天佑星"

                def run_turn(self, *, label, prompt, result_contract, timeout_sec):  # noqa: ANN001
                    calls.append((label, result_contract.mode))
                    if result_contract.mode == "a05_detailed_design_generate":
                        paths["detailed_design_path"].write_text("详细设计正文\n", encoding="utf-8")
                    return SimpleNamespace(
                        ok=True,
                        clean_output=json.dumps({"status": result_contract.expected_statuses[0]}, ensure_ascii=False),
                        exit_code=0,
                    )

            handoff = RequirementsAnalystHandoff(
                worker=_StructuredResultWorker(),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )

            returned_handoff = generate_detailed_design_document(
                handoff,
                project_dir=project_dir,
                paths=paths,
                initialize_first=True,
            )

        self.assertIs(returned_handoff, handoff)
        self.assertEqual(
            calls,
            [
                ("detailed_design_ba_init", "a05_ba_init"),
                ("generate_detailed_design", "a05_detailed_design_generate"),
            ],
        )

    def test_run_ba_modify_loop_uses_feedback_result_contract_for_modify_and_hitl_reply(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_detailed_design_paths(project_dir, "需求A")
            for file_path in (
                paths["original_requirement_path"],
                paths["requirements_clear_path"],
                paths["hitl_record_path"],
                paths["detailed_design_path"],
            ):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("ready\n", encoding="utf-8")

            calls: list[tuple[str, str]] = []
            ask_human_before_reply: list[str] = []

            class _StructuredResultWorker:
                session_name = "需求分析师-天佑星"

                def run_turn(self, *, label, prompt, result_contract, timeout_sec):  # noqa: ANN001
                    calls.append((label, result_contract.mode))
                    if len(calls) == 1:
                        paths["ask_human_path"].write_text("请补充边界条件\n", encoding="utf-8")
                        paths["ba_feedback_path"].write_text("", encoding="utf-8")
                        return SimpleNamespace(
                            ok=True,
                            clean_output=json.dumps({"status": "hitl"}, ensure_ascii=False),
                            exit_code=0,
                        )
                    ask_human_before_reply.append(paths["ask_human_path"].read_text(encoding="utf-8"))
                    paths["ask_human_path"].write_text("", encoding="utf-8")
                    paths["ba_feedback_path"].write_text("已修订\n", encoding="utf-8")
                    paths["detailed_design_path"].write_text("更新后的详细设计\n", encoding="utf-8")
                    return SimpleNamespace(
                        ok=True,
                        clean_output=json.dumps({"status": "completed"}, ensure_ascii=False),
                        exit_code=0,
                    )

            handoff = RequirementsAnalystHandoff(
                worker=_StructuredResultWorker(),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )

            with patch("A05_DetailedDesign._collect_design_hitl_response", return_value="补充说明"):
                returned_handoff = run_ba_modify_loop(
                    handoff,
                    project_dir=project_dir,
                    paths=paths,
                    review_msg="需要补充边界",
                )

        self.assertIs(returned_handoff, handoff)
        self.assertEqual(
            calls,
            [
                ("modify_detailed_design", "a05_detailed_design_feedback"),
                ("detailed_design_hitl_reply_round_1", "a05_detailed_design_feedback"),
            ],
        )
        self.assertEqual(ask_human_before_reply, [""])

    def test_run_reviewer_turn_with_recreation_drops_dead_reviewer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reviewer = create_reviewer_runtime(
                project_dir=root,
                requirement_name="需求A",
                reviewer_spec=DetailedDesignReviewerSpec(
                    role_name="审核员",
                    role_prompt="一致性检查",
                    reviewer_key="审核员#1",
                ),
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
            )

            with patch.object(
                reviewer.worker,
                "run_turn",
                side_effect=RuntimeError("tmux pane died while waiting for turn artifacts"),
            ), patch(
                "A05_DetailedDesign.recreate_reviewer_runtime",
                return_value=None,
            ) as recreate_mock:
                returned = run_reviewer_turn_with_recreation(
                    reviewer,
                    project_dir=root,
                    requirement_name="需求A",
                    reviewer_spec=DetailedDesignReviewerSpec(
                        role_name="审核员",
                        role_prompt="一致性检查",
                        reviewer_key="审核员#1",
                    ),
                    label="review",
                    prompt="do review",
                )

        self.assertIsNone(returned)
        recreate_mock.assert_called_once()

    def test_run_reviewer_turn_with_recreation_reuses_materialized_outputs_after_runtime_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reviewer = create_reviewer_runtime(
                project_dir=root,
                requirement_name="需求A",
                reviewer_spec=DetailedDesignReviewerSpec(
                    role_name="架构师",
                    role_prompt="架构检查",
                    reviewer_key="架构师#1",
                ),
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
            )

            def fake_run_turn(**kwargs):  # noqa: ANN001
                reviewer.review_md_path.write_text("- [Ambiguity] still unresolved\n", encoding="utf-8")
                reviewer.review_json_path.write_text(
                    json.dumps([{"task_name": "详细设计", "review_pass": True}], ensure_ascii=False),
                    encoding="utf-8",
                )
                return SimpleNamespace(ok=False, clean_output="runtime failed")

            with patch.object(reviewer.worker, "run_turn", side_effect=fake_run_turn):
                returned = run_reviewer_turn_with_recreation(
                    reviewer,
                    project_dir=root,
                    requirement_name="需求A",
                    reviewer_spec=DetailedDesignReviewerSpec(
                        role_name="架构师",
                        role_prompt="架构检查",
                        reviewer_key="架构师#1",
                    ),
                    label="review",
                    prompt="do review",
                )

        self.assertIs(returned, reviewer)

    def test_run_stage_requires_nonempty_ba_feedback_before_again_review(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_detailed_design_paths(project_dir, "需求A")
            reviewers = [
                SimpleNamespace(
                    reviewer_name="开发工程师#1",
                    review_md_path=project_dir / "reviewer.md",
                    review_json_path=project_dir / "reviewer.json",
                )
            ]
            reviewer_specs = [
                DetailedDesignReviewerSpec(
                    role_name="开发工程师",
                    role_prompt="实现视角",
                    reviewer_key="开发工程师#1",
                )
            ]
            fake_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(runtime_root=project_dir / ".detailed_design_runtime", runtime_dir=project_dir / ".detailed_design_runtime" / "ba"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            parallel_calls: list[int] = []

            def fake_generate(handoff, *, project_dir, paths, initialize_first, progress=None):  # noqa: ANN001
                _ = project_dir
                _ = initialize_first
                _ = progress
                paths["detailed_design_path"].write_text("详细设计正文\n", encoding="utf-8")
                return handoff

            def fake_parallel(reviewers_arg, *, round_index, **kwargs):  # noqa: ANN001
                _ = kwargs
                parallel_calls.append(round_index)
                return list(reviewers_arg)

            def fake_run_ba_modify_loop(
                handoff,
                *,
                project_dir,
                paths,
                review_msg,
                progress=None,
                audit_context=None,
                review_round_index=None,
            ):  # noqa: ANN001
                _ = project_dir
                _ = review_msg
                _ = progress
                paths["ask_human_path"].write_text("", encoding="utf-8")
                paths["ba_feedback_path"].write_text("", encoding="utf-8")
                return handoff

            def fake_task_done(**kwargs):  # noqa: ANN001
                merged_path = Path(kwargs["directory"]) / kwargs["md_output_name"]
                merged_path.write_text("需要补充字段含义\n", encoding="utf-8")
                return False

            with patch("A05_DetailedDesign.ensure_detailed_design_inputs", return_value=paths), patch(
                "A05_DetailedDesign.update_pre_development_task_status",
                return_value=None,
            ), patch(
                "A05_DetailedDesign.cleanup_stale_detailed_design_runtime_state",
                return_value=(),
            ), patch(
                "A05_DetailedDesign.cleanup_existing_detailed_design_artifacts",
                return_value=(),
            ), patch(
                "A05_DetailedDesign.prepare_design_ba_handoff",
                return_value=(fake_handoff, False),
            ), patch(
                "A05_DetailedDesign.generate_detailed_design_document",
                side_effect=fake_generate,
            ), patch(
                "A05_DetailedDesign.resolve_reviewer_specs",
                return_value=reviewer_specs,
            ), patch(
                "A05_DetailedDesign.build_reviewer_workers",
                return_value=reviewers,
            ), patch(
                "A05_DetailedDesign._run_parallel_reviewers",
                side_effect=fake_parallel,
            ), patch(
                "A05_DetailedDesign.repair_reviewer_outputs",
                side_effect=lambda reviewers_arg, **kwargs: list(reviewers_arg),
            ), patch(
                "A05_DetailedDesign.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A05_DetailedDesign.run_ba_modify_loop",
                side_effect=fake_run_ba_modify_loop,
            ), patch(
                "A05_DetailedDesign._shutdown_workers",
                return_value=(),
            ):
                with self.assertRaisesRegex(RuntimeError, "需求分析师反馈.*为空"):
                    run_detailed_design_stage(
                        ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5"],
                    )

        self.assertEqual(parallel_calls, [1])

    def test_run_stage_round_two_keeps_all_reviewers_and_calls_ba_modify_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paths = build_detailed_design_paths(project_dir, "需求A")
            reviewer_specs = [
                DetailedDesignReviewerSpec(role_name="开发工程师", role_prompt="实现视角", reviewer_key="开发工程师"),
                DetailedDesignReviewerSpec(role_name="测试工程师", role_prompt="测试视角", reviewer_key="测试工程师"),
                DetailedDesignReviewerSpec(role_name="架构师", role_prompt="架构视角", reviewer_key="架构师"),
                DetailedDesignReviewerSpec(role_name="审核员", role_prompt="一致性视角", reviewer_key="审核员"),
            ]
            reviewers = [
                SimpleNamespace(
                    reviewer_name=item.reviewer_key,
                    review_md_path=project_dir / f"{item.reviewer_key}.md",
                    review_json_path=project_dir / f"{item.reviewer_key}.json",
                )
                for item in reviewer_specs
            ]
            fake_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(runtime_root=project_dir / ".detailed_design_runtime", runtime_dir=project_dir / ".detailed_design_runtime" / "ba"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            parallel_rounds: list[tuple[int, list[str]]] = []
            ba_modify_calls: list[str] = []

            def fake_generate(handoff, *, project_dir, paths, initialize_first, progress=None):  # noqa: ANN001
                _ = project_dir
                _ = initialize_first
                _ = progress
                paths["detailed_design_path"].write_text("详细设计正文\n", encoding="utf-8")
                return handoff

            def fake_parallel(reviewers_arg, *, round_index, **kwargs):  # noqa: ANN001
                _ = kwargs
                parallel_rounds.append((round_index, [item.reviewer_name for item in reviewers_arg]))
                return list(reviewers_arg)

            def fake_task_done(**kwargs):  # noqa: ANN001
                merged_path = Path(kwargs["directory"]) / kwargs["md_output_name"]
                if len(parallel_rounds) == 1:
                    merged_path.write_text("第一轮未通过\n", encoding="utf-8")
                    return False
                merged_path.write_text("", encoding="utf-8")
                return True

            def fake_run_ba_modify_loop(
                handoff,
                *,
                project_dir,
                paths,
                review_msg,
                progress=None,
                audit_context=None,
                review_round_index=None,
            ):  # noqa: ANN001
                _ = handoff
                _ = project_dir
                _ = progress
                ba_modify_calls.append(review_msg)
                paths["ask_human_path"].write_text("", encoding="utf-8")
                paths["ba_feedback_path"].write_text("已按建议修订\n", encoding="utf-8")
                return fake_handoff

            with patch("A05_DetailedDesign.ensure_detailed_design_inputs", return_value=paths), patch(
                "A05_DetailedDesign.update_pre_development_task_status",
                return_value=None,
            ), patch(
                "A05_DetailedDesign.cleanup_stale_detailed_design_runtime_state",
                return_value=(),
            ), patch(
                "A05_DetailedDesign.cleanup_existing_detailed_design_artifacts",
                return_value=(),
            ), patch(
                "A05_DetailedDesign.prepare_design_ba_handoff",
                return_value=(fake_handoff, False),
            ), patch(
                "A05_DetailedDesign.generate_detailed_design_document",
                side_effect=fake_generate,
            ), patch(
                "A05_DetailedDesign.resolve_reviewer_specs",
                return_value=reviewer_specs,
            ), patch(
                "A05_DetailedDesign.build_reviewer_workers",
                return_value=reviewers,
            ), patch(
                "A05_DetailedDesign._run_parallel_reviewers",
                side_effect=fake_parallel,
            ), patch(
                "A05_DetailedDesign.repair_reviewer_outputs",
                side_effect=lambda reviewers_arg, **kwargs: list(reviewers_arg),
            ), patch(
                "A05_DetailedDesign.task_done",
                side_effect=fake_task_done,
            ), patch(
                "A05_DetailedDesign.run_ba_modify_loop",
                side_effect=fake_run_ba_modify_loop,
            ), patch(
                "A05_DetailedDesign.mark_detailed_design_completed",
                return_value=paths["pre_development_path"],
            ), patch(
                "A05_DetailedDesign._shutdown_workers",
                return_value=(),
            ):
                result = run_detailed_design_stage(
                    ["--project-dir", str(project_dir), "--requirement-name", "需求A", "--review-max-rounds", "5"],
                )

        self.assertTrue(result.passed)
        self.assertEqual(
            parallel_rounds,
            [
                (1, ["开发工程师", "测试工程师", "架构师", "审核员"]),
                (2, ["开发工程师", "测试工程师", "架构师", "审核员"]),
            ],
        )
        self.assertEqual(ba_modify_calls, ["第一轮未通过"])

    def test_cleanup_stale_detailed_design_runtime_state_scopes_by_requirement_and_keeps_live_legacy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_root = root / ".detailed_design_runtime"
            target_dir = runtime_root / "target-reviewer"
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "详设-当前需求",
                        "project_dir": str(root.resolve()),
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a05.start",
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
                        "session_name": "详设-其他需求",
                        "project_dir": str(root.resolve()),
                        "requirement_name": "需求B",
                        "workflow_action": "stage.a05.start",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            legacy_live_dir = runtime_root / "legacy-live"
            legacy_live_dir.mkdir(parents=True, exist_ok=True)
            (legacy_live_dir / "worker.state.json").write_text(
                json.dumps({"session_name": "详设-遗留存活"}, ensure_ascii=False),
                encoding="utf-8",
            )
            legacy_dead_dir = runtime_root / "legacy-dead"
            legacy_dead_dir.mkdir(parents=True, exist_ok=True)
            (legacy_dead_dir / "worker.state.json").write_text(
                json.dumps({"session_name": "详设-遗留死亡", "agent_state": "DEAD"}, ensure_ascii=False),
                encoding="utf-8",
            )
            killed_sessions: list[str] = []

            class FakeTmuxRuntimeController:
                def session_exists(self, session_name: str) -> bool:
                    return session_name in {"详设-遗留存活", "详设-其他需求"}

                def kill_session(self, session_name: str, *, missing_ok: bool = True):  # noqa: ANN001
                    killed_sessions.append(session_name)
                    return session_name

            with patch("tmux_core.stage_kernel.runtime_scope_cleanup.TmuxRuntimeController", FakeTmuxRuntimeController):
                removed = cleanup_stale_detailed_design_runtime_state(root, "需求A")

            self.assertFalse(target_dir.exists())
            self.assertFalse(legacy_dead_dir.exists())
            self.assertTrue(other_requirement_dir.exists())
            self.assertTrue(legacy_live_dir.exists())
            self.assertIn("详设-当前需求", killed_sessions)
            self.assertIn("详设-遗留死亡", killed_sessions)
            self.assertIn(str(target_dir.resolve()), removed)
            self.assertIn(str(legacy_dead_dir.resolve()), removed)


if __name__ == "__main__":
    unittest.main()
