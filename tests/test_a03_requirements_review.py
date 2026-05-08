from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from Prompt_04_RequirementsReview import resume_ba
from A03_RequirementsReview import (
    REQUIREMENTS_REVIEW_TASK_NAME,
    MAX_REVIEWER_REPAIR_ATTEMPTS,
    RequirementsAnalystHandoff,
    ReviewAgentSelection,
    ReviewerRuntime,
    build_reviewer_artifact_paths,
    build_reviewer_workers,
    build_ba_human_feedback_result_contract,
    build_ba_resume_result_contract,
    build_ba_review_feedback_result_contract,
    build_parser,
    build_reviewer_completion_contract,
    build_requirements_review_paths,
    cleanup_stale_review_runtime_state,
    create_reviewer_runtime,
    prepare_ba_handoff,
    repair_reviewer_outputs,
    run_ba_turn_with_recreation,
    run_reviewer_turn_with_recreation,
    run_human_check_loop,
    run_requirements_review_limit_hitl_loop,
    run_requirements_review_stage,
    resolve_review_max_rounds,
    _shutdown_workers,
    cleanup_existing_review_artifacts,
)
from T05_hitl_runtime import build_prefixed_sha256
from T09_terminal_ops import BridgePromptRequest, BridgeTerminalUI, PROMPT_BACK_VALUE, PromptBackRequested, use_terminal_ui
from T08_pre_development import (
    build_pre_development_task_record_path,
    ensure_pre_development_task_record,
)


class _FakeWorker:
    def __init__(self, runtime_root: str | Path = "/tmp/runtime", runtime_dir: str | Path = "/tmp/runtime/worker"):
        self.runtime_root = Path(runtime_root)
        self.runtime_dir = Path(runtime_dir)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.prompts: list[tuple[str, str]] = []
        self.killed = False
        self.session_name = "demo-session"

    def run_turn(self, *, label, prompt, **kwargs):  # noqa: ANN001
        self.prompts.append((label, prompt))
        return SimpleNamespace(ok=True, clean_output="完成", exit_code=0)

    def request_kill(self):
        self.killed = True
        return self.session_name


class A03RequirementsReviewTests(unittest.TestCase):
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

        with patch("A04_RequirementsReview.stdin_is_interactive", return_value=True), patch(
            "A04_RequirementsReview.prompt_review_max_rounds",
            return_value=7,
        ) as prompt_mock:
            value = resolve_review_max_rounds(args, progress=object())

        self.assertEqual(value, 7)
        prompt_mock.assert_called_once()

    def test_resume_ba_prompt_locks_scope_to_current_requirement_files(self):
        prompt = resume_ba(
            human_msg="",
            original_requirement_md="/tmp/基金数据生成器_原始需求.md",
            requirements_clear_md="/tmp/基金数据生成器_需求澄清.md",
            ask_human_md="/tmp/基金数据生成器_与人类交流.md",
            hitl_record_md="/tmp/基金数据生成器_人机交互澄清记录.md",
        )
        self.assertIn("/tmp/基金数据生成器_原始需求.md", prompt)
        self.assertIn("/tmp/基金数据生成器_需求澄清.md", prompt)
        self.assertIn("/tmp/基金数据生成器_人机交互澄清记录.md", prompt)
        self.assertIn("只允许回复 `准备完毕`", prompt)

    def test_shutdown_workers_preserves_ba_when_requested(self):
        ba_worker = _FakeWorker(runtime_root="/tmp/a04-runtime", runtime_dir="/tmp/a04-runtime/ba")
        reviewer_worker = _FakeWorker(runtime_root="/tmp/a04-runtime", runtime_dir="/tmp/a04-runtime/r1")
        handoff = RequirementsAnalystHandoff(
            worker=ba_worker,
            vendor="codex",
            model="gpt-5.4",
            reasoning_effort="high",
            proxy_url="",
        )
        reviewer = ReviewerRuntime(
            reviewer_name="R1",
            selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
            worker=reviewer_worker,
            review_md_path=Path("/tmp/review.md"),
            review_json_path=Path("/tmp/review.json"),
            contract=build_reviewer_completion_contract(
                requirement_name="需求A",
                reviewer_name="R1",
                review_md_path=Path("/tmp/review.md"),
                review_json_path=Path("/tmp/review.json"),
            ),
        )

        removed = _shutdown_workers(handoff, [reviewer], cleanup_runtime=False, preserve_ba_worker=True)

        self.assertEqual(removed, ())
        self.assertFalse(reviewer_worker.killed)
        self.assertFalse(ba_worker.killed)

    def test_build_ba_resume_result_contract_uses_file_contract_ready_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_requirements_review_paths(tmpdir, "需求A")
            contract = build_ba_resume_result_contract(paths)

        self.assertEqual(contract.expected_statuses, ("ready",))
        self.assertEqual(contract.mode, "a03_ba_resume")
        self.assertIn("requirements_clear", contract.optional_artifacts)

    def test_review_limit_hitl_does_not_gate_ba_turn_on_reviewer_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = build_requirements_review_paths(root, "需求A")
            ba_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(runtime_root=root / ".requirements_review_runtime", runtime_dir=root / ".requirements_review_runtime" / "ba"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            review_md_path, review_json_path = build_reviewer_artifact_paths(root, "需求A", "R1")
            reviewer = ReviewerRuntime(
                reviewer_name="R1",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(runtime_root=root / ".requirements_review_runtime", runtime_dir=root / ".requirements_review_runtime" / "r1"),
                review_md_path=review_md_path,
                review_json_path=review_json_path,
                contract=build_reviewer_completion_contract(
                    requirement_name="需求A",
                    reviewer_name="R1",
                    review_md_path=review_md_path,
                    review_json_path=review_json_path,
                ),
            )
            observed_reviewer_groups: list[list[ReviewerRuntime]] = []

            def fake_run_main_phase(main_owner, *, reviewers, run_phase, **kwargs):  # noqa: ANN001
                _ = run_phase, kwargs
                observed_reviewer_groups.append(list(reviewers))
                return SimpleNamespace(ok=True), list(reviewers), main_owner

            def fake_limit_cycle(*, initial_turn, human_reply_turn, **kwargs):  # noqa: ANN001
                _ = kwargs
                initial_turn()
                human_reply_turn("确认继续")
                return SimpleNamespace(post_hitl_continue_completed=True)

            with patch(
                "A03_RequirementsReview.run_main_phase_with_death_handling",
                side_effect=fake_run_main_phase,
            ), patch(
                "A03_RequirementsReview.run_review_limit_hitl_cycle",
                side_effect=fake_limit_cycle,
            ):
                result_handoff, result_reviewers, post_hitl_continue_completed = run_requirements_review_limit_hitl_loop(
                    ba_handoff,
                    reviewers=[reviewer],
                    paths=paths,
                    requirement_name="需求A",
                    review_msg="需要人工确认",
                    review_limit=1,
                    review_rounds_used=1,
                )

            self.assertIs(result_handoff, ba_handoff)
            self.assertEqual(result_reviewers, [reviewer])
            self.assertTrue(post_hitl_continue_completed)
            self.assertEqual(observed_reviewer_groups, [[], []])

    def test_create_reviewer_runtime_uses_worker_session_name_for_artifacts(self):
        import A04_RequirementsReview as review_module

        class FakeTmuxWorker(_FakeWorker):
            def __init__(self, *, runtime_root, **kwargs):  # noqa: ANN001
                super().__init__(runtime_root=runtime_root, runtime_dir=Path(runtime_root) / "worker")
                self.session_name = "审核器-鬼金羊"

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "A04_RequirementsReview.TmuxBatchWorker",
            FakeTmuxWorker,
        ), patch("sys.stdout", io.StringIO()):
            reviewer = create_reviewer_runtime(
                project_dir=tmpdir,
                requirement_name="需求A",
                reviewer_name="R1",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
            )

        self.assertEqual(reviewer.worker.session_name, "审核器-鬼金羊")
        self.assertTrue(reviewer.review_md_path.name.endswith("需求评审记录_审核器-鬼金羊.md"))
        self.assertTrue(reviewer.review_json_path.name.endswith("评审记录_审核器-鬼金羊.json"))

    def test_run_human_check_loop_exposes_requirements_clear_preview_path(self):
        import A04_RequirementsReview as review_module

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_requirements_review_paths(tmpdir, "需求A")
            paths["requirements_clear_path"].write_text("需求澄清正文\n", encoding="utf-8")
            handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            prompts: list[dict[str, object]] = []

            def fake_prompt(prompt_text, default=False, **kwargs):  # noqa: ANN001
                prompts.append(
                    {
                        "prompt_text": prompt_text,
                        **kwargs,
                    }
                )
                return False

            with patch("A04_RequirementsReview.prompt_yes_no_choice", side_effect=fake_prompt):
                result = run_human_check_loop(handoff=handoff, paths=paths, requirement_name="需求A")

        self.assertIs(result, handoff)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0]["prompt_text"], "是否向需求分析师提出建议或问题")
        self.assertEqual(prompts[1]["prompt_text"], "是否跳过需求评审阶段")
        self.assertEqual(prompts[0]["preview_path"], paths["requirements_clear_path"])
        self.assertEqual(prompts[1]["preview_path"], paths["requirements_clear_path"])
        self.assertEqual(prompts[0]["preview_title"], "需求澄清文档")
        self.assertEqual(prompts[1]["preview_title"], "需求澄清文档")
        self.assertFalse(prompts[0]["allow_back"])
        self.assertTrue(prompts[1]["allow_back"])
        self.assertEqual(prompts[0]["stage_key"], "requirements_review")
        self.assertEqual(prompts[1]["stage_key"], "requirements_review")
        self.assertEqual(prompts[0]["stage_step_index"], 0)
        self.assertEqual(prompts[1]["stage_step_index"], 1)

    def test_run_human_check_loop_auto_confirm_skips_feedback_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_requirements_review_paths(tmpdir, "需求A")
            paths["requirements_clear_path"].write_text("需求澄清正文\n", encoding="utf-8")
            handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )

            with patch("A04_RequirementsReview.prompt_yes_no_choice", side_effect=AssertionError("should not prompt")):
                result = run_human_check_loop(
                    handoff=handoff,
                    paths=paths,
                    requirement_name="需求A",
                    auto_confirm=True,
                )

        self.assertIs(result, handoff)

    def test_run_human_check_loop_back_from_skip_review_returns_to_human_question_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_requirements_review_paths(tmpdir, "需求A")
            paths["requirements_clear_path"].write_text("需求澄清正文\n", encoding="utf-8")
            handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            prompts: list[str] = []
            skip_calls = 0

            def fake_prompt(prompt_text, default=False, **kwargs):  # noqa: ANN001
                nonlocal skip_calls
                _ = default, kwargs
                prompts.append(prompt_text)
                if prompt_text == "是否跳过需求评审阶段":
                    skip_calls += 1
                    if skip_calls == 1:
                        raise PromptBackRequested()
                return False

            with patch("A04_RequirementsReview.prompt_yes_no_choice", side_effect=fake_prompt):
                result = run_human_check_loop(handoff=handoff, paths=paths, requirement_name="需求A")

        self.assertIs(result, handoff)
        self.assertEqual(
            prompts,
            [
                "是否向需求分析师提出建议或问题",
                "是否跳过需求评审阶段",
                "是否向需求分析师提出建议或问题",
                "是否跳过需求评审阶段",
            ],
        )

    def test_run_human_check_loop_skip_review_prompt_allows_back_under_bridge_ui(self):
        captured_requests: list[BridgePromptRequest] = []

        def emit_event(_event_type: str, _payload: dict[str, object]) -> None:
            return None

        def request_prompt(request: BridgePromptRequest) -> dict[str, object]:
            captured_requests.append(request)
            return {"value": "no"}

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_requirements_review_paths(tmpdir, "需求A")
            paths["requirements_clear_path"].write_text("需求澄清正文\n", encoding="utf-8")
            handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            with use_terminal_ui(BridgeTerminalUI(emit_event=emit_event, request_prompt=request_prompt)):
                result = run_human_check_loop(
                    handoff=handoff,
                    paths=paths,
                    requirement_name="需求A",
                    allow_previous_stage_back=True,
                )

        skip_requests = [
            request
            for request in captured_requests
            if request.payload.get("prompt_text") == "是否跳过需求评审阶段"
        ]
        self.assertIs(result, handoff)
        self.assertEqual(len(skip_requests), 1)
        self.assertEqual(skip_requests[0].prompt_type, "select")
        self.assertTrue(skip_requests[0].payload["allow_back"])
        self.assertEqual(skip_requests[0].payload["back_value"], PROMPT_BACK_VALUE)
        self.assertEqual(skip_requests[0].payload["stage_key"], "requirements_review")
        self.assertEqual(skip_requests[0].payload["stage_step_index"], 1)

    def test_run_human_check_loop_raises_skip_to_detailed_design_when_user_skips_review(self):
        import A04_RequirementsReview as review_module

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_requirements_review_paths(tmpdir, "需求A")
            handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )

            with patch(
                "A04_RequirementsReview.prompt_yes_no_choice",
                side_effect=[False, True],
            ):
                with self.assertRaises(review_module._SkipToDetailedDesign) as caught:
                    run_human_check_loop(handoff=handoff, paths=paths, requirement_name="需求A")

        self.assertIs(caught.exception.handoff, handoff)

    def test_run_requirements_review_stage_skip_does_not_resolve_review_max_rounds(self):
        import A04_RequirementsReview as review_module

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "需求A_原始需求.md").write_text("原始需求正文\n", encoding="utf-8")
            (root / "需求A_需求澄清.md").write_text("需求澄清正文\n", encoding="utf-8")
            ensure_pre_development_task_record(root, "需求A")
            handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(runtime_root=root / ".requirements_review_runtime", runtime_dir=root / ".requirements_review_runtime" / "ba"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )

            with patch(
                "A03_RequirementsReview.run_human_check_loop",
                side_effect=review_module._SkipToDetailedDesign(handoff),
            ), patch(
                "A03_RequirementsReview.resolve_review_max_rounds",
            ) as resolve_rounds_mock, patch(
                "A03_RequirementsReview.cleanup_stale_review_runtime_state",
            ) as cleanup_runtime_mock, patch(
                "A03_RequirementsReview._shutdown_workers",
                return_value=(),
            ):
                result = run_requirements_review_stage(
                    ["--project-dir", tmpdir, "--requirement-name", "需求A"],
                    ba_handoff=handoff,
                )

        self.assertFalse(result.passed)
        self.assertEqual(result.rounds_used, 0)
        resolve_rounds_mock.assert_not_called()
        cleanup_runtime_mock.assert_not_called()

    def test_run_requirements_review_stage_resolves_review_rounds_after_human_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "需求A_原始需求.md").write_text("原始需求正文\n", encoding="utf-8")
            (root / "需求A_需求澄清.md").write_text("需求澄清正文\n", encoding="utf-8")
            ensure_pre_development_task_record(root, "需求A")
            handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(runtime_root=root / ".requirements_review_runtime", runtime_dir=root / ".requirements_review_runtime" / "ba"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            calls: list[str] = []

            with patch(
                "A03_RequirementsReview.run_human_check_loop",
                side_effect=lambda handoff, paths, requirement_name=None, **kwargs: calls.append("human_check") or handoff,
            ), patch(
                "A03_RequirementsReview.resolve_review_max_rounds",
                side_effect=lambda *args, **kwargs: calls.append("review_max_rounds") or 5,
            ), patch(
                "A03_RequirementsReview.cleanup_stale_review_runtime_state",
                side_effect=lambda *args, **kwargs: calls.append("cleanup_runtime") or (),
            ), patch(
                "A03_RequirementsReview.cleanup_existing_review_artifacts",
                side_effect=lambda *args, **kwargs: calls.append("cleanup_artifacts") or (),
            ), patch(
                "A03_RequirementsReview.build_reviewer_workers",
                side_effect=lambda *args, **kwargs: calls.append("build_reviewers") or (_ for _ in ()).throw(RuntimeError("stop-after-order-check")),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop-after-order-check"):
                    run_requirements_review_stage(
                        ["--project-dir", tmpdir, "--requirement-name", "需求A"],
                        ba_handoff=handoff,
                    )

        self.assertEqual(
            calls,
            ["human_check", "review_max_rounds", "cleanup_runtime", "cleanup_artifacts", "build_reviewers"],
        )

    def test_prompt_yes_no_choice_suspends_and_resumes_progress(self):
        import A04_RequirementsReview as review_module

        class FakeSpinner:
            def __init__(self, *args, **kwargs):  # noqa: ANN001
                self.start_calls = 0
                self.stop_calls = 0

            def start(self):
                self.start_calls += 1

            def stop(self):
                self.stop_calls += 1

        created: list[FakeSpinner] = []

        def fake_spinner(*args, **kwargs):  # noqa: ANN001
            spinner = FakeSpinner()
            created.append(spinner)
            return spinner

        with patch("A04_RequirementsReview.SingleLineSpinnerMonitor", side_effect=fake_spinner), patch(
            "A04_RequirementsReview.terminal_prompt_yes_no",
            return_value=True,
        ):
            progress = review_module.ReviewStageProgress()
            progress.set_phase("等待人工确认")
            result = review_module.prompt_yes_no_choice("是否继续", False, progress=progress)

        self.assertTrue(result)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].start_calls, 2)
        self.assertEqual(created[0].stop_calls, 1)

    def test_parallel_reviewers_enter_round_progress_phase(self):
        import A04_RequirementsReview as review_module

        review_md_path = Path("/tmp/review.md")
        review_json_path = Path("/tmp/review.json")
        reviewer = ReviewerRuntime(
            reviewer_name="R1",
            selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
            worker=_FakeWorker(),
            review_md_path=review_md_path,
            review_json_path=review_json_path,
            contract=build_reviewer_completion_contract(
                requirement_name="需求A",
                reviewer_name="R1",
                review_md_path=review_md_path,
                review_json_path=review_json_path,
            ),
        )
        observed: dict[str, object] = {}

        class FakeProgress:
            def set_phase(self, phase: str, *, start: bool = True) -> None:
                observed["phase"] = phase
                observed["start"] = start

        def fake_run_reviewer_turn_with_recreation(reviewer_runtime, **kwargs):  # noqa: ANN001
            observed["label"] = kwargs["label"]
            observed["progress"] = kwargs["progress"]
            return reviewer_runtime

        with patch("A04_RequirementsReview.run_reviewer_turn_with_recreation", side_effect=fake_run_reviewer_turn_with_recreation):
            result = review_module._run_parallel_reviewers(
                [reviewer],
                project_dir="/tmp/project",
                requirement_name="需求A",
                round_index=2,
                prompt_builder=lambda item: f"prompt for {item.reviewer_name}",
                label_prefix="requirements_review_init",
                progress=FakeProgress(),
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(observed["phase"], "需求评审第 2 轮")
        self.assertEqual(observed["label"], "requirements_review_init_R1_round_2")
        self.assertIsNotNone(observed["progress"])

    def test_prepare_ba_handoff_reuses_live_worker(self):
        handoff = RequirementsAnalystHandoff(
            worker=_FakeWorker(),
            vendor="codex",
            model="gpt-5.4",
            reasoning_effort="high",
            proxy_url="",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_requirements_review_paths(tmpdir, "需求A")
            reused, cleanup_paths = prepare_ba_handoff(
                project_dir=tmpdir,
                requirement_name="需求A",
                ba_handoff=handoff,
                paths=paths,
            )
        self.assertIs(reused, handoff)
        self.assertEqual(cleanup_paths, ())

    def test_prompt_review_agent_selection_forwards_role_label_to_prompt_steps(self):
        import A04_RequirementsReview as review_module

        observed: dict[str, object] = {}

        def fake_prompt_vendor(default_vendor, *, role_label=""):  # noqa: ANN001
            observed["vendor_role_label"] = role_label
            return "codex"

        def fake_prompt_model(vendor, preferred_model, *, role_label=""):  # noqa: ANN001
            observed["model_role_label"] = role_label
            return "gpt-5.4"

        def fake_prompt_effort(vendor, model, default_effort, *, role_label=""):  # noqa: ANN001
            observed["effort_role_label"] = role_label
            return "high"

        def fake_prompt_proxy_url(default_proxy_url="", *, role_label=""):  # noqa: ANN001
            observed["proxy_role_label"] = role_label
            return ""

        with patch("A04_RequirementsReview.prompt_vendor", side_effect=fake_prompt_vendor), patch(
            "A04_RequirementsReview.prompt_model",
            side_effect=fake_prompt_model,
        ), patch(
            "A04_RequirementsReview.prompt_effort",
            side_effect=fake_prompt_effort,
        ), patch(
            "A04_RequirementsReview.prompt_proxy_url",
            side_effect=fake_prompt_proxy_url,
        ):
            selection = review_module.prompt_review_agent_selection(
                default_vendor="codex",
                role_label="需求分析师",
            )

        self.assertEqual(selection.vendor, "codex")
        self.assertEqual(selection.model, "gpt-5.4")
        self.assertEqual(selection.reasoning_effort, "high")
        self.assertEqual(observed["vendor_role_label"], "需求分析师")
        self.assertEqual(observed["model_role_label"], "需求分析师")
        self.assertEqual(observed["effort_role_label"], "需求分析师")
        self.assertEqual(observed["proxy_role_label"], "需求分析师")

    def test_create_review_ba_handoff_uses_predicted_constellation_role_label(self):
        import A04_RequirementsReview as review_module

        observed: dict[str, object] = {}
        rendered_messages: list[str] = []

        class FakeTmuxWorker(_FakeWorker):
            def __init__(self, *, runtime_root, **kwargs):  # noqa: ANN001
                super().__init__(runtime_root=runtime_root, runtime_dir=Path(runtime_root) / "ba-worker")
                self.session_name = "需求分析师-天佑星"

        def fake_prompt_review_agent_selection(default_vendor, default_model="", default_reasoning_effort="high", default_proxy_url="", *, role_label="", progress=None, **kwargs):  # noqa: ANN001
            observed["role_label"] = role_label
            return ReviewAgentSelection("codex", "gpt-5.4", "high", "")

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "A04_RequirementsReview.build_session_name",
            return_value="需求分析师-天佑星",
        ), patch(
            "A04_RequirementsReview.list_tmux_session_names",
            return_value=[],
        ), patch(
            "A04_RequirementsReview.list_registered_tmux_workers",
            return_value=[],
        ), patch(
            "A04_RequirementsReview.prompt_review_agent_selection",
            side_effect=fake_prompt_review_agent_selection,
        ), patch(
            "A04_RequirementsReview.TmuxBatchWorker",
            FakeTmuxWorker,
        ), patch(
            "A04_RequirementsReview.message",
            side_effect=rendered_messages.append,
        ):
            handoff = review_module._create_review_ba_handoff(
                project_dir=tmpdir,
                selection_title="进入需求评审阶段（需求分析师）",
            )

        self.assertEqual(observed["role_label"], "需求分析师-天佑星")
        self.assertEqual(handoff.worker.session_name, "需求分析师-天佑星")
        self.assertTrue(any("需求分析师-天佑星 已创建" in item for item in rendered_messages))

    def test_build_reviewer_workers_uses_predicted_constellation_role_labels(self):
        observed_role_labels: list[str] = []
        observed_occupied_sets: list[set[str]] = []

        def fake_prompt_review_agent_selection(default_vendor, default_model="", default_reasoning_effort="high", *, role_label="", progress=None, **kwargs):  # noqa: ANN001
            observed_role_labels.append(role_label)
            return ReviewAgentSelection("codex", "gpt-5.4", "high", "")

        def fake_create_reviewer_runtime(*, project_dir, requirement_name, reviewer_name, selection):  # noqa: ANN001
            runtime_root = Path(project_dir) / ".requirements_review_runtime"
            return ReviewerRuntime(
                reviewer_name=reviewer_name,
                selection=selection,
                worker=_FakeWorker(
                    runtime_root=runtime_root,
                    runtime_dir=runtime_root / reviewer_name.lower(),
                ),
                review_md_path=Path(project_dir) / f"{reviewer_name}.md",
                review_json_path=Path(project_dir) / f"{reviewer_name}.json",
                contract=SimpleNamespace(status_path=Path(project_dir) / f"{reviewer_name}.status.json"),
            )

        def fake_build_session_name(worker_id, work_dir, vendor, instance_id="", occupied_session_names=None):  # noqa: ANN001
            observed_occupied_sets.append({str(item).strip() for item in occupied_session_names or () if str(item).strip()})
            mapping = {
                "requirements-review-r1": "审核器-天平星",
                "requirements-review-r2": "审核器-地隐星",
            }
            return mapping[worker_id]

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "A04_RequirementsReview.prompt_positive_int",
            return_value=2,
        ), patch(
            "A04_RequirementsReview.build_session_name",
            side_effect=fake_build_session_name,
        ), patch(
            "A04_RequirementsReview.list_tmux_session_names",
            return_value=["审核器-天勇星"],
        ), patch(
            "A04_RequirementsReview.list_registered_tmux_workers",
            return_value=[],
        ), patch(
            "A04_RequirementsReview.prompt_review_agent_selection",
            side_effect=fake_prompt_review_agent_selection,
        ), patch(
            "A04_RequirementsReview.create_reviewer_runtime",
            side_effect=fake_create_reviewer_runtime,
        ), patch("sys.stdout", io.StringIO()):
            reviewers = build_reviewer_workers(project_dir=tmpdir, requirement_name="需求A")

        self.assertEqual([item.reviewer_name for item in reviewers], ["R1", "R2"])
        self.assertEqual(observed_role_labels, ["审核器-天平星", "审核器-地隐星"])
        self.assertTrue(any("审核器-天勇星" in occupied for occupied in observed_occupied_sets))

    def test_requirements_reviewer_count_prompt_allows_previous_step_back(self):
        captured_requests: list[BridgePromptRequest] = []

        def request_prompt(request: BridgePromptRequest) -> dict[str, object]:
            captured_requests.append(request)
            return {"value": PROMPT_BACK_VALUE}

        ui = BridgeTerminalUI(emit_event=lambda *_args, **_kwargs: None, request_prompt=request_prompt)

        with tempfile.TemporaryDirectory() as tmpdir:
            with use_terminal_ui(ui), self.assertRaises(PromptBackRequested):
                build_reviewer_workers(project_dir=tmpdir, requirement_name="需求A", allow_back_first_prompt=True)

        self.assertEqual(len(captured_requests), 1)
        self.assertEqual(captured_requests[0].prompt_type, "text")
        self.assertEqual(captured_requests[0].payload["prompt_text"], "请输入审核器数量")
        self.assertTrue(captured_requests[0].payload["allow_back"])
        self.assertEqual(captured_requests[0].payload["back_value"], PROMPT_BACK_VALUE)
        self.assertEqual(captured_requests[0].payload["stage_key"], "requirements_review")
        self.assertEqual(captured_requests[0].payload["stage_step_index"], 1)

    def test_prompt_review_agent_selection_accepts_back_kwargs(self):
        import A04_RequirementsReview as review_module

        expected = ReviewAgentSelection("codex", "gpt-5.4", "high", "")
        with patch(
            "A04_RequirementsReview.shared_review.prompt_review_agent_selection",
            return_value=expected,
        ) as prompt_mock:
            result = review_module.prompt_review_agent_selection(
                "codex",
                role_label="审核器-天慧星",
                allow_back_first_step=True,
                stage_key="requirements_review_reviewer_selection",
            )

        self.assertEqual(result, expected)
        self.assertTrue(prompt_mock.call_args.kwargs["allow_back_first_step"])
        self.assertEqual(prompt_mock.call_args.kwargs["stage_key"], "requirements_review_reviewer_selection")

    def test_prepare_ba_handoff_creates_new_ba_when_missing(self):
        created_workers: list[_FakeWorker] = []
        observed: list[tuple[str, str]] = []

        class FakeTmuxWorker(_FakeWorker):
            def __init__(self, *, runtime_root, **kwargs):  # noqa: ANN001
                super().__init__(runtime_root=runtime_root, runtime_dir=Path(runtime_root) / "ba-worker")
                created_workers.append(self)

        def fake_run_ba_turn_with_recreation(handoff, *, project_dir, label, prompt, result_contract):  # noqa: ANN001
            observed.append((label, prompt))
            self.assertEqual(result_contract.mode, "a03_ba_resume")
            return handoff, {"status": "ready"}

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "A03_RequirementsReview.TmuxBatchWorker",
            FakeTmuxWorker,
        ), patch(
            "A03_RequirementsReview.prompt_review_agent_selection",
            return_value=ReviewAgentSelection(
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            ),
        ), patch(
            "A03_RequirementsReview.run_ba_turn_with_recreation",
            side_effect=fake_run_ba_turn_with_recreation,
        ):
            paths = build_requirements_review_paths(tmpdir, "需求A")
            handoff, cleanup_paths = prepare_ba_handoff(
                project_dir=tmpdir,
                requirement_name="需求A",
                ba_handoff=None,
                paths=paths,
            )

        self.assertEqual(handoff.vendor, "codex")
        self.assertEqual(handoff.model, "gpt-5.4")
        self.assertEqual(handoff.reasoning_effort, "high")
        self.assertEqual(len(created_workers), 1)
        self.assertEqual(observed[0][0], "resume_requirements_review_ba")
        self.assertIn(str(paths["requirements_clear_path"].resolve()), observed[0][1])
        self.assertTrue(cleanup_paths)

    def test_run_human_check_loop_sends_feedback_until_user_enters_review(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_requirements_review_paths(tmpdir, "需求A")
            paths["ask_human_path"].write_text("", encoding="utf-8")
            created_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            call_order: list[str] = []
            observed: dict[str, object] = {}
            prompt_answers = iter([True, False, False])

            def fake_prompt(*args, **kwargs):  # noqa: ANN001
                call_order.append("prompt")
                return next(prompt_answers)

            def fake_create(*, project_dir, selection_title, progress=None):  # noqa: ANN001
                call_order.append("create")
                observed["selection_title"] = selection_title
                return created_handoff

            def fake_continue(*, handoff, paths, initial_prompt, label_prefix, progress=None):  # noqa: ANN001
                call_order.append("continuation")
                observed["handoff"] = handoff
                observed["initial_prompt"] = initial_prompt
                observed["label_prefix"] = label_prefix
                return handoff

            with patch(
                "A03_RequirementsReview.prompt_yes_no_choice",
                side_effect=fake_prompt,
            ), patch(
                "A03_RequirementsReview.collect_multiline_input",
                return_value="请补充边界条件",
            ), patch(
                "A03_RequirementsReview._create_review_ba_handoff",
                side_effect=fake_create,
            ), patch(
                "A03_RequirementsReview.resume_ba",
                return_value="RESUME_WITH_HUMAN_MSG",
            ), patch(
                "A03_RequirementsReview._run_review_clarification_continuation",
                side_effect=fake_continue,
            ), patch("sys.stdout", io.StringIO()):
                result = run_human_check_loop(handoff=None, paths=paths, requirement_name="需求A")

            self.assertIs(result, created_handoff)
            self.assertEqual(call_order, ["prompt", "create", "continuation", "prompt", "prompt"])
            self.assertEqual(observed["selection_title"], "按人类建议启动需求分析师")
            self.assertIs(observed["handoff"], created_handoff)
            self.assertEqual(observed["label_prefix"], "requirements_review_human_audit")
            self.assertEqual(observed["initial_prompt"], "RESUME_WITH_HUMAN_MSG")

    def test_run_human_check_loop_uses_human_feed_bck_when_ba_handoff_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_requirements_review_paths(tmpdir, "需求A")
            paths["ask_human_path"].write_text("", encoding="utf-8")
            existing_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            call_order: list[str] = []
            observed: dict[str, object] = {}
            prompt_answers = iter([True, False, False])

            def fake_prompt(*args, **kwargs):  # noqa: ANN001
                call_order.append("prompt")
                return next(prompt_answers)

            def fake_continue(*, handoff, paths, initial_prompt, label_prefix, progress=None):  # noqa: ANN001
                call_order.append("continuation")
                observed["handoff"] = handoff
                observed["initial_prompt"] = initial_prompt
                observed["label_prefix"] = label_prefix
                return handoff

            with patch(
                "A03_RequirementsReview.prompt_yes_no_choice",
                side_effect=fake_prompt,
            ), patch(
                "A03_RequirementsReview.collect_multiline_input",
                return_value="请补充边界条件",
            ), patch(
                "A03_RequirementsReview.human_feed_bck",
                return_value="HUMAN_FEED_BCK_PROMPT",
            ), patch(
                "A03_RequirementsReview.resume_ba",
                side_effect=AssertionError("已有需求分析师时不应走 resume_ba"),
            ), patch(
                "A03_RequirementsReview._run_review_clarification_continuation",
                side_effect=fake_continue,
            ), patch("sys.stdout", io.StringIO()):
                result = run_human_check_loop(handoff=existing_handoff, paths=paths, requirement_name="需求A")

            self.assertIs(result, existing_handoff)
            self.assertEqual(call_order, ["prompt", "continuation", "prompt", "prompt"])
            self.assertIs(observed["handoff"], existing_handoff)
            self.assertEqual(observed["label_prefix"], "requirements_review_human_audit")
            self.assertEqual(observed["initial_prompt"], "HUMAN_FEED_BCK_PROMPT")

    def test_run_human_check_loop_creates_ba_when_user_has_no_feedback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_requirements_review_paths(tmpdir, "需求A")
            created_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            observed: dict[str, object] = {}

            with patch(
                "A03_RequirementsReview.prompt_yes_no_choice",
                return_value=False,
            ), patch(
                "A03_RequirementsReview._create_review_ba_handoff",
                side_effect=lambda *, project_dir, selection_title, progress=None: observed.update(
                    {
                        "project_dir": Path(project_dir).resolve(),
                        "selection_title": selection_title,
                    }
                ) or created_handoff,
            ), patch("sys.stdout", io.StringIO()):
                result = run_human_check_loop(handoff=None, paths=paths, requirement_name="需求A")

            self.assertIs(result, created_handoff)
            self.assertEqual(observed["project_dir"], Path(tmpdir).resolve())
            self.assertEqual(observed["selection_title"], "进入需求评审阶段（需求分析师）")

    def test_run_review_clarification_continuation_uses_initial_prompt_then_hitl_bck(self):
        import A04_RequirementsReview as review_module
        from T05_hitl_runtime import HitlPromptContext

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_requirements_review_paths(tmpdir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("澄清结果\n", encoding="utf-8")
            paths["hitl_record_path"].write_text("", encoding="utf-8")
            handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(runtime_root=Path(tmpdir) / ".requirements_review_runtime", runtime_dir=Path(tmpdir) / ".requirements_review_runtime" / "ba"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            observed: dict[str, str] = {}

            def fake_run_hitl_agent_loop(**kwargs):  # noqa: ANN001
                context = HitlPromptContext(
                    stage_name="requirements_clarification",
                    hitl_round=1,
                    turn_id="review_turn_1",
                    turn_phase="requirements_clarification",
                    output_path=str(paths["requirements_clear_path"]),
                    question_path=str(paths["ask_human_path"]),
                    record_path=str(paths["hitl_record_path"]),
                    stage_status_path=str(handoff.worker.runtime_dir / "status.json"),
                    turn_status_path=str(handoff.worker.runtime_dir / "turn_status.json"),
                )
                observed["initial_prompt"] = kwargs["initial_prompt_builder"](context)
                observed["followup_prompt"] = kwargs["hitl_prompt_builder"]("需要补充的信息", context)
                return SimpleNamespace(decision=SimpleNamespace(payload={"status": "completed"}))

            with patch("A04_RequirementsReview.run_hitl_agent_loop", side_effect=fake_run_hitl_agent_loop):
                review_module._run_review_clarification_continuation(
                    handoff=handoff,
                    paths=paths,
                    initial_prompt="初始评审反馈提示词",
                    label_prefix="requirements_review_feedback_round_2",
                )

            self.assertEqual(observed["initial_prompt"], "初始评审反馈提示词")
            self.assertIn("需要补充的信息", observed["followup_prompt"])
            self.assertIn(str(paths["requirements_clear_path"].resolve()), observed["followup_prompt"])

    def test_collect_review_hitl_response_avoids_duplicate_log_under_bridge_ui(self):
        import A04_RequirementsReview as review_module

        with tempfile.TemporaryDirectory() as tmpdir:
            question_path = Path(tmpdir) / "贪吃蛇_与人类交流.md"
            question_path.write_text("请补充成绩记录规则。", encoding="utf-8")
            bridge_ui = BridgeTerminalUI(
                emit_event=lambda *_args, **_kwargs: None,
                request_prompt=lambda request: {"value": "补充如下"},
            )
            message_calls: list[tuple[object, ...]] = []

            with patch(
                "A04_RequirementsReview.get_terminal_ui",
                return_value=bridge_ui,
            ), patch(
                "A04_RequirementsReview.message",
                side_effect=lambda *args, **kwargs: message_calls.append(args),
            ), patch(
                "A04_RequirementsReview.collect_multiline_input",
                return_value="补充如下",
            ):
                result = review_module._collect_review_hitl_response(question_path, hitl_round=1)

        self.assertEqual(result, "补充如下")
        self.assertEqual(message_calls, [])

    def test_run_review_clarification_continuation_collects_hitl_response_when_record_file_is_empty(self):
        import A04_RequirementsReview as review_module

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_requirements_review_paths(tmpdir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("已有需求澄清\n", encoding="utf-8")
            paths["hitl_record_path"].write_text("", encoding="utf-8")
            handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(
                    runtime_root=Path(tmpdir) / ".requirements_review_runtime",
                    runtime_dir=Path(tmpdir) / ".requirements_review_runtime" / "ba",
                ),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            handoff.worker.ensure_agent_ready = lambda timeout_sec=60.0: None
            observed: dict[str, object] = {}

            def fake_run_turn(*, label, prompt, completion_contract, **kwargs):  # noqa: ANN001
                if len(observed) == 0:
                    observed["initial_prompt"] = prompt
                    paths["ask_human_path"].write_text("- [阻断] 需要补充业务边界\n", encoding="utf-8")
                    paths["hitl_record_path"].write_text("", encoding="utf-8")
                else:
                    observed["followup_prompt"] = prompt
                    paths["requirements_clear_path"].write_text("更新后的需求澄清\n", encoding="utf-8")
                    paths["ask_human_path"].write_text("", encoding="utf-8")
                    paths["hitl_record_path"].write_text("", encoding="utf-8")
                completion_contract.validator(completion_contract.status_path)
                return SimpleNamespace(ok=True, clean_output="", exit_code=0)

            human_inputs: list[str] = []

            with patch.object(handoff.worker, "run_turn", side_effect=fake_run_turn), patch(
                "A04_RequirementsReview.collect_multiline_input",
                side_effect=lambda **kwargs: human_inputs.append(kwargs["title"]) or "请补充业务边界",
            ), patch("sys.stdout", io.StringIO()):
                returned_handoff = review_module._run_review_clarification_continuation(
                    handoff=handoff,
                    paths=paths,
                    initial_prompt="初始评审反馈提示词",
                    label_prefix="requirements_review_feedback_round_2",
                )

        self.assertIs(returned_handoff, handoff)
        self.assertEqual(human_inputs, ["HITL 第 1 轮回复"])
        self.assertEqual(observed["initial_prompt"], "初始评审反馈提示词")
        self.assertIn("请补充业务边界", str(observed["followup_prompt"]))

    def test_run_review_clarification_continuation_allows_completed_result_with_non_empty_hitl_record(self):
        import A04_RequirementsReview as review_module

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_requirements_review_paths(tmpdir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("更新后的需求澄清\n", encoding="utf-8")
            paths["hitl_record_path"].write_text("- [已确认] 历史澄清事实\n", encoding="utf-8")
            handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(
                    runtime_root=Path(tmpdir) / ".requirements_review_runtime",
                    runtime_dir=Path(tmpdir) / ".requirements_review_runtime" / "ba",
                ),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )

            with patch(
                "A04_RequirementsReview.run_hitl_agent_loop",
                return_value=SimpleNamespace(decision=SimpleNamespace(payload={"status": "completed"})),
            ):
                returned_handoff = review_module._run_review_clarification_continuation(
                    handoff=handoff,
                    paths=paths,
                    initial_prompt="初始评审反馈提示词",
                    label_prefix="requirements_review_feedback_round_2",
                )

        self.assertIs(returned_handoff, handoff)

    def test_run_review_clarification_continuation_allows_idempotent_completed_first_round(self):
        import A04_RequirementsReview as review_module

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_requirements_review_paths(tmpdir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("已有需求澄清\n", encoding="utf-8")
            paths["hitl_record_path"].write_text("- [已确认] 历史澄清事实\n", encoding="utf-8")
            handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(
                    runtime_root=Path(tmpdir) / ".requirements_review_runtime",
                    runtime_dir=Path(tmpdir) / ".requirements_review_runtime" / "ba",
                ),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            handoff.worker.ensure_agent_ready = lambda timeout_sec=60.0: None
            observed: dict[str, object] = {}

            def fake_run_turn(*, label, prompt, completion_contract, **kwargs):  # noqa: ANN001
                observed["prompt"] = prompt
                completion_contract.validator(completion_contract.status_path)
                return SimpleNamespace(ok=True, clean_output="", exit_code=0)

            with patch.object(handoff.worker, "run_turn", side_effect=fake_run_turn), patch(
                "sys.stdout", io.StringIO()
            ):
                returned_handoff = review_module._run_review_clarification_continuation(
                    handoff=handoff,
                    paths=paths,
                    initial_prompt="初始评审反馈提示词",
                    label_prefix="requirements_review_human_audit",
                )

        self.assertIs(returned_handoff, handoff)
        self.assertEqual(observed["prompt"], "初始评审反馈提示词")

    def test_run_review_clarification_continuation_recovers_from_invalid_completed_status_with_pending_question(self):
        import A04_RequirementsReview as review_module

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_requirements_review_paths(tmpdir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("已有需求澄清\n", encoding="utf-8")
            paths["hitl_record_path"].write_text("", encoding="utf-8")
            handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(
                    runtime_root=Path(tmpdir) / ".requirements_review_runtime",
                    runtime_dir=Path(tmpdir) / ".requirements_review_runtime" / "ba",
                ),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            handoff.worker.ensure_agent_ready = lambda timeout_sec=60.0: None
            observed: dict[str, object] = {}
            stage_status_path = handoff.worker.runtime_dir / "requirements_review_feedback_round_2_clarification_status.json"

            def fake_run_turn(*, label, prompt, completion_contract, **kwargs):  # noqa: ANN001
                if len(observed) == 0:
                    observed["initial_prompt"] = prompt
                    paths["ask_human_path"].write_text("- [阻断] 需要补充业务边界\n", encoding="utf-8")
                    paths["hitl_record_path"].write_text("- [待确认] 平台边界\n", encoding="utf-8")
                    stage_status_path.write_text(
                        json.dumps(
                            {
                                "schema_version": "1.0",
                                "stage": review_module.REVIEW_CLARIFICATION_STAGE_NAME,
                                "turn_id": completion_contract.turn_id,
                                "hitl_round": 1,
                                "status": "completed",
                                "summary": "done",
                                "output_path": str(paths["requirements_clear_path"].resolve()),
                                "question_path": "",
                                "record_path": "",
                                "artifact_hashes": {
                                    str(paths["requirements_clear_path"].resolve()): build_prefixed_sha256(paths["requirements_clear_path"])
                                },
                                "written_at": "2026-04-17T12:00:00+08:00",
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                else:
                    observed["followup_prompt"] = prompt
                    paths["requirements_clear_path"].write_text("更新后的需求澄清\n", encoding="utf-8")
                    paths["ask_human_path"].write_text("", encoding="utf-8")
                    paths["hitl_record_path"].write_text("", encoding="utf-8")
                completion_contract.validator(completion_contract.status_path)
                return SimpleNamespace(ok=True, clean_output="", exit_code=0)

            human_inputs: list[str] = []

            with patch.object(handoff.worker, "run_turn", side_effect=fake_run_turn), patch(
                "A04_RequirementsReview.collect_multiline_input",
                side_effect=lambda **kwargs: human_inputs.append(kwargs["title"]) or "请补充业务边界",
            ), patch("sys.stdout", io.StringIO()):
                returned_handoff = review_module._run_review_clarification_continuation(
                    handoff=handoff,
                    paths=paths,
                    initial_prompt="初始评审反馈提示词",
                    label_prefix="requirements_review_feedback_round_2",
                )

        self.assertIs(returned_handoff, handoff)
        self.assertEqual(human_inputs, ["HITL 第 1 轮回复"])
        self.assertEqual(observed["initial_prompt"], "初始评审反馈提示词")
        self.assertIn("请补充业务边界", str(observed["followup_prompt"]))

    def test_run_review_clarification_continuation_waits_for_fresh_artifact_after_hitl_reply(self):
        import A04_RequirementsReview as review_module

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_requirements_review_paths(tmpdir, "需求A")
            paths["original_requirement_path"].write_text("原始需求\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("已有需求澄清\n", encoding="utf-8")
            paths["hitl_record_path"].write_text("", encoding="utf-8")
            handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(
                    runtime_root=Path(tmpdir) / ".requirements_review_runtime",
                    runtime_dir=Path(tmpdir) / ".requirements_review_runtime" / "ba",
                ),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            handoff.worker.ensure_agent_ready = lambda timeout_sec=60.0: None
            observed: dict[str, object] = {}

            def fake_run_turn(*, label, prompt, completion_contract, **kwargs):  # noqa: ANN001
                if "initial_prompt" not in observed:
                    observed["initial_prompt"] = prompt
                    paths["ask_human_path"].write_text("- [阻断] 需要补充业务边界\n", encoding="utf-8")
                    paths["hitl_record_path"].write_text("- [待确认] 业务边界\n", encoding="utf-8")
                    completion_contract.validator(completion_contract.status_path)
                    return SimpleNamespace(ok=True, clean_output="", exit_code=0)

                observed["followup_prompt"] = prompt
                with self.assertRaisesRegex(ValueError, "completed 状态未生成新的阶段产物"):
                    completion_contract.validator(completion_contract.status_path)
                observed["freshness_blocked"] = True
                paths["requirements_clear_path"].write_text("更新后的需求澄清\n", encoding="utf-8")
                paths["hitl_record_path"].write_text("- [已确认] 业务边界\n", encoding="utf-8")
                completion_contract.validator(completion_contract.status_path)
                return SimpleNamespace(ok=True, clean_output="", exit_code=0)

            human_inputs: list[str] = []

            with patch.object(handoff.worker, "run_turn", side_effect=fake_run_turn), patch(
                "A04_RequirementsReview.collect_multiline_input",
                side_effect=lambda **kwargs: human_inputs.append(kwargs["title"]) or "请补充业务边界",
            ), patch("sys.stdout", io.StringIO()):
                returned_handoff = review_module._run_review_clarification_continuation(
                    handoff=handoff,
                    paths=paths,
                    initial_prompt="初始评审反馈提示词",
                    label_prefix="requirements_review_feedback_round_2",
                )

        self.assertIs(returned_handoff, handoff)
        self.assertEqual(human_inputs, ["HITL 第 1 轮回复"])
        self.assertTrue(observed.get("freshness_blocked"))
        self.assertIn("请补充业务边界", str(observed["followup_prompt"]))

    def test_run_review_feedback_loop_uses_clarification_continuation_before_reviewer_reply(self):
        import A04_RequirementsReview as review_module

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_requirements_review_paths(tmpdir, "需求A")
            paths["merged_review_path"].write_text("仍有问题\n", encoding="utf-8")
            paths["requirements_clear_path"].write_text("需求澄清正文\n", encoding="utf-8")
            reviewer = ReviewerRuntime(
                reviewer_name="R1",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(),
                review_md_path=Path(tmpdir) / "需求A_需求评审记录_R1.md",
                review_json_path=Path(tmpdir) / "需求A_评审记录_R1.json",
                contract=SimpleNamespace(status_path=Path(tmpdir) / "需求A_评审记录_R1.json"),
            )
            handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            observed: dict[str, object] = {}

            def fake_continue(*, handoff, paths, initial_prompt, label_prefix, progress=None):  # noqa: ANN001
                observed["label_prefix"] = label_prefix
                observed["initial_prompt"] = initial_prompt
                paths["ba_feedback_path"].write_text("需求分析师反馈正文\n", encoding="utf-8")
                return handoff

            def fake_parallel(reviewers, *, project_dir, requirement_name, round_index, prompt_builder, label_prefix):  # noqa: ANN001
                observed["reply_prompt"] = prompt_builder(reviewers[0])
                return list(reviewers)

            with patch(
                "A03_RequirementsReview._run_review_clarification_continuation",
                side_effect=fake_continue,
            ), patch(
                "A03_RequirementsReview._run_parallel_reviewers",
                side_effect=fake_parallel,
            ), patch(
                "A03_RequirementsReview.repair_reviewer_outputs",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A03_RequirementsReview.collect_multiline_input",
                side_effect=AssertionError("review feedback loop should not use inline HITL input"),
            ):
                returned_handoff, returned_reviewers = review_module._run_review_feedback_loop(
                    handoff=handoff,
                    reviewers=[reviewer],
                    paths=paths,
                    requirement_name="需求A",
                    round_index=2,
                )

            self.assertIs(returned_handoff, handoff)
            self.assertEqual(returned_reviewers, [reviewer])
            self.assertEqual(observed["label_prefix"], "requirements_review_feedback_round_2")
            self.assertIn("仍有问题", str(observed["initial_prompt"]))
            self.assertIn("需求分析师反馈正文", str(observed["reply_prompt"]))

    def test_run_review_feedback_loop_prepares_ba_on_demand_when_missing(self):
        import A04_RequirementsReview as review_module

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_requirements_review_paths(tmpdir, "需求A")
            paths["merged_review_path"].write_text("仍有问题\n", encoding="utf-8")
            reviewer = ReviewerRuntime(
                reviewer_name="R1",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(),
                review_md_path=Path(tmpdir) / "需求A_需求评审记录_R1.md",
                review_json_path=Path(tmpdir) / "需求A_评审记录_R1.json",
                contract=SimpleNamespace(status_path=Path(tmpdir) / "需求A_评审记录_R1.json"),
            )
            created_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            observed: dict[str, object] = {}

            def fake_continue_with_feedback(**kwargs):  # noqa: ANN003
                kwargs["paths"]["ba_feedback_path"].write_text("需求分析师反馈正文\n", encoding="utf-8")
                return kwargs["handoff"]

            with patch(
                "A03_RequirementsReview.prepare_ba_handoff",
                return_value=(created_handoff, ()),
            ) as mocked_prepare, patch(
                "A03_RequirementsReview._run_review_clarification_continuation",
                side_effect=fake_continue_with_feedback,
            ), patch(
                "A03_RequirementsReview._run_parallel_reviewers",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A03_RequirementsReview.repair_reviewer_outputs",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ):
                returned_handoff, returned_reviewers = review_module._run_review_feedback_loop(
                    handoff=None,
                    reviewers=[reviewer],
                    paths=paths,
                    requirement_name="需求A",
                    round_index=2,
                )
                observed["prepare_calls"] = mocked_prepare.call_count

            self.assertIs(returned_handoff, created_handoff)
            self.assertEqual(returned_reviewers, [reviewer])
            self.assertEqual(observed["prepare_calls"], 1)

    def test_run_requirements_review_stage_does_not_start_ba_turn_before_first_reviewer_round_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "需求A_原始需求.md").write_text("原始需求正文\n", encoding="utf-8")
            (root / "需求A_需求澄清.md").write_text("需求澄清正文\n", encoding="utf-8")
            ensure_pre_development_task_record(root, "需求A")
            reviewer = ReviewerRuntime(
                reviewer_name="R1",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(runtime_root=root / ".requirements_review_runtime", runtime_dir=root / ".requirements_review_runtime" / "r1"),
                review_md_path=root / "需求A_需求评审记录_R1.md",
                review_json_path=root / "需求A_评审记录_R1.json",
                contract=SimpleNamespace(status_path=root / "需求A_评审记录_R1.json"),
            )

            def fake_parallel_reviewers(reviewers, *, project_dir, requirement_name, round_index, prompt_builder, label_prefix):  # noqa: ANN001
                reviewer_path = reviewer.contract.status_path
                reviewer_path.write_text(
                    json.dumps([{"task_name": REQUIREMENTS_REVIEW_TASK_NAME, "review_pass": True}], ensure_ascii=False),
                    encoding="utf-8",
                )
                reviewer.review_md_path.write_text("", encoding="utf-8")
                observed_prompts.append(prompt_builder(reviewers[0]))
                return list(reviewers)

            created_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            observed: dict[str, object] = {}
            observed_prompts: list[str] = []
            with patch(
                "A03_RequirementsReview.prompt_yes_no_choice",
                return_value=False,
            ), patch(
                "A03_RequirementsReview._create_review_ba_handoff",
                side_effect=lambda *, project_dir, selection_title, progress=None: observed.update(
                    {
                        "project_dir": Path(project_dir).resolve(),
                        "selection_title": selection_title,
                    }
                ) or created_handoff,
            ), patch(
                "A03_RequirementsReview.build_reviewer_workers",
                side_effect=lambda **kwargs: [reviewer],
            ), patch(
                "A03_RequirementsReview._run_parallel_reviewers",
                side_effect=fake_parallel_reviewers,
            ), patch(
                "A03_RequirementsReview.repair_reviewer_outputs",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A03_RequirementsReview._shutdown_workers",
                return_value=(),
            ):
                result = run_requirements_review_stage(["--project-dir", tmpdir, "--requirement-name", "需求A"])

            self.assertTrue(result.passed)
            self.assertEqual(observed["project_dir"], root.resolve())
            self.assertEqual(observed["selection_title"], "进入需求评审阶段（需求分析师）")
            self.assertEqual(len(observed_prompts), 1)
            self.assertNotIn("Task Routing Assessment", observed_prompts[0])
            self.assertNotIn("Output the Task Routing Assessment first", observed_prompts[0])

    def test_repair_reviewer_outputs_repairs_single_reviewer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            review_md_path, review_json_path = build_reviewer_artifact_paths(tmpdir, "需求A", "R1")
            review_md_path.write_text("待修复问题\n", encoding="utf-8")
            review_json_path.write_text("[]", encoding="utf-8")
            reviewer = ReviewerRuntime(
                reviewer_name="R1",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(),
                review_md_path=review_md_path,
                review_json_path=review_json_path,
                contract=build_reviewer_completion_contract(
                    requirement_name="需求A",
                    reviewer_name="R1",
                    review_md_path=review_md_path,
                    review_json_path=review_json_path,
                ),
            )
            observed_labels: list[str] = []

            def fake_check_reviewer_job(*args, **kwargs):  # noqa: ANN001
                if len(observed_labels) == 0:
                    return {"demo-session": "请补齐 JSON"}
                return {}

            def fake_run_reviewer_turn(reviewer_runtime, *, label, prompt):  # noqa: ANN001
                observed_labels.append(label)
                review_md_path.write_text("", encoding="utf-8")
                review_json_path.write_text(
                    json.dumps([{"task_name": REQUIREMENTS_REVIEW_TASK_NAME, "review_pass": True}], ensure_ascii=False),
                    encoding="utf-8",
                )

            with patch("A03_RequirementsReview.check_reviewer_job", side_effect=fake_check_reviewer_job), patch(
                "A03_RequirementsReview._run_reviewer_turn",
                side_effect=fake_run_reviewer_turn,
            ):
                repair_reviewer_outputs(
                    [reviewer],
                    project_dir=tmpdir,
                    requirement_name="需求A",
                    round_index=1,
                )

            self.assertEqual(len(observed_labels), 1)
            self.assertIn("requirements_review_fix_R1_round_1_attempt_1", observed_labels[0])

    def test_run_requirements_review_stage_passes_and_marks_pre_development_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "需求A_原始需求.md").write_text("原始需求正文\n", encoding="utf-8")
            (root / "需求A_需求澄清.md").write_text("需求澄清正文\n", encoding="utf-8")
            ensure_pre_development_task_record(root, "需求A")
            review_md_path, review_json_path = build_reviewer_artifact_paths(root, "需求A", "R1")
            review_md_path.write_text("", encoding="utf-8")
            review_json_path.write_text("[]", encoding="utf-8")
            reviewer = ReviewerRuntime(
                reviewer_name="R1",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(runtime_root=root / ".requirements_review_runtime", runtime_dir=root / ".requirements_review_runtime" / "r1"),
                review_md_path=review_md_path,
                review_json_path=review_json_path,
                contract=build_reviewer_completion_contract(
                    requirement_name="需求A",
                    reviewer_name="R1",
                    review_md_path=review_md_path,
                    review_json_path=review_json_path,
                ),
            )

            def fake_parallel_reviewers(reviewers, *, project_dir, requirement_name, round_index, prompt_builder, label_prefix):  # noqa: ANN001
                self.assertEqual(round_index, 1)
                _ = prompt_builder(reviewers[0])
                review_json_path.write_text(
                    json.dumps([{"task_name": REQUIREMENTS_REVIEW_TASK_NAME, "review_pass": True}], ensure_ascii=False),
                    encoding="utf-8",
                )
                review_md_path.write_text("", encoding="utf-8")
                return list(reviewers)

            cleanup_observed: dict[str, object] = {}

            def fake_cleanup_stale_review_runtime_state(project_dir, requirement_name, *, preserve_workers=()):  # noqa: ANN001
                cleanup_observed["project_dir"] = Path(project_dir).resolve()
                cleanup_observed["requirement_name"] = str(requirement_name)
                cleanup_observed["preserve_workers"] = list(preserve_workers)
                return ()

            with patch(
                "A03_RequirementsReview.prepare_ba_handoff",
                return_value=(
                    RequirementsAnalystHandoff(
                        worker=_FakeWorker(),
                        vendor="codex",
                        model="gpt-5.4",
                        reasoning_effort="high",
                        proxy_url="",
                    ),
                    (),
                ),
            ), patch(
                "A03_RequirementsReview.run_human_check_loop",
                side_effect=lambda handoff, paths, requirement_name=None, **kwargs: handoff,
            ), patch(
                "A03_RequirementsReview.build_reviewer_workers",
                return_value=[reviewer],
            ), patch(
                "A03_RequirementsReview._run_parallel_reviewers",
                side_effect=fake_parallel_reviewers,
            ), patch(
                "A03_RequirementsReview.repair_reviewer_outputs",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A03_RequirementsReview.cleanup_stale_review_runtime_state",
                side_effect=fake_cleanup_stale_review_runtime_state,
            ), patch(
                "A03_RequirementsReview._shutdown_workers",
                return_value=(),
            ):
                result = run_requirements_review_stage(
                    ["--project-dir", tmpdir, "--requirement-name", "需求A"],
                    ba_handoff=RequirementsAnalystHandoff(
                        worker=_FakeWorker(),
                        vendor="codex",
                        model="gpt-5.4",
                        reasoning_effort="high",
                        proxy_url="",
                    ),
                )

            pre_dev_path = build_pre_development_task_record_path(root, "需求A")
            pre_dev_payload = json.loads(pre_dev_path.read_text(encoding="utf-8"))
            self.assertTrue(result.passed)
            self.assertEqual(result.rounds_used, 1)
            self.assertTrue(pre_dev_payload["需求评审"]["需求评审"])
            self.assertEqual(cleanup_observed["project_dir"], root.resolve())
            self.assertEqual(cleanup_observed["requirement_name"], "需求A")
            self.assertEqual(cleanup_observed["preserve_workers"], [])

    def test_run_requirements_review_stage_loops_once_after_failed_review(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "需求A_原始需求.md").write_text("原始需求正文\n", encoding="utf-8")
            (root / "需求A_需求澄清.md").write_text("需求澄清正文\n", encoding="utf-8")
            ensure_pre_development_task_record(root, "需求A")
            review_md_path, review_json_path = build_reviewer_artifact_paths(root, "需求A", "R1")
            review_md_path.write_text("", encoding="utf-8")
            review_json_path.write_text("[]", encoding="utf-8")
            reviewer = ReviewerRuntime(
                reviewer_name="R1",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(runtime_root=root / ".requirements_review_runtime", runtime_dir=root / ".requirements_review_runtime" / "r1"),
                review_md_path=review_md_path,
                review_json_path=review_json_path,
                contract=build_reviewer_completion_contract(
                    requirement_name="需求A",
                    reviewer_name="R1",
                    review_md_path=review_md_path,
                    review_json_path=review_json_path,
                ),
            )
            round_counter = {"count": 0}

            def fake_parallel_reviewers(reviewers, *, project_dir, requirement_name, round_index, prompt_builder, label_prefix):  # noqa: ANN001
                round_counter["count"] += 1
                _ = prompt_builder(reviewers[0])
                review_json_path.write_text(
                    json.dumps([{"task_name": REQUIREMENTS_REVIEW_TASK_NAME, "review_pass": False}], ensure_ascii=False),
                    encoding="utf-8",
                )
                review_md_path.write_text("仍有问题\n", encoding="utf-8")
                return list(reviewers)

            def fake_review_feedback_loop(**kwargs):  # noqa: ANN001
                self.assertIs(kwargs["handoff"], active_handoff)
                review_json_path.write_text(
                    json.dumps([{"task_name": REQUIREMENTS_REVIEW_TASK_NAME, "review_pass": True}], ensure_ascii=False),
                    encoding="utf-8",
                )
                review_md_path.write_text("", encoding="utf-8")

            active_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            observed: dict[str, object] = {}

            with patch(
                "A03_RequirementsReview.prepare_ba_handoff",
                return_value=(
                    RequirementsAnalystHandoff(
                        worker=_FakeWorker(),
                        vendor="codex",
                        model="gpt-5.4",
                        reasoning_effort="high",
                        proxy_url="",
                    ),
                    (),
                ),
            ), patch(
                "A03_RequirementsReview.run_human_check_loop",
                side_effect=lambda handoff, paths, requirement_name=None, **kwargs: handoff,
            ), patch(
                "A03_RequirementsReview.build_reviewer_workers",
                return_value=[reviewer],
            ), patch(
                "A03_RequirementsReview._run_parallel_reviewers",
                side_effect=fake_parallel_reviewers,
            ), patch(
                "A03_RequirementsReview.repair_reviewer_outputs",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A03_RequirementsReview._run_review_feedback_loop",
                side_effect=lambda **kwargs: (kwargs["handoff"], list(kwargs["reviewers"])) if not fake_review_feedback_loop(**kwargs) else None,
            ), patch(
                "A03_RequirementsReview._shutdown_workers",
                return_value=(),
            ):
                result = run_requirements_review_stage(
                    ["--project-dir", tmpdir, "--requirement-name", "需求A"],
                    ba_handoff=active_handoff,
                )

            self.assertTrue(result.passed)
            self.assertEqual(result.rounds_used, 2)
            self.assertEqual(round_counter["count"], 1)

    def test_run_requirements_review_stage_preserves_ba_handoff_after_first_feedback_round(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "需求A_原始需求.md").write_text("原始需求正文\n", encoding="utf-8")
            (root / "需求A_需求澄清.md").write_text("需求澄清正文\n", encoding="utf-8")
            ensure_pre_development_task_record(root, "需求A")
            review_md_path, review_json_path = build_reviewer_artifact_paths(root, "需求A", "R1")
            review_md_path.write_text("", encoding="utf-8")
            review_json_path.write_text("[]", encoding="utf-8")
            reviewer = ReviewerRuntime(
                reviewer_name="R1",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(runtime_root=root / ".requirements_review_runtime", runtime_dir=root / ".requirements_review_runtime" / "r1"),
                review_md_path=review_md_path,
                review_json_path=review_json_path,
                contract=build_reviewer_completion_contract(
                    requirement_name="需求A",
                    reviewer_name="R1",
                    review_md_path=review_md_path,
                    review_json_path=review_json_path,
                ),
            )

            def fake_parallel_reviewers(reviewers, *, project_dir, requirement_name, round_index, prompt_builder, label_prefix):  # noqa: ANN001
                self.assertEqual(round_index, 1)
                _ = prompt_builder(reviewers[0])
                review_json_path.write_text(
                    json.dumps([{"task_name": REQUIREMENTS_REVIEW_TASK_NAME, "review_pass": False}], ensure_ascii=False),
                    encoding="utf-8",
                )
                review_md_path.write_text("仍有问题\n", encoding="utf-8")
                return list(reviewers)

            active_handoff = RequirementsAnalystHandoff(
                worker=_FakeWorker(),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            observed: dict[str, object] = {
                "feedback_handoffs": [],
            }
            feedback_round = {"count": 0}

            def fake_review_feedback_loop(**kwargs):  # noqa: ANN001
                feedback_round["count"] += 1
                observed["feedback_handoffs"].append(kwargs["handoff"])
                if feedback_round["count"] == 1:
                    review_json_path.write_text(
                        json.dumps([{"task_name": REQUIREMENTS_REVIEW_TASK_NAME, "review_pass": False}], ensure_ascii=False),
                        encoding="utf-8",
                    )
                    review_md_path.write_text("仍有问题\n", encoding="utf-8")
                else:
                    review_json_path.write_text(
                        json.dumps([{"task_name": REQUIREMENTS_REVIEW_TASK_NAME, "review_pass": True}], ensure_ascii=False),
                        encoding="utf-8",
                    )
                    review_md_path.write_text("", encoding="utf-8")
                return kwargs["handoff"], list(kwargs["reviewers"])

            with patch(
                "A03_RequirementsReview.run_human_check_loop",
                side_effect=lambda handoff, paths, requirement_name=None, **kwargs: handoff,
            ), patch(
                "A03_RequirementsReview.build_reviewer_workers",
                return_value=[reviewer],
            ), patch(
                "A03_RequirementsReview._run_parallel_reviewers",
                side_effect=fake_parallel_reviewers,
            ), patch(
                "A03_RequirementsReview.repair_reviewer_outputs",
                side_effect=lambda reviewers, **kwargs: list(reviewers),
            ), patch(
                "A03_RequirementsReview._run_review_feedback_loop",
                side_effect=fake_review_feedback_loop,
            ), patch(
                "A03_RequirementsReview._shutdown_workers",
                return_value=(),
            ):
                result = run_requirements_review_stage(
                    ["--project-dir", tmpdir, "--requirement-name", "需求A"],
                    ba_handoff=active_handoff,
                )

            self.assertTrue(result.passed)
            self.assertEqual(result.rounds_used, 3)
            self.assertEqual(observed["feedback_handoffs"], [active_handoff, active_handoff])

    def test_run_requirements_review_stage_failure_keeps_review_status_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "需求A_原始需求.md").write_text("原始需求正文\n", encoding="utf-8")
            (root / "需求A_需求澄清.md").write_text("需求澄清正文\n", encoding="utf-8")
            ensure_pre_development_task_record(root, "需求A")
            reviewer = ReviewerRuntime(
                reviewer_name="R1",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(runtime_root=root / ".requirements_review_runtime", runtime_dir=root / ".requirements_review_runtime" / "r1"),
                review_md_path=root / "需求A_需求评审记录_R1.md",
                review_json_path=root / "需求A_评审记录_R1.json",
                contract=SimpleNamespace(status_path=root / "需求A_评审记录_R1.json"),
            )

            with patch(
                "A03_RequirementsReview.prepare_ba_handoff",
                return_value=(
                    RequirementsAnalystHandoff(
                        worker=_FakeWorker(),
                        vendor="codex",
                        model="gpt-5.4",
                        reasoning_effort="high",
                        proxy_url="",
                    ),
                    (),
                ),
            ), patch(
                "A03_RequirementsReview.run_human_check_loop",
                side_effect=lambda handoff, paths, requirement_name=None, **kwargs: handoff,
            ), patch(
                "A03_RequirementsReview.build_reviewer_workers",
                return_value=[reviewer],
            ), patch(
                "A03_RequirementsReview._run_parallel_reviewers",
                side_effect=RuntimeError("reviewer exploded"),
            ), patch(
                "A03_RequirementsReview._shutdown_workers",
                return_value=(),
            ):
                with self.assertRaises(RuntimeError):
                    run_requirements_review_stage(
                        ["--project-dir", tmpdir, "--requirement-name", "需求A"],
                        ba_handoff=RequirementsAnalystHandoff(
                            worker=_FakeWorker(),
                            vendor="codex",
                            model="gpt-5.4",
                            reasoning_effort="high",
                            proxy_url="",
                        ),
                    )

            pre_dev_path = build_pre_development_task_record_path(root, "需求A")
            pre_dev_payload = json.loads(pre_dev_path.read_text(encoding="utf-8"))
            self.assertFalse(pre_dev_payload["需求评审"]["需求评审"])

    def test_run_requirements_review_stage_can_skip_review_and_continue_to_a05(self):
        import A04_RequirementsReview as review_module

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "需求A_原始需求.md").write_text("原始需求正文\n", encoding="utf-8")
            (root / "需求A_需求澄清.md").write_text("需求澄清正文\n", encoding="utf-8")
            ensure_pre_development_task_record(root, "需求A")
            live_ba = RequirementsAnalystHandoff(
                worker=_FakeWorker(
                    runtime_root=root / ".requirements_clarification_runtime",
                    runtime_dir=root / ".requirements_clarification_runtime" / "ba",
                ),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            observed: dict[str, object] = {}

            def fake_shutdown(ba_handoff, reviewers, *, cleanup_runtime, preserve_ba_worker=False):  # noqa: ANN001
                observed["ba_handoff"] = ba_handoff
                observed["reviewers"] = list(reviewers)
                observed["cleanup_runtime"] = cleanup_runtime
                observed["preserve_ba_worker"] = preserve_ba_worker
                return ("worker-cleanup",)

            with patch(
                "A03_RequirementsReview.run_human_check_loop",
                side_effect=review_module._SkipToDetailedDesign(live_ba),
            ), patch(
                "A03_RequirementsReview.build_reviewer_workers",
                side_effect=AssertionError("skip review 不应创建 reviewer"),
            ), patch(
                "A03_RequirementsReview._run_parallel_reviewers",
                side_effect=AssertionError("skip review 不应执行 reviewer"),
            ), patch(
                "A03_RequirementsReview.cleanup_existing_review_artifacts",
                return_value=("artifact-cleanup",),
            ), patch(
                "A03_RequirementsReview._shutdown_workers",
                side_effect=fake_shutdown,
            ):
                result = run_requirements_review_stage(
                    ["--project-dir", tmpdir, "--requirement-name", "需求A"],
                    ba_handoff=live_ba,
                    preserve_ba_worker=True,
                )

            pre_dev_path = build_pre_development_task_record_path(root, "需求A")
            pre_dev_payload = json.loads(pre_dev_path.read_text(encoding="utf-8"))
            self.assertFalse(result.passed)
            self.assertEqual(result.rounds_used, 0)
            self.assertIsNone(result.ba_handoff)
            self.assertEqual(result.cleanup_paths, ("artifact-cleanup", "worker-cleanup"))
            self.assertIs(observed["ba_handoff"], live_ba)
            self.assertEqual(observed["reviewers"], [])
            self.assertTrue(observed["cleanup_runtime"])
            self.assertFalse(observed["preserve_ba_worker"])
            self.assertFalse(pre_dev_payload["需求评审"]["需求评审"])

    def test_cleanup_existing_review_artifacts_removes_stale_files_for_same_requirement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = build_requirements_review_paths(root, "需求A")
            stale_json = root / "需求A_评审记录_R2.json"
            stale_md = root / "需求A_需求评审记录_R2.md"
            stale_merged = root / "需求A_需求评审记录.md"
            stale_feedback = root / "需求A_需求分析师反馈.md"
            other_requirement_json = root / "需求B_评审记录_R1.json"
            for target in (stale_json, stale_md, stale_merged, stale_feedback, other_requirement_json):
                target.write_text("stale\n", encoding="utf-8")

            removed = cleanup_existing_review_artifacts(paths, "需求A")

            self.assertFalse(stale_json.exists())
            self.assertFalse(stale_md.exists())
            self.assertFalse(stale_merged.exists())
            self.assertFalse(stale_feedback.exists())
            self.assertTrue(other_requirement_json.exists())
            self.assertTrue(removed)

    def test_cleanup_stale_review_runtime_state_scopes_by_requirement_and_preserves_live_legacy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_root = root / ".requirements_review_runtime"
            target_dir = runtime_root / "target-reviewer"
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "审核器-R3-天哭星",
                        "project_dir": str(root.resolve()),
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a04.start",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            other_requirement_dir = runtime_root / "other-requirement-reviewer"
            other_requirement_dir.mkdir(parents=True, exist_ok=True)
            (other_requirement_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "审核器-R4-天牢星",
                        "project_dir": str(root.resolve()),
                        "requirement_name": "需求B",
                        "workflow_action": "stage.a04.start",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            legacy_live_dir = runtime_root / "legacy-live-reviewer"
            legacy_live_dir.mkdir(parents=True, exist_ok=True)
            (legacy_live_dir / "worker.state.json").write_text(
                json.dumps({"session_name": "遗留-存活会话"}, ensure_ascii=False),
                encoding="utf-8",
            )
            legacy_dead_dir = runtime_root / "legacy-dead-reviewer"
            legacy_dead_dir.mkdir(parents=True, exist_ok=True)
            (legacy_dead_dir / "worker.state.json").write_text(
                json.dumps({"session_name": "遗留-死亡会话", "agent_state": "DEAD"}, ensure_ascii=False),
                encoding="utf-8",
            )
            preserved_dir = runtime_root / "preserved-reviewer"
            preserved_dir.mkdir(parents=True, exist_ok=True)
            (preserved_dir / "worker.state.json").write_text(
                json.dumps(
                    {
                        "session_name": "需求分析师-天佑星",
                        "project_dir": str(root.resolve()),
                        "requirement_name": "需求A",
                        "workflow_action": "stage.a04.start",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            preserved_worker = _FakeWorker(runtime_root=runtime_root, runtime_dir=preserved_dir)
            preserved_worker.session_name = "需求分析师-天佑星"
            killed_sessions: list[str] = []

            class FakeTmuxRuntimeController:
                def session_exists(self, session_name: str) -> bool:
                    return session_name in {"遗留-存活会话", "审核器-R4-天牢星", "需求分析师-天佑星"}

                def kill_session(self, session_name: str, *, missing_ok: bool = True):  # noqa: ANN001
                    killed_sessions.append(session_name)
                    return session_name

            with patch("tmux_core.stage_kernel.runtime_scope_cleanup.TmuxRuntimeController", FakeTmuxRuntimeController):
                removed = cleanup_stale_review_runtime_state(root, "需求A", preserve_workers=(preserved_worker,))

            self.assertFalse(target_dir.exists())
            self.assertFalse(legacy_dead_dir.exists())
            self.assertTrue(other_requirement_dir.exists())
            self.assertTrue(legacy_live_dir.exists())
            self.assertTrue(preserved_dir.exists())
            self.assertIn("审核器-R3-天哭星", killed_sessions)
            self.assertIn("遗留-死亡会话", killed_sessions)
            self.assertIn(str(target_dir.resolve()), removed)
            self.assertIn(str(legacy_dead_dir.resolve()), removed)

    def test_run_reviewer_turn_with_recreation_drops_dead_reviewer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            review_md_path, review_json_path = build_reviewer_artifact_paths(root, "需求A", "R1")
            review_md_path.write_text("", encoding="utf-8")
            review_json_path.write_text("[]", encoding="utf-8")
            original = ReviewerRuntime(
                reviewer_name="R1",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(runtime_root=root / ".requirements_review_runtime", runtime_dir=root / ".requirements_review_runtime" / "old"),
                review_md_path=review_md_path,
                review_json_path=review_json_path,
                contract=build_reviewer_completion_contract(
                    requirement_name="需求A",
                    reviewer_name="R1",
                    review_md_path=review_md_path,
                    review_json_path=review_json_path,
                ),
            )
            recreated = ReviewerRuntime(
                reviewer_name="R1",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(runtime_root=root / ".requirements_review_runtime", runtime_dir=root / ".requirements_review_runtime" / "new"),
                review_md_path=review_md_path,
                review_json_path=review_json_path,
                contract=build_reviewer_completion_contract(
                    requirement_name="需求A",
                    reviewer_name="R1",
                    review_md_path=review_md_path,
                    review_json_path=review_json_path,
                ),
            )
            call_counter = {"count": 0}

            def fake_run_reviewer_turn(reviewer, *, label, prompt):  # noqa: ANN001
                call_counter["count"] += 1
                if call_counter["count"] == 1:
                    raise RuntimeError("tmux pane died while waiting for turn artifacts")
                return None

            with patch(
                "A03_RequirementsReview._run_reviewer_turn",
                side_effect=fake_run_reviewer_turn,
            ), patch(
                "A03_RequirementsReview.recreate_reviewer_runtime",
                return_value=recreated,
            ) as recreate_mock:
                result = run_reviewer_turn_with_recreation(
                    original,
                    project_dir=root,
                    requirement_name="需求A",
                    label="review",
                    prompt="do review",
                )

            self.assertIs(result, recreated)
            self.assertEqual(call_counter["count"], 2)
            recreate_mock.assert_called_once()

    def test_run_reviewer_turn_with_recreation_returns_current_reviewer_on_contract_violation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            review_md_path, review_json_path = build_reviewer_artifact_paths(root, "需求A", "R1")
            review_md_path.write_text("", encoding="utf-8")
            review_json_path.write_text("[]", encoding="utf-8")
            reviewer = ReviewerRuntime(
                reviewer_name="R1",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(
                    runtime_root=root / ".requirements_review_runtime",
                    runtime_dir=root / ".requirements_review_runtime" / "worker",
                ),
                review_md_path=review_md_path,
                review_json_path=review_json_path,
                contract=build_reviewer_completion_contract(
                    requirement_name="需求A",
                    reviewer_name="R1",
                    review_md_path=review_md_path,
                    review_json_path=review_json_path,
                ),
            )

            with patch(
                "A03_RequirementsReview._run_reviewer_turn",
                side_effect=RuntimeError(
                    "turn artifacts contract violation after task completion: "
                    "phase=需求评审 status_path=/tmp/review.json error=审核器未通过，但评审 markdown 为空"
                ),
            ), patch(
                "A03_RequirementsReview.recreate_reviewer_runtime",
                side_effect=AssertionError("contract violation should enter repair loop instead of rebuilding reviewer"),
            ):
                result = run_reviewer_turn_with_recreation(
                    reviewer,
                    project_dir=root,
                    requirement_name="需求A",
                    label="review",
                    prompt="do review",
                )

            self.assertIs(result, reviewer)

    def test_run_reviewer_turn_with_recreation_rebuilds_auth_failed_reviewer_with_new_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            review_md_path, review_json_path = build_reviewer_artifact_paths(root, "需求A", "R1")
            review_md_path.write_text("", encoding="utf-8")
            review_json_path.write_text("[]", encoding="utf-8")

            class AuthFailedWorker(_FakeWorker):
                def read_state(self):
                    return {
                        "health_status": "provider_auth_error",
                        "health_note": "provider_auth_error",
                    }

            original = ReviewerRuntime(
                reviewer_name="R1",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=AuthFailedWorker(
                    runtime_root=root / ".requirements_review_runtime",
                    runtime_dir=root / ".requirements_review_runtime" / "old",
                ),
                review_md_path=review_md_path,
                review_json_path=review_json_path,
                contract=build_reviewer_completion_contract(
                    requirement_name="需求A",
                    reviewer_name="R1",
                    review_md_path=review_md_path,
                    review_json_path=review_json_path,
                ),
            )
            recreated = ReviewerRuntime(
                reviewer_name="R1",
                selection=ReviewAgentSelection("codex", "gpt-5.4-mini", "high", ""),
                worker=_FakeWorker(
                    runtime_root=root / ".requirements_review_runtime",
                    runtime_dir=root / ".requirements_review_runtime" / "new",
                ),
                review_md_path=review_md_path,
                review_json_path=review_json_path,
                contract=build_reviewer_completion_contract(
                    requirement_name="需求A",
                    reviewer_name="R1",
                    review_md_path=review_md_path,
                    review_json_path=review_json_path,
                ),
            )
            call_counter = {"count": 0}

            def fake_run_reviewer_turn(reviewer, *, label, prompt):  # noqa: ANN001
                call_counter["count"] += 1
                if call_counter["count"] == 1:
                    raise RuntimeError("API Error: 401 invalid access token or token expired")
                return None

            with patch(
                "A03_RequirementsReview._run_reviewer_turn",
                side_effect=fake_run_reviewer_turn,
            ), patch(
                "A03_RequirementsReview.create_reviewer_runtime",
                return_value=recreated,
            ), patch(
                "A03_RequirementsReview.prompt_yes_no_choice",
                return_value=True,
            ), patch(
                "A03_RequirementsReview.prompt_review_agent_selection",
                return_value=ReviewAgentSelection("codex", "gpt-5.4-mini", "high", ""),
            ), patch("sys.stdout", io.StringIO()):
                result = run_reviewer_turn_with_recreation(
                    original,
                    project_dir=root,
                    requirement_name="需求A",
                    label="review",
                    prompt="do review",
                )

            self.assertIs(result, recreated)
            self.assertEqual(call_counter["count"], 2)

    def test_run_ba_turn_with_recreation_rebuilds_ba_after_agent_ready_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original = RequirementsAnalystHandoff(
                worker=_FakeWorker(runtime_root=root / ".requirements_review_runtime", runtime_dir=root / ".requirements_review_runtime" / "old"),
                vendor="codex",
                model="gpt-5.4",
                reasoning_effort="high",
                proxy_url="",
            )
            recreated_worker = _FakeWorker(
                runtime_root=root / ".requirements_review_runtime",
                runtime_dir=root / ".requirements_review_runtime" / "new",
            )
            call_counter = {"count": 0}

            def fake_run_ba_turn(handoff, *, label, prompt, result_contract):  # noqa: ANN001
                call_counter["count"] += 1
                if call_counter["count"] == 1:
                    raise RuntimeError("Timed out waiting for agent ready.\nmock screen")
                self.assertEqual(result_contract.mode, "a03_human_feedback")
                return {"status": "completed"}

            class FakeTmuxWorker(_FakeWorker):
                pass

            with patch(
                "A03_RequirementsReview._run_ba_turn",
                side_effect=fake_run_ba_turn,
            ), patch(
                "A03_RequirementsReview.prompt_yes_no_choice",
                return_value=True,
            ), patch(
                "A03_RequirementsReview.prompt_review_agent_selection",
                return_value=ReviewAgentSelection("codex", "gpt-5.4-mini", "high", ""),
            ), patch(
                "A03_RequirementsReview.TmuxBatchWorker",
                return_value=recreated_worker,
            ), patch("sys.stdout", io.StringIO()):
                handoff, payload = run_ba_turn_with_recreation(
                    original,
                    project_dir=root,
                    label="resume_ba",
                    prompt="resume",
                    result_contract=build_ba_human_feedback_result_contract(
                        build_requirements_review_paths(root, "需求A")
                    ),
                )

            self.assertEqual(payload["status"], "completed")
            self.assertEqual(handoff.model, "gpt-5.4-mini")
            self.assertEqual(call_counter["count"], 2)

    def test_run_reviewer_turn_with_recreation_rebuilds_reviewer_after_agent_ready_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            review_md_path, review_json_path = build_reviewer_artifact_paths(root, "需求A", "R1")
            review_md_path.write_text("", encoding="utf-8")
            review_json_path.write_text("[]", encoding="utf-8")
            original = ReviewerRuntime(
                reviewer_name="R1",
                selection=ReviewAgentSelection("codex", "gpt-5.4", "high", ""),
                worker=_FakeWorker(runtime_root=root / ".requirements_review_runtime", runtime_dir=root / ".requirements_review_runtime" / "old"),
                review_md_path=review_md_path,
                review_json_path=review_json_path,
                contract=build_reviewer_completion_contract(
                    requirement_name="需求A",
                    reviewer_name="R1",
                    review_md_path=review_md_path,
                    review_json_path=review_json_path,
                ),
            )
            recreated = ReviewerRuntime(
                reviewer_name="R1",
                selection=ReviewAgentSelection("codex", "gpt-5.4-mini", "high", ""),
                worker=_FakeWorker(
                    runtime_root=root / ".requirements_review_runtime",
                    runtime_dir=root / ".requirements_review_runtime" / "new",
                ),
                review_md_path=review_md_path,
                review_json_path=review_json_path,
                contract=build_reviewer_completion_contract(
                    requirement_name="需求A",
                    reviewer_name="R1",
                    review_md_path=review_md_path,
                    review_json_path=review_json_path,
                ),
            )
            call_counter = {"count": 0}

            def fake_run_reviewer_turn(reviewer, *, label, prompt):  # noqa: ANN001
                call_counter["count"] += 1
                if call_counter["count"] == 1:
                    raise RuntimeError("Timed out waiting for agent ready.\nmock screen")
                return None

            with patch(
                "A03_RequirementsReview._run_reviewer_turn",
                side_effect=fake_run_reviewer_turn,
            ), patch(
                "A03_RequirementsReview.create_reviewer_runtime",
                return_value=recreated,
            ), patch(
                "A03_RequirementsReview.prompt_yes_no_choice",
                return_value=True,
            ), patch(
                "A03_RequirementsReview.prompt_review_agent_selection",
                return_value=ReviewAgentSelection("codex", "gpt-5.4-mini", "high", ""),
            ), patch("sys.stdout", io.StringIO()):
                result = run_reviewer_turn_with_recreation(
                    original,
                    project_dir=root,
                    requirement_name="需求A",
                    label="review",
                    prompt="do review",
                )

            self.assertIs(result, recreated)
            self.assertEqual(call_counter["count"], 2)


if __name__ == "__main__":
    unittest.main()
