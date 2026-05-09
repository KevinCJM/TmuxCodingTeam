from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import A01_Routing_LayerPlanning as routing_stage
from tmux_core.stage_kernel import detailed_design, requirements_review, reviewer_orchestration, shared_review
from tmux_core.stage_kernel.agent_intervention import AGENT_INTERVENTION_RECREATE
from T09_terminal_ops import PromptBackRequested


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StageKernelSharedTests(unittest.TestCase):
    def test_a04_a05_share_review_support_types(self):
        self.assertIs(requirements_review.ReviewAgentSelection, shared_review.ReviewAgentSelection)
        self.assertIs(detailed_design.ReviewAgentSelection, shared_review.ReviewAgentSelection)
        self.assertIs(requirements_review.ReviewerRuntime, shared_review.ReviewerRuntime)
        self.assertIs(detailed_design.ReviewerRuntime, shared_review.ReviewerRuntime)
        self.assertTrue(issubclass(requirements_review.ReviewStageProgress, shared_review.ReviewStageProgress))
        self.assertIs(detailed_design.ReviewStageProgress, shared_review.ReviewStageProgress)
        self.assertIs(requirements_review.ensure_empty_file, shared_review.ensure_empty_file)
        self.assertIs(detailed_design.ensure_empty_file, shared_review.ensure_empty_file)
        self.assertIs(requirements_review.worker_has_provider_auth_error, shared_review.worker_has_provider_auth_error)
        self.assertIs(detailed_design.worker_has_provider_auth_error, shared_review.worker_has_provider_auth_error)

    def test_a04_a05_delegate_reviewer_orchestration_to_shared_kernel(self):
        review_source = (PROJECT_ROOT / "tmux_core/stage_kernel/requirements_review.py").read_text(encoding="utf-8")
        design_source = (PROJECT_ROOT / "tmux_core/stage_kernel/detailed_design.py").read_text(encoding="utf-8")

        for source in (review_source, design_source):
            self.assertIn("run_parallel_reviewer_round(", source)
            self.assertIn("repair_reviewer_round_outputs(", source)
            self.assertIn("shutdown_stage_workers(", source)

        self.assertTrue(hasattr(reviewer_orchestration, "run_parallel_reviewer_round"))
        self.assertTrue(hasattr(reviewer_orchestration, "repair_reviewer_round_outputs"))
        self.assertTrue(hasattr(reviewer_orchestration, "shutdown_stage_workers"))

    def test_reviewer_orchestration_does_not_drop_prelaunch_missing_session_reviewer(self):
        class _PrelaunchReviewer:
            agent_started = False
            pane_id = ""

            def get_agent_state(self):
                return "DEAD"

            def read_state(self):
                return {
                    "status": "running",
                    "result_status": "running",
                    "agent_state": "DEAD",
                    "agent_started": False,
                    "pane_id": "",
                    "health_status": "missing_session",
                    "workflow_stage": "pending",
                }

        reviewer = SimpleNamespace(worker=_PrelaunchReviewer(), reviewer_name="审核员")

        self.assertFalse(reviewer_orchestration._owner_is_dead(reviewer))  # noqa: SLF001

    def test_reviewer_repair_hitl_can_recreate_reviewer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            old_reviewer = SimpleNamespace(
                reviewer_name="测试工程师",
                worker=SimpleNamespace(session_name="测试工程师-天暴星"),
                review_md_path=root / "old.md",
                review_json_path=root / "old.json",
            )
            new_reviewer = SimpleNamespace(
                reviewer_name="测试工程师",
                worker=SimpleNamespace(session_name="测试工程师-天勇星"),
                review_md_path=root / "new.md",
                review_json_path=root / "new.json",
            )
            fixed = {"done": False}
            check_names: list[tuple[str, ...]] = []
            recreated: list[object] = []
            fix_calls: list[tuple[object, str, int]] = []

            def check_job(names):
                check_names.append(tuple(names))
                if fixed["done"]:
                    return {}
                return {names[0]: "请修复审核输出"} if names else {}

            def recreate(reviewer):
                recreated.append(reviewer)
                return new_reviewer

            def run_fix_turn(reviewer, prompt, repair_attempt):
                fix_calls.append((reviewer, prompt, repair_attempt))
                fixed["done"] = True
                return reviewer

            with patch.object(
                reviewer_orchestration,
                "request_file_noncompliance_intervention",
                return_value=AGENT_INTERVENTION_RECREATE,
            ) as intervention:
                result = reviewer_orchestration.repair_reviewer_round_outputs(
                    [old_reviewer],
                    key_func=lambda reviewer: reviewer.reviewer_name,
                    artifact_name_func=lambda reviewer: str(reviewer.worker.session_name),
                    check_job=check_job,
                    run_fix_turn=run_fix_turn,
                    max_attempts=0,
                    error_prefix="审核修复失败",
                    final_error="仍未按协议更新文档",
                    recreate_reviewer=recreate,
                )

        self.assertEqual(result, [new_reviewer])
        self.assertEqual(recreated, [old_reviewer])
        self.assertEqual(fix_calls, [(new_reviewer, "请修复审核输出", 1)])
        self.assertTrue(intervention.call_args.kwargs["allow_recreate"])
        self.assertIn(("测试工程师-天勇星",), check_names)

    def test_parse_review_max_rounds_supports_default_and_infinite(self):
        self.assertEqual(shared_review.parse_review_max_rounds("", source="--review-max-rounds"), 5)
        self.assertIsNone(shared_review.parse_review_max_rounds("infinite", source="--review-max-rounds"))

    def test_prompt_review_max_rounds_retries_until_valid(self):
        with patch.object(shared_review, "prompt_with_default", side_effect=["abc", "infinite"]) as prompt_mock, patch.object(
            shared_review,
            "message",
        ) as message_mock:
            value = shared_review.prompt_review_max_rounds()

        self.assertIsNone(value)
        self.assertEqual(prompt_mock.call_args_list[0].args[0], "输入最大审核轮次（输入 infinite 表示不设上限）")
        self.assertTrue(any("必须是正整数或 infinite" in str(call.args[0]) for call in message_mock.call_args_list if call.args))

    def test_collect_reviewer_agent_selections_bubbles_back_from_first_prompt_when_enabled(self):
        reviewer_spec = type("ReviewerSpec", (), {"reviewer_key": "R1", "role_name": "Reviewer"})()
        with patch.object(shared_review, "stdin_is_interactive", return_value=True), patch.object(
            shared_review,
            "prompt_review_agent_selection",
            side_effect=PromptBackRequested(),
        ) as prompt_mock:
            with self.assertRaises(PromptBackRequested):
                shared_review.collect_reviewer_agent_selections(
                    project_dir="/tmp/project",
                    reviewer_specs=[reviewer_spec],
                    display_name_resolver=lambda *_args: "R1-display",
                    allow_back_first_prompt=True,
                    stage_key="requirements_review_reviewer_selection",
                )

        self.assertTrue(prompt_mock.call_args.kwargs["allow_back_first_step"])
        self.assertEqual(prompt_mock.call_args.kwargs["stage_key"], "requirements_review_reviewer_selection")

    def test_review_round_policy_resets_quota_without_resetting_initial_flag(self):
        policy = shared_review.ReviewRoundPolicy(max_rounds=2)

        policy.record_review_attempt()
        policy.record_review_attempt()
        self.assertTrue(policy.initial_review_done)
        self.assertTrue(policy.should_escalate_before_next_review())

        policy.reset_after_hitl()
        self.assertEqual(policy.quota_count, 0)
        self.assertTrue(policy.initial_review_done)
        self.assertFalse(policy.should_escalate_before_next_review())

    def test_auto_review_limit_hitl_response_includes_question_and_non_interactive_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            question_path = Path(tmpdir) / "ask.md"
            question_path.write_text("请选择方案1原样同步。", encoding="utf-8")

            with patch.object(shared_review, "message") as message_mock:
                response = shared_review.collect_auto_review_limit_hitl_response(
                    question_path,
                    stage_label="需求评审超限",
                    hitl_round=2,
                )

        self.assertIn("--yes", response)
        self.assertIn("不跳过评审", response)
        self.assertIn("请选择方案1原样同步。", response)
        self.assertTrue(any("--yes 自动回复" in str(call.args[0]) for call in message_mock.call_args_list if call.args))

    def test_review_limit_hitl_cycle_uses_human_input_provider(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ask_path = Path(tmpdir) / "ask.md"
            hitl_record_path = Path(tmpdir) / "hitl.md"
            ask_path.write_text("需要人工确认", encoding="utf-8")
            received_messages: list[str] = []

            def initial_turn():
                return "owner-before-hitl"

            def human_reply_turn(human_msg: str):
                received_messages.append(human_msg)
                ask_path.write_text("", encoding="utf-8")
                return "owner-after-hitl"

            def provider(question_path: Path, hitl_round: int) -> str:
                self.assertEqual(question_path, ask_path.resolve())
                self.assertEqual(hitl_round, 1)
                return "自动继续"

            with patch.object(
                shared_review,
                "collect_review_limit_hitl_response",
                side_effect=AssertionError("should not prompt"),
            ):
                result = shared_review.run_review_limit_hitl_cycle(
                    stage_label="需求评审超限",
                    ask_human_path=ask_path,
                    hitl_record_path=hitl_record_path,
                    initial_turn=initial_turn,
                    human_reply_turn=human_reply_turn,
                    human_input_provider=provider,
                )

        self.assertEqual(result.owner, "owner-after-hitl")
        self.assertEqual(result.rounds_used, 1)
        self.assertTrue(result.post_hitl_continue_completed)
        self.assertEqual(received_messages, ["自动继续"])

    def test_shell_initialization_timeout_is_recoverable_startup_failure(self):
        error = RuntimeError("Shell initialization timed out.\nzsh prompt")
        self.assertTrue(shared_review.is_recoverable_startup_failure(error))

    def test_ensure_review_artifacts_creates_empty_markdown_and_empty_json_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "review.md"
            json_path = Path(tmpdir) / "review.json"

            shared_review.ensure_review_artifacts(md_path, json_path)

            self.assertEqual(md_path.read_text(encoding="utf-8"), "")
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), [])

    def test_resolve_stage_agent_config_accepts_fixed_two_reviewers(self):
        args = type(
            "Args",
            (),
            {
                "agent_config": "",
                "main_agent": "vendor=codex,model=gpt-5.4,effort=high,proxy=10900",
                "reviewer_agent": [
                    "name=R1,vendor=codex,model=gpt-5.4-mini,effort=medium,proxy=10900",
                    "name=R2,vendor=codex,model=gpt-5.4,effort=high",
                ],
            },
        )()

        config = shared_review.resolve_stage_agent_config(args)

        self.assertIsNotNone(config.main)
        self.assertEqual(config.main.vendor, "codex")
        self.assertEqual(config.main.proxy_url, "10900")
        self.assertEqual(config.reviewer_order, ("R1", "R2"))
        self.assertEqual(config.reviewer_selection("R1").model, "gpt-5.4-mini")
        self.assertEqual(config.reviewer_selection("R2").model, "gpt-5.4")

    def test_resolve_stage_agent_config_uses_stage_specific_overrides_with_global_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "agents.json"
            config_path.write_text(
                json.dumps(
                    {
                        "main": {"vendor": "codex", "model": "gpt-5.4", "effort": "high"},
                        "reviewers": [{"name": "R1", "vendor": "codex", "model": "gpt-5.4-mini", "effort": "medium"}],
                        "stages": {
                            "development": {
                                "main": {"vendor": "gemini", "model": "flash", "effort": "medium", "proxy": "10809"},
                                "reviewers": [{"name": "R1", "vendor": "opencode", "model": "opencode/big-pickle", "effort": "xhigh"}],
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            args = type("Args", (), {"agent_config": str(config_path), "main_agent": "", "reviewer_agent": []})()

            development = shared_review.resolve_stage_agent_config(args, stage_key="development")
            design = shared_review.resolve_stage_agent_config(args, stage_key="detailed_design")

        self.assertEqual(development.main.vendor, "gemini")
        self.assertEqual(development.main.model, "flash")
        self.assertEqual(development.main.proxy_url, "10809")
        self.assertEqual(development.reviewer_selection("R1").vendor, "opencode")
        self.assertEqual(development.reviewer_selection("R1").reasoning_effort, "xhigh")
        self.assertEqual(design.main.vendor, "codex")
        self.assertEqual(design.reviewer_selection("R1").model, "gpt-5.4-mini")

    def test_prompt_effort_falls_back_when_default_is_not_allowed(self):
        with patch.object(routing_stage, "normalize_vendor_choice", return_value="codex"), patch.object(
            routing_stage,
            "normalize_model_choice",
            return_value="gpt-x",
        ), patch.object(
            routing_stage,
            "get_normalized_effort_choices",
            return_value=("high",),
        ), patch.object(
            routing_stage,
            "normalize_effort_choice",
            side_effect=[ValueError("gpt-x 不支持的推理强度: max"), "high"],
        ), patch.object(
            routing_stage,
            "prompt_select_option",
            return_value="high",
        ) as prompt_mock:
            effort = routing_stage.prompt_effort("codex", "gpt-x", "max", role_label="审核员")

        self.assertEqual(effort, "high")
        self.assertEqual(prompt_mock.call_args.kwargs["default_value"], "high")

    def test_resolve_stage_agent_config_records_invalid_reviewer_without_raising(self):
        args = type(
            "Args",
            (),
            {
                "agent_config": "",
                "main_agent": "",
                "reviewer_agent": [
                    "name=审核员,vendor=codex,model=gpt-5.4,effort=max",
                ],
            },
        )()
        with patch.object(
            shared_review,
            "parse_agent_selection_spec",
            side_effect=ValueError("gpt-5.4 不支持的推理强度: max"),
        ):
            config = shared_review.resolve_stage_agent_config(args)

        self.assertEqual(config.reviewer_order, ("审核员",))
        self.assertIsNone(config.reviewer_selection("审核员"))
        self.assertIn("审核员", config.invalid_reviewers)
        self.assertIn("不支持的推理强度", config.invalid_reviewers["审核员"].error)

    def test_agent_run_config_creation_error_prompts_reselection(self):
        invalid = shared_review.ReviewAgentSelection("opencode", "ark-coding-plan/doubao-seed-2.0-pro", "max", "")
        replacement = shared_review.ReviewAgentSelection("opencode", "ark-coding-plan/doubao-seed-2.0-pro", "high", "")
        fake_config = object()

        with patch.object(shared_review, "stdin_is_interactive", return_value=True), patch.object(
            shared_review,
            "AgentRunConfig",
            side_effect=[ValueError("ark-coding-plan/doubao-seed-2.0-pro 不支持的推理强度: max"), fake_config],
        ), patch.object(
            shared_review,
            "prompt_review_agent_selection",
            return_value=replacement,
        ) as prompt_mock, patch.object(shared_review, "message"):
            selection, config = shared_review.resolve_agent_run_config_with_recovery(
                invalid,
                role_label="审核员",
            )

        self.assertEqual(selection.reasoning_effort, "high")
        self.assertIs(config, fake_config)
        prompt_mock.assert_called_once()

    def test_requirements_review_proxy_prompt_does_not_recurse_after_binding_sync(self):
        requirements_review._sync_shared_review_bindings()
        with patch.object(requirements_review, "prompt_with_default", return_value="10900") as prompt_mock:
            proxy_url = requirements_review.prompt_proxy_url("7890", role_label="审核员")

        self.assertEqual(proxy_url, "10900")
        prompt_mock.assert_called_once_with(
            "为 审核员 输入代理端口或完整代理 URL（可留空）",
            "7890",
            allow_empty=True,
        )
