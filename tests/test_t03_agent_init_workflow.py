from __future__ import annotations

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from T02_tmux_agents import AgentRunConfig, AgentRuntimeState, CommandResult, WorkerResult
from T03_agent_init_workflow import (
    BatchInitResult,
    DirectoryInitResult,
    ROUTING_AUDIT_RECORD_FILE,
    ROUTING_AUDIT_PASS_TOKEN,
    ROUTING_AUDIT_REVISE_TOKEN,
    ROUTING_AUDIT_STATUS_FAIL,
    ROUTING_AUDIT_STATUS_FILE,
    ROUTING_AUDIT_STATUS_PASS,
    ROUTING_LAYER_REQUIRED_FILES,
    ROUTING_RUNTIME_ROOT_NAME,
    PHASE_ROUTING_LAYER_AUDIT,
    PHASE_ROUTING_LAYER_CREATE,
    PHASE_ROUTING_LAYER_REFINE,
    RunStore,
    TURN_STATUS_SCHEMA_VERSION,
    TURN_STATUS_FILE,
    TargetSelection,
    audit_pass_has_extra_text,
    audit_output_requires_revise,
    build_audit_prompt,
    build_create_prompt,
    build_refine_prompt,
    build_routing_runtime_root,
    build_turn_file_contract,
    build_prefixed_sha256,
    cleanup_routing_stage_artifacts,
    determine_batch_worker_count,
    extract_protocol_token,
    has_overlapping_scope_paths,
    has_complete_routing_layer,
    load_routing_audit_decision,
    load_existing_run,
    missing_routing_layer_files,
    normalize_audit_output,
    prepare_revise_audit_output,
    prepare_live_workers,
    project_has_business_files,
    resolve_target_selection,
    run_batch_initialization,
    run_directory_initialization_with_worker,
    routing_turn_status_path,
    validate_routing_layer_artifacts,
)


def _write_valid_routing_layer(project_dir: Path) -> None:
    (project_dir / "docs").mkdir(exist_ok=True)
    (project_dir / "AGENTS.md").write_text("ok\n", encoding="utf-8")
    (project_dir / "docs" / "repo_map.json").write_text(
        json.dumps({"modules": [{"id": "M01", "name": "root"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (project_dir / "docs" / "task_routes.json").write_text(
        json.dumps({"routes": [{"id": "R01", "first_read_modules": ["M01"], "pitfall_ids": ["P01"]}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (project_dir / "docs" / "pitfalls.json").write_text(
        json.dumps({"pitfalls": [{"id": "P01", "title": "risk"}]}, ensure_ascii=False),
        encoding="utf-8",
    )


class FakeWorker:
    scripts: dict[str, list[dict[str, object]]] = {}
    alive_sessions: dict[str, bool] = {}

    def __init__(
        self,
        *,
        worker_id,
        work_dir,
        config,
        runtime_root,
        existing_runtime_dir=None,
        existing_session_name="",
        existing_pane_id="",
    ):
        self.worker_id = worker_id
        self.work_dir = Path(work_dir)
        self.config = config
        self.runtime_root = Path(runtime_root)
        self.runtime_dir = Path(existing_runtime_dir) if existing_runtime_dir else (self.runtime_root / worker_id)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.session_name = existing_session_name or f"fake-{worker_id}"
        self.pane_id = existing_pane_id or "pane-0"
        self.log_path = self.runtime_dir / "worker.log"
        self.raw_log_path = self.runtime_dir / "worker.raw.log"
        self.state_path = self.runtime_dir / "worker.state.json"
        self.transcript_path = self.runtime_dir / "transcript.md"
        self.results = []
        self.script = list(self.scripts.get(str(self.work_dir), []))
        self._write_state(
            {
                "session_name": self.session_name,
                "pane_id": self.pane_id,
                "runtime_dir": str(self.runtime_dir),
                "workflow_stage": "pending",
                "workflow_round": 0,
                "result_status": "pending",
                "agent_state": "STARTING",
                "retry_count": 0,
                "last_log_offset": 0,
                "log_path": str(self.log_path),
                "raw_log_path": str(self.raw_log_path),
                "state_path": str(self.state_path),
                "transcript_path": str(self.transcript_path),
            }
        )

    def _write_state(self, payload):
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def read_state(self):
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def runtime_metadata(self):
        return {
            "worker_id": self.worker_id,
            "session_name": self.session_name,
            "pane_id": self.pane_id,
            "runtime_dir": str(self.runtime_dir),
            "work_dir": str(self.work_dir),
            "log_path": str(self.log_path),
            "raw_log_path": str(self.raw_log_path),
            "state_path": str(self.state_path),
            "transcript_path": str(self.transcript_path),
        }

    def session_exists(self):
        return self.alive_sessions.get(self.session_name, False)

    def _build_default_turn_status(self, *, contract, label):
        if contract.phase == PHASE_ROUTING_LAYER_AUDIT:
            artifact_paths = [ROUTING_AUDIT_STATUS_FILE, ROUTING_AUDIT_RECORD_FILE]
            artifacts = {
                "audit_status": ROUTING_AUDIT_STATUS_FILE,
                "review_record": ROUTING_AUDIT_RECORD_FILE,
            }
        else:
            artifact_paths = list(ROUTING_LAYER_REQUIRED_FILES)
            artifacts = {"routing_layer_files": list(ROUTING_LAYER_REQUIRED_FILES)}
        artifact_hashes = {}
        for relative_path in artifact_paths:
            file_path = self.work_dir / relative_path
            if file_path.exists():
                artifact_hashes[relative_path] = build_prefixed_sha256(file_path)
        status_payload = {
            "schema_version": TURN_STATUS_SCHEMA_VERSION,
            "turn_id": contract.turn_id,
            "phase": contract.phase,
            "status": "done",
            "artifacts": artifacts,
            "artifact_hashes": artifact_hashes,
            "written_at": "2026-04-12T00:00:03",
        }
        contract.status_path.parent.mkdir(parents=True, exist_ok=True)
        contract.status_path.write_text(json.dumps(status_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def run_turn(self, *, label, prompt, required_tokens=(), completion_contract=None, timeout_sec=None):
        step = self.script.pop(0)
        create_files = step.get("create_files", ())
        handled_create_files: set[str] = set()
        if set(str(item) for item in create_files) >= set(ROUTING_LAYER_REQUIRED_FILES):
            _write_valid_routing_layer(self.work_dir)
            handled_create_files = set(ROUTING_LAYER_REQUIRED_FILES)
        for relative_path in create_files:
            if str(relative_path) in handled_create_files:
                continue
            path = self.work_dir / str(relative_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("ok\n", encoding="utf-8")
        for relative_path, content in dict(step.get("write_files", {})).items():
            path = self.work_dir / str(relative_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(content), encoding="utf-8")
        if completion_contract is not None and not step.get("skip_turn_status", False):
            status_payload = step.get("turn_status_payload")
            if status_payload is None:
                self._build_default_turn_status(contract=completion_contract, label=label)
            else:
                completion_contract.status_path.parent.mkdir(parents=True, exist_ok=True)
                completion_contract.status_path.write_text(
                    json.dumps(status_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        result = CommandResult(
            label=label,
            command=prompt,
            exit_code=int(step.get("exit_code", 0)),
            raw_output=str(step.get("output", "")),
            clean_output=str(step.get("output", "")),
            started_at="2026-04-12T00:00:00",
            finished_at="2026-04-12T00:00:01",
        )
        self.results.append(result)
        state = self.read_state()
        state.update(
            {
                "workflow_stage": label,
                "workflow_round": int(state.get("workflow_round", 0)),
                "result_status": "succeeded" if result.ok else "failed",
                "last_turn_token": label,
                "last_prompt_hash": "fakehash",
                "last_heartbeat_at": "2026-04-12T00:00:02",
            }
        )
        self._write_state(state)
        return result

    def collect_result(self):
        return WorkerResult(
            worker_id=self.worker_id,
            session_name=self.session_name,
            pane_id=self.pane_id,
            runtime_dir=str(self.runtime_dir),
            work_dir=str(self.work_dir),
            config=self.config.to_summary(),
            status="succeeded",
            commands=list(self.results),
        )


class AgentInitWorkflowTests(unittest.TestCase):
    def setUp(self):
        FakeWorker.scripts = {}
        FakeWorker.alive_sessions = {}

    @staticmethod
    def _strong_pass_audit_output() -> str:
        return ROUTING_AUDIT_PASS_TOKEN

    @staticmethod
    def _audit_write_files(*, round_index: int, status: str, record_text: str) -> dict[str, str]:
        return {
            ROUTING_AUDIT_RECORD_FILE: record_text,
            ROUTING_AUDIT_STATUS_FILE: json.dumps(
                {
                    "schema_version": "1.0",
                    "stage": PHASE_ROUTING_LAYER_AUDIT,
                    "audit_round": round_index,
                    "status": status,
                    "review_record_path": ROUTING_AUDIT_RECORD_FILE,
                },
                ensure_ascii=False,
                indent=2,
            ),
        }

    def test_missing_routing_files_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "docs").mkdir(parents=True)
            (root / "AGENTS.md").write_text("ok\n", encoding="utf-8")
            missing = missing_routing_layer_files(root)
            self.assertEqual(
                missing,
                ["docs/repo_map.json", "docs/task_routes.json", "docs/pitfalls.json"],
            )
            self.assertFalse(has_complete_routing_layer(root))

    def test_validate_routing_layer_artifacts_accepts_minimal_valid_docs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_valid_routing_layer(root)

            validate_routing_layer_artifacts(root)
            self.assertTrue(has_complete_routing_layer(root))

    def test_validate_routing_layer_artifacts_rejects_empty_json_objects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "docs").mkdir(parents=True)
            (root / "AGENTS.md").write_text("ok\n", encoding="utf-8")
            for name in ("repo_map.json", "task_routes.json", "pitfalls.json"):
                (root / "docs" / name).write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "modules|routes|pitfalls"):
                validate_routing_layer_artifacts(root)
            self.assertFalse(has_complete_routing_layer(root))

    def test_validate_routing_layer_artifacts_rejects_unresolved_refs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_valid_routing_layer(root)
            (root / "docs" / "task_routes.json").write_text(
                json.dumps(
                    {
                        "routes": [
                            {
                                "id": "R01",
                                "first_read_modules": ["M99"],
                                "pitfall_ids": ["P99"],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "M99|P99"):
                validate_routing_layer_artifacts(root)

    def test_cleanup_routing_stage_artifacts_removes_audit_files_and_runtime_dir_on_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project_dir = root / "project"
            project_dir.mkdir()
            (project_dir / ROUTING_AUDIT_STATUS_FILE).write_text("{}", encoding="utf-8")
            (project_dir / ROUTING_AUDIT_RECORD_FILE).write_text("record", encoding="utf-8")
            runtime_dir = root / "runtime" / "run_demo"
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "manifest.json").write_text("{}", encoding="utf-8")
            cleanup = cleanup_routing_stage_artifacts(
                batch_result=BatchInitResult(
                    run_id="run_demo",
                    runtime_dir=str(runtime_dir),
                    selection=TargetSelection(
                        project_dir=str(project_dir),
                        selected_dirs=(str(project_dir),),
                        skipped_dirs=(),
                        forced_dirs=(),
                        project_missing_files=(),
                    ),
                    config={"vendor": "codex", "model": "gpt-5.4", "reasoning_effort": "high", "proxy_url": ""},
                    results=[DirectoryInitResult(work_dir=str(project_dir), forced=False, status="passed", rounds_used=0)],
                )
            )
            self.assertGreaterEqual(cleanup.removed_intermediate_count, 2)
            self.assertGreaterEqual(cleanup.removed_runtime_count, 1)
            self.assertFalse((project_dir / ROUTING_AUDIT_STATUS_FILE).exists())
            self.assertFalse((project_dir / ROUTING_AUDIT_RECORD_FILE).exists())
            self.assertFalse(runtime_dir.exists())

    def test_cleanup_routing_stage_artifacts_preserves_debug_files_when_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project_dir = root / "project"
            project_dir.mkdir()
            (project_dir / ROUTING_AUDIT_STATUS_FILE).write_text("{}", encoding="utf-8")
            (project_dir / ROUTING_AUDIT_RECORD_FILE).write_text("record", encoding="utf-8")
            runtime_dir = root / "runtime" / "run_demo"
            runtime_dir.mkdir(parents=True)
            cleanup = cleanup_routing_stage_artifacts(
                batch_result=BatchInitResult(
                    run_id="run_demo",
                    runtime_dir=str(runtime_dir),
                    selection=TargetSelection(
                        project_dir=str(project_dir),
                        selected_dirs=(str(project_dir),),
                        skipped_dirs=(),
                        forced_dirs=(),
                        project_missing_files=(),
                    ),
                    config={"vendor": "codex", "model": "gpt-5.4", "reasoning_effort": "high", "proxy_url": ""},
                    results=[DirectoryInitResult(work_dir=str(project_dir), forced=False, status="failed", rounds_used=0)],
                )
            )
            self.assertEqual(cleanup.removed_intermediate_count, 0)
            self.assertEqual(cleanup.removed_runtime_count, 0)
            self.assertTrue((project_dir / ROUTING_AUDIT_STATUS_FILE).exists())
            self.assertTrue((project_dir / ROUTING_AUDIT_RECORD_FILE).exists())
            self.assertTrue(runtime_dir.exists())

    def test_resolve_target_selection_forces_project_dir_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            calc_dir = (project_dir / "calc").resolve()
            calc_dir.mkdir(parents=True)
            (project_dir / "app.py").write_text("print('ok')\n", encoding="utf-8")
            selection = resolve_target_selection(
                project_dir=project_dir,
                target_dirs=("calc",),
                run_init=False,
            )
            self.assertEqual(selection.selected_dirs, (str(project_dir),))
            self.assertEqual(selection.skipped_dirs, (str(calc_dir),))
            self.assertEqual(selection.forced_dirs, (str(project_dir),))
            self.assertTrue(selection.project_missing_files)

    def test_resolve_target_selection_skips_empty_missing_project_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            (project_dir / ".routing_init_runtime").mkdir()
            (project_dir / ".routing_init_runtime" / "manifest.json").write_text("{}", encoding="utf-8")
            (project_dir / ".pytest_cache").mkdir()
            (project_dir / "__pycache__").mkdir()
            (project_dir / "__pycache__" / "module.pyc").write_bytes(b"cache")
            (project_dir / ".gitignore").write_text("*.pyc\n", encoding="utf-8")

            selection = resolve_target_selection(
                project_dir=project_dir,
                run_init=False,
            )
            self.assertFalse(project_has_business_files(project_dir))
            self.assertFalse(selection.should_run)
            self.assertEqual(selection.selected_dirs, ())
            self.assertEqual(selection.skipped_dirs, (str(project_dir),))
            self.assertEqual(selection.forced_dirs, ())
            self.assertTrue(selection.project_missing_files)

    def test_project_has_business_files_counts_unignored_symlink(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            outside_dir = (Path(tmpdir) / "outside").resolve()
            project_dir.mkdir()
            outside_dir.mkdir()
            (outside_dir / "app.py").write_text("print('ok')\n", encoding="utf-8")
            (project_dir / "linked_app").symlink_to(outside_dir, target_is_directory=True)

            self.assertTrue(project_has_business_files(project_dir))

    def test_resolve_target_selection_skips_all_when_disabled_and_complete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            calc_dir = (project_dir / "calc").resolve()
            calc_dir.mkdir(parents=True)
            _write_valid_routing_layer(project_dir)

            selection = resolve_target_selection(
                project_dir=project_dir,
                target_dirs=("calc",),
                run_init=False,
            )
            self.assertFalse(selection.should_run)
            self.assertEqual(selection.selected_dirs, ())
            self.assertEqual(selection.skipped_dirs, (str(project_dir), str(calc_dir)))

    def test_prompt_builders_keep_runtime_contracts_out_of_prompt_body(self):
        create_prompt = build_create_prompt()
        audit_prompt = build_audit_prompt(audit_round=1)
        refine_prompt = build_refine_prompt("/tmp/project/路由层审核记录.md")
        self.assertIn("AGENTS.md", create_prompt)
        self.assertIn("inspect at most about 12 files total in the first pass", create_prompt)
        self.assertIn("read every source file in a large homogeneous directory", create_prompt)
        self.assertIn("CI/workflow/editor metadata is **low priority**", create_prompt)
        self.assertIn("creating or overwriting all 4 required output files", create_prompt)
        self.assertIn("your next shell action must be a write action", create_prompt)
        self.assertIn("Do not do another read/search/list command before that first write", create_prompt)
        self.assertIn("Write the file skeletons first, then fill them in place", create_prompt)
        self.assertIn("If you catch yourself writing a planning note instead of files", create_prompt)
        self.assertIn("Do not plan code changes, patches, or review workflows", create_prompt)
        self.assertNotIn("<function create_routing_layer_file", create_prompt)
        self.assertNotIn("<function routing_layer_file_audit", audit_prompt)
        self.assertNotIn("<function routing_layer_refine", refine_prompt)
        self.assertNotIn(TURN_STATUS_FILE, create_prompt)
        self.assertNotIn(PHASE_ROUTING_LAYER_CREATE, create_prompt)
        self.assertNotIn("artifact_hashes", create_prompt)
        self.assertNotIn("written_at", create_prompt)
        self.assertNotIn("turn_id", create_prompt)
        self.assertIn(ROUTING_AUDIT_STATUS_FILE, audit_prompt)
        self.assertIn(ROUTING_AUDIT_RECORD_FILE, audit_prompt)
        self.assertIn(ROUTING_AUDIT_STATUS_PASS, audit_prompt)
        self.assertIn(ROUTING_AUDIT_STATUS_FAIL, audit_prompt)
        self.assertNotIn(TURN_STATUS_FILE, audit_prompt)
        self.assertNotIn("artifact_hashes", audit_prompt)
        self.assertNotIn("written_at", audit_prompt)
        self.assertNotIn("turn_id", audit_prompt)
        self.assertIn("/tmp/project/路由层审核记录.md", refine_prompt)
        self.assertIn("不要修改", refine_prompt)
        self.assertIn("下一轮审核由外层 Python 编排器负责", refine_prompt)
        self.assertNotIn(TURN_STATUS_FILE, refine_prompt)
        self.assertNotIn(PHASE_ROUTING_LAYER_REFINE, refine_prompt)
        self.assertNotIn("artifact_hashes", refine_prompt)
        self.assertNotIn("written_at", refine_prompt)
        self.assertNotIn("turn_id", refine_prompt)

    def test_build_turn_file_contract_materializes_turn_status_for_routing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir).resolve()
            _write_valid_routing_layer(project_dir)
            runtime_dir = project_dir / "runtime"
            contract = build_turn_file_contract(
                runtime_dir=runtime_dir,
                work_dir=project_dir,
                turn_id="create_routing_layer_1",
                phase=PHASE_ROUTING_LAYER_CREATE,
                required_artifacts=ROUTING_LAYER_REQUIRED_FILES,
            )
            self.assertEqual(contract.kind, "routing_file_contract")
            result = contract.validator(contract.status_path)
            payload = json.loads(Path(result.status_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["turn_id"], "create_routing_layer_1")
            self.assertEqual(payload["phase"], PHASE_ROUTING_LAYER_CREATE)
            self.assertEqual(payload["artifacts"]["routing_layer_files"], list(ROUTING_LAYER_REQUIRED_FILES))
            self.assertEqual(
                payload["artifact_hashes"]["AGENTS.md"],
                build_prefixed_sha256(project_dir / "AGENTS.md"),
            )

    def test_build_turn_file_contract_rejects_empty_routing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir).resolve()
            for relative in ROUTING_LAYER_REQUIRED_FILES:
                path = project_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            runtime_dir = project_dir / "runtime"
            contract = build_turn_file_contract(
                runtime_dir=runtime_dir,
                work_dir=project_dir,
                turn_id="create_routing_layer_1",
                phase=PHASE_ROUTING_LAYER_CREATE,
                required_artifacts=ROUTING_LAYER_REQUIRED_FILES,
            )
            with self.assertRaisesRegex(ValueError, "缺少路由层文件"):
                contract.validator(contract.status_path)

    def test_build_turn_file_contract_materializes_turn_status_for_audit_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir).resolve()
            (project_dir / ROUTING_AUDIT_STATUS_FILE).write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "stage": PHASE_ROUTING_LAYER_AUDIT,
                        "audit_round": 1,
                        "status": ROUTING_AUDIT_STATUS_PASS,
                        "review_record_path": ROUTING_AUDIT_RECORD_FILE,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (project_dir / ROUTING_AUDIT_RECORD_FILE).write_text("- status: 审核通过\n", encoding="utf-8")
            runtime_dir = project_dir / "runtime"
            contract = build_turn_file_contract(
                runtime_dir=runtime_dir,
                work_dir=project_dir,
                turn_id="audit_routing_layer_1",
                phase=PHASE_ROUTING_LAYER_AUDIT,
                required_artifacts=(ROUTING_AUDIT_STATUS_FILE, ROUTING_AUDIT_RECORD_FILE),
            )
            result = contract.validator(contract.status_path)
            payload = json.loads(Path(result.status_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["artifacts"]["audit_status"], ROUTING_AUDIT_STATUS_FILE)
            self.assertEqual(payload["artifacts"]["review_record"], ROUTING_AUDIT_RECORD_FILE)
            self.assertEqual(
                payload["artifact_hashes"][ROUTING_AUDIT_RECORD_FILE],
                build_prefixed_sha256(project_dir / ROUTING_AUDIT_RECORD_FILE),
            )

    def test_build_turn_file_contract_rematerializes_when_artifact_hash_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir).resolve()
            _write_valid_routing_layer(project_dir)
            runtime_dir = project_dir / "runtime"
            contract = build_turn_file_contract(
                runtime_dir=runtime_dir,
                work_dir=project_dir,
                turn_id="create_routing_layer_1",
                phase=PHASE_ROUTING_LAYER_CREATE,
                required_artifacts=ROUTING_LAYER_REQUIRED_FILES,
            )
            first = contract.validator(contract.status_path)
            first_payload = json.loads(Path(first.status_path).read_text(encoding="utf-8"))
            (project_dir / "AGENTS.md").write_text("AGENTS:v2\n", encoding="utf-8")
            second = contract.validator(contract.status_path)
            second_payload = json.loads(Path(second.status_path).read_text(encoding="utf-8"))
            self.assertNotEqual(
                first_payload["artifact_hashes"]["AGENTS.md"],
                second_payload["artifact_hashes"]["AGENTS.md"],
            )
            self.assertEqual(
                second_payload["artifact_hashes"]["AGENTS.md"],
                build_prefixed_sha256(project_dir / "AGENTS.md"),
            )

    def test_build_turn_file_contract_requires_refine_change_from_baseline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir).resolve()
            _write_valid_routing_layer(project_dir)
            runtime_dir = project_dir / "runtime"
            baseline_hashes = {
                relative: build_prefixed_sha256(project_dir / relative)
                for relative in ROUTING_LAYER_REQUIRED_FILES
            }
            contract = build_turn_file_contract(
                runtime_dir=runtime_dir,
                work_dir=project_dir,
                turn_id="refine_routing_layer_1",
                phase=PHASE_ROUTING_LAYER_REFINE,
                required_artifacts=ROUTING_LAYER_REQUIRED_FILES,
                baseline_artifact_hashes=baseline_hashes,
                require_artifact_change=True,
            )
            with self.assertRaisesRegex(ValueError, "required_artifact_change_missing"):
                contract.validator(contract.status_path)
            (project_dir / "AGENTS.md").write_text("AGENTS:v2\n", encoding="utf-8")
            result = contract.validator(contract.status_path)
            payload = json.loads(Path(result.status_path).read_text(encoding="utf-8"))
            self.assertEqual(
                payload["artifact_hashes"]["AGENTS.md"],
                build_prefixed_sha256(project_dir / "AGENTS.md"),
            )

    def test_extract_protocol_token_uses_last_matching_line(self):
        text = "\n".join(["line 1", ROUTING_AUDIT_PASS_TOKEN, "extra tail"])
        token = extract_protocol_token(
            text,
            [ROUTING_AUDIT_PASS_TOKEN, ROUTING_AUDIT_REVISE_TOKEN],
        )
        self.assertEqual(token, "")

    def test_audit_output_requires_revise_when_semantics_conflict_with_pass(self):
        audit_text = "\n".join(
            [
                "- verdict: strong",
                "- recommendation: use_as_is",
                "- missing: area=escalation_conditions | effect=agents may not know when to stop | direction=add stop_and_verify_when",
                ROUTING_AUDIT_PASS_TOKEN,
            ]
        )
        self.assertTrue(audit_output_requires_revise(audit_text))

        clean_pass = ROUTING_AUDIT_PASS_TOKEN
        self.assertFalse(audit_output_requires_revise(clean_pass))
        self.assertTrue(audit_pass_has_extra_text(f"extra prose\n{ROUTING_AUDIT_PASS_TOKEN}"))
        self.assertFalse(audit_pass_has_extra_text(ROUTING_AUDIT_PASS_TOKEN))

    def test_normalize_audit_output_keeps_only_final_revise_body_or_pass_token(self):
        revise_raw = "\n".join(
            [
                "⠼ Thinking... (esc to cancel, 14s)",
                "* - missing: area=escalation_conditions | effect=agents may not know when to stop | direction=add stop_and_verify_when",
                "- recommendation: minor_structural_fixes",
                ROUTING_AUDIT_REVISE_TOKEN,
            ]
        )
        normalized_revise = normalize_audit_output(revise_raw, ROUTING_AUDIT_REVISE_TOKEN)
        self.assertEqual(
            normalized_revise,
            "\n".join(
                [
                    "- missing: area=escalation_conditions | effect=agents may not know when to stop | direction=add stop_and_verify_when",
                    "- recommendation: minor_structural_fixes",
                    ROUTING_AUDIT_REVISE_TOKEN,
                ]
            ),
        )
        self.assertEqual(normalize_audit_output("noise\n[[ROUTING_AUDIT:PASS]]", ROUTING_AUDIT_PASS_TOKEN), ROUTING_AUDIT_PASS_TOKEN)
        self.assertEqual(
            prepare_revise_audit_output(
                "\n".join(
                    [
                        "⠼ Thinking... (esc to cancel, 14s)",
                        "*   Type your message or @path/to/file",
                        "- verdict: usable_but_drift_prone",
                        "- missing: area=escalation_conditions | effect=agents may not know when to stop | direction=add stop_and_verify_when",
                        "- recommendation: minor_structural_fixes",
                        ROUTING_AUDIT_REVISE_TOKEN,
                    ]
                )
            ),
            "\n".join(
                [
                    "- verdict: usable_but_drift_prone",
                    "- missing: area=escalation_conditions | effect=agents may not know when to stop | direction=add stop_and_verify_when",
                    "- recommendation: minor_structural_fixes",
                    ROUTING_AUDIT_REVISE_TOKEN,
                ]
            ),
        )
        self.assertEqual(
            prepare_revise_audit_output(
                "\n".join(
                    [
                        "- verdict: usable_but_drift_prone",
                        "note: keep this final reviewer line",
                        "[[ROUTING_AUDIT:REVISE]]",
                    ]
                )
            ),
            "\n".join(
                [
                    "- verdict: usable_but_drift_prone",
                    "note: keep this final reviewer line",
                    ROUTING_AUDIT_REVISE_TOKEN,
                ]
            ),
        )

    def test_load_routing_audit_decision_reads_status_json_and_review_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir).resolve()
            (project_dir / ROUTING_AUDIT_RECORD_FILE).write_text(
                "- verdict: weak\n- recommendation: major_structural_redesign\n",
                encoding="utf-8",
            )
            (project_dir / ROUTING_AUDIT_STATUS_FILE).write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "stage": PHASE_ROUTING_LAYER_AUDIT,
                        "audit_round": 2,
                        "status": ROUTING_AUDIT_STATUS_FAIL,
                        "review_record_path": ROUTING_AUDIT_RECORD_FILE,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            decision = load_routing_audit_decision(project_dir, expected_round=2)
            self.assertEqual(decision.status, ROUTING_AUDIT_STATUS_FAIL)
            self.assertIn("- verdict: weak", decision.record_text)
            self.assertEqual(Path(decision.record_path).name, ROUTING_AUDIT_RECORD_FILE)

    def test_refine_prompt_disables_inner_review_loop(self):
        refine_prompt = build_refine_prompt("/tmp/project/路由层审核记录.md")
        self.assertIn("不要启动 subagent", refine_prompt)
        self.assertIn("下一轮审核由外层 Python 编排器负责", refine_prompt)
        self.assertIn("/tmp/project/路由层审核记录.md", refine_prompt)
        self.assertIn("系统会在你完成后重新执行 audit", refine_prompt)

    def test_has_overlapping_scope_paths_detects_nested_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            calc_dir = (project_dir / "calc").resolve()
            other_dir = (Path(tmpdir) / "other").resolve()
            calc_dir.mkdir(parents=True)
            other_dir.mkdir(parents=True)
            self.assertTrue(has_overlapping_scope_paths([project_dir, calc_dir]))
            self.assertFalse(has_overlapping_scope_paths([project_dir, other_dir]))

    def test_determine_batch_worker_count_keeps_parallelism_for_nested_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            calc_dir = (project_dir / "calc").resolve()
            calc_dir.mkdir(parents=True)
            self.assertEqual(determine_batch_worker_count([project_dir, calc_dir], max_workers=4), 4)

    def test_run_batch_initialization_passes_after_refine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            FakeWorker.scripts = {
                str(project_dir): [
                    {
                        "label": "create_routing_layer",
                        "output": "",
                        "create_files": ROUTING_LAYER_REQUIRED_FILES,
                    },
                    {
                        "label": "audit_routing_layer_1",
                        "output": "",
                        "write_files": self._audit_write_files(
                            round_index=1,
                            status=ROUTING_AUDIT_STATUS_FAIL,
                            record_text="\n".join(
                                [
                                    "- verdict: usable_but_drift_prone",
                                    "- finding: severity=medium | title=ownership drift | problem=task routing lacks escalation branch | impact=agents may stop too late | evidence=docs/task_routes.json | direction=add stop_and_verify_when",
                                    "- missing: area=escalation_conditions | effect=agents may not know when to stop | direction=add stop_and_verify_when",
                                    "- recommendation: minor_structural_fixes",
                                    "- top_priority: add escalation branch",
                                    "- top_priority: preserve ID stability",
                                    "- top_priority: keep subtree_only",
                                ]
                            ),
                        ),
                    },
                    {
                        "label": "refine_routing_layer_1",
                        "output": "",
                        "write_files": {"AGENTS.md": "refined\n"},
                    },
                    {
                        "label": "audit_routing_layer_2",
                        "output": "",
                        "write_files": self._audit_write_files(
                            round_index=2,
                            status=ROUTING_AUDIT_STATUS_PASS,
                            record_text="- status: 审核通过\n- note: routing layer can be used as-is\n",
                        ),
                    },
                ]
            }
            selection = resolve_target_selection(project_dir=project_dir, run_init=True)
            config = AgentRunConfig(vendor="codex", model="gpt-5")
            result = run_batch_initialization(
                selection=selection,
                config=config,
                runtime_root=Path(tmpdir) / "runtime",
                worker_factory=FakeWorker,
            )
            self.assertEqual(len(result.results), 1)
            directory_result = result.results[0]
            self.assertEqual(directory_result.status, "passed")
            self.assertEqual(directory_result.rounds_used, 1)
            self.assertFalse(directory_result.missing_after)

    def test_run_batch_initialization_rejects_inconsistent_pass_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            FakeWorker.scripts = {
                str(project_dir): [
                    {
                        "label": "create_routing_layer",
                        "output": "",
                        "create_files": ROUTING_LAYER_REQUIRED_FILES,
                    },
                    {
                        "label": "audit_routing_layer_1",
                        "output": "",
                        "write_files": self._audit_write_files(
                            round_index=1,
                            status=ROUTING_AUDIT_STATUS_PASS,
                            record_text="- status: 审核通过\n- note: routing layer can be used as-is\n",
                        ),
                    },
                ]
            }
            selection = resolve_target_selection(project_dir=project_dir, run_init=True)
            config = AgentRunConfig(vendor="codex", model="gpt-5")
            result = run_batch_initialization(
                selection=selection,
                config=config,
                runtime_root=Path(tmpdir) / "runtime",
                worker_factory=FakeWorker,
            )
            self.assertEqual(len(result.results), 1)
            directory_result = result.results[0]
            self.assertEqual(directory_result.status, "passed")

    def test_run_directory_initialization_normalizes_revise_output_before_refine_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            FakeWorker.scripts = {
                str(project_dir): [
                    {
                        "label": "create_routing_layer",
                        "output": "",
                        "create_files": ROUTING_LAYER_REQUIRED_FILES,
                    },
                    {
                        "label": "audit_routing_layer_1",
                        "output": "",
                        "write_files": self._audit_write_files(
                            round_index=1,
                            status=ROUTING_AUDIT_STATUS_FAIL,
                            record_text="\n".join(
                                [
                                    "- verdict: usable_but_drift_prone",
                                    "- missing: area=escalation_conditions | effect=Agents may not know when to stop | direction=Add stop_and_verify_when.",
                                    "- recommendation: minor_structural_fixes",
                                ]
                            ),
                        ),
                    },
                    {
                        "label": "refine_routing_layer_1",
                        "output": "",
                        "write_files": {"AGENTS.md": "refined\n"},
                    },
                    {
                        "label": "audit_routing_layer_2",
                        "output": "",
                        "write_files": self._audit_write_files(
                            round_index=2,
                            status=ROUTING_AUDIT_STATUS_PASS,
                            record_text="- status: 审核通过\n- note: routing layer can be used as-is\n",
                        ),
                    },
                ]
            }
            worker = FakeWorker(
                worker_id="project",
                work_dir=project_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5"),
                runtime_root=Path(tmpdir) / "runtime",
            )
            result = run_directory_initialization_with_worker(worker=worker, forced=True)
            self.assertEqual(result.status, "passed")
            refine_prompt = worker.results[2].command
            self.assertIn(str(project_dir / ROUTING_AUDIT_RECORD_FILE), refine_prompt)
            self.assertNotIn("- verdict: usable_but_drift_prone", refine_prompt)
            self.assertNotIn(ROUTING_AUDIT_REVISE_TOKEN, refine_prompt)

    def test_run_batch_initialization_fails_when_audit_json_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            FakeWorker.scripts = {
                str(project_dir): [
                    {
                        "label": "create_routing_layer",
                        "output": "",
                        "create_files": ROUTING_LAYER_REQUIRED_FILES,
                    },
                    {
                        "label": "audit_routing_layer_1",
                        "output": "",
                    },
                ]
            }
            selection = resolve_target_selection(project_dir=project_dir, run_init=True)
            config = AgentRunConfig(vendor="codex", model="gpt-5")
            result = run_batch_initialization(
                selection=selection,
                config=config,
                runtime_root=Path(tmpdir) / "runtime",
                worker_factory=FakeWorker,
            )
            self.assertEqual(result.results[0].status, "failed")
            self.assertIn("audit_turn_status_invalid", result.results[0].failure_reason)

    def test_run_batch_initialization_fails_when_audit_json_round_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            FakeWorker.scripts = {
                str(project_dir): [
                    {
                        "label": "create_routing_layer",
                        "output": "",
                        "create_files": ROUTING_LAYER_REQUIRED_FILES,
                    },
                    {
                        "label": "audit_routing_layer_1",
                        "output": "",
                        "write_files": self._audit_write_files(
                            round_index=99,
                            status=ROUTING_AUDIT_STATUS_PASS,
                            record_text="- status: 审核通过\n",
                        ),
                    },
                ]
            }
            selection = resolve_target_selection(project_dir=project_dir, run_init=True)
            config = AgentRunConfig(vendor="codex", model="gpt-5")
            result = run_batch_initialization(
                selection=selection,
                config=config,
                runtime_root=Path(tmpdir) / "runtime",
                worker_factory=FakeWorker,
            )
            self.assertEqual(result.results[0].status, "failed")
            self.assertIn("audit_status_invalid", result.results[0].failure_reason)

    def test_run_batch_initialization_materializes_audit_turn_status_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            FakeWorker.scripts = {
                str(project_dir): [
                    {
                        "label": "create_routing_layer",
                        "output": "",
                        "create_files": ROUTING_LAYER_REQUIRED_FILES,
                    },
                    {
                        "label": "audit_routing_layer_1",
                        "output": "some prose before final token",
                        "skip_turn_status": True,
                        "write_files": self._audit_write_files(
                            round_index=1,
                            status=ROUTING_AUDIT_STATUS_FAIL,
                            record_text="- verdict: weak\n- recommendation: major_structural_redesign\n",
                        ),
                    },
                    {
                        "label": "refine_routing_layer_1",
                        "output": "",
                        "write_files": {"AGENTS.md": "refined\n"},
                    },
                    {
                        "label": "audit_routing_layer_2",
                        "output": "",
                        "write_files": self._audit_write_files(
                            round_index=2,
                            status=ROUTING_AUDIT_STATUS_PASS,
                            record_text="- status: 审核通过\n",
                        ),
                    },
                ]
            }
            selection = resolve_target_selection(project_dir=project_dir, run_init=True)
            config = AgentRunConfig(vendor="codex", model="gpt-5")
            result = run_batch_initialization(
                selection=selection,
                config=config,
                runtime_root=Path(tmpdir) / "runtime",
                worker_factory=FakeWorker,
            )
            self.assertEqual(result.results[0].status, "passed")

    def test_run_batch_initialization_fails_when_refine_leaves_routing_files_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            FakeWorker.scripts = {
                str(project_dir): [
                    {
                        "label": "create_routing_layer",
                        "output": "",
                        "create_files": ROUTING_LAYER_REQUIRED_FILES,
                    },
                    {
                        "label": "audit_routing_layer_1",
                        "output": "",
                        "write_files": self._audit_write_files(
                            round_index=1,
                            status=ROUTING_AUDIT_STATUS_FAIL,
                            record_text="- verdict: usable_but_drift_prone\n- recommendation: minor_structural_fixes\n",
                        ),
                    },
                    {
                        "label": "refine_routing_layer_1",
                        "output": "",
                    },
                ]
            }
            selection = resolve_target_selection(project_dir=project_dir, run_init=True)
            config = AgentRunConfig(vendor="codex", model="gpt-5")
            result = run_batch_initialization(
                selection=selection,
                config=config,
                runtime_root=Path(tmpdir) / "runtime",
                worker_factory=FakeWorker,
            )
            self.assertEqual(result.results[0].status, "failed")
            self.assertIn("required_artifact_change_missing", result.results[0].failure_reason)

    def test_run_batch_initialization_fails_when_create_rerun_leaves_existing_routing_files_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            _write_valid_routing_layer(project_dir)
            FakeWorker.scripts = {
                str(project_dir): [
                    {
                        "label": "create_routing_layer",
                        "output": "",
                    },
                ]
            }
            selection = resolve_target_selection(project_dir=project_dir, run_init=True)
            config = AgentRunConfig(vendor="codex", model="gpt-5")
            result = run_batch_initialization(
                selection=selection,
                config=config,
                runtime_root=Path(tmpdir) / "runtime",
                worker_factory=FakeWorker,
            )
            self.assertEqual(result.results[0].status, "failed")
            self.assertIn("create_turn_status_invalid", result.results[0].failure_reason)
            self.assertIn("required_artifact_change_missing", result.results[0].failure_reason)

    def test_run_batch_initialization_includes_command_output_in_failure_reason(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            FakeWorker.scripts = {
                str(project_dir): [
                    {
                        "label": "create_routing_layer",
                        "output": "",
                        "create_files": ROUTING_LAYER_REQUIRED_FILES,
                    },
                    {
                        "label": "audit_routing_layer_1",
                        "output": "tmux pane exited while waiting for turn artifacts",
                        "exit_code": 1,
                    },
                ]
            }
            selection = resolve_target_selection(project_dir=project_dir, run_init=True)
            config = AgentRunConfig(vendor="codex", model="gpt-5")
            result = run_batch_initialization(
                selection=selection,
                config=config,
                runtime_root=Path(tmpdir) / "runtime",
                worker_factory=FakeWorker,
            )
            self.assertEqual(result.results[0].status, "failed")
            self.assertIn("audit_command_failed", result.results[0].failure_reason)
            self.assertIn("tmux pane exited while waiting for turn artifacts", result.results[0].failure_reason)

    def test_run_batch_initialization_marks_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            _write_valid_routing_layer(project_dir)

            selection = resolve_target_selection(project_dir=project_dir, run_init=False)
            config = AgentRunConfig(vendor="codex", model="gpt-5")
            result = run_batch_initialization(
                selection=selection,
                config=config,
                runtime_root=Path(tmpdir) / "runtime",
                worker_factory=FakeWorker,
            )
            self.assertEqual(result.results[0].status, "skipped")

    def test_run_batch_initialization_fails_when_create_turn_status_leaves_missing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            FakeWorker.scripts = {
                str(project_dir): [
                    {
                        "label": "create_routing_layer",
                        "output": "",
                        "create_files": ("AGENTS.md",),
                    },
                    {
                        "label": "audit_routing_layer_1",
                        "output": "",
                        "write_files": self._audit_write_files(
                            round_index=1,
                            status=ROUTING_AUDIT_STATUS_PASS,
                            record_text="- status: 审核通过\n",
                        ),
                    },
                ]
            }
            selection = resolve_target_selection(project_dir=project_dir, run_init=True)
            config = AgentRunConfig(vendor="codex", model="gpt-5")
            result = run_batch_initialization(
                selection=selection,
                config=config,
                runtime_root=Path(tmpdir) / "runtime",
                worker_factory=FakeWorker,
            )
            self.assertEqual(result.results[0].status, "failed")
            self.assertIn("create_turn_status_invalid", result.results[0].failure_reason)

    def test_run_batch_initialization_isolates_worker_factory_failure(self):
        class BrokenWorker:
            def __init__(self, **kwargs):
                raise RuntimeError("worker init failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            selection = resolve_target_selection(project_dir=project_dir, run_init=True)
            config = AgentRunConfig(vendor="codex", model="gpt-5")
            result = run_batch_initialization(
                selection=selection,
                config=config,
                runtime_root=Path(tmpdir) / "runtime",
                worker_factory=BrokenWorker,
            )
            self.assertEqual(result.results[0].status, "failed")
            self.assertIn("worker init failed", result.results[0].failure_reason)

    def test_prepare_live_workers_writes_manifest_immediately(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            selection = resolve_target_selection(project_dir=project_dir, run_init=True)
            config = AgentRunConfig(vendor="codex", model="gpt-5")
            store, live_workers, immediate_results = prepare_live_workers(
                selection=selection,
                config=config,
                runtime_root=Path(tmpdir) / "runtime",
                worker_factory=FakeWorker,
            )
            manifest_path = store.run_root / "manifest.json"
            events_path = store.run_root / "events.jsonl"
            self.assertTrue(manifest_path.exists())
            self.assertTrue(events_path.exists())
            self.assertEqual(len(live_workers), 1)
            self.assertFalse(immediate_results)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_id"], store.manifest.run_id)
            self.assertEqual(manifest["workers"][0]["work_dir"], str(project_dir))
            self.assertEqual(manifest["workers"][0]["result_status"], "pending")
            self.assertEqual(manifest["workers"][0]["agent_state"], AgentRuntimeState.STARTING.value)

    def test_run_store_write_manifest_allows_concurrent_loaded_instances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            selection = resolve_target_selection(project_dir=project_dir, run_init=True)
            config = AgentRunConfig(vendor="codex", model="gpt-5")
            store = RunStore.create(
                selection=selection,
                config=config,
                runtime_root=Path(tmpdir) / "runtime",
                run_id="run_demo",
            )
            loaded_stores = [
                RunStore.load(run_id=store.manifest.run_id, runtime_root=Path(tmpdir) / "runtime"),
                RunStore.load(run_id=store.manifest.run_id, runtime_root=Path(tmpdir) / "runtime"),
            ]
            original_write_text = Path.write_text
            write_barrier = threading.Barrier(2)

            def delayed_write_text(path, data, *args, **kwargs):  # noqa: ANN001
                result = original_write_text(path, data, *args, **kwargs)
                if str(Path(path).name).startswith("manifest.json.tmp."):
                    write_barrier.wait(timeout=5.0)
                return result

            def write_loaded_store(index: int) -> Path:
                loaded_stores[index].manifest.status = f"running-{index}"
                return loaded_stores[index].write_manifest()

            with patch.object(Path, "write_text", delayed_write_text):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(write_loaded_store, index) for index in range(2)]
                    for future in futures:
                        self.assertEqual(future.result(timeout=5), store.manifest_path)

            payload = json.loads(store.manifest_path.read_text(encoding="utf-8"))
            self.assertIn(payload["status"], {"running-0", "running-1"})
            self.assertFalse(list(store.run_root.glob("manifest.json.tmp.*")))

    def test_run_store_preserves_active_prelaunch_starting_from_dead_state_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            selection = resolve_target_selection(project_dir=project_dir, run_init=True)
            config = AgentRunConfig(vendor="codex", model="gpt-5")
            store = RunStore.create(
                selection=selection,
                config=config,
                runtime_root=Path(tmpdir) / "runtime",
                run_id="run_demo",
            )
            state_path = Path(tmpdir) / "worker.state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "session_name": "sess-launching",
                        "pane_id": "",
                        "runtime_dir": str(state_path.parent),
                        "result_status": "running",
                        "agent_state": "DEAD",
                        "agent_alive": False,
                        "agent_started": False,
                        "health_status": "missing_session",
                        "health_note": "missing_session",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            store.update_worker_binding(
                str(project_dir),
                session_name="sess-launching",
                pane_id="",
                workflow_stage="create_running",
                result_status="running",
                agent_state=AgentRuntimeState.STARTING.value,
                agent_started=False,
                health_status="unknown",
                note="create_routing_layer",
                state_path=str(state_path),
            )

            entry = store.update_worker_state_from_file(
                str(project_dir),
                state_path,
                preserve_workflow_fields=True,
            )

        self.assertIsNotNone(entry)
        self.assertEqual(entry.agent_state, AgentRuntimeState.STARTING.value)
        self.assertEqual(entry.health_status, "unknown")
        self.assertEqual(entry.health_note, "launch pending")

    def test_run_store_keeps_dead_state_for_launched_active_worker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            selection = resolve_target_selection(project_dir=project_dir, run_init=True)
            config = AgentRunConfig(vendor="codex", model="gpt-5")
            store = RunStore.create(
                selection=selection,
                config=config,
                runtime_root=Path(tmpdir) / "runtime",
                run_id="run_demo",
            )
            state_path = Path(tmpdir) / "worker.state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "session_name": "sess-launched",
                        "pane_id": "%1",
                        "runtime_dir": str(state_path.parent),
                        "result_status": "running",
                        "agent_state": "DEAD",
                        "agent_alive": False,
                        "agent_started": True,
                        "health_status": "missing_session",
                        "health_note": "missing_session",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            store.update_worker_binding(
                str(project_dir),
                session_name="sess-launched",
                pane_id="%1",
                workflow_stage="create_running",
                result_status="running",
                agent_state=AgentRuntimeState.READY.value,
                agent_started=True,
                health_status="alive",
                note="create_routing_layer",
                state_path=str(state_path),
            )

            entry = store.update_worker_state_from_file(
                str(project_dir),
                state_path,
                preserve_workflow_fields=True,
            )

        self.assertIsNotNone(entry)
        self.assertEqual(entry.agent_state, AgentRuntimeState.DEAD.value)
        self.assertEqual(entry.health_status, "missing_session")

    def test_run_store_marks_ready_agent_busy_while_routing_turn_is_active(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            selection = resolve_target_selection(project_dir=project_dir, run_init=True)
            config = AgentRunConfig(vendor="codex", model="gpt-5")
            store = RunStore.create(
                selection=selection,
                config=config,
                runtime_root=Path(tmpdir) / "runtime",
                run_id="run_demo",
            )
            state_path = Path(tmpdir) / "worker.state.json"
            turn_status_path = Path(tmpdir) / "turn_status.json"
            state_path.write_text(
                json.dumps(
                    {
                        "session_name": "sess-routing",
                        "pane_id": "%1",
                        "runtime_dir": str(state_path.parent),
                        "result_status": "running",
                        "agent_state": "READY",
                        "agent_alive": True,
                        "agent_started": True,
                        "current_task_runtime_status": "running",
                        "current_turn_status_path": str(turn_status_path),
                        "health_status": "alive",
                        "health_note": "alive",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            store.update_worker_binding(
                str(project_dir),
                session_name="sess-routing",
                pane_id="%1",
                workflow_stage="create_running",
                result_status="running",
                agent_state=AgentRuntimeState.READY.value,
                agent_started=True,
                health_status="alive",
                note="create_routing_layer",
                state_path=str(state_path),
                current_turn_status_path=str(turn_status_path),
            )

            entry = store.update_worker_state_from_file(
                str(project_dir),
                state_path,
                preserve_workflow_fields=True,
            )

        self.assertIsNotNone(entry)
        self.assertEqual(entry.agent_state, AgentRuntimeState.BUSY.value)
        self.assertEqual(entry.current_task_runtime_status, "running")

    def test_run_store_load_maps_legacy_worker_manifest_phase_to_agent_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            selection = resolve_target_selection(project_dir=project_dir, run_init=True)
            config = AgentRunConfig(vendor="codex", model="gpt-5")
            store, _, _ = prepare_live_workers(
                selection=selection,
                config=config,
                runtime_root=Path(tmpdir) / "runtime",
                worker_factory=FakeWorker,
            )
            manifest_path = store.run_root / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["workers"][0].pop("agent_state", None)
            payload["workers"][0].pop("agent_started", None)
            payload["workers"][0]["agent_ready"] = True
            payload["workers"][0]["current_command"] = "codex"
            payload["workers"][0]["provider_phase"] = "waiting_input"
            manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            loaded = RunStore.load(run_id=store.manifest.run_id, runtime_root=Path(tmpdir) / "runtime")

        self.assertEqual(loaded.manifest.workers[0].agent_state, "READY")
        self.assertTrue(loaded.manifest.workers[0].agent_started)

    def test_run_store_load_maps_prelaunch_dead_manifest_worker_to_starting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            selection = resolve_target_selection(project_dir=project_dir, run_init=True)
            config = AgentRunConfig(vendor="codex", model="gpt-5")
            store = RunStore.create(
                selection=selection,
                config=config,
                runtime_root=Path(tmpdir) / "runtime",
                run_id="run_demo",
            )
            payload = json.loads((store.run_root / "manifest.json").read_text(encoding="utf-8"))
            payload["workers"] = [
                {
                    "work_dir": str(project_dir),
                    "session_name": "sess-prelaunch",
                    "pane_id": "",
                    "workflow_stage": "create_running",
                    "result_status": "running",
                    "agent_state": "DEAD",
                    "agent_started": False,
                    "agent_alive": False,
                    "health_status": "missing_session",
                    "health_note": "missing_session",
                }
            ]
            (store.run_root / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            loaded = RunStore.load(run_id=store.manifest.run_id, runtime_root=Path(tmpdir) / "runtime")

        self.assertEqual(loaded.manifest.workers[0].agent_state, "STARTING")
        self.assertFalse(loaded.manifest.workers[0].agent_started)

    def test_run_store_defaults_to_project_local_routing_runtime_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            selection = resolve_target_selection(project_dir=project_dir, run_init=True)
            config = AgentRunConfig(vendor="codex", model="gpt-5")

            store = RunStore.create(
                selection=selection,
                config=config,
                run_id="run_demo",
            )
            loaded = RunStore.load(run_id="run_demo", project_dir=project_dir)
            expected_root = project_dir / ROUTING_RUNTIME_ROOT_NAME / "run_demo"
            self.assertEqual(store.run_root, expected_root)
            self.assertEqual(loaded.run_root, expected_root)
            self.assertEqual(build_routing_runtime_root(project_dir), project_dir / ROUTING_RUNTIME_ROOT_NAME)

    def test_load_existing_run_uses_project_local_runtime_root_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            selection = resolve_target_selection(project_dir=project_dir, run_init=True)
            config = AgentRunConfig(vendor="codex", model="gpt-5")
            store, live_workers, _ = prepare_live_workers(
                selection=selection,
                config=config,
                worker_factory=FakeWorker,
            )
            alive_session = live_workers[0].session_name
            FakeWorker.alive_sessions = {alive_session: True}

            loaded_store, loaded_selection, loaded_config, resumed_live_workers, _ = load_existing_run(
                run_id=store.manifest.run_id,
                project_dir=project_dir,
                worker_factory=FakeWorker,
            )

        self.assertEqual(loaded_store.run_root.parent, project_dir / ROUTING_RUNTIME_ROOT_NAME)
        self.assertEqual(loaded_selection.project_dir, str(project_dir))
        self.assertEqual(loaded_config.vendor.value, "codex")
        self.assertEqual([item.session_name for item in resumed_live_workers], [alive_session])

    def test_load_existing_run_binds_alive_sessions_and_marks_dead_sessions_stale_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            calc_dir = (project_dir / "calc").resolve()
            calc_dir.mkdir(parents=True)
            selection = resolve_target_selection(project_dir=project_dir, target_dirs=("calc",), run_init=True)
            config = AgentRunConfig(vendor="codex", model="gpt-5")
            store, live_workers, _ = prepare_live_workers(
                selection=selection,
                config=config,
                runtime_root=Path(tmpdir) / "runtime",
                worker_factory=FakeWorker,
            )
            self.assertEqual(len(live_workers), 2)
            alive_session = live_workers[0].session_name
            dead_session = live_workers[1].session_name
            FakeWorker.alive_sessions = {alive_session: True, dead_session: False}

            loaded_store, loaded_selection, loaded_config, resumed_live_workers, results_by_dir = load_existing_run(
                run_id=store.manifest.run_id,
                runtime_root=Path(tmpdir) / "runtime",
                worker_factory=FakeWorker,
            )
            self.assertEqual(loaded_selection.selected_dirs, selection.selected_dirs)
            self.assertEqual(loaded_config.vendor.value, "codex")
            self.assertEqual([item.session_name for item in resumed_live_workers], [alive_session])
            self.assertIn(live_workers[1].work_dir, results_by_dir)
            self.assertEqual(results_by_dir[live_workers[1].work_dir].failure_reason, "stale_failed")
            manifest_entry = loaded_store.ensure_worker(work_dir=live_workers[1].work_dir)
            self.assertEqual(manifest_entry.result_status, "stale_failed")

    def test_load_existing_run_keeps_create_running_live_when_session_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            selection = resolve_target_selection(project_dir=project_dir, run_init=True)
            config = AgentRunConfig(vendor="codex", model="gpt-5")
            store, live_workers, _ = prepare_live_workers(
                selection=selection,
                config=config,
                runtime_root=Path(tmpdir) / "runtime",
                worker_factory=FakeWorker,
            )
            handle = live_workers[0]
            store.update_worker_binding(
                handle.work_dir,
                workflow_stage="create_running",
                workflow_round=0,
                result_status="running",
                session_name=handle.session_name,
                runtime_dir=str(handle.worker.runtime_dir),
                pane_id=handle.worker.pane_id,
                current_turn_id="create_routing_layer_1",
                current_turn_phase=PHASE_ROUTING_LAYER_CREATE,
                current_turn_status_path=str(routing_turn_status_path(handle.worker.runtime_dir, "create_routing_layer_1")),
            )
            FakeWorker.alive_sessions = {handle.session_name: True}

            _, _, _, resumed_live_workers, results_by_dir = load_existing_run(
                run_id=store.manifest.run_id,
                runtime_root=Path(tmpdir) / "runtime",
                worker_factory=FakeWorker,
            )
            self.assertEqual([item.session_name for item in resumed_live_workers], [handle.session_name])
            self.assertFalse(results_by_dir)

    def test_run_directory_initialization_resumes_from_refine_pending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = (Path(tmpdir) / "project").resolve()
            project_dir.mkdir(parents=True)
            _write_valid_routing_layer(project_dir)
            (project_dir / ROUTING_AUDIT_RECORD_FILE).write_text(
                "- verdict: usable_but_drift_prone\n- missing: area=escalation_conditions | effect=agents may not know when to stop | direction=add stop_and_verify_when\n- recommendation: minor_structural_fixes\n",
                encoding="utf-8",
            )
            (project_dir / ROUTING_AUDIT_STATUS_FILE).write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "stage": PHASE_ROUTING_LAYER_AUDIT,
                        "audit_round": 1,
                        "status": ROUTING_AUDIT_STATUS_FAIL,
                        "review_record_path": ROUTING_AUDIT_RECORD_FILE,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            FakeWorker.scripts = {
                str(project_dir): [
                    {
                        "label": "refine_routing_layer_1",
                        "output": "",
                        "write_files": {"AGENTS.md": "refined\n"},
                    },
                    {
                        "label": "audit_routing_layer_2",
                        "output": "",
                        "write_files": self._audit_write_files(
                            round_index=2,
                            status=ROUTING_AUDIT_STATUS_PASS,
                            record_text="- status: 审核通过\n",
                        ),
                    },
                ]
            }
            worker = FakeWorker(
                worker_id="project",
                work_dir=project_dir,
                config=AgentRunConfig(vendor="codex", model="gpt-5"),
                runtime_root=Path(tmpdir) / "runtime",
            )
            result = run_directory_initialization_with_worker(
                worker=worker,
                forced=True,
                resume_state={
                    "workflow_stage": "refine_pending",
                    "workflow_round": 1,
                    "last_audit_token": ROUTING_AUDIT_REVISE_TOKEN,
                },
            )
            self.assertEqual(result.status, "passed")
            self.assertEqual(result.rounds_used, 1)
            self.assertEqual([item.label for item in worker.results], ["refine_routing_layer_1", "audit_routing_layer_2"])


if __name__ == "__main__":
    unittest.main()
