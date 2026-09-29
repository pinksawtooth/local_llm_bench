from __future__ import annotations

import copy
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .conditions import capture_conditions, comparison_metadata, condition_differences, execution_contract, unknown_conditions
from .persistence import atomic_write_json
from .diagnostics import host_snapshot


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunExecution:
    """Durable attempt/question results and the model lifetime for one run."""

    def __init__(self, config, runtime, model: str, parallelism=None, *, resume_id=None, retry_id=None, questions=None, allow_unverified=False):
        self.config, self.runtime, self.model, self.parallelism = config, runtime, model, parallelism
        self.run_id = resume_id or retry_id or uuid.uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.run_id):
            raise ValueError("run ID が不正です。")
        self.directory = config.output.run_logs_dir / self.run_id
        self.checkpoint = self.directory / "checkpoint.json"
        self.contract = execution_contract(config, model, parallelism)
        self.protocol = self.contract["protocol"]
        self.externally_managed = config.provider == "ds4" and self.contract["ds4_launch_config"]["management"] == "external"
        if self.externally_managed and config.runs.cold_runs:
            raise ValueError("ds4 は外部管理です。cold_runs: 0 を指定してください。")
        self.recovering = bool(resume_id or retry_id)
        self.allow_unverified = allow_unverified
        self.mode = "retry_errors" if retry_id else ("resume" if resume_id else "new")
        self.pending_retries = set()
        if self.recovering:
            if not self.checkpoint.is_file():
                raise ValueError(f"再開用checkpointがありません: {self.run_id}")
            self.state = json.loads(self.checkpoint.read_text())
            if self.state.get("schema_version") != 1:
                raise ValueError("未対応のcheckpoint形式です。")
            self._replay_units()
            if resume_id:
                self.pending_retries = set(self.state.get("pending_retries", []))
            if self.state["contract"] != self.contract:
                fields = ", ".join(item["field"] for item in condition_differences(self.state["contract"], self.contract))
                raise ValueError(f"保存された実行条件と一致しません: {fields}")
            if resume_id and self.state["status"] == "completed" and self.state.get("outputs_saved"):
                raise ValueError("このrunは完了しています。エラーの再実行には --retry-errors-run-id を使用してください。")
            if retry_id:
                from .conditions import workload
                question_ids = [q["id"] for q in workload(config)["questions"]] if config.mode == "docker_task" else ["prompt"]
                if config.mode == "performance":
                    from .performance import scenario_units
                    question_ids = scenario_units(config.performance)
                expected = {self.key(phase, i, qid) for phase, count in (("cold", config.runs.cold_runs), ("warm", config.runs.warm_runs)) for i in range(1, count + 1) for qid in question_ids}
                if expected - self.state["units"].keys():
                    raise ValueError("未完了の問題があります。先に --resume-run-id で完了させてください。")
                failed = {key for key, unit in self.state["units"].items() if unit["result"].get("status") != "success"}
                selected = set(questions or [])
                failed_questions = {json.loads(key)[2] for key in failed}
                if selected - failed_questions:
                    raise ValueError(f"記録済みエラーのないquestion IDです: {sorted(selected - failed_questions)}")
                self.pending_retries = {key for key in failed if not selected or json.loads(key)[2] in selected}
                if not self.pending_retries:
                    raise ValueError("再試行できる記録済みエラーがありません。")
        else:
            self.state = {"schema_version": 1, "run_id": self.run_id, "started_at": utc_now(), "status": "preparing", "contract": self.contract, "units": {}, "segments": [], "lifecycle": []}
            if config.mode == "docker_task":
                from .docker_task.spec import load_spec
                spec = load_spec(config.benchmark_spec_path, config.benchmark_answer_key_path)
                self.state["display"] = {"benchmark_id": spec.id, "benchmark_title": spec.title, "question_count": len(spec.questions), "prompt_text": "\n\n".join(question.prompt for question in spec.questions)}
            elif config.mode == "performance":
                from dataclasses import asdict
                from .performance import WORKLOAD_VERSION
                self.state["display"] = {"benchmark_id": "runtime-performance-v1", "benchmark_title": "入力長別・同時実行性能",
                    "prompt_text": "入力長別の共通Pythonコーパス", "performance": {**asdict(config.performance), "input_length_method": WORKLOAD_VERSION}}
            self._flush()
        self.started_at = self.state["started_at"]
        self.api_model = model
        self.model_info = None
        self.prepared = False
        self.started = False
        self.attempts_started = 0
        self.has_inference = False
        self.attempted_inference = False
        self.segment_started = time.perf_counter()

    @staticmethod
    def key(phase, iteration, question_id):
        return json.dumps([phase, iteration, question_id], ensure_ascii=False)

    def _flush(self):
        self.state["updated_at"] = utc_now()
        atomic_write_json(self.checkpoint, self.state)

    def _replay_units(self):
        """Recover an fsynced result even if checkpoint replacement was interrupted."""
        self.valid_event_bytes = 0
        path = self.directory / "events.jsonl"
        if not path.exists():
            return
        self.state["unit_timings"] = []
        lines = path.read_bytes().splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not line.endswith(b"\n") and index == len(lines) - 1:
                break
            try:
                entry = json.loads(line)
                if entry["event"] == "unit_completed":
                    self.state["units"][entry["data"]["key"]] = entry["data"]["unit"]
                    self._record_unit_timing(entry["data"]["unit"]["result"])
                    if entry["data"]["key"] in self.state.get("pending_retries", []):
                        self.state["pending_retries"].remove(entry["data"]["key"])
                elif entry["event"] == "unit_annotated":
                    self._apply_annotation(entry["data"]["key"], entry["data"]["updates"])
                elif entry["event"] == "retry_plan":
                    self.state["pending_retries"] = entry["data"]["keys"]
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError("イベントログが破損しています。元のログを保持して終了します。") from exc
            self.valid_event_bytes += len(line)

    def event(self, kind: str, data: dict):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at": utc_now(), "event": kind, "data": data}, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def cached(self, phase, iteration, question_id):
        key = self.key(phase, iteration, question_id)
        if key in self.pending_retries:
            return None
        unit = self.state["units"].get(key)
        if unit is None and self.mode == "retry_errors":
            raise ValueError("未完了の問題があります。先に --resume-run-id で完了させてください。")
        return copy.deepcopy(unit)

    def _unload(self):
        results = self.runtime.unload_model(self.api_model)
        failures = [item.message for item in results if item.status != "unloaded"]
        if failures:
            raise RuntimeError("モデルのアンロードに失敗しました: " + "; ".join(failures))
        self.prepared = False

    def _load(self):
        if not self.externally_managed:
            self._unload()
        began = time.perf_counter()
        self.api_model, self.model_info = self.runtime.prepare_model(self.model, lmstudio_parallelism=self.parallelism)
        self.model_info = self.model_info or {}
        self.prepared = True
        if self.model_info.get("load_status") == "already_loaded":
            raise RuntimeError("モデルが既にロード済みと報告されました。cold条件の再ロードを確認できません。")
        self.has_inference = False
        self.attempted_inference = False
        entry = {"event": "attach" if self.externally_managed else "load", "at": utc_now(), "wall_sec": time.perf_counter() - began, "provider_load_sec": self.model_info.get("load_time_seconds"), "cache_state": "unknown", "model": self.api_model}
        self.model_info = self.runtime.measurement_metadata(self.model_info)
        entry["host_after_attach" if self.externally_managed else "host_after_load"] = host_snapshot()
        self.state["lifecycle"].append(entry)
        if self.started:
            self.event(entry["event"], entry)
            self._flush()

    def start(self, preflight: dict):
        self.preflight = preflight
        self.runtime.begin_run(self.directory, ds4_launch_config=self.contract.get("ds4_launch_config"))
        self._load()
        conditions = capture_conditions(self.config, self.model_info, self.contract, preflight)
        previous = self.state.get("conditions")
        if self.recovering and previous:
            differences = condition_differences(previous, conditions)
            if differences:
                fields = ", ".join(item["field"] for item in differences)
                raise ValueError(f"再開時の測定条件が変わっています: {fields}")
            # Reloading starts a fresh cache lifecycle; unknown cache state alone
            # does not block recovery, but still prevents cross-run aggregation.
            unknown = [field for field in unknown_conditions(conditions) if field != "cache.state"]
            if unknown and not self.allow_unverified:
                raise ValueError("実効条件を確認できないため再開できません: " + ", ".join(unknown) + ". 不明条件を許容する場合は --allow-unverified-resume を明示してください。")
        if self.recovering:
            # Preserve the entire prior checkpoint, including successful answers and logs.
            previous_checkpoint = json.loads(self.checkpoint.read_text())
            atomic_write_json(self.directory / "revisions" / f"{uuid.uuid4().hex}.json", previous_checkpoint)
            event_path = self.directory / "events.jsonl"
            if event_path.exists():
                with event_path.open("r+b") as stream:
                    stream.truncate(self.valid_event_bytes)
            for segment in self.state["segments"]:
                if "ended_at" not in segment:
                    segment.update(status="interrupted", wall_sec=None)
        self.started = True
        self.state.update(status="running", conditions=conditions, model_info=self.model_info, preflight=preflight)
        self.state["outputs_saved"] = False
        self.state["pending_retries"] = sorted(self.pending_retries)
        if self.mode == "retry_errors":
            self.event("retry_plan", {"keys": sorted(self.pending_retries)})
        self.state["segments"].append({"started_at": utc_now(), "mode": self.mode, "verification": "incomplete" if unknown_conditions(conditions) else "recorded"})
        self.event("segment_started", {"mode": self.mode, "conditions": conditions})
        self._flush()

    def before_attempt(self, phase: str, iteration: int, warmup: Callable[[str], dict]) -> str:
        if phase == "cold" and self.attempts_started:
            self._load()
            current = capture_conditions(self.config, self.model_info, self.contract, self.preflight)
            if condition_differences(self.state["conditions"], current):
                raise RuntimeError("再ロード時に測定条件が変わりました。")
        # Prime only a freshly loaded model. A failed measured attempt must not
        # trigger an extra, unscored retry of the same workload. Its successors
        # retain unknown_after_error until a measured request succeeds.
        if phase == "warm" and not self.has_inference and not self.attempted_inference:
            began = time.perf_counter()
            result = warmup(self.api_model)
            entry = {"event": "warmup", "at": utc_now(), "wall_sec": time.perf_counter() - began, "excluded_from_statistics": True, "result": result}
            self.state["lifecycle"].append(entry)
            self.event("warmup", entry)
            self._flush()
            if result.get("status", "success") != "success":
                raise RuntimeError("warm測定の準備推論に失敗しました。")
            self.has_inference = True
        self.attempts_started += 1
        return self.api_model

    def save_unit(self, phase, iteration, question_id, result: dict, log: dict):
        key = self.key(phase, iteration, question_id)
        stage = "repeat" if self.has_inference else "unknown_after_error" if self.attempted_inference else "first_after_load"
        if self.externally_managed and stage == "first_after_load":
            stage = "unknown_before_attach"
        measured = result.get("measurement", {}) if self.config.mode == "performance" else {}
        result["measurement"] = {"protocol": self.protocol, "inference_stage": measured.get("inference_stage", stage),
                                 "cache_state": measured.get("cache_state", (result.get("metrics") or {}).get("cache_state", "unknown")),
                                 "segment": len(self.state["segments"])}
        self.attempted_inference = True
        self.has_inference = self.has_inference or result.get("status") == "success"
        unit = {"result": copy.deepcopy(result), "log": copy.deepcopy(log)}
        self.event("unit_completed", {"key": key, "unit": unit})
        self._record_unit_timing(result)
        self.state["units"][key] = unit
        self.pending_retries.discard(key)
        self.state["pending_retries"] = sorted(self.pending_retries)
        self._flush()

    def _apply_annotation(self, key, updates):
        unit = self.state["units"][key]
        unit["result"].update(copy.deepcopy(updates))
        unit["log"].setdefault("response", {}).update(copy.deepcopy(updates))
        if "inspect" in updates:
            unit["log"]["inspect"] = copy.deepcopy(updates["inspect"])

    def annotate_unit(self, phase, iteration, question_id, updates):
        key = self.key(phase, iteration, question_id)
        self.event("unit_annotated", {"key": key, "updates": updates})
        self._apply_annotation(key, updates)
        self._flush()

    def _record_unit_timing(self, result: dict):
        self.state.setdefault("unit_timings", []).append({
            "stage": result.get("measurement", {}).get("inference_stage"),
            "wall_sec": result.get("host_wall_ms", result.get("attempt_wall_ms", 0)) / 1000,
        })

    def finish(self, status: str, error: str | None = None):
        if self.recovering and not self.started:
            return
        self.state["status"] = status
        self.state["error"] = error
        if self.started and self.state["segments"]:
            self.state["segments"][-1].update(ended_at=utc_now(), wall_sec=time.perf_counter() - self.segment_started, status=status)
        self.event("run_status", {"status": status, "error": error})
        self._flush()

    def mark_outputs_saved(self):
        self.state["outputs_saved"] = True
        self._flush()

    def enrich(self, result: dict) -> dict:
        from .conditions import evaluation_identity
        result["evaluation"] = evaluation_identity(self.config)
        conditions = self.state.get("conditions", {})
        result.update(run_id=self.run_id, api_model=self.api_model, schema_version=2, status=self.state["status"], conditions=conditions,
                      comparison=comparison_metadata(conditions, self.run_id, recovered=len(self.state["segments"]) > 1),
                      lifecycle=self.state["lifecycle"], execution_segments=self.state["segments"],
                      preflight=self.state.get("preflight"), model_info=self.state.get("model_info", {}))
        result["timings"] = {
            "load_wall_sec": None if self.externally_managed else sum(item.get("wall_sec", 0) for item in self.state["lifecycle"] if item["event"] == "load"),
            "attach_wall_sec": sum(item.get("wall_sec", 0) for item in self.state["lifecycle"] if item["event"] == "attach"),
            "warmup_wall_sec": sum(item.get("wall_sec", 0) for item in self.state["lifecycle"] if item["event"] == "warmup"),
            "inference_wall_sec": sum(unit["wall_sec"] for unit in self.state.get("unit_timings", [])),
            "inference_scope": "all_completed_units_including_replaced_errors",
            "total_wall_sec": sum(item.get("wall_sec", 0) for item in self.state["segments"]) if all(item.get("wall_sec") is not None for item in self.state["segments"]) else None,
        }
        for stage in ("first_after_load", "repeat", "unknown_after_error", "unknown_before_attach"):
            units = [unit for unit in self.state.get("unit_timings", []) if unit["stage"] == stage]
            result["timings"][f"{stage}_count"] = len(units)
            result["timings"][f"{stage}_wall_sec"] = sum(unit["wall_sec"] for unit in units)
        if self.externally_managed:
            result["timings"]["first_after_load_wall_sec"] = None
        if self.config.mode == "performance":
            # Requests overlap. Their sum is not inference wall-clock time.
            result["timings"]["request_wall_sum_sec"] = result["timings"]["inference_wall_sec"]
            result["timings"]["inference_wall_sec"] = None
            result["timings"]["inference_scope"] = "concurrent_cohorts; see each batch_metrics.elapsed_sec"
        return result

    def partial_result(self) -> dict:
        records, attempts = [], []
        display = self.state.get("display", {})
        prompt_text = display.get("prompt_text", self.config.prompt_text)
        grouped = {}
        for key, unit in self.state["units"].items():
            phase, iteration, question_id = json.loads(key)
            grouped.setdefault((phase, iteration), []).append((question_id, unit))
        for (phase, iteration), units in sorted(grouped.items()):
            if self.config.mode == "performance":
                for _, unit in units:
                    records.append(copy.deepcopy(unit["result"]))
                    attempts.append({"record_index": len(records) - 1, "phase": phase,
                                     "iteration": iteration, "payload": copy.deepcopy(unit["log"])})
                continue
            if self.config.mode == "docker_task":
                from .docker_task.runner import _aggregate_attempt_record
                record = _aggregate_attempt_record(phase=phase, iteration=iteration, started_at=self.started_at, prompt_text=prompt_text, question_results=[copy.deepcopy(unit["result"]) for _, unit in units])
                record["partial"] = True
                record["attempt_wall_ms"] = sum(unit["result"].get("host_wall_ms", 0) for _, unit in units)
                record["measurement"] = {"protocol": self.protocol, "scope": "suite", "cache_state": "unknown", "recovered": self.recovering}
                attempt = {"question_logs": [{"question_index": index, "question_id": qid, "payload": unit["log"]} for index, (qid, unit) in enumerate(units)]}
            else:
                record = copy.deepcopy(units[0][1]["result"])
                attempt = {"payload": units[0][1]["log"]}
            records.append(record)
            attempts.append({"record_index": len(records) - 1, "phase": phase, "iteration": iteration, **attempt})
        return self.enrich({"run_id": self.run_id, "started_at": self.started_at, "ended_at": utc_now(), "provider": self.config.provider,
                            "model": self.model, "api_model": self.api_model, "prompt_text": prompt_text, **display,
                            "benchmark_mode": self.config.mode, "records": records, "error": self.state.get("error"),
                            "_log_bundle": {"console_lines": [], "attempts": attempts}})

    def close(self, *, keep_loaded=False):
        if self.config.provider == "ds4" and not self.externally_managed:
            # This also owns cleanup when startup was interrupted before
            # prepare_model returned. A child must not outlive its run.
            self._unload()
            return
        if self.prepared and not keep_loaded and not self.externally_managed:
            self._unload()
