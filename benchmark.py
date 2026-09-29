from __future__ import annotations

import argparse
import json
import signal
from dataclasses import asdict, replace
from pathlib import Path

from local_llm_bench.config import DEFAULT_CONFIG_PATH, DOCKER_TASK_MODE, LMSTUDIO_PROVIDER, load_config
from local_llm_bench.docker_task.runner import run_docker_task_benchmark
from local_llm_bench.history import update_history, write_latest
from local_llm_bench.provider_runtime import build_provider_runtime
from local_llm_bench.report import ensure_report_html
from local_llm_bench.runner import run_benchmark
from local_llm_bench.performance import run_performance_benchmark
from local_llm_bench.run_logs import persist_run_logs
from local_llm_bench.diagnostics import run_preflight
from local_llm_bench.execution import RunExecution
from local_llm_bench.persistence import server_lock, FileLock


def _lmstudio_parallelism_values(config) -> list[int | None]:
    if config.provider != LMSTUDIO_PROVIDER:
        return [None]
    if config.lmstudio_load.parallelism_sweep:
        return list(config.lmstudio_load.parallelism_sweep)
    if config.lmstudio_load.parallelism is not None:
        return [config.lmstudio_load.parallelism]
    return [None]


def _with_lmstudio_parallelism_metadata(run_data: dict[str, object], parallelism: int | None) -> dict[str, object]:
    if parallelism is None:
        return run_data
    run_data["lmstudio_parallelism"] = parallelism
    lmstudio = run_data.get("lmstudio")
    if not isinstance(lmstudio, dict):
        lmstudio = {}
        run_data["lmstudio"] = lmstudio
    lmstudio["parallelism"] = parallelism
    model_info = run_data.get("model_info")
    if isinstance(model_info, dict):
        load_config = model_info.get("load_config")
        if isinstance(load_config, dict):
            lmstudio["load_config"] = load_config
    return run_data


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ローカルLLM性能を比較し、HTMLダッシュボードを生成します。"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help=f"設定ファイルのパス（既定: {DEFAULT_CONFIG_PATH}）")
    parser.add_argument("--docker-image", help="使用するローカルDocker image（新しいタグへの切替用）")
    parser.add_argument("--model", action="append", help="比較対象モデル。複数指定可")
    parser.add_argument("--prompt-text", help="プロンプト本文。指定時は設定ファイルの prompt.text を上書き")
    parser.add_argument("--api-base", help="provider=lmstudio / omlx / mlx_serve 用 OpenAI互換APIのベースURL")
    parser.add_argument("--cold-runs", type=int, help="coldフェーズの反復回数")
    parser.add_argument("--warm-runs", type=int, help="warmフェーズの反復回数")
    parser.add_argument("--timeout-sec", type=float, help="各リクエストのタイムアウト秒")
    parser.add_argument("--parallelism", type=int, help="LM Studio の Max Concurrent Predictions")
    parser.add_argument("--parallelism-sweep", help="LM Studio parallelism をカンマ区切りで複数指定（例: 2,3,4）")
    parser.add_argument("--max-tokens", type=int, help="max_tokens")
    parser.add_argument("--temperature", type=float, help="temperature")
    parser.add_argument("--performance", action="store_true", help="Inspect AIで入力長別・同時実行性能を測定")
    parser.add_argument("--input-tokens", type=int, nargs="+", help="入力トークン数の目安（例: 1024 4096 8192）")
    parser.add_argument("--concurrency", type=int, nargs="+", help="クライアント同時実行数（例: 1 2 4）。サーバー設定は変更しません")
    parser.add_argument("--memory-pid", type=int, help="Peak RSSを観測する推論プロセスのPID")
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument("--resume-run-id", help="中断したrunの未完了部分を再開")
    recovery.add_argument("--retry-errors-run-id", help="同じrunの記録済みエラーだけを再試行")
    parser.add_argument("--questions", nargs="+", help="再試行するquestion ID（完全一致）")
    parser.add_argument("--allow-unverified-resume", action="store_true", help="不明な実効条件を明記した上で再開（既知の条件差は許可しない）")
    parser.add_argument("--check", action="store_true", help="モデルをロードせずに実行前診断のみ実行")
    parser.add_argument("--out-dir", type=Path, help="出力先ディレクトリ。指定時は JSON/HTML をここに集約")
    parser.add_argument(
        "--refresh-report",
        action="store_true",
        help="index.html ビュアーを強制再生成します。通常は既存ファイルを再利用します。",
    )
    parser.add_argument(
        "--keep-loaded",
        action="store_true",
        help="終了時のアンロードを省略します（ds4 の managed モードは常に自分のサーバーを停止します）。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_config(
        args.config,
        cli_models=args.model,
        cli_prompt_text=args.prompt_text,
        cli_api_base=args.api_base,
        cli_cold_runs=args.cold_runs,
        cli_warm_runs=args.warm_runs,
        cli_timeout_sec=args.timeout_sec,
        cli_parallelism=args.parallelism,
        cli_parallelism_sweep=args.parallelism_sweep,
        cli_max_tokens=args.max_tokens,
        cli_temperature=args.temperature,
        cli_out_dir=args.out_dir,
    )
    if args.docker_image:
        config = replace(config, docker_image=args.docker_image)
    if args.performance or args.input_tokens is not None or args.concurrency is not None or args.memory_pid is not None:
        if config.mode == DOCKER_TASK_MODE:
            parser.error("性能測定にはprompt用のconfigを指定してください。")
        from local_llm_bench.config import _performance_settings
        options = asdict(config.performance)
        for key, value in (("input_tokens", args.input_tokens), ("concurrency", args.concurrency), ("memory_pid", args.memory_pid)):
            if value is not None:
                options[key] = value
        config = replace(config, mode="performance", performance=_performance_settings(options))
    from local_llm_bench.inspect_harness import require_inspect
    require_inspect()
    runtime = build_provider_runtime(config)

    produced_runs = []
    parallelism_values = _lmstudio_parallelism_values(config)
    if (args.resume_run_id or args.retry_errors_run_id) and (len(config.models) != 1 or len(parallelism_values) != 1):
        parser.error("再開・再試行時は --model で1モデルを指定し、parallelismも1値にしてください。")
    if args.questions and not args.retry_errors_run_id:
        parser.error("--questions は --retry-errors-run-id と併用してください。")
    if args.check:
        with server_lock(config.api_base):
            for model in config.models:
                print(json.dumps(run_preflight(config, runtime, model), ensure_ascii=False, indent=2))
        return 0
    for model in config.models:
        for lmstudio_parallelism in parallelism_values:
            with server_lock(config.api_base):
                execution = RunExecution(config, runtime, model, lmstudio_parallelism,
                    resume_id=args.resume_run_id, retry_id=args.retry_errors_run_id,
                    questions=args.questions, allow_unverified=args.allow_unverified_resume)
                with FileLock(execution.directory / ".run.lock"):
                    print(f"[Preparing {execution.run_id}] {model}")
                    try:
                        preflight = run_preflight(config, runtime, model)
                        execution.start(preflight)
                        run_config = replace(config, docker_image=preflight["docker"]["image_id"]) if preflight.get("docker") else config
                        if config.mode == DOCKER_TASK_MODE:
                            run_data = run_docker_task_benchmark(
                                run_config, model=execution.api_model, requested_model=model,
                                docker_env=runtime.docker_environment(), execution=execution,
                            )
                        elif config.mode == "performance":
                            run_data = run_performance_benchmark(
                                run_config, model=execution.api_model, requested_model=model,
                                client=runtime.chat_client(), execution=execution,
                            )
                        else:
                            run_data = run_benchmark(
                                run_config, model=execution.api_model, requested_model=model,
                                client=runtime.chat_client(), execution=execution,
                            )
                        execution.close(keep_loaded=args.keep_loaded)
                        execution.finish("completed")
                        run_data = execution.enrich(run_data)
                        run_data = _with_lmstudio_parallelism_metadata(run_data, lmstudio_parallelism)
                        run_data = persist_run_logs(config.output, run_data)
                        update_history(config.output.history_json, run_data)
                        write_latest(config.output.latest_json, run_data)
                        execution.mark_outputs_saved()
                        produced_runs.append(run_data)
                    except BaseException as exc:
                        try:
                            execution.close(keep_loaded=args.keep_loaded)
                        except Exception as cleanup_error:
                            if execution.started:
                                execution.event("cleanup_error", {"error": str(cleanup_error)})
                        execution.finish("interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed", str(exc))
                        if not execution.recovering or execution.started:
                            partial = persist_run_logs(config.output, execution.partial_result())
                            update_history(config.output.history_json, partial)
                            write_latest(config.output.latest_json, partial)
                        raise

    report_updated = ensure_report_html(
        config.output.report_html,
        config.output.history_json,
        force=args.refresh_report,
    )

    print("")
    print("[Completed]")
    print(f"history : {config.output.history_json}")
    print(f"latest  : {config.output.latest_json}")
    print(f"logs    : {config.output.run_logs_dir}")
    report_state = "updated" if report_updated else "reused"
    print(f"report  : {config.output.report_html} ({report_state})")
    for run_data in produced_runs:
        if run_data.get("benchmark_mode") == "performance":
            print(f"- {run_data['model']}: {len(run_data['records'])} samples / performance profile complete")
            continue
        row = run_data["summary"]["models"][0]
        model_info = run_data.get("model_info") or {}
        display_name = model_info.get("display_name") or row["model"]
        format_label = model_info.get("format") or "N/A"
        quantization_label = model_info.get("quantization") or "N/A"
        latency = row.get("warm_mean_total_latency_ms")
        ttft = row.get("warm_mean_ttft_ms")
        decode = row.get("warm_mean_decode_tps")
        duration = run_data.get("duration_sec")
        lmstudio_parallelism = run_data.get("lmstudio_parallelism")
        print(
            f"- {display_name} [{format_label} / {quantization_label}] ({row['model']}): "
            f"lmstudio_parallelism={lmstudio_parallelism} "
            if isinstance(lmstudio_parallelism, int)
            else f"- {display_name} [{format_label} / {quantization_label}] ({row['model']}): ",
            end="",
        )
        print(
            f"duration={duration:.1f}s " if isinstance(duration, (int, float)) else "duration=N/A ",
            end="",
        )
        print(
            f"warm_latency={latency:.1f}ms "
            if isinstance(latency, (int, float))
            else "warm_latency=N/A ",
            end="",
        )
        print(
            f"warm_ttft={ttft:.1f}ms " if isinstance(ttft, (int, float)) else "warm_ttft=N/A ",
            end="",
        )
        print(f"decode={decode:.2f} tok/s" if isinstance(decode, (int, float)) else "decode=N/A", end="")
        if run_data.get("benchmark_mode") == DOCKER_TASK_MODE:
            score = row.get("warm_mean_benchmark_score")
            correct_rate = row.get("warm_benchmark_correct_rate")
            error_rate = row.get("warm_benchmark_error_rate")
            print(
                f" score={score:.3f}" if isinstance(score, (int, float)) else " score=N/A",
                end="",
            )
            print(
                f" correct={correct_rate * 100:.1f}%"
                if isinstance(correct_rate, (int, float))
                else " correct=N/A",
                end="",
            )
            print(
                f" error={error_rate * 100:.1f}%"
                if isinstance(error_rate, (int, float))
                else " error=N/A"
            )
        else:
            print("")
    return 0


if __name__ == "__main__":
    def terminate(signum, frame):
        raise KeyboardInterrupt("SIGTERM")
    signal.signal(signal.SIGTERM, terminate)
    raise SystemExit(main())
