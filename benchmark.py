from __future__ import annotations

import argparse
from pathlib import Path

from local_llm_bench.config import DEFAULT_CONFIG_PATH, DOCKER_TASK_MODE, load_config
from local_llm_bench.docker_task.runner import run_docker_task_benchmark
from local_llm_bench.history import update_history, write_latest
from local_llm_bench.provider_runtime import build_provider_runtime
from local_llm_bench.report import ensure_report_html
from local_llm_bench.runner import run_benchmark
from local_llm_bench.run_logs import persist_run_logs


def _resolved_target(model: str, model_info: dict[str, object] | None) -> str:
    if not model_info:
        return model
    for key in ("identifier", "model_key", "indexed_model_identifier", "path"):
        value = model_info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ローカルLLM性能を比較し、HTMLダッシュボードを生成します。"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="設定ファイルのパス")
    parser.add_argument("--model", action="append", help="比較対象モデル。複数指定可")
    parser.add_argument("--prompt-text", help="プロンプト本文。指定時は bench.yaml の prompt.text を上書き")
    parser.add_argument("--api-base", help="provider=lmstudio 用 OpenAI互換APIのベースURL")
    parser.add_argument("--cold-runs", type=int, help="coldフェーズの反復回数")
    parser.add_argument("--warm-runs", type=int, help="warmフェーズの反復回数")
    parser.add_argument("--timeout-sec", type=float, help="各リクエストのタイムアウト秒")
    parser.add_argument("--max-tokens", type=int, help="max_tokens")
    parser.add_argument("--temperature", type=float, help="temperature")
    parser.add_argument("--out-dir", type=Path, help="出力先ディレクトリ。指定時は JSON/HTML をここに集約")
    parser.add_argument(
        "--refresh-report",
        action="store_true",
        help="index.html ビュアーを強制再生成します。通常は既存ファイルを再利用します。",
    )
    parser.add_argument(
        "--keep-loaded",
        action="store_true",
        help="ベンチ後に `lms unload` しません。既定では各モデル計測後にアンロードします。",
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
        cli_max_tokens=args.max_tokens,
        cli_temperature=args.temperature,
        cli_out_dir=args.out_dir,
    )
    runtime = build_provider_runtime(config)

    produced_runs = []
    for model in config.models:
        run_data = None
        requested_model = model
        model_info = None
        api_model = requested_model
        model_prepared = False
        try:
            api_model, model_info = runtime.prepare_model(requested_model)
            model_prepared = True
            if config.mode == DOCKER_TASK_MODE:
                run_data = run_docker_task_benchmark(
                    config,
                    model=api_model,
                    requested_model=requested_model,
                    docker_env=runtime.docker_environment(),
                )
            else:
                run_data = run_benchmark(
                    config,
                    model=api_model,
                    requested_model=requested_model,
                    client=runtime.chat_client(),
                )
            if not model_info:
                model_info = runtime.describe_model(api_model)
            if model_info:
                run_data["model_info"] = model_info
                api_model = _resolved_target(api_model, model_info)
            run_data["api_model"] = api_model
            run_data = persist_run_logs(config.output, run_data)
            update_history(config.output.history_json, run_data)
            write_latest(config.output.latest_json, run_data)
            produced_runs.append(run_data)
        finally:
            if model_prepared and not args.keep_loaded:
                unload_results = runtime.unload_model(api_model)
                for unload_result in unload_results:
                    prefix = "[Unload]"
                    if unload_result.status == "unloaded":
                        print(f"{prefix} {unload_result.target}: {unload_result.message}")
                    else:
                        print(
                            f"{prefix} {unload_result.target}: {unload_result.status}: "
                            f"{unload_result.message}"
                        )

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
        row = run_data["summary"]["models"][0]
        model_info = run_data.get("model_info") or {}
        display_name = model_info.get("display_name") or row["model"]
        format_label = model_info.get("format") or "N/A"
        quantization_label = model_info.get("quantization") or "N/A"
        latency = row.get("warm_mean_total_latency_ms")
        ttft = row.get("warm_mean_ttft_ms")
        decode = row.get("warm_mean_decode_tps")
        print(
            f"- {display_name} [{format_label} / {quantization_label}] ({row['model']}): "
            f"warm_latency={latency:.1f}ms "
            if isinstance(latency, (int, float))
            else f"- {display_name} [{format_label} / {quantization_label}] ({row['model']}): warm_latency=N/A ",
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
    raise SystemExit(main())
