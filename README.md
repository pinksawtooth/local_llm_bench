# local_llm_bench

LM Studio / oMLX / mlx-serve（MLX Core） / Unsloth Studio / ds4 を対象に、Inspect AIで評価・記録するローカルLLMベンチマークです。モデルのロード、cold/warm、測定条件の照合、逐次保存・再開は既存の実行管理と統合しています。

ベンチの実行設定は [`configs/`](configs/) にまとめています。既定は `configs/bench.yaml` です。YAML内の相対パスはそのYAMLのディレクトリを基準とし、同梱設定の結果は従来どおり `runs/`、レポートは `docs/index.html` に出力します。課題仕様・正解YAMLは対象データと組になるため `benchmarks/<課題名>/` に置いています。

## Inspect AI

すべての評価にInspect AIを使います。通常測定の1試行・Docker測定の1問をそれぞれInspectのTask/Sampleとして実行し、Docker内のエージェントにはInspect ReActを使います。正答判定はホスト側の型別ルールをInspect Scorerから呼び、その結果をWeb・集計・checkpointに保存します。通常測定は速度測定のみで、正答率は付けません。

```yaml
inspect:
  max_turns: null
  max_tool_calls: null
  tool_timeout_sec: 120
benchmark:
  question_timeout_sec: 3600
```

- 通常測定・課題評価は1 sample・1 epoch・同時実行1。性能プロファイルでは同時実行グループを複数のInspect Sampleとして実行します。Inspectの自動再試行・生成キャッシュは無効にします。
- Inspectに渡す生成設定からHTTPリクエストを再構成せず、既存の通信処理と明示された設定を使います。LM Studio・oMLX・mlx-serveの保存設定とds4の方針を全ターンで維持します。通常測定でも出力上限による自動継続は行いません。
- ターン数・ツール呼び出し回数は省略または `null` で無制限です。課題評価は既定で1問3600秒（1時間）まで継続し、`benchmark.question_timeout_sec` で変更できます。同梱の課題用設定も3600秒です。明示的な回数上限が必要な場合だけ正の整数を指定します。各ツールの待機上限は `tool_timeout_sec` と問題の残り時間の短い方です。不正なツール引数、時間切れ、通信エラーは成功扱いにしません。誤答と実行エラーは別に集計します。
- 正解ファイルはコンテナに渡しません。Dockerのbuild contextとimageにも含めず、実行コード・対象バイナリとその問題のリクエストを渡します。
- `.eval` ログは `runs/logs/<run-id>/inspect/<unit>/<attempt>/` に保存します。Docker内の会話・ツールイベントはその `worker/`、送信前から逐次保存する履歴は `worker/trace.json` です。タイムアウト時もホスト側に残ります。再試行は別ディレクトリを作り、前回ログを保持します。
- RevBenchと同じInspect標準の圧縮 `.eval` 形式を明示指定します。固定版 `0.3.263` が会話の重複排除とZIP内のZstandard圧縮を行い、通常測定・性能プロファイル・Dockerエージェントで共通です。追加設定は不要で、`INSPECT_LOG_FORMAT=json` が環境にあっても `.eval` で保存します。[Inspectの保存形式](https://inspect.aisi.org.uk/eval-logs.html#storage-optimization)
- 終了したInspectログは権限 `0600` とし、実際の圧縮方式・ファイル容量・ZIP内の展開時容量を `inspect.log_storage` に記録します。展開時容量はJSONエクスポートの容量とは異なります。保存確認に失敗した場合は `log_storage.status: unavailable` とし、完了した推論を再実行しません。統合Webのログ一覧APIも本文を全展開せず容量を取得し、Inspect Viewとダウンロードは `.eval` をそのまま利用します。
- 既存ログの変換・再圧縮は自動実行しません。`trace.json`・`checkpoint.json`・`events.jsonl` は中断復旧用の逐次記録として従来どおり保存します。圧縮対象はInspectのネイティブログです。
- Inspectの版・依存lockハッシュ・上限設定を比較条件に含めます。Webのモデル名とRun Detailsにも評価基盤を表示します。

通常プロンプト・性能測定・Dockerタスク（d-compileなど）は共通のストリーミング通信と計測処理を使います。各推論でPP/TG TPS・TTFT・TPOTを記録し、クライアントの時計とAPIが報告する統計を区別します。Inspectやツールの処理時間はタスク全体のwall timeに含め、推論のE2Eからは除きます。ホストの評価ログとworkerのログは同じ推論を別の範囲で記録するため、両方のトークン数を足さないでください。

## 入力長別・同時実行の性能測定

oMLX・mlx-serve用の設定を用意しています。起動済みのサーバーに合わせて、以下のいずれかを実行してください。実際のモデルロード・推論を行います。

```bash
cd /Users/samsepi0l/local_llm_bench
.venv/bin/python benchmark.py --config configs/bench_performance_omlx.yaml
.venv/bin/python benchmark.py --config configs/bench_performance_mlx_serve.yaml
```

他のランタイムでも既存のprompt用configを使えます。`--concurrency` はリクエストの同時実行数で、LM Studioのロード設定 `--parallelism` とは別です。Docker課題用configとの併用は拒否します。

```bash
.venv/bin/python benchmark.py --config configs/bench_lmstudio_4_models.yaml --performance --input-tokens 1024 4096 8192 --concurrency 1 2 4 --cold-runs 0
.venv/bin/python benchmark.py --config configs/bench_ds4.yaml --performance --input-tokens 1024 4096 8192 --concurrency 1 2 4 --cold-runs 0
```

YAMLでは `mode: performance` と以下を指定します。

```yaml
performance:
  input_tokens: [1024, 4096, 8192] # 必要に応じて16384、32768も追加
  concurrency: [1, 2, 4]
  # memory_pid: 12345            # 推論プロセスと子プロセスのRSSを測る場合
runs:
  cold_runs: 0
  warm_runs: 3
  timeout_sec: 43200
```

入力目安は共通Pythonテキストの「文字数÷4」です。モデル固有の正確なトークン数ではなく、実入力数はAPIのusage（チャットテンプレート等を含む）から保存・表示します。ダウンロードや事前推論によるトークン数調整は行いません。サーバーがコンテキスト上限エラーを返した場合は、そのエラーを保存します。生成上限・温度・サンプリング・思考設定は元のconfigの設定元を維持し、出力長を128等に固定しません。

上のoMLX・mlx-serve設定は1モデルにつき63リクエスト（3入力長×3反復×並列数の合計7）と、統計から除外する準備推論1回です。`cold_runs: 0` でも開始時のモデル準備は行います。ロード・初回グループ・反復グループと、APIが報告するキャッシュ状態を分けて記録します。warmはキャッシュヒットを保証しません。

| 指標 | 定義・出所 |
| --- | --- |
| pp TPS | `timings.prompt_per_second`、またはprocessed token数÷`timings.prompt_ms`、次に`usage.prompt_tokens_per_second`を優先。ない場合は実入力数÷クライアントTTFT。出所・算出方法はツールチップと計測定義に表示 |
| tg TPS | `timings.predicted_per_second` / `usage.generation_tokens_per_second`。ない場合は `(出力数−1)÷(E2E−TTFT)`。出所・算出方法はツールチップと計測定義に表示 |
| TTFT | リクエスト開始から最初の本文・思考・ツール関数名/引数チャンクまで。role/idだけの通知は数えず、最終回答の開始時間とも区別する |
| TPOT | `(E2E−TTFT)÷(出力数−1)` のクライアント概算。複数トークンのチャンクや終端処理の影響を含む。出力1トークン以下・非ストリーミングでは未取得 |
| E2E | リクエスト開始から応答ストリーム終端まで |
| 総合TPS | `(実入力数＋実出力数)÷E2E` |
| 全体入力/出力tok/s | 同時実行グループの合計トークン数÷最初のリクエスト開始から最後の完了まで。Inspectの準備時間は含まない |

`usage.prompt_eval_duration` はAPI定義の区間です。oMLXではサーバーTTFT由来の場合があるため、独立したプレフィル時間として扱いません。`metrics.prefill_ms` / `turn_usage.prefill_sec` にクライアントTTFTをコピーする処理を廃止しました。`raw_usage` / `raw_timings` / `raw_stats` と `metrics.version` / `metrics.sources` を残し、後から測定方法を確認できます。未取得値はnullで、キャッシュ未取得を0に補完しません。旧ログから純粋なプレフィルは復元しません。表のセルは数値だけにし、API報告値かクライアント算出値かはツールチップと計測定義から確認できます。

全モードで`stream: true`と`stream_options.include_usage: true`を要求します。空のストリームや非ストリーミングJSONは計測エラーとし、非ストリーミングでの自動再送は行いません。タスクの各推論は`inference_requests[]`に計測値とAPIの生の統計を保存し、設問結果・試行結果・history・Inspectログへ引き継ぎます。タスクの`metrics.scope`は`inference_requests`で、PP/TG TPS・TTFT・TPOTは各試行内の全推論の中央値、入出力トークン数と推論E2Eは合計です。複数設問も個々の推論を集めて集計し、設問ごとの中央値を再度平均しません。`metrics.aggregation`に計測件数、`metrics.source_details`に各推論の計測元を残します。途中の通信失敗でも完了済み推論の記録を保持し、欠損した推論を含む全体値はnullにします。

Inspectの `max_samples` / `max_connections` をグループの実行数に合わせ、各サンプルを同期して開始します。通常のサーバーロック内で動作し、別ベンチとの重複を防ぎます。1件の完了ごとに結果をfsyncし、グループ完了後にログの参照先・グループ統計を追記します。

中断時は従来の `--resume-run-id`、完了後のエラー再試行は `--retry-errors-run-id` を使います。再試行対象の `--questions` には保存された `sample_id`（例 `pp1024-c4-r2`）を指定できます。完了済みは再送しません。4並列のうち1件を再試行した結果は「実行1 / 予定4」と記録し、4並列の倍率・集計に混ぜません。失敗・未完了グループには比較用スループットを付けません。同時実行の観測ピークと開始時刻の差も記録し、予定数のリクエストが重ならなかったグループには倍率を付けません。

Webのメイン表は同じ入力目安・並列数・フェーズ・キャッシュ状態・測定方法の中央値と範囲・nを表示し、課題の正答率は別表に表示します。倍率は同一run内の条件が一致する1並列グループを基準にします。APIから取得できない設定やキャッシュ制御を同一条件とみなす保証はしません。

Peak Mem列・カードと入力長別グラフの表示実装は削除しました。同時実行・課題成績は該当データがある場合だけ表示します。性能プロファイルの入力長別測定と監査用のメモリ採取は保存処理として残し、表では入力長ごとの結果を確認できます。

```bash
.venv/bin/python -m local_llm_bench.dashboard --port 7575
```

表示コードの反映にはダッシュボードプロセスの再起動が必要です。オフライン検証は `.venv/bin/python tests/run_offline.py` で実行し、実サーバー接続・推論・Docker起動を禁止します。

依存関係はInspect AI 0.3.263を含むハッシュ付き `requirements.lock` に固定しています。既存のベンチ用venvへ追加せず、このプロジェクト専用のPython 3.12環境を用意してください。

```bash
cd /Users/samsepi0l/local_llm_bench
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.lock
```

実行中のベンチが終了してから、新しいタグでDocker imageを作ります。旧imageは事前診断で拒否し、自動でbuildしません。

```bash
./build_bench_image.sh --platform linux/arm64 --tag local-llm-bench:inspect-v1
.venv/bin/python benchmark.py --config configs/bench_d_compile_arm64_lmstudio_4_models.yaml
.venv/bin/python benchmark.py --config configs/bench_mafc_arm64_lmstudio_4_models.yaml
```

通常測定は `.venv/bin/python benchmark.py --config configs/bench_lmstudio_4_models.yaml` です。ds4も既存の設定をそのまま使えます。検証には以下を使います（モデル/API/Docker/MCPの実プロセスを使わず、Inspect本体とメモリ上のfixtureを実行します）。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tests/run_offline.py
```

比較ダッシュボードとInspectの詳細ログは、専用venvから1つのコマンドで起動できます。

```bash
cd /Users/samsepi0l/local_llm_bench
.venv/bin/python -m local_llm_bench.dashboard --port 7575
```

[統合ダッシュボード](http://127.0.0.1:7575/) を開いてください。Leaderboard、Compare、Cold vs Warm、Timing Analysis、Error Analysis、Run Detailsに加えて、**Inspect Logs**タブにInspect標準ビューアーを表示します。

- **Run Details → Inspect で開く**で、その試行・問題のログへ移動します。保存された結果のログを選ぶので、新しい再試行と取り違えません。
- メイン表の「最新ログ」はその集計グループの最新試行を開きます。Run Detailsは正答率・誤答/エラー数・速度・トークン数・問題別成績に絞り、会話・思考・ツール入出力の全文と採点根拠はInspectで確認します。測定条件・診断・ロード時間は折りたたんで保持します。
- **評価ログ**では採点と結果、**エージェントログ**では会話とツール呼び出しを確認できます。選択欄にはLM Studio / ds4などの実行元、モデル、問題、試行を表示します。両者は同じ推論の記録なので、使用トークン数を合算しません。
- 「Reload History」で比較結果、「ログを更新」で新しく保存されたInspectログを再読込できます。削除済み・未保存のログは、その状態を表示します。
- 起動は閲覧用サーバーのみです。モデル、Docker、ベンチマークを起動せず、ログの編集・削除APIも無効です。設定YAMLなどのプロジェクト全体は配信しません。

ポートを変更する場合は `--port 8081`、別の保存先は `--runs-dir /path/to/runs`（配下に`history.json`と`logs/`）を指定します。使用中のポートではエラーで終了し、既存のサーバーを停止しません。従来の `/docs/` URLも同じ画面を表示します。静的な`docs/index.html`は比較用として残りますが、Inspect統合には上の起動コマンドを使ってください。

固定版Inspectの標準画面には、ログのパスに文字`%`が含まれると表示エラーになる制約があります。現在のプロジェクトの既定保存先と、自動生成されるログ名では発生しません。

Inspectだけを開く場合は `.venv/bin/inspect view --log-dir runs/logs --recursive --host 127.0.0.1 --port 7575` も使えます。ログは[Inspectのログ仕様](https://inspect.aisi.org.uk/eval-logs.html)に従い、統合画面は固定版InspectのビューアーAPIと同梱アセットを利用しています。会話やツール出力を含むため、ログを公開する場合は内容を確認してください。

評価基盤の切替と旧エージェントは削除しました。`--harness` は使用できず、YAMLに `harness` が残っている場合は削除が必要です。Inspect AIが未導入・版不一致の場合は実行前に停止します。保存済みの履歴は引き続き参照できます。checkpointの再開では実装ハッシュも照合するため、変更前のコード・依存環境で実行中のベンチを完了してから切り替えてください。

## 使い方

```bash
python /Users/samsepi0l/local_llm_bench/benchmark.py --config /Users/samsepi0l/local_llm_bench/configs/bench.yaml
```

LM Studio / Unsloth Studio では、既定で各モデルのベンチが終わるたびに provider に応じた unload を実行します。LM Studio では REST API の `/api/v1/models/unload` を使い、既に未ロードならそのままロードへ進みます。ロード状態の取得に失敗した場合は停止します。Unsloth Studio では `/api/inference/unload` を使います。終了時にロード状態を残したいときだけ `--keep-loaded` を付けてください。この指定でも測定開始前と各 cold 試行前には再ロードします。

`provider` を省略した場合は `lmstudio` として扱います。Unsloth Studio を使うときは `provider: unsloth_studio` を設定し、`api_base` は書かずに認証情報だけ渡してください。OpenAI 互換 API は内部で固定の `http://127.0.0.1:8888/v1` を使います。

Unsloth Studio の設定例:

```yaml
provider: unsloth_studio
models:
  - "unsloth/gpt-oss-20b"
auth:
  bearer_token: "..."
```

Bearer token の代わりに username/password を使う場合は、`auth.username` / `auth.password` か環境変数 `UNSLOTH_STUDIO_USERNAME` / `UNSLOTH_STUDIO_PASSWORD` を指定してください。token は `UNSLOTH_STUDIO_BEARER_TOKEN` でも渡せます。

モデルをCLIで上書きする例:

```bash
python /Users/samsepi0l/local_llm_bench/benchmark.py \
  --model openai/gpt-oss-20b \
  --model openai/gpt-oss-120b
```

プロンプトをCLIで上書きする例:

```bash
python /Users/samsepi0l/local_llm_bench/benchmark.py \
  --config /Users/samsepi0l/local_llm_bench/configs/bench.yaml \
  --prompt-text "pythonでライブラリを使わずにAESを実装して"
```

LM Studio の `api_base` をCLIで上書きする例:

```bash
python /Users/samsepi0l/local_llm_bench/benchmark.py \
  --config /Users/samsepi0l/local_llm_bench/configs/bench.yaml \
  --api-base http://localhost:1234/v1
```

LM Studio の Max Concurrent Predictions を変えて測る例:

```bash
python /Users/samsepi0l/local_llm_bench/benchmark.py \
  --config /Users/samsepi0l/local_llm_bench/configs/bench_d_compile_arm64.yaml \
  --parallelism-sweep 2,3,4
```

YAML では次のように指定できます。各値でモデルを unload/load し直し、`duration_sec` と `lmstudio_parallelism` を履歴へ別 run として保存します。

```yaml
lmstudio:
  parallelism_sweep: [2, 3, 4]
```

出力先をまとめて変える例:

```bash
python /Users/samsepi0l/local_llm_bench/benchmark.py --out-dir /Users/samsepi0l/local_llm_bench/out
```

アンロードせずに終える例:

```bash
python /Users/samsepi0l/local_llm_bench/benchmark.py \
  --config /Users/samsepi0l/local_llm_bench/configs/bench.yaml \
  --keep-loaded
```

`request.max_tokens` はAPIへ送る出力上限です。`finish_reason=length` でも追加要求を行いません。LM Studio保存設定を使う場合は、出力上限もLM Studio側の値を使います。

LM Studioは既定で、保存した出力上限・温度・サンプリング・思考設定を使います。明示する場合は次のように指定します。[configs/bench_qwen3_8_flash_next.yaml](configs/bench_qwen3_8_flash_next.yaml)（通常測定）と [configs/bench_d_compile_arm64_qwen3_8_flash_next.yaml](configs/bench_d_compile_arm64_qwen3_8_flash_next.yaml)（Docker測定）を含め、同梱のLM Studio設定はすべてこの方針です。`configs/bench_d_compile_arm64.yaml` の並列数sweepも既定では行いません。

LM Studioの4モデルをまとめて測る設定は [configs/bench_lmstudio_4_models.yaml](configs/bench_lmstudio_4_models.yaml)（通常測定）と [configs/bench_d_compile_arm64_lmstudio_4_models.yaml](configs/bench_d_compile_arm64_lmstudio_4_models.yaml)（Docker測定）です。2026-09-14にローカル索引とモデルファイルを確認したQwopus3.8-27B-Flash、Qwen3.8-Flash-Next、Muse-Glimmer-30B、Gemma-4-31B-it-QATを順番に測定します。各モデルはcold 1回・warm 3回で、推論設定は各モデルのLM Studio保存設定を使います。

```yaml
request:
  use_lmstudio_defaults: true
```

このモードでは推論パラメータをAPIへ送らず、準備推論・再送・Dockerの全ツールターンもLM Studioの設定に任せます。`request.temperature` / `max_tokens` / `top_p` / `reasoning_effort` やCLIの `--temperature` / `--max-tokens` との併用はエラーになります。通常測定では `finish_reason=length` から自動で続きを取得しません。履歴には `settings_source: lmstudio_saved` を保存し、APIで確認できない実効値は不明のまま記録します。古い設定に推論パラメータがある場合は削除してください。意図的にベンチ側の要求値を使う場合だけ `use_lmstudio_defaults: false` を明示します。

`provider=unsloth_studio` では `api_base` と `docker.api_base` の override は受け付けません。docker_task ではコンテナ側から host の Studio に到達するため、内部で `host.docker.internal` 側の URL を使います。

## oMLX

`provider: omlx` は、起動済みのoMLXへ接続し、ベンチ実行時に対象モデルをロードします。通常測定とDocker内のInspect ReActに対応し、WebのLeaderboard・Compare・Run Details・Inspect Logsでは実行元を **oMLX** と表示します。

| 測定 | 設定 |
| --- | --- |
| 通常測定 | [configs/bench_omlx.yaml](configs/bench_omlx.yaml) |
| D Compile | [configs/bench_d_compile_arm64_omlx.yaml](configs/bench_d_compile_arm64_omlx.yaml) |
| MAFC | [configs/bench_mafc_arm64_omlx.yaml](configs/bench_mafc_arm64_omlx.yaml) |
| 入力長・同時実行性能 | [configs/bench_performance_omlx.yaml](configs/bench_performance_omlx.yaml) |

同梱の4設定は、2026-09-17にローカルのoMLXモデル設定・ディレクトリで確認した `Qwopus3.8-27B-Flash-oQ4e-mtp` と `Qwen3.8-Flash-Next-oQ4e-mtp` を順番に測定します。モデル名は `/v1/models` に表示されるID・エイリアスを完全一致で指定します。LM StudioのGGUF名からMLXモデルへ自動変換・置換は行いません。

```yaml
provider: omlx
api_base: http://127.0.0.1:8000/v1
models: ["Qwopus3.8-27B-Flash-oQ4e-mtp", "Qwen3.8-Flash-Next-oQ4e-mtp"]
request:
  use_omlx_defaults: true
```

oMLXも既定で保存済みの出力上限・温度・サンプリング・思考設定を使います。準備推論とInspect ReActの全ターンで生成パラメータを送らず、oMLXの保存設定・ピン留め・キャッシュ設定も変更しません。履歴には `settings_source: omlx_saved` を保存します。`request` に温度や出力上限を併記するとエラーになります。明示的に要求値を使う場合だけ `use_omlx_defaults: false` を設定できます。

oMLXの公開API `GET /api/status`、`GET /v1/models/status`、`POST /v1/models/{id}/load`・`unload` が必要です。[oMLXの実装](https://github.com/jundot/omlx/blob/main/omlx/server.py)とインストール済み `0.7.0.dev2` のソースを基に対応しています。サーバーの起動・停止やモデルのダウンロードはベンチ側では行いません。未対応API、認証エラー、モデル不一致、ロード中・処理中・待機中の要求を検出した場合は停止します。管理APIへのログインや別ランタイムへの自動切替は行いません。

認証キーは `auth.bearer_token` → 環境変数 `OMLX_API_KEY` → oMLXの保存済み `settings.json` の順で参照します。保存先はoMLXと同じく `OMLX_BASE_PATH` → macOSアプリの保存先指定 → `~/.omlx` の順です。保存キーの自動利用は、localhost / 127.0.0.1 / ::1 のHTTP接続で、保存済みの待受ポートと一致する場合に限ります。リモート接続・別ポート・HTTPSでは `OMLX_API_KEY` を明示してください。明示したキーが拒否されても別のキーで再送しません。ホストで選んだキーをDockerにも渡し、ベンチの出力ログにはキーを記録しません。認証が通らない場合は、oMLXで現在有効なキーを `OMLX_API_KEY` に設定してください。

URLの既定値はホストが `http://127.0.0.1:8000/v1`、Dockerが `http://host.docker.internal:8000/v1` です。DockerからoMLXへ到達可能なlisten設定をoMLX側で用意し、`docker.api_base` を変更する場合も同じサーバーを指定します。ds4の既定ポートも8000なので同時には起動できません。ダッシュボードの7575とは別です。

```bash
cd /Users/samsepi0l/local_llm_bench
.venv/bin/python benchmark.py --config configs/bench_omlx.yaml
# Docker測定はInspect対応imageがある場合に実行
.venv/bin/python benchmark.py --config configs/bench_d_compile_arm64_omlx.yaml
.venv/bin/python benchmark.py --config configs/bench_mafc_arm64_omlx.yaml
```

各cold試行前に対象モデルだけをunload → loadし、ロード時間を推論時間と分離します。終了時は対象モデルをアンロードし、`--keep-loaded` なら残します。oMLXのサーバー自体は維持します。同じサーバーへのlocal_llm_benchの重複実行は既存ロックで防止し、モデル操作前にも処理中の要求を確認します。外部クライアントとの完全な排他制御ではないため、測定中は他のクライアントから同じサーバーを使わないでください。

実行中のoMLXの版・モデルID・APIで確認できるコンテキスト長などを保存します。ローカルモデルはsafetensorsの全shardとモデル設定・tokenizer・chat templateをSHA-256で記録します（初回ハッシュ計算には時間がかかります）。リモートのパスをホストのファイルとみなしてハッシュしません。APIが返さない実効サンプリング・思考・MTPの有効状態は不明のままにし、モデル名の `mtp` から有効とは判定しません。SSD/KVキャッシュは削除せず、再ロードをキャッシュ消去済みとは扱いません。不明条件を含む別runは同じ平均に混ぜません。

この対応の検証は通信・実プロセス起動を遮断したfixtureテストのみです。実行中のベンチには触れず、実機のロード・推論・Docker操作は実施していません。

## mlx-serve（MLX Core）

`provider: mlx_serve` は、起動済みの [mlx-serve](https://github.com/ddalcu/mlx-serve)（MLX Core.app 同梱、API形式を確認した版は 26.9.4）へ接続し、ベンチ実行時に対象モデルをロードします。通常測定・性能測定・Docker内のInspect ReActに対応し、Webでは実行元を **mlx-serve** と表示します。

| 測定 | 設定 |
| --- | --- |
| 通常測定 | [configs/bench_mlx_serve.yaml](configs/bench_mlx_serve.yaml) |
| D Compile | [configs/bench_d_compile_arm64_mlx_serve.yaml](configs/bench_d_compile_arm64_mlx_serve.yaml) |
| MAFC | [configs/bench_mafc_arm64_mlx_serve.yaml](configs/bench_mafc_arm64_mlx_serve.yaml) |
| 入力長・同時実行性能 | [configs/bench_performance_mlx_serve.yaml](configs/bench_performance_mlx_serve.yaml) |

```yaml
provider: mlx_serve
api_base: http://127.0.0.1:11234/v1
models:
  - "ddalcu/Qwen3.8-27B-MLX-Serve-4bit"
  - "ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit"
  - "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
mlx_serve:
  model_dirs: ["~/.mlx-serve/models"]
request:
  use_mlx_serve_defaults: true
```

同梱の4設定は、上記3モデルをこの順に測定します。モデルは `~/.mlx-serve/models/` 配下の `ddalcu/` と `prism-ml/` に配置します。モデル名は `GET /v1/models` の `id` を完全一致で指定します。ダウンロード完了後、MLX Core のモデルフォルダに `~/.mlx-serve/models` が含まれていることを確認し、`curl http://127.0.0.1:11234/v1/models` で3モデルのIDが表示されてから実行してください。`mlx_serve.model_dirs` は重みの照合用で、サーバーへのモデル登録やダウンロードは行いません。LAN経由の `<id>@<peer>` や絶対パスによる登録は受け付けません。

既定ではリクエストに温度・出力上限・サンプリング・思考設定を含めず、mlx-serve の起動オプション（`--temp` / `--top-p` / `--top-k` / `--max-tokens`）とモデルの `generation_config.json` に任せます。履歴には `settings_source: mlx_serve_saved` を保存し、ロード後の `/v1/models` が返す既定値（`gen_temperature` / `gen_top_p` / `gen_top_k`）を実効サンプリングとして記録します。明示する場合は `use_mlx_serve_defaults: false` と `temperature` / `max_tokens` / `top_p` / `top_k` / `reasoning_effort` を指定できます。`min_p` / `seed` は mlx-serve のAPIに文書化されていないため拒否します。

使用する公開APIは `GET /health`、`GET /api/version`、`GET /v1/models`、`GET /props`、`POST /v1/load-model`・`/v1/unload-model`（本文 `{"model": "<id>"}`）です。`--metrics` が有効なら `GET /metrics.json` の実行中・待機中リクエスト数を確認し、処理中はモデル操作を行いません。無効な場合は確認できないとして記録します。PLD・MTP・KVキャッシュ量子化などの実効設定は、ロード後の `/props` が対象モデルを既定モデルとして返す場合に `runtime_features` と投機デコード状態として保存します。思考設定はAPIから確認できないため不明のままです。サーバーの起動・停止・モデルのダウンロード・`/v1/models/rescan` はベンチ側で行いません。

各cold試行前に対象モデルを unload → load し、ロード時間を推論時間と分離します。既にロード済みなら cold を確認できないため停止します。終了時は対象モデルをアンロードし、`--keep-loaded` なら残します。ストリーム終端後もサーバーの実行中カウンタが残る場合、アンロード前に最大30秒待機します。この待機は推論時間に含めず、実行中・待機中リクエストが残ればアンロードせず停止します。`--max-resident-models` により他のモデルが残っていても操作しません。mlx-serve 側の同時実行上限（`--max-concurrent`、既定1）を超える性能測定の並列リクエストはサーバーで待機します。

ロード時の `HTTP 503: out_of_memory` は、空きメモリ不足またはサーバーの常駐メモリ上限による拒否です。必要量・利用可能量は mlx-serve のサーバーログ（MLX Core の既定では `~/.mlx-serve/logs/mlx-serve-11234.log`）で確認してください。Flash-Next mixed-4-8bit は重みに加えてwarmup・KVキャッシュ用の余裕が必要です。他の大きな処理の終了後に再実行し、メモリ事前チェックの無効化で回避しないでください。ベンチ側では既知のエラー種別と対処を表示し、モデルロードを自動再試行しません。

APIはモデルのパスを返さないため、重みのSHA-256は `mlx_serve.model_dirs`（既定 `~/.mlx-serve/models`）配下の `<model_dir>/<id>` に safetensors のモデルディレクトリまたはGGUFがある場合だけ計算し、`local_artifact_identity_status: inferred_from_model_dirs` として mlx-serve の `--model-dir` と同じフォルダからの推定であることを記録します。見つからない場合は `unavailable_from_api` で、モデルの実体ハッシュは不明条件になります。複数のフォルダに同じIDがある場合は停止します。リモート接続ではハッシュしません。

認証は mlx-serve を `--api-key` 付きで起動した場合だけ必要で、`auth.bearer_token` または環境変数 `MLX_SERVE_API_KEY` に同じ値を設定します。loopback からの接続は mlx-serve 側で既定免除です。キーはDockerにも渡し、出力ログには記録しません。URLの既定値はホストが `http://127.0.0.1:11234/v1`、Dockerが `http://host.docker.internal:11234/v1` です。mlx-serve の既定 `--host 0.0.0.0` ならコンテナから到達できます。

```bash
cd /Users/samsepi0l/local_llm_bench
.venv/bin/python benchmark.py --config configs/bench_mlx_serve.yaml --check
.venv/bin/python benchmark.py --config configs/bench_mlx_serve.yaml
```

Docker測定のコンテナ側ワーカーもこの対応を含む image が必要です。既存の `local-llm-bench:inspect-v1` を古いソースで作成している場合は、実行中のベンチ終了後に再ビルドしてください。この対応の検証は通信・実プロセス起動を遮断したfixtureテストと、稼働中の mlx-serve に対する読み取り専用API（`/health`・`/v1/models`・`/props`・`/metrics.json`）の形式確認のみで、実機のロード・推論・Docker操作は実施していません。

## ds4

RevBench の managed ds4 実装に合わせ、`provider: ds4` のベンチ開始時に `ds4-server -m <GGUF>` を起動してモデルをロードします。既定のAPIは `http://127.0.0.1:8000/v1`。サーバーを手動起動する必要はありません。

| モデル | 通常測定 | Docker測定 |
| --- | --- | --- |
| DeepSeek-V4-Flash | [configs/bench_ds4.yaml](configs/bench_ds4.yaml) | [configs/bench_d_compile_arm64_ds4.yaml](configs/bench_d_compile_arm64_ds4.yaml) |
| GLM-5.3-Flash | [configs/bench_ds4_glm_5_3_flash.yaml](configs/bench_ds4_glm_5_3_flash.yaml) | [configs/bench_d_compile_arm64_ds4_glm_5_3_flash.yaml](configs/bench_d_compile_arm64_ds4_glm_5_3_flash.yaml) |
| Qwen3.8-Flash-Next Q2 / MTP | [configs/bench_ds4_qwen3_8_flash_next.yaml](configs/bench_ds4_qwen3_8_flash_next.yaml) | [configs/bench_d_compile_arm64_ds4_qwen3_8_flash_next.yaml](configs/bench_d_compile_arm64_ds4_qwen3_8_flash_next.yaml) |

DeepSeekの3種類のプリセットは0731用DSparkを有効化します。実行前にds4のディレクトリで `./download_model.sh ds4f-dspark` を完了してください。`--mtp-model` の相対パスはYAMLではなくds4-serverのディレクトリ基準です。Qwen用は `./download_model.sh qwen38-q2` のGGUFを指定し、内蔵MTPを有効化します。LM Studio向けの別GGUFをそのまま流用できるとは限りません。

両モデルとも `--mtp-exact-sampling` を指定し、非ゼロtemperatureでも通常のサンプリング分布を維持します。DeepSeekは[モデル公式推奨値](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)（temperature=1、top_p=1、追加のtop_k/min_pフィルターは無効）、Qwenは[モデル公式Thinking推奨値](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)（temperature=1、top_p=0.95、top_k=20、min_p=0、reasoning_effort=xhigh）を使用します。Qwenのpresence/repetition penaltyはds4の設定として送らず、ペナルティを追加しません。context=32768とmax_tokens=8192は既存ベンチに合わせたローカル上限で、モデル公式の最大長ではありません。

Qwenのprefill chunkは[ds4公式Mac導入例](https://github.com/antirez/ds4/blob/main/docs/QWEN38_FLASH_NEXT.md)の1024です。これはメモリを抑える開始値で、最速実測値ではありません。MTPの深さやDSparkのconfidenceはランタイム既定の自動制御に任せます。[投機デコード](https://github.com/antirez/ds4/blob/main/docs/SPECULATIVE_DECODING.md)はdecodeを対象とし、速度向上は採用率に依存します。これらは単一セッション向け設定です。セッションバッチを追加した場合、DeepSeek DSparkは適用されず、Qwenのexact samplingも非ゼロtemperatureでは通常バッチデコードになります。

各設定はRevBenchと同じ個別のGGUFパスを指定します。共通リンク `ds4flash.gguf` はモデルを追加すると参照先が変わるため使いません。実際に読み込むモデルはこのパスで決まります。

```yaml
provider: ds4
models: ["deepseek-v4-flash"]
ds4:
  management: managed
  server_path: "../../RevBench/ds4/ds4-server"
  model_path: "../../RevBench/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
  context_length: 32768
  startup_timeout_sec: 1200
  server_args: []
```

パスはYAMLのディレクトリからの相対指定です。例は隣のRevBench checkoutにある実行ファイルとGGUFを参照します。利用するGGUFに合わせて `model_path` を変更してください。ファイルの有無と実行権限を確認し、ダウンロードやビルドは自動実行しません。追加引数は `server_args: ["--ssd-streaming"]` のようにargvの配列で指定でき、モデル・context・listen設定を上書きする引数は拒否します。LM Studio用の設定・`--parallelism` は使いません。

起動設定はrun開始前に固定し、checkpointと測定条件に保存します。自分のプロセスが出した今回のlisten完了ログと `GET /v1/models` のエイリアス・コンテキスト長を確認してから推論を開始します。ロード待ち時間は問題のtimeoutとは別です。ログは `runs/logs/<run-id>/ds4-server.log` に追記します。起動前の診断ではAPIへ接続せず、ファイルと設定を検証します。起動失敗時にも、同じポートの別プロセスをこのrunのモデルとして記録しません。

managedモードでは他のランタイムと同じ `load-first-repeat-v1` を使い、既定は `cold_runs: 1` です。各coldは再ロード直後の初回推論、warmはその後の反復です。coldを複数回指定すると自分のサーバーを停止・再起動します。warmだけの場合や再開後は、統計から除外する準備推論を1回行います。ロード時間は `timings.load_wall_sec` に分離し、OS・KVキャッシュ状態は未確認として記録します。

終了・失敗・Ctrl+C・CLIへのSIGTERM時は、起動した子プロセスだけを停止します。ds4のmanagedモードは `--keep-loaded` を指定しても終了時に停止します。ポートが使用中、または別の local_llm_bench が予約中なら既存プロセスを操作せずエラーになります。予約ロックはRevBenchとは別ですが、既にlistenしているRevBenchのサーバーも使用中として拒否します。別クライアントとの同時起動は避けてください。GGUFやディスク上のKVキャッシュを削除しません。

管理対象はローカルHTTP `/v1`です。ホスト測定では `127.0.0.1`、Docker測定ではコンテナから届く `0.0.0.0` でlistenします。Docker APIを省略すると、同じポート・パスの `host.docker.internal` に置き換えます。認証が必要なら `DS4_API_KEY` または `auth.bearer_token` を指定できますが、これはAPIリクエスト用でサーバーに認証機能を設定するものではありません。tokenは保存JSONやコマンド引数に含めません。HTTPリダイレクトと環境変数由来のHTTPプロキシは使いません。

サンプリングはモデルごとのYAMLの `ds4_sampling` で選びます。RevBenchの `revbench_sampling` と同じ優先順位です。

| `source` | 動作 |
| --- | --- |
| `server_defaults` | サンプリング値を送らずDS4に任せる。`ds4_sampling` を省略した場合もこれを使用。 |
| `model_config` | `ds4_sampling` 内の数値を送る。`request` やCLIより優先し、省略した項目はサーバーに任せる。 |
| `run_config` | `request` / CLIの値を使う。共通既定値の温度0.0も適用される。 |

DeepSeekの例は `server_defaults`、GLMの例はRevBenchと同じ以下の設定です。

```yaml
ds4_sampling:
  source: model_config
  temperature: 1.0
  top_p: 0.95
  top_k: 0
  min_p: 0.0
```

モデル設定で指定できる値はtemperature 0〜2、top_p 0より大きく1以下、top_k 0〜1024、min_p 0〜1、seed 0〜9007199254740991です。top_k/min_pの0はフィルター無効、seedの0はDS4では固定seedを使わない指定です。`run_config` では同じ項目を `request` 内に書けます。`server_defaults` / `run_config` の `ds4_sampling` に数値は併記できません。以前のds4設定で `request.temperature` / `top_p` を引き続き送る場合は `source: run_config` を明示してください。

方針は設定読み込み時に確定し、準備推論・継続応答・Dockerの全ツールターンへ適用します。制御用の `ds4_sampling` 自体はAPIへ送りません。履歴・比較条件・各リクエストログには送信値と `sampling_source: ds4_server_defaults / ds4_model_config / ds4_run_config` を記録します。省略した値をサーバーの既定値で補完せず、内部の実効値は未確認として扱います。

出力上限と思考設定は別に扱い、各例は `request.max_tokens: 8192`、`request.reasoning_effort: high` を送ります。`minimal/low/medium/high/xhigh` は通常の思考、`max` は十分なコンテキストがある場合の Think Max で、不足時は通常の思考へ戻ります。`ultra` は拒否します。出力上限がコンテキスト長以上ならロード前に拒否し、入力を含む超過はサーバーのエラーとして記録します。

GLM-5.3-Flashがモデル一覧になく `glm-5.2` がある場合のみ、個別の `/v1/models/glm-5.3-flash` の完全一致するIDを確認します。これはAPI互換名の確認で、別のモデルへの自動置換やGGUF照合ではありません。

API名は互換エイリアスで、実際の重みは起動時のGGUFで決まります。起動設定のSHA-256は重みのSHA-256とは別です。GGUFハッシュ・実行バイナリの版・実効サンプリング・思考・投機状態はAPIから照合できないため、`capture_status: partial` として別runの平均に混ぜません。再開時は保存した起動設定との一致が必要で、既知の差は拒否します。不明な実効条件を許容する再開には `--allow-unverified-resume` を使います。

手動起動済み・リモートのサーバーを使う場合だけ、`ds4` ブロック全体を次に置き換えます。

```yaml
ds4:
  management: external
runs:
  cold_runs: 0
  warm_runs: 3
```

externalモードは起動・停止・ロード操作をせず、準備推論後のwarmだけを `external-server-repeat-v1` として記録します。ロード・ロード直後の初回時間は不明です。

今回の検証は通信と実プロセス起動を禁止したモックテストに限定し、実機のロード・推論・Docker操作は実施していません。実行中のベンチ終了後に設定例を `--config` へ渡してください。`--check` もホスト診断やDocker確認（externalではモデルAPI確認）を行うため、他の計測中には実行しないでください。

## 測定条件と cold / warm

各 run の `conditions` に、モデル本体の SHA-256（分割 GGUF は全 shard）、実行エンジンの版、要求設定とプロバイダーが返した実効設定、思考・投機デコードの状態、問題・入力バイナリ・ベンチ実装のハッシュ、Docker image ID、ホスト構成を保存します。ローカルで読めないモデルファイルや API が返さない値は不明です。Unsloth の推奨 inference 設定、要求された投機モード、インストール済みエンジンの版は、実際に使用された値と分けて保存します。認証用 token/password は測定条件に含めません。

- **ロード**: 対象モデルを unload → load し、準備操作の時間を `lifecycle` / `timings.load_wall_sec` に保存します。モデルが「既にロード済み」と返した場合は cold の成立を確認できないため停止します。
- **cold**: 毎試行、再ロード後に最初の推論を実行します。ロード時間は TTFT / 推論時間に含めません。
- **warm**: 初回推論後の反復です。warm だけを実行する場合や再開後は、統計から除外する準備推論を1回実行します。Docker では最初の未完了問題を準備推論に使います。
- **キャッシュ**: OS のページキャッシュやサーバーの KV/prompt キャッシュを強制消去したとは見なしません。確認できない状態は `unknown` です。

Docker の cold/warm は問題セット単位です。セット内の最初の問題だけがロード後の初回となり、各問には `measurement.inference_stage` を保存します。初回・反復時間の合計も別保存します。時間の合計には、再試行で置き換えたエラー試行も含め、成績・速度の集計とは分けて保持します。失敗した要求の後で初回かどうか確認できない推論は `unknown_after_error` として分けます。1試行または1問が複数の API 呼び出しを含む場合、その区間全体の時間であり、個々の呼び出しは既存の `turn_usage` を参照します。

`Compare` の比較軸（モデル / 実行エンジン / 量子化 / 並列数）を選ぶと、比較軸以外の条件差と未確認項目を表示します。条件の異なる結果、不明条件を含む別 run、再開を含む別 run、測定条件のない旧形式は同じ平均に混ぜません。中断・失敗した run はランキング集計から除外し、保存済みの試行を Run Details に残します。Run Details ではロード時間、初回・反復時間、条件、診断結果を確認できます。

## 実行前診断と競合防止

```bash
python benchmark.py --config configs/bench.yaml --check
```

接続・認証・対象モデルを確認します。Docker Task では spec/answer key、Docker daemon、ローカルの image と platform も確認し、実行には検証した image ID を使います。同じサーバーを使うベンチ用コンテナが残っている場合も、重複実行を防ぐため停止します。診断だけではモデルをロードしません。実行時にも同じ診断を行い、メモリ・swap・推論プロセスの RSS を実行前とロード直後に記録します。これらはベンチを起動したホストの情報です。

同じユーザー・同じマシンで、この CLI が同じサーバーへ重複実行することを OS ロックで防ぎます。`localhost` / `127.0.0.1` / `::1` / `host.docker.internal` と API パスの違いは同一視します。終了・強制終了時にロックは解放されます。別のツール・別ユーザー・別マシンからの推論はこのロックの対象外です。

## 逐次保存・中断再開・エラー再試行

1試行・Docker の1問が終了するたびに `runs/logs/<run-id>/events.jsonl` と `checkpoint.json` を同期保存します。checkpoint は一時ファイルから置換し、置換前に停止した場合はイベントログから完了分を復元します。`Ctrl+C` / SIGTERM では完了分を履歴にも残します。強制終了時は checkpoint を使って再開してください。Docker の中断・タイムアウト時は、その問題のコンテナを削除します。

```bash
# 中断した run の未完了部分を再開
python benchmark.py --config configs/bench.yaml --model MODEL --resume-run-id RUN_ID

# 完了した run の記録済みエラーだけを再試行（正解・不正解の回答は維持）
python benchmark.py --config configs/bench.yaml --model MODEL --retry-errors-run-id RUN_ID

# Docker の指定したエラー問題だけを再試行
python benchmark.py --config configs/bench_d_compile_arm64.yaml --model MODEL \
  --retry-errors-run-id RUN_ID --questions q2 q5
```

再開時は元と同じ設定・モデル・parallelism を指定します。sweep で作った run なら `--parallelism` で元の1値を指定してください。保存済みの問題や設定、GGUF、実装、Docker image ID などに差があれば再開を拒否します。実効設定や実行エンジンの版を API で確認できない場合に限り、`--allow-unverified-resume` を追加して不明条件を許容できます。既知の条件差を無視する指定ではありません。キャッシュ状態だけの不明は再開を妨げませんが、別 run との合算を防ぎます。

再試行前の checkpoint（元回答・ログを含む）は `revisions/` に保存し、同じ run ID の履歴を更新します。再試行中に中断しても `--resume-run-id` で残りの再試行対象を引き継げます。未完了問題と記録済みエラーが混在する run は、まず resume で未完了部分を終えてから retry します。旧形式には checkpoint がないため再開できません。

## Docker Task ベンチ

`configs/bench_d_compile_arm64.yaml` のような `mode: docker_task` 設定を使う場合は、先に Docker イメージを build してください。

```bash
./build_bench_image.sh --platform linux/arm64
python benchmark.py --config configs/bench_d_compile_arm64.yaml
```

この build では Ghidra 本体に加えて、`mecha_ghidra` (`ghidra_mcp`) も GitHub から取得して image に同梱します。clone 後に host 側へ `ghidra_mcp` を別インストールしなくても、そのまま `docker_task` ベンチを実行できます。手元の `ghidra_mcp` checkout を優先して試したい場合だけ、`LOCAL_LLM_BENCH_DOCKER_GHIDRA_MCP_SOURCE_PATH` か `REV_BENCH_DOCKER_GHIDRA_MCP_SOURCE_PATH` を設定してください。

## MAFC ベンチ

SECCON Beginners CTF 2025のMAFCを追加しています。公式配布ZIP（Windows x64実行ファイルと暗号化データ）をそのまま使い、Ghidraによる静的解析とフラグ復元を評価します。正解はホスト側に分離し、大文字・小文字を区別して採点します。取得元・固定コミット・ハッシュは [MAFCの説明](benchmarks/mafc/README.md) と [source.json](benchmarks/mafc/source.json) を参照してください。

| 設定 | 対象 |
| --- | --- |
| [configs/bench_mafc_arm64_lmstudio_4_models.yaml](configs/bench_mafc_arm64_lmstudio_4_models.yaml) | LM Studioの4モデル、保存済み推論設定 |
| [configs/bench_mafc_arm64_omlx.yaml](configs/bench_mafc_arm64_omlx.yaml) | oMLX / Qwopus3.8、保存済み推論設定 |
| [configs/bench_mafc_arm64_mlx_serve.yaml](configs/bench_mafc_arm64_mlx_serve.yaml) | mlx-serve / MLXモデル、サーバー既定の推論設定 |
| [configs/bench_mafc_arm64_ds4.yaml](configs/bench_mafc_arm64_ds4.yaml) | ds4 / DeepSeek-V4-Flash |
| [configs/bench_mafc_arm64_ds4_qwen3_8_flash_next.yaml](configs/bench_mafc_arm64_ds4_qwen3_8_flash_next.yaml) | ds4 / Qwen3.8-Flash-Next Q2 / MTP |
| [configs/bench_mafc_arm64_ds4_glm_5_3_flash.yaml](configs/bench_mafc_arm64_ds4_glm_5_3_flash.yaml) | ds4 / GLM-5.3-Flash |

Inspect AIの専用venvと `local-llm-bench:inspect-v1` イメージを用意し、既存ベンチが終了してから実行します。

```bash
cd /Users/samsepi0l/local_llm_bench
.venv/bin/python benchmark.py --config configs/bench_mafc_arm64_lmstudio_4_models.yaml
```

1モデルだけなら `--model qwen3.8-flash-next` を追加してください。cold 1回・warm 3回で測定し、既存WebビューアーでMAFCを選択して結果を確認できます。配布実行ファイルの起動は行わない問題設定です。

## 生成物

- `runs/history.json`: 実行履歴
- `runs/latest_run.json`: 最新の生データ
- `docs/index.html`: `history.json` を読む可視化ビュアー

`docs/index.html` はデータを埋め込まず、既定で `../runs/history.json` を読みに行きます。通常のベンチ実行ではテンプレートと同一なら既存のビュアーを再利用し、更新があれば自動再生成します。`--refresh-report` は強制再生成です。`file://` で開いて自動読込できないブラウザでは、`Open history.json` ボタンから手動で読み込めます。

複数の prompt を同じ `history.json` に溜めても、Viewer 上部の `Prompt` セレクタから prompt 単位で絞り込んで Leaderboard / Compare / Run Details を見比べられます。複数 prompt がある場合は、既定で最新 run の prompt が選ばれます。

Leaderboardの「推論性能 — 中央値」は、RC4などの通常プロンプト・性能測定・d-compileなどのタスク結果を表示します。タスク名・設問構成・測定条件・cold/warm・計測元ごとに集計し、成功した試行の中央値を表示します。新しいタスク履歴のE2Eと総合TPSはツール実行を除く推論時間に基づきます。旧履歴のタスク全体の時間とは集計を分け、旧履歴に保存されていないPP/TG TPS・TTFT・TPOTは「—」とします。計測実装の修正は次回の実行から反映され、過去の時刻は再実行なしには復元できません。

Leaderboardの「実行元」列、Compare、Run Detailsに `LM Studio` / `oMLX` / `mlx-serve` / `ds4` / `Unsloth Studio` を表示します。モデルの選択肢にも実行元を付け、同じモデル名でも区別できます。表示は履歴に保存されたproviderに基づき、記録がない古い結果は「不明」です。サーバーの現在の稼働状態を確認する表示ではありません。

モデル名には元の表示名だけを使い、形式名・評価基盤・長い条件IDを連結しません。同じモデル・実行元で条件の異なる結果がある場合は「測定 1」「測定 2」で区別します。選択肢は「モデル名 · 実行元 · 測定番号」、Inspect AIの版と測定条件IDはRun DetailsのRun Contextで確認できます。表示を短くしても、集計・比較・選択には完全な識別子を使い、別条件の結果を混ぜません。

LM Studio / Unsloth Studio から取得できる場合は、各 run に `display_name` / `format` / `quantization` も保存し、Leaderboard と Run Details に表示します。同じ `model` 名でも `selectedVariant` や `identifier` が異なる場合は、viewer 側で別モデルとして分けて表示します。

`Compare` タブでは、任意の 2 モデルを選んで Warm TTFT / Latency / Decode Speed などの主要指標を横並びで比較できます。

各 run には `telemetry` ブロックも保存されます。cold/warm の attempt、Docker Task の question、cooldown を span として記録し、wall time、LLM latency、TTFT、decode speed、token 数、process CPU/RSS sample を後から追えるようにしています。各 record / question result には `turn_usage[]` も保存され、turn ごとの `prompt_tokens` / `completion_tokens` / `cumulative_total_tokens` / `elapsed_sec` / `completion_tokens_per_sec` と、文字数ベース概算の `prompt_breakdown` を確認できます。Viewer の `Telemetry` タブでは RevBench と同じ形式の Run Averages / Question Averages を表示し、測定条件を含めて同一の scope 内を実行回数で平均化して、生成量と生成時間、生成量と生成速度、入力文脈と待ち時間、入力文脈の増分と待ち時間、per-turn context/速度、prompt composition を確認できます。古い履歴に `telemetry` や `turn_usage` が無い場合も、既存の `records` や Docker trace から可能な範囲で復元して表示します。

## テスト

```bash
python -m unittest discover -s /Users/samsepi0l/local_llm_bench/tests -v
```
