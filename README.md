# local_llm_bench

LM Studio / Unsloth Studio の OpenAI互換 API を使って、複数のローカルLLMを同一プロンプト条件で比較する簡易ベンチマークです。

## 使い方

```bash
python /Users/samsepi0l/local_llm_bench/benchmark.py --config /Users/samsepi0l/local_llm_bench/bench.yaml
```

既定では各モデルのベンチが終わるたびに provider に応じた unload を実行します。LM Studio では `lms unload`、Unsloth Studio では `/api/inference/unload` を使います。ロード状態を残したいときだけ `--keep-loaded` を付けてください。

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
  --config /Users/samsepi0l/local_llm_bench/bench.yaml \
  --prompt-text "pythonでライブラリを使わずにAESを実装して"
```

LM Studio の `api_base` をCLIで上書きする例:

```bash
python /Users/samsepi0l/local_llm_bench/benchmark.py \
  --config /Users/samsepi0l/local_llm_bench/bench.yaml \
  --api-base http://localhost:1234/v1
```

出力先をまとめて変える例:

```bash
python /Users/samsepi0l/local_llm_bench/benchmark.py --out-dir /Users/samsepi0l/local_llm_bench/out
```

アンロードせずに終える例:

```bash
python /Users/samsepi0l/local_llm_bench/benchmark.py \
  --config /Users/samsepi0l/local_llm_bench/bench.yaml \
  --keep-loaded
```

`bench.yaml` の `request.max_tokens` は、1試行で保存したい総出力上限です。モデルが `finish_reason=length` で途中停止した場合は、残り予算ぶんだけ自動で続きを取得して連結保存します。

`provider=unsloth_studio` では `api_base` と `docker.api_base` の override は受け付けません。docker_task ではコンテナ側から host の Studio に到達するため、内部で `host.docker.internal` 側の URL を使います。

## Docker Task ベンチ

`bench_d_compile_arm64.yaml` のような `mode: docker_task` 設定を使う場合は、先に Docker イメージを build してください。

```bash
./build_bench_image.sh --platform linux/arm64
python benchmark.py --config bench_d_compile_arm64.yaml
```

この build では Ghidra 本体に加えて、`mecha_ghidra` (`ghidra_mcp`) も GitHub から取得して image に同梱します。clone 後に host 側へ `ghidra_mcp` を別インストールしなくても、そのまま `docker_task` ベンチを実行できます。手元の `ghidra_mcp` checkout を優先して試したい場合だけ、`LOCAL_LLM_BENCH_DOCKER_GHIDRA_MCP_SOURCE_PATH` か `REV_BENCH_DOCKER_GHIDRA_MCP_SOURCE_PATH` を設定してください。

## 生成物

- `runs/history.json`: 実行履歴
- `runs/latest_run.json`: 最新の生データ
- `docs/index.html`: `history.json` を読む可視化ビュアー

`docs/index.html` はデータを埋め込まず、既定で `../runs/history.json` を読みに行きます。通常のベンチ実行では既存のビュアーを再利用し、テンプレートを更新したいときだけ `--refresh-report` で再生成します。`file://` で開いて自動読込できないブラウザでは、`Open history.json` ボタンから手動で読み込めます。

複数の prompt を同じ `history.json` に溜めても、Viewer 上部の `Prompt` セレクタから prompt 単位で絞り込んで Leaderboard / Compare / Run Details を見比べられます。複数 prompt がある場合は、既定で最新 run の prompt が選ばれます。

LM Studio / Unsloth Studio から取得できる場合は、各 run に `display_name` / `format` / `quantization` も保存し、Leaderboard と Run Details に表示します。同じ `model` 名でも `selectedVariant` や `identifier` が異なる場合は、viewer 側で別モデルとして分けて表示します。

`Compare` タブでは、任意の 2 モデルを選んで Warm TTFT / Latency / Decode Speed などの主要指標を横並びで比較できます。

## テスト

```bash
python -m unittest discover -s /Users/samsepi0l/local_llm_bench/tests -v
```
