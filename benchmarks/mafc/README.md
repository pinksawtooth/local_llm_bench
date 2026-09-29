# MAFC

[SECCON Beginners CTF 2025の公式問題](https://github.com/SECCON/SECCON_Beginners_CTF_2025/tree/4634eb65935a085def1c88abab2c67752d5fe822/reversing/MAFC)（author: JUCK、reversing / hard）を、フラグ復元の単問ベンチマークとして追加しています。

公式配布の `data/MAFC.zip` は無変更です。Windows x64の `MAFC/MalwareAnalysis-FirstChallenge.exe` と暗号化データ `MAFC/flag.encrypted` が入っています。ZIPから実行ファイルを選んでGhidraに取り込み、モデルにはZIPのパスを提示します。暗号化データはPythonの `zipfile` 等で同じZIPから読み出せます。実行ファイルは起動せず、静的解析・補助計算・復号で解くプロンプトです。

正解は公式 `FLAG` に基づく `spec.answers.yaml` に分離し、ホスト側で大文字・小文字を区別した完全一致相当の採点を行います。既存の `exact` は大小文字を無視するため、この問題では先頭・末尾を固定した `regex` を使います。正解、公式解説、solverはコンテナに渡しません。測定条件の `workload_hash` はZIP全体のハッシュを含むため、実行ファイルと暗号化データの両方を識別します。

取得元のコミット、ZIPと内容物のSHA-256は [source.json](source.json) に記録しています。

| 設定 | 対象 |
| --- | --- |
| `bench_mafc_arm64_lmstudio_4_models.yaml` | LM Studioの4モデル、各モデルの保存済み推論設定 |
| `bench_mafc_arm64_ds4.yaml` | ds4 / DeepSeek-V4-Flash |
| `bench_mafc_arm64_ds4_glm_5_3_flash.yaml` | ds4 / GLM-5.3-Flash |

すべてInspect AI・cold 1回 / warm 3回です。推論条件と制限は対応するd-compile設定を引き継ぎ、Docker imageは `local-llm-bench:inspect-v1` を指定しています。`linux/arm64` は解析用コンテナのアーキテクチャで、解析対象PEのx86_64とは独立です。

既存ベンチの終了後、プロジェクト専用venvとInspect対応imageを用意して実行してください（導入手順はリポジトリのREADME参照）。

```bash
cd /Users/samsepi0l/local_llm_bench
.venv/bin/python benchmark.py --config bench_mafc_arm64_lmstudio_4_models.yaml
```

1モデルだけなら `--model qwen3.8-flash-next` を追加します。ds4は上表の該当設定を `--config` に指定します。結果は既存の `runs/history.json` / Webビューアーへ保存され、MAFCとして選択できます。
