# 自前イメージを使う TP=2 検証の準備

目的は、対話側が実機で実行できる構成と手順を用意すること。実機検証そのものは行わない。
この依頼と AGENTS.md、CLAUDE.md の準備作業の境界を、計画・テスト・実装・レビュー・修正の全段階で守る。
cc-sdd のスキルは使わない。Git の commit・push・マージ、外部への投稿も行わない。

## 確定していること

- 固定した vLLM に C++ 修正と FlashInfer 0.7.0 を組み合わせ、head の 4 層・ダミー重みで HTTP 200 を確認済み。
- イメージ ID: `sha256:9df45888d2d726a1818be1005ace819808d4a1e8b4ec01a3efa7bc7f10a40c90`。
  linux/arm64、23,438,274,807 バイト。タグは `vllm-nope:0961bbae-fi070`。
- 実重み、TP=2、品質、性能は未検証。`pip check` の既知の 2 件は実行結果の説明を参照する。
- 重みは `RedHatAI/GLM-5.3-Flash-NVFP4@18d55bfd5a2194887738da73753975c9d3842f46`。
  マニフェストは 19 ファイル、197,881,153,655 バイト (約 184.29 GiB)。各ノードに全体が必要。
- 2026-09-23 の読み取り確認では、models/ は両ノードとも空、GPU プロセスと Serving Kit のコンテナも 0 件。
  空き容量は head が 2,516,964,708,352 バイト、worker が 2,803,729,174,528 バイトだった。
  これは過去の観測値であり、実行直前には対話側が再確認する。
- 通信の根拠は既存の実測記録。ケーブル 1 本、IB の all-reduce が 186.9 Gbps。
  この値からイメージ転送やインターネットのダウンロード時間を見積もらない。

## 読む資料

- `serving/config/configs.toml` の `p1-nvfp4-tp2`、`p1-fetch-nvfp4`、NCCL の記録設定。
- `serving/config/nodes.toml` と上記の重みマニフェスト。
- `docs/vllm-baseline/procedure.md` の関門 B と段 2.1〜2.3、停止と記録回収の説明。
- `docs/results/2026-09-22-nope-build.md`、`docs/results/2026-09-22-netcheck-bandwidth.md`。
- `experiments/nope-mla/configure_probe.py`、構成を読み込み・選択・起動計画に変換する `serving_kit` の公開 API と試験。
- 生データのうち読んでよいのは `serving/var/nope-build-0961bbae/image-inspect.json` のみ。
  他の `serving/var/`、取得した上流ソース、第三者のレシピ、モデルカードは読まない。

## 成果物

書き込み先は次の 3 ファイルに限る。実行時に生まれる一時ファイルと takt のレポートは除く。
既存の構成、Serving Kit 本体、既存テスト、ワークフローや規範ファイルは変更しない。

1. `experiments/nope-mla/configure_tp2.py`
   - image inspect の JSON と出力先を引数に取る、ネットワークも subprocess も使わない生成器。
   - 上記の完全なイメージ ID と linux/arm64 を要求する。サイズは正の整数。入力が不正なら出力を作らない。
   - 出力先が既存なら上書きしない。
   - `p1-nvfp4-tp2` を基に、新しい `p2-nope-tp2-smoke` を生成する。
   - 変更は名前・説明・イメージの節、`max-model-len=4096`、`max-num-seqs=1`、初回の NCCL 記録の 3 変数だけ。
   - `NCCL_DEBUG=INFO`、`NCCL_DEBUG_SUBSYS=INIT,NET`、`NCCL_DEBUG_FILE=/logs/nccl.%h.%p.log`。
     根拠は既存の設定と手順の同じ項目から引き継ぐ。
   - 短い文脈と同時実行 1 は、初回に起動と応答だけを確認するための運用上の選択として理由を書く。
   - head / worker、TP=2、nnodes=2、node-rank、headless、モデルの版、IB の指定、その他の引数と出典は維持する。
   - イメージの根拠は実行結果の文書に結び付ける。実重みで起動済みとは書かない。
2. `serving/tests/unit/test_configure_tp2.py`
   - 生成した TOML を実際の `load_configs` / `load_nodes` / `select_config` / `build_plans` で検証する。
   - 2 台の起動計画の役割・イメージ・短い文脈・IB・NCCL 記録を確認する。既存の契約も維持されることを確認する。
   - 不正な ID、アーキテクチャ、不正サイズ、既存出力の拒否と、既存出力の非破壊を確認する。
   - ソース文字列検査だけで済ませず、生成器を通る検証にする。実機には接続しない。
3. `docs/vllm-baseline/patched-tp2-procedure.md`
   - Mac での準備、イメージを head から worker に移す操作、両側の ID 照合、重み取得・照合、起動、IB 確認、短い応答確認、ログ回収、停止の順。
   - Spark 上のソース編集や再ビルドは不要。同じイメージを保存・転送・読み込みする計画にする。
   - `docker save` / 転送 / `docker load` は実行せずコマンドとして示す。Mac を経由し、Spark 間の SSH 鍵があるとは仮定しない。
   - 重み取得には既存の `p1-fetch-nvfp4` を使い、モデルカード・トークンを取得しない。
   - 2 台分の実重み約 368.58 GiB に、イメージと一時アーカイブが加わる。必要容量と回収・保持方針を明示する。
   - 所要時間は実効帯域を仮定した計算として示し、実測とは区別する。取得の既定 8 時間を超えた場合や中断時の取得コンテナの扱いも記す。
   - 起動前の 8 関門、初回 `--timeout 3h`、2 台の NCCL が IB であること、ready と短い要求の HTTP 200 を進む条件とする。
   - `serve smoke` は応答本文を stderr に出す。意味の確認は対話側で行い、stderr をファイルへ保存しない。記録するのは状態・長さ・判定だけ。
   - 最後は `serve logs` の後で `serve stop`。停止対象は事前に所有を照合する。この試行でサーバーを常駐させない。
   - 失敗・不一致時は記録して停止する。勝手に別の重み・別の設定へ進まない。
   - 実行前には対象の状態変更と容量を示し、計測者の了承を得る。長文脈・性能・長時間運転の検証は次の段階とする。

## 完了条件

- 上記 3 ファイルが揃い、構成生成とローカルの起動計画の検証が通る。
- `cd serving && uv run pytest && uv run ruff check . ../experiments/nope-mla && uv run ruff format --check . ../experiments/nope-mla && uv run mypy` が通る。
- 実機操作は 0 件。作成したファイル、検証結果、実機で未確認の事項を日本語で報告する。
- 現在の未コミットの takt 導入ファイルは先行する別作業であり、このタスクの不備として変更・差し戻しを要求しない。
