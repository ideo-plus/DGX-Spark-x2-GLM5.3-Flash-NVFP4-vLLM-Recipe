# serving-kit

`serving-kit` は、作業用の Mac から DGX Spark 2 台に vLLM の推論サーバーを立てて、確かめて、
止めるためのコマンドラインの道具 (Serving Kit) である。`bench/` と同じ流儀 (src レイアウト、
uv、`uv.lock` をコミット、ruff、mypy strict、pytest) で作る。

リポジトリの正本は Mac にある。Spark は実行するだけの場所で、Spark の上でファイルを編集しない
(配布は `serve push`、実行は ssh)。

```bash
cd serving
uv sync
uv run serve --help
uv run serve --version
```

## 使い方 (段の順)

段の順の詳細は [`docs/vllm-baseline/procedure.md`](../docs/vllm-baseline/procedure.md) に譲る。
サブコマンドと引数は、下の「サブコマンド」の表が正である。

```bash
# 1. 2 台に置き場所を作って、payload/ を配る (状態を変える)
uv run serve push --yes

# 2. イメージを、ダイジェストで 2 台に取得する (状態を変える)
uv run serve pull-image p1-fetch-nvfp4-probe --yes
uv run serve image-licenses p1-image-licenses --yes

# 3. 重みのマニフェストを Mac で作り、段 0 に要る設定とトークナイザだけを 2 台に取得して照合する
uv run serve manifest RedHatAI/GLM-5.3-Flash-NVFP4 <40 桁の revision>
uv run serve fetch p1-fetch-nvfp4-probe --probe-files --yes
uv run serve verify probe-pinned --probe-files --yes

# 4. 1 台の縮小の確認
uv run serve probe probe-pinned --timeout 45m --yes

# 5. 通信の確認 (links は読み取りだけ)
uv run serve netcheck links
uv run serve netcheck bandwidth netcheck-bandwidth --yes
uv run serve netcheck sanity netcheck-sanity --yes

# 6. 重みの本体を 2 台に取得して照合し、2 台で TP=2 を起動する
uv run serve fetch p1-fetch-nvfp4 --yes
uv run serve verify p1-fetch-nvfp4 --yes
uv run serve start p1-nvfp4-tp2 --timeout 3h --yes

# 7. 確かめる、見張る、片付ける
uv run serve status
uv run serve smoke p1-nvfp4-tp2
# 応答の長さの上限は、既定の 64 のまま。その回だけ上げたいときに --max-tokens を書く
uv run serve smoke p1-nvfp4-tp2 --max-tokens 512
uv run serve watch p1-nvfp4-tp2 --duration 2h --interval 30s
uv run serve logs p1-nvfp4-tp2
uv run serve stop --yes
```

構成の定義は `serving/config/configs.toml`、ノードの定義は `serving/config/nodes.toml` にある
(設定の 1 つ 1 つに、値と理由と出典が付いている)。ノードの直結の値 (`fabric_*`) は
`serve netcheck links` の実測で埋めるまで空で、埋めるまで 2 台の構成は選べない (終了コード 1)。

## サブコマンド

| コマンド | すること | Spark の状態を変えるか (了承が要るか) |
|---|---|---|
| `serve check <構成>` | 構成の検査と、8 つの関門を流して結果を並べる | 変えない |
| `serve push` | 2 台の `remote_root` の下に 6 つの置き場所を作り、`serving/payload/` を `payload/` に配る | **変える** |
| `serve pull-image <構成>` | イメージを、ダイジェストで 2 台に取得して照合する | **変える** |
| `serve image-licenses <構成>` | イメージの中のライセンスの表記を読んで出す (`kind = "inspect"` のコンテナを起こし、読み終えたら消す) | **変える** |
| `serve manifest <repo> <revision>` | Mac で、重みのマニフェストを作る | 変えない (Spark に触らない) |
| `serve fetch <構成> [--probe-files]` | 重み (または、設定とトークナイザだけ) を 2 台に取得して照合する | **変える** |
| `serve verify <構成> [--probe-files]` | 2 台の重みを、マニフェストと照合する | **変える** (照合の記録の 1 ファイルだけ) |
| `serve start <構成> [--timeout <秒>]` | 2 台で起こし、受け付けの開始まで待つ | **変える** |
| `serve stop` | 自分のコンテナを 2 台とも止めて消し、GPU が空くまで待つ (`kind` を問わない) | **変える** |
| `serve status` | 2 台と推論サーバーの、いまの状態を示す | 変えない |
| `serve smoke <構成> [--max-tokens <数>]` | 英語と日本語の短い要求を 1 つずつ送る (応答の長さの上限は、既定 64。`--max-tokens` は 1 以上の整数で、その回だけ上書きする) | 変えない |
| `serve logs <構成>` | 2 台の記録を `serving/var/` に写す | 変えない |
| `serve probe <構成> [--timeout <秒>]` | 1 台の縮小の確認 | **変える** |
| `serve netcheck links` | 直結のインターフェースを読み、ケーブルの本数を判断する | 変えない |
| `serve netcheck bandwidth <構成>` | 自前の all-reduce で、2 台の間の帯域を測る | **変える** |
| `serve netcheck sanity <構成>` | 推論サーバーの公式の資料が示す、事前の確認を流す | **変える** |
| `serve netcheck ab <構成> --env K=V [--repeat <回>]` | 足す設定の A/B を、交互に流して比べる | **変える** |
| `serve watch <構成> [--thermal-threshold <℃>]` | 連続の負荷の間、外から見張る (読み取りだけ。GPU の温度・SM クロック・電力・使用率、ACPI 熱区域、hwmon (`mlx5`/`nvme`/`acpitz`) の温度、コアごとの CPU 使用率を観察し、`unresponsive`/`stalled`/`thermal` の出来事を見つける) | 変えない |
| `serve thinking <構成>` | thinking の深さの渡し方を確かめる (HTTP だけ) | 変えない |

すべてのコマンドに共通の引数:

| 引数 | 既定 | 意味 |
|---|---|---|
| `--configs PATH` | `serving/config/configs.toml` | 構成の定義のファイル |
| `--nodes PATH` | `serving/config/nodes.toml` | ノードの定義のファイル |
| `--var-root PATH` | `serving/var/` | 記録の置き場所の根 (git の管理の外) |
| `--yes` | なし | 計画を見せたうえで、了承を求めずに進む |

秒を受ける引数 (`--timeout`、`--duration`、`--interval`、`--stall-window`) は、単位を付けて
書ける (`30`、`30s`、`5m`、`2h`)。正で有限の値だけを受け、`nan`、`inf`、0、負は、Spark に
触る前に断る。

`serve watch` の `--thermal-threshold <℃>` は、摂氏の温度で、既定は 90。いずれかの熱区域が
この値以上になった観察の立ち上がりで `thermal` の出来事を 1 件にする (`unresponsive`/`stalled`
と違い、記録の回収はしない。熱では推論サーバーは壊れていないため)。熱区域と hwmon の一覧は、
見張りの開始時に 1 回だけ発見する。台ごとに、熱区域の番号 32 個と hwmon の番号 32 個を順に
`cat` するので少なくとも 64 回、さらに `mlx5`/`nvme`/`acpitz` の名前が採用された hwmon 1 つ
ごとに、温度センサーの番号を最大 32 回 (見つかった分は、ラベルの読み取りも加わる) 試す。
**最初の観察は、この発見が終わってから始まる**ので、少し時間がかかる。台に届かなければ、
その台の発見だけを打ち切り、`detail` に記録して見張りは続ける (README の「終了コード」の表
は変わらない)。

## ローカルでビルドしたイメージ

自前ビルドのイメージは、完全なローカル ID (`sha256:` と 64 桁) でも指定できる。
この場合は、各ノードの `docker image inspect` の `.Id` と照合してから起動する。
タグと短縮 ID は受け付けず、`serve pull-image` もローカル ID を拒否する。
ビルドの手順は [NoPE 修正イメージの検証](../docs/vllm-baseline/patched-build-procedure.md) を参照する。

## 終了コード

| 値 | 意味 | 例 |
|---|---|---|
| 0 | 正常 (すでに望む状態だった場合を含む) | すでに同じ中身で動いている構成への `start`、停止済みへの `stop` |
| 1 | 前提の不足、断り | 構成の定義の誤り、関門の不通過、了承されなかった、ssh で入れない、`serve status` で読めなかった台がある、`serve probe` の「判定できなかった」、`serve thinking` の「判定できなかった」 |
| 2 | 実行して失敗した | 時間切れ、コンテナの終了、名前の衝突、照合の不一致、GPU が空かない、確認の不合格、`serve watch` が出来事を見つけた |
| 130 | 中断 (Ctrl-C) | — |

例外から終了コードへの写しは、`cli.main` の 1 か所 (`_EXIT_BY_ERROR` の表) にある。誤りは
traceback を出さずに、`エラー: <1 行>` として標準エラーに出す。予期しない失敗も 1 行で 2 に
なる (`SERVE_DEBUG=1` を付けたときだけ、詳しい出所を足す)。

`serve` が引数の使い方そのものを断るとき (知らないフラグ、足りない位置の引数) は、`argparse`
の決まりで 2 になる (`bench` と同じ。上の表の「実行しての失敗」とは別のもの)。

## 出力の形

- **標準出力**は、後の処理が読める決まった形だけである。1 行に 1 つの `key=value` と、`|` で
  始まる表 (`serve status`) を出す。`detail=` は、複数行でも 1 行に収める (改行、復帰、タブ、
  逆斜線を `\n`、`\r`、`\t`、`\\` に書き換える)
- **標準エラー**には、進捗、計画、了承の問いかけ、警告、記録の末尾、`detail` の全文、誤りを
  出す
- 例外は `serve image-licenses` の 1 つだけで、`key=value` の行のあとに、空の行を 1 つ置いて、
  読み取った表記の本文をそのまま出す

## Spark の上の置き場所

`serve push` が、2 台の `remote_root` (ノードの定義に書く) の下に、この 6 つだけを作る。

| 置き場所 | 何を置くか | 配布の宛先にできるか |
|---|---|---|
| `payload/` | Spark で流すスクリプト (`serving/payload/` の中身) | できる |
| `models/` | 重み (本体) | できない |
| `probe/` | 縮小の確認用の、設定とトークナイザ | できない |
| `cache/` | JIT などのキャッシュ | できない |
| `logs/` | 通信の記録 (NCCL) | できない |
| `state/` | 起動の記録と、照合の結果の記録 | できる |

作ることと配ることは別である。`serve push` は、6 つを `mkdir -p` で作ってから、`payload/`
だけを配る (置き場所を消す道は、この道具のどこにもない)。重み、確認用、キャッシュ、記録の
置き場所には配れない (`remote` が断る)。

回収した記録は、Mac の `serving/var/<UTC の日時>-<コマンド>-<構成>/` に入る (`.gitignore`
済み。`bench` の `results/` とは別の場所)。

## 了承の流れ (requirements 2.1)

状態を変えるコマンドは、次の順に進む。

1. 読み取りだけの関門を流す (入れるか、自分のコンテナ、GPU、置き場所、イメージ、重みの照合、
   ディスクの空き、ポート)
2. これから流すコマンドと、その巻き戻しと、対象の機械を、**標準エラーに全部見せる**
3. 端末で `yes` と打つのを待つ (`yes` との完全な一致だけを了承として通す)
4. 了承を得た計画に入っていない、状態を変える呼び出しは、`remote` が断る

端末でなく `--yes` もないときは、**状態を変える呼び出しを 1 つも出さずに**終了コード 1 で
終わる。

**`--yes` の扱い**: `--yes` は、計画を見せたうえで了承を求めずに進める。Claude が計測者の
代わりにコマンドを打つときは、**会話の中で計測者の了承を得てから** `--yes` を付ける
(CLAUDE.md の「状態を変える操作は、実行する前に確認を取る」)。自動で付けてはいけない。

## 試験の流し方

```bash
cd serving
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

試験は、実物の ssh、rsync、docker、推論サーバーに、どの段でもつながない (`FakeRunner` と
`fake_vllm.py` を相手にする)。実際の時間も待たない。実機での確かめは、試験ではなく、手順書の
段として行う。

依存のライセンスは、リポジトリ直下の [`LICENSES.md`](../LICENSES.md) にある。
