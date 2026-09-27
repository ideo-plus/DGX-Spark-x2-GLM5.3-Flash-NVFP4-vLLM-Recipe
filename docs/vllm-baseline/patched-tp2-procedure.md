# 自前イメージで TP=2 の初回起動を確認する

この手順は、Mac から 2 台の DGX Spark に同じイメージと固定版の重みを用意し、TP=2 の初回起動と短い応答だけを確認するためのものです。自前イメージは `sha256:9df45888d2d726a1818be1005ace819808d4a1e8b4ec01a3efa7bc7f10a40c90`、タグは `vllm-nope:0961bbae-fi070`、サイズは 23,438,274,807 B (約 21.83 GiB) です。イメージの根拠は [2026-09-22 の実行結果](../results/2026-09-22-nope-build.md)です。4 層・ダミー重みの縮小起動は確認済みですが、実重み、TP=2、品質、性能、長時間運転は未確認です。Spark 上の編集や再ビルドは行いません。

書かないこと: 要求・応答の本文、認証情報、`exl3-tp2` の中身。

実行位置: 以降のコマンドはすべて、このリポジトリ直下 (`DGX-Spark-GLM5.3-Flash-Recipe/`) を作業ディレクトリとして実行します。`serve` サブコマンドは `cd serving` を使わず `uv run --directory serving serve …` の形で呼び、`--configs` はその起点 (`serving/`) からの相対パスで書きます。`python3`、`docker image save/load`、`ssh` の相対パスは、リポジトリ直下を起点とします。ブロックをまたいで作業ディレクトリを持ち越す `cd` はどこでも使いません。

## 1. 実行前の了承と容量

⚠ 計測者に、次の状態変更と容量を示し、了承を得てから進みます。worker へのイメージ読み込み、両ノードの `models/` への重み取得、`state/` の照合記録、コンテナの起動・停止、`logs/` の生成が発生します。

| 項目 | 必要量・扱い |
|---|---|
| 重み | 197,881,153,655 B × 2 台 = 395,762,307,310 B (約 368.58 GiB) |
| イメージ | 23,438,274,807 B (約 21.83 GiB)。worker に追加。head の既存イメージは ID を照合 |
| Mac の一時アーカイブ | 約 21.83 GiB。両側の ID 照合後に削除 |
| 過去の空き容量 | head 2,516,964,708,352 B、worker 2,803,729,174,528 B (2026-09-23 の観測値) |

空き容量は実行直前に両ノードで再確認します。`gate_disk_space` の余裕 (`SPACE_MARGIN_PERCENT`) を重みの合計に加えて判定し、イメージと一時領域も含めます。過去の観測値だけで実行を承認しません。

## 2. Mac で構成を生成し、読み取り検査する

```bash
python3 experiments/nope-mla/configure_tp2.py \
  serving/var/nope-build-0961bbae/image-inspect.json \
  serving/var/nope-build-0961bbae/tp2.toml
uv run --directory serving serve check p2-nope-tp2-smoke --configs var/nope-build-0961bbae/tp2.toml
```

構成生成はネットワークや `subprocess` を使いません。既存出力がある場合は上書きせず停止します。`serve check` は読み取り検査で、状態を変えません。

`layout` は、構成が bind mount するディレクトリの存在を検査します。起動用構成は `{remote_root}/models/glm-5-3-flash-nvfp4`、`cache`、`logs` を使います。一方、取得用の `p1-fetch-nvfp4` が mount するのは親の `{remote_root}/models` です。`serve push` は親を含む 6 ディレクトリまでを作り、モデル専用の子ディレクトリは §5 の取得が作ります（`serving/config/configs.toml` の取得用構成を参照）。

したがって、取得前は次の不通過を想定します。

- `layout`（両台）: **モデル専用の子ディレクトリだけ**がない場合。§5 で解消します。
- `image_digest`（worker）: 自前イメージが未移送の場合。§3 で解消します。
- `weights_verified`（両台）: 実重みをまだ取得・照合していない場合。§5 で解消します。

不通過の詳細を確認し、上記に該当する場合だけ準備を進めます。親の `models`、`cache`、`logs` の欠落、接続失敗、他の関門の不通過はここで止めて調べます。全 9 関門の通過は §6 の起動条件です。

## 3. Mac を経由してイメージを移す

Spark 間の SSH 鍵は仮定しません。まず head に期待する ID があり、worker にまだ同じ ID がないことを確認します。

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-153d \
  'docker image inspect --format "{{.Id}}" vllm-nope:0961bbae-fi070'
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-5083 \
  'docker image inspect --format "{{.Id}}" vllm-nope:0961bbae-fi070'
```

⚠ Mac に一時アーカイブを作り、Mac から worker に読み込みます。Docker 操作は実機で実行し、ここではコマンドだけを示します。

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-153d \
  'docker image save vllm-nope:0961bbae-fi070' \
  > serving/var/nope-build-0961bbae/vllm-nope-0961bbae-fi070.tar
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-5083 \
  'docker image load' \
  < serving/var/nope-build-0961bbae/vllm-nope-0961bbae-fi070.tar
```

転送時間は実測ではありません。アーカイブの実サイズを `S` B、head→Mac と Mac→worker の実効帯域をそれぞれ `X1`、`X2` B/s とすると、逐次転送は `S/X1 + S/X2` 秒に保存・読み込みの処理時間を加えたものです。イメージの `Size` は tar の実サイズとは限らないため、保存後にサイズを確認します。仮に `S = 23,438,274,807` B で両経路が同じ速度なら、50 MB/s で約 15.6 分、100 MB/s で約 7.8 分、500 MB/s で約 94 秒です。186.9 Gbps は IB all-reduce の実測であり、この転送時間には使いません。

## 4. 両側のイメージ ID を照合する

両ノードの次の出力が完全な ID と一致することを確認します。不一致、接続失敗、タグだけの一致は合格にしません。

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-153d \
  'docker image inspect --format "{{.Id}}" vllm-nope:0961bbae-fi070'
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-5083 \
  'docker image inspect --format "{{.Id}}" vllm-nope:0961bbae-fi070'
uv run --directory serving serve check p2-nope-tp2-smoke --configs var/nope-build-0961bbae/tp2.toml
```

この時点では `image_digest` が head・worker とも通ることを確認します。モデル専用ディレクトリの欠落による `layout` と、`weights_verified` は §5 まで不通過のままで構いません。いずれも tar の削除を妨げません。`image_digest` が 2 台とも通った後、Mac の tar を削除します。Spark に一時アーカイブは置きません。

## 5. 固定版の重みを取得し、照合する

⚠ `models/` の状態と容量を直前に再確認し、了承後に両ノードへ取得します。既存の `p1-fetch-nvfp4` を使い、モデルカードやトークンは取得しません。

```bash
uv run --directory serving serve fetch p1-fetch-nvfp4 --yes
uv run --directory serving serve verify p1-fetch-nvfp4 --yes
```

対象は `RedHatAI/GLM-5.3-Flash-NVFP4@18d55bfd5a2194887738da73753975c9d3842f46` です。既定の待ち時間は 28,800 秒 (8 時間) です。8 時間を超えても取得コンテナは止まらず続き、もう一度 `serve fetch` を実行すると待ちに戻ります。Ctrl-C や SSH 切断でも同じ扱いです。

中止する場合は、取得用構成 `p1-fetch-nvfp4` を指定して記録を回収してから停止します（§8 は推論用構成 `p2-nope-tp2-smoke` が対象で、コンテナが違うためここでは使えません）。

```bash
uv run --directory serving serve logs p1-fetch-nvfp4
```

回収先に head・worker 両方の `container.stdout.log` / `container.stderr.log` があり、`collect.json` の欠落内容を確認できることを確かめます。取得では `serve start` 専用の `<構成>.launch.json` を作らないため、両台の `item=launch_record` の欠落は想定どおりです。その 2 件だけは取得ログの回収失敗と区別します。それ以外の欠落や読み取り不能は、該当する台と項目を記録し、回収できたとは扱いません。`serve status` で自分のラベル (`vllm-baseline.owner=serving-kit`) のコンテナだけが停止対象であることを照合してから停止します。

```bash
uv run --directory serving serve status
uv run --directory serving serve stop --yes
```

8 時間で終えるための目安は、1 台あたり 197,881,153,655 B / 28,800 s ≈ 6.87 MB/s 以上です。これは仮定からの計算であり、実測ではありません。`mismatched=` が空でなければ停止し、別の重みを黙って取得し直しません。照合記録は重みの slug で参照されるため、TP=2 の起動構成の関門にも使われます。

## 6. 起動前の 9 関門と起動

```bash
uv run --directory serving serve check p2-nope-tp2-smoke --configs var/nope-build-0961bbae/tp2.toml
```

出力の `gate.N.name` が `reachable` / `own_state` / `gpu_idle` / `layout` / `image_digest` / `weights_verified` / `disk_space` / `memory_free` / `ports_free` の 9 種であること、head・worker とも `gate.N.passed` がすべて `true`、`gates_failed=0`、`status=passed` であることを確認します。構成のハッシュ (`config-sha256`) の照合は、この `serve check` の確認項目ではなく、次の `serve start` が同名で中身の違うコンテナを見つけたときに別途行う判定です。

起動の前の空きは、`MemFree` を読んで確かめます。`--load-format instanttensor` の構成なら下限を確かめます (issue #84。断られたときの対処は `ops/spark-drop-caches/README.md`)。

⚠ 起動による GPU プロセス、コンテナ、ログの状態変更を了承した後、初回だけ次を実行します。

```bash
uv run --directory serving serve start p2-nope-tp2-smoke \
  --configs var/nope-build-0961bbae/tp2.toml --timeout 3h --yes
```

進む条件は `status=ready` だけです。`status=already_running`（同じ構成のコンテナがすでに動いている。関門も了承も通していない）が出た場合は §7 へ進まず、§8 の所有確認・記録回収・停止へ回してから `serve check` をやり直します。`refused` または `failed` が出た場合は記録して停止します。別の重みや設定へ切り替えません。

## 7. IB と短い応答を確認する

まずログを回収します。

```bash
uv run --directory serving serve logs p2-nope-tp2-smoke --configs var/nope-build-0961bbae/tp2.toml
```

`collect.json` の `missing` が空配列であることを確認します。1 件以上あれば、その台の記録が写っていないため今回分を確定できず、計測へ進まず記録して停止します。

ログ回収は過去の NCCL ファイルも含むため、今回のコンテナに対応するファイルだけを読みます。両台で、まず所有ラベルと構成名で絞った一覧を確認します。

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-153d \
  'docker ps --no-trunc --filter label=vllm-baseline.owner=serving-kit --filter label=vllm-baseline.config=p2-nope-tp2-smoke --format "{{.ID}} {{.Names}}"'
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-5083 \
  'docker ps --no-trunc --filter label=vllm-baseline.owner=serving-kit --filter label=vllm-baseline.config=p2-nope-tp2-smoke --format "{{.ID}} {{.Names}}"'
```

各台で期待する名前 `vb-p2-nope-tp2-smoke-head` / `vb-p2-nope-tp2-smoke-worker` の 1 件だけがあることを確認します。その一覧から得た完全な ID **だけ**を対象に、`docker inspect --format '{{.Config.Hostname}} {{.State.StartedAt}}' <確認済みのID>` を各台の SSH 経由で読みます。対象を一意に確認できなければ止めます。他のコンテナは inspect しません。

`serving/var/<UTC>-logs-p2-nope-tp2-smoke/<役割>/logs/` のうち、`nccl.<確認したhostname>.<PID>.log` に一致するファイルを今回分として選び、コンテナの開始時刻とファイルの更新時刻も照合します。`%h` はコンテナ内の hostname、`%p` はプロセス PID であり、PID だけでは新旧を識別できません。ファイルは複数でも構いません。台ごとに今回分を一意に帰属でき、少なくとも 1 行の `Using network IB` があること、今回分にある経路確定行がすべて IB であることを確認します。`Socket`、不明な経路、確定行の欠落、過去分との識別不能なら計測へ進まず、記録して停止します。

```bash
uv run --directory serving serve status --configs var/nope-build-0961bbae/tp2.toml
uv run --directory serving serve smoke p2-nope-tp2-smoke --configs var/nope-build-0961bbae/tp2.toml
```

進む条件は `status=answered`、短い要求 2 件の HTTP 200、両方の応答の意味が通ることです。空の応答、意味不明な応答、判定不能の場合は成功扱いにしません。`serve smoke` の応答本文は stderr に出ます。本文の意味は対話側で確認し、stderr をファイルへ保存しません。記録するのは状態、`reply.*.http_status`、`*_tokens`、`stop_reason`、意味の通る文かの判定だけです。

### 7.1 上限を変えて短い応答を確認し直す

上の `serve smoke` は、応答の長さの上限が既定の 64 トークンです。`stop_reason` が `max_tokens` で応答の本文が空、または意味を判定できない場合は、上の条件のとおり成功扱いにせず、§8 のログ回収・停止へ回します。上限に達したこと自体を根拠に、原因を思考の長さと断定しません。

確認し直すときは、§8 で停止したうえで、別の試行としてやり直します。⚠ この別試行には §6 の起動と §8 の停止が含まれます。状態を変えるので、それぞれの節に書いた了承の条件をそのまま満たしてから進みます。前の試行の結果を根拠に、次の 3 つを省きません。

1. §6 の `serve check` を流し直し、`gate.N.name` が 9 種そろい、head・worker とも `gate.N.passed` がすべて `true`、`gates_failed=0`、`status=passed` であることを確認する。
2. §6 の `serve start` を、同じ構成・同じ重み・同じイメージで実行する。進む条件は `status=ready` だけです。`already_running`、`refused`、`failed` の扱いは §6 のとおりです。
3. §7 の手順で、今回のコンテナに帰属する NCCL 記録だけを選び直し、少なくとも 1 行の `Using network IB` があること、今回分にある経路確定行がすべて IB であることを確認する。前の試行のファイルを今回分として使いません。

この 3 つを満たした後だけ、上限を上げて短い要求を送り直します。

```bash
uv run --directory serving serve status --configs var/nope-build-0961bbae/tp2.toml
uv run --directory serving serve smoke p2-nope-tp2-smoke \
  --configs var/nope-build-0961bbae/tp2.toml --max-tokens 512
```

再送でも、`status` が `answered` でない、HTTP 200 でない、`stop_reason` が上限到達 (`max_tokens`)、応答の本文が欠落している、意味を判定できない、のいずれかに当たれば合格にしません。「上限を上げたから合格」とは扱いません。

記録するのは、変更した上限値 (512)、`status`、`reply.*.http_status`、`stop_reason`、`*_tokens`、意味の通る文かの判定だけです。要求・応答の本文と思考の内容は記録せず、stderr をファイルへ保存しません。

再送の可否にかかわらず、§8 に従ってログを回収してから停止し、サーバーを常駐させません。重み、構成 TOML、イメージ、サーバーの設定は変えません。変えるのは `--max-tokens` に渡す値だけです。

## 8. 推論用コンテナのログを回収して停止する

§7 の応答確認が終わった後（または §6・§7 で記録して停止すると判断した後）、推論用構成 `p2-nope-tp2-smoke` のコンテナを対象に、ログを回収してから停止します（§5 の取得中止の回収は取得用構成 `p1-fetch-nvfp4` が対象で、コンテナが別のためここでは流用しません）。

```bash
uv run --directory serving serve logs p2-nope-tp2-smoke --configs var/nope-build-0961bbae/tp2.toml
uv run --directory serving serve status --configs var/nope-build-0961bbae/tp2.toml
```

停止前に、自分のラベル `vllm-baseline.owner=serving-kit` のコンテナだけが対象であることを照合します。⚠ この試行でサーバーを常駐させないことを確認してから停止します。

```bash
uv run --directory serving serve stop --yes
uv run --directory serving serve status --configs var/nope-build-0961bbae/tp2.toml
```

停止後に GPU プロセスが 0 件であることを確認します。失敗・不一致・中断時は状態と長さだけを記録し、`serve logs` の後に停止します。停止後も、要求・応答の本文や認証情報は記録しません。

## 9. 次の段階

長文脈、性能、長時間運転、`/v1/messages` 互換はこの試行の対象外です。今回の ready、IB、短い要求の HTTP 200 が揃った後に、別の計画として扱います。
