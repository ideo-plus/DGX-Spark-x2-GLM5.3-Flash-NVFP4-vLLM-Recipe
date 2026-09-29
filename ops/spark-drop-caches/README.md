# spark-drop-caches

両台の Spark で、ページキャッシュを捨てる固定の 1 コマンド (issue #84)。

## 何をするか

`sync` のあとに `/proc/sys/vm/drop_caches` へ `3` を書き、ページキャッシュ・dentry・inode の
キャッシュを捨てる。これで、その台の `MemFree` が増える。

## なぜこの対策か

GB10 は CPU と GPU がメモリを共有する。変換 (約 190 GB の書き込み)・`serve verify`
(`sha256sum`)・別構成の mmap 起動の直後は、ページキャッシュが埋まっていて、CUDA から見える
空きが小さくなり、`--load-format instanttensor` の起動が
`buffer_size ... exceeds device memory budget` で落ちる。関門 `memory_free` が、起動の前に
`/proc/meminfo` の `MemFree` を読んでこの状態を検出し、下限に足りなければ起動せずに断る
(`serve` の許可一覧に `sudo` は足していない。断ったあとの対処が、この一式である)。

検討した対策は 3 つある。

- **(a) 起動の前にページキャッシュを捨てる** (採用): 原因 (ページキャッシュ) を直接消す唯一の
  手段。`ops/spark-power-caps/` の既存の形 (root 所有の POSIX sh + sudoers.d の固定形) に
  合わせ、`serve` からは呼ばずに、計測者が ⚠ 了承したうえで手打ちで流す。
- **(b) instanttensor の予算・バッファの大きさを vLLM の設定で渡す** (不採用): 固定した vLLM
  (`0961bbae`) の `instanttensor_weights_iterator` は `model_loader_extra_config` も
  `buffer_size`/`max_free_mem_usage` も渡さない。渡す口がない。上流の InstantTensor の
  環境変数 (`INSTANTTENSOR_MAX_FREE_MEM_USAGE` など) はあるが、イメージ内の版との対応が
  未確認で、予算の元 (`torch.cuda.mem_get_info()`) が変わらないので原因は消えない。
- **(c) `serve start` の前の関門で断る** (採用): 関門 `memory_free` として実装した。読み取り
  だけで、`serve` の安全の決まりを緩めない。この README は、(c) が断ったときの対処である。

## 前提

- 両台に `/proc/sys/vm/drop_caches` がある (既知の事実。Linux の一般的な入り口)。
- `serve` の遠隔の許可一覧に `sudo` は足していない (`test_safety.py` の契約)。この一式は
  `serve` の外から、対話側が実行する。

## 初回の設置 (状態を変える。⚠ 計測者が sudo で行う。了承を得てから実行する)

このリポジトリのコード・構成の準備までがこの作業の範囲であり、実機への設置は対話側が行う。
両台 (head: `spark-153d`、worker: `spark-5083`) とも、この節は 1 回だけでよい。

1. Mac のリポジトリ直下で、sudoers の断片を置く前に構文を検査する (読み取りだけ)。
   ```sh
   visudo -cf ops/spark-drop-caches/spark-drop-caches.sudoers
   ```
2. Mac から、2 つのファイルを両台の `/home/j5ik2o/spark-drop-caches/` へ rsync で移す
   (⚠ 状態を変える。了承を得てから実行する)。
   ```sh
   rsync -e 'ssh -o BatchMode=yes -o ConnectTimeout=5' ops/spark-drop-caches/spark-drop-caches ops/spark-drop-caches/spark-drop-caches.sudoers spark-153d:/home/j5ik2o/spark-drop-caches/
   rsync -e 'ssh -o BatchMode=yes -o ConnectTimeout=5' ops/spark-drop-caches/spark-drop-caches ops/spark-drop-caches/spark-drop-caches.sudoers spark-5083:/home/j5ik2o/spark-drop-caches/
   ```
3. Mac から ssh で、両台で、移した先の sudoers の断片の構文を検査する (⚠ 状態を変える。
   了承を得てから実行する。sudo がパスワードを求めるので `-t` で端末を割り当てる)。検査が
   通らない台では、sudoers の断片を `/etc/sudoers.d/` に置く段 (段 5) に進まない。
   ```sh
   ssh -t spark-153d 'sudo visudo -cf /home/j5ik2o/spark-drop-caches/spark-drop-caches.sudoers'
   ssh -t spark-5083 'sudo visudo -cf /home/j5ik2o/spark-drop-caches/spark-drop-caches.sudoers'
   ```
4. Mac から ssh で、移した先の道筋を写し元にして、両台にスクリプトを root 所有・実行可能で
   置く (⚠ 状態を変える。了承を得てから実行する。sudo がパスワードを求めるので `-t` で
   端末を割り当てる)。
   ```sh
   ssh -t spark-153d 'sudo install -m 0755 -o root -g root /home/j5ik2o/spark-drop-caches/spark-drop-caches /usr/local/sbin/spark-drop-caches'
   ssh -t spark-5083 'sudo install -m 0755 -o root -g root /home/j5ik2o/spark-drop-caches/spark-drop-caches /usr/local/sbin/spark-drop-caches'
   ```
5. Mac から ssh で、両台に sudoers の断片を `0440` で置く (⚠ 状態を変える。了承を得てから
   実行する)。
   ```sh
   ssh -t spark-153d 'sudo install -m 0440 -o root -g root /home/j5ik2o/spark-drop-caches/spark-drop-caches.sudoers /etc/sudoers.d/spark-drop-caches'
   ssh -t spark-5083 'sudo install -m 0440 -o root -g root /home/j5ik2o/spark-drop-caches/spark-drop-caches.sudoers /etc/sudoers.d/spark-drop-caches'
   ```

## 使い方 (状態を変える。⚠ 了承を得てから実行する)

Mac から、両台に、ssh で固定のコマンドを流す。

```sh
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-153d 'sudo /usr/local/sbin/spark-drop-caches'
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-5083 'sudo /usr/local/sbin/spark-drop-caches'
```

いつ流すか: 変換・`serve verify`・mmap (`--load-format auto`) 起動の直後、または
`serve check`/`serve start` の関門 `memory_free` が断ったとき。

## 確認 (読み取りだけ)

- 前後で `/proc/meminfo` の 3 つの値を比べる。
  ```sh
  ssh -o BatchMode=yes -o ConnectTimeout=5 spark-153d "grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo"
  ```
- `uv run --directory serving serve check <構成>` の `memory_free` が `passed=true` になる
  ことを確かめる。

## 原因の確かめ方

- `serve check` の `memory_free` の `detail` に、その時点の `MemFree`・`MemAvailable`・
  `Cached` が残る。
- `serve start` が `ready` になった回は、`serve logs` で回収した `<構成>.launch.json` の
  `gates` に、起動の直前の `memory_free` の値が残る。
- 任意で、コンテナの中の `torch.cuda.mem_get_info()` を読む (⚠ 状態を変える。イメージを
  1 回起こすので、了承を得てから実行する)。
  ```sh
  ssh -o BatchMode=yes -o ConnectTimeout=5 spark-153d 'docker run --rm --gpus all --entrypoint python3 <イメージ ID> -c "import torch; print(torch.cuda.mem_get_info())"'
  ```

## 撤去 (状態を変える。⚠ 了承を得てから実行する)

```sh
sudo rm /etc/sudoers.d/spark-drop-caches /usr/local/sbin/spark-drop-caches
```
この手順はこのリポジトリのコマンドでは自動化しない (この作業の範囲外)。

## 実機確認記録 (2026-09-29)

head (`spark-153d`) と worker (`spark-5083`) の両台で、次を確認した。

- `/usr/local/sbin/spark-drop-caches` を root 所有・`0755` で設置した。
- `/etc/sudoers.d/spark-drop-caches` を root 所有・`0440` で設置し、`visudo -cf /etc/sudoers` が通った。
- sudoers の引数なし指定 (`""`) が実機の `sudo -n -l` に表示され、`sudo -n /usr/local/sbin/spark-drop-caches` がパスワードなしで完了した。
- drop-caches 後に `serve check glm53-tp2-mtp3-marlin` を実行し、18 件の関門がすべて通った。`MemFree` は head 117.5 GiB、worker 111.1 GiB だった。
- `serve start glm53-tp2-mtp3-marlin --yes` は、はじめの 3 回が worker の vLLM 初期化 (`init_device`) で終了した。
  `gpu_memory_utilization=0.90` の必要量は `109.52 GiB` である。いずれの回も、worker では別の重みの取得
  (`hf download`、約 200 GB) と、その NAS への同期が動いていた、または止めたまま残っていた。

  | 回 | worker の状態 | 初期化時の CUDA 空き |
  |---|---|---:|
  | 1 | 取得が動いたまま。drop-caches → `serve check` → 起動 (GPU 上限 2000 MHz) | 106.16 GiB |
  | 2 | 同上 (GPU 上限 2200 MHz) | 106.57 GiB |
  | 3 | 取得を `SIGSTOP` で止め、drop-caches の直後に起動 | 108.75 GiB |
  | 4 | 取得のプロセスを終了させ、drop-caches の直後に起動 | 通過 (`status=ready`) |

  3 回目は、止めたプロセスが匿名メモリ約 2.3 GiB (RSS) を持ったままだった。止めただけではメモリは空かない。
  4 回目の直前の worker は `MemFree` 117.8 GiB、`AnonPages` 0.16 GB だった。
- 起動の余裕は小さい (3 回目の不足は 0.77 GiB)。drop-caches は起動の直前に流す。そのとき、両台で大きな読み書きや
  取得をするプロセスが動いていない (止めたまま残してもいない) ことを確かめる。

## 実機で未確認のこと

- drop-caches 後に `MemFree` が増える量を、実行前後の同一条件で定量比較すること。
- drop-caches 後に `serve start` の CUDA 初期化条件を安定して通せるかどうか。
- イメージ内の instanttensor の版と、上流の環境変数 (`INSTANTTENSOR_MAX_FREE_MEM_USAGE` など)
  の対応。
