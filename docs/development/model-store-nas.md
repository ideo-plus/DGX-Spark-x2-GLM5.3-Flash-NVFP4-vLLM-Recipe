# モデルの重みを NAS (vega-nas) に置く

Spark のディスクは 8 割前後まで埋まっている (2026-09-30、head 86%、worker 77%)。今の構成で使わない重みは、
NAS に写してから Spark から消す。消すのは、NAS の写しを照合したあとだけにする。
もう一度使うときは、NAS から戻す。実機の状態を変える操作 (写す、消す、戻す) は、対話側が計測者の了承を取ってから行う。

## 置き場所

| 項目 | 値 |
|---|---|
| NAS | vega-nas (Synology)。LAN の `10.0.1.30`、SSH のポート `1022`、ユーザー `j5ik2o` |
| 置き場所 | `/volume1/docker/models/<レシピ>-<モデル>` (共有フォルダー `docker` の下) |
| 空き | 21 TB のうち 19 TB (2026-09-30) |

同じ `docker` の下の `registry` は、Docker のレジストリ (`registry.j5ik2o-vega.synology.me:5001`) の置き場所である。

## 置いてあるもの

| NAS の上の名前 | 元 | ファイル | 大きさ | 置いた日 |
|---|---|--:|--:|---|
| `miaai-glm-5.3-flash-exl3-4.05bpw` | MiaAI の比較用の EXL3 の重み | 96 | 154 GiB | 2026-09-19 |
| `mmastrac-glm-5.3-flash-nvfp4` | `nvidia/GLM-5.3-Flash-NVFP4` の `09b04e5e74bca08ca8549fc736d4cdd8624bfde3` (mmastrac の比較用) | 44 | 191 GiB | 2026-09-29 |
| `ideo-plus-glm-5.3-flash-k2s1` | このリポジトリの K2 の途中の段の派生の重み `k2s1` | 20 | 183 GiB | 2026-09-30 |
| `ideo-plus-glm-5.3-flash-k2s1b` | 同 `k2s1b` | 20 | 183 GiB | 2026-09-30 |
| `ideo-plus-glm-5.3-flash-k2s2a` | 同 `k2s2a` | 20 | 181 GiB | 2026-10-01 |
| `ideo-plus-glm-5.3-flash-k2s2b` | 同 `k2s2b` | 20 | 177 GiB | 2026-10-01 |
| `ideo-plus-glm-5.3-flash-k2s3` | 同 `k2s3` | 20 | 174 GiB | 2026-09-30 |
| `ideo-plus-glm-5.3-flash-k2s4` | 今の構成 (`glm53-tp2-mtp3-marlin`) の重み。Spark にも残す | 20 | 171 GiB | 2026-09-30 |
| `redhatai-glm-5.3-flash-nvfp4` | `RedHatAI/GLM-5.3-Flash-NVFP4` の `18d55bfd5a2194887738da73753975c9d3842f46` (k2s4 の元)。Spark にも残す | 62 | 185 GiB | 2026-10-01 |
| `miaai-glm-5.3-flash-exl3-tr3-4bpw` | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` の `25a44fdbf16862a46b7cc9921142c6c81350af2f` (MiaAI の比較用) | 144 | 164 GiB | 2026-10-01 |

`nvidia-glm-5.3-flash-nvfp4` は空のディレクトリである。
2026-09-30〜10-01 に写したものは、送った rsync が正常に終わったことまでを確かめた。チェックサムでの照合 (下の「照合する」) はまだで、Spark から消すのは照合の後にする。
`redhatai-glm-5.3-flash-nvfp4` は、Hugging Face の取得記録 `.cache/huggingface/trees/…json` が読めず (権限)、その 1 ファイルだけ写していない (rsync の終わりの値は 23)。重みのファイルは全部ある。
派生の重みのファイルの一覧と SHA-256 は、`serving/weights/*.manifest.json` にある (`k2s1` の分は無い)。

## 入り方

- **Mac から**: `~/.ssh/config` の `Host vega-nas` で入れる。
- **Spark から**: 次のように入る。
  - 鍵は、Mac と共有の `~/.ssh/id_ed25519.tailnet` を指定する。NAS に登録済みで、既定の鍵では断られる。
  - コマンド:

    ```sh
    ssh -i ~/.ssh/id_ed25519.tailnet -p 1022 j5ik2o@10.0.1.30
    ```

## Spark から NAS に写す

```bash
E="ssh -o BatchMode=yes -c aes128-gcm@openssh.com -i $HOME/.ssh/id_ed25519.tailnet -p 1022"
rsync -aL --partial --info=stats1 --rsync-path=/usr/bin/rsync -e "$E" \
  "$HOME/vllm-baseline/models/k2s3/" "j5ik2o@10.0.1.30:/volume1/docker/models/ideo-plus-glm-5.3-flash-k2s3/"
```

- **`--rsync-path=/usr/bin/rsync` は必ず付ける**: 付けないと、Synology の rsync の入口が DSM の認証を求め、`Permission denied` になる。
- **`-L` を付ける**: Hugging Face のキャッシュ (`hub/models--…/snapshots/<rev>/`) はリンクの集まりなので、`-L` で実体を写す。
- **速さ**: 大きな写しは、一晩かかる前提で組む。
  - **送り始め**: 送り始めの短い間は、NAS の `sshd` の暗号化で 1 本あたり約 200 MB/s が上限になる。2 台から同時に送ると、合わせて約 420 MB/s 出る。
  - **長く送ったときの平均** (2026-09-30〜10-01): 2 本同時で、1 本あたり約 23〜29 MB/s に落ちた (最初の約 40 分だけは 70〜82 MB/s)。ほぼ 1 本だけで送った `k2s2b` は、約 69 MB/s だった。1 本ずつのほうが速い可能性があるが、条件をそろえて比べてはいない。
  - **遅い理由**: NAS の書き込みで詰まる。回線 (`eth2` の 10GbE) は余っている。
    - HDD 4 台 (Seagate ST8000DM004、SMR) の RAID5 で、長く書き続けると、RAID5 の使用率 100% で約 23 MB/s しか書けなかった。
    - NVMe の SSD キャッシュ (RAID1、write-back) は、1 MB を超える連続の書き込みを素通りさせる (`skip_seq_thresh_kb = 1024`)。そのため、写しの書き込みの約 98% が HDD に直接行っていた。
    - DSM 7.4 の画面には、この値を変える項目が無い。変えるなら、root の権限で `sysctl -w dev.flashcache_shared_cache_vg1_alloc_cache_1+volume_1.skip_seq_thresh_kb=0` を実行する (DSM のタスクスケジューラの root のスクリプトなど)。
  - **どちらから送るか**: 2 台の重みは同じ記録 (manifest) と照合済みで中身が同じなので、どちらから送ってもよい。
- **止めて再開するとき**: `--partial` で送るので、同じコマンドで送り直せば続きから送る。ただし途中のファイルは、NAS の側で読み直して差分を取るので、そのぶん時間がかかる (2026-10-01、`k2s2a` の残り約 60 GB に約 2.5 時間)。
- **長く流すとき**: SSH が切れても止まらないよう、Spark の上で `nohup bash <スクリプト> &` で動かす。
- **シェル**: Spark の対話のシェルは zsh で、変数に入れたコマンドが空白で分かれない。上の例のように `$E` を使うときは、bash で動かす。
- **止め方**: 送っている途中で止めるときは、Spark の `rsync` と `ssh` だけでなく、NAS の上の `rsync --server` も残っていないかを確かめる。残っていると、次の送りと同じ置き場所に書いてしまう。

## 照合する

写した後、チェックサムで差分が無いことを確かめる (両側で全体を読むので、写すのと同じくらい時間がかかる)。

```bash
rsync -aL -c -n -i --rsync-path=/usr/bin/rsync -e "$E" <元>/ j5ik2o@10.0.1.30:/volume1/docker/models/<名前>/
```

何も表示されなければ、中身が同じである。

## NAS から Spark に戻す

```bash
rsync -a --partial --info=stats1 --rsync-path=/usr/bin/rsync -e "$E" \
  "j5ik2o@10.0.1.30:/volume1/docker/models/ideo-plus-glm-5.3-flash-k2s3/" "$HOME/vllm-baseline/models/k2s3/"
```

- **このリポジトリの派生の重みを戻したとき**: 使う前に `serve verify` で manifest と照合する。
- **起動の前にページキャッシュを捨てる**: 大量に読み書きした直後は、`--load-format instanttensor` の起動が落ちやすい。
  - 両台で `find /home/j5ik2o/vllm-baseline/models -type f -exec dd if={} iflag=nocache count=0 status=none \;` を流す。
  - 関門の `memory_free` が通ることを確かめてから起動する。
