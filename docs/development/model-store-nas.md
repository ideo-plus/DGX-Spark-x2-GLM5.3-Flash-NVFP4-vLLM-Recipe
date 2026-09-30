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
| `miaai-glm-5.3-flash-exl3-4.05bpw` | MiaAI の比較用の EXL3 の重み | 96 | 154 GB | 2026-09-19 |
| `mmastrac-glm-5.3-flash-nvfp4` | `nvidia/GLM-5.3-Flash-NVFP4` の `09b04e5e74bca08ca8549fc736d4cdd8624bfde3` (mmastrac の比較用) | 44 | 191 GB | 2026-09-29 |
| `ideo-plus-glm-5.3-flash-k2s1` 〜 `-k2s3` | このリポジトリの K2 の途中の段の派生の重み (`k2s1`、`k2s1b`、`k2s2a`、`k2s2b`、`k2s3`) | 写している途中 | — | 2026-09-30 |
| `ideo-plus-glm-5.3-flash-k2s4` | 今の構成 (`glm53-tp2-mtp3-marlin`) の重み。Spark にも残す | 写している途中 | — | 2026-09-30 |
| `redhatai-glm-5.3-flash-nvfp4` | `RedHatAI/GLM-5.3-Flash-NVFP4` の `18d55bfd5a2194887738da73753975c9d3842f46` (k2s4 の元)。Spark にも残す | 写している途中 | — | 2026-09-30 |
| `miaai-glm-5.3-flash-exl3-tr3-4bpw` | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` の `25a44fdbf16862a46b7cc9921142c6c81350af2f` (MiaAI の比較用) | 写している途中 | — | 2026-09-30 |

`nvidia-glm-5.3-flash-nvfp4` は空のディレクトリである。
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
- **速さ**:
  - NAS の `sshd` の暗号化で、1 本あたり約 200 MB/s で頭打ちになる (2026-09-30)。
  - head と worker から別々のものを同時に送ると、合わせて約 420 MB/s になる。
  - 2 台の重みは同じ記録 (manifest) と照合済みで中身が同じなので、どちらから送ってもよい。
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
