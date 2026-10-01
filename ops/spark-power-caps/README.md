# spark-power-caps

再起動すると既定に戻る、2 つの上限を、両台の起動のたびに入れ直す systemd の oneshot サービス (issue #10)。

## 何を保つか

- GPU クロックの上限 `nvidia-smi -lgc 300,1800` — 根拠: `docs/results/2026-09-23-decode-gpu-clock-cap.md`
- X925 の周波数の上限 `cpupower -c 5-9,15-19 frequency-set -u 3000MHz` — 根拠: `docs/results/2026-09-23-thermal-source.md`（#11 段階 2）

計測ごとに上限が効いていたことを確かめる仕組みは `serve watch` 側にある（`result.json` の `gpu_sm_clock_range_mhz` と `cpu_cluster_max_freq_khz`。`serving/README.md` の watch の節を参照）。

## 前提

- 両台に `/usr/bin/nvidia-smi` と `/usr/bin/cpupower` がある（既知の事実）。
- 両台の既存の `/etc/sudoers.d/nvidia-clock`・`/etc/sudoers.d/cpu-freq-cap`（`docs/results/` に記載）には触れない。この一式は別名 `/etc/sudoers.d/spark-power-caps` で追加する。
- 読み取りだけの確認: `ssh <host> 'command -v install systemctl'` で、両コマンドが両台にあることを確かめられる。

## 初回の設置（状態を変える。⚠ 計測者が sudo で行う）

このリポジトリのコード・構成の準備までがこの作業の範囲であり、実機への設置と実機検証は対話側が行う（issue #10 の前提）。設置は、計測者が両台に sudo で行う。

1. sudoers の断片を置く前に、構文を検査する（読み取りだけ）。
   ```sh
   visudo -cf ops/spark-power-caps/spark-power-caps.sudoers
   ```
2. `serve push` で、default の写し元を配る（`serving/payload/spark-power-caps.default` が両台の `<remote_root>/payload/` に届く）。
3. 4 つのファイルを、ssh で両台に置く（状態を変える。sudo が要る）。
   - `ops/spark-power-caps/spark-power-caps.service` → `/etc/systemd/system/spark-power-caps.service`
   - `ops/spark-power-caps/spark-power-caps-apply` → `/usr/local/sbin/spark-power-caps-apply`（root 所有、実行可能）
   - `ops/spark-power-caps/spark-power-caps.sudoers` → `/etc/sudoers.d/spark-power-caps`（visudo -cf を通してから）
   - `serving/payload/spark-power-caps.default` の写し元は、この時点では手で置いてよい（`<remote_root>/payload/` から `/etc/default/spark-power-caps` へ、値の変更の手順と同じ形の `install`）
4. `systemctl daemon-reload` → `systemctl enable --now spark-power-caps.service`。
5. `systemctl status spark-power-caps.service` で、`active (exited)` になっていることを確かめる（読み取りだけ）。

## 確認（読み取りだけ）

- `systemctl is-enabled spark-power-caps.service`
- `systemctl is-active spark-power-caps.service`
- `systemctl status spark-power-caps.service`
- `serve watch` の記録 (`result.json`) の `gpu_sm_clock_range_mhz`（最大が 1800 以下）と `cpu_cluster_max_freq_khz.x925`（3000000 以下）で、計測中に上限が効いていたことを確かめる。

## 値の変更（初回設置後。状態を変える手順を含む）

1. `serving/payload/spark-power-caps.default` を編集し、`serve push --yes` で写し元を両台に配る（読み取りではなく状態を変える。`serve push` 自体が了承つきの操作）。
2. sudoers で許した固定形だけを、ssh で両台に実行する（⚠ 状態を変える）。
   ```sh
   sudo install -m 0600 -o root -g root <remote_root>/payload/spark-power-caps.default /etc/default/spark-power-caps
   sudo systemctl restart spark-power-caps.service
   ```
   `systemctl restart` は、`ExecStop`（reset）を実行してから `ExecStart`（apply）を行う。値が不正だと、reset で GPU と X925 の両方の上限がいったん既定に戻ったあと、apply の検査で `ExecStart` が失敗し、`spark-power-caps.service` は**上限が既定に戻ったまま** `failed` になる（apply の途中で失敗し中途半端な上限のまま止まることはない）。
3. `systemctl status spark-power-caps.service` と `serve watch` の記録で確かめる（読み取りだけ）。写した値が不正でも、この 2 つのコマンド自体は通ってしまう。値の書式・範囲の妥当性は `spark-power-caps-apply` の検査が唯一の防御で、不正な値は `ExecStart` が失敗し、`systemctl status` が `failed` になる（`journalctl -u spark-power-caps.service` に理由が残る）。この関係により、sudoers が値の中身までは検査しなくても、反映される前に必ず食い止められる。ただし、この防御は値の書式・範囲の検査に限られる。`EnvironmentFile` を使って実行ファイルや sysfs の読み取り先そのものを差し替えることや、動的ローダーに効く環境変数は、値の検査では止められない（前者は `spark-power-caps-apply` が引数だけを差し替え口にすることで、後者は `spark-power-caps.service` の `UnsetEnvironment=` で別途防ぐ。「未解決の前提」の節を参照）。
4. `failed` になったら、正しい値に直すまで計測しない。`docs/vllm-baseline/full-context-procedure.md` §2 の確認（`systemctl is-active`）がここで `active` にならないので、その場で気づける。直すには、1〜3 を正しい値でやり直す（`spark-power-caps.default` を修正して配り直し、`install` と `restart` を実行し、`systemctl status` が `active (exited)` に戻ったことを確かめる）。

## 無効化（状態を変える。⚠ 了承）

```sh
sudo systemctl disable --now spark-power-caps.service
```
`disable --now` は `ExecStop`（`spark-power-caps-apply reset`）を実行してから止まるので、GPU とX925 の上限は既定に戻る。

## 実機確認記録 (2026-09-29)

GPU クロックの上限を上げると速くなるかを確かめるため、両台の上限を一時的に手で変えた
（`sudo -n /usr/bin/nvidia-smi -lgc 300,2000`、続けて `300,2200`。下限は 300 MHz のまま）。
`/etc/default/spark-power-caps` は変えていないので、サービスの既定は `300,1800` のままである。

- 2000 MHz と 2200 MHz での起動は、worker のメモリ不足で落ちた。原因はクロックではなく、worker で動いていた
  別の重みの取得だった（`ops/spark-drop-caches/README.md` の実機確認記録）。
- 取得を止めたあと、2200 MHz で起動し、decode・同時処理・入力の処理を測った。続けて、X925 の上限も
  `sudo -n /usr/bin/cpupower -c 5-9,15-19 frequency-set -u 3900MHz` で外して同じ計測をした。
  実クロックは上限まで上がったが（GPU 2171〜2190 MHz、X925 3.9 GHz）、どの値も標準の上限のときと変わらなかった
  （[記録](../../docs/results/2026-09-29-gpu-clock-2200.md)）。
- このため、上限の既定は GPU 1800 MHz・X925 3.0 GHz のままにする。2000 MHz は測っていない。
- 計測のあと、両台で `nvidia-smi -lgc 300,1800` と `cpupower ... -u 3000MHz` を流して、上限を標準に戻した
  （17:40 JST）。

## 撤去（状態を変える。⚠ 了承）

上の無効化のあと、4 つのファイル（unit、適用スクリプト、sudoers の断片、`/etc/default/spark-power-caps`）を手で削除し、`systemctl daemon-reload` する。この手順はこのリポジトリのコマンドでは自動化しない（この作業の範囲外）。

## 未解決の前提

この一式は `EnvironmentFile` と、NOPASSWD の固定形 `install` という要求どおりの方式で作られているが、この方式自体が持つ限界が 2 つ残る。

1. 動的ローダーに効く環境変数は、`spark-power-caps.service` の `UnsetEnvironment=` で代表的な 5 つ（`LD_PRELOAD`・`LD_LIBRARY_PATH`・`LD_AUDIT`・`GLIBC_TUNABLES`・`PATH`）を外しているが、一次情報（glibc の `unsecvars` の一覧）を確かめられていないため、この列挙に漏れが残りうる。
2. sudoers の `install` は、写し先を root だけが読めるモード（`0600`）にしているが、写し元（利用者が書ける置き場所）がシンボリックリンクであっても root が写してしまう関係までは閉じていない。

どちらも、`EnvironmentFile` と NOPASSWD の `install` という方式のままでは完全には閉じられない。閉じきるには、次のいずれかのように要求の方式そのものを変える必要があり、利用者の判断が要る。

- root 所有の写し役（デーモンやラッパースクリプト）に置き換える
- `install` をパスワードつきにする（NOPASSWD をやめる）

## 実機で未確認のこと

- `nvidia-persistenced.service` が両台にあるか、`After=` の順序指定で GPU ドライバの準備が起動時に間に合うか。
- `nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits` が GB10 で整数を 1 行返すか。
- `/usr/bin/install`・`/usr/bin/systemctl` の実際の道筋（sudoers の固定形と一致するか）。
- `cpupower -c <集合> frequency-set -u <値>kHz`（kHz 単位）が実機で受理されるか（`reset` が使う形）。
- `cpuinfo_min_freq` の実測値と、実機のコア数（20 コアである前提）。
- `UnsetEnvironment=` が `EnvironmentFile=` より後に効くこと（systemd の仕様どおりか、この環境の systemd バージョンで確かめていない）。
- `install` が写し元のシンボリックリンクをたどること。
- `journalctl -u spark-power-caps.service` に、写し元の中身が出るかどうか。

これらは実機への設置時に対話側が確かめる。

## 上限を外した記録 (2026-10-02)

GLM-5.3-Flash のレシピの計測を終え、別のプロジェクトで両台を使うため、計測者の指示で上限を外した。

- **起動のたびに入れ直す値**: `/etc/default/spark-power-caps` を、ハードウェアの最大にした (GPU `300,3003`、X925 `3900`)。
  - sudoers で許した固定形 (`install` と `systemctl restart`) で入れた。
  - サービスは `active` のままである。再起動しても、上限は実質かからない。
- **今の上限**: `sudo -n nvidia-smi -rgc` で、GPU のクロックの固定を外した。X925 の上限は 3.9 GHz になっている。
- **残っているもの**: サービスの登録、適用スクリプト、sudoers の断片は、両台に残っている。
  - 撤去 (上の「撤去」の手順) には sudo のパスワードが要るので、行っていない。
- **このレシピをまた測るとき**:
  1. `serving/payload/spark-power-caps.default` (`300,1800` と `3000`) を `serve push --yes` で配り直す。
  2. 「値の変更」の手順で、`install` と `restart` を流す。
