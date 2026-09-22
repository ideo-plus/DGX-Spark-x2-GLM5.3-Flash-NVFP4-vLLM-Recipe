# 試した構成の記録 (attempts.md)

P1 (Baseline Procedure) で試した構成と、その結果を、1 行 1 試行で積み上げる記録である
(要件 8.2)。書く手順は [`procedure.md`](procedure.md) の各段の「記録」の節にある。

## 書かないこと (要件 10.5)

1. **送った内容と、応答の本文**。書くのは、長さ、トークンの数、終わりの理由、HTTP の状態だけ
2. **認証の情報** (鍵、トークン)。Spark にも置かない (要件 2.6)
3. **計測者が別に起動していた構成 (`exl3-tp2`) の中身** (起動の引数、設定、差し込まれた
   ファイル、記録)。読んでよいのは、GPU を使っているプロセスの名前とメモリの量だけである

## 試す前にこの表を見る (要件 8.3)

**どの段でも、1 つめのコマンドを打つ前に、この表を読み、同じ構成で同じ失敗を、設定を
変えずに繰り返さない。** 同じ構成をもう一度試すなら、何を変えたのかを、その行の「次に進む
理由」に書ける状態にしておく。構成の値を書き換えたとき (段 1、段 3、段 4、`ready_timeout_s`
の直し、NCCL の記録の変数の足し外しなど) は、その行に、**その構成で何を変えたか**を書く。

## 1 行の書式 (要件 8.2)

1 行は、次の項目をすべて持つ (design.md 「Baseline Procedure (文書)」)。

| 項目 | 書くこと |
|---|---|
| 日時 | 試した日時 (UTC、`YYYY-MM-DDTHH:MM:SSZ`) |
| 段 | 段 0〜段 4、関門 A、関門 B、準備 1〜4 のいずれか (`procedure.md` の見出しに合わせる) |
| 構成の名前 | `configs.toml` の構成の名前 |
| 結果 | 起動できたか、計測 (`bench`) を流し切れたか |
| 止まった場所 | 起動できなかった、または計測を流し切れなかった場合の、止まった場所 |
| 次に進む理由 | 次にどの段・関門に進むか (または打ち切るか) と、その理由 |
| 記録の置き場所 | `serving/var/…` など、回収した記録の道筋 (`docs/` にコミットしない) |

## 表 (試すたびに 1 行足す)

| 日時 | 段 | 構成の名前 | 結果 | 止まった場所 | 次に進む理由 | 記録の置き場所 |
|---|---|---|---|---|---|---|
| 2026-09-22T10:02:15Z | 段 0 | `probe-pinned` | 起動できなかった (計測は流していない) | 重みの読み込みのあと、CUDA graph の取得の途中で、`concat_and_cache_mla` の `pe_dim must be 64 for fp8_ds_mla` の assert (コンテナ終了コード 1)。選ばれたアテンションは `FLASHINFER_MLA_SPARSE_SM120` だけ | 段 1 へ。今日の `nightly` (commit `0961bbae…`) にも同じ assert が残り、issue #57578 / #55773 は open だが、候補のバックエンドが変わっていないかを 1 回だけ確かめ、要件 8.7 の打ち切りの判断の材料にする | `serving/var/20260922T100215Z-start-probe-pinned/` (not-working.md の件 1) |
| 2026-09-22T10:23:11Z | 段 1 | `probe-nightly` | 起動できなかった (計測は流していない) | 段 0 と**同じ場所** (`concat_and_cache_mla` の `pe_dim must be 64 for fp8_ds_mla`。vLLM `0.29.1rc1.dev513+g0961bbae2`)。候補のアテンションも `FLASHINFER_MLA_SPARSE_SM120` の 1 つだけで変わらない | 要件 8.7 に当たる (2 つの公式のイメージで同じ場所。ダミーの重みで落ちるので重みの形式に依らず、1 台なので並列の取り方にも依らない)。重みの取得に進まず、打ち切りの手順 (8.5) で計測者に尋ねる | `serving/var/20260922T102311Z-start-probe-nightly/` (not-working.md の件 1) |
| 2026-09-22T12:32:52Z | 関門 A | `netcheck-bandwidth` | 計測は流れたが不合格 (経路が Socket) | NCCL が `NET/IB : No device found.` で TCP に落ち、1 GiB の busbw 16 Gbps | `--device /dev/infiniband` を 1 つだけ足した B (`netcheck-bandwidth-ib`) で A/B | `serving/var/20260922T123252Z-netcheck-netcheck-bandwidth/`、`docs/results/2026-09-22-netcheck-bandwidth.md` |
| 2026-09-22T12:35:35Z | 関門 A | `netcheck-bandwidth-ib` | 合格 (2 台とも IB。1 GiB の busbw 186.9 Gbps) | — | `--device /dev/infiniband` を採用 (ADR 0003)。事前の確認へ | `serving/var/20260922T123535Z-netcheck-netcheck-bandwidth-ib/`、同上 |
| 2026-09-22T12:37:20Z | 関門 A | `netcheck-sanity` | 合格 (4 段すべて、2 台とも IB) | — | 通信の確認は済み。段 2 は要件 8.7 の打ち切りにより行わない (ADR 0005) | `serving/var/20260922T123720Z-netcheck-netcheck-sanity/` |
