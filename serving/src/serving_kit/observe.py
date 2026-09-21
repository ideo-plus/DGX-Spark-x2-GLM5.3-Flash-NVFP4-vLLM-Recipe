r"""起動の記録と NCCL の記録から、事実を抜き出す (design.md 「組み立てと読み取り › observe」)。

**入出力のない、純粋な関数だけを置く**。ファイルも、時刻も、ネットワークも読まない。
依存の向きにより、読み込むのは `serving_kit.types` だけである (`plan` も `guards` も
読み込まない)。同じ文字列からは、いつでも同じ結果ができる。

守る決まり:

- **読めなかった項目は、断らずに `None` (または `()`、`False`) のまま返す**。空の文字列、
  関係のない行だけの記録は、すべて空で返る (design.md 「observe」、requirements 4.4、
  5.2、5.3、6.3)
- **探す文字列は、この module の先頭に、名前つきの定数でまとめる**。それぞれに、出どころ
  (research.md のどの節か、上流のどのファイルか) を 1 行のコメントで書く。イメージを
  替えたら、ここを確かめ直す (design.md 「Revalidation Triggers」)
- **行の途中に現れる文字列を拾える**。記録の行には、vLLM のログの接頭辞
  (`(EngineCore_DP0 pid=123) INFO 09-21 12:00:00 [file.py:123]`) や、NCCL の
  `<hostname>:<pid>:<tid> [<cudaDev>]` の形、docker の `--timestamps` の時刻が付くので、
  正規表現は `re.search` (行頭に縛らない) で、行の内側で使う
- **同じ項目が 2 度現れたとき (head と worker、再試行) は、最初に現れたものを返す**。
  記録を行に分けて (下記)、行の並びの順に探し、最初に一致した行の一致を、値を持つ項目の
  すべてで一貫して使う。真偽の項目 (`ib_no_device`、`merged_nic`、`gdrdma_seen`、
  `socket_channel_seen`) は、位置を問わず「どこかの行に 1 回でも出たか」で判定するので、
  この決め方の対象ではない
- **知っている失敗が 2 種類ある記録では、`types.KnownFailure` の並び順で先に来るものを
  返す** (`_KNOWN_FAILURE_PATTERNS` の並びが、その優先順位そのもの)。`UNCLASSIFIED` は
  ここでは付けない (この module は、6 つの知っている型を見分けるだけで、それ以外の終了が
  「失敗」かどうかを知る手段 (終了コードなど) を持たないため。付けるなら、その手段を持つ
  呼び出し元 (`probe` など) の仕事にする)
- **`failure_excerpt` の行数の上限**: `types.LaunchObservation.failure_excerpt` の
  `max_length` (40) を、この module の `_MAX_FAILURE_EXCERPT_LINES` に同じ値で持つ
  (`types.py` は凍結されているので、私の名前 (アンダースコア始まり) を直接 import せず、
  値を揃える。`test_observe.py` の `test_failure_excerpt_is_capped_at_forty_lines` が、
  食い違えば pydantic の `ValidationError` で気づける形にして固定する)
- **正規表現の後戻りは、入力の長さの二乗以上にならないようにする**
  (task 2.2 のレビューの差し戻し、指摘 1、および 2 回目の差し戻し)。これを、**構造の上限**
  と、**個々の正規表現の書き方**の、両方で守る (どちらか一方には頼らない)。

  1. **構造の上限 (`_split_lines` / `_truncated_lines`)**: `observe_launch` /
     `observe_nccl` は、記録を行に分け (`_split_lines`)、**1 行ごとに** (`re.DOTALL` を
     使わず、行をまたがせずに) 正規表現を当てる。探す文字列の集まりは、**2 つの組**に
     分かれる (3 回目の差し戻し、指摘: 「手間の上限が要る探索」と「読み落としのなさを
     優先する探索」の分類が module になかった):

     a. **量指定子を持たない探索** (`_SPECULATIVE_CONFIG_MARKER`、
        `_KNOWN_FAILURE_PATTERNS` の 6 件、`_NETWORK_RE`、`_IB_NO_DEVICE_RE`。合わせて
        9 件) は、切り詰めていない行 (`_split_lines` がそのまま返す行) に当てる。これらは
        リテラルの `in` か、量指定子を持たないリテラルの選択だけなので、手間は入力の長さに
        線形で、後戻りが起こりようがない (実測: 5,000 万文字の 1 行で、`in` が 1.70 ms、
        リテラルの `re.search` 6 件が 37.85 ms)。切り詰めると、長い 1 行 (vLLM は全設定を
        1 行でダンプする。数キロバイトになりうる) の `_MAX_LINE_LENGTH` より後ろにある印を
        読み落とすので、読み落としのなさを優先し、切り詰めない
     b. **量指定子を持つ探索** (上記以外のすべて) は、`_truncated_lines` で
        `_MAX_LINE_LENGTH` (4,096 文字。vLLM と NCCL の記録の行は、これよりずっと短い) に
        切り詰めた行に当てる。これで、**個々の正規表現にどんな後戻りの性質が残っていても**、
        1 行あたりの手間が、その行の実際の長さによらず定数で抑えられ、記録の全体に掛かる
        手間は、行数に対して線形になる。上限より長い行の、上限より後ろの文字列は読まれない
        (この組み分けの対象の行に限る。書いた決まりとして試験で固定する)。`failure_excerpt`
        に入れる行は、一致した行だけ、一致の位置の前 `_EXCERPT_WINDOW_BEFORE` (512) 文字
        から `_MAX_LINE_LENGTH` 文字を取り、ほかの行は先頭から `_MAX_LINE_LENGTH` 文字を
        取る (窓の決まり。`_excerpt_around` 参照)。**新しい探す定数を足すときは、(a)(b) の
        どちらの組かを決め、`test_observe.py` の分類の表 (`_UNTRUNCATED_PATTERNS` /
        `_TRUNCATED_PATTERNS`) にも足すこと。忘れると
        `test_every_regex_pattern_is_classified_as_truncated_or_untruncated` が落ちる**
  2. **個々の正規表現の書き方**: 構造の上限だけに頼らず、それぞれの正規表現も、後戻りが
     線形になる形にする (二重に守る)。見直すのは、少なくとも次の 3 つの形:
     a. 隣り合う 2 つの無制限の量指定子の文字集合が重なり、そのあとに一致しないことの
        あるリテラルが続く形 (`_NCCL_VERSION_RE` で見つけた。量指定子のすぐあとに、その
        量指定子が自分では食えない文字を置いて、量指定子の終わりが一意に決まるように直す)
     b. 量指定子がパターンの先頭にあり、その前にリテラルの錨がない形 (`_COLL_CHANNELS_RE`
        で見つけた。錨がないと、`re.search` が試すすべての開始位置で量指定子が後戻りする
        ので、1 つの開始位置での後戻りが線形でも、全体では二次関数的になる。量指定子の
        桁数を現実的な上限で区切って直す)
     c. **錨のリテラルが記録の中で繰り返されると、除く文字を持たない量指定子 (`.*`、
        `.+`) が、錨の出現のたびに、末尾のリテラルを求めて記録の残り全体を探し直す形**
        (`_MERGED_NIC_VIRTUAL_DEVICE_RE` で見つけた。2 回目の差し戻しの指摘)。
        **「量指定子が 1 つだけで、パターンの先頭がリテラルの錨なら安全」という、前回
        この docstring に書いた原則は誤りだった。** 安全なのは、(i) 量指定子の直後に
        何も続かない形 (`.*$` など。一致すればそれで終わり、失敗して後戻りをやり直す
        必要がない)、または (ii) 量指定子の文字集合が、錨の中にもよく出る文字 (空白や、
        錨自身の終わりの文字) を除いている形 (`\S+`、`[^']+`、`[^\s+]+` など。錨が
        繰り返されると、その除いた文字も一緒に繰り返されるので、量指定子の後戻りが、
        錨とは無関係に、いつも短い範囲で終わる) だけである。**`.` はどんな文字も除かない
        ので、この保証がない**。直すには、量指定子の幅を、現実的な上限で区切る
        (`.{0,200}` のように)

  この 2 つを両方守っていることを、`test_observe.py` の `*_is_linear_time` (錨が 1 回の
  形) と、`*_anchor_repeated_*` (錨を 1 行の中で繰り返す形と、行をまたいで繰り返す形の
  両方) の試験群が、この module の正規表現の定数のすべてについて固定する

**CONCERNS として残すこと**:

- `vllm_version` は、この module では読まない (つねに `None`)。design.md の観察の表は、
  版の出どころを「`GET /version`、または起動時のバナー」としているが、ログの文字列としての
  確かな探す文字列が、research.md にも design.md にも見つからなかった (バナーの原文が引かれ
  ていない)。推測で文字列を作らずに、この項目は空のままにする。`GET /version` は HTTP の
  応答なので、ログの文字列だけを読む、この入出力のない関数の外 (`lifecycle` など、通信を
  持つ部品) の仕事になる
- `engine_init_s` は、`init engine (profile, create kv cache, warmup model) took` という
  接頭ぎだけが research.md の原文 (`…` で終わる) で確かで、続く数値の書式 (単位、小数点の
  桁) は確かめられていない。接頭ぎ (確かな部分) を探す文字列にし、続く数値をそのまま浮動小数
  として読む。イメージを替えたときに、まずここを確かめ直す
- `attention_candidates`、`max_concurrency_note`、`ib_devices_line` は、行の末尾を取るので、
  4,096 文字を超える行では値が途中で切れうる。上流の原文は、どれも短い行なので、受け入れる
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final, Literal

from serving_kit.types import KnownFailure, LaunchObservation, NcclObservation

__all__ = ["observe_launch", "observe_nccl"]

# --- 構造の上限 (二次関数的な後戻りへの、正規表現の書き方に頼らない歯止め) -

_MAX_LINE_LENGTH: Final[int] = 4096
"""正規表現に掛ける前に、1 行を切り詰める長さの上限 (module の docstring の「構造の上限」)。

vLLM と NCCL の記録の 1 行は、これよりずっと短い。これにより、個々の正規表現にどんな
後戻りの性質が残っていても、1 行あたりの手間が、その行の実際の長さによらず定数で抑えられ、
記録の全体に掛かる手間は、行数に対して線形になる (task 2.2 の 2 回目のレビューの差し戻し)。
上限より後ろの文字列は読まれない。
"""

# --- 誤りの前後の抜き出し ------------------------------------------------

_MAX_FAILURE_EXCERPT_LINES: Final[int] = 40
"""`types.LaunchObservation.failure_excerpt` の `max_length` と同じ値 (module の docstring)。"""

_FAILURE_EXCERPT_BEFORE: Final[int] = 19
_FAILURE_EXCERPT_AFTER: Final[int] = 20
"""誤りの行の前後に抜き出す行数 (前 19 + 一致した行 1 + 後 20 = 40 で、上の上限に合わせる)。"""

_EXCERPT_WINDOW_BEFORE: Final[int] = 512
"""一致した行を切り出すとき、一致の位置より手前に残す文字数 (module の docstring の「構造の
上限」)。`_KNOWN_FAILURE_PATTERNS` は切り詰めていない行 (`raw_lines`) に当てるので、一致の
位置が `_MAX_LINE_LENGTH` より後ろのこともある。一致した行だけ、一致の位置の前 512 文字から
`_MAX_LINE_LENGTH` 文字を取り、失敗の文面を必ず抜き出しに入れつつ、巨大な行を丸ごとは
抜き出しに入れない (窓の決まり)。"""


# --- 起動の記録: アテンションとエキスパートのバックエンド ---------------
# 出どころ: research.md §d-2 の観察の表、`vllm/platforms/cuda.py` L536。
# 原文: "Using %s attention backend out of potential backends: %s."
# `\S+` は、直後の必須の空白 (文字集合の外) で区切られる。錨 ("Using ") が繰り返されても、
# 空白も一緒に繰り返されるので、後戻りはいつも短い (module の docstring の 2-c)。1 行の
# 中でしか使わないので、`re.MULTILINE` は要らない (`$` は文字列の終わりに一致する)。
_ATTENTION_BACKEND_RE: Final[re.Pattern[str]] = re.compile(
    r"Using (\S+) attention backend out of potential backends: (.+)$"
)

# 出どころ: research.md §d-2 の観察の表、
# `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` L228-233。
# 原文: "Using '{backend}' NvFp4 MoE backend out of potential backends: {...}."
# `[^']+` は、直後の必須の `'` (文字集合の外) で区切られる。錨 ("Using '") 自身が `'` で
# 終わるので、錨が繰り返されても後戻りは短い。
_MOE_BACKEND_RE: Final[re.Pattern[str]] = re.compile(
    r"Using '([^']+)' NvFp4 MoE backend out of potential backends"
)

# --- 起動の記録: KV キャッシュ、ロードと起動の所要 -----------------------
# 出どころ: research.md §d-2 の観察の表、`vllm/v1/core/kv_cache_utils.py` L2405-2406。
# 原文: "GPU KV cache size: N tokens, Maximum concurrency for M tokens per request: X.XXx"
_KV_CACHE_SIZE_RE: Final[re.Pattern[str]] = re.compile(
    r"GPU KV cache size: ([\d,]+) tokens, (Maximum concurrency for [^\n]+)"
)

# 出どころ: research.md §d-2 の観察の表、`vllm/v1/worker/gpu_worker.py` L645-648。
# 原文: "Available KV cache memory: N GiB"
_KV_CACHE_MEMORY_RE: Final[re.Pattern[str]] = re.compile(r"Available KV cache memory: ([\d.]+) GiB")

# 出どころ: research.md §d-2 の観察の表 (重みのロード時間)。
# 原文: "Model loading took %s GiB memory and %.6f seconds"
_MODEL_LOADING_RE: Final[re.Pattern[str]] = re.compile(
    r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds"
)

# 出どころ: research.md §d-2 の観察の表 (起動全体)。原文の接頭ぎだけが確か (module の
# docstring の CONCERNS)。原文: "init engine (profile, create kv cache, warmup model) took …"
_ENGINE_INIT_RE: Final[re.Pattern[str]] = re.compile(
    r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+)"
)

# --- 起動の記録: 投機的デコード ------------------------------------------
# 出どころ: research.md §d-7 (投機的デコードを確実に切る)。
# 原文: "起動時の設定のダンプに `SpeculativeConfig(...)` が出ないこと"
_SPECULATIVE_CONFIG_MARKER: Final[str] = "SpeculativeConfig("

# --- 起動の記録: 知っている失敗 (design.md 「observe」の表) --------------
# 並び順が、判定の優先順位である (`types.KnownFailure` の並びと同じ)。1 つの記録に 2 つの
# 失敗の文面が両方あるときは、この並びで先に来るものを返す (module の docstring、
# tasks.md 2.2 の完了の状態、test_two_known_failure_strings_prefer_the_earlier_enum_member)。
# いずれも量指定子のないリテラルだけなので、後戻りの穴はない。
_KNOWN_FAILURE_PATTERNS: Final[tuple[tuple[KnownFailure, re.Pattern[str]], ...]] = (
    # 出どころ: research.md §d-1、issue #57578。`csrc/libtorch_stable/cache_kernels.cu`。
    (KnownFailure.PE_DIM_ASSERT, re.compile(r"pe_dim must be 64")),
    # 出どころ: research.md §d-1、issue #53963。`vllm/v1/attention/backend.py`。
    (KnownFailure.NO_ATTENTION_BACKEND, re.compile(r"No valid attention backend found")),
    # 出どころ: research.md §a-3。sm_121 がビルドの対象にない。
    (
        KnownFailure.NO_KERNEL_IMAGE,
        re.compile(r"no kernel image is available for execution on the device"),
    ),
    # 出どころ: research.md §d-2 の観察の表。`vllm/models/glm5next/common/attention.py`。
    (KnownFailure.KPOOL_BLOCK_SIZE, re.compile(r"kpool indexer requires cache block_size")),
    # 出どころ: research.md §d-6。`vllm/v1/worker/gpu_worker.py` L541-550。
    (
        KnownFailure.STARTUP_MEMORY_CHECK,
        re.compile(r"is less than desired GPU memory utilization"),
    ),
    # 出どころ: research.md §d-9。`vllm/models/glm5next/nvidia/sparse_indexer.py`。
    (KnownFailure.DEEP_GEMM_MISSING, re.compile(r"requires DeepGEMM to be installed")),
)

# --- 通信の記録 (research.md §e-4、NCCL のソースの書式) -------------------
# 原文: "NCCL version [0-9.]+[^ ]*\+cuda[0-9.]+" (`src/init.cc` の `VERSION_STRING`)。
# **実装は、原文の正規表現のまま書き写さない**。`[0-9.]+` と `[^ ]*` は、文字集合が
# ([0-9.] はどちらも [^ ] に含まれる) 完全に重なる、隣り合う 2 つの無制限の量指定子で、
# その直後に、一致しないことのあるリテラル `\+cuda` が続く形 (`(a+)(a*)b` の典型) なので、
# `+cuda` を含まない数字とピリオドの塊を与えると、2 つの量指定子の間で分け方を総当たりし、
# 二次関数的な後戻りが起きる (task 2.2 のレビューでの実測: 正規表現だけで 80,000 文字に
# 1.39 秒)。代わりに、後に続くリテラルの最初の文字 (`+`) を、手前の量指定子の文字集合から
# 除き、自分では `+cuda` を食えない 1 つの量指定子にする (`[^\s+]+`)。これで、量指定子の
# 終わりが一意に決まり (`+` が現れた位置、またはテキストの終わり)、後戻りが起こらない。
# 錨 ("NCCL version ") 自身が空白で終わるので、錨が繰り返されても後戻りは短い。
_NCCL_VERSION_RE: Final[re.Pattern[str]] = re.compile(r"NCCL version ([^\s+]+\+cuda[0-9.]+)")

# 原文: "NCCL INFO Using network (IB|Socket)" (`src/init.cc:557`)。最も確実な 1 行。
# 量指定子を持たない (2 つのリテラルの選択のみ) ので、後戻りの穴はない。
_NETWORK_RE: Final[re.Pattern[str]] = re.compile(r"NCCL INFO Using network (IB|Socket)")

# 原文: "NET/IB : No device found\." (`src/transport/net_ib/init.cc`)。量指定子のない
# リテラルだけなので、後戻りの穴はない。
_IB_NO_DEVICE_RE: Final[re.Pattern[str]] = re.compile(r"NET/IB : No device found\.")

# 原文: "NET/IB : Using" (`"NET/IB : Using%s %s; OOB %s:%s"`)。行の残りを、選ばれた
# IB デバイスの情報として、そのまま持つ。量指定子の直後に何も続かない (`$` だけ) ので、
# 一致すればそれで終わり、後戻りをやり直す必要がない。1 行の中でしか使わないので、
# `re.MULTILINE` は要らない。
_IB_USING_RE: Final[re.Pattern[str]] = re.compile(r"(NET/IB : Using.*)$")

# 出どころ: research.md §e-4。原文: "NET/IB : Made virtual device \[[0-9]+\] name=([^ ]+)
# .* ndevs=([2-9])"。`ndevs=1` の素のデバイスにも毎回出るので、判定は `ndevs>=2` のときだけ
# にする (design.md 「observe」の `NcclObservation` の欄の注記のとおり:
# `merged_nic: bool  # "TOPO/NET : Made vNic" か、ndevs>=2`。`types.py` の
# `NcclObservation.merged_nic` 自身には、この注記はない)。research.md §e-4 にある、もう
# 1 つの代替の判定 (name に `+` を含むこと) は、design.md の上の注記がこの 2 つに絞って
# いるので、ここでは実装しない。
#
# **`.*` は、原文の正規表現のまま書き写さない (task 2.2 の 2 回目のレビューの差し戻し)。**
# `.` はどんな文字も除かないので、`name=` と `ndevs=` の間の錨 (`NET/IB : Made virtual
# device [0] name=x `) が記録の中で繰り返されると、`re.search` は、錨の出現のたびに、
# `.*` に `ndevs=` を求めて記録の残り全体を探し直す。これが、錨の出現の回数だけ繰り返されて
# 二次関数的になる (実測: 錨を 5,000 回で 0.487 秒、50,000 回で 52.8 秒)。研究資料の原文の
# 形では、`name=` と `ndevs=` の間は `speed=…` 程度の短い記述なので、`.*` を `.{0,200}`
# (現実的な上限つきの量指定子) に区切る。これで、量指定子自身の後戻りの幅が、錨の繰り返しの
# 有無によらず、つねに高々 200 文字に収まる。`_MAX_LINE_LENGTH` による行の切り詰め
# (module の docstring の「構造の上限」) と、二重に後戻りを防ぐ。
_MERGED_NIC_VIRTUAL_DEVICE_RE: Final[re.Pattern[str]] = re.compile(
    r"NET/IB : Made virtual device \[[0-9]+\] name=([^ ]+) .{0,200} ndevs=([2-9])"
)

# 原文: "TOPO/NET : Made vNic [0-9]+"。`ndevs==1` では出ないので、直接の証拠。量指定子の
# 直後に何も続かないので、一致すればそれで終わり、後戻りをやり直す必要がない。
_MERGED_NIC_VNIC_RE: Final[re.Pattern[str]] = re.compile(r"TOPO/NET : Made vNic [0-9]+")

# 原文: "%d coll channels, %d collnet channels" (`src/init.cc`)。coll channels の数だけを
# 読む (`collnet channels` に対応する項目は `NcclObservation` にない)。
# **この定数だけ、量指定子の前にリテラルの錨がない** (パターンが `[0-9]+` から始まる)。
# 記録が数字だけの塊 (錨になる文字列を含まない) だと、`re.search` は 200,000 文字なら
# 200,000 個の開始位置のすべてで一致を試み、位置ごとに `[0-9]+` が最後まで後戻りしてから
# 失敗するので、全体で二次関数的になる (指摘 1 と同じ「見直すこと」で見つけた、もう 1 つの
# 穴)。チャンネルの数は、現実にはたかだか数桁なので、桁数を 10 に区切って、後戻りの幅を
# 開始位置によらず一定にする (`src/init.cc` の `%d` は int なので、10 進で最大 10 桁あれば
# 十分に余裕がある)。
_COLL_CHANNELS_RE: Final[re.Pattern[str]] = re.compile(
    r"([0-9]{1,10}) coll channels, ([0-9]{1,10}) collnet channels"
)

# 原文: "Channel [0-9]{2}/[0-9]+ : .* \[(send|receive)\] via NET/(IB|Socket)/[0-9]+" の
# うち、Socket の側だけを見る (IB の側は `network` と `ib_devices_line` で足りる)。量指定子
# の直後に何も続かないので、一致すればそれで終わり、後戻りをやり直す必要がない。
_SOCKET_CHANNEL_RE: Final[re.Pattern[str]] = re.compile(
    r"\[(?:send|receive)\] via NET/Socket/[0-9]+"
)

# 原文: "via NET/IB/[0-9]+/GDRDMA" (`req.useGdr ? "/GDRDMA" : ""`)。DGX Spark では出ない
# のが正常 (research.md §e-2)。`[0-9]+` は、直後の必須の `/` (文字集合の外) で区切られる。
_GDRDMA_RE: Final[re.Pattern[str]] = re.compile(r"via NET/IB/[0-9]+/GDRDMA")


def _split_lines(log_text: str) -> tuple[str, ...]:
    """記録を行に分ける (切り詰めない)。

    量指定子を持たない探索 (module の docstring の「構造の上限」の組 (a)) は、この関数が
    返す、切り詰めていない行にそのまま当てる。手間は入力の長さに線形で、後戻りが起こらない
    ので、`_MAX_LINE_LENGTH` より後ろにある印を読み落とさない。
    """
    return tuple(log_text.splitlines())


def _truncated_lines(lines: Sequence[str]) -> tuple[str, ...]:
    """各行を `_MAX_LINE_LENGTH` で切り詰める (module の docstring の「構造の上限」の組 (b))。

    量指定子を持つ探索は、この関数が返す、切り詰めた行にだけ、行をまたがせずに
    (`re.DOTALL` を使わず) 当てる。上限より長い行の、上限より後ろの文字列は、ここで
    捨てられ、以後は読まれない。
    """
    return tuple(line[:_MAX_LINE_LENGTH] for line in lines)


def _first_match(pattern: re.Pattern[str], lines: Sequence[str]) -> re.Match[str] | None:
    """行の並びの順に正規表現を当て、最初に一致した行の一致を返す (行をまたがない)。"""
    for line in lines:
        match = pattern.search(line)
        if match is not None:
            return match
    return None


def _any_match(pattern: re.Pattern[str], lines: Sequence[str]) -> bool:
    """どれかの行に、正規表現が 1 回でも一致するか (真偽の項目に使う)。"""
    return any(pattern.search(line) is not None for line in lines)


def _parse_backend_list(text: str) -> tuple[str, ...]:
    """`out of potential backends: ` の続きを、バックエンドの名前の列に直す。

    レンダリングの形 (Python の `%s` に list を渡したときの `['A', 'B']` の形か、角かっこの
    ない `A, B` の形か) が research.md に示されていないので、両方を受け入れる: 末尾の
    `.` を落とし、外側の `[` `]` があれば剥がし、`,` で割って、前後の空白と引用符を落とす。
    """
    stripped = text.strip()
    if stripped.endswith("."):
        stripped = stripped[:-1]
    stripped = stripped.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1]
    return tuple(part.strip().strip("'\"") for part in stripped.split(",") if part.strip())


def _excerpt_around(lines: Sequence[str], line_index: int, match_start: int) -> tuple[str, ...]:
    """一致した行の前後の行を抜き出す (上限は `_MAX_FAILURE_EXCERPT_LINES`)。

    `lines` は、**切り詰めていない行** (`raw_lines`) を渡す。`_KNOWN_FAILURE_PATTERNS` は
    切り詰めていない行に当てるので、一致の位置 (`match_start`) が `_MAX_LINE_LENGTH` より
    後ろのこともある。一致した行だけ、一致の位置の前 `_EXCERPT_WINDOW_BEFORE` 文字から
    `_MAX_LINE_LENGTH` 文字を取り (窓の決まり)、失敗の文面を必ず抜き出しに入れる。ほかの行は
    先頭から `_MAX_LINE_LENGTH` 文字を取る (module の docstring の「構造の上限」)。
    """
    start = max(0, line_index - _FAILURE_EXCERPT_BEFORE)
    end = min(len(lines), line_index + _FAILURE_EXCERPT_AFTER + 1)
    window_start = max(0, match_start - _EXCERPT_WINDOW_BEFORE)
    excerpted = tuple(
        line[window_start : window_start + _MAX_LINE_LENGTH]
        if i == line_index
        else line[:_MAX_LINE_LENGTH]
        for i, line in enumerate(lines[start:end], start=start)
    )
    return excerpted[:_MAX_FAILURE_EXCERPT_LINES]


def _known_failure(lines: Sequence[str]) -> tuple[KnownFailure | None, tuple[str, ...]]:
    """知っている失敗を、優先順位 (`_KNOWN_FAILURE_PATTERNS` の並び) で 1 つだけ選ぶ。

    優先順位が先のパターンから順に、行の並びの順に探し、最初に一致した行を見つけた時点で
    決める (優先順位が、一致した位置より優先する。既存の決め方をそのまま保つ)。`lines` は
    切り詰めていない行 (`raw_lines`) を渡す (module の docstring の「構造の上限」の組 (a))。
    """
    for failure, pattern in _KNOWN_FAILURE_PATTERNS:
        for line_index, line in enumerate(lines):
            match = pattern.search(line)
            if match is not None:
                return failure, _excerpt_around(lines, line_index, match.start())
    return None, ()


def observe_launch(log_text: str) -> LaunchObservation:
    """起動の記録から、事実を抜き出す (design.md 「observe」、requirements 4.4、5.2、5.3、6.3)。

    入出力のない純粋な関数。読めなかった項目は、断らずに `None` (または `()`、`False`) の
    まま返す (module の docstring の決め方)。記録は行に分け、1 行ごとに正規表現を当てる
    (行をまたがせない)。1 行が `_MAX_LINE_LENGTH` より長い場合、上限より後ろの文字列は
    読まれない (module の docstring の「構造の上限」)。

    引数:
        log_text: 起動の記録の全文 (`docker logs` の出力。時刻やログの接頭辞を含んでいて
            よい。2 台ぶんや再試行ぶんを連ねたものでもよいが、同じ項目が 2 度現れたときは
            最初に現れたものを返す)。

    返り値:
        `LaunchObservation`。`vllm_version` は、確かな探す文字列がないので、つねに `None`
        (module の docstring の CONCERNS)。
    """
    raw_lines = _split_lines(log_text)
    lines = _truncated_lines(raw_lines)
    known_failure, failure_excerpt = _known_failure(raw_lines)

    attention_match = _first_match(_ATTENTION_BACKEND_RE, lines)
    attention_backend = attention_match.group(1) if attention_match is not None else None
    attention_candidates = (
        _parse_backend_list(attention_match.group(2)) if attention_match is not None else ()
    )

    moe_match = _first_match(_MOE_BACKEND_RE, lines)
    moe_backend = moe_match.group(1) if moe_match is not None else None

    kv_size_match = _first_match(_KV_CACHE_SIZE_RE, lines)
    kv_cache_tokens = (
        int(kv_size_match.group(1).replace(",", "")) if kv_size_match is not None else None
    )
    max_concurrency_note = kv_size_match.group(2).strip() if kv_size_match is not None else None

    kv_memory_match = _first_match(_KV_CACHE_MEMORY_RE, lines)
    kv_cache_gib = float(kv_memory_match.group(1)) if kv_memory_match is not None else None

    loading_match = _first_match(_MODEL_LOADING_RE, lines)
    model_loading_gib = float(loading_match.group(1)) if loading_match is not None else None
    model_loading_s = float(loading_match.group(2)) if loading_match is not None else None

    engine_init_match = _first_match(_ENGINE_INIT_RE, lines)
    engine_init_s = float(engine_init_match.group(1)) if engine_init_match is not None else None

    speculative_config_seen = any(_SPECULATIVE_CONFIG_MARKER in line for line in raw_lines)

    return LaunchObservation(
        vllm_version=None,
        attention_backend=attention_backend,
        attention_candidates=attention_candidates,
        moe_backend=moe_backend,
        kv_cache_tokens=kv_cache_tokens,
        kv_cache_gib=kv_cache_gib,
        max_concurrency_note=max_concurrency_note,
        model_loading_gib=model_loading_gib,
        model_loading_s=model_loading_s,
        engine_init_s=engine_init_s,
        speculative_config_seen=speculative_config_seen,
        known_failure=known_failure,
        failure_excerpt=failure_excerpt,
    )


def _network_literal(value: str | None) -> Literal["IB", "Socket"] | None:
    """マッチした文字列を、`NcclObservation.network` の `Literal` に絞る。"""
    if value == "IB":
        return "IB"
    if value == "Socket":
        return "Socket"
    return None


def observe_nccl(log_text: str) -> NcclObservation:
    """通信の記録から、事実を抜き出す (design.md 「observe」、requirements 4.4)。

    入出力のない純粋な関数。読めなかった項目は、断らずに `None` (または `False`) のまま
    返す (module の docstring の決め方)。真偽の項目は、位置を問わず「どこかの行に 1 回でも
    出たか」で判定する。記録は行に分け、1 行ごとに正規表現を当てる (行をまたがせない)。
    1 行が `_MAX_LINE_LENGTH` より長い場合、上限より後ろの文字列は読まれない (module の
    docstring の「構造の上限」)。

    引数:
        log_text: `NCCL_DEBUG=INFO` で採った通信の記録の全文 (2 台ぶんを連ねたものでも
            よいが、値を持つ項目 (`nccl_version`、`network`、`ib_devices_line`、
            `coll_channels`) が 2 度現れたときは、最初に現れたものを返す)。

    返り値:
        `NcclObservation`。`gdrdma_seen` は、DGX Spark では偽が正常 (research.md §e-2)。
    """
    raw_lines = _split_lines(log_text)
    lines = _truncated_lines(raw_lines)

    version_match = _first_match(_NCCL_VERSION_RE, lines)
    nccl_version = version_match.group(1) if version_match is not None else None

    network_match = _first_match(_NETWORK_RE, raw_lines)
    network = _network_literal(network_match.group(1) if network_match is not None else None)

    ib_no_device = _any_match(_IB_NO_DEVICE_RE, raw_lines)

    ib_using_match = _first_match(_IB_USING_RE, lines)
    ib_devices_line = ib_using_match.group(1) if ib_using_match is not None else None

    merged_nic = _any_match(_MERGED_NIC_VNIC_RE, lines) or _any_match(
        _MERGED_NIC_VIRTUAL_DEVICE_RE, lines
    )

    coll_match = _first_match(_COLL_CHANNELS_RE, lines)
    coll_channels = int(coll_match.group(1)) if coll_match is not None else None

    gdrdma_seen = _any_match(_GDRDMA_RE, lines)
    socket_channel_seen = _any_match(_SOCKET_CHANNEL_RE, lines)

    return NcclObservation(
        nccl_version=nccl_version,
        network=network,
        ib_no_device=ib_no_device,
        ib_devices_line=ib_devices_line,
        merged_nic=merged_nic,
        coll_channels=coll_channels,
        gdrdma_seen=gdrdma_seen,
        socket_channel_seen=socket_channel_seen,
    )
