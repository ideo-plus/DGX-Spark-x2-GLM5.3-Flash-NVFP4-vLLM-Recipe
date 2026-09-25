"""同時処理の合成入力の、対象サーバーに数えた長さでの合わせ込み (issue #9、#26)。

別のテストファイルからは import しない (助けは `test_suites_concurrency.py` から
要る分だけ写す)。concurrency の合成入力が、狙いのトークン数 (`target_input_tokens`)
より長くなる問題を確かめる。issue #9 は、要求の決まった分量 (包み: 識別子の行、
指示、`system`、チャットテンプレート) を計測ランの最初に 1 回数えて差し引いたが、
実機では狙いより +3〜4% (約 +65 トークン) 長い側に残った (issue #26)。#26 は、
`agent` と同じく、文書ごとに組み立てた要求そのものを数えて文章の長さを合わせ込む。

確かめるのは 4 つのまとまり。

1. **+65 の出どころの切り分け** (#26): 修正前の組み立て (`target − 包み` の文章) を、
   実機の数え方の候補 (線形、改行、識別子のばらつき、文章に依存する固定分、文章の
   比の差) で数えると、どの候補が +65 を作り、どれが作らないかを試験内の模型で示す
2. **合わせ込みの到達** (#26): 文書ごとに組み立てた要求を数えて合わせ込むと、+65 を
   作るどの数え方でも、測った試行が `length_tolerance` の内側に入り、
   `LENGTH_OFF_TARGET` が付かない
3. **合わせ込み用の識別子** (#26): 合わせ込みの数える要求の先頭は、試行の先頭とも
   包みの先頭とも重ならず、`run_id` を含まない (計測ランをまたいで同じ)。文章は
   計測ランをまたいで同じ
4. **失敗の扱い** (issue #9、#26): 狙いが包み以下なら試行を送らずに飛ばし、
   数えられない `ProbeError` は見積もりに戻さず伝播する
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest
from pydantic import JsonValue

from bench_harness.client.messages import HttpxMessagesClient, build_request_body
from bench_harness.client.probe import ProbeError, TokenCount
from bench_harness.corpus.synth import TemplateCorpus
from bench_harness.suites.base import (
    ConditionAborted,
    SuiteContext,
    cold_prefix_nonce,
    iter_trials,
    make_suite_context,
    single_user_message,
    trial_seed,
)
from bench_harness.suites.concurrency import (
    _FRAME_NONCE_KEY,
    _LONG_OUTPUT_INSTRUCTION,
    ConcurrencySuite,
    _system_for,
    _user_text,
)
from bench_harness.types import (
    ConditionPlan,
    MessagesRequest,
    Profile,
    SuiteName,
    TargetDef,
    TrialFlag,
    TrialRecord,
)
from fake_server import (
    FakeServer,
    InputTokenCounter,
    estimator_aligned_counter,
    token_stream_response,
)

RUN_A = "20260920-000000-aaaaaa"
RUN_B = "20260920-111111-bbbbbb"

_INSTRUCTION_SUFFIX = "\n\n" + _LONG_OUTPUT_INSTRUCTION
"""指示だけの本文 (包みを測る要求の text)。"""

_NONCE_RE = re.compile(r"[0-9a-f]{32}")
"""システムプロンプト 1 行目の、32 桁の 16 進の識別子。"""


# --- 助け (test_suites_concurrency.py と同じ形。別のテストファイルからは import しない) --


class BodySink:
    """`RunStore.put_body` の代わり。保存された本文をそのまま覚えておく。"""

    def __init__(self) -> None:
        self.bodies: list[dict[str, JsonValue]] = []
        self.refs: list[str] = []

    def put(self, body: dict[str, JsonValue]) -> str:
        canonical = json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ref = hashlib.sha256(canonical).hexdigest()
        self.bodies.append(body)
        self.refs.append(ref)
        return ref


def make_profile(**overrides: Any) -> Profile:
    data: dict[str, Any] = {
        "name": "test",
        "seed": 7,
        "timeout": {"connect_s": 5.0, "first_event_s": 5.0, "idle_s": 5.0, "total_s": 20.0},
        "length_tolerance": 0.05,
        "chars_per_token": {"prose_en": 4.0, "prose_ja": 4.0, "code": 4.0, "log": 4.0},
        "concurrency": {"levels": [2], "rounds": 2, "max_tokens": 8, "input_tokens": 2000},
    }
    data.update(overrides)
    return Profile.model_validate(data)


def target_for(server: FakeServer) -> TargetDef:
    return TargetDef.model_validate(
        {"name": "fake", "base_url": server.base_url, "model": "fake-model"}
    )


@asynccontextmanager
async def suite_ctx(
    server: FakeServer,
    *,
    profile: Profile | None = None,
    run_id: str = RUN_A,
    count_input_tokens: Any = None,
) -> AsyncIterator[tuple[SuiteContext, BodySink]]:
    """偽のサーバーにつないだ `SuiteContext` と、本文の受け皿を渡す。"""
    sink = BodySink()
    async with HttpxMessagesClient(server.base_url) as client:
        kwargs: dict[str, Any] = {}
        if count_input_tokens is not None:
            kwargs["count_input_tokens"] = count_input_tokens
        ctx = make_suite_context(
            client=client,
            profile=profile if profile is not None else make_profile(),
            target=target_for(server),
            run_id=run_id,
            put_body=sink.put,
            **kwargs,
        )
        yield ctx, sink


def plans(suite: ConcurrencySuite, ctx: SuiteContext) -> list[ConditionPlan]:
    return [item for item in suite.plan(ctx) if isinstance(item, ConditionPlan)]


async def run_all(suite: ConcurrencySuite, ctx: SuiteContext) -> list[TrialRecord]:
    return [record for cond in plans(suite, ctx) async for record in suite.run_condition(ctx, cond)]


def first_text(body: dict[str, Any]) -> str:
    return str(body["messages"][0]["content"][0]["text"])


def system_first_line(body: dict[str, Any]) -> str:
    return str(body["system"]).splitlines()[0]


def frame_tokens_of(server: FakeServer) -> int:
    """偽のサーバーが記録した包みの計測 (count_tokens の 1 件目) のトークン数。"""
    return server.requests_for("/v1/messages/count_tokens")[0].input_tokens


def _content_texts(body: dict[str, Any]) -> list[str]:
    """要求の本文の、text ブロックの文字列を集める (数え方の差し替えの助け)。"""
    texts: list[str] = []
    for message in body.get("messages") or []:
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        for block in content or []:
            if block.get("type") == "text":
                texts.append(str(block.get("text", "")))
    return texts


def _document_part(text: str) -> str:
    """本文のうち、末尾の指示より前の部分 (文章そのもの)。"""
    if text.endswith(_INSTRUCTION_SUFFIX):
        return text[: -len(_INSTRUCTION_SUFFIX)]
    return ""


# --- 実機の数え方の候補 (issue #26 の +65 の出どころ) -----------------------


def linear_counter() -> InputTokenCounter:
    """文字数に線形な数え方 (実機の比の模型。補正の残差は丸めだけ)。"""
    return estimator_aligned_counter(fixed=150, chars_per_token=4.0)


def per_newline_counter() -> InputTokenCounter:
    """本文の改行 1 つにつき +1。文書の段落の切れ目を、比の外に数える模型。"""
    base = linear_counter()

    def count(body: dict[str, Any] | None) -> int:
        return base(body) + sum(text.count("\n") for text in _content_texts(body or {}))

    return count


def nonce_jitter_counter() -> InputTokenCounter:
    """システムプロンプト 1 行目の識別子から、決定的に −5〜+5 を足す模型。"""
    base = linear_counter()

    def count(body: dict[str, Any] | None) -> int:
        line = system_first_line(body) if body else ""
        match = _NONCE_RE.search(line)
        jitter = int(match.group(0), 16) % 11 - 5 if match else 0
        return base(body) + jitter

    return count


def fixed_with_document_counter() -> InputTokenCounter:
    """文章があるときだけ +65。比にも包みにも入らない固定分の模型。"""
    base = linear_counter()

    def count(body: dict[str, Any] | None) -> int:
        longer = any(len(text) > len(_INSTRUCTION_SUFFIX) for text in _content_texts(body or {}))
        return base(body) + (65 if longer else 0)

    return count


def dense_document_counter() -> InputTokenCounter:
    """文章を 4.0 / 1.034 で数える。文章の比が較正の標本と違う模型。"""
    base = linear_counter()

    def count(body: dict[str, Any] | None) -> int:
        extra = 0
        for text in _content_texts(body or {}):
            document = _document_part(text)
            if document:
                extra += round(len(document) / (4.0 / 1.034)) - round(len(document) / 4.0)
        return base(body) + extra

    return count


_CANDIDATE_COUNTERS: tuple[tuple[str, Callable[[], InputTokenCounter]], ...] = (
    ("linear", linear_counter),
    ("per_newline", per_newline_counter),
    ("nonce_jitter", nonce_jitter_counter),
    ("fixed_with_document", fixed_with_document_counter),
    ("dense_document", dense_document_counter),
)


async def unfitted_residuals(
    counter: InputTokenCounter, ctx: SuiteContext, cond: ConditionPlan
) -> list[int]:
    """修正前の組み立て (`target − 包み` の文章) の残差 (`counter(本文) − 狙い`) を返す。

    包み (文章を空にした要求) を同じ数え方で数え、測った回・ストリームごとに
    `prose("en", 狙い − 包み, seed)` で本文を組み立てる。計測は偽のサーバーに
    送らない (数え方の模型だけを通す)。
    """
    target = cond.target_input_tokens
    assert target is not None
    frame_nonce = ctx.corpus.prefix_nonce(ctx.profile.seed, _FRAME_NONCE_KEY, ctx.run_id)
    frame_body = build_request_body(
        MessagesRequest(
            model=ctx.target.model,
            max_tokens=1,
            system=_system_for(frame_nonce),
            messages=single_user_message(_user_text("")),
        )
    )
    frame = counter(frame_body)

    residuals: list[int] = []
    for round_index, warmup in iter_trials(cond):
        if warmup:
            continue
        for stream_index in range(cond.concurrency):
            seed = trial_seed(
                ctx.profile.seed, cond.key, round_index, warmup=False, stream_index=stream_index
            )
            nonce = cold_prefix_nonce(
                ctx, cond, trial_index=round_index, warmup=False, stream_index=stream_index
            )
            document = ctx.corpus.prose("en", target - frame, seed)
            body = build_request_body(
                MessagesRequest(
                    model=ctx.target.model,
                    max_tokens=cond.max_tokens,
                    system=_system_for(nonce),
                    messages=single_user_message(_user_text(document)),
                )
            )
            residuals.append(counter(body) - target)
    return residuals


# --- C1: +65 を作る数え方と作らない数え方 -----------------------------------


async def test_candidate_counters_show_which_can_produce_the_bias_before_the_fit(
    fake_server: FakeServer,
) -> None:
    """修正前の組み立てでは、線形・改行・識別子のばらつきは +65 を作らず、
    文章に依存する固定分と比の差だけが +65 を作る (線形 1 つでは結論できない)。
    """
    profile = make_profile(
        concurrency={"levels": [1, 2, 4, 8], "rounds": 2, "max_tokens": 8, "input_tokens": 2000}
    )
    suite = ConcurrencySuite()

    async with suite_ctx(fake_server, profile=profile) as (ctx, _sink):
        conditions = plans(suite, ctx)
        medians: dict[str, float] = {}
        for name, factory in _CANDIDATE_COUNTERS:
            residuals: list[int] = []
            for cond in conditions:
                residuals.extend(await unfitted_residuals(factory(), ctx, cond))
            medians[name] = statistics.median(residuals)

    assert abs(medians["linear"]) <= 0.02 * 1850
    assert 0 < medians["per_newline"] < 65
    assert abs(medians["nonce_jitter"]) < 65
    assert medians["fixed_with_document"] >= 50
    assert medians["dense_document"] >= 50


# --- C2: 数えて合わせ込むと、どの数え方でも狙いに入る -----------------------


@pytest.mark.parametrize(
    "counter_factory",
    [factory for _, factory in _CANDIDATE_COUNTERS],
    ids=[name for name, _ in _CANDIDATE_COUNTERS],
)
async def test_the_fit_lands_within_the_tolerance_for_every_modelled_counter(
    fake_server: FakeServer, counter_factory: Callable[[], InputTokenCounter]
) -> None:
    """+65 を作る数え方でも、文書ごとに組み立てた要求を数えて合わせ込めば狙いに入る。"""
    fake_server.set_response(token_stream_response(output_tokens=4))
    fake_server.set_input_token_counter(counter_factory())
    profile = make_profile(
        concurrency={"levels": [2], "rounds": 2, "max_tokens": 8, "input_tokens": 2000}
    )
    suite = ConcurrencySuite()

    async with suite_ctx(fake_server, profile=profile) as (ctx, _sink):
        records = await run_all(suite, ctx)

    measured = [record for record in records if not record.warmup]
    assert measured
    usages: list[int] = []
    for record in measured:
        usage = record.result.usage
        assert usage is not None
        usages.append(usage.total_input_tokens)
        assert TrialFlag.LENGTH_OFF_TARGET not in record.flags
    assert all(abs(usage - 2000) / 2000 <= 0.05 for usage in usages)
    assert abs(statistics.median(usages) - 2000) <= 20


async def test_the_document_is_fitted_by_counting_the_built_request(
    fake_server: FakeServer,
) -> None:
    """最初の候補は `target − 包み` の文章で、送る文章は訪れた候補のうち狙いに最も近い。"""
    counter = estimator_aligned_counter(fixed=150, chars_per_token=4.0)
    fake_server.set_response(token_stream_response(output_tokens=4))
    fake_server.set_input_token_counter(counter)
    profile = make_profile(
        concurrency={"levels": [2], "rounds": 2, "max_tokens": 8, "input_tokens": 2000}
    )
    suite = ConcurrencySuite()
    calls: list[tuple[dict[str, Any], int]] = []

    async def recording(request: MessagesRequest) -> TokenCount:
        body = build_request_body(request)
        counted = counter(body)
        calls.append((body, counted))
        return TokenCount(tokens=counted, method="count_tokens")

    async with suite_ctx(fake_server, profile=profile, count_input_tokens=recording) as (ctx, sink):
        records = await run_all(suite, ctx)

    frame_bodies = [counted for body, counted in calls if first_text(body) == _INSTRUCTION_SUFFIX]
    assert len(frame_bodies) == 1
    frame = frame_bodies[0]

    corpus = TemplateCorpus(profile.chars_per_token)
    by_ref = dict(zip(sink.refs, sink.bodies, strict=True))
    measured = [record for record in records if not record.warmup]
    assert measured
    for record in measured:
        assert record.round_id is not None
        assert record.stream_index is not None
        seed = trial_seed(
            profile.seed,
            "concurrency/c2",
            record.round_id,
            warmup=False,
            stream_index=record.stream_index,
        )
        first_expected = f"{corpus.prose('en', 2000 - frame, seed)}{_INSTRUCTION_SUFFIX}"
        assert any(first_text(body) == first_expected for body, _ in calls)

        short = corpus.prose("en", 20, seed)
        document_calls = [
            (body, counted) for body, counted in calls if first_text(body).startswith(short)
        ]
        assert len(document_calls) >= 1
        sent_body = by_ref[record.request_body_ref]
        sent_text = first_text(sent_body)
        sent_counted = [
            counted for body, counted in document_calls if first_text(body) == sent_text
        ]
        assert sent_counted
        closest = min(abs(counted - 2000) for _, counted in document_calls)
        assert abs(sent_counted[0] - 2000) == closest


async def test_the_fit_stops_when_the_next_candidate_is_below_one(
    fake_server: FakeServer,
) -> None:
    """次の候補が 1 未満なら数え続けず、訪れた最初の候補を送る。"""
    counter = estimator_aligned_counter(fixed=150, chars_per_token=4.0)
    fake_server.set_response(token_stream_response(output_tokens=4))
    fake_server.set_input_token_counter(counter)
    profile = make_profile(
        concurrency={"levels": [2], "rounds": 1, "max_tokens": 8, "input_tokens": 2000}
    )
    suite = ConcurrencySuite()
    calls = {"count": 0}

    async def huge_after_the_frame(request: MessagesRequest) -> TokenCount:
        calls["count"] += 1
        if calls["count"] == 1:
            return TokenCount(tokens=1, method="count_tokens")
        return TokenCount(tokens=50_000, method="count_tokens")

    async with suite_ctx(fake_server, profile=profile, count_input_tokens=huge_after_the_frame) as (
        ctx,
        sink,
    ):
        records = await run_all(suite, ctx)

    # 文書ごとに 1 回で止まる ((慣らし 1 + 本番 1) の 2 回 × 2 本 = 4 文書)
    assert calls["count"] == 1 + 4
    corpus = TemplateCorpus(profile.chars_per_token)
    by_ref = dict(zip(sink.refs, sink.bodies, strict=True))
    assert len(records) == 4
    for record in records:
        assert record.round_id is not None
        assert record.stream_index is not None
        seed = trial_seed(
            profile.seed,
            "concurrency/c2",
            record.round_id,
            warmup=record.warmup,
            stream_index=record.stream_index,
        )
        expected = f"{corpus.prose('en', 2000 - 1, seed)}{_INSTRUCTION_SUFFIX}"
        assert first_text(by_ref[record.request_body_ref]) == expected


@pytest.mark.parametrize(
    ("counted_script", "visited", "sent"),
    [
        # 収束しない数え方。4 回で打ち切り、訪れた 4 つのうち狙い (2000) に最も近い
        # 2 つ目 (誤差 20) を送る。最後の候補 (誤差 200) ではない
        pytest.param(
            (2400, 2020, 1700, 2200),
            (1999, 1599, 1579, 1879),
            1,
            id="stops-at-four-counts-and-sends-the-closest",
        ),
        # 誤差が同じ大きさで符号だけ違う数え方。2 つ目の候補の次は最初の候補に戻り、
        # 訪れた候補と同じなので止める。同点なので先の候補 (最初) を送る
        pytest.param(
            (2400, 1600),
            (1999, 1599),
            0,
            id="stops-on-a-visited-candidate-and-a-tie-goes-to-the-first",
        ),
    ],
)
async def test_the_fit_stops_at_the_limit_and_sends_the_closest_visited_candidate(
    fake_server: FakeServer,
    counted_script: tuple[int, ...],
    visited: tuple[int, ...],
    sent: int,
) -> None:
    """文書ごとの数える回数の上限と、送る候補の選び方 (最も近いもの、同点は先)。

    数えた長さは、包み (1 トークン) のあとは、文書ごとに `counted_script` を順に返す。
    最初の候補は `2000 − 1` で、次の候補は `今の候補 + (2000 − 数えた長さ)` になる。
    数える要求の回数が README の要求数の表 (文書ごとに最大 4 回) の根拠になる。
    """
    fake_server.set_response(token_stream_response(output_tokens=4))
    profile = make_profile(
        concurrency={"levels": [1], "rounds": 1, "max_tokens": 8, "input_tokens": 2000}
    )
    suite = ConcurrencySuite()
    calls: list[dict[str, Any]] = []

    async def scripted(request: MessagesRequest) -> TokenCount:
        body = build_request_body(request)
        index = len(calls)
        calls.append(body)
        if index == 0:
            return TokenCount(tokens=1, method="count_tokens")  # 包み
        return TokenCount(
            tokens=counted_script[(index - 1) % len(counted_script)], method="count_tokens"
        )

    async with suite_ctx(fake_server, profile=profile, count_input_tokens=scripted) as (ctx, sink):
        records = await run_all(suite, ctx)

    documents = len(records)  # 慣らし 1 + 本番 1、1 本
    assert documents == 2
    assert len(calls) == 1 + len(counted_script) * documents

    corpus = TemplateCorpus(profile.chars_per_token)
    by_ref = dict(zip(sink.refs, sink.bodies, strict=True))
    for position, record in enumerate(records):
        assert record.round_id is not None
        assert record.stream_index is not None
        seed = trial_seed(
            profile.seed,
            "concurrency/c1",
            record.round_id,
            warmup=record.warmup,
            stream_index=record.stream_index,
        )
        texts = [f"{corpus.prose('en', length, seed)}{_INSTRUCTION_SUFFIX}" for length in visited]
        start = 1 + len(counted_script) * position
        assert [first_text(body) for body in calls[start : start + len(visited)]] == texts
        assert first_text(by_ref[record.request_body_ref]) == texts[sent]


async def test_the_frame_is_measured_once_per_run_and_again_for_a_new_run(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(token_stream_response(output_tokens=4))
    profile = make_profile(
        concurrency={"levels": [1, 2], "rounds": 2, "max_tokens": 8, "input_tokens": 2000}
    )
    suite = ConcurrencySuite()
    documents = (1 + 2) * 1 + (1 + 2) * 2  # 水準ごとの (慣らし 1 + 本番 2) × 本数

    for run_id in (RUN_A, RUN_B):
        fake_server.reset()
        async with suite_ctx(fake_server, profile=profile, run_id=run_id) as (ctx, _sink):
            await run_all(suite, ctx)
        counted = fake_server.requests_for("/v1/messages/count_tokens")
        frame_requests = [
            record
            for record in counted
            if record.body is not None and first_text(record.body) == _INSTRUCTION_SUFFIX
        ]
        assert len(frame_requests) == 1
        assert len(counted) >= 1 + documents
        assert len(counted) <= 1 + 4 * documents


# --- C3: 合わせ込み用の識別子は、試行とも包みとも重ならない -----------------


async def test_the_fit_nonce_and_documents_are_stable_across_runs(
    fake_server: FakeServer,
) -> None:
    """合わせ込みの識別子は `run_id` を含まないので、計測ランをまたいで同じになる。"""
    fake_server.set_response(token_stream_response(output_tokens=4))
    profile = make_profile(
        concurrency={"levels": [2], "rounds": 2, "max_tokens": 8, "input_tokens": 2000}
    )
    suite = ConcurrencySuite()

    async def fit_fingerprint(run_id: str) -> tuple[set[str], set[str], set[str]]:
        fake_server.reset()
        async with suite_ctx(fake_server, profile=profile, run_id=run_id) as (ctx, _sink):
            await run_all(suite, ctx)
        bodies = [
            record.body
            for record in fake_server.requests_for("/v1/messages/count_tokens")
            if record.body is not None
        ]
        frame_lines = {
            system_first_line(body) for body in bodies if first_text(body) == _INSTRUCTION_SUFFIX
        }
        fit_bodies = [body for body in bodies if first_text(body) != _INSTRUCTION_SUFFIX]
        fit_lines = {system_first_line(body) for body in fit_bodies}
        fit_documents = {first_text(body) for body in fit_bodies}
        return frame_lines, fit_lines, fit_documents

    frame_a, fit_lines_a, docs_a = await fit_fingerprint(RUN_A)
    frame_b, fit_lines_b, docs_b = await fit_fingerprint(RUN_B)

    assert len(frame_a) == 1
    assert len(frame_b) == 1
    assert frame_a != frame_b
    assert len(fit_lines_a) == 1
    assert fit_lines_a == fit_lines_b
    assert docs_a == docs_b


async def test_the_fit_nonce_collides_with_neither_frame_nor_trials(
    fake_server: FakeServer,
) -> None:
    """口がない対象でも、合わせ込みの先頭は包みとも試行とも重ならない。"""
    fake_server.set_count_tokens_enabled(False)
    fake_server.set_response(token_stream_response(output_tokens=4))
    profile = make_profile(
        concurrency={"levels": [2], "rounds": 2, "max_tokens": 8, "input_tokens": 2000}
    )
    suite = ConcurrencySuite()

    async with suite_ctx(fake_server, profile=profile) as (ctx, _sink):
        await run_all(suite, ctx)

    requests = [
        record for record in fake_server.requests_for("/v1/messages") if record.body is not None
    ]
    counting = [
        record for record in requests if record.body is not None and record.body["max_tokens"] == 1
    ]
    trials = [
        record for record in requests if record.body is not None and record.body["max_tokens"] != 1
    ]
    assert len(trials) == 6  # (慣らし 1 + 本番 2) × 2 本

    frame = [record for record in counting if first_text(record.body or {}) == _INSTRUCTION_SUFFIX]
    fits = [record for record in counting if first_text(record.body or {}) != _INSTRUCTION_SUFFIX]
    assert len(frame) == 1
    assert frame[0].body is not None
    fit_lines = {system_first_line(record.body or {}) for record in fits}
    assert len(fit_lines) == 1
    fit_line = next(iter(fit_lines))
    frame_line = system_first_line(frame[0].body)
    trial_lines = {system_first_line(record.body or {}) for record in trials}
    assert fit_line != frame_line
    assert fit_line not in trial_lines
    # 包みの先頭も、どの試行の先頭とも重ならない (要件 12)
    assert frame_line not in trial_lines

    documents = (1 + 2) * 2
    assert len(counting) >= 1 + documents
    assert len(counting) <= 1 + 4 * documents


# --- C4: 狙いが包み以下なら飛ばす・数えられないものは伝播する ---------------


async def test_a_target_at_or_below_the_frame_is_skipped_with_a_reason_before_sending(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(token_stream_response(output_tokens=4))
    fake_server.set_input_token_counter(estimator_aligned_counter(fixed=50))
    profile = make_profile(
        concurrency={"levels": [2], "rounds": 2, "max_tokens": 8, "input_tokens": 10}
    )
    suite = ConcurrencySuite()

    async with suite_ctx(fake_server, profile=profile) as (ctx, _sink):
        with pytest.raises(ConditionAborted) as raised:
            await run_all(suite, ctx)

    skipped = raised.value.skipped
    assert skipped.suite is SuiteName.CONCURRENCY
    assert skipped.key == "concurrency/c2"
    assert "10" in skipped.reason
    assert str(frame_tokens_of(fake_server)) in skipped.reason
    assert fake_server.call_count("/v1/messages") == 0
    assert fake_server.call_count("/v1/messages/count_tokens") == 1


@pytest.mark.parametrize(("input_tokens", "skipped"), [(60, True), (61, False)])
async def test_the_frame_boundary_decides_between_skipping_and_sending(
    fake_server: FakeServer, input_tokens: int, skipped: bool
) -> None:
    """境界: 包みが 60 トークンなら、狙い 60 は飛ばし、狙い 61 (文章 1 トークン) は送る。"""
    fake_server.set_response(token_stream_response(output_tokens=4))
    fake_server.set_input_token_counter(lambda body: 60)
    profile = make_profile(
        concurrency={"levels": [2], "rounds": 1, "max_tokens": 8, "input_tokens": input_tokens}
    )
    suite = ConcurrencySuite()

    async with suite_ctx(fake_server, profile=profile) as (ctx, _sink):
        if skipped:
            with pytest.raises(ConditionAborted):
                await run_all(suite, ctx)
        else:
            assert await run_all(suite, ctx)

    if skipped:
        assert fake_server.call_count("/v1/messages") == 0
        assert fake_server.call_count("/v1/messages/count_tokens") == 1
    else:
        assert fake_server.call_count("/v1/messages") == 4  # (慣らし 1 + 本番 1) × 2 本
        counts = fake_server.call_count("/v1/messages/count_tokens")
        assert 1 + 4 <= counts <= 1 + 4 * 4


async def test_a_probe_error_is_not_turned_into_an_estimate(fake_server: FakeServer) -> None:
    fake_server.set_response(token_stream_response(output_tokens=4))

    async def failing(request: MessagesRequest) -> TokenCount:
        raise ProbeError("boom")

    suite = ConcurrencySuite()
    async with suite_ctx(fake_server, count_input_tokens=failing) as (ctx, _sink):
        with pytest.raises(ProbeError):
            await run_all(suite, ctx)

    assert fake_server.call_count("/v1/messages") == 0


async def test_a_document_probe_error_is_not_turned_into_an_estimate(
    fake_server: FakeServer,
) -> None:
    """包みは数えられても、文書の合わせ込みの計測が失敗したら、そのまま伝播する。"""
    fake_server.set_response(token_stream_response(output_tokens=4))
    calls = {"count": 0}

    async def failing_after_the_frame(request: MessagesRequest) -> TokenCount:
        calls["count"] += 1
        if calls["count"] > 1:
            raise ProbeError("boom")
        return TokenCount(tokens=1000, method="count_tokens")

    suite = ConcurrencySuite()
    async with suite_ctx(fake_server, count_input_tokens=failing_after_the_frame) as (ctx, _sink):
        with pytest.raises(ProbeError):
            await run_all(suite, ctx)

    assert fake_server.call_count("/v1/messages") == 0
