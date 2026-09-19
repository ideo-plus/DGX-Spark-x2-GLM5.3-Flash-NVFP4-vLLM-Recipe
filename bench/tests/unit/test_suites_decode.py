"""生成速度のまとまりの試験 (task 3.2: suites/decode)。

要求が絡む試験は、実際のソケット越しに偽のサーバーを相手に行う。時刻を判定
する試験は 1 つもない (このファイルが確かめるのは、条件の計画、送った要求の
指示の中身、慣らしと本番の分かれ方、共通の印の伝わり方であって、速さでは
ない)。助けの関数は `test_suites_base.py` から輸入せず、必要なものだけを
ここに書き写す (suite_common.md「Read first」)。
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from bench_harness.client.messages import HttpxMessagesClient
from bench_harness.suites.base import SuiteContext, make_suite_context
from bench_harness.suites.decode import DecodeSuite, decode_suite
from bench_harness.types import ConditionPlan, Profile, SuiteName, TargetDef, TrialFlag
from fake_server import (
    FakeServer,
    RecordedRequest,
    http_error_response,
    replace,
    text_response,
    token_stream_response,
)

RUN_ID = "20260920-000000-abcdef"

_JAPANESE_RE = re.compile(r"[぀-ヿ㐀-鿿]")

_ALL_KEYS: tuple[str, ...] = (
    "decode/code/en",
    "decode/code/ja",
    "decode/prose/en",
    "decode/prose/ja",
)


# --- 助け -------------------------------------------------------------------


def make_profile(**overrides: Any) -> Profile:
    """試験用の設定。サンプリングは既定と違う値にしてある。"""
    data: dict[str, Any] = {
        "name": "test",
        "seed": 7,
        "sampling": {"temperature": 0.3, "top_p": 0.9, "top_k": 40},
        "timeout": {"connect_s": 5.0, "first_event_s": 5.0, "idle_s": 5.0, "total_s": 20.0},
        "chars_per_token": {"prose_en": 8.0, "prose_ja": 2.0, "code": 3.0, "log": 3.0},
    }
    data.update(overrides)
    return Profile.model_validate(data)


def target_for(server: FakeServer, **overrides: Any) -> TargetDef:
    payload: dict[str, Any] = {"name": "fake", "base_url": server.base_url, "model": "fake-model"}
    payload.update(overrides)
    return TargetDef.model_validate(payload)


@asynccontextmanager
async def suite_ctx(
    server: FakeServer, *, profile: Profile | None = None, run_id: str = RUN_ID
) -> AsyncIterator[SuiteContext]:
    """偽のサーバーにつないだ `SuiteContext` を渡す。本文の保存先は参照だけ返す入れ物。"""
    async with HttpxMessagesClient(server.base_url) as client:
        yield make_suite_context(
            client=client,
            profile=profile if profile is not None else make_profile(),
            target=target_for(server),
            run_id=run_id,
            put_body=lambda _body: "ref",
        )


def _condition(ctx: SuiteContext, key: str) -> ConditionPlan:
    for plan in DecodeSuite().plan(ctx):
        if isinstance(plan, ConditionPlan) and plan.key == key:
            return plan
    raise AssertionError(f"decode: condition not planned: {key}")


def _user_text(body: dict[str, Any] | None) -> str:
    assert body is not None
    content = body["messages"][-1]["content"]
    text = content[-1]["text"]
    assert isinstance(text, str)
    return text


def _requests(server: FakeServer) -> list[RecordedRequest]:
    return server.requests_for("/v1/messages")


# --- 条件の計画 (2.1、2.2、2.7) ---------------------------------------------


async def test_plan_returns_the_four_conditions_with_the_profiles_numbers(
    fake_server: FakeServer,
) -> None:
    profile = make_profile(decode={"trials": 15, "warmup_trials": 3, "max_tokens": 777})
    async with suite_ctx(fake_server, profile=profile) as ctx:
        planned = decode_suite.plan(ctx)

    assert len(planned) == 4
    assert all(isinstance(plan, ConditionPlan) for plan in planned)
    keys = {plan.key for plan in planned if isinstance(plan, ConditionPlan)}
    assert keys == set(_ALL_KEYS)
    for plan in planned:
        assert isinstance(plan, ConditionPlan)
        assert plan.suite is SuiteName.DECODE
        assert plan.tier == "primary"
        assert plan.trials == 15
        assert plan.warmup_trials == 3
        assert plan.max_tokens == 777
        assert plan.concurrency == 1
        assert plan.target_input_tokens is None
        assert plan.sampling == profile.sampling


# --- 慣らしと本番 (2.5) ------------------------------------------------------


async def test_running_a_condition_sends_warmup_plus_trials_requests_warmup_first(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(text_response("ok"))
    profile = make_profile(decode={"trials": 10, "warmup_trials": 3, "max_tokens": 64})
    async with suite_ctx(fake_server, profile=profile) as ctx:
        cond = _condition(ctx, "decode/code/en")
        records = [record async for record in decode_suite.run_condition(ctx, cond)]

    assert len(records) == 13
    assert fake_server.call_count("/v1/messages") == 13
    assert [record.warmup for record in records] == [True, True, True] + [False] * 10
    assert [record.trial_index for record in records[:3]] == [0, 1, 2]
    assert [record.trial_index for record in records[3:]] == list(range(10))


# --- 同じ条件のすべての試行が同じ設定 (2.7) ---------------------------------


async def test_every_request_of_a_condition_has_identical_temperature_and_max_tokens(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(text_response("ok"))
    profile = make_profile(decode={"trials": 10, "warmup_trials": 2, "max_tokens": 77})
    async with suite_ctx(fake_server, profile=profile) as ctx:
        cond = _condition(ctx, "decode/prose/ja")
        records = [record async for record in decode_suite.run_condition(ctx, cond)]

    bodies = _requests(fake_server)
    assert len(bodies) == 12 == len(records)
    for recorded in bodies:
        body = recorded.body
        assert body is not None
        assert body["temperature"] == 0.3
        assert body["top_p"] == 0.9
        assert body["top_k"] == 40
        assert body["max_tokens"] == 77
        assert body["stream"] is True


# --- 指示の中身: 試行ごと、条件ごとの違い -----------------------------------


async def test_instructions_differ_between_trial_indices_and_between_conditions(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(text_response("ok"))
    profile = make_profile(decode={"trials": 10, "warmup_trials": 0, "max_tokens": 64})
    texts: dict[str, list[str]] = {}
    async with suite_ctx(fake_server, profile=profile) as ctx:
        for key in ("decode/code/en", "decode/prose/en"):
            fake_server.reset()
            cond = _condition(ctx, key)
            [record async for record in decode_suite.run_condition(ctx, cond)]
            texts[key] = [_user_text(recorded.body) for recorded in _requests(fake_server)]

    assert len(texts["decode/code/en"]) == 10
    assert len(set(texts["decode/code/en"])) == 10
    assert len(set(texts["decode/prose/en"])) == 10
    assert set(texts["decode/code/en"]).isdisjoint(texts["decode/prose/en"])


async def test_ja_instructions_contain_japanese_characters_and_en_do_not(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(text_response("ok"))
    profile = make_profile(decode={"trials": 10, "warmup_trials": 0, "max_tokens": 64})
    async with suite_ctx(fake_server, profile=profile) as ctx:
        for key in _ALL_KEYS:
            fake_server.reset()
            cond = _condition(ctx, key)
            [record async for record in decode_suite.run_condition(ctx, cond)]
            texts = [_user_text(recorded.body) for recorded in _requests(fake_server)]
            has_japanese = [bool(_JAPANESE_RE.search(text)) for text in texts]
            if key.endswith("/ja"):
                assert all(has_japanese), texts
            else:
                assert not any(has_japanese), texts


# --- run_id をまたいだ決定性 (base.py trial_seed の注) ----------------------


async def test_two_runs_with_different_run_id_send_identical_user_messages_but_different_system(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(text_response("ok"))
    profile = make_profile(decode={"trials": 10, "warmup_trials": 0, "max_tokens": 64})

    fake_server.reset()
    async with suite_ctx(fake_server, profile=profile, run_id="20260920-000000-aaaaaa") as ctx1:
        cond1 = _condition(ctx1, "decode/prose/en")
        [record async for record in decode_suite.run_condition(ctx1, cond1)]
    body1 = _requests(fake_server)[0].body  # 本番の試行 0 番 (慣らしなし)

    fake_server.reset()
    async with suite_ctx(fake_server, profile=profile, run_id="20260920-111111-bbbbbb") as ctx2:
        cond2 = _condition(ctx2, "decode/prose/en")
        [record async for record in decode_suite.run_condition(ctx2, cond2)]
    body2 = _requests(fake_server)[0].body  # 同じ番号の試行 0 番

    assert body1 is not None
    assert body2 is not None
    assert body1["messages"] == body2["messages"]
    assert body1["system"] != body2["system"]


# --- 共通の印: 早く終わった、出力が少なすぎる (2.6) --------------------------


async def test_end_turn_before_max_tokens_flags_short_output(fake_server: FakeServer) -> None:
    fake_server.set_response(
        replace(token_stream_response(output_tokens=20), stop_reason="end_turn")
    )
    profile = make_profile(decode={"trials": 10, "warmup_trials": 0, "max_tokens": 1024})
    async with suite_ctx(fake_server, profile=profile) as ctx:
        cond = _condition(ctx, "decode/code/en")
        records = [record async for record in decode_suite.run_condition(ctx, cond)]

    assert len(records) == 10
    assert all(TrialFlag.SHORT_OUTPUT in record.flags for record in records)


async def test_fewer_than_16_output_tokens_flags_too_few_output_tokens(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(token_stream_response(output_tokens=8, stop_reason="max_tokens"))
    profile = make_profile(decode={"trials": 10, "warmup_trials": 0, "max_tokens": 1024})
    async with suite_ctx(fake_server, profile=profile) as ctx:
        cond = _condition(ctx, "decode/prose/ja")
        records = [record async for record in decode_suite.run_condition(ctx, cond)]

    assert len(records) == 10
    assert all(TrialFlag.TOO_FEW_OUTPUT_TOKENS in record.flags for record in records)


# --- 要求の失敗でも記録して続ける (8.1、10.1) --------------------------------


async def test_failed_request_still_yields_a_record_and_the_condition_continues(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(http_error_response(500))
    profile = make_profile(decode={"trials": 10, "warmup_trials": 0, "max_tokens": 64})
    async with suite_ctx(fake_server, profile=profile) as ctx:
        cond = _condition(ctx, "decode/code/ja")
        records = [record async for record in decode_suite.run_condition(ctx, cond)]

    assert len(records) == 10
    assert all(record.result.error is not None for record in records)
    assert all(record.result.error.http_status == 500 for record in records)  # type: ignore[union-attr]


# --- 型紙の埋め込み忘れがない --------------------------------------------


async def test_no_leftover_template_slots_in_any_instruction(fake_server: FakeServer) -> None:
    fake_server.set_response(text_response("ok"))
    profile = make_profile(decode={"trials": 10, "warmup_trials": 2, "max_tokens": 64})
    async with suite_ctx(fake_server, profile=profile) as ctx:
        for key in _ALL_KEYS:
            fake_server.reset()
            cond = _condition(ctx, key)
            [record async for record in decode_suite.run_condition(ctx, cond)]
            for recorded in _requests(fake_server):
                body = recorded.body
                assert body is not None
                texts = [_user_text(body)]
                system = body.get("system")
                if isinstance(system, str):
                    texts.append(system)
                for text in texts:
                    assert "{" not in text, (key, text)
                    assert "}" not in text, (key, text)


# --- 話題の多様さ (最初の 20 本の本番の試行) --------------------------------


async def test_first_20_measured_trials_have_distinct_user_messages(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(text_response("ok"))
    profile = make_profile(decode={"trials": 20, "warmup_trials": 0, "max_tokens": 64})
    async with suite_ctx(fake_server, profile=profile) as ctx:
        for key in _ALL_KEYS:
            fake_server.reset()
            cond = _condition(ctx, key)
            [record async for record in decode_suite.run_condition(ctx, cond)]
            texts = [_user_text(recorded.body) for recorded in _requests(fake_server)]
            assert len(texts) == 20
            assert len(set(texts)) == 20, (key, texts)
            # 文中の架空の管理番号 (4 桁) を除いても、題材そのものが互いに違うこと。
            # 番号だけが違う同じ題材の繰り返しでは、この条件の代表にならない
            without_numbers = {re.sub(r"\d{4}", "", text) for text in texts}
            assert len(without_numbers) == 20, (key, sorted(without_numbers)[:3])
