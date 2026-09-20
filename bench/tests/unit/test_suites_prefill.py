"""入力の処理と最初のトークンまでの時間のまとまりの試験 (task 3.3: suites/prefill)。

小さな長さ (2000 / 4000 トークン、8k/32k/128k の代わり) の `Profile` を使う。
`chars_per_token` は、内容の種類によらず同じ比 (4.0) にそろえ、偽のサーバーの
`set_chars_per_token` も同じ比に合わせる。これで、DOCUMENT の狙いの計算
(prefill.py の docstring を参照) と、偽のサーバーがそこから逆算するトークン数
が、実際のトークン数の許容の幅 (`Profile.length_tolerance`) の中でそろう。

時刻を判定する試験は 1 つもない (条件の計画、先頭の行、DOCUMENT と QUESTION の
中身、共通の印、トークン数の許容を確かめる試験だけ)。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from pydantic import JsonValue

from bench_harness.client.messages import HttpxMessagesClient
from bench_harness.suites.base import ConditionAborted, SuiteContext, make_suite_context
from bench_harness.suites.prefill import (
    SUITE,
    PrefillSuite,
    _document_for_length,
    _question_for,
    _short_warmup_document,
)
from bench_harness.types import (
    ConditionPlan,
    Profile,
    SkippedCondition,
    TargetDef,
    TrialFlag,
    TrialRecord,
)
from fake_server import FakeServer, RecordedRequest, text_response

RUN_A = "20260920-000000-aaaaaa"
RUN_B = "20260920-111111-bbbbbb"

# 8k/32k/128k の代わりに使う、小さな長さ。プレフィックスキャッシュの偽の実装
# (16 トークン刻みで切り下げる) が意味のある値を返せるだけの余裕を持たせてある
_LENGTHS: tuple[int, int] = (2000, 4000)


# --- 助け (test_suites_base.py から、この試験に要る分だけを写す) -----------


class BodySink:
    """`RunStore.put_body` の代わり。保存された本文をそのまま覚えておく。"""

    def __init__(self) -> None:
        self.bodies: list[dict[str, JsonValue]] = []

    def put(self, body: dict[str, JsonValue]) -> str:
        canonical = json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def make_profile(**overrides: Any) -> Profile:
    """試験用の小さな設定。既定の長さは 2000 / 4000。"""
    data: dict[str, Any] = {
        "name": "test",
        "seed": 11,
        "sampling": {"temperature": 0.0},
        "timeout": {"connect_s": 5.0, "first_event_s": 5.0, "idle_s": 5.0, "total_s": 20.0},
        "length_tolerance": 0.05,
        "chars_per_token": {"prose_en": 4.0, "prose_ja": 4.0, "code": 4.0, "log": 4.0},
        "prefill": {
            "trials": 2,
            "warmup_trials": 1,
            "max_tokens": 16,
            "target_input_tokens": list(_LENGTHS),
        },
    }
    data.update(overrides)
    return Profile.model_validate(data)


def target_for(server: FakeServer, **overrides: Any) -> TargetDef:
    payload: dict[str, Any] = {"name": "fake", "base_url": server.base_url, "model": "fake-model"}
    payload.update(overrides)
    return TargetDef.model_validate(payload)


@asynccontextmanager
async def suite_ctx(
    server: FakeServer,
    *,
    profile: Profile | None = None,
    context_limit: int | None = None,
    run_id: str = RUN_A,
) -> AsyncIterator[SuiteContext]:
    """偽のサーバーにつないだ `SuiteContext`。"""
    async with HttpxMessagesClient(server.base_url) as client:
        yield make_suite_context(
            client=client,
            profile=profile if profile is not None else make_profile(),
            target=target_for(server),
            run_id=run_id,
            put_body=BodySink().put,
            context_limit=context_limit,
        )


def planned_condition(ctx: SuiteContext, key: str) -> ConditionPlan:
    """`PrefillSuite.plan` が返す条件から、鍵で 1 つを選ぶ (飛ばされていないことを確かめる)。"""
    plan = next(p for p in SUITE.plan(ctx) if p.key == key)
    assert isinstance(plan, ConditionPlan)
    return plan


async def run_all(ctx: SuiteContext, cond: ConditionPlan) -> list[TrialRecord]:
    return [record async for record in SUITE.run_condition(ctx, cond)]


def message_text(body: dict[str, JsonValue] | None) -> str:
    assert body is not None
    messages = body["messages"]
    assert isinstance(messages, list)
    message = messages[0]
    assert isinstance(message, dict)
    content = message["content"]
    assert isinstance(content, list)
    block = content[0]
    assert isinstance(block, dict)
    text = block["text"]
    assert isinstance(text, str)
    return text


def sent_requests(server: FakeServer) -> list[RecordedRequest]:
    return server.requests_for("/v1/messages")


# --- 計画: 鍵、tier、試行と慣らしの回数 (3.1) -------------------------------


async def test_plan_produces_cold_and_warm_conditions_with_profile_numbers(
    fake_server: FakeServer,
) -> None:
    profile = make_profile()
    async with suite_ctx(fake_server, profile=profile) as ctx:
        planned = SUITE.plan(ctx)

    assert all(isinstance(plan, ConditionPlan) for plan in planned)
    by_key = {plan.key: plan for plan in planned if isinstance(plan, ConditionPlan)}
    assert set(by_key) == {
        "prefill/cold/2k",
        "prefill/warm/2k",
        "prefill/cold/4k",
        "prefill/warm/4k",
    }

    for length, label in ((2000, "2k"), (4000, "4k")):
        cold = by_key[f"prefill/cold/{label}"]
        warm = by_key[f"prefill/warm/{label}"]
        assert (cold.tier, warm.tier) == ("primary", "primary")
        assert cold.trials == warm.trials == profile.prefill.trials
        assert cold.max_tokens == warm.max_tokens == profile.prefill.max_tokens
        assert cold.target_input_tokens == warm.target_input_tokens == length
        assert cold.warmup_trials == 1  # DEFAULT_CONNECTION_WARMUP_TRIALS
        assert warm.warmup_trials == max(profile.prefill.warmup_trials, 1)


async def test_warm_warmup_trials_is_at_least_one_even_if_the_profile_says_zero(
    fake_server: FakeServer,
) -> None:
    profile = make_profile(
        prefill={"trials": 2, "warmup_trials": 0, "max_tokens": 16, "target_input_tokens": [2000]}
    )
    async with suite_ctx(fake_server, profile=profile) as ctx:
        planned = SUITE.plan(ctx)
    warm = next(p for p in planned if p.key == "prefill/warm/2k")
    assert isinstance(warm, ConditionPlan)
    assert warm.warmup_trials == 1


async def test_plan_skips_the_longer_length_beyond_an_advertised_limit(
    fake_server: FakeServer,
) -> None:
    """tasks.md 3.3 の完了の状態: 短いほうは計画され、長いほうは理由つきで飛ぶ。"""
    profile = make_profile()
    # 2000 + 16 <= 3000 (収まる)。4000 + 16 > 3000 (収まらない)
    async with suite_ctx(fake_server, profile=profile, context_limit=3000) as ctx:
        planned = SUITE.plan(ctx)

    by_key = {plan.key: plan for plan in planned}
    assert isinstance(by_key["prefill/cold/2k"], ConditionPlan)
    assert isinstance(by_key["prefill/warm/2k"], ConditionPlan)
    skipped_cold = by_key["prefill/cold/4k"]
    skipped_warm = by_key["prefill/warm/4k"]
    assert isinstance(skipped_cold, SkippedCondition)
    assert isinstance(skipped_warm, SkippedCondition)
    assert "4000" in skipped_cold.reason and "3000" in skipped_cold.reason
    assert "4000" in skipped_warm.reason and "3000" in skipped_warm.reason


# --- 実行中に上限に当たったとき (3.6) --------------------------------------


async def test_run_condition_records_the_400_then_aborts_when_the_limit_is_not_advertised(
    fake_server: FakeServer,
) -> None:
    """上限を申告しないが守るサーバー: レコードは残り、条件は打ち切られる (base.py の契約)。"""
    fake_server.set_chars_per_token(4.0)
    fake_server.set_context_limit(100, advertise=False)  # 慣らしの短い DOCUMENT すら超える
    fake_server.set_response(text_response("ok"))
    profile = make_profile(
        prefill={"trials": 2, "warmup_trials": 1, "max_tokens": 16, "target_input_tokens": [2000]}
    )
    async with suite_ctx(fake_server, profile=profile, context_limit=None) as ctx:
        cond = planned_condition(ctx, "prefill/cold/2k")
        records: list[TrialRecord] = []
        with pytest.raises(ConditionAborted) as raised:
            async for record in SUITE.run_condition(ctx, cond):
                records.append(record)

    assert len(records) == 1  # 8.1: 要求そのものは残す
    assert records[0].result.error is not None
    assert records[0].result.error.http_status == 400
    assert raised.value.skipped.key == "prefill/cold/2k"


# --- cold: 先頭の行が重ならない (3.4) ---------------------------------------


async def test_cold_first_lines_are_pairwise_distinct_including_warmup_and_across_runs(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(text_response("ok"))
    profile = make_profile(
        prefill={"trials": 2, "warmup_trials": 1, "max_tokens": 16, "target_input_tokens": [2000]}
    )

    fake_server.reset()
    async with suite_ctx(fake_server, profile=profile, run_id=RUN_A) as ctx:
        records_a = await run_all(ctx, planned_condition(ctx, "prefill/cold/2k"))
    systems_a = [req.body["system"] for req in sent_requests(fake_server) if req.body is not None]

    fake_server.reset()
    async with suite_ctx(fake_server, profile=profile, run_id=RUN_B) as ctx:
        records_b = await run_all(ctx, planned_condition(ctx, "prefill/cold/2k"))
    systems_b = [req.body["system"] for req in sent_requests(fake_server) if req.body is not None]

    assert len(records_a) == len(records_b) == 3  # 1 慣らし + 2 本番
    assert len(systems_a) == len(systems_b) == 3
    assert len(set(systems_a)) == 3  # 同じ run の中で重ならない
    assert set(systems_a).isdisjoint(systems_b)  # run をまたいでも重ならない


# --- DOCUMENT: cold と warm で同じ、試行をまたいでも同じ (3.1、3.5) --------


async def test_document_is_identical_across_trials_and_between_cold_and_warm(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(text_response("ok"))
    profile = make_profile(
        prefill={"trials": 2, "warmup_trials": 1, "max_tokens": 16, "target_input_tokens": [2000]}
    )

    async with suite_ctx(fake_server, profile=profile) as ctx:
        document = _document_for_length(ctx, 2000)
        cold_cond = planned_condition(ctx, "prefill/cold/2k")
        warm_cond = planned_condition(ctx, "prefill/warm/2k")

        fake_server.reset()
        cold_records = await run_all(ctx, cold_cond)
        cold_bodies = [req.body for req in sent_requests(fake_server)]

        fake_server.reset()
        await run_all(ctx, warm_cond)
        warm_bodies = [req.body for req in sent_requests(fake_server)]

        # 呼び出しのたびに同じ (純粋な関数)
        assert _document_for_length(ctx, 2000) == document

    for record, body in zip(cold_records, cold_bodies, strict=True):
        if record.warmup:
            continue
        text = message_text(body)
        assert text.startswith(document)
    for body in warm_bodies:
        text = message_text(body)
        assert text.startswith(document)


async def test_cold_warmup_uses_a_short_document_and_gets_flagged(fake_server: FakeServer) -> None:
    fake_server.set_response(text_response("ok"))
    profile = make_profile(
        prefill={"trials": 1, "warmup_trials": 1, "max_tokens": 16, "target_input_tokens": [2000]}
    )

    async with suite_ctx(fake_server, profile=profile) as ctx:
        short_document = _short_warmup_document(ctx)
        cond = planned_condition(ctx, "prefill/cold/2k")
        records = await run_all(ctx, cond)
        bodies = [req.body for req in sent_requests(fake_server)]

    warmup_record, warmup_body = records[0], bodies[0]
    assert warmup_record.warmup is True
    text = message_text(warmup_body)
    assert text.startswith(short_document)
    assert not text.startswith(_document_for_length(ctx, 2000)[: len(short_document) + 1])
    # 慣らしのトークン数は条件の狙い (2000) と大きく外れるので印が付く。実害は
    # ない (慣らしは集計から外れる。prefill.py の docstring を参照)
    assert TrialFlag.LENGTH_OFF_TARGET in warmup_record.flags


# --- warm: 先頭の行と DOCUMENT が同じ、末尾の QUESTION だけ違う (3.5) ------


async def test_warm_requests_share_the_header_and_document_only_the_tail_differs(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(text_response("ok"))
    profile = make_profile(
        prefill={"trials": 2, "warmup_trials": 1, "max_tokens": 16, "target_input_tokens": [2000]}
    )

    async with suite_ctx(fake_server, profile=profile) as ctx:
        cond = planned_condition(ctx, "prefill/warm/2k")
        document = _document_for_length(ctx, 2000)
        records = await run_all(ctx, cond)
        bodies = [req.body for req in sent_requests(fake_server)]

    systems = {body["system"] for body in bodies if body is not None}
    assert len(systems) == 1  # 先頭の行はどの要求でも同じ

    for record, body in zip(records, bodies, strict=True):
        text = message_text(body)
        expected_question = _question_for(
            ctx, cond, trial_index=record.trial_index, warmup=record.warmup
        )
        assert text == f"{document}\n\n{expected_question}"


# --- プレフィックスキャッシュ: warm は当たり、cold は当たらない (3.4、3.5) -


async def test_prefix_cache_hits_for_warm_measured_trials_and_zero_for_cold(
    fake_server: FakeServer,
) -> None:
    fake_server.set_chars_per_token(4.0)
    fake_server.set_prefix_cache(True)
    fake_server.set_response(text_response("ok"))
    profile = make_profile(
        prefill={"trials": 2, "warmup_trials": 1, "max_tokens": 16, "target_input_tokens": [2000]}
    )

    async with suite_ctx(fake_server, profile=profile) as ctx:
        fake_server.reset()
        cold_records = await run_all(ctx, planned_condition(ctx, "prefill/cold/2k"))
        fake_server.reset()
        warm_records = await run_all(ctx, planned_condition(ctx, "prefill/warm/2k"))

    cold_measured = [r for r in cold_records if not r.warmup]
    warm_measured = [r for r in warm_records if not r.warmup]
    assert len(cold_measured) == len(warm_measured) == 2

    for record in cold_measured:
        assert record.result.usage is not None
        assert (record.result.usage.cache_read_input_tokens or 0) == 0

    for record in warm_measured:
        assert record.result.usage is not None
        assert (record.result.usage.cache_read_input_tokens or 0) > 0


# --- 狙った長さに近いこと、印が付かないこと (3.2、3.3) ---------------------


async def test_measured_trials_are_not_length_off_target_when_chars_per_token_matches(
    fake_server: FakeServer,
) -> None:
    fake_server.set_chars_per_token(4.0)  # profile の chars_per_token と一致させる
    fake_server.set_response(text_response("ok"))
    profile = make_profile(
        prefill={
            "trials": 2,
            "warmup_trials": 1,
            "max_tokens": 16,
            "target_input_tokens": [2000, 4000],
        }
    )

    async with suite_ctx(fake_server, profile=profile) as ctx:
        cold_2k = await run_all(ctx, planned_condition(ctx, "prefill/cold/2k"))
        fake_server.reset()
        warm_4k = await run_all(ctx, planned_condition(ctx, "prefill/warm/4k"))

    for length, records in ((2000, cold_2k), (4000, warm_4k)):
        for record in records:
            if record.warmup:
                continue
            assert record.result.usage is not None
            actual = record.result.usage.total_input_tokens
            tolerance = profile.length_tolerance
            assert abs(actual - length) / length <= tolerance
            assert TrialFlag.LENGTH_OFF_TARGET not in record.flags


# --- max_tokens は条件のとおり (2.7) ----------------------------------------


async def test_max_tokens_matches_the_profile_in_every_recorded_body(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(text_response("ok"))
    profile = make_profile(
        prefill={"trials": 2, "warmup_trials": 1, "max_tokens": 16, "target_input_tokens": [2000]}
    )

    async with suite_ctx(fake_server, profile=profile) as ctx:
        await run_all(ctx, planned_condition(ctx, "prefill/cold/2k"))
        await run_all(ctx, planned_condition(ctx, "prefill/warm/2k"))

    bodies = [req.body for req in sent_requests(fake_server)]
    assert len(bodies) == 6  # (1 慣らし + 2 本番) × 2 条件
    for body in bodies:
        assert body is not None
        assert body["max_tokens"] == profile.prefill.max_tokens


# --- 同じ試行の番号は、計測ランをまたいでも同じ QUESTION (3.1 の注) --------


async def test_measured_question_is_stable_across_runs(fake_server: FakeServer) -> None:
    fake_server.set_response(text_response("ok"))
    profile = make_profile(
        prefill={"trials": 2, "warmup_trials": 1, "max_tokens": 16, "target_input_tokens": [2000]}
    )

    fake_server.reset()
    async with suite_ctx(fake_server, profile=profile, run_id=RUN_A) as ctx:
        records_a = await run_all(ctx, planned_condition(ctx, "prefill/warm/2k"))
    texts_a = [message_text(req.body) for req in sent_requests(fake_server)]

    fake_server.reset()
    async with suite_ctx(fake_server, profile=profile, run_id=RUN_B) as ctx:
        records_b = await run_all(ctx, planned_condition(ctx, "prefill/warm/2k"))
    texts_b = [message_text(req.body) for req in sent_requests(fake_server)]

    # warm の先頭行も DOCUMENT も QUESTION も run_id に依存しないので、本文が丸ごと一致する
    assert [(r.trial_index, r.warmup) for r in records_a] == [
        (r.trial_index, r.warmup) for r in records_b
    ]
    assert texts_a == texts_b


def test_prefill_suite_exposes_a_module_level_instance() -> None:
    assert isinstance(SUITE, PrefillSuite)
    assert SUITE.name.value == "prefill"
