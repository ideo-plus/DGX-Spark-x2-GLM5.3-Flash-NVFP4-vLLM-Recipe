"""共有の型の試験。

確かめること:

- すべての型が JSON に書き出せて読み戻せる (タスク 1.2 の完了の状態)
- 入力の全長が、キャッシュの内訳の有無にかかわらず正しい (3.2)
- 要約の型のスキーマに、送った内容と応答の本文の項目がない (8.3)
- 対象サーバーの定義に、認証の情報の値を入れる項目がない (1.8)
- 設定の下限と、時刻の順序の不変の決まりが、検証で弾かれる
- `types.py` が `bench_harness` のほかの module を読み込まない (依存の向き)
"""

from __future__ import annotations

import ast
import inspect
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, HttpUrl, ValidationError

from bench_harness import types as t

# --- 例の組み立て -------------------------------------------------------

AT = datetime(2026, 9, 19, 12, 0, 0, tzinfo=UTC)

SAMPLING = t.Sampling(temperature=0.0, top_p=0.9, top_k=40, thinking="server_default")
TIMEOUT = t.TimeoutPolicy(connect_s=5.0, first_event_s=120.0, idle_s=60.0, total_s=900.0)
TARGET = t.TargetDef(
    name="candidate-d",
    base_url=HttpUrl("http://10.0.1.60:8001"),
    model="glm-5.3-flash",
    notes="比較の基準",
    api_key_env="BENCH_OWN_API_KEY",
    metrics_url=HttpUrl("http://10.0.1.60:8001/metrics"),
    max_context_tokens=131072,
    metric_map={t.LogicalMetric.KV_USAGE: "vllm:kv_cache_usage_perc"},
    tool_markup_markers=["<tool_call>", "</tool_call>"],
)
PROFILE = t.Profile(name="quick", seed=7, sampling=SAMPLING, timeout=TIMEOUT)

TOOL = t.ToolDef(
    name="read_file",
    description="ファイルを読む",
    input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
)
TEXT_BLOCK = t.TextBlockParam(text="README.md を読んで")
TOOL_USE_BLOCK = t.ToolUseBlockParam(id="toolu_1", name="read_file", input={"path": "README.md"})
TOOL_RESULT_BLOCK = t.ToolResultBlockParam(tool_use_id="toolu_1", content="# README")
MESSAGE = t.InputMessage(role="user", content=[TEXT_BLOCK])
REQUEST = t.MessagesRequest(
    model="glm-5.3-flash",
    max_tokens=1024,
    messages=[MESSAGE, t.InputMessage(role="assistant", content=[TOOL_USE_BLOCK])],
    system="あなたは計測の対象です",
    tools=[TOOL],
    temperature=0.0,
    top_p=0.9,
    top_k=40,
    extra={"chat_template_kwargs": {"enable_thinking": False}},
)

USAGE = t.Usage(
    input_tokens=1000,
    output_tokens=64,
    cache_read_input_tokens=200,
    cache_creation_input_tokens=50,
)
TIMING = t.StreamTiming(
    sent_at_utc=AT,
    sent_at_ns=1_000,
    message_start_ns=2_000,
    first_token_ns=3_000,
    first_text_ns=3_500,
    last_token_ns=9_000,
    end_ns=10_000,
    event_count=42,
)
CONTENT_BLOCK = t.ContentBlock(
    type="tool_use",
    tool_name="read_file",
    tool_input_raw='{"path": "README.md"}',
    tool_input={"path": "README.md"},
)
STREAM_RESULT = t.StreamResult(
    timing=TIMING,
    usage=USAGE,
    stop_reason="end_turn",
    blocks=[t.ContentBlock(type="text", text="はい"), CONTENT_BLOCK],
    server_model="glm-5.3-flash",
)

SNAPSHOT = t.MetricSnapshot(
    taken_at_utc=AT,
    raw_text="vllm:num_requests_running 0.0\n",
    values={t.LogicalMetric.RUNNING_REQUESTS: 0.0},
    missing=[t.LogicalMetric.SPEC_DRAFTS],
)
DERIVED = t.DerivedMetrics(
    spec_acceptance_rate=0.62,
    mean_acceptance_length=1.62,
    decode_steps=512.0,
    tokens_per_step=1.62,
    prefix_cache_hit_rate=0.0,
    kv_usage_peak=0.31,
    preemptions=0.0,
    missing=[t.LogicalMetric.PREEMPTIONS],
)

TOOL_TASK = t.ToolTask(
    prompt="README.md を読んでください",
    tools=[TOOL],
    expected_tool="read_file",
    expected_input={"path": "README.md"},
)
SANDBOX_RESULT = t.SandboxResult(passed=True, timed_out=False, exit_code=0, stderr_tail="")
DATASET = t.DatasetRef(
    name="HumanEval+",
    version="v0.1.10",
    source_url="https://github.com/evalplus/evalplus",
    license="Apache-2.0",
    scoring_method="pass@1 (greedy)",
    sha256="0" * 64,
)

CONDITION = t.ConditionPlan(
    suite=t.SuiteName.DECODE,
    key="decode/code/en",
    tier="primary",
    trials=10,
    warmup_trials=2,
    sampling=SAMPLING,
    max_tokens=1024,
)
TRIAL = t.TrialRecord(
    run_id="20260919T120000Z-candidate-d-ab12cd",
    suite=t.SuiteName.DECODE,
    condition="decode/code/en",
    tier="primary",
    trial_index=3,
    warmup=False,
    round_id=None,
    stream_index=None,
    request_body_ref="f" * 64,
    result=STREAM_RESULT,
    flags=[t.TrialFlag.SHORT_OUTPUT],
    verdict=t.ToolCallVerdict(outcome=t.ToolCallOutcome.CORRECT, detail=""),
    target_input_tokens=8000,
)
MANIFEST = t.RunManifest(
    run_id="20260919T120000Z-candidate-d-ab12cd",
    status=t.RunStatus.COMPLETED,
    target=TARGET,
    server_model="glm-5.3-flash",
    server_version="0.11.1",
    started_at=AT,
    finished_at=AT,
    suites=[t.SuiteName.DECODE, t.SuiteName.PREFILL],
    profile_name="quick",
    profile=PROFILE,
    harness_version="0.1.0+g1234567.dirty",
    context_limit=131072,
    warnings=["始める前に実行中の要求が 1 本あった"],
    skipped=[
        t.SkippedCondition(
            suite=t.SuiteName.PREFILL, key="prefill/cold/128k", reason="context_limit"
        )
    ],
    datasets=[DATASET],
)

DESCRIBE = t.Describe(n=10, mean=42.0, median=41.5, min=38.0, max=45.0, stdev=2.0, iqr=3.0, cv=0.05)
PROPORTION = t.ProportionStat(
    numerator=0,
    denominator=299,
    rate=0.0,
    ci95_low=0.0,
    ci95_high=0.0122,
    upper95_one_sided=0.00996,
)
METRIC_RESULT = t.MetricResult(
    condition="decode/code/en",
    metric="decode_tps",
    tier="primary",
    continuous=DESCRIBE,
    proportion=None,
    failures=1,
    flags=[t.MetricFlag.SHORT_OUTPUTS],
    flag_counts={"SHORT_OUTPUT": 2},
)
STAGE = t.AgentStageResult(
    stage_key="agent/stage/020k",
    target_input_tokens=20000,
    actual_input_tokens=DESCRIBE,
    trials=50,
    outcome_counts={t.ToolCallOutcome.CORRECT: 50},
    request_failures=0,
    break_rate=PROPORTION,
    verdict=t.ThresholdVerdict.UNDETERMINED,
)
AGENT_SUMMARY = t.AgentSummary(
    threshold=0.01,
    stages=[STAGE],
    first_exceeded_tokens=None,
    reached_tokens=120000,
    stopped_reason=None,
)
SUMMARY = t.Summary(
    conditions=MANIFEST,
    incomplete=False,
    results=[METRIC_RESULT],
    agent=AGENT_SUMMARY,
    server_metrics={"decode/code/en": DERIVED},
    datasets=[DATASET],
)
DIFF = t.DiffVerdict(
    verdict="within",
    paired=True,
    median_diff=0.3,
    relative_diff=0.007,
    ci95_low=-0.9,
    ci95_high=1.4,
    tolerance=0.02,
    n=10,
)
COMPARISON_ROW = t.ComparisonRow(
    condition="decode/code/en",
    metric="decode_tps",
    tier="primary",
    value_a=42.0,
    value_b=41.7,
    diff=-0.3,
    relative_diff=-0.007,
    verdict=DIFF,
    proportion_verdict=None,
)

EXAMPLES: dict[str, BaseModel] = {
    "Sampling": SAMPLING,
    "TimeoutPolicy": TIMEOUT,
    "TargetDef": TARGET,
    "OutputSanity": t.OutputSanity(),
    "SandboxSettings": t.SandboxSettings(
        runtime="podman", image="python:3.12-slim", image_digest="sha256:" + "a" * 64
    ),
    "DecodeSettings": t.DecodeSettings(),
    "PrefillSettings": t.PrefillSettings(),
    "ConcurrencySettings": t.ConcurrencySettings(),
    "QualitySettings": t.QualitySettings(),
    "AgentSettings": t.AgentSettings(),
    "Profile": PROFILE,
    "ToolDef": TOOL,
    "TextBlockParam": TEXT_BLOCK,
    "ToolUseBlockParam": TOOL_USE_BLOCK,
    "ToolResultBlockParam": TOOL_RESULT_BLOCK,
    "InputMessage": MESSAGE,
    "MessagesRequest": REQUEST,
    "Usage": USAGE,
    "StreamTiming": TIMING,
    "ContentBlock": CONTENT_BLOCK,
    "RequestError": t.RequestError(kind="timeout_idle", http_status=None, message="idle 60s"),
    "StreamResult": STREAM_RESULT,
    "PreflightOk": t.PreflightOk(
        server_model="glm-5.3-flash",
        server_version="0.11.1",
        context_limit=131072,
        running_requests=0,
    ),
    "PreflightFailure": t.PreflightFailure(unmet="no_usage", detail="usage が応答にない"),
    "MetricSnapshot": SNAPSHOT,
    "MetricsUnavailable": t.MetricsUnavailable(reason="/metrics が 404"),
    "DerivedMetrics": DERIVED,
    "ToolTask": TOOL_TASK,
    "ToolCallVerdict": t.ToolCallVerdict(
        outcome=t.ToolCallOutcome.WRONG_CALL, detail="extra_calls=2"
    ),
    "QualityVerdict": t.QualityVerdict(
        task="code",
        outcome=t.QualityOutcome.CORRECT,
        detail="HumanEval/0",
        sandbox=SANDBOX_RESULT,
    ),
    "NeedleCase": t.NeedleCase(
        haystack="干し草" * 100,
        question="合言葉は何ですか",
        answer="紫のカワウソ",
        target_tokens=8000,
        depth_pct=50,
        index=0,
        seed=7,
        inserted_char_offset=150,
    ),
    "CodeProblem": t.CodeProblem(
        task_id="HumanEval/0",
        prompt="def has_close_elements(...):",
        entry_point="has_close_elements",
        test="def check(candidate): ...",
        canonical_solution=None,
    ),
    "DatasetRef": DATASET,
    "ConversationPrefix": t.ConversationPrefix(
        system="あなたは takt の作業者です",
        tools=[TOOL],
        messages=[MESSAGE, t.InputMessage(role="assistant", content=[TOOL_USE_BLOCK])],
        approx_tokens=20000,
        conversation_seed=3,
    ),
    "SandboxResult": SANDBOX_RESULT,
    "SandboxUnavailable": t.SandboxUnavailable(reason="podman も docker も見つからない"),
    "SkippedCondition": t.SkippedCondition(
        suite=t.SuiteName.AGENT, key="agent/stage/120k", reason="context_limit"
    ),
    "ConditionPlan": CONDITION,
    "TrialRecord": TRIAL,
    "RunManifest": MANIFEST,
    "RunRequest": t.RunRequest(
        target_name="candidate-d",
        suites=[t.SuiteName.DECODE],
        profile_name="quick",
        trials_override={"decode": 20},
    ),
    "RunOutcome": t.RunOutcome(
        run_id="20260919T120000Z-candidate-d-ab12cd",
        status=t.RunStatus.COMPLETED,
        run_dir=Path("/tmp/results/20260919T120000Z-candidate-d-ab12cd"),
        exit_code=0,
    ),
    "Describe": DESCRIBE,
    "ProportionStat": PROPORTION,
    "DiffVerdict": DIFF,
    "MetricResult": METRIC_RESULT,
    "AgentStageResult": STAGE,
    "AgentSummary": AGENT_SUMMARY,
    "Summary": SUMMARY,
    "ComparisonRow": COMPARISON_ROW,
    "RepeatabilityConclusion": t.RepeatabilityConclusion(
        all_within=False, outside=["decode/code/en"]
    ),
    "ComparisonReport": t.ComparisonReport(
        run_a="20260919T120000Z-candidate-d-ab12cd",
        run_b="20260919T130000Z-candidate-d-ef34gh",
        warnings=["profile_name が違う"],
        rows=[COMPARISON_ROW],
        excluded=["agent/stage/120k"],
        repeatability=t.RepeatabilityConclusion(all_within=False, outside=["decode/code/en"]),
    ),
}


def _public_models() -> dict[str, type[BaseModel]]:
    """`types.py` が定義する、公開の pydantic の型をすべて集める。"""
    found: dict[str, type[BaseModel]] = {}
    for name, obj in vars(t).items():
        if name.startswith("_") or not isinstance(obj, type):
            continue
        if issubclass(obj, BaseModel) and obj.__module__ == t.__name__:
            found[name] = obj
    return found


# --- JSON の往復 --------------------------------------------------------


def test_every_public_model_has_an_example() -> None:
    assert set(_public_models()) == set(EXAMPLES)


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_model_round_trips_through_json(name: str) -> None:
    model = EXAMPLES[name]
    restored = type(model).model_validate_json(model.model_dump_json())
    assert restored == model


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_model_round_trips_through_python_objects(name: str) -> None:
    model = EXAMPLES[name]
    restored = type(model).model_validate(model.model_dump())
    assert restored == model


# --- 3.2 入力の全長 -----------------------------------------------------


def test_total_input_tokens_without_cache_breakdown() -> None:
    usage = t.Usage(input_tokens=1200, output_tokens=64)

    assert usage.total_input_tokens == 1200


def test_total_input_tokens_with_cache_breakdown() -> None:
    usage = t.Usage(
        input_tokens=1000,
        output_tokens=64,
        cache_read_input_tokens=200,
        cache_creation_input_tokens=50,
    )

    assert usage.total_input_tokens == 1250


def test_total_input_tokens_with_partial_cache_breakdown() -> None:
    read_only = t.Usage(input_tokens=1000, output_tokens=64, cache_read_input_tokens=200)
    creation_only = t.Usage(input_tokens=1000, output_tokens=64, cache_creation_input_tokens=50)

    assert read_only.total_input_tokens == 1200
    assert creation_only.total_input_tokens == 1050


def test_total_input_tokens_is_not_serialized() -> None:
    assert "total_input_tokens" not in USAGE.model_dump()


# --- 8.3 要約に本文を入れない -------------------------------------------

BANNED_SCHEMA_MODELS = frozenset(
    {
        "StreamResult",
        "ContentBlock",
        "MessagesRequest",
        "InputMessage",
        "TextBlockParam",
        "ToolUseBlockParam",
        "ToolResultBlockParam",
        "ToolDef",
        "ToolTask",
        "NeedleCase",
        "CodeProblem",
        "ConversationPrefix",
        "TrialRecord",
        "MetricSnapshot",
    }
)

BANNED_PROPERTY_TOKENS = frozenset(
    {"text", "body", "content", "contents", "prompt", "response", "messages", "system"}
)

BANNED_PROPERTY_NAMES = frozenset(
    {"tool_input", "tool_input_raw", "blocks", "input", "output", "haystack"}
)


def _collect_schema(schema: Any, props: set[str], refs: set[str]) -> None:
    """スキーマを再帰で歩き、項目の名前と `$ref` の参照先を集める。"""
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "properties" and isinstance(value, dict):
                props.update(str(k) for k in value)
            if key == "$ref" and isinstance(value, str):
                refs.add(value.rsplit("/", 1)[-1])
            if key == "$defs" and isinstance(value, dict):
                refs.update(str(k) for k in value)
            _collect_schema(value, props, refs)
    elif isinstance(schema, list):
        for item in schema:
            _collect_schema(item, props, refs)


def test_summary_schema_never_references_a_body_carrying_model() -> None:
    props: set[str] = set()
    refs: set[str] = set()
    _collect_schema(t.Summary.model_json_schema(), props, refs)

    assert refs, "参照を 1 つも集められていない (歩き方の誤り)"
    assert refs & BANNED_SCHEMA_MODELS == set()


def test_summary_schema_has_no_body_like_property() -> None:
    props: set[str] = set()
    refs: set[str] = set()
    _collect_schema(t.Summary.model_json_schema(), props, refs)

    assert props, "項目を 1 つも集められていない (歩き方の誤り)"
    assert props & BANNED_PROPERTY_NAMES == set()
    offending = {name for name in props if set(name.split("_")) & BANNED_PROPERTY_TOKENS}
    assert offending == set()


def test_summary_carries_the_run_manifest_at_the_head() -> None:
    assert t.Summary.model_fields["conditions"].annotation is t.RunManifest


def test_run_manifest_schema_has_no_body_like_property() -> None:
    props: set[str] = set()
    refs: set[str] = set()
    _collect_schema(t.RunManifest.model_json_schema(), props, refs)

    assert refs & BANNED_SCHEMA_MODELS == set()
    assert props & BANNED_PROPERTY_NAMES == set()
    assert {name for name in props if set(name.split("_")) & BANNED_PROPERTY_TOKENS} == set()


def test_summary_json_does_not_contain_request_or_response_markers() -> None:
    dumped = SUMMARY.model_dump_json()

    assert "README.md を読んで" not in dumped
    assert "はい" not in dumped


def test_the_schema_walk_does_flag_a_body_carrying_model() -> None:
    """対照: 本文を持つ `TrialRecord` は、同じ検査に引っかかる (検査が空振りでないこと)。"""
    props: set[str] = set()
    refs: set[str] = set()
    _collect_schema(t.TrialRecord.model_json_schema(), props, refs)

    assert refs & BANNED_SCHEMA_MODELS != set()
    assert props & BANNED_PROPERTY_NAMES != set()
    assert {name for name in props if set(name.split("_")) & BANNED_PROPERTY_TOKENS} != set()


# --- 1.8 認証の情報 -----------------------------------------------------

SECRET_TOKENS = frozenset(
    {"key", "apikey", "token", "secret", "password", "credential", "auth", "bearer"}
)


def test_target_def_has_only_the_env_var_name_for_credentials() -> None:
    properties = set(t.TargetDef.model_json_schema()["properties"])

    assert properties == {
        "name",
        "base_url",
        "model",
        "notes",
        "api_key_env",
        "metrics_url",
        "max_context_tokens",
        "metric_map",
        "tool_markup_markers",
    }
    secretish = {
        name
        for name in properties
        if name != "api_key_env" and set(name.split("_")) & SECRET_TOKENS
    }
    assert secretish == set()


def test_target_def_json_keeps_only_the_env_var_name() -> None:
    dumped = TARGET.model_dump_json()

    assert "BENCH_OWN_API_KEY" in dumped
    assert "api_key" not in dumped.replace("api_key_env", "")


# --- 設定の下限と不変の決まり -------------------------------------------


def test_decode_trials_below_ten_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        t.DecodeSettings(trials=9)

    assert "trials" in str(exc_info.value)


def test_decode_trials_at_ten_is_accepted() -> None:
    assert t.DecodeSettings(trials=10).trials == 10


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.05, 1.5])
def test_out_of_range_tolerances_are_rejected(bad: float) -> None:
    with pytest.raises(ValidationError):
        t.Profile(name="quick", length_tolerance=bad)
    with pytest.raises(ValidationError):
        t.Profile(name="quick", compare_tolerance=bad)


def test_default_tolerances_match_the_design() -> None:
    profile = t.Profile(name="quick")

    assert profile.length_tolerance == pytest.approx(0.05)
    assert profile.compare_tolerance == pytest.approx(0.02)
    assert profile.output_sanity.repeat_min_chars == 12
    assert profile.output_sanity.repeat_min_count == 8
    assert profile.agent.threshold == pytest.approx(0.01)
    assert profile.agent.start_tokens == 20000
    assert profile.agent.end_tokens == 120000
    assert profile.agent.step_tokens == 20000
    assert profile.agent.conversations_per_stage == 5
    assert profile.decode.warmup_trials == 2
    assert profile.decode.max_tokens == 1024
    assert profile.prefill.max_tokens == 16
    assert profile.prefill.target_input_tokens == [8000, 32000, 128000]
    assert profile.concurrency.levels == [1, 2, 4, 8]
    assert profile.quality.needle_depths == [0, 25, 50, 75, 100]
    assert set(profile.chars_per_token) == set(t.ContentKind)


def test_agent_stage_range_must_not_be_inverted() -> None:
    with pytest.raises(ValidationError):
        t.AgentSettings(start_tokens=120000, end_tokens=20000)


def test_concurrency_levels_must_be_non_empty_and_positive() -> None:
    with pytest.raises(ValidationError):
        t.ConcurrencySettings(levels=[])
    with pytest.raises(ValidationError):
        t.ConcurrencySettings(levels=[1, 0])


def test_stream_timing_rejects_out_of_order_timestamps() -> None:
    with pytest.raises(ValidationError):
        t.StreamTiming(
            sent_at_utc=AT,
            sent_at_ns=5_000,
            message_start_ns=4_000,
            first_token_ns=6_000,
            last_token_ns=9_000,
            end_ns=10_000,
            event_count=3,
        )


def test_stream_timing_rejects_an_end_before_the_last_token() -> None:
    with pytest.raises(ValidationError):
        t.StreamTiming(
            sent_at_utc=AT,
            sent_at_ns=1_000,
            first_token_ns=3_000,
            last_token_ns=9_000,
            end_ns=8_000,
            event_count=3,
        )


def test_stream_timing_ignores_missing_timestamps() -> None:
    timing = t.StreamTiming(
        sent_at_utc=AT,
        sent_at_ns=1_000,
        message_start_ns=None,
        first_token_ns=None,
        first_text_ns=None,
        last_token_ns=None,
        end_ns=2_000,
        event_count=1,
    )

    assert timing.first_token_ns is None


def test_proportion_stat_rejects_a_numerator_above_the_denominator() -> None:
    with pytest.raises(ValidationError):
        t.ProportionStat(
            numerator=5,
            denominator=4,
            rate=1.25,
            ci95_low=0.0,
            ci95_high=1.0,
            upper95_one_sided=1.0,
        )


def test_messages_request_requires_at_least_one_output_token() -> None:
    with pytest.raises(ValidationError):
        t.MessagesRequest(model="m", max_tokens=0, messages=[MESSAGE])


def test_messages_request_always_streams() -> None:
    assert REQUEST.stream is True
    with pytest.raises(ValidationError):
        t.MessagesRequest.model_validate(
            {"model": "m", "max_tokens": 1, "messages": [MESSAGE.model_dump()], "stream": False}
        )


def test_unknown_field_is_rejected_so_typos_surface() -> None:
    with pytest.raises(ValidationError):
        t.Sampling.model_validate({"temperatur": 0.0})


# --- 印と分類の顔ぶれ ---------------------------------------------------


def test_tool_call_outcome_has_exactly_the_nine_classes() -> None:
    assert [member.value for member in t.ToolCallOutcome] == [
        "correct",
        "no_call",
        "unknown_tool",
        "args_unparseable",
        "args_schema_invalid",
        "markup_leaked",
        "wrong_call",
        "empty_or_truncated",
        "request_failed",
    ]


def test_quality_outcome_separates_not_scored_from_incorrect() -> None:
    assert [member.value for member in t.QualityOutcome] == [
        "correct",
        "incorrect",
        "not_scored",
    ]


def test_trial_flag_members() -> None:
    assert {member.name for member in t.TrialFlag} == {
        "SHORT_OUTPUT",
        "LENGTH_OFF_TARGET",
        "TOO_FEW_OUTPUT_TOKENS",
        "REPLACEMENT_CHAR",
        "REPETITION_LOOP",
    }


def test_metric_flag_members() -> None:
    assert {member.name for member in t.MetricFlag} == {
        "INSUFFICIENT_TRIALS",
        "PARTIAL_FAILURES",
        "SHORT_OUTPUTS",
        "LENGTH_OFF_TARGET",
        "SUSPECT_OUTPUTS",
    }


def test_run_status_members_match_the_state_machine() -> None:
    assert {member.value for member in t.RunStatus} == {
        "running",
        "completed",
        "aborted",
        "interrupted",
    }


def test_suite_names_match_the_five_suites() -> None:
    assert {member.value for member in t.SuiteName} == {
        "decode",
        "prefill",
        "concurrency",
        "quality",
        "agent",
    }


def test_threshold_verdict_members() -> None:
    assert {member.value for member in t.ThresholdVerdict} == {
        "below",
        "above",
        "undetermined",
    }


def test_trial_verdict_union_keeps_the_two_kinds_apart() -> None:
    tool_call = t.TrialRecord.model_validate_json(
        TRIAL.model_copy(
            update={"verdict": t.ToolCallVerdict(outcome=t.ToolCallOutcome.NO_CALL)}
        ).model_dump_json()
    )
    quality = t.TrialRecord.model_validate_json(
        TRIAL.model_copy(
            update={"verdict": t.QualityVerdict(task="needle", outcome=t.QualityOutcome.NOT_SCORED)}
        ).model_dump_json()
    )

    assert isinstance(tool_call.verdict, t.ToolCallVerdict)
    assert isinstance(quality.verdict, t.QualityVerdict)


def test_schema_and_generator_versions_start_at_one() -> None:
    assert t.SCHEMA_VERSION == 1
    assert t.GENERATOR_VERSION == 1
    assert MANIFEST.schema_version == t.SCHEMA_VERSION
    assert TRIAL.schema_version == t.SCHEMA_VERSION
    assert SUMMARY.schema_version == t.SCHEMA_VERSION
    assert MANIFEST.generator_version == t.GENERATOR_VERSION


# --- 依存の向き ---------------------------------------------------------

ALLOWED_IMPORT_ROOTS = frozenset(
    {"__future__", "collections", "datetime", "enum", "pathlib", "typing", "pydantic"}
)


def test_types_module_imports_nothing_from_bench_harness() -> None:
    tree = ast.parse(inspect.getsource(t))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "相対の読み込みがある"
            assert node.module is not None
            roots.add(node.module.split(".")[0])

    assert "bench_harness" not in roots
    assert roots <= ALLOWED_IMPORT_ROOTS, f"許していない読み込み: {roots - ALLOWED_IMPORT_ROOTS}"
