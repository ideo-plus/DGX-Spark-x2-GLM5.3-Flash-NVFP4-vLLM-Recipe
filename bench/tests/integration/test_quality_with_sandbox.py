"""品質の検査のコードの課題を、実物のコンテナで流す確認 (task 6.7、5.2、5.4)。

単体の試験 (`tests/unit/test_suites_quality.py`) は隔離を差し替えるので、
「取り出したコードが、本当に隔離の中で動いて合否が決まる」ことは、ここでしか
確かめられない。実行環境 (podman / docker) と、`bench/config/profiles.toml` の
`quick` に書いてあるイメージが手元にないときは、理由を添えて飛ばす。したがって、
コンテナのない機械でも、この試験の入ったまとまりは緑のままになる。

課題は手で書いた 2 問だけ (HumanEval+ の取得を待たない)。1 問は正しい解答、
もう 1 問は誤った解答を返させ、正解と不正解が分かれることを確かめる。
"""

from __future__ import annotations

import tomllib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from bench_harness.client.messages import HttpxMessagesClient
from bench_harness.corpus import humaneval
from bench_harness.scoring.sandbox import ContainerSandbox
from bench_harness.suites.base import SuiteContext, make_suite_context
from bench_harness.suites.quality import CODE_CONDITION_KEY, ProblemsLoader, QualitySuite
from bench_harness.types import (
    CodeProblem,
    ConditionPlan,
    DatasetRef,
    Profile,
    QualityOutcome,
    QualityVerdict,
    SandboxSettings,
    SandboxUnavailable,
    TargetDef,
    TrialRecord,
)
from fake_server import FakeServer, Script, text_response

PROFILES_TOML = Path(__file__).resolve().parents[2] / "config" / "profiles.toml"

CORRECT = CodeProblem(
    task_id="Handwritten/correct",
    prompt='def add_one(x: int) -> int:\n    """Return x + 1."""\n',
    entry_point="add_one",
    test="def check(candidate):\n    assert candidate(1) == 2\n    assert candidate(-3) == -2\n",
)
WRONG = CodeProblem(
    task_id="Handwritten/wrong",
    prompt='def double(x: int) -> int:\n    """Return x * 2."""\n',
    entry_point="double",
    test="def check(candidate):\n    assert candidate(3) == 6\n",
)


def _quick_sandbox_settings() -> SandboxSettings:
    """計測で実際に使う設定 (`quick`) を、そのまま読む。"""
    data: dict[str, Any] = tomllib.loads(PROFILES_TOML.read_text(encoding="utf-8"))
    return SandboxSettings.model_validate(data["profiles"]["quick"]["sandbox"])


@pytest.fixture(scope="module")
def settings() -> SandboxSettings:
    return _quick_sandbox_settings()


@pytest.fixture(scope="module")
def sandbox(settings: SandboxSettings) -> ContainerSandbox:
    box = ContainerSandbox(settings)
    status = box.available()
    if isinstance(status, SandboxUnavailable):
        pytest.skip(f"隔離の実行環境が使えない: {status.reason}")
    return box


def _profile(settings: SandboxSettings) -> Profile:
    return Profile.model_validate(
        {
            "name": "integration",
            "seed": 3,
            "sampling": {"temperature": 0.0},
            "timeout": {"connect_s": 5.0, "first_event_s": 5.0, "idle_s": 5.0, "total_s": 30.0},
            "sandbox": settings.model_dump(),
            "quality": {
                "toolcall_tasks": 1,
                "needle_lengths": [800],
                "needle_depths": [50],
                "trials_per_cell": 1,
                "code_max_tokens": 256,
            },
        }
    )


def _loader() -> ProblemsLoader:
    ref = humaneval.dataset_ref()

    def load() -> tuple[DatasetRef, list[CodeProblem]]:
        return ref, [CORRECT, WRONG]

    return load


def _responder(body: dict[str, Any]) -> Script:
    """正しい解答と、誤った解答を出し分ける (指示の中の関数の名前で見分ける)。"""
    messages = body["messages"]
    assert isinstance(messages, list)
    text = messages[-1]["content"][0]["text"]
    assert isinstance(text, str)
    if f"def {CORRECT.entry_point}(" in text:
        return text_response(
            f"```python\ndef {CORRECT.entry_point}(x: int) -> int:\n    return x + 1\n```"
        )
    return text_response(
        f"```python\ndef {WRONG.entry_point}(x: int) -> int:\n    return x + 2\n```"
    )


@asynccontextmanager
async def _ctx(server: FakeServer, profile: Profile) -> AsyncIterator[SuiteContext]:
    async with HttpxMessagesClient(server.base_url) as client:
        yield make_suite_context(
            client=client,
            profile=profile,
            target=TargetDef.model_validate(
                {"name": "fake", "base_url": server.base_url, "model": "fake-model"}
            ),
            run_id="20260920-000000-integration",
            put_body=lambda body: "ref",
        )


async def test_the_code_condition_scores_in_a_real_container(
    fake_server: FakeServer, sandbox: ContainerSandbox, settings: SandboxSettings
) -> None:
    """正しい解答が正解、誤った解答が不正解になる (実物のコンテナで動かす)。"""
    profile = _profile(settings)
    suite = QualitySuite(sandbox_factory=lambda _settings: sandbox, problems_loader=_loader())
    fake_server.set_response_factory(_responder)

    async with _ctx(fake_server, profile) as ctx:
        cond = next(item for item in suite.plan(ctx) if item.key == CODE_CONDITION_KEY)
        assert isinstance(cond, ConditionPlan)
        records: list[TrialRecord] = [record async for record in suite.run_condition(ctx, cond)]

    assert [record.verdict.outcome for record in records if record.verdict is not None] == [
        QualityOutcome.CORRECT,
        QualityOutcome.INCORRECT,
    ]
    first = records[0].verdict
    assert isinstance(first, QualityVerdict)
    assert first.sandbox is not None and first.sandbox.passed
    assert CORRECT.task_id in first.detail

    # 隔離の識別子が、計測ランに書ける形で取れる (5.5、design.md scoring/sandbox)
    assert suite.run_info().sandbox_image_ref is not None
