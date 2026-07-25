"""Day 20 guards: the self-repair loop fixes crashes without weakening anything.

What must hold: the loop only fires on *repairable* failures (a crash — never a
timeout, memory kill, or clean run), every candidate passes the static guard
(compiles, no held-out references, bounded size) before it earns a subprocess,
attempt N+1 is shown attempt N's failure (the loop learns), the attempt budget
is hard, mock mode honestly refuses, and both integrations — Trainer and FE —
adopt a repaired result only when it passes the exact same acceptance gate as a
first-try result. All offline: the "LLM" is a monkeypatched stand-in; the
sandboxed executor runs for real.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from crewml import config, llm, repair
from crewml.crew import feature_engineer as fe
from crewml.crew import trainer as tr
from crewml.crew.planner import build_plan
from crewml.crew.profiler import build_profile
from crewml.crew.trainer import run_trainer
from crewml.datasets import REGISTRY, load_train, verify_holdout_untouched
from crewml.repair import _static_guard, is_repairable, repair_loop

KEY = "credit-g"


def _plan() -> dict:
    return build_plan(build_profile(REGISTRY[KEY], load_train(KEY)))


def _fake_chat(replies):
    """A monkeypatch stand-in for llm.chat that pops canned replies in order."""
    queue = list(replies)
    calls: list[dict] = []

    def chat(system, user, **kwargs):
        calls.append({"system": system, "user": user})
        text = queue.pop(0) if queue else queue_exhausted()
        return llm.LLMResult(
            text=text, provider="fake", model="fake-1",
            prompt_tokens=100, completion_tokens=50,
        )

    def queue_exhausted():
        raise AssertionError("llm.chat called more times than replies were canned")

    chat.calls = calls
    return chat


GOOD = "x = 1\n"
BROKEN_SYNTAX = "def f(:\n"


def _run_fn_ok(source):
    return True, None, {"ran": source}


def _run_fn_fail(source):
    return False, "RuntimeError: still broken", None


# --- is_repairable: only genuine crashes qualify -----------------------------

@pytest.mark.parametrize(
    "result, expected",
    [
        (SimpleNamespace(ok=True, timed_out=False, oom=False, returncode=0), False),
        (SimpleNamespace(ok=False, timed_out=True, oom=False, returncode=None), False),
        (SimpleNamespace(ok=False, timed_out=False, oom=True, returncode=None), False),
        (SimpleNamespace(ok=False, timed_out=False, oom=False, returncode=1), True),
    ],
    ids=["clean", "timeout", "oom", "crash"],
)
def test_only_crashes_are_repairable(result, expected):
    assert is_repairable(result) is expected


# --- The static guard: what never earns a subprocess -------------------------

def test_guard_rejects_noncompiling_code():
    assert "does not compile" in _static_guard(BROKEN_SYNTAX)


def test_guard_rejects_holdout_loading_surfaces():
    for poison in (
        "from crewml.datasets import load_holdout",
        "pd.read_parquet('holdout.parquet')",
        "pd.read_parquet('test.parquet')",
    ):
        assert "forbidden token" in _static_guard(GOOD + poison)


def test_guard_permits_the_honest_holdout_vocabulary():
    # The Trainer's own script says `cv_score_is_holdout=False` — the guard must
    # not reject the label that exists to keep the numbers honest.
    assert _static_guard(GOOD + "flags = {'cv_score_is_holdout': False}") is None


def test_guard_rejects_runaway_size():
    assert "exceeds" in _static_guard("x = 1\n" * 20_000)


def test_guard_accepts_clean_code():
    assert _static_guard(GOOD) is None


# --- The loop itself ---------------------------------------------------------

def test_recovers_on_first_attempt(monkeypatch):
    monkeypatch.setattr(config, "is_mock_mode", lambda: False)
    monkeypatch.setattr(llm, "chat", _fake_chat([f"```python\n{GOOD}```"]))
    out = repair_loop("bad = )", "SyntaxError", run_fn=_run_fn_ok, context="ctx")
    assert out["attempted"] and out["recovered"]
    assert out["recovered_on_attempt"] == 1
    assert out["code"] == GOOD.strip()
    assert out["payload"]["ran"] == GOOD.strip()
    assert out["total_prompt_tokens"] == 100
    assert out["total_completion_tokens"] == 50


def test_second_attempt_sees_first_attempts_failure(monkeypatch):
    """The loop must learn: attempt 2's prompt carries attempt 1's error+code."""
    monkeypatch.setattr(config, "is_mock_mode", lambda: False)
    first = "y = 2  # attempt one\n"
    chat = _fake_chat([f"```python\n{first}```", f"```python\n{GOOD}```"])
    monkeypatch.setattr(llm, "chat", chat)

    def run_fn(source):
        if "attempt one" in source:
            return False, "ValueError: attempt one failed", None
        return True, None, {}

    out = repair_loop("orig", "orig error", run_fn=run_fn, context="ctx")
    assert out["recovered_on_attempt"] == 2
    assert "attempt one failed" in chat.calls[1]["user"]
    assert "attempt one" in chat.calls[1]["user"]  # its own broken fix, not orig


def test_attempt_budget_is_hard(monkeypatch):
    monkeypatch.setattr(config, "is_mock_mode", lambda: False)
    chat = _fake_chat([f"```python\n{GOOD}```"] * 2)
    monkeypatch.setattr(llm, "chat", chat)
    out = repair_loop("orig", "err", run_fn=_run_fn_fail, context="ctx", max_attempts=2)
    assert out["attempted"] and not out["recovered"]
    assert len(out["attempts"]) == 2 == len(chat.calls)


def test_noncompiling_candidate_is_bounced_not_run(monkeypatch):
    monkeypatch.setattr(config, "is_mock_mode", lambda: False)
    executed = []
    chat = _fake_chat([f"```python\n{BROKEN_SYNTAX}```", f"```python\n{GOOD}```"])
    monkeypatch.setattr(llm, "chat", chat)

    def run_fn(source):
        executed.append(source)
        return True, None, {}

    out = repair_loop("orig", "err", run_fn=run_fn, context="ctx")
    assert out["recovered_on_attempt"] == 2
    assert executed == [GOOD.strip()]          # the broken candidate never ran
    assert out["attempts"][0]["stage"] == "guard"
    assert "does not compile" in chat.calls[1]["user"]  # and was bounced back


def test_mock_mode_refuses_honestly(monkeypatch):
    monkeypatch.setattr(config, "is_mock_mode", lambda: True)
    out = repair_loop("orig", "err", run_fn=_run_fn_ok, context="ctx")
    assert out == {**out, "attempted": False, "reason_not_attempted": "mock_mode",
                   "recovered": False, "attempts": []}


def test_provider_error_is_recorded_not_raised(monkeypatch):
    monkeypatch.setattr(config, "is_mock_mode", lambda: False)

    def boom(system, user, **kwargs):
        raise ConnectionError("provider down")

    monkeypatch.setattr(llm, "chat", boom)
    out = repair_loop("orig", "err", run_fn=_run_fn_ok, context="ctx")
    assert out["attempted"] and not out["recovered"]
    assert out["attempts"][0]["stage"] == "llm"
    assert "provider down" in out["attempts"][0]["error"]


# --- Trainer integration (real sandbox runs, fake LLM) -----------------------

FAULTY_FE = """\
import pandas as pd


def add_features(df):
    out = df.copy()
    out["row_nan_count"] = df.isna().sum(axis=1) + undefined_offset
    return out
"""

FIXED_LINE = '    out["row_nan_count"] = df.isna().sum(axis=1)'
BROKEN_LINE = '    out["row_nan_count"] = df.isna().sum(axis=1) + undefined_offset'


def _script_fixing_chat():
    """A fake provider that extracts the failing script from the prompt and
    repairs the planted bug — the shape of what a real model does, minus the
    model."""
    calls = []

    def chat(system, user, **kwargs):
        calls.append(user)
        blocks = llm._CODE_FENCE.findall(user)
        assert blocks, "repair prompt must carry the source in a python fence"
        source = max(blocks, key=len)
        fixed = source.replace(BROKEN_LINE, FIXED_LINE)
        assert fixed != source, "the planted bug must be visible in the prompt"
        return llm.LLMResult(
            text=f"```python\n{fixed}\n```", provider="fake", model="fake-1",
            prompt_tokens=1000, completion_tokens=800,
        )

    chat.calls = calls
    return chat


@pytest.fixture(scope="module")
def repaired_training():
    """One shared repaired Trainer run: faulty FE in, recovered metrics out."""
    plan = _plan()
    import unittest.mock as mock

    with mock.patch.object(config, "is_mock_mode", lambda: False), \
         mock.patch.object(llm, "chat", _script_fixing_chat()):
        return run_trainer(plan, FAULTY_FE, KEY, param_search=False, self_repair=True)


def test_trainer_recovers_a_crashed_script(repaired_training):
    t = repaired_training
    assert t["ok"] is True
    assert t["repaired"] is True
    assert t["repair"]["recovered"] and t["repair"]["recovered_on_attempt"] == 1
    assert isinstance(t["cv_score"], float) and 0.5 <= t["cv_score"] <= 1.0


def test_repaired_run_keeps_full_provenance(repaired_training):
    r = repaired_training["repair"]
    assert r["attempts"][0]["provider"] == "fake"
    assert r["total_prompt_tokens"] >= 1000
    # The bulky fields stay out of the state record.
    assert "code" not in r and "payload" not in r


def test_trainer_repair_off_preserves_old_behaviour():
    t = run_trainer(_plan(), FAULTY_FE, KEY, param_search=False, self_repair=False)
    assert t["ok"] is False and t["repaired"] is False
    assert t["repair"]["attempted"] is False
    assert "undefined_offset" in (t["error"] or "")


def test_clean_run_never_invokes_repair(monkeypatch):
    def no_llm(system, user, **kwargs):
        raise AssertionError("repair must not fire on a clean run")

    monkeypatch.setattr(llm, "chat", no_llm)
    t = run_trainer(_plan(), fe.DEFAULT_FE_SOURCE, KEY, param_search=False,
                    self_repair=True)
    assert t["ok"] is True and t["repaired"] is False
    assert t["repair"]["reason_not_attempted"] == "not_needed"


def test_holdout_seal_survives_repair(repaired_training):
    assert verify_holdout_untouched(KEY)


# --- FE integration ----------------------------------------------------------

BAD_FE_GEN = """\
import pandas as pd


def add_features(df):
    out = df.copy()
    out["broken"] = out["no_such_column_here"] * 2
    return out
"""

GOOD_FE_FIX = """\
import pandas as pd


def add_features(df):
    out = df.copy()
    out["row_nan_count"] = df.isna().sum(axis=1).astype("int64")
    return out
"""


def test_fe_repair_recovers_failed_generation(monkeypatch):
    monkeypatch.setattr(config, "is_mock_mode", lambda: False)
    # Call 1 = generation (bad), call 2 = repair (good).
    monkeypatch.setattr(
        llm, "chat",
        _fake_chat([f"```python\n{BAD_FE_GEN}```", f"```python\n{GOOD_FE_FIX}```"]),
    )
    out = fe.run_feature_engineer(_plan(), KEY, with_llm=True, self_repair=True)
    assert out["meta"]["source"] == "llm_repaired"
    assert out["meta"]["validation"]["ok"] is True           # the repaired code's pass
    assert out["meta"]["llm_validation"]["ok"] is False      # the original's failure
    assert out["meta"]["repair"]["recovered_on_attempt"] == 1
    assert "row_nan_count" in out["code"]


def test_fe_falls_back_when_repair_also_fails(monkeypatch):
    monkeypatch.setattr(config, "is_mock_mode", lambda: False)
    monkeypatch.setattr(
        llm, "chat",
        _fake_chat([f"```python\n{BAD_FE_GEN}```"] * 3),  # gen + 2 repair attempts
    )
    out = fe.run_feature_engineer(_plan(), KEY, with_llm=True, self_repair=True)
    assert out["meta"]["source"] == "fallback"
    assert out["meta"]["repair"]["attempted"] is True
    assert out["meta"]["repair"]["recovered"] is False
    assert out["code"] == fe.DEFAULT_FE_SOURCE


# --- The study's honesty gates -----------------------------------------------

def test_study_refuses_mock_mode(monkeypatch):
    from crewml import self_repair_study as study

    monkeypatch.setattr(study, "is_mock_mode", lambda: True)
    with pytest.raises(RuntimeError, match="mock-mode"):
        study.run_self_repair_study()


def test_scripted_repairer_is_labelled_as_not_an_llm_measurement():
    """The stand-in mode's output must be unmistakable at every layer."""
    from crewml import self_repair_study as study

    report = {
        "provider": "scripted_stand_in", "model": "deterministic-repair-policy",
        "is_measurement_of_llm_capability": False, "max_attempts": 2,
        "recovered_runs": 3, "n_injected_runs": 3, "recovery_rate": 1.0,
        "false_positive_repairs_on_clean": 0, "fe_artifact_inconsistencies": 0,
        "holdout_seal_intact": True, "mean_abs_score_fidelity": 0.0,
        "datasets": ["credit-g"], "runs": [],
    }
    md = study.render_table_md(report)
    assert "NOT AN LLM MEASUREMENT" in md
    assert "mechanism recovery" in md and "recovery rate" not in md


def test_scripted_repairer_is_fault_blind():
    """The stand-in must not be an oracle over the fault list: it applies one
    fixed policy to whatever source it is shown."""
    from crewml.self_repair_study import scripted_repairer

    chat = scripted_repairer()
    broken = (
        "import requests\n"
        "import pandas as pd\n\n\n"
        "def add_features(df):\n"
        "    return df.nonexistent_method()\n\n\n"
        "FE_SOURCE_TEXT = 'stale'\n"
        "print('tail preserved')\n"
    )
    out = llm.extract_python(chat("sys", f"```python\n{broken}\n```").text)
    assert "import requests" not in out          # sandbox-refused import stripped
    assert "nonexistent_method" not in out       # broken body replaced
    assert "row_nan_count" in out                # with the contract-minimal one
    assert "print('tail preserved')" in out      # the rest of the module survives
    assert "'stale'" not in out                  # artifact constant re-synced
    compile(out, "<scripted>", "exec")


def test_study_refuses_a_dead_provider(monkeypatch):
    """A configured-but-dead provider (revoked key, restricted org) must abort
    the study BEFORE any trainer run — a 0% rate made of failed API calls would
    measure the provider's billing state, not the repair loop."""
    from crewml import self_repair_study as study

    monkeypatch.setattr(study, "is_mock_mode", lambda: False)

    def dead(system, user, **kwargs):
        raise ConnectionError("organization_restricted")

    monkeypatch.setattr(llm, "chat", dead)
    ran = []
    monkeypatch.setattr(study, "run_trainer", lambda *a, **k: ran.append(1))
    with pytest.raises(RuntimeError, match="dead provider"):
        study.run_self_repair_study()
    assert not ran


# --- Structural no-peeking ---------------------------------------------------

def test_repair_module_cannot_reach_the_holdout():
    # The module never imports the dataset layer at all — the only place the
    # holdout loader's name appears is inside the guard list that FORBIDS it.
    source = inspect.getsource(repair)
    assert "crewml.datasets" not in source
    assert "from crewml import config, llm" in source  # its only crewml imports
    assert "load_holdout" in str(repair.FORBIDDEN_TOKENS)
