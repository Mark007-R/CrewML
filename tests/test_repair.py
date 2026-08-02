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

# Full-crew / model-fit module: minute-scale by nature (Day 28 speed lanes).
pytestmark = pytest.mark.slow

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


def _stub_study(monkeypatch, repairer):
    """Run the REAL report-assembly path with training stubbed out.

    The point is that the honesty stamp must be PRODUCED by production code. An
    earlier version of this test hand-fed `is_measurement_of_llm_capability` into
    a literal dict and then asserted it — which would have passed even if the
    study never set the field at all.
    """
    from crewml import self_repair_study as study

    monkeypatch.setattr(study, "_plan_for", lambda key: {"dummy": "plan"})
    monkeypatch.setattr(study, "verify_holdout_untouched", lambda key: True)
    monkeypatch.setattr(
        study, "run_trainer",
        lambda *a, **k: {
            "ok": True, "cv_score": 0.8, "run_id": None,
            "repair": {"attempted": False, "reason_not_attempted": "not_needed"},
        },
    )
    fault = ({"key": "f1", "taxonomy": "exec_error", "description": "d",
              "source": "import pandas as pd\n\n\ndef add_features(df):\n    return df\n"},)
    return study._execute_study(
        ("credit-g",), fault, progress=False, repairer=repairer,
    )


def test_scripted_stamp_is_produced_by_the_study_not_the_test(monkeypatch):
    from crewml import self_repair_study as study

    report = _stub_study(monkeypatch, "scripted")
    assert report["is_measurement_of_llm_capability"] is False
    assert report["provider"] == "scripted_stand_in"
    # No provider was contacted, so EVAL_PROTOCOL §5 makes this a mock result.
    assert report["is_mock"] is True
    md = study.render_table_md(report)
    assert "NOT AN LLM MEASUREMENT" in md
    assert "mechanism recovery" in md


def test_live_mode_stamp_differs_from_scripted(monkeypatch):
    """The stamp must actually DISCRIMINATE, not be a constant."""
    report = _stub_study(monkeypatch, "live")
    assert report["is_measurement_of_llm_capability"] is True
    assert report["provider"] != "scripted_stand_in"


def test_empty_fault_selection_is_refused_before_anything_is_spent(monkeypatch):
    from crewml import self_repair_study as study

    monkeypatch.setattr(study, "is_mock_mode", lambda: False)
    ran = []
    monkeypatch.setattr(study, "run_trainer", lambda *a, **k: ran.append(1))
    with pytest.raises(ValueError, match="no faults selected"):
        study.run_self_repair_study(("credit-g",), (), repairer="scripted")
    assert not ran


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


# --- Provider failure must never masquerade as a repair failure --------------
#
# On 2026-07-25 a mid-run Groq daily-token exhaustion (100k TPD) made every
# repair call in an 18-run pass return 429. The study happily computed
# "recovery rate 0%" from calls that never reached the model. These guard that.

def _training_with_repair(attempts, recovered=False):
    return {
        "ok": recovered, "run_id": None, "cv_score": 0.5 if recovered else None,
        "repair": {
            "attempted": True, "recovered": recovered,
            "recovered_on_attempt": 1 if recovered else None,
            "attempts": attempts, "total_prompt_tokens": 0, "total_completion_tokens": 0,
        },
    }


def test_all_llm_stage_failures_are_flagged_unmeasured():
    from crewml import self_repair_study as study

    rec = study._run_record(
        "credit-g", {"key": "f", "taxonomy": "exec_error", "description": "d"},
        _training_with_repair([
            {"attempt": 1, "stage": "llm", "ok": False, "error": "RateLimitError 429 TPD"},
        ]),
        1.0, 0.79,
    )
    assert rec["llm_unavailable"] is True
    assert rec["recovered"] is False
    # The reason survives into the committed record — without it a rate limit is
    # indistinguishable from a model that could not fix the bug.
    assert "429" in rec["attempt_errors"][0]["error"]


def test_a_real_run_failure_is_not_flagged_unmeasured():
    from crewml import self_repair_study as study

    rec = study._run_record(
        "credit-g", {"key": "f", "taxonomy": "exec_error", "description": "d"},
        _training_with_repair([
            {"attempt": 1, "stage": "run", "ok": False, "error": "NameError again"},
            {"attempt": 2, "stage": "run", "ok": False, "error": "NameError again"},
        ]),
        1.0, 0.79,
    )
    assert rec["llm_unavailable"] is False   # the model tried and genuinely failed


def test_unmeasured_runs_are_excluded_from_the_rate_and_flagged_in_the_table():
    from crewml import self_repair_study as study

    report = {
        "provider": "groq", "model": "llama-3.3-70b-versatile",
        "is_measurement_of_llm_capability": True, "max_attempts": 2,
        "recovered_runs": 2, "n_injected_runs": 4, "measurable_runs": 2,
        "unmeasured_runs": 2, "recovery_rate": 1.0,
        "measurement_valid": True,
        "measurement_caveat": "2 of 4 injected runs never reached the provider",
        "false_positive_repairs_on_clean": 0, "fe_artifact_inconsistencies": 0,
        "holdout_seal_intact": True, "mean_abs_score_fidelity": 0.0,
        "datasets": ["credit-g"],
        "runs": [{
            "dataset": "credit-g", "fault": "non_finite", "recovered": False,
            "llm_unavailable": True, "recovered_on_attempt": None, "cv_score": None,
            "score_fidelity_vs_clean": None, "prompt_tokens": 0,
            "completion_tokens": 0, "wall_s": 49.1,
        }],
    }
    md = study.render_table_md(report)
    assert "PARTIALLY UNMEASURED" in md
    assert "unmeasured (provider)" in md
    # The rate is over measurable runs (2/2), never 2/4.
    assert "2/2 = 100%" in md and "2/4" not in md


def test_no_measurable_runs_yields_no_rate_at_all():
    from crewml import self_repair_study as study

    report = {
        "provider": "groq", "model": "m", "is_measurement_of_llm_capability": True,
        "max_attempts": 2, "recovered_runs": 0, "n_injected_runs": 4,
        "measurable_runs": 0, "unmeasured_runs": 4, "recovery_rate": None,
        "measurement_valid": False,
        "measurement_caveat": "4 of 4 injected runs never reached the provider",
        "false_positive_repairs_on_clean": 0, "fe_artifact_inconsistencies": 0,
        "holdout_seal_intact": True, "mean_abs_score_fidelity": None,
        "datasets": ["credit-g"], "runs": [],
    }
    md = study.render_table_md(report)
    assert "not measurable" in md
    assert "0%" not in md    # the failure mode being guarded: a fabricated zero


def test_provider_error_text_is_scrubbed_before_it_reaches_a_record(monkeypatch):
    """Rate-limit bodies name the org; those records get committed publicly."""
    monkeypatch.setattr(config, "is_mock_mode", lambda: False)
    # A verbatim-shaped Groq 429 body with a FAKE org id — the real one must
    # never be committed, which is the whole point of the scrub under test.
    real_429 = (
        "RateLimitError: Error code: 429 - Rate limit reached for model X in "
        "organization org_01fake0test0org0id0000000 service tier on_demand on "
        "tokens per day (TPD): Limit 100000, Used 96964"
    )

    def boom(system, user, **kwargs):
        raise RuntimeError(real_429)

    monkeypatch.setattr(llm, "chat", boom)
    out = repair_loop("orig", "err", run_fn=_run_fn_ok, context="ctx")
    recorded = out["attempts"][0]["error"]
    assert "org_01fake0test0org0id0000000" not in recorded
    assert "[redacted]" in recorded
    # The diagnostic value must survive the scrub — this is how a 429 stays
    # distinguishable from a model that genuinely could not fix the bug.
    assert "429" in recorded and "tokens per day" in recorded


def test_scrub_covers_key_shaped_strings():
    from crewml.repair import scrub

    for secret in ("gsk_abcdefgh12345678", "sk-ant-api03-abcdefgh1234",
                   "Bearer abcdefgh.12345678"):
        assert secret not in scrub(f"failed with {secret} in header")
    assert scrub(None) is None
    assert scrub("plain error") == "plain error"


# --- Post-review fixes (adversarial review of the Day-20 diff) ---------------

def test_repaired_fe_is_written_back_to_state():
    """The Ensembler is called with state["fe_code"]. If a repair rewrote
    add_features and we don't write it back, the Ensembler re-runs the code that
    just crashed and later holdout scoring re-applies an FE the shipped model was
    never fitted with."""
    from crewml.crew import nodes

    captured = {}

    def fake_run_trainer(plan, fe_code, key, **kwargs):
        captured["fe_code_in"] = fe_code
        return {"ok": True, "repaired": True, "fe_code_used": GOOD_FE_FIX,
                "repair": {"recovered": True}}

    import unittest.mock as mock
    with mock.patch.object(nodes, "run_trainer", fake_run_trainer):
        update = nodes.trainer({
            "plan": {}, "fe_code": FAULTY_FE, "dataset_key": KEY, "iteration": 0,
        })
    assert captured["fe_code_in"] == FAULTY_FE          # the crashed code went in
    assert update["fe_code"] == GOOD_FE_FIX             # the repaired code comes out


def test_clean_run_does_not_overwrite_fe_code():
    from crewml.crew import nodes
    import unittest.mock as mock

    with mock.patch.object(
        nodes, "run_trainer",
        lambda *a, **k: {"ok": True, "repaired": False, "fe_code_used": None},
    ):
        update = nodes.trainer({
            "plan": {}, "fe_code": FAULTY_FE, "dataset_key": KEY, "iteration": 0,
        })
    assert "fe_code" not in update


def test_repair_acceptance_requires_the_model_and_fe_artifacts(monkeypatch):
    """A 'fix' that scores but persists no model.joblib must NOT be adopted."""
    monkeypatch.setattr(config, "is_mock_mode", lambda: False)

    scoring_but_artifactless = (
        "from crew_io import emit_metrics\n"
        "emit_metrics(best_cv_score=0.99, best_model='fake', per_model=[])\n"
    )
    monkeypatch.setattr(
        llm, "chat",
        _fake_chat([f"```python\n{scoring_but_artifactless}```"] * 2),
    )
    t = run_trainer(_plan(), FAULTY_FE, KEY, param_search=False, self_repair=True)
    assert t["ok"] is False and t["repaired"] is False
    assert t["repair"]["recovered"] is False
    reasons = " ".join(a.get("error") or "" for a in t["repair"]["attempts"])
    assert "did not persist required artifact" in reasons
    assert "model.joblib" in reasons


def test_forbidden_token_rejection_informs_the_next_attempt(monkeypatch):
    """At temperature 0 an unchanged prompt returns the identical rejected reply,
    so a bare `continue` would burn the whole budget on one bad candidate."""
    monkeypatch.setattr(config, "is_mock_mode", lambda: False)
    poison = GOOD + "df = pd.read_parquet('test.parquet')\n"
    chat = _fake_chat([f"```python\n{poison}```", f"```python\n{GOOD}```"])
    monkeypatch.setattr(llm, "chat", chat)

    executed = []

    def run_fn(source):
        executed.append(source)
        return True, None, {}

    out = repair_loop("orig", "original error", run_fn=run_fn, context="ctx")
    assert out["recovered_on_attempt"] == 2
    assert executed == [GOOD.strip()]                       # poison never ran
    assert out["attempts"][0]["stage"] == "guard"
    # Attempt 2's prompt carries the rejection, and does NOT adopt the candidate.
    second = chat.calls[1]["user"]
    assert "REJECTED without being run" in second
    assert "test.parquet" not in second.split("[Your previous fix was REJECTED")[0]


def test_a_candidate_that_times_out_stops_the_loop(monkeypatch):
    """'Timeouts and OOM are never repaired' must hold for the loop's own runs."""
    monkeypatch.setattr(config, "is_mock_mode", lambda: False)
    chat = _fake_chat([f"```python\n{GOOD}```"] * 2)
    monkeypatch.setattr(llm, "chat", chat)

    timed_out = SimpleNamespace(ok=False, timed_out=True, oom=False, returncode=None)
    out = repair_loop(
        "orig", "err",
        run_fn=lambda src: (False, "execution exceeded timeout of 120s", timed_out),
        context="ctx",
        not_repairable_fn=lambda res: res is not None and not is_repairable(res),
    )
    assert out["recovered"] is False
    assert out["stopped_early"] == "candidate_run_hit_resource_limit"
    assert len(out["attempts"]) == 1 == len(chat.calls)   # budget 2, only 1 spent


def test_provider_unavailable_is_flagged_on_the_record(monkeypatch):
    monkeypatch.setattr(config, "is_mock_mode", lambda: False)

    def boom(system, user, **kwargs):
        raise ConnectionError("429 rate limit")

    monkeypatch.setattr(llm, "chat", boom)
    out = repair_loop("orig", "err", run_fn=_run_fn_ok, context="ctx")
    assert out["provider_unavailable"] is True


def test_successful_attempt_entry_is_schema_uniform(monkeypatch):
    monkeypatch.setattr(config, "is_mock_mode", lambda: False)
    monkeypatch.setattr(llm, "chat", _fake_chat([f"```python\n{GOOD}```"]))
    out = repair_loop("orig", "err", run_fn=_run_fn_ok, context="ctx")
    entry = out["attempts"][0]
    assert entry["ok"] is True
    assert entry["stage"] == "run"     # present on success too, not only failure


def _reporter_state(fe_source="llm_repaired", repaired=True):
    """Minimal terminal state exercising the Reporter's repair accounting."""
    return {
        "dataset_key": KEY, "iteration": 1, "max_iterations": 3,
        "profile": {"n_rows": 800, "n_features": 20, "assessment": {"flags": []}},
        "plan": {"task": "classification", "subtype": "binary", "metric": "roc_auc",
                 "drop_columns": [], "candidate_models": [], "cv": {"scheme": "StratifiedKFold"},
                 "imbalance_strategy": {}},
        "fe_meta": {
            "source": fe_source, "provider": "groq", "model": "llama-3.3-70b-versatile",
            "prompt_tokens": 1200, "completion_tokens": 400, "is_mock": False,
            "validation": {"new_columns": ["row_nan_count"]},
            "repair": {"attempted": True, "recovered": True,
                       "attempts": [{"attempt": 1, "ok": True}],
                       "total_prompt_tokens": 3000, "total_completion_tokens": 900},
        },
        "training": {
            "ok": True, "best_model": "random_forest", "cv_score": 0.79,
            "metrics": {"per_model": []}, "repaired": repaired,
            "repair": {"attempted": True, "recovered": True, "recovered_on_attempt": 1,
                       "max_attempts": 2,
                       "attempts": [{"attempt": 1, "ok": True, "provider": "groq",
                                     "model": "llama-3.3-70b-versatile"}],
                       "total_prompt_tokens": 5000, "total_completion_tokens": 1500},
        },
        "critiques": [{"decision": "finalize", "finding_codes": []}],
        "ensemble": {"attempted": False},
    }


def test_reporter_counts_llm_repaired_as_a_live_llm_surface():
    """The MODEL_CARD must never claim no LLM ran live on a run that made live
    calls — that is the one assertion the honesty artifact exists to get right."""
    from crewml.crew.reporter import build_report

    report = build_report(_reporter_state())
    usage = report["llm_usage"]
    assert usage["any_live"] is True
    assert usage["n_live"] >= 3          # fe generation + fe repair + trainer repair
    # Repair tokens are real spend and must be counted, not silently dropped.
    assert usage["prompt_tokens"] >= 1200 + 3000 + 5000
    assert usage["completion_tokens"] >= 400 + 900 + 1500
    nodes_seen = {n["node"] for n in usage["narratives"]}
    assert {"feature_engineer", "feature_engineer_repair", "trainer_repair"} <= nodes_seen
    assert not any("No advisory LLM narrative ran live" in w for w in report["warnings"])


def test_reporter_surfaces_the_repair_instead_of_a_silent_save():
    from crewml.crew.reporter import build_report

    report = build_report(_reporter_state())
    assert report["training"]["repaired"] is True
    assert report["training"]["repair_attempts"] == 1
    assert any("CRASHED and was repaired" in w for w in report["warnings"])
    assert any("failed validation on first generation" in w for w in report["warnings"])


def test_reporter_stays_quiet_about_repair_on_a_clean_run():
    from crewml.crew.reporter import build_report

    state = _reporter_state(fe_source="llm", repaired=False)
    state["training"]["repair"] = {"attempted": False, "reason_not_attempted": "not_needed"}
    report = build_report(state)
    assert report["training"]["repaired"] is False
    assert report["training"]["repair_attempts"] == 0
    assert not any("CRASHED" in w for w in report["warnings"])


def test_taxonomy_records_a_repaired_run_instead_of_reading_it_as_clean():
    """A crashed-then-repaired run is an exec_error the guard HANDLED. Emitting
    nothing would undercount the family and hide both fault and guard."""
    from crewml.failure_taxonomy import classify_run

    record = {
        "dataset_key": KEY, "run_id": "r1",
        "training": {"ok": True, "repaired": True,
                     "repair": {"attempted": True, "recovered": True,
                                "recovered_on_attempt": 1, "max_attempts": 2,
                                "attempts": [{"attempt": 1, "ok": True}]}},
        "critiques": [], "ensemble": {},
    }
    events = classify_run(record, run=record["run_id"])
    repair_events = [e for e in events if e["category"] == "exec_error"]
    assert len(repair_events) == 1
    assert repair_events[0]["outcome"] == "handled"
    assert "self-repair loop recovered" in repair_events[0]["evidence"]


def test_taxonomy_notes_a_failed_repair_on_a_fatal_run():
    from crewml.failure_taxonomy import classify_run

    record = {
        "dataset_key": KEY, "run_id": "r2",
        "training": {"ok": False, "timed_out": False, "error": "NameError: x",
                     "repaired": False,
                     "repair": {"attempted": True, "recovered": False,
                                "attempts": [{"attempt": 1}, {"attempt": 2}]}},
        "critiques": [], "ensemble": {},
    }
    events = classify_run(record, run=record["run_id"])
    fatal = [e for e in events if e["category"] == "exec_error" and e["outcome"] == "fatal"]
    assert len(fatal) == 1
    assert "self-repair attempted, 2 attempt(s), did not recover" in fatal[0]["evidence"]
