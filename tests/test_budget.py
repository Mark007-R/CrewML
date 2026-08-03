"""Day 21 guards: run budgets cap cost/latency and the crew stops GRACEFULLY.

``CREWML_RUN_TOKEN_BUDGET`` existed since Day 5 but was never enforced. Day 21's
contract, asserted here:

  * **Ledger** — every LLM call is charged to the active :class:`RunBudget` with a
    per-agent breakdown; token and wall-clock caps exhaust exactly at their limits
    (``<= 0`` = uncapped), and refused calls are counted, not lost.
  * **Gate** — an exhausted budget makes :func:`crewml.llm.chat` raise
    :class:`BudgetExhaustedError` *before* the provider is touched; with no active
    budget behaviour is exactly pre-Day-21 (dormant, nothing charged).
  * **Budget-aware Critic** — an exhausted or unaffordable budget finalises the loop
    with an honest reason; affordability is measured from observed per-pass cost, so
    a zero-cost (deterministic/mock) run is never blocked.
  * **Graceful early-stop, end to end** — a real offline crew run under a spent
    budget COMPLETES (model, report, card), labels itself budget-constrained, and
    never crashes: degradation, not failure.
"""
from __future__ import annotations

import pytest

from crewml import budget as budget_mod
from crewml import config, llm
from crewml.budget import BudgetExhaustedError, RunBudget
from crewml.crew import critic as cr
from crewml.crew.critic import build_critique, decide, run_critic
from crewml.crew.reporter import build_report

from tests.test_critic import _plan, _profile, _training
from tests.test_reporter import _state

# Full-crew / model-fit module: minute-scale by nature (Day 28 speed lanes).
pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _no_leftover_budget():
    """Every test starts and ends with no active ledger (the dormant default)."""
    budget_mod.end_run()
    yield
    budget_mod.end_run()


class _Clock:
    """Injectable monotonic clock so time-cap tests need no sleeping."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


# --- The ledger ---------------------------------------------------------------

def test_charges_accumulate_with_per_agent_breakdown():
    b = RunBudget(token_budget=10_000)
    b.charge(agent="profiler", prompt_tokens=300, completion_tokens=100)
    b.charge(agent="critic", prompt_tokens=500, completion_tokens=200)
    b.charge(agent="critic", prompt_tokens=400, completion_tokens=100)
    assert b.tokens_spent == 1600
    assert b.tokens_remaining == 8400
    snap = b.snapshot()
    assert snap["n_calls"] == 3
    assert snap["per_agent"]["critic"] == {"calls": 2, "tokens": 1200, "refused": 0,
                                           "llm_time_s": 0.0}
    assert snap["per_agent"]["profiler"]["tokens"] == 400
    assert snap["exhausted"] is False and snap["stop_reason"] is None


def test_token_cap_exhausts_at_the_limit_and_enforce_refuses():
    b = RunBudget(token_budget=1000)
    b.charge(agent="planner", prompt_tokens=600, completion_tokens=400)  # exactly spent
    assert b.tokens_exhausted and b.exhausted and b.stop_reason == "tokens"
    with pytest.raises(BudgetExhaustedError) as exc:
        b.enforce(agent="critic")
    # The refusal is recorded — globally and against the agent that was turned away.
    snap = b.snapshot()
    assert snap["n_refused"] == 1
    assert snap["per_agent"]["critic"]["refused"] == 1
    assert exc.value.status["stop_reason"] == "tokens"


def test_granted_call_may_overshoot_but_is_always_charged_in_full():
    # The gate is pre-call: a call granted at 900/1000 spent may finish over cap.
    b = RunBudget(token_budget=1000)
    b.charge(agent="fe", prompt_tokens=700, completion_tokens=200)
    b.enforce(agent="fe")  # 900 < 1000 -> granted
    b.charge(agent="fe", prompt_tokens=300, completion_tokens=200)  # lands at 1400
    assert b.tokens_spent == 1400
    assert b.tokens_remaining == 0  # clamped, never negative
    assert b.tokens_exhausted


def test_time_cap_exhausts_on_the_injected_clock():
    clock = _Clock()
    b = RunBudget(token_budget=0, time_budget_s=60, clock=clock)
    assert b.time_remaining_s == 60
    clock.now += 59
    assert not b.exhausted
    clock.now += 2
    assert b.time_exhausted and b.stop_reason == "time"
    with pytest.raises(BudgetExhaustedError):
        b.enforce(agent="critic")


def test_nonpositive_caps_mean_uncapped():
    b = RunBudget(token_budget=0, time_budget_s=-1)
    b.charge(agent="x", prompt_tokens=10**9, completion_tokens=0)
    assert b.token_budget is None and b.time_budget_s is None
    assert b.tokens_remaining is None and b.time_remaining_s is None
    assert not b.exhausted
    b.enforce(agent="x")  # never raises


def test_run_budget_context_installs_and_always_retires(monkeypatch):
    assert budget_mod.active() is None
    with pytest.raises(RuntimeError, match="boom"):
        with budget_mod.run_budget(token_budget=5):
            assert budget_mod.active() is not None
            raise RuntimeError("boom")
    assert budget_mod.active() is None  # retired even on the exception path


# --- The gate in llm.chat ------------------------------------------------------

def _fake_live_provider(monkeypatch, *, prompt_tokens=100, completion_tokens=50):
    """Make config look live and stub the Groq transport with a fixed-cost reply."""
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key-never-used")
    calls = {"n": 0}

    def fake(system, user, *, temperature, max_tokens):
        calls["n"] += 1
        return llm.LLMResult(
            text="ok", provider="groq", model="fake",
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        )

    monkeypatch.setattr(llm, "_chat_groq", fake)
    return calls


def test_chat_charges_the_active_budget_under_the_agent_label(monkeypatch):
    _fake_live_provider(monkeypatch, prompt_tokens=120, completion_tokens=30)
    with budget_mod.run_budget(token_budget=10_000) as b:
        llm.chat("s", "u", agent="planner")
        llm.chat("s", "u", agent="planner")
    assert b.tokens_spent == 300
    assert b.snapshot()["per_agent"]["planner"]["calls"] == 2


def test_chat_refuses_before_touching_the_provider_when_exhausted(monkeypatch):
    calls = _fake_live_provider(monkeypatch)
    with budget_mod.run_budget(token_budget=100) as b:
        b.charge(agent="setup", prompt_tokens=100, completion_tokens=0)  # spend it all
        with pytest.raises(BudgetExhaustedError):
            llm.chat("s", "u", agent="critic")
    assert calls["n"] == 0  # the network was NEVER touched
    assert b.snapshot()["n_refused"] == 1


def test_chat_without_an_active_budget_is_unchanged(monkeypatch):
    calls = _fake_live_provider(monkeypatch)
    result = llm.chat("s", "u", agent="whoever")
    assert result.text == "ok" and calls["n"] == 1  # dormant: no gate, no ledger


def test_mock_mode_still_wins_over_the_budget_gate(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    with budget_mod.run_budget(token_budget=10_000) as b:
        with pytest.raises(llm.MockModeError):
            llm.chat("s", "u", agent="planner")
    assert b.tokens_spent == 0 and b.snapshot()["n_refused"] == 0


# --- The budget-aware Critic ---------------------------------------------------

_ACTIONABLE = [{
    "code": "overfit", "keyword": "overfit", "severity": "medium",
    "detail": "injected.", "directive": "reduce capacity.",
}]


def _snap(**over):
    base = {
        "token_budget": 200_000, "time_budget_s": 1800.0,
        "prompt_tokens": 0, "completion_tokens": 0, "tokens_spent": 0,
        "tokens_remaining": 200_000, "elapsed_s": 0.0, "time_remaining_s": 1800.0,
        "n_calls": 0, "n_refused": 0, "per_agent": {},
        "tokens_exhausted": False, "time_exhausted": False,
        "exhausted": False, "stop_reason": None,
    }
    base.update(over)
    return base


def test_decide_finalizes_when_the_run_budget_is_exhausted():
    decision, reason, _ = decide(
        _ACTIONABLE, _training(), [], iteration=1, max_iterations=3,
        budget=_snap(tokens_spent=200_000, tokens_remaining=0,
                     tokens_exhausted=True, exhausted=True, stop_reason="tokens"),
    )
    assert decision == "finalize"
    assert "run budget exhausted" in reason and "tokens" in reason


def test_decide_finalizes_when_another_pass_is_unaffordable():
    # Pass 1 cost ~5000 tokens; 1000 remain -> starting pass 2 would be cut off.
    decision, reason, _ = decide(
        _ACTIONABLE, _training(), [], iteration=1, max_iterations=3,
        budget=_snap(tokens_spent=5000, tokens_remaining=1000),
    )
    assert decision == "finalize"
    assert "cannot afford another pass" in reason


def test_decide_finalizes_when_remaining_time_is_below_per_pass_cost():
    decision, reason, _ = decide(
        _ACTIONABLE, _training(), [], iteration=1, max_iterations=3,
        budget=_snap(elapsed_s=100.0, time_remaining_s=10.0),
    )
    assert decision == "finalize"
    assert "cannot afford another pass" in reason


def test_decide_iterates_when_the_budget_can_afford_it():
    decision, _, _ = decide(
        _ACTIONABLE, _training(), [], iteration=1, max_iterations=3,
        budget=_snap(tokens_spent=5000, tokens_remaining=150_000,
                     elapsed_s=60.0, time_remaining_s=1740.0),
    )
    assert decision == "iterate"


def test_zero_cost_runs_are_never_blocked_by_affordability():
    # Mock/deterministic runs spend no tokens: estimate is 0 -> always affordable.
    decision, _, _ = decide(
        _ACTIONABLE, _training(), [], iteration=1, max_iterations=3,
        budget=_snap(),
    )
    assert decision == "iterate"


def test_decide_without_a_budget_is_exactly_pre_day21():
    decision, reason, _ = decide(
        _ACTIONABLE, _training(), [], iteration=1, max_iterations=3, budget=None,
    )
    assert decision == "iterate" and "under budget" in reason


def test_critique_embeds_the_budget_snapshot_at_the_pass_boundary():
    snap = _snap(tokens_spent=1234)
    critique = build_critique(
        _profile(), _plan(), _training(),
        critiques_so_far=[], iteration=1, max_iterations=3, budget=snap,
    )
    assert critique["budget"] == snap


def test_run_critic_reads_the_active_ledger_automatically():
    with budget_mod.run_budget(token_budget=50_000) as b:
        b.charge(agent="profiler", prompt_tokens=400, completion_tokens=100)
        critique = run_critic(
            _profile(), _plan(), _training(), iteration=1, with_llm=False,
        )
    assert critique["budget"]["tokens_spent"] == 500
    assert critique["budget"]["per_agent"]["profiler"]["calls"] == 1


def test_critic_narrative_degrades_to_unavailable_on_a_spent_budget(monkeypatch):
    # Provider looks live, but the ledger is spent: the narrative call is refused by
    # the gate and the node records it as unavailable — decision untouched, no crash.
    calls = _fake_live_provider(monkeypatch)
    with budget_mod.run_budget(token_budget=100) as b:
        b.charge(agent="setup", prompt_tokens=100, completion_tokens=0)
        critique = run_critic(_profile(), _plan(), _training(), iteration=1, with_llm=True)
    assert calls["n"] == 0
    narr = critique["llm_narrative"]
    assert narr["source"] == "unavailable" and narr["text"] is None
    assert "BudgetExhaustedError" in narr["reason"]
    assert critique["decision"] in ("iterate", "finalize")  # verdict stands on the core


# --- The Reporter surfaces the ledger ------------------------------------------

def test_report_carries_the_run_budget_ledger():
    snap = _snap(tokens_spent=4321, n_calls=5)
    report = build_report(_state(), run_budget=snap)
    assert report["run_budget"] == snap
    assert not any("BUDGET-CONSTRAINED" in w for w in report["warnings"])


def test_report_without_a_budget_records_none_and_no_budget_warning():
    report = build_report(_state())
    assert report["run_budget"] is None
    assert not any("BUDGET-CONSTRAINED" in w for w in report["warnings"])


def test_refused_calls_become_an_explicit_warning():
    report = build_report(_state(), run_budget=_snap(tokens_spent=99_000, n_refused=3))
    warning = next(w for w in report["warnings"] if "BUDGET-CONSTRAINED" in w)
    assert "3 LLM call(s) were refused" in warning


def test_budget_stopped_loop_becomes_an_explicit_warning():
    st = _state(extra={"critiques": [{
        "iteration": 1, "decision": "finalize",
        "reason": "run budget exhausted (time; ...) — finalising gracefully",
        "cv_score": 0.79, "finding_codes": ["overfit"],
        "llm_narrative": {"source": "unavailable", "text": None},
    }]})
    report = build_report(st, run_budget=_snap(time_exhausted=True, exhausted=True,
                                               stop_reason="time"))
    warning = next(w for w in report["warnings"] if "BUDGET-CONSTRAINED" in w)
    assert "finalised early on budget grounds" in warning


# --- End to end: graceful early-stop, never a crash ----------------------------

def test_full_crew_run_under_a_spent_budget_completes_gracefully(monkeypatch):
    """A real (offline) crew run whose wall-clock budget is already spent: the Critic
    wants to iterate (injected actionable finding) but finalises on budget grounds on
    pass 1, and the run still ships a model, a report, and the honesty warning —
    graceful early-stop, not a crash.
    """
    monkeypatch.setenv("CREWML_PROFILER_LLM", "0")
    monkeypatch.setenv("CREWML_PLANNER_LLM", "0")
    monkeypatch.setenv("CREWML_FE_LLM", "0")
    monkeypatch.setenv("CREWML_CRITIC_LLM", "0")
    monkeypatch.setenv("CREWML_TRAINER_PARAM_SEARCH", "0")

    # Every pass has an actionable finding — without the budget the loop would run
    # to max_iterations (cf. test_critic's loop test); with a spent clock it must not.
    monkeypatch.setattr(cr, "diagnose", lambda p, pl, t: list(_ACTIONABLE))

    from crewml.crew import build_crew, initial_state
    from crewml.datasets import REGISTRY

    app = build_crew()
    st = initial_state(REGISTRY["credit-g"], max_iterations=3)
    with budget_mod.run_budget(time_budget_s=1e-6):  # spent before the crew starts
        final = app.invoke(st, config={"recursion_limit": 60})

    assert final["iteration"] == 1  # the loop never got a second pass
    assert final["critiques"][0]["decision"] == "finalize"
    assert "run budget exhausted" in final["critiques"][0]["reason"]
    assert final["trace"][-1] == "reporter"  # the run COMPLETED
    report = final["report"]
    assert report["final_model"]["kind"] in ("single", "ensemble")  # a model shipped
    assert report["run_budget"]["time_exhausted"] is True
    assert any("BUDGET-CONSTRAINED" in w for w in report["warnings"])
