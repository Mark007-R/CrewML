"""Day 16 guards: the provider study probes honestly, prices only measured tokens,
and proves resilience rather than asserting it.

The study's claims rest on the properties pinned here:

  * ``cost_usd`` prices only real, complete token measurements — a missing count is
    ``None``, never a zero-dollar figure that reads as "free";
  * ``forced_provider`` switches the live config attribute AND the env var, restores
    both on exit (including on error), and rejects unknown providers;
  * ``probe_provider`` classifies each provider truthfully: no key → ``not_configured``
    with no network call; a raising provider → ``error`` carrying the exception text
    verbatim (the Groq restriction is evidence, not noise); a working provider →
    ``ok`` with measured latency, tokens and a priced cost; mock → ``offline``;
  * ``equality_check`` compares only pairs of real scores, counts equality strictly,
    and reports itself not-computable when there is no baseline — it never manufactures
    a resilience claim;
  * ``assemble_report`` flags mock arms, lists blocked providers with their evidence,
    and the rendered board carries the mock caveat and the blocked section;
  * the chart renders from the committed report shape, including with zero live arms.

Everything here is pure/in-memory — probes are exercised via a monkeypatched
``llm.chat``, never a live call.
"""
from __future__ import annotations

import os

import pytest

from crewml import config, llm
from crewml import provider_study as ps


# --- cost_usd: only measured tokens are priced (pure) -------------------------

def test_cost_usd_prices_measured_tokens_at_published_rates():
    # 1M in + 1M out at Groq's published rates.
    assert ps.cost_usd(1_000_000, 1_000_000, "groq") == pytest.approx(0.59 + 0.79)
    assert ps.cost_usd(1_000_000, 1_000_000, "anthropic") == pytest.approx(2.00 + 10.00)
    assert ps.cost_usd(1_000_000, 1_000_000, "mock") == 0.0


def test_cost_usd_refuses_missing_or_partial_measurements():
    assert ps.cost_usd(None, None, "groq") is None
    assert ps.cost_usd(1000, None, "groq") is None
    assert ps.cost_usd(None, 1000, "groq") is None


def test_cost_usd_rejects_unknown_providers():
    with pytest.raises(KeyError):
        ps.cost_usd(1, 1, "openai")


# --- forced_provider: scoped, restoring, strict -------------------------------

def test_forced_provider_switches_and_restores_config_and_env():
    prev_attr = config.LLM_PROVIDER
    prev_env = os.environ.get("CREWML_LLM_PROVIDER")
    with ps.forced_provider("mock"):
        assert config.LLM_PROVIDER == "mock"
        assert os.environ["CREWML_LLM_PROVIDER"] == "mock"
        assert config.is_mock_mode() is True
    assert config.LLM_PROVIDER == prev_attr
    assert os.environ.get("CREWML_LLM_PROVIDER") == prev_env


def test_forced_provider_restores_even_when_the_body_raises():
    prev_attr = config.LLM_PROVIDER
    with pytest.raises(RuntimeError):
        with ps.forced_provider("anthropic"):
            raise RuntimeError("boom")
    assert config.LLM_PROVIDER == prev_attr


def test_forced_provider_rejects_unknown_providers():
    with pytest.raises(KeyError):
        with ps.forced_provider("openai"):
            pass  # pragma: no cover


# --- probe_provider: truthful classification (monkeypatched llm.chat) ---------

def test_probe_mock_is_offline_and_never_calls_the_network(monkeypatch):
    def _explode(*a, **k):  # any call would be a bug
        raise AssertionError("mock probe must not call llm.chat")
    monkeypatch.setattr(llm, "chat", _explode)
    rec = ps.probe_provider("mock")
    assert rec["status"] == "offline"
    assert rec["error"] is None


def test_probe_without_a_key_is_not_configured_and_makes_no_call(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    def _explode(*a, **k):
        raise AssertionError("not_configured probe must not call llm.chat")
    monkeypatch.setattr(llm, "chat", _explode)
    rec = ps.probe_provider("anthropic")
    assert rec["status"] == "not_configured"
    assert rec["latency_s"] is None


def test_probe_carries_a_provider_failure_verbatim(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_test_not_a_real_key")
    def _restricted(*a, **k):
        raise RuntimeError("Organization has been restricted.")
    monkeypatch.setattr(llm, "chat", _restricted)
    rec = ps.probe_provider("groq")
    assert rec["status"] == "error"
    assert "Organization has been restricted." in rec["error"]
    assert rec["cost_usd"] is None
    assert isinstance(rec["latency_s"], float)


def test_probe_success_measures_latency_tokens_and_cost(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_test_not_a_real_key")
    result = llm.LLMResult(text="PONG", provider="groq", model="llama-3.3-70b-versatile",
                           prompt_tokens=20, completion_tokens=2)
    monkeypatch.setattr(llm, "chat", lambda *a, **k: result)
    rec = ps.probe_provider("groq")
    assert rec["status"] == "ok"
    assert rec["reply"] == "PONG"
    assert rec["prompt_tokens"] == 20 and rec["completion_tokens"] == 2
    assert rec["cost_usd"] == ps.cost_usd(20, 2, "groq")


# --- equality_check: strict, honest, never manufactured -----------------------

def _run(value, *, ok=True, metric="roc_auc"):
    return {"ok": ok, "value": value if ok else None, "metric": metric}


def test_equality_check_counts_exact_matches_only():
    fresh = {"a": _run(0.79130952380), "b": _run(0.5)}
    arch = {"a": _run(0.79130952380), "b": _run(0.5000001)}
    res = ps.equality_check(fresh, arch)
    assert res["n_compared"] == 2
    assert res["n_equal"] == 1
    assert res["all_equal"] is False
    by_ds = {r["dataset"]: r for r in res["rows"]}
    assert by_ds["a"]["equal"] is True
    assert by_ds["b"]["equal"] is False


def test_equality_check_skips_failed_or_missing_baselines():
    fresh = {"a": _run(0.8), "b": _run(None, ok=False), "c": _run(0.7)}
    arch = {"a": _run(None, ok=False)}  # b: failed fresh; c: no archival at all
    res = ps.equality_check(fresh, arch)
    assert res["n_compared"] == 0
    assert res["all_equal"] is False  # no comparison ≠ proven equal
    assert all(r["abs_diff"] is None for r in res["rows"])


def test_equality_check_with_no_data_is_not_computable():
    res = ps.equality_check({}, {})
    assert res["rows"] == []
    assert res["all_equal"] is False
    assert res["all_within_noise"] is False
    assert res["max_abs_diff"] is None


def test_sub_noise_wobble_is_named_float_noise_not_inequality():
    # A 1e-7 R² wobble (parallel-learner reduction order) is strictly unequal but
    # below the float-noise line — the verdict must say "float noise", never promote
    # it to "identical" and never cry "NOT equivalent".
    fresh = {"a": _run(0.9749972211802477, metric="r2"), "b": _run(0.5)}
    arch = {"a": _run(0.9749971043511926, metric="r2"), "b": _run(0.5)}
    res = ps.equality_check(fresh, arch)
    assert res["all_equal"] is False
    assert res["all_within_noise"] is True
    assert res["max_abs_diff"] == pytest.approx(1.168e-07, rel=1e-2)
    md = ps.render_markdown(ps.assemble_report(_probes(), {"mock": _mock_arm()}, res))
    assert "float-noise" in md
    assert "NOT equivalent" not in md


def test_a_real_inequality_is_still_called_out():
    fresh = {"a": _run(0.80)}
    arch = {"a": _run(0.75)}  # a 0.05 gap is a modelling difference, full stop
    res = ps.equality_check(fresh, arch)
    assert res["all_within_noise"] is False
    md = ps.render_markdown(ps.assemble_report(_probes(), {"mock": _mock_arm()}, res))
    assert "NOT equivalent" in md


# --- assemble_report + rendering ----------------------------------------------

def _probes(groq_status="error", anthropic_status="not_configured"):
    return {
        "groq": {"provider": "groq", "label": ps.PROVIDERS["groq"]["label"],
                 "model": "llama-3.3-70b-versatile", "status": groq_status,
                 "latency_s": 0.61, "prompt_tokens": None, "completion_tokens": None,
                 "cost_usd": None, "reply": None,
                 "error": "BadRequestError: organization_restricted" if groq_status == "error" else None,
                 "checked": "2026-07-21"},
        "anthropic": {"provider": "anthropic", "label": ps.PROVIDERS["anthropic"]["label"],
                      "model": "claude-sonnet-5", "status": anthropic_status,
                      "latency_s": None, "prompt_tokens": None, "completion_tokens": None,
                      "cost_usd": None, "reply": None, "error": None, "checked": "2026-07-21"},
        "mock": {"provider": "mock", "label": ps.PROVIDERS["mock"]["label"], "model": None,
                 "status": "offline", "latency_s": None, "prompt_tokens": None,
                 "completion_tokens": None, "cost_usd": None, "reply": None,
                 "error": None, "checked": "2026-07-21"},
    }


def _mock_arm():
    return {
        "credit-g": {"ok": True, "value": 0.7913, "metric": "roc_auc", "mock": True,
                     "crew_seconds": 33.1, "llm_narratives_live": 0,
                     "llm_prompt_tokens": 0, "llm_completion_tokens": 0,
                     "llm_cost_usd": None, "provider": "mock"},
    }


def test_assemble_report_flags_mock_and_lists_blocked_providers():
    report = ps.assemble_report(_probes(), {"mock": _mock_arm()},
                                ps.equality_check({}, {}))
    assert report["any_mock"] is True
    assert report["live_arms_run"] == []
    assert set(report["blocked_providers"]) == {"groq", "anthropic"}
    assert "organization_restricted" in report["blocked_providers"]["groq"]
    assert report["blocked_providers"]["anthropic"] == "not_configured"


def test_rendered_board_carries_the_evidence_and_the_caveats():
    report = ps.assemble_report(
        _probes(), {"mock": _mock_arm()},
        ps.equality_check(_mock_arm(), {"credit-g": _run(0.7913)}),
    )
    md = ps.render_markdown(report)
    assert "organization_restricted" in md          # the outage is on the board
    assert "NOT CONFIGURED" in md                   # the missing key is on the board
    assert "*(mock)*" in md                         # the honesty caveat
    assert "Blocked — what the live comparison still needs" in md
    assert "$ / 1M input" in md                     # the cost model is committed


def test_rendered_board_reports_resilience_verdict():
    fresh = {"credit-g": _run(0.7913), "kin8nm": _run(0.61, metric="r2")}
    arch = {"credit-g": _run(0.7913), "kin8nm": _run(0.61, metric="r2")}
    report = ps.assemble_report(_probes(), {"mock": _mock_arm()},
                                ps.equality_check(fresh, arch))
    assert report["resilience"]["all_equal"] is True
    md = ps.render_markdown(report)
    assert "2/2 datasets bit-identical" in md


def test_write_report_persists_json_and_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "PROVIDER_STUDY_RESULT_PATH", tmp_path / "day16.json")
    monkeypatch.setattr(ps, "PROVIDER_STUDY_TABLE_MD_PATH", tmp_path / "day16.md")
    report = ps.assemble_report(_probes(), {"mock": _mock_arm()}, ps.equality_check({}, {}))
    ps.write_report(report)
    assert (tmp_path / "day16.json").exists()
    assert (tmp_path / "day16.md").read_text(encoding="utf-8").startswith("# Day 16")


# --- Chart renders from the committed shape -----------------------------------

def test_provider_chart_renders_with_zero_live_arms(tmp_path):
    from crewml.charts import plot_provider_study
    report = ps.assemble_report(
        _probes(), {"mock": _mock_arm()},
        ps.equality_check(_mock_arm(), {"credit-g": _run(0.7913)}),
    )
    out = plot_provider_study(report, path=tmp_path / "day16.png")
    assert out.exists() and out.stat().st_size > 0
