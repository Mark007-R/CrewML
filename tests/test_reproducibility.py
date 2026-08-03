"""Day 23 — run manifest + result fingerprint + same-pins-same-result.

Three layers of guarantee, tested cheapest-first:

1. The fingerprint hashes ONLY the deterministic outcome — volatile facts
   (latency, tokens, narrative prose, run ids) must not move it, and every
   fact that IS the result (scores, chosen model, FE code) must.
2. The run manifest records the pins honestly and never leaks a secret.
3. End-to-end: two full offline crew runs in this process produce identical
   fingerprints. (The stronger fresh-process claim is measured by
   ``scripts/run_repro_check.py`` and committed as
   ``results/day23_reproducibility.json``.)
"""
from __future__ import annotations

import copy
import json

import pytest

from crewml import config, manifest, repro_study
from crewml.crew import build_crew, initial_state
from crewml.datasets import REGISTRY

# Full-crew / model-fit module: minute-scale by nature (Day 28 speed lanes).
pytestmark = pytest.mark.slow

SPEC = REGISTRY["credit-g"]


def _fake_state() -> dict:
    """A minimal finished-run state with realistic fields, volatile bits included."""
    return {
        "dataset_key": "credit-g",
        "task": "classification",
        "subtype": "binary",
        "metric": "roc_auc",
        "iteration": 1,
        "trace": ["profiler", "planner", "feature_engineer", "trainer",
                  "critic", "ensembler", "reporter"],
        "fe_code": "def add_features(df):\n    return df\n",
        "fe_meta": {"source": "default", "validation": {"ok": True}},
        "profile": {"llm_narrative": {"text": "volatile prose", "prompt_tokens": 500}},
        "training": {
            "ok": True, "best_model": "hgb", "cv_score": 0.7972,
            "param_search": False, "repaired": False,
            "run_id": "volatile-1234", "duration_s": 21.7,
        },
        "critiques": [{"decision": "finalize", "llm_narrative": {"text": "volatile"}}],
        "report": {
            "final_model": {
                "kind": "ensemble", "chosen": "ensemble",
                "members": ["hgb", "rf"], "single_best_model": "hgb",
                "cv_score": 0.7972, "ensemble_cv_score": 0.7972,
                "single_best_cv_score": 0.7891,
            },
            "run_budget": {"tokens_spent": 2057, "elapsed_s": 29.0},
        },
    }


# --- 1. Fingerprint semantics ------------------------------------------------

def test_fingerprint_is_stable_across_calls():
    st = _fake_state()
    assert manifest.result_fingerprint(st) == manifest.result_fingerprint(st)


def test_volatile_fields_do_not_move_the_fingerprint():
    st = _fake_state()
    base = manifest.result_fingerprint(st)

    changed = copy.deepcopy(st)
    changed["training"]["run_id"] = "different-run-id"
    changed["training"]["duration_s"] = 999.0
    changed["profile"]["llm_narrative"]["text"] = "totally different prose"
    changed["critiques"][0]["llm_narrative"]["text"] = "other words"
    changed["report"]["run_budget"] = {"tokens_spent": 1, "elapsed_s": 1.0}
    assert manifest.result_fingerprint(changed) == base


@pytest.mark.parametrize("mutate, what", [
    (lambda s: s["report"]["final_model"].update(cv_score=0.5), "final CV score"),
    (lambda s: s["training"].update(best_model="rf"), "winning model"),
    (lambda s: s.update(fe_code="def add_features(df):\n    return df * 2\n"), "FE code"),
    (lambda s: s["critiques"].append({"decision": "iterate"}), "Critic decisions"),
    (lambda s: s.update(trace=s["trace"] + ["trainer"]), "node trace"),
])
def test_result_facts_move_the_fingerprint(mutate, what):
    st = _fake_state()
    base = manifest.result_fingerprint(st)
    mutate(st)
    assert manifest.result_fingerprint(st) != base, f"{what} silently ignored"


def test_canonical_json_is_key_order_independent():
    a = manifest.canonical_json({"b": 1, "a": {"y": 2, "x": 3}})
    b = manifest.canonical_json({"a": {"x": 3, "y": 2}, "b": 1})
    assert a == b


# --- 2. Manifest pins --------------------------------------------------------

def test_manifest_records_the_pins_and_the_seals():
    m = manifest.build_run_manifest(_fake_state())
    assert m["schema_version"] == manifest.MANIFEST_SCHEMA_VERSION
    pins = m["pins"]
    assert pins["seed"] == config.SEED
    for pkg in ("numpy", "pandas", "scikit-learn"):
        assert pins["packages"][pkg], f"{pkg} version missing from the manifest"
    seals = m["dataset"]["seals"]
    recorded = json.loads(
        (config.RESULTS_DIR / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert seals["holdout_sha256"] == recorded["datasets"]["credit-g"]["holdout_sha256"]
    assert m["result_fingerprint"] == manifest.result_fingerprint(_fake_state())


def test_manifest_never_contains_a_secret():
    m = manifest.build_run_manifest(_fake_state())
    blob = json.dumps(m)
    assert "gsk_" not in blob and "sk-ant-" not in blob
    for key in m["pins"]["crewml_env"]:
        assert key.startswith("CREWML_")  # allowlist by construction
        assert "KEY" not in key.upper()


# --- 3. End-to-end: same pins, same fingerprint ------------------------------

@pytest.fixture(scope="module")
def two_offline_runs():
    """The full crew twice, offline pins, same seed — the in-process repro claim."""
    mp = pytest.MonkeyPatch()
    for var in ("CREWML_PROFILER_LLM", "CREWML_PLANNER_LLM",
                "CREWML_FE_LLM", "CREWML_CRITIC_LLM"):
        mp.setenv(var, "0")
    mp.setenv("CREWML_TRAINER_PARAM_SEARCH", "0")
    finals = []
    for _ in range(2):
        app = build_crew()
        st = initial_state(SPEC, max_iterations=1)
        finals.append(app.invoke(st, config={"recursion_limit": 50}))
    mp.undo()
    return finals


def test_same_pins_produce_the_same_result_fingerprint(two_offline_runs):
    a, b = two_offline_runs
    assert manifest.result_fingerprint(a) == manifest.result_fingerprint(b)


def test_the_fingerprint_covers_a_real_score(two_offline_runs):
    result = manifest.canonical_result(two_offline_runs[0])
    assert isinstance(result["final_model"]["cv_score"], float)
    assert result["fe"]["fe_code_sha256"], "FE code must be fingerprinted"
    assert result["cv_score_is_holdout"] is False


# --- 4. Study renderer -------------------------------------------------------

def _fake_report(live_measured: bool) -> dict:
    run = {
        "result_fingerprint": "a" * 64, "cv_score": 0.797173,
        "final_model": "ensemble", "fe_code_sha256": "b" * 64,
        "narratives_sha256": "c" * 64, "holdout_untouched": True,
        "duration_s": 22.6, "tokens": 0,
    }
    live = {"measured": False, "dataset": "credit-g",
            "reason_not_measured": "mock mode — no LLM key configured"}
    if live_measured:
        run2 = dict(run, narratives_sha256="d" * 64)
        live = {"measured": True, "dataset": "credit-g", "provider": "groq",
                "model": "llama-3.3-70b-versatile", "runs": [run, run2],
                "result_identical": True, "fe_code_identical": True,
                "narratives_identical": False}
    return {
        "study": "day23_reproducibility", "param_search": False,
        "max_iterations": 3,
        "pins_reference": manifest.environment_pins(),
        "arms": {
            "deterministic_core": {
                "datasets": {"credit-g": {"runs": [run, run], "identical": True}},
                "all_identical": True,
            },
            "seed_sensitivity": {
                "dataset": "credit-g", "seed_base": 42, "seed_alt": 43,
                "fingerprint_base": "a" * 64, "fingerprint_alt": "e" * 64,
                "cv_base": 0.797173, "cv_alt": 0.801020,
                "fingerprint_moved": True,
            },
            "live_llm": live,
        },
        "holdout_untouched_throughout": True,
    }


def test_renderer_is_deterministic_and_covers_both_live_branches():
    for measured in (False, True):
        report = _fake_report(measured)
        md = repro_study.render_markdown(report)
        assert md == repro_study.render_markdown(report)
        assert "same pins ⇒ same result fingerprint" in md
        assert "bit-identical" in md
        assert "seed reaches the model" in md
        if measured:
            assert "scored pipeline reproduced" in md
        else:
            assert "Not measured" in md


def test_renderer_reports_divergence_loudly_not_quietly():
    report = _fake_report(True)
    report["arms"]["deterministic_core"]["datasets"]["credit-g"]["identical"] = False
    report["arms"]["deterministic_core"]["all_identical"] = False
    report["arms"]["seed_sensitivity"]["fingerprint_moved"] = False
    md = repro_study.render_markdown(report)
    assert "DIVERGENCE FOUND" in md
    assert "SEED IGNORED" in md
