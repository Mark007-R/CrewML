"""Day 14 guards: the per-agent ablations are substitutional, honest, and well-paired.

The attribution's validity rests on the properties pinned here:

  * the ``no_planner`` / ``no_feature_engineer`` variants keep the FULL topology —
    same nodes, Critic loop included — and swap exactly one node body for its naive
    floor, so a score difference is attributable to that specialist alone;
  * the naive plan is genuinely profile-blind (no drops, cardinality-blind one-hot,
    no imbalance strategy, one default model, empty grid) yet keeps the protocol's
    positive class, so no drop can be an artifact of a flipped ROC-AUC orientation —
    and it is critique-deaf, so the loop has no actuator without a Planner;
  * the identity FE really is the identity, and both stand-ins present the same
    state/meta shape the real agents do;
  * a drop is only computed between two real held-out numbers, and the per-agent
    summaries count helps AND hurts honestly (a negative drop is a finding, not a bug).

The end-to-end run is kept offline + fast (no LLM, no grid search), mirroring
``test_ablation.py``; drop arithmetic and rendering are pure and tested in memory.
"""
from __future__ import annotations

import pytest

from crewml import agent_ablation
from crewml.crew import build_crew, build_graph, initial_state
from crewml.crew.feature_engineer import IDENTITY_FE_SOURCE
from crewml.crew.graph import _NODE_OVERRIDES
from crewml.crew.planner import build_naive_plan
from crewml.crew.trainer import _training_config
from crewml.datasets import REGISTRY

SPEC = REGISTRY["credit-g"]


# --- Variant topology: same graph, one node body swapped ---------------------

@pytest.mark.parametrize("variant", ["no_planner", "no_feature_engineer"])
def test_agent_ablations_keep_the_full_topology(variant):
    full = build_graph("full")
    ablated = build_graph(variant)
    # Same node set — the Critic (and its loop edge) survives both removals.
    assert set(ablated.nodes) == set(full.nodes)
    assert "critic" in ablated.nodes


@pytest.mark.parametrize("variant", ["no_planner", "no_feature_engineer"])
def test_agent_ablations_swap_exactly_one_node_body(variant):
    overrides = _NODE_OVERRIDES[variant]
    assert len(overrides) == 1  # one specialist removed per variant — never two


# --- The naive plan: profile-blind, protocol-true, Trainer-consumable --------

@pytest.fixture(scope="module")
def profile():
    from crewml.crew.profiler import run_profiler
    return run_profiler("credit-g")


@pytest.fixture(scope="module")
def naive_plan(profile):
    return build_naive_plan(profile)


def test_naive_plan_is_profile_blind(naive_plan):
    assert naive_plan["ablated"] == "planner"
    assert naive_plan["drop_columns"] == []                      # no leakage screen
    pre = naive_plan["preprocessing"]
    assert pre["numeric"]["zero_as_missing"] == []               # no disguised-missing handling
    assert pre["categorical"]["ordinal_columns"] == []           # cardinality-blind
    assert set(pre["categorical"]["onehot_columns"]) == set(pre["categorical"]["columns"])
    assert naive_plan["imbalance_strategy"]["recommended"] is False


def test_naive_plan_has_one_default_model_and_no_search(naive_plan):
    (cand,) = naive_plan["candidate_models"]
    assert cand["name"] == "random_forest"
    assert cand["param_grid"] == {}  # empty grid => Trainer skips its search


def test_naive_plan_keeps_the_protocol_positive_class(naive_plan, profile):
    # The 0/1 label mapping is part of the eval protocol, not a planning decision —
    # it must survive the ablation or a "drop" could be a flipped-AUC artifact.
    assert (
        naive_plan["imbalance_strategy"]["positive_class"]
        == (profile.get("target") or {}).get("positive_class")
    )
    assert naive_plan["imbalance_strategy"]["positive_class"] is not None


def test_naive_plan_feeds_the_trainer_unchanged(naive_plan):
    # The Trainer's config distiller must accept the naive plan as-is.
    cfg = _training_config(naive_plan)
    assert cfg["candidates"][0]["estimator"] == "RandomForestClassifier"
    assert cfg["cv_scheme"] == "StratifiedKFold"
    assert cfg["positive_class"] is not None


def test_naive_plan_is_critique_deaf(profile):
    # build_naive_plan takes no critique parameter — re-entry rebuilds the same plan.
    a = build_naive_plan(profile)
    b = build_naive_plan(profile)
    assert a == b


# --- The identity FE: truly the identity -------------------------------------

def test_identity_fe_is_the_identity():
    import pandas as pd

    ns: dict = {}
    exec(IDENTITY_FE_SOURCE, ns)
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out = ns["add_features"](df)
    assert list(out.columns) == ["a", "b"]
    assert out.equals(df)


# --- Drop arithmetic + honest per-agent aggregation (pure, in-memory) --------

def _rec(value, *, ok=True, mock=False):
    return {"ok": ok, "value": value, "mock": mock, "error": None if ok else "boom"}


def _result(metric, full_v, np_v, nfe_v, *, full_ok=True, np_ok=True, nfe_ok=True):
    arms = {
        "full": _rec(full_v, ok=full_ok),
        "no_planner": _rec(np_v, ok=np_ok),
        "no_feature_engineer": _rec(nfe_v, ok=nfe_ok),
    }
    fv = full_v if full_ok else None

    def _drop(v, ok):
        return round(fv - v, 6) if (fv is not None and ok) else None

    return {"metric": metric, "arms": arms,
            "drops": {"planner": _drop(np_v, np_ok), "feature_engineer": _drop(nfe_v, nfe_ok)}}


def test_summary_counts_helps_and_hurts():
    study = {
        "a": _result("roc_auc", 0.80, 0.72, 0.81),   # planner helped, FE hurt
        "b": _result("r2", 0.90, 0.90, 0.85),        # planner flat, FE helped
    }
    s = agent_ablation.assemble_report(study)["summary"]
    assert s["planner"]["helped_count"] == 1 and s["planner"]["hurt_count"] == 0
    assert s["planner"]["best_dataset"] == "a"
    assert s["feature_engineer"]["helped_count"] == 1 and s["feature_engineer"]["hurt_count"] == 1
    assert s["feature_engineer"]["mean_drop"] == pytest.approx((0.05 - 0.01) / 2)


def test_drop_is_none_when_an_arm_failed():
    study = {"broken": _result("r2", 0.5, 0.0, 0.4, np_ok=False)}
    assert study["broken"]["drops"]["planner"] is None
    s = agent_ablation.assemble_report(study)["summary"]
    # A failed arm contributes no drop — never a flattering zero.
    assert s["planner"]["compared"] == 0 and s["planner"]["mean_drop"] is None
    assert s["feature_engineer"]["compared"] == 1


def test_render_reports_negative_drops_as_is():
    study = {"d": _result("roc_auc", 0.78, 0.80, 0.78)}  # naive plan WON by 0.02
    report = agent_ablation.assemble_report(study)
    md = agent_ablation.render_markdown(report)
    assert "-0.0200" in md            # the hurt is printed, sign included
    assert "negative drop" in md.lower()


def test_any_mock_flag_propagates():
    study = {"d": _result("r2", 0.5, 0.5, 0.5)}
    study["d"]["arms"]["full"]["mock"] = True
    assert agent_ablation.assemble_report(study)["any_mock"] is True


# --- End-to-end offline: both variants run the loop-capable graph ------------

@pytest.fixture(scope="module")
def offline_env():
    mp = pytest.MonkeyPatch()
    for var in ("CREWML_PROFILER_LLM", "CREWML_PLANNER_LLM", "CREWML_FE_LLM", "CREWML_CRITIC_LLM"):
        mp.setenv(var, "0")
    mp.setenv("CREWML_TRAINER_PARAM_SEARCH", "0")
    yield
    mp.undo()


@pytest.fixture(scope="module")
def no_planner_final(offline_env):
    app = build_crew(variant="no_planner")
    return app.invoke(initial_state(SPEC, max_iterations=3), config={"recursion_limit": 50})


def test_no_planner_run_ships_with_the_naive_plan(no_planner_final):
    plan = no_planner_final["plan"]
    assert plan["ablated"] == "planner"
    assert plan["drop_columns"] == []
    # The Critic is still in the graph and passed judgement.
    assert "critic" in no_planner_final["trace"]
    assert no_planner_final["iteration"] >= 1
    assert no_planner_final["report"] is not None


def test_no_feature_engineer_run_ships_raw_features(offline_env):
    app = build_crew(variant="no_feature_engineer")
    final = app.invoke(initial_state(SPEC, max_iterations=3), config={"recursion_limit": 50})
    assert final["fe_meta"]["ablated"] == "feature_engineer"
    assert final["fe_code"] == IDENTITY_FE_SOURCE
    metrics = final["training"]["metrics"]
    assert metrics.get("n_engineered") == 0            # nothing was engineered
    assert "critic" in final["trace"]
    assert final["report"] is not None
