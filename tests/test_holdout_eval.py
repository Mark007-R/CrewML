"""Day 12 guards: the final holdout scorer is honest and correctly oriented.

The scorer is the single place where the locked split is allowed out of the vault, so
these tests are less about "does it compute a number" and more about the properties
that make the number *mean* something:

  * it scores what the crew actually shipped (Ensembler's model, or the Trainer's when
    the Ensembler stood down) — and reports nothing at all when the crew shipped nothing;
  * it puts binary predictions back in the dataset's own vocabulary before scoring, so
    ROC AUC points at the protocol's positive class rather than its complement;
  * the sandbox it drives never receives the holdout labels and never refits;
  * the seal is still intact afterwards.

The end-to-end test is skipped unless a real crew run exists on disk, so the suite stays
runnable on a fresh clone with no data.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from crewml.datasets import REGISTRY, holdout_path, load_manifest
from crewml.holdout_eval import (
    _PREDICT_TEMPLATE,
    _build_script,
    _decode_predictions,
    final_model_ref,
    score_on_holdout,
)

# --- Which model the crew shipped (unit, hand-built states) ------------------


def _state(*, ensemble=None, training=None) -> dict:
    return {"ensemble": ensemble, "training": training}


def test_final_model_ref_prefers_the_ensemblers_model():
    ref = final_model_ref(_state(
        ensemble={"attempted": True, "ok": True, "run_id": "ens1", "final_model_kind": "ensemble",
                  "final_cv_score": 0.9, "metrics": {"final_model_artifact": "final_model.joblib"}},
        training={"ok": True, "run_id": "trn1", "metrics": {"model_artifact": "model.joblib"}},
    ))
    assert ref["source"] == "ensembler"
    assert ref["run_id"] == "ens1"
    assert ref["model_artifact"] == "final_model.joblib"


def test_final_model_ref_falls_back_to_trainer_when_ensembler_stood_down():
    # An Ensembler that never attempted (too few candidates) means the Trainer's
    # single best IS what the crew shipped — score that, don't report nothing.
    ref = final_model_ref(_state(
        ensemble={"attempted": False, "ok": True, "chosen": "single"},
        training={"ok": True, "run_id": "trn1", "cv_score": 0.8,
                  "metrics": {"model_artifact": "model.joblib", "positive_class": "bad"}},
    ))
    assert ref["source"] == "trainer"
    assert ref["run_id"] == "trn1"
    assert ref["model_kind"] == "single"
    assert ref["positive_class"] == "bad"


def test_final_model_ref_falls_back_when_ensembler_crashed():
    ref = final_model_ref(_state(
        ensemble={"attempted": True, "ok": False, "run_id": "ens1", "error": "boom"},
        training={"ok": True, "run_id": "trn1", "metrics": {}},
    ))
    assert ref["source"] == "trainer"


def test_final_model_ref_is_none_when_the_crew_shipped_nothing():
    assert final_model_ref(_state(training={"ok": False}, ensemble={"attempted": False})) is None
    assert final_model_ref(_state()) is None


def test_no_usable_model_is_reported_not_scored():
    rec = score_on_holdout(REGISTRY["credit-g"], _state(training={"ok": False}))
    assert rec["ok"] is False
    assert "no usable fitted model" in rec["error"]
    assert "value" not in rec  # never fabricate a number for a failed run


# --- Binary label orientation (the bug that would silently invert AUC) -------


def test_decode_predictions_restores_the_datasets_own_vocabulary():
    # The Trainer fits 1 = positive ("bad"); holdout labels are the originals.
    assert _decode_predictions([1, 0, 1], {"1": "bad", "0": "good"}) == ["bad", "good", "bad"]


def test_decode_predictions_is_identity_without_a_mapping():
    # Multiclass / regression never map, so the raw predictions must pass through.
    assert _decode_predictions(["a", "b"], None) == ["a", "b"]


def test_decode_predictions_passes_through_unknown_labels():
    assert _decode_predictions([7], {"1": "bad"}) == ["7"]


# --- Structural honesty of the generated script -----------------------------


def test_prediction_script_never_fits_and_never_names_the_holdout_target():
    script = _build_script("def add_features(df):\n    return df\n",
                           {"classification": True, "needs_proba": True})
    assert ".fit(" not in script          # the crew's modeling is over before this runs
    assert "holdout.parquet" not in script  # only the FEATURES file is ever staged
    assert "add_features" in script
    assert "holdout_features.parquet" in script


def test_prediction_template_declares_no_refit():
    assert "refit_on_holdout=False" in _PREDICT_TEMPLATE


# --- End-to-end on a real crew run (skipped without data) -------------------

_RUN = Path("artifacts/crew/credit-g/final_run.json")


@pytest.mark.skipif(
    not (_RUN.exists() and holdout_path("credit-g").exists()),
    reason="needs a materialised dataset + a completed crew run",
)
def test_end_to_end_scores_credit_g_above_chance_and_keeps_the_seal():
    spec = REGISTRY["credit-g"]
    state = json.loads(_RUN.read_text())
    pos = load_manifest()["datasets"]["credit-g"]["target"]["positive_class"]

    rec = score_on_holdout(spec, state, positive_class=pos)

    assert rec["ok"] is True
    assert rec["metric"] == "roc_auc"
    # A correctly-oriented AUC must beat chance; an inverted positive class would
    # land near 1-value (~0.2) and this is the test that catches it.
    assert 0.55 < rec["value"] <= 1.0
    assert rec["refit_on_holdout"] is False
    assert rec["holdout_score_is_holdout"] is True
    assert rec["holdout_untouched"] is True
    assert rec["n_holdout"] > 0
    assert isinstance(rec["cv_minus_holdout"], float)
