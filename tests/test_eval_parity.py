"""Day 28 guards: every competing system is scored by the same ruler.

The headline claim — "the crew beats the solo agent and holds its own against
AutoML" — is only meaningful if all four systems (dummy/RF baselines, solo
agent, AutoML ceiling, crew) are measured by the *same* metric, computed by the
*same* code, on the *same* sealed holdout. These tests pin that parity three
ways:

  * **one scoring authority** — ``crewml.scoring`` is the only module in the
    package allowed to import ``sklearn.metrics``, and every competing system
    calls its ``score_predictions``;
  * **one metric mapping** — the registry's per-dataset metric, the upload
    path's derived metric, and the scorer's dispatch all agree on
    binary→roc_auc, multiclass→f1_macro, regression→r2;
  * **one number** — identical predictions produce identical scores regardless
    of label dtype or class-column order, and each primary matches the
    protocol's definition computed independently.

Artifact checks (results/*.json carry the registry's metric) are skipped when
the git-ignored results are absent, so the suite runs on a fresh clone.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, r2_score, roc_auc_score

from crewml.datasets import BENCHMARK_KEYS, REGISTRY, DatasetSpec
from crewml.scoring import HIGHER_IS_BETTER, score_predictions
from crewml.uploads import derive_target

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "crewml"

# The protocol's metric-per-subtype mapping, stated once for all parity checks.
PROTOCOL_METRIC = {"binary": "roc_auc", "multiclass": "f1_macro", "regression": "r2"}

# The four systems the comparison tables put side by side.
COMPETING_MODULES = ("baselines", "solo_agent", "automl_baseline", "holdout_eval")


def _spec(subtype: str) -> DatasetSpec:
    task = "regression" if subtype == "regression" else "classification"
    return DatasetSpec(
        key=f"parity-{subtype}", openml_name="synthetic", version=1,
        task=task, subtype=subtype, metric=PROTOCOL_METRIC[subtype],
        note="synthetic spec for parity tests",
    )


# --- One scoring authority ---------------------------------------------------


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_only_scoring_imports_sklearn_metrics():
    """A second metrics import is a second, driftable definition of the score."""
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        if path.relative_to(PACKAGE).as_posix() == "scoring.py":
            continue
        if any(m == "sklearn.metrics" or m.startswith("sklearn.metrics.")
               for m in _imported_modules(path)):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == [], (
        f"these modules compute metrics outside crewml.scoring: {offenders}"
    )


@pytest.mark.parametrize("module", COMPETING_MODULES)
def test_every_competing_system_calls_the_canonical_scorer(module):
    source = (PACKAGE / f"{module}.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports_scorer = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "crewml.scoring"
        and any(alias.name == "score_predictions" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert imports_scorer, f"crewml/{module}.py does not import score_predictions"
    assert "score_predictions(" in source, (
        f"crewml/{module}.py imports the scorer but never calls it"
    )


# --- One metric mapping ------------------------------------------------------


@pytest.mark.parametrize("key", BENCHMARK_KEYS)
def test_registry_metric_matches_protocol_mapping(key):
    spec = REGISTRY[key]
    assert spec.metric == PROTOCOL_METRIC[spec.subtype]


@pytest.mark.parametrize(
    "column, subtype",
    [
        (pd.Series(["yes", "no"] * 30), "binary"),
        (pd.Series([True, False] * 30), "binary"),
        (pd.Series(["a", "b", "c", "d"] * 15), "multiclass"),
        (pd.Series([0, 1, 2] * 20), "multiclass"),
        (pd.Series(np.linspace(0.0, 5.0, 60)), "regression"),
    ],
)
def test_upload_derivation_agrees_with_protocol_mapping(column, subtype):
    """The Day-26 upload path derives the same metric the registry would assign."""
    derived = derive_target(column)
    assert derived.subtype == subtype
    assert derived.metric == PROTOCOL_METRIC[subtype]


def test_scorer_reports_the_specs_metric_name():
    for subtype in PROTOCOL_METRIC:
        spec = _spec(subtype)
        if subtype == "binary":
            result = score_predictions(
                spec, ["g", "b", "g", "b"],
                y_proba=np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]]),
                class_labels=["g", "b"], positive_class="b",
            )
        elif subtype == "multiclass":
            result = score_predictions(spec, [0, 1, 2, 1], y_pred=[0, 1, 2, 2])
        else:
            result = score_predictions(spec, [1.0, 2.0, 3.0], y_pred=[1.1, 1.9, 3.2])
        assert result["metric"] == PROTOCOL_METRIC[subtype]


# --- One number: caller-invariant scores -------------------------------------


def test_binary_auc_is_dtype_invariant():
    """Int-labeled and str-labeled callers of the same predictions score identically.

    The manifest stores classes as strings while estimators may keep native
    dtypes — the scorer's string-form comparison is what keeps e.g. the solo
    agent (raw sklearn labels) and the crew (decoded vocabulary) on one ruler.
    """
    spec = _spec("binary")
    rng = np.random.default_rng(0)
    y_true_int = rng.integers(0, 2, size=40)
    proba_pos = rng.random(40)
    y_proba = np.column_stack([1 - proba_pos, proba_pos])

    as_int = score_predictions(
        spec, y_true_int, y_proba=y_proba, class_labels=[0, 1], positive_class="1",
    )
    as_str = score_predictions(
        spec, [str(v) for v in y_true_int], y_proba=y_proba,
        class_labels=["0", "1"], positive_class="1",
    )
    assert as_int["value"] == as_str["value"]


def test_binary_auc_is_class_order_invariant():
    """Estimators order ``classes_`` differently; the score must not care."""
    spec = _spec("binary")
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=40)
    proba_pos = rng.random(40)

    forward = score_predictions(
        spec, y_true, y_proba=np.column_stack([1 - proba_pos, proba_pos]),
        class_labels=[0, 1], positive_class="1",
    )
    reversed_cols = score_predictions(
        spec, y_true, y_proba=np.column_stack([proba_pos, 1 - proba_pos]),
        class_labels=[1, 0], positive_class="1",
    )
    assert forward["value"] == pytest.approx(reversed_cols["value"])


def test_primaries_match_their_protocol_definitions():
    """Each primary equals the metric named in EVAL_PROTOCOL, computed directly."""
    rng = np.random.default_rng(2)

    y_bin = rng.integers(0, 2, size=50)
    p = rng.random(50)
    got = score_predictions(
        _spec("binary"), y_bin, y_proba=np.column_stack([1 - p, p]),
        class_labels=[0, 1], positive_class="1",
    )
    assert got["value"] == pytest.approx(roc_auc_score(y_bin, p))

    y_mc = rng.integers(0, 4, size=50)
    pred_mc = rng.integers(0, 4, size=50)
    got = score_predictions(_spec("multiclass"), y_mc, y_pred=pred_mc)
    assert got["value"] == pytest.approx(f1_score(y_mc, pred_mc, average="macro"))

    y_reg = rng.normal(size=50)
    pred_reg = y_reg + rng.normal(scale=0.3, size=50)
    got = score_predictions(_spec("regression"), y_reg, y_pred=pred_reg)
    assert got["value"] == pytest.approx(r2_score(y_reg, pred_reg))


def test_all_primaries_are_maximised():
    """Perfect predictions beat noisy ones on every primary — one direction only."""
    assert HIGHER_IS_BETTER is True
    rng = np.random.default_rng(3)

    y_bin = np.array([0, 1] * 25)
    perfect = score_predictions(
        _spec("binary"), y_bin,
        y_proba=np.column_stack([1.0 - y_bin, y_bin.astype(float)]),
        class_labels=[0, 1], positive_class="1",
    )
    noisy_p = rng.random(50)
    noisy = score_predictions(
        _spec("binary"), y_bin, y_proba=np.column_stack([1 - noisy_p, noisy_p]),
        class_labels=[0, 1], positive_class="1",
    )
    assert perfect["value"] == 1.0 >= noisy["value"]

    y_mc = rng.integers(0, 3, size=60)
    assert score_predictions(_spec("multiclass"), y_mc, y_pred=y_mc)["value"] == 1.0

    y_reg = rng.normal(size=60)
    assert score_predictions(_spec("regression"), y_reg, y_pred=y_reg)["value"] == 1.0


# --- Recorded artifacts carry the registry's metric --------------------------

RESULTS = ROOT / "results"
ARTIFACT_FILES = ("baseline_metrics.json", "automl_metrics.json")


@pytest.mark.parametrize("filename", ARTIFACT_FILES)
def test_recorded_metrics_match_the_registry(filename):
    path = RESULTS / filename
    if not path.exists():
        pytest.skip(f"{filename} not materialised on this clone")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    datasets = recorded.get("datasets", {})
    assert datasets, f"{filename} records no per-dataset results"
    for key, entry in datasets.items():
        assert key in REGISTRY, f"{filename} scores unknown dataset {key!r}"
        assert entry["metric"] == REGISTRY[key].metric, (
            f"{filename} recorded {entry['metric']!r} for {key!r}; "
            f"the registry says {REGISTRY[key].metric!r}"
        )
