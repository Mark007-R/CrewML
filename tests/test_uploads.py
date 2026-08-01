"""Day 26 — uploaded-CSV ingestion: chosen target, derived protocol, sealed split.

Three invariant families:

* **Derivation** — task/subtype/metric come from the CHOSEN column's dtype +
  cardinality per EVAL_PROTOCOL, never from a guess; undecidable picks
  (constant, id-like class explosion) are loud errors, not heuristics.
* **Sealing** — ingestion splits seed-locked, SHA-256s both splits into a
  per-upload manifest, and `verify_holdout_untouched` / `dataset_seals`
  dispatch to that manifest so upload runs carry benchmark-grade seal evidence.
* **Structure** — an upload registers as an ordinary DatasetSpec; CrewState
  still carries only the dataset key, so no-peeking stays structural.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import crewml.datasets as datasets_mod
import crewml.uploads as uploads_mod
from crewml.crew.state import initial_state
from crewml.datasets import REGISTRY, sha256_of_frame, verify_holdout_untouched
from crewml.manifest import dataset_seals
from crewml.uploads import (
    UploadError,
    derive_target,
    ingest_csv,
    restore_uploaded_datasets,
)

RNG = np.random.default_rng(0)


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _binary_frame(n: int = 120) -> pd.DataFrame:
    return pd.DataFrame({
        "age": RNG.integers(18, 90, n),
        "income": RNG.normal(50_000, 12_000, n).round(2),
        "city": RNG.choice(["pune", "mumbai", "goa"], n),
        "churned": RNG.choice(["yes", "no"], n, p=[0.3, 0.7]),
    })


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Redirect upload storage to tmp and restore the registry afterwards."""
    monkeypatch.setattr(datasets_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(uploads_mod, "DATA_DIR", tmp_path)
    before = set(REGISTRY)
    yield tmp_path
    for key in set(REGISTRY) - before:
        REGISTRY.pop(key, None)


# --- derivation: from the CHOSEN column, never a guess -----------------------

def test_binary_strings_derive_roc_auc():
    d = derive_target(pd.Series(["yes"] * 70 + ["no"] * 30))
    assert (d.task, d.subtype, d.metric) == ("classification", "binary", "roc_auc")
    # rarer class = positive, matching prepare_datasets / EVAL_PROTOCOL
    assert d.target_summary["positive_class"] == "no"


def test_bool_derives_binary():
    d = derive_target(pd.Series([True, False] * 20))
    assert (d.subtype, d.metric) == ("binary", "roc_auc")


def test_multiclass_strings_derive_f1_macro():
    d = derive_target(pd.Series(list("abc") * 20))
    assert (d.subtype, d.metric) == ("multiclass", "f1_macro")


def test_float_derives_regression():
    d = derive_target(pd.Series(RNG.normal(size=100)))
    assert (d.task, d.metric) == ("regression", "r2")


def test_high_cardinality_int_derives_regression():
    d = derive_target(pd.Series(np.arange(1000) * 3))
    assert d.task == "regression"


def test_low_cardinality_int_derives_classification():
    d = derive_target(pd.Series([0, 1, 2, 3] * 30))
    assert (d.task, d.subtype) == ("classification", "multiclass")


def test_constant_column_rejected():
    with pytest.raises(UploadError, match="constant"):
        derive_target(pd.Series(["same"] * 50))


def test_id_like_text_column_rejected():
    ids = pd.Series([f"row-{i}" for i in range(200)])
    with pytest.raises(UploadError, match="identifier"):
        derive_target(ids)


def test_id_like_integer_warns():
    d = derive_target(pd.Series(np.arange(500)))
    assert d.task == "regression"
    assert any("identifier" in w for w in d.warnings)


def test_rare_class_warns():
    d = derive_target(pd.Series(["a"] * 96 + ["b"] * 3))
    assert d.subtype == "binary"
    assert any("rarest class" in w for w in d.warnings)


# --- ingestion: validate, split, SEAL, register ------------------------------

def test_ingest_happy_path_registers_and_seals(sandbox):
    frame = _binary_frame()
    man = ingest_csv(_csv(frame), target_column="churned", filename="churn.csv")
    key = man["key"]
    assert key.startswith("upload-") and key in REGISTRY
    spec = REGISTRY[key]
    assert (spec.task, spec.subtype, spec.metric) == \
        ("classification", "binary", "roc_auc")
    # both splits exist, target column is normalised, seals match the files
    train = pd.read_parquet(sandbox / key / "train.parquet")
    holdout = pd.read_parquet(sandbox / key / "holdout.parquet")
    assert "target" in train.columns and "churned" not in train.columns
    assert man["train_sha256"] == sha256_of_frame(train)
    assert man["holdout_sha256"] == sha256_of_frame(holdout)
    assert man["n_train"] + man["n_holdout"] == len(frame)
    # stratified: the sealed holdout kept both classes
    assert holdout["target"].nunique() == 2


def test_ingest_never_guesses_target(sandbox):
    with pytest.raises(UploadError, match="never guesses"):
        ingest_csv(_csv(_binary_frame()), target_column="", filename="x.csv")


def test_ingest_unknown_column_lists_available(sandbox):
    with pytest.raises(UploadError, match="churned"):
        ingest_csv(_csv(_binary_frame()), target_column="nope", filename="x.csv")


def test_ingest_existing_target_column_steps_aside(sandbox):
    frame = _binary_frame().rename(columns={"income": "target"})
    man = ingest_csv(_csv(frame), target_column="churned", filename="x.csv")
    train = pd.read_parquet(sandbox / man["key"] / "train.parquet")
    assert "target_feature" in train.columns  # old `target` kept as a feature
    assert set(train["target"].unique()) <= {"yes", "no"}


def test_ingest_drops_missing_targets_never_imputes(sandbox):
    frame = _binary_frame(100)
    frame.loc[:9, "churned"] = None
    man = ingest_csv(_csv(frame), target_column="churned", filename="x.csv")
    assert man["source"]["n_rows_dropped_missing_target"] == 10
    assert man["n_train"] + man["n_holdout"] == 90


def test_ingest_rejects_garbage_and_tiny_files(sandbox):
    with pytest.raises(UploadError):
        ingest_csv(b"", target_column="y", filename="empty.csv")
    with pytest.raises(UploadError):
        ingest_csv(b"\x00\x01\x02\xff", target_column="y", filename="bin.dat")
    small = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
    with pytest.raises(UploadError, match="rows"):
        ingest_csv(_csv(small), target_column="y", filename="small.csv")


def test_ingest_is_idempotent_one_dataset_one_seal(sandbox):
    payload = _csv(_binary_frame())
    first = ingest_csv(payload, target_column="churned", filename="a.csv")
    second = ingest_csv(payload, target_column="churned", filename="a.csv")
    assert second["key"] == first["key"]
    assert second["already_ingested"] is True
    assert second["holdout_sha256"] == first["holdout_sha256"]


def test_different_target_choice_is_a_different_dataset(sandbox):
    payload = _csv(_binary_frame())
    a = ingest_csv(payload, target_column="churned", filename="a.csv")
    b = ingest_csv(payload, target_column="city", filename="a.csv")
    assert a["key"] != b["key"]  # the chosen target is part of the identity


def test_split_is_seed_locked(tmp_path, monkeypatch):
    payload = _csv(_binary_frame())  # one payload, two fresh ingests
    shas = []
    for sub in ("one", "two"):
        d = tmp_path / sub
        d.mkdir()
        monkeypatch.setattr(datasets_mod, "DATA_DIR", d)
        monkeypatch.setattr(uploads_mod, "DATA_DIR", d)
        man = ingest_csv(payload, target_column="churned", filename="x.csv")
        REGISTRY.pop(man["key"], None)
        shas.append((man["train_sha256"], man["holdout_sha256"]))
    assert shas[0] == shas[1]


# --- seals: benchmark-grade evidence for user data ---------------------------

def test_verify_holdout_untouched_dispatches_to_upload_manifest(sandbox):
    man = ingest_csv(_csv(_binary_frame()), target_column="churned",
                     filename="x.csv")
    key = man["key"]
    assert verify_holdout_untouched(key) is True
    # tamper with the sealed holdout -> the seal must break
    hpath = sandbox / key / "holdout.parquet"
    df = pd.read_parquet(hpath)
    df.iloc[0, 0] = 99999
    df.to_parquet(hpath, index=False)
    assert verify_holdout_untouched(key) is False


def test_dataset_seals_reads_upload_manifest(sandbox):
    man = ingest_csv(_csv(_binary_frame()), target_column="churned",
                     filename="x.csv")
    seals = dataset_seals(man["key"])
    assert seals["holdout_sha256"] == man["holdout_sha256"]
    assert seals["sealed_at_ingestion"] is True
    assert seals["n_holdout"] == man["n_holdout"]


def test_restore_reregisters_sealed_uploads(sandbox):
    man = ingest_csv(_csv(_binary_frame()), target_column="churned",
                     filename="x.csv")
    key = man["key"]
    REGISTRY.pop(key)  # simulate an API restart losing the in-memory registry
    restored = restore_uploaded_datasets()
    assert key in restored and key in REGISTRY


def test_restore_skips_broken_upload_dirs(sandbox):
    bad = sandbox / "upload-deadbeef0000"
    bad.mkdir()
    (bad / "upload_manifest.json").write_text("{not json", encoding="utf-8")
    assert "upload-deadbeef0000" not in restore_uploaded_datasets()


# --- structure: no-peeking stays structural for user data --------------------

def test_crew_state_for_upload_carries_no_paths(sandbox):
    man = ingest_csv(_csv(_binary_frame()), target_column="churned",
                     filename="x.csv")
    state = initial_state(REGISTRY[man["key"]], max_iterations=2)
    blob = json.dumps(state, default=str).lower()
    assert "holdout" not in blob and "parquet" not in blob
    assert state["dataset_key"] == man["key"]
